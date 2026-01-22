//! Synchronization primitives and lock hierarchy
//!
//! This module provides synchronization utilities for the memory system,
//! including the eviction coordinator and lock hierarchy management.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Lock hierarchy levels for preventing deadlocks
///
/// Locks should always be acquired in order of increasing level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LockLevel {
    /// Level 1 (Global): Eviction mutex
    Eviction = 1,
    /// Level 2 (Per-Region): Region locks
    Region = 2,
    /// Level 3 (Per-Thread): Thread cache locks
    ThreadCache = 3,
}

/// Lock hierarchy tracker for debugging deadlocks
#[cfg(debug_assertions)]
thread_local! {
    static HELD_LOCKS: std::cell::RefCell<Vec<LockLevel>> = std::cell::RefCell::new(Vec::new());
}

/// Lock hierarchy utilities
pub struct LockHierarchy;

impl LockHierarchy {
    /// Check if acquiring this lock level would violate the hierarchy
    #[cfg(debug_assertions)]
    pub fn check_acquire(level: LockLevel) -> bool {
        HELD_LOCKS.with(|held| {
            let held = held.borrow();
            if let Some(&highest) = held.last() {
                if level <= highest {
                    return false; // Would violate hierarchy
                }
            }
            true
        })
    }

    /// Record that a lock was acquired
    #[cfg(debug_assertions)]
    pub fn record_acquire(level: LockLevel) {
        HELD_LOCKS.with(|held| {
            held.borrow_mut().push(level);
        });
    }

    /// Record that a lock was released
    #[cfg(debug_assertions)]
    pub fn record_release(level: LockLevel) {
        HELD_LOCKS.with(|held| {
            let mut held = held.borrow_mut();
            if let Some(pos) = held.iter().rposition(|&l| l == level) {
                held.remove(pos);
            }
        });
    }

    /// No-op versions for release builds
    #[cfg(not(debug_assertions))]
    pub fn check_acquire(_level: LockLevel) -> bool {
        true
    }

    #[cfg(not(debug_assertions))]
    pub fn record_acquire(_level: LockLevel) {}

    #[cfg(not(debug_assertions))]
    pub fn record_release(_level: LockLevel) {}
}

/// Coordinator for eviction operations
///
/// Ensures only one eviction operation runs at a time while allowing
/// other threads to wait efficiently.
pub struct EvictionCoordinator {
    /// Mutex for eviction serialization
    mutex: Mutex<()>,
    /// Flag indicating eviction in progress
    in_progress: AtomicBool,
    /// Number of threads waiting for eviction
    waiting_threads: AtomicUsize,
}

impl EvictionCoordinator {
    /// Create a new eviction coordinator
    pub fn new() -> Self {
        Self {
            mutex: Mutex::new(()),
            in_progress: AtomicBool::new(false),
            waiting_threads: AtomicUsize::new(0),
        }
    }

    /// Execute an eviction operation, or wait if one is in progress
    ///
    /// Returns true if eviction succeeded, false if it failed
    pub fn maybe_evict<F>(&self, evict_fn: F) -> bool
    where
        F: FnOnce() -> bool,
    {
        // Fast path: check if eviction already in progress
        if self.in_progress.load(Ordering::Acquire) {
            return self.wait_for_eviction();
        }

        // Try to acquire eviction lock
        let _guard = match self.mutex.try_lock() {
            Some(guard) => guard,
            None => {
                // Another thread is evicting, wait for it
                return self.wait_for_eviction();
            }
        };

        // Check hierarchy
        debug_assert!(LockHierarchy::check_acquire(LockLevel::Eviction));
        LockHierarchy::record_acquire(LockLevel::Eviction);

        // Mark eviction in progress
        self.in_progress.store(true, Ordering::Release);

        // Perform eviction
        let result = evict_fn();

        // Mark eviction complete
        self.in_progress.store(false, Ordering::Release);

        LockHierarchy::record_release(LockLevel::Eviction);

        result
    }

    /// Wait for an ongoing eviction to complete
    fn wait_for_eviction(&self) -> bool {
        self.waiting_threads.fetch_add(1, Ordering::Relaxed);

        // Spin-wait for eviction to complete
        while self.in_progress.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }

        self.waiting_threads.fetch_sub(1, Ordering::Relaxed);

        true // Assume eviction succeeded if we got here
    }

    /// Check if eviction is in progress
    pub fn is_evicting(&self) -> bool {
        self.in_progress.load(Ordering::Acquire)
    }

    /// Get the number of threads waiting for eviction
    pub fn waiting_threads(&self) -> usize {
        self.waiting_threads.load(Ordering::Relaxed)
    }
}

impl Default for EvictionCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free page list node
pub struct PageNode {
    /// Page ID
    pub page: u32,
    /// Next node pointer
    pub next: *mut PageNode,
}

impl PageNode {
    /// Create a new node
    pub fn new(page: u32) -> Box<Self> {
        Box::new(Self {
            page,
            next: std::ptr::null_mut(),
        })
    }
}

/// Lock-free page list using compare-and-swap
///
/// Based on the ADR-006 specification for the free list
pub struct LockFreePageList {
    /// Head of the list
    head: std::sync::atomic::AtomicPtr<PageNode>,
    /// Size of the list
    size: AtomicUsize,
}

impl LockFreePageList {
    /// Create a new empty list
    pub fn new() -> Self {
        Self {
            head: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            size: AtomicUsize::new(0),
        }
    }

    /// Push a page to the list
    pub fn push(&self, page: u32) {
        let mut new_node = PageNode::new(page);

        loop {
            let old_head = self.head.load(Ordering::Acquire);
            new_node.next = old_head;

            let new_node_ptr = Box::into_raw(new_node);

            match self.head.compare_exchange_weak(
                old_head,
                new_node_ptr,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.size.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(_) => {
                    // Retry with the node
                    new_node = unsafe { Box::from_raw(new_node_ptr) };
                }
            }
        }
    }

    /// Pop a page from the list
    pub fn pop(&self) -> Option<u32> {
        loop {
            let old_head = self.head.load(Ordering::Acquire);

            if old_head.is_null() {
                return None;
            }

            let next = unsafe { (*old_head).next };

            match self.head.compare_exchange_weak(
                old_head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.size.fetch_sub(1, Ordering::Relaxed);
                    let node = unsafe { Box::from_raw(old_head) };
                    return Some(node.page);
                }
                Err(_) => continue,
            }
        }
    }

    /// Get the current size
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for LockFreePageList {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LockFreePageList {
    fn drop(&mut self) {
        // Clean up remaining nodes
        while self.pop().is_some() {}
    }
}

// LockFreePageList is Send + Sync
unsafe impl Send for LockFreePageList {}
unsafe impl Sync for LockFreePageList {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_eviction_coordinator() {
        let coordinator = EvictionCoordinator::new();

        assert!(!coordinator.is_evicting());
        assert_eq!(coordinator.waiting_threads(), 0);

        let result = coordinator.maybe_evict(|| {
            assert!(coordinator.is_evicting());
            true
        });

        assert!(result);
        assert!(!coordinator.is_evicting());
    }

    #[test]
    fn test_lock_free_list_basic() {
        let list = LockFreePageList::new();

        assert!(list.is_empty());

        list.push(1);
        list.push(2);
        list.push(3);

        assert_eq!(list.len(), 3);

        // LIFO order
        assert_eq!(list.pop(), Some(3));
        assert_eq!(list.pop(), Some(2));
        assert_eq!(list.pop(), Some(1));
        assert_eq!(list.pop(), None);
    }

    #[test]
    fn test_lock_free_list_concurrent() {
        let list = Arc::new(LockFreePageList::new());
        let mut handles = vec![];

        // Spawn multiple producers
        for i in 0..4 {
            let list = Arc::clone(&list);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    list.push(i * 100 + j);
                }
            }));
        }

        // Wait for producers
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(list.len(), 400);

        // Consume all
        let mut count = 0;
        while list.pop().is_some() {
            count += 1;
        }

        assert_eq!(count, 400);
    }

    #[test]
    fn test_eviction_coordinator_concurrent() {
        let coordinator = Arc::new(EvictionCoordinator::new());
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn multiple threads trying to evict
        for _ in 0..4 {
            let coord = Arc::clone(&coordinator);
            let cnt = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    coord.maybe_evict(|| {
                        cnt.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(std::time::Duration::from_micros(100));
                        true
                    });
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Each eviction should have run exactly once
        // (some may have waited and returned without running)
        let total = counter.load(Ordering::Relaxed);
        assert!(total > 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_lock_hierarchy() {
        // Acquiring in order should work
        assert!(LockHierarchy::check_acquire(LockLevel::Eviction));
        LockHierarchy::record_acquire(LockLevel::Eviction);

        assert!(LockHierarchy::check_acquire(LockLevel::Region));
        LockHierarchy::record_acquire(LockLevel::Region);

        assert!(LockHierarchy::check_acquire(LockLevel::ThreadCache));

        // Clean up
        LockHierarchy::record_release(LockLevel::Region);
        LockHierarchy::record_release(LockLevel::Eviction);
    }
}
