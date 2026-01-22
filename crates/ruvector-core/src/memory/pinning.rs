//! Page pinning with reference counting and RAII guards
//!
//! This module provides the pinning mechanism to prevent page eviction
//! during active use, using RAII guards for automatic cleanup.

use std::sync::Arc;

use super::page::PageRange;
use super::pool::UnifiedPool;

/// Error type for pin operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinError {
    /// Page has been freed
    PageFreed,
    /// Page is in invalid state
    InvalidState,
    /// Timeout waiting for pin
    Timeout,
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinError::PageFreed => write!(f, "Cannot pin freed page"),
            PinError::InvalidState => write!(f, "Page is in invalid state"),
            PinError::Timeout => write!(f, "Timeout waiting for pin"),
        }
    }
}

impl std::error::Error for PinError {}

impl From<super::metadata::PinError> for PinError {
    fn from(e: super::metadata::PinError) -> Self {
        match e {
            super::metadata::PinError::PageFreed => PinError::PageFreed,
        }
    }
}

/// Result type for pin operations
pub type PinResult<T> = Result<T, PinError>;

/// RAII guard that automatically unpins pages on drop
///
/// This struct ensures that pages are properly unpinned when they are
/// no longer needed, preventing pin leaks.
pub struct PinGuard {
    pool: Arc<UnifiedPool>,
    range: PageRange,
    active: bool,
}

impl PinGuard {
    /// Create a new pin guard
    ///
    /// Note: Pages should already be pinned before creating this guard
    pub(crate) fn new(pool: Arc<UnifiedPool>, range: PageRange) -> Self {
        Self {
            pool,
            range,
            active: true,
        }
    }

    /// Get the pinned page range
    #[inline]
    pub fn range(&self) -> &PageRange {
        &self.range
    }

    /// Get the number of pinned pages
    #[inline]
    pub fn num_pages(&self) -> usize {
        self.range.len()
    }

    /// Get the byte size of the pinned region
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.range.byte_size(self.pool.page_size())
    }

    /// Check if the guard is still active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Manually release the pin (same as drop but explicit)
    pub fn release(mut self) {
        self.do_unpin();
    }

    /// Get a raw pointer to the pinned data
    ///
    /// # Safety
    /// The caller must ensure the data is properly initialized
    #[inline]
    pub unsafe fn as_ptr(&self) -> *mut u8 {
        self.pool.page_ptr(self.range.start)
    }

    /// Get the pinned data as a slice
    ///
    /// # Safety
    /// The caller must ensure the data is properly initialized
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.as_ptr(), self.byte_size())
    }

    /// Get the pinned data as a mutable slice
    ///
    /// # Safety
    /// The caller must ensure they have exclusive access
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.as_ptr(), self.byte_size())
    }

    /// Downgrade to a weak reference (unpin but keep tracking)
    ///
    /// This is useful when you want to track pages but allow eviction
    pub fn downgrade(mut self) -> WeakPinRef {
        self.do_unpin();
        WeakPinRef {
            pool: Arc::clone(&self.pool),
            range: self.range.clone(),
        }
    }

    fn do_unpin(&mut self) {
        if self.active {
            self.pool.unpin(&self.range);
            self.active = false;
        }
    }
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        self.do_unpin();
    }
}

// PinGuard cannot be Clone - each guard represents a unique pin
// It can be Send if the pool is Send
unsafe impl Send for PinGuard {}
// PinGuard is not Sync - access should be through a single owner

/// Weak reference to a page range (not pinned, may be evicted)
pub struct WeakPinRef {
    pool: Arc<UnifiedPool>,
    range: PageRange,
}

impl WeakPinRef {
    /// Try to upgrade to a strong pin guard
    ///
    /// Returns None if the pages have been evicted
    pub fn upgrade(&self) -> Option<PinGuard> {
        // Check if pages are still allocated
        for page_id in self.range.iter() {
            if let Some(meta) = self.pool.metadata(page_id) {
                if meta.is_free() {
                    return None;
                }
            } else {
                return None;
            }
        }

        // Try to pin
        self.pool.pin(&self.range).ok()
    }

    /// Get the page range this reference points to
    #[inline]
    pub fn range(&self) -> &PageRange {
        &self.range
    }

    /// Check if the pages are still valid (not evicted)
    pub fn is_valid(&self) -> bool {
        for page_id in self.range.iter() {
            if let Some(meta) = self.pool.metadata(page_id) {
                if meta.is_free() {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

impl Clone for WeakPinRef {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            range: self.range.clone(),
        }
    }
}

/// Scoped pin that automatically acquires and releases pins
///
/// Useful for temporary pin operations in a controlled scope
pub struct ScopedPin<'a, F>
where
    F: FnOnce(&PinGuard),
{
    guard: Option<PinGuard>,
    on_release: Option<F>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, F> ScopedPin<'a, F>
where
    F: FnOnce(&PinGuard),
{
    /// Create a new scoped pin
    pub fn new(guard: PinGuard, on_release: F) -> Self {
        Self {
            guard: Some(guard),
            on_release: Some(on_release),
            _marker: std::marker::PhantomData,
        }
    }

    /// Get a reference to the guard
    pub fn guard(&self) -> &PinGuard {
        self.guard.as_ref().unwrap()
    }

    /// Get a mutable reference to the guard
    pub fn guard_mut(&mut self) -> &mut PinGuard {
        self.guard.as_mut().unwrap()
    }
}

impl<'a, F> Drop for ScopedPin<'a, F>
where
    F: FnOnce(&PinGuard),
{
    fn drop(&mut self) {
        if let (Some(guard), Some(on_release)) = (self.guard.take(), self.on_release.take()) {
            on_release(&guard);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{ContentType, PoolConfig};

    fn create_test_pool() -> Arc<UnifiedPool> {
        let config = PoolConfig {
            page_size: 4096,
            total_pages: 100,
            ..Default::default()
        };
        UnifiedPool::new(config).unwrap()
    }

    #[test]
    fn test_pin_guard_basic() {
        let pool = create_test_pool();
        let handle = pool.allocate(10, ContentType::KvCache).unwrap();

        let guard = handle.pin().unwrap();
        assert!(guard.is_active());
        assert_eq!(guard.num_pages(), 10);

        // Verify pages are pinned
        for page_id in guard.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert!(meta.is_pinned());
        }

        drop(guard);

        // Verify pages are unpinned
        for page_id in handle.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert!(!meta.is_pinned());
        }
    }

    #[test]
    fn test_pin_guard_release() {
        let pool = create_test_pool();
        let handle = pool.allocate(5, ContentType::TempBuffer).unwrap();

        let guard = handle.pin().unwrap();
        guard.release(); // Explicit release

        // Pages should be unpinned
        for page_id in handle.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert!(!meta.is_pinned());
        }
    }

    #[test]
    fn test_weak_pin_ref() {
        let pool = create_test_pool();
        let handle = pool.allocate(5, ContentType::Activation).unwrap();

        let guard = handle.pin().unwrap();
        let weak = guard.downgrade();

        // Should be valid
        assert!(weak.is_valid());

        // Should be able to upgrade
        let new_guard = weak.upgrade();
        assert!(new_guard.is_some());

        drop(new_guard);
        drop(handle);

        // After free, weak ref should be invalid
        // Note: In real usage, the weak ref would become invalid after eviction
    }

    #[test]
    fn test_multiple_pins() {
        let pool = create_test_pool();
        let handle = pool.allocate(5, ContentType::KvCache).unwrap();

        let guard1 = handle.pin().unwrap();
        let guard2 = handle.pin().unwrap();

        // Both pins should be active
        for page_id in handle.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert_eq!(meta.ref_count(), 2);
        }

        drop(guard1);

        // Still pinned with one ref
        for page_id in handle.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert_eq!(meta.ref_count(), 1);
            assert!(meta.is_pinned());
        }

        drop(guard2);

        // Now unpinned
        for page_id in handle.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert_eq!(meta.ref_count(), 0);
            assert!(!meta.is_pinned());
        }
    }
}
