//! Per-thread page cache for fast allocations
//!
//! This module provides thread-local caching of free pages to reduce
//! contention on the global allocator for small allocations.

use std::cell::RefCell;

use super::page::PageId;

/// Per-thread cache of free pages
pub struct ThreadPageCache {
    /// Cached page IDs
    pages: Vec<PageId>,
    /// Maximum cache size
    max_size: usize,
    /// Statistics: cache hits
    hits: u64,
    /// Statistics: cache misses
    misses: u64,
}

impl ThreadPageCache {
    /// Create a new thread page cache
    pub fn new(max_size: usize) -> Self {
        Self {
            pages: Vec::with_capacity(max_size),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Try to allocate pages from the cache
    ///
    /// Returns Some if the cache has enough pages, None otherwise
    pub fn allocate(&mut self, count: usize) -> Option<Vec<PageId>> {
        if self.pages.len() >= count {
            self.hits += 1;
            Some(self.pages.drain(self.pages.len() - count..).collect())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Try to allocate a single page from the cache
    pub fn allocate_one(&mut self) -> Option<PageId> {
        if let Some(page) = self.pages.pop() {
            self.hits += 1;
            Some(page)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Return pages to the cache
    ///
    /// Returns any pages that exceed the cache capacity
    pub fn return_pages(&mut self, pages: Vec<PageId>) -> Vec<PageId> {
        let available_space = self.max_size.saturating_sub(self.pages.len());
        let to_cache = pages.len().min(available_space);

        if to_cache > 0 {
            self.pages.extend(pages.iter().take(to_cache).copied());
        }

        if pages.len() > to_cache {
            pages.into_iter().skip(to_cache).collect()
        } else {
            Vec::new()
        }
    }

    /// Return a single page to the cache
    ///
    /// Returns true if cached, false if cache is full
    pub fn return_one(&mut self, page: PageId) -> bool {
        if self.pages.len() < self.max_size {
            self.pages.push(page);
            true
        } else {
            false
        }
    }

    /// Get the number of cached pages
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Get the maximum cache size
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Clear the cache, returning all pages
    pub fn drain(&mut self) -> Vec<PageId> {
        std::mem::take(&mut self.pages)
    }

    /// Get cache statistics
    pub fn stats(&self) -> ThreadCacheStats {
        ThreadCacheStats {
            cached_pages: self.pages.len(),
            max_size: self.max_size,
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

/// Statistics for thread page cache
#[derive(Debug, Clone, Default)]
pub struct ThreadCacheStats {
    /// Number of pages currently cached
    pub cached_pages: usize,
    /// Maximum cache capacity
    pub max_size: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
}

impl ThreadCacheStats {
    /// Get hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Thread-local storage for page cache
thread_local! {
    static THREAD_PAGE_CACHE: RefCell<ThreadPageCache> = RefCell::new(
        ThreadPageCache::new(super::DEFAULT_THREAD_CACHE_SIZE)
    );
}

/// Get a reference to the thread-local page cache
///
/// Use this to interact with the thread's cache:
/// ```ignore
/// with_thread_cache(|cache| {
///     if let Some(page) = cache.allocate_one() {
///         // Use page
///     }
/// });
/// ```
pub fn with_thread_cache<F, R>(f: F) -> R
where
    F: FnOnce(&mut ThreadPageCache) -> R,
{
    THREAD_PAGE_CACHE.with(|cache| f(&mut cache.borrow_mut()))
}

/// Try to allocate pages from the thread cache
///
/// Returns None if cache doesn't have enough pages
pub fn try_allocate_from_cache(count: usize) -> Option<Vec<PageId>> {
    with_thread_cache(|cache| cache.allocate(count))
}

/// Return pages to the thread cache
///
/// Returns pages that couldn't fit in the cache
pub fn return_to_cache(pages: Vec<PageId>) -> Vec<PageId> {
    with_thread_cache(|cache| cache.return_pages(pages))
}

/// Get thread cache statistics
pub fn thread_cache_stats() -> ThreadCacheStats {
    with_thread_cache(|cache| cache.stats())
}

/// Global cache manager for coordinating thread caches
pub struct CacheManager {
    /// Target cache size per thread
    target_size: usize,
    /// Minimum cache size
    min_size: usize,
    /// Maximum cache size
    max_size: usize,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(target_size: usize) -> Self {
        Self {
            target_size,
            min_size: 4,
            max_size: target_size * 4,
        }
    }

    /// Resize thread cache based on pressure
    pub fn adjust_cache_size(&self, pressure: f64) -> usize {
        // Higher pressure = smaller cache
        let factor = 1.0 - pressure.min(1.0);
        let size = (self.target_size as f64 * factor) as usize;
        size.clamp(self.min_size, self.max_size)
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new(super::DEFAULT_THREAD_CACHE_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_cache_basic() {
        let mut cache = ThreadPageCache::new(10);

        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 10);

        // Add some pages
        let returned = cache.return_pages(vec![PageId::new(0), PageId::new(1), PageId::new(2)]);
        assert!(returned.is_empty());
        assert_eq!(cache.len(), 3);

        // Allocate
        let pages = cache.allocate(2);
        assert!(pages.is_some());
        assert_eq!(pages.unwrap().len(), 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_thread_cache_overflow() {
        let mut cache = ThreadPageCache::new(3);

        // Add more pages than capacity
        let returned = cache.return_pages(vec![
            PageId::new(0),
            PageId::new(1),
            PageId::new(2),
            PageId::new(3),
            PageId::new(4),
        ]);

        assert_eq!(cache.len(), 3);
        assert_eq!(returned.len(), 2);
    }

    #[test]
    fn test_thread_cache_underflow() {
        let mut cache = ThreadPageCache::new(10);

        cache.return_pages(vec![PageId::new(0), PageId::new(1)]);

        // Try to allocate more than available
        let result = cache.allocate(5);
        assert!(result.is_none());
    }

    #[test]
    fn test_thread_cache_stats() {
        let mut cache = ThreadPageCache::new(10);

        // Initial stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        // Miss
        cache.allocate_one();
        assert_eq!(cache.stats().misses, 1);

        // Add pages and hit
        cache.return_one(PageId::new(0));
        cache.allocate_one();
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_thread_cache_drain() {
        let mut cache = ThreadPageCache::new(10);

        cache.return_pages(vec![PageId::new(0), PageId::new(1), PageId::new(2)]);

        let drained = cache.drain();
        assert_eq!(drained.len(), 3);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_manager() {
        let manager = CacheManager::new(16);

        // No pressure = full size
        assert_eq!(manager.adjust_cache_size(0.0), 16);

        // Full pressure = minimum size
        assert_eq!(manager.adjust_cache_size(1.0), 4);

        // Medium pressure
        let size = manager.adjust_cache_size(0.5);
        assert!(size >= 4 && size <= 16);
    }

    #[test]
    fn test_thread_local_cache() {
        // Test the thread-local functions
        let returned = return_to_cache(vec![PageId::new(100), PageId::new(101)]);
        assert!(returned.is_empty());

        let pages = try_allocate_from_cache(1);
        assert!(pages.is_some());

        let stats = thread_cache_stats();
        assert!(stats.hits >= 1);
    }
}
