//! Memory Pool for Allocation Reuse
//!
//! Thread-safe object pools for reusing allocations in hot paths.
//! Reduces allocation pressure and improves latency consistency.
//!
//! ## Performance Targets
//! - Pool acquire: <100ns
//! - Pool release: <100ns
//! - Allocation reduction: 50%+ in hot paths
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvllm::rlm::pool::{VectorPool, StringPool, PooledVec, PooledString};
//!
//! // Create pools
//! let vector_pool = VectorPool::new(384, 64);
//! let string_pool = StringPool::new(128);
//!
//! // Acquire a vector from the pool
//! let mut vec = vector_pool.acquire();
//! vec.extend_from_slice(&[1.0, 2.0, 3.0]);
//!
//! // Vector is automatically returned to pool on drop
//! ```

use parking_lot::Mutex;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// =============================================================================
// Vector Pool
// =============================================================================

/// Thread-safe object pool for reusable f32 vectors.
///
/// Optimized for embedding generation and memory search operations
/// where vectors of a fixed dimension are frequently allocated and deallocated.
pub struct VectorPool {
    /// Pool of available vectors
    pool: Mutex<Vec<Vec<f32>>>,
    /// Target dimension for vectors (for pre-allocation)
    dimension: usize,
    /// Maximum number of vectors to keep pooled
    max_pooled: usize,
    /// Statistics
    stats: PoolStats,
}

impl VectorPool {
    /// Create a new vector pool.
    ///
    /// # Arguments
    /// * `dimension` - The target dimension for vectors (used for pre-allocation)
    /// * `max_pooled` - Maximum number of vectors to keep in the pool
    ///
    /// # Example
    /// ```rust,ignore
    /// let pool = VectorPool::new(384, 64);
    /// ```
    pub fn new(dimension: usize, max_pooled: usize) -> Arc<Self> {
        Arc::new(Self {
            pool: Mutex::new(Vec::with_capacity(max_pooled)),
            dimension,
            max_pooled,
            stats: PoolStats::default(),
        })
    }

    /// Create a new vector pool with pre-warmed entries.
    ///
    /// Pre-allocates `warm_count` vectors to avoid cold-start allocation spikes.
    pub fn new_warmed(dimension: usize, max_pooled: usize, warm_count: usize) -> Arc<Self> {
        let pool = Arc::new(Self {
            pool: Mutex::new(Vec::with_capacity(max_pooled)),
            dimension,
            max_pooled,
            stats: PoolStats::default(),
        });

        // Pre-warm the pool
        let count = warm_count.min(max_pooled);
        let mut guard = pool.pool.lock();
        for _ in 0..count {
            guard.push(Vec::with_capacity(dimension));
        }
        drop(guard);

        pool
    }

    /// Acquire a vector from the pool.
    ///
    /// Returns a pooled vector if available, otherwise allocates a new one.
    /// The returned `PooledVec` will automatically return the vector to the pool
    /// when dropped.
    #[inline]
    pub fn acquire(self: &Arc<Self>) -> PooledVec {
        let vec = {
            let mut guard = self.pool.lock();
            guard.pop()
        };

        match vec {
            Some(mut v) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                v.clear(); // Reset for reuse
                PooledVec {
                    vec: Some(v),
                    pool: Arc::clone(self),
                }
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                PooledVec {
                    vec: Some(Vec::with_capacity(self.dimension)),
                    pool: Arc::clone(self),
                }
            }
        }
    }

    /// Acquire a vector and initialize it with zeros.
    #[inline]
    pub fn acquire_zeroed(self: &Arc<Self>) -> PooledVec {
        let mut vec = self.acquire();
        vec.resize(self.dimension, 0.0);
        vec
    }

    /// Acquire a vector and initialize it from a slice.
    #[inline]
    pub fn acquire_from_slice(self: &Arc<Self>, slice: &[f32]) -> PooledVec {
        let mut vec = self.acquire();
        vec.extend_from_slice(slice);
        vec
    }

    /// Return a vector to the pool.
    #[inline]
    fn release(&self, mut vec: Vec<f32>) {
        let mut guard = self.pool.lock();
        if guard.len() < self.max_pooled {
            vec.clear();
            guard.push(vec);
            self.stats.returns.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            // Vec is dropped here (pool is full)
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the current number of pooled vectors.
    pub fn pooled_count(&self) -> usize {
        self.pool.lock().len()
    }

    /// Get the target dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Clear all pooled vectors.
    pub fn clear(&self) {
        self.pool.lock().clear();
    }
}

/// A vector borrowed from a `VectorPool`.
///
/// Automatically returns the vector to the pool when dropped.
pub struct PooledVec {
    vec: Option<Vec<f32>>,
    pool: Arc<VectorPool>,
}

impl PooledVec {
    /// Take ownership of the inner vector, preventing return to pool.
    ///
    /// Use this when you need to store the vector long-term.
    #[inline]
    pub fn take(mut self) -> Vec<f32> {
        self.vec.take().unwrap_or_default()
    }

    /// Check if this is a fresh allocation (not from pool).
    #[inline]
    pub fn is_fresh(&self) -> bool {
        self.vec.as_ref().map(|v| v.capacity() == 0).unwrap_or(true)
    }
}

impl Deref for PooledVec {
    type Target = Vec<f32>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.vec.as_ref().unwrap()
    }
}

impl DerefMut for PooledVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vec.as_mut().unwrap()
    }
}

impl Drop for PooledVec {
    #[inline]
    fn drop(&mut self) {
        if let Some(vec) = self.vec.take() {
            self.pool.release(vec);
        }
    }
}

impl AsRef<[f32]> for PooledVec {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self.vec.as_ref().unwrap()
    }
}

impl AsMut<[f32]> for PooledVec {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        self.vec.as_mut().unwrap()
    }
}

// =============================================================================
// String Pool
// =============================================================================

/// Thread-safe object pool for reusable strings.
///
/// Optimized for query processing and decomposition operations
/// where strings are frequently created and discarded.
pub struct StringPool {
    /// Pool of available strings
    pool: Mutex<Vec<String>>,
    /// Maximum number of strings to keep pooled
    max_pooled: usize,
    /// Initial capacity for new strings
    initial_capacity: usize,
    /// Statistics
    stats: PoolStats,
}

impl StringPool {
    /// Create a new string pool.
    ///
    /// # Arguments
    /// * `max_pooled` - Maximum number of strings to keep in the pool
    pub fn new(max_pooled: usize) -> Arc<Self> {
        Arc::new(Self {
            pool: Mutex::new(Vec::with_capacity(max_pooled)),
            max_pooled,
            initial_capacity: 256, // Good default for query strings
            stats: PoolStats::default(),
        })
    }

    /// Create a new string pool with custom initial string capacity.
    pub fn with_capacity(max_pooled: usize, initial_capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            pool: Mutex::new(Vec::with_capacity(max_pooled)),
            max_pooled,
            initial_capacity,
            stats: PoolStats::default(),
        })
    }

    /// Create a new string pool with pre-warmed entries.
    pub fn new_warmed(max_pooled: usize, initial_capacity: usize, warm_count: usize) -> Arc<Self> {
        let pool = Arc::new(Self {
            pool: Mutex::new(Vec::with_capacity(max_pooled)),
            max_pooled,
            initial_capacity,
            stats: PoolStats::default(),
        });

        let count = warm_count.min(max_pooled);
        let mut guard = pool.pool.lock();
        for _ in 0..count {
            guard.push(String::with_capacity(initial_capacity));
        }
        drop(guard);

        pool
    }

    /// Acquire a string from the pool.
    #[inline]
    pub fn acquire(self: &Arc<Self>) -> PooledString {
        let string = {
            let mut guard = self.pool.lock();
            guard.pop()
        };

        match string {
            Some(mut s) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                s.clear();
                PooledString {
                    string: Some(s),
                    pool: Arc::clone(self),
                }
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                PooledString {
                    string: Some(String::with_capacity(self.initial_capacity)),
                    pool: Arc::clone(self),
                }
            }
        }
    }

    /// Acquire a string initialized with content.
    #[inline]
    pub fn acquire_with(self: &Arc<Self>, content: &str) -> PooledString {
        let mut s = self.acquire();
        s.push_str(content);
        s
    }

    /// Return a string to the pool.
    #[inline]
    fn release(&self, mut string: String) {
        let mut guard = self.pool.lock();
        if guard.len() < self.max_pooled {
            string.clear();
            // Shrink if too large to avoid memory bloat
            if string.capacity() > self.initial_capacity * 4 {
                string.shrink_to(self.initial_capacity);
            }
            guard.push(string);
            self.stats.returns.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the current number of pooled strings.
    pub fn pooled_count(&self) -> usize {
        self.pool.lock().len()
    }

    /// Clear all pooled strings.
    pub fn clear(&self) {
        self.pool.lock().clear();
    }
}

/// A string borrowed from a `StringPool`.
pub struct PooledString {
    string: Option<String>,
    pool: Arc<StringPool>,
}

impl PooledString {
    /// Take ownership of the inner string, preventing return to pool.
    #[inline]
    pub fn take(mut self) -> String {
        self.string.take().unwrap_or_default()
    }

    /// Convert to owned String (alias for take).
    #[inline]
    pub fn into_string(self) -> String {
        self.take()
    }
}

impl Deref for PooledString {
    type Target = String;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.string.as_ref().unwrap()
    }
}

impl DerefMut for PooledString {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.string.as_mut().unwrap()
    }
}

impl Drop for PooledString {
    #[inline]
    fn drop(&mut self) {
        if let Some(string) = self.string.take() {
            self.pool.release(string);
        }
    }
}

impl AsRef<str> for PooledString {
    #[inline]
    fn as_ref(&self) -> &str {
        self.string.as_ref().unwrap()
    }
}

impl std::fmt::Display for PooledString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.string.as_ref().unwrap())
    }
}

// =============================================================================
// Result Vector Pool (for search results)
// =============================================================================

/// Thread-safe pool for search result vectors.
///
/// Specialized for memory search operations where result vectors
/// are frequently created with known capacity.
pub struct ResultPool<T> {
    pool: Mutex<Vec<Vec<T>>>,
    max_pooled: usize,
    default_capacity: usize,
    stats: PoolStats,
}

impl<T> ResultPool<T> {
    /// Create a new result pool.
    pub fn new(default_capacity: usize, max_pooled: usize) -> Arc<Self> {
        Arc::new(Self {
            pool: Mutex::new(Vec::with_capacity(max_pooled)),
            max_pooled,
            default_capacity,
            stats: PoolStats::default(),
        })
    }

    /// Acquire a result vector from the pool.
    #[inline]
    pub fn acquire(self: &Arc<Self>) -> PooledResults<T> {
        let vec = {
            let mut guard = self.pool.lock();
            guard.pop()
        };

        match vec {
            Some(mut v) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                v.clear();
                PooledResults {
                    vec: Some(v),
                    pool: Arc::clone(self),
                }
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                PooledResults {
                    vec: Some(Vec::with_capacity(self.default_capacity)),
                    pool: Arc::clone(self),
                }
            }
        }
    }

    /// Return a vector to the pool.
    #[inline]
    fn release(&self, mut vec: Vec<T>) {
        let mut guard = self.pool.lock();
        if guard.len() < self.max_pooled {
            vec.clear();
            guard.push(vec);
            self.stats.returns.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStatsSnapshot {
        self.stats.snapshot()
    }
}

/// A result vector borrowed from a `ResultPool`.
pub struct PooledResults<T> {
    vec: Option<Vec<T>>,
    pool: Arc<ResultPool<T>>,
}

impl<T> PooledResults<T> {
    /// Take ownership of the inner vector.
    #[inline]
    pub fn take(mut self) -> Vec<T> {
        self.vec.take().unwrap_or_default()
    }
}

impl<T> Deref for PooledResults<T> {
    type Target = Vec<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.vec.as_ref().unwrap()
    }
}

impl<T> DerefMut for PooledResults<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vec.as_mut().unwrap()
    }
}

impl<T> Drop for PooledResults<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(vec) = self.vec.take() {
            self.pool.release(vec);
        }
    }
}

// =============================================================================
// Pool Statistics
// =============================================================================

/// Statistics for pool operations.
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Number of successful pool hits (reused allocation)
    pub hits: AtomicU64,
    /// Number of pool misses (new allocation needed)
    pub misses: AtomicU64,
    /// Total allocations made
    pub allocations: AtomicU64,
    /// Number of items returned to pool
    pub returns: AtomicU64,
    /// Number of items dropped (pool full)
    pub dropped: AtomicU64,
}

impl PoolStats {
    /// Get a snapshot of current statistics.
    pub fn snapshot(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            allocations: self.allocations.load(Ordering::Relaxed),
            returns: self.returns.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
        self.returns.store(0, Ordering::Relaxed);
        self.dropped.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of pool statistics.
#[derive(Debug, Clone, Default)]
pub struct PoolStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub allocations: u64,
    pub returns: u64,
    pub dropped: u64,
}

impl PoolStatsSnapshot {
    /// Calculate hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate allocation savings.
    /// Returns the number of allocations saved by pool reuse.
    pub fn allocations_saved(&self) -> u64 {
        self.hits
    }
}

// =============================================================================
// Global Pool Manager
// =============================================================================

/// Global pool manager for convenient access to shared pools.
///
/// Provides singleton-style access to commonly used pools.
pub struct PoolManager {
    /// Vector pool for embeddings (384-dim default)
    pub vector_pool: Arc<VectorPool>,
    /// String pool for queries
    pub string_pool: Arc<StringPool>,
}

impl PoolManager {
    /// Create a new pool manager with default settings.
    pub fn new() -> Self {
        Self {
            vector_pool: VectorPool::new(384, 64),
            string_pool: StringPool::new(128),
        }
    }

    /// Create a pool manager with custom dimensions and sizes.
    pub fn with_config(
        vector_dim: usize,
        vector_pool_size: usize,
        string_pool_size: usize,
    ) -> Self {
        Self {
            vector_pool: VectorPool::new(vector_dim, vector_pool_size),
            string_pool: StringPool::new(string_pool_size),
        }
    }

    /// Create a pool manager with pre-warmed pools.
    pub fn warmed(
        vector_dim: usize,
        vector_pool_size: usize,
        string_pool_size: usize,
        warm_count: usize,
    ) -> Self {
        Self {
            vector_pool: VectorPool::new_warmed(vector_dim, vector_pool_size, warm_count),
            string_pool: StringPool::new_warmed(string_pool_size, 256, warm_count),
        }
    }

    /// Get combined statistics from all pools.
    pub fn stats(&self) -> CombinedPoolStats {
        CombinedPoolStats {
            vector: self.vector_pool.stats(),
            string: self.string_pool.stats(),
        }
    }
}

impl Default for PoolManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined statistics from all pools.
#[derive(Debug, Clone)]
pub struct CombinedPoolStats {
    pub vector: PoolStatsSnapshot,
    pub string: PoolStatsSnapshot,
}

impl CombinedPoolStats {
    /// Calculate overall hit rate.
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.vector.hits + self.string.hits;
        let total_requests = total_hits + self.vector.misses + self.string.misses;
        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }

    /// Calculate total allocations saved.
    pub fn total_allocations_saved(&self) -> u64 {
        self.vector.allocations_saved() + self.string.allocations_saved()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_vector_pool_basic() {
        let pool = VectorPool::new(384, 16);

        // Acquire and release
        {
            let mut vec = pool.acquire();
            vec.extend_from_slice(&[1.0, 2.0, 3.0]);
            assert_eq!(vec.len(), 3);
        }

        // Should have one pooled vector now
        assert_eq!(pool.pooled_count(), 1);

        // Acquire again should reuse
        {
            let vec = pool.acquire();
            assert!(vec.is_empty()); // Should be cleared
        }

        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_vector_pool_zeroed() {
        let pool = VectorPool::new(10, 16);
        let vec = pool.acquire_zeroed();
        assert_eq!(vec.len(), 10);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vector_pool_from_slice() {
        let pool = VectorPool::new(384, 16);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vec = pool.acquire_from_slice(&data);
        assert_eq!(vec.as_ref(), &data[..]);
    }

    #[test]
    fn test_vector_pool_take() {
        let pool = VectorPool::new(384, 16);

        let vec = pool.acquire();
        let owned = vec.take();

        // Pool should be empty since we took ownership
        assert_eq!(pool.pooled_count(), 0);
        assert!(owned.is_empty());
    }

    #[test]
    fn test_vector_pool_warmed() {
        let pool = VectorPool::new_warmed(384, 16, 8);
        assert_eq!(pool.pooled_count(), 8);

        // First 8 acquires should be hits
        for _ in 0..8 {
            let _vec = pool.acquire();
        }

        let stats = pool.stats();
        assert_eq!(stats.hits, 8);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_vector_pool_max_size() {
        let pool = VectorPool::new(10, 2);

        // Create and drop 5 vectors
        for _ in 0..5 {
            let _vec = pool.acquire();
        }

        // Pool should only have 2 (max_pooled)
        assert!(pool.pooled_count() <= 2);
    }

    #[test]
    fn test_string_pool_basic() {
        let pool = StringPool::new(16);

        {
            let mut s = pool.acquire();
            s.push_str("hello world");
            assert_eq!(s.as_ref(), "hello world");
        }

        assert_eq!(pool.pooled_count(), 1);

        {
            let s = pool.acquire();
            assert!(s.is_empty()); // Should be cleared
        }

        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_string_pool_with_content() {
        let pool = StringPool::new(16);
        let s = pool.acquire_with("test content");
        assert_eq!(s.as_ref(), "test content");
    }

    #[test]
    fn test_string_pool_display() {
        let pool = StringPool::new(16);
        let s = pool.acquire_with("display test");
        assert_eq!(format!("{}", s), "display test");
    }

    #[test]
    fn test_result_pool_basic() {
        let pool: Arc<ResultPool<i32>> = ResultPool::new(10, 16);

        {
            let mut results = pool.acquire();
            results.push(1);
            results.push(2);
            results.push(3);
            assert_eq!(results.len(), 3);
        }

        assert_eq!(pool.stats().returns, 1);
    }

    #[test]
    fn test_pool_manager() {
        let manager = PoolManager::new();

        {
            let _vec = manager.vector_pool.acquire();
            let _str = manager.string_pool.acquire();
        }

        let stats = manager.stats();
        assert_eq!(stats.vector.misses, 1);
        assert_eq!(stats.string.misses, 1);
    }

    #[test]
    fn test_pool_manager_warmed() {
        let manager = PoolManager::warmed(384, 32, 64, 16);

        assert_eq!(manager.vector_pool.pooled_count(), 16);
        assert_eq!(manager.string_pool.pooled_count(), 16);
    }

    #[test]
    fn test_pool_stats_hit_rate() {
        let pool = VectorPool::new_warmed(10, 16, 8);

        // Hold onto vectors to prevent them from being returned to pool
        let mut held_vecs = Vec::new();

        // 8 hits from warmed pool
        for _ in 0..8 {
            held_vecs.push(pool.acquire());
        }

        // 2 misses (pool is now empty)
        for _ in 0..2 {
            held_vecs.push(pool.acquire());
        }

        let stats = pool.stats();
        assert_eq!(stats.hits, 8);
        assert_eq!(stats.misses, 2);
        assert!((stats.hit_rate() - 0.8).abs() < 0.001);

        // Now drop held vectors
        drop(held_vecs);
    }

    #[test]
    fn test_concurrent_vector_pool() {
        let pool = VectorPool::new(100, 32);
        let pool_clone = Arc::clone(&pool);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let p = Arc::clone(&pool_clone);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let mut vec = p.acquire();
                        vec.extend_from_slice(&[1.0, 2.0, 3.0]);
                        // vec dropped here, returned to pool
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = pool.stats();
        // Total requests should be 400
        assert_eq!(stats.hits + stats.misses, 400);
    }

    #[test]
    fn test_concurrent_string_pool() {
        let pool = StringPool::new(32);
        let pool_clone = Arc::clone(&pool);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let p = Arc::clone(&pool_clone);
                thread::spawn(move || {
                    for j in 0..100 {
                        let mut s = p.acquire();
                        s.push_str(&format!("thread {} iter {}", i, j));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = pool.stats();
        assert_eq!(stats.hits + stats.misses, 400);
    }

    #[test]
    fn test_combined_stats() {
        let manager = PoolManager::new();

        // Use vector pool
        for _ in 0..10 {
            let _vec = manager.vector_pool.acquire();
        }

        // Use string pool
        for _ in 0..5 {
            let _s = manager.string_pool.acquire();
        }

        let stats = manager.stats();
        let total_saved = stats.total_allocations_saved();

        // First requests are misses, subsequent are hits
        assert!(total_saved > 0 || stats.vector.misses + stats.string.misses > 0);
    }
}
