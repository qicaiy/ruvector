//! Memoization Cache
//!
//! High-performance cache for storing intermediate query results.
//! Uses O(1) LRU operations via the `lru` crate for optimal performance.

use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for the memoization cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,

    /// Time-to-live for cache entries in seconds.
    pub ttl_secs: u64,

    /// Enable LRU eviction when cache is full.
    pub enable_lru: bool,

    /// Minimum quality score to cache an entry.
    pub min_cache_quality: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            ttl_secs: 3600, // 1 hour
            enable_lru: true,
            min_cache_quality: 0.5,
        }
    }
}

/// A cached query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The query text (key).
    pub query: String,

    /// The answer text.
    pub answer: String,

    /// Quality score of the answer.
    pub quality_score: f32,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,

    /// Number of times this entry was accessed.
    pub access_count: u64,

    /// Timestamp when entry was created (as Unix timestamp).
    pub created_at: u64,

    /// Last access timestamp.
    pub last_accessed: u64,

    /// Optional metadata.
    pub metadata: Option<String>,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(query: &str, answer: &str, quality_score: f32, processing_time_ms: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            query: query.to_string(),
            answer: answer.to_string(),
            quality_score,
            processing_time_ms,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            metadata: None,
        }
    }

    /// Check if the entry has expired.
    pub fn is_expired(&self, ttl_secs: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now.saturating_sub(self.created_at) > ttl_secs
    }

    /// Update access statistics.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

/// Memoization cache for RLM query results.
///
/// Uses O(1) LRU operations for optimal performance in high-traffic scenarios.
/// All cache operations (get, put, contains) are O(1) amortized.
pub struct MemoizationCache {
    /// Configuration.
    config: CacheConfig,

    /// LRU cache storage - O(1) get/put with automatic eviction.
    /// Uses Mutex for thread-safety (LruCache is not thread-safe).
    cache: Mutex<LruCache<u64, CacheEntry>>,

    /// Statistics.
    stats: CacheStatsInner,
}

struct CacheStatsInner {
    hits: AtomicU64,
    misses: AtomicU64,
    insertions: AtomicU64,
    evictions: AtomicU64,
    expired_evictions: AtomicU64,
}

impl MemoizationCache {
    /// Create a new memoization cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new memoization cache with custom configuration.
    pub fn with_config(config: CacheConfig) -> Self {
        let capacity =
            NonZeroUsize::new(config.max_entries.max(1)).expect("max_entries must be >= 1");

        Self {
            config,
            cache: Mutex::new(LruCache::new(capacity)),
            stats: CacheStatsInner {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                insertions: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
                expired_evictions: AtomicU64::new(0),
            },
        }
    }

    /// Look up a query in the cache - O(1) operation.
    pub fn get(&self, query: &str) -> Option<CacheEntry> {
        let key = self.hash_key(query);
        let mut cache = self.cache.lock();

        if let Some(entry) = cache.get_mut(&key) {
            // Check expiration
            if entry.is_expired(self.config.ttl_secs) {
                // Remove expired entry - O(1)
                cache.pop(&key);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.stats.expired_evictions.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            entry.record_access();
            self.stats.hits.fetch_add(1, Ordering::Relaxed);

            // LruCache::get_mut automatically promotes to most-recently-used
            Some(entry.clone())
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert an entry into the cache - O(1) operation.
    pub fn insert(&self, entry: CacheEntry) {
        // Check quality threshold
        if entry.quality_score < self.config.min_cache_quality {
            return;
        }

        let key = self.hash_key(&entry.query);
        let mut cache = self.cache.lock();

        // Check if we'll evict (cache is full and key doesn't exist)
        let will_evict = cache.len() >= self.config.max_entries && !cache.contains(&key);

        // LruCache::put handles eviction automatically - O(1)
        cache.put(key, entry);
        self.stats.insertions.fetch_add(1, Ordering::Relaxed);

        if will_evict {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Remove an entry from the cache - O(1) operation.
    pub fn remove(&self, query: &str) -> Option<CacheEntry> {
        let key = self.hash_key(query);
        self.cache.lock().pop(&key)
    }

    /// Check if a query exists in the cache - O(1) operation.
    /// Note: Does not update LRU order.
    pub fn contains(&self, query: &str) -> bool {
        let key = self.hash_key(query);
        self.cache.lock().contains(&key)
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        self.cache.lock().clear();
    }

    /// Get the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        CacheStats {
            hits,
            misses,
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
            insertions: self.stats.insertions.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            expired_evictions: self.stats.expired_evictions.load(Ordering::Relaxed),
            current_size: self.len(),
            max_size: self.config.max_entries,
        }
    }

    /// Prune expired entries.
    /// Note: This is O(n) as we must check all entries for expiration.
    pub fn prune_expired(&self) -> usize {
        let mut cache = self.cache.lock();
        let ttl = self.config.ttl_secs;

        // Collect expired keys
        let expired_keys: Vec<u64> = cache
            .iter()
            .filter(|(_, entry)| entry.is_expired(ttl))
            .map(|(k, _)| *k)
            .collect();

        let pruned = expired_keys.len();

        // Remove expired entries - O(1) per entry
        for key in expired_keys {
            cache.pop(&key);
            self.stats.expired_evictions.fetch_add(1, Ordering::Relaxed);
        }

        pruned
    }

    /// Compute a 64-bit hash key from the normalized query string.
    /// Uses FNV-1a for fast, good distribution.
    #[inline]
    fn hash_key(&self, query: &str) -> u64 {
        let normalized = query.trim().to_lowercase();

        // FNV-1a hash
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for byte in normalized.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

impl Default for MemoizationCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,

    /// Number of cache misses.
    pub misses: u64,

    /// Cache hit rate (0.0 - 1.0).
    pub hit_rate: f64,

    /// Total number of insertions.
    pub insertions: u64,

    /// Number of evictions due to capacity.
    pub evictions: u64,

    /// Number of evictions due to expiration.
    pub expired_evictions: u64,

    /// Current number of entries.
    pub current_size: usize,

    /// Maximum cache capacity.
    pub max_size: usize,
}

impl CacheStats {
    /// Get total lookups (hits + misses).
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }

    /// Get cache utilization as percentage.
    pub fn utilization(&self) -> f64 {
        if self.max_size > 0 {
            self.current_size as f64 / self.max_size as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = MemoizationCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let cache = MemoizationCache::new();
        let entry = CacheEntry::new("What is AI?", "AI is artificial intelligence.", 0.9, 100);

        cache.insert(entry);
        assert_eq!(cache.len(), 1);

        let result = cache.get("What is AI?");
        assert!(result.is_some());
        assert_eq!(result.unwrap().answer, "AI is artificial intelligence.");
    }

    #[test]
    fn test_cache_normalization() {
        let cache = MemoizationCache::new();
        let entry = CacheEntry::new("What is AI?", "Answer", 0.9, 100);

        cache.insert(entry);

        // Different casing should still hit
        let result = cache.get("  WHAT IS AI?  ");
        assert!(result.is_some());
    }

    #[test]
    fn test_cache_miss() {
        let cache = MemoizationCache::new();
        let result = cache.get("nonexistent query");
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_cache_quality_threshold() {
        let config = CacheConfig {
            min_cache_quality: 0.7,
            ..Default::default()
        };
        let cache = MemoizationCache::with_config(config);

        // Low quality - should not be cached
        let low_quality = CacheEntry::new("Query 1", "Answer 1", 0.5, 100);
        cache.insert(low_quality);
        assert!(cache.get("Query 1").is_none());

        // High quality - should be cached
        let high_quality = CacheEntry::new("Query 2", "Answer 2", 0.9, 100);
        cache.insert(high_quality);
        assert!(cache.get("Query 2").is_some());
    }

    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            max_entries: 3,
            enable_lru: true,
            min_cache_quality: 0.0,
            ..Default::default()
        };
        let cache = MemoizationCache::with_config(config);

        for i in 0..5 {
            let entry = CacheEntry::new(&format!("Query {}", i), "Answer", 0.9, 100);
            cache.insert(entry);
        }

        assert!(cache.len() <= 3);

        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_cache_lru_order() {
        let config = CacheConfig {
            max_entries: 3,
            enable_lru: true,
            min_cache_quality: 0.0,
            ..Default::default()
        };
        let cache = MemoizationCache::with_config(config);

        // Insert 3 entries
        for i in 0..3 {
            let entry = CacheEntry::new(&format!("Query {}", i), "Answer", 0.9, 100);
            cache.insert(entry);
        }

        // Access Query 0 to make it most recently used
        cache.get("Query 0");

        // Insert a 4th entry - should evict Query 1 (LRU)
        let entry = CacheEntry::new("Query 3", "Answer", 0.9, 100);
        cache.insert(entry);

        // Query 0 should still exist (was accessed)
        assert!(cache.contains("Query 0"));
        // Query 1 should be evicted (was LRU)
        assert!(!cache.contains("Query 1"));
        // Query 2 and 3 should exist
        assert!(cache.contains("Query 2"));
        assert!(cache.contains("Query 3"));
    }

    #[test]
    fn test_cache_removal() {
        let cache = MemoizationCache::new();
        let entry = CacheEntry::new("Query", "Answer", 0.9, 100);

        cache.insert(entry);
        assert_eq!(cache.len(), 1);

        let removed = cache.remove("Query");
        assert!(removed.is_some());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_clear() {
        let cache = MemoizationCache::new();

        for i in 0..10 {
            let entry = CacheEntry::new(&format!("Query {}", i), "Answer", 0.9, 100);
            cache.insert(entry);
        }

        assert_eq!(cache.len(), 10);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats() {
        let cache = MemoizationCache::new();
        let entry = CacheEntry::new("Query", "Answer", 0.9, 100);

        cache.insert(entry);

        // Hits
        cache.get("Query");
        cache.get("Query");

        // Misses
        cache.get("Missing");

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.insertions, 1);
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_entry_access_tracking() {
        let mut entry = CacheEntry::new("Query", "Answer", 0.9, 100);
        assert_eq!(entry.access_count, 0);

        entry.record_access();
        assert_eq!(entry.access_count, 1);

        entry.record_access();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_cache_contains() {
        let cache = MemoizationCache::new();
        let entry = CacheEntry::new("Query", "Answer", 0.9, 100);

        assert!(!cache.contains("Query"));
        cache.insert(entry);
        assert!(cache.contains("Query"));
        assert!(!cache.contains("Other"));
    }

    #[test]
    fn test_hash_consistency() {
        let cache = MemoizationCache::new();

        // Same query should produce same hash
        let h1 = cache.hash_key("test query");
        let h2 = cache.hash_key("test query");
        assert_eq!(h1, h2);

        // Normalized queries should match
        let h3 = cache.hash_key("  TEST QUERY  ");
        assert_eq!(h1, h3);

        // Different queries should produce different hashes
        let h4 = cache.hash_key("different query");
        assert_ne!(h1, h4);
    }
}
