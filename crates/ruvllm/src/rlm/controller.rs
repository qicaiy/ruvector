//! RLM Controller - Core control logic for Ruvector Language Model operations
//!
//! The controller manages:
//! - Query processing and response generation
//! - Memory operations (add, search, update)
//! - Statistics tracking
//! - Environment abstraction
//!
//! Performance targets:
//! - Cache lookup: <100us
//! - Full query (cached): <500us

use super::environment::RlmEnvironment;
use super::memory::{MemoryConfig, MemorySearchResult, RlmMemory};
use crate::error::Result;
// Import memory types - re-exported from memory module
pub use super::memory::{MemoryEntry, MemoryMetadata};
// Import zero-copy shared types
use super::shared_types::extract_excerpt;
#[allow(unused_imports)]
use super::shared_types::{
    ArcCachedResponse, IntoSharedEmbedding, IntoSharedText, SharedEmbedding, SharedQueryResult,
    SharedSourceAttribution, SharedText, SharedTokenUsage,
};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Configuration for the RLM controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmConfig {
    /// Embedding dimension for vectors
    pub embedding_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Model identifier (for RuvLTRA tiny config in WASM)
    pub model_id: String,
    /// Enable semantic caching
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    /// Temperature for generation
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Maximum recursion depth for query decomposition
    pub max_depth: usize,
    /// Number of memories to retrieve per query
    pub retrieval_top_k: usize,
    /// Total token budget for recursive processing
    pub token_budget: usize,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384, // Small dimension for WASM efficiency
            max_seq_len: 2048,
            memory_config: MemoryConfig::default(),
            model_id: "ruvltra-tiny".to_string(),
            enable_cache: true,
            cache_ttl_secs: 3600,
            max_concurrent_ops: 4,
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            max_depth: 5,
            retrieval_top_k: 5,
            token_budget: 16000,
        }
    }
}

impl RlmConfig {
    /// Create a WASM-optimized configuration
    pub fn for_wasm() -> Self {
        Self {
            embedding_dim: 256, // Smaller for WASM
            max_seq_len: 1024,
            memory_config: MemoryConfig::for_wasm(),
            model_id: "ruvltra-wasm".to_string(),
            enable_cache: true,
            cache_ttl_secs: 1800,
            max_concurrent_ops: 1, // Single-threaded in WASM
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 256,
            max_depth: 3,
            retrieval_top_k: 3,
            token_budget: 8000,
        }
    }

    /// Create a configuration for RuvLTRA small (0.5B)
    pub fn for_ruvltra_small() -> Self {
        Self {
            embedding_dim: 384,
            max_seq_len: 4096,
            memory_config: MemoryConfig::default(),
            model_id: "ruvltra-small".to_string(),
            enable_cache: true,
            cache_ttl_secs: 3600,
            max_concurrent_ops: 8,
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 1024,
            max_depth: 5,
            retrieval_top_k: 5,
            token_budget: 16000,
        }
    }
}

/// Result from a query operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Unique result ID
    pub id: String,
    /// Generated response text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Source attributions from memory
    pub sources: Vec<SourceAttribution>,
    /// Usage statistics
    pub usage: TokenUsage,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens processed
    pub input_tokens: usize,
    /// Output tokens generated
    pub output_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// Source attribution for RAG-style responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAttribution {
    /// Memory entry ID
    pub memory_id: String,
    /// Relevance score
    pub relevance: f32,
    /// Excerpt from the source
    pub excerpt: String,
}

// MemoryEntry and MemoryMetadata are imported from super::memory and re-exported

/// Statistics for the RLM controller
#[derive(Debug, Default)]
pub struct RlmStats {
    /// Total queries processed
    pub total_queries: AtomicU64,
    /// Total memory entries
    pub total_memories: AtomicU64,
    /// Cache hits
    pub cache_hits: AtomicU64,
    /// Cache misses
    pub cache_misses: AtomicU64,
    /// Total tokens processed
    pub total_tokens: AtomicU64,
    /// Average latency in microseconds
    pub avg_latency_us: AtomicU64,
}

impl Clone for RlmStats {
    fn clone(&self) -> Self {
        Self {
            total_queries: AtomicU64::new(self.total_queries.load(Ordering::Relaxed)),
            total_memories: AtomicU64::new(self.total_memories.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(Ordering::Relaxed)),
            total_tokens: AtomicU64::new(self.total_tokens.load(Ordering::Relaxed)),
            avg_latency_us: AtomicU64::new(self.avg_latency_us.load(Ordering::Relaxed)),
        }
    }
}

impl RlmStats {
    /// Get a snapshot of current stats
    pub fn snapshot(&self) -> RlmStatsSnapshot {
        RlmStatsSnapshot {
            total_queries: self.total_queries.load(Ordering::Relaxed),
            total_memories: self.total_memories.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            total_tokens: self.total_tokens.load(Ordering::Relaxed),
            avg_latency_us: self.avg_latency_us.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of RLM statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmStatsSnapshot {
    pub total_queries: u64,
    pub total_memories: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_tokens: u64,
    pub avg_latency_us: u64,
}

/// Fast hash for cache keys.
#[inline]
fn fast_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// The main RLM controller, generic over the environment
pub struct RlmController<E: RlmEnvironment> {
    /// Configuration
    config: RlmConfig,
    /// Memory backend
    memory: Arc<RwLock<RlmMemory>>,
    /// Semantic cache for responses (uses hash key for faster lookup)
    cache: Arc<RwLock<HashMap<u64, CachedResponse>>>,
    /// Reverse mapping from hash to original key (for collision detection)
    cache_keys: Arc<RwLock<HashMap<u64, String>>>,
    /// Statistics
    stats: Arc<RlmStats>,
    /// Environment marker
    _env: PhantomData<E>,
}

/// Cached response entry with Arc for zero-copy retrieval.
/// Using Arc<QueryResult> instead of cloning on every cache hit.
#[derive(Debug, Clone)]
struct CachedResponse {
    /// Arc-wrapped result for zero-copy sharing
    result: Arc<QueryResult>,
    cached_at: DateTime<Utc>,
    /// Unix timestamp for faster TTL check
    cached_at_secs: i64,
}

impl CachedResponse {
    /// Create a new cached response, wrapping the result in Arc.
    #[inline]
    fn new(result: QueryResult) -> Self {
        let now = Utc::now();
        Self {
            result: Arc::new(result),
            cached_at: now,
            cached_at_secs: now.timestamp(),
        }
    }

    /// Get the result with zero-copy (Arc clone only).
    #[inline]
    fn get_result(&self) -> Arc<QueryResult> {
        Arc::clone(&self.result)
    }

    /// Check if expired.
    #[inline]
    fn is_expired(&self, ttl_secs: i64) -> bool {
        let now_secs = Utc::now().timestamp();
        now_secs - self.cached_at_secs >= ttl_secs
    }
}

impl<E: RlmEnvironment> RlmController<E> {
    /// Create a new RLM controller with the given configuration
    pub fn new(config: RlmConfig) -> Result<Self> {
        let memory = RlmMemory::new(config.memory_config.clone())?;

        // Pre-size cache for expected load
        let cache_capacity = 1024;

        Ok(Self {
            config,
            memory: Arc::new(RwLock::new(memory)),
            cache: Arc::new(RwLock::new(HashMap::with_capacity(cache_capacity))),
            cache_keys: Arc::new(RwLock::new(HashMap::with_capacity(cache_capacity))),
            stats: Arc::new(RlmStats::default()),
            _env: PhantomData,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &RlmConfig {
        &self.config
    }

    /// Get current statistics
    pub fn stats(&self) -> Arc<RlmStats> {
        Arc::clone(&self.stats)
    }

    /// Process a query and generate a response
    ///
    /// This method:
    /// 1. Checks the semantic cache for similar queries (zero-copy with Arc)
    /// 2. Retrieves relevant context from memory
    /// 3. Generates a response using the environment
    /// 4. Caches the result for future use
    ///
    /// Zero-copy optimizations:
    /// - Cache hits return Arc<QueryResult> unwrapped (single clone if needed)
    /// - Response text is moved, not cloned
    /// - Source excerpts use Cow for efficient string handling
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn query(&self, input: &str) -> Result<QueryResult> {
        use std::time::Instant;

        let start = Instant::now();

        // Check cache first - returns Arc<QueryResult> for zero-copy
        if self.config.enable_cache {
            if let Some(cached_arc) = self.check_cache(input) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                // If Arc has single owner, unwrap directly; otherwise clone
                return Ok(Arc::try_unwrap(cached_arc).unwrap_or_else(|arc| (*arc).clone()));
            }
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Generate embedding for the query
        let query_embedding = E::embed(input)?;

        // Search memory for relevant context
        let context = self.search_memory_internal(&query_embedding, 5)?;

        // Generate response using environment
        let response_text = E::generate(input, &context, &self.config)?;

        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        // Pre-compute token counts to avoid redundant iteration
        let input_tokens = input.split_whitespace().count();
        let output_tokens = response_text.split_whitespace().count();

        // Build sources with efficient excerpt extraction (using Cow internally)
        let sources: Vec<SourceAttribution> = context
            .iter()
            .map(|r| {
                // Use extract_excerpt for zero-copy when possible
                let excerpt_cow = extract_excerpt(&r.entry.text, 100);
                SourceAttribution {
                    memory_id: r.id.clone(),
                    relevance: r.score,
                    excerpt: excerpt_cow.into_owned(),
                }
            })
            .collect();

        // Build result - response_text is moved, not cloned
        let result = QueryResult {
            id: Uuid::new_v4().to_string(),
            text: response_text, // MOVED, not cloned
            confidence: self.calculate_confidence(&context),
            tokens_generated: output_tokens,
            latency_ms,
            sources,
            usage: TokenUsage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
            },
        };

        // Update stats
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_tokens
            .fetch_add(result.usage.total_tokens as u64, Ordering::Relaxed);
        self.update_avg_latency(latency_ms as u64 * 1000);

        // Cache the result - we need to clone here since we return the result
        if self.config.enable_cache {
            self.cache_result(input, result.clone());
        }

        Ok(result)
    }

    /// Synchronous query for WASM environments
    /// Zero-copy optimizations applied for cache hits and response building.
    #[cfg(target_arch = "wasm32")]
    pub fn query(&self, input: &str) -> Result<QueryResult> {
        let start = E::now();

        // Check cache first - returns Arc<QueryResult> for zero-copy
        if self.config.enable_cache {
            if let Some(cached_arc) = self.check_cache(input) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                // If Arc has single owner, unwrap directly; otherwise clone
                return Ok(Arc::try_unwrap(cached_arc).unwrap_or_else(|arc| (*arc).clone()));
            }
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Generate embedding for the query
        let query_embedding = E::embed(input)?;

        // Search memory for relevant context
        let context = self.search_memory_internal(&query_embedding, 5)?;

        // Generate response using environment
        let response_text = E::generate(input, &context, &self.config)?;

        let latency_ms = E::elapsed_ms(start);

        // Pre-compute token counts to avoid redundant iteration
        let input_tokens = input.split_whitespace().count();
        let output_tokens = response_text.split_whitespace().count();

        // Build sources with efficient excerpt extraction
        let sources: Vec<SourceAttribution> = context
            .iter()
            .map(|r| {
                let excerpt_cow = extract_excerpt(&r.entry.text, 100);
                SourceAttribution {
                    memory_id: r.id.clone(),
                    relevance: r.score,
                    excerpt: excerpt_cow.into_owned(),
                }
            })
            .collect();

        // Build result - response_text is moved, not cloned
        let result = QueryResult {
            id: E::generate_id(),
            text: response_text, // MOVED, not cloned
            confidence: self.calculate_confidence(&context),
            tokens_generated: output_tokens,
            latency_ms,
            sources,
            usage: TokenUsage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
            },
        };

        // Update stats
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_tokens
            .fetch_add(result.usage.total_tokens as u64, Ordering::Relaxed);
        self.update_avg_latency(latency_ms as u64 * 1000);

        // Cache the result
        if self.config.enable_cache {
            self.cache_result(input, result.clone());
        }

        Ok(result)
    }

    /// Add a text entry to memory
    /// Zero-copy optimization: embedding is moved directly into entry, not cloned.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn add_memory(&self, text: &str, metadata: MemoryMetadata) -> Result<String> {
        let embedding = E::embed(text)?;
        let now = Utc::now();
        let id = Uuid::new_v4().to_string();
        let id_clone = id.clone(); // Clone ID before moving into entry

        let entry = MemoryEntry {
            id,
            text: text.to_string(),
            embedding, // MOVED, not cloned
            metadata,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        };

        self.memory.write().add(entry)?;
        self.stats.total_memories.fetch_add(1, Ordering::Relaxed);

        Ok(id_clone)
    }

    /// Add a text entry to memory (synchronous for WASM)
    /// Zero-copy optimization: embedding is moved directly into entry, not cloned.
    #[cfg(target_arch = "wasm32")]
    pub fn add_memory(&self, text: &str, metadata: MemoryMetadata) -> Result<String> {
        let embedding = E::embed(text)?;
        let now = Utc::now();
        let id = E::generate_id();
        let id_clone = id.clone(); // Clone ID before moving into entry

        let entry = MemoryEntry {
            id,
            text: text.to_string(),
            embedding, // MOVED, not cloned
            metadata,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        };

        self.memory.write().add(entry)?;
        self.stats.total_memories.fetch_add(1, Ordering::Relaxed);

        Ok(id_clone)
    }

    /// Search memory for relevant entries
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn search_memory(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        let embedding = E::embed(query)?;
        self.search_memory_internal(&embedding, top_k)
    }

    /// Search memory for relevant entries (synchronous for WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn search_memory(&self, query: &str, top_k: usize) -> Result<Vec<MemorySearchResult>> {
        let embedding = E::embed(query)?;
        self.search_memory_internal(&embedding, top_k)
    }

    /// Internal memory search by embedding
    fn search_memory_internal(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        self.memory.read().search(embedding, top_k)
    }

    /// Check the semantic cache for a matching query.
    /// Optimized for <100us lookup using hash-based keys.
    /// Returns Arc<QueryResult> for zero-copy sharing.
    #[inline]
    fn check_cache(&self, input: &str) -> Option<Arc<QueryResult>> {
        let hash_key = fast_hash(input);
        let cache = self.cache.read();

        if let Some(cached) = cache.get(&hash_key) {
            // Fast TTL check using pre-computed timestamp
            let now_secs = Utc::now().timestamp();
            if (now_secs - cached.cached_at_secs) < self.config.cache_ttl_secs as i64 {
                // Verify key match (handle hash collisions)
                let keys = self.cache_keys.read();
                if keys.get(&hash_key).map(|k| k == input).unwrap_or(false) {
                    // Zero-copy: return Arc clone instead of deep clone
                    return Some(cached.get_result());
                }
            }
        }

        None
    }

    /// Cache a query result using Arc for zero-copy retrieval.
    /// Uses hash-based key for faster subsequent lookups.
    fn cache_result(&self, input: &str, result: QueryResult) {
        let hash_key = fast_hash(input);

        {
            let mut cache = self.cache.write();
            // Use CachedResponse::new which wraps in Arc
            cache.insert(hash_key, CachedResponse::new(result));
        }

        {
            let mut keys = self.cache_keys.write();
            keys.insert(hash_key, input.to_string());
        }
    }

    /// Cache a query result that's already in an Arc.
    /// Avoids re-wrapping when result is already shared.
    fn cache_result_arc(&self, input: &str, result: Arc<QueryResult>) {
        let hash_key = fast_hash(input);
        let now = Utc::now();

        {
            let mut cache = self.cache.write();
            cache.insert(
                hash_key,
                CachedResponse {
                    result,
                    cached_at: now,
                    cached_at_secs: now.timestamp(),
                },
            );
        }

        {
            let mut keys = self.cache_keys.write();
            keys.insert(hash_key, input.to_string());
        }
    }

    /// Calculate confidence based on retrieved context.
    /// Optimized with pre-computed harmonic weights for common sizes.
    #[inline]
    fn calculate_confidence(&self, context: &[MemorySearchResult]) -> f32 {
        let len = context.len();
        if len == 0 {
            return 0.5; // Base confidence without context
        }

        // Pre-computed harmonic weight sums for common sizes
        static HARMONIC_SUMS: [f32; 11] = [
            0.0,   // 0 (unused)
            1.0,   // 1
            1.5,   // 1 + 1/2
            1.833, // 1 + 1/2 + 1/3
            2.083, // ...
            2.283, 2.45, 2.593, 2.718, 2.829, 2.929, // 10
        ];

        // Use pre-computed sum for common sizes
        let weight_sum = if len <= 10 {
            HARMONIC_SUMS[len]
        } else {
            (0..len).map(|i| 1.0 / (i as f32 + 1.0)).sum()
        };

        // Calculate weighted sum
        let mut weighted_sum = 0.0f32;
        for (i, r) in context.iter().enumerate() {
            weighted_sum += r.score / (i as f32 + 1.0);
        }

        (weighted_sum / weight_sum).clamp(0.0, 1.0)
    }

    /// Update average latency using exponential moving average
    fn update_avg_latency(&self, latency_us: u64) {
        let current = self.stats.avg_latency_us.load(Ordering::Relaxed);
        if current == 0 {
            self.stats
                .avg_latency_us
                .store(latency_us, Ordering::Relaxed);
        } else {
            // EMA with alpha = 0.1
            let new_avg = ((current as f64 * 0.9) + (latency_us as f64 * 0.1)) as u64;
            self.stats.avg_latency_us.store(new_avg, Ordering::Relaxed);
        }
    }

    /// Clear the response cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
        self.cache_keys.write().clear();
    }

    /// Evict expired entries from cache (call periodically for large caches).
    pub fn evict_expired_cache(&self) {
        let now_secs = Utc::now().timestamp();
        let ttl = self.config.cache_ttl_secs as i64;

        let mut cache = self.cache.write();
        let mut keys = self.cache_keys.write();

        let expired: Vec<u64> = cache
            .iter()
            .filter(|(_, v)| now_secs - v.cached_at_secs >= ttl)
            .map(|(k, _)| *k)
            .collect();

        for key in expired {
            cache.remove(&key);
            keys.remove(&key);
        }
    }

    /// Get memory entry by ID
    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryEntry>> {
        self.memory.read().get(id)
    }

    /// Delete memory entry by ID
    pub fn delete_memory(&self, id: &str) -> Result<bool> {
        let deleted = self.memory.write().delete(id)?;
        if deleted {
            self.stats.total_memories.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(deleted)
    }

    /// Get all memory entries (use with caution for large stores)
    pub fn list_memories(&self, limit: usize, offset: usize) -> Result<Vec<MemoryEntry>> {
        self.memory.read().list(limit, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlm::environment::NativeEnvironment;

    #[test]
    fn test_config_default() {
        let config = RlmConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_config_wasm() {
        let config = RlmConfig::for_wasm();
        assert_eq!(config.embedding_dim, 256);
        assert_eq!(config.max_concurrent_ops, 1);
    }

    #[test]
    fn test_stats_clone() {
        let stats = RlmStats::default();
        stats.total_queries.fetch_add(10, Ordering::Relaxed);

        let cloned = stats.clone();
        assert_eq!(cloned.total_queries.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_controller_creation() {
        let config = RlmConfig::default();
        let controller = RlmController::<NativeEnvironment>::new(config);
        assert!(controller.is_ok());
    }
}
