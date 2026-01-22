//! RLM Memory System
//!
//! Vector-based memory storage and retrieval for the RLM system.
//! Integrates with ruvector HNSW index for fast similarity search.
//!
//! Performance targets:
//! - Memory search (k=5): <0.5ms with HNSW (was <5ms brute-force)
//! - Cache lookup: <100us
//!
//! ## HNSW Integration
//!
//! When `use_hnsw: true` in config, the memory system uses the HNSW index
//! from ruvector-core for O(log n) approximate nearest neighbor search.
//! This provides 8-10x speedup for datasets >1k entries compared to
//! brute-force O(n) linear scan.

use crate::error::{Result, RuvLLMError};
use crate::rlm::simd_ops::{batch_cosine_similarity_4_prenorm, l2_norm as simd_l2_norm};
use chrono::{DateTime, Utc};
#[allow(unused_imports)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

// HNSW index from ruvector-core
use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};

// SIMD imports for optimized vector operations (fallback for local functions)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Configuration for the RLM memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Embedding dimension for vectors.
    pub embedding_dim: usize,
    /// Maximum number of entries to store.
    pub max_entries: usize,
    /// Enable HNSW index for fast search (vs brute force).
    pub use_hnsw: bool,
    /// HNSW M parameter (connections per node).
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter.
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search parameter.
    pub hnsw_ef_search: usize,
    /// Similarity threshold for retrieval (0.0 - 1.0).
    pub similarity_threshold: f32,
    /// Enable automatic cleanup of old entries.
    pub enable_ttl: bool,
    /// Default TTL in seconds.
    pub default_ttl_secs: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            max_entries: 10000,
            use_hnsw: true,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            similarity_threshold: 0.5,
            enable_ttl: false,
            default_ttl_secs: 3600,
        }
    }
}

impl MemoryConfig {
    /// Create a configuration optimized for WASM.
    pub fn for_wasm() -> Self {
        Self {
            embedding_dim: 256,
            max_entries: 1000,
            use_hnsw: false, // Simpler search for WASM
            hnsw_m: 8,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 20,
            similarity_threshold: 0.6,
            enable_ttl: true,
            default_ttl_secs: 1800,
        }
    }

    /// Create a configuration for testing.
    pub fn for_testing() -> Self {
        Self {
            embedding_dim: 64,
            max_entries: 100,
            use_hnsw: false,
            ..Default::default()
        }
    }
}

/// A memory entry stored in the RLM memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: String,
    /// Text content.
    pub text: String,
    /// Vector embedding.
    pub embedding: Vec<f32>,
    /// Metadata.
    pub metadata: MemoryMetadata,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp.
    pub last_accessed: DateTime<Utc>,
    /// Access count.
    pub access_count: u64,
}

/// Metadata for a memory entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMetadata {
    /// Source identifier.
    pub source: Option<String>,
    /// Category or type.
    pub category: Option<String>,
    /// Custom tags.
    pub tags: Vec<String>,
    /// TTL override in seconds.
    pub ttl_secs: Option<u64>,
    /// Additional key-value pairs.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Result from a memory search.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    /// The matched entry ID.
    pub id: String,
    /// The matched entry.
    pub entry: MemoryEntry,
    /// Similarity score (0.0 - 1.0).
    pub score: f32,
}

// ============================================================================
// Zero-Copy Types for Optimized Memory Operations
// ============================================================================

/// Zero-copy search result using Arc for shared ownership.
/// Avoids cloning MemoryEntry on each search result.
///
/// # Performance
/// - Cache hit: O(1) clone (just Arc ref count increment)
/// - Memory: Single copy of entry data shared across all references
#[derive(Debug, Clone)]
pub struct ArcMemorySearchResult {
    /// The matched entry ID (Arc<str> for zero-copy sharing).
    pub id: Arc<str>,
    /// Shared reference to the matched entry (no clone needed).
    pub entry: Arc<MemoryEntry>,
    /// Similarity score (0.0 - 1.0).
    pub score: f32,
}

impl ArcMemorySearchResult {
    /// Create from an Arc-wrapped entry.
    #[inline]
    pub fn new(entry: Arc<MemoryEntry>, score: f32) -> Self {
        Self {
            id: Arc::from(entry.id.as_str()),
            entry,
            score,
        }
    }

    /// Get the entry text (zero-copy).
    #[inline]
    pub fn text(&self) -> &str {
        &self.entry.text
    }

    /// Get the entry embedding (zero-copy).
    #[inline]
    pub fn embedding(&self) -> &[f32] {
        &self.entry.embedding
    }

    /// Convert to owned MemorySearchResult if needed for API compatibility.
    /// This performs a clone, use sparingly.
    pub fn to_owned(&self) -> MemorySearchResult {
        MemorySearchResult {
            id: self.id.to_string(),
            entry: (*self.entry).clone(),
            score: self.score,
        }
    }
}

/// Extract a text excerpt using Cow to avoid allocation when possible.
/// Returns borrowed slice if text is short enough, otherwise truncates.
#[inline]
pub fn extract_text_excerpt<'a>(text: &'a str, max_chars: usize) -> Cow<'a, str> {
    if text.len() <= max_chars {
        Cow::Borrowed(text)
    } else {
        // Find char boundary for safe truncation
        let mut end = max_chars;
        while !text.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        Cow::Borrowed(&text[..end])
    }
}

/// RLM Memory system for storing and retrieving vector embeddings.
///
/// Uses HNSW index from ruvector-core for O(log n) search when enabled,
/// falling back to brute-force O(n) scan when disabled or for small datasets.
pub struct RlmMemory {
    /// Configuration.
    config: MemoryConfig,
    /// Stored entries by ID.
    entries: HashMap<String, MemoryEntry>,
    /// ID list for iteration (maintains insertion order).
    id_list: Vec<String>,
    /// HNSW index for fast approximate nearest neighbor search.
    /// Only initialized when `config.use_hnsw` is true.
    hnsw_index: Option<HnswIndex>,
    /// Set of deleted IDs (HNSW doesn't support true deletion).
    /// Entries are filtered out during search.
    deleted_ids: std::collections::HashSet<String>,
}

impl RlmMemory {
    /// Create a new RLM memory system with the given configuration.
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let hnsw_index = if config.use_hnsw {
            // Create HNSW index with configured parameters
            let hnsw_config = HnswConfig {
                m: config.hnsw_m,
                ef_construction: config.hnsw_ef_construction,
                ef_search: config.hnsw_ef_search,
                max_elements: config.max_entries,
            };

            // Use cosine distance for semantic similarity (most common for embeddings)
            let index = HnswIndex::new(config.embedding_dim, DistanceMetric::Cosine, hnsw_config)
                .map_err(|e| {
                RuvLLMError::Ruvector(format!("Failed to create HNSW index: {}", e))
            })?;

            Some(index)
        } else {
            None
        };

        Ok(Self {
            config,
            entries: HashMap::new(),
            id_list: Vec::new(),
            hnsw_index,
            deleted_ids: std::collections::HashSet::new(),
        })
    }

    /// Add a new memory entry.
    ///
    /// When HNSW is enabled, the entry's embedding is also added to the HNSW index
    /// for O(log n) approximate nearest neighbor search.
    pub fn add(&mut self, entry: MemoryEntry) -> Result<()> {
        // Check capacity
        if self.entries.len() >= self.config.max_entries {
            // Remove oldest entry
            if let Some(oldest_id) = self.id_list.first().cloned() {
                self.delete(&oldest_id)?;
            }
        }

        let id = entry.id.clone();
        let embedding = entry.embedding.clone();

        // Add to HNSW index if enabled
        if let Some(ref mut hnsw) = self.hnsw_index {
            hnsw.add(id.clone(), embedding)
                .map_err(|e| RuvLLMError::Ruvector(format!("Failed to add to HNSW: {}", e)))?;
        }

        // Remove from deleted set if re-adding
        self.deleted_ids.remove(&id);

        self.entries.insert(id.clone(), entry);
        self.id_list.push(id);

        Ok(())
    }

    /// Search for similar memories.
    ///
    /// When HNSW is enabled, uses O(log n) approximate nearest neighbor search
    /// for dramatically improved performance on large datasets (8-10x faster
    /// for >1k entries). Falls back to SIMD-optimized brute-force when disabled.
    ///
    /// Performance targets:
    /// - HNSW enabled: <0.5ms for 10k entries
    /// - Brute-force: <5ms for 10k entries (with SIMD)
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<MemorySearchResult>> {
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }

        // Validate embedding dimension
        if query_embedding.len() != self.config.embedding_dim {
            return Err(RuvLLMError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: query_embedding.len(),
            });
        }

        // Use HNSW index if available and we have enough entries to benefit
        // (HNSW has overhead, so brute-force may be faster for very small datasets)
        if let Some(ref hnsw) = self.hnsw_index {
            if self.entries.len() > 100 {
                return self.search_hnsw(hnsw, query_embedding, top_k);
            }
        }

        // Fall back to brute-force search
        self.search_brute_force(query_embedding, top_k)
    }

    /// HNSW-based approximate nearest neighbor search.
    /// O(log n) complexity with high recall (>95% typically).
    fn search_hnsw(
        &self,
        hnsw: &HnswIndex,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        let threshold = self.config.similarity_threshold;

        // HNSW returns distance (lower = closer), we need to convert to similarity
        // For cosine distance: similarity = 1 - distance
        // Request more results than needed to account for deleted entries
        let request_k = (top_k + self.deleted_ids.len()).min(self.entries.len());

        let hnsw_results = hnsw
            .search_with_ef(query_embedding, request_k, self.config.hnsw_ef_search)
            .map_err(|e| RuvLLMError::Ruvector(format!("HNSW search failed: {}", e)))?;

        let mut results: Vec<MemorySearchResult> = hnsw_results
            .into_iter()
            .filter_map(|result| {
                // Skip deleted entries
                if self.deleted_ids.contains(&result.id) {
                    return None;
                }

                // Get the full entry
                let entry = self.entries.get(&result.id)?;

                // Convert distance to similarity (cosine distance = 1 - cosine_similarity)
                // HNSW stores raw cosine distance, so similarity = 1 - distance
                let similarity = 1.0 - result.score;

                // Apply threshold
                if similarity >= threshold {
                    Some(MemorySearchResult {
                        id: result.id,
                        entry: entry.clone(),
                        score: similarity,
                    })
                } else {
                    None
                }
            })
            .take(top_k)
            .collect();

        // Results from HNSW are already sorted by distance (ascending),
        // which means they're sorted by similarity (descending) after conversion
        // But we should ensure this since we filtered some out
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    /// Brute-force O(n) search with SIMD optimization.
    /// Used when HNSW is disabled or for small datasets.
    fn search_brute_force(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        // Pre-compute query norm for efficiency
        let query_norm = vector_norm(query_embedding);
        if query_norm < 1e-8 {
            return Ok(Vec::new());
        }

        let threshold = self.config.similarity_threshold;
        let num_entries = self.entries.len();

        // Use partial sort for better performance when top_k << num_entries
        if top_k < num_entries / 4 {
            // Heap-based top-k selection
            return self.search_topk_heap_batch(query_embedding, query_norm, top_k, threshold);
        }

        // Compute similarities with SIMD-optimized function
        let mut results: Vec<_> = self
            .entries
            .values()
            .filter(|entry| !self.deleted_ids.contains(&entry.id))
            .filter_map(|entry| {
                let score = cosine_similarity_fast(query_embedding, &entry.embedding, query_norm);
                if score >= threshold {
                    Some(MemorySearchResult {
                        id: entry.id.clone(),
                        entry: entry.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }

    /// Optimized top-k search using batch SIMD and min-heap for large datasets.
    /// Processes 4 similarity computations in parallel for ~4x throughput.
    #[inline]
    fn search_topk_heap_batch(
        &self,
        query_embedding: &[f32],
        query_norm: f32,
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<MemorySearchResult>> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        // Min-heap wrapper (BinaryHeap is max-heap by default)
        #[derive(Clone)]
        struct MinScoreEntry {
            score: f32,
            id: String,
            entry: MemoryEntry,
        }

        impl PartialEq for MinScoreEntry {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }
        impl Eq for MinScoreEntry {}

        impl PartialOrd for MinScoreEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // Reverse ordering for min-heap behavior
                other.score.partial_cmp(&self.score)
            }
        }
        impl Ord for MinScoreEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<MinScoreEntry> = BinaryHeap::with_capacity(top_k + 1);
        let mut min_score = threshold;

        // Collect entries for batch processing
        let entries: Vec<_> = self
            .entries
            .values()
            .filter(|entry| !self.deleted_ids.contains(&entry.id))
            .collect();

        // Process in chunks of 4 for batch SIMD
        let chunks = entries.len() / 4;
        let remainder = entries.len() % 4;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;
            let batch_scores = batch_cosine_similarity_4_prenorm(
                query_embedding,
                query_norm,
                [
                    &entries[base].embedding,
                    &entries[base + 1].embedding,
                    &entries[base + 2].embedding,
                    &entries[base + 3].embedding,
                ],
            );

            for (i, &score) in batch_scores.iter().enumerate() {
                if score >= min_score {
                    let entry = entries[base + i];
                    heap.push(MinScoreEntry {
                        score,
                        id: entry.id.clone(),
                        entry: entry.clone(),
                    });

                    if heap.len() > top_k {
                        heap.pop();
                        if let Some(min_entry) = heap.peek() {
                            min_score = min_entry.score;
                        }
                    }
                }
            }
        }

        // Handle remainder
        let start = chunks * 4;
        for i in 0..remainder {
            let entry = entries[start + i];
            let score = cosine_similarity_fast(query_embedding, &entry.embedding, query_norm);

            if score >= min_score {
                heap.push(MinScoreEntry {
                    score,
                    id: entry.id.clone(),
                    entry: entry.clone(),
                });

                if heap.len() > top_k {
                    heap.pop();
                    if let Some(min_entry) = heap.peek() {
                        min_score = min_entry.score;
                    }
                }
            }
        }

        // Convert heap to sorted results (descending)
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|e| MemorySearchResult {
                id: e.id,
                entry: e.entry,
                score: e.score,
            })
            .collect();

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(results)
    }

    /// Get a memory entry by ID.
    pub fn get(&self, id: &str) -> Result<Option<MemoryEntry>> {
        Ok(self.entries.get(id).cloned())
    }

    /// Delete a memory entry by ID.
    ///
    /// Note: HNSW doesn't support true deletion. The entry is removed from
    /// the entries map and marked as deleted, but remains in the HNSW graph.
    /// Deleted entries are filtered out during search.
    ///
    /// For workloads with many deletions, consider calling `rebuild_index()`
    /// periodically to reclaim space and improve search performance.
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        if self.entries.remove(id).is_some() {
            self.id_list.retain(|i| i != id);

            // Mark as deleted for HNSW filtering (HNSW doesn't support true deletion)
            if self.hnsw_index.is_some() {
                self.deleted_ids.insert(id.to_string());
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Rebuild the HNSW index from scratch.
    ///
    /// This reclaims space from deleted entries and can improve search
    /// performance after many deletions. Call this periodically in
    /// workloads with heavy deletion patterns.
    ///
    /// Cost: O(n log n) for n entries.
    pub fn rebuild_index(&mut self) -> Result<()> {
        if !self.config.use_hnsw {
            return Ok(());
        }

        // Create fresh HNSW index
        let hnsw_config = HnswConfig {
            m: self.config.hnsw_m,
            ef_construction: self.config.hnsw_ef_construction,
            ef_search: self.config.hnsw_ef_search,
            max_elements: self.config.max_entries,
        };

        let mut new_index = HnswIndex::new(
            self.config.embedding_dim,
            DistanceMetric::Cosine,
            hnsw_config,
        )
        .map_err(|e| RuvLLMError::Ruvector(format!("Failed to create HNSW index: {}", e)))?;

        // Re-insert all non-deleted entries
        for entry in self.entries.values() {
            new_index
                .add(entry.id.clone(), entry.embedding.clone())
                .map_err(|e| RuvLLMError::Ruvector(format!("Failed to add to HNSW: {}", e)))?;
        }

        self.hnsw_index = Some(new_index);
        self.deleted_ids.clear();

        Ok(())
    }

    /// Get the number of deleted entries pending cleanup.
    ///
    /// If this number grows large relative to total entries,
    /// consider calling `rebuild_index()`.
    pub fn deleted_count(&self) -> usize {
        self.deleted_ids.len()
    }

    /// Check if HNSW indexing is enabled and active.
    pub fn is_hnsw_enabled(&self) -> bool {
        self.hnsw_index.is_some()
    }

    /// List memory entries with pagination.
    pub fn list(&self, limit: usize, offset: usize) -> Result<Vec<MemoryEntry>> {
        Ok(self
            .id_list
            .iter()
            .skip(offset)
            .take(limit)
            .filter_map(|id| self.entries.get(id).cloned())
            .collect())
    }

    /// Get the number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    ///
    /// This also rebuilds the HNSW index if enabled.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.id_list.clear();
        self.deleted_ids.clear();

        // Rebuild empty HNSW index
        if self.config.use_hnsw {
            let hnsw_config = HnswConfig {
                m: self.config.hnsw_m,
                ef_construction: self.config.hnsw_ef_construction,
                ef_search: self.config.hnsw_ef_search,
                max_elements: self.config.max_entries,
            };

            self.hnsw_index = HnswIndex::new(
                self.config.embedding_dim,
                DistanceMetric::Cosine,
                hnsw_config,
            )
            .ok();
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Cleanup expired entries (if TTL is enabled).
    pub fn cleanup_expired(&mut self) {
        if !self.config.enable_ttl {
            return;
        }

        let now = Utc::now();
        let default_ttl = self.config.default_ttl_secs;

        let expired_ids: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                let ttl = entry.metadata.ttl_secs.unwrap_or(default_ttl);
                let age = now.signed_duration_since(entry.created_at);
                age.num_seconds() > ttl as i64
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired_ids {
            let _ = self.delete(&id);
        }
    }
}

/// Compute vector L2 norm.
#[inline]
fn vector_norm(v: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return vector_norm_avx2(v);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return vector_norm_neon(v);
    }

    // Fallback: scalar computation
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Optimized cosine similarity with pre-computed query norm.
#[inline]
fn cosine_similarity_fast(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    if a.len() != b.len() || a_norm < 1e-8 {
        return 0.0;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return cosine_similarity_avx2(a, b, a_norm);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return cosine_similarity_neon(a, b, a_norm);
    }

    // Fallback: optimized scalar
    cosine_similarity_scalar(a, b, a_norm)
}

/// Scalar fallback for cosine similarity.
#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let len = a.len();

    // Unroll loop for better ILP (Instruction Level Parallelism)
    let chunks = len / 4;
    let remainder = len % 4;

    let mut dot_product = 0.0f32;
    let mut b_norm_sq = 0.0f32;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        dot_product += a[idx] * b[idx];
        dot_product += a[idx + 1] * b[idx + 1];
        dot_product += a[idx + 2] * b[idx + 2];
        dot_product += a[idx + 3] * b[idx + 3];

        b_norm_sq += b[idx] * b[idx];
        b_norm_sq += b[idx + 1] * b[idx + 1];
        b_norm_sq += b[idx + 2] * b[idx + 2];
        b_norm_sq += b[idx + 3] * b[idx + 3];
    }

    // Process remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let idx = start + i;
        dot_product += a[idx] * b[idx];
        b_norm_sq += b[idx] * b[idx];
    }

    let b_norm = b_norm_sq.sqrt();
    if b_norm < 1e-8 {
        return 0.0;
    }

    (dot_product / (a_norm * b_norm)).clamp(-1.0, 1.0)
}

/// AVX2 SIMD cosine similarity (x86_64).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn cosine_similarity_avx2(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut dot_sum = _mm256_setzero_ps();
        let mut b_norm_sum = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let idx = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(idx));
            let vb = _mm256_loadu_ps(b_ptr.add(idx));

            // dot += a * b
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            // b_norm += b * b
            b_norm_sum = _mm256_fmadd_ps(vb, vb, b_norm_sum);
        }

        // Horizontal sum
        let dot_product = hsum256_ps(dot_sum);
        let b_norm_sq = hsum256_ps(b_norm_sum);

        // Handle remainder with scalar
        let mut dot_rem = 0.0f32;
        let mut b_rem = 0.0f32;
        let start = chunks * 8;
        for i in 0..remainder {
            let idx = start + i;
            dot_rem += a[idx] * b[idx];
            b_rem += b[idx] * b[idx];
        }

        let total_dot = dot_product + dot_rem;
        let total_b_norm = (b_norm_sq + b_rem).sqrt();

        if total_b_norm < 1e-8 {
            return 0.0;
        }

        (total_dot / (a_norm * total_b_norm)).clamp(-1.0, 1.0)
    }
}

/// Horizontal sum for AVX2 __m256.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let vsum = _mm_add_ps(vlow, vhigh);
    let shuf = _mm_movehdup_ps(vsum);
    let sums = _mm_add_ps(vsum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}

/// AVX2 vector norm.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn vector_norm_avx2(v: &[f32]) -> f32 {
    unsafe {
        let len = v.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = _mm256_setzero_ps();
        let v_ptr = v.as_ptr();

        for i in 0..chunks {
            let idx = i * 8;
            let va = _mm256_loadu_ps(v_ptr.add(idx));
            sum = _mm256_fmadd_ps(va, va, sum);
        }

        let mut total = hsum256_ps(sum);

        // Remainder
        let start = chunks * 8;
        for i in 0..remainder {
            let val = v[start + i];
            total += val * val;
        }

        total.sqrt()
    }
}

/// NEON SIMD cosine similarity (ARM).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn cosine_similarity_neon(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut dot_sum = vdupq_n_f32(0.0);
        let mut b_norm_sum = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let idx = i * 4;
            let va = vld1q_f32(a_ptr.add(idx));
            let vb = vld1q_f32(b_ptr.add(idx));

            dot_sum = vfmaq_f32(dot_sum, va, vb);
            b_norm_sum = vfmaq_f32(b_norm_sum, vb, vb);
        }

        // Horizontal sum
        let dot_product = vaddvq_f32(dot_sum);
        let b_norm_sq = vaddvq_f32(b_norm_sum);

        // Remainder
        let mut dot_rem = 0.0f32;
        let mut b_rem = 0.0f32;
        let start = chunks * 4;
        for i in 0..remainder {
            let idx = start + i;
            dot_rem += a[idx] * b[idx];
            b_rem += b[idx] * b[idx];
        }

        let total_dot = dot_product + dot_rem;
        let total_b_norm = (b_norm_sq + b_rem).sqrt();

        if total_b_norm < 1e-8 {
            return 0.0;
        }

        (total_dot / (a_norm * total_b_norm)).clamp(-1.0, 1.0)
    }
}

/// NEON vector norm.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn vector_norm_neon(v: &[f32]) -> f32 {
    unsafe {
        let len = v.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = vdupq_n_f32(0.0);
        let v_ptr = v.as_ptr();

        for i in 0..chunks {
            let idx = i * 4;
            let va = vld1q_f32(v_ptr.add(idx));
            sum = vfmaq_f32(sum, va, va);
        }

        let mut total = vaddvq_f32(sum);

        // Remainder
        let start = chunks * 4;
        for i in 0..remainder {
            let val = v[start + i];
            total += val * val;
        }

        total.sqrt()
    }
}

/// Compute cosine similarity between two vectors.
/// Original API preserved for compatibility.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let norm_a = vector_norm(a);
    if norm_a < 1e-8 {
        return 0.0;
    }

    cosine_similarity_fast(a, b, norm_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entry(id: &str, text: &str) -> MemoryEntry {
        create_test_entry_with_dim(id, text, 64)
    }

    fn create_test_entry_with_dim(id: &str, text: &str, dim: usize) -> MemoryEntry {
        let mut embedding = vec![0.0f32; dim];
        // Create simple deterministic embedding based on text
        for (i, c) in text.bytes().enumerate() {
            embedding[i % dim] += c as f32 / 255.0;
        }
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        MemoryEntry {
            id: id.to_string(),
            text: text.to_string(),
            embedding,
            metadata: MemoryMetadata::default(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
        }
    }

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert!(config.use_hnsw);
    }

    #[test]
    fn test_memory_config_wasm() {
        let config = MemoryConfig::for_wasm();
        assert_eq!(config.embedding_dim, 256);
        assert!(!config.use_hnsw);
        assert!(config.enable_ttl);
    }

    #[test]
    fn test_memory_add_and_get() {
        let config = MemoryConfig::for_testing();
        let mut memory = RlmMemory::new(config).unwrap();

        let entry = create_test_entry("test-1", "hello world");
        memory.add(entry).unwrap();

        let retrieved = memory.get("test-1").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().text, "hello world");
    }

    #[test]
    fn test_memory_search() {
        let config = MemoryConfig::for_testing();
        let mut memory = RlmMemory::new(config).unwrap();

        memory.add(create_test_entry("1", "hello world")).unwrap();
        memory.add(create_test_entry("2", "hello there")).unwrap();
        memory.add(create_test_entry("3", "goodbye world")).unwrap();

        // Search with a query similar to "hello world"
        let query_entry = create_test_entry("q", "hello world");
        let results = memory.search(&query_entry.embedding, 2).unwrap();

        assert!(!results.is_empty());
        // The exact match should be first
        assert_eq!(results[0].entry.text, "hello world");
    }

    #[test]
    fn test_memory_delete() {
        let config = MemoryConfig::for_testing();
        let mut memory = RlmMemory::new(config).unwrap();

        memory.add(create_test_entry("1", "test")).unwrap();
        assert_eq!(memory.len(), 1);

        let deleted = memory.delete("1").unwrap();
        assert!(deleted);
        assert_eq!(memory.len(), 0);

        let deleted_again = memory.delete("1").unwrap();
        assert!(!deleted_again);
    }

    #[test]
    fn test_memory_capacity_eviction() {
        let mut config = MemoryConfig::for_testing();
        config.max_entries = 3;
        let mut memory = RlmMemory::new(config).unwrap();

        memory.add(create_test_entry("1", "first")).unwrap();
        memory.add(create_test_entry("2", "second")).unwrap();
        memory.add(create_test_entry("3", "third")).unwrap();
        memory.add(create_test_entry("4", "fourth")).unwrap();

        // Should have evicted the first entry
        assert_eq!(memory.len(), 3);
        assert!(memory.get("1").unwrap().is_none());
        assert!(memory.get("4").unwrap().is_some());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_memory_list_pagination() {
        let config = MemoryConfig::for_testing();
        let mut memory = RlmMemory::new(config).unwrap();

        for i in 0..10 {
            memory
                .add(create_test_entry(
                    &format!("{}", i),
                    &format!("entry {}", i),
                ))
                .unwrap();
        }

        let page1 = memory.list(3, 0).unwrap();
        assert_eq!(page1.len(), 3);

        let page2 = memory.list(3, 3).unwrap();
        assert_eq!(page2.len(), 3);

        let last_page = memory.list(3, 9).unwrap();
        assert_eq!(last_page.len(), 1);
    }

    // ==========================================================================
    // HNSW Integration Tests
    // ==========================================================================

    /// Create a config with HNSW enabled for testing
    fn create_hnsw_test_config() -> MemoryConfig {
        MemoryConfig {
            embedding_dim: 128,
            max_entries: 10000,
            use_hnsw: true,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 50,
            similarity_threshold: 0.0, // No threshold for testing
            enable_ttl: false,
            default_ttl_secs: 3600,
        }
    }

    #[test]
    fn test_hnsw_memory_creation() {
        let config = create_hnsw_test_config();
        let memory = RlmMemory::new(config).unwrap();

        assert!(memory.is_hnsw_enabled());
        assert_eq!(memory.len(), 0);
    }

    #[test]
    fn test_hnsw_memory_add_and_search() {
        let config = create_hnsw_test_config();
        let mut memory = RlmMemory::new(config).unwrap();

        // Add entries
        for i in 0..200 {
            let entry = create_test_entry_with_dim(
                &format!("entry-{}", i),
                &format!("Document {} about topic {}", i, i % 10),
                128,
            );
            memory.add(entry).unwrap();
        }

        assert_eq!(memory.len(), 200);

        // Search - should use HNSW since we have >100 entries
        let query = create_test_entry_with_dim("query", "Document 0 about topic 0", 128);
        let results = memory.search(&query.embedding, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // First result should be the exact match or very similar
        assert!(
            results[0].score > 0.9,
            "Expected high similarity, got {}",
            results[0].score
        );
    }

    #[test]
    fn test_hnsw_memory_delete() {
        let config = create_hnsw_test_config();
        let mut memory = RlmMemory::new(config).unwrap();

        // Add entries
        for i in 0..150 {
            let entry = create_test_entry_with_dim(
                &format!("entry-{}", i),
                &format!("Content for entry {}", i),
                128,
            );
            memory.add(entry).unwrap();
        }

        // Delete some entries
        memory.delete("entry-0").unwrap();
        memory.delete("entry-1").unwrap();
        memory.delete("entry-2").unwrap();

        assert_eq!(memory.len(), 147);
        assert_eq!(memory.deleted_count(), 3);

        // Search should not return deleted entries
        let query = create_test_entry_with_dim("query", "Content for entry 0", 128);
        let results = memory.search(&query.embedding, 10).unwrap();

        for result in &results {
            // Check exact matches for deleted entries (not starts_with, as that would
            // incorrectly match "entry-10" for "entry-1")
            assert_ne!(
                result.id.as_str(),
                "entry-0",
                "Deleted entry-0 should not be returned"
            );
            assert_ne!(
                result.id.as_str(),
                "entry-1",
                "Deleted entry-1 should not be returned"
            );
            assert_ne!(
                result.id.as_str(),
                "entry-2",
                "Deleted entry-2 should not be returned"
            );
        }
    }

    #[test]
    fn test_hnsw_memory_rebuild() {
        let config = create_hnsw_test_config();
        let mut memory = RlmMemory::new(config).unwrap();

        // Add entries
        for i in 0..200 {
            let entry =
                create_test_entry_with_dim(&format!("entry-{}", i), &format!("Content {}", i), 128);
            memory.add(entry).unwrap();
        }

        // Delete half
        for i in 0..100 {
            memory.delete(&format!("entry-{}", i)).unwrap();
        }

        assert_eq!(memory.deleted_count(), 100);

        // Rebuild index
        memory.rebuild_index().unwrap();

        assert_eq!(memory.deleted_count(), 0);
        assert_eq!(memory.len(), 100);

        // Search should still work
        let query = create_test_entry_with_dim("query", "Content 150", 128);
        let results = memory.search(&query.embedding, 5).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_hnsw_memory_clear() {
        let config = create_hnsw_test_config();
        let mut memory = RlmMemory::new(config).unwrap();

        // Add entries
        for i in 0..100 {
            let entry =
                create_test_entry_with_dim(&format!("entry-{}", i), &format!("Content {}", i), 128);
            memory.add(entry).unwrap();
        }

        // Clear
        memory.clear();

        assert_eq!(memory.len(), 0);
        assert_eq!(memory.deleted_count(), 0);
        assert!(memory.is_hnsw_enabled()); // HNSW should still be enabled after clear
    }

    #[test]
    fn test_hnsw_vs_brute_force_consistency() {
        // Test that HNSW and brute-force return similar results
        let mut hnsw_config = create_hnsw_test_config();
        hnsw_config.use_hnsw = true;

        let mut brute_config = create_hnsw_test_config();
        brute_config.use_hnsw = false;

        let mut hnsw_memory = RlmMemory::new(hnsw_config).unwrap();
        let mut brute_memory = RlmMemory::new(brute_config).unwrap();

        // Add same entries to both
        for i in 0..500 {
            let entry = create_test_entry_with_dim(
                &format!("entry-{}", i),
                &format!("Document about topic {} with content {}", i % 20, i),
                128,
            );
            hnsw_memory.add(entry.clone()).unwrap();
            brute_memory.add(entry).unwrap();
        }

        // Search with same query
        let query = create_test_entry_with_dim("query", "Document about topic 5", 128);

        let hnsw_results = hnsw_memory.search(&query.embedding, 10).unwrap();
        let brute_results = brute_memory.search(&query.embedding, 10).unwrap();

        // HNSW is approximate, so we check that most top results overlap
        let hnsw_ids: std::collections::HashSet<_> = hnsw_results.iter().map(|r| &r.id).collect();
        let brute_ids: std::collections::HashSet<_> = brute_results.iter().map(|r| &r.id).collect();

        let overlap = hnsw_ids.intersection(&brute_ids).count();

        // Expect at least 60% overlap (HNSW recall should be higher with good params)
        assert!(
            overlap >= 6,
            "Expected at least 60% overlap between HNSW and brute-force results, got {}/10",
            overlap
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = create_hnsw_test_config();
        let mut memory = RlmMemory::new(config).unwrap();

        // Add an entry with correct dimension
        let entry = create_test_entry_with_dim("test", "content", 128);
        memory.add(entry).unwrap();

        // Try to search with wrong dimension
        let wrong_query = vec![0.1f32; 64];
        let result = memory.search(&wrong_query, 5);

        assert!(result.is_err());
        match result.unwrap_err() {
            RuvLLMError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 128);
                assert_eq!(actual, 64);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }
}
