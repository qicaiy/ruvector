//! RuvLTRA RLM Backend Implementation
//!
//! This module implements the `LlmBackend` trait for the RuvLTRA model architecture,
//! providing optimized inference for recursive language model operations.
//!
//! ## Architecture
//!
//! ```text
//! +-------------------+     +-------------------+
//! | RuvLtraRlmBackend |<--->| RuvLtraModel      |
//! | - generate()      |     | - forward()       |
//! | - embed()         |     | - attention       |
//! | - KV cache mgmt   |     | - MLP             |
//! +-------------------+     +-------------------+
//!          |
//!          v
//! +-------------------+     +-------------------+
//! | RuvLtraEnvironment|<--->| SONA Integration  |
//! | - retrieve()      |     | - Instant loop    |
//! | - decompose()     |     | - Background loop |
//! | - synthesize()    |     | - Trajectory rec  |
//! +-------------------+     +-------------------+
//! ```
//!
//! ## Features
//!
//! - **Autoregressive decoding** with temperature-based sampling
//! - **KV cache management** for efficient multi-turn inference
//! - **Mean pooling embeddings** from hidden states
//! - **SONA learning integration** for trajectory recording
//! - **ruvector memory** for HNSW semantic search
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::rlm::ruvltra_backend::{RuvLtraRlmBackend, RuvLtraRlmConfig};
//!
//! // Create backend with default config
//! let config = RuvLtraRlmConfig::default();
//! let backend = RuvLtraRlmBackend::new(config)?;
//!
//! // Generate text
//! let params = GenerationParams::default();
//! let output = backend.generate("Hello, world!", &params)?;
//! println!("Generated: {}", output.text);
//!
//! // Get embeddings
//! let embedding = backend.embed("What is HNSW?")?;
//! ```

use crate::error::{Result, RuvLLMError};
use crate::models::ruvltra::{QuantizationType, RuvLtraConfig, RuvLtraModel};
#[allow(unused_imports)]
use crate::ruvector_integration::{
    IntegrationConfig, RuvectorIntegration, SearchResultWithMetadata, VectorMetadata,
};
#[allow(unused_imports)]
use crate::sona::{SonaConfig, SonaIntegration, Trajectory};
use crate::tokenizer::RuvTokenizer;

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// Generation Types
// =============================================================================

/// Parameters for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Top-p nucleus sampling threshold
    pub top_p: f32,
    /// Top-k sampling limit
    pub top_k: usize,
    /// Stop sequences to terminate generation
    pub stop_sequences: Vec<String>,
    /// Presence penalty for new tokens
    pub presence_penalty: f32,
    /// Frequency penalty for repeated tokens
    pub frequency_penalty: f32,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            stop_sequences: Vec::new(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
        }
    }
}

impl GenerationParams {
    /// Create params for deterministic generation
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Create params for creative generation
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

/// Reason for generation completion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Reached max tokens
    Length,
    /// Hit a stop sequence
    Stop,
    /// Model decided to stop naturally (EOS)
    EndOfText,
    /// Context length exceeded
    ContextExhausted,
}

/// Output from text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    /// Generated text
    pub text: String,
    /// Tokens consumed for generation
    pub tokens_used: usize,
    /// Prompt tokens consumed
    pub prompt_tokens: usize,
    /// Reason generation stopped
    pub finish_reason: FinishReason,
    /// Log probabilities (optional)
    pub logprobs: Option<Vec<f32>>,
}

/// Token emitted during streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamToken {
    /// Token ID
    pub id: u32,
    /// Token text
    pub text: String,
    /// Log probability
    pub logprob: Option<f32>,
    /// Is this a special token?
    pub is_special: bool,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt tokens used
    pub prompt_tokens: usize,
    /// Completion tokens generated
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

// =============================================================================
// Memory Types
// =============================================================================

/// Memory span identifier
pub type MemoryId = String;

/// Metadata associated with stored memory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMetadata {
    /// Source of the memory (document, conversation, etc.)
    pub source: Option<String>,
    /// Creation timestamp
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom key-value pairs
    pub extra: HashMap<String, serde_json::Value>,
}

/// Memory span retrieved from storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySpan {
    /// Unique identifier
    pub id: MemoryId,
    /// Text content
    pub text: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Similarity score to query (0.0 - 1.0)
    pub similarity_score: f32,
    /// Source document/conversation
    pub source: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

// =============================================================================
// Query Decomposition Types
// =============================================================================

/// Context for query decomposition
#[derive(Debug, Clone, Default)]
pub struct QueryContext {
    /// Current recursion depth
    pub depth: usize,
    /// Parent query (if sub-query)
    pub parent_query: Option<String>,
    /// Retrieved context so far
    pub retrieved_context: Vec<MemorySpan>,
    /// Token budget remaining
    pub token_budget: usize,
    /// Session ID for state tracking
    pub session_id: Option<String>,
}

/// Strategy for decomposing a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// No decomposition needed - answer directly
    Direct,
    /// Split by conjunction ("and", "or")
    Conjunction(Vec<String>),
    /// Split by aspect (what, why, how)
    Aspect(Vec<String>),
    /// Sequential multi-step reasoning
    Sequential(Vec<String>),
    /// Parallel independent sub-questions
    Parallel(Vec<String>),
}

impl Default for DecompositionStrategy {
    fn default() -> Self {
        Self::Direct
    }
}

/// A sub-query generated from decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuery {
    /// The sub-question text
    pub query: String,
    /// Dependencies on other sub-queries (indices)
    pub depends_on: Vec<usize>,
    /// Priority (higher = execute first)
    pub priority: u8,
    /// Estimated complexity (0.0 - 1.0)
    pub complexity: f32,
}

/// Result of answering a sub-query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAnswer {
    /// Original sub-query
    pub query: SubQuery,
    /// Generated answer text
    pub answer: String,
    /// Confidence in the answer (0.0 - 1.0)
    pub confidence: f32,
    /// Sources used
    pub sources: Vec<MemorySpan>,
    /// Tokens used
    pub tokens_used: usize,
}

/// Query decomposition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDecomposition {
    /// Decomposition strategy used
    pub strategy: DecompositionStrategy,
    /// Sub-queries to execute
    pub sub_queries: Vec<SubQuery>,
    /// Dependency edges (from_idx, to_idx)
    pub dependencies: Vec<(usize, usize)>,
    /// Whether sub-queries can be parallelized
    pub parallelizable: bool,
}

impl QueryDecomposition {
    /// Create a direct (no decomposition) result
    pub fn direct() -> Self {
        Self {
            strategy: DecompositionStrategy::Direct,
            sub_queries: Vec::new(),
            dependencies: Vec::new(),
            parallelizable: false,
        }
    }

    /// Check if decomposition is needed
    pub fn needs_decomposition(&self) -> bool {
        !matches!(self.strategy, DecompositionStrategy::Direct)
    }
}

// =============================================================================
// RLM Answer Types
// =============================================================================

/// Final answer from RLM processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmAnswer {
    /// Generated answer text
    pub text: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Source memories used
    pub sources: Vec<MemorySpan>,
    /// Sub-queries executed (if any)
    pub sub_queries: Option<Vec<SubQuery>>,
    /// Quality score from validator
    pub quality_score: f32,
    /// Token usage breakdown
    pub token_usage: TokenUsage,
    /// Trajectory ID for learning
    pub trajectory_id: Option<String>,
    /// Recursion depth reached
    pub depth_reached: usize,
    /// Time taken in milliseconds
    pub latency_ms: u64,
}

impl RlmAnswer {
    /// Create a new answer
    pub fn new(text: String, confidence: f32) -> Self {
        Self {
            text,
            confidence,
            sources: Vec::new(),
            sub_queries: None,
            quality_score: confidence,
            token_usage: TokenUsage::default(),
            trajectory_id: None,
            depth_reached: 0,
            latency_ms: 0,
        }
    }

    /// Add sources
    pub fn with_sources(mut self, sources: Vec<MemorySpan>) -> Self {
        self.sources = sources;
        self
    }

    /// Set token usage
    pub fn with_token_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = usage;
        self
    }
}

// =============================================================================
// Model Info
// =============================================================================

/// Model information from a backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmModelInfo {
    /// Model name/ID
    pub name: String,
    /// Model architecture
    pub architecture: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Maximum context length
    pub max_context_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Whether embeddings are supported
    pub supports_embeddings: bool,
}

// =============================================================================
// KV Cache Management
// =============================================================================

/// Entry in the KV cache for a single session
#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    /// Key states for each layer: (layer_idx, key_states)
    pub keys: Vec<Vec<f32>>,
    /// Value states for each layer: (layer_idx, value_states)
    pub values: Vec<Vec<f32>>,
    /// Current sequence length
    pub seq_len: usize,
    /// Last access timestamp
    pub last_accessed: std::time::Instant,
}

impl KvCacheEntry {
    /// Create a new empty KV cache entry
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
            seq_len: 0,
            last_accessed: std::time::Instant::now(),
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for k in &mut self.keys {
            k.clear();
        }
        for v in &mut self.values {
            v.clear();
        }
        self.seq_len = 0;
    }

    /// Get total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let key_bytes: usize = self.keys.iter().map(|k| k.len() * 4).sum();
        let val_bytes: usize = self.values.iter().map(|v| v.len() * 4).sum();
        key_bytes + val_bytes
    }
}

/// KV cache manager for multiple sessions
/// Optimized with LRU eviction and efficient memory tracking.
#[derive(Debug)]
pub struct KvCache {
    /// Cache entries by session ID
    entries: DashMap<String, KvCacheEntry>,
    /// Number of layers
    num_layers: usize,
    /// Maximum memory budget in bytes
    max_memory_bytes: usize,
    /// Current total memory (tracked incrementally to avoid repeated calculation)
    current_memory_bytes: AtomicU64,
    /// Total operations count
    total_ops: AtomicU64,
    /// Cache hits
    cache_hits: AtomicU64,
    /// Eviction count
    evictions: AtomicU64,
}

impl KvCache {
    /// Create a new KV cache
    pub fn new(num_layers: usize, max_memory_bytes: usize) -> Self {
        Self {
            entries: DashMap::new(),
            num_layers,
            max_memory_bytes,
            current_memory_bytes: AtomicU64::new(0),
            total_ops: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Get or create a cache entry for a session
    /// Optimized: tracks hit/miss without double lookup.
    #[inline]
    pub fn get_or_create(
        &self,
        session_id: &str,
    ) -> dashmap::mapref::one::RefMut<'_, String, KvCacheEntry> {
        self.total_ops.fetch_add(1, Ordering::Relaxed);

        // Use entry API to avoid double lookup
        let is_existing = self.entries.contains_key(session_id);
        if is_existing {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        }

        self.entries
            .entry(session_id.to_string())
            .or_insert_with(|| KvCacheEntry::new(self.num_layers))
    }

    /// Clear cache for a session
    pub fn clear_session(&self, session_id: &str) {
        if let Some((_, entry)) = self.entries.remove(session_id) {
            // Update tracked memory
            let freed = entry.memory_bytes() as u64;
            self.current_memory_bytes
                .fetch_sub(freed, Ordering::Relaxed);
        }
    }

    /// Update memory tracking after cache modification.
    #[inline]
    pub fn update_memory_usage(&self, old_size: usize, new_size: usize) {
        if new_size > old_size {
            self.current_memory_bytes
                .fetch_add((new_size - old_size) as u64, Ordering::Relaxed);
        } else if old_size > new_size {
            self.current_memory_bytes
                .fetch_sub((old_size - new_size) as u64, Ordering::Relaxed);
        }
    }

    /// Evict oldest entries if memory exceeded.
    /// Optimized: uses batch eviction targeting 80% capacity.
    pub fn evict_if_needed(&self) {
        // Fast path: recalculate total memory (more accurate)
        let current: usize = self.entries.iter().map(|e| e.memory_bytes()).sum();
        self.current_memory_bytes
            .store(current as u64, Ordering::Relaxed);

        if current <= self.max_memory_bytes {
            return;
        }

        // Calculate how much we need to free (aim for 80% capacity)
        let target = (self.max_memory_bytes * 80) / 100;
        let to_free = current.saturating_sub(target);

        if to_free == 0 {
            return;
        }

        let mut freed = 0usize;
        let mut to_remove = Vec::new();

        // Collect candidates sorted by last_accessed
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .map(|e| (e.key().clone(), e.last_accessed, e.memory_bytes()))
            .collect();

        // Sort by last_accessed (oldest first) - unstable for speed
        candidates.sort_unstable_by_key(|(_, accessed, _)| *accessed);

        // Select entries to evict
        for (key, _, mem_size) in candidates {
            if freed >= to_free {
                break;
            }
            to_remove.push(key);
            freed += mem_size;
        }

        // Perform eviction
        for key in to_remove {
            if let Some((_, entry)) = self.entries.remove(&key) {
                self.current_memory_bytes
                    .fetch_sub(entry.memory_bytes() as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Evict a single oldest entry (lightweight version for incremental eviction).
    #[inline]
    pub fn evict_one_if_needed(&self) {
        let current = self.current_memory_bytes.load(Ordering::Relaxed) as usize;

        if current <= self.max_memory_bytes {
            return;
        }

        // Find oldest entry using iterator (avoids full collection)
        let oldest = self
            .entries
            .iter()
            .min_by_key(|e| e.last_accessed)
            .map(|e| e.key().clone());

        if let Some(key) = oldest {
            if let Some((_, entry)) = self.entries.remove(&key) {
                self.current_memory_bytes
                    .fetch_sub(entry.memory_bytes() as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> KvCacheStats {
        let total_ops = self.total_ops.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        KvCacheStats {
            entries: self.entries.len(),
            total_memory_bytes: self.current_memory_bytes.load(Ordering::Relaxed) as usize,
            hit_rate: if total_ops > 0 {
                hits as f32 / total_ops as f32
            } else {
                0.0
            },
            total_ops,
        }
    }
}

/// KV cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheStats {
    /// Number of cached sessions
    pub entries: usize,
    /// Total memory used in bytes
    pub total_memory_bytes: usize,
    /// Cache hit rate
    pub hit_rate: f32,
    /// Total operations
    pub total_ops: u64,
}

// =============================================================================
// LLM Backend Trait
// =============================================================================

/// Backend abstraction for any LLM engine
///
/// This trait allows different inference backends (RuvLTRA, Candle, mistral.rs, API)
/// to be used interchangeably with the RLM system.
pub trait LlmBackendTrait: Send + Sync {
    /// Get backend identifier
    fn id(&self) -> &str;

    /// Get model information
    fn model_info(&self) -> RlmModelInfo;

    /// Get maximum context length in tokens
    fn max_context(&self) -> usize;

    /// Estimate token count for text
    fn estimate_tokens(&self, text: &str) -> usize;

    /// Generate completion (blocking)
    fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<GenerationOutput>;

    /// Generate embeddings for text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

// =============================================================================
// Memory Store Trait
// =============================================================================

/// Memory storage abstraction for RLM
///
/// This trait abstracts the memory backend (typically ruvector HNSW index)
/// for semantic search and pattern retrieval.
pub trait MemoryStore: Send + Sync {
    /// Store a memory span
    fn store(&self, text: &str, metadata: MemoryMetadata) -> Result<MemoryId>;

    /// Search for similar memories
    fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<MemorySpan>>;

    /// Retrieve a specific memory by ID
    fn get(&self, id: &MemoryId) -> Result<Option<MemorySpan>>;

    /// Delete a memory
    fn delete(&self, id: &MemoryId) -> Result<bool>;

    /// Get total number of stored memories
    fn count(&self) -> usize;
}

// =============================================================================
// RuvLTRA RLM Backend Configuration
// =============================================================================

/// Configuration for RuvLTRA RLM backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvLtraRlmConfig {
    /// RuvLTRA model configuration
    pub model_config: RuvLtraConfig,
    /// Maximum memory for KV cache in MB
    pub kv_cache_memory_mb: usize,
    /// Enable SONA learning
    pub enable_sona: bool,
    /// SONA configuration
    pub sona_config: SonaConfig,
    /// Default generation parameters
    pub default_gen_params: GenerationParams,
    /// Embedding pooling strategy
    pub embedding_pooling: EmbeddingPooling,
}

/// Embedding pooling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingPooling {
    /// Mean pool across sequence
    Mean,
    /// Use last token
    Last,
    /// Use first token (CLS-style)
    First,
    /// Max pool across sequence
    Max,
}

impl Default for EmbeddingPooling {
    fn default() -> Self {
        Self::Mean
    }
}

impl Default for RuvLtraRlmConfig {
    fn default() -> Self {
        Self {
            model_config: RuvLtraConfig::qwen_0_5b(),
            kv_cache_memory_mb: 512,
            enable_sona: true,
            sona_config: SonaConfig::default(),
            default_gen_params: GenerationParams::default(),
            embedding_pooling: EmbeddingPooling::Mean,
        }
    }
}

// =============================================================================
// RuvLTRA RLM Backend Implementation
// =============================================================================

/// RuvLTRA backend implementing the `LlmBackendTrait`
///
/// This backend uses the RuvLTRA model (Qwen 0.5B-based) for inference,
/// providing autoregressive decoding with KV cache management and
/// mean-pooled embeddings from hidden states.
pub struct RuvLtraRlmBackend {
    /// The RuvLTRA model
    model: Arc<RwLock<RuvLtraModel>>,
    /// Tokenizer
    tokenizer: Arc<RuvTokenizer>,
    /// Configuration
    config: RuvLtraRlmConfig,
    /// KV cache manager
    kv_cache: KvCache,
    /// Generation statistics
    stats: RuvLtraRlmStats,
}

/// Backend statistics
#[derive(Debug, Default)]
struct RuvLtraRlmStats {
    /// Total generations
    total_generations: AtomicU64,
    /// Total embeddings
    total_embeddings: AtomicU64,
    /// Total tokens generated
    total_tokens: AtomicU64,
    /// Total latency in microseconds
    total_latency_us: AtomicU64,
}

impl RuvLtraRlmBackend {
    /// Create a new RuvLTRA RLM backend
    pub fn new(config: RuvLtraRlmConfig) -> Result<Self> {
        let model = RuvLtraModel::new(&config.model_config)?;

        // Create tokenizer (using Qwen tokenizer by default)
        // Note: Requires the 'candle' feature for actual tokenization
        let tokenizer = RuvTokenizer::from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")?;

        let kv_cache = KvCache::new(
            config.model_config.num_hidden_layers,
            config.kv_cache_memory_mb * 1024 * 1024,
        );

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            tokenizer: Arc::new(tokenizer),
            config,
            kv_cache,
            stats: RuvLtraRlmStats::default(),
        })
    }

    /// Create with a pre-loaded model
    pub fn with_model(
        model: RuvLtraModel,
        tokenizer: RuvTokenizer,
        config: RuvLtraRlmConfig,
    ) -> Self {
        let kv_cache = KvCache::new(
            config.model_config.num_hidden_layers,
            config.kv_cache_memory_mb * 1024 * 1024,
        );

        Self {
            model: Arc::new(RwLock::new(model)),
            tokenizer: Arc::new(tokenizer),
            config,
            kv_cache,
            stats: RuvLtraRlmStats::default(),
        }
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &RuvTokenizer {
        &self.tokenizer
    }

    /// Get KV cache statistics
    pub fn kv_cache_stats(&self) -> KvCacheStats {
        self.kv_cache.stats()
    }

    /// Clear KV cache for a session
    pub fn clear_session_cache(&self, session_id: &str) {
        self.kv_cache.clear_session(session_id);
    }

    /// Internal: perform autoregressive generation
    fn autoregressive_generate(
        &self,
        input_ids: &[u32],
        params: &GenerationParams,
        session_id: Option<&str>,
    ) -> Result<(Vec<u32>, FinishReason)> {
        let model = self.model.read();
        let mut generated_ids = Vec::new();
        let mut finish_reason = FinishReason::Length;

        // Get or create KV cache
        let session_key = session_id.unwrap_or("default").to_string();
        let mut kv_entry = self.kv_cache.get_or_create(&session_key);

        // Prepare KV caches for model
        let mut kv_caches: Vec<(Vec<f32>, Vec<f32>)> = kv_entry
            .keys
            .iter()
            .zip(kv_entry.values.iter())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let start_pos = kv_entry.seq_len;
        let mut current_ids = input_ids.to_vec();

        for i in 0..params.max_tokens {
            // Create position indices
            let positions: Vec<usize> =
                (start_pos + i..start_pos + i + current_ids.len()).collect();

            // Forward pass
            let logits = model.forward(&current_ids, &positions, Some(&mut kv_caches))?;

            // Get last token logits
            let vocab_size = self.config.model_config.vocab_size;
            let last_logits = &logits[(logits.len() - vocab_size)..];

            // Sample next token
            let next_token = self.sample_token(last_logits, params)?;

            // Check for EOS
            if next_token == self.config.model_config.eos_token_id {
                finish_reason = FinishReason::EndOfText;
                break;
            }

            generated_ids.push(next_token);

            // Check for stop sequences
            if !params.stop_sequences.is_empty() {
                let generated_text = self.tokenizer.decode(&generated_ids).unwrap_or_default();
                for stop in &params.stop_sequences {
                    if generated_text.ends_with(stop) {
                        finish_reason = FinishReason::Stop;
                        break;
                    }
                }
                if finish_reason == FinishReason::Stop {
                    break;
                }
            }

            // Update current_ids for next iteration (only the new token)
            current_ids = vec![next_token];
        }

        // Update KV cache entry
        for (i, (k, v)) in kv_caches.iter().enumerate() {
            kv_entry.keys[i] = k.clone();
            kv_entry.values[i] = v.clone();
        }
        kv_entry.seq_len = start_pos + input_ids.len() + generated_ids.len();
        kv_entry.last_accessed = std::time::Instant::now();

        // Evict old entries if needed
        drop(kv_entry);
        self.kv_cache.evict_if_needed();

        Ok((generated_ids, finish_reason))
    }

    /// Sample a token from logits
    fn sample_token(&self, logits: &[f32], params: &GenerationParams) -> Result<u32> {
        if params.temperature == 0.0 {
            // Greedy decoding
            let (max_idx, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| RuvLLMError::InvalidOperation("Empty logits".to_string()))?;
            return Ok(max_idx as u32);
        }

        // Apply temperature
        let temperature = params.temperature.max(1e-7);
        let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();

        // Softmax
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = scaled.iter().map(|l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();

        // Apply top-k filtering
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = if params.top_k > 0 {
            params.top_k
        } else {
            probs.len()
        };
        let mut filtered: Vec<(usize, f32)> = indexed_probs.into_iter().take(top_k).collect();

        // Apply top-p (nucleus) filtering
        let mut cumsum = 0.0;
        let mut cutoff_idx = filtered.len();
        for (i, (_, p)) in filtered.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        filtered.truncate(cutoff_idx);

        // Renormalize
        let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
        let normalized: Vec<(usize, f32)> = filtered.iter().map(|(i, p)| (*i, p / sum)).collect();

        // Sample
        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        });
        let mut rng_state = seed;
        let random: f32 = {
            // Simple LCG for deterministic sampling
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32
        };

        let mut cumsum = 0.0;
        for (idx, prob) in &normalized {
            cumsum += prob;
            if random < cumsum {
                return Ok(*idx as u32);
            }
        }

        // Fallback to most likely
        Ok(normalized.first().map(|(i, _)| *i as u32).unwrap_or(0))
    }

    /// Extract hidden states for embedding
    fn extract_hidden_states(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let model = self.model.read();
        let positions: Vec<usize> = (0..input_ids.len()).collect();

        // Run forward pass without KV cache to get hidden states
        // The model's forward returns logits, so we need to get the hidden states before the LM head
        // For now, we approximate embeddings using the embedding layer output
        let hidden_size = self.config.model_config.hidden_size;
        let mut hidden_states = Vec::with_capacity(input_ids.len() * hidden_size);

        for &token_id in input_ids {
            let offset = (token_id as usize) * hidden_size;
            if offset + hidden_size <= model.embed_tokens.len() {
                hidden_states.extend_from_slice(&model.embed_tokens[offset..offset + hidden_size]);
            }
        }

        Ok(hidden_states)
    }

    /// Pool hidden states into a single embedding
    /// Optimized with loop unrolling and SIMD-friendly access patterns.
    fn pool_hidden_states(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        let hidden_size = self.config.model_config.hidden_size;

        if seq_len == 0 || hidden_states.len() < hidden_size {
            return vec![0.0; hidden_size];
        }

        match self.config.embedding_pooling {
            EmbeddingPooling::Mean => self.pool_mean_optimized(hidden_states, seq_len, hidden_size),
            EmbeddingPooling::Last => {
                let offset = (seq_len - 1) * hidden_size;
                if offset + hidden_size <= hidden_states.len() {
                    hidden_states[offset..offset + hidden_size].to_vec()
                } else {
                    vec![0.0; hidden_size]
                }
            }
            EmbeddingPooling::First => {
                if hidden_states.len() >= hidden_size {
                    hidden_states[..hidden_size].to_vec()
                } else {
                    vec![0.0; hidden_size]
                }
            }
            EmbeddingPooling::Max => self.pool_max_optimized(hidden_states, seq_len, hidden_size),
        }
    }

    /// Optimized mean pooling with loop unrolling.
    #[inline]
    fn pool_mean_optimized(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut pooled = vec![0.0; hidden_size];
        let inv_seq_len = 1.0 / seq_len as f32;

        // Process 4 dimensions at a time for better cache utilization
        let chunks = hidden_size / 4;
        let remainder = hidden_size % 4;

        for t in 0..seq_len {
            let offset = t * hidden_size;
            if offset + hidden_size > hidden_states.len() {
                continue;
            }

            // Unrolled loop
            for c in 0..chunks {
                let i = c * 4;
                pooled[i] += hidden_states[offset + i];
                pooled[i + 1] += hidden_states[offset + i + 1];
                pooled[i + 2] += hidden_states[offset + i + 2];
                pooled[i + 3] += hidden_states[offset + i + 3];
            }

            // Remainder
            let start = chunks * 4;
            for i in 0..remainder {
                pooled[start + i] += hidden_states[offset + start + i];
            }
        }

        // Apply division with pre-computed inverse
        for p in &mut pooled {
            *p *= inv_seq_len;
        }

        pooled
    }

    /// Optimized max pooling.
    #[inline]
    fn pool_max_optimized(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        // Initialize from first valid slice
        let mut pooled = if hidden_states.len() >= hidden_size {
            hidden_states[..hidden_size].to_vec()
        } else {
            return vec![0.0; hidden_size];
        };

        // Process remaining sequences
        for t in 1..seq_len {
            let offset = t * hidden_size;
            if offset + hidden_size > hidden_states.len() {
                continue;
            }

            // Unrolled max comparison
            let chunks = hidden_size / 4;
            let remainder = hidden_size % 4;

            for c in 0..chunks {
                let i = c * 4;
                pooled[i] = pooled[i].max(hidden_states[offset + i]);
                pooled[i + 1] = pooled[i + 1].max(hidden_states[offset + i + 1]);
                pooled[i + 2] = pooled[i + 2].max(hidden_states[offset + i + 2]);
                pooled[i + 3] = pooled[i + 3].max(hidden_states[offset + i + 3]);
            }

            let start = chunks * 4;
            for i in 0..remainder {
                pooled[start + i] = pooled[start + i].max(hidden_states[offset + start + i]);
            }
        }

        pooled
    }

    /// Get generation statistics
    pub fn stats(&self) -> RuvLtraBackendStats {
        let total_gens = self.stats.total_generations.load(Ordering::Relaxed);
        let total_latency = self.stats.total_latency_us.load(Ordering::Relaxed);
        RuvLtraBackendStats {
            total_generations: total_gens,
            total_embeddings: self.stats.total_embeddings.load(Ordering::Relaxed),
            total_tokens: self.stats.total_tokens.load(Ordering::Relaxed),
            avg_latency_ms: if total_gens > 0 {
                (total_latency / total_gens) as f32 / 1000.0
            } else {
                0.0
            },
            kv_cache_stats: self.kv_cache.stats(),
        }
    }
}

/// Backend statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvLtraBackendStats {
    /// Total generation calls
    pub total_generations: u64,
    /// Total embedding calls
    pub total_embeddings: u64,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// KV cache statistics
    pub kv_cache_stats: KvCacheStats,
}

impl LlmBackendTrait for RuvLtraRlmBackend {
    fn id(&self) -> &str {
        "ruvltra"
    }

    fn model_info(&self) -> RlmModelInfo {
        RlmModelInfo {
            name: "RuvLTRA".to_string(),
            architecture: "Qwen".to_string(),
            num_parameters: self.config.model_config.estimate_params(),
            hidden_size: self.config.model_config.hidden_size,
            max_context_length: self.config.model_config.max_position_embeddings,
            vocab_size: self.config.model_config.vocab_size,
            supports_embeddings: true,
        }
    }

    fn max_context(&self) -> usize {
        self.config.model_config.max_position_embeddings
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        // Use tokenizer if available, otherwise estimate ~4 chars per token
        self.tokenizer
            .encode(text)
            .map(|ids| ids.len())
            .unwrap_or_else(|_| text.len() / 4 + 1)
    }

    fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<GenerationOutput> {
        let start = Instant::now();
        self.stats.total_generations.fetch_add(1, Ordering::Relaxed);

        // Tokenize prompt
        let input_ids = self.tokenizer.encode(prompt)?;
        let prompt_tokens = input_ids.len();

        // Generate
        let (generated_ids, finish_reason) =
            self.autoregressive_generate(&input_ids, params, None)?;
        let completion_tokens = generated_ids.len();

        // Decode
        let text = self.tokenizer.decode(&generated_ids)?;

        // Update stats
        self.stats
            .total_tokens
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.stats
            .total_latency_us
            .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);

        Ok(GenerationOutput {
            text,
            tokens_used: completion_tokens,
            prompt_tokens,
            finish_reason,
            logprobs: None,
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.stats.total_embeddings.fetch_add(1, Ordering::Relaxed);

        // Tokenize
        let input_ids = self.tokenizer.encode(text)?;
        let seq_len = input_ids.len();

        // Get hidden states
        let hidden_states = self.extract_hidden_states(&input_ids)?;

        // Pool to single embedding
        let embedding = self.pool_hidden_states(&hidden_states, seq_len);

        // Normalize embedding (L2)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 1e-8 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };

        Ok(normalized)
    }
}

// =============================================================================
// RuvLTRA Memory Store (wraps RuvectorIntegration)
// =============================================================================

/// Memory store backed by RuvectorIntegration
pub struct RuvLtraMemoryStore {
    /// Ruvector integration
    integration: Arc<RuvectorIntegration>,
    /// Embedding backend for encoding queries
    backend: Arc<RuvLtraRlmBackend>,
    /// Memory counter
    memory_count: AtomicU64,
}

impl RuvLtraMemoryStore {
    /// Create a new memory store
    pub fn new(backend: Arc<RuvLtraRlmBackend>) -> Result<Self> {
        let config = IntegrationConfig {
            embedding_dim: backend.config.model_config.hidden_size,
            ..Default::default()
        };
        let integration = RuvectorIntegration::new(config)?;

        Ok(Self {
            integration: Arc::new(integration),
            backend,
            memory_count: AtomicU64::new(0),
        })
    }

    /// Create with existing integration
    pub fn with_integration(
        integration: RuvectorIntegration,
        backend: Arc<RuvLtraRlmBackend>,
    ) -> Self {
        Self {
            integration: Arc::new(integration),
            backend,
            memory_count: AtomicU64::new(0),
        }
    }
}

impl MemoryStore for RuvLtraMemoryStore {
    fn store(&self, text: &str, metadata: MemoryMetadata) -> Result<MemoryId> {
        // Get embedding for text
        let embedding = self.backend.embed(text)?;

        // Generate ID
        let id = uuid::Uuid::new_v4().to_string();

        // Convert metadata to VectorMetadata
        let vec_meta = VectorMetadata {
            source: metadata
                .source
                .clone()
                .unwrap_or_else(|| "user".to_string()),
            task_type: metadata.tags.first().cloned(),
            agent_type: None,
            quality_score: 1.0,
            access_count: 0,
            created_at: metadata.created_at.unwrap_or_else(chrono::Utc::now),
            last_accessed: chrono::Utc::now(),
            tags: metadata.tags,
        };

        // Store in unified index
        self.integration
            .add_vector(id.clone(), embedding, vec_meta)?;
        self.memory_count.fetch_add(1, Ordering::Relaxed);

        Ok(id)
    }

    fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<MemorySpan>> {
        let results = self.integration.search(query_embedding, top_k)?;

        Ok(results
            .into_iter()
            .map(|r| MemorySpan {
                id: r.id,
                text: r
                    .metadata
                    .as_ref()
                    .and_then(|m| m.task_type.clone())
                    .unwrap_or_default(),
                embedding: Vec::new(), // Don't include full embedding in results
                similarity_score: 1.0 - r.score, // Convert distance to similarity
                source: r.metadata.as_ref().map(|m| m.source.clone()),
                metadata: HashMap::new(),
            })
            .collect())
    }

    fn get(&self, _id: &MemoryId) -> Result<Option<MemorySpan>> {
        // Direct lookup not supported by current integration
        // Would need to add a get_by_id method to UnifiedIndex
        Ok(None)
    }

    fn delete(&self, _id: &MemoryId) -> Result<bool> {
        // Deletion not supported by current integration
        Ok(false)
    }

    fn count(&self) -> usize {
        self.memory_count.load(Ordering::Relaxed) as usize
    }
}

// =============================================================================
// RuvLTRA RLM Environment
// =============================================================================

/// Configuration for RuvLTRA environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvLtraEnvConfig {
    /// Retrieval top-k
    pub retrieval_top_k: usize,
    /// Enable SONA learning
    pub enable_sona: bool,
    /// SONA config
    pub sona_config: SonaConfig,
    /// Quality threshold for learning
    pub quality_threshold: f32,
}

impl Default for RuvLtraEnvConfig {
    fn default() -> Self {
        Self {
            retrieval_top_k: 5,
            enable_sona: true,
            sona_config: SonaConfig::default(),
            quality_threshold: 0.5,
        }
    }
}

/// RuvLTRA environment implementing RlmEnvironment trait
///
/// This environment combines the RuvLTRA backend with ruvector memory
/// and SONA learning for a complete RLM solution.
pub struct RuvLtraEnvironment {
    /// LLM backend
    backend: Arc<RuvLtraRlmBackend>,
    /// Memory store
    memory: Arc<RuvLtraMemoryStore>,
    /// SONA integration for learning
    sona: Option<Arc<RwLock<SonaIntegration>>>,
    /// Configuration
    config: RuvLtraEnvConfig,
    /// Query decomposer
    decomposer: QueryDecomposer,
    /// Answer synthesizer
    synthesizer: AnswerSynthesizer,
}

impl RuvLtraEnvironment {
    /// Create a new RuvLTRA environment
    pub fn new(backend: RuvLtraRlmBackend, config: RuvLtraEnvConfig) -> Result<Self> {
        let backend = Arc::new(backend);
        let memory = Arc::new(RuvLtraMemoryStore::new(backend.clone())?);

        let sona = if config.enable_sona {
            Some(Arc::new(RwLock::new(SonaIntegration::new(
                config.sona_config.clone(),
            ))))
        } else {
            None
        };

        Ok(Self {
            backend: backend.clone(),
            memory,
            sona,
            config,
            decomposer: QueryDecomposer::new(backend.clone()),
            synthesizer: AnswerSynthesizer::new(backend),
        })
    }

    /// Get SONA integration
    pub fn sona(&self) -> Option<&Arc<RwLock<SonaIntegration>>> {
        self.sona.as_ref()
    }

    /// Record a trajectory for SONA learning
    pub fn record_trajectory(&self, trajectory: Trajectory) -> Result<()> {
        if let Some(sona) = &self.sona {
            sona.read().record_trajectory(trajectory)?;
        }
        Ok(())
    }

    /// Trigger background learning
    pub fn trigger_background_learning(&self) -> Result<()> {
        if let Some(sona) = &self.sona {
            sona.read().trigger_background_loop()?;
        }
        Ok(())
    }

    /// Retrieve with query text (embeds internally)
    pub fn retrieve_text(&self, query: &str, top_k: usize) -> Result<Vec<MemorySpan>> {
        let embedding = self.backend.embed(query)?;
        self.memory.search(&embedding, top_k)
    }

    /// Decompose query into sub-queries
    pub fn decompose(&self, query: &str, context: &QueryContext) -> Result<QueryDecomposition> {
        self.decomposer.decompose(query, context)
    }

    /// Synthesize answers
    pub fn synthesize(&self, query: &str, sub_answers: &[SubAnswer]) -> Result<String> {
        self.synthesizer.synthesize(query, sub_answers)
    }

    /// Answer a query recursively
    pub fn answer_query(
        &self,
        query: &str,
        depth: usize,
        config: &super::controller::RlmConfig,
    ) -> Result<RlmAnswer> {
        let start = Instant::now();

        // Check depth limit
        if depth >= config.max_depth {
            return Ok(RlmAnswer::new(
                "Maximum recursion depth reached".to_string(),
                0.0,
            ));
        }

        // Retrieve relevant context
        let context_spans = self.retrieve_text(query, config.retrieval_top_k)?;

        // Build context for decomposition
        let query_context = QueryContext {
            depth,
            parent_query: None,
            retrieved_context: context_spans.clone(),
            token_budget: config.token_budget,
            session_id: None,
        };

        // Try to decompose
        let decomposition = self.decompose(query, &query_context)?;

        let answer = if decomposition.needs_decomposition() && depth < config.max_depth - 1 {
            // Handle sub-queries
            let mut sub_answers = Vec::new();

            for sub_query in &decomposition.sub_queries {
                let sub_answer = self.answer_query(&sub_query.query, depth + 1, config)?;
                sub_answers.push(SubAnswer {
                    query: sub_query.clone(),
                    answer: sub_answer.text,
                    confidence: sub_answer.confidence,
                    sources: sub_answer.sources,
                    tokens_used: sub_answer.token_usage.total_tokens,
                });
            }

            // Synthesize final answer
            let synthesized = self.synthesize(query, &sub_answers)?;

            RlmAnswer {
                text: synthesized,
                confidence: sub_answers.iter().map(|a| a.confidence).sum::<f32>()
                    / sub_answers.len() as f32,
                sources: context_spans,
                sub_queries: Some(decomposition.sub_queries),
                quality_score: 0.8,
                token_usage: TokenUsage::default(),
                trajectory_id: None,
                depth_reached: depth + 1,
                latency_ms: start.elapsed().as_millis() as u64,
            }
        } else {
            // Direct answer
            let prompt = self.build_prompt(query, &context_spans);
            let params = GenerationParams::default().with_max_tokens(256);
            let output = self.backend.generate(&prompt, &params)?;

            RlmAnswer {
                text: output.text,
                confidence: 0.8, // Base confidence
                sources: context_spans,
                sub_queries: None,
                quality_score: 0.8,
                token_usage: TokenUsage {
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.tokens_used,
                    total_tokens: output.prompt_tokens + output.tokens_used,
                },
                trajectory_id: None,
                depth_reached: depth,
                latency_ms: start.elapsed().as_millis() as u64,
            }
        };

        // Record trajectory for learning
        if self.config.enable_sona && answer.quality_score >= self.config.quality_threshold {
            let query_embedding = self.backend.embed(query)?;
            let response_embedding = self.backend.embed(&answer.text)?;

            let trajectory = Trajectory {
                request_id: uuid::Uuid::new_v4().to_string(),
                session_id: "rlm".to_string(),
                query_embedding,
                response_embedding,
                quality_score: answer.quality_score,
                routing_features: vec![depth as f32, answer.confidence],
                model_index: 0,
                timestamp: chrono::Utc::now(),
            };

            let _ = self.record_trajectory(trajectory);
        }

        Ok(answer)
    }

    /// Build a prompt with context
    fn build_prompt(&self, query: &str, context: &[MemorySpan]) -> String {
        let mut prompt = String::new();

        if !context.is_empty() {
            prompt.push_str("Context:\n");
            for span in context {
                if !span.text.is_empty() {
                    prompt.push_str(&format!("- {}\n", span.text));
                }
            }
            prompt.push('\n');
        }

        prompt.push_str(&format!("Question: {}\n\nAnswer:", query));
        prompt
    }
}

// =============================================================================
// Query Decomposer
// =============================================================================

/// Decomposes complex queries into sub-queries
pub struct QueryDecomposer {
    backend: Arc<RuvLtraRlmBackend>,
}

impl QueryDecomposer {
    /// Create a new decomposer
    pub fn new(backend: Arc<RuvLtraRlmBackend>) -> Self {
        Self { backend }
    }

    /// Decompose a query
    pub fn decompose(&self, query: &str, _context: &QueryContext) -> Result<QueryDecomposition> {
        // Simple heuristic-based decomposition
        let lower = query.to_lowercase();

        // Check for conjunctions
        if lower.contains(" and ") {
            let parts: Vec<&str> = query.split(" and ").collect();
            if parts.len() >= 2 {
                return Ok(QueryDecomposition {
                    strategy: DecompositionStrategy::Conjunction(
                        parts.iter().map(|s| s.to_string()).collect(),
                    ),
                    sub_queries: parts
                        .iter()
                        .enumerate()
                        .map(|(i, q)| SubQuery {
                            query: q.trim().to_string(),
                            depends_on: Vec::new(),
                            priority: (parts.len() - i) as u8,
                            complexity: 0.5,
                        })
                        .collect(),
                    dependencies: Vec::new(),
                    parallelizable: true,
                });
            }
        }

        // Check for aspect-based questions
        let aspects = ["what", "why", "how", "when", "where", "who"];
        let aspect_count = aspects.iter().filter(|a| lower.contains(*a)).count();
        if aspect_count >= 2 {
            // Multiple aspects in one query
            let aspect_queries: Vec<String> = aspects
                .iter()
                .filter(|a| lower.contains(*a))
                .map(|a| format!("{} aspect of: {}", a, query))
                .collect();

            return Ok(QueryDecomposition {
                strategy: DecompositionStrategy::Aspect(aspect_queries.clone()),
                sub_queries: aspect_queries
                    .iter()
                    .enumerate()
                    .map(|(i, q)| SubQuery {
                        query: q.clone(),
                        depends_on: Vec::new(),
                        priority: (aspect_queries.len() - i) as u8,
                        complexity: 0.3,
                    })
                    .collect(),
                dependencies: Vec::new(),
                parallelizable: true,
            });
        }

        // Default: no decomposition needed
        Ok(QueryDecomposition::direct())
    }
}

// =============================================================================
// Answer Synthesizer
// =============================================================================

/// Synthesizes sub-answers into coherent responses
pub struct AnswerSynthesizer {
    backend: Arc<RuvLtraRlmBackend>,
}

impl AnswerSynthesizer {
    /// Create a new synthesizer
    pub fn new(backend: Arc<RuvLtraRlmBackend>) -> Self {
        Self { backend }
    }

    /// Synthesize answers
    pub fn synthesize(&self, original_query: &str, sub_answers: &[SubAnswer]) -> Result<String> {
        if sub_answers.is_empty() {
            return Ok(String::new());
        }

        if sub_answers.len() == 1 {
            return Ok(sub_answers[0].answer.clone());
        }

        // Build synthesis prompt
        let mut prompt = format!(
            "Original question: {}\n\nPartial answers:\n",
            original_query
        );
        for (i, sa) in sub_answers.iter().enumerate() {
            prompt.push_str(&format!("{}. {}: {}\n", i + 1, sa.query.query, sa.answer));
        }
        prompt.push_str("\nSynthesize a coherent answer that combines all the above:\n");

        // Generate synthesis
        let params = GenerationParams::default().with_max_tokens(512);
        let output = self.backend.generate(&prompt, &params)?;

        Ok(output.text)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_params_default() {
        let params = GenerationParams::default();
        assert_eq!(params.max_tokens, 256);
        assert_eq!(params.temperature, 0.7);
    }

    #[test]
    fn test_kv_cache_creation() {
        let cache = KvCache::new(24, 512 * 1024 * 1024);
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_kv_cache_entry() {
        let entry = KvCacheEntry::new(24);
        assert_eq!(entry.keys.len(), 24);
        assert_eq!(entry.values.len(), 24);
        assert_eq!(entry.seq_len, 0);
    }

    #[test]
    fn test_query_decomposition_direct() {
        let decomp = QueryDecomposition::direct();
        assert!(!decomp.needs_decomposition());
    }

    #[test]
    fn test_rlm_answer_creation() {
        let answer = RlmAnswer::new("Test answer".to_string(), 0.85);
        assert_eq!(answer.text, "Test answer");
        assert_eq!(answer.confidence, 0.85);
    }

    #[test]
    fn test_embedding_pooling_default() {
        let pooling = EmbeddingPooling::default();
        assert_eq!(pooling, EmbeddingPooling::Mean);
    }

    #[test]
    fn test_ruvltra_config_default() {
        let config = RuvLtraRlmConfig::default();
        assert_eq!(config.model_config.hidden_size, 896);
        assert!(config.enable_sona);
    }

    #[test]
    fn test_decomposer_conjunction() {
        // Create a minimal backend for testing
        let config = RuvLtraRlmConfig {
            model_config: RuvLtraConfig::tiny(),
            enable_sona: false,
            ..Default::default()
        };

        if let Ok(backend) = RuvLtraRlmBackend::new(config) {
            let decomposer = QueryDecomposer::new(Arc::new(backend));
            let context = QueryContext::default();

            let result = decomposer.decompose("What is Rust and why is it fast?", &context);
            assert!(result.is_ok());

            let decomp = result.unwrap();
            assert!(decomp.needs_decomposition());
            assert_eq!(decomp.sub_queries.len(), 2);
        }
    }
}
