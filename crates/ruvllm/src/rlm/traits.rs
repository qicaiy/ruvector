//! RLM Trait Definitions
//!
//! Core traits for the Recursive Language Model system:
//! - `LlmBackendTrait`: Abstraction for LLM inference engines
//! - `RlmEnvironment`: Environment trait for recursive reasoning
//! - `MemoryStore`: Memory storage abstraction

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

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
    /// Model decided to stop naturally
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
// RLM Configuration
// =============================================================================

/// Configuration for RLM processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmConfig {
    /// Maximum recursion depth (default: 5)
    pub max_depth: usize,
    /// Maximum sub-queries per level (default: 4)
    pub max_sub_queries: usize,
    /// Token budget for entire query chain
    pub token_budget: usize,
    /// Enable memoization cache
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Number of context chunks to retrieve
    pub retrieval_top_k: usize,
    /// Minimum quality score to accept answer
    pub min_quality_score: f32,
    /// Enable reflection loops
    pub enable_reflection: bool,
    /// Maximum reflection iterations
    pub max_reflection_iterations: usize,
    /// Parallelism for independent sub-queries
    pub parallel_sub_queries: bool,
    /// Enable SONA learning
    pub enable_sona_learning: bool,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_sub_queries: 4,
            token_budget: 16000,
            enable_cache: true,
            cache_ttl_secs: 3600,
            retrieval_top_k: 5,
            min_quality_score: 0.7,
            enable_reflection: true,
            max_reflection_iterations: 2,
            parallel_sub_queries: true,
            enable_sona_learning: true,
        }
    }
}

// =============================================================================
// LLM Backend Trait
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

/// Async version of the backend trait
#[allow(async_fn_in_trait)]
pub trait LlmBackendAsync: LlmBackendTrait {
    /// Generate completion asynchronously
    fn generate_async<'a>(
        &'a self,
        prompt: &'a str,
        params: &'a GenerationParams,
    ) -> Pin<Box<dyn Future<Output = Result<GenerationOutput>> + Send + 'a>>;

    /// Embed text asynchronously
    fn embed_async<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>>> + Send + 'a>>;
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
// RLM Environment Trait
// =============================================================================

/// RLM Environment - the sandbox in which recursive reasoning operates
///
/// This trait combines an LLM backend with a memory store and provides
/// the high-level operations needed for recursive language model execution.
pub trait RlmEnvironment: Send + Sync {
    /// Associated backend type
    type Backend: LlmBackendTrait;

    /// Associated memory store type
    type Memory: MemoryStore;

    /// Get the LLM backend
    fn backend(&self) -> &Self::Backend;

    /// Get the memory store
    fn memory(&self) -> &Self::Memory;

    /// Retrieve relevant context for a query
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<MemorySpan>>;

    /// Store a new memory span
    fn store_memory(&self, text: &str, metadata: MemoryMetadata) -> Result<MemoryId>;

    /// Decompose a complex query into sub-queries
    fn decompose_query(&self, query: &str, context: &QueryContext) -> Result<QueryDecomposition>;

    /// Synthesize partial answers into a coherent response
    fn synthesize_answers(&self, original_query: &str, sub_answers: &[SubAnswer])
        -> Result<String>;

    /// Main recursive query answering
    fn answer_query(&self, query: &str, depth: usize, config: &RlmConfig) -> Result<RlmAnswer>;
}

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
    fn test_generation_params_builder() {
        let params = GenerationParams::default()
            .with_max_tokens(512)
            .with_temperature(0.5);
        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.temperature, 0.5);
    }

    #[test]
    fn test_rlm_config_default() {
        let config = RlmConfig::default();
        assert_eq!(config.max_depth, 5);
        assert_eq!(config.max_sub_queries, 4);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_query_decomposition_direct() {
        let decomp = QueryDecomposition::direct();
        assert!(!decomp.needs_decomposition());
    }

    #[test]
    fn test_rlm_answer_builder() {
        let answer = RlmAnswer::new("Test answer".to_string(), 0.85).with_token_usage(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        });
        assert_eq!(answer.confidence, 0.85);
        assert_eq!(answer.token_usage.total_tokens, 30);
    }
}
