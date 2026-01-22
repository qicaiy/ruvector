//! RLM Value Objects and Domain Types
//!
//! This module defines the core value objects used throughout the RLM system,
//! following Domain-Driven Design principles for clear bounded contexts.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

// ============================================================================
// Memory Types
// ============================================================================

/// Unique identifier for a memory span
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// Create a new unique memory ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mem-{}", self.0)
    }
}

/// A span of memory retrieved from the vector store (ruvector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySpan {
    /// Unique identifier
    pub id: MemoryId,
    /// Text content of the memory
    pub text: String,
    /// Vector embedding
    pub embedding: Vec<f32>,
    /// Similarity score from retrieval (0.0 - 1.0)
    pub similarity_score: f32,
    /// Source identifier (e.g., document name, URL)
    pub source: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp when this memory was stored
    pub stored_at: DateTime<Utc>,
    /// Number of times this memory has been retrieved
    pub access_count: u32,
}

impl MemorySpan {
    /// Create a new memory span
    pub fn new(text: String, embedding: Vec<f32>) -> Self {
        Self {
            id: MemoryId::new(),
            text,
            embedding,
            similarity_score: 0.0,
            source: None,
            metadata: HashMap::new(),
            stored_at: Utc::now(),
            access_count: 0,
        }
    }

    /// Set the source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set similarity score
    pub fn with_score(mut self, score: f32) -> Self {
        self.similarity_score = score;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get estimated token count (rough estimate: ~4 chars per token)
    pub fn estimated_tokens(&self) -> usize {
        self.text.len() / 4
    }
}

/// Metadata for storing new memories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMetadata {
    /// Source identifier
    pub source: Option<String>,
    /// Content type (e.g., "text", "code", "conversation")
    pub content_type: Option<String>,
    /// Language (for code or text)
    pub language: Option<String>,
    /// Custom tags
    pub tags: Vec<String>,
    /// Custom key-value attributes
    pub attributes: HashMap<String, String>,
    /// Time-to-live in seconds (None = permanent)
    pub ttl_seconds: Option<u64>,
}

// ============================================================================
// Query Types
// ============================================================================

/// Global query ID counter
static QUERY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique identifier for a query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(pub u64);

impl QueryId {
    /// Generate a new unique query ID
    pub fn new() -> Self {
        Self(QUERY_COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for QueryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "q-{}", self.0)
    }
}

/// A query to be processed by the RLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Unique identifier
    pub id: QueryId,
    /// The query text
    pub text: String,
    /// Query embedding (computed lazily)
    pub embedding: Option<Vec<f32>>,
    /// Context for the query
    pub context: QueryContext,
    /// Constraints on the query processing
    pub constraints: QueryConstraints,
    /// Timestamp
    pub created_at: DateTime<Utc>,
}

impl Query {
    /// Create a new query
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: QueryId::new(),
            text: text.into(),
            embedding: None,
            context: QueryContext::default(),
            constraints: QueryConstraints::default(),
            created_at: Utc::now(),
        }
    }

    /// Set the embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the context
    pub fn with_context(mut self, context: QueryContext) -> Self {
        self.context = context;
        self
    }

    /// Set constraints
    pub fn with_constraints(mut self, constraints: QueryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Estimate token count
    pub fn estimated_tokens(&self) -> usize {
        self.text.len() / 4
    }
}

/// Context for a query
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryContext {
    /// Session ID if part of a conversation
    pub session_id: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// Previous queries in this session (for context)
    pub history: Vec<String>,
    /// Domain hints (e.g., "programming", "science")
    pub domain_hints: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Constraints on query processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConstraints {
    /// Maximum tokens to use for answer
    pub max_answer_tokens: usize,
    /// Maximum recursion depth
    pub max_depth: usize,
    /// Maximum sub-queries to generate
    pub max_sub_queries: usize,
    /// Require source citations
    pub require_citations: bool,
    /// Minimum quality threshold
    pub min_quality: f32,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

impl Default for QueryConstraints {
    fn default() -> Self {
        Self {
            max_answer_tokens: 2048,
            max_depth: 5,
            max_sub_queries: 4,
            require_citations: false,
            min_quality: 0.7,
            timeout_ms: None,
        }
    }
}

// ============================================================================
// Answer Types
// ============================================================================

/// Global answer ID counter
static ANSWER_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique identifier for an answer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AnswerId(pub u64);

impl AnswerId {
    /// Generate a new unique answer ID
    pub fn new() -> Self {
        Self(ANSWER_COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for AnswerId {
    fn default() -> Self {
        Self::new()
    }
}

/// A sub-answer from a decomposed query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAnswer {
    /// The sub-query that was answered
    pub sub_query: String,
    /// The answer text
    pub answer: String,
    /// Confidence in this answer (0.0 - 1.0)
    pub confidence: f32,
    /// Sources used for this answer
    pub sources: Vec<MemorySpan>,
    /// Tokens used
    pub tokens_used: usize,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Recursion depth at which this was generated
    pub depth: usize,
}

impl SubAnswer {
    /// Create a new sub-answer
    pub fn new(sub_query: impl Into<String>, answer: impl Into<String>) -> Self {
        Self {
            sub_query: sub_query.into(),
            answer: answer.into(),
            confidence: 1.0,
            sources: Vec::new(),
            tokens_used: 0,
            latency_ms: 0,
            depth: 0,
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add a source
    pub fn with_source(mut self, source: MemorySpan) -> Self {
        self.sources.push(source);
        self
    }

    /// Set sources
    pub fn with_sources(mut self, sources: Vec<MemorySpan>) -> Self {
        self.sources = sources;
        self
    }
}

/// Complete answer from the RLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmAnswer {
    /// Unique identifier
    pub id: AnswerId,
    /// The original query
    pub query: String,
    /// The synthesized answer text
    pub text: String,
    /// Confidence in the answer (0.0 - 1.0)
    pub confidence: f32,
    /// Quality score (0.0 - 1.0)
    pub quality_score: f32,
    /// Sources used (citations)
    pub sources: Vec<MemorySpan>,
    /// Sub-answers that were combined
    pub sub_answers: Vec<SubAnswer>,
    /// Total tokens used across all calls
    pub total_tokens: usize,
    /// Breakdown of token usage
    pub token_usage: TokenUsage,
    /// Maximum recursion depth reached
    pub max_depth_reached: usize,
    /// Total latency in milliseconds
    pub total_latency_ms: u64,
    /// Whether cache was used
    pub cache_hit: bool,
    /// Whether reflection was triggered
    pub reflection_triggered: bool,
    /// Number of reflection iterations
    pub reflection_iterations: usize,
    /// Timestamp
    pub created_at: DateTime<Utc>,
    /// Trajectory ID for learning
    pub trajectory_id: Option<crate::reasoning_bank::TrajectoryId>,
}

impl RlmAnswer {
    /// Create a new RLM answer
    pub fn new(query: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: AnswerId::new(),
            query: query.into(),
            text: text.into(),
            confidence: 1.0,
            quality_score: 1.0,
            sources: Vec::new(),
            sub_answers: Vec::new(),
            total_tokens: 0,
            token_usage: TokenUsage::default(),
            max_depth_reached: 0,
            total_latency_ms: 0,
            cache_hit: false,
            reflection_triggered: false,
            reflection_iterations: 0,
            created_at: Utc::now(),
            trajectory_id: None,
        }
    }

    /// Check if the answer is high quality
    pub fn is_high_quality(&self, threshold: f32) -> bool {
        self.quality_score >= threshold && self.confidence >= threshold
    }

    /// Get citation text
    pub fn format_citations(&self) -> String {
        if self.sources.is_empty() {
            return String::new();
        }

        let mut citations = String::from("\n\nSources:\n");
        for (i, source) in self.sources.iter().enumerate() {
            let source_name = source.source.as_deref().unwrap_or("Unknown");
            citations.push_str(&format!("[{}] {}\n", i + 1, source_name));
        }
        citations
    }
}

/// Token usage breakdown
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Tokens used for retrieval queries
    pub retrieval_tokens: usize,
    /// Tokens used for decomposition
    pub decomposition_tokens: usize,
    /// Tokens used for generation
    pub generation_tokens: usize,
    /// Tokens used for synthesis
    pub synthesis_tokens: usize,
    /// Tokens used for reflection
    pub reflection_tokens: usize,
    /// Total input tokens
    pub input_tokens: usize,
    /// Total output tokens
    pub output_tokens: usize,
}

impl TokenUsage {
    /// Get total tokens
    pub fn total(&self) -> usize {
        self.input_tokens + self.output_tokens
    }

    /// Add usage from another
    pub fn add(&mut self, other: &TokenUsage) {
        self.retrieval_tokens += other.retrieval_tokens;
        self.decomposition_tokens += other.decomposition_tokens;
        self.generation_tokens += other.generation_tokens;
        self.synthesis_tokens += other.synthesis_tokens;
        self.reflection_tokens += other.reflection_tokens;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

// ============================================================================
// Decomposition Types
// ============================================================================

/// Strategy for decomposing a complex query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// No decomposition needed - answer directly
    Direct,
    /// Split by conjunction ("and", "or")
    Conjunction {
        /// The conjunctions found
        conjunctions: Vec<String>,
    },
    /// Split by aspect (what, why, how, when, where)
    Aspect {
        /// The aspects to explore
        aspects: Vec<String>,
    },
    /// Sequential multi-step reasoning (step 1, then step 2, ...)
    Sequential {
        /// Steps in order
        steps: Vec<String>,
    },
    /// Parallel independent sub-questions
    Parallel {
        /// Independent sub-questions
        sub_questions: Vec<String>,
    },
    /// Hierarchical decomposition (main topic -> subtopics)
    Hierarchical {
        /// Main topic
        main_topic: String,
        /// Subtopics
        subtopics: Vec<String>,
    },
    /// Comparison decomposition (compare A vs B on criteria)
    Comparison {
        /// Items to compare
        items: Vec<String>,
        /// Criteria for comparison
        criteria: Vec<String>,
    },
}

impl DecompositionStrategy {
    /// Get the number of sub-queries implied by this strategy
    pub fn sub_query_count(&self) -> usize {
        match self {
            Self::Direct => 0,
            Self::Conjunction { conjunctions } => conjunctions.len(),
            Self::Aspect { aspects } => aspects.len(),
            Self::Sequential { steps } => steps.len(),
            Self::Parallel { sub_questions } => sub_questions.len(),
            Self::Hierarchical { subtopics, .. } => subtopics.len() + 1,
            Self::Comparison { items, criteria } => items.len() * criteria.len(),
        }
    }

    /// Check if this strategy requires sequential processing
    pub fn is_sequential(&self) -> bool {
        matches!(self, Self::Sequential { .. })
    }

    /// Check if this can be parallelized
    pub fn can_parallelize(&self) -> bool {
        matches!(
            self,
            Self::Parallel { .. } | Self::Aspect { .. } | Self::Comparison { .. }
        )
    }
}

/// Sub-query generated from decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuery {
    /// The sub-query text
    pub text: String,
    /// Index in the decomposition
    pub index: usize,
    /// Dependencies on other sub-queries (by index)
    pub depends_on: Vec<usize>,
    /// Priority (lower = higher priority)
    pub priority: usize,
    /// Estimated complexity (0.0 - 1.0)
    pub complexity: f32,
}

impl SubQuery {
    /// Create a new sub-query
    pub fn new(text: impl Into<String>, index: usize) -> Self {
        Self {
            text: text.into(),
            index,
            depends_on: Vec::new(),
            priority: 0,
            complexity: 0.5,
        }
    }

    /// Add a dependency
    pub fn with_dependency(mut self, dep_index: usize) -> Self {
        self.depends_on.push(dep_index);
        self
    }

    /// Check if this query can be executed (dependencies satisfied)
    pub fn can_execute(&self, completed: &[usize]) -> bool {
        self.depends_on.iter().all(|dep| completed.contains(dep))
    }
}

/// Result of query decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDecomposition {
    /// The original query
    pub original_query: String,
    /// Decomposition strategy chosen
    pub strategy: DecompositionStrategy,
    /// Generated sub-queries
    pub sub_queries: Vec<SubQuery>,
    /// Dependency graph edges (from, to)
    pub dependencies: Vec<(usize, usize)>,
    /// Confidence in the decomposition
    pub confidence: f32,
    /// Reasoning for the decomposition choice
    pub reasoning: Option<String>,
}

impl QueryDecomposition {
    /// Create a direct (no decomposition) result
    pub fn direct(query: impl Into<String>) -> Self {
        Self {
            original_query: query.into(),
            strategy: DecompositionStrategy::Direct,
            sub_queries: Vec::new(),
            dependencies: Vec::new(),
            confidence: 1.0,
            reasoning: Some("Query is simple enough to answer directly".to_string()),
        }
    }

    /// Check if decomposition is needed
    pub fn needs_decomposition(&self) -> bool {
        !matches!(self.strategy, DecompositionStrategy::Direct)
    }

    /// Get sub-queries that can be executed in parallel (no unmet dependencies)
    pub fn get_parallel_batch(&self, completed: &[usize]) -> Vec<&SubQuery> {
        self.sub_queries
            .iter()
            .filter(|sq| !completed.contains(&sq.index) && sq.can_execute(completed))
            .collect()
    }

    /// Get topologically sorted sub-queries
    pub fn topological_order(&self) -> Vec<usize> {
        let mut result = Vec::new();
        let mut visited = vec![false; self.sub_queries.len()];

        fn visit(
            index: usize,
            deps: &[(usize, usize)],
            visited: &mut [bool],
            result: &mut Vec<usize>,
        ) {
            if visited[index] {
                return;
            }
            visited[index] = true;

            for &(from, to) in deps {
                if to == index {
                    visit(from, deps, visited, result);
                }
            }
            result.push(index);
        }

        for i in 0..self.sub_queries.len() {
            visit(i, &self.dependencies, &mut visited, &mut result);
        }

        result
    }
}

// ============================================================================
// Generation Types
// ============================================================================

/// Reason why generation finished
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Reached max tokens
    MaxTokens,
    /// Reached a stop sequence
    StopSequence,
    /// Model decided to stop (EOS token)
    EndOfSequence,
    /// Timeout
    Timeout,
    /// Error occurred
    Error,
}

/// Output from a generation call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Reason for finishing
    pub finish_reason: FinishReason,
    /// Log probabilities (if requested)
    pub logprobs: Option<Vec<f32>>,
    /// Latency in milliseconds
    pub latency_ms: u64,
}

impl GenerationOutput {
    /// Create a new generation output
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens_generated: 0,
            finish_reason: FinishReason::EndOfSequence,
            logprobs: None,
            latency_ms: 0,
        }
    }
}

/// Parameters for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy, 1.0 = creative)
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Whether to return log probabilities
    pub return_logprobs: bool,
    /// Number of log probabilities to return per token
    pub logprobs_count: usize,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 0,
            stop_sequences: Vec::new(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            return_logprobs: false,
            logprobs_count: 0,
        }
    }
}

impl GenerationParams {
    /// Create params for deterministic generation
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        }
    }

    /// Create params for creative generation
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            ..Default::default()
        }
    }
}

// ============================================================================
// Model Information
// ============================================================================

/// Information about an LLM model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model family (e.g., "llama", "mistral", "phi")
    pub family: String,
    /// Parameter count
    pub parameters: u64,
    /// Maximum context length
    pub max_context: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Whether the model supports embeddings
    pub supports_embedding: bool,
    /// Quantization type if quantized
    pub quantization: Option<String>,
}

impl ModelInfo {
    /// Create basic model info
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            family: String::new(),
            parameters: 0,
            max_context: 4096,
            vocab_size: 32000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            supports_embedding: false,
            quantization: None,
        }
    }

    /// Get approximate memory requirement in bytes
    pub fn estimated_memory_bytes(&self) -> u64 {
        let bytes_per_param = match &self.quantization {
            Some(q) if q.contains("q4") => 0.5,
            Some(q) if q.contains("q8") => 1.0,
            Some(q) if q.contains("f16") => 2.0,
            _ => 4.0,
        };
        (self.parameters as f64 * bytes_per_param) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_id_uniqueness() {
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_memory_span_creation() {
        let span = MemorySpan::new("test text".to_string(), vec![0.1; 768])
            .with_source("test.txt")
            .with_score(0.95);

        assert_eq!(span.text, "test text");
        assert_eq!(span.source, Some("test.txt".to_string()));
        assert_eq!(span.similarity_score, 0.95);
    }

    #[test]
    fn test_query_creation() {
        let query = Query::new("What is RLM?");
        assert!(!query.text.is_empty());
        assert!(query.embedding.is_none());
    }

    #[test]
    fn test_decomposition_strategy() {
        let strategy = DecompositionStrategy::Parallel {
            sub_questions: vec!["q1".to_string(), "q2".to_string()],
        };
        assert_eq!(strategy.sub_query_count(), 2);
        assert!(strategy.can_parallelize());
        assert!(!strategy.is_sequential());
    }

    #[test]
    fn test_sub_query_dependencies() {
        let sq = SubQuery::new("test", 0)
            .with_dependency(1)
            .with_dependency(2);
        assert!(!sq.can_execute(&[]));
        assert!(!sq.can_execute(&[1]));
        assert!(sq.can_execute(&[1, 2]));
    }

    #[test]
    fn test_token_usage_addition() {
        let mut usage1 = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let usage2 = TokenUsage {
            input_tokens: 200,
            output_tokens: 100,
            ..Default::default()
        };
        usage1.add(&usage2);
        assert_eq!(usage1.total(), 450);
    }

    #[test]
    fn test_generation_params_defaults() {
        let params = GenerationParams::default();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.95);

        let deterministic = GenerationParams::deterministic();
        assert_eq!(deterministic.temperature, 0.0);
    }
}
