//! RLM Error Types
//!
//! Comprehensive error handling for Recursive Language Model operations.

use thiserror::Error;

/// Result type alias for RLM operations
pub type RlmResult<T> = std::result::Result<T, RlmError>;

/// RLM-specific error types
#[derive(Error, Debug)]
pub enum RlmError {
    /// Query decomposition failed
    #[error("Decomposition error: {0}")]
    Decomposition(String),

    /// Answer synthesis failed
    #[error("Synthesis error: {0}")]
    Synthesis(String),

    /// Maximum recursion depth exceeded
    #[error("Maximum recursion depth exceeded: {depth}/{max_depth}")]
    MaxRecursionDepthExceeded {
        /// Current depth
        depth: usize,
        /// Maximum allowed depth
        max_depth: usize,
    },

    /// Token budget exhausted
    #[error("Token budget exhausted: consumed {consumed}, budget {budget}")]
    TokenBudgetExhausted {
        /// Tokens consumed
        consumed: usize,
        /// Total budget
        budget: usize,
    },

    /// Memory retrieval failed
    #[error("Memory retrieval error: {0}")]
    MemoryRetrieval(String),

    /// Memory storage failed
    #[error("Memory storage error: {0}")]
    MemoryStorage(String),

    /// LLM generation failed
    #[error("Generation error: {0}")]
    Generation(String),

    /// Embedding generation failed
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Embedding not supported by backend
    #[error("Embedding not supported by this backend")]
    EmbeddingNotSupported,

    /// Quality score below threshold
    #[error("Quality score {score:.2} below threshold {threshold:.2}")]
    QualityBelowThreshold {
        /// Actual quality score
        score: f32,
        /// Required threshold
        threshold: f32,
    },

    /// Reflection loop failed to improve answer
    #[error("Reflection failed to improve answer after {iterations} iterations")]
    ReflectionFailed {
        /// Number of iterations attempted
        iterations: usize,
    },

    /// Query not found in session
    #[error("Query not found: {query_id}")]
    QueryNotFound {
        /// The query ID that was not found
        query_id: String,
    },

    /// Trajectory not found
    #[error("Trajectory not found: {trajectory_id}")]
    TrajectoryNotFound {
        /// The trajectory ID that was not found
        trajectory_id: String,
    },

    /// Session expired or invalid
    #[error("Session expired or invalid: {session_id}")]
    SessionExpired {
        /// The session ID
        session_id: String,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Backend not initialized
    #[error("Backend not initialized")]
    BackendNotInitialized,

    /// Cycle detected in query decomposition
    #[error("Cycle detected in query decomposition")]
    CycleDetected,

    /// Timeout during operation
    #[error("Operation timed out after {duration_ms}ms")]
    Timeout {
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// Invalid query format
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapped RuvLLM error
    #[error("RuvLLM error: {0}")]
    RuvLLM(String),
}

impl From<crate::error::RuvLLMError> for RlmError {
    fn from(err: crate::error::RuvLLMError) -> Self {
        RlmError::RuvLLM(err.to_string())
    }
}

impl From<serde_json::Error> for RlmError {
    fn from(err: serde_json::Error) -> Self {
        RlmError::Serialization(err.to_string())
    }
}
