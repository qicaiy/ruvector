//! RLM Configuration
//!
//! This module defines the configuration options for the Recursive Language Model,
//! including recursion limits, caching, retrieval settings, and reflection parameters.

use serde::{Deserialize, Serialize};

/// Configuration for recursive reasoning settings in the RLM Controller
///
/// This config focuses on recursion depth, decomposition strategies, quality thresholds,
/// and other parameters related to the recursive answer generation pipeline.
///
/// For controller-level settings like embedding dimensions and memory configuration,
/// see `controller::RlmConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveConfig {
    // ========================================================================
    // Recursion Control
    // ========================================================================
    /// Maximum recursion depth (default: 5)
    ///
    /// Higher values allow more complex query decomposition but increase latency.
    /// Recommended range: 3-10
    pub max_depth: usize,

    /// Maximum sub-queries per level (default: 4)
    ///
    /// Limits the branching factor to prevent exponential blowup.
    /// Recommended range: 2-6
    pub max_sub_queries: usize,

    /// Enable parallel execution of independent sub-queries
    pub parallel_sub_queries: bool,

    /// Maximum concurrent sub-query executions
    pub max_parallel_concurrent: usize,

    // ========================================================================
    // Token Budget
    // ========================================================================
    /// Total token budget for entire query chain (default: 16000)
    ///
    /// This is the maximum total tokens across all recursive calls.
    pub token_budget: usize,

    /// Reserved tokens for synthesis step
    pub synthesis_token_reserve: usize,

    /// Reserved tokens for reflection (if enabled)
    pub reflection_token_reserve: usize,

    /// Early termination when budget drops below this threshold
    pub budget_termination_threshold: usize,

    // ========================================================================
    // Caching
    // ========================================================================
    /// Enable memoization cache for repeated queries
    pub enable_cache: bool,

    /// Cache time-to-live in seconds (default: 3600 = 1 hour)
    pub cache_ttl_secs: u64,

    /// Maximum cache entries
    pub max_cache_entries: usize,

    /// Enable fuzzy cache matching (similar queries may hit cache)
    pub fuzzy_cache_matching: bool,

    /// Similarity threshold for fuzzy cache matching (0.0 - 1.0)
    pub fuzzy_cache_threshold: f32,

    // ========================================================================
    // Retrieval Settings
    // ========================================================================
    /// Number of context chunks to retrieve per query (default: 5)
    pub retrieval_top_k: usize,

    /// Minimum similarity score for retrieved chunks (0.0 - 1.0)
    pub retrieval_min_similarity: f32,

    /// Maximum total retrieved chunks across all recursive calls
    pub max_total_retrievals: usize,

    /// Enable retrieval reranking
    pub enable_reranking: bool,

    // ========================================================================
    // Quality & Reflection
    // ========================================================================
    /// Minimum quality score to accept answer (0.0 - 1.0)
    pub min_quality_score: f32,

    /// Enable reflection loops for low-quality answers
    pub enable_reflection: bool,

    /// Maximum reflection iterations (default: 2)
    pub max_reflection_iterations: usize,

    /// Quality improvement threshold to continue reflection
    pub reflection_improvement_threshold: f32,

    // ========================================================================
    // Decomposition Settings
    // ========================================================================
    /// Enable LLM-driven decomposition (vs. heuristic only)
    pub llm_decomposition: bool,

    /// Complexity threshold for LLM decomposition (0.0 - 1.0)
    /// Queries below this complexity use heuristics only
    pub llm_decomposition_threshold: f32,

    /// Minimum query length for decomposition consideration
    pub min_decomposition_length: usize,

    /// Decomposition configuration
    pub decomposition: DecompositionConfig,

    /// Strategy for aggregating sub-query answers
    pub aggregation_strategy: AggregationStrategy,

    // ========================================================================
    // Learning Integration
    // ========================================================================
    /// Record trajectories to ReasoningBank
    pub record_trajectories: bool,

    /// Store successful patterns
    pub store_patterns: bool,

    /// Minimum quality for pattern storage
    pub pattern_store_threshold: f32,

    // ========================================================================
    // Timeouts
    // ========================================================================
    /// Per-query timeout in milliseconds (0 = no timeout)
    pub query_timeout_ms: u64,

    /// Per-sub-query timeout in milliseconds
    pub sub_query_timeout_ms: u64,

    /// Retrieval timeout in milliseconds
    pub retrieval_timeout_ms: u64,

    // ========================================================================
    // Generation Settings
    // ========================================================================
    /// Default temperature for generation
    pub default_temperature: f32,

    /// Temperature for decomposition prompts (typically lower)
    pub decomposition_temperature: f32,

    /// Temperature for synthesis prompts
    pub synthesis_temperature: f32,
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self {
            // Recursion Control
            max_depth: 5,
            max_sub_queries: 4,
            parallel_sub_queries: true,
            max_parallel_concurrent: 4,

            // Token Budget
            token_budget: 16000,
            synthesis_token_reserve: 2000,
            reflection_token_reserve: 1500,
            budget_termination_threshold: 500,

            // Caching
            enable_cache: true,
            cache_ttl_secs: 3600,
            max_cache_entries: 10000,
            fuzzy_cache_matching: true,
            fuzzy_cache_threshold: 0.95,

            // Retrieval
            retrieval_top_k: 5,
            retrieval_min_similarity: 0.5,
            max_total_retrievals: 50,
            enable_reranking: false,

            // Quality & Reflection
            min_quality_score: 0.7,
            enable_reflection: true,
            max_reflection_iterations: 2,
            reflection_improvement_threshold: 0.05,

            // Decomposition
            llm_decomposition: true,
            llm_decomposition_threshold: 0.6,
            min_decomposition_length: 20,
            decomposition: DecompositionConfig::default(),
            aggregation_strategy: AggregationStrategy::WeightedMerge,

            // Learning
            record_trajectories: true,
            store_patterns: true,
            pattern_store_threshold: 0.8,

            // Timeouts
            query_timeout_ms: 30000,
            sub_query_timeout_ms: 10000,
            retrieval_timeout_ms: 2000,

            // Generation
            default_temperature: 0.7,
            decomposition_temperature: 0.3,
            synthesis_temperature: 0.5,
        }
    }
}

impl RecursiveConfig {
    /// Create a config optimized for high quality
    pub fn high_quality() -> Self {
        Self {
            max_depth: 7,
            max_sub_queries: 6,
            token_budget: 32000,
            min_quality_score: 0.8,
            enable_reflection: true,
            max_reflection_iterations: 3,
            retrieval_top_k: 10,
            enable_reranking: true,
            ..Default::default()
        }
    }

    /// Create a config optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_depth: 3,
            max_sub_queries: 2,
            token_budget: 8000,
            parallel_sub_queries: true,
            max_parallel_concurrent: 2,
            enable_reflection: false,
            retrieval_top_k: 3,
            llm_decomposition: false,
            query_timeout_ms: 10000,
            sub_query_timeout_ms: 3000,
            ..Default::default()
        }
    }

    /// Create a config optimized for minimal token usage
    pub fn token_efficient() -> Self {
        Self {
            max_depth: 4,
            max_sub_queries: 3,
            token_budget: 8000,
            synthesis_token_reserve: 1000,
            reflection_token_reserve: 0,
            enable_reflection: false,
            retrieval_top_k: 3,
            max_total_retrievals: 20,
            llm_decomposition: false,
            ..Default::default()
        }
    }

    /// Create a config for simple queries (no decomposition)
    pub fn simple() -> Self {
        Self {
            max_depth: 1,
            max_sub_queries: 0,
            enable_reflection: false,
            llm_decomposition: false,
            min_decomposition_length: usize::MAX,
            ..Default::default()
        }
    }

    /// Calculate remaining budget after reserves
    pub fn available_budget(&self) -> usize {
        let reserves = self.synthesis_token_reserve
            + if self.enable_reflection {
                self.reflection_token_reserve
            } else {
                0
            };
        self.token_budget.saturating_sub(reserves)
    }

    /// Calculate per-level budget
    pub fn per_level_budget(&self) -> usize {
        self.available_budget() / self.max_depth.max(1)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.max_depth == 0 {
            return Err(ConfigValidationError::InvalidValue {
                field: "max_depth".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }

        if self.token_budget < 1000 {
            return Err(ConfigValidationError::InvalidValue {
                field: "token_budget".to_string(),
                reason: "must be at least 1000".to_string(),
            });
        }

        if self.min_quality_score < 0.0 || self.min_quality_score > 1.0 {
            return Err(ConfigValidationError::InvalidValue {
                field: "min_quality_score".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        if self.fuzzy_cache_matching && self.fuzzy_cache_threshold < 0.5 {
            return Err(ConfigValidationError::InvalidValue {
                field: "fuzzy_cache_threshold".to_string(),
                reason: "must be at least 0.5 for fuzzy matching".to_string(),
            });
        }

        let reserves = self.synthesis_token_reserve + self.reflection_token_reserve;
        if reserves >= self.token_budget {
            return Err(ConfigValidationError::InvalidValue {
                field: "reserves".to_string(),
                reason: "token reserves exceed total budget".to_string(),
            });
        }

        Ok(())
    }
}

/// Configuration for query decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionConfig {
    /// Keywords that trigger conjunction decomposition
    pub conjunction_keywords: Vec<String>,

    /// Keywords that trigger comparison decomposition
    pub comparison_keywords: Vec<String>,

    /// Keywords that trigger sequential decomposition
    pub sequential_keywords: Vec<String>,

    /// Minimum query length to attempt decomposition
    pub min_decomposition_length: usize,

    /// Maximum complexity score for direct answering
    pub direct_complexity_threshold: f32,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            conjunction_keywords: vec![
                "and".to_string(),
                "also".to_string(),
                "as well as".to_string(),
                "along with".to_string(),
                "in addition".to_string(),
            ],
            comparison_keywords: vec![
                "compare".to_string(),
                "versus".to_string(),
                "vs".to_string(),
                "difference".to_string(),
                "contrast".to_string(),
                "similarities".to_string(),
            ],
            sequential_keywords: vec![
                "then".to_string(),
                "after".to_string(),
                "next".to_string(),
                "followed by".to_string(),
                "subsequently".to_string(),
                "first".to_string(),
                "finally".to_string(),
            ],
            min_decomposition_length: 20,
            direct_complexity_threshold: 0.3,
        }
    }
}

/// Strategy for aggregating sub-query answers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple concatenation of answers
    Concatenate,

    /// Weighted merge based on quality scores
    WeightedMerge,

    /// Hierarchical summarization
    Summarize,

    /// Best answer selection based on quality
    BestAnswer,

    /// Voting-based consensus for multiple answers
    Consensus,
}

impl AggregationStrategy {
    /// Returns true if this strategy requires quality scores
    pub fn requires_quality_scores(&self) -> bool {
        matches!(
            self,
            Self::WeightedMerge | Self::BestAnswer | Self::Consensus
        )
    }

    /// Returns true if this strategy can handle partial results
    pub fn supports_partial_results(&self) -> bool {
        matches!(self, Self::Concatenate | Self::BestAnswer)
    }
}

/// Configuration validation error
#[derive(Debug, Clone)]
pub enum ConfigValidationError {
    /// Invalid field value
    InvalidValue {
        /// Field name
        field: String,
        /// Reason for invalidity
        reason: String,
    },
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValue { field, reason } => {
                write!(f, "Invalid config value for '{}': {}", field, reason)
            }
        }
    }
}

impl std::error::Error for ConfigValidationError {}

/// Builder for RecursiveConfig
#[derive(Debug, Default)]
pub struct RecursiveConfigBuilder {
    config: RecursiveConfig,
}

impl RecursiveConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum recursion depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth;
        self
    }

    /// Set maximum sub-queries per level
    pub fn max_sub_queries(mut self, count: usize) -> Self {
        self.config.max_sub_queries = count;
        self
    }

    /// Set token budget
    pub fn token_budget(mut self, budget: usize) -> Self {
        self.config.token_budget = budget;
        self
    }

    /// Enable or disable caching
    pub fn enable_cache(mut self, enable: bool) -> Self {
        self.config.enable_cache = enable;
        self
    }

    /// Set cache TTL
    pub fn cache_ttl(mut self, ttl_secs: u64) -> Self {
        self.config.cache_ttl_secs = ttl_secs;
        self
    }

    /// Set retrieval top-k
    pub fn retrieval_top_k(mut self, k: usize) -> Self {
        self.config.retrieval_top_k = k;
        self
    }

    /// Set minimum quality score
    pub fn min_quality(mut self, score: f32) -> Self {
        self.config.min_quality_score = score;
        self
    }

    /// Enable or disable reflection
    pub fn enable_reflection(mut self, enable: bool) -> Self {
        self.config.enable_reflection = enable;
        self
    }

    /// Set maximum reflection iterations
    pub fn max_reflection_iterations(mut self, iterations: usize) -> Self {
        self.config.max_reflection_iterations = iterations;
        self
    }

    /// Enable or disable parallel sub-queries
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel_sub_queries = enable;
        self
    }

    /// Enable or disable LLM decomposition
    pub fn llm_decomposition(mut self, enable: bool) -> Self {
        self.config.llm_decomposition = enable;
        self
    }

    /// Enable or disable trajectory recording
    pub fn record_trajectories(mut self, enable: bool) -> Self {
        self.config.record_trajectories = enable;
        self
    }

    /// Set query timeout
    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.config.query_timeout_ms = timeout_ms;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<RecursiveConfig, ConfigValidationError> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation
    pub fn build_unchecked(self) -> RecursiveConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RecursiveConfig::default();
        assert_eq!(config.max_depth, 5);
        assert_eq!(config.max_sub_queries, 4);
        assert!(config.enable_cache);
        assert!(config.enable_reflection);
    }

    #[test]
    fn test_config_validation() {
        let mut config = RecursiveConfig::default();
        assert!(config.validate().is_ok());

        config.max_depth = 0;
        assert!(config.validate().is_err());

        config.max_depth = 5;
        config.min_quality_score = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_presets() {
        let hq = RecursiveConfig::high_quality();
        assert!(hq.max_depth > RecursiveConfig::default().max_depth);
        assert!(hq.enable_reflection);

        let ll = RecursiveConfig::low_latency();
        assert!(ll.max_depth < RecursiveConfig::default().max_depth);
        assert!(!ll.enable_reflection);

        let simple = RecursiveConfig::simple();
        assert_eq!(simple.max_depth, 1);
        assert_eq!(simple.max_sub_queries, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = RecursiveConfigBuilder::new()
            .max_depth(10)
            .token_budget(50000)
            .enable_reflection(false)
            .build()
            .unwrap();

        assert_eq!(config.max_depth, 10);
        assert_eq!(config.token_budget, 50000);
        assert!(!config.enable_reflection);
    }

    #[test]
    fn test_available_budget() {
        let config = RecursiveConfig {
            token_budget: 10000,
            synthesis_token_reserve: 2000,
            reflection_token_reserve: 1500,
            enable_reflection: true,
            ..Default::default()
        };
        assert_eq!(config.available_budget(), 6500);

        let config_no_reflect = RecursiveConfig {
            token_budget: 10000,
            synthesis_token_reserve: 2000,
            reflection_token_reserve: 1500,
            enable_reflection: false,
            ..Default::default()
        };
        assert_eq!(config_no_reflect.available_budget(), 8000);
    }

    #[test]
    fn test_decomposition_config_default() {
        let config = DecompositionConfig::default();
        assert!(config.conjunction_keywords.contains(&"and".to_string()));
        assert!(config.comparison_keywords.contains(&"compare".to_string()));
        assert!(config.sequential_keywords.contains(&"then".to_string()));
        assert_eq!(config.min_decomposition_length, 20);
    }

    #[test]
    fn test_aggregation_strategy_properties() {
        assert!(AggregationStrategy::WeightedMerge.requires_quality_scores());
        assert!(AggregationStrategy::BestAnswer.requires_quality_scores());
        assert!(!AggregationStrategy::Concatenate.requires_quality_scores());

        assert!(AggregationStrategy::Concatenate.supports_partial_results());
        assert!(AggregationStrategy::BestAnswer.supports_partial_results());
        assert!(!AggregationStrategy::Summarize.supports_partial_results());
    }
}
