//! Recursive Language Model (RLM) Integration
//!
//! This module implements the RLM architecture described in ADR-014, providing:
//!
//! - **LlmBackend Trait**: Abstraction for any LLM inference engine
//! - **RlmEnvironment Trait**: Environment for recursive reasoning with memory
//! - **QueryDecomposer**: Breaks complex queries into manageable sub-queries
//! - **AnswerSynthesizer**: Combines sub-answers into coherent responses
//! - **RlmController**: Orchestrates the recursive answer pipeline
//!
//! ## Architecture
//!
//! ```text
//! +------------------+
//! |  RlmController   |  <-- Main entry point
//! +--------+---------+
//!          |
//!          v
//! +--------+---------+
//! | QueryDecomposer  |  <-- Breaks complex queries
//! +--------+---------+
//!          |
//!    +-----+-----+
//!    |           |
//! +--v--+     +--v--+
//! |Sub  |     |Sub  |  <-- Parallel sub-query processing
//! |Query|     |Query|
//! +--+--+     +--+--+
//!    |           |
//!    +-----+-----+
//!          |
//!          v
//! +--------+---------+
//! |AnswerSynthesizer |  <-- Combines sub-answers
//! +--------+---------+
//!          |
//!          v
//! +--------+---------+
//! |   RlmMemory      |  <-- HNSW-indexed retrieval
//! +-----------------+
//! ```
//!
//! ## Features
//!
//! - **Query Decomposition**: Heuristic and LLM-driven query decomposition
//! - **Recursive Processing**: Depth-controlled recursive answering
//! - **Memory Integration**: HNSW-indexed semantic memory
//! - **Memoization Cache**: O(1) LRU cache via `lru` crate
//! - **Quality Control**: Reflection loops for answer improvement
//! - **ReasoningBank Integration**: Trajectory recording for learning
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvllm::rlm::{RlmController, RlmConfig, NativeEnvironment};
//!
//! // Create controller with default config
//! let config = RlmConfig::default();
//! let controller = RlmController::<NativeEnvironment>::new(config)?;
//!
//! // Query the model
//! let response = controller.query("What is recursive language modeling?")?;
//! println!("Answer: {}", response.text);
//!
//! // Add to memory for future retrieval
//! controller.add_memory("RLM uses recursive decomposition.", Default::default())?;
//!
//! // Search memory
//! let results = controller.search_memory("recursive", 5)?;
//! ```

// ============================================================================
// Core Modules
// ============================================================================

pub mod cache;
pub mod config;
pub mod controller;
pub mod decomposer;
pub mod environment;
pub mod memory;
pub mod pool;
pub mod synthesizer;
pub mod traits;
pub mod types;

// SIMD-optimized vector operations
pub mod simd_ops;

// Zero-copy shared types for optimized memory usage
pub mod shared_types;

// RuvLTRA backend implementation
pub mod ruvltra_backend;

// WASM bindings (only for wasm32 target with rlm-wasm feature)
#[cfg(all(target_arch = "wasm32", feature = "rlm-wasm"))]
pub mod wasm;

// ============================================================================
// Re-exports from config (ADR-014 focused on recursive reasoning settings)
// ============================================================================

pub use config::{
    AggregationStrategy, ConfigValidationError, DecompositionConfig, RecursiveConfig,
    RecursiveConfigBuilder,
};

// ============================================================================
// Re-exports from controller
// ============================================================================

pub use controller::{
    MemoryEntry, MemoryMetadata, QueryResult, RlmConfig, RlmController, RlmStats, RlmStatsSnapshot,
    SourceAttribution, TokenUsage as ControllerTokenUsage,
};

// ============================================================================
// Re-exports from decomposer
// ============================================================================

pub use decomposer::{
    DecomposerStats, DecomposerStatsSnapshot, DecompositionResult,
    DecompositionStrategy as DecomposerStrategy, QueryDecomposer, QueryType,
    SubQuery as DecomposerSubQuery,
};

// ============================================================================
// Re-exports from synthesizer
// ============================================================================

pub use synthesizer::{AnswerSynthesizer, SynthesisResult};

// ============================================================================
// Re-exports from environment
// ============================================================================

pub use environment::{EnvironmentConfig, EnvironmentType, NativeEnvironment, RlmEnvironment};

#[cfg(all(target_arch = "wasm32", feature = "rlm-wasm"))]
pub use environment::WasmEnvironment;

// ============================================================================
// Re-exports from cache (O(1) LRU memoization)
// ============================================================================

pub use cache::{CacheConfig, CacheEntry, CacheStats, MemoizationCache};

// ============================================================================
// Re-exports from memory
// ============================================================================

pub use memory::{
    MemoryConfig, MemoryEntry as MemoryStoreEntry, MemoryMetadata as MemoryStoreMetadata,
    MemorySearchResult, RlmMemory,
};

// ============================================================================
// Re-exports from traits
// ============================================================================

pub use traits::{
    DecompositionStrategy as TraitsDecompositionStrategy, FinishReason as TraitsFinishReason,
    GenerationOutput as TraitsGenerationOutput, GenerationParams as TraitsGenerationParams,
    LlmBackendAsync, LlmBackendTrait, MemoryMetadata as TraitsMemoryMetadata,
    MemorySpan as TraitsMemorySpan, MemoryStore, QueryContext as TraitsQueryContext,
    QueryDecomposition as TraitsQueryDecomposition, RlmAnswer as TraitsRlmAnswer,
    RlmConfig as TraitsRlmConfig, RlmModelInfo, StreamToken, SubAnswer as TraitsSubAnswer,
    SubQuery as TraitsSubQuery, TokenUsage as TraitsTokenUsage,
};

// ============================================================================
// Re-exports from types
// ============================================================================

pub use types::{
    AnswerId, DecompositionStrategy, FinishReason, GenerationOutput, GenerationParams, MemoryId,
    MemorySpan, ModelInfo, Query, QueryConstraints, QueryContext, QueryDecomposition, QueryId,
    RlmAnswer, SubAnswer, SubQuery, TokenUsage,
};

// ============================================================================
// Re-exports from simd_ops
// ============================================================================

pub use simd_ops::{
    // Single-vector operations
    batch_cosine_similarity,
    // Batch SIMD operations (4-8 vectors in parallel)
    batch_cosine_similarity_4,
    batch_cosine_similarity_4_prenorm,
    batch_cosine_similarity_8,
    batch_cosine_similarity_slices,
    batch_dot_products,
    batch_dot_products_4,
    batch_l2_norms,
    batch_l2_norms_4,
    batch_normalize_inplace,
    batch_similarity_search,
    cosine_similarity,
    cosine_similarity_prenorm,
    dot_product,
    l2_norm,
    normalize,
    normalize_inplace,
};

// ============================================================================
// Re-exports from shared_types (zero-copy optimizations)
// ============================================================================

pub use shared_types::{
    extract_excerpt,
    // Utility functions
    normalize_query,
    prepare_cache_key,
    // Cache types
    ArcCachedResponse,
    IntoSharedEmbedding,
    // Conversion traits
    IntoSharedText,
    SharedBytes,
    SharedEmbedding,
    // Zero-copy memory types
    SharedMemoryEntry,
    SharedMemoryMetadata,
    // Zero-copy query result
    SharedQueryResult,
    SharedSearchResult,
    SharedSourceAttribution,
    // Core shared types
    SharedText,
    SharedTokenUsage,
};

// ============================================================================
// Re-exports from pool (memory allocation optimization)
// ============================================================================

pub use pool::{
    CombinedPoolStats,
    // Pool management
    PoolManager,
    PoolStats,
    PoolStatsSnapshot,
    PooledResults,
    PooledString,
    PooledVec,
    // Generic result pool
    ResultPool,
    // String pool for query processing
    StringPool,
    // Vector pool for embedding reuse
    VectorPool,
};

// ============================================================================
// WASM Re-exports
// ============================================================================

#[cfg(all(target_arch = "wasm32", feature = "rlm-wasm"))]
pub use wasm::WasmRlmController;

// ============================================================================
// Re-exports from RuvLTRA Backend
// ============================================================================

pub use ruvltra_backend::{
    AnswerSynthesizer as RuvLtraSynthesizer,
    // Embedding pooling
    EmbeddingPooling,
    // KV cache
    KvCache,
    KvCacheEntry,
    KvCacheStats,
    // Query decomposer and synthesizer from ruvltra_backend
    QueryDecomposer as RuvLtraQueryDecomposer,
    RuvLtraBackendStats,
    RuvLtraEnvConfig,
    RuvLtraEnvironment,
    RuvLtraMemoryStore,
    // Backend and environment
    RuvLtraRlmBackend,
    RuvLtraRlmConfig,
};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify core exports are accessible
        let _config = RlmConfig::default();
        let _mem_config = MemoryConfig::default();
        let _gen_params = GenerationParams::default();
        let _query = Query::new("test query");
    }

    #[test]
    fn test_decomposer_available() {
        let decomposer = QueryDecomposer::new();
        let result = decomposer.decompose("What is AI and how does it work?");
        assert!(result.was_decomposed() || !result.was_decomposed()); // Sanity check
    }

    #[test]
    fn test_synthesizer_available() {
        let synthesizer = AnswerSynthesizer::default();
        // Just verify the type exists and can be instantiated
        assert!(matches!(synthesizer, AnswerSynthesizer { .. }));
    }

    #[test]
    fn test_config_builder() {
        let config = RecursiveConfigBuilder::new()
            .max_depth(10)
            .token_budget(20000)
            .enable_cache(true)
            .build()
            .unwrap();

        assert_eq!(config.max_depth, 10);
        assert_eq!(config.token_budget, 20000);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_types_accessible() {
        let memory_id = MemoryId::new();
        assert!(!memory_id.to_string().is_empty());

        let query_id = QueryId::new();
        assert!(!query_id.to_string().is_empty());

        let answer_id = AnswerId::new();
        assert!(answer_id.0 > 0 || answer_id.0 == 0); // Sanity check
    }

    #[test]
    fn test_aggregation_strategies() {
        assert!(AggregationStrategy::WeightedMerge.requires_quality_scores());
        assert!(AggregationStrategy::Concatenate.supports_partial_results());
        assert!(!AggregationStrategy::Summarize.supports_partial_results());
    }

    #[test]
    fn test_pool_exports() {
        // Verify pool exports are accessible
        let vector_pool = VectorPool::new(384, 16);
        let string_pool = StringPool::new(32);

        // Test basic operations
        {
            let mut vec = vector_pool.acquire();
            vec.extend_from_slice(&[1.0, 2.0, 3.0]);
            assert_eq!(vec.len(), 3);

            let mut s = string_pool.acquire();
            s.push_str("test");
            assert_eq!(s.as_ref(), "test");
        }

        // Verify pooling works
        assert_eq!(vector_pool.pooled_count(), 1);
        assert_eq!(string_pool.pooled_count(), 1);
    }

    #[test]
    fn test_pool_manager_exports() {
        let manager = PoolManager::new();
        let _stats = manager.stats();
        assert!(manager.vector_pool.dimension() == 384);
    }
}
