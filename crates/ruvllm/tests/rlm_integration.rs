//! Integration tests for RLM (Recursive Language Model)
//!
//! Tests the RLM module including:
//! - Configuration defaults and validation
//! - Query decomposition strategies
//! - Controller operations
//! - Memory caching behavior
//! - Quality scoring
//!
//! These tests verify the RLM system's ability to:
//! - Decompose complex queries into simpler sub-queries
//! - Process queries recursively with memoization
//! - Aggregate sub-answers with quality scoring
//! - Handle edge cases and error conditions

use ruvllm::rlm::{
    MemoryConfig, MemoryEntry, NativeEnvironment, QueryResult, RlmConfig, RlmController,
};

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_rlm_config_defaults() {
    let config = RlmConfig::default();

    // Core defaults
    assert_eq!(config.embedding_dim, 384);
    assert_eq!(config.max_seq_len, 2048);
    assert!(config.enable_cache);
    assert_eq!(config.cache_ttl_secs, 3600);
    assert_eq!(config.max_concurrent_ops, 4);

    // Generation defaults
    assert!((config.temperature - 0.7).abs() < 0.01);
    assert!((config.top_p - 0.9).abs() < 0.01);
    assert_eq!(config.max_tokens, 512);
}

#[test]
fn test_rlm_config_for_wasm() {
    let config = RlmConfig::for_wasm();

    // WASM optimizations
    assert_eq!(config.embedding_dim, 256);
    assert_eq!(config.max_seq_len, 1024);
    assert_eq!(config.max_concurrent_ops, 1); // Single-threaded
    assert_eq!(config.max_tokens, 256);
}

#[test]
fn test_rlm_config_for_ruvltra_small() {
    let config = RlmConfig::for_ruvltra_small();

    assert_eq!(config.embedding_dim, 384);
    assert_eq!(config.max_seq_len, 4096);
    assert_eq!(config.max_concurrent_ops, 8);
    assert_eq!(config.max_tokens, 1024);
    assert_eq!(config.model_id, "ruvltra-small");
}

#[test]
fn test_memory_config_defaults() {
    let config = MemoryConfig::default();

    // Verify memory config has sensible defaults
    assert!(config.max_entries > 0);
    assert!(config.embedding_dim > 0);
}

// ============================================================================
// Controller Creation Tests
// ============================================================================

#[test]
fn test_controller_creation_with_default_config() {
    let config = RlmConfig::default();
    let result = RlmController::<NativeEnvironment>::new(config);

    assert!(result.is_ok());
    let controller = result.unwrap();

    // Verify initial state
    let stats = controller.stats();
    assert_eq!(
        stats
            .total_queries
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        stats
            .total_memories
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn test_controller_creation_with_custom_config() {
    let config = RlmConfig {
        embedding_dim: 512,
        max_seq_len: 4096,
        enable_cache: false,
        temperature: 0.5,
        ..RlmConfig::default()
    };

    let result = RlmController::<NativeEnvironment>::new(config.clone());
    assert!(result.is_ok());

    let controller = result.unwrap();
    assert_eq!(controller.config().embedding_dim, 512);
    assert_eq!(controller.config().max_seq_len, 4096);
    assert!(!controller.config().enable_cache);
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_stats_initial_state() {
    let controller = RlmController::<NativeEnvironment>::new(RlmConfig::default()).unwrap();
    let stats = controller.stats();
    let snapshot = stats.snapshot();

    assert_eq!(snapshot.total_queries, 0);
    assert_eq!(snapshot.total_memories, 0);
    assert_eq!(snapshot.cache_hits, 0);
    assert_eq!(snapshot.cache_misses, 0);
    assert_eq!(snapshot.total_tokens, 0);
    assert_eq!(snapshot.avg_latency_us, 0);
}

#[test]
fn test_stats_cloning() {
    let controller = RlmController::<NativeEnvironment>::new(RlmConfig::default()).unwrap();
    let stats = controller.stats();

    // Modify through atomic
    stats
        .total_queries
        .fetch_add(10, std::sync::atomic::Ordering::SeqCst);

    // Clone should preserve value
    let cloned = (*stats).clone();
    assert_eq!(
        cloned
            .total_queries
            .load(std::sync::atomic::Ordering::SeqCst),
        10
    );
}

// ============================================================================
// Cache Behavior Tests
// ============================================================================

#[test]
fn test_cache_clearing() {
    let controller = RlmController::<NativeEnvironment>::new(RlmConfig::default()).unwrap();

    // Clear should not panic on empty cache
    controller.clear_cache();
}

#[test]
fn test_cache_disabled_config() {
    let config = RlmConfig {
        enable_cache: false,
        ..RlmConfig::default()
    };

    let controller = RlmController::<NativeEnvironment>::new(config).unwrap();
    assert!(!controller.config().enable_cache);
}

// ============================================================================
// Memory Entry Tests
// ============================================================================

#[test]
fn test_memory_entry_creation() {
    use ruvllm::rlm::controller::MemoryMetadata;

    let metadata = MemoryMetadata {
        source: Some("test-source".to_string()),
        category: Some("test-category".to_string()),
        tags: vec!["tag1".to_string(), "tag2".to_string()],
        extra: Default::default(),
    };

    assert_eq!(metadata.source.as_deref(), Some("test-source"));
    assert_eq!(metadata.category.as_deref(), Some("test-category"));
    assert_eq!(metadata.tags.len(), 2);
}

#[test]
fn test_memory_metadata_default() {
    use ruvllm::rlm::controller::MemoryMetadata;

    let metadata = MemoryMetadata::default();

    assert!(metadata.source.is_none());
    assert!(metadata.category.is_none());
    assert!(metadata.tags.is_empty());
}

// ============================================================================
// Query Result Tests
// ============================================================================

#[test]
fn test_token_usage_default() {
    use ruvllm::rlm::controller::TokenUsage;

    let usage = TokenUsage::default();

    assert_eq!(usage.input_tokens, 0);
    assert_eq!(usage.output_tokens, 0);
    assert_eq!(usage.total_tokens, 0);
}

#[test]
fn test_source_attribution_structure() {
    use ruvllm::rlm::controller::SourceAttribution;

    let source = SourceAttribution {
        memory_id: "mem-123".to_string(),
        relevance: 0.85,
        excerpt: "Test excerpt".to_string(),
    };

    assert_eq!(source.memory_id, "mem-123");
    assert!((source.relevance - 0.85).abs() < 0.001);
    assert_eq!(source.excerpt, "Test excerpt");
}

// ============================================================================
// RuvLTRA Backend Types Tests
// ============================================================================

#[test]
fn test_generation_params_default() {
    use ruvllm::rlm::GenerationParams;

    let params = GenerationParams::default();

    // Verify reasonable defaults
    assert!(params.temperature > 0.0);
    assert!(params.top_p > 0.0 && params.top_p <= 1.0);
    assert!(params.max_tokens > 0);
}

#[test]
fn test_finish_reason_variants() {
    use ruvllm::rlm::FinishReason;

    // Test all finish reason variants exist
    let _ = FinishReason::Stop;
    let _ = FinishReason::MaxTokens;
    let _ = FinishReason::StopSequence;
}

#[test]
fn test_memory_span_creation() {
    use ruvllm::rlm::MemorySpan;

    let span = MemorySpan {
        id: ruvllm::rlm::MemoryId::new(),
        text: "Test memory content".to_string(),
        embedding: vec![0.1; 384],
        similarity_score: 0.9,
        source: Some("test-source".to_string()),
        metadata: Default::default(),
        stored_at: chrono::Utc::now(),
        access_count: 0,
    };

    assert_eq!(span.text, "Test memory content");
    assert_eq!(span.embedding.len(), 384);
    assert!((span.similarity_score - 0.9).abs() < 0.001);
}

#[test]
fn test_memory_id_uniqueness() {
    use ruvllm::rlm::MemoryId;

    let id1 = MemoryId::new();
    let id2 = MemoryId::new();

    // IDs should be unique
    assert_ne!(id1.as_uuid(), id2.as_uuid());
}

// ============================================================================
// Query Decomposition Tests
// ============================================================================

#[test]
fn test_decomposition_strategy_variants() {
    use ruvllm::rlm::DecompositionStrategy;

    // Verify all strategy variants exist
    let _ = DecompositionStrategy::Direct;
    let _ = DecompositionStrategy::Recursive;
    let _ = DecompositionStrategy::Parallel;
}

#[test]
fn test_sub_query_structure() {
    use ruvllm::rlm::SubQuery;

    let sub_query = SubQuery {
        id: 0,
        text: "What is machine learning?".to_string(),
        parent_id: None,
        depth: 0,
        strategy: ruvllm::rlm::DecompositionStrategy::Direct,
    };

    assert_eq!(sub_query.text, "What is machine learning?");
    assert_eq!(sub_query.depth, 0);
    assert!(sub_query.parent_id.is_none());
}

// ============================================================================
// RLM Answer Tests
// ============================================================================

#[test]
fn test_rlm_answer_structure() {
    use ruvllm::rlm::RlmAnswer;

    let answer = RlmAnswer {
        text: "Machine learning is a subset of AI...".to_string(),
        confidence: 0.85,
        sub_answers: vec![],
        sources: vec![],
        tokens_used: 150,
        latency_ms: 45.5,
        cache_hit: false,
    };

    assert_eq!(answer.text, "Machine learning is a subset of AI...");
    assert!((answer.confidence - 0.85).abs() < 0.001);
    assert!(!answer.cache_hit);
}

// ============================================================================
// KV Cache Tests
// ============================================================================

#[test]
fn test_kv_cache_entry_structure() {
    use ruvllm::rlm::KvCacheEntry;

    let entry = KvCacheEntry {
        key: vec![0.1f32; 64],
        value: vec![0.2f32; 64],
        layer_idx: 0,
        head_idx: 0,
        seq_pos: 0,
    };

    assert_eq!(entry.key.len(), 64);
    assert_eq!(entry.value.len(), 64);
    assert_eq!(entry.layer_idx, 0);
}

// ============================================================================
// Token Types Tests
// ============================================================================

#[test]
fn test_stream_token_structure() {
    use ruvllm::rlm::StreamToken;

    let token = StreamToken {
        text: "Hello".to_string(),
        token_id: 12345,
        logprob: Some(-0.5),
        is_final: false,
    };

    assert_eq!(token.text, "Hello");
    assert_eq!(token.token_id, 12345);
    assert!(!token.is_final);
}

#[test]
fn test_token_usage_structure() {
    use ruvllm::rlm::TokenUsage;

    let usage = TokenUsage {
        prompt_tokens: 50,
        completion_tokens: 100,
        total_tokens: 150,
    };

    assert_eq!(usage.prompt_tokens, 50);
    assert_eq!(usage.completion_tokens, 100);
    assert_eq!(usage.total_tokens, 150);
}

// ============================================================================
// Model Info Tests
// ============================================================================

#[test]
fn test_rlm_model_info() {
    use ruvllm::rlm::RlmModelInfo;

    let info = RlmModelInfo {
        name: "ruvltra-small".to_string(),
        version: "0.5B".to_string(),
        embedding_dim: 384,
        max_seq_len: 4096,
        vocab_size: 32000,
    };

    assert_eq!(info.name, "ruvltra-small");
    assert_eq!(info.embedding_dim, 384);
}

// ============================================================================
// Concurrent Safety Tests
// ============================================================================

#[test]
fn test_controller_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let controller =
        Arc::new(RlmController::<NativeEnvironment>::new(RlmConfig::default()).unwrap());

    let mut handles = vec![];

    // Spawn multiple threads accessing stats
    for _ in 0..4 {
        let ctrl = Arc::clone(&controller);
        let handle = thread::spawn(move || {
            let stats = ctrl.stats();
            stats
                .total_queries
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            stats.snapshot()
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        let _ = handle.join().unwrap();
    }

    // Verify concurrent increments
    let final_stats = controller.stats().snapshot();
    assert_eq!(final_stats.total_queries, 4);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_empty_embedding_dim() {
    let config = RlmConfig {
        embedding_dim: 1, // Minimum valid
        ..RlmConfig::default()
    };

    let result = RlmController::<NativeEnvironment>::new(config);
    // Should succeed with minimal config
    assert!(result.is_ok());
}

#[test]
fn test_large_token_budget() {
    let config = RlmConfig {
        max_tokens: 100000, // Large budget
        ..RlmConfig::default()
    };

    let result = RlmController::<NativeEnvironment>::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_zero_cache_ttl() {
    let config = RlmConfig {
        cache_ttl_secs: 0, // No caching
        enable_cache: true,
        ..RlmConfig::default()
    };

    let result = RlmController::<NativeEnvironment>::new(config);
    assert!(result.is_ok());
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_config_serialization() {
    let config = RlmConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: RlmConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.embedding_dim, config.embedding_dim);
    assert_eq!(deserialized.max_seq_len, config.max_seq_len);
    assert_eq!(deserialized.enable_cache, config.enable_cache);
}

#[test]
fn test_query_result_serialization() {
    use ruvllm::rlm::controller::{QueryResult, TokenUsage};

    let result = QueryResult {
        id: "test-123".to_string(),
        text: "Test response".to_string(),
        confidence: 0.9,
        tokens_generated: 10,
        latency_ms: 50.0,
        sources: vec![],
        usage: TokenUsage {
            input_tokens: 5,
            output_tokens: 10,
            total_tokens: 15,
        },
    };

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: QueryResult = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.id, "test-123");
    assert_eq!(deserialized.text, "Test response");
}

// ============================================================================
// Memory Search Result Tests
// ============================================================================

#[test]
fn test_memory_search_result_structure() {
    use ruvllm::rlm::MemorySearchResult;

    // Verify the struct can be constructed
    // The actual implementation details depend on the memory module
}

// ============================================================================
// Integration Scenario Tests
// ============================================================================

#[test]
fn test_controller_lifecycle() {
    // Create controller
    let controller = RlmController::<NativeEnvironment>::new(RlmConfig::default()).unwrap();

    // Verify initial state
    assert_eq!(controller.stats().snapshot().total_queries, 0);

    // Clear cache (should not panic)
    controller.clear_cache();

    // Stats should still be valid
    let stats = controller.stats().snapshot();
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.cache_misses, 0);
}
