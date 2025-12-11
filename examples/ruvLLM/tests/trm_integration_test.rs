//! TRM Integration Tests
//!
//! Tests the full TRM pipeline from question to refined answer.

use ruvllm::trm::{
    AttentionLatentUpdater, ConfidenceScorer, MlpLatentUpdater, SonaBridge,
    TrmConfig, TrmEngine, TrmEngineBuilder, TrmTrajectory,
    LatentUpdate, RecursiveReasoner,
};

/// Test basic TRM reasoning pipeline
#[test]
fn test_trm_full_pipeline() {
    // Create engine with small dimensions for fast testing
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(5)
        .latent_iterations(2)
        .confidence_threshold(0.95)
        .early_stopping(true)
        .build()
        .expect("Failed to create TRM engine");

    // Simulate question and answer embeddings
    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    // Run reasoning
    let result = engine.reason(&question, &mut answer);

    // Verify results
    assert!(result.iterations_used > 0, "Should use at least 1 iteration");
    assert!(result.iterations_used <= 5, "Should not exceed default_k");
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0, "Confidence should be in [0, 1]");
    assert_eq!(result.answer.len(), 64, "Answer should maintain dimension");
    assert_eq!(result.trajectory.states.len(), result.iterations_used);
}

/// Test TRM with MLP updater
#[test]
fn test_trm_with_mlp() {
    let config = TrmConfig {
        hidden_dim: 64,
        embedding_dim: 64,
        use_attention: false,
        default_k: 3,
        ..Default::default()
    };

    let mut engine = TrmEngine::with_mlp(config).expect("Failed to create MLP engine");

    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    let result = engine.reason(&question, &mut answer);

    assert!(!engine.config().use_attention);
    assert!(result.iterations_used > 0);
}

/// Test TRM with Attention updater
#[test]
fn test_trm_with_attention() {
    let config = TrmConfig {
        hidden_dim: 64,
        embedding_dim: 64,
        use_attention: true,
        num_heads: 4,
        default_k: 3,
        ..Default::default()
    };

    let mut engine = TrmEngine::with_attention(config).expect("Failed to create Attention engine");

    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    let result = engine.reason(&question, &mut answer);

    assert!(engine.config().use_attention);
    assert!(result.iterations_used > 0);
}

/// Test SONA bridge routing
#[test]
fn test_sona_routing() {
    let config = TrmConfig {
        hidden_dim: 64,
        embedding_dim: 64,
        max_k: 20,
        ..Default::default()
    };

    let bridge = SonaBridge::new(&config);

    // Simple query
    let simple_query = vec![0.1; 64];
    let decision = bridge.route(&simple_query);

    assert!(decision.k_value >= 1 && decision.k_value <= 20);

    // Complex query (high variance)
    let complex_query: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
    let complex_decision = bridge.route(&complex_query);

    assert!(complex_decision.k_value >= 1 && complex_decision.k_value <= 20);
}

/// Test SONA learning from trajectories
#[test]
fn test_sona_learning() {
    let config = TrmConfig {
        hidden_dim: 64,
        embedding_dim: 64,
        max_k: 20,
        ..Default::default()
    };

    let mut bridge = SonaBridge::new(&config);

    // Learn from trajectories
    for k in [5, 6, 7, 5, 6, 5, 7, 6] {
        let mut trajectory = TrmTrajectory::new(vec![0.5; 64]);
        trajectory.optimal_k = k;
        bridge.learn(&trajectory);
    }

    // Check statistics updated
    let stats = bridge.get_stats();
    assert!(stats.count >= 8);
    assert!((stats.mean_k - 6.0).abs() < 1.0);  // Mean should be around 6
}

/// Test TRM with routing decision
#[test]
fn test_trm_with_routing() {
    let config = TrmConfig {
        hidden_dim: 64,
        embedding_dim: 64,
        max_k: 20,
        ..Default::default()
    };

    let mut engine = TrmEngine::new(config.clone()).expect("Failed to create engine");
    let bridge = SonaBridge::new(&config);

    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    // Get routing decision
    let routing = bridge.route(&question);

    // Run with routing
    let result = engine.reason_with_routing(&question, &mut answer, &routing);

    if routing.use_trm {
        assert!(result.iterations_used > 0);
    } else {
        assert_eq!(result.iterations_used, 0);
    }
}

/// Test MLP and Attention updaters produce different results
#[test]
fn test_updater_variants_differ() {
    let mlp = MlpLatentUpdater::new(64, 64);
    let attention = AttentionLatentUpdater::new(64, 64, 4);

    let question = vec![0.5; 64];
    let answer = vec![0.5; 64];

    let mut mlp_latent = vec![0.0; 64];
    let mut attn_latent = vec![0.0; 64];

    mlp.update(&question, &answer, &mut mlp_latent);
    attention.update(&question, &answer, &mut attn_latent);

    // Both should modify latent
    assert!(mlp_latent.iter().any(|&x| x.abs() > 1e-6));
    assert!(attn_latent.iter().any(|&x| x.abs() > 1e-6));

    // Should produce different results
    let diff: f32 = mlp_latent.iter()
        .zip(attn_latent.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(diff > 0.01, "MLP and Attention should produce different latents");
}

/// Test confidence scoring
#[test]
fn test_confidence_scoring() {
    let config = TrmConfig {
        embedding_dim: 64,
        ..Default::default()
    };

    let scorer = ConfidenceScorer::new(&config);

    // Low values
    let low = vec![0.1; 64];
    let low_conf = scorer.score(&low);

    // High values
    let high = vec![0.9; 64];
    let high_conf = scorer.score(&high);

    // Both should be valid
    assert!(low_conf >= 0.0 && low_conf <= 1.0);
    assert!(high_conf >= 0.0 && high_conf <= 1.0);
}

/// Test trajectory recording
#[test]
fn test_trajectory_recording() {
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(5)
        .early_stopping(false)  // Don't early stop to get all iterations
        .build()
        .expect("Failed to create engine");

    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    let result = engine.reason(&question, &mut answer);

    // Should have recorded all iterations
    assert_eq!(result.trajectory.states.len(), 5);

    // Each state should have proper data
    for (i, state) in result.trajectory.states.iter().enumerate() {
        assert_eq!(state.iteration, i);
        assert_eq!(state.latent_state.len(), 64);
        assert_eq!(state.answer_state.len(), 64);
        assert!(state.confidence >= 0.0 && state.confidence <= 1.0);
        assert!(state.latency_us > 0);
    }
}

/// Test early stopping
#[test]
fn test_early_stopping() {
    // Low threshold to encourage early stopping
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(20)
        .confidence_threshold(0.3)  // Very low threshold
        .early_stopping(true)
        .build()
        .expect("Failed to create engine");

    let question = vec![0.5; 64];
    let mut answer = vec![0.5; 64];

    let result = engine.reason(&question, &mut answer);

    // May or may not early stop depending on initialization
    // But should not exceed max_k
    assert!(result.iterations_used <= 20);
}

/// Test variable length inputs
#[test]
fn test_variable_input_lengths() {
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(3)
        .build()
        .expect("Failed to create engine");

    // Short inputs
    let short_q = vec![0.5; 32];
    let mut short_a = vec![0.1; 32];
    let result = engine.reason(&short_q, &mut short_a);
    assert!(result.confidence >= 0.0);

    // Long inputs
    let long_q = vec![0.5; 256];
    let mut long_a = vec![0.1; 256];
    let result = engine.reason(&long_q, &mut long_a);
    assert!(result.confidence >= 0.0);

    // Exact size
    let exact_q = vec![0.5; 64];
    let mut exact_a = vec![0.1; 64];
    let result = engine.reason(&exact_q, &mut exact_a);
    assert!(result.confidence >= 0.0);
}

/// Test state persistence with SONA bridge
#[test]
fn test_sona_state_persistence() {
    let config = TrmConfig {
        hidden_dim: 64,
        embedding_dim: 64,
        max_k: 20,
        ..Default::default()
    };

    let mut bridge = SonaBridge::new(&config);

    // Learn something
    for _ in 0..10 {
        let mut trajectory = TrmTrajectory::new(vec![0.5; 64]);
        trajectory.optimal_k = 8;
        bridge.learn(&trajectory);
    }

    // Export state
    let state = bridge.export_state();
    assert_eq!(state.stats.count, 10);

    // Create new bridge and import
    let mut new_bridge = SonaBridge::new(&config);
    new_bridge.import_state(state);

    // Should have same statistics
    assert_eq!(new_bridge.get_stats().count, 10);
    assert!((new_bridge.get_stats().mean_k - 8.0).abs() < 0.1);
}

/// Test multiple reasoning sessions
#[test]
fn test_multiple_sessions() {
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(3)
        .build()
        .expect("Failed to create engine");

    // Run multiple queries
    for i in 0..5 {
        let question: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
        let mut answer = vec![0.1; 64];

        let result = engine.reason(&question, &mut answer);

        assert!(result.iterations_used > 0);
        assert!(result.confidence >= 0.0);
    }

    // Check statistics
    let (total_iters, _) = engine.stats();
    assert!(total_iters >= 5);  // At least 1 iteration per query
}

/// Test reset functionality
#[test]
fn test_engine_reset() {
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(3)
        .build()
        .expect("Failed to create engine");

    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    engine.reason(&question, &mut answer);
    engine.reset();

    // Should work after reset
    let mut answer2 = vec![0.1; 64];
    let result = engine.reason(&question, &mut answer2);
    assert!(result.iterations_used > 0);
}

/// Test with custom K
#[test]
fn test_custom_k() {
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(10)
        .max_k(20)
        .early_stopping(false)
        .build()
        .expect("Failed to create engine");

    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    let result = engine.reason_with_k(&question, &mut answer, 7);

    assert_eq!(result.iterations_used, 7);
}

/// Test entropy-based confidence
#[test]
fn test_entropy_confidence() {
    let mut engine = TrmEngineBuilder::new()
        .hidden_dim(64)
        .embedding_dim(64)
        .default_k(3)
        .use_entropy_confidence(true)
        .build()
        .expect("Failed to create engine");

    let question = vec![0.5; 64];
    let mut answer = vec![0.1; 64];

    let result = engine.reason(&question, &mut answer);

    // Should still produce valid confidence
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}
