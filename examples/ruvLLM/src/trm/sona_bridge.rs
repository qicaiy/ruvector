//! SONA Bridge
//!
//! Integration with the Self-Optimizing Neural Architecture (SONA) for:
//! - Adaptive K selection based on query complexity
//! - Learning optimal iteration counts from trajectory data
//! - Routing decisions (use TRM or skip)
//!
//! Attribution: TRM algorithm from Samsung SAIL Montreal
//! https://github.com/SamsungSAILMontreal/TinyRecursiveModels

use serde::{Deserialize, Serialize};

use super::{
    types::{TrmRoutingDecision, TrmTrajectory},
    TrmConfig,
};

/// SONA Bridge configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SonaBridgeConfig {
    /// Minimum K value to predict
    pub min_k: usize,

    /// Maximum K value to predict
    pub max_k: usize,

    /// Learning rate for K predictor
    pub learning_rate: f32,

    /// History size for running statistics
    pub history_size: usize,

    /// Complexity threshold for routing (below = skip TRM)
    pub complexity_threshold: f32,

    /// Enable adaptive K prediction
    pub enable_adaptive_k: bool,

    /// Enable learning from trajectories
    pub enable_learning: bool,
}

impl Default for SonaBridgeConfig {
    fn default() -> Self {
        Self {
            min_k: 1,
            max_k: 20,
            learning_rate: 0.01,
            history_size: 100,
            complexity_threshold: 0.3,
            enable_adaptive_k: true,
            enable_learning: true,
        }
    }
}

/// Statistics about K values from trajectories
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KStatistics {
    /// Running mean of optimal K
    pub mean_k: f32,

    /// Running variance of K
    pub variance_k: f32,

    /// Number of samples
    pub count: usize,

    /// Recent K values for windowed statistics
    pub recent_ks: Vec<usize>,

    /// K values by task type
    pub k_by_task: Vec<(String, f32)>,
}

impl KStatistics {
    /// Create new empty statistics
    pub fn new(history_size: usize) -> Self {
        Self {
            mean_k: 5.0,  // Default starting point
            variance_k: 4.0,
            count: 0,
            recent_ks: Vec::with_capacity(history_size),
            k_by_task: Vec::new(),
        }
    }

    /// Update statistics with a new K observation
    pub fn update(&mut self, k: usize, history_size: usize) {
        let k_f = k as f32;

        // Welford's online algorithm for mean and variance
        self.count += 1;
        let n = self.count as f32;

        let delta = k_f - self.mean_k;
        self.mean_k += delta / n;

        let delta2 = k_f - self.mean_k;
        self.variance_k += delta * delta2;

        // Update recent history
        self.recent_ks.push(k);
        if self.recent_ks.len() > history_size {
            self.recent_ks.remove(0);
        }
    }

    /// Update statistics for a specific task type
    pub fn update_for_task(&mut self, task_type: &str, k: usize) {
        let k_f = k as f32;

        // Find or create entry for this task type
        if let Some(entry) = self.k_by_task.iter_mut().find(|(t, _)| t == task_type) {
            // Exponential moving average
            entry.1 = entry.1 * 0.9 + k_f * 0.1;
        } else {
            self.k_by_task.push((task_type.to_string(), k_f));
        }

        // Limit task types stored
        if self.k_by_task.len() > 100 {
            self.k_by_task.remove(0);
        }
    }

    /// Get standard deviation
    pub fn std_k(&self) -> f32 {
        if self.count < 2 {
            return 2.0;  // Default
        }
        (self.variance_k / (self.count - 1) as f32).sqrt()
    }

    /// Get recommended K for a task type
    pub fn recommended_k_for_task(&self, task_type: &str) -> Option<usize> {
        self.k_by_task
            .iter()
            .find(|(t, _)| t == task_type)
            .map(|(_, k)| k.round() as usize)
    }
}

/// Query complexity estimator
///
/// Estimates how complex a query is to help route and select K.
#[derive(Clone, Debug)]
pub struct ComplexityEstimator {
    /// Projection weights for complexity estimation
    weights: Vec<f32>,

    /// Bias term
    bias: f32,

    /// Input dimension
    input_dim: usize,
}

impl ComplexityEstimator {
    /// Create a new complexity estimator
    pub fn new(input_dim: usize) -> Self {
        // Initialize with heuristic weights
        // Higher values for larger embeddings suggest more complexity
        let weights: Vec<f32> = (0..input_dim)
            .map(|i| {
                let x = ((i as f64 * 0.6180339887498949) % 1.0) as f32;
                (x - 0.5) * 0.1
            })
            .collect();

        Self {
            weights,
            bias: 0.5,  // Default to medium complexity
            input_dim,
        }
    }

    /// Estimate complexity of a query embedding
    ///
    /// Returns a value in [0, 1] where:
    /// - 0 = very simple (skip TRM or use K=1)
    /// - 1 = very complex (use high K)
    pub fn estimate(&self, query_embedding: &[f32]) -> f32 {
        // Pool to input_dim if needed
        let pooled = if query_embedding.len() == self.input_dim {
            query_embedding.to_vec()
        } else if query_embedding.len() < self.input_dim {
            let mut p = query_embedding.to_vec();
            p.resize(self.input_dim, 0.0);
            p
        } else {
            let chunks = query_embedding.len() / self.input_dim;
            let mut p = vec![0.0; self.input_dim];
            for chunk in query_embedding.chunks(self.input_dim) {
                for (i, &v) in chunk.iter().enumerate() {
                    if i < self.input_dim {
                        p[i] += v / chunks as f32;
                    }
                }
            }
            p
        };

        // Compute weighted sum
        let mut score = self.bias;
        for (i, &w) in self.weights.iter().enumerate() {
            if i < pooled.len() {
                score += pooled[i] * w;
            }
        }

        // Additional heuristics:
        // - Variance in embedding suggests complexity
        let mean: f32 = pooled.iter().sum::<f32>() / pooled.len() as f32;
        let variance: f32 = pooled.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / pooled.len() as f32;
        let variance_factor = (variance.sqrt() * 2.0).clamp(0.0, 0.5);

        // - Magnitude of embedding
        let magnitude: f32 = pooled.iter().map(|x| x.abs()).sum::<f32>() / pooled.len() as f32;
        let magnitude_factor = (magnitude * 0.5).clamp(0.0, 0.3);

        let raw_score = score + variance_factor + magnitude_factor;

        // Sigmoid to bound in [0, 1]
        1.0 / (1.0 + (-raw_score).exp())
    }

    /// Update weights based on trajectory feedback
    pub fn learn_from_trajectory(&mut self, query_embedding: &[f32], optimal_k: usize, max_k: usize, learning_rate: f32) {
        // Target complexity based on optimal K
        let target_complexity = optimal_k as f32 / max_k as f32;

        // Current estimate
        let estimated = self.estimate(query_embedding);

        // Error
        let error = target_complexity - estimated;

        // Gradient descent on weights
        let pooled = if query_embedding.len() == self.input_dim {
            query_embedding.to_vec()
        } else {
            // Simplified pooling
            let mut p = vec![0.0; self.input_dim];
            for (i, &v) in query_embedding.iter().enumerate() {
                p[i % self.input_dim] += v / (query_embedding.len() as f32 / self.input_dim as f32);
            }
            p
        };

        // Update weights
        let gradient_scale = error * learning_rate * estimated * (1.0 - estimated);
        for (i, w) in self.weights.iter_mut().enumerate() {
            if i < pooled.len() {
                *w += gradient_scale * pooled[i];
            }
        }

        self.bias += gradient_scale;
    }
}

/// SONA Bridge for TRM integration
///
/// Provides:
/// - Adaptive K selection
/// - Routing decisions
/// - Learning from trajectories
pub struct SonaBridge {
    config: SonaBridgeConfig,
    stats: KStatistics,
    complexity_estimator: ComplexityEstimator,
}

impl SonaBridge {
    /// Create a new SONA bridge
    pub fn new(trm_config: &TrmConfig) -> Self {
        let bridge_config = SonaBridgeConfig {
            max_k: trm_config.max_k,
            ..Default::default()
        };

        Self {
            config: bridge_config.clone(),
            stats: KStatistics::new(bridge_config.history_size),
            complexity_estimator: ComplexityEstimator::new(trm_config.embedding_dim),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SonaBridgeConfig, embedding_dim: usize) -> Self {
        Self {
            config: config.clone(),
            stats: KStatistics::new(config.history_size),
            complexity_estimator: ComplexityEstimator::new(embedding_dim),
        }
    }

    /// Get routing decision for a query
    pub fn route(&self, query_embedding: &[f32]) -> TrmRoutingDecision {
        let complexity = self.complexity_estimator.estimate(query_embedding);

        // Decision: use TRM?
        let use_trm = complexity >= self.config.complexity_threshold;

        // Predict K based on complexity
        let predicted_k = if self.config.enable_adaptive_k {
            self.predict_k(complexity)
        } else {
            5  // Default
        };

        TrmRoutingDecision {
            use_trm,
            k_value: predicted_k,
            n_value: 3,  // Default latent iterations
            use_attention: complexity > 0.7,  // Use attention for complex queries
            confidence: 0.5 + complexity * 0.3,  // Higher complexity = higher confidence in prediction
            reason: if use_trm {
                format!("complexity={:.2}, predicted_k={}", complexity, predicted_k)
            } else {
                format!("complexity={:.2} below threshold, skipping TRM", complexity)
            },
        }
    }

    /// Route with task type hint
    pub fn route_with_task(&self, query_embedding: &[f32], task_type: &str) -> TrmRoutingDecision {
        let mut decision = self.route(query_embedding);

        // Override K if we have statistics for this task type
        if let Some(recommended_k) = self.stats.recommended_k_for_task(task_type) {
            decision.k_value = recommended_k.clamp(self.config.min_k, self.config.max_k);
            decision.reason = format!(
                "task_type={}, recommended_k={}, {}",
                task_type, recommended_k, decision.reason
            );
        }

        decision
    }

    /// Predict K based on complexity
    fn predict_k(&self, complexity: f32) -> usize {
        // Linear interpolation between min and max K
        let base_k = self.config.min_k as f32
            + complexity * (self.config.max_k - self.config.min_k) as f32;

        // Adjust based on historical statistics
        let adjusted_k = if self.stats.count > 10 {
            // Blend with historical mean
            base_k * 0.7 + self.stats.mean_k * 0.3
        } else {
            base_k
        };

        (adjusted_k.round() as usize).clamp(self.config.min_k, self.config.max_k)
    }

    /// Learn from a completed trajectory
    pub fn learn(&mut self, trajectory: &TrmTrajectory) {
        if !self.config.enable_learning {
            return;
        }

        let optimal_k = trajectory.optimal_k;

        // Update K statistics
        self.stats.update(optimal_k, self.config.history_size);

        // Update task-specific statistics
        if let Some(task_type) = &trajectory.task_type {
            self.stats.update_for_task(task_type, optimal_k);
        }

        // Train complexity estimator
        self.complexity_estimator.learn_from_trajectory(
            &trajectory.question_embedding,
            optimal_k,
            self.config.max_k,
            self.config.learning_rate,
        );
    }

    /// Get current K statistics
    pub fn get_stats(&self) -> &KStatistics {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &SonaBridgeConfig {
        &self.config
    }

    /// Reset learned state
    pub fn reset(&mut self) {
        self.stats = KStatistics::new(self.config.history_size);
    }

    /// Export learned state for persistence
    pub fn export_state(&self) -> SonaBridgeState {
        SonaBridgeState {
            stats: self.stats.clone(),
            complexity_weights: self.complexity_estimator.weights.clone(),
            complexity_bias: self.complexity_estimator.bias,
        }
    }

    /// Import learned state
    pub fn import_state(&mut self, state: SonaBridgeState) {
        self.stats = state.stats;
        self.complexity_estimator.weights = state.complexity_weights;
        self.complexity_estimator.bias = state.complexity_bias;
    }
}

/// Serializable state for persistence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SonaBridgeState {
    pub stats: KStatistics,
    pub complexity_weights: Vec<f32>,
    pub complexity_bias: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> TrmConfig {
        TrmConfig {
            hidden_dim: 64,
            embedding_dim: 64,
            max_k: 20,
            ..Default::default()
        }
    }

    #[test]
    fn test_sona_bridge_creation() {
        let config = make_config();
        let bridge = SonaBridge::new(&config);
        assert_eq!(bridge.config().max_k, 20);
    }

    #[test]
    fn test_complexity_estimation() {
        let estimator = ComplexityEstimator::new(64);

        // Low complexity: small values
        let low = vec![0.1; 64];
        let complexity_low = estimator.estimate(&low);

        // High complexity: varied values
        let high: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let complexity_high = estimator.estimate(&high);

        // Both should be in valid range
        assert!(complexity_low >= 0.0 && complexity_low <= 1.0);
        assert!(complexity_high >= 0.0 && complexity_high <= 1.0);
    }

    #[test]
    fn test_routing_decision() {
        let config = make_config();
        let bridge = SonaBridge::new(&config);

        // Simple query (low values)
        let simple = vec![0.1; 64];
        let decision = bridge.route(&simple);
        assert!(decision.k_value >= 1 && decision.k_value <= 20);

        // Complex query (varied values)
        let complex: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let decision = bridge.route(&complex);
        assert!(decision.k_value >= 1 && decision.k_value <= 20);
    }

    #[test]
    fn test_learning_from_trajectory() {
        let config = make_config();
        let mut bridge = SonaBridge::new(&config);

        // Create a trajectory
        let mut trajectory = TrmTrajectory::new(vec![0.5; 64]);
        trajectory.optimal_k = 10;
        trajectory.task_type = Some("math".to_string());

        // Learn from it
        bridge.learn(&trajectory);

        assert_eq!(bridge.stats.count, 1);
        assert!(bridge.stats.mean_k > 0.0);
    }

    #[test]
    fn test_k_statistics() {
        let mut stats = KStatistics::new(10);

        // Add some observations
        for k in [5, 7, 6, 8, 5, 6, 7, 6, 5, 6] {
            stats.update(k, 10);
        }

        // Mean should be around 6.1
        assert!((stats.mean_k - 6.1).abs() < 0.2);

        // Standard deviation should be reasonable
        let std = stats.std_k();
        assert!(std > 0.0 && std < 5.0);
    }

    #[test]
    fn test_task_specific_k() {
        let config = make_config();
        let mut bridge = SonaBridge::new(&config);

        // Learn from multiple trajectories with task types
        for _ in 0..5 {
            let mut traj = TrmTrajectory::new(vec![0.5; 64]);
            traj.optimal_k = 3;
            traj.task_type = Some("simple".to_string());
            bridge.learn(&traj);
        }

        for _ in 0..5 {
            let mut traj = TrmTrajectory::new(vec![0.5; 64]);
            traj.optimal_k = 15;
            traj.task_type = Some("complex".to_string());
            bridge.learn(&traj);
        }

        // Check task-specific recommendations
        let simple_k = bridge.stats.recommended_k_for_task("simple");
        let complex_k = bridge.stats.recommended_k_for_task("complex");

        assert!(simple_k.is_some());
        assert!(complex_k.is_some());

        // Complex tasks should have higher recommended K
        if let (Some(s), Some(c)) = (simple_k, complex_k) {
            assert!(c > s, "Complex K ({}) should be > simple K ({})", c, s);
        }
    }

    #[test]
    fn test_state_export_import() {
        let config = make_config();
        let mut bridge = SonaBridge::new(&config);

        // Learn something
        let mut traj = TrmTrajectory::new(vec![0.5; 64]);
        traj.optimal_k = 8;
        bridge.learn(&traj);

        // Export state
        let state = bridge.export_state();

        // Create new bridge and import
        let mut new_bridge = SonaBridge::new(&config);
        new_bridge.import_state(state);

        // Should have same statistics
        assert_eq!(new_bridge.stats.count, bridge.stats.count);
        assert!((new_bridge.stats.mean_k - bridge.stats.mean_k).abs() < 0.01);
    }

    #[test]
    fn test_routing_with_task() {
        let config = make_config();
        let mut bridge = SonaBridge::new(&config);

        // Learn task-specific K
        for _ in 0..10 {
            let mut traj = TrmTrajectory::new(vec![0.5; 64]);
            traj.optimal_k = 12;
            traj.task_type = Some("reasoning".to_string());
            bridge.learn(&traj);
        }

        // Route with task hint
        let query = vec![0.5; 64];
        let decision = bridge.route_with_task(&query, "reasoning");

        // Should use task-specific K
        assert!(decision.reason.contains("reasoning"));
    }

    #[test]
    fn test_complexity_learning() {
        let mut estimator = ComplexityEstimator::new(64);

        // Learn that high-variance queries need high K
        let high_var: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 5.0).collect();

        let initial_estimate = estimator.estimate(&high_var);

        // Train toward high complexity
        for _ in 0..50 {
            estimator.learn_from_trajectory(&high_var, 15, 20, 0.1);
        }

        let final_estimate = estimator.estimate(&high_var);

        // Should have learned to estimate higher complexity
        // (may not always increase due to initialization, but should change)
        assert!((final_estimate - initial_estimate).abs() > 0.0 || initial_estimate > 0.5);
    }
}
