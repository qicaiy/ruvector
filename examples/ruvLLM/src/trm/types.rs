//! TRM Types
//!
//! Core data structures for TRM recursive reasoning.

use serde::{Deserialize, Serialize};

/// Result of TRM recursive reasoning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrmResult {
    /// Final answer embedding
    pub answer: Vec<f32>,

    /// Confidence score [0, 1]
    pub confidence: f32,

    /// Number of K iterations actually used
    pub iterations_used: usize,

    /// Whether early stopping was triggered
    pub early_stopped: bool,

    /// Full trajectory of reasoning
    pub trajectory: TrmTrajectory,

    /// Total latency in microseconds
    pub latency_us: u64,

    /// Per-iteration latencies
    pub iteration_latencies_us: Vec<u64>,
}

impl TrmResult {
    /// Create a new TRM result
    pub fn new(
        answer: Vec<f32>,
        confidence: f32,
        iterations_used: usize,
        early_stopped: bool,
        trajectory: TrmTrajectory,
        latency_us: u64,
    ) -> Self {
        let iteration_latencies_us = trajectory.states.iter()
            .map(|s| s.latency_us)
            .collect();

        Self {
            answer,
            confidence,
            iterations_used,
            early_stopped,
            trajectory,
            latency_us,
            iteration_latencies_us,
        }
    }

    /// Get average latency per iteration
    pub fn avg_iteration_latency_us(&self) -> f64 {
        if self.iterations_used == 0 {
            return 0.0;
        }
        self.latency_us as f64 / self.iterations_used as f64
    }
}

/// Single iteration state in TRM trajectory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrmIterationState {
    /// Iteration number (0-indexed)
    pub iteration: usize,

    /// Latent state after this iteration
    pub latent_state: Vec<f32>,

    /// Answer state after this iteration
    pub answer_state: Vec<f32>,

    /// Confidence at this iteration
    pub confidence: f32,

    /// Latency for this iteration in microseconds
    pub latency_us: u64,

    /// Confidence delta from previous iteration
    pub confidence_delta: f32,
}

impl TrmIterationState {
    /// Create a new iteration state
    pub fn new(
        iteration: usize,
        latent_state: Vec<f32>,
        answer_state: Vec<f32>,
        confidence: f32,
        latency_us: u64,
        prev_confidence: f32,
    ) -> Self {
        Self {
            iteration,
            latent_state,
            answer_state,
            confidence,
            latency_us,
            confidence_delta: confidence - prev_confidence,
        }
    }
}

/// Full trajectory of TRM reasoning process
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrmTrajectory {
    /// All iteration states
    pub states: Vec<TrmIterationState>,

    /// Optimal K value (actual iterations used)
    pub optimal_k: usize,

    /// Total latency in microseconds
    pub total_latency_us: u64,

    /// Initial question embedding (for similarity matching)
    pub question_embedding: Vec<f32>,

    /// Task type hint (if known)
    pub task_type: Option<String>,

    /// Whether convergence was detected
    pub converged: bool,
}

impl TrmTrajectory {
    /// Create a new empty trajectory
    pub fn new(question_embedding: Vec<f32>) -> Self {
        Self {
            states: Vec::new(),
            optimal_k: 0,
            total_latency_us: 0,
            question_embedding,
            task_type: None,
            converged: false,
        }
    }

    /// Add a state to the trajectory
    pub fn push(&mut self, state: TrmIterationState) {
        self.total_latency_us += state.latency_us;
        self.optimal_k = state.iteration + 1;
        self.states.push(state);
    }

    /// Mark as converged
    pub fn set_converged(&mut self, converged: bool) {
        self.converged = converged;
    }

    /// Get final confidence
    pub fn final_confidence(&self) -> f32 {
        self.states.last().map(|s| s.confidence).unwrap_or(0.0)
    }

    /// Get confidence improvement from first to last
    pub fn confidence_improvement(&self) -> f32 {
        if self.states.len() < 2 {
            return 0.0;
        }
        self.states.last().unwrap().confidence - self.states.first().unwrap().confidence
    }

    /// Check if confidence is plateauing
    pub fn is_plateauing(&self, window: usize, threshold: f32) -> bool {
        if self.states.len() < window {
            return false;
        }

        let recent: Vec<&TrmIterationState> = self.states.iter()
            .rev()
            .take(window)
            .collect();

        let delta = recent.first().unwrap().confidence - recent.last().unwrap().confidence;
        delta.abs() < threshold
    }
}

/// Routing decision for TRM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrmRoutingDecision {
    /// Use TRM recursive reasoning
    pub use_trm: bool,

    /// Predicted optimal K
    pub k_value: usize,

    /// Latent iterations per K
    pub n_value: usize,

    /// Use attention variant
    pub use_attention: bool,

    /// Confidence in routing decision
    pub confidence: f32,

    /// Reasoning for decision
    pub reason: String,
}

impl Default for TrmRoutingDecision {
    fn default() -> Self {
        Self {
            use_trm: true,
            k_value: 5,
            n_value: 3,
            use_attention: false,
            confidence: 0.5,
            reason: "default".to_string(),
        }
    }
}

/// Metadata about TRM execution (for response)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrmInfo {
    /// Iterations used
    pub iterations_used: usize,

    /// Predicted K (if prediction was used)
    pub predicted_k: Option<usize>,

    /// Whether K was predicted by SONA
    pub k_was_predicted: bool,

    /// Whether early stopping occurred
    pub early_stopped: bool,

    /// Final confidence
    pub confidence: f32,

    /// Whether recursion actually ran (vs cache hit)
    pub recursion_ran: bool,

    /// Latency in milliseconds
    pub latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trm_result_creation() {
        let trajectory = TrmTrajectory::new(vec![0.1; 64]);
        let result = TrmResult::new(
            vec![0.5; 64],
            0.85,
            5,
            false,
            trajectory,
            10000,
        );

        assert_eq!(result.iterations_used, 5);
        assert_eq!(result.confidence, 0.85);
        assert!(!result.early_stopped);
    }

    #[test]
    fn test_trajectory_operations() {
        let mut trajectory = TrmTrajectory::new(vec![0.1; 64]);

        for i in 0..5 {
            let state = TrmIterationState::new(
                i,
                vec![0.5; 64],
                vec![0.6; 64],
                0.5 + i as f32 * 0.1,
                1000,
                if i == 0 { 0.0 } else { 0.5 + (i - 1) as f32 * 0.1 },
            );
            trajectory.push(state);
        }

        assert_eq!(trajectory.optimal_k, 5);
        assert_eq!(trajectory.total_latency_us, 5000);
        assert!((trajectory.final_confidence() - 0.9).abs() < 0.01);
        assert!((trajectory.confidence_improvement() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_plateauing_detection() {
        let mut trajectory = TrmTrajectory::new(vec![0.1; 64]);

        // Add states with plateauing confidence
        for i in 0..5 {
            let state = TrmIterationState::new(
                i,
                vec![0.5; 64],
                vec![0.6; 64],
                0.8 + 0.001 * i as f32, // Very small increase
                1000,
                if i == 0 { 0.0 } else { 0.8 + 0.001 * (i - 1) as f32 },
            );
            trajectory.push(state);
        }

        assert!(trajectory.is_plateauing(3, 0.01));
    }

    #[test]
    fn test_serialization() {
        let trajectory = TrmTrajectory::new(vec![0.1; 64]);
        let result = TrmResult::new(
            vec![0.5; 64],
            0.85,
            5,
            false,
            trajectory,
            10000,
        );

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: TrmResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.iterations_used, deserialized.iterations_used);
        assert_eq!(result.confidence, deserialized.confidence);
    }
}
