//! Adaptive Threshold Learning
//!
//! This module provides self-tuning thresholds that adapt based on historical
//! error patterns and system behavior. Uses exponential moving averages and
//! online learning to optimize gate decisions.
//!
//! ## How It Works
//!
//! 1. **Baseline Learning**: Establish normal operating ranges during warmup
//! 2. **Anomaly Detection**: Identify when metrics deviate from baseline
//! 3. **Threshold Adjustment**: Gradually tune thresholds to reduce false positives/negatives
//! 4. **Feedback Integration**: Learn from downstream outcomes (if available)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruqu::adaptive::{AdaptiveThresholds, LearningConfig};
//!
//! let config = LearningConfig::default();
//! let mut adaptive = AdaptiveThresholds::new(config);
//!
//! // During operation
//! let thresholds = adaptive.current_thresholds();
//! let decision = evaluate_with_thresholds(&metrics, &thresholds);
//!
//! // Feed back outcome
//! adaptive.record_outcome(decision, was_correct);
//! ```

use crate::tile::GateThresholds;

/// Configuration for adaptive learning
#[derive(Clone, Debug)]
pub struct LearningConfig {
    /// Learning rate (0.0-1.0), higher = faster adaptation
    pub learning_rate: f64,
    /// History window size for baseline computation
    pub history_window: usize,
    /// Warmup period (samples before adaptation starts)
    pub warmup_samples: usize,
    /// Minimum threshold for structural min-cut
    pub min_structural_threshold: f64,
    /// Maximum threshold for structural min-cut
    pub max_structural_threshold: f64,
    /// Decay factor for exponential moving average
    pub ema_decay: f64,
    /// Enable automatic threshold adjustment
    pub auto_adjust: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            history_window: 10_000,
            warmup_samples: 1_000,
            min_structural_threshold: 1.0,
            max_structural_threshold: 20.0,
            ema_decay: 0.99,
            auto_adjust: true,
        }
    }
}

impl LearningConfig {
    /// Conservative configuration (slow adaptation)
    pub fn conservative() -> Self {
        Self {
            learning_rate: 0.001,
            history_window: 50_000,
            warmup_samples: 5_000,
            ema_decay: 0.999,
            auto_adjust: true,
            ..Default::default()
        }
    }

    /// Aggressive configuration (fast adaptation)
    pub fn aggressive() -> Self {
        Self {
            learning_rate: 0.1,
            history_window: 1_000,
            warmup_samples: 100,
            ema_decay: 0.95,
            auto_adjust: true,
            ..Default::default()
        }
    }
}

/// Running statistics using Welford's algorithm
#[derive(Clone, Debug, Default)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Exponential moving average tracker
#[derive(Clone, Debug)]
struct EMA {
    value: f64,
    decay: f64,
    initialized: bool,
}

impl EMA {
    fn new(decay: f64) -> Self {
        Self {
            value: 0.0,
            decay,
            initialized: false,
        }
    }

    fn update(&mut self, sample: f64) {
        if !self.initialized {
            self.value = sample;
            self.initialized = true;
        } else {
            self.value = self.decay * self.value + (1.0 - self.decay) * sample;
        }
    }

    fn get(&self) -> f64 {
        self.value
    }
}

/// Adaptive threshold manager
pub struct AdaptiveThresholds {
    /// Configuration
    config: LearningConfig,
    /// Current thresholds
    current: GateThresholds,
    /// Statistics for structural cut values
    cut_stats: RunningStats,
    /// Statistics for shift scores
    shift_stats: RunningStats,
    /// Statistics for e-values
    evidence_stats: RunningStats,
    /// EMA of false positive rate
    false_positive_ema: EMA,
    /// EMA of false negative rate
    false_negative_ema: EMA,
    /// Total samples processed
    samples: u64,
    /// Outcomes recorded
    outcomes: OutcomeTracker,
}

/// Tracks decision outcomes for learning
#[derive(Clone, Debug, Default)]
struct OutcomeTracker {
    /// True positives (Deny when should deny)
    true_positives: u64,
    /// True negatives (Permit when should permit)
    true_negatives: u64,
    /// False positives (Deny when should permit)
    false_positives: u64,
    /// False negatives (Permit when should deny)
    false_negatives: u64,
}

impl OutcomeTracker {
    fn record(&mut self, predicted_deny: bool, actual_bad: bool) {
        match (predicted_deny, actual_bad) {
            (true, true) => self.true_positives += 1,
            (false, false) => self.true_negatives += 1,
            (true, false) => self.false_positives += 1,
            (false, true) => self.false_negatives += 1,
        }
    }

    fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            return 1.0;
        }
        self.true_positives as f64 / denom as f64
    }

    fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            return 1.0;
        }
        self.true_positives as f64 / denom as f64
    }

    fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    fn false_positive_rate(&self) -> f64 {
        let denom = self.false_positives + self.true_negatives;
        if denom == 0 {
            return 0.0;
        }
        self.false_positives as f64 / denom as f64
    }

    fn false_negative_rate(&self) -> f64 {
        let denom = self.false_negatives + self.true_positives;
        if denom == 0 {
            return 0.0;
        }
        self.false_negatives as f64 / denom as f64
    }
}

impl AdaptiveThresholds {
    /// Create new adaptive threshold manager
    pub fn new(config: LearningConfig) -> Self {
        let current = GateThresholds::default();

        Self {
            false_positive_ema: EMA::new(config.ema_decay),
            false_negative_ema: EMA::new(config.ema_decay),
            config,
            current,
            cut_stats: RunningStats::new(),
            shift_stats: RunningStats::new(),
            evidence_stats: RunningStats::new(),
            samples: 0,
            outcomes: OutcomeTracker::default(),
        }
    }

    /// Record observed metrics (call every cycle)
    pub fn record_metrics(&mut self, cut: f64, shift: f64, e_value: f64) {
        self.cut_stats.update(cut);
        self.shift_stats.update(shift);
        self.evidence_stats.update(e_value);
        self.samples += 1;

        // Adjust thresholds after warmup
        if self.config.auto_adjust && self.samples > self.config.warmup_samples as u64 {
            self.adjust_thresholds();
        }
    }

    /// Record decision outcome for learning
    ///
    /// # Arguments
    /// * `was_deny` - True if gate decided Deny
    /// * `was_actually_bad` - True if there was an actual error (ground truth)
    pub fn record_outcome(&mut self, was_deny: bool, was_actually_bad: bool) {
        self.outcomes.record(was_deny, was_actually_bad);

        // Update EMAs
        let fp = if was_deny && !was_actually_bad { 1.0 } else { 0.0 };
        let fn_rate = if !was_deny && was_actually_bad { 1.0 } else { 0.0 };

        self.false_positive_ema.update(fp);
        self.false_negative_ema.update(fn_rate);

        // Adjust thresholds based on outcome
        if self.config.auto_adjust && self.samples > self.config.warmup_samples as u64 {
            self.adjust_from_outcome(was_deny, was_actually_bad);
        }
    }

    /// Get current thresholds
    pub fn current_thresholds(&self) -> &GateThresholds {
        &self.current
    }

    /// Get mutable thresholds for manual adjustment
    pub fn current_thresholds_mut(&mut self) -> &mut GateThresholds {
        &mut self.current
    }

    /// Check if warmup period is complete
    pub fn is_warmed_up(&self) -> bool {
        self.samples >= self.config.warmup_samples as u64
    }

    /// Get learning statistics
    pub fn stats(&self) -> AdaptiveStats {
        AdaptiveStats {
            samples: self.samples,
            cut_mean: self.cut_stats.mean,
            cut_std: self.cut_stats.std_dev(),
            shift_mean: self.shift_stats.mean,
            shift_std: self.shift_stats.std_dev(),
            evidence_mean: self.evidence_stats.mean,
            precision: self.outcomes.precision(),
            recall: self.outcomes.recall(),
            f1_score: self.outcomes.f1_score(),
            false_positive_rate: self.false_positive_ema.get(),
            false_negative_rate: self.false_negative_ema.get(),
        }
    }

    /// Reset learning state
    pub fn reset(&mut self) {
        self.cut_stats = RunningStats::new();
        self.shift_stats = RunningStats::new();
        self.evidence_stats = RunningStats::new();
        self.false_positive_ema = EMA::new(self.config.ema_decay);
        self.false_negative_ema = EMA::new(self.config.ema_decay);
        self.samples = 0;
        self.outcomes = OutcomeTracker::default();
    }

    // Private methods

    fn adjust_thresholds(&mut self) {
        let lr = self.config.learning_rate;

        // Adjust structural threshold based on observed cut distribution
        // Target: threshold = mean - 2*std (catch 95% of normal operation)
        if self.cut_stats.count > 100 {
            let target = self.cut_stats.mean - 2.0 * self.cut_stats.std_dev();
            let target = target.clamp(
                self.config.min_structural_threshold,
                self.config.max_structural_threshold,
            );

            self.current.structural_min_cut =
                self.current.structural_min_cut * (1.0 - lr) + target * lr;
        }

        // Adjust shift threshold based on observed distribution
        // Target: threshold = mean + 2*std
        if self.shift_stats.count > 100 {
            let target = (self.shift_stats.mean + 2.0 * self.shift_stats.std_dev()).min(1.0);
            self.current.shift_max =
                self.current.shift_max * (1.0 - lr) + target * lr;
        }

        // Adjust evidence thresholds
        if self.evidence_stats.count > 100 {
            // tau_deny should be well below normal (5th percentile estimate)
            let tau_deny_target = (self.evidence_stats.mean - 2.0 * self.evidence_stats.std_dev())
                .max(0.001);
            self.current.tau_deny =
                self.current.tau_deny * (1.0 - lr) + tau_deny_target * lr;

            // tau_permit should be above normal (75th percentile estimate)
            let tau_permit_target = self.evidence_stats.mean + 0.5 * self.evidence_stats.std_dev();
            self.current.tau_permit =
                self.current.tau_permit * (1.0 - lr) + tau_permit_target * lr;
        }
    }

    fn adjust_from_outcome(&mut self, was_deny: bool, was_actually_bad: bool) {
        let lr = self.config.learning_rate * 0.1; // Slower adjustment from outcomes

        match (was_deny, was_actually_bad) {
            (true, false) => {
                // False positive: we denied but it was fine
                // → Relax thresholds (lower structural, raise shift)
                self.current.structural_min_cut *= 1.0 - lr;
                self.current.shift_max = (self.current.shift_max + lr).min(1.0);
            }
            (false, true) => {
                // False negative: we permitted but it was bad
                // → Tighten thresholds (raise structural, lower shift)
                self.current.structural_min_cut *= 1.0 + lr;
                self.current.shift_max = (self.current.shift_max - lr).max(0.1);
            }
            _ => {
                // Correct decision: no adjustment needed
            }
        }

        // Clamp thresholds to valid ranges
        self.current.structural_min_cut = self.current.structural_min_cut.clamp(
            self.config.min_structural_threshold,
            self.config.max_structural_threshold,
        );
    }
}

/// Statistics from adaptive learning
#[derive(Clone, Debug, Default)]
pub struct AdaptiveStats {
    /// Total samples processed
    pub samples: u64,
    /// Mean observed cut value
    pub cut_mean: f64,
    /// Standard deviation of cut values
    pub cut_std: f64,
    /// Mean observed shift score
    pub shift_mean: f64,
    /// Standard deviation of shift scores
    pub shift_std: f64,
    /// Mean observed e-value
    pub evidence_mean: f64,
    /// Precision (true positives / predicted positives)
    pub precision: f64,
    /// Recall (true positives / actual positives)
    pub recall: f64,
    /// F1 score (harmonic mean of precision and recall)
    pub f1_score: f64,
    /// Current false positive rate (EMA)
    pub false_positive_rate: f64,
    /// Current false negative rate (EMA)
    pub false_negative_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_config_default() {
        let config = LearningConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert!(config.auto_adjust);
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();

        for i in 1..=100 {
            stats.update(i as f64);
        }

        assert_eq!(stats.count, 100);
        assert!((stats.mean - 50.5).abs() < 0.001);
        assert!(stats.std_dev() > 0.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 100.0);
    }

    #[test]
    fn test_ema() {
        let mut ema = EMA::new(0.9);

        ema.update(100.0);
        assert_eq!(ema.get(), 100.0);

        ema.update(0.0);
        assert!((ema.get() - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_thresholds_creation() {
        let config = LearningConfig::default();
        let adaptive = AdaptiveThresholds::new(config);

        assert!(!adaptive.is_warmed_up());
        assert_eq!(adaptive.samples, 0);
    }

    #[test]
    fn test_adaptive_metrics_recording() {
        let config = LearningConfig {
            warmup_samples: 10,
            ..Default::default()
        };
        let mut adaptive = AdaptiveThresholds::new(config);

        for i in 0..20 {
            adaptive.record_metrics(10.0 + i as f64 * 0.1, 0.2, 100.0);
        }

        assert!(adaptive.is_warmed_up());
        assert_eq!(adaptive.samples, 20);
    }

    #[test]
    fn test_outcome_tracker() {
        let mut tracker = OutcomeTracker::default();

        // 8 true positives
        for _ in 0..8 {
            tracker.record(true, true);
        }
        // 2 false positives
        for _ in 0..2 {
            tracker.record(true, false);
        }

        assert_eq!(tracker.precision(), 0.8);
    }

    #[test]
    fn test_adaptive_stats() {
        let config = LearningConfig {
            warmup_samples: 5,
            ..Default::default()
        };
        let mut adaptive = AdaptiveThresholds::new(config);

        for _ in 0..10 {
            adaptive.record_metrics(10.0, 0.2, 100.0);
        }

        let stats = adaptive.stats();
        assert_eq!(stats.samples, 10);
        assert!((stats.cut_mean - 10.0).abs() < 0.001);
    }
}
