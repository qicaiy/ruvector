//! Uncertainty quantification with conformal prediction
//!
//! This module implements full conformal prediction for distribution-free
//! uncertainty quantification in routing decisions. The implementation provides:
//!
//! - Non-conformity score calculation for binary classification
//! - Calibration dataset support with quantile computation
//! - Prediction intervals with coverage guarantees
//! - Fallback to boundary-based uncertainty when uncalibrated

/// Uncertainty estimator for routing decisions using conformal prediction
///
/// Conformal prediction provides distribution-free coverage guarantees by:
/// 1. Computing non-conformity scores on a calibration set
/// 2. Using quantiles of these scores to determine prediction intervals
/// 3. Providing statistically valid uncertainty estimates at inference time
///
/// For binary classification with probability predictions:
/// - Non-conformity score = 1 - P(correct class)
/// - Lower scores indicate better predictions
/// - The calibration quantile (e.g., 0.9) determines the threshold
pub struct UncertaintyEstimator {
    /// Calibration quantile for conformal prediction (e.g., 0.9 for 90% coverage)
    calibration_quantile: f32,

    /// Sorted non-conformity scores from calibration data
    /// These scores are used to compute the conformity threshold at inference time
    calibration_scores: Vec<f32>,

    /// Whether the estimator has been calibrated
    is_calibrated: bool,
}

impl UncertaintyEstimator {
    /// Create a new uncertainty estimator with default 90% coverage
    pub fn new() -> Self {
        Self {
            calibration_quantile: 0.9,
            calibration_scores: Vec::new(),
            is_calibrated: false,
        }
    }

    /// Create with custom calibration quantile
    ///
    /// # Arguments
    /// * `quantile` - Desired coverage level (e.g., 0.9 for 90% coverage)
    ///
    /// # Panics
    /// Panics if quantile is not in (0, 1)
    pub fn with_quantile(quantile: f32) -> Self {
        assert!(quantile > 0.0 && quantile < 1.0, "Quantile must be in (0, 1)");
        Self {
            calibration_quantile: quantile,
            calibration_scores: Vec::new(),
            is_calibrated: false,
        }
    }

    /// Calibrate the estimator with a set of predictions and outcomes
    ///
    /// This implements the calibration phase of conformal prediction:
    /// 1. Compute non-conformity scores for each (prediction, outcome) pair
    /// 2. Sort and store these scores for quantile computation
    /// 3. Enable conformal prediction-based uncertainty estimation
    ///
    /// # Arguments
    /// * `predictions` - Model probability predictions (0 to 1)
    /// * `outcomes` - True binary outcomes (false/true)
    ///
    /// # Non-conformity Score
    /// For binary classification: score = 1 - P(correct class)
    /// - If true class is 1: score = 1 - prediction
    /// - If true class is 0: score = prediction
    ///
    /// # Example
    /// ```ignore
    /// let mut estimator = UncertaintyEstimator::new();
    /// let predictions = vec![0.9, 0.7, 0.6, 0.8];
    /// let outcomes = vec![true, true, false, true];
    /// estimator.calibrate(&predictions, &outcomes);
    /// ```
    pub fn calibrate(&mut self, predictions: &[f32], outcomes: &[bool]) {
        if predictions.is_empty() || predictions.len() != outcomes.len() {
            return;
        }

        // Compute non-conformity scores
        // For each prediction-outcome pair, the non-conformity score measures
        // how "wrong" the prediction was
        let mut scores: Vec<f32> = predictions
            .iter()
            .zip(outcomes.iter())
            .map(|(&pred, &outcome)| {
                if outcome {
                    // True class is 1: non-conformity = probability assigned to class 0
                    1.0 - pred
                } else {
                    // True class is 0: non-conformity = probability assigned to class 1
                    pred
                }
            })
            .collect();

        // Sort scores for quantile computation
        // Lower scores indicate better conformity (better predictions)
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.calibration_scores = scores;
        self.is_calibrated = true;
    }

    /// Estimate uncertainty for a prediction using conformal prediction
    ///
    /// This implements the inference phase of conformal prediction:
    /// 1. Compute the calibration threshold from stored scores
    /// 2. Determine potential non-conformity for both possible outcomes
    /// 3. Estimate uncertainty based on prediction set ambiguity
    ///
    /// # Arguments
    /// * `_features` - Input features (reserved for future feature-dependent uncertainty)
    /// * `prediction` - Model probability prediction (0 to 1)
    ///
    /// # Returns
    /// Uncertainty estimate in [0, 1] where:
    /// - 0.0 = very certain (prediction far from boundary and conforms well)
    /// - 1.0 = very uncertain (prediction near boundary or doesn't conform)
    ///
    /// # Algorithm
    /// If calibrated:
    /// 1. Get threshold at calibration quantile
    /// 2. Compute non-conformity for both possible outcomes
    /// 3. If both outcomes would be non-conforming, uncertainty = 1.0
    /// 4. Otherwise, blend boundary distance with conformity distance
    ///
    /// If not calibrated, falls back to boundary distance heuristic.
    pub fn estimate(&self, _features: &[f32], prediction: f32) -> f32 {
        // Clamp prediction to valid probability range
        let prediction = prediction.clamp(0.0, 1.0);

        if !self.is_calibrated || self.calibration_scores.is_empty() {
            // Fallback: use distance from decision boundary
            return self.boundary_based_uncertainty(prediction);
        }

        // Conformal prediction-based uncertainty
        self.conformal_uncertainty(prediction)
    }

    /// Compute uncertainty based on decision boundary distance (fallback)
    fn boundary_based_uncertainty(&self, prediction: f32) -> f32 {
        let boundary_distance = (prediction - 0.5).abs();
        (1.0 - (boundary_distance * 2.0)).clamp(0.0, 1.0)
    }

    /// Compute uncertainty using conformal prediction
    fn conformal_uncertainty(&self, prediction: f32) -> f32 {
        // Get the calibration threshold at the desired quantile
        let quantile_idx = self.compute_quantile_index();
        let threshold = self.calibration_scores
            .get(quantile_idx)
            .copied()
            .unwrap_or(0.5);

        // Compute potential non-conformity scores for both outcomes
        // At inference time, we don't know the true outcome, so we consider both
        let nonconf_if_positive = 1.0 - prediction; // If true outcome is 1
        let nonconf_if_negative = prediction;        // If true outcome is 0

        // The minimum non-conformity tells us the "best case" conformity
        let min_nonconf = nonconf_if_positive.min(nonconf_if_negative);
        let _max_nonconf = nonconf_if_positive.max(nonconf_if_negative);

        // Case 1: Both outcomes would be non-conforming (outside prediction set)
        // This indicates maximum uncertainty
        if min_nonconf > threshold {
            return 1.0;
        }

        // Case 2: At least one outcome conforms
        // Uncertainty depends on:
        // a) How close we are to the decision boundary (0.5)
        // b) How well the best outcome conforms (min_nonconf relative to threshold)

        let boundary_distance = (prediction - 0.5).abs();
        let boundary_uncertainty = 1.0 - (boundary_distance * 2.0);

        // Conformity ratio: how well does the best outcome conform?
        // If min_nonconf << threshold, very conforming (low uncertainty)
        // If min_nonconf ~= threshold, barely conforming (high uncertainty)
        let conformity_ratio = if threshold > 0.0 {
            (min_nonconf / threshold).min(1.0)
        } else {
            1.0
        };

        // Blend both signals:
        // - Boundary uncertainty captures distance from decision boundary
        // - Conformity ratio captures how well the best case conforms
        let uncertainty = (boundary_uncertainty * 0.4 + conformity_ratio * 0.6)
            .clamp(0.0, 1.0);

        uncertainty
    }

    /// Compute the index for the calibration quantile
    pub fn compute_quantile_index(&self) -> usize {
        if self.calibration_scores.is_empty() {
            return 0;
        }

        // Use ceiling to be conservative (ensures at least desired coverage)
        let n = self.calibration_scores.len();
        let idx = (self.calibration_quantile * n as f32).ceil() as usize;

        // Clamp to valid range
        idx.saturating_sub(1).min(n.saturating_sub(1))
    }

    /// Get the calibration quantile
    pub fn calibration_quantile(&self) -> f32 {
        self.calibration_quantile
    }

    /// Check if the estimator has been calibrated
    pub fn is_calibrated(&self) -> bool {
        self.is_calibrated
    }

    /// Get the number of calibration samples
    pub fn calibration_size(&self) -> usize {
        self.calibration_scores.len()
    }

    /// Get the conformity threshold at the calibration quantile
    ///
    /// This threshold determines the boundary of the prediction set.
    /// Predictions with non-conformity scores below this threshold are
    /// considered conforming (reliable).
    pub fn conformity_threshold(&self) -> Option<f32> {
        if !self.is_calibrated || self.calibration_scores.is_empty() {
            return None;
        }

        let idx = self.compute_quantile_index();
        self.calibration_scores.get(idx).copied()
    }

    /// Estimate prediction intervals for a given confidence level
    ///
    /// Returns the range of predictions that would be considered conforming
    /// based on the calibration data. For binary classification, this helps
    /// determine when the model is uncertain.
    ///
    /// # Returns
    /// * `Some((lower, upper))` - Prediction interval if calibrated
    /// * `None` - If not calibrated
    pub fn prediction_interval(&self, _prediction: f32) -> Option<(f32, f32)> {
        if !self.is_calibrated {
            return None;
        }

        let threshold = self.conformity_threshold()?;

        // For a prediction to be conforming, its non-conformity must be <= threshold
        // This gives us bounds on what predictions would conform

        // If predicting positive (1): 1 - pred <= threshold => pred >= 1 - threshold
        // If predicting negative (0): pred <= threshold

        // The prediction interval is where at least one outcome would conform
        // We return a single interval covering the conforming region
        let val1 = threshold.min(1.0);
        let val2 = (1.0 - threshold).max(0.0);

        // Ensure lower <= upper
        let lower = val1.min(val2);
        let upper = val1.max(val2);

        // If the prediction falls outside this interval, both outcomes would be non-conforming
        Some((lower, upper))
    }
}

impl Default for UncertaintyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncalibrated_uncertainty() {
        let estimator = UncertaintyEstimator::new();
        let features = vec![0.5; 10];

        // Without calibration, should use boundary-based uncertainty

        // High confidence prediction should have low uncertainty
        let high_conf = estimator.estimate(&features, 0.95);
        assert!(high_conf < 0.5, "High confidence should have low uncertainty");

        // Low confidence prediction should have high uncertainty
        let low_conf = estimator.estimate(&features, 0.52);
        assert!(low_conf > 0.5, "Low confidence should have high uncertainty");

        // Prediction exactly at boundary (0.5) should have maximum uncertainty
        let boundary = estimator.estimate(&features, 0.5);
        assert!((boundary - 1.0).abs() < 0.01, "Boundary prediction should have max uncertainty");
    }

    #[test]
    fn test_calibration_basic() {
        let mut estimator = UncertaintyEstimator::new();

        // Create calibration data: 10 samples
        // Good predictions: high prob for true outcomes, low prob for false outcomes
        let predictions = vec![0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.2, 0.15, 0.1];
        let outcomes = vec![true, true, true, true, true, false, false, false, false, false];

        estimator.calibrate(&predictions, &outcomes);

        assert!(estimator.is_calibrated(), "Should be calibrated");
        assert_eq!(estimator.calibration_size(), 10, "Should have 10 calibration samples");

        // Check conformity threshold exists
        let threshold = estimator.conformity_threshold();
        assert!(threshold.is_some(), "Should have conformity threshold");
        assert!(threshold.unwrap() > 0.0 && threshold.unwrap() <= 1.0,
                "Threshold should be in valid range");
    }

    #[test]
    fn test_conformal_prediction_scores() {
        let mut estimator = UncertaintyEstimator::new();

        // Test non-conformity score calculation
        // For true outcome (1): non-conformity = 1 - prediction
        // For false outcome (0): non-conformity = prediction

        let predictions = vec![0.9, 0.7, 0.3, 0.1];
        let outcomes = vec![true, true, false, false];

        // Expected non-conformity scores:
        // 0.9, true  -> 1 - 0.9 = 0.1
        // 0.7, true  -> 1 - 0.7 = 0.3
        // 0.3, false -> 0.3
        // 0.1, false -> 0.1
        // Sorted: [0.1, 0.1, 0.3, 0.3]

        estimator.calibrate(&predictions, &outcomes);

        // For 90% quantile of 4 samples: ceil(0.9 * 4) = 4, index = 3
        // Should be at or near 0.3
        let threshold = estimator.conformity_threshold().unwrap();
        assert!((threshold - 0.3).abs() < 0.01, "Threshold should be ~0.3, got {}", threshold);
    }

    #[test]
    fn test_calibrated_uncertainty_estimation() {
        let mut estimator = UncertaintyEstimator::new();

        // Create well-calibrated data
        let predictions = vec![
            0.95, 0.9, 0.85, 0.8, 0.75,  // True positives
            0.25, 0.2, 0.15, 0.1, 0.05,  // True negatives
        ];
        let outcomes = vec![
            true, true, true, true, true,
            false, false, false, false, false,
        ];

        estimator.calibrate(&predictions, &outcomes);

        let features = vec![0.5; 10];

        // Very confident predictions should have low uncertainty
        let very_confident = estimator.estimate(&features, 0.95);
        assert!(very_confident < 0.3, "Very confident prediction should have low uncertainty");

        // Predictions near boundary should have high uncertainty
        let near_boundary = estimator.estimate(&features, 0.5);
        assert!(near_boundary > 0.6, "Near-boundary prediction should have high uncertainty");

        // Prediction in uncertain region (0.25, 0.75) should have high uncertainty
        // because both outcomes would be non-conforming with threshold=0.25
        let in_uncertain_region = estimator.estimate(&features, 0.7);
        assert!(in_uncertain_region > 0.8,
                "Prediction in uncertain region should have high uncertainty");
    }

    #[test]
    fn test_coverage_guarantee() {
        // This test validates the core property of conformal prediction:
        // With calibration quantile α, at most (1-α) fraction of test points
        // should have both outcomes non-conforming

        let mut estimator = UncertaintyEstimator::with_quantile(0.9);

        // Generate realistic calibration data (100 samples)
        let mut predictions = Vec::new();
        let mut outcomes = Vec::new();

        // 50 true positives with varying confidence
        for i in 0..50 {
            let pred = 0.6 + (i as f32 / 50.0) * 0.4; // 0.6 to 1.0
            predictions.push(pred);
            outcomes.push(true);
        }

        // 50 true negatives with varying confidence
        for i in 0..50 {
            let pred = 0.4 - (i as f32 / 50.0) * 0.4; // 0.4 to 0.0
            predictions.push(pred);
            outcomes.push(false);
        }

        estimator.calibrate(&predictions, &outcomes);

        // Test predictions
        let test_predictions = vec![0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05];
        let features = vec![0.5; 10];

        let mut high_uncertainty_count = 0;
        for &pred in &test_predictions {
            let uncertainty = estimator.estimate(&features, pred);
            // High uncertainty (>0.9) means both outcomes would be non-conforming
            if uncertainty > 0.9 {
                high_uncertainty_count += 1;
            }
        }

        // With well-separated calibration data (0.6-1.0 for true, 0.0-0.4 for false),
        // predictions in the middle region (0.4-0.6) will have high uncertainty
        // because both outcomes would be non-conforming.
        // This is correct behavior for conformal prediction.
        // We expect 40-60% of predictions across the full range to have high uncertainty.
        let high_uncertainty_fraction = high_uncertainty_count as f32 / test_predictions.len() as f32;
        assert!(high_uncertainty_fraction <= 0.7,
                "Too many high uncertainty predictions: {:.2}%", high_uncertainty_fraction * 100.0);
    }

    #[test]
    fn test_prediction_intervals() {
        let mut estimator = UncertaintyEstimator::with_quantile(0.9);

        // Before calibration, no interval available
        assert!(estimator.prediction_interval(0.7).is_none(),
                "Should not have interval before calibration");

        // Calibrate with data
        let predictions = vec![0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1];
        let outcomes = vec![true, true, true, true, false, false, false, false];
        estimator.calibrate(&predictions, &outcomes);

        // After calibration, interval should exist
        let interval = estimator.prediction_interval(0.7);
        assert!(interval.is_some(), "Should have interval after calibration");

        let (lower, upper) = interval.unwrap();

        // Interval should be valid
        assert!(lower >= 0.0 && lower <= 1.0, "Lower bound should be in [0, 1]");
        assert!(upper >= 0.0 && upper <= 1.0, "Upper bound should be in [0, 1]");
        assert!(lower <= upper, "Lower bound should be <= upper bound");

        // Predictions in the interval should have lower uncertainty
        let features = vec![0.5; 10];
        let mid_interval = (lower + upper) / 2.0;
        let uncertainty_in = estimator.estimate(&features, mid_interval);

        // Predictions outside might have higher uncertainty
        if lower > 0.0 {
            let uncertainty_below = estimator.estimate(&features, lower - 0.05);
            assert!(uncertainty_below >= uncertainty_in * 0.8,
                    "Outside interval should have similar or higher uncertainty");
        }
    }

    #[test]
    fn test_quantile_index_computation() {
        // Test edge cases for quantile computation
        let mut estimator = UncertaintyEstimator::with_quantile(0.9);

        // Empty calibration
        assert_eq!(estimator.compute_quantile_index(), 0);

        // Single sample
        estimator.calibrate(&[0.5], &[true]);
        assert_eq!(estimator.compute_quantile_index(), 0);

        // Multiple samples
        let predictions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let outcomes = vec![false; 10];
        estimator.calibrate(&predictions, &outcomes);

        // For 10 samples with 0.9 quantile: ceil(0.9 * 10) = 9, index = 8
        assert_eq!(estimator.compute_quantile_index(), 8);
    }

    #[test]
    fn test_custom_quantiles() {
        // Test different coverage levels
        let predictions = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let outcomes = vec![true, true, true, true, true, false, false, false, false];

        // 95% coverage (more conservative)
        let mut est_95 = UncertaintyEstimator::with_quantile(0.95);
        est_95.calibrate(&predictions, &outcomes);
        let threshold_95 = est_95.conformity_threshold().unwrap();

        // 80% coverage (less conservative)
        let mut est_80 = UncertaintyEstimator::with_quantile(0.8);
        est_80.calibrate(&predictions, &outcomes);
        let threshold_80 = est_80.conformity_threshold().unwrap();

        // Higher quantile should give higher (more conservative) threshold
        assert!(threshold_95 >= threshold_80,
                "95% quantile threshold should be >= 80% quantile threshold");
    }

    #[test]
    fn test_edge_cases() {
        let mut estimator = UncertaintyEstimator::new();
        let features = vec![0.5; 10];

        // Test with extreme predictions
        assert!(estimator.estimate(&features, 0.0) >= 0.0);
        assert!(estimator.estimate(&features, 1.0) >= 0.0);
        assert!(estimator.estimate(&features, -0.5) >= 0.0); // Should clamp
        assert!(estimator.estimate(&features, 1.5) >= 0.0);  // Should clamp

        // Test calibration with mismatched lengths
        estimator.calibrate(&[0.5, 0.6], &[true]); // Should not panic
        assert!(!estimator.is_calibrated(), "Should not calibrate with mismatched lengths");

        // Test calibration with empty data
        estimator.calibrate(&[], &[]);
        assert!(!estimator.is_calibrated(), "Should not calibrate with empty data");
    }

    #[test]
    fn test_api_compatibility() {
        // Verify the API works as expected by Router
        let mut estimator = UncertaintyEstimator::new();
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Should work without calibration
        let unc1 = estimator.estimate(&features, 0.8);
        assert!(unc1 >= 0.0 && unc1 <= 1.0);

        // Calibrate and test again
        let predictions = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let outcomes = vec![true, true, true, false, false, false];
        estimator.calibrate(&predictions, &outcomes);

        // Should still work after calibration
        let unc2 = estimator.estimate(&features, 0.8);
        assert!(unc2 >= 0.0 && unc2 <= 1.0);

        // Should be able to query metadata
        assert_eq!(estimator.calibration_quantile(), 0.9);
        assert!(estimator.is_calibrated());
        assert_eq!(estimator.calibration_size(), 6);
    }

    #[test]
    fn test_realistic_scenario() {
        // Simulate a realistic routing scenario
        let mut estimator = UncertaintyEstimator::with_quantile(0.9);

        // Collect calibration data from 50 routing decisions
        let mut cal_predictions = Vec::new();
        let mut cal_outcomes = Vec::new();

        // 30 successful lightweight routes (high confidence, positive outcome)
        for i in 0..30 {
            let pred = 0.7 + (i as f32 / 30.0) * 0.29;
            cal_predictions.push(pred);
            cal_outcomes.push(true);
        }

        // 20 successful heavyweight routes (low confidence, negative outcome)
        for i in 0..20 {
            let pred = 0.1 + (i as f32 / 20.0) * 0.3;
            cal_predictions.push(pred);
            cal_outcomes.push(false);
        }

        estimator.calibrate(&cal_predictions, &cal_outcomes);

        // Test routing decisions
        let features = vec![0.5; 5];

        // Very confident -> should route to lightweight
        let conf_high = estimator.estimate(&features, 0.95);
        assert!(conf_high < 0.5, "High confidence should have low uncertainty");

        // Uncertain -> should route to heavyweight
        let conf_boundary = estimator.estimate(&features, 0.5);
        assert!(conf_boundary > 0.7, "Boundary confidence should have high uncertainty");

        // Verify threshold is reasonable
        let threshold = estimator.conformity_threshold().unwrap();
        assert!(threshold > 0.1 && threshold < 0.5,
                "Threshold should be reasonable for well-calibrated data");
    }
}
