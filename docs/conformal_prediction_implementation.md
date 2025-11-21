# Conformal Prediction Implementation for Tiny Dancer

## Overview

This document describes the full conformal prediction implementation for uncertainty quantification in the Tiny Dancer routing system (`/home/user/ruvector/crates/ruvector-tiny-dancer-core/src/uncertainty.rs`).

## What is Conformal Prediction?

Conformal prediction is a framework for producing prediction intervals with distribution-free coverage guarantees. Unlike traditional uncertainty quantification methods that rely on assumptions about the data distribution, conformal prediction provides valid uncertainty estimates regardless of the underlying distribution.

### Key Concepts

1. **Non-conformity Score**: A measure of how "strange" or unexpected a prediction is
2. **Calibration Set**: A held-out dataset used to compute the distribution of non-conformity scores
3. **Coverage Guarantee**: With calibration quantile α (e.g., 0.9), at least α fraction of predictions will have valid prediction intervals

## Implementation Details

### Data Structure

```rust
pub struct UncertaintyEstimator {
    /// Calibration quantile (e.g., 0.9 for 90% coverage)
    calibration_quantile: f32,

    /// Sorted non-conformity scores from calibration data
    calibration_scores: Vec<f32>,

    /// Whether the estimator has been calibrated
    is_calibrated: bool,
}
```

### Non-conformity Score Calculation

For binary classification with probability predictions, we use:

**Non-conformity score = 1 - P(correct class)**

- If true class is 1 (positive): score = 1 - prediction
- If true class is 0 (negative): score = prediction

Lower scores indicate better conformity (more accurate predictions).

### Calibration Phase

```rust
pub fn calibrate(&mut self, predictions: &[f32], outcomes: &[bool])
```

**Algorithm:**
1. For each (prediction, outcome) pair, compute non-conformity score
2. Sort all scores in ascending order
3. Store sorted scores for quantile computation at inference time

**Example:**
```rust
let mut estimator = UncertaintyEstimator::new();

// Calibration data: predictions and their true outcomes
let predictions = vec![0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.2, 0.15, 0.1];
let outcomes = vec![true, true, true, true, true, false, false, false, false, false];

estimator.calibrate(&predictions, &outcomes);
```

### Inference Phase

```rust
pub fn estimate(&self, features: &[f32], prediction: f32) -> f32
```

**Algorithm:**

1. **If not calibrated**: Fall back to boundary-based uncertainty
   - Uncertainty = 1 - 2 × |prediction - 0.5|
   - High uncertainty near decision boundary (0.5)

2. **If calibrated**: Use conformal prediction
   - Compute conformity threshold at calibration quantile
   - Calculate potential non-conformity for both outcomes:
     - `nonconf_if_positive = 1 - prediction`
     - `nonconf_if_negative = prediction`
   - If both outcomes would be non-conforming: uncertainty = 1.0
   - Otherwise: blend boundary distance with threshold distance

**Uncertainty Score Interpretation:**
- 0.0: Very certain (prediction far from boundary and conforms well)
- 1.0: Very uncertain (prediction near boundary or doesn't conform)

### Additional Methods

#### Get Conformity Threshold
```rust
pub fn conformity_threshold(&self) -> Option<f32>
```

Returns the non-conformity threshold at the calibration quantile. Predictions with non-conformity scores below this threshold are considered reliable.

#### Prediction Intervals
```rust
pub fn prediction_interval(&self, prediction: f32) -> Option<(f32, f32)>
```

Returns the range of predictions that would be considered conforming based on calibration data:
- Lower bound: `1 - threshold` (minimum prediction for positive class)
- Upper bound: `threshold` (maximum prediction for negative class)

#### Calibration Metadata
```rust
pub fn is_calibrated(&self) -> bool
pub fn calibration_size(&self) -> usize
pub fn calibration_quantile(&self) -> f32
```

Query calibration status and parameters.

## Coverage Guarantee

**Theorem**: With calibration quantile α and n calibration samples, conformal prediction guarantees that at most ⌈(1-α)(n+1)⌉ / n fraction of test predictions will have both outcomes non-conforming.

For α = 0.9 and large n, this means approximately 90% coverage.

## API Compatibility

The implementation maintains full API compatibility with the existing Router usage:

```rust
// In Router::route()
let uncertainty = self.uncertainty_estimator.estimate(&features.features, score);

let use_lightweight = score >= self.config.confidence_threshold
    && uncertainty <= self.config.max_uncertainty;
```

The API works both before and after calibration, gracefully falling back to boundary-based uncertainty when not calibrated.

## Test Coverage

The implementation includes 12 comprehensive tests:

### Basic Functionality
1. `test_uncalibrated_uncertainty` - Boundary-based fallback
2. `test_calibration_basic` - Basic calibration workflow
3. `test_conformal_prediction_scores` - Non-conformity score calculation
4. `test_calibrated_uncertainty_estimation` - Uncertainty after calibration

### Coverage Guarantees
5. `test_coverage_guarantee` - Validates 90% coverage property with 100 calibration samples

### Advanced Features
6. `test_prediction_intervals` - Prediction interval computation
7. `test_quantile_index_computation` - Edge cases for quantile calculation
8. `test_custom_quantiles` - Different coverage levels (80%, 95%)

### Robustness
9. `test_edge_cases` - Extreme values, mismatched lengths, empty data
10. `test_api_compatibility` - Verifies Router integration
11. `test_realistic_scenario` - End-to-end routing simulation

## Usage Example

### Offline Calibration
```rust
// Collect routing decisions and outcomes
let mut cal_predictions = Vec::new();
let mut cal_outcomes = Vec::new();

for routing_decision in historical_data {
    cal_predictions.push(routing_decision.confidence);
    // true = lightweight was correct choice
    // false = heavyweight was correct choice
    cal_outcomes.push(routing_decision.was_lightweight_correct);
}

// Calibrate the estimator
let mut estimator = UncertaintyEstimator::with_quantile(0.9);
estimator.calibrate(&cal_predictions, &cal_outcomes);
```

### Online Inference
```rust
// During routing
let confidence = model.forward(&features)?;
let uncertainty = estimator.estimate(&features, confidence);

// Route based on uncertainty-aware threshold
let use_lightweight = confidence >= 0.7 && uncertainty <= 0.3;
```

## Performance Characteristics

- **Calibration**: O(n log n) due to sorting, where n = calibration set size
- **Inference**: O(1) - constant time lookup after calibration
- **Memory**: O(n) - stores sorted calibration scores
- **Thread Safety**: Immutable after calibration, safe for concurrent inference

## Key Benefits

1. **Distribution-Free**: No assumptions about data distribution
2. **Rigorous Guarantees**: Mathematical coverage guarantees
3. **Adaptive**: Uncertainty estimates adapt to calibration data quality
4. **Graceful Degradation**: Falls back to heuristic when not calibrated
5. **Low Overhead**: Minimal computational cost at inference time

## Integration with Router

The uncertainty estimator integrates seamlessly with the existing router:

```rust
pub struct Router {
    config: RouterConfig,
    model: Arc<RwLock<FastGRNN>>,
    feature_engineer: FeatureEngineer,
    uncertainty_estimator: UncertaintyEstimator,  // ← Uses conformal prediction
    circuit_breaker: Option<CircuitBreaker>,
}
```

The router uses uncertainty estimates to make routing decisions:
- **High confidence, low uncertainty** → Route to lightweight model
- **Low confidence or high uncertainty** → Route to heavyweight model

## Future Enhancements

Potential improvements:
1. **Feature-dependent uncertainty**: Use `features` parameter for more nuanced estimates
2. **Online calibration**: Update calibration scores incrementally
3. **Multiple quantiles**: Support different coverage levels for different use cases
4. **Adaptive quantiles**: Automatically adjust based on observed error rates
5. **Class-conditional calibration**: Separate calibration for positive/negative classes

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*
- Shafer, G., & Vovk, V. (2008). *A Tutorial on Conformal Prediction*
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*

## Conclusion

This implementation provides a theoretically grounded, practically efficient uncertainty quantification system for the Tiny Dancer routing engine. The conformal prediction framework ensures reliable uncertainty estimates with distribution-free coverage guarantees, enabling more robust and trustworthy routing decisions.
