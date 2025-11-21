//! Main routing engine combining all components

use crate::circuit_breaker::{CircuitBreaker, CircuitState};
use crate::error::{Result, TinyDancerError};
use crate::feature_engineering::FeatureEngineer;
use crate::metrics::MetricsCollector;
use crate::model::FastGRNN;
use crate::types::{RouterConfig, RoutingDecision, RoutingRequest, RoutingResponse};
use crate::uncertainty::UncertaintyEstimator;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};

/// Main router for AI agent routing
pub struct Router {
    config: RouterConfig,
    model: Arc<RwLock<FastGRNN>>,
    feature_engineer: FeatureEngineer,
    uncertainty_estimator: UncertaintyEstimator,
    circuit_breaker: Option<CircuitBreaker>,
    metrics: MetricsCollector,
}

impl Router {
    /// Create a new router with the given configuration
    pub fn new(config: RouterConfig) -> Result<Self> {
        info!("Initializing Tiny Dancer router");

        // Load or create model
        let model = if std::path::Path::new(&config.model_path).exists() {
            debug!("Loading model from: {}", config.model_path);
            FastGRNN::load(&config.model_path)?
        } else {
            warn!("Model path not found, creating new model: {}", config.model_path);
            FastGRNN::new(Default::default())?
        };

        let circuit_breaker = if config.enable_circuit_breaker {
            info!(
                "Circuit breaker enabled with threshold: {}",
                config.circuit_breaker_threshold
            );
            Some(CircuitBreaker::new(config.circuit_breaker_threshold))
        } else {
            None
        };

        Ok(Self {
            config,
            model: Arc::new(RwLock::new(model)),
            feature_engineer: FeatureEngineer::new(),
            uncertainty_estimator: UncertaintyEstimator::new(),
            circuit_breaker,
            metrics: MetricsCollector::new(),
        })
    }

    /// Create a router with default configuration
    pub fn default() -> Result<Self> {
        Self::new(RouterConfig::default())
    }

    /// Route a request through the system
    #[tracing::instrument(skip(self, request), fields(candidate_count = request.candidates.len()))]
    pub fn route(&self, request: RoutingRequest) -> Result<RoutingResponse> {
        let start = Instant::now();
        let candidate_count = request.candidates.len();

        info!(
            candidate_count = candidate_count,
            "Processing routing request"
        );

        // Check circuit breaker
        if let Some(ref cb) = self.circuit_breaker {
            let _span = tracing::debug_span!("circuit_breaker_check").entered();

            if !cb.is_closed() {
                let state = cb.state();
                error!("Circuit breaker is open, rejecting request");

                // Update metrics
                self.metrics.record_routing_failure("circuit_breaker_open");
                self.metrics.set_circuit_breaker_state(match state {
                    CircuitState::Closed => 0.0,
                    CircuitState::HalfOpen => 1.0,
                    CircuitState::Open => 2.0,
                });

                return Err(TinyDancerError::CircuitBreakerError(
                    "Circuit breaker is open".to_string(),
                ));
            }

            // Update circuit breaker state metric
            self.metrics.set_circuit_breaker_state(match cb.state() {
                CircuitState::Closed => 0.0,
                CircuitState::HalfOpen => 1.0,
                CircuitState::Open => 2.0,
            });
        }

        // Feature engineering
        let feature_start = Instant::now();
        let feature_vectors = {
            let _span = tracing::debug_span!(
                "feature_engineering",
                batch_size = candidate_count
            )
            .entered();

            debug!(batch_size = candidate_count, "Extracting features");

            self.feature_engineer.extract_batch_features(
                &request.query_embedding,
                &request.candidates,
                request.metadata.as_ref(),
            )?
        };
        let feature_time_us = feature_start.elapsed().as_micros() as u64;

        // Record feature engineering metrics
        self.metrics.record_feature_engineering_duration(
            candidate_count,
            feature_time_us as f64 / 1_000_000.0,
        );

        // Model inference
        let model = self.model.read();
        let mut decisions = Vec::new();
        let mut lightweight_count = 0;
        let mut powerful_count = 0;

        for (candidate, features) in request.candidates.iter().zip(feature_vectors.iter()) {
            let inference_start = Instant::now();
            let _span = tracing::debug_span!(
                "model_inference",
                candidate_id = %candidate.id
            )
            .entered();

            match model.forward(&features.features, None) {
                Ok(score) => {
                    let inference_duration = inference_start.elapsed().as_secs_f64();

                    debug!(
                        candidate_id = %candidate.id,
                        confidence = score,
                        "Model inference completed"
                    );

                    // Estimate uncertainty
                    let uncertainty = {
                        let _span = tracing::debug_span!(
                            "uncertainty_estimation",
                            candidate_id = %candidate.id
                        )
                        .entered();

                        self.uncertainty_estimator.estimate(&features.features, score)
                    };

                    // Determine routing decision
                    let use_lightweight = score >= self.config.confidence_threshold
                        && uncertainty <= self.config.max_uncertainty;

                    if use_lightweight {
                        lightweight_count += 1;
                    } else {
                        powerful_count += 1;
                    }

                    debug!(
                        candidate_id = %candidate.id,
                        confidence = score,
                        uncertainty = uncertainty,
                        use_lightweight = use_lightweight,
                        "Routing decision made"
                    );

                    // Record metrics
                    self.metrics.record_model_inference_duration("fastgrnn", inference_duration);
                    self.metrics.record_routing_decision(use_lightweight);
                    self.metrics.record_confidence_score(use_lightweight, score);
                    self.metrics
                        .record_uncertainty_estimate(use_lightweight, uncertainty);

                    decisions.push(RoutingDecision {
                        candidate_id: candidate.id.clone(),
                        confidence: score,
                        use_lightweight,
                        uncertainty,
                    });

                    // Record success with circuit breaker
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_success();
                    }
                }
                Err(e) => {
                    error!(
                        candidate_id = %candidate.id,
                        error = %e,
                        "Model inference failed"
                    );

                    // Record failure with circuit breaker
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_failure();
                    }

                    self.metrics.record_routing_failure("inference_error");
                    return Err(e);
                }
            }
        }

        // Sort by confidence (descending)
        decisions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let inference_time_us = start.elapsed().as_micros() as u64;

        // Record overall metrics
        self.metrics.record_routing_success();
        self.metrics.record_candidates_processed(candidate_count);
        self.metrics
            .record_routing_latency("total", inference_time_us as f64 / 1_000_000.0);

        info!(
            inference_time_us = inference_time_us,
            feature_time_us = feature_time_us,
            lightweight_routes = lightweight_count,
            powerful_routes = powerful_count,
            "Routing request completed successfully"
        );

        Ok(RoutingResponse {
            decisions,
            inference_time_us,
            candidates_processed: request.candidates.len(),
            feature_time_us,
        })
    }

    /// Reload the model from disk
    pub fn reload_model(&self) -> Result<()> {
        let new_model = FastGRNN::load(&self.config.model_path)?;
        let mut model = self.model.write();
        *model = new_model;
        Ok(())
    }

    /// Get router configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get circuit breaker status
    pub fn circuit_breaker_status(&self) -> Option<bool> {
        self.circuit_breaker.as_ref().map(|cb| cb.is_closed())
    }

    /// Get metrics collector
    pub fn metrics(&self) -> &MetricsCollector {
        &self.metrics
    }

    /// Export current metrics in Prometheus format
    pub fn export_metrics(&self) -> Result<String> {
        self.metrics
            .export_metrics()
            .map_err(|e| TinyDancerError::Unknown(format!("Failed to export metrics: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Candidate;
    use chrono::Utc;
    use std::collections::HashMap;

    #[test]
    fn test_router_creation() {
        let router = Router::default().unwrap();
        assert!(router.circuit_breaker_status().is_some());
    }

    #[test]
    fn test_routing() {
        let router = Router::default().unwrap();

        // The default FastGRNN model expects input dimension to match feature count (5)
        // Features: semantic_similarity, recency, frequency, success_rate, metadata_overlap
        let candidates = vec![
            Candidate {
                id: "1".to_string(),
                embedding: vec![0.5; 384], // Embeddings can be any size
                metadata: HashMap::new(),
                created_at: Utc::now().timestamp(),
                access_count: 10,
                success_rate: 0.95,
            },
            Candidate {
                id: "2".to_string(),
                embedding: vec![0.3; 384],
                metadata: HashMap::new(),
                created_at: Utc::now().timestamp(),
                access_count: 5,
                success_rate: 0.85,
            },
        ];

        let request = RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates,
            metadata: None,
        };

        let response = router.route(request).unwrap();
        assert_eq!(response.decisions.len(), 2);
        assert!(response.inference_time_us > 0);
    }
}
