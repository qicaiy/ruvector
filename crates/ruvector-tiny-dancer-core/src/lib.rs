//! # Tiny Dancer: Production-Grade AI Agent Routing System
//!
//! High-performance neural routing system for optimizing LLM inference costs.
//!
//! This crate provides:
//! - FastGRNN model inference (sub-millisecond latency)
//! - Feature engineering for candidate scoring
//! - Model optimization (quantization, pruning)
//! - Model training with knowledge distillation
//! - Uncertainty quantification with conformal prediction
//! - Circuit breaker patterns for graceful degradation
//! - SQLite/AgentDB integration

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rustdoc::broken_intra_doc_links)]

pub mod circuit_breaker;
pub mod error;
pub mod feature_engineering;
pub mod metrics;
pub mod model;
pub mod optimization;
pub mod router;
pub mod storage;
pub mod tracing;
pub mod training;
pub mod types;

#[cfg(feature = "admin-api")]
pub mod api;
pub mod uncertainty;

// Re-exports for convenience
pub use error::{Result, TinyDancerError};
pub use metrics::MetricsCollector;
pub use model::FastGRNN;
pub use router::Router;
pub use tracing::{RoutingSpan, TraceContext, TracingConfig, TracingSystem};
pub use types::{Candidate, RouterConfig, RoutingDecision, RoutingRequest, RoutingResponse};

/// Version of the Tiny Dancer library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
