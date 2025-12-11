//! # TRM (Tiny Recursive Model) Module
//!
//! This module implements Samsung SAIL Montreal's TinyRecursiveModels approach
//! for parameter-efficient recursive reasoning.
//!
//! ## Overview
//!
//! TRM achieves strong reasoning performance with only 7M parameters by
//! iteratively refining answers through recursive latent updates.
//!
//! ## Architecture
//!
//! ```text
//! Question ──┬► Latent Update (n times) ──► Answer Refine ──┐
//!            └─────────────────────────────────────────────┘
//!                            (repeat K times)
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::trm::{TrmEngine, TrmEngineBuilder, TrmConfig};
//!
//! let mut engine = TrmEngineBuilder::new()
//!     .hidden_dim(256)
//!     .embedding_dim(256)
//!     .default_k(10)
//!     .build()
//!     .unwrap();
//!
//! let question = vec![0.5; 256];
//! let mut answer = vec![0.1; 256];
//!
//! let result = engine.reason(&question, &mut answer);
//!
//! println!("Confidence: {}", result.confidence);
//! println!("Iterations: {}", result.iterations_used);
//! ```
//!
//! ## Attribution
//!
//! Based on research from Samsung AI Lab Montreal.
//! Repository: <https://github.com/SamsungSAILMontreal/TinyRecursiveModels>

pub mod attention;
pub mod confidence;
pub mod config;
pub mod engine;
pub mod error;
pub mod mlp;
pub mod refiner;
pub mod sona_bridge;
pub mod types;

// Re-exports for convenient access
pub use attention::AttentionLatentUpdater;
pub use confidence::ConfidenceScorer;
pub use config::{TrmConfig, TrmConfigBuilder};
pub use engine::{TrmEngine, TrmEngineBuilder};
pub use error::TrmError;
pub use mlp::MlpLatentUpdater;
pub use refiner::AnswerRefiner;
pub use sona_bridge::{SonaBridge, SonaBridgeConfig, SonaBridgeState};
pub use types::{TrmIterationState, TrmResult, TrmRoutingDecision, TrmTrajectory, TrmInfo};

/// Trait for latent state updaters
///
/// Implementations update the latent state given question/answer context.
pub trait LatentUpdate: Send + Sync {
    /// Update the latent state given question, answer, and current latent
    ///
    /// # Arguments
    /// * `question_pooled` - Pooled question embedding
    /// * `answer_pooled` - Pooled answer embedding
    /// * `latent` - Mutable latent state to update in-place
    fn update(&self, question_pooled: &[f32], answer_pooled: &[f32], latent: &mut [f32]);

    /// Get the hidden dimension of the latent state
    fn hidden_dim(&self) -> usize;

    /// Reset any internal state/buffers
    fn reset(&mut self) {}
}

/// Trait for recursive reasoning engines
///
/// Implementations perform the full TRM reasoning loop with K iterations.
pub trait RecursiveReasoner: Send + Sync {
    /// Perform recursive reasoning with default K iterations
    ///
    /// # Arguments
    /// * `question` - Question embedding
    /// * `answer` - Answer embedding (modified in-place)
    ///
    /// # Returns
    /// TrmResult with refined answer and metadata
    fn reason(&mut self, question: &[f32], answer: &mut [f32]) -> TrmResult;

    /// Perform recursive reasoning with specified K iterations
    ///
    /// # Arguments
    /// * `question` - Question embedding
    /// * `answer` - Answer embedding (modified in-place)
    /// * `k` - Number of K iterations to perform
    fn reason_with_k(&mut self, question: &[f32], answer: &mut [f32], k: usize) -> TrmResult;

    /// Perform recursive reasoning with routing decision
    ///
    /// # Arguments
    /// * `question` - Question embedding
    /// * `answer` - Answer embedding (modified in-place)
    /// * `routing` - Routing decision from SONA
    fn reason_with_routing(
        &mut self,
        question: &[f32],
        answer: &mut [f32],
        routing: &TrmRoutingDecision,
    ) -> TrmResult;

    /// Reset internal state
    fn reset(&mut self);
}
