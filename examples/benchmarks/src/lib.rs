//! RuVector Benchmarks Library
//!
//! Comprehensive benchmarking suite for:
//! - Temporal reasoning (TimePuzzles-style constraint inference)
//! - Vector index operations (IVF, coherence-gated search)
//! - Swarm controller regret tracking
//!
//! Based on research from:
//! - TimePuzzles benchmark (arXiv:2601.07148)
//! - Sublinear regret in multi-agent control
//! - Tool-augmented iterative temporal reasoning

pub mod temporal;
pub mod vector_index;
pub mod swarm_regret;
pub mod logging;
pub mod timepuzzles;

pub use temporal::*;
pub use vector_index::*;
pub use swarm_regret::*;
pub use logging::*;
pub use timepuzzles::*;
