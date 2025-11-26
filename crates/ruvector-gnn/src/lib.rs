//! # RuVector GNN
//!
//! Graph Neural Network capabilities for RuVector, providing tensor operations,
//! GNN layers, compression, and differentiable search.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod error;
pub mod layer;
pub mod tensor;
pub mod compress;
pub mod search;
pub mod training;
pub mod query;

#[cfg(all(not(target_arch = "wasm32"), feature = "mmap"))]
pub mod mmap;

// Re-export commonly used types
pub use error::{GnnError, Result};
pub use layer::RuvectorLayer;
pub use compress::{CompressedTensor, CompressionLevel, TensorCompress};
pub use search::{cosine_similarity, differentiable_search, hierarchical_forward};
pub use training::{TrainConfig, OnlineConfig, info_nce_loss, local_contrastive_loss, sgd_step};
pub use query::{QueryMode, RuvectorQuery, QueryResult, SubGraph};

#[cfg(all(not(target_arch = "wasm32"), feature = "mmap"))]
pub use mmap::{AtomicBitmap, MmapManager, MmapGradientAccumulator};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Basic smoke test to ensure the crate compiles
        assert!(true);
    }
}
