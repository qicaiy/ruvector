//! # RuVector Math
//!
//! Advanced mathematics for next-generation vector search, featuring:
//!
//! - **Optimal Transport**: Wasserstein distances, Sinkhorn algorithm, Sliced Wasserstein
//! - **Information Geometry**: Fisher Information, Natural Gradient, K-FAC
//! - **Product Manifolds**: Mixed-curvature spaces (Euclidean × Hyperbolic × Spherical)
//! - **Spherical Geometry**: Geodesics on the n-sphere for cyclical patterns
//!
//! ## Design Principles
//!
//! 1. **Pure Rust**: No BLAS/LAPACK dependencies for full WASM compatibility
//! 2. **SIMD-Ready**: Hot paths optimized for auto-vectorization
//! 3. **Numerically Stable**: Log-domain arithmetic, clamping, and stable softmax
//! 4. **Modular**: Each component usable independently
//!
//! ## Quick Start
//!
//! ```rust
//! use ruvector_math::optimal_transport::{SlicedWasserstein, SinkhornSolver, OptimalTransport};
//! use ruvector_math::information_geometry::FisherInformation;
//! use ruvector_math::product_manifold::ProductManifold;
//!
//! // Sliced Wasserstein distance between point clouds
//! let sw = SlicedWasserstein::new(100).with_seed(42);
//! let points_a = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
//! let points_b = vec![vec![0.5, 0.5], vec![1.5, 0.5]];
//! let dist = sw.distance(&points_a, &points_b);
//! assert!(dist > 0.0);
//!
//! // Sinkhorn optimal transport
//! let solver = SinkhornSolver::new(0.1, 100);
//! let cost_matrix = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
//! let weights_a = vec![0.5, 0.5];
//! let weights_b = vec![0.5, 0.5];
//! let result = solver.solve(&cost_matrix, &weights_a, &weights_b).unwrap();
//! assert!(result.converged);
//!
//! // Product manifold operations (Euclidean only for simplicity)
//! let manifold = ProductManifold::new(2, 0, 0);
//! let point_a = vec![0.0, 0.0];
//! let point_b = vec![3.0, 4.0];
//! let dist = manifold.distance(&point_a, &point_b).unwrap();
//! assert!((dist - 5.0).abs() < 1e-10);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod error;
pub mod optimal_transport;
pub mod information_geometry;
pub mod spherical;
pub mod product_manifold;
pub mod utils;

// Re-exports for convenience
pub use error::{MathError, Result};
pub use optimal_transport::{
    SlicedWasserstein, SinkhornSolver, GromovWasserstein,
    TransportPlan, WassersteinConfig,
};
pub use information_geometry::{
    FisherInformation, NaturalGradient, KFACApproximation,
};
pub use spherical::{SphericalSpace, SphericalConfig};
pub use product_manifold::{ProductManifold, ProductManifoldConfig, CurvatureType};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::optimal_transport::*;
    pub use crate::information_geometry::*;
    pub use crate::spherical::*;
    pub use crate::product_manifold::*;
    pub use crate::error::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_version() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
    }
}
