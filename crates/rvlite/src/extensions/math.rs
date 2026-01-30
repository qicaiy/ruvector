//! Advanced math integration via ruvector-math-wasm
//!
//! Provides Optimal Transport (Wasserstein), Information Geometry,
//! Product Manifolds, and advanced distance metrics.

use wasm_bindgen::prelude::*;

/// Math Engine for advanced geometric computations
#[wasm_bindgen]
pub struct MathEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl MathEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        MathEngine { initialized: true }
    }

    /// Check if math module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported distance metrics
    pub fn supported_metrics(&self) -> JsValue {
        let metrics = vec![
            "wasserstein",
            "sinkhorn",
            "fisher-rao",
            "bures",
            "fubini-study",
            "mahalanobis",
            "bregman",
            "kl-divergence",
            "js-divergence",
            "hellinger",
        ];
        serde_wasm_bindgen::to_value(&metrics).unwrap_or(JsValue::NULL)
    }

    /// Get supported manifold types
    pub fn supported_manifolds(&self) -> JsValue {
        let manifolds = vec![
            "euclidean",
            "hyperbolic",
            "spherical",
            "product",
            "grassmann",
            "stiefel",
        ];
        serde_wasm_bindgen::to_value(&manifolds).unwrap_or(JsValue::NULL)
    }
}
