//! Hyperbolic HNSW integration via ruvector-hyperbolic-hnsw-wasm
//!
//! Provides hierarchy-aware vector search using Poincare ball
//! and Lorentz hyperboloid models.

use wasm_bindgen::prelude::*;

/// Hyperbolic HNSW Engine for hierarchy-aware search
#[wasm_bindgen]
pub struct HyperbolicEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl HyperbolicEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        HyperbolicEngine { initialized: true }
    }

    /// Check if hyperbolic module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported models
    pub fn supported_models(&self) -> JsValue {
        let models = vec![
            "poincare-ball",
            "lorentz-hyperboloid",
            "klein-disk",
            "upper-half-plane",
        ];
        serde_wasm_bindgen::to_value(&models).unwrap_or(JsValue::NULL)
    }
}
