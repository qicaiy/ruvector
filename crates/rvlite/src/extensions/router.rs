//! Router integration via ruvector-router-wasm
//!
//! Provides intelligent request routing with embedding-based
//! classification and load balancing.

use wasm_bindgen::prelude::*;

/// Router Engine for intelligent request routing
#[wasm_bindgen]
pub struct RouterEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl RouterEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        RouterEngine { initialized: true }
    }

    /// Check if router module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported routing strategies
    pub fn supported_strategies(&self) -> JsValue {
        let strategies = vec![
            "embedding-similarity",
            "keyword-first",
            "hybrid",
            "round-robin",
            "least-connections",
            "weighted",
            "content-based",
        ];
        serde_wasm_bindgen::to_value(&strategies).unwrap_or(JsValue::NULL)
    }
}
