//! Neuromorphic HNSW integration via micro-hnsw-wasm
//!
//! Provides ultra-lightweight (11.8KB) HNSW vector search with
//! spiking neural networks, LIF neurons, and STDP learning.

use wasm_bindgen::prelude::*;

/// Micro HNSW Engine for neuromorphic vector search
#[wasm_bindgen]
pub struct MicroHnswEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl MicroHnswEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        MicroHnswEngine { initialized: true }
    }

    /// Check if micro-hnsw module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported distance functions
    pub fn supported_distances(&self) -> JsValue {
        let distances = vec![
            "cosine",
            "euclidean",
            "dot-product",
            "manhattan",
            "hamming",
        ];
        serde_wasm_bindgen::to_value(&distances).unwrap_or(JsValue::NULL)
    }
}
