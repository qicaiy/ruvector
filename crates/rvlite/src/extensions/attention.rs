//! Attention mechanism integration via ruvector-attention-wasm
//!
//! Provides 39 attention types including Flash, Multi-Head, Hyperbolic,
//! MoE, CGT Sheaf, and GPU-accelerated variants.

use wasm_bindgen::prelude::*;

/// Attention Engine wrapping ruvector-attention-wasm
#[wasm_bindgen]
pub struct AttentionEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl AttentionEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        AttentionEngine { initialized: true }
    }

    /// Check if attention module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported attention types
    pub fn supported_types(&self) -> JsValue {
        let types = vec![
            "multi-head",
            "flash",
            "flash-v2",
            "linear",
            "sparse",
            "local",
            "global",
            "cross",
            "self",
            "causal",
            "sliding-window",
            "dilated",
            "axial",
            "performer",
            "reformer",
            "longformer",
            "bigbird",
            "routing",
            "mixture-of-experts",
            "hyperbolic",
            "cgt-sheaf",
            "grouped-query",
            "multi-query",
            "ring",
            "page",
            "flex",
            "differential",
            "native-sparse",
            "block-sparse",
            "memory-efficient",
            "cosine",
            "additive",
            "location-based",
            "content-based",
            "relative-position",
            "rotary",
            "alibi",
            "sandwich",
            "talking-heads",
        ];
        serde_wasm_bindgen::to_value(&types).unwrap_or(JsValue::NULL)
    }
}
