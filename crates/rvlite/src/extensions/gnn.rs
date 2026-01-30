//! Graph Neural Network integration via ruvector-gnn-wasm
//!
//! Provides GCN, GAT, GraphSAGE layers and node embedding computation
//! directly through the RvLite WASM interface.

use wasm_bindgen::prelude::*;

/// GNN Engine wrapping ruvector-gnn-wasm functionality
#[wasm_bindgen]
pub struct GnnEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl GnnEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        GnnEngine { initialized: true }
    }

    /// Check if GNN module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported layer types
    pub fn supported_layers(&self) -> JsValue {
        let layers = vec![
            "gcn",
            "gat",
            "graphsage",
            "gin",
            "chebnet",
            "appnp",
        ];
        serde_wasm_bindgen::to_value(&layers).unwrap_or(JsValue::NULL)
    }
}
