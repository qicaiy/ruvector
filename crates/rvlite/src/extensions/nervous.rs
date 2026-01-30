//! Bio-inspired neural integration via ruvector-nervous-system-wasm
//!
//! Provides spiking neural networks (SNN), LIF neurons, STDP learning,
//! winner-take-all, and dendritic computation for neuromorphic AI.

use wasm_bindgen::prelude::*;

/// Nervous System Engine for bio-inspired neural computation
#[wasm_bindgen]
pub struct NervousSystemEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl NervousSystemEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        NervousSystemEngine { initialized: true }
    }

    /// Check if nervous system module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported neuron models
    pub fn supported_models(&self) -> JsValue {
        let models = vec![
            "lif",
            "izhikevich",
            "hodgkin-huxley",
            "adaptive-exponential",
            "stochastic-lif",
        ];
        serde_wasm_bindgen::to_value(&models).unwrap_or(JsValue::NULL)
    }

    /// Get supported learning rules
    pub fn supported_learning_rules(&self) -> JsValue {
        let rules = vec![
            "stdp",
            "btsp",
            "bcm",
            "oja",
            "hebb",
        ];
        serde_wasm_bindgen::to_value(&rules).unwrap_or(JsValue::NULL)
    }
}
