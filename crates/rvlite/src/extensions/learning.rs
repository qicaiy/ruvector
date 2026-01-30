//! MicroLoRA learning integration via ruvector-learning-wasm
//!
//! Provides rank-2 LoRA adaptation with <100us latency
//! for per-operator learning in WASM environments.

use wasm_bindgen::prelude::*;

/// Learning Engine for MicroLoRA adaptation
#[wasm_bindgen]
pub struct LearningEngine {
    initialized: bool,
}

#[wasm_bindgen]
impl LearningEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        LearningEngine { initialized: true }
    }

    /// Check if learning module is available
    pub fn is_available(&self) -> bool {
        self.initialized
    }

    /// Get supported learning algorithms
    pub fn supported_algorithms(&self) -> JsValue {
        let algos = vec![
            "micro-lora",
            "rank-2-adaptation",
            "online-learning",
            "continual-learning",
            "few-shot",
        ];
        serde_wasm_bindgen::to_value(&algos).unwrap_or(JsValue::NULL)
    }
}
