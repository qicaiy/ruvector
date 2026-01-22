//! WASM bindings for RLM
//!
//! This module provides JavaScript/TypeScript bindings for the RLM controller
//! using wasm-bindgen.
//!
//! ## Usage from JavaScript
//!
//! ```javascript
//! import { WasmRlmController } from 'ruvllm-wasm';
//!
//! // Create controller with default config
//! const controller = new WasmRlmController();
//!
//! // Or with custom config
//! const controller = WasmRlmController.withConfig({
//!     embedding_dim: 256,
//!     max_entries: 1000,
//! });
//!
//! // Query
//! const result = await controller.query("What is Rust?");
//! console.log(result.text, result.confidence);
//!
//! // Add memory
//! const id = await controller.addMemory("Rust is a systems language", {
//!     source: "docs",
//!     tags: ["programming", "rust"]
//! });
//!
//! // Search memory
//! const results = await controller.searchMemory("programming", 5);
//! ```

#![cfg(all(target_arch = "wasm32", feature = "rlm-wasm"))]

use js_sys::{Array, Object, Promise, Reflect};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use super::controller::{
    MemoryEntry, MemoryMetadata, QueryResult, RlmConfig, RlmController, SourceAttribution,
    TokenUsage,
};
use super::environment::WasmEnvironment;
use super::memory::{MemoryConfig, MemorySearchResult};
use crate::error::Result;

/// WASM-compatible RLM Controller
///
/// This struct wraps the generic `RlmController` with the `WasmEnvironment`
/// and provides JavaScript-friendly bindings.
#[wasm_bindgen]
pub struct WasmRlmController {
    inner: Arc<RwLock<RlmController<WasmEnvironment>>>,
}

/// JavaScript configuration object for WasmRlmController
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsRlmConfig {
    /// Embedding dimension (default: 256)
    pub embedding_dim: Option<usize>,
    /// Maximum sequence length (default: 1024)
    pub max_seq_len: Option<usize>,
    /// Maximum memory entries (default: 1000)
    pub max_entries: Option<usize>,
    /// Enable response caching (default: true)
    pub enable_cache: Option<bool>,
    /// Cache TTL in seconds (default: 1800)
    pub cache_ttl_secs: Option<u64>,
    /// Temperature for generation (default: 0.7)
    pub temperature: Option<f32>,
    /// Top-p sampling (default: 0.9)
    pub top_p: Option<f32>,
    /// Maximum tokens to generate (default: 256)
    pub max_tokens: Option<usize>,
}

impl Default for JsRlmConfig {
    fn default() -> Self {
        Self {
            embedding_dim: None,
            max_seq_len: None,
            max_entries: None,
            enable_cache: None,
            cache_ttl_secs: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
        }
    }
}

impl From<JsRlmConfig> for RlmConfig {
    fn from(js: JsRlmConfig) -> Self {
        let base = RlmConfig::for_wasm();
        Self {
            embedding_dim: js.embedding_dim.unwrap_or(base.embedding_dim),
            max_seq_len: js.max_seq_len.unwrap_or(base.max_seq_len),
            memory_config: MemoryConfig {
                embedding_dim: js.embedding_dim.unwrap_or(base.memory_config.embedding_dim),
                max_entries: js.max_entries.unwrap_or(base.memory_config.max_entries),
                ..base.memory_config
            },
            model_id: base.model_id,
            enable_cache: js.enable_cache.unwrap_or(base.enable_cache),
            cache_ttl_secs: js.cache_ttl_secs.unwrap_or(base.cache_ttl_secs),
            max_concurrent_ops: base.max_concurrent_ops,
            temperature: js.temperature.unwrap_or(base.temperature),
            top_p: js.top_p.unwrap_or(base.top_p),
            max_tokens: js.max_tokens.unwrap_or(base.max_tokens),
        }
    }
}

/// JavaScript-compatible query result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsQueryResult {
    pub id: String,
    pub text: String,
    pub confidence: f32,
    pub tokens_generated: usize,
    pub latency_ms: f64,
    pub sources: Vec<JsSourceAttribution>,
    pub usage: JsTokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsSourceAttribution {
    pub memory_id: String,
    pub relevance: f32,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsTokenUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

impl From<QueryResult> for JsQueryResult {
    fn from(r: QueryResult) -> Self {
        Self {
            id: r.id,
            text: r.text,
            confidence: r.confidence,
            tokens_generated: r.tokens_generated,
            latency_ms: r.latency_ms,
            sources: r
                .sources
                .into_iter()
                .map(|s| JsSourceAttribution {
                    memory_id: s.memory_id,
                    relevance: s.relevance,
                    excerpt: s.excerpt,
                })
                .collect(),
            usage: JsTokenUsage {
                input_tokens: r.usage.input_tokens,
                output_tokens: r.usage.output_tokens,
                total_tokens: r.usage.total_tokens,
            },
        }
    }
}

/// JavaScript-compatible memory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsMemoryMetadata {
    pub source: Option<String>,
    pub category: Option<String>,
    pub tags: Option<Vec<String>>,
}

impl Default for JsMemoryMetadata {
    fn default() -> Self {
        Self {
            source: None,
            category: None,
            tags: None,
        }
    }
}

impl From<JsMemoryMetadata> for MemoryMetadata {
    fn from(js: JsMemoryMetadata) -> Self {
        Self {
            source: js.source,
            category: js.category,
            tags: js.tags.unwrap_or_default(),
            ttl_secs: None,
            extra: Default::default(),
        }
    }
}

/// JavaScript-compatible search result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsSearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: JsMemoryMetadata,
}

impl From<MemorySearchResult> for JsSearchResult {
    fn from(r: MemorySearchResult) -> Self {
        Self {
            id: r.id,
            text: r.entry.text,
            score: r.score,
            metadata: JsMemoryMetadata {
                source: r.entry.metadata.source,
                category: r.entry.metadata.category,
                tags: Some(r.entry.metadata.tags),
            },
        }
    }
}

/// JavaScript-compatible stats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsStats {
    pub total_queries: u64,
    pub total_memories: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_tokens: u64,
    pub avg_latency_ms: f64,
}

#[wasm_bindgen]
impl WasmRlmController {
    /// Create a new WasmRlmController with default WASM-optimized configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmRlmController, JsValue> {
        console_error_panic_hook::set_once();

        let config = RlmConfig::for_wasm();
        let controller = RlmController::<WasmEnvironment>::new(config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(controller)),
        })
    }

    /// Create a new WasmRlmController with custom configuration
    #[wasm_bindgen(js_name = "withConfig")]
    pub fn with_config(config: JsValue) -> Result<WasmRlmController, JsValue> {
        console_error_panic_hook::set_once();

        let js_config: JsRlmConfig = serde_wasm_bindgen::from_value(config).unwrap_or_default();

        let config: RlmConfig = js_config.into();
        let controller = RlmController::<WasmEnvironment>::new(config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(controller)),
        })
    }

    /// Query the RLM and get a response
    ///
    /// Returns a Promise that resolves to a QueryResult object
    #[wasm_bindgen]
    pub fn query(&self, input: &str) -> Result<JsValue, JsValue> {
        let controller = self.inner.read();
        let result = controller
            .query(input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let js_result: JsQueryResult = result.into();
        serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Add text to memory with optional metadata
    ///
    /// Returns the ID of the created memory entry
    #[wasm_bindgen(js_name = "addMemory")]
    pub fn add_memory(&self, text: &str, metadata: JsValue) -> Result<String, JsValue> {
        let js_metadata: JsMemoryMetadata = if metadata.is_undefined() || metadata.is_null() {
            JsMemoryMetadata::default()
        } else {
            serde_wasm_bindgen::from_value(metadata).unwrap_or_default()
        };

        let metadata: MemoryMetadata = js_metadata.into();
        let controller = self.inner.read();

        controller
            .add_memory(text, metadata)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Search memory for similar entries
    ///
    /// Returns an array of search results
    #[wasm_bindgen(js_name = "searchMemory")]
    pub fn search_memory(&self, query: &str, top_k: usize) -> Result<JsValue, JsValue> {
        let controller = self.inner.read();
        let results = controller
            .search_memory(query, top_k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let js_results: Vec<JsSearchResult> = results.into_iter().map(|r| r.into()).collect();

        serde_wasm_bindgen::to_value(&js_results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get a memory entry by ID
    #[wasm_bindgen(js_name = "getMemory")]
    pub fn get_memory(&self, id: &str) -> Result<JsValue, JsValue> {
        let controller = self.inner.read();
        let entry = controller
            .get_memory(id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        match entry {
            Some(e) => {
                let js_entry = JsSearchResult {
                    id: e.id,
                    text: e.text,
                    score: 1.0,
                    metadata: JsMemoryMetadata {
                        source: e.metadata.source,
                        category: e.metadata.category,
                        tags: Some(e.metadata.tags),
                    },
                };
                serde_wasm_bindgen::to_value(&js_entry)
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Delete a memory entry by ID
    #[wasm_bindgen(js_name = "deleteMemory")]
    pub fn delete_memory(&self, id: &str) -> Result<bool, JsValue> {
        let controller = self.inner.read();
        controller
            .delete_memory(id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// List memory entries with pagination
    #[wasm_bindgen(js_name = "listMemories")]
    pub fn list_memories(&self, limit: usize, offset: usize) -> Result<JsValue, JsValue> {
        let controller = self.inner.read();
        let entries = controller
            .list_memories(limit, offset)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let js_entries: Vec<JsSearchResult> = entries
            .into_iter()
            .map(|e| JsSearchResult {
                id: e.id,
                text: e.text,
                score: 1.0,
                metadata: JsMemoryMetadata {
                    source: e.metadata.source,
                    category: e.metadata.category,
                    tags: Some(e.metadata.tags),
                },
            })
            .collect();

        serde_wasm_bindgen::to_value(&js_entries).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clear the response cache
    #[wasm_bindgen(js_name = "clearCache")]
    pub fn clear_cache(&self) {
        self.inner.read().clear_cache();
    }

    /// Get current statistics
    #[wasm_bindgen(js_name = "getStats")]
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        let controller = self.inner.read();
        let stats = controller.stats();
        let snapshot = stats.snapshot();

        let js_stats = JsStats {
            total_queries: snapshot.total_queries,
            total_memories: snapshot.total_memories,
            cache_hits: snapshot.cache_hits,
            cache_misses: snapshot.cache_misses,
            total_tokens: snapshot.total_tokens,
            avg_latency_ms: snapshot.avg_latency_us as f64 / 1000.0,
        };

        serde_wasm_bindgen::to_value(&js_stats).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the configuration
    #[wasm_bindgen(js_name = "getConfig")]
    pub fn get_config(&self) -> Result<JsValue, JsValue> {
        let controller = self.inner.read();
        let config = controller.config();

        let js_config = JsRlmConfig {
            embedding_dim: Some(config.embedding_dim),
            max_seq_len: Some(config.max_seq_len),
            max_entries: Some(config.memory_config.max_entries),
            enable_cache: Some(config.enable_cache),
            cache_ttl_secs: Some(config.cache_ttl_secs),
            temperature: Some(config.temperature),
            top_p: Some(config.top_p),
            max_tokens: Some(config.max_tokens),
        };

        serde_wasm_bindgen::to_value(&js_config).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Initialize the WASM module
///
/// This should be called once when the module is loaded
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();

    // Log initialization
    web_sys::console::log_1(&JsValue::from_str("RuvLLM WASM module initialized"));
}

/// Console error panic hook for better error messages in WASM
mod console_error_panic_hook {
    use std::sync::Once;

    static SET_HOOK: Once = Once::new();

    pub fn set_once() {
        SET_HOOK.call_once(|| {
            #[cfg(feature = "rlm-wasm")]
            std::panic::set_hook(Box::new(|panic_info| {
                let msg = panic_info.to_string();
                web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&msg));
            }));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: WASM tests need to be run with wasm-pack test
    // These are compile-time checks

    #[test]
    fn test_js_config_default() {
        let config = JsRlmConfig::default();
        assert!(config.embedding_dim.is_none());
    }

    #[test]
    fn test_js_config_to_rlm_config() {
        let js_config = JsRlmConfig {
            embedding_dim: Some(128),
            max_entries: Some(500),
            ..Default::default()
        };

        let config: RlmConfig = js_config.into();
        assert_eq!(config.embedding_dim, 128);
        assert_eq!(config.memory_config.max_entries, 500);
    }
}
