//! RLM Environment Abstraction
//!
//! This module provides environment traits and implementations for both
//! native and WASM execution contexts.
//!
//! ## Environment Trait
//!
//! The `RlmEnvironment` trait abstracts over platform-specific operations:
//! - Embedding generation
//! - Response generation
//! - Time measurement
//! - ID generation
//!
//! ## Implementations
//!
//! - `NativeEnvironment`: Full-featured environment for native targets
//! - `WasmEnvironment`: Lightweight environment for WASM targets

use super::controller::RlmConfig;
use super::memory::MemorySearchResult;
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Trait for RLM execution environments
///
/// Implementations provide platform-specific operations for embedding,
/// generation, timing, and ID generation.
pub trait RlmEnvironment: Send + Sync + 'static {
    /// Generate an embedding for the given text
    fn embed(text: &str) -> Result<Vec<f32>>;

    /// Generate a response given input and context
    fn generate(input: &str, context: &[MemorySearchResult], config: &RlmConfig) -> Result<String>;

    /// Get current timestamp for timing (platform-specific)
    #[cfg(not(target_arch = "wasm32"))]
    fn now() -> std::time::Instant {
        std::time::Instant::now()
    }

    /// Get current timestamp for timing (WASM version)
    #[cfg(target_arch = "wasm32")]
    fn now() -> f64;

    /// Calculate elapsed time in milliseconds (native version)
    #[cfg(not(target_arch = "wasm32"))]
    fn elapsed_ms(start: std::time::Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1000.0
    }

    /// Calculate elapsed time in milliseconds (WASM version)
    #[cfg(target_arch = "wasm32")]
    fn elapsed_ms(start: f64) -> f64;

    /// Generate a unique ID
    fn generate_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

/// Native environment for full-featured execution
///
/// This environment uses:
/// - Simple bag-of-words embedding (can be replaced with model-based)
/// - Template-based response generation
/// - std::time for timing
/// - UUID v4 for ID generation
pub struct NativeEnvironment;

impl RlmEnvironment for NativeEnvironment {
    fn embed(text: &str) -> Result<Vec<f32>> {
        // Simple bag-of-words embedding for demonstration
        // In production, this would use a proper embedding model
        let vocab_size = 384; // Match config embedding_dim
        let mut embedding = vec![0.0f32; vocab_size];

        // Hash words into embedding dimensions
        for word in text.split_whitespace() {
            let hash = simple_hash(word);
            let idx = (hash % vocab_size as u64) as usize;
            embedding[idx] += 1.0;
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    fn generate(
        input: &str,
        context: &[MemorySearchResult],
        _config: &RlmConfig,
    ) -> Result<String> {
        // Simple template-based generation for demonstration
        // In production, this would use an LLM backend

        let context_str = if context.is_empty() {
            String::new()
        } else {
            let excerpts: Vec<_> = context
                .iter()
                .take(3)
                .map(|r| format!("- {}", truncate(&r.entry.text, 100)))
                .collect();
            format!("\n\nRelevant context:\n{}", excerpts.join("\n"))
        };

        Ok(format!(
            "Based on your query: \"{}\"{}\n\nI don't have a full language model available in this environment. \
             Please configure a proper LLM backend for actual generation.",
            truncate(input, 50),
            context_str
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn now() -> std::time::Instant {
        std::time::Instant::now()
    }

    #[cfg(target_arch = "wasm32")]
    fn now() -> f64 {
        0.0 // Will be overridden by WasmEnvironment
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn elapsed_ms(start: std::time::Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1000.0
    }

    #[cfg(target_arch = "wasm32")]
    fn elapsed_ms(_start: f64) -> f64 {
        0.0 // Will be overridden by WasmEnvironment
    }

    fn generate_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

/// WASM environment for browser/Node.js execution
///
/// This environment is optimized for WASM:
/// - Uses smaller embedding dimension
/// - Single-threaded execution
/// - Performance.now() for timing
/// - Crypto.randomUUID() for ID generation
#[cfg(all(target_arch = "wasm32", feature = "rlm-wasm"))]
pub struct WasmEnvironment;

#[cfg(all(target_arch = "wasm32", feature = "rlm-wasm"))]
impl RlmEnvironment for WasmEnvironment {
    fn embed(text: &str) -> Result<Vec<f32>> {
        // Smaller embedding for WASM
        let vocab_size = 256; // Smaller for WASM efficiency
        let mut embedding = vec![0.0f32; vocab_size];

        // Hash words into embedding dimensions
        for word in text.split_whitespace() {
            let hash = simple_hash(word);
            let idx = (hash % vocab_size as u64) as usize;
            embedding[idx] += 1.0;
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    fn generate(
        input: &str,
        context: &[MemorySearchResult],
        _config: &RlmConfig,
    ) -> Result<String> {
        // Simple template-based generation for WASM demo
        let context_str = if context.is_empty() {
            String::new()
        } else {
            let excerpts: Vec<_> = context
                .iter()
                .take(2) // Fewer for WASM
                .map(|r| format!("- {}", truncate(&r.entry.text, 80)))
                .collect();
            format!("\n\nContext:\n{}", excerpts.join("\n"))
        };

        Ok(format!(
            "Query: \"{}\"{}\n\n[WASM environment - configure LLM backend for full generation]",
            truncate(input, 40),
            context_str
        ))
    }

    fn now() -> f64 {
        // Use web_sys Performance API if available
        #[cfg(feature = "rlm-wasm")]
        {
            use wasm_bindgen::JsCast;
            if let Some(window) = web_sys::window() {
                if let Some(perf) = window.performance() {
                    return perf.now();
                }
            }
        }
        0.0
    }

    fn elapsed_ms(start: f64) -> f64 {
        Self::now() - start
    }

    fn generate_id() -> String {
        // Use crypto.randomUUID if available, fallback to timestamp-based
        #[cfg(feature = "rlm-wasm")]
        {
            if let Some(crypto) = web_sys::window().and_then(|w| w.crypto().ok()) {
                if let Ok(uuid) = crypto.random_uuid() {
                    return uuid;
                }
            }
        }

        // Fallback: timestamp + random
        format!(
            "wasm-{:x}-{:x}",
            (Self::now() * 1000.0) as u64,
            rand::random::<u32>()
        )
    }
}

/// Simple hash function for embedding
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u64);
    }
    hash
}

/// Truncate string to max length with ellipsis
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Environment-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Environment type
    pub env_type: EnvironmentType,
    /// Enable debug logging
    pub debug: bool,
    /// Maximum memory usage in bytes (WASM)
    pub max_memory_bytes: Option<usize>,
}

/// Environment type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvironmentType {
    /// Native environment with full features
    Native,
    /// WASM environment with constraints
    Wasm,
    /// Node.js WASM environment
    NodeWasm,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            env_type: if cfg!(target_arch = "wasm32") {
                EnvironmentType::Wasm
            } else {
                EnvironmentType::Native
            },
            debug: false,
            max_memory_bytes: if cfg!(target_arch = "wasm32") {
                Some(64 * 1024 * 1024) // 64MB default for WASM
            } else {
                None
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_embed() {
        let embedding = NativeEnvironment::embed("hello world test").unwrap();
        assert_eq!(embedding.len(), 384);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_native_generate() {
        let config = RlmConfig::default();
        let result = NativeEnvironment::generate("test query", &[], &config).unwrap();
        assert!(result.contains("test query"));
    }

    #[test]
    fn test_simple_hash() {
        let h1 = simple_hash("hello");
        let h2 = simple_hash("world");
        let h3 = simple_hash("hello");

        assert_ne!(h1, h2);
        assert_eq!(h1, h3);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a longer string", 10), "this is...");
    }

    #[test]
    fn test_environment_config_default() {
        let config = EnvironmentConfig::default();
        assert_eq!(config.env_type, EnvironmentType::Native);
        assert!(!config.debug);
    }
}
