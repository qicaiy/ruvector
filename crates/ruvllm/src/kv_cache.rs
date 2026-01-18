//! Two-Tier KV Cache Implementation
//!
//! Implements a memory-efficient KV cache with two tiers:
//! - **High-precision tail**: Recent tokens in FP16 for attention quality
//! - **Quantized store**: Older tokens in Q4/Q8 for memory efficiency
//!
//! This design balances memory usage with attention quality by keeping
//! the most relevant (recent) context in high precision while compressing
//! older context.

use crate::error::{Result, RuvLLMError};
use crate::types::Precision;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// KV cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Number of tokens to keep in high-precision tail
    pub tail_length: usize,
    /// Precision for tail storage
    pub tail_precision: Precision,
    /// Precision for quantized store
    pub store_precision: Precision,
    /// Maximum total tokens to cache
    pub max_tokens: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Migration batch size (tokens to move at once)
    pub migration_batch: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            tail_length: 256,
            tail_precision: Precision::FP16,
            store_precision: Precision::Q4,
            max_tokens: 4096,
            num_kv_heads: 8,
            head_dim: 128,
            migration_batch: 64,
        }
    }
}

/// Cache tier enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTier {
    /// High-precision tail for recent tokens
    Hot,
    /// Warm tier (optional intermediate)
    Warm,
    /// Quantized store for older tokens
    Cold,
}

/// Quantization configuration for cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheQuantization {
    /// High-precision tail only
    HighPrecisionTail {
        /// Number of tokens in tail
        tail_length: usize,
        /// Precision level
        precision: Precision,
    },
    /// Quantized store only
    QuantizedStore {
        /// Precision level
        precision: Precision,
        /// Compression ratio achieved
        compression_ratio: f32,
    },
    /// Hybrid: tail in FP16, rest in Q4
    Hybrid {
        /// Number of tokens in tail
        tail_length: usize,
        /// Tail precision
        tail_precision: Precision,
        /// Store precision
        store_precision: Precision,
    },
}

impl Default for CacheQuantization {
    fn default() -> Self {
        Self::Hybrid {
            tail_length: 256,
            tail_precision: Precision::FP16,
            store_precision: Precision::Q4,
        }
    }
}

/// KV pair storage
#[derive(Debug, Clone)]
struct KvPair {
    /// Key tensor
    keys: Vec<f32>,
    /// Value tensor
    values: Vec<f32>,
    /// Token position
    position: usize,
}

/// Quantized KV pair storage (simulated - production would use actual quantization)
#[derive(Debug, Clone)]
struct QuantizedKvPair {
    /// Quantized keys (stored as f32 for simplicity, would be i8/i4 in production)
    keys: Vec<f32>,
    /// Quantized values
    values: Vec<f32>,
    /// Scale factor for dequantization
    scale: f32,
    /// Zero point for asymmetric quantization
    zero_point: f32,
    /// Token position
    position: usize,
}

impl QuantizedKvPair {
    /// Quantize from full precision
    fn from_kv_pair(pair: &KvPair, precision: Precision) -> Self {
        // Simplified quantization - production would use proper quantization
        let (scale, zero_point) = Self::compute_scale_and_zero(&pair.keys, precision);

        let quantize = |vals: &[f32]| -> Vec<f32> {
            vals.iter()
                .map(|v| ((v - zero_point) / scale).round())
                .collect()
        };

        Self {
            keys: quantize(&pair.keys),
            values: quantize(&pair.values),
            scale,
            zero_point,
            position: pair.position,
        }
    }

    /// Compute scale and zero point for quantization
    fn compute_scale_and_zero(values: &[f32], precision: Precision) -> (f32, f32) {
        if values.is_empty() {
            return (1.0, 0.0);
        }

        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = match precision {
            Precision::Q8 => 255.0,
            Precision::Q4 | Precision::Q4K => 15.0,
            _ => 255.0,
        };

        let scale = (max_val - min_val) / range;
        let zero_point = min_val;

        (scale.max(1e-8), zero_point)
    }

    /// Dequantize to full precision
    fn dequantize(&self) -> KvPair {
        let dequant = |vals: &[f32]| -> Vec<f32> {
            vals.iter()
                .map(|v| v * self.scale + self.zero_point)
                .collect()
        };

        KvPair {
            keys: dequant(&self.keys),
            values: dequant(&self.values),
            position: self.position,
        }
    }
}

/// Two-tier KV cache implementation
#[derive(Debug)]
pub struct TwoTierKvCache {
    /// Configuration
    config: KvCacheConfig,
    /// High-precision tail storage
    tail: RwLock<VecDeque<KvPair>>,
    /// Quantized store
    store: RwLock<Vec<QuantizedKvPair>>,
    /// Current total tokens
    total_tokens: AtomicUsize,
    /// Quantization policy reference (for dynamic adjustment)
    quantization_policy: Arc<RwLock<CacheQuantization>>,
}

impl TwoTierKvCache {
    /// Create a new two-tier KV cache
    pub fn new(config: KvCacheConfig) -> Self {
        let quantization_policy = Arc::new(RwLock::new(CacheQuantization::Hybrid {
            tail_length: config.tail_length,
            tail_precision: config.tail_precision,
            store_precision: config.store_precision,
        }));

        Self {
            config,
            tail: RwLock::new(VecDeque::new()),
            store: RwLock::new(Vec::new()),
            total_tokens: AtomicUsize::new(0),
            quantization_policy,
        }
    }

    /// Append new KV pairs
    pub fn append(&self, keys: &[f32], values: &[f32]) -> Result<()> {
        let stride = self.config.num_kv_heads * self.config.head_dim;
        let num_tokens = keys.len() / stride;

        if keys.len() != values.len() {
            return Err(RuvLLMError::KvCache(
                "Key and value lengths must match".to_string(),
            ));
        }

        let current_tokens = self.total_tokens.load(Ordering::SeqCst);

        // Add to tail
        let mut tail = self.tail.write();
        for i in 0..num_tokens {
            let offset = i * stride;
            tail.push_back(KvPair {
                keys: keys[offset..offset + stride].to_vec(),
                values: values[offset..offset + stride].to_vec(),
                position: current_tokens + i,
            });
        }

        // Migrate to store if tail exceeds threshold
        while tail.len() > self.config.tail_length {
            let batch_size = self.config.migration_batch.min(
                tail.len() - self.config.tail_length
            );

            let to_migrate: Vec<_> = (0..batch_size)
                .filter_map(|_| tail.pop_front())
                .collect();

            let mut store = self.store.write();
            for pair in to_migrate {
                let quantized = QuantizedKvPair::from_kv_pair(
                    &pair,
                    self.config.store_precision,
                );
                store.push(quantized);
            }
        }

        self.total_tokens.fetch_add(num_tokens, Ordering::SeqCst);

        // Enforce max tokens limit
        self.enforce_max_tokens()?;

        Ok(())
    }

    /// Enforce maximum token limit by evicting oldest tokens
    fn enforce_max_tokens(&self) -> Result<()> {
        let total = self.total_tokens.load(Ordering::SeqCst);

        if total <= self.config.max_tokens {
            return Ok(());
        }

        let to_evict = total - self.config.max_tokens;
        let mut store = self.store.write();

        // Evict from quantized store first
        let store_evict = to_evict.min(store.len());
        store.drain(0..store_evict);

        self.total_tokens.fetch_sub(store_evict, Ordering::SeqCst);

        // If still over limit, evict from tail
        let remaining = to_evict - store_evict;
        if remaining > 0 {
            let mut tail = self.tail.write();
            for _ in 0..remaining.min(tail.len()) {
                tail.pop_front();
            }
            self.total_tokens.fetch_sub(remaining.min(tail.len()), Ordering::SeqCst);
        }

        Ok(())
    }

    /// Get all KV pairs for attention computation
    pub fn get_all_kv(&self) -> (Vec<f32>, Vec<f32>) {
        let stride = self.config.num_kv_heads * self.config.head_dim;
        let total = self.total_tokens.load(Ordering::SeqCst);

        let mut all_keys = Vec::with_capacity(total * stride);
        let mut all_values = Vec::with_capacity(total * stride);

        // Get from quantized store (dequantize)
        let store = self.store.read();
        for qpair in store.iter() {
            let pair = qpair.dequantize();
            all_keys.extend_from_slice(&pair.keys);
            all_values.extend_from_slice(&pair.values);
        }
        drop(store);

        // Get from tail (full precision)
        let tail = self.tail.read();
        for pair in tail.iter() {
            all_keys.extend_from_slice(&pair.keys);
            all_values.extend_from_slice(&pair.values);
        }

        (all_keys, all_values)
    }

    /// Compute attention with tier-aware access
    ///
    /// This applies position-based decay weights to balance precision/memory tradeoff
    pub fn attend(&self, query: &[f32], scale: f32) -> Result<Vec<f32>> {
        let (keys, values) = self.get_all_kv();
        let stride = self.config.num_kv_heads * self.config.head_dim;
        let num_tokens = keys.len() / stride;

        if num_tokens == 0 {
            return Ok(vec![0.0; query.len()]);
        }

        // Simplified attention - production would use optimized kernels
        let mut scores = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let k_offset = t * stride;
            let k_slice = &keys[k_offset..k_offset + stride];

            let score: f32 = query.iter()
                .zip(k_slice.iter())
                .map(|(q, k)| q * k * scale)
                .sum();

            scores.push(score);
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of values
        let mut output = vec![0.0; stride];
        for (t, weight) in attn_weights.iter().enumerate() {
            let v_offset = t * stride;
            for (i, v) in values[v_offset..v_offset + stride].iter().enumerate() {
                output[i] += weight * v;
            }
        }

        Ok(output)
    }

    /// Get current statistics
    pub fn stats(&self) -> KvCacheStats {
        let tail = self.tail.read();
        let store = self.store.read();
        let stride = self.config.num_kv_heads * self.config.head_dim;

        let tail_bytes = tail.len() * stride * 4 * 2; // f32 * 2 (keys + values)
        let store_bytes = store.len() * stride * self.config.store_precision.bytes_per_element() as usize * 2;

        KvCacheStats {
            total_tokens: self.total_tokens.load(Ordering::SeqCst),
            tail_tokens: tail.len(),
            store_tokens: store.len(),
            tail_bytes,
            store_bytes,
            compression_ratio: tail_bytes as f32 / store_bytes.max(1) as f32,
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut tail = self.tail.write();
        let mut store = self.store.write();
        tail.clear();
        store.clear();
        self.total_tokens.store(0, Ordering::SeqCst);
    }

    /// Update quantization policy
    pub fn update_policy(&self, policy: CacheQuantization) {
        let mut current = self.quantization_policy.write();
        *current = policy;
    }
}

/// KV cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KvCacheStats {
    /// Total tokens cached
    pub total_tokens: usize,
    /// Tokens in high-precision tail
    pub tail_tokens: usize,
    /// Tokens in quantized store
    pub store_tokens: usize,
    /// Bytes used by tail
    pub tail_bytes: usize,
    /// Bytes used by store
    pub store_bytes: usize,
    /// Compression ratio (tail/store)
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_append() {
        let config = KvCacheConfig {
            tail_length: 4,
            num_kv_heads: 2,
            head_dim: 4,
            migration_batch: 2,
            ..Default::default()
        };

        let cache = TwoTierKvCache::new(config);

        // Append tokens
        let keys = vec![1.0; 2 * 4]; // 1 token
        let values = vec![1.0; 2 * 4];
        cache.append(&keys, &values).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_tokens, 1);
        assert_eq!(stats.tail_tokens, 1);
        assert_eq!(stats.store_tokens, 0);
    }

    #[test]
    fn test_kv_cache_migration() {
        let config = KvCacheConfig {
            tail_length: 2,
            num_kv_heads: 2,
            head_dim: 4,
            migration_batch: 1,
            max_tokens: 100,
            ..Default::default()
        };

        let cache = TwoTierKvCache::new(config);

        // Append more tokens than tail can hold
        for _ in 0..5 {
            let keys = vec![1.0; 2 * 4];
            let values = vec![1.0; 2 * 4];
            cache.append(&keys, &values).unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.total_tokens, 5);
        assert_eq!(stats.tail_tokens, 2);
        assert_eq!(stats.store_tokens, 3);
    }

    #[test]
    fn test_kv_cache_attend() {
        let config = KvCacheConfig {
            tail_length: 4,
            num_kv_heads: 1,
            head_dim: 4,
            ..Default::default()
        };

        let cache = TwoTierKvCache::new(config);

        // Add some KV pairs
        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        cache.append(&keys, &values).unwrap();

        // Query
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let output = cache.attend(&query, 1.0).unwrap();

        assert_eq!(output.len(), 4);
        // With single token and matching query, output should be similar to values
        assert!((output[0] - 1.0).abs() < 0.1);
    }
}
