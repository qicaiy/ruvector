# RuvLLM v2.0 - High-Performance LLM Inference for Rust

RuvLLM is a production-ready Rust LLM inference engine optimized for Apple Silicon (M1-M4), featuring real-time fine-tuning, NEON SIMD acceleration, Apple Neural Engine integration, and the SONA self-optimizing neural architecture.

## What's New in v2.0

### Major Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **RLM (Recursive Language Model)** | Recursive query decomposition for complex reasoning | Break down complex questions, parallel sub-query processing |
| **RuvLTRA-Medium 3B** | Purpose-built 3B model for Claude Flow | 42 layers, 256K context, speculative decode |
| **HuggingFace Hub** | Full Hub integration (download/upload) | Easy model sharing & distribution |
| **Task-Specific LoRA** | 5 pre-trained adapters for agent types | Optimized for coder/researcher/security/architect/reviewer |
| **Adapter Merging** | TIES, DARE, SLERP, Task Arithmetic | Combine adapters for multi-task models |
| **Hot-Swap Adapters** | Zero-downtime adapter switching | Runtime task specialization |
| **WASM Support** | WebAssembly target for browser-based inference | Run LLMs in the browser |
| **HNSW Routing** | 150x faster semantic pattern matching | <25us pattern retrieval |

### Performance Optimizations (NEW)

The v2.0 release includes significant performance improvements across all hot paths:

| Optimization | Description | Benefit |
|--------------|-------------|---------|
| **HNSW Index** | O(log n) approximate nearest neighbor search | 10x faster at 10k entries vs linear scan |
| **O(1) LRU Cache** | Using `lru` crate for cache operations | 23.5ns cache lookup (vs 500ns+ HashMap) |
| **Zero-Copy Types** | `Arc<str>`, `Arc<[f32]>` for shared data | 100-1000x improvement in cache hit paths |
| **Batch SIMD** | AVX2/NEON vectorized batch operations | 4x throughput for similarity search |
| **Memory Pools** | Pre-allocated vector/string pools | 50% fewer allocations in hot paths |

### Benchmark Results

Measured on Apple M4 Pro with 384-dimensional embeddings:

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Query decomposition | 340 ns | Pattern-based keyword extraction |
| Cache lookup | 23.5 ns | O(1) LRU with FNV-1a hashing |
| Memory search (10k entries) | ~0.4 ms | With HNSW index (vs 4ms linear) |
| Embeddings (384d) | 293 ns | SIMD-accelerated dot product |
| Batch cosine (4x384d) | ~1.1 us | AVX2/NEON batch processing |
| Pool acquire/release | <100 ns | Zero-allocation in steady state |

### New Optimization Modules

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `rlm/pool.rs` | Memory pools for allocation reuse | `VectorPool`, `StringPool`, `PooledVec` |
| `rlm/shared_types.rs` | Zero-copy shared types | `SharedText`, `SharedEmbedding`, `SharedQueryResult` |
| `rlm/simd_ops.rs` | SIMD-accelerated vector operations | `batch_cosine_similarity_4`, `batch_dot_products` |
| `rlm/cache.rs` | O(1) LRU memoization cache | `MemoizationCache`, `CacheEntry` |

### RLM (Recursive Language Model) Architecture

RLM provides a sophisticated recursive reasoning pipeline:

```text
+------------------+
|  RlmController   |  <-- Main entry point
+--------+---------+
         |
         v
+--------+---------+
| QueryDecomposer  |  <-- Breaks complex queries
+--------+---------+
         |
   +-----+-----+
   |           |
+--v--+     +--v--+
|Sub  |     |Sub  |  <-- Parallel sub-query processing
|Query|     |Query|
+--+--+     +--+--+
   |           |
   +-----+-----+
         |
         v
+--------+---------+
|AnswerSynthesizer |  <-- Combines sub-answers
+--------+---------+
         |
         v
+--------+---------+
|   RlmMemory      |  <-- HNSW-indexed retrieval
+-----------------+
```

### RLM Quick Start

```rust
use ruvllm::rlm::{RlmController, RlmConfig, NativeEnvironment};

// Create controller with default config
let config = RlmConfig::default();
let controller = RlmController::<NativeEnvironment>::new(config)?;

// Query the model with recursive decomposition
let response = controller.query("What is recursive language modeling and how does it improve reasoning?")?;
println!("Answer: {}", response.text);

// Add to memory for future retrieval
controller.add_memory("RLM uses recursive decomposition.", Default::default())?;

// Search memory semantically
let results = controller.search_memory("recursive", 5)?;
```

### RLM Configuration

```rust
use ruvllm::rlm::{RecursiveConfig, RecursiveConfigBuilder, AggregationStrategy};

let config = RecursiveConfigBuilder::new()
    .max_depth(5)                              // Maximum recursion depth
    .token_budget(16000)                       // Total token budget
    .enable_cache(true)                        // Enable memoization
    .aggregation(AggregationStrategy::WeightedMerge)
    .parallel_subqueries(true)                 // Process sub-queries in parallel
    .build()?;
```

### Previous Features (v1.x-2.x)

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Apple Neural Engine** | Core ML backend with ANE routing | 38 TOPS, 3-4x power efficiency |
| **Hybrid GPU+ANE Pipeline** | Intelligent operation routing | Best of both accelerators |
| **Multi-threaded GEMM** | Rayon parallelization | 4-12x speedup on M4 Pro |
| **Flash Attention 2** | Auto block sizing, online softmax | O(N) memory, +10% throughput |
| **Quantized Inference** | INT8/INT4/Q4_K/Q8_K kernels | 4-8x memory reduction |
| **Metal GPU Shaders** | simdgroup_matrix operations | 3x speedup on Apple Silicon |
| **GGUF Support** | Memory-mapped model loading | Fast loading, reduced RAM |
| **Continuous Batching** | Dynamic batch scheduling | 2-3x throughput improvement |
| **Speculative Decoding** | Draft model acceleration | 2-3x faster generation |
| **Gemma-2 & Phi-3** | New model architectures | Extended model support |

## Features

### Multiple Backends
- **Candle Backend**: HuggingFace's Candle framework with Metal/CUDA GPU acceleration
- **Core ML Backend**: Apple Neural Engine for maximum efficiency on Apple Silicon
- **Hybrid Pipeline**: Automatic routing between GPU and ANE based on operation type
- **RuvLTRA Backend**: Custom backend optimized for Claude Flow integration

### Optimized Kernels
- **NEON SIMD**: ARM64-optimized kernels with 4x loop unrolling and FMA instructions
- **Flash Attention 2**: Memory-efficient attention with O(N) complexity and online softmax
- **Paged Attention**: Efficient KV cache management for long-context inference
- **ANE Operations**: GELU, SiLU, softmax, layer norm optimized for Neural Engine

### Real-Time Learning (SONA)
- **MicroLoRA**: Per-request fine-tuning with rank 1-2 adapters (<1ms latency)
- **EWC++**: Elastic Weight Consolidation to prevent catastrophic forgetting
- **Three-Tier Learning**: Instant (<1ms), Background (~100ms), Deep (minutes)

### Memory Efficiency
- **Two-Tier KV Cache**: FP16 tail + Q4/Q8 quantized store
- **Grouped-Query Attention (GQA)**: 4-8x KV memory reduction
- **Memory Pool**: Arena allocator for zero-allocation inference
- **GGUF Memory Mapping**: Efficient large model loading

## Quick Start

```rust
use ruvllm::prelude::*;

// Initialize backend with Metal GPU + ANE hybrid
let mut backend = CandleBackend::with_device(DeviceType::Metal)?;

// Load a GGUF model
backend.load_gguf("models/qwen2.5-7b-q4_k.gguf", ModelConfig::default())?;

// Or load from HuggingFace
backend.load_model("Qwen/Qwen2.5-7B-Instruct", ModelConfig {
    quantization: Quantization::Q4K,
    use_flash_attention: true,
    ..Default::default()
})?;

// Generate text
let response = backend.generate("Explain quantum computing in simple terms.",
    GenerateParams {
        max_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
        ..Default::default()
    }
)?;

println!("{}", response);

// Check SONA learning stats
if let Some(stats) = backend.sona_stats() {
    println!("Patterns learned: {}", stats.patterns_learned);
    println!("Quality improvement: {:.1}%", stats.quality_improvement * 100.0);
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
# Recommended for Apple Silicon Mac
ruvllm = { version = "2.0", features = ["inference-metal", "coreml", "parallel"] }

# For NVIDIA GPUs
ruvllm = { version = "2.0", features = ["inference-cuda", "parallel"] }

# With RLM recursive reasoning
ruvllm = { version = "2.0", features = ["rlm-full"] }

# Minimal (CPU only)
ruvllm = { version = "2.0" }
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `candle` | Enable Candle backend (HuggingFace) |
| `metal` | Apple Silicon GPU acceleration via Candle |
| `metal-compute` | Native Metal compute shaders (M4 Pro optimized) |
| `cuda` | NVIDIA GPU acceleration |
| `coreml` | Apple Neural Engine via Core ML |
| `hybrid-ane` | GPU+ANE hybrid pipeline (recommended for Mac) |
| `inference-metal` | Full Metal inference stack |
| `inference-metal-native` | Metal + native shaders (best M4 Pro perf) |
| `inference-cuda` | Full CUDA inference stack |
| `parallel` | Multi-threaded GEMM/GEMV with Rayon |
| `accelerate` | Apple Accelerate BLAS (~2x GEMV speedup) |
| `gguf-mmap` | Memory-mapped GGUF loading |
| `async-runtime` | Tokio async support |
| `wasm` | WebAssembly support |
| **`rlm-core`** | RLM recursive reasoning core (includes cache, pools, SIMD) |
| **`rlm-wasm`** | RLM with WASM support for browsers |
| **`rlm-full`** | Full RLM with async runtime |
| `attention` | Ruvector attention mechanisms |
| `graph` | Ruvector graph integration |
| `gnn` | Graph neural network support |
| `ruvector-full` | All Ruvector integrations |

## Architecture

```
+----------------------------------+
|         Application              |
+----------------------------------+
              |
+----------------------------------+
|        RuvLLM Backend            |
|  +----------------------------+  |
|  |   Hybrid Pipeline Router   |  |
|  |  +----------+ +----------+ |  |
|  |  |  Metal   | |   ANE    | |  |
|  |  |   GPU    | | Core ML  | |  |
|  |  +----+-----+ +----+-----+ |  |
|  |       |    v      |        |  |
|  |  Attention    MLP/FFN      |  |
|  |  RoPE         Activations  |  |
|  |  Softmax      LayerNorm    |  |
|  +----------------------------+  |
|              |                   |
|  +----------------------------+  |
|  |     SONA Learning          |  |
|  |  - Instant (<1ms)          |  |
|  |  - Background (~100ms)     |  |
|  |  - Deep (minutes)          |  |
|  +----------------------------+  |
|              |                   |
|  +----------------------------+  |
|  |     NEON/SIMD Kernels      |  |
|  |  - Flash Attention 2       |  |
|  |  - Paged KV Cache          |  |
|  |  - Quantized MatMul        |  |
|  +----------------------------+  |
+----------------------------------+
```

## Supported Models

| Model Family | Sizes | Quantization | Backend |
|--------------|-------|--------------|---------|
| **RuvLTRA-Small** | 0.5B | Q4K, Q5K, Q8, FP16 | Candle/Metal/ANE |
| **RuvLTRA-Medium** | 3B | Q4K, Q5K, Q8, FP16 | Candle/Metal |
| Qwen 2.5 | 0.5B-72B | Q4K, Q8, FP16 | Candle/Metal |
| Llama 3.x | 8B-70B | Q4K, Q8, FP16 | Candle/Metal |
| Mistral | 7B-22B | Q4K, Q8, FP16 | Candle/Metal |
| Phi-3 | 3.8B-14B | Q4K, Q8, FP16 | Candle/Metal |
| Gemma-2 | 2B-27B | Q4K, Q8, FP16 | Candle/Metal |

### RuvLTRA Models (Claude Flow Optimized)

| Model | Parameters | Hidden | Layers | Context | Features |
|-------|------------|--------|--------|---------|----------|
| RuvLTRA-Small | 494M | 896 | 24 | 32K | GQA 7:1, SONA hooks |
| RuvLTRA-Medium | 3.0B | 2560 | 42 | 256K | Flash Attention 2, Speculative Decode |

### HuggingFace Model Links

Pre-trained RuvLTRA models are available on HuggingFace:

- **Repository**: [huggingface.co/ruv/ruvltra](https://huggingface.co/ruv/ruvltra)

| Model | File | Size | Purpose |
|-------|------|------|---------|
| RuvLTRA Claude Code 0.5B | `ruvltra-claude-code-0.5b-q4_k_m.gguf` | ~400MB | Agent routing (100% accuracy with hybrid) |
| RuvLTRA Small 0.5B | `ruvltra-0.5b-q4_k_m.gguf` | ~400MB | General embeddings |
| RuvLTRA Medium 3B | `ruvltra-3b-q4_k_m.gguf` | ~2GB | Full LLM inference |

Download models:
```bash
# Using huggingface-cli
huggingface-cli download ruv/ruvltra ruvltra-claude-code-0.5b-q4_k_m.gguf --local-dir ~/.ruvllm/models

# Or via the API
curl -L https://huggingface.co/ruv/ruvltra/resolve/main/ruvltra-claude-code-0.5b-q4_k_m.gguf -o ~/.ruvllm/models/ruvltra-claude-code-0.5b-q4_k_m.gguf
```

## Performance Benchmarks

### Inference (M4 Pro 14-core)

| Model | Quant | Prefill (tok/s) | Decode (tok/s) | Memory |
|-------|-------|-----------------|----------------|--------|
| Qwen2.5-7B | Q4K | 2,800 | 95 | 4.2 GB |
| Qwen2.5-7B | Q8 | 2,100 | 72 | 7.8 GB |
| Llama3-8B | Q4K | 2,600 | 88 | 4.8 GB |
| Mistral-7B | Q4K | 2,500 | 85 | 4.1 GB |
| Phi-3-3.8B | Q4K | 3,500 | 135 | 2.3 GB |
| Gemma2-9B | Q4K | 2,200 | 75 | 5.2 GB |

### RLM Decomposition Performance

| Query Complexity | Sub-queries | Decomposition Time | Total Time |
|-----------------|-------------|-------------------|------------|
| Simple | 1 | <1ms | 50-100ms |
| Moderate | 2-3 | 2-5ms | 150-300ms |
| Complex | 4-6 | 5-10ms | 400-800ms |
| Deep reasoning | 6-10 | 10-20ms | 1-3s |

### ANE vs GPU Performance (M4 Pro)

| Dimension | ANE | GPU | Winner |
|-----------|-----|-----|--------|
| < 512 | +30-50% | - | ANE |
| 512-1024 | +10-30% | - | ANE |
| 1024-1536 | ~Similar | ~Similar | Either |
| 1536-2048 | - | +10-20% | GPU |
| > 2048 | - | +30-50% | GPU |

### Kernel Benchmarks

| Kernel | Single-thread | Multi-thread (10-core) |
|--------|---------------|------------------------|
| GEMM 4096x4096 | 1.2 GFLOPS | 12.7 GFLOPS |
| GEMV 4096x4096 | 0.8 GFLOPS | 6.4 GFLOPS |
| Flash Attention (seq=2048) | 850us | 320us |
| RMS Norm (4096) | 2.1us | 0.8us |
| RoPE (4096, 128) | 4.3us | 1.6us |

## RLM Usage Examples

### Basic Recursive Query

```rust
use ruvllm::rlm::{RlmController, RlmConfig, NativeEnvironment};

let controller = RlmController::<NativeEnvironment>::new(RlmConfig::default())?;

// Complex query gets automatically decomposed
let result = controller.query(
    "Compare the economic policies of keynesian and monetarist schools,
     and explain how each would respond to stagflation"
)?;

println!("Answer: {}", result.text);
println!("Sub-queries processed: {}", result.stats.sub_queries_count);
println!("Memory retrievals: {}", result.stats.memory_hits);
```

### With Memory Context

```rust
// Add domain knowledge to memory
controller.add_memory(
    "Keynesian economics emphasizes government intervention and aggregate demand",
    Default::default()
)?;
controller.add_memory(
    "Monetarism focuses on controlling money supply to manage inflation",
    Default::default()
)?;

// Query now uses memory context
let result = controller.query("What are the key differences between these economic schools?")?;
```

### Custom Decomposition Strategy

```rust
use ruvllm::rlm::{RecursiveConfigBuilder, AggregationStrategy, DecomposerStrategy};

let config = RecursiveConfigBuilder::new()
    .max_depth(3)
    .token_budget(8000)
    .decomposition_strategy(DecomposerStrategy::Semantic)
    .aggregation(AggregationStrategy::Summarize)
    .build()?;

let controller = RlmController::<NativeEnvironment>::new(config)?;
```

### Using Memory Pools for High-Throughput

```rust
use ruvllm::rlm::pool::{VectorPool, PoolManager};

// Create pre-warmed pools for embedding operations
let vector_pool = VectorPool::new_warmed(384, 64, 32);

// Or use the pool manager for convenience
let manager = PoolManager::warmed(384, 64, 128, 32);

// Acquire vectors from pool (zero allocation if pool has capacity)
let mut embedding = manager.vector_pool.acquire();
embedding.extend_from_slice(&query_embedding);

// Vector automatically returns to pool on drop
// Check pool statistics
let stats = manager.stats();
println!("Hit rate: {:.1}%", stats.overall_hit_rate() * 100.0);
println!("Allocations saved: {}", stats.total_allocations_saved());
```

### WASM Usage (Browser)

```rust
#[cfg(target_arch = "wasm32")]
use ruvllm::rlm::{WasmRlmController, WasmEnvironment};

#[cfg(target_arch = "wasm32")]
async fn query_in_browser() -> Result<String, JsValue> {
    let controller = WasmRlmController::new(Default::default()).await?;
    let result = controller.query("What is machine learning?").await?;
    Ok(result.text)
}
```

## Apple Neural Engine (ANE) Integration

RuvLLM includes full ANE support via Core ML:

```rust
use ruvllm::backends::coreml::{CoreMLBackend, AneStrategy};

// Create ANE-optimized backend
let backend = CoreMLBackend::new(AneStrategy::PreferAneForMlp)?;

// Or use hybrid pipeline for best performance
use ruvllm::backends::HybridPipeline;

let pipeline = HybridPipeline::new(HybridConfig {
    ane_strategy: AneStrategy::Adaptive,
    gpu_for_attention: true,  // Attention on GPU
    ane_for_mlp: true,        // MLP/FFN on ANE
    ..Default::default()
})?;
```

### ANE Routing Recommendations

| Operation | Recommended | Reason |
|-----------|-------------|--------|
| Attention | GPU | Better for variable sequence lengths |
| Flash Attention | GPU | GPU memory bandwidth advantage |
| MLP/FFN | ANE | Optimal for fixed-size matmuls |
| GELU/SiLU | ANE | Dedicated activation units |
| LayerNorm/RMSNorm | ANE | Good for small dimensions |
| Embedding | GPU | Sparse operations |

## MicroLoRA Real-Time Adaptation

RuvLLM supports per-request fine-tuning using MicroLoRA:

```rust
use ruvllm::lora::{MicroLoRA, MicroLoraConfig, AdaptFeedback};

// Create MicroLoRA adapter
let config = MicroLoraConfig::for_hidden_dim(4096);
let lora = MicroLoRA::new(config);

// Adapt on user feedback
let feedback = AdaptFeedback::from_quality(0.9);
lora.adapt(&input_embedding, feedback)?;

// Apply learned updates
lora.apply_updates(0.01); // learning rate

// Get adaptation stats
let stats = lora.stats();
println!("Samples: {}, Avg quality: {:.2}", stats.samples, stats.avg_quality);
```

## SONA Three-Tier Learning

Continuous improvement with three learning loops:

```rust
use ruvllm::optimization::{SonaLlm, SonaLlmConfig, ConsolidationStrategy};

let config = SonaLlmConfig {
    instant_lr: 0.01,
    background_interval_ms: 100,
    deep_trigger_threshold: 100.0,
    consolidation_strategy: ConsolidationStrategy::EwcMerge,
    ..Default::default()
};

let sona = SonaLlm::new(config);

// 1. Instant Loop (<1ms): Per-request MicroLoRA
let result = sona.instant_adapt("user query", "model response", 0.85);
println!("Instant adapt: {}us", result.latency_us);

// 2. Background Loop (~100ms): Pattern consolidation
if let result = sona.maybe_background() {
    if result.applied {
        println!("Consolidated {} samples", result.samples_used);
    }
}

// 3. Deep Loop (minutes): Full optimization
if sona.should_trigger_deep() {
    let result = sona.deep_optimize(OptimizationTrigger::QualityThreshold(100.0));
    println!("Deep optimization: {:.1}s", result.latency_us as f64 / 1_000_000.0);
}
```

## Two-Tier KV Cache

Memory-efficient caching with automatic tiering:

```rust
use ruvllm::kv_cache::{TwoTierKvCache, KvCacheConfig};

let config = KvCacheConfig {
    tail_length: 256,              // Recent tokens in FP16
    tail_precision: Precision::FP16,
    store_precision: Precision::Q4,  // Older tokens in Q4
    max_tokens: 8192,
    num_layers: 32,
    num_kv_heads: 8,
    head_dim: 128,
};

let cache = TwoTierKvCache::new(config);
cache.append(&keys, &values)?;

// Automatic migration from tail to quantized store
let stats = cache.stats();
println!("Tail: {} tokens, Store: {} tokens", stats.tail_tokens, stats.store_tokens);
println!("Compression ratio: {:.2}x", stats.compression_ratio);
```

## HuggingFace Hub Integration

Download and upload models to HuggingFace Hub:

```rust
use ruvllm::hub::{ModelDownloader, ModelUploader, RuvLtraRegistry, DownloadConfig};

// Download from Hub
let downloader = ModelDownloader::new(DownloadConfig::default());
let model_path = downloader.download(
    "ruvector/ruvltra-small-q4km",
    Some("./models"),
)?;

// Or use the registry for RuvLTRA models
let registry = RuvLtraRegistry::new();
let model = registry.get("ruvltra-medium", "Q4_K_M")?;

// Upload to Hub (requires HF_TOKEN)
let uploader = ModelUploader::new("hf_your_token");
let url = uploader.upload(
    "./my-model.gguf",
    "username/my-ruvltra-model",
    Some(metadata),
)?;
println!("Uploaded to: {}", url);
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUVLLM_CACHE_DIR` | Model cache directory | `~/.cache/ruvllm` |
| `RUVLLM_LOG_LEVEL` | Logging level | `info` |
| `RUVLLM_METAL_DEVICE` | Metal device index | `0` |
| `RUVLLM_ANE_ENABLED` | Enable ANE routing | `true` |
| `RUVLLM_SONA_ENABLED` | Enable SONA learning | `true` |
| `HF_TOKEN` | HuggingFace API token | - |

### Model Configuration

```rust
let config = ModelConfig {
    max_context: 8192,
    use_flash_attention: true,
    quantization: Quantization::Q4K,
    kv_cache_config: KvCacheConfig::default(),
    rope_scaling: Some(RopeScaling::Linear { factor: 2.0 }),
    sliding_window: Some(4096),
    ..Default::default()
};
```

## Benchmarks

Run benchmarks with:

```bash
# Attention benchmarks
cargo bench --bench attention_bench --features inference-metal

# ANE benchmarks (Mac only)
cargo bench --bench ane_bench --features coreml

# LoRA benchmarks
cargo bench --bench lora_bench

# RLM benchmarks
cargo bench --bench rlm_bench --features rlm-full

# End-to-end inference
cargo bench --bench e2e_bench --features inference-metal

# Metal shader benchmarks
cargo bench --bench metal_bench --features metal-compute

# Serving benchmarks
cargo bench --bench serving_bench --features inference-metal

# RuvLTRA router benchmarks
cargo bench --bench ruvltra_benchmark
```

## npm Package

RuvLLM is also available as an npm package with native bindings:

```bash
npm install @ruvector/ruvllm
```

```typescript
import { RuvLLM } from '@ruvector/ruvllm';

const llm = new RuvLLM();
const response = llm.query('Explain quantum computing');
console.log(response.text);
```

See [@ruvector/ruvllm on npm](https://www.npmjs.com/package/@ruvector/ruvllm) for full documentation.

## Error Handling

```rust
use ruvllm::error::{Result, RuvLLMError};

match backend.generate(prompt, params) {
    Ok(response) => println!("{}", response),
    Err(RuvLLMError::Model(e)) => eprintln!("Model error: {}", e),
    Err(RuvLLMError::OutOfMemory(e)) => eprintln!("OOM: {}", e),
    Err(RuvLLMError::Generation(e)) => eprintln!("Generation failed: {}", e),
    Err(RuvLLMError::Ane(e)) => eprintln!("ANE error: {}", e),
    Err(RuvLLMError::Gguf(e)) => eprintln!("GGUF loading error: {}", e),
    Err(e) => eprintln!("Error: {}", e),
}
```

## License

Apache-2.0 / MIT dual license.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [API Documentation](https://docs.rs/ruvllm)
- [npm Package](https://www.npmjs.com/package/@ruvector/ruvllm)
- [Issue Tracker](https://github.com/ruvnet/ruvector/issues)
- [crates.io](https://crates.io/crates/ruvllm)
- [HuggingFace Models](https://huggingface.co/ruv/ruvltra)
