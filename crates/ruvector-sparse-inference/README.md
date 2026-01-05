# ruvector-sparse-inference

PowerInfer-style activation locality inference engine for RuVector.

## Overview

This crate implements efficient sparse neural network inference by predicting which neurons will be active before performing the full computation. This dramatically reduces computation for large feed-forward networks.

### Key Features

- 🎯 **Low-rank Prediction**: Uses P·Q factorization to predict active neurons
- ⚡ **Sparse Computation**: Only computes predicted active neurons
- 🔥 **Hot/Cold Caching**: Separates frequently-used and rarely-used neuron weights
- 📦 **Quantization**: GGUF-compatible weight compression (F16, Int8, Int4)
- 🚀 **SIMD Optimized**: Fast backends for CPU and WASM
- 🧪 **WASM Compatible**: No OS-specific dependencies

## Architecture

```text
Input (d dimensions)
   ↓
[Low-rank Predictor]
   P·Q factorization (rank r << d)
   ↓
Active Neurons (top-K or threshold)
   ↓
[Sparse FFN]
   Only compute selected neurons
   W1: [hidden, input] → sparse matmul
   Activation (ReLU/GeLU/SiLU)
   W2: [output, hidden] → sparse accumulate
   ↓
Output
```

## Usage

```rust
use ruvector_sparse_inference::{
    config::{ModelConfig, SparsityConfig, ActivationType},
    predictor::LowRankPredictor,
    sparse::SparseFfn,
};

// Configure sparsity
let sparsity = SparsityConfig::with_top_k(50); // Top 50 neurons

// Create predictor
let predictor = LowRankPredictor::new(
    128,  // input_dim
    512,  // hidden_dim
    64,   // rank
    sparsity,
).unwrap();

// Create sparse FFN
let ffn = SparseFfn::new(
    128,  // input_dim
    512,  // hidden_dim
    128,  // output_dim
    ActivationType::Gelu,
);

// Run inference
let input = vec![0.1; 128];
let active_neurons = predictor.predict(&input)?;
let output = ffn.forward_sparse(&input, &active_neurons)?;
```

## Configuration Options

### Sparsity Selection

```rust
// Top-K selection
SparsityConfig::with_top_k(100);

// Threshold-based selection
SparsityConfig::with_threshold(0.01);

// Target sparsity ratio
SparsityConfig::with_target_sparsity(0.95); // 95% sparse
```

### Activation Functions

- `Relu`: max(0, x)
- `Gelu`: Gaussian Error Linear Unit
- `Silu`/`Swish`: x * sigmoid(x)
- `Identity`: No activation

### Quantization (with `quantization` feature)

```rust
use ruvector_sparse_inference::memory::QuantizedWeights;

// F32 (no quantization)
let weights = QuantizedWeights::from_f32(matrix);

// Int8 quantization
let weights = QuantizedWeights::Int8 {
    data: quantized_matrix,
    scale: 0.01,
    zero_point: 0,
};

// Int4 quantization (GGUF-style)
let weights = QuantizedWeights::Int4 {
    data: packed_bytes,
    scales: per_group_scales,
    zero_points: per_group_zeros,
    group_size: 32,
    shape: (hidden_dim, input_dim),
};
```

## Performance

Expected speedups for 90% sparsity:
- **Computation**: ~10x faster (only 10% of neurons computed)
- **Memory bandwidth**: ~10x reduction (hot/cold caching)
- **Energy**: Proportional savings on mobile/edge devices

## Features

- `default = ["simd"]`
- `simd`: Enable SIMD optimizations
- `parallel`: Enable parallel computation with rayon
- `quantization`: Enable quantization support
- `npu`: Enable ARM NPU support (experimental)

## Examples

See `examples/` directory:
- `basic_inference.rs`: Simple prediction and computation
- `calibration.rs`: Calibrate predictor on sample data
- `quantization.rs`: Use quantized weights
- `benchmarks.rs`: Performance comparison

## License

Same as workspace (MIT OR Apache-2.0)
