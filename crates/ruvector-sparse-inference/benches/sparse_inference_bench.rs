//! Benchmark tests for sparse inference

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use ruvector_sparse_inference::*;
use rand::Rng;

// Test utilities
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn load_benchmark_model() -> model::LlamaModel {
    model::LlamaModel::new(512, 2048, 4, 32000)
}

fn benchmark_sparse_vs_dense(c: &mut Criterion) {
    let model = load_benchmark_model();
    let dense_engine = SparseInferenceEngine::new_dense(model.clone());
    let sparse_engine = SparseInferenceEngine::new_sparse(model, 0.1);
    let input = random_vector(512);

    let mut group = c.benchmark_group("inference");

    group.bench_function("dense", |b| {
        b.iter(|| {
            black_box(dense_engine.infer(&input).unwrap())
        })
    });

    group.bench_function("sparse_10pct", |b| {
        b.iter(|| {
            black_box(sparse_engine.infer(&input).unwrap())
        })
    });

    group.finish();
}

fn benchmark_sparsity_levels(c: &mut Criterion) {
    let model = load_benchmark_model();
    let input = random_vector(512);

    let mut group = c.benchmark_group("sparsity_levels");

    for sparsity in [0.3, 0.5, 0.7, 0.9] {
        let engine = SparseInferenceEngine::new_sparse(model.clone(), sparsity);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.0}%", sparsity * 100.0)),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(engine.infer(input).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn benchmark_predictor(c: &mut Criterion) {
    let predictor = predictor::LowRankPredictor::new(512, 4096, 128, 0.1);
    let input = random_vector(512);

    c.bench_function("predictor_predict", |b| {
        b.iter(|| {
            black_box(predictor.predict(&input))
        })
    });
}

fn benchmark_predictor_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("predictor_top_k");
    let input = random_vector(512);

    for k in [100, 500, 1000, 2000] {
        let mut predictor = predictor::LowRankPredictor::new(512, 4096, 128, 0.0);
        predictor.set_top_k(Some(k));

        group.bench_with_input(
            BenchmarkId::from_parameter(k),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(predictor.predict(input))
                })
            },
        );
    }

    group.finish();
}

fn benchmark_sparse_ffn(c: &mut Criterion) {
    let ffn = sparse::SparseFfn::new(512, 2048, sparse::ActivationType::Silu);
    let input = random_vector(512);

    let mut group = c.benchmark_group("sparse_ffn");

    group.bench_function("dense_forward", |b| {
        b.iter(|| {
            black_box(ffn.forward_dense(&input))
        })
    });

    let active_10pct: Vec<usize> = (0..204).collect();
    group.bench_function("sparse_10pct", |b| {
        b.iter(|| {
            black_box(ffn.forward_sparse(&input, &active_10pct))
        })
    });

    let active_50pct: Vec<usize> = (0..1024).collect();
    group.bench_function("sparse_50pct", |b| {
        b.iter(|| {
            black_box(ffn.forward_sparse(&input, &active_50pct))
        })
    });

    group.finish();
}

fn benchmark_activation_functions(c: &mut Criterion) {
    let input = random_vector(512);
    let active: Vec<usize> = (0..500).collect();

    let mut group = c.benchmark_group("activation_functions");

    for activation in [
        sparse::ActivationType::Relu,
        sparse::ActivationType::Gelu,
        sparse::ActivationType::Silu,
    ] {
        let ffn = sparse::SparseFfn::new(512, 2048, activation);
        let name = format!("{:?}", activation);

        group.bench_with_input(
            BenchmarkId::from_parameter(&name),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(ffn.forward_sparse(input, &active))
                })
            },
        );
    }

    group.finish();
}

fn benchmark_quantized_dequantization(c: &mut Criterion) {
    let data: Vec<f32> = (0..4096 * 512).map(|i| (i as f32) * 0.001).collect();
    let quantized = memory::quantization::QuantizedWeights::quantize_int8(&data);

    let mut group = c.benchmark_group("quantization");

    group.bench_function("dequantize_1_row", |b| {
        b.iter(|| {
            black_box(quantized.dequantize_row(0))
        })
    });

    let rows_10: Vec<usize> = (0..10).collect();
    group.bench_function("dequantize_10_rows", |b| {
        b.iter(|| {
            black_box(quantized.dequantize_rows(&rows_10))
        })
    });

    let rows_100: Vec<usize> = (0..100).collect();
    group.bench_function("dequantize_100_rows", |b| {
        b.iter(|| {
            black_box(quantized.dequantize_rows(&rows_100))
        })
    });

    group.finish();
}

fn benchmark_int4_vs_int8(c: &mut Criterion) {
    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01).collect();

    let mut group = c.benchmark_group("quantization_types");

    group.bench_function("int8_quantize", |b| {
        b.iter(|| {
            black_box(memory::quantization::QuantizedWeights::quantize_int8(&data))
        })
    });

    group.bench_function("int4_quantize", |b| {
        b.iter(|| {
            black_box(memory::quantization::QuantizedWeights::quantize_int4(&data, 64))
        })
    });

    let int8_quantized = memory::quantization::QuantizedWeights::quantize_int8(&data);
    group.bench_function("int8_dequantize", |b| {
        b.iter(|| {
            black_box(int8_quantized.dequantize_row(0))
        })
    });

    let int4_quantized = memory::quantization::QuantizedWeights::quantize_int4(&data, 64);
    group.bench_function("int4_dequantize", |b| {
        b.iter(|| {
            black_box(int4_quantized.dequantize_row(0))
        })
    });

    group.finish();
}

fn benchmark_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration");

    for num_samples in [10, 50, 100, 500] {
        let samples: Vec<Vec<f32>> = (0..num_samples)
            .map(|_| random_vector(512))
            .collect();

        let activations: Vec<Vec<usize>> = (0..num_samples)
            .map(|_| (0..500).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_samples),
            &(samples, activations),
            |b, (samples, activations)| {
                b.iter(|| {
                    let mut predictor = predictor::LowRankPredictor::new(512, 4096, 128, 0.1);
                    black_box(predictor.calibrate(samples, activations))
                })
            },
        );
    }

    group.finish();
}

fn benchmark_swiglu(c: &mut Criterion) {
    let ffn = sparse::SwiGLUFfn::new(512, 2048);
    let input = random_vector(512);

    let mut group = c.benchmark_group("swiglu");

    group.bench_function("dense", |b| {
        b.iter(|| {
            black_box(ffn.forward_dense(&input))
        })
    });

    let active_pairs: Vec<usize> = (0..500).map(|i| i * 2).collect();
    group.bench_function("sparse", |b| {
        b.iter(|| {
            black_box(ffn.forward_sparse(&input, &active_pairs))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_sparse_vs_dense,
    benchmark_sparsity_levels,
    benchmark_predictor,
    benchmark_predictor_top_k,
    benchmark_sparse_ffn,
    benchmark_activation_functions,
    benchmark_quantized_dequantization,
    benchmark_int4_vs_int8,
    benchmark_calibration,
    benchmark_swiglu,
);

criterion_main!(benches);
