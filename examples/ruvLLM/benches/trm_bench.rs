//! TRM Performance Benchmarks
//!
//! Benchmarks for TRM components to measure:
//! - Latent update throughput (MLP vs Attention)
//! - Full reasoning pipeline latency
//! - SONA routing overhead
//! - Memory allocation patterns

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ruvllm::trm::{
    AttentionLatentUpdater, ConfidenceScorer, MlpLatentUpdater, SonaBridge,
    TrmConfig, TrmEngine, TrmEngineBuilder, AnswerRefiner,
    LatentUpdate, RecursiveReasoner,
};

/// Benchmark MLP latent updater
fn bench_mlp_updater(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_updater");

    for dim in [64, 128, 256, 512].iter() {
        group.throughput(Throughput::Elements(*dim as u64));

        let updater = MlpLatentUpdater::new(*dim, *dim);
        let question = vec![0.5; *dim];
        let answer = vec![0.5; *dim];
        let mut latent = vec![0.0; *dim];

        group.bench_with_input(BenchmarkId::new("update", dim), dim, |b, _| {
            b.iter(|| {
                updater.update(
                    black_box(&question),
                    black_box(&answer),
                    black_box(&mut latent),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark Attention latent updater
fn bench_attention_updater(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_updater");

    for (dim, heads) in [(64, 4), (128, 8), (256, 8), (512, 16)].iter() {
        group.throughput(Throughput::Elements(*dim as u64));

        let updater = AttentionLatentUpdater::new(*dim, *dim, *heads);
        let question = vec![0.5; *dim];
        let answer = vec![0.5; *dim];
        let mut latent = vec![0.0; *dim];

        group.bench_with_input(BenchmarkId::new("update", dim), dim, |b, _| {
            b.iter(|| {
                updater.update(
                    black_box(&question),
                    black_box(&answer),
                    black_box(&mut latent),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark answer refiner
fn bench_refiner(c: &mut Criterion) {
    let mut group = c.benchmark_group("refiner");

    for dim in [64, 128, 256].iter() {
        group.throughput(Throughput::Elements(*dim as u64));

        let config = TrmConfig {
            hidden_dim: *dim,
            embedding_dim: *dim,
            residual_scale: 0.1,
            ..Default::default()
        };
        let refiner = AnswerRefiner::new(&config);
        let question = vec![0.5; *dim];
        let latent = vec![0.5; *dim];
        let mut answer = vec![0.1; *dim * 10];  // 10 tokens

        group.bench_with_input(BenchmarkId::new("refine", dim), dim, |b, _| {
            b.iter(|| {
                refiner.refine(
                    black_box(&question),
                    black_box(&latent),
                    black_box(&mut answer),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark confidence scorer
fn bench_scorer(c: &mut Criterion) {
    let mut group = c.benchmark_group("confidence_scorer");

    for dim in [64, 128, 256, 512].iter() {
        group.throughput(Throughput::Elements(*dim as u64));

        let config = TrmConfig {
            embedding_dim: *dim,
            ..Default::default()
        };
        let scorer = ConfidenceScorer::new(&config);
        let answer = vec![0.5; *dim];

        group.bench_with_input(BenchmarkId::new("score", dim), dim, |b, _| {
            b.iter(|| {
                scorer.score(black_box(&answer))
            });
        });

        group.bench_with_input(BenchmarkId::new("score_entropy", dim), dim, |b, _| {
            b.iter(|| {
                scorer.score_with_entropy(black_box(&answer))
            });
        });
    }

    group.finish();
}

/// Benchmark full TRM reasoning with MLP
fn bench_trm_mlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("trm_mlp");

    for k in [1, 3, 5, 10, 20].iter() {
        let mut engine = TrmEngineBuilder::new()
            .hidden_dim(256)
            .embedding_dim(256)
            .default_k(*k)
            .latent_iterations(3)
            .use_attention(false)
            .early_stopping(false)
            .build()
            .expect("Failed to create engine");

        let question = vec![0.5; 256];
        let mut answer = vec![0.1; 256];

        group.bench_with_input(BenchmarkId::new("k", k), k, |b, _| {
            b.iter(|| {
                engine.reason(black_box(&question), black_box(&mut answer))
            });
        });
    }

    group.finish();
}

/// Benchmark full TRM reasoning with Attention
fn bench_trm_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("trm_attention");

    for k in [1, 3, 5, 10].iter() {
        let mut engine = TrmEngineBuilder::new()
            .hidden_dim(256)
            .embedding_dim(256)
            .default_k(*k)
            .latent_iterations(3)
            .use_attention(true)
            .num_heads(8)
            .early_stopping(false)
            .build()
            .expect("Failed to create engine");

        let question = vec![0.5; 256];
        let mut answer = vec![0.1; 256];

        group.bench_with_input(BenchmarkId::new("k", k), k, |b, _| {
            b.iter(|| {
                engine.reason(black_box(&question), black_box(&mut answer))
            });
        });
    }

    group.finish();
}

/// Benchmark SONA routing
fn bench_sona_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sona_routing");

    let config = TrmConfig {
        hidden_dim: 256,
        embedding_dim: 256,
        max_k: 20,
        ..Default::default()
    };
    let bridge = SonaBridge::new(&config);

    for dim in [64, 128, 256, 512].iter() {
        let query = vec![0.5; *dim];

        group.bench_with_input(BenchmarkId::new("route", dim), dim, |b, _| {
            b.iter(|| {
                bridge.route(black_box(&query))
            });
        });
    }

    group.finish();
}

/// Benchmark MLP vs Attention comparison
fn bench_mlp_vs_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_vs_attention");

    let dim = 256;
    let question = vec![0.5; dim];
    let answer = vec![0.5; dim];
    let mut latent = vec![0.0; dim];

    // MLP
    let mlp = MlpLatentUpdater::new(dim, dim);
    group.bench_function("mlp", |b| {
        b.iter(|| {
            mlp.update(
                black_box(&question),
                black_box(&answer),
                black_box(&mut latent),
            );
        });
    });

    // Attention
    let attention = AttentionLatentUpdater::new(dim, dim, 8);
    group.bench_function("attention", |b| {
        b.iter(|| {
            attention.update(
                black_box(&question),
                black_box(&answer),
                black_box(&mut latent),
            );
        });
    });

    group.finish();
}

/// Benchmark throughput with different answer sizes
fn bench_answer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("answer_throughput");

    for num_tokens in [1, 10, 50, 100].iter() {
        let dim = 256;
        let answer_len = dim * num_tokens;

        let mut engine = TrmEngineBuilder::new()
            .hidden_dim(dim)
            .embedding_dim(dim)
            .default_k(5)
            .early_stopping(false)
            .build()
            .expect("Failed to create engine");

        let question = vec![0.5; dim];
        let mut answer = vec![0.1; answer_len];

        group.throughput(Throughput::Elements(*num_tokens as u64));
        group.bench_with_input(BenchmarkId::new("tokens", num_tokens), num_tokens, |b, _| {
            b.iter(|| {
                engine.reason(black_box(&question), black_box(&mut answer))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mlp_updater,
    bench_attention_updater,
    bench_refiner,
    bench_scorer,
    bench_trm_mlp,
    bench_trm_attention,
    bench_sona_routing,
    bench_mlp_vs_attention,
    bench_answer_throughput,
);

criterion_main!(benches);
