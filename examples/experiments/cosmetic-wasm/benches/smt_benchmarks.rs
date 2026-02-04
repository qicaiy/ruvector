use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cosmetic_wasm::hasher::{self, Hash};
use cosmetic_wasm::proof::{self, CompactProof};
use cosmetic_wasm::tree::SparseMerkleTree;
use cosmetic_wasm::attestation::AttestationBuilder;

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("smt_insert");

    for count in [1, 10, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            b.iter(|| {
                let mut tree = SparseMerkleTree::new();
                for i in 0..n {
                    let key = hasher::compute_key(format!("key_{}", i).as_bytes());
                    tree.insert(key, b"value".to_vec(), None);
                }
                black_box(tree.root())
            });
        });
    }

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("smt_batch_insert");

    for count in [10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            let entries: Vec<(Hash, Vec<u8>, Option<String>)> = (0..n)
                .map(|i| {
                    let key = hasher::compute_key(format!("batch_{}", i).as_bytes());
                    (key, format!("value_{}", i).into_bytes(), None)
                })
                .collect();

            b.iter(|| {
                let mut tree = SparseMerkleTree::new();
                tree.insert_batch(entries.clone());
                black_box(tree.root())
            });
        });
    }

    group.finish();
}

fn bench_prove_inclusion(c: &mut Criterion) {
    let mut tree = SparseMerkleTree::new();
    let keys: Vec<Hash> = (0..100)
        .map(|i| {
            let key = hasher::compute_key(format!("prove_{}", i).as_bytes());
            tree.insert(key, b"data".to_vec(), None);
            key
        })
        .collect();

    c.bench_function("prove_inclusion", |b| {
        let mut idx = 0;
        b.iter(|| {
            let proof = tree.prove_inclusion(&keys[idx % keys.len()]).unwrap();
            idx += 1;
            black_box(proof)
        });
    });
}

fn bench_prove_exclusion(c: &mut Criterion) {
    let mut tree = SparseMerkleTree::new();
    for i in 0..100 {
        let key = hasher::compute_key(format!("exist_{}", i).as_bytes());
        tree.insert(key, b"data".to_vec(), None);
    }

    let absent_keys: Vec<Hash> = (0..100)
        .map(|i| hasher::compute_key(format!("absent_{}", i).as_bytes()))
        .collect();

    c.bench_function("prove_exclusion", |b| {
        let mut idx = 0;
        b.iter(|| {
            let proof = tree.prove_exclusion(&absent_keys[idx % absent_keys.len()]).unwrap();
            idx += 1;
            black_box(proof)
        });
    });
}

fn bench_verify_inclusion(c: &mut Criterion) {
    let mut tree = SparseMerkleTree::new();
    let key = hasher::compute_key(b"verify_bench");
    tree.insert(key, b"data".to_vec(), None);
    let inc_proof = tree.prove_inclusion(&key).unwrap();

    c.bench_function("verify_inclusion", |b| {
        b.iter(|| black_box(proof::verify_inclusion(&inc_proof)))
    });
}

fn bench_verify_exclusion(c: &mut Criterion) {
    let mut tree = SparseMerkleTree::new();
    let key = hasher::compute_key(b"present_key");
    tree.insert(key, b"data".to_vec(), None);

    let absent = hasher::compute_key(b"absent_key");
    let exc_proof = tree.prove_exclusion(&absent).unwrap();

    c.bench_function("verify_exclusion", |b| {
        b.iter(|| black_box(proof::verify_exclusion(&exc_proof)))
    });
}

fn bench_compact_proof(c: &mut Criterion) {
    let mut tree = SparseMerkleTree::new();
    let key = hasher::compute_key(b"compact_bench");
    tree.insert(key, b"data".to_vec(), None);
    let inc_proof = tree.prove_inclusion(&key).unwrap();

    let mut group = c.benchmark_group("compact_proof");

    group.bench_function("compress", |b| {
        b.iter(|| black_box(CompactProof::from_inclusion(&inc_proof)))
    });

    let compact = CompactProof::from_inclusion(&inc_proof);
    group.bench_function("decompress", |b| {
        b.iter(|| black_box(compact.decompress_siblings()))
    });

    group.bench_function("verify", |b| {
        b.iter(|| black_box(compact.verify()))
    });

    group.finish();
}

fn bench_hex_encoding(c: &mut Criterion) {
    let hash = hasher::sha256(b"benchmark");

    let mut group = c.benchmark_group("hex");
    group.bench_function("to_hex", |b| {
        b.iter(|| black_box(hasher::to_hex(&hash)))
    });

    let hex_str = hasher::to_hex(&hash);
    group.bench_function("from_hex", |b| {
        b.iter(|| black_box(hasher::from_hex(&hex_str)))
    });
    group.finish();
}

fn bench_hash_functions(c: &mut Criterion) {
    let data = b"benchmark_data_for_hashing_performance";
    let key = hasher::sha256(b"key");
    let left = hasher::sha256(b"left");
    let right = hasher::sha256(b"right");

    let mut group = c.benchmark_group("hash");
    group.bench_function("sha256", |b| {
        b.iter(|| black_box(hasher::sha256(data)))
    });
    group.bench_function("hash_leaf", |b| {
        b.iter(|| black_box(hasher::hash_leaf(&key, data)))
    });
    group.bench_function("hash_internal", |b| {
        b.iter(|| black_box(hasher::hash_internal(&left, &right)))
    });
    group.bench_function("hash_attestation", |b| {
        b.iter(|| black_box(hasher::hash_attestation(&left, &right, b"fn_id", b"params")))
    });
    group.finish();
}

fn bench_attestation(c: &mut Criterion) {
    let input_root = hasher::sha256(b"input");
    let output_root = hasher::sha256(b"output");

    c.bench_function("attestation_build_verify", |b| {
        b.iter(|| {
            let key1 = hasher::compute_key(b"p1");
            let key2 = hasher::compute_key(b"p2");
            let att = AttestationBuilder::new("filter")
                .with_parameters(b"age>=18".to_vec())
                .include(key1, "eligible", "age>=18")
                .exclude(key2, "too young", "age>=18")
                .build(input_root, output_root);
            black_box(cosmetic_wasm::attestation::verify_attestation(&att))
        });
    });
}

fn bench_snapshot_restore(c: &mut Criterion) {
    let mut tree = SparseMerkleTree::new();
    for i in 0..50 {
        let key = hasher::compute_key(format!("snap_{}", i).as_bytes());
        tree.insert(key, format!("val_{}", i).into_bytes(), None);
    }

    c.bench_function("snapshot_restore_50", |b| {
        let snap = tree.snapshot();
        b.iter(|| {
            let restored = SparseMerkleTree::from_snapshot(snap.clone());
            black_box(restored.root())
        });
    });
}

criterion_group!(
    benches,
    bench_insert,
    bench_batch_insert,
    bench_prove_inclusion,
    bench_prove_exclusion,
    bench_verify_inclusion,
    bench_verify_exclusion,
    bench_compact_proof,
    bench_hex_encoding,
    bench_hash_functions,
    bench_attestation,
    bench_snapshot_restore,
);
criterion_main!(benches);
