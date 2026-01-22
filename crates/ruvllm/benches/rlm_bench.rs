//! RLM (Recursive Language Model) Benchmarks
//!
//! Performance benchmarks for the RLM module including:
//! - Query decomposition latency
//! - Cache lookup performance
//! - Embedding generation throughput
//! - Memory search latency
//! - End-to-end query processing
//!
//! Performance targets:
//! - Query decomposition: <1ms for simple queries
//! - Cache lookup: <100us
//! - Memory search (k=5): <5ms
//! - Full query (cached): <500us
//! - Full query (uncached): <50ms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use std::time::Duration;

// ============================================================================
// Mock Structures for Benchmarking
// ============================================================================

/// Simple query decomposer for benchmarking (mirrors real implementation)
#[derive(Clone)]
struct MockQueryDecomposer {
    conjunction_keywords: Vec<String>,
    comparison_keywords: Vec<String>,
    sequential_keywords: Vec<String>,
}

impl MockQueryDecomposer {
    fn new() -> Self {
        Self {
            conjunction_keywords: vec![
                "and".to_string(),
                "also".to_string(),
                "as well as".to_string(),
            ],
            comparison_keywords: vec![
                "compare".to_string(),
                "versus".to_string(),
                "vs".to_string(),
            ],
            sequential_keywords: vec!["then".to_string(), "after".to_string(), "next".to_string()],
        }
    }

    fn decompose(&self, query: &str) -> DecompositionResult {
        let query_lower = query.to_lowercase();

        // Check for comparison
        for keyword in &self.comparison_keywords {
            if query_lower.contains(keyword) {
                return DecompositionResult {
                    strategy: DecompositionStrategy::Comparison,
                    sub_queries: self.split_comparison(query, keyword),
                    complexity: self.compute_complexity(query),
                };
            }
        }

        // Check for conjunction
        for keyword in &self.conjunction_keywords {
            if query_lower.contains(keyword) {
                return DecompositionResult {
                    strategy: DecompositionStrategy::Conjunction,
                    sub_queries: self.split_conjunction(query, keyword),
                    complexity: self.compute_complexity(query),
                };
            }
        }

        // Check for sequential
        for keyword in &self.sequential_keywords {
            if query_lower.contains(keyword) {
                return DecompositionResult {
                    strategy: DecompositionStrategy::Sequential,
                    sub_queries: self.split_sequential(query, keyword),
                    complexity: self.compute_complexity(query),
                };
            }
        }

        // Direct query
        DecompositionResult {
            strategy: DecompositionStrategy::Direct,
            sub_queries: vec![query.to_string()],
            complexity: self.compute_complexity(query),
        }
    }

    fn compute_complexity(&self, query: &str) -> f32 {
        let mut score = 0.0f32;
        let query_lower = query.to_lowercase();

        // Length factor
        score += (query.len() as f32 / 200.0).min(0.3);

        // Conjunction factor
        let conj_count = self
            .conjunction_keywords
            .iter()
            .filter(|kw| query_lower.contains(kw.as_str()))
            .count();
        score += (conj_count as f32 * 0.15).min(0.3);

        // Question word factor
        let question_words = ["what", "why", "how", "when", "where", "who"];
        let q_count = question_words
            .iter()
            .filter(|w| query_lower.contains(*w))
            .count();
        score += (q_count as f32 * 0.1).min(0.2);

        score.min(1.0)
    }

    fn split_comparison(&self, query: &str, _keyword: &str) -> Vec<String> {
        vec![
            format!(
                "What is {}?",
                query
                    .split_whitespace()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            format!(
                "What is {}?",
                query
                    .split_whitespace()
                    .rev()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        ]
    }

    fn split_conjunction(&self, query: &str, keyword: &str) -> Vec<String> {
        query
            .split(keyword)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn split_sequential(&self, query: &str, keyword: &str) -> Vec<String> {
        query
            .split(keyword)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

#[derive(Debug, Clone)]
enum DecompositionStrategy {
    Direct,
    Conjunction,
    Comparison,
    Sequential,
}

#[derive(Debug, Clone)]
struct DecompositionResult {
    strategy: DecompositionStrategy,
    sub_queries: Vec<String>,
    complexity: f32,
}

/// Mock memoization cache
struct MockCache {
    entries: std::collections::HashMap<String, CacheEntry>,
    max_entries: usize,
}

#[derive(Clone)]
struct CacheEntry {
    answer: String,
    quality: f32,
    created_at: std::time::Instant,
}

impl MockCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(max_entries),
            max_entries,
        }
    }

    fn get(&self, query: &str) -> Option<&CacheEntry> {
        self.entries.get(&query.to_lowercase())
    }

    fn insert(&mut self, query: &str, answer: String, quality: f32) {
        if self.entries.len() >= self.max_entries {
            // Simple eviction: remove first entry
            if let Some(key) = self.entries.keys().next().cloned() {
                self.entries.remove(&key);
            }
        }
        self.entries.insert(
            query.to_lowercase(),
            CacheEntry {
                answer,
                quality,
                created_at: std::time::Instant::now(),
            },
        );
    }

    fn contains(&self, query: &str) -> bool {
        self.entries.contains_key(&query.to_lowercase())
    }
}

/// Mock quality scorer
struct MockQualityScorer {
    weights: QualityWeights,
}

struct QualityWeights {
    completeness: f32,
    coherence: f32,
    relevance: f32,
    confidence: f32,
}

impl Default for QualityWeights {
    fn default() -> Self {
        Self {
            completeness: 0.3,
            coherence: 0.25,
            relevance: 0.3,
            confidence: 0.15,
        }
    }
}

impl MockQualityScorer {
    fn new() -> Self {
        Self {
            weights: QualityWeights::default(),
        }
    }

    fn score(&self, query: &str, answer: &str) -> f32 {
        let completeness = self.score_completeness(answer);
        let coherence = self.score_coherence(answer);
        let relevance = self.score_relevance(query, answer);
        let confidence = self.score_confidence(answer);

        completeness * self.weights.completeness
            + coherence * self.weights.coherence
            + relevance * self.weights.relevance
            + confidence * self.weights.confidence
    }

    fn score_completeness(&self, answer: &str) -> f32 {
        let len = answer.len();
        if len < 10 {
            0.2
        } else if len > 2000 {
            0.8
        } else {
            1.0
        }
    }

    fn score_coherence(&self, answer: &str) -> f32 {
        let sentences: Vec<&str> = answer
            .split(|c| c == '.' || c == '!')
            .filter(|s| !s.trim().is_empty())
            .collect();
        if sentences.len() >= 2 {
            0.9
        } else {
            0.7
        }
    }

    fn score_relevance(&self, query: &str, answer: &str) -> f32 {
        let query_words: std::collections::HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(String::from)
            .collect();

        let answer_lower = answer.to_lowercase();
        let matched = query_words
            .iter()
            .filter(|w| answer_lower.contains(w.as_str()))
            .count();

        if query_words.is_empty() {
            0.5
        } else {
            0.3 + 0.5 * (matched as f32 / query_words.len() as f32)
        }
    }

    fn score_confidence(&self, answer: &str) -> f32 {
        let lower = answer.to_lowercase();
        let mut score = 0.7f32;

        if lower.contains("i'm not sure") || lower.contains("maybe") {
            score -= 0.15;
        }
        if lower.contains("specifically") || lower.contains("is defined as") {
            score += 0.1;
        }

        score.clamp(0.0, 1.0)
    }
}

/// Mock embedding generator
struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // Simple hash-based embedding for benchmarking
        let mut embedding = vec![0.0f32; self.dim];
        let text_bytes = text.as_bytes();

        for (i, &byte) in text_bytes.iter().enumerate() {
            embedding[i % self.dim] += (byte as f32 / 255.0) - 0.5;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }
}

/// Mock memory store
struct MockMemoryStore {
    entries: Vec<MemoryEntry>,
    embedder: MockEmbedder,
}

struct MemoryEntry {
    id: String,
    text: String,
    embedding: Vec<f32>,
}

impl MockMemoryStore {
    fn new(dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            embedder: MockEmbedder::new(dim),
        }
    }

    fn add(&mut self, text: &str) -> String {
        let id = format!("mem-{}", self.entries.len());
        let embedding = self.embedder.embed(text);
        self.entries.push(MemoryEntry {
            id: id.clone(),
            text: text.to_string(),
            embedding,
        });
        id
    }

    fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(&MemoryEntry, f32)> {
        let mut results: Vec<_> = self
            .entries
            .iter()
            .map(|entry| {
                let similarity = cosine_similarity(&entry.embedding, query_embedding);
                (entry, similarity)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// Helper function to generate random queries
fn generate_random_queries(count: usize) -> Vec<String> {
    let templates = [
        "What is {}?",
        "How does {} work?",
        "Compare {} and {}",
        "What are the causes and effects of {}?",
        "Explain {} then describe {}",
    ];

    let topics = [
        "machine learning",
        "deep learning",
        "neural networks",
        "artificial intelligence",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "transformers",
    ];

    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let template = templates[rng.gen_range(0..templates.len())];
            let topic1 = topics[rng.gen_range(0..topics.len())];
            let topic2 = topics[rng.gen_range(0..topics.len())];
            template.replace("{}", topic1).replace("{}", topic2)
        })
        .collect()
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_query_decomposition(c: &mut Criterion) {
    let decomposer = MockQueryDecomposer::new();

    let queries = vec![
        "What is AI?",
        "What are the causes and effects of climate change?",
        "Compare Python and JavaScript. Also explain when to use each.",
        "First explain machine learning, then describe deep learning, finally discuss transformers",
    ];

    let mut group = c.benchmark_group("query_decomposition");
    group.sample_size(1000);

    for query in queries {
        let id = BenchmarkId::new("query", query.len());
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(id, query, |b, q| {
            b.iter(|| decomposer.decompose(black_box(q)))
        });
    }

    group.finish();
}

fn bench_complexity_scoring(c: &mut Criterion) {
    let decomposer = MockQueryDecomposer::new();

    let queries = vec![
        "What is X?",
        "What are the primary causes of climate change, and how do they contribute to global warming?",
        "Compare A and B, then explain C and D, also discuss E and F",
    ];

    let mut group = c.benchmark_group("complexity_scoring");
    group.sample_size(1000);

    for query in queries {
        let id = BenchmarkId::new("complexity", query.len());
        group.bench_with_input(id, query, |b, q| {
            b.iter(|| decomposer.compute_complexity(black_box(q)))
        });
    }

    group.finish();
}

fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");
    group.sample_size(1000);

    // Cache lookup (hit)
    {
        let mut cache = MockCache::new(10000);
        cache.insert("test query", "test answer".to_string(), 0.9);

        group.bench_function("cache_lookup_hit", |b| {
            b.iter(|| cache.get(black_box("test query")))
        });
    }

    // Cache lookup (miss)
    {
        let cache = MockCache::new(10000);

        group.bench_function("cache_lookup_miss", |b| {
            b.iter(|| cache.get(black_box("nonexistent query")))
        });
    }

    // Cache insert
    {
        group.bench_function("cache_insert", |b| {
            b.iter_batched(
                || MockCache::new(10000),
                |mut cache| {
                    cache.insert(
                        black_box("test query"),
                        black_box("test answer".to_string()),
                        0.9,
                    );
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    // Cache contains check
    {
        let mut cache = MockCache::new(10000);
        for i in 0..1000 {
            cache.insert(&format!("query {}", i), format!("answer {}", i), 0.9);
        }

        group.bench_function("cache_contains_1000", |b| {
            b.iter(|| cache.contains(black_box("query 500")))
        });
    }

    group.finish();
}

fn bench_quality_scoring(c: &mut Criterion) {
    let scorer = MockQualityScorer::new();

    let test_cases = vec![
        ("What is AI?", "AI is artificial intelligence."),
        (
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance over time without being explicitly programmed. It uses algorithms to identify patterns and make decisions."
        ),
        (
            "Compare Python and JavaScript",
            "Python is specifically designed for general-purpose programming, while JavaScript is defined as a scripting language primarily for web development. Python has simpler syntax, whereas JavaScript excels in browser environments."
        ),
    ];

    let mut group = c.benchmark_group("quality_scoring");
    group.sample_size(500);

    for (query, answer) in test_cases {
        let id = BenchmarkId::new("score", answer.len());
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(id, &(query, answer), |b, (q, a)| {
            b.iter(|| scorer.score(black_box(q), black_box(a)))
        });
    }

    group.finish();
}

fn bench_embedding_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_generation");
    group.sample_size(200);

    for dim in [256, 384, 768, 1024] {
        let embedder = MockEmbedder::new(dim);
        let text = "This is a sample text for embedding generation benchmark.";

        let id = BenchmarkId::new("embed", dim);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(id, &embedder, |b, e| b.iter(|| e.embed(black_box(text))));
    }

    group.finish();
}

fn bench_memory_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_search");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));

    // Test with different store sizes
    for store_size in [100, 1000, 10000] {
        let mut store = MockMemoryStore::new(384);

        // Populate store
        for i in 0..store_size {
            store.add(&format!(
                "Memory entry {} about {} and {}",
                i,
                ["AI", "ML", "DL", "NLP", "CV"][i % 5],
                ["Python", "Rust", "JavaScript", "Go"][i % 4]
            ));
        }

        let query_embedding = store.embedder.embed("machine learning in Python");

        for top_k in [5, 10, 20] {
            let id = BenchmarkId::new(
                format!("search_{}k_top{}", store_size / 1000, top_k),
                store_size,
            );

            group.throughput(Throughput::Elements(top_k as u64));
            group.bench_with_input(id, &(store_size, top_k), |b, _| {
                b.iter(|| store.search(black_box(&query_embedding), top_k))
            });
        }
    }

    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    group.sample_size(1000);

    for dim in [256, 384, 768, 1024] {
        let mut rng = rand::thread_rng();
        let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let id = BenchmarkId::new("dim", dim);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(id, &(a, b), |bench, (a, b)| {
            bench.iter(|| cosine_similarity(black_box(a), black_box(b)))
        });
    }

    group.finish();
}

fn bench_batch_decomposition(c: &mut Criterion) {
    let decomposer = MockQueryDecomposer::new();
    let queries = generate_random_queries(100);

    let mut group = c.benchmark_group("batch_decomposition");
    group.sample_size(50);

    for batch_size in [10, 50, 100] {
        let batch: Vec<_> = queries.iter().take(batch_size).collect();

        let id = BenchmarkId::new("batch", batch_size);
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(id, &batch, |b, queries| {
            b.iter(|| {
                queries
                    .iter()
                    .map(|q| decomposer.decompose(q))
                    .collect::<Vec<_>>()
            })
        });
    }

    group.finish();
}

fn bench_end_to_end_cached(c: &mut Criterion) {
    let mut cache = MockCache::new(10000);
    let queries = generate_random_queries(1000);

    // Pre-populate cache
    for query in &queries {
        cache.insert(query, format!("Cached answer for: {}", query), 0.9);
    }

    let mut group = c.benchmark_group("e2e_cached");
    group.sample_size(200);

    group.bench_function("cached_lookup", |b| {
        let mut i = 0;
        b.iter(|| {
            let query = &queries[i % queries.len()];
            i += 1;
            cache.get(black_box(query))
        })
    });

    group.finish();
}

fn bench_end_to_end_uncached(c: &mut Criterion) {
    let decomposer = MockQueryDecomposer::new();
    let scorer = MockQualityScorer::new();
    let embedder = MockEmbedder::new(384);
    let mut store = MockMemoryStore::new(384);

    // Populate memory store
    for i in 0..1000 {
        store.add(&format!("Knowledge about topic {}", i));
    }

    let queries = generate_random_queries(100);

    let mut group = c.benchmark_group("e2e_uncached");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("full_pipeline", |b| {
        let mut i = 0;
        b.iter(|| {
            let query = &queries[i % queries.len()];
            i += 1;

            // 1. Decompose
            let decomp = decomposer.decompose(query);

            // 2. Generate embedding
            let embedding = embedder.embed(query);

            // 3. Search memory
            let results = store.search(&embedding, 5);

            // 4. Generate mock answer
            let answer = format!(
                "Answer for '{}' with {} sub-queries and {} memory results",
                query,
                decomp.sub_queries.len(),
                results.len()
            );

            // 5. Score quality
            let score = scorer.score(query, &answer);

            black_box((decomp, answer, score))
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    decomposition_benches,
    bench_query_decomposition,
    bench_complexity_scoring,
    bench_batch_decomposition,
);

criterion_group!(cache_benches, bench_cache_operations,);

criterion_group!(quality_benches, bench_quality_scoring,);

criterion_group!(
    embedding_benches,
    bench_embedding_generation,
    bench_cosine_similarity,
);

criterion_group!(memory_benches, bench_memory_search,);

criterion_group!(
    e2e_benches,
    bench_end_to_end_cached,
    bench_end_to_end_uncached,
);

// ============================================================================
// Pool Benchmarks
// ============================================================================

/// Mock vector pool for benchmarking (mirrors real implementation)
struct MockVectorPool {
    pool: std::sync::Mutex<Vec<Vec<f32>>>,
    dimension: usize,
    max_pooled: usize,
}

impl MockVectorPool {
    fn new(dimension: usize, max_pooled: usize) -> Self {
        Self {
            pool: std::sync::Mutex::new(Vec::with_capacity(max_pooled)),
            dimension,
            max_pooled,
        }
    }

    fn acquire(&self) -> Vec<f32> {
        let mut guard = self.pool.lock().unwrap();
        guard
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.dimension))
    }

    fn release(&self, mut vec: Vec<f32>) {
        vec.clear();
        let mut guard = self.pool.lock().unwrap();
        if guard.len() < self.max_pooled {
            guard.push(vec);
        }
    }
}

/// Mock string pool for benchmarking
struct MockStringPool {
    pool: std::sync::Mutex<Vec<String>>,
    max_pooled: usize,
}

impl MockStringPool {
    fn new(max_pooled: usize) -> Self {
        Self {
            pool: std::sync::Mutex::new(Vec::with_capacity(max_pooled)),
            max_pooled,
        }
    }

    fn acquire(&self) -> String {
        let mut guard = self.pool.lock().unwrap();
        guard.pop().unwrap_or_else(|| String::with_capacity(256))
    }

    fn release(&self, mut s: String) {
        s.clear();
        let mut guard = self.pool.lock().unwrap();
        if guard.len() < self.max_pooled {
            guard.push(s);
        }
    }
}

fn bench_pool_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_vector_ops");
    group.sample_size(1000);

    // Benchmark: Pool acquire (cold - empty pool)
    {
        group.bench_function("vector_acquire_cold", |b| {
            let pool = MockVectorPool::new(384, 64);
            b.iter(|| {
                let vec = pool.acquire();
                black_box(vec)
            })
        });
    }

    // Benchmark: Pool acquire (warm - populated pool)
    {
        let pool = MockVectorPool::new(384, 64);
        // Pre-warm the pool
        for _ in 0..32 {
            let vec = Vec::with_capacity(384);
            pool.release(vec);
        }

        group.bench_function("vector_acquire_warm", |b| {
            b.iter(|| {
                let vec = pool.acquire();
                pool.release(vec);
                black_box(())
            })
        });
    }

    // Benchmark: Pool release
    {
        let pool = MockVectorPool::new(384, 64);

        group.bench_function("vector_release", |b| {
            b.iter_batched(
                || Vec::<f32>::with_capacity(384),
                |vec| pool.release(vec),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    // Benchmark: Acquire + fill + release cycle
    {
        let pool = MockVectorPool::new(384, 64);
        // Pre-warm
        for _ in 0..16 {
            pool.release(Vec::with_capacity(384));
        }

        group.bench_function("vector_cycle_384", |b| {
            b.iter(|| {
                let mut vec = pool.acquire();
                vec.resize(384, 0.0f32);
                for i in 0..384 {
                    vec[i] = i as f32 * 0.001;
                }
                pool.release(vec);
            })
        });
    }

    // Compare with direct allocation
    {
        group.bench_function("vector_direct_alloc_384", |b| {
            b.iter(|| {
                let mut vec = Vec::<f32>::with_capacity(384);
                vec.resize(384, 0.0f32);
                for i in 0..384 {
                    vec[i] = i as f32 * 0.001;
                }
                black_box(vec)
            })
        });
    }

    group.finish();
}

fn bench_pool_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_string_ops");
    group.sample_size(1000);

    // Benchmark: String pool acquire (cold)
    {
        group.bench_function("string_acquire_cold", |b| {
            let pool = MockStringPool::new(64);
            b.iter(|| {
                let s = pool.acquire();
                black_box(s)
            })
        });
    }

    // Benchmark: String pool acquire (warm)
    {
        let pool = MockStringPool::new(64);
        for _ in 0..32 {
            pool.release(String::with_capacity(256));
        }

        group.bench_function("string_acquire_warm", |b| {
            b.iter(|| {
                let s = pool.acquire();
                pool.release(s);
                black_box(())
            })
        });
    }

    // Benchmark: String cycle (typical query processing)
    {
        let pool = MockStringPool::new(64);
        for _ in 0..16 {
            pool.release(String::with_capacity(256));
        }

        group.bench_function("string_cycle_query", |b| {
            b.iter(|| {
                let mut s = pool.acquire();
                s.push_str("What is machine learning and how does it work?");
                let lower = s.to_lowercase();
                pool.release(s);
                black_box(lower)
            })
        });
    }

    // Compare with direct allocation
    {
        group.bench_function("string_direct_alloc_query", |b| {
            b.iter(|| {
                let mut s = String::with_capacity(256);
                s.push_str("What is machine learning and how does it work?");
                let lower = s.to_lowercase();
                black_box(lower)
            })
        });
    }

    group.finish();
}

fn bench_pool_vs_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_vs_direct");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(5));

    // Simulate embedding generation workflow with pool
    {
        let pool = MockVectorPool::new(384, 64);
        for _ in 0..32 {
            pool.release(Vec::with_capacity(384));
        }

        let embedder = MockEmbedder::new(384);
        let texts = vec![
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Natural language processing enables text understanding",
        ];

        group.bench_function("embedding_with_pool", |b| {
            let mut idx = 0;
            b.iter(|| {
                let text = texts[idx % texts.len()];
                idx += 1;

                // Use pooled vector for output
                let mut output = pool.acquire();
                let embedding = embedder.embed(text);
                output.extend_from_slice(&embedding);

                // Simulate using the embedding
                let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

                pool.release(output);
                black_box(norm)
            })
        });
    }

    // Simulate embedding generation workflow without pool
    {
        let embedder = MockEmbedder::new(384);
        let texts = vec![
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Natural language processing enables text understanding",
        ];

        group.bench_function("embedding_without_pool", |b| {
            let mut idx = 0;
            b.iter(|| {
                let text = texts[idx % texts.len()];
                idx += 1;

                let embedding = embedder.embed(text);
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

                black_box(norm)
            })
        });
    }

    // Simulate query decomposition workflow with string pool
    {
        let pool = MockStringPool::new(64);
        for _ in 0..16 {
            pool.release(String::with_capacity(256));
        }

        let decomposer = MockQueryDecomposer::new();
        let queries = generate_random_queries(50);

        group.bench_function("decomposition_with_pool", |b| {
            let mut idx = 0;
            b.iter(|| {
                let query = &queries[idx % queries.len()];
                idx += 1;

                // Use pooled string for intermediate processing
                let mut lower = pool.acquire();
                lower.push_str(&query.to_lowercase());

                let result = decomposer.decompose(query);

                pool.release(lower);
                black_box(result)
            })
        });
    }

    // Simulate query decomposition without pool
    {
        let decomposer = MockQueryDecomposer::new();
        let queries = generate_random_queries(50);

        group.bench_function("decomposition_without_pool", |b| {
            let mut idx = 0;
            b.iter(|| {
                let query = &queries[idx % queries.len()];
                idx += 1;

                let _lower = query.to_lowercase();
                let result = decomposer.decompose(query);

                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_pool_concurrent(c: &mut Criterion) {
    use std::sync::Arc;

    let mut group = c.benchmark_group("pool_concurrent");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // Concurrent vector pool access
    {
        group.bench_function("vector_pool_4_threads", |b| {
            b.iter_batched(
                || {
                    let pool = Arc::new(MockVectorPool::new(384, 128));
                    // Pre-warm
                    for _ in 0..64 {
                        pool.release(Vec::with_capacity(384));
                    }
                    pool
                },
                |pool| {
                    let handles: Vec<_> = (0..4)
                        .map(|_| {
                            let p = Arc::clone(&pool);
                            std::thread::spawn(move || {
                                for _ in 0..100 {
                                    let mut vec = p.acquire();
                                    vec.extend_from_slice(&[1.0; 384]);
                                    p.release(vec);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(
    pool_benches,
    bench_pool_vector_operations,
    bench_pool_string_operations,
    bench_pool_vs_direct,
    bench_pool_concurrent,
);

// ============================================================================
// Zero-Copy Cache Benchmarks
// ============================================================================

/// Mock zero-copy cache entry using Arc for shared ownership.
/// This demonstrates the performance benefit of Arc<QueryResult> vs cloning.
#[derive(Clone)]
struct MockArcCacheEntry {
    result: std::sync::Arc<MockQueryResult>,
    cached_at_secs: i64,
}

#[derive(Clone)]
struct MockQueryResult {
    id: String,
    text: String,
    confidence: f32,
    tokens_generated: usize,
    latency_ms: f64,
}

/// Traditional cache that clones on retrieval
struct CloneCache {
    entries: std::collections::HashMap<u64, MockQueryResult>,
}

impl CloneCache {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    fn insert(&mut self, key: u64, result: MockQueryResult) {
        self.entries.insert(key, result);
    }

    fn get(&self, key: &u64) -> Option<MockQueryResult> {
        self.entries.get(key).cloned() // Full clone
    }
}

/// Zero-copy cache using Arc<QueryResult>
struct ArcCache {
    entries: std::collections::HashMap<u64, MockArcCacheEntry>,
}

impl ArcCache {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    fn insert(&mut self, key: u64, result: MockQueryResult) {
        self.entries.insert(
            key,
            MockArcCacheEntry {
                result: std::sync::Arc::new(result),
                cached_at_secs: 0,
            },
        );
    }

    fn get(&self, key: &u64) -> Option<std::sync::Arc<MockQueryResult>> {
        self.entries
            .get(key)
            .map(|e| std::sync::Arc::clone(&e.result)) // Arc clone only
    }
}

fn create_mock_query_result(size: usize) -> MockQueryResult {
    MockQueryResult {
        id: format!("result-{}", size),
        text: "a".repeat(size), // Vary text size
        confidence: 0.95,
        tokens_generated: size / 4,
        latency_ms: 25.5,
    }
}

fn bench_zero_copy_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_cache");
    group.sample_size(1000);

    // Test with different response sizes
    for text_size in [100, 500, 2000, 10000] {
        // Benchmark: Clone cache retrieval
        {
            let mut cache = CloneCache::new();
            for i in 0..1000 {
                cache.insert(i, create_mock_query_result(text_size));
            }

            let id = BenchmarkId::new("clone_cache_get", text_size);
            group.bench_with_input(id, &text_size, |b, _| {
                let mut i: u64 = 0;
                b.iter(|| {
                    i = (i + 1) % 1000;
                    let result = cache.get(&i);
                    black_box(result)
                })
            });
        }

        // Benchmark: Arc cache retrieval (zero-copy)
        {
            let mut cache = ArcCache::new();
            for i in 0..1000 {
                cache.insert(i, create_mock_query_result(text_size));
            }

            let id = BenchmarkId::new("arc_cache_get", text_size);
            group.bench_with_input(id, &text_size, |b, _| {
                let mut i: u64 = 0;
                b.iter(|| {
                    i = (i + 1) % 1000;
                    let result = cache.get(&i);
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

fn bench_zero_copy_string_vs_arc(c: &mut Criterion) {
    use std::borrow::Cow;
    use std::sync::Arc;

    let mut group = c.benchmark_group("zero_copy_strings");
    group.sample_size(1000);

    // Test Arc<str> vs String clone
    {
        let text = "This is a reasonably long text that would be part of a query result, containing multiple sentences to simulate realistic response data.".to_string();
        let arc_text: Arc<str> = Arc::from(text.as_str());

        group.bench_function("string_clone", |b| {
            b.iter(|| {
                let cloned = text.clone();
                black_box(cloned)
            })
        });

        group.bench_function("arc_str_clone", |b| {
            b.iter(|| {
                let cloned = Arc::clone(&arc_text);
                black_box(cloned)
            })
        });
    }

    // Test Cow<str> for excerpt extraction
    {
        let short_text = "Short text";
        let long_text = "This is a much longer text that would need to be truncated when extracting an excerpt for display purposes in the search results.";

        group.bench_function("excerpt_short_cow", |b| {
            b.iter(|| {
                let excerpt: Cow<str> = if short_text.len() <= 50 {
                    Cow::Borrowed(short_text)
                } else {
                    Cow::Owned(short_text[..50].to_string())
                };
                black_box(excerpt)
            })
        });

        group.bench_function("excerpt_long_cow", |b| {
            b.iter(|| {
                let excerpt: Cow<str> = if long_text.len() <= 50 {
                    Cow::Borrowed(long_text)
                } else {
                    Cow::Owned(long_text[..50].to_string())
                };
                black_box(excerpt)
            })
        });

        group.bench_function("excerpt_short_direct", |b| {
            b.iter(|| {
                let excerpt = if short_text.len() <= 50 {
                    short_text.to_string()
                } else {
                    short_text[..50].to_string()
                };
                black_box(excerpt)
            })
        });

        group.bench_function("excerpt_long_direct", |b| {
            b.iter(|| {
                let excerpt = if long_text.len() <= 50 {
                    long_text.to_string()
                } else {
                    long_text[..50].to_string()
                };
                black_box(excerpt)
            })
        });
    }

    group.finish();
}

fn bench_zero_copy_embedding(c: &mut Criterion) {
    use std::sync::Arc;

    let mut group = c.benchmark_group("zero_copy_embedding");
    group.sample_size(500);

    for dim in [256, 384, 768] {
        // Create test data
        let embedding: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let arc_embedding: Arc<[f32]> = Arc::from(embedding.as_slice());

        // Benchmark: Vec<f32> clone
        {
            let id = BenchmarkId::new("vec_clone", dim);
            group.bench_with_input(id, &embedding, |b, e| {
                b.iter(|| {
                    let cloned = e.clone();
                    black_box(cloned)
                })
            });
        }

        // Benchmark: Arc<[f32]> clone (zero-copy)
        {
            let id = BenchmarkId::new("arc_clone", dim);
            group.bench_with_input(id, &arc_embedding, |b, e| {
                b.iter(|| {
                    let cloned = Arc::clone(e);
                    black_box(cloned)
                })
            });
        }
    }

    group.finish();
}

/// Benchmark simulating cache hit path with zero-copy vs clone
fn bench_cache_hit_path(c: &mut Criterion) {
    use std::sync::Arc;

    let mut group = c.benchmark_group("cache_hit_path");
    group.sample_size(500);

    // Simulate a realistic query result
    let result = MockQueryResult {
        id: "query-12345".to_string(),
        text: "The answer to your question involves several key concepts. First, you need to understand the fundamental principles of machine learning. Then, consider how neural networks process information through multiple layers of abstraction.".to_string(),
        confidence: 0.92,
        tokens_generated: 48,
        latency_ms: 156.7,
    };

    // Clone-based cache hit
    {
        let mut cache = CloneCache::new();
        cache.insert(42, result.clone());

        group.bench_function("cache_hit_clone", |b| {
            b.iter(|| {
                // Simulate cache lookup + return
                let cached = cache.get(&42).expect("entry should exist");
                // "Use" the result
                let _len = cached.text.len();
                let _conf = cached.confidence;
                black_box(cached)
            })
        });
    }

    // Arc-based cache hit (zero-copy)
    {
        let mut cache = ArcCache::new();
        cache.insert(42, result.clone());

        group.bench_function("cache_hit_arc", |b| {
            b.iter(|| {
                // Simulate cache lookup + return
                let cached = cache.get(&42).expect("entry should exist");
                // "Use" the result
                let _len = cached.text.len();
                let _conf = cached.confidence;
                black_box(cached)
            })
        });
    }

    // Measure the cost of Arc::try_unwrap path
    {
        let mut cache = ArcCache::new();
        cache.insert(42, result.clone());

        group.bench_function("cache_hit_arc_unwrap", |b| {
            b.iter(|| {
                let cached_arc = cache.get(&42).expect("entry should exist");
                // Simulate what the controller does: try_unwrap or clone
                let result = Arc::try_unwrap(cached_arc).unwrap_or_else(|arc| (*arc).clone());
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(
    zero_copy_benches,
    bench_zero_copy_cache,
    bench_zero_copy_string_vs_arc,
    bench_zero_copy_embedding,
    bench_cache_hit_path,
);

criterion_main!(
    decomposition_benches,
    cache_benches,
    quality_benches,
    embedding_benches,
    memory_benches,
    e2e_benches,
    pool_benches,
    zero_copy_benches,
);
