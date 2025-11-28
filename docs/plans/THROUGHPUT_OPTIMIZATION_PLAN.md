# RuVector Throughput Optimization GOAP Plan

**Proof ID:** TO-GOAP-2025-001
**Version:** 1.0
**Date:** 2025-11-28
**Planning Method:** Goal-Oriented Action Planning (GOAP)

---

## Executive Summary

This document outlines a systematic, goal-oriented action plan to optimize RuVector's throughput from **~490 QPS** (current 10K vector benchmark) to **5,000+ QPS** (10x improvement), achieving parity with state-of-the-art vector databases while maintaining **99%+ recall**.

### Current State vs Goal State

| Metric | Current State | Goal State | Gap |
|--------|--------------|------------|-----|
| **QPS (10K vectors)** | ~490 | 5,000+ | **10x** |
| **QPS (100K vectors)** | 15,000-22,000 | 50,000+ | **3x** |
| **Recall@10** | 95-100% | 99%+ | ✅ Maintained |
| **p99 Latency** | 1.6-2.0ms | <1.0ms | **2x** |
| **SOTA Comparison** | 20x gap | Competitive | Close gap |

### Optimization Strategy

Using GOAP methodology, we've identified **7 critical optimization paths** with clear dependencies, measurable outcomes, and risk mitigation:

1. **SIMD Optimization** (Expected: +200-300% QPS)
2. **Batch Query Processing** (Expected: +150-200% QPS)
3. **Memory Layout Optimization** (Expected: +50-100% QPS)
4. **Multi-threading Enhancement** (Expected: +100-200% QPS)
5. **Lock-free Data Structures** (Expected: +30-50% QPS)
6. **Distance Computation Optimization** (Expected: +20-40% QPS)
7. **Query Compilation/Specialization** (Expected: +50-80% QPS)

**Cumulative Expected Improvement:** 10-15x throughput increase

---

## GOAP State Analysis

### World State Definition

```rust
struct WorldState {
    // Performance Metrics
    qps_10k: f64,           // Current: 490, Goal: 5000+
    qps_100k: f64,          // Current: 22000, Goal: 50000+
    p99_latency_ms: f64,    // Current: 1.8, Goal: <1.0
    recall_at_10: f64,      // Current: 0.99, Goal: >=0.99

    // Implementation State
    simd_level: SimdLevel,               // Current: AVX2, Goal: AVX512
    batch_processing: bool,              // Current: false, Goal: true
    memory_layout: MemoryLayout,         // Current: SoA, Goal: Optimized SoA
    threading_model: ThreadingModel,     // Current: Rayon, Goal: Custom
    lockfree_structures: LockfreeLevel,  // Current: Partial, Goal: Full
    distance_optimization: DistOptLevel, // Current: SimSIMD, Goal: Specialized
    query_compilation: bool,             // Current: false, Goal: true

    // Resource Constraints
    cpu_cores: usize,       // Runtime detection
    cache_size_kb: usize,   // L1: 32KB, L2: 256KB, L3: 8MB (typical)
    memory_gb: usize,       // Available RAM

    // Quality Assurance
    benchmarks_passing: bool,
    regression_tests_passing: bool,
    production_validation: bool,
}
```

### Goal State

```rust
const GOAL_STATE: WorldState = WorldState {
    qps_10k: 5000.0,
    qps_100k: 50000.0,
    p99_latency_ms: 1.0,
    recall_at_10: 0.99,

    simd_level: SimdLevel::AVX512,
    batch_processing: true,
    memory_layout: MemoryLayout::OptimizedSoA,
    threading_model: ThreadingModel::CustomPooled,
    lockfree_structures: LockfreeLevel::Full,
    distance_optimization: DistOptLevel::Specialized,
    query_compilation: true,

    benchmarks_passing: true,
    regression_tests_passing: true,
    production_validation: true,
};
```

---

## Milestone 1: SIMD Optimization (Phase 1)

**Expected Impact:** +200-300% QPS
**Risk Level:** Medium
**Timeline:** 2-3 weeks
**Priority:** Critical Path

### Current State Analysis

**Existing SIMD Implementation:**
- Location: `crates/ruvector-core/src/simd_intrinsics.rs`, `src/distance.rs`
- Technology: SimSIMD library + custom AVX2 intrinsics
- Coverage: Euclidean, Cosine, Dot Product
- Performance: 22-167ns per distance calc (128-1536D)

**Limitations:**
- No AVX-512 support (2x throughput vs AVX2)
- No ARM NEON support (mobile/edge deployment)
- Horizontal sum inefficiency in AVX2 (transmute overhead)
- No FMA (Fused Multiply-Add) exploitation
- Scalar fallback for remainder elements

### Action Sequence

#### Action 1.1: AVX-512 Distance Kernels
**Preconditions:**
- CPU supports AVX-512 (check via `is_x86_feature_detected!`)
- Dimensions divisible by 16 (or handle remainder)

**Implementation:**
```rust
// File: crates/ruvector-core/src/simd_intrinsics_avx512.rs

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn euclidean_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm512_setzero_ps();

    // Process 16 floats at a time (2x AVX2)
    let chunks = len / 16;
    for i in 0..chunks {
        let idx = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm512_sub_ps(va, vb);

        // Use FMA for better performance: sum = sum + diff * diff
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    // Horizontal reduction (much faster on AVX-512)
    let result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    let mut remainder_sum = 0.0f32;
    for i in (chunks * 16)..len {
        let diff = a[i] - b[i];
        remainder_sum += diff * diff;
    }

    (result + remainder_sum).sqrt()
}
```

**Effects:**
- `simd_level = AVX512`
- `distance_compute_ns` reduced by 40-50%
- `qps` increased by 60-100%

**Success Criteria:**
- [ ] AVX-512 distance calculations 2x faster than AVX2
- [ ] Benchmark: Euclidean 128D < 15ns, 384D < 25ns, 768D < 50ns
- [ ] No regression in recall (<0.1% acceptable)
- [ ] Tests pass on both AVX-512 and non-AVX-512 hardware

#### Action 1.2: ARM NEON Support
**Preconditions:**
- ARM target architecture
- NEON feature available

**Implementation:**
```rust
// File: crates/ruvector-core/src/simd_intrinsics_neon.rs

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 floats at a time
    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let diff = vsubq_f32(va, vb);

        // FMA: sum = sum + diff * diff
        sum = vfmaq_f32(sum, diff, diff);
    }

    // Horizontal sum
    let sum_arr = [
        vgetq_lane_f32(sum, 0),
        vgetq_lane_f32(sum, 1),
        vgetq_lane_f32(sum, 2),
        vgetq_lane_f32(sum, 3),
    ];
    let mut total = sum_arr.iter().sum::<f32>();

    // Remainder
    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}
```

**Effects:**
- `arm_support = true`
- Mobile/edge deployment enabled

**Success Criteria:**
- [ ] NEON performance within 80% of x86 AVX2
- [ ] Cross-compilation working for ARM targets
- [ ] CI tests on ARM runners

#### Action 1.3: Optimize Horizontal Reductions
**Preconditions:**
- AVX-512 action completed

**Implementation:**
```rust
// Avoid expensive transmute, use SIMD horizontal operations

// AVX2 optimized horizontal sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Hadd: [a,b,c,d,e,f,g,h] -> [a+b, c+d, e+f, g+h, ...]
    let sum1 = _mm256_hadd_ps(v, v);
    let sum2 = _mm256_hadd_ps(sum1, sum1);

    // Extract high and low 128-bit lanes and add
    let low = _mm256_castps256_ps128(sum2);
    let high = _mm256_extractf128_ps(sum2, 1);
    let sum3 = _mm_add_ps(low, high);

    _mm_cvtss_f32(sum3)
}
```

**Effects:**
- Reduce horizontal sum overhead by 30-40%
- Small QPS improvement (5-10%)

**Success Criteria:**
- [ ] No std::mem::transmute usage in hot paths
- [ ] Horizontal sum < 2ns overhead

#### Action 1.4: Dynamic SIMD Dispatch
**Preconditions:**
- All SIMD kernels implemented (AVX2, AVX-512, NEON)

**Implementation:**
```rust
// File: crates/ruvector-core/src/distance.rs

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy)]
enum SimdCapability {
    AVX512,
    AVX2,
    NEON,
    Scalar,
}

static SIMD_CAPABILITY: Lazy<SimdCapability> = Lazy::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdCapability::AVX512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdCapability::AVX2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return SimdCapability::NEON;
        }
    }

    SimdCapability::Scalar
});

#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    match *SIMD_CAPABILITY {
        #[cfg(target_arch = "x86_64")]
        SimdCapability::AVX512 => unsafe { euclidean_distance_avx512(a, b) },
        #[cfg(target_arch = "x86_64")]
        SimdCapability::AVX2 => unsafe { euclidean_distance_avx2(a, b) },
        #[cfg(target_arch = "aarch64")]
        SimdCapability::NEON => unsafe { euclidean_distance_neon(a, b) },
        SimdCapability::Scalar => euclidean_distance_scalar(a, b),
    }
}
```

**Effects:**
- Zero-overhead runtime dispatch
- Optimal path selection per CPU

**Success Criteria:**
- [ ] Dispatch overhead < 1ns
- [ ] Correct fallback on all platforms

### Testing & Validation

```bash
# Benchmark suite
cargo bench --bench distance_metrics -- --save-baseline simd-opt

# Verify correctness
cargo test --release simd_intrinsics

# Cross-platform validation
cargo test --target x86_64-unknown-linux-gnu
cargo test --target aarch64-unknown-linux-gnu

# Performance regression check
cargo bench --bench distance_metrics -- --baseline simd-opt --test
```

### Success Metrics

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| Euclidean 128D | 25ns | <15ns | ✅ 40% faster |
| Euclidean 384D | 47ns | <25ns | ✅ 47% faster |
| Euclidean 768D | 90ns | <45ns | ✅ 50% faster |
| Cosine 384D | 42ns | <22ns | ✅ 48% faster |
| QPS Improvement | - | +60-100% | Latency benchmark |
| Recall Regression | - | <0.1% | Integration tests |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AVX-512 not available | Medium | High | Graceful fallback to AVX2 |
| Correctness bugs | Low | Critical | Extensive fuzzing, property tests |
| Precision loss | Low | Medium | f64 accumulators where needed |
| Platform fragmentation | Medium | Medium | Comprehensive CI matrix |

### Dependencies for Next Milestone

✅ SIMD optimization must complete before:
- Batch query processing (uses SIMD kernels)
- Distance computation specialization (builds on SIMD)

---

## Milestone 2: Batch Query Processing (Phase 1)

**Expected Impact:** +150-200% QPS
**Risk Level:** Medium-High
**Timeline:** 2-3 weeks
**Priority:** Critical Path

### Current State Analysis

**Existing Implementation:**
- HNSW searches processed one-at-a-time
- No query batching infrastructure
- Potential for vectorization across queries

**Opportunity:**
- Process 8-16 queries simultaneously
- Amortize graph traversal overhead
- Better CPU cache utilization

### Action Sequence

#### Action 2.1: Batch Distance Computation

**Preconditions:**
- SIMD optimization (Milestone 1) completed
- Cache-optimized SoA storage exists

**Implementation:**
```rust
// File: crates/ruvector-core/src/batch_search.rs

use rayon::prelude::*;
use crate::cache_optimized::SoAVectorStorage;

/// Batch distance computation with SIMD
pub struct BatchDistanceComputer {
    storage: Arc<SoAVectorStorage>,
    batch_size: usize,
}

impl BatchDistanceComputer {
    /// Compute distances from multiple queries to all vectors
    ///
    /// # Layout
    /// queries: [Q0, Q1, Q2, ..., Q7] (batch_size queries)
    /// vectors: [V0, V1, V2, ..., VN] (N vectors in database)
    ///
    /// Output: [batch_size x N] distance matrix
    pub fn compute_batch_distances(
        &self,
        queries: &[Vec<f32>],  // batch_size x dimensions
        candidate_ids: &[usize],
    ) -> Vec<Vec<f32>> {
        let batch_size = queries.len();
        let num_candidates = candidate_ids.len();

        // Pre-allocate output matrix
        let mut distances = vec![vec![0.0f32; num_candidates]; batch_size];

        // Process dimension-by-dimension for cache efficiency
        for dim_idx in 0..self.storage.dimensions() {
            let dim_slice = self.storage.dimension_slice(dim_idx);

            // For each query in the batch
            for (query_idx, query) in queries.iter().enumerate() {
                let query_val = query[dim_idx];

                // Vectorize over candidates (SIMD friendly)
                for (cand_idx, &vec_id) in candidate_ids.iter().enumerate() {
                    let diff = dim_slice[vec_id] - query_val;
                    distances[query_idx][cand_idx] += diff * diff;
                }
            }
        }

        // Take square roots
        distances.par_iter_mut().for_each(|row| {
            for dist in row.iter_mut() {
                *dist = dist.sqrt();
            }
        });

        distances
    }
}
```

**Effects:**
- `batch_processing = true`
- Amortized cache misses
- Better instruction pipeline utilization

**Success Criteria:**
- [ ] Batch computation 2-3x faster than sequential
- [ ] Scalable to 8-16 queries per batch
- [ ] Memory overhead <10% vs sequential

#### Action 2.2: Parallel HNSW Traversal

**Preconditions:**
- Batch distance computation working
- HNSW graph structure is thread-safe (DashMap)

**Implementation:**
```rust
// File: crates/ruvector-core/src/batch_search.rs

use crossbeam::channel::{bounded, Sender, Receiver};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct BatchHNSWSearch {
    index: Arc<HnswIndex>,
    thread_pool: rayon::ThreadPool,
    batch_size: usize,
}

impl BatchHNSWSearch {
    /// Search multiple queries in parallel with work stealing
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: usize,
    ) -> Vec<Vec<SearchResult>> {
        let num_queries = queries.len();

        // Use rayon's parallel iterator for work stealing
        queries
            .par_iter()
            .map(|query| {
                self.index.search_with_ef(query, k, ef_search)
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Optimized batch search with candidate pooling
    pub fn search_batch_optimized(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: usize,
    ) -> Vec<Vec<SearchResult>> {
        let num_queries = queries.len();

        // Phase 1: Find candidate pools for each query (coarse search)
        let candidate_pools: Vec<Vec<usize>> = queries
            .par_iter()
            .map(|query| {
                // Quick navigation to candidate region
                self.find_candidate_pool(query, ef_search * 2)
            })
            .collect();

        // Phase 2: Batch distance computation within candidate pools
        let distance_matrices: Vec<Vec<(usize, f32)>> = queries
            .par_iter()
            .zip(&candidate_pools)
            .map(|(query, candidates)| {
                // Compute distances to all candidates in batch
                self.compute_distances_to_candidates(query, candidates)
            })
            .collect();

        // Phase 3: Top-k selection per query
        distance_matrices
            .into_par_iter()
            .map(|mut distances| {
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                distances.truncate(k);

                distances
                    .into_iter()
                    .map(|(id, dist)| SearchResult {
                        id: id.to_string(),
                        score: dist,
                        vector: None,
                        metadata: None,
                    })
                    .collect()
            })
            .collect()
    }
}
```

**Effects:**
- Parallel query processing
- Work stealing for load balancing
- Candidate pool reuse

**Success Criteria:**
- [ ] Linear scaling up to CPU core count
- [ ] 8-thread: 6-7x throughput vs single-threaded
- [ ] 16-thread: 10-14x throughput vs single-threaded

#### Action 2.3: Query Grouping by Similarity

**Preconditions:**
- Batch HNSW search working
- Performance profiling showing cache benefits

**Implementation:**
```rust
/// Group similar queries together for better cache locality
pub fn group_queries_by_similarity(
    queries: &[Vec<f32>],
    num_groups: usize,
) -> Vec<Vec<usize>> {
    // Use simple clustering (k-means) to group queries
    // Queries in same group likely visit similar graph regions

    let mut groups = vec![Vec::new(); num_groups];

    // Simple hash-based grouping for speed
    for (idx, query) in queries.iter().enumerate() {
        let hash = compute_query_hash(query);
        let group_id = hash % num_groups;
        groups[group_id].push(idx);
    }

    groups
}

fn compute_query_hash(query: &[f32]) -> usize {
    // Use first few dimensions for quick hashing
    let mut hash = 0usize;
    for (i, &val) in query.iter().take(8).enumerate() {
        hash ^= (val.to_bits() as usize).wrapping_mul(0x9e3779b9) >> i;
    }
    hash
}
```

**Effects:**
- Better cache locality
- Reduced graph traversal overhead
- 10-20% additional throughput

**Success Criteria:**
- [ ] Grouped batches 10-20% faster than random batches
- [ ] Low grouping overhead (<5% of total time)

### Testing & Validation

```bash
# Batch processing benchmark
cargo bench --bench batch_operations -- --save-baseline batch-opt

# Correctness verification
cargo test batch_search::tests

# Scalability test
for threads in 1 2 4 8 16; do
    cargo bench --features batch -- --threads $threads
done

# Memory overhead check
valgrind --tool=massif cargo bench batch_search
```

### Success Metrics

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| Batch-8 Throughput | 490 QPS | 1,200+ QPS | ✅ 2.4x |
| Batch-16 Throughput | 490 QPS | 1,500+ QPS | ✅ 3x |
| Parallel-8-threads | 3,920 QPS | 9,800+ QPS | ✅ 2.5x |
| Memory Overhead | - | <10% | Memory profiler |
| Recall Regression | - | <0.1% | Integration tests |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Thread contention | Medium | Medium | Lock-free structures (Milestone 5) |
| Memory overhead | Medium | Medium | Pooled allocations |
| Load imbalance | Low | Low | Work stealing scheduler |
| Cache thrashing | Low | Medium | Query grouping optimization |

---

## Milestone 3: Memory Layout Optimization (Phase 1)

**Expected Impact:** +50-100% QPS
**Risk Level:** Low-Medium
**Timeline:** 1-2 weeks
**Priority:** Medium

### Current State Analysis

**Existing Implementation:**
- Location: `crates/ruvector-core/src/cache_optimized.rs`
- Structure-of-Arrays (SoA) layout implemented
- 64-byte cache line alignment
- Dimension-wise storage for SIMD

**Opportunity:**
- Further prefetch optimization
- NUMA-aware allocation
- Huge pages support

### Action Sequence

#### Action 3.1: Software Prefetching

**Preconditions:**
- SoA storage in use
- Batch query processing working

**Implementation:**
```rust
// File: crates/ruvector-core/src/cache_optimized.rs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl SoAVectorStorage {
    /// Batch distance computation with manual prefetching
    pub fn batch_distances_with_prefetch(
        &self,
        query: &[f32],
        candidate_ids: &[usize],
        output: &mut [f32],
    ) {
        assert_eq!(candidate_ids.len(), output.len());

        // Initialize output
        output.fill(0.0);

        // Prefetch distance (in cache lines)
        const PREFETCH_DISTANCE: usize = 8;

        // Process dimension by dimension
        for dim_idx in 0..self.dimensions {
            let dim_slice = self.dimension_slice(dim_idx);
            let query_val = query[dim_idx];

            for (i, &vec_id) in candidate_ids.iter().enumerate() {
                // Prefetch ahead
                if i + PREFETCH_DISTANCE < candidate_ids.len() {
                    let next_id = candidate_ids[i + PREFETCH_DISTANCE];
                    let next_ptr = unsafe {
                        dim_slice.as_ptr().add(next_id)
                    };

                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        _mm_prefetch(next_ptr as *const i8, _MM_HINT_T0);
                    }
                }

                // Compute distance contribution
                let diff = dim_slice[vec_id] - query_val;
                output[i] += diff * diff;
            }
        }

        // Take square roots
        for dist in output.iter_mut() {
            *dist = dist.sqrt();
        }
    }
}
```

**Effects:**
- Reduced cache miss latency
- 15-30% throughput improvement

**Success Criteria:**
- [ ] Cache miss rate reduced by 20-30%
- [ ] Prefetch overhead <5% of total time
- [ ] Measurable with `perf stat -e cache-misses`

#### Action 3.2: Huge Pages Support

**Preconditions:**
- Linux huge pages configured (`/proc/sys/vm/nr_hugepages`)

**Implementation:**
```rust
// File: crates/ruvector-core/src/cache_optimized.rs

use std::alloc::{alloc, dealloc, Layout};
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB

impl SoAVectorStorage {
    /// Create storage using huge pages for large datasets
    pub fn new_with_huge_pages(
        dimensions: usize,
        initial_capacity: usize,
    ) -> Result<Self> {
        let total_elements = dimensions * initial_capacity;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        // Round up to huge page boundary
        let aligned_size = (total_bytes + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE;

        // Try to allocate with huge pages
        let data = unsafe {
            #[cfg(target_os = "linux")]
            {
                use libc::{mmap, MAP_ANONYMOUS, MAP_HUGETLB, MAP_PRIVATE, PROT_READ, PROT_WRITE};

                let ptr = mmap(
                    std::ptr::null_mut(),
                    aligned_size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1,
                    0,
                );

                if ptr == libc::MAP_FAILED {
                    // Fallback to regular pages
                    eprintln!("Huge pages not available, using regular pages");
                    alloc(Layout::from_size_align(total_bytes, 64).unwrap()) as *mut f32
                } else {
                    ptr as *mut f32
                }
            }

            #[cfg(not(target_os = "linux"))]
            {
                alloc(Layout::from_size_align(total_bytes, 64).unwrap()) as *mut f32
            }
        };

        Ok(Self {
            count: 0,
            dimensions,
            capacity: initial_capacity,
            data,
        })
    }
}
```

**Effects:**
- Reduced TLB misses
- 5-15% throughput improvement on large datasets

**Success Criteria:**
- [ ] TLB miss rate reduced by 50%+
- [ ] Performance gain on 1M+ vector datasets
- [ ] Graceful fallback when huge pages unavailable

#### Action 3.3: NUMA-Aware Allocation

**Preconditions:**
- Multi-socket system available for testing

**Implementation:**
```rust
// File: crates/ruvector-core/src/numa_optimized.rs

#[cfg(target_os = "linux")]
use libc::{numa_available, numa_alloc_onnode, numa_free};

pub struct NumaAwareStorage {
    nodes: Vec<SoAVectorStorage>,
    node_count: usize,
}

impl NumaAwareStorage {
    /// Create NUMA-aware storage with data distributed across nodes
    pub fn new(dimensions: usize, total_capacity: usize) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let node_count = unsafe { numa_num_configured_nodes() };

            if node_count > 1 {
                // Split capacity across NUMA nodes
                let capacity_per_node = total_capacity / node_count;

                let nodes = (0..node_count)
                    .map(|node_id| {
                        // Allocate on specific NUMA node
                        allocate_on_node(dimensions, capacity_per_node, node_id)
                    })
                    .collect::<Result<Vec<_>>>()?;

                return Ok(Self { nodes, node_count });
            }
        }

        // Fallback to single node
        Ok(Self {
            nodes: vec![SoAVectorStorage::new(dimensions, total_capacity)],
            node_count: 1,
        })
    }

    /// Search with NUMA-aware parallelism
    pub fn search_numa_aware(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<SearchResult> {
        // Search each NUMA node in parallel (thread affinity to node)
        let node_results: Vec<Vec<SearchResult>> = self.nodes
            .par_iter()
            .enumerate()
            .map(|(node_id, storage)| {
                // Pin thread to NUMA node
                #[cfg(target_os = "linux")]
                set_thread_affinity_to_node(node_id);

                storage.search_local(query, k)
            })
            .collect();

        // Merge results from all nodes
        merge_search_results(node_results, k)
    }
}
```

**Effects:**
- Reduced remote memory access on multi-socket systems
- 20-40% throughput on NUMA systems

**Success Criteria:**
- [ ] NUMA-local access >90% of total
- [ ] Performance gain proportional to node count
- [ ] `numactl --hardware` shows balanced allocation

### Testing & Validation

```bash
# Prefetch effectiveness
perf stat -e cache-misses,cache-references cargo bench memory_layout

# Huge pages validation
echo 512 | sudo tee /proc/sys/vm/nr_hugepages
cargo bench --features huge-pages

# NUMA awareness
numactl --cpunodebind=0 --membind=0 cargo bench numa_aware
```

### Success Metrics

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| L1 Cache Miss Rate | 15% | <10% | perf stat |
| L3 Cache Miss Rate | 25% | <18% | perf stat |
| TLB Miss Rate | 2% | <1% | perf stat |
| NUMA Remote Access | 40% | <10% | numastat |
| QPS Improvement | - | +50-100% | Benchmark |

---

## Milestone 4: Multi-threading Enhancement (Phase 1)

**Expected Impact:** +100-200% QPS
**Risk Level:** Medium
**Timeline:** 2 weeks
**Priority:** High

### Current State Analysis

**Existing Implementation:**
- Uses Rayon for parallel iteration
- DashMap for concurrent access
- RwLock for HNSW index

**Opportunity:**
- Custom thread pool with work stealing
- Thread-local caching
- Reduced synchronization overhead

### Action Sequence

#### Action 4.1: Custom Thread Pool

**Preconditions:**
- Batch query processing implemented
- Profiling shows Rayon overhead

**Implementation:**
```rust
// File: crates/ruvector-core/src/threading/work_stealing_pool.rs

use crossbeam::deque::{Injector, Stealer, Worker};
use crossbeam::channel::{bounded, Sender, Receiver};
use std::sync::Arc;
use std::thread;

pub struct WorkStealingThreadPool {
    workers: Vec<WorkerThread>,
    injector: Arc<Injector<Task>>,
    stealers: Vec<Stealer<Task>>,
}

type Task = Box<dyn FnOnce() + Send + 'static>;

struct WorkerThread {
    thread: Option<thread::JoinHandle<()>>,
}

impl WorkStealingThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let injector = Arc::new(Injector::new());
        let mut stealers = Vec::new();
        let mut workers = Vec::new();

        for id in 0..num_threads {
            let worker_queue = Worker::new_fifo();
            stealers.push(worker_queue.stealer());

            let injector_clone = Arc::clone(&injector);
            let stealers_clone = stealers.clone();

            let thread = thread::Builder::new()
                .name(format!("worker-{}", id))
                .spawn(move || {
                    worker_main(id, worker_queue, injector_clone, stealers_clone);
                })
                .unwrap();

            workers.push(WorkerThread { thread: Some(thread) });
        }

        Self { workers, injector: Arc::new(injector), stealers }
    }

    pub fn execute<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.injector.push(Box::new(task));
    }

    pub fn scope<'env, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Scope<'env>) -> R + Send,
        R: Send,
    {
        // Scoped task execution with join semantics
        let scope = Scope::new(self);
        f(&scope)
    }
}

fn worker_main(
    id: usize,
    worker: Worker<Task>,
    injector: Arc<Injector<Task>>,
    stealers: Vec<Stealer<Task>>,
) {
    // Set CPU affinity for this worker
    #[cfg(target_os = "linux")]
    set_cpu_affinity(id);

    loop {
        // Try local queue first
        let task = worker.pop()
            .or_else(|| {
                // Steal from global injector
                loop {
                    match injector.steal_batch_and_pop(&worker) {
                        crossbeam::deque::Steal::Success(t) => return Some(t),
                        crossbeam::deque::Steal::Empty => break,
                        crossbeam::deque::Steal::Retry => continue,
                    }
                }

                // Try stealing from other workers
                stealers.iter()
                    .filter(|s| !s.is_empty())
                    .find_map(|s| s.steal().success())
            });

        match task {
            Some(t) => t(),
            None => {
                // No work available, backoff
                std::hint::spin_loop();
                thread::yield_now();
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn set_cpu_affinity(cpu_id: usize) {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

    unsafe {
        let mut cpuset: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        CPU_SET(cpu_id, &mut cpuset);

        sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpuset);
    }
}
```

**Effects:**
- Reduced thread pool overhead
- Better CPU cache affinity
- 15-25% throughput improvement

**Success Criteria:**
- [ ] Task dispatch overhead <100ns
- [ ] Linear scaling up to 80% of core count
- [ ] No thread starvation under load

#### Action 4.2: Thread-Local Caching

**Preconditions:**
- Custom thread pool in use

**Implementation:**
```rust
// File: crates/ruvector-core/src/threading/thread_local_cache.rs

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Thread-local cache for distance computations
    static DISTANCE_CACHE: RefCell<DistanceCache> = RefCell::new(DistanceCache::new());

    /// Thread-local buffer pool for scratch space
    static BUFFER_POOL: RefCell<BufferPool> = RefCell::new(BufferPool::new());
}

struct DistanceCache {
    cache: HashMap<(usize, usize), f32>,
    capacity: usize,
}

impl DistanceCache {
    fn new() -> Self {
        Self {
            cache: HashMap::with_capacity(1024),
            capacity: 1024,
        }
    }

    fn get_or_compute<F>(
        &mut self,
        id_a: usize,
        id_b: usize,
        compute_fn: F,
    ) -> f32
    where
        F: FnOnce() -> f32,
    {
        let key = if id_a < id_b { (id_a, id_b) } else { (id_b, id_a) };

        *self.cache.entry(key).or_insert_with(compute_fn)
    }

    fn clear(&mut self) {
        if self.cache.len() > self.capacity {
            self.cache.clear();
        }
    }
}

struct BufferPool {
    buffers: Vec<Vec<f32>>,
    max_size: usize,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            buffers: Vec::new(),
            max_size: 16,
        }
    }

    fn acquire(&mut self, size: usize) -> Vec<f32> {
        self.buffers.pop().unwrap_or_else(|| Vec::with_capacity(size))
    }

    fn release(&mut self, mut buffer: Vec<f32>) {
        if self.buffers.len() < self.max_size {
            buffer.clear();
            self.buffers.push(buffer);
        }
    }
}

/// Use thread-local cache in search
pub fn search_with_cache(query: &[f32], candidates: &[usize]) -> Vec<f32> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let mut distances = pool.acquire(candidates.len());
        distances.resize(candidates.len(), 0.0);

        // Compute distances
        for (i, &cand_id) in candidates.iter().enumerate() {
            distances[i] = compute_distance(query, cand_id);
        }

        let result = distances.clone();
        pool.release(distances);

        result
    })
}
```

**Effects:**
- Reduced allocation overhead
- Better cache hit rates
- 10-20% throughput improvement

**Success Criteria:**
- [ ] Allocation rate reduced by 50%+
- [ ] Cache hit rate >30% on similar queries
- [ ] Memory overhead per thread <1MB

#### Action 4.3: Lock-Free HNSW Graph Access

**Preconditions:**
- Understanding of HNSW read access patterns
- Lock contention identified in profiling

**Implementation:**
```rust
// File: crates/ruvector-core/src/index/hnsw_lockfree.rs

use dashmap::DashMap;
use crossbeam::epoch::{self, Atomic, Owned, Shared};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct LockFreeHNSWGraph {
    /// Adjacency lists stored in lock-free map
    /// Key: node_id, Value: list of neighbor IDs
    neighbors: DashMap<usize, Vec<usize>>,

    /// Node levels (read-only after construction)
    levels: Vec<AtomicUsize>,

    /// Entry point (updated rarely with CAS)
    entry_point: AtomicUsize,
}

impl LockFreeHNSWGraph {
    /// Read neighbors without locking
    pub fn get_neighbors(&self, node_id: usize, level: usize) -> Vec<usize> {
        let key = self.encode_key(node_id, level);

        self.neighbors.get(&key)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }

    /// Search without write locks (read-only traversal)
    pub fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        num_to_return: usize,
        level: usize,
    ) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();
        let mut best = std::collections::BinaryHeap::new();

        // Initialize
        for &ep in entry_points {
            let dist = self.compute_distance(query, ep);
            candidates.push((-dist, ep));
            best.push((dist, ep));
            visited.insert(ep);
        }

        // Greedy search
        while let Some((-curr_dist, curr_id)) = candidates.pop() {
            if curr_dist > best.peek().map(|&(d, _)| d).unwrap_or(f32::INFINITY) {
                break;
            }

            // Read neighbors without locking
            let neighbors = self.get_neighbors(curr_id, level);

            for &neighbor_id in &neighbors {
                if visited.insert(neighbor_id) {
                    let dist = self.compute_distance(query, neighbor_id);

                    if dist < best.peek().map(|&(d, _)| d).unwrap_or(f32::INFINITY)
                        || best.len() < num_to_return {
                        candidates.push((-dist, neighbor_id));
                        best.push((dist, neighbor_id));

                        if best.len() > num_to_return {
                            best.pop();
                        }
                    }
                }
            }
        }

        best.into_iter().map(|(_, id)| id).collect()
    }
}
```

**Effects:**
- Eliminated read lock contention
- 40-60% throughput improvement on multi-threaded workloads

**Success Criteria:**
- [ ] No lock contention in read-heavy workloads
- [ ] 16-thread scaling efficiency >85%
- [ ] Correctness maintained (same results as locked version)

### Testing & Validation

```bash
# Thread scaling benchmark
for threads in 1 2 4 8 16 32; do
    cargo bench --bench threading -- --threads $threads
done

# Lock contention analysis
perf record -e contention cargo bench threading
perf report

# Thread-local cache effectiveness
cargo bench --features thread-local-cache

# Correctness under concurrency
cargo test --release --features stress-test threading::stress
```

### Success Metrics

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| 8-thread Scaling | 5x | 7x | Linear scaling |
| 16-thread Scaling | 8x | 13x | 80%+ efficiency |
| Lock Contention | High | None | perf analysis |
| Task Dispatch Latency | 500ns | <100ns | Microbenchmark |
| Thread-local Cache Hit | 0% | 30%+ | Instrumentation |

---

## Milestone 5: Lock-Free Data Structures (Phase 2)

**Expected Impact:** +30-50% QPS
**Risk Level:** Low
**Timeline:** 1 week
**Priority:** Medium

### Current State Analysis

**Existing Implementation:**
- Some lock-free structures in `crates/ruvector-core/src/lockfree.rs`
- LockFreeCounter, LockFreeStats, ObjectPool
- DashMap for concurrent hash maps

**Opportunity:**
- Replace remaining locks with lock-free alternatives
- Optimize hot paths with relaxed ordering

### Action Sequence

#### Action 5.1: Lock-Free Result Buffer Pool

**Implementation:**
```rust
// File: crates/ruvector-core/src/lockfree_pool.rs

use crossbeam::queue::SegQueue;
use std::sync::Arc;

/// Lock-free object pool for search result buffers
pub struct LockFreeResultPool<T> {
    queue: Arc<SegQueue<T>>,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
    current_size: AtomicUsize,
}

impl<T: Send> LockFreeResultPool<T> {
    pub fn new<F>(max_size: usize, factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            queue: Arc::new(SegQueue::new()),
            factory: Arc::new(factory),
            max_size,
            current_size: AtomicUsize::new(0),
        }
    }

    pub fn acquire(&self) -> PooledObject<T> {
        let object = self.queue.pop().unwrap_or_else(|| {
            let size = self.current_size.fetch_add(1, Ordering::Relaxed);
            if size < self.max_size {
                (self.factory)()
            } else {
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                // Spin until an object is available
                loop {
                    if let Some(obj) = self.queue.pop() {
                        break obj;
                    }
                    std::hint::spin_loop();
                }
            }
        });

        PooledObject {
            object: Some(object),
            pool: Arc::clone(&self.queue),
        }
    }
}
```

**Effects:**
- No lock contention for buffer allocation
- 5-10% throughput improvement

#### Action 5.2: Optimized Atomic Ordering

**Implementation:**
```rust
// Replace Ordering::SeqCst with Relaxed where possible

// BEFORE: Expensive sequential consistency
counter.fetch_add(1, Ordering::SeqCst);

// AFTER: Relaxed ordering (sufficient for counters)
counter.fetch_add(1, Ordering::Relaxed);

// Stats collection with relaxed ordering
impl LockFreeStats {
    #[inline]
    pub fn record_query(&self, latency_ns: u64) {
        // Relaxed is fine - we don't need ordering guarantees
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
    }
}
```

**Effects:**
- Reduced memory fence overhead
- 3-5% throughput improvement

### Success Metrics

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| Pool Allocation Latency | 200ns | <50ns | Microbenchmark |
| Lock Contention | Medium | None | perf analysis |
| QPS Improvement | - | +30-50% | Integration bench |

---

## Milestone 6: Distance Computation Specialization (Phase 2)

**Expected Impact:** +20-40% QPS
**Risk Level:** Low
**Timeline:** 1 week
**Priority:** Low

### Action Sequence

#### Action 6.1: Dimension-Specific Kernels

**Implementation:**
```rust
// File: crates/ruvector-core/src/distance_specialized.rs

/// Generate specialized kernels for common dimensions
macro_rules! generate_distance_kernel {
    ($dim:expr) => {
        paste::paste! {
            #[target_feature(enable = "avx512f")]
            unsafe fn [<euclidean_distance_ $dim>](a: &[f32], b: &[f32]) -> f32 {
                debug_assert_eq!(a.len(), $dim);
                debug_assert_eq!(b.len(), $dim);

                // Fully unrolled loop for this dimension
                let mut sum = _mm512_setzero_ps();

                $(
                    let idx = $i * 16;
                    let va = _mm512_loadu_ps(a.as_ptr().add(idx));
                    let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
                    let diff = _mm512_sub_ps(va, vb);
                    sum = _mm512_fmadd_ps(diff, diff, sum);
                )*

                _mm512_reduce_add_ps(sum).sqrt()
            }
        }
    };
}

// Generate specialized kernels
generate_distance_kernel!(128);  // Common: sentence embeddings
generate_distance_kernel!(384);  // Common: MiniLM
generate_distance_kernel!(768);  // Common: BERT
generate_distance_kernel!(1536); // Common: OpenAI

/// Dispatch to specialized kernel
#[inline]
pub fn euclidean_distance_optimized(a: &[f32], b: &[f32]) -> f32 {
    match a.len() {
        128 => unsafe { euclidean_distance_128(a, b) },
        384 => unsafe { euclidean_distance_384(a, b) },
        768 => unsafe { euclidean_distance_768(a, b) },
        1536 => unsafe { euclidean_distance_1536(a, b) },
        _ => euclidean_distance_generic(a, b),
    }
}
```

**Effects:**
- Optimized instruction scheduling
- 10-20% faster for common dimensions

#### Action 6.2: Approximate Distance Early Termination

**Implementation:**
```rust
/// Early termination for distance computation
#[inline]
pub fn euclidean_distance_with_threshold(
    a: &[f32],
    b: &[f32],
    threshold: f32,
) -> Option<f32> {
    let threshold_sq = threshold * threshold;
    let mut sum = 0.0f32;

    // Process in chunks, check threshold periodically
    const CHUNK_SIZE: usize = 64;

    for chunk_start in (0..a.len()).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(a.len());

        // Compute chunk distance
        for i in chunk_start..chunk_end {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }

        // Early exit if already exceeded threshold
        if sum > threshold_sq {
            return None;
        }
    }

    Some(sum.sqrt())
}
```

**Effects:**
- Skip expensive computations for far candidates
- 10-20% throughput in HNSW search

### Success Metrics

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| Distance Compute (128D) | 15ns | <12ns | Specialized kernel |
| Distance Compute (384D) | 25ns | <20ns | Specialized kernel |
| Early Termination Rate | 0% | 30%+ | HNSW search |
| QPS Improvement | - | +20-40% | Integration bench |

---

## Milestone 7: Query Compilation & Specialization (Phase 3)

**Expected Impact:** +50-80% QPS
**Risk Level:** High
**Timeline:** 3-4 weeks
**Priority:** Low (future work)

### Concept

Generate specialized query execution code at runtime based on:
- Query pattern (sparse vs dense)
- Distance metric
- Dimension count
- Filter complexity

### Implementation Strategy

```rust
// File: crates/ruvector-core/src/query_compiler.rs

use cranelift::prelude::*;

pub struct QueryCompiler {
    builder: FunctionBuilder,
    module: JITModule,
}

impl QueryCompiler {
    /// Compile a specialized search function
    pub fn compile_search_fn(
        &mut self,
        metric: DistanceMetric,
        dimensions: usize,
        filter: Option<&FilterExpr>,
    ) -> CompiledSearchFn {
        // Generate specialized LLVM IR / Cranelift IR
        // for this specific query pattern

        // 1. Inline distance computation
        // 2. Inline filter evaluation
        // 3. Unroll loops for known dimensions
        // 4. Optimize register allocation

        // Return function pointer to compiled code
        todo!()
    }
}
```

**Effects:**
- Eliminate function call overhead
- Optimal instruction scheduling
- 50-80% throughput for hot queries

**Success Criteria:**
- [ ] Compilation time <100ms
- [ ] Compiled code 50%+ faster than interpreted
- [ ] Cache compiled queries for reuse

---

## Overall Success Criteria & Validation

### Performance Targets

| Metric | Current | Milestone 3 | Milestone 5 | Final Goal | Status |
|--------|---------|-------------|-------------|------------|--------|
| **QPS (10K vectors)** | 490 | 1,500 | 3,000 | 5,000+ | 🎯 |
| **QPS (100K vectors)** | 22,000 | 35,000 | 45,000 | 50,000+ | 🎯 |
| **p99 Latency (10K)** | 1.8ms | 1.2ms | 0.8ms | <1.0ms | 🎯 |
| **Recall@10** | 99% | 99% | 99% | 99%+ | 🎯 |
| **Memory Overhead** | - | +10% | +15% | <20% | 🎯 |

### Dependency Graph

```
Milestone 1: SIMD Optimization (Foundation)
    ├─> Milestone 2: Batch Query Processing (Requires SIMD kernels)
    └─> Milestone 6: Distance Specialization (Requires SIMD kernels)

Milestone 2: Batch Query Processing
    ├─> Milestone 3: Memory Layout Optimization (Uses batch patterns)
    └─> Milestone 4: Multi-threading Enhancement (Parallelizes batches)

Milestone 3: Memory Layout Optimization
    └─> Milestone 4: Multi-threading Enhancement (NUMA-aware threading)

Milestone 4: Multi-threading Enhancement
    └─> Milestone 5: Lock-Free Structures (Eliminates thread contention)

Milestone 5: Lock-Free Structures
    └─> Milestone 7: Query Compilation (Lock-free runtime)

Milestone 6: Distance Specialization
    └─> Milestone 7: Query Compilation (Inlines specialized distance)
```

### Critical Path

**Phase 1 (Weeks 1-6): Foundation**
1. SIMD Optimization (Milestone 1) - 3 weeks
2. Batch Query Processing (Milestone 2) - 3 weeks

**Phase 2 (Weeks 7-11): Scaling**
3. Memory Layout Optimization (Milestone 3) - 2 weeks
4. Multi-threading Enhancement (Milestone 4) - 2 weeks
5. Lock-Free Structures (Milestone 5) - 1 week

**Phase 3 (Weeks 12-13): Refinement**
6. Distance Specialization (Milestone 6) - 1 week

**Phase 4 (Future): Advanced**
7. Query Compilation (Milestone 7) - 4 weeks (optional)

---

## Risk Assessment & Mitigation

### High-Risk Items

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **AVX-512 bugs** | Medium | Critical | Extensive testing, fuzzing, fallback to AVX2 |
| **Precision loss in SIMD** | Low | High | Use f64 accumulators, validation tests |
| **Batch processing complexity** | Medium | Medium | Incremental rollout, A/B testing |
| **Thread contention** | Medium | High | Lock-free data structures, profiling |
| **Memory overhead** | Medium | Medium | Memory budget tracking, pooling |
| **Query compilation bugs** | High | Critical | Sandbox execution, validation against reference |

### Medium-Risk Items

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **NUMA allocation failures** | Medium | Medium | Graceful fallback to uniform allocation |
| **Huge pages unavailable** | High | Low | Fallback to regular pages |
| **Cache pollution** | Low | Medium | Prefetch tuning, profiling |
| **Load imbalance** | Low | Low | Work stealing scheduler |

---

## Monitoring & Continuous Validation

### Automated Benchmarking

```bash
# CI/CD pipeline integration
.github/workflows/performance.yml

name: Performance Regression Check
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: cargo bench --bench comprehensive_bench -- --save-baseline pr-${{ github.event.pull_request.number }}

      - name: Compare against main
        run: |
          cargo bench --bench comprehensive_bench -- --baseline main --test

      - name: Fail if regression >5%
        run: |
          python scripts/check_regression.py --threshold 0.05
```

### Performance Dashboard

```rust
// Export metrics to Prometheus
use prometheus::{Counter, Histogram, register_counter, register_histogram};

lazy_static! {
    static ref QPS: Counter = register_counter!("ruvector_qps", "Queries per second").unwrap();
    static ref LATENCY: Histogram = register_histogram!(
        "ruvector_latency_seconds",
        "Query latency in seconds"
    ).unwrap();
}
```

---

## Conclusion

This GOAP plan provides a systematic, measurable path to achieve **10x throughput improvement** for RuVector, bringing it from **~490 QPS** to **5,000+ QPS** while maintaining **99%+ recall**.

### Key Success Factors

1. **Phased Approach**: Clear milestones with dependencies
2. **Measurable Outcomes**: Every action has quantifiable success criteria
3. **Risk Management**: Identified risks with mitigation strategies
4. **Continuous Validation**: Automated testing and benchmarking
5. **Production Focus**: Emphasis on correctness and stability

### Next Steps

1. ✅ Review and approve this GOAP plan
2. 🔄 Begin Milestone 1: SIMD Optimization (Week 1)
3. 📊 Establish baseline benchmarks and CI
4. 🏗️ Implement actions sequentially following dependency graph
5. 📈 Monitor progress against success metrics
6. 🔁 Iterate based on profiling and real-world feedback

**Estimated Timeline:** 12-13 weeks for Phase 1-3 (core optimizations)
**Expected Outcome:** Competitive SOTA performance with novel GNN features

---

**Document Owner:** RuVector Performance Team
**Last Updated:** 2025-11-28
**Status:** Draft for Review
