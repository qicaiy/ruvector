//! SIMD-Optimized Vector Operations
//!
//! High-performance vector operations using SIMD instructions when available.
//! Provides fallbacks for non-SIMD platforms.
//!
//! ## Performance Targets
//! - Dot product 384-dim: <1us
//! - Cosine similarity 384-dim: <2us
//! - L2 norm 384-dim: <1us
//! - Batch dot product (100x384): <100us

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// Public API
// =============================================================================

/// Compute dot product of two vectors.
/// Uses SIMD when available.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have equal length");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { dot_product_avx2(a, b) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { dot_product_neon(a, b) };
    }

    // Fallback
    dot_product_scalar(a, b)
}

/// Compute L2 norm of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { l2_norm_avx2(v) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { l2_norm_neon(v) };
    }

    // Fallback
    l2_norm_scalar(v)
}

/// Compute cosine similarity between two vectors.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { cosine_similarity_avx2(a, b) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { cosine_similarity_neon(a, b) };
    }

    // Fallback
    cosine_similarity_scalar(a, b)
}

/// Compute cosine similarity with pre-computed norm for `a`.
#[inline]
pub fn cosine_similarity_prenorm(a: &[f32], a_norm: f32, b: &[f32]) -> f32 {
    if a.len() != b.len() || a_norm < 1e-8 {
        return 0.0;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { cosine_similarity_prenorm_avx2(a, a_norm, b) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { cosine_similarity_prenorm_neon(a, a_norm, b) };
    }

    // Fallback
    cosine_similarity_prenorm_scalar(a, a_norm, b)
}

/// Normalize a vector in-place.
#[inline]
pub fn normalize_inplace(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-8 {
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// Normalize a vector, returning a new vector.
#[inline]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm > 1e-8 {
        let inv_norm = 1.0 / norm;
        v.iter().map(|x| x * inv_norm).collect()
    } else {
        v.to_vec()
    }
}

/// Batch cosine similarity: compute similarity between query and all candidates.
/// Returns Vec<(index, similarity)>.
#[inline]
pub fn batch_cosine_similarity(query: &[f32], candidates: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let query_norm = l2_norm(query);
    if query_norm < 1e-8 {
        return candidates
            .iter()
            .enumerate()
            .map(|(i, _)| (i, 0.0))
            .collect();
    }

    candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity_prenorm(query, query_norm, c)))
        .collect()
}

/// Batch cosine similarity with slice of slices (avoids Vec overhead).
#[inline]
pub fn batch_cosine_similarity_slices<'a>(
    query: &[f32],
    candidates: impl Iterator<Item = &'a [f32]>,
) -> Vec<f32> {
    let query_norm = l2_norm(query);
    if query_norm < 1e-8 {
        return candidates.map(|_| 0.0).collect();
    }

    candidates
        .map(|c| cosine_similarity_prenorm(query, query_norm, c))
        .collect()
}

// =============================================================================
// Batch SIMD Operations - Process 4-8 embeddings in parallel
// =============================================================================

/// Compute cosine similarity between query and 4 target vectors simultaneously.
/// Uses SIMD to process all 4 dot products in parallel for ~4x throughput.
///
/// # Arguments
/// * `query` - The query vector
/// * `targets` - Array of 4 target vectors (must all have same length as query)
///
/// # Returns
/// Array of 4 similarity scores
#[inline]
pub fn batch_cosine_similarity_4(query: &[f32], targets: [&[f32]; 4]) -> [f32; 4] {
    let query_norm = l2_norm(query);
    if query_norm < 1e-8 {
        return [0.0; 4];
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { batch_cosine_similarity_4_avx2(query, targets, query_norm) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { batch_cosine_similarity_4_neon(query, targets, query_norm) };
    }

    // Fallback: sequential computation
    [
        cosine_similarity_prenorm(query, query_norm, targets[0]),
        cosine_similarity_prenorm(query, query_norm, targets[1]),
        cosine_similarity_prenorm(query, query_norm, targets[2]),
        cosine_similarity_prenorm(query, query_norm, targets[3]),
    ]
}

/// Compute cosine similarity between query and 8 target vectors simultaneously.
/// Maximum SIMD utilization for AVX2 (256-bit registers).
#[inline]
pub fn batch_cosine_similarity_8(query: &[f32], targets: [&[f32]; 8]) -> [f32; 8] {
    let query_norm = l2_norm(query);
    if query_norm < 1e-8 {
        return [0.0; 8];
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { batch_cosine_similarity_8_avx2(query, targets, query_norm) };
    }

    // Fallback: use two batch_4 operations
    let first_4 =
        batch_cosine_similarity_4(query, [targets[0], targets[1], targets[2], targets[3]]);
    let second_4 =
        batch_cosine_similarity_4(query, [targets[4], targets[5], targets[6], targets[7]]);

    [
        first_4[0],
        first_4[1],
        first_4[2],
        first_4[3],
        second_4[0],
        second_4[1],
        second_4[2],
        second_4[3],
    ]
}

/// Normalize multiple vectors in-place using SIMD.
/// Processes vectors in chunks of 4 for better cache utilization.
#[inline]
pub fn batch_normalize_inplace(vectors: &mut [&mut [f32]]) {
    // Process in chunks of 4
    let chunks = vectors.len() / 4;
    let remainder = vectors.len() % 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        // Compute all 4 norms first
        let norms = [
            l2_norm(vectors[base]),
            l2_norm(vectors[base + 1]),
            l2_norm(vectors[base + 2]),
            l2_norm(vectors[base + 3]),
        ];

        // Normalize each vector
        for i in 0..4 {
            if norms[i] > 1e-8 {
                let inv_norm = 1.0 / norms[i];
                normalize_vector_inplace(vectors[base + i], inv_norm);
            }
        }
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        normalize_inplace(vectors[start + i]);
    }
}

/// Normalize a vector with pre-computed inverse norm using SIMD.
#[inline]
fn normalize_vector_inplace(v: &mut [f32], inv_norm: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { normalize_vector_inplace_avx2(v, inv_norm) };
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { normalize_vector_inplace_neon(v, inv_norm) };
        return;
    }

    // Fallback: scalar
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

/// Compute dot products between query and N target vectors.
/// Uses chunked SIMD processing for optimal throughput.
///
/// # Arguments
/// * `query` - The query vector
/// * `targets` - Slice of target vectors
///
/// # Returns
/// Vector of dot products
#[inline]
pub fn batch_dot_products(query: &[f32], targets: &[Vec<f32>]) -> Vec<f32> {
    let n = targets.len();
    if n == 0 {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(n);

    // Process in chunks of 4 for better SIMD utilization
    let chunks = n / 4;
    let remainder = n % 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let dots = batch_dot_products_4(
            query,
            [
                &targets[base],
                &targets[base + 1],
                &targets[base + 2],
                &targets[base + 3],
            ],
        );
        results.extend_from_slice(&dots);
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        results.push(dot_product(query, &targets[start + i]));
    }

    results
}

/// Compute dot products between query and 4 target vectors simultaneously.
#[inline]
pub fn batch_dot_products_4(query: &[f32], targets: [&[f32]; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { batch_dot_products_4_avx2(query, targets) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { batch_dot_products_4_neon(query, targets) };
    }

    // Fallback
    [
        dot_product(query, targets[0]),
        dot_product(query, targets[1]),
        dot_product(query, targets[2]),
        dot_product(query, targets[3]),
    ]
}

/// Batch L2 norms for multiple vectors.
#[inline]
pub fn batch_l2_norms(vectors: &[&[f32]]) -> Vec<f32> {
    let n = vectors.len();
    if n == 0 {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(n);

    // Process in chunks of 4
    let chunks = n / 4;
    let remainder = n % 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let norms = batch_l2_norms_4([
            vectors[base],
            vectors[base + 1],
            vectors[base + 2],
            vectors[base + 3],
        ]);
        results.extend_from_slice(&norms);
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        results.push(l2_norm(vectors[start + i]));
    }

    results
}

/// Compute L2 norms for 4 vectors simultaneously.
#[inline]
pub fn batch_l2_norms_4(vectors: [&[f32]; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { batch_l2_norms_4_avx2(vectors) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { batch_l2_norms_4_neon(vectors) };
    }

    // Fallback
    [
        l2_norm(vectors[0]),
        l2_norm(vectors[1]),
        l2_norm(vectors[2]),
        l2_norm(vectors[3]),
    ]
}

/// Optimized batch similarity search for memory retrieval.
/// Processes candidates in chunks of 4/8 for maximum SIMD throughput.
///
/// Returns indices and scores of top-k matches above threshold.
#[inline]
pub fn batch_similarity_search(
    query: &[f32],
    candidates: &[Vec<f32>],
    top_k: usize,
    threshold: f32,
) -> Vec<(usize, f32)> {
    let query_norm = l2_norm(query);
    if query_norm < 1e-8 || candidates.is_empty() {
        return Vec::new();
    }

    let n = candidates.len();
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n.min(top_k * 2));

    // Process in chunks of 4 for SIMD efficiency
    let chunks = n / 4;
    let remainder = n % 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        // Get references for batch processing
        let batch_scores = batch_cosine_similarity_4_prenorm(
            query,
            query_norm,
            [
                &candidates[base],
                &candidates[base + 1],
                &candidates[base + 2],
                &candidates[base + 3],
            ],
        );

        // Filter by threshold
        for (i, &score) in batch_scores.iter().enumerate() {
            if score >= threshold {
                scores.push((base + i, score));
            }
        }
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let idx = start + i;
        let score = cosine_similarity_prenorm(query, query_norm, &candidates[idx]);
        if score >= threshold {
            scores.push((idx, score));
        }
    }

    // Sort by score descending and take top_k
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);

    scores
}

/// Batch cosine similarity with pre-computed query norm.
#[inline]
pub fn batch_cosine_similarity_4_prenorm(
    query: &[f32],
    query_norm: f32,
    targets: [&[f32]; 4],
) -> [f32; 4] {
    if query_norm < 1e-8 {
        return [0.0; 4];
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { batch_cosine_similarity_4_avx2(query, targets, query_norm) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { batch_cosine_similarity_4_neon(query, targets, query_norm) };
    }

    // Fallback
    [
        cosine_similarity_prenorm(query, query_norm, targets[0]),
        cosine_similarity_prenorm(query, query_norm, targets[1]),
        cosine_similarity_prenorm(query, query_norm, targets[2]),
        cosine_similarity_prenorm(query, query_norm, targets[3]),
    ]
}

// =============================================================================
// Scalar Implementations (Fallback)
// =============================================================================

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum = 0.0f32;

    // Unrolled loop
    for i in 0..chunks {
        let idx = i * 4;
        sum += a[idx] * b[idx];
        sum += a[idx + 1] * b[idx + 1];
        sum += a[idx + 2] * b[idx + 2];
        sum += a[idx + 3] * b[idx + 3];
    }

    // Remainder
    let start = chunks * 4;
    for i in 0..remainder {
        sum += a[start + i] * b[start + i];
    }

    sum
}

#[inline]
fn l2_norm_scalar(v: &[f32]) -> f32 {
    let len = v.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let idx = i * 4;
        sum += v[idx] * v[idx];
        sum += v[idx + 1] * v[idx + 1];
        sum += v[idx + 2] * v[idx + 2];
        sum += v[idx + 3] * v[idx + 3];
    }

    let start = chunks * 4;
    for i in 0..remainder {
        sum += v[start + i] * v[start + i];
    }

    sum.sqrt()
}

#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut dot = 0.0f32;
    let mut a_norm_sq = 0.0f32;
    let mut b_norm_sq = 0.0f32;

    for i in 0..chunks {
        let idx = i * 4;
        dot += a[idx] * b[idx];
        dot += a[idx + 1] * b[idx + 1];
        dot += a[idx + 2] * b[idx + 2];
        dot += a[idx + 3] * b[idx + 3];

        a_norm_sq += a[idx] * a[idx];
        a_norm_sq += a[idx + 1] * a[idx + 1];
        a_norm_sq += a[idx + 2] * a[idx + 2];
        a_norm_sq += a[idx + 3] * a[idx + 3];

        b_norm_sq += b[idx] * b[idx];
        b_norm_sq += b[idx + 1] * b[idx + 1];
        b_norm_sq += b[idx + 2] * b[idx + 2];
        b_norm_sq += b[idx + 3] * b[idx + 3];
    }

    let start = chunks * 4;
    for i in 0..remainder {
        dot += a[start + i] * b[start + i];
        a_norm_sq += a[start + i] * a[start + i];
        b_norm_sq += b[start + i] * b[start + i];
    }

    let norm_product = (a_norm_sq * b_norm_sq).sqrt();
    if norm_product < 1e-8 {
        0.0
    } else {
        (dot / norm_product).clamp(-1.0, 1.0)
    }
}

#[inline]
fn cosine_similarity_prenorm_scalar(a: &[f32], a_norm: f32, b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut dot = 0.0f32;
    let mut b_norm_sq = 0.0f32;

    for i in 0..chunks {
        let idx = i * 4;
        dot += a[idx] * b[idx];
        dot += a[idx + 1] * b[idx + 1];
        dot += a[idx + 2] * b[idx + 2];
        dot += a[idx + 3] * b[idx + 3];

        b_norm_sq += b[idx] * b[idx];
        b_norm_sq += b[idx + 1] * b[idx + 1];
        b_norm_sq += b[idx + 2] * b[idx + 2];
        b_norm_sq += b[idx + 3] * b[idx + 3];
    }

    let start = chunks * 4;
    for i in 0..remainder {
        dot += a[start + i] * b[start + i];
        b_norm_sq += b[start + i] * b[start + i];
    }

    let b_norm = b_norm_sq.sqrt();
    if b_norm < 1e-8 {
        0.0
    } else {
        (dot / (a_norm * b_norm)).clamp(-1.0, 1.0)
    }
}

// =============================================================================
// AVX2 Implementations (x86_64)
// =============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let vsum = _mm_add_ps(vlow, vhigh);
    let shuf = _mm_movehdup_ps(vsum);
    let sums = _mm_add_ps(vsum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(idx));
        let vb = _mm256_loadu_ps(b_ptr.add(idx));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut total = hsum256(sum);

    // Remainder
    let start = chunks * 8;
    for i in 0..remainder {
        total += a[start + i] * b[start + i];
    }

    total
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn l2_norm_avx2(v: &[f32]) -> f32 {
    let len = v.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();
    let v_ptr = v.as_ptr();

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(v_ptr.add(idx));
        sum = _mm256_fmadd_ps(va, va, sum);
    }

    let mut total = hsum256(sum);

    let start = chunks * 8;
    for i in 0..remainder {
        let val = v[start + i];
        total += val * val;
    }

    total.sqrt()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut dot_sum = _mm256_setzero_ps();
    let mut a_sum = _mm256_setzero_ps();
    let mut b_sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(idx));
        let vb = _mm256_loadu_ps(b_ptr.add(idx));

        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        a_sum = _mm256_fmadd_ps(va, va, a_sum);
        b_sum = _mm256_fmadd_ps(vb, vb, b_sum);
    }

    let mut dot = hsum256(dot_sum);
    let mut a_norm_sq = hsum256(a_sum);
    let mut b_norm_sq = hsum256(b_sum);

    let start = chunks * 8;
    for i in 0..remainder {
        dot += a[start + i] * b[start + i];
        a_norm_sq += a[start + i] * a[start + i];
        b_norm_sq += b[start + i] * b[start + i];
    }

    let norm_product = (a_norm_sq * b_norm_sq).sqrt();
    if norm_product < 1e-8 {
        0.0
    } else {
        (dot / norm_product).clamp(-1.0, 1.0)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn cosine_similarity_prenorm_avx2(a: &[f32], a_norm: f32, b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut dot_sum = _mm256_setzero_ps();
    let mut b_sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(idx));
        let vb = _mm256_loadu_ps(b_ptr.add(idx));

        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        b_sum = _mm256_fmadd_ps(vb, vb, b_sum);
    }

    let mut dot = hsum256(dot_sum);
    let mut b_norm_sq = hsum256(b_sum);

    let start = chunks * 8;
    for i in 0..remainder {
        dot += a[start + i] * b[start + i];
        b_norm_sq += b[start + i] * b[start + i];
    }

    let b_norm = b_norm_sq.sqrt();
    if b_norm < 1e-8 {
        0.0
    } else {
        (dot / (a_norm * b_norm)).clamp(-1.0, 1.0)
    }
}

// =============================================================================
// Batch AVX2 Implementations - Process 4-8 vectors simultaneously
// =============================================================================

/// Batch cosine similarity for 4 vectors using AVX2.
/// Interleaves computations across all 4 targets for better ILP.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn batch_cosine_similarity_4_avx2(
    query: &[f32],
    targets: [&[f32]; 4],
    query_norm: f32,
) -> [f32; 4] {
    let len = query.len();
    let chunks = len / 8;
    let remainder = len % 8;

    // Accumulators for each target
    let mut dot_sum_0 = _mm256_setzero_ps();
    let mut dot_sum_1 = _mm256_setzero_ps();
    let mut dot_sum_2 = _mm256_setzero_ps();
    let mut dot_sum_3 = _mm256_setzero_ps();

    let mut norm_sum_0 = _mm256_setzero_ps();
    let mut norm_sum_1 = _mm256_setzero_ps();
    let mut norm_sum_2 = _mm256_setzero_ps();
    let mut norm_sum_3 = _mm256_setzero_ps();

    let q_ptr = query.as_ptr();
    let t0_ptr = targets[0].as_ptr();
    let t1_ptr = targets[1].as_ptr();
    let t2_ptr = targets[2].as_ptr();
    let t3_ptr = targets[3].as_ptr();

    // Process 8 elements at a time across all 4 targets
    for i in 0..chunks {
        let idx = i * 8;

        // Load query once, reuse for all 4 targets
        let vq = _mm256_loadu_ps(q_ptr.add(idx));

        // Target 0
        let vt0 = _mm256_loadu_ps(t0_ptr.add(idx));
        dot_sum_0 = _mm256_fmadd_ps(vq, vt0, dot_sum_0);
        norm_sum_0 = _mm256_fmadd_ps(vt0, vt0, norm_sum_0);

        // Target 1
        let vt1 = _mm256_loadu_ps(t1_ptr.add(idx));
        dot_sum_1 = _mm256_fmadd_ps(vq, vt1, dot_sum_1);
        norm_sum_1 = _mm256_fmadd_ps(vt1, vt1, norm_sum_1);

        // Target 2
        let vt2 = _mm256_loadu_ps(t2_ptr.add(idx));
        dot_sum_2 = _mm256_fmadd_ps(vq, vt2, dot_sum_2);
        norm_sum_2 = _mm256_fmadd_ps(vt2, vt2, norm_sum_2);

        // Target 3
        let vt3 = _mm256_loadu_ps(t3_ptr.add(idx));
        dot_sum_3 = _mm256_fmadd_ps(vq, vt3, dot_sum_3);
        norm_sum_3 = _mm256_fmadd_ps(vt3, vt3, norm_sum_3);
    }

    // Horizontal sums
    let mut dots = [
        hsum256(dot_sum_0),
        hsum256(dot_sum_1),
        hsum256(dot_sum_2),
        hsum256(dot_sum_3),
    ];
    let mut norms_sq = [
        hsum256(norm_sum_0),
        hsum256(norm_sum_1),
        hsum256(norm_sum_2),
        hsum256(norm_sum_3),
    ];

    // Handle remainder
    let start = chunks * 8;
    for i in 0..remainder {
        let idx = start + i;
        let q = query[idx];
        dots[0] += q * targets[0][idx];
        dots[1] += q * targets[1][idx];
        dots[2] += q * targets[2][idx];
        dots[3] += q * targets[3][idx];

        norms_sq[0] += targets[0][idx] * targets[0][idx];
        norms_sq[1] += targets[1][idx] * targets[1][idx];
        norms_sq[2] += targets[2][idx] * targets[2][idx];
        norms_sq[3] += targets[3][idx] * targets[3][idx];
    }

    // Compute final similarities
    let mut results = [0.0f32; 4];
    for i in 0..4 {
        let t_norm = norms_sq[i].sqrt();
        if t_norm >= 1e-8 {
            results[i] = (dots[i] / (query_norm * t_norm)).clamp(-1.0, 1.0);
        }
    }

    results
}

/// Batch cosine similarity for 8 vectors using AVX2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn batch_cosine_similarity_8_avx2(
    query: &[f32],
    targets: [&[f32]; 8],
    query_norm: f32,
) -> [f32; 8] {
    // Process as two batches of 4 for better register utilization
    let first_4 = batch_cosine_similarity_4_avx2(
        query,
        [targets[0], targets[1], targets[2], targets[3]],
        query_norm,
    );
    let second_4 = batch_cosine_similarity_4_avx2(
        query,
        [targets[4], targets[5], targets[6], targets[7]],
        query_norm,
    );

    [
        first_4[0],
        first_4[1],
        first_4[2],
        first_4[3],
        second_4[0],
        second_4[1],
        second_4[2],
        second_4[3],
    ]
}

/// Batch dot products for 4 vectors using AVX2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn batch_dot_products_4_avx2(query: &[f32], targets: [&[f32]; 4]) -> [f32; 4] {
    let len = query.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum_0 = _mm256_setzero_ps();
    let mut sum_1 = _mm256_setzero_ps();
    let mut sum_2 = _mm256_setzero_ps();
    let mut sum_3 = _mm256_setzero_ps();

    let q_ptr = query.as_ptr();
    let t0_ptr = targets[0].as_ptr();
    let t1_ptr = targets[1].as_ptr();
    let t2_ptr = targets[2].as_ptr();
    let t3_ptr = targets[3].as_ptr();

    for i in 0..chunks {
        let idx = i * 8;
        let vq = _mm256_loadu_ps(q_ptr.add(idx));

        sum_0 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(t0_ptr.add(idx)), sum_0);
        sum_1 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(t1_ptr.add(idx)), sum_1);
        sum_2 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(t2_ptr.add(idx)), sum_2);
        sum_3 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(t3_ptr.add(idx)), sum_3);
    }

    let mut results = [
        hsum256(sum_0),
        hsum256(sum_1),
        hsum256(sum_2),
        hsum256(sum_3),
    ];

    // Handle remainder
    let start = chunks * 8;
    for i in 0..remainder {
        let idx = start + i;
        let q = query[idx];
        results[0] += q * targets[0][idx];
        results[1] += q * targets[1][idx];
        results[2] += q * targets[2][idx];
        results[3] += q * targets[3][idx];
    }

    results
}

/// Batch L2 norms for 4 vectors using AVX2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn batch_l2_norms_4_avx2(vectors: [&[f32]; 4]) -> [f32; 4] {
    // Find minimum length (should all be equal, but be safe)
    let len = vectors[0].len();

    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum_0 = _mm256_setzero_ps();
    let mut sum_1 = _mm256_setzero_ps();
    let mut sum_2 = _mm256_setzero_ps();
    let mut sum_3 = _mm256_setzero_ps();

    let v0_ptr = vectors[0].as_ptr();
    let v1_ptr = vectors[1].as_ptr();
    let v2_ptr = vectors[2].as_ptr();
    let v3_ptr = vectors[3].as_ptr();

    for i in 0..chunks {
        let idx = i * 8;

        let v0 = _mm256_loadu_ps(v0_ptr.add(idx));
        let v1 = _mm256_loadu_ps(v1_ptr.add(idx));
        let v2 = _mm256_loadu_ps(v2_ptr.add(idx));
        let v3 = _mm256_loadu_ps(v3_ptr.add(idx));

        sum_0 = _mm256_fmadd_ps(v0, v0, sum_0);
        sum_1 = _mm256_fmadd_ps(v1, v1, sum_1);
        sum_2 = _mm256_fmadd_ps(v2, v2, sum_2);
        sum_3 = _mm256_fmadd_ps(v3, v3, sum_3);
    }

    let mut totals = [
        hsum256(sum_0),
        hsum256(sum_1),
        hsum256(sum_2),
        hsum256(sum_3),
    ];

    // Handle remainder
    let start = chunks * 8;
    for i in 0..remainder {
        let idx = start + i;
        totals[0] += vectors[0][idx] * vectors[0][idx];
        totals[1] += vectors[1][idx] * vectors[1][idx];
        totals[2] += vectors[2][idx] * vectors[2][idx];
        totals[3] += vectors[3][idx] * vectors[3][idx];
    }

    [
        totals[0].sqrt(),
        totals[1].sqrt(),
        totals[2].sqrt(),
        totals[3].sqrt(),
    ]
}

/// Normalize a vector in-place using AVX2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn normalize_vector_inplace_avx2(v: &mut [f32], inv_norm: f32) {
    let len = v.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let inv_norm_vec = _mm256_set1_ps(inv_norm);
    let v_ptr = v.as_mut_ptr();

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(v_ptr.add(idx));
        let result = _mm256_mul_ps(va, inv_norm_vec);
        _mm256_storeu_ps(v_ptr.add(idx), result);
    }

    // Handle remainder
    let start = chunks * 8;
    for i in 0..remainder {
        v[start + i] *= inv_norm;
    }
}

// =============================================================================
// NEON Implementations (aarch64)
// =============================================================================

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a_ptr.add(idx));
        let vb = vld1q_f32(b_ptr.add(idx));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut total = vaddvq_f32(sum);

    let start = chunks * 4;
    for i in 0..remainder {
        total += a[start + i] * b[start + i];
    }

    total
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn l2_norm_neon(v: &[f32]) -> f32 {
    let len = v.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum = vdupq_n_f32(0.0);
    let v_ptr = v.as_ptr();

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(v_ptr.add(idx));
        sum = vfmaq_f32(sum, va, va);
    }

    let mut total = vaddvq_f32(sum);

    let start = chunks * 4;
    for i in 0..remainder {
        let val = v[start + i];
        total += val * val;
    }

    total.sqrt()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut dot_sum = vdupq_n_f32(0.0);
    let mut a_sum = vdupq_n_f32(0.0);
    let mut b_sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a_ptr.add(idx));
        let vb = vld1q_f32(b_ptr.add(idx));

        dot_sum = vfmaq_f32(dot_sum, va, vb);
        a_sum = vfmaq_f32(a_sum, va, va);
        b_sum = vfmaq_f32(b_sum, vb, vb);
    }

    let mut dot = vaddvq_f32(dot_sum);
    let mut a_norm_sq = vaddvq_f32(a_sum);
    let mut b_norm_sq = vaddvq_f32(b_sum);

    let start = chunks * 4;
    for i in 0..remainder {
        dot += a[start + i] * b[start + i];
        a_norm_sq += a[start + i] * a[start + i];
        b_norm_sq += b[start + i] * b[start + i];
    }

    let norm_product = (a_norm_sq * b_norm_sq).sqrt();
    if norm_product < 1e-8 {
        0.0
    } else {
        (dot / norm_product).clamp(-1.0, 1.0)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn cosine_similarity_prenorm_neon(a: &[f32], a_norm: f32, b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut dot_sum = vdupq_n_f32(0.0);
    let mut b_sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a_ptr.add(idx));
        let vb = vld1q_f32(b_ptr.add(idx));

        dot_sum = vfmaq_f32(dot_sum, va, vb);
        b_sum = vfmaq_f32(b_sum, vb, vb);
    }

    let mut dot = vaddvq_f32(dot_sum);
    let mut b_norm_sq = vaddvq_f32(b_sum);

    let start = chunks * 4;
    for i in 0..remainder {
        dot += a[start + i] * b[start + i];
        b_norm_sq += b[start + i] * b[start + i];
    }

    let b_norm = b_norm_sq.sqrt();
    if b_norm < 1e-8 {
        0.0
    } else {
        (dot / (a_norm * b_norm)).clamp(-1.0, 1.0)
    }
}

// =============================================================================
// Batch NEON Implementations (aarch64) - Process 4 vectors simultaneously
// =============================================================================

/// Batch cosine similarity for 4 vectors using NEON.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn batch_cosine_similarity_4_neon(
    query: &[f32],
    targets: [&[f32]; 4],
    query_norm: f32,
) -> [f32; 4] {
    let len = query.len();
    let chunks = len / 4;
    let remainder = len % 4;

    // Accumulators for each target
    let mut dot_sum_0 = vdupq_n_f32(0.0);
    let mut dot_sum_1 = vdupq_n_f32(0.0);
    let mut dot_sum_2 = vdupq_n_f32(0.0);
    let mut dot_sum_3 = vdupq_n_f32(0.0);

    let mut norm_sum_0 = vdupq_n_f32(0.0);
    let mut norm_sum_1 = vdupq_n_f32(0.0);
    let mut norm_sum_2 = vdupq_n_f32(0.0);
    let mut norm_sum_3 = vdupq_n_f32(0.0);

    let q_ptr = query.as_ptr();
    let t0_ptr = targets[0].as_ptr();
    let t1_ptr = targets[1].as_ptr();
    let t2_ptr = targets[2].as_ptr();
    let t3_ptr = targets[3].as_ptr();

    for i in 0..chunks {
        let idx = i * 4;

        // Load query once
        let vq = vld1q_f32(q_ptr.add(idx));

        // Target 0
        let vt0 = vld1q_f32(t0_ptr.add(idx));
        dot_sum_0 = vfmaq_f32(dot_sum_0, vq, vt0);
        norm_sum_0 = vfmaq_f32(norm_sum_0, vt0, vt0);

        // Target 1
        let vt1 = vld1q_f32(t1_ptr.add(idx));
        dot_sum_1 = vfmaq_f32(dot_sum_1, vq, vt1);
        norm_sum_1 = vfmaq_f32(norm_sum_1, vt1, vt1);

        // Target 2
        let vt2 = vld1q_f32(t2_ptr.add(idx));
        dot_sum_2 = vfmaq_f32(dot_sum_2, vq, vt2);
        norm_sum_2 = vfmaq_f32(norm_sum_2, vt2, vt2);

        // Target 3
        let vt3 = vld1q_f32(t3_ptr.add(idx));
        dot_sum_3 = vfmaq_f32(dot_sum_3, vq, vt3);
        norm_sum_3 = vfmaq_f32(norm_sum_3, vt3, vt3);
    }

    // Horizontal sums
    let mut dots = [
        vaddvq_f32(dot_sum_0),
        vaddvq_f32(dot_sum_1),
        vaddvq_f32(dot_sum_2),
        vaddvq_f32(dot_sum_3),
    ];
    let mut norms_sq = [
        vaddvq_f32(norm_sum_0),
        vaddvq_f32(norm_sum_1),
        vaddvq_f32(norm_sum_2),
        vaddvq_f32(norm_sum_3),
    ];

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let idx = start + i;
        let q = query[idx];
        dots[0] += q * targets[0][idx];
        dots[1] += q * targets[1][idx];
        dots[2] += q * targets[2][idx];
        dots[3] += q * targets[3][idx];

        norms_sq[0] += targets[0][idx] * targets[0][idx];
        norms_sq[1] += targets[1][idx] * targets[1][idx];
        norms_sq[2] += targets[2][idx] * targets[2][idx];
        norms_sq[3] += targets[3][idx] * targets[3][idx];
    }

    // Compute final similarities
    let mut results = [0.0f32; 4];
    for i in 0..4 {
        let t_norm = norms_sq[i].sqrt();
        if t_norm >= 1e-8 {
            results[i] = (dots[i] / (query_norm * t_norm)).clamp(-1.0, 1.0);
        }
    }

    results
}

/// Batch dot products for 4 vectors using NEON.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn batch_dot_products_4_neon(query: &[f32], targets: [&[f32]; 4]) -> [f32; 4] {
    let len = query.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum_0 = vdupq_n_f32(0.0);
    let mut sum_1 = vdupq_n_f32(0.0);
    let mut sum_2 = vdupq_n_f32(0.0);
    let mut sum_3 = vdupq_n_f32(0.0);

    let q_ptr = query.as_ptr();
    let t0_ptr = targets[0].as_ptr();
    let t1_ptr = targets[1].as_ptr();
    let t2_ptr = targets[2].as_ptr();
    let t3_ptr = targets[3].as_ptr();

    for i in 0..chunks {
        let idx = i * 4;
        let vq = vld1q_f32(q_ptr.add(idx));

        sum_0 = vfmaq_f32(sum_0, vq, vld1q_f32(t0_ptr.add(idx)));
        sum_1 = vfmaq_f32(sum_1, vq, vld1q_f32(t1_ptr.add(idx)));
        sum_2 = vfmaq_f32(sum_2, vq, vld1q_f32(t2_ptr.add(idx)));
        sum_3 = vfmaq_f32(sum_3, vq, vld1q_f32(t3_ptr.add(idx)));
    }

    let mut results = [
        vaddvq_f32(sum_0),
        vaddvq_f32(sum_1),
        vaddvq_f32(sum_2),
        vaddvq_f32(sum_3),
    ];

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let idx = start + i;
        let q = query[idx];
        results[0] += q * targets[0][idx];
        results[1] += q * targets[1][idx];
        results[2] += q * targets[2][idx];
        results[3] += q * targets[3][idx];
    }

    results
}

/// Batch L2 norms for 4 vectors using NEON.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn batch_l2_norms_4_neon(vectors: [&[f32]; 4]) -> [f32; 4] {
    let len = vectors[0].len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum_0 = vdupq_n_f32(0.0);
    let mut sum_1 = vdupq_n_f32(0.0);
    let mut sum_2 = vdupq_n_f32(0.0);
    let mut sum_3 = vdupq_n_f32(0.0);

    let v0_ptr = vectors[0].as_ptr();
    let v1_ptr = vectors[1].as_ptr();
    let v2_ptr = vectors[2].as_ptr();
    let v3_ptr = vectors[3].as_ptr();

    for i in 0..chunks {
        let idx = i * 4;

        let v0 = vld1q_f32(v0_ptr.add(idx));
        let v1 = vld1q_f32(v1_ptr.add(idx));
        let v2 = vld1q_f32(v2_ptr.add(idx));
        let v3 = vld1q_f32(v3_ptr.add(idx));

        sum_0 = vfmaq_f32(sum_0, v0, v0);
        sum_1 = vfmaq_f32(sum_1, v1, v1);
        sum_2 = vfmaq_f32(sum_2, v2, v2);
        sum_3 = vfmaq_f32(sum_3, v3, v3);
    }

    let mut totals = [
        vaddvq_f32(sum_0),
        vaddvq_f32(sum_1),
        vaddvq_f32(sum_2),
        vaddvq_f32(sum_3),
    ];

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let idx = start + i;
        totals[0] += vectors[0][idx] * vectors[0][idx];
        totals[1] += vectors[1][idx] * vectors[1][idx];
        totals[2] += vectors[2][idx] * vectors[2][idx];
        totals[3] += vectors[3][idx] * vectors[3][idx];
    }

    [
        totals[0].sqrt(),
        totals[1].sqrt(),
        totals[2].sqrt(),
        totals[3].sqrt(),
    ]
}

/// Normalize a vector in-place using NEON.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn normalize_vector_inplace_neon(v: &mut [f32], inv_norm: f32) {
    let len = v.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let inv_norm_vec = vdupq_n_f32(inv_norm);
    let v_ptr = v.as_mut_ptr();

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(v_ptr.add(idx));
        let result = vmulq_f32(va, inv_norm_vec);
        vst1q_f32(v_ptr.add(idx), result);
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        v[start + i] *= inv_norm;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = dot_product(&a, &b);
        assert!((result - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let result = l2_norm(&v);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let result = cosine_similarity(&a, &a);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = cosine_similarity(&a, &b);
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = normalize(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];
        let results = batch_cosine_similarity(&query, &candidates);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        assert!(results[1].1.abs() < 1e-6);
    }

    #[test]
    fn test_large_vector() {
        let size = 384;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.01).collect();

        let result = cosine_similarity(&a, &b);
        assert!(result > 0.0 && result < 1.0);
    }

    // =============================================================================
    // Batch Operation Tests
    // =============================================================================

    #[test]
    fn test_batch_cosine_similarity_4() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let targets: [&[f32]; 4] = [
            &[1.0, 0.0, 0.0, 0.0],  // Should be 1.0 (identical)
            &[0.0, 1.0, 0.0, 0.0],  // Should be 0.0 (orthogonal)
            &[0.5, 0.5, 0.0, 0.0],  // Should be ~0.707
            &[-1.0, 0.0, 0.0, 0.0], // Should be -1.0 (opposite)
        ];

        let results = batch_cosine_similarity_4(&query, targets);

        assert!(
            (results[0] - 1.0).abs() < 1e-5,
            "Expected 1.0, got {}",
            results[0]
        );
        assert!(results[1].abs() < 1e-5, "Expected 0.0, got {}", results[1]);
        assert!(
            (results[2] - 0.7071).abs() < 0.01,
            "Expected ~0.707, got {}",
            results[2]
        );
        assert!(
            (results[3] - (-1.0)).abs() < 1e-5,
            "Expected -1.0, got {}",
            results[3]
        );
    }

    #[test]
    fn test_batch_cosine_similarity_8() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let targets: [&[f32]; 8] = [
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.5, 0.5, 0.0, 0.0],
            &[-1.0, 0.0, 0.0, 0.0],
            &[0.8, 0.6, 0.0, 0.0],
            &[0.6, 0.8, 0.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
        ];

        let results = batch_cosine_similarity_8(&query, targets);

        assert!((results[0] - 1.0).abs() < 1e-5);
        assert!(results[1].abs() < 1e-5);
        assert!((results[2] - 0.7071).abs() < 0.01);
        assert!((results[3] - (-1.0)).abs() < 1e-5);
        assert!((results[4] - 0.8).abs() < 1e-5);
        assert!((results[5] - 0.6).abs() < 1e-5);
        assert!((results[6] - 0.7071).abs() < 0.01);
        assert!(results[7].abs() < 1e-5);
    }

    #[test]
    fn test_batch_dot_products_4() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let targets: [&[f32]; 4] = [
            &[1.0, 1.0, 1.0, 1.0], // 10
            &[2.0, 2.0, 2.0, 2.0], // 20
            &[0.5, 0.5, 0.5, 0.5], // 5
            &[1.0, 0.0, 0.0, 0.0], // 1
        ];

        let results = batch_dot_products_4(&query, targets);

        assert!((results[0] - 10.0).abs() < 1e-5);
        assert!((results[1] - 20.0).abs() < 1e-5);
        assert!((results[2] - 5.0).abs() < 1e-5);
        assert!((results[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_dot_products_n() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let results = batch_dot_products(&query, &targets);

        assert_eq!(results.len(), 6);
        assert!((results[0] - 10.0).abs() < 1e-5);
        assert!((results[1] - 20.0).abs() < 1e-5);
        assert!((results[2] - 5.0).abs() < 1e-5);
        assert!((results[3] - 1.0).abs() < 1e-5);
        assert!((results[4] - 2.0).abs() < 1e-5);
        assert!((results[5] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_l2_norms_4() {
        // Note: Batch operations require all vectors to have the same length
        let vectors: [&[f32]; 4] = [
            &[3.0, 4.0, 0.0, 0.0], // 5.0 (sqrt(9+16))
            &[1.0, 0.0, 0.0, 0.0], // 1.0
            &[0.0, 0.0, 0.0, 0.0], // 0.0
            &[1.0, 1.0, 1.0, 1.0], // 2.0 (sqrt(1+1+1+1))
        ];

        let results = batch_l2_norms_4(vectors);

        assert!((results[0] - 5.0).abs() < 1e-5);
        assert!((results[1] - 1.0).abs() < 1e-5);
        assert!(results[2].abs() < 1e-5);
        assert!((results[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_l2_norms_n() {
        let v1 = vec![3.0, 4.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 0.0, 0.0];
        let v4 = vec![1.0, 1.0, 1.0, 1.0];
        let v5 = vec![2.0, 0.0, 0.0, 0.0];

        let vectors: Vec<&[f32]> = vec![&v1, &v2, &v3, &v4, &v5];
        let results = batch_l2_norms(&vectors);

        assert_eq!(results.len(), 5);
        assert!((results[0] - 5.0).abs() < 1e-5);
        assert!((results[1] - 1.0).abs() < 1e-5);
        assert!(results[2].abs() < 1e-5);
        assert!((results[3] - 2.0).abs() < 1e-5);
        assert!((results[4] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_similarity_search() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0, 0.0],  // 1.0
            vec![0.9, 0.1, 0.0, 0.0],  // ~0.99
            vec![0.0, 1.0, 0.0, 0.0],  // 0.0
            vec![0.5, 0.5, 0.0, 0.0],  // ~0.707
            vec![-1.0, 0.0, 0.0, 0.0], // -1.0
        ];

        // Get top 3 with threshold 0.5
        let results = batch_similarity_search(&query, &candidates, 3, 0.5);

        assert_eq!(results.len(), 3);
        // Should be sorted by score descending
        assert_eq!(results[0].0, 0); // Index 0 with score 1.0
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_cosine_similarity_large_vectors() {
        // Test with 384-dim vectors (common embedding size)
        let size = 384;
        let query: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();

        let t0: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        let t1: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).cos()).collect();
        let t2: Vec<f32> = (0..size).map(|i| (i as f32 * 0.02).sin()).collect();
        let t3: Vec<f32> = (0..size).map(|_| 0.1).collect();

        let targets: [&[f32]; 4] = [&t0, &t1, &t2, &t3];
        let results = batch_cosine_similarity_4(&query, targets);

        // First should be identical (score ~1.0)
        assert!(
            (results[0] - 1.0).abs() < 1e-4,
            "Identical vectors should have similarity 1.0"
        );
        // Others should be in valid range
        for (i, &score) in results.iter().enumerate() {
            assert!(
                score >= -1.0 && score <= 1.0,
                "Score {} out of range: {}",
                i,
                score
            );
        }
    }

    #[test]
    fn test_batch_operations_consistency() {
        // Verify batch operations produce same results as single operations
        let query: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let t0: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).cos()).collect();
        let t1: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
        let t2: Vec<f32> = (0..64).map(|i| (i as f32 * 0.15).cos()).collect();
        let t3: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).sin()).collect();

        // Single operation results
        let single_results = [
            cosine_similarity(&query, &t0),
            cosine_similarity(&query, &t1),
            cosine_similarity(&query, &t2),
            cosine_similarity(&query, &t3),
        ];

        // Batch operation results
        let batch_results = batch_cosine_similarity_4(&query, [&t0, &t1, &t2, &t3]);

        // Compare
        for i in 0..4 {
            assert!(
                (single_results[i] - batch_results[i]).abs() < 1e-5,
                "Mismatch at index {}: single={}, batch={}",
                i,
                single_results[i],
                batch_results[i]
            );
        }
    }
}
