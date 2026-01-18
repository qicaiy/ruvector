//! Custom SIMD intrinsics for performance-critical operations
//!
//! This module provides hand-optimized SIMD implementations:
//! - AVX2/AVX-512 for x86_64 processors
//! - NEON for ARM64/Apple Silicon processors (M1/M2/M3/M4)
//!
//! Distance calculations and other vectorized operations are automatically
//! dispatched to the optimal implementation based on the target architecture.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SIMD-optimized euclidean distance
/// Uses AVX2 on x86_64, NEON on ARM64/Apple Silicon, falls back to scalar otherwise
#[inline]
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { euclidean_distance_avx2_impl(a, b) }
        } else {
            euclidean_distance_scalar(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { euclidean_distance_neon_impl(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_distance_scalar(a, b)
    }
}

/// Legacy alias for backward compatibility
#[inline]
pub fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_simd(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    // SECURITY: Ensure both arrays have the same length to prevent out-of-bounds access
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    // Process 8 floats at a time
    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;

        // Load 8 floats from each array
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

        // Compute difference: (a - b)
        let diff = _mm256_sub_ps(va, vb);

        // Square the difference: (a - b)^2
        let sq = _mm256_mul_ps(diff, diff);

        // Accumulate
        sum = _mm256_add_ps(sum, sq);
    }

    // Horizontal sum of the 8 floats in the AVX register
    let sum_arr: [f32; 8] = std::mem::transmute(sum);
    let mut total = sum_arr.iter().sum::<f32>();

    // Handle remaining elements (if len not divisible by 8)
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}

// ============================================================================
// NEON implementations for ARM64/Apple Silicon (M1/M2/M3/M4)
// ============================================================================

/// NEON-optimized euclidean distance for ARM64
/// Processes 4 floats at a time using 128-bit NEON registers
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn euclidean_distance_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 floats at a time with NEON
    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));

        // Compute difference: (a - b)
        let diff = vsubq_f32(va, vb);

        // Square and accumulate: sum += (a - b)^2
        sum = vfmaq_f32(sum, diff, diff);
    }

    // Horizontal sum of the 4 floats
    let mut total = vaddvq_f32(sum);

    // Handle remaining elements
    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}

/// NEON-optimized dot product for ARM64
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_product_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));

        // Fused multiply-add: sum += a * b
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut total = vaddvq_f32(sum);

    for i in (chunks * 4)..len {
        total += a[i] * b[i];
    }

    total
}

/// NEON-optimized cosine similarity for ARM64
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_similarity_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut dot = vdupq_n_f32(0.0);
    let mut norm_a = vdupq_n_f32(0.0);
    let mut norm_b = vdupq_n_f32(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));

        // Dot product
        dot = vfmaq_f32(dot, va, vb);

        // Norms (squared)
        norm_a = vfmaq_f32(norm_a, va, va);
        norm_b = vfmaq_f32(norm_b, vb, vb);
    }

    let mut dot_sum = vaddvq_f32(dot);
    let mut norm_a_sum = vaddvq_f32(norm_a);
    let mut norm_b_sum = vaddvq_f32(norm_b);

    for i in (chunks * 4)..len {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }

    dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
}

/// NEON-optimized Manhattan distance for ARM64
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn manhattan_distance_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));

        // Absolute difference
        let diff = vsubq_f32(va, vb);
        let abs_diff = vabsq_f32(diff);
        sum = vaddq_f32(sum, abs_diff);
    }

    let mut total = vaddvq_f32(sum);

    for i in (chunks * 4)..len {
        total += (a[i] - b[i]).abs();
    }

    total
}

// ============================================================================
// Public API with architecture dispatch
// ============================================================================

/// SIMD-optimized dot product
/// Uses AVX2 on x86_64, NEON on ARM64/Apple Silicon
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { dot_product_avx2_impl(a, b) }
        } else {
            dot_product_scalar(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_neon_impl(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_product_scalar(a, b)
    }
}

/// Legacy alias for backward compatibility
#[inline]
pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    dot_product_simd(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    // SECURITY: Ensure both arrays have the same length to prevent out-of-bounds access
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }

    let sum_arr: [f32; 8] = std::mem::transmute(sum);
    let mut total = sum_arr.iter().sum::<f32>();

    for i in (chunks * 8)..len {
        total += a[i] * b[i];
    }

    total
}

/// SIMD-optimized cosine similarity
/// Uses AVX2 on x86_64, NEON on ARM64/Apple Silicon
#[inline]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { cosine_similarity_avx2_impl(a, b) }
        } else {
            cosine_similarity_scalar(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { cosine_similarity_neon_impl(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        cosine_similarity_scalar(a, b)
    }
}

/// Legacy alias for backward compatibility
#[inline]
pub fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity_simd(a, b)
}

/// SIMD-optimized Manhattan distance
/// Uses NEON on ARM64/Apple Silicon, scalar on other platforms
#[inline]
pub fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { manhattan_distance_neon_impl(a, b) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        manhattan_distance_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cosine_similarity_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    // SECURITY: Ensure both arrays have the same length to prevent out-of-bounds access
    assert_eq!(a.len(), b.len(), "Input arrays must have the same length");

    let len = a.len();
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

        // Dot product
        dot = _mm256_add_ps(dot, _mm256_mul_ps(va, vb));

        // Norms
        norm_a = _mm256_add_ps(norm_a, _mm256_mul_ps(va, va));
        norm_b = _mm256_add_ps(norm_b, _mm256_mul_ps(vb, vb));
    }

    let dot_arr: [f32; 8] = std::mem::transmute(dot);
    let norm_a_arr: [f32; 8] = std::mem::transmute(norm_a);
    let norm_b_arr: [f32; 8] = std::mem::transmute(norm_b);

    let mut dot_sum = dot_arr.iter().sum::<f32>();
    let mut norm_a_sum = norm_a_arr.iter().sum::<f32>();
    let mut norm_b_sum = norm_b_arr.iter().sum::<f32>();

    for i in (chunks * 8)..len {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }

    dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
}

// Scalar fallback implementations

fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = euclidean_distance_simd(&a, &b);
        let expected = euclidean_distance_scalar(&a, &b);

        assert!(
            (result - expected).abs() < 0.001,
            "SIMD result {} differs from scalar result {}",
            result,
            expected
        );
    }

    #[test]
    fn test_euclidean_distance_large() {
        // Test with 128-dim vectors (common embedding size)
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1) + 0.5).collect();

        let result = euclidean_distance_simd(&a, &b);
        let expected = euclidean_distance_scalar(&a, &b);

        assert!(
            (result - expected).abs() < 0.01,
            "Large vector: SIMD {} vs scalar {}",
            result,
            expected
        );
    }

    #[test]
    fn test_dot_product_simd() {
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];

        let result = dot_product_simd(&a, &b);
        assert!((result - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_dot_product_large() {
        let a: Vec<f32> = (0..256).map(|i| (i % 10) as f32).collect();
        let b: Vec<f32> = (0..256).map(|i| ((i + 5) % 10) as f32).collect();

        let result = dot_product_simd(&a, &b);
        let expected = dot_product_scalar(&a, &b);

        assert!(
            (result - expected).abs() < 0.1,
            "Large dot product: SIMD {} vs scalar {}",
            result,
            expected
        );
    }

    #[test]
    fn test_cosine_similarity_simd() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let result = cosine_similarity_simd(&a, &b);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        let result = cosine_similarity_simd(&a, &b);
        assert!(
            result.abs() < 0.001,
            "Orthogonal vectors should have ~0 similarity, got {}",
            result
        );
    }

    #[test]
    fn test_manhattan_distance_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = manhattan_distance_simd(&a, &b);
        let expected = manhattan_distance_scalar(&a, &b);

        assert!(
            (result - expected).abs() < 0.001,
            "Manhattan: SIMD {} vs scalar {}",
            result,
            expected
        );
        assert!((result - 16.0).abs() < 0.001); // |4| + |4| + |4| + |4| = 16
    }

    #[test]
    fn test_non_aligned_lengths() {
        // Test vectors not aligned to SIMD width (4 for NEON, 8 for AVX2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 7 elements
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = euclidean_distance_simd(&a, &b);
        let expected = euclidean_distance_scalar(&a, &b);

        assert!(
            (result - expected).abs() < 0.001,
            "Non-aligned: SIMD {} vs scalar {}",
            result,
            expected
        );
    }

    // Legacy function tests (ensure backward compatibility)
    #[test]
    fn test_legacy_avx2_aliases() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // These should work identically to the _simd versions
        let _ = euclidean_distance_avx2(&a, &b);
        let _ = dot_product_avx2(&a, &b);
        let _ = cosine_similarity_avx2(&a, &b);
    }
}
