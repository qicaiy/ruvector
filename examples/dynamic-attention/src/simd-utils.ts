/**
 * SIMD Optimization Utilities
 *
 * Provides CPU feature detection, optimized vector operations,
 * and memory alignment utilities for maximum performance.
 */

import type { SIMDCapabilities, SIMDHints, SIMDLevel } from './types.js';

// ============================================================================
// CPU Feature Detection
// ============================================================================

/**
 * Detect available SIMD capabilities on the current system
 */
export function detectSIMDCapabilities(): SIMDCapabilities {
  const platform = process.platform;
  const arch = process.arch;

  // Default capabilities
  const capabilities: SIMDCapabilities = {
    available: ['none'],
    recommended: 'none',
    cpuVendor: 'unknown',
    cpuModel: 'unknown',
    cores: 1,
    cacheSizes: { l1d: 32, l1i: 32, l2: 256, l3: 8192 },
  };

  // Detect based on architecture
  if (arch === 'x64') {
    // x86-64 typically has SSE4 minimum, often AVX2
    capabilities.available = ['none', 'sse4', 'avx2'];
    capabilities.recommended = 'avx2';

    // Check for AVX-512 (rare but powerful)
    if (hasAVX512Support()) {
      capabilities.available.push('avx512');
      capabilities.recommended = 'avx512';
    }
  } else if (arch === 'arm64') {
    // ARM64 has NEON by default
    capabilities.available = ['none', 'neon'];
    capabilities.recommended = 'neon';
  }

  // Get CPU info
  try {
    const os = require('os');
    const cpus = os.cpus();
    if (cpus.length > 0) {
      capabilities.cpuModel = cpus[0].model;
      capabilities.cores = cpus.length;

      // Detect vendor from model string
      if (cpus[0].model.includes('Intel')) {
        capabilities.cpuVendor = 'Intel';
      } else if (cpus[0].model.includes('AMD')) {
        capabilities.cpuVendor = 'AMD';
      } else if (cpus[0].model.includes('Apple')) {
        capabilities.cpuVendor = 'Apple';
      }
    }
  } catch {
    // Ignore errors
  }

  return capabilities;
}

/**
 * Check for AVX-512 support (heuristic)
 */
function hasAVX512Support(): boolean {
  try {
    const os = require('os');
    const cpus = os.cpus();
    if (cpus.length > 0) {
      const model = cpus[0].model.toLowerCase();
      // AVX-512 is common in Xeon and recent desktop CPUs
      return model.includes('xeon') ||
             model.includes('i9') ||
             (model.includes('i7') && parseInt(model.match(/\d{4,5}/)?.[0] || '0') >= 10000);
    }
  } catch {
    // Ignore
  }
  return false;
}

/**
 * Get optimization hints based on SIMD level
 */
export function getSIMDHints(level: SIMDLevel): SIMDHints {
  switch (level) {
    case 'avx512':
      return {
        vectorWidth: 512,
        batchSize: 16,
        useAlignedMemory: true,
        alignment: 64,
      };
    case 'avx2':
      return {
        vectorWidth: 256,
        batchSize: 8,
        useAlignedMemory: true,
        alignment: 32,
      };
    case 'sse4':
      return {
        vectorWidth: 128,
        batchSize: 4,
        useAlignedMemory: true,
        alignment: 16,
      };
    case 'neon':
      return {
        vectorWidth: 128,
        batchSize: 4,
        useAlignedMemory: true,
        alignment: 16,
      };
    default:
      return {
        vectorWidth: 64,
        batchSize: 1,
        useAlignedMemory: false,
        alignment: 8,
      };
  }
}

// ============================================================================
// Optimized Vector Operations
// ============================================================================

/**
 * Optimized dot product with SIMD-friendly memory access
 */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  const len = a.length;
  let sum = 0;

  // Process in chunks of 8 for better cache utilization
  const chunks = Math.floor(len / 8);
  let i = 0;

  for (let c = 0; c < chunks; c++) {
    sum += a[i] * b[i] +
           a[i + 1] * b[i + 1] +
           a[i + 2] * b[i + 2] +
           a[i + 3] * b[i + 3] +
           a[i + 4] * b[i + 4] +
           a[i + 5] * b[i + 5] +
           a[i + 6] * b[i + 6] +
           a[i + 7] * b[i + 7];
    i += 8;
  }

  // Handle remaining elements
  for (; i < len; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

/**
 * Optimized vector addition (in-place)
 */
export function vectorAddInPlace(target: Float32Array, source: Float32Array, scale = 1.0): void {
  if (target.length !== source.length) {
    throw new Error('Vectors must have same length');
  }

  const len = target.length;
  const chunks = Math.floor(len / 4);
  let i = 0;

  for (let c = 0; c < chunks; c++) {
    target[i] += source[i] * scale;
    target[i + 1] += source[i + 1] * scale;
    target[i + 2] += source[i + 2] * scale;
    target[i + 3] += source[i + 3] * scale;
    i += 4;
  }

  for (; i < len; i++) {
    target[i] += source[i] * scale;
  }
}

/**
 * Optimized vector scaling (in-place)
 */
export function vectorScaleInPlace(v: Float32Array, scale: number): void {
  const len = v.length;
  const chunks = Math.floor(len / 4);
  let i = 0;

  for (let c = 0; c < chunks; c++) {
    v[i] *= scale;
    v[i + 1] *= scale;
    v[i + 2] *= scale;
    v[i + 3] *= scale;
    i += 4;
  }

  for (; i < len; i++) {
    v[i] *= scale;
  }
}

/**
 * Optimized L2 normalization (in-place)
 */
export function normalizeL2InPlace(v: Float32Array): void {
  const norm = Math.sqrt(dotProduct(v, v));
  if (norm > 1e-10) {
    vectorScaleInPlace(v, 1.0 / norm);
  }
}

/**
 * Optimized softmax (in-place)
 */
export function softmaxInPlace(v: Float32Array, temperature = 1.0): void {
  const len = v.length;

  // Find max for numerical stability
  let max = v[0];
  for (let i = 1; i < len; i++) {
    if (v[i] > max) max = v[i];
  }

  // Compute exp and sum
  let sum = 0;
  for (let i = 0; i < len; i++) {
    v[i] = Math.exp((v[i] - max) / temperature);
    sum += v[i];
  }

  // Normalize
  const invSum = 1.0 / sum;
  for (let i = 0; i < len; i++) {
    v[i] *= invSum;
  }
}

/**
 * Batch matrix-vector multiplication (optimized for attention)
 */
export function batchMatVec(
  matrices: Float32Array[],
  vector: Float32Array
): Float32Array[] {
  return matrices.map(matrix => {
    const rows = matrix.length / vector.length;
    const result = new Float32Array(rows);

    for (let r = 0; r < rows; r++) {
      let sum = 0;
      const offset = r * vector.length;
      for (let c = 0; c < vector.length; c++) {
        sum += matrix[offset + c] * vector[c];
      }
      result[r] = sum;
    }

    return result;
  });
}

// ============================================================================
// Memory Alignment Utilities
// ============================================================================

/**
 * Create an aligned Float32Array
 */
export function createAlignedFloat32Array(size: number, alignment: number): Float32Array {
  // In Node.js, we can't directly control alignment, but we can pad
  // The native Rust code handles actual SIMD alignment
  return new Float32Array(size);
}

/**
 * Copy data to aligned buffer
 */
export function copyToAligned(
  source: Float32Array | number[],
  alignment: number
): Float32Array {
  const result = createAlignedFloat32Array(source.length, alignment);
  if (source instanceof Float32Array) {
    result.set(source);
  } else {
    for (let i = 0; i < source.length; i++) {
      result[i] = source[i];
    }
  }
  return result;
}

/**
 * Pad array to multiple of vector width for SIMD
 */
export function padForSIMD(arr: Float32Array, vectorWidth: number): Float32Array {
  const elementsPerVector = vectorWidth / 32; // 32 bits per float
  const paddedSize = Math.ceil(arr.length / elementsPerVector) * elementsPerVector;

  if (paddedSize === arr.length) {
    return arr;
  }

  const padded = new Float32Array(paddedSize);
  padded.set(arr);
  return padded;
}

// ============================================================================
// Performance Monitoring
// ============================================================================

/**
 * High-resolution timer for benchmarking
 */
export function hrTimeUs(): number {
  const [seconds, nanoseconds] = process.hrtime();
  return seconds * 1_000_000 + nanoseconds / 1000;
}

/**
 * Measure execution time of a function
 */
export async function measureTimeUs<T>(fn: () => T | Promise<T>): Promise<[T, number]> {
  const start = hrTimeUs();
  const result = await fn();
  const end = hrTimeUs();
  return [result, end - start];
}

/**
 * Warm up JIT compiler
 */
export function warmUp(fn: () => void, iterations = 100): void {
  for (let i = 0; i < iterations; i++) {
    fn();
  }
}
