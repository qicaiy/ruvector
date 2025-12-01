/**
 * SIMD Optimization Benchmark
 *
 * Compares SIMD-optimized vs naive implementations
 */

import {
  detectSIMDCapabilities,
  getSIMDHints,
  dotProduct,
  vectorAddInPlace,
  vectorScaleInPlace,
  normalizeL2InPlace,
  softmaxInPlace,
  padForSIMD,
  hrTimeUs,
  warmUp,
} from '../src/simd-utils.js';
import type { SIMDLevel } from '../src/types.js';

// ============================================================================
// Naive Reference Implementations
// ============================================================================

function naiveDotProduct(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function naiveVectorAdd(a: Float32Array, b: Float32Array, scale: number): Float32Array {
  const result = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] + b[i] * scale;
  }
  return result;
}

function naiveSoftmax(v: Float32Array): Float32Array {
  const result = new Float32Array(v.length);
  let max = v[0];
  for (let i = 1; i < v.length; i++) {
    if (v[i] > max) max = v[i];
  }
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    result[i] = Math.exp(v[i] - max);
    sum += result[i];
  }
  for (let i = 0; i < v.length; i++) {
    result[i] /= sum;
  }
  return result;
}

// ============================================================================
// Benchmark Utilities
// ============================================================================

function randomFloat32Array(size: number): Float32Array {
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = (Math.random() - 0.5) * 2;
  }
  return arr;
}

interface BenchResult {
  name: string;
  size: number;
  naiveUs: number;
  optimizedUs: number;
  speedup: number;
}

function runBench(
  name: string,
  size: number,
  naiveFn: () => void,
  optimizedFn: () => void,
  iterations: number = 10000
): BenchResult {
  // Warmup
  warmUp(naiveFn, 100);
  warmUp(optimizedFn, 100);

  // Benchmark naive
  const naiveStart = hrTimeUs();
  for (let i = 0; i < iterations; i++) {
    naiveFn();
  }
  const naiveTime = (hrTimeUs() - naiveStart) / iterations;

  // Benchmark optimized
  const optStart = hrTimeUs();
  for (let i = 0; i < iterations; i++) {
    optimizedFn();
  }
  const optTime = (hrTimeUs() - optStart) / iterations;

  return {
    name,
    size,
    naiveUs: naiveTime,
    optimizedUs: optTime,
    speedup: naiveTime / optTime,
  };
}

// ============================================================================
// Main Benchmark
// ============================================================================

async function main() {
  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║              SIMD Optimization Benchmark                      ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  // System Info
  const caps = detectSIMDCapabilities();
  console.log('System Information:');
  console.log(`  CPU Model:    ${caps.cpuModel}`);
  console.log(`  CPU Vendor:   ${caps.cpuVendor}`);
  console.log(`  Cores:        ${caps.cores}`);
  console.log(`  SIMD Support: ${caps.available.join(', ')}`);
  console.log(`  Recommended:  ${caps.recommended}\n`);

  // SIMD Hints
  const hints = getSIMDHints(caps.recommended);
  console.log('SIMD Hints:');
  console.log(`  Vector Width: ${hints.vectorWidth} bits`);
  console.log(`  Batch Size:   ${hints.batchSize}`);
  console.log(`  Alignment:    ${hints.alignment} bytes\n`);

  const results: BenchResult[] = [];
  const sizes = [64, 128, 256, 384, 512, 768, 1024, 2048];

  // ─────────────────────────────────────────────────────────────────
  // Dot Product Benchmark
  // ─────────────────────────────────────────────────────────────────
  console.log('📊 Dot Product Performance:\n');
  console.log('Size     | Naive (μs) | Optimized (μs) | Speedup');
  console.log('---------|------------|----------------|--------');

  for (const size of sizes) {
    const a = randomFloat32Array(size);
    const b = randomFloat32Array(size);

    const result = runBench(
      'Dot Product',
      size,
      () => naiveDotProduct(a, b),
      () => dotProduct(a, b)
    );
    results.push(result);

    console.log(
      `${size.toString().padEnd(8)} | ${result.naiveUs.toFixed(3).padStart(10)} | ${result.optimizedUs.toFixed(3).padStart(14)} | ${result.speedup.toFixed(2)}x`
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // Vector Addition Benchmark
  // ─────────────────────────────────────────────────────────────────
  console.log('\n📊 Vector Addition Performance:\n');
  console.log('Size     | Naive (μs) | Optimized (μs) | Speedup');
  console.log('---------|------------|----------------|--------');

  for (const size of sizes) {
    const a = randomFloat32Array(size);
    const b = randomFloat32Array(size);

    const result = runBench(
      'Vector Add',
      size,
      () => naiveVectorAdd(a, b, 1.0),
      () => {
        const target = new Float32Array(a);
        vectorAddInPlace(target, b, 1.0);
      }
    );
    results.push(result);

    console.log(
      `${size.toString().padEnd(8)} | ${result.naiveUs.toFixed(3).padStart(10)} | ${result.optimizedUs.toFixed(3).padStart(14)} | ${result.speedup.toFixed(2)}x`
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // Softmax Benchmark
  // ─────────────────────────────────────────────────────────────────
  console.log('\n📊 Softmax Performance:\n');
  console.log('Size     | Naive (μs) | Optimized (μs) | Speedup');
  console.log('---------|------------|----------------|--------');

  for (const size of sizes.filter(s => s <= 1024)) {
    const v = randomFloat32Array(size);

    const result = runBench(
      'Softmax',
      size,
      () => naiveSoftmax(v),
      () => {
        const copy = new Float32Array(v);
        softmaxInPlace(copy);
      }
    );
    results.push(result);

    console.log(
      `${size.toString().padEnd(8)} | ${result.naiveUs.toFixed(3).padStart(10)} | ${result.optimizedUs.toFixed(3).padStart(14)} | ${result.speedup.toFixed(2)}x`
    );
  }

  // ─────────────────────────────────────────────────────────────────
  // Memory Alignment Impact
  // ─────────────────────────────────────────────────────────────────
  console.log('\n📊 Memory Alignment Impact (dim=1024):\n');

  const unalignedSize = 1023; // Not a multiple of SIMD width
  const alignedSize = 1024;   // Multiple of SIMD width

  const unaligned = randomFloat32Array(unalignedSize);
  const aligned = randomFloat32Array(alignedSize);
  const paddedUnaligned = padForSIMD(unaligned, hints.vectorWidth);

  const unalignedResult = runBench(
    'Unaligned',
    unalignedSize,
    () => naiveDotProduct(unaligned, unaligned),
    () => dotProduct(unaligned, unaligned)
  );

  const alignedResult = runBench(
    'Aligned',
    alignedSize,
    () => naiveDotProduct(aligned, aligned),
    () => dotProduct(aligned, aligned)
  );

  const paddedResult = runBench(
    'Padded',
    paddedUnaligned.length,
    () => naiveDotProduct(paddedUnaligned, paddedUnaligned),
    () => dotProduct(paddedUnaligned, paddedUnaligned)
  );

  console.log('Case      | Size | Optimized (μs) | Speedup');
  console.log('----------|------|----------------|--------');
  console.log(`Unaligned | ${unalignedSize.toString().padStart(4)} | ${unalignedResult.optimizedUs.toFixed(3).padStart(14)} | ${unalignedResult.speedup.toFixed(2)}x`);
  console.log(`Aligned   | ${alignedSize.toString().padStart(4)} | ${alignedResult.optimizedUs.toFixed(3).padStart(14)} | ${alignedResult.speedup.toFixed(2)}x`);
  console.log(`Padded    | ${paddedUnaligned.length.toString().padStart(4)} | ${paddedResult.optimizedUs.toFixed(3).padStart(14)} | ${paddedResult.speedup.toFixed(2)}x`);

  // ─────────────────────────────────────────────────────────────────
  // Summary
  // ─────────────────────────────────────────────────────────────────
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('                          SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const avgSpeedup = results.reduce((s, r) => s + r.speedup, 0) / results.length;
  const maxSpeedup = results.reduce((a, b) => a.speedup > b.speedup ? a : b);
  const minSpeedup = results.reduce((a, b) => a.speedup < b.speedup ? a : b);

  console.log(`Average Speedup: ${avgSpeedup.toFixed(2)}x`);
  console.log(`Max Speedup:     ${maxSpeedup.speedup.toFixed(2)}x (${maxSpeedup.name}, size=${maxSpeedup.size})`);
  console.log(`Min Speedup:     ${minSpeedup.speedup.toFixed(2)}x (${minSpeedup.name}, size=${minSpeedup.size})`);

  console.log('\n📋 OPTIMIZATION TIPS:\n');
  console.log(`  1. Use vector sizes that are multiples of ${hints.batchSize} for best performance`);
  console.log(`  2. Align memory to ${hints.alignment} bytes when possible`);
  console.log(`  3. Batch operations to amortize function call overhead`);
  console.log(`  4. Use native Rust bindings for maximum SIMD utilization`);
  console.log(`  5. Enable AVX2/AVX-512 in Rust builds with RUSTFLAGS="-C target-cpu=native"`);
}

main().catch(console.error);
