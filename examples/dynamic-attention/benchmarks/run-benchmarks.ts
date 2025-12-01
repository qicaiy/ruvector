/**
 * Comprehensive Benchmark Suite
 *
 * Benchmarks all components of the Dynamic Attention pipeline
 */

import { createPipeline } from '../src/dynamic-attention.js';
import {
  hrTimeUs,
  detectSIMDCapabilities,
  dotProduct,
  normalizeL2InPlace,
  softmaxInPlace,
} from '../src/simd-utils.js';
import type { BenchmarkResult, AttentionType, RoutingCandidate } from '../src/types.js';

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

function generateCandidates(count: number, dim: number): RoutingCandidate[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `candidate-${i}`,
    embedding: randomFloat32Array(dim),
    successRate: Math.random() * 0.3 + 0.7,
    avgLatency: Math.random() * 200 + 50,
    cost: Math.random() * 0.1,
  }));
}

async function runBenchmark(
  name: string,
  fn: () => void | Promise<void>,
  iterations: number = 1000,
  warmupIterations: number = 100
): Promise<BenchmarkResult> {
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await fn();
  }

  // Force GC if available
  if (global.gc) {
    global.gc();
  }

  // Collect samples
  const samples: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = hrTimeUs();
    await fn();
    const end = hrTimeUs();
    samples.push(end - start);
  }

  // Sort for percentiles
  samples.sort((a, b) => a - b);

  // Calculate statistics
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const variance = samples.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / samples.length;
  const stdDev = Math.sqrt(variance);

  const memoryUsage = process.memoryUsage();

  return {
    name,
    iterations,
    meanLatencyUs: mean,
    stdDevUs: stdDev,
    p50Us: samples[Math.floor(iterations * 0.5)],
    p95Us: samples[Math.floor(iterations * 0.95)],
    p99Us: samples[Math.floor(iterations * 0.99)],
    minUs: samples[0],
    maxUs: samples[samples.length - 1],
    opsPerSecond: 1_000_000 / mean,
    memoryBytes: memoryUsage.heapUsed,
  };
}

function formatResult(result: BenchmarkResult): string {
  return `
┌─────────────────────────────────────────────────────────────┐
│ ${result.name.padEnd(59)} │
├─────────────────────────────────────────────────────────────┤
│ Iterations:    ${result.iterations.toString().padStart(10)}                              │
│ Mean Latency:  ${result.meanLatencyUs.toFixed(2).padStart(10)} μs                         │
│ Std Dev:       ${result.stdDevUs.toFixed(2).padStart(10)} μs                         │
│ P50:           ${result.p50Us.toFixed(2).padStart(10)} μs                         │
│ P95:           ${result.p95Us.toFixed(2).padStart(10)} μs                         │
│ P99:           ${result.p99Us.toFixed(2).padStart(10)} μs                         │
│ Min:           ${result.minUs.toFixed(2).padStart(10)} μs                         │
│ Max:           ${result.maxUs.toFixed(2).padStart(10)} μs                         │
│ Throughput:    ${result.opsPerSecond.toFixed(0).padStart(10)} ops/sec                    │
│ Memory:        ${(result.memoryBytes / 1024 / 1024).toFixed(2).padStart(10)} MB                          │
└─────────────────────────────────────────────────────────────┘`;
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

async function main() {
  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║       Dynamic Attention Benchmark Suite                       ║');
  console.log('║       FastGRNN + Attention Performance Analysis              ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  // Print system info
  const simd = detectSIMDCapabilities();
  console.log('System Information:');
  console.log(`  CPU: ${simd.cpuModel}`);
  console.log(`  Cores: ${simd.cores}`);
  console.log(`  SIMD Support: ${simd.available.join(', ')}`);
  console.log(`  Recommended: ${simd.recommended}`);
  console.log('');

  const results: BenchmarkResult[] = [];

  // ─────────────────────────────────────────────────────────────────
  // 1. SIMD Vector Operations
  // ─────────────────────────────────────────────────────────────────
  console.log('📊 Benchmarking SIMD Vector Operations...\n');

  const dims = [128, 256, 384, 512, 1024];

  for (const dim of dims) {
    const a = randomFloat32Array(dim);
    const b = randomFloat32Array(dim);

    const result = await runBenchmark(
      `Dot Product (dim=${dim})`,
      () => dotProduct(a, b),
      10000
    );
    results.push(result);
    console.log(formatResult(result));
  }

  // Softmax benchmark
  for (const dim of [64, 128, 256]) {
    const v = randomFloat32Array(dim);
    const result = await runBenchmark(
      `Softmax (dim=${dim})`,
      () => {
        const copy = new Float32Array(v);
        softmaxInPlace(copy);
      },
      10000
    );
    results.push(result);
    console.log(formatResult(result));
  }

  // ─────────────────────────────────────────────────────────────────
  // 2. Attention Mechanisms
  // ─────────────────────────────────────────────────────────────────
  console.log('\n📊 Benchmarking Attention Mechanisms...\n');

  const attentionTypes: AttentionType[] = [
    'dot-product',
    'multi-head',
    'hyperbolic',
    'flash',
    'linear',
    'moe',
  ];

  for (const attentionType of attentionTypes) {
    const pipeline = createPipeline({
      dim: 384,
      numHeads: 8,
      attentionType,
    });

    const query = randomFloat32Array(384);
    const context = Array.from({ length: 8 }, () => randomFloat32Array(384));
    const candidates = generateCandidates(10, 384);

    const result = await runBenchmark(
      `Attention: ${attentionType}`,
      async () => {
        await pipeline.process({ embedding: query, context }, candidates);
      },
      1000
    );
    results.push(result);
    console.log(formatResult(result));
  }

  // ─────────────────────────────────────────────────────────────────
  // 3. Full Pipeline Scaling
  // ─────────────────────────────────────────────────────────────────
  console.log('\n📊 Benchmarking Pipeline Scaling...\n');

  const candidateCounts = [1, 5, 10, 25, 50, 100];

  for (const count of candidateCounts) {
    const pipeline = createPipeline({ dim: 384, numHeads: 8 });
    const query = randomFloat32Array(384);
    const context = Array.from({ length: 4 }, () => randomFloat32Array(384));
    const candidates = generateCandidates(count, 384);

    const result = await runBenchmark(
      `Pipeline (${count} candidates)`,
      async () => {
        await pipeline.process({ embedding: query, context }, candidates);
      },
      1000
    );
    results.push(result);
    console.log(formatResult(result));
  }

  // ─────────────────────────────────────────────────────────────────
  // 4. Dimension Scaling
  // ─────────────────────────────────────────────────────────────────
  console.log('\n📊 Benchmarking Dimension Scaling...\n');

  const embedDims = [128, 256, 384, 512, 768, 1024];

  for (const dim of embedDims) {
    const pipeline = createPipeline({ dim, numHeads: 8 });
    const query = randomFloat32Array(dim);
    const context = Array.from({ length: 4 }, () => randomFloat32Array(dim));
    const candidates = generateCandidates(10, dim);

    const result = await runBenchmark(
      `Pipeline (dim=${dim})`,
      async () => {
        await pipeline.process({ embedding: query, context }, candidates);
      },
      1000
    );
    results.push(result);
    console.log(formatResult(result));
  }

  // ─────────────────────────────────────────────────────────────────
  // Summary
  // ─────────────────────────────────────────────────────────────────
  console.log('\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║                     BENCHMARK SUMMARY                         ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  // Find fastest attention type
  const attentionResults = results.filter(r => r.name.startsWith('Attention:'));
  const fastestAttention = attentionResults.reduce((a, b) =>
    a.meanLatencyUs < b.meanLatencyUs ? a : b
  );

  console.log(`Fastest Attention: ${fastestAttention.name} (${fastestAttention.meanLatencyUs.toFixed(2)} μs)`);

  // Print scaling analysis
  const scalingResults = results.filter(r => r.name.includes('candidates'));
  if (scalingResults.length >= 2) {
    const first = scalingResults[0];
    const last = scalingResults[scalingResults.length - 1];
    const scalingFactor = last.meanLatencyUs / first.meanLatencyUs;
    const candidatesFactor = 100; // Assuming 1 to 100 scaling

    console.log(`\nScaling Analysis:`);
    console.log(`  1 candidate:   ${first.meanLatencyUs.toFixed(2)} μs`);
    console.log(`  100 candidates: ${last.meanLatencyUs.toFixed(2)} μs`);
    console.log(`  Scaling factor: ${scalingFactor.toFixed(2)}x for ${candidatesFactor}x candidates`);
  }

  // Memory analysis
  const avgMemory = results.reduce((sum, r) => sum + r.memoryBytes, 0) / results.length;
  console.log(`\nMemory Usage: ${(avgMemory / 1024 / 1024).toFixed(2)} MB average`);

  // Export results
  const output = {
    timestamp: new Date().toISOString(),
    system: simd,
    results: results.map(r => ({
      name: r.name,
      meanUs: r.meanLatencyUs,
      p95Us: r.p95Us,
      p99Us: r.p99Us,
      opsPerSec: r.opsPerSecond,
    })),
  };

  console.log('\n📁 Results exported to stdout as JSON:');
  console.log(JSON.stringify(output, null, 2));
}

main().catch(console.error);
