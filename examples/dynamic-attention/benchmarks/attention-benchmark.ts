/**
 * Attention Mechanism Benchmark
 *
 * Detailed benchmarks for each attention type
 */

import { createPipeline } from '../src/dynamic-attention.js';
import { hrTimeUs, detectSIMDCapabilities } from '../src/simd-utils.js';
import type { AttentionType } from '../src/types.js';

// ============================================================================
// Test Data Generation
// ============================================================================

function randomFloat32Array(size: number): Float32Array {
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = (Math.random() - 0.5) * 2;
  }
  // Normalize for stable attention
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  for (let i = 0; i < size; i++) {
    arr[i] /= norm;
  }
  return arr;
}

function generateTestData(dim: number, contextSize: number, numCandidates: number) {
  return {
    query: randomFloat32Array(dim),
    context: Array.from({ length: contextSize }, () => randomFloat32Array(dim)),
    candidates: Array.from({ length: numCandidates }, (_, i) => ({
      id: `model-${i}`,
      embedding: randomFloat32Array(dim),
      successRate: 0.7 + Math.random() * 0.3,
      avgLatency: 50 + Math.random() * 200,
    })),
  };
}

// ============================================================================
// Benchmark Runner
// ============================================================================

interface AttentionBenchmarkResult {
  type: AttentionType;
  dim: number;
  contextSize: number;
  iterations: number;
  meanLatencyUs: number;
  p50Us: number;
  p95Us: number;
  p99Us: number;
  stdDevUs: number;
  throughputOpsPerSec: number;
}

async function benchmarkAttention(
  type: AttentionType,
  dim: number,
  contextSize: number,
  iterations: number = 1000
): Promise<AttentionBenchmarkResult> {
  const pipeline = createPipeline({
    dim,
    numHeads: Math.min(8, dim / 32),
    attentionType: type,
  });

  const { query, context, candidates } = generateTestData(dim, contextSize, 5);

  // Warmup
  for (let i = 0; i < 100; i++) {
    await pipeline.process({ embedding: query, context }, candidates);
  }

  // Benchmark
  const latencies: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = hrTimeUs();
    await pipeline.process({ embedding: query, context }, candidates);
    latencies.push(hrTimeUs() - start);
  }

  latencies.sort((a, b) => a - b);
  const mean = latencies.reduce((a, b) => a + b, 0) / iterations;
  const variance = latencies.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / iterations;

  return {
    type,
    dim,
    contextSize,
    iterations,
    meanLatencyUs: mean,
    p50Us: latencies[Math.floor(iterations * 0.5)],
    p95Us: latencies[Math.floor(iterations * 0.95)],
    p99Us: latencies[Math.floor(iterations * 0.99)],
    stdDevUs: Math.sqrt(variance),
    throughputOpsPerSec: 1_000_000 / mean,
  };
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║         Attention Mechanism Benchmark Suite                   ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  const simd = detectSIMDCapabilities();
  console.log(`System: ${simd.cpuModel}`);
  console.log(`SIMD: ${simd.recommended}\n`);

  const attentionTypes: AttentionType[] = [
    'dot-product',
    'multi-head',
    'hyperbolic',
    'flash',
    'linear',
    'local-global',
    'moe',
  ];

  const dims = [128, 256, 384, 512];
  const contextSizes = [4, 8, 16, 32];

  const results: AttentionBenchmarkResult[] = [];

  // Test each attention type with various dimensions
  for (const type of attentionTypes) {
    console.log(`\n📊 Benchmarking ${type.toUpperCase()}...\n`);

    for (const dim of dims) {
      const result = await benchmarkAttention(type, dim, 8, 500);
      results.push(result);

      console.log(`  dim=${dim}: ${result.meanLatencyUs.toFixed(1)}μs (p95: ${result.p95Us.toFixed(1)}μs) - ${result.throughputOpsPerSec.toFixed(0)} ops/s`);
    }
  }

  // Context size scaling for multi-head
  console.log('\n📊 Context Size Scaling (multi-head, dim=384)...\n');
  for (const contextSize of contextSizes) {
    const result = await benchmarkAttention('multi-head', 384, contextSize, 500);
    results.push(result);
    console.log(`  context=${contextSize}: ${result.meanLatencyUs.toFixed(1)}μs (p95: ${result.p95Us.toFixed(1)}μs)`);
  }

  // Summary table
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('                         SUMMARY TABLE');
  console.log('═══════════════════════════════════════════════════════════════\n');

  console.log('Attention Type     | Dim  | Mean (μs) | P95 (μs) | Throughput');
  console.log('-------------------|------|-----------|----------|------------');

  for (const r of results.filter(r => r.contextSize === 8)) {
    console.log(
      `${r.type.padEnd(18)} | ${r.dim.toString().padStart(4)} | ${r.meanLatencyUs.toFixed(1).padStart(9)} | ${r.p95Us.toFixed(1).padStart(8)} | ${r.throughputOpsPerSec.toFixed(0).padStart(10)}`
    );
  }

  // Recommendations
  console.log('\n📋 RECOMMENDATIONS:\n');

  const dim384Results = results.filter(r => r.dim === 384 && r.contextSize === 8);
  const fastest = dim384Results.reduce((a, b) => a.meanLatencyUs < b.meanLatencyUs ? a : b);
  const mostStable = dim384Results.reduce((a, b) => a.stdDevUs < b.stdDevUs ? a : b);

  console.log(`  Fastest:     ${fastest.type} (${fastest.meanLatencyUs.toFixed(1)}μs)`);
  console.log(`  Most Stable: ${mostStable.type} (σ=${mostStable.stdDevUs.toFixed(1)}μs)`);
  console.log(`  Best P99:    ${dim384Results.reduce((a, b) => a.p99Us < b.p99Us ? a : b).type}`);

  console.log('\n  Use Case Recommendations:');
  console.log('    - Low latency:     dot-product or linear');
  console.log('    - Hierarchical:    hyperbolic');
  console.log('    - Long sequences:  flash or local-global');
  console.log('    - Complex tasks:   moe or multi-head');
}

main().catch(console.error);
