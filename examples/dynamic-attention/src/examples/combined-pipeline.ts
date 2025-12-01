/**
 * Combined Pipeline Demo
 *
 * Full end-to-end demonstration of the Dynamic Attention system
 * integrating FastGRNN routing with attention-enhanced features
 */

import { createPipeline, DynamicAttentionPipeline } from '../dynamic-attention.js';
import { detectSIMDCapabilities, hrTimeUs } from '../simd-utils.js';
import type { RoutingCandidate, PipelineResult } from '../types.js';

// ============================================================================
// Real-World Scenario: Multi-Model Inference Router
// ============================================================================

async function multiModelRouterDemo() {
  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║        Production Multi-Model Inference Router               ║');
  console.log('║        FastGRNN + Attention Dynamic Routing                  ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  // System information
  const simd = detectSIMDCapabilities();
  console.log('System Configuration:');
  console.log(`  CPU:          ${simd.cpuModel}`);
  console.log(`  SIMD:         ${simd.recommended}`);
  console.log(`  Cores:        ${simd.cores}\n`);

  // Initialize pipeline
  const pipeline = createPipeline({
    dim: 384,
    numHeads: 8,
    hiddenDim: 64,
    attentionType: 'multi-head',
    enableSIMD: true,
    temperature: 0.8,
  });

  console.log('Pipeline Configuration:');
  const config = pipeline.getConfig();
  console.log(`  Dimension:    ${config.dim}`);
  console.log(`  Heads:        ${config.numHeads}`);
  console.log(`  Hidden:       ${config.hiddenDim}`);
  console.log(`  Attention:    ${config.attentionType}`);
  console.log(`  SIMD:         ${config.enableSIMD ? 'Enabled' : 'Disabled'}\n`);

  // Define model pool
  const modelPool: RoutingCandidate[] = [
    {
      id: 'gpt-4-turbo',
      embedding: generateModelEmbedding(384, 'gpt4-powerful-reasoning'),
      successRate: 0.96,
      avgLatency: 1800,
      cost: 0.01,
      capabilities: ['reasoning', 'code', 'math', 'analysis'],
    },
    {
      id: 'gpt-3.5-turbo',
      embedding: generateModelEmbedding(384, 'gpt35-fast-general'),
      successRate: 0.89,
      avgLatency: 400,
      cost: 0.0005,
      capabilities: ['general', 'chat', 'fast'],
    },
    {
      id: 'claude-3-opus',
      embedding: generateModelEmbedding(384, 'claude-deep-thinking'),
      successRate: 0.94,
      avgLatency: 2200,
      cost: 0.015,
      capabilities: ['reasoning', 'creative', 'nuanced'],
    },
    {
      id: 'claude-3-sonnet',
      embedding: generateModelEmbedding(384, 'claude-balanced'),
      successRate: 0.91,
      avgLatency: 1000,
      cost: 0.003,
      capabilities: ['balanced', 'code', 'writing'],
    },
    {
      id: 'claude-3-haiku',
      embedding: generateModelEmbedding(384, 'claude-fast-efficient'),
      successRate: 0.85,
      avgLatency: 250,
      cost: 0.00025,
      capabilities: ['fast', 'efficient', 'simple'],
    },
    {
      id: 'gemini-1.5-pro',
      embedding: generateModelEmbedding(384, 'gemini-multimodal'),
      successRate: 0.92,
      avgLatency: 1200,
      cost: 0.007,
      capabilities: ['multimodal', 'long-context', 'reasoning'],
    },
    {
      id: 'mixtral-8x7b',
      embedding: generateModelEmbedding(384, 'mixtral-open-moe'),
      successRate: 0.87,
      avgLatency: 600,
      cost: 0.0006,
      capabilities: ['open-source', 'code', 'general'],
    },
    {
      id: 'llama-3-70b',
      embedding: generateModelEmbedding(384, 'llama-open-powerful'),
      successRate: 0.88,
      avgLatency: 800,
      cost: 0.0008,
      capabilities: ['open-source', 'reasoning', 'chat'],
    },
  ];

  console.log('Model Pool (8 models):');
  console.log('┌───────────────────┬─────────┬──────────┬──────────┬────────────────────────┐');
  console.log('│ Model             │ Success │ Latency  │ Cost/1K  │ Capabilities           │');
  console.log('├───────────────────┼─────────┼──────────┼──────────┼────────────────────────┤');
  for (const m of modelPool) {
    console.log(
      `│ ${m.id.padEnd(17)} │ ${((m.successRate || 0) * 100).toFixed(0).padStart(5)}%  │ ${(m.avgLatency || 0).toString().padStart(6)}ms │ $${(m.cost || 0).toFixed(5).padStart(7)} │ ${(m.capabilities?.slice(0, 3).join(', ') || '').padEnd(22)} │`
    );
  }
  console.log('└───────────────────┴─────────┴──────────┴──────────┴────────────────────────┘\n');

  // Test queries
  const testQueries = [
    {
      name: 'Simple greeting',
      query: 'Hello, how are you today?',
      expectedComplexity: 'low',
    },
    {
      name: 'Code debugging',
      query: 'Debug this Python function that calculates Fibonacci numbers recursively',
      expectedComplexity: 'medium',
    },
    {
      name: 'Complex reasoning',
      query: 'Analyze the implications of quantum computing on current cryptographic standards',
      expectedComplexity: 'high',
    },
    {
      name: 'Creative writing',
      query: 'Write a short story about a robot discovering emotions',
      expectedComplexity: 'medium',
    },
    {
      name: 'Math problem',
      query: 'Solve this differential equation: dy/dx + 2xy = x',
      expectedComplexity: 'high',
    },
    {
      name: 'Quick fact',
      query: 'What is the capital of France?',
      expectedComplexity: 'low',
    },
  ];

  console.log('Routing Decisions:\n');
  console.log('┌─────────────────────┬────────────────────┬────────────┬─────────────┬───────────────┐');
  console.log('│ Query               │ Routed Model       │ Confidence │ Lightweight │ Latency (μs)  │');
  console.log('├─────────────────────┼────────────────────┼────────────┼─────────────┼───────────────┤');

  const results: PipelineResult[] = [];

  for (const test of testQueries) {
    // Generate query embedding (in production, use actual embedding model)
    const queryEmbedding = generateQueryEmbedding(384, test.query);

    // Optional: conversation context
    const context = [
      generateQueryEmbedding(384, 'previous message context'),
      generateQueryEmbedding(384, 'system prompt context'),
    ];

    const result = await pipeline.process(
      { embedding: queryEmbedding, context },
      modelPool
    );
    results.push(result);

    const best = result.decisions[0];
    console.log(
      `│ ${test.name.padEnd(19)} │ ${best.candidateId.padEnd(18)} │ ${(best.confidence * 100).toFixed(1).padStart(8)}%  │ ${best.useLightweight ? '     Yes' : '      No'}    │ ${result.metrics.totalLatencyUs.toFixed(0).padStart(11)}  │`
    );
  }

  console.log('└─────────────────────┴────────────────────┴────────────┴─────────────┴───────────────┘\n');

  // Performance summary
  const avgLatency = results.reduce((s, r) => s + r.metrics.totalLatencyUs, 0) / results.length;
  const avgAttention = results.reduce((s, r) => s + r.metrics.attentionLatencyUs, 0) / results.length;
  const avgFastgrnn = results.reduce((s, r) => s + r.metrics.fastgrnnLatencyUs, 0) / results.length;

  console.log('Performance Summary:');
  console.log(`  Average Pipeline Latency:  ${avgLatency.toFixed(0)} μs`);
  console.log(`  Average Attention Time:    ${avgAttention.toFixed(0)} μs`);
  console.log(`  Average FastGRNN Time:     ${avgFastgrnn.toFixed(0)} μs`);
  console.log(`  Estimated Throughput:      ${(1_000_000 / avgLatency).toFixed(0)} routes/sec\n`);

  // Cost savings analysis
  await costSavingsAnalysis(pipeline, modelPool, results);
}

// ============================================================================
// Cost Savings Analysis
// ============================================================================

async function costSavingsAnalysis(
  pipeline: DynamicAttentionPipeline,
  modelPool: RoutingCandidate[],
  routingResults: PipelineResult[]
) {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('                   Cost Savings Analysis');
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Simulate 10,000 queries
  const numQueries = 10000;
  const avgTokensPerQuery = 500;

  // Calculate costs with intelligent routing
  let routedCost = 0;
  let lightweightRoutes = 0;

  for (let i = 0; i < numQueries; i++) {
    // Use sample results distribution
    const result = routingResults[i % routingResults.length];
    const selectedModel = modelPool.find(m => m.id === result.decisions[0].candidateId);

    if (result.decisions[0].useLightweight) {
      // Use lightweight fallback (cheapest model)
      const cheapest = modelPool.reduce((a, b) => (a.cost || 1) < (b.cost || 1) ? a : b);
      routedCost += (cheapest.cost || 0) * avgTokensPerQuery / 1000;
      lightweightRoutes++;
    } else {
      routedCost += (selectedModel?.cost || 0) * avgTokensPerQuery / 1000;
    }
  }

  // Compare with always using most expensive model
  const mostExpensive = modelPool.reduce((a, b) => (a.cost || 0) > (b.cost || 0) ? a : b);
  const premiumCost = (mostExpensive.cost || 0) * avgTokensPerQuery * numQueries / 1000;

  // Compare with always using cheapest model
  const cheapest = modelPool.reduce((a, b) => (a.cost || 0) < (b.cost || 0) ? a : b);
  const budgetCost = (cheapest.cost || 0) * avgTokensPerQuery * numQueries / 1000;

  // Calculate average
  const avgCost = modelPool.reduce((s, m) => s + (m.cost || 0), 0) / modelPool.length;
  const randomCost = avgCost * avgTokensPerQuery * numQueries / 1000;

  console.log(`Scenario: ${numQueries.toLocaleString()} queries, ${avgTokensPerQuery} avg tokens/query\n`);

  console.log('Strategy               │ Total Cost    │ vs Premium   │ vs Random');
  console.log('───────────────────────┼───────────────┼──────────────┼────────────');
  console.log(`Always Premium         │ $${premiumCost.toFixed(2).padStart(11)} │      -       │ +${((premiumCost / randomCost - 1) * 100).toFixed(0)}%`);
  console.log(`Random Selection       │ $${randomCost.toFixed(2).padStart(11)} │ -${((1 - randomCost / premiumCost) * 100).toFixed(0)}%        │     -`);
  console.log(`Intelligent Routing    │ $${routedCost.toFixed(2).padStart(11)} │ -${((1 - routedCost / premiumCost) * 100).toFixed(0)}%        │ -${((1 - routedCost / randomCost) * 100).toFixed(0)}%`);
  console.log(`Always Budget          │ $${budgetCost.toFixed(2).padStart(11)} │ -${((1 - budgetCost / premiumCost) * 100).toFixed(0)}%        │ -${((1 - budgetCost / randomCost) * 100).toFixed(0)}%`);

  console.log(`\n💰 Intelligent Routing Savings:`);
  console.log(`   vs Premium: $${(premiumCost - routedCost).toFixed(2)} saved (${((1 - routedCost / premiumCost) * 100).toFixed(1)}%)`);
  console.log(`   vs Random:  $${(randomCost - routedCost).toFixed(2)} saved (${((1 - routedCost / randomCost) * 100).toFixed(1)}%)`);
  console.log(`   Lightweight routes: ${lightweightRoutes} (${((lightweightRoutes / numQueries) * 100).toFixed(1)}%)`);

  // Routing overhead
  const routingOverheadMs = numQueries * 0.1; // ~100μs per route
  console.log(`\n⚡ Routing Overhead: ${routingOverheadMs.toFixed(0)}ms total (${(routingOverheadMs / numQueries * 1000).toFixed(0)}μs/query)`);
}

// ============================================================================
// Utility Functions
// ============================================================================

function generateModelEmbedding(dim: number, seed: string): Float32Array {
  return seededEmbedding(dim, seed);
}

function generateQueryEmbedding(dim: number, text: string): Float32Array {
  return seededEmbedding(dim, text);
}

function seededEmbedding(dim: number, seed: string): Float32Array {
  const arr = new Float32Array(dim);
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
  }

  for (let i = 0; i < dim; i++) {
    hash = ((hash * 1103515245 + 12345) | 0) >>> 0;
    arr[i] = (hash / 0xFFFFFFFF - 0.5) * 2;
  }

  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  for (let i = 0; i < dim; i++) arr[i] /= norm;
  return arr;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  await multiModelRouterDemo();
}

main().catch(console.error);
