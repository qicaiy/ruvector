/**
 * Routing Demo
 *
 * Demonstrates FastGRNN-based neural routing for AI agent orchestration
 */

import { createPipeline } from '../dynamic-attention.js';
import { hrTimeUs } from '../simd-utils.js';

// ============================================================================
// Demo: LLM Model Routing
// ============================================================================

async function llmRoutingDemo() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('         LLM Model Routing with FastGRNN + Attention');
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Create pipeline with multi-head attention
  const pipeline = createPipeline({
    dim: 384,
    numHeads: 8,
    hiddenDim: 64,
    attentionType: 'multi-head',
    enableSIMD: true,
  });

  // Simulate query embeddings (in production, use actual embeddings)
  const queries = [
    {
      name: 'Simple greeting',
      embedding: createEmbedding(384, 'greeting'),
      expectedModel: 'gpt-3.5-turbo',
    },
    {
      name: 'Complex code review',
      embedding: createEmbedding(384, 'code'),
      expectedModel: 'gpt-4',
    },
    {
      name: 'Creative writing',
      embedding: createEmbedding(384, 'creative'),
      expectedModel: 'claude-3-opus',
    },
    {
      name: 'Math problem',
      embedding: createEmbedding(384, 'math'),
      expectedModel: 'gpt-4',
    },
  ];

  // Available models as candidates
  const candidates = [
    {
      id: 'gpt-4',
      embedding: createEmbedding(384, 'powerful'),
      successRate: 0.95,
      avgLatency: 2000,
      cost: 0.06,
      capabilities: ['reasoning', 'code', 'math', 'creative'],
    },
    {
      id: 'gpt-3.5-turbo',
      embedding: createEmbedding(384, 'fast'),
      successRate: 0.88,
      avgLatency: 500,
      cost: 0.002,
      capabilities: ['general', 'fast'],
    },
    {
      id: 'claude-3-opus',
      embedding: createEmbedding(384, 'creative'),
      successRate: 0.93,
      avgLatency: 1500,
      cost: 0.045,
      capabilities: ['reasoning', 'creative', 'long-context'],
    },
    {
      id: 'claude-3-haiku',
      embedding: createEmbedding(384, 'efficient'),
      successRate: 0.85,
      avgLatency: 300,
      cost: 0.001,
      capabilities: ['general', 'fast', 'efficient'],
    },
    {
      id: 'gemini-pro',
      embedding: createEmbedding(384, 'multimodal'),
      successRate: 0.90,
      avgLatency: 800,
      cost: 0.01,
      capabilities: ['multimodal', 'reasoning'],
    },
  ];

  console.log('Available Models:');
  for (const c of candidates) {
    console.log(`  • ${c.id}: ${c.capabilities?.join(', ')} (${c.avgLatency}ms, $${c.cost}/1K)`);
  }
  console.log('');

  // Process each query
  for (const query of queries) {
    console.log(`\n─────────────────────────────────────────────────────────────────`);
    console.log(`Query: "${query.name}"`);
    console.log(`─────────────────────────────────────────────────────────────────`);

    const result = await pipeline.process(
      { embedding: query.embedding },
      candidates
    );

    const best = result.decisions[0];
    const runner = result.decisions[1];

    console.log(`\n  Routing Decision:`);
    console.log(`    Primary:    ${best.candidateId} (confidence: ${(best.confidence * 100).toFixed(1)}%)`);
    console.log(`    Fallback:   ${runner.candidateId} (confidence: ${(runner.confidence * 100).toFixed(1)}%)`);
    console.log(`    Lightweight: ${best.useLightweight ? 'Yes' : 'No'}`);
    console.log(`    Uncertainty: ${(best.uncertainty * 100).toFixed(1)}%`);
    console.log(`    Reason:     ${best.reason}`);

    console.log(`\n  Performance:`);
    console.log(`    Total:      ${result.metrics.totalLatencyUs.toFixed(0)} μs`);
    console.log(`    Attention:  ${result.metrics.attentionLatencyUs.toFixed(0)} μs`);
    console.log(`    FastGRNN:   ${result.metrics.fastgrnnLatencyUs.toFixed(0)} μs`);
    console.log(`    Throughput: ${result.metrics.throughputQps.toFixed(0)} qps`);
  }

  // Batch throughput test
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('                    Throughput Test');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const batchSize = 1000;
  const start = hrTimeUs();

  for (let i = 0; i < batchSize; i++) {
    await pipeline.process(
      { embedding: queries[i % queries.length].embedding },
      candidates
    );
  }

  const elapsed = hrTimeUs() - start;
  const qps = (batchSize * 1_000_000) / elapsed;

  console.log(`  Batch Size:    ${batchSize} queries`);
  console.log(`  Total Time:    ${(elapsed / 1000).toFixed(2)} ms`);
  console.log(`  Throughput:    ${qps.toFixed(0)} queries/second`);
  console.log(`  Avg Latency:   ${(elapsed / batchSize).toFixed(2)} μs/query`);
}

// ============================================================================
// Demo: Agent Orchestration
// ============================================================================

async function agentOrchestrationDemo() {
  console.log('\n\n═══════════════════════════════════════════════════════════════');
  console.log('         Agent Orchestration with FastGRNN + Attention');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const pipeline = createPipeline({
    dim: 256,
    numHeads: 4,
    attentionType: 'hyperbolic', // Good for hierarchical agent structures
  });

  // Specialized agents
  const agents = [
    {
      id: 'code-agent',
      embedding: createEmbedding(256, 'programming'),
      successRate: 0.92,
      capabilities: ['code', 'debug', 'review'],
    },
    {
      id: 'research-agent',
      embedding: createEmbedding(256, 'research'),
      successRate: 0.89,
      capabilities: ['search', 'summarize', 'cite'],
    },
    {
      id: 'data-agent',
      embedding: createEmbedding(256, 'analysis'),
      successRate: 0.91,
      capabilities: ['sql', 'visualization', 'statistics'],
    },
    {
      id: 'creative-agent',
      embedding: createEmbedding(256, 'creative'),
      successRate: 0.87,
      capabilities: ['writing', 'brainstorm', 'design'],
    },
    {
      id: 'coordinator-agent',
      embedding: createEmbedding(256, 'planning'),
      successRate: 0.94,
      capabilities: ['plan', 'delegate', 'synthesize'],
    },
  ];

  // Sample tasks
  const tasks = [
    { name: 'Fix authentication bug', embedding: createEmbedding(256, 'debug') },
    { name: 'Research market trends', embedding: createEmbedding(256, 'research') },
    { name: 'Analyze sales data', embedding: createEmbedding(256, 'data') },
    { name: 'Write blog post', embedding: createEmbedding(256, 'writing') },
    { name: 'Plan sprint tasks', embedding: createEmbedding(256, 'planning') },
  ];

  console.log('Specialized Agents:');
  for (const a of agents) {
    console.log(`  • ${a.id}: ${a.capabilities?.join(', ')}`);
  }
  console.log('');

  for (const task of tasks) {
    const result = await pipeline.process({ embedding: task.embedding }, agents);
    const best = result.decisions[0];

    console.log(`Task: "${task.name}"`);
    console.log(`  → Route to: ${best.candidateId} (${(best.confidence * 100).toFixed(0)}% confidence)\n`);
  }
}

// ============================================================================
// Utilities
// ============================================================================

function createEmbedding(dim: number, seed: string): Float32Array {
  // Deterministic pseudo-random embedding based on seed
  const arr = new Float32Array(dim);
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
  }

  for (let i = 0; i < dim; i++) {
    hash = ((hash * 1103515245 + 12345) | 0) >>> 0;
    arr[i] = (hash / 0xFFFFFFFF - 0.5) * 2;
  }

  // Normalize
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  for (let i = 0; i < dim; i++) {
    arr[i] /= norm;
  }

  return arr;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  await llmRoutingDemo();
  await agentOrchestrationDemo();
}

main().catch(console.error);
