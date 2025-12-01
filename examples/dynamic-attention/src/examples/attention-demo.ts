/**
 * Attention Mechanisms Demo
 *
 * Demonstrates different attention mechanisms and their use cases
 */

import { createPipeline } from '../dynamic-attention.js';
import type { AttentionType } from '../types.js';
import { hrTimeUs } from '../simd-utils.js';

// ============================================================================
// Utility Functions
// ============================================================================

function randomEmbedding(dim: number): Float32Array {
  const arr = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    arr[i] = (Math.random() - 0.5) * 2;
  }
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  for (let i = 0; i < dim; i++) arr[i] /= norm;
  return arr;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ============================================================================
// Demo: Compare Attention Types
// ============================================================================

async function compareAttentionTypes() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('            Attention Mechanisms Comparison');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const dim = 256;
  const attentionTypes: AttentionType[] = [
    'dot-product',
    'multi-head',
    'hyperbolic',
    'flash',
    'linear',
    'local-global',
    'moe',
  ];

  const query = randomEmbedding(dim);
  const context = Array.from({ length: 16 }, () => randomEmbedding(dim));
  const candidates = Array.from({ length: 5 }, (_, i) => ({
    id: `candidate-${i}`,
    embedding: randomEmbedding(dim),
    successRate: 0.7 + Math.random() * 0.3,
  }));

  console.log('Test Setup:');
  console.log(`  Dimension:     ${dim}`);
  console.log(`  Context Size:  ${context.length}`);
  console.log(`  Candidates:    ${candidates.length}\n`);

  const results: Array<{
    type: AttentionType;
    latency: number;
    enrichmentChange: number;
    topCandidate: string;
    confidence: number;
  }> = [];

  for (const type of attentionTypes) {
    const pipeline = createPipeline({
      dim,
      numHeads: 8,
      attentionType: type,
    });

    // Run multiple times for stable timing
    const iterations = 100;
    const start = hrTimeUs();
    let result;
    for (let i = 0; i < iterations; i++) {
      result = await pipeline.process({ embedding: query, context }, candidates);
    }
    const avgLatency = (hrTimeUs() - start) / iterations;

    // Measure how much attention changes the embedding
    const enrichmentChange = 1 - cosineSimilarity(query, result!.enrichedEmbedding);

    results.push({
      type,
      latency: avgLatency,
      enrichmentChange,
      topCandidate: result!.decisions[0].candidateId,
      confidence: result!.decisions[0].confidence,
    });
  }

  // Print results table
  console.log('Attention Type   | Latency (μs) | Enrichment Δ | Top Choice  | Confidence');
  console.log('-----------------|--------------|--------------|-------------|------------');

  for (const r of results) {
    console.log(
      `${r.type.padEnd(16)} | ${r.latency.toFixed(1).padStart(12)} | ${r.enrichmentChange.toFixed(4).padStart(12)} | ${r.topCandidate.padStart(11)} | ${(r.confidence * 100).toFixed(1).padStart(10)}%`
    );
  }

  // Recommendations
  const fastest = results.reduce((a, b) => a.latency < b.latency ? a : b);
  const mostEnriching = results.reduce((a, b) => a.enrichmentChange > b.enrichmentChange ? a : b);

  console.log('\n📋 Analysis:\n');
  console.log(`  Fastest:          ${fastest.type} (${fastest.latency.toFixed(1)} μs)`);
  console.log(`  Most Enriching:   ${mostEnriching.type} (Δ=${mostEnriching.enrichmentChange.toFixed(4)})`);
}

// ============================================================================
// Demo: Context Aggregation
// ============================================================================

async function contextAggregationDemo() {
  console.log('\n\n═══════════════════════════════════════════════════════════════');
  console.log('            Context Aggregation Demo');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const dim = 128;
  const pipeline = createPipeline({
    dim,
    numHeads: 4,
    attentionType: 'multi-head',
  });

  // Scenario: Query with conversation history
  console.log('Scenario: Chat context with conversation history\n');

  const userQuery = randomEmbedding(dim);

  // Simulate conversation turns (some more relevant than others)
  const conversationHistory = [
    { role: 'system', embedding: randomEmbedding(dim), relevance: 0.3 },
    { role: 'user', embedding: randomEmbedding(dim), relevance: 0.5 },
    { role: 'assistant', embedding: randomEmbedding(dim), relevance: 0.6 },
    { role: 'user', embedding: randomEmbedding(dim), relevance: 0.9 }, // Most relevant
    { role: 'assistant', embedding: randomEmbedding(dim), relevance: 0.7 },
  ];

  console.log('Conversation History:');
  conversationHistory.forEach((turn, i) => {
    console.log(`  ${i + 1}. [${turn.role}] relevance: ${turn.relevance.toFixed(1)}`);
  });

  // Process with different context windows
  const contextSizes = [1, 2, 3, 5];

  console.log('\nContext Window Analysis:\n');
  console.log('Window Size | Enrichment Δ | Top Confidence');
  console.log('------------|--------------|---------------');

  for (const size of contextSizes) {
    const context = conversationHistory
      .slice(-size)
      .map(t => t.embedding);

    const result = await pipeline.process(
      { embedding: userQuery, context },
      Array.from({ length: 3 }, (_, i) => ({
        id: `model-${i}`,
        embedding: randomEmbedding(dim),
        successRate: 0.8 + i * 0.05,
      }))
    );

    const enrichment = 1 - cosineSimilarity(userQuery, result.enrichedEmbedding);

    console.log(
      `${size.toString().padStart(11)} | ${enrichment.toFixed(4).padStart(12)} | ${(result.decisions[0].confidence * 100).toFixed(1).padStart(13)}%`
    );
  }
}

// ============================================================================
// Demo: Hyperbolic Attention for Hierarchies
// ============================================================================

async function hyperbolicHierarchyDemo() {
  console.log('\n\n═══════════════════════════════════════════════════════════════');
  console.log('            Hyperbolic Attention for Hierarchies');
  console.log('═══════════════════════════════════════════════════════════════\n');

  console.log('Hyperbolic space is ideal for hierarchical data because:');
  console.log('  • Distance grows exponentially with depth');
  console.log('  • Parent-child relationships are preserved');
  console.log('  • Low distortion for tree-like structures\n');

  const dim = 64;

  // Compare Euclidean (dot-product) vs Hyperbolic
  const euclidean = createPipeline({ dim, attentionType: 'dot-product' });
  const hyperbolic = createPipeline({ dim, attentionType: 'hyperbolic' });

  // Simulate hierarchical structure: root → branches → leaves
  const root = createHierarchyEmbedding(dim, 0);
  const branches = [
    createHierarchyEmbedding(dim, 1),
    createHierarchyEmbedding(dim, 1),
  ];
  const leaves = [
    createHierarchyEmbedding(dim, 2),
    createHierarchyEmbedding(dim, 2),
    createHierarchyEmbedding(dim, 2),
    createHierarchyEmbedding(dim, 2),
  ];

  const query = leaves[0]; // Query from a leaf node
  const context = [root, ...branches, ...leaves.slice(1)];

  const candidates = [
    { id: 'leaf-sibling', embedding: leaves[1], successRate: 0.9 },
    { id: 'parent-branch', embedding: branches[0], successRate: 0.85 },
    { id: 'root', embedding: root, successRate: 0.8 },
    { id: 'distant-branch', embedding: branches[1], successRate: 0.75 },
  ];

  console.log('Hierarchy: root → branches → leaves');
  console.log('Query: leaf node (depth=2)');
  console.log('Candidates: sibling leaf, parent branch, root, distant branch\n');

  const eucResult = await euclidean.process({ embedding: query, context }, candidates);
  const hypResult = await hyperbolic.process({ embedding: query, context }, candidates);

  console.log('Results:');
  console.log('              | Euclidean        | Hyperbolic');
  console.log('--------------|------------------|------------------');
  console.log(`Top Choice    | ${eucResult.decisions[0].candidateId.padEnd(16)} | ${hypResult.decisions[0].candidateId}`);
  console.log(`Confidence    | ${(eucResult.decisions[0].confidence * 100).toFixed(1).padStart(14)}% | ${(hypResult.decisions[0].confidence * 100).toFixed(1)}%`);
  console.log(`Uncertainty   | ${(eucResult.decisions[0].uncertainty * 100).toFixed(1).padStart(14)}% | ${(hypResult.decisions[0].uncertainty * 100).toFixed(1)}%`);

  console.log('\n📋 Insight: Hyperbolic attention better captures hierarchical');
  console.log('   relationships, preferring candidates at similar depths.');
}

function createHierarchyEmbedding(dim: number, depth: number): Float32Array {
  const arr = new Float32Array(dim);
  // Embeddings at same depth are more similar
  const seed = depth * 1000;
  for (let i = 0; i < dim; i++) {
    const noise = (Math.sin(seed + i * 0.1) + Math.random() * 0.5) * Math.exp(-depth * 0.3);
    arr[i] = noise;
  }
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  for (let i = 0; i < dim; i++) arr[i] /= norm;
  return arr;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  await compareAttentionTypes();
  await contextAggregationDemo();
  await hyperbolicHierarchyDemo();
}

main().catch(console.error);
