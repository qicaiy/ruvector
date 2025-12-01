/**
 * Pipeline Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { createPipeline, DynamicAttentionPipeline } from '../src/dynamic-attention.js';
import { dotProduct, softmaxInPlace, detectSIMDCapabilities } from '../src/simd-utils.js';
import type { RoutingCandidate } from '../src/types.js';

// ============================================================================
// Test Utilities
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
    successRate: 0.7 + Math.random() * 0.3,
    avgLatency: Math.random() * 200 + 50,
  }));
}

// ============================================================================
// SIMD Utils Tests
// ============================================================================

describe('SIMD Utils', () => {
  describe('dotProduct', () => {
    it('should compute dot product correctly', () => {
      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([1, 0, 2, 0]);
      expect(dotProduct(a, b)).toBeCloseTo(7, 5);
    });

    it('should handle large vectors', () => {
      const a = randomFloat32Array(1024);
      const b = randomFloat32Array(1024);
      const result = dotProduct(a, b);
      expect(typeof result).toBe('number');
      expect(isFinite(result)).toBe(true);
    });

    it('should throw on mismatched lengths', () => {
      const a = new Float32Array([1, 2, 3]);
      const b = new Float32Array([1, 2]);
      expect(() => dotProduct(a, b)).toThrow();
    });
  });

  describe('softmaxInPlace', () => {
    it('should produce valid probability distribution', () => {
      const v = new Float32Array([1, 2, 3, 4]);
      softmaxInPlace(v);

      // Should sum to 1
      const sum = v.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);

      // Should be ordered (ascending input → ascending output)
      expect(v[0]).toBeLessThan(v[1]);
      expect(v[1]).toBeLessThan(v[2]);
      expect(v[2]).toBeLessThan(v[3]);
    });

    it('should handle extreme values', () => {
      const v = new Float32Array([1000, -1000, 0]);
      softmaxInPlace(v);
      expect(v[0]).toBeCloseTo(1, 5);
      expect(v[1]).toBeCloseTo(0, 5);
    });
  });

  describe('detectSIMDCapabilities', () => {
    it('should return valid capabilities', () => {
      const caps = detectSIMDCapabilities();
      expect(caps.available).toContain('none');
      expect(caps.cores).toBeGreaterThan(0);
      expect(['none', 'sse4', 'avx2', 'avx512', 'neon']).toContain(caps.recommended);
    });
  });
});

// ============================================================================
// Pipeline Tests
// ============================================================================

describe('DynamicAttentionPipeline', () => {
  let pipeline: DynamicAttentionPipeline;
  const dim = 128;

  beforeEach(() => {
    pipeline = createPipeline({
      dim,
      numHeads: 4,
      hiddenDim: 32,
      attentionType: 'multi-head',
    });
  });

  describe('creation', () => {
    it('should create pipeline with default config', () => {
      const p = createPipeline();
      expect(p).toBeDefined();
      expect(p.getConfig().dim).toBe(384);
    });

    it('should create pipeline with custom config', () => {
      const config = pipeline.getConfig();
      expect(config.dim).toBe(dim);
      expect(config.numHeads).toBe(4);
    });
  });

  describe('process', () => {
    it('should route single candidate', async () => {
      const query = randomFloat32Array(dim);
      const candidates = generateCandidates(1, dim);

      const result = await pipeline.process({ embedding: query }, candidates);

      expect(result.decisions).toHaveLength(1);
      expect(result.decisions[0].candidateId).toBe('candidate-0');
      expect(result.decisions[0].confidence).toBeGreaterThan(0);
      expect(result.decisions[0].confidence).toBeLessThanOrEqual(1);
    });

    it('should rank multiple candidates', async () => {
      const query = randomFloat32Array(dim);
      const candidates = generateCandidates(5, dim);

      const result = await pipeline.process({ embedding: query }, candidates);

      expect(result.decisions).toHaveLength(5);
      // Should be sorted by confidence
      for (let i = 1; i < result.decisions.length; i++) {
        expect(result.decisions[i - 1].confidence)
          .toBeGreaterThanOrEqual(result.decisions[i].confidence);
      }
    });

    it('should enrich embedding with context', async () => {
      const query = randomFloat32Array(dim);
      const context = [randomFloat32Array(dim), randomFloat32Array(dim)];
      const candidates = generateCandidates(3, dim);

      const resultWithContext = await pipeline.process(
        { embedding: query, context },
        candidates
      );

      const resultWithoutContext = await pipeline.process(
        { embedding: query },
        candidates
      );

      // Enriched embeddings should differ
      let diff = 0;
      for (let i = 0; i < dim; i++) {
        diff += Math.abs(
          resultWithContext.enrichedEmbedding[i] -
          resultWithoutContext.enrichedEmbedding[i]
        );
      }
      expect(diff).toBeGreaterThan(0);
    });

    it('should return valid metrics', async () => {
      const query = randomFloat32Array(dim);
      const candidates = generateCandidates(5, dim);

      const result = await pipeline.process({ embedding: query }, candidates);

      expect(result.metrics.totalLatencyUs).toBeGreaterThan(0);
      expect(result.metrics.attentionLatencyUs).toBeGreaterThanOrEqual(0);
      expect(result.metrics.fastgrnnLatencyUs).toBeGreaterThan(0);
      expect(result.metrics.candidatesProcessed).toBe(5);
      expect(result.metrics.throughputQps).toBeGreaterThan(0);
    });
  });

  describe('attention types', () => {
    const attentionTypes = [
      'dot-product',
      'multi-head',
      'hyperbolic',
      'flash',
      'linear',
      'moe',
    ] as const;

    it.each(attentionTypes)('should work with %s attention', async (type) => {
      const p = createPipeline({ dim: 64, attentionType: type });
      const query = randomFloat32Array(64);
      const candidates = generateCandidates(3, 64);

      const result = await p.process({ embedding: query }, candidates);

      expect(result.decisions).toHaveLength(3);
      expect(result.enrichedEmbedding).toHaveLength(64);
    });
  });

  describe('edge cases', () => {
    it('should handle empty context', async () => {
      const query = randomFloat32Array(dim);
      const candidates = generateCandidates(2, dim);

      const result = await pipeline.process(
        { embedding: query, context: [] },
        candidates
      );

      expect(result.decisions).toHaveLength(2);
    });

    it('should handle single candidate', async () => {
      const query = randomFloat32Array(dim);
      const candidates = generateCandidates(1, dim);

      const result = await pipeline.process({ embedding: query }, candidates);

      expect(result.decisions).toHaveLength(1);
      expect(result.decisions[0].confidence).toBeGreaterThan(0);
    });
  });

  describe('performance', () => {
    it('should process within acceptable latency', async () => {
      const query = randomFloat32Array(dim);
      const candidates = generateCandidates(10, dim);

      const start = performance.now();
      await pipeline.process({ embedding: query }, candidates);
      const elapsed = performance.now() - start;

      // Should be under 10ms (very generous for test stability)
      expect(elapsed).toBeLessThan(10);
    });

    it('should handle batch processing', async () => {
      const queries = Array.from({ length: 100 }, () => randomFloat32Array(dim));
      const candidates = generateCandidates(5, dim);

      const start = performance.now();
      await Promise.all(
        queries.map(q => pipeline.process({ embedding: q }, candidates))
      );
      const elapsed = performance.now() - start;

      // 100 queries should be under 500ms
      expect(elapsed).toBeLessThan(500);
    });
  });
});
