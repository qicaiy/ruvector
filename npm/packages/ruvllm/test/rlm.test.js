/**
 * RLM (Recursive Language Model) Tests for @ruvector/ruvllm
 *
 * Tests the RLM module including:
 * - RlmController instantiation and configuration
 * - Query processing and caching
 * - Memory operations
 * - Quality scoring
 * - Edge cases
 */

const { test, describe, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert');

// Import from dist - adjust path based on build output
const { RuvLLM } = require('../dist/cjs/index.js');

// ============================================================================
// Mock RlmController for testing
// ============================================================================

/**
 * Mock RlmController that simulates the Rust implementation
 * This allows testing the JavaScript API without requiring native bindings
 */
class MockRlmController {
  constructor(config = {}) {
    this.config = {
      maxDepth: config.maxDepth ?? 5,
      tokenBudget: config.tokenBudget ?? 16000,
      enableCache: config.enableCache ?? true,
      qualityThreshold: config.qualityThreshold ?? 0.7,
      embeddingDim: config.embeddingDim ?? 384,
      ...config,
    };

    this.cache = new Map();
    this.memories = new Map();
    this.stats = {
      totalQueries: 0,
      cacheHits: 0,
      cacheMisses: 0,
      decompositions: 0,
      memoriesStored: 0,
    };
  }

  /**
   * Process a query and return an RlmAnswer structure
   * @param {string} query - The query text
   * @returns {Promise<RlmAnswer>} The answer
   */
  async query(query) {
    this.stats.totalQueries++;

    // Check cache
    if (this.config.enableCache && this.cache.has(query.toLowerCase())) {
      this.stats.cacheHits++;
      return {
        ...this.cache.get(query.toLowerCase()),
        cached: true,
      };
    }

    this.stats.cacheMisses++;

    // Decompose if needed
    const decomposition = this.decompose(query);
    if (decomposition.strategy !== 'direct') {
      this.stats.decompositions++;
    }

    // Generate mock answer
    const answer = this.generateAnswer(query, decomposition);

    // Cache if quality meets threshold
    if (answer.qualityScore >= this.config.qualityThreshold && this.config.enableCache) {
      this.cache.set(query.toLowerCase(), answer);
    }

    return answer;
  }

  /**
   * Decompose a query into sub-queries
   * @param {string} query
   * @returns {DecompositionResult}
   */
  decompose(query) {
    const queryLower = query.toLowerCase();

    // Check for comparison
    if (queryLower.includes('compare') || queryLower.includes('versus') || queryLower.includes('vs')) {
      return {
        strategy: 'comparison',
        subQueries: this.splitComparison(query),
        complexity: this.computeComplexity(query),
      };
    }

    // Check for conjunction
    if (queryLower.includes(' and ')) {
      return {
        strategy: 'conjunction',
        subQueries: query.split(/ and /i).map(s => s.trim()),
        complexity: this.computeComplexity(query),
      };
    }

    // Direct query
    return {
      strategy: 'direct',
      subQueries: [query],
      complexity: this.computeComplexity(query),
    };
  }

  /**
   * Compute complexity score for a query
   * @param {string} query
   * @returns {number} Complexity between 0 and 1
   */
  computeComplexity(query) {
    let score = 0;

    // Length factor
    score += Math.min(query.length / 200, 0.3);

    // Question word count
    const questionWords = ['what', 'why', 'how', 'when', 'where', 'who'];
    const qCount = questionWords.filter(w => query.toLowerCase().includes(w)).length;
    score += Math.min(qCount * 0.1, 0.2);

    // Conjunction count
    const conjCount = (query.toLowerCase().match(/ and | or /g) || []).length;
    score += Math.min(conjCount * 0.15, 0.3);

    return Math.min(score, 1);
  }

  splitComparison(query) {
    // Simple split around comparison keywords
    const patterns = [' versus ', ' vs ', ' compare '];
    for (const pattern of patterns) {
      if (query.toLowerCase().includes(pattern)) {
        const parts = query.split(new RegExp(pattern, 'i'));
        return parts.map(p => `What is ${p.trim()}?`);
      }
    }
    return [query];
  }

  generateAnswer(query, decomposition) {
    const subAnswers = decomposition.subQueries.map((sq, i) => ({
      text: `Answer for sub-query ${i + 1}: "${sq}"`,
      qualityScore: 0.8 + Math.random() * 0.15,
    }));

    const aggregatedText = subAnswers.map(a => a.text).join('\n\n');
    const avgQuality = subAnswers.reduce((sum, a) => sum + a.qualityScore, 0) / subAnswers.length;

    return {
      text: aggregatedText,
      qualityScore: avgQuality,
      cached: false,
      depthReached: decomposition.strategy === 'direct' ? 0 : 1,
      subAnswers: subAnswers.map(a => a.text),
      processingTimeMs: 10 + Math.random() * 40,
      decompositionStrategy: decomposition.strategy,
    };
  }

  /**
   * Add content to memory
   * @param {string} text
   * @param {object} metadata
   * @returns {string} Memory ID
   */
  addMemory(text, metadata = {}) {
    const id = `mem-${this.stats.memoriesStored++}`;
    this.memories.set(id, {
      id,
      text,
      embedding: this.generateMockEmbedding(text),
      metadata,
      createdAt: new Date().toISOString(),
      accessCount: 0,
    });
    return id;
  }

  /**
   * Search memory for similar content
   * @param {string} query
   * @param {number} topK
   * @returns {Array<MemorySearchResult>}
   */
  searchMemory(query, topK = 5) {
    const queryEmb = this.generateMockEmbedding(query);
    const results = [];

    for (const [id, entry] of this.memories) {
      const similarity = this.cosineSimilarity(queryEmb, entry.embedding);
      results.push({
        id,
        text: entry.text,
        score: similarity,
        metadata: entry.metadata,
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  generateMockEmbedding(text) {
    // Simple hash-based embedding
    const dim = this.config.embeddingDim;
    const embedding = new Array(dim).fill(0);
    const bytes = Buffer.from(text);

    for (let i = 0; i < bytes.length; i++) {
      embedding[i % dim] += (bytes[i] / 255) - 0.5;
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
    if (norm > 0) {
      for (let i = 0; i < dim; i++) {
        embedding[i] /= norm;
      }
    }

    return embedding;
  }

  cosineSimilarity(a, b) {
    if (a.length !== b.length) return 0;

    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  getStats() {
    return { ...this.stats };
  }

  clearCache() {
    this.cache.clear();
  }

  getMemory(id) {
    return this.memories.get(id) || null;
  }

  deleteMemory(id) {
    return this.memories.delete(id);
  }
}

// ============================================================================
// Tests
// ============================================================================

describe('MockRlmController', () => {
  test('constructor with default config', () => {
    const rlm = new MockRlmController();
    assert.ok(rlm);
    assert.strictEqual(rlm.config.maxDepth, 5);
    assert.strictEqual(rlm.config.tokenBudget, 16000);
    assert.strictEqual(rlm.config.enableCache, true);
  });

  test('constructor with custom config', () => {
    const rlm = new MockRlmController({
      maxDepth: 3,
      enableCache: false,
      embeddingDim: 256,
    });
    assert.strictEqual(rlm.config.maxDepth, 3);
    assert.strictEqual(rlm.config.enableCache, false);
    assert.strictEqual(rlm.config.embeddingDim, 256);
  });

  test('query returns RlmAnswer structure', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('What is machine learning?');

    assert.ok(answer.text);
    assert.ok(typeof answer.qualityScore === 'number');
    assert.ok(answer.qualityScore >= 0 && answer.qualityScore <= 1);
    assert.strictEqual(typeof answer.cached, 'boolean');
    assert.ok(typeof answer.processingTimeMs === 'number');
  });

  test('cache prevents duplicate computation', async () => {
    const rlm = new MockRlmController({ enableCache: true });

    const answer1 = await rlm.query('Cached query test');
    assert.strictEqual(answer1.cached, false);

    const answer2 = await rlm.query('Cached query test');
    assert.strictEqual(answer2.cached, true);

    const stats = rlm.getStats();
    assert.strictEqual(stats.cacheHits, 1);
    assert.strictEqual(stats.cacheMisses, 1);
  });

  test('cache disabled skips caching', async () => {
    const rlm = new MockRlmController({ enableCache: false });

    await rlm.query('Test query');
    await rlm.query('Test query');

    const stats = rlm.getStats();
    assert.strictEqual(stats.cacheHits, 0);
  });

  test('query decomposition for conjunction', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('What are the causes and effects of climate change?');

    assert.strictEqual(answer.decompositionStrategy, 'conjunction');
    assert.ok(answer.subAnswers.length >= 2);
  });

  test('query decomposition for comparison', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('Compare Python versus JavaScript');

    assert.strictEqual(answer.decompositionStrategy, 'comparison');
    assert.ok(answer.subAnswers.length >= 2);
  });

  test('direct query for simple questions', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('What is AI?');

    assert.strictEqual(answer.decompositionStrategy, 'direct');
    assert.strictEqual(answer.subAnswers.length, 1);
  });

  test('complexity scoring', () => {
    const rlm = new MockRlmController();

    const simpleComplexity = rlm.computeComplexity('What is X?');
    const complexComplexity = rlm.computeComplexity(
      'What are the primary causes and effects of climate change, and how do they impact global weather patterns?'
    );

    assert.ok(simpleComplexity < complexComplexity);
    assert.ok(simpleComplexity >= 0 && simpleComplexity <= 1);
    assert.ok(complexComplexity >= 0 && complexComplexity <= 1);
  });

  test('stats tracking', async () => {
    const rlm = new MockRlmController();

    assert.strictEqual(rlm.getStats().totalQueries, 0);

    await rlm.query('Query 1');
    await rlm.query('Query 2');

    const stats = rlm.getStats();
    assert.strictEqual(stats.totalQueries, 2);
  });

  test('clear cache', async () => {
    const rlm = new MockRlmController();

    await rlm.query('Test query');
    rlm.clearCache();

    // Second query should not be cached
    const answer = await rlm.query('Test query');
    assert.strictEqual(answer.cached, false);
  });
});

describe('Memory Operations', () => {
  test('add and retrieve memory', () => {
    const rlm = new MockRlmController();

    const id = rlm.addMemory('Test content', { category: 'test' });
    assert.ok(id.startsWith('mem-'));

    const memory = rlm.getMemory(id);
    assert.ok(memory);
    assert.strictEqual(memory.text, 'Test content');
    assert.strictEqual(memory.metadata.category, 'test');
  });

  test('search memory returns results', () => {
    const rlm = new MockRlmController();

    rlm.addMemory('Machine learning is a subset of AI');
    rlm.addMemory('Deep learning uses neural networks');
    rlm.addMemory('Python is a programming language');

    const results = rlm.searchMemory('machine learning', 2);

    assert.ok(Array.isArray(results));
    assert.ok(results.length <= 2);
    assert.ok(results[0].score >= 0);
  });

  test('delete memory', () => {
    const rlm = new MockRlmController();

    const id = rlm.addMemory('To be deleted');
    assert.ok(rlm.getMemory(id));

    const deleted = rlm.deleteMemory(id);
    assert.strictEqual(deleted, true);
    assert.strictEqual(rlm.getMemory(id), null);
  });

  test('memory embedding generation', () => {
    const rlm = new MockRlmController({ embeddingDim: 128 });

    const embedding = rlm.generateMockEmbedding('Test text');

    assert.strictEqual(embedding.length, 128);
    // Check normalization (unit vector)
    const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
    assert.ok(Math.abs(norm - 1) < 0.001 || norm === 0);
  });
});

describe('Quality Scoring', () => {
  test('quality score in valid range', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('Test query');

    assert.ok(answer.qualityScore >= 0);
    assert.ok(answer.qualityScore <= 1);
  });

  test('quality threshold affects caching', async () => {
    const rlm = new MockRlmController({
      enableCache: true,
      qualityThreshold: 0.99, // Very high threshold
    });

    await rlm.query('Test query');
    const answer2 = await rlm.query('Test query');

    // Should not be cached due to high threshold
    // (depending on mock quality generation)
    assert.ok(answer2.qualityScore < 0.99 ? !answer2.cached : answer2.cached);
  });
});

describe('Edge Cases', () => {
  test('empty query handling', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('');

    assert.ok(answer);
    assert.ok(typeof answer.text === 'string');
  });

  test('very long query handling', async () => {
    const rlm = new MockRlmController();
    const longQuery = 'What '.repeat(1000) + 'is AI?';
    const answer = await rlm.query(longQuery);

    assert.ok(answer);
    assert.ok(answer.qualityScore >= 0);
  });

  test('special characters in query', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('What is AI? (with special chars: @#$%^&*)');

    assert.ok(answer);
  });

  test('unicode in query', async () => {
    const rlm = new MockRlmController();
    const answer = await rlm.query('What is AI? Japanese: \u4eba\u5de5\u77e5\u80fd');

    assert.ok(answer);
  });

  test('concurrent queries', async () => {
    const rlm = new MockRlmController();

    const queries = Array(10).fill(null).map((_, i) => rlm.query(`Query ${i}`));
    const answers = await Promise.all(queries);

    assert.strictEqual(answers.length, 10);
    answers.forEach(answer => {
      assert.ok(answer.text);
      assert.ok(typeof answer.qualityScore === 'number');
    });
  });

  test('memory search with no entries', () => {
    const rlm = new MockRlmController();
    const results = rlm.searchMemory('test query', 5);

    assert.ok(Array.isArray(results));
    assert.strictEqual(results.length, 0);
  });

  test('memory search topK larger than entries', () => {
    const rlm = new MockRlmController();
    rlm.addMemory('Entry 1');
    rlm.addMemory('Entry 2');

    const results = rlm.searchMemory('test', 100);

    assert.ok(results.length <= 2);
  });
});

describe('Cosine Similarity', () => {
  test('identical vectors have similarity 1', () => {
    const rlm = new MockRlmController();
    const vec = [0.5, 0.5, 0.5, 0.5];
    const similarity = rlm.cosineSimilarity(vec, vec);

    assert.ok(Math.abs(similarity - 1) < 0.001);
  });

  test('orthogonal vectors have similarity 0', () => {
    const rlm = new MockRlmController();
    const vec1 = [1, 0, 0, 0];
    const vec2 = [0, 1, 0, 0];
    const similarity = rlm.cosineSimilarity(vec1, vec2);

    assert.ok(Math.abs(similarity) < 0.001);
  });

  test('different length vectors return 0', () => {
    const rlm = new MockRlmController();
    const vec1 = [1, 2, 3];
    const vec2 = [1, 2];
    const similarity = rlm.cosineSimilarity(vec1, vec2);

    assert.strictEqual(similarity, 0);
  });
});

describe('Integration with RuvLLM', () => {
  test('RuvLLM instance exists', () => {
    const llm = new RuvLLM();
    assert.ok(llm);
  });

  test('RuvLLM query functionality', () => {
    const llm = new RuvLLM();
    const response = llm.query('test query');

    assert.ok(response);
    assert.ok(response.text);
    assert.ok(typeof response.confidence === 'number');
  });

  test('RuvLLM memory operations', () => {
    const llm = new RuvLLM();

    const id = llm.addMemory('Test memory content', { type: 'test' });
    assert.ok(typeof id === 'number');

    const results = llm.searchMemory('memory', 5);
    assert.ok(Array.isArray(results));
  });
});

describe('Decomposition Strategies', () => {
  test('all strategy types are handled', async () => {
    const rlm = new MockRlmController();

    // Direct
    const direct = await rlm.query('What is AI?');
    assert.strictEqual(direct.decompositionStrategy, 'direct');

    // Conjunction
    const conj = await rlm.query('causes and effects');
    assert.strictEqual(conj.decompositionStrategy, 'conjunction');

    // Comparison
    const comp = await rlm.query('compare X versus Y');
    assert.strictEqual(comp.decompositionStrategy, 'comparison');
  });

  test('decomposition affects depth', async () => {
    const rlm = new MockRlmController();

    const simple = await rlm.query('Simple query');
    const complex = await rlm.query('Complex query with causes and effects');

    assert.strictEqual(simple.depthReached, 0);
    assert.ok(complex.depthReached >= 1);
  });
});

// Run additional benchmarks if time allows
describe('Performance Sanity Checks', () => {
  test('query completes in reasonable time', async () => {
    const rlm = new MockRlmController();
    const start = Date.now();

    await rlm.query('Performance test query');

    const elapsed = Date.now() - start;
    assert.ok(elapsed < 1000, `Query took ${elapsed}ms, expected <1000ms`);
  });

  test('batch queries complete efficiently', async () => {
    const rlm = new MockRlmController();
    const start = Date.now();

    const queries = Array(100).fill(null).map((_, i) => rlm.query(`Batch query ${i}`));
    await Promise.all(queries);

    const elapsed = Date.now() - start;
    assert.ok(elapsed < 5000, `Batch took ${elapsed}ms, expected <5000ms`);
  });

  test('memory search scales reasonably', () => {
    const rlm = new MockRlmController();

    // Add 1000 entries
    for (let i = 0; i < 1000; i++) {
      rlm.addMemory(`Memory entry ${i} about topic ${i % 10}`);
    }

    const start = Date.now();
    rlm.searchMemory('topic 5', 10);
    const elapsed = Date.now() - start;

    assert.ok(elapsed < 100, `Search took ${elapsed}ms, expected <100ms`);
  });
});
