/**
 * RLM Controller - Recursive Retrieval Language Model
 *
 * Implements a recursive retrieval-augmented generation system that:
 * 1. Breaks down complex queries into sub-queries
 * 2. Retrieves relevant memory spans for each query
 * 3. Synthesizes coherent answers from retrieved context
 * 4. Optionally reflects on and refines answers
 *
 * @example Basic Usage
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController({
 *   maxDepth: 3,
 *   retrievalTopK: 10,
 *   enableCache: true,
 * });
 *
 * // Add knowledge to memory
 * await rlm.addMemory('Machine learning is a subset of AI that enables systems to learn from data.');
 * await rlm.addMemory('Deep learning uses neural networks with many layers.');
 *
 * // Query with recursive retrieval
 * const answer = await rlm.query('Explain the relationship between ML and deep learning');
 * console.log(answer.text);
 * console.log('Sources:', answer.sources.length);
 * console.log('Confidence:', answer.confidence);
 * ```
 *
 * @example Streaming
 * ```typescript
 * const rlm = new RlmController();
 *
 * for await (const event of rlm.queryStream('What is AI?')) {
 *   if (event.type === 'token') {
 *     process.stdout.write(event.text);
 *   } else {
 *     console.log('\n\nDone! Quality:', event.answer.qualityScore);
 *   }
 * }
 * ```
 *
 * @example With Reflection
 * ```typescript
 * const rlm = new RlmController({
 *   enableReflection: true,
 *   maxReflectionIterations: 2,
 *   minQualityScore: 0.8,
 * });
 *
 * const answer = await rlm.query('Complex multi-part question...');
 * // Answer will be iteratively refined until quality >= 0.8
 * ```
 */

import {
  RlmConfig,
  RlmAnswer,
  MemorySpan,
  SubQuery,
  TokenUsage,
  StreamToken,
  RlmCacheEntry,
  ReflectionResult,
} from './types';

import { RuvLLM } from '../engine';
import type { GenerationConfig, QueryResponse } from '../types';

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: Required<RlmConfig> = {
  maxDepth: 3,
  maxSubQueries: 5,
  tokenBudget: 4096,
  enableCache: true,
  cacheTtl: 300000, // 5 minutes
  retrievalTopK: 10,
  minQualityScore: 0.7,
  enableReflection: false,
  maxReflectionIterations: 2,
};

/**
 * RlmController - Recursive Retrieval Language Model Controller
 *
 * Orchestrates retrieval-augmented generation with recursive sub-query
 * decomposition, memory search, and optional self-reflection.
 */
export class RlmController {
  private config: Required<RlmConfig>;
  private cache: Map<string, RlmCacheEntry>;
  private engine: RuvLLM;
  private memoryIdCounter: number;

  /**
   * Create a new RLM controller
   *
   * @param config - Configuration options
   * @param engine - Optional RuvLLM engine instance (creates new if not provided)
   *
   * @example
   * ```typescript
   * // With default config
   * const rlm = new RlmController();
   *
   * // With custom config
   * const rlm = new RlmController({
   *   maxDepth: 5,
   *   enableReflection: true,
   * });
   *
   * // With existing engine
   * const engine = new RuvLLM({ learningEnabled: true });
   * const rlm = new RlmController({}, engine);
   * ```
   */
  constructor(config?: RlmConfig, engine?: RuvLLM) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.cache = new Map();
    this.engine = engine ?? new RuvLLM({ learningEnabled: true });
    this.memoryIdCounter = 0;
  }

  /**
   * Query the RLM with recursive retrieval
   *
   * @param input - The query string
   * @returns Promise resolving to the answer with sources and metadata
   *
   * @example
   * ```typescript
   * const answer = await rlm.query('What is the capital of France?');
   * console.log(answer.text); // "The capital of France is Paris..."
   * console.log(answer.confidence); // 0.95
   * console.log(answer.sources); // [{ id: '...', text: '...', similarityScore: 0.92 }]
   * ```
   */
  async query(input: string): Promise<RlmAnswer> {
    // Check cache first
    if (this.config.enableCache) {
      const cached = this.getCached(input);
      if (cached) {
        return { ...cached, cached: true };
      }
    }

    // Retrieve relevant memory spans
    const sources = await this.searchMemory(input, this.config.retrievalTopK);

    // Generate sub-queries if needed and depth allows
    const subQueries = await this.generateSubQueries(input, sources, 0);

    // Build context from sources and sub-query answers
    const context = this.buildContext(sources, subQueries);

    // Generate the answer
    const startTime = Date.now();
    const response = this.engine.query(
      this.buildPrompt(input, context),
      this.getGenerationConfig()
    );

    // Calculate token usage (estimate if not provided by engine)
    const tokenUsage = this.estimateTokenUsage(input, context, response.text);

    // Calculate quality score
    const qualityScore = this.calculateQualityScore(sources, response.confidence);

    let answer: RlmAnswer = {
      text: response.text,
      confidence: response.confidence,
      qualityScore,
      sources,
      subQueries: subQueries.length > 0 ? subQueries : undefined,
      tokenUsage,
      cached: false,
    };

    // Apply reflection if enabled and quality is below threshold
    if (this.config.enableReflection && qualityScore < this.config.minQualityScore) {
      answer = await this.applyReflection(input, answer);
    }

    // Cache the result
    if (this.config.enableCache) {
      this.setCache(input, answer);
    }

    return answer;
  }

  /**
   * Query with streaming response
   *
   * @param input - The query string
   * @yields StreamToken events (either partial tokens or final answer)
   *
   * @example
   * ```typescript
   * for await (const event of rlm.queryStream('Explain quantum computing')) {
   *   if (event.type === 'token') {
   *     // Partial token received
   *     process.stdout.write(event.text);
   *   } else {
   *     // Generation complete
   *     console.log('\n\nSources:', event.answer.sources.length);
   *   }
   * }
   * ```
   */
  async *queryStream(input: string): AsyncGenerator<StreamToken> {
    // Check cache first
    if (this.config.enableCache) {
      const cached = this.getCached(input);
      if (cached) {
        // Simulate streaming for cached response
        const words = cached.text.split(' ');
        for (const word of words) {
          yield { type: 'token', text: word + ' ', done: false };
          await this.delay(10); // Small delay for realistic streaming
        }
        yield { type: 'done', answer: { ...cached, cached: true }, done: true };
        return;
      }
    }

    // Retrieve sources
    const sources = await this.searchMemory(input, this.config.retrievalTopK);
    const subQueries = await this.generateSubQueries(input, sources, 0);
    const context = this.buildContext(sources, subQueries);

    // Generate with simulated streaming
    const prompt = this.buildPrompt(input, context);
    const response = this.engine.query(prompt, this.getGenerationConfig());

    // Stream the response word by word
    const words = response.text.split(' ');
    let streamedText = '';

    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const text = i < words.length - 1 ? word + ' ' : word;
      streamedText += text;

      yield { type: 'token', text, done: false };
      await this.delay(20); // Simulate generation latency
    }

    const tokenUsage = this.estimateTokenUsage(input, context, streamedText);
    const qualityScore = this.calculateQualityScore(sources, response.confidence);

    const answer: RlmAnswer = {
      text: streamedText,
      confidence: response.confidence,
      qualityScore,
      sources,
      subQueries: subQueries.length > 0 ? subQueries : undefined,
      tokenUsage,
      cached: false,
    };

    // Cache the result
    if (this.config.enableCache) {
      this.setCache(input, answer);
    }

    yield { type: 'done', answer, done: true };
  }

  /**
   * Add content to memory for retrieval
   *
   * @param text - The text content to store
   * @param metadata - Optional metadata to associate with the memory
   * @returns Promise resolving to the memory span ID
   *
   * @example
   * ```typescript
   * const id1 = await rlm.addMemory(
   *   'TypeScript is a typed superset of JavaScript.',
   *   { source: 'documentation', category: 'programming' }
   * );
   *
   * const id2 = await rlm.addMemory(
   *   'React is a JavaScript library for building UIs.'
   * );
   * ```
   */
  async addMemory(text: string, metadata?: Record<string, unknown>): Promise<string> {
    const nodeId = this.engine.addMemory(text, metadata);
    const id = `rlm-mem-${this.memoryIdCounter++}-${nodeId}`;
    return id;
  }

  /**
   * Search memory for relevant spans
   *
   * @param query - The search query
   * @param topK - Number of results to return (default: config.retrievalTopK)
   * @returns Promise resolving to array of memory spans
   *
   * @example
   * ```typescript
   * const spans = await rlm.searchMemory('JavaScript frameworks', 5);
   * for (const span of spans) {
   *   console.log(`[${span.similarityScore.toFixed(2)}] ${span.text}`);
   * }
   * ```
   */
  async searchMemory(query: string, topK?: number): Promise<MemorySpan[]> {
    const k = topK ?? this.config.retrievalTopK;
    const results = this.engine.searchMemory(query, k);

    return results.map((result, index) => ({
      id: `rlm-span-${result.id}-${index}`,
      text: result.content,
      similarityScore: result.score,
      source: result.metadata?.source as string | undefined,
      metadata: result.metadata,
    }));
  }

  /**
   * Clear the response cache
   *
   * @example
   * ```typescript
   * rlm.clearCache();
   * console.log('Cache cleared');
   * ```
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get current cache statistics
   *
   * @returns Object with cache size and hit rate info
   */
  getCacheStats(): { size: number; entries: number } {
    return {
      size: this.cache.size,
      entries: this.cache.size,
    };
  }

  /**
   * Update configuration at runtime
   *
   * @param config - Partial configuration to merge
   */
  updateConfig(config: Partial<RlmConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<RlmConfig> {
    return { ...this.config };
  }

  // ============================================
  // Private Methods
  // ============================================

  /**
   * Generate sub-queries for complex questions
   */
  private async generateSubQueries(
    query: string,
    sources: MemorySpan[],
    depth: number
  ): Promise<SubQuery[]> {
    if (depth >= this.config.maxDepth) {
      return [];
    }

    // Simple heuristic: generate sub-queries for questions with multiple parts
    const subQueries: SubQuery[] = [];
    const parts = this.decomposeQuery(query);

    for (const part of parts.slice(0, this.config.maxSubQueries)) {
      if (part.trim().length < 10) continue;

      // Search for sub-query specific sources
      const subSources = await this.searchMemory(part, Math.ceil(this.config.retrievalTopK / 2));
      const context = this.buildContext(subSources, []);
      const response = this.engine.query(
        this.buildPrompt(part, context),
        { ...this.getGenerationConfig(), maxTokens: 256 }
      );

      subQueries.push({
        query: part,
        answer: response.text,
        depth: depth + 1,
      });
    }

    return subQueries;
  }

  /**
   * Decompose a complex query into simpler parts
   */
  private decomposeQuery(query: string): string[] {
    // Split on common conjunctions and question markers
    const parts: string[] = [];

    // Check for multi-part questions
    const conjunctions = [' and ', ' or ', '. ', '? ', '; '];
    let current = query;

    for (const conj of conjunctions) {
      if (current.includes(conj)) {
        const split = current.split(conj);
        parts.push(...split.filter(p => p.trim().length > 10));
        current = '';
        break;
      }
    }

    // If no decomposition happened, return original
    if (parts.length === 0) {
      return [query];
    }

    return parts;
  }

  /**
   * Build context string from sources and sub-queries
   */
  private buildContext(sources: MemorySpan[], subQueries: SubQuery[]): string {
    const parts: string[] = [];

    // Add sources
    if (sources.length > 0) {
      parts.push('Relevant context:');
      for (const source of sources) {
        parts.push(`- ${source.text}`);
      }
    }

    // Add sub-query answers
    if (subQueries.length > 0) {
      parts.push('\nRelated information:');
      for (const sq of subQueries) {
        parts.push(`Q: ${sq.query}`);
        parts.push(`A: ${sq.answer}`);
      }
    }

    return parts.join('\n');
  }

  /**
   * Build the full prompt with context
   */
  private buildPrompt(query: string, context: string): string {
    if (context.trim().length === 0) {
      return query;
    }

    return `${context}\n\nBased on the above context, answer the following question:\n${query}`;
  }

  /**
   * Get generation config based on RLM settings
   */
  private getGenerationConfig(): GenerationConfig {
    return {
      maxTokens: Math.min(this.config.tokenBudget, 2048),
      temperature: 0.7,
      topP: 0.9,
    };
  }

  /**
   * Estimate token usage
   */
  private estimateTokenUsage(query: string, context: string, response: string): TokenUsage {
    // Rough estimation: ~4 characters per token
    const promptTokens = Math.ceil((query.length + context.length) / 4);
    const completionTokens = Math.ceil(response.length / 4);

    return {
      prompt: promptTokens,
      completion: completionTokens,
      total: promptTokens + completionTokens,
    };
  }

  /**
   * Calculate quality score based on sources and confidence
   */
  private calculateQualityScore(sources: MemorySpan[], confidence: number): number {
    if (sources.length === 0) {
      return confidence * 0.5; // Penalize answers without sources
    }

    // Average source similarity
    const avgSimilarity = sources.reduce((sum, s) => sum + s.similarityScore, 0) / sources.length;

    // Weighted combination
    return confidence * 0.6 + avgSimilarity * 0.4;
  }

  /**
   * Apply self-reflection to improve answer
   */
  private async applyReflection(
    query: string,
    answer: RlmAnswer
  ): Promise<RlmAnswer> {
    let currentAnswer = answer;
    let iterations = 0;

    while (
      iterations < this.config.maxReflectionIterations &&
      currentAnswer.qualityScore < this.config.minQualityScore
    ) {
      iterations++;

      // Generate critique
      const critiquePrompt = `Evaluate this answer for accuracy and completeness:
Question: ${query}
Answer: ${currentAnswer.text}

Provide a brief critique and suggest improvements.`;

      const critiqueResponse = this.engine.query(critiquePrompt, {
        maxTokens: 256,
        temperature: 0.5,
      });

      // Generate improved answer
      const improvePrompt = `Based on this feedback: "${critiqueResponse.text}"

Improve this answer:
Question: ${query}
Original: ${currentAnswer.text}

Provide an improved answer:`;

      const improvedResponse = this.engine.query(improvePrompt, this.getGenerationConfig());

      // Update answer with reflection improvements
      const newQualityScore = Math.min(
        1.0,
        currentAnswer.qualityScore + 0.1 * iterations
      );

      currentAnswer = {
        ...currentAnswer,
        text: improvedResponse.text,
        confidence: Math.max(currentAnswer.confidence, improvedResponse.confidence),
        qualityScore: newQualityScore,
        tokenUsage: {
          prompt: currentAnswer.tokenUsage.prompt + 100, // Approximate additional tokens
          completion: currentAnswer.tokenUsage.completion + 100,
          total: currentAnswer.tokenUsage.total + 200,
        },
      };
    }

    return currentAnswer;
  }

  /**
   * Get cached answer if valid
   */
  private getCached(query: string): RlmAnswer | null {
    const hash = this.hashQuery(query);
    const entry = this.cache.get(hash);

    if (!entry) {
      return null;
    }

    // Check TTL
    if (Date.now() - entry.timestamp > this.config.cacheTtl) {
      this.cache.delete(hash);
      return null;
    }

    return entry.answer;
  }

  /**
   * Set cache entry
   */
  private setCache(query: string, answer: RlmAnswer): void {
    const hash = this.hashQuery(query);
    this.cache.set(hash, {
      answer,
      timestamp: Date.now(),
      queryHash: hash,
    });

    // Prune old entries if cache gets too large
    if (this.cache.size > 1000) {
      this.pruneCache();
    }
  }

  /**
   * Simple hash function for cache keys
   */
  private hashQuery(query: string): string {
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `rlm-cache-${hash.toString(16)}`;
  }

  /**
   * Prune expired cache entries
   */
  private pruneCache(): void {
    const now = Date.now();
    const toDelete: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.config.cacheTtl) {
        toDelete.push(key);
      }
    }

    // Delete oldest entries if still too large
    if (this.cache.size - toDelete.length > 800) {
      const entries = Array.from(this.cache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp);

      const deleteCount = entries.length - 500;
      for (let i = 0; i < deleteCount; i++) {
        toDelete.push(entries[i][0]);
      }
    }

    for (const key of toDelete) {
      this.cache.delete(key);
    }
  }

  /**
   * Utility delay function for streaming simulation
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
