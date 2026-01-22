/**
 * RLM (Retrieval Language Model) Type Definitions
 *
 * Types for the recursive retrieval-augmented generation system
 * that breaks down complex queries into sub-queries and synthesizes
 * answers from retrieved memory spans.
 */

/**
 * Configuration for the RLM controller
 *
 * @example
 * ```typescript
 * const config: RlmConfig = {
 *   maxDepth: 3,
 *   maxSubQueries: 5,
 *   tokenBudget: 4096,
 *   enableCache: true,
 *   cacheTtl: 300000, // 5 minutes
 *   retrievalTopK: 10,
 *   minQualityScore: 0.7,
 *   enableReflection: true,
 *   maxReflectionIterations: 2,
 * };
 * ```
 */
export interface RlmConfig {
  /** Maximum recursion depth for sub-queries (default: 3) */
  maxDepth?: number;

  /** Maximum number of sub-queries per level (default: 5) */
  maxSubQueries?: number;

  /** Token budget for generation (default: 4096) */
  tokenBudget?: number;

  /** Enable response caching (default: true) */
  enableCache?: boolean;

  /** Cache TTL in milliseconds (default: 300000 = 5 minutes) */
  cacheTtl?: number;

  /** Number of memory spans to retrieve (default: 10) */
  retrievalTopK?: number;

  /** Minimum quality score to accept answer (default: 0.7) */
  minQualityScore?: number;

  /** Enable self-reflection loop (default: false) */
  enableReflection?: boolean;

  /** Maximum reflection iterations (default: 2) */
  maxReflectionIterations?: number;
}

/**
 * Answer produced by the RLM controller
 *
 * @example
 * ```typescript
 * const answer: RlmAnswer = {
 *   text: 'Machine learning is a subset of artificial intelligence...',
 *   confidence: 0.92,
 *   qualityScore: 0.88,
 *   sources: [
 *     { id: 'mem-1', text: 'ML definition from textbook', similarityScore: 0.95, metadata: {} },
 *   ],
 *   subQueries: [
 *     { query: 'What is artificial intelligence?', answer: 'AI is...', depth: 1 },
 *   ],
 *   tokenUsage: { prompt: 512, completion: 256, total: 768 },
 *   cached: false,
 * };
 * ```
 */
export interface RlmAnswer {
  /** The generated answer text */
  text: string;

  /** Overall confidence in the answer (0.0 - 1.0) */
  confidence: number;

  /** Quality score based on source coverage and coherence (0.0 - 1.0) */
  qualityScore: number;

  /** Memory spans used to generate the answer */
  sources: MemorySpan[];

  /** Sub-queries generated and answered (if recursive) */
  subQueries?: SubQuery[];

  /** Token usage statistics */
  tokenUsage: TokenUsage;

  /** Whether this answer was served from cache */
  cached: boolean;
}

/**
 * A span of memory retrieved for context
 *
 * @example
 * ```typescript
 * const span: MemorySpan = {
 *   id: 'mem-abc123',
 *   text: 'Relevant context from memory...',
 *   similarityScore: 0.89,
 *   source: 'documentation',
 *   metadata: { timestamp: Date.now(), category: 'technical' },
 * };
 * ```
 */
export interface MemorySpan {
  /** Unique identifier for the memory span */
  id: string;

  /** The text content of the memory span */
  text: string;

  /** Cosine similarity score to the query (0.0 - 1.0) */
  similarityScore: number;

  /** Optional source identifier (e.g., document name, URL) */
  source?: string;

  /** Additional metadata associated with this span */
  metadata: Record<string, unknown>;
}

/**
 * A sub-query generated during recursive retrieval
 */
export interface SubQuery {
  /** The generated sub-query text */
  query: string;

  /** The answer to the sub-query */
  answer: string;

  /** Recursion depth at which this sub-query was generated */
  depth: number;
}

/**
 * Token usage statistics for a query
 */
export interface TokenUsage {
  /** Tokens used in the prompt (including context) */
  prompt: number;

  /** Tokens generated in the completion */
  completion: number;

  /** Total tokens used (prompt + completion) */
  total: number;
}

/**
 * Streaming token event
 *
 * Discriminated union for streaming responses:
 * - `type: 'token'` - A partial token was generated
 * - `type: 'done'` - Generation complete with final answer
 *
 * @example
 * ```typescript
 * for await (const event of controller.queryStream('What is AI?')) {
 *   if (event.type === 'token') {
 *     process.stdout.write(event.text);
 *   } else {
 *     console.log('\n\nFinal answer:', event.answer.text);
 *   }
 * }
 * ```
 */
export type StreamToken =
  | {
      /** Token event type */
      type: 'token';
      /** The partial text token */
      text: string;
      /** Always false for token events */
      done: false;
    }
  | {
      /** Done event type */
      type: 'done';
      /** The complete answer */
      answer: RlmAnswer;
      /** Always true for done events */
      done: true;
    };

/**
 * Internal cache entry for RLM answers
 */
export interface RlmCacheEntry {
  /** The cached answer */
  answer: RlmAnswer;

  /** Timestamp when the entry was cached */
  timestamp: number;

  /** Query hash for cache key */
  queryHash: string;
}

/**
 * Reflection result from self-evaluation loop
 */
export interface ReflectionResult {
  /** Whether the answer passed reflection criteria */
  passed: boolean;

  /** Critique of the current answer */
  critique?: string;

  /** Suggested improvements */
  suggestions?: string[];

  /** Updated quality score after reflection */
  updatedScore: number;

  /** Number of reflection iterations performed */
  iterations: number;
}
