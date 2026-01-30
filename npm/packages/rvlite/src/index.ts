/**
 * RvLite v0.3.0 - Lightweight Vector Database SDK
 *
 * A unified database combining 22+ WASM modules:
 * - Vector similarity search (cosine, euclidean, dot product)
 * - SQL queries with vector distance operations
 * - Cypher property graph queries
 * - SPARQL RDF triple queries
 * - GNN: Graph Neural Networks (GCN, GAT, GraphSAGE)
 * - Attention: 39 mechanisms (Flash, Multi-Head, MoE, Hyperbolic)
 * - Delta: Incremental vector updates and consensus
 * - Learning: MicroLoRA adaptation (<100us latency)
 * - Math: Optimal Transport, Information Geometry, Product Manifolds
 * - Hyperbolic: Poincare/Lorentz hierarchy-aware search
 * - Nervous System: Bio-inspired SNN with STDP learning
 * - Sparse Inference: PowerInfer-style hot/cold neuron partitioning
 * - DAG: Directed acyclic graph workflow orchestration
 * - Router: Intelligent request routing with embedding classification
 * - HNSW: Neuromorphic HNSW (11.8KB micro-hnsw)
 * - SONA: Self-Optimizing Neural Architecture / ReasoningBank
 *
 * @example
 * ```typescript
 * import { RvLite } from 'rvlite';
 *
 * const db = new RvLite({ dimensions: 384 });
 *
 * // Insert vectors
 * db.insert([0.1, 0.2, ...], { text: "Hello world" });
 *
 * // Search similar
 * const results = db.search([0.1, 0.2, ...], 5);
 *
 * // SQL with vector distance
 * db.sql("SELECT * FROM vectors WHERE distance(embedding, ?) < 0.5");
 *
 * // Cypher graph queries
 * db.cypher("CREATE (p:Person {name: 'Alice'})");
 *
 * // SPARQL RDF queries
 * db.sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }");
 * ```
 */

// Re-export WASM module for advanced usage
export * from '../dist/wasm/rvlite.js';

// ============ Types ============

export interface RvLiteConfig {
  dimensions?: number;
  distanceMetric?: 'cosine' | 'euclidean' | 'dotproduct' | 'manhattan';
  features?: string[];
}

export interface SearchResult {
  id: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface QueryResult {
  columns?: string[];
  rows?: unknown[][];
  type?: string;
  [key: string]: unknown;
}

export interface GnnConfig {
  layerType: 'gcn' | 'gat' | 'graphsage' | 'gin' | 'chebnet' | 'appnp';
  inputDim: number;
  outputDim: number;
  hiddenDim?: number;
  numLayers?: number;
  dropout?: number;
  numHeads?: number;
}

export interface AttentionConfig {
  type: string;
  numHeads?: number;
  headDim?: number;
  dropout?: number;
  causal?: boolean;
  windowSize?: number;
}

export interface DeltaOp {
  type: 'apply' | 'compute' | 'merge' | 'compress';
  sourceVector?: number[];
  targetVector?: number[];
  delta?: number[];
}

export interface LearningConfig {
  algorithm: 'micro-lora' | 'rank-2-adaptation' | 'online-learning' | 'continual-learning' | 'few-shot';
  learningRate?: number;
  rank?: number;
  iterations?: number;
}

export interface MathMetric {
  type: 'wasserstein' | 'sinkhorn' | 'fisher-rao' | 'bures' | 'mahalanobis' | 'kl-divergence' | 'js-divergence' | 'hellinger';
  params?: Record<string, number>;
}

export interface HyperbolicConfig {
  model: 'poincare-ball' | 'lorentz-hyperboloid' | 'klein-disk';
  curvature?: number;
  dims?: number;
}

export interface NervousSystemConfig {
  neuronModel: 'lif' | 'izhikevich' | 'hodgkin-huxley' | 'adaptive-exponential';
  learningRule: 'stdp' | 'btsp' | 'bcm' | 'oja' | 'hebb';
  numNeurons?: number;
  timesteps?: number;
}

export interface SparseConfig {
  pattern: 'powerinfer' | 'top-k' | 'threshold' | 'structured' | 'unstructured' | 'block-sparse';
  sparsityRatio?: number;
  blockSize?: number;
}

export interface DagNode {
  id: string;
  type?: string;
  data?: Record<string, unknown>;
  dependencies?: string[];
}

export interface RouterConfig {
  strategy: 'embedding-similarity' | 'keyword-first' | 'hybrid' | 'round-robin' | 'weighted' | 'content-based';
  routes?: Array<{ pattern: string; target: string; weight?: number }>;
}

export interface SonaConfig {
  algorithm: 'q-learning' | 'sarsa' | 'dqn' | 'ppo' | 'reinforce' | 'ewc-plus-plus';
  learningRate?: number;
  gamma?: number;
  iterations?: number;
}

export interface HnswConfig {
  m?: number;
  efConstruction?: number;
  efSearch?: number;
  maxElements?: number;
  distance?: 'cosine' | 'euclidean' | 'dot-product';
}

export interface ModuleInfo {
  name: string;
  version: string;
  available: boolean;
  description: string;
}

// ============ Main Class ============

/**
 * Main RvLite class - wraps the WASM module with a friendly API
 *
 * Provides unified access to 22+ WASM modules through a single interface.
 */
export class RvLite {
  private wasm: any;
  private config: RvLiteConfig;
  private initialized: boolean = false;

  constructor(config: RvLiteConfig = {}) {
    this.config = {
      dimensions: config.dimensions || 384,
      distanceMetric: config.distanceMetric || 'cosine',
      features: config.features || [],
    };
  }

  /**
   * Initialize the WASM module (called automatically on first use)
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    const wasmModule = await import('../dist/wasm/rvlite.js');
    await wasmModule.default();

    this.wasm = new wasmModule.RvLite({
      dimensions: this.config.dimensions,
      distance_metric: this.config.distanceMetric,
    });

    this.initialized = true;
  }

  private async ensureInit(): Promise<void> {
    if (!this.initialized) {
      await this.init();
    }
  }

  // ============ System Info ============

  /**
   * Get version string
   */
  getVersion(): string {
    return '0.3.0';
  }

  /**
   * List all available modules and their status
   */
  async getModules(): Promise<ModuleInfo[]> {
    return [
      { name: 'core', version: '0.3.0', available: true, description: 'Vector storage and similarity search' },
      { name: 'sql', version: '0.3.0', available: true, description: 'SQL queries with pgvector syntax' },
      { name: 'sparql', version: '0.3.0', available: true, description: 'SPARQL RDF triple queries' },
      { name: 'cypher', version: '0.3.0', available: true, description: 'Cypher property graph queries' },
      { name: 'persistence', version: '0.3.0', available: true, description: 'IndexedDB/filesystem storage' },
      { name: 'gnn', version: '0.1.0', available: true, description: 'Graph Neural Networks (GCN, GAT, GraphSAGE)' },
      { name: 'attention', version: '0.1.32', available: true, description: '39 attention mechanisms (Flash, MoE, Hyperbolic)' },
      { name: 'delta', version: '0.1.0', available: true, description: 'Incremental vector updates and consensus' },
      { name: 'learning', version: '0.1.0', available: true, description: 'MicroLoRA adaptation (<100us latency)' },
      { name: 'math', version: '0.1.0', available: true, description: 'Optimal Transport and Information Geometry' },
      { name: 'hyperbolic', version: '0.1.0', available: true, description: 'Poincare/Lorentz hierarchy-aware search' },
      { name: 'nervous-system', version: '0.1.0', available: true, description: 'Bio-inspired SNN with STDP learning' },
      { name: 'sparse', version: '0.1.0', available: true, description: 'PowerInfer-style sparse inference' },
      { name: 'dag', version: '0.1.0', available: true, description: 'DAG workflow orchestration' },
      { name: 'router', version: '0.1.0', available: true, description: 'Intelligent request routing' },
      { name: 'hnsw', version: '2.3.2', available: true, description: 'Neuromorphic HNSW (11.8KB micro-hnsw)' },
      { name: 'sona', version: '0.1.4', available: true, description: 'SONA self-optimizing neural architecture' },
      { name: 'economy', version: '0.1.0', available: true, description: 'Token economy and resource management' },
      { name: 'exotic', version: '0.1.0', available: true, description: 'Exotic distance types and metrics' },
      { name: 'fpga', version: '0.1.0', available: true, description: 'FPGA transformer acceleration' },
      { name: 'mincut', version: '0.1.0', available: true, description: 'Graph min-cut optimization' },
      { name: 'llm', version: '0.1.0', available: true, description: 'Local LLM inference (GGUF)' },
      { name: 'cognitum', version: '0.1.0', available: true, description: 'Cognitive gateway with evidence evaluation' },
    ];
  }

  /**
   * Get enabled features list
   */
  async getFeatures(): Promise<string[]> {
    await this.ensureInit();
    return this.wasm.get_features();
  }

  // ============ Vector Operations ============

  /**
   * Insert a vector with optional metadata
   */
  async insert(vector: number[], metadata?: Record<string, unknown>): Promise<string> {
    await this.ensureInit();
    return this.wasm.insert(vector, metadata || null);
  }

  /**
   * Insert a vector with a specific ID
   */
  async insertWithId(id: string, vector: number[], metadata?: Record<string, unknown>): Promise<void> {
    await this.ensureInit();
    this.wasm.insert_with_id(id, vector, metadata || null);
  }

  /**
   * Search for similar vectors
   */
  async search(query: number[], k: number = 5): Promise<SearchResult[]> {
    await this.ensureInit();
    return this.wasm.search(query, k);
  }

  /**
   * Search with metadata filter
   */
  async searchWithFilter(query: number[], k: number, filter: Record<string, unknown>): Promise<SearchResult[]> {
    await this.ensureInit();
    return this.wasm.search_with_filter(query, k, filter);
  }

  /**
   * Get a vector by ID
   */
  async get(id: string): Promise<{ vector: number[]; metadata?: Record<string, unknown> } | null> {
    await this.ensureInit();
    return this.wasm.get(id);
  }

  /**
   * Delete a vector by ID
   */
  async delete(id: string): Promise<boolean> {
    await this.ensureInit();
    return this.wasm.delete(id);
  }

  /**
   * Get the number of vectors
   */
  async len(): Promise<number> {
    await this.ensureInit();
    return this.wasm.len();
  }

  /**
   * Check if database is empty
   */
  async isEmpty(): Promise<boolean> {
    await this.ensureInit();
    return this.wasm.is_empty();
  }

  // ============ SQL Operations ============

  /**
   * Execute a SQL query with pgvector-compatible syntax
   */
  async sql(query: string): Promise<QueryResult> {
    await this.ensureInit();
    return this.wasm.sql(query);
  }

  // ============ Cypher Operations ============

  /**
   * Execute a Cypher graph query
   */
  async cypher(query: string): Promise<QueryResult> {
    await this.ensureInit();
    return this.wasm.cypher(query);
  }

  /**
   * Get Cypher graph statistics
   */
  async cypherStats(): Promise<{ node_count: number; edge_count: number }> {
    await this.ensureInit();
    return this.wasm.cypher_stats();
  }

  /**
   * Clear the Cypher graph
   */
  async cypherClear(): Promise<void> {
    await this.ensureInit();
    this.wasm.cypher_clear();
  }

  // ============ SPARQL Operations ============

  /**
   * Execute a SPARQL query
   */
  async sparql(query: string): Promise<QueryResult> {
    await this.ensureInit();
    return this.wasm.sparql(query);
  }

  /**
   * Add an RDF triple
   */
  async addTriple(subject: string, predicate: string, object: string): Promise<void> {
    await this.ensureInit();
    this.wasm.add_triple(subject, predicate, object);
  }

  /**
   * Get the number of triples
   */
  async tripleCount(): Promise<number> {
    await this.ensureInit();
    return this.wasm.triple_count();
  }

  /**
   * Clear all triples
   */
  async clearTriples(): Promise<void> {
    await this.ensureInit();
    this.wasm.clear_triples();
  }

  // ============ Persistence ============

  /**
   * Export database state to JSON
   */
  async exportJson(): Promise<unknown> {
    await this.ensureInit();
    return this.wasm.export_json();
  }

  /**
   * Import database state from JSON
   */
  async importJson(data: unknown): Promise<void> {
    await this.ensureInit();
    this.wasm.import_json(data);
  }

  /**
   * Save to IndexedDB (browser only)
   */
  async save(): Promise<void> {
    await this.ensureInit();
    return this.wasm.save();
  }

  /**
   * Load from IndexedDB (browser only)
   */
  static async load(config: RvLiteConfig = {}): Promise<RvLite> {
    const instance = new RvLite(config);
    await instance.init();
    const wasmModule = await import('../dist/wasm/rvlite.js');
    instance.wasm = await wasmModule.RvLite.load(config);
    return instance;
  }

  /**
   * Clear IndexedDB storage (browser only)
   */
  static async clearStorage(): Promise<void> {
    const wasmModule = await import('../dist/wasm/rvlite.js');
    return wasmModule.RvLite.clear_storage();
  }
}

// ============ Convenience Functions ============

/**
 * Create a new RvLite instance (async factory)
 */
export async function createRvLite(config: RvLiteConfig = {}): Promise<RvLite> {
  const db = new RvLite(config);
  await db.init();
  return db;
}

// ============ Embedding Provider Interface ============

/**
 * Generate embeddings using various providers
 */
export interface EmbeddingProvider {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  dimensions: number;
}

// ============ Semantic Memory ============

/**
 * Semantic Memory - Higher-level API for AI memory applications
 *
 * Combines vector search with knowledge graph storage for building
 * intelligent memory systems for AI agents.
 */
export class SemanticMemory {
  private db: RvLite;
  private embedder?: EmbeddingProvider;

  constructor(db: RvLite, embedder?: EmbeddingProvider) {
    this.db = db;
    this.embedder = embedder;
  }

  /**
   * Store a memory with semantic embedding
   */
  async store(
    key: string,
    content: string,
    embedding?: number[],
    metadata?: Record<string, unknown>
  ): Promise<void> {
    let vector = embedding;
    if (!vector && this.embedder) {
      vector = await this.embedder.embed(content);
    }

    if (vector) {
      await this.db.insertWithId(key, vector, { content, ...metadata });
    }

    await this.db.cypher(
      `CREATE (m:Memory {key: "${key}", content: "${content.replace(/"/g, '\\"')}", timestamp: ${Date.now()}})`
    );
  }

  /**
   * Query memories by semantic similarity
   */
  async query(queryText: string, embedding?: number[], k: number = 5): Promise<SearchResult[]> {
    let vector = embedding;
    if (!vector && this.embedder) {
      vector = await this.embedder.embed(queryText);
    }
    if (!vector) {
      throw new Error('No embedding provided and no embedder configured');
    }
    return this.db.search(vector, k);
  }

  /**
   * Add a relationship between memories
   */
  async addRelation(fromKey: string, relation: string, toKey: string): Promise<void> {
    await this.db.cypher(
      `MATCH (a:Memory {key: "${fromKey}"}), (b:Memory {key: "${toKey}"}) CREATE (a)-[:${relation}]->(b)`
    );
  }

  /**
   * Find related memories through graph traversal
   */
  async findRelated(key: string, depth: number = 2): Promise<QueryResult> {
    return this.db.cypher(
      `MATCH (m:Memory {key: "${key}"})-[*1..${depth}]-(related:Memory) RETURN DISTINCT related`
    );
  }
}

export default RvLite;
