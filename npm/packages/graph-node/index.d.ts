/**
 * RuVector Graph Database - Native Node.js Bindings
 *
 * High-performance graph database with hypergraph support,
 * Cypher-like queries, and vector similarity search.
 */

export type DistanceMetric = 'Euclidean' | 'Cosine' | 'DotProduct' | 'Manhattan';
export type TemporalGranularity = 'Hourly' | 'Daily' | 'Monthly' | 'Yearly';

export interface GraphOptions {
  distanceMetric?: DistanceMetric;
  dimensions?: number;
  storagePath?: string;
}

export interface Node {
  id: string;
  embedding: Float32Array | number[];
  properties?: Record<string, string>;
}

export interface Edge {
  from: string;
  to: string;
  description: string;
  embedding: Float32Array | number[];
  confidence?: number;
  metadata?: Record<string, string>;
}

export interface Hyperedge {
  nodes: string[];
  description: string;
  embedding: Float32Array | number[];
  confidence?: number;
  metadata?: Record<string, string>;
}

export interface HyperedgeQuery {
  embedding: Float32Array | number[];
  k: number;
}

export interface HyperedgeResult {
  id: string;
  score: number;
}

export interface QueryResult {
  nodes: Node[];
  edges: Edge[];
  stats?: GraphStats;
}

export interface GraphStats {
  totalNodes: number;
  totalEdges: number;
  avgDegree: number;
}

export interface BatchInsert {
  nodes: Node[];
  edges: Edge[];
}

export interface BatchResult {
  nodeIds: string[];
  edgeIds: string[];
}

export interface TemporalHyperedge {
  hyperedge: Hyperedge;
  timestamp: number;
  expiresAt?: number;
  granularity: TemporalGranularity;
}

/**
 * High-performance graph database with hypergraph support
 *
 * @example
 * ```typescript
 * const db = new GraphDatabase({
 *   distanceMetric: 'Cosine',
 *   dimensions: 384
 * });
 *
 * // Create nodes
 * await db.createNode({
 *   id: 'alice',
 *   embedding: new Float32Array([0.1, 0.2, 0.3]),
 *   properties: { name: 'Alice', age: '30' }
 * });
 *
 * // Create edges
 * await db.createEdge({
 *   from: 'alice',
 *   to: 'bob',
 *   description: 'knows',
 *   embedding: new Float32Array([0.5, 0.5, 0.5])
 * });
 *
 * // Query with Cypher-like syntax
 * const results = await db.query('MATCH (n) RETURN n LIMIT 10');
 * ```
 */
export class GraphDatabase {
  /**
   * Create a new graph database
   * @param options - Configuration options
   */
  constructor(options?: GraphOptions);

  /**
   * Create a node in the graph
   * @param node - Node data
   * @returns Node ID
   */
  createNode(node: Node): Promise<string>;

  /**
   * Create an edge between two nodes
   * @param edge - Edge data
   * @returns Edge ID
   */
  createEdge(edge: Edge): Promise<string>;

  /**
   * Create a hyperedge connecting multiple nodes
   * @param hyperedge - Hyperedge data
   * @returns Hyperedge ID
   */
  createHyperedge(hyperedge: Hyperedge): Promise<string>;

  /**
   * Query the graph using Cypher-like syntax
   * @param cypher - Cypher query string
   * @returns Query results
   */
  query(cypher: string): Promise<QueryResult>;

  /**
   * Query the graph synchronously
   * @param cypher - Cypher query string
   * @returns Query results
   */
  querySync(cypher: string): QueryResult;

  /**
   * Search for similar hyperedges
   * @param query - Search query
   * @returns Hyperedge results sorted by similarity
   */
  searchHyperedges(query: HyperedgeQuery): Promise<HyperedgeResult[]>;

  /**
   * Get k-hop neighbors from a starting node
   * @param startNode - Starting node ID
   * @param k - Number of hops
   * @returns List of neighbor node IDs
   */
  kHopNeighbors(startNode: string, k: number): Promise<string[]>;

  /**
   * Begin a new transaction
   * @returns Transaction ID
   */
  begin(): Promise<string>;

  /**
   * Commit a transaction
   * @param txId - Transaction ID
   */
  commit(txId: string): Promise<void>;

  /**
   * Rollback a transaction
   * @param txId - Transaction ID
   */
  rollback(txId: string): Promise<void>;

  /**
   * Batch insert nodes and edges
   * @param batch - Batch data
   * @returns Batch result with IDs
   */
  batchInsert(batch: BatchInsert): Promise<BatchResult>;

  /**
   * Subscribe to graph changes
   * @param callback - Change callback function
   */
  subscribe(callback: (change: any) => void): void;

  /**
   * Get graph statistics
   * @returns Graph statistics
   */
  stats(): Promise<GraphStats>;
}

/**
 * Streaming query result iterator
 */
export class QueryResultStream {
  /**
   * Get the next result from the stream
   * @returns Next query result or null if exhausted
   */
  next(): Promise<QueryResult | null>;
}

/**
 * Streaming hyperedge result iterator
 */
export class HyperedgeStream {
  /**
   * Get the next hyperedge result
   * @returns Next result or null if exhausted
   */
  next(): Promise<HyperedgeResult | null>;

  /**
   * Collect all remaining results
   * @returns Array of all remaining results
   */
  collect(): HyperedgeResult[];
}

/**
 * Node stream iterator
 */
export class NodeStream {
  /**
   * Get the next node
   * @returns Next node or null if exhausted
   */
  next(): Promise<Node | null>;

  /**
   * Collect all remaining nodes
   * @returns Array of all remaining nodes
   */
  collect(): Node[];
}

/**
 * Get library version
 * @returns Version string
 */
export function version(): string;

/**
 * Test function to verify bindings
 * @returns Greeting message
 */
export function hello(): string;
