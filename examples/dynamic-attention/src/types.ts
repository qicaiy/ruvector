/**
 * Type definitions for Dynamic Attention system
 *
 * Combines FastGRNN neural routing with attention-enhanced feature extraction
 */

// ============================================================================
// Core Types
// ============================================================================

/**
 * Attention mechanism type selection
 */
export type AttentionType =
  | 'dot-product'
  | 'multi-head'
  | 'hyperbolic'
  | 'flash'
  | 'linear'
  | 'local-global'
  | 'moe';

/**
 * SIMD optimization level
 */
export type SIMDLevel = 'none' | 'sse4' | 'avx2' | 'avx512' | 'neon';

/**
 * Configuration for the dynamic attention pipeline
 */
export interface DynamicAttentionConfig {
  /** Embedding dimension */
  dim: number;
  /** Number of attention heads (for multi-head attention) */
  numHeads: number;
  /** Hidden dimension for FastGRNN */
  hiddenDim: number;
  /** Attention type to use */
  attentionType: AttentionType;
  /** Enable SIMD optimization */
  enableSIMD: boolean;
  /** SIMD optimization level */
  simdLevel: SIMDLevel;
  /** Temperature for attention softmax */
  temperature: number;
  /** Dropout rate (training only) */
  dropout: number;
  /** Enable quantization for memory efficiency */
  enableQuantization: boolean;
  /** Quantization bits (8 or 16) */
  quantizationBits: 8 | 16;
}

/**
 * Default configuration
 */
export const DEFAULT_CONFIG: DynamicAttentionConfig = {
  dim: 384,
  numHeads: 8,
  hiddenDim: 64,
  attentionType: 'multi-head',
  enableSIMD: true,
  simdLevel: 'avx2',
  temperature: 1.0,
  dropout: 0.1,
  enableQuantization: false,
  quantizationBits: 8,
};

// ============================================================================
// Input/Output Types
// ============================================================================

/**
 * Query input for the dynamic attention system
 */
export interface QueryInput {
  /** Query embedding vector */
  embedding: Float32Array;
  /** Optional context vectors for attention */
  context?: Float32Array[];
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Candidate for routing decisions
 */
export interface RoutingCandidate {
  /** Unique identifier */
  id: string;
  /** Candidate embedding */
  embedding: Float32Array;
  /** Historical success rate (0-1) */
  successRate?: number;
  /** Average latency in ms */
  avgLatency?: number;
  /** Cost per query */
  cost?: number;
  /** Capability tags */
  capabilities?: string[];
}

/**
 * Routing decision from the system
 */
export interface RoutingDecision {
  /** Selected candidate ID */
  candidateId: string;
  /** Confidence score (0-1) */
  confidence: number;
  /** Attention weights used */
  attentionWeights: Float32Array;
  /** Whether to use lightweight model */
  useLightweight: boolean;
  /** Uncertainty estimate */
  uncertainty: number;
  /** Routing reason explanation */
  reason: string;
}

/**
 * Full pipeline result
 */
export interface PipelineResult {
  /** Ranked routing decisions */
  decisions: RoutingDecision[];
  /** Enriched query embedding after attention */
  enrichedEmbedding: Float32Array;
  /** Performance metrics */
  metrics: PipelineMetrics;
}

// ============================================================================
// Metrics Types
// ============================================================================

/**
 * Performance metrics for the pipeline
 */
export interface PipelineMetrics {
  /** Total pipeline latency in microseconds */
  totalLatencyUs: number;
  /** Attention computation time in microseconds */
  attentionLatencyUs: number;
  /** FastGRNN inference time in microseconds */
  fastgrnnLatencyUs: number;
  /** Feature engineering time in microseconds */
  featureLatencyUs: number;
  /** Memory usage in bytes */
  memoryBytes: number;
  /** Whether SIMD was used */
  simdUsed: boolean;
  /** SIMD level used */
  simdLevel: SIMDLevel;
  /** Number of candidates processed */
  candidatesProcessed: number;
  /** Throughput in queries per second */
  throughputQps: number;
}

/**
 * Benchmark result
 */
export interface BenchmarkResult {
  /** Benchmark name */
  name: string;
  /** Number of iterations */
  iterations: number;
  /** Mean latency in microseconds */
  meanLatencyUs: number;
  /** Standard deviation */
  stdDevUs: number;
  /** P50 latency */
  p50Us: number;
  /** P95 latency */
  p95Us: number;
  /** P99 latency */
  p99Us: number;
  /** Min latency */
  minUs: number;
  /** Max latency */
  maxUs: number;
  /** Throughput in operations per second */
  opsPerSecond: number;
  /** Memory usage in bytes */
  memoryBytes: number;
}

// ============================================================================
// SIMD Types
// ============================================================================

/**
 * SIMD capability detection result
 */
export interface SIMDCapabilities {
  /** Available SIMD levels */
  available: SIMDLevel[];
  /** Recommended level for this hardware */
  recommended: SIMDLevel;
  /** CPU vendor */
  cpuVendor: string;
  /** CPU model */
  cpuModel: string;
  /** Number of cores */
  cores: number;
  /** Cache sizes in KB */
  cacheSizes: {
    l1d: number;
    l1i: number;
    l2: number;
    l3: number;
  };
}

/**
 * SIMD optimization hints
 */
export interface SIMDHints {
  /** Optimal vector width for this CPU */
  vectorWidth: number;
  /** Optimal batch size for SIMD */
  batchSize: number;
  /** Whether to use aligned memory */
  useAlignedMemory: boolean;
  /** Memory alignment in bytes */
  alignment: number;
}

// ============================================================================
// Training Types
// ============================================================================

/**
 * Training sample for the combined model
 */
export interface TrainingSample {
  /** Query embedding */
  query: Float32Array;
  /** Context embeddings */
  context: Float32Array[];
  /** Candidates */
  candidates: RoutingCandidate[];
  /** Ground truth best candidate ID */
  bestCandidateId: string;
  /** Ground truth was lightweight sufficient */
  lightweightSufficient: boolean;
}

/**
 * Training configuration
 */
export interface TrainingConfig {
  /** Learning rate */
  learningRate: number;
  /** Batch size */
  batchSize: number;
  /** Number of epochs */
  epochs: number;
  /** Weight decay for regularization */
  weightDecay: number;
  /** Gradient clipping threshold */
  gradientClip: number;
  /** Warmup steps */
  warmupSteps: number;
  /** Use knowledge distillation */
  useDistillation: boolean;
  /** Teacher model path (for distillation) */
  teacherModelPath?: string;
}

/**
 * Training metrics
 */
export interface TrainingMetrics {
  /** Current epoch */
  epoch: number;
  /** Training loss */
  trainLoss: number;
  /** Validation loss */
  valLoss: number;
  /** Routing accuracy */
  routingAccuracy: number;
  /** Lightweight prediction accuracy */
  lightweightAccuracy: number;
  /** Learning rate */
  currentLr: number;
}
