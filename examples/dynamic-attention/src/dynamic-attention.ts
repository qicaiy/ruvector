/**
 * Dynamic Attention Pipeline
 *
 * Combines FastGRNN neural routing with attention-enhanced feature extraction
 * for intelligent query routing with sub-millisecond inference.
 */

import type {
  DynamicAttentionConfig,
  QueryInput,
  RoutingCandidate,
  RoutingDecision,
  PipelineResult,
  PipelineMetrics,
  AttentionType,
} from './types.js';
import { DEFAULT_CONFIG } from './types.js';
import {
  detectSIMDCapabilities,
  getSIMDHints,
  dotProduct,
  normalizeL2InPlace,
  softmaxInPlace,
  hrTimeUs,
} from './simd-utils.js';

// ============================================================================
// Attention Factory
// ============================================================================

/**
 * Create an attention mechanism based on type
 */
function createAttention(type: AttentionType, config: DynamicAttentionConfig): AttentionMechanism {
  // In production, this would use the actual @ruvector/attention bindings
  // Here we provide a reference implementation
  switch (type) {
    case 'multi-head':
      return new MultiHeadAttention(config.dim, config.numHeads);
    case 'dot-product':
      return new ScaledDotProductAttention(config.dim);
    case 'hyperbolic':
      return new HyperbolicAttention(config.dim, 1.0);
    case 'flash':
      return new FlashAttention(config.dim, 64);
    case 'linear':
      return new LinearAttention(config.dim, config.dim);
    case 'local-global':
      return new LocalGlobalAttention(config.dim, 128, 8);
    case 'moe':
      return new MoEAttention(config.dim, 8, 2);
    default:
      return new ScaledDotProductAttention(config.dim);
  }
}

// ============================================================================
// Attention Mechanism Interface & Implementations
// ============================================================================

interface AttentionMechanism {
  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array;
  readonly dim: number;
}

/**
 * Scaled Dot-Product Attention
 */
class ScaledDotProductAttention implements AttentionMechanism {
  readonly dim: number;
  private scale: number;

  constructor(dim: number) {
    this.dim = dim;
    this.scale = 1.0 / Math.sqrt(dim);
  }

  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array {
    if (keys.length === 0) return new Float32Array(query);

    // Compute attention scores
    const scores = new Float32Array(keys.length);
    for (let i = 0; i < keys.length; i++) {
      scores[i] = dotProduct(query, keys[i]) * this.scale;
    }

    // Apply softmax
    softmaxInPlace(scores);

    // Compute weighted sum of values
    const output = new Float32Array(this.dim);
    for (let i = 0; i < values.length; i++) {
      for (let j = 0; j < this.dim; j++) {
        output[j] += scores[i] * values[i][j];
      }
    }

    return output;
  }
}

/**
 * Multi-Head Attention
 */
class MultiHeadAttention implements AttentionMechanism {
  readonly dim: number;
  private numHeads: number;
  private headDim: number;
  private heads: ScaledDotProductAttention[];

  constructor(dim: number, numHeads: number) {
    if (dim % numHeads !== 0) {
      throw new Error(`Dimension ${dim} must be divisible by number of heads ${numHeads}`);
    }
    this.dim = dim;
    this.numHeads = numHeads;
    this.headDim = dim / numHeads;
    this.heads = Array.from({ length: numHeads }, () =>
      new ScaledDotProductAttention(this.headDim)
    );
  }

  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array {
    if (keys.length === 0) return new Float32Array(query);

    const output = new Float32Array(this.dim);

    // Process each head
    for (let h = 0; h < this.numHeads; h++) {
      const offset = h * this.headDim;

      // Extract head slices
      const qHead = query.subarray(offset, offset + this.headDim);
      const kHeads = keys.map(k => k.subarray(offset, offset + this.headDim));
      const vHeads = values.map(v => v.subarray(offset, offset + this.headDim));

      // Compute attention for this head
      const headOutput = this.heads[h].compute(
        new Float32Array(qHead),
        kHeads.map(k => new Float32Array(k)),
        vHeads.map(v => new Float32Array(v))
      );

      // Copy to output
      for (let i = 0; i < this.headDim; i++) {
        output[offset + i] = headOutput[i];
      }
    }

    return output;
  }
}

/**
 * Hyperbolic Attention (Poincare ball model)
 */
class HyperbolicAttention implements AttentionMechanism {
  readonly dim: number;
  private curvature: number;

  constructor(dim: number, curvature: number) {
    this.dim = dim;
    this.curvature = curvature;
  }

  private poincareDistance(a: Float32Array, b: Float32Array): number {
    const diff = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      diff[i] = a[i] - b[i];
    }
    const normDiffSq = dotProduct(diff, diff);
    const normASq = dotProduct(a, a);
    const normBSq = dotProduct(b, b);

    const numerator = normDiffSq;
    const denominator = (1 - normASq) * (1 - normBSq);

    if (denominator <= 0) return 0;

    const x = 1 + 2 * this.curvature * numerator / Math.max(denominator, 1e-10);
    return Math.acosh(Math.max(1, x)) / Math.sqrt(this.curvature);
  }

  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array {
    if (keys.length === 0) return new Float32Array(query);

    // Project to Poincare ball
    const projQuery = this.projectToBall(query);
    const projKeys = keys.map(k => this.projectToBall(k));

    // Compute hyperbolic distances
    const scores = new Float32Array(keys.length);
    for (let i = 0; i < keys.length; i++) {
      scores[i] = -this.poincareDistance(projQuery, projKeys[i]);
    }

    // Apply softmax
    softmaxInPlace(scores);

    // Weighted Frechet mean approximation
    const output = new Float32Array(this.dim);
    for (let i = 0; i < values.length; i++) {
      for (let j = 0; j < this.dim; j++) {
        output[j] += scores[i] * values[i][j];
      }
    }

    return output;
  }

  private projectToBall(v: Float32Array): Float32Array {
    const norm = Math.sqrt(dotProduct(v, v));
    const maxNorm = 1 - 1e-5;
    if (norm >= maxNorm) {
      const scale = maxNorm / norm;
      const projected = new Float32Array(v.length);
      for (let i = 0; i < v.length; i++) {
        projected[i] = v[i] * scale;
      }
      return projected;
    }
    return new Float32Array(v);
  }
}

/**
 * Flash Attention (memory-efficient tiled computation)
 */
class FlashAttention implements AttentionMechanism {
  readonly dim: number;
  private blockSize: number;

  constructor(dim: number, blockSize: number) {
    this.dim = dim;
    this.blockSize = blockSize;
  }

  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array {
    // Simplified flash attention (actual implementation uses tiling)
    const base = new ScaledDotProductAttention(this.dim);
    return base.compute(query, keys, values);
  }
}

/**
 * Linear Attention (O(n) complexity via kernel approximation)
 */
class LinearAttention implements AttentionMechanism {
  readonly dim: number;
  private numFeatures: number;

  constructor(dim: number, numFeatures: number) {
    this.dim = dim;
    this.numFeatures = numFeatures;
  }

  private featureMap(x: Float32Array): Float32Array {
    // ELU feature map: max(0, x) + exp(min(0, x)) - 1
    const result = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      result[i] = x[i] >= 0 ? x[i] + 1 : Math.exp(x[i]);
    }
    return result;
  }

  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array {
    if (keys.length === 0) return new Float32Array(query);

    const qPhi = this.featureMap(query);

    // Compute KV sum
    const kvSum = new Float32Array(this.dim * this.dim);
    let kSum = new Float32Array(this.dim);

    for (let i = 0; i < keys.length; i++) {
      const kPhi = this.featureMap(keys[i]);
      for (let j = 0; j < this.dim; j++) {
        kSum[j] += kPhi[j];
        for (let k = 0; k < this.dim; k++) {
          kvSum[j * this.dim + k] += kPhi[j] * values[i][k];
        }
      }
    }

    // Compute output
    const output = new Float32Array(this.dim);
    let normalizer = 0;

    for (let i = 0; i < this.dim; i++) {
      normalizer += qPhi[i] * kSum[i];
      for (let j = 0; j < this.dim; j++) {
        output[j] += qPhi[i] * kvSum[i * this.dim + j];
      }
    }

    if (normalizer > 1e-10) {
      for (let i = 0; i < this.dim; i++) {
        output[i] /= normalizer;
      }
    }

    return output;
  }
}

/**
 * Local-Global Attention (Longformer-style)
 */
class LocalGlobalAttention implements AttentionMechanism {
  readonly dim: number;
  private localWindow: number;
  private globalTokens: number;

  constructor(dim: number, localWindow: number, globalTokens: number) {
    this.dim = dim;
    this.localWindow = localWindow;
    this.globalTokens = globalTokens;
  }

  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array {
    // Use standard attention for simplicity (actual impl uses sparse patterns)
    const base = new ScaledDotProductAttention(this.dim);
    return base.compute(query, keys, values);
  }
}

/**
 * Mixture of Experts Attention
 */
class MoEAttention implements AttentionMechanism {
  readonly dim: number;
  private numExperts: number;
  private topK: number;
  private experts: ScaledDotProductAttention[];

  constructor(dim: number, numExperts: number, topK: number) {
    this.dim = dim;
    this.numExperts = numExperts;
    this.topK = topK;
    this.experts = Array.from({ length: numExperts }, () =>
      new ScaledDotProductAttention(dim)
    );
  }

  compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array {
    if (keys.length === 0) return new Float32Array(query);

    // Simple router: use first topK experts
    const routerScores = new Float32Array(this.numExperts);
    for (let i = 0; i < this.numExperts; i++) {
      routerScores[i] = dotProduct(query, new Float32Array(query.length).fill(1 / (i + 1)));
    }
    softmaxInPlace(routerScores);

    // Combine expert outputs
    const output = new Float32Array(this.dim);
    for (let i = 0; i < this.topK; i++) {
      const expertOutput = this.experts[i].compute(query, keys, values);
      const weight = routerScores[i];
      for (let j = 0; j < this.dim; j++) {
        output[j] += weight * expertOutput[j];
      }
    }

    return output;
  }
}

// ============================================================================
// FastGRNN Simulator (Reference Implementation)
// ============================================================================

/**
 * FastGRNN cell for routing decisions
 * This is a reference implementation - production uses native Rust bindings
 */
class FastGRNNCell {
  private inputDim: number;
  private hiddenDim: number;
  private nu: number;
  private zeta: number;

  // Weight matrices (would be loaded from model in production)
  private wReset: Float32Array;
  private wUpdate: Float32Array;
  private wCandidate: Float32Array;
  private wRecurrent: Float32Array;

  constructor(inputDim: number, hiddenDim: number, nu = 1.0, zeta = 1.0) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.nu = nu;
    this.zeta = zeta;

    // Initialize weights randomly (would be loaded from model)
    this.wReset = this.randomMatrix(hiddenDim, inputDim);
    this.wUpdate = this.randomMatrix(hiddenDim, inputDim);
    this.wCandidate = this.randomMatrix(hiddenDim, inputDim);
    this.wRecurrent = this.randomMatrix(hiddenDim, hiddenDim);
  }

  private randomMatrix(rows: number, cols: number): Float32Array {
    const size = rows * cols;
    const arr = new Float32Array(size);
    const scale = Math.sqrt(2.0 / (rows + cols));
    for (let i = 0; i < size; i++) {
      arr[i] = (Math.random() - 0.5) * 2 * scale;
    }
    return arr;
  }

  private sigmoid(x: number): number {
    if (x > 0) {
      return 1.0 / (1.0 + Math.exp(-x * this.nu));
    } else {
      const ex = Math.exp(x * this.nu);
      return ex / (1.0 + ex);
    }
  }

  private tanh(x: number): number {
    return Math.tanh(x * this.zeta);
  }

  private matVec(matrix: Float32Array, vec: Float32Array, rows: number, cols: number): Float32Array {
    const result = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        sum += matrix[i * cols + j] * vec[j];
      }
      result[i] = sum;
    }
    return result;
  }

  forward(input: Float32Array, hidden: Float32Array): { output: Float32Array; hidden: Float32Array } {
    // Reset gate
    const r = this.matVec(this.wReset, input, this.hiddenDim, this.inputDim);
    for (let i = 0; i < this.hiddenDim; i++) {
      r[i] = this.sigmoid(r[i]);
    }

    // Update gate
    const u = this.matVec(this.wUpdate, input, this.hiddenDim, this.inputDim);
    for (let i = 0; i < this.hiddenDim; i++) {
      u[i] = this.sigmoid(u[i]);
    }

    // Candidate
    const rh = new Float32Array(this.hiddenDim);
    for (let i = 0; i < this.hiddenDim; i++) {
      rh[i] = r[i] * hidden[i];
    }
    const recurrent = this.matVec(this.wRecurrent, rh, this.hiddenDim, this.hiddenDim);
    const c = this.matVec(this.wCandidate, input, this.hiddenDim, this.inputDim);
    for (let i = 0; i < this.hiddenDim; i++) {
      c[i] = this.tanh(c[i] + recurrent[i]);
    }

    // New hidden state
    const newHidden = new Float32Array(this.hiddenDim);
    for (let i = 0; i < this.hiddenDim; i++) {
      newHidden[i] = u[i] * hidden[i] + (1 - u[i]) * c[i];
    }

    return { output: newHidden, hidden: newHidden };
  }
}

// ============================================================================
// Dynamic Attention Pipeline
// ============================================================================

/**
 * Dynamic Attention Pipeline
 *
 * Combines attention mechanisms with FastGRNN for intelligent routing
 */
export class DynamicAttentionPipeline {
  private config: DynamicAttentionConfig;
  private attention: AttentionMechanism;
  private fastgrnn: FastGRNNCell;
  private simdCapabilities: ReturnType<typeof detectSIMDCapabilities>;

  constructor(config: Partial<DynamicAttentionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.attention = createAttention(this.config.attentionType, this.config);
    this.fastgrnn = new FastGRNNCell(
      this.config.dim + 5, // embedding + 5 features
      this.config.hiddenDim
    );
    this.simdCapabilities = detectSIMDCapabilities();
  }

  /**
   * Process a query through the full pipeline
   */
  async process(input: QueryInput, candidates: RoutingCandidate[]): Promise<PipelineResult> {
    const startTime = hrTimeUs();

    // Step 1: Attention-based context aggregation
    const attentionStart = hrTimeUs();
    const enrichedEmbedding = this.computeAttention(input);
    const attentionLatency = hrTimeUs() - attentionStart;

    // Step 2: Feature engineering
    const featureStart = hrTimeUs();
    const features = candidates.map(c => this.extractFeatures(enrichedEmbedding, c));
    const featureLatency = hrTimeUs() - featureStart;

    // Step 3: FastGRNN inference for routing scores
    const fastgrnnStart = hrTimeUs();
    const scores = this.computeRoutingScores(features);
    const fastgrnnLatency = hrTimeUs() - fastgrnnStart;

    // Step 4: Generate routing decisions
    const decisions = this.generateDecisions(candidates, scores, enrichedEmbedding);

    const totalLatency = hrTimeUs() - startTime;

    const metrics: PipelineMetrics = {
      totalLatencyUs: totalLatency,
      attentionLatencyUs: attentionLatency,
      fastgrnnLatencyUs: fastgrnnLatency,
      featureLatencyUs: featureLatency,
      memoryBytes: this.estimateMemoryUsage(),
      simdUsed: this.config.enableSIMD,
      simdLevel: this.config.simdLevel,
      candidatesProcessed: candidates.length,
      throughputQps: 1_000_000 / totalLatency,
    };

    return {
      decisions,
      enrichedEmbedding,
      metrics,
    };
  }

  /**
   * Compute attention over context
   */
  private computeAttention(input: QueryInput): Float32Array {
    if (!input.context || input.context.length === 0) {
      return new Float32Array(input.embedding);
    }

    return this.attention.compute(input.embedding, input.context, input.context);
  }

  /**
   * Extract features for FastGRNN input
   */
  private extractFeatures(query: Float32Array, candidate: RoutingCandidate): Float32Array {
    // Combine embedding similarity with metadata features
    const similarity = dotProduct(query, candidate.embedding) /
      (Math.sqrt(dotProduct(query, query)) * Math.sqrt(dotProduct(candidate.embedding, candidate.embedding)));

    const features = new Float32Array(this.config.dim + 5);

    // Copy query embedding
    features.set(query.subarray(0, Math.min(query.length, this.config.dim)));

    // Add metadata features
    features[this.config.dim] = similarity;
    features[this.config.dim + 1] = candidate.successRate ?? 0.5;
    features[this.config.dim + 2] = candidate.avgLatency ? 1.0 / (1.0 + candidate.avgLatency / 100) : 0.5;
    features[this.config.dim + 3] = candidate.cost ? 1.0 / (1.0 + candidate.cost) : 0.5;
    features[this.config.dim + 4] = candidate.capabilities?.length ?? 0 / 10;

    return features;
  }

  /**
   * Compute routing scores using FastGRNN
   */
  private computeRoutingScores(features: Float32Array[]): number[] {
    const hidden = new Float32Array(this.config.hiddenDim);
    const scores: number[] = [];

    for (const feature of features) {
      const { output } = this.fastgrnn.forward(feature, hidden);

      // Average hidden state as score
      let score = 0;
      for (let i = 0; i < output.length; i++) {
        score += output[i];
      }
      score /= output.length;

      // Sigmoid to [0, 1]
      scores.push(1.0 / (1.0 + Math.exp(-score)));
    }

    return scores;
  }

  /**
   * Generate final routing decisions
   */
  private generateDecisions(
    candidates: RoutingCandidate[],
    scores: number[],
    enrichedEmbedding: Float32Array
  ): RoutingDecision[] {
    const decisions: RoutingDecision[] = candidates.map((candidate, i) => {
      const confidence = scores[i];
      const uncertainty = this.estimateUncertainty(scores[i]);

      return {
        candidateId: candidate.id,
        confidence,
        attentionWeights: enrichedEmbedding, // Simplified
        useLightweight: confidence < 0.7 || uncertainty > 0.3,
        uncertainty,
        reason: this.generateReason(candidate, confidence, uncertainty),
      };
    });

    // Sort by confidence (descending)
    decisions.sort((a, b) => b.confidence - a.confidence);

    return decisions;
  }

  /**
   * Estimate uncertainty using simple heuristics
   */
  private estimateUncertainty(score: number): number {
    // Higher uncertainty near 0.5
    return 1 - 2 * Math.abs(score - 0.5);
  }

  /**
   * Generate human-readable routing reason
   */
  private generateReason(candidate: RoutingCandidate, confidence: number, uncertainty: number): string {
    const parts: string[] = [];

    if (confidence > 0.8) {
      parts.push('High confidence match');
    } else if (confidence > 0.6) {
      parts.push('Moderate confidence');
    } else {
      parts.push('Low confidence, consider alternatives');
    }

    if (candidate.successRate && candidate.successRate > 0.9) {
      parts.push('excellent track record');
    }

    if (candidate.avgLatency && candidate.avgLatency < 100) {
      parts.push('fast response time');
    }

    if (uncertainty > 0.4) {
      parts.push('high uncertainty');
    }

    return parts.join(', ');
  }

  /**
   * Estimate memory usage
   */
  private estimateMemoryUsage(): number {
    const attentionParams = this.config.dim * this.config.dim * 4; // Rough estimate
    const fastgrnnParams = (this.config.dim + 5) * this.config.hiddenDim * 4 * 4; // 4 weight matrices
    return attentionParams + fastgrnnParams;
  }

  /**
   * Get current configuration
   */
  getConfig(): DynamicAttentionConfig {
    return { ...this.config };
  }

  /**
   * Get SIMD capabilities
   */
  getSIMDCapabilities() {
    return this.simdCapabilities;
  }

  /**
   * Update attention type
   */
  setAttentionType(type: AttentionType): void {
    this.config.attentionType = type;
    this.attention = createAttention(type, this.config);
  }
}

/**
 * Create a new pipeline with default configuration
 */
export function createPipeline(config?: Partial<DynamicAttentionConfig>): DynamicAttentionPipeline {
  return new DynamicAttentionPipeline(config);
}
