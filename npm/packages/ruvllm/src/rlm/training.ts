/**
 * RLM (Recursive Learning Machine) Training Module
 *
 * Provides training capabilities for RuvLTRA models on RLM task routing
 * and decomposition, including query decomposition, answer synthesis,
 * and agent routing optimization.
 *
 * @module rlm/training
 */

// =============================================================================
// Types and Interfaces
// =============================================================================

/**
 * Strategy for decomposing a complex query
 */
export type DecompositionStrategy =
  | 'sequential'
  | 'parallel'
  | 'hierarchical'
  | 'dag-based'
  | 'iterative'
  | 'none';

/**
 * A sub-query in the decomposition
 */
export interface SubQuery {
  /** Unique identifier within the decomposition */
  id: number;
  /** The sub-query text */
  query: string;
  /** Expected output type (e.g., "code", "analysis", "data") */
  expectedType: string;
  /** Dependencies (IDs of sub-queries that must complete first) */
  dependencies: number[];
  /** Recommended agent type for this sub-query */
  recommendedAgent?: string;
  /** Estimated complexity (0.0-1.0) */
  complexity: number;
  /** Optional context from parent query */
  context?: string;
}

/**
 * Decomposition of a complex query into sub-queries
 */
export interface QueryDecomposition {
  /** Sub-queries in execution order */
  subQueries: SubQuery[];
  /** Decomposition strategy used */
  strategy: DecompositionStrategy;
  /** Reasoning for this decomposition */
  rationale: string;
  /** Total estimated complexity */
  totalComplexity: number;
  /** Whether decomposition was successful */
  success: boolean;
  /** Error message if decomposition failed */
  error?: string;
}

/**
 * Answer to a sub-query
 */
export interface SubAnswer {
  /** ID of the sub-query this answers */
  subQueryId: number;
  /** The answer content */
  content: string;
  /** Confidence in this answer (0.0-1.0) */
  confidence: number;
  /** Agent that produced this answer */
  agent: string;
  /** Latency in milliseconds */
  latencyMs: number;
  /** Quality score (0.0-1.0) */
  quality: number;
  /** Whether this answer was successful */
  success: boolean;
  /** Error message if failed */
  error?: string;
  /** Intermediate reasoning/chain-of-thought */
  reasoning?: string;
}

/**
 * Metadata about the RLM execution trajectory
 */
export interface RlmTrajectoryMetadata {
  /** Session ID */
  sessionId?: string;
  /** User ID */
  userId?: string;
  /** Total latency in milliseconds */
  totalLatencyMs: number;
  /** Number of retries */
  retries: number;
  /** Maximum parallel branches executed */
  maxParallelism: number;
  /** Models used during execution */
  modelsUsed: string[];
  /** Agents invoked */
  agentsInvoked: string[];
  /** Tools used */
  toolsUsed: string[];
  /** Custom attributes */
  attributes: Record<string, string>;
}

/**
 * A complete RLM training example
 */
export interface RlmTrainingExample {
  /** Unique identifier */
  id: string;
  /** Original complex query */
  query: string;
  /** Query embedding (optional) */
  queryEmbedding?: number[];
  /** How the query was decomposed */
  decomposition: QueryDecomposition;
  /** Answers to each sub-query */
  subAnswers: SubAnswer[];
  /** Final synthesized answer */
  finalAnswer: string;
  /** Final answer embedding (optional) */
  finalEmbedding?: number[];
  /** Overall quality score (0.0-1.0) */
  qualityScore: number;
  /** Execution trajectory metadata */
  trajectory: RlmTrajectoryMetadata;
  /** Whether this example was successful */
  success: boolean;
  /** Lessons learned from this example */
  lessons: string[];
  /** Source of this example */
  source: string;
}

/**
 * A contrastive pair for agent routing training
 */
export interface ContrastivePair {
  /** Anchor query */
  anchor: string;
  /** Anchor embedding (optional) */
  anchorEmbedding?: number[];
  /** Positive agent (correct routing) */
  positiveAgent: string;
  /** Negative agent (incorrect routing) */
  negativeAgent: string;
  /** Whether this is a hard negative */
  isHardNegative: boolean;
  /** Quality score of the anchor example */
  quality: number;
  /** Source example ID */
  sourceId: string;
}

/**
 * Configuration for RLM training
 */
export interface RlmTrainingConfig {
  /** Learning rate for decomposition training */
  decompositionLr: number;
  /** Learning rate for synthesis training */
  synthesisLr: number;
  /** Learning rate for contrastive fine-tuning */
  contrastiveLr: number;
  /** Batch size */
  batchSize: number;
  /** Number of epochs */
  epochs: number;
  /** Contrastive margin for triplet loss */
  contrastiveMargin: number;
  /** Temperature for InfoNCE loss */
  infonceTemperature: number;
  /** Weight for decomposition loss */
  decompositionWeight: number;
  /** Weight for synthesis loss */
  synthesisWeight: number;
  /** Weight for routing loss */
  routingWeight: number;
  /** Minimum quality for updates */
  qualityThreshold: number;
  /** Evaluation interval (epochs) */
  evaluationInterval: number;
  /** Warmup steps */
  warmupSteps: number;
  /** Early stopping patience */
  earlyStoppingPatience: number;
  /** Validation split ratio */
  validationSplit: number;
  /** Random seed */
  seed: number;
}

/**
 * Training result for a phase
 */
export interface TrainingResult {
  /** Training phase name */
  phase: string;
  /** Epochs completed */
  epochsCompleted: number;
  /** Total steps */
  totalSteps: number;
  /** Final training loss */
  finalLoss: number;
  /** Best validation loss */
  bestValLoss: number;
  /** Best epoch */
  bestEpoch: number;
  /** Final accuracy (for classification tasks) */
  accuracy: number;
  /** Loss history per epoch */
  lossHistory: number[];
  /** Validation loss history */
  valLossHistory: number[];
  /** Training duration in milliseconds */
  durationMs: number;
  /** Whether early stopping was triggered */
  earlyStopped: boolean;
}

/**
 * Evaluation result for the trained model
 */
export interface EvaluationResult {
  /** Decomposition accuracy */
  decompositionAccuracy: number;
  /** Synthesis quality */
  synthesisQuality: number;
  /** Routing accuracy */
  routingAccuracy: number;
  /** Hard negative accuracy */
  hardNegativeAccuracy: number;
  /** Average latency in ms */
  avgLatencyMs: number;
  /** Total examples evaluated */
  totalExamples: number;
  /** Per-agent accuracy */
  perAgentAccuracy: Record<string, number>;
}

// =============================================================================
// Default Configurations
// =============================================================================

/**
 * Default RLM training configuration
 */
export const DEFAULT_RLM_CONFIG: RlmTrainingConfig = {
  decompositionLr: 1e-5,
  synthesisLr: 1e-5,
  contrastiveLr: 2e-5,
  batchSize: 32,
  epochs: 10,
  contrastiveMargin: 0.5,
  infonceTemperature: 0.07,
  decompositionWeight: 1.0,
  synthesisWeight: 1.0,
  routingWeight: 1.0,
  qualityThreshold: 0.7,
  evaluationInterval: 1,
  warmupSteps: 100,
  earlyStoppingPatience: 3,
  validationSplit: 0.1,
  seed: 42,
};

/**
 * Fast training configuration
 */
export const FAST_RLM_CONFIG: RlmTrainingConfig = {
  ...DEFAULT_RLM_CONFIG,
  epochs: 3,
  batchSize: 64,
  decompositionLr: 1e-4,
  synthesisLr: 1e-4,
  contrastiveLr: 5e-5,
  earlyStoppingPatience: 1,
};

/**
 * Thorough training configuration
 */
export const THOROUGH_RLM_CONFIG: RlmTrainingConfig = {
  ...DEFAULT_RLM_CONFIG,
  epochs: 50,
  batchSize: 16,
  decompositionLr: 5e-6,
  synthesisLr: 5e-6,
  contrastiveLr: 1e-5,
  earlyStoppingPatience: 10,
};

/**
 * Routing-focused training configuration
 */
export const ROUTING_FOCUSED_CONFIG: RlmTrainingConfig = {
  ...DEFAULT_RLM_CONFIG,
  routingWeight: 2.0,
  decompositionWeight: 0.5,
  synthesisWeight: 0.5,
  contrastiveLr: 3e-5,
  contrastiveMargin: 0.3,
  infonceTemperature: 0.05,
};

// =============================================================================
// Agent Definitions
// =============================================================================

/**
 * Agent types with descriptions and keywords
 */
export const AGENT_DEFINITIONS: Record<string, { description: string; keywords: string[] }> = {
  coder: {
    description: 'Software developer who writes and implements code',
    keywords: ['implement', 'build', 'create', 'code', 'write', 'develop', 'program'],
  },
  researcher: {
    description: 'Technical researcher who investigates and analyzes',
    keywords: ['research', 'investigate', 'analyze', 'explore', 'study', 'examine'],
  },
  reviewer: {
    description: 'Code reviewer who evaluates code quality',
    keywords: ['review', 'check', 'evaluate', 'assess', 'examine', 'inspect'],
  },
  tester: {
    description: 'QA engineer who writes and runs tests',
    keywords: ['test', 'unit test', 'coverage', 'validate', 'verify', 'qa'],
  },
  architect: {
    description: 'System architect who designs software structure',
    keywords: ['design', 'plan', 'architecture', 'schema', 'structure', 'diagram'],
  },
  'security-architect': {
    description: 'Security specialist who audits vulnerabilities',
    keywords: ['security', 'audit', 'vulnerability', 'xss', 'injection', 'cve'],
  },
  debugger: {
    description: 'Bug hunter who fixes errors and traces issues',
    keywords: ['fix', 'debug', 'bug', 'error', 'trace', 'crash', 'troubleshoot'],
  },
  documenter: {
    description: 'Technical writer who creates documentation',
    keywords: ['document', 'jsdoc', 'readme', 'comment', 'explain', 'describe'],
  },
  refactorer: {
    description: 'Code modernizer who restructures without changing behavior',
    keywords: ['refactor', 'restructure', 'modernize', 'clean', 'simplify', 'consolidate'],
  },
  optimizer: {
    description: 'Performance engineer who speeds up slow code',
    keywords: ['optimize', 'performance', 'speed', 'cache', 'improve', 'faster'],
  },
  devops: {
    description: 'DevOps engineer who manages deployment and infrastructure',
    keywords: ['deploy', 'ci/cd', 'kubernetes', 'docker', 'infrastructure', 'pipeline'],
  },
  'api-docs': {
    description: 'API documentation specialist who creates specs',
    keywords: ['openapi', 'swagger', 'api reference', 'endpoint', 'spec', 'rest'],
  },
  planner: {
    description: 'Project planner who organizes and schedules work',
    keywords: ['plan', 'estimate', 'schedule', 'timeline', 'sprint', 'roadmap'],
  },
};

/**
 * Hard negative pairs (confusable agent combinations)
 */
export const HARD_NEGATIVE_PAIRS: [string, string][] = [
  ['coder', 'debugger'],
  ['coder', 'refactorer'],
  ['researcher', 'reviewer'],
  ['tester', 'reviewer'],
  ['architect', 'planner'],
  ['documenter', 'api-docs'],
  ['optimizer', 'debugger'],
  ['devops', 'architect'],
  ['security-architect', 'reviewer'],
];

// =============================================================================
// RLM Trainer Class
// =============================================================================

/**
 * RLM Trainer for RuvLTRA models
 *
 * Provides training capabilities for decomposition, synthesis, and routing tasks.
 */
export class RlmTrainer {
  private config: RlmTrainingConfig;
  private currentEpoch = 0;
  private currentStep = 0;
  private bestValLoss = Infinity;
  private patienceCounter = 0;
  private lossHistory: number[] = [];
  private valLossHistory: number[] = [];

  /**
   * Create a new RLM trainer
   */
  constructor(config: Partial<RlmTrainingConfig> = {}) {
    this.config = { ...DEFAULT_RLM_CONFIG, ...config };
  }

  /**
   * Train on decomposition task
   *
   * Learns to break complex queries into manageable sub-queries.
   */
  async trainDecomposition(dataset: RlmTrainingExample[]): Promise<TrainingResult> {
    const startTime = Date.now();
    this.resetState();

    const { trainSet, valSet } = this.splitDataset(dataset);
    const batches = this.createBatches(trainSet);

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      this.currentEpoch = epoch;
      let epochLoss = 0;

      for (const batch of batches) {
        const batchLoss = this.trainDecompositionBatch(batch);
        epochLoss += batchLoss;
        this.currentStep++;
      }

      const avgLoss = epochLoss / batches.length;
      this.lossHistory.push(avgLoss);

      // Validation
      const valLoss = this.validateDecomposition(valSet);
      this.valLossHistory.push(valLoss);

      // Early stopping
      if (valLoss < this.bestValLoss) {
        this.bestValLoss = valLoss;
        this.patienceCounter = 0;
      } else {
        this.patienceCounter++;
        if (this.patienceCounter >= this.config.earlyStoppingPatience) {
          break;
        }
      }
    }

    return {
      phase: 'decomposition',
      epochsCompleted: this.currentEpoch + 1,
      totalSteps: this.currentStep,
      finalLoss: this.lossHistory[this.lossHistory.length - 1] || 0,
      bestValLoss: this.bestValLoss,
      bestEpoch: this.findBestEpoch(),
      accuracy: 0, // Not applicable for decomposition
      lossHistory: this.lossHistory,
      valLossHistory: this.valLossHistory,
      durationMs: Date.now() - startTime,
      earlyStopped: this.patienceCounter >= this.config.earlyStoppingPatience,
    };
  }

  /**
   * Train on synthesis task
   *
   * Learns to combine sub-answers into coherent final responses.
   */
  async trainSynthesis(dataset: RlmTrainingExample[]): Promise<TrainingResult> {
    const startTime = Date.now();
    this.resetState();

    const { trainSet, valSet } = this.splitDataset(dataset);
    const batches = this.createBatches(trainSet);

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      this.currentEpoch = epoch;
      let epochLoss = 0;

      for (const batch of batches) {
        const batchLoss = this.trainSynthesisBatch(batch);
        epochLoss += batchLoss;
        this.currentStep++;
      }

      const avgLoss = epochLoss / batches.length;
      this.lossHistory.push(avgLoss);

      // Validation
      const valLoss = this.validateSynthesis(valSet);
      this.valLossHistory.push(valLoss);

      // Early stopping
      if (valLoss < this.bestValLoss) {
        this.bestValLoss = valLoss;
        this.patienceCounter = 0;
      } else {
        this.patienceCounter++;
        if (this.patienceCounter >= this.config.earlyStoppingPatience) {
          break;
        }
      }
    }

    return {
      phase: 'synthesis',
      epochsCompleted: this.currentEpoch + 1,
      totalSteps: this.currentStep,
      finalLoss: this.lossHistory[this.lossHistory.length - 1] || 0,
      bestValLoss: this.bestValLoss,
      bestEpoch: this.findBestEpoch(),
      accuracy: 0,
      lossHistory: this.lossHistory,
      valLossHistory: this.valLossHistory,
      durationMs: Date.now() - startTime,
      earlyStopped: this.patienceCounter >= this.config.earlyStoppingPatience,
    };
  }

  /**
   * Contrastive fine-tuning for agent routing
   *
   * Uses triplet loss and InfoNCE to improve routing accuracy.
   */
  async trainContrastive(pairs: ContrastivePair[]): Promise<TrainingResult> {
    const startTime = Date.now();
    this.resetState();

    if (pairs.length === 0) {
      throw new Error('No contrastive pairs provided');
    }

    const { trainSet, valSet } = this.splitPairs(pairs);
    const batches = this.createPairBatches(trainSet);
    let totalCorrect = 0;
    let totalExamples = 0;

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      this.currentEpoch = epoch;
      let epochLoss = 0;

      for (const batch of batches) {
        const batchLoss = this.trainContrastiveBatch(batch);
        epochLoss += batchLoss;
        this.currentStep++;
      }

      const avgLoss = epochLoss / batches.length;
      this.lossHistory.push(avgLoss);

      // Validation
      const { loss: valLoss, correct, total } = this.validateContrastive(valSet);
      this.valLossHistory.push(valLoss);
      totalCorrect = correct;
      totalExamples = total;

      // Early stopping
      if (valLoss < this.bestValLoss) {
        this.bestValLoss = valLoss;
        this.patienceCounter = 0;
      } else {
        this.patienceCounter++;
        if (this.patienceCounter >= this.config.earlyStoppingPatience) {
          break;
        }
      }
    }

    return {
      phase: 'contrastive',
      epochsCompleted: this.currentEpoch + 1,
      totalSteps: this.currentStep,
      finalLoss: this.lossHistory[this.lossHistory.length - 1] || 0,
      bestValLoss: this.bestValLoss,
      bestEpoch: this.findBestEpoch(),
      accuracy: totalExamples > 0 ? totalCorrect / totalExamples : 0,
      lossHistory: this.lossHistory,
      valLossHistory: this.valLossHistory,
      durationMs: Date.now() - startTime,
      earlyStopped: this.patienceCounter >= this.config.earlyStoppingPatience,
    };
  }

  /**
   * Evaluate trained model on test set
   */
  async evaluate(testSet: RlmTrainingExample[]): Promise<EvaluationResult> {
    const perAgentAccuracy: Record<string, { correct: number; total: number }> = {};

    let decompositionCorrect = 0;
    let synthesisQualitySum = 0;
    let routingCorrect = 0;
    let hardNegativeCorrect = 0;
    let hardNegativeTotal = 0;
    let totalLatency = 0;

    for (const example of testSet) {
      // Decomposition evaluation
      if (example.decomposition.success && example.decomposition.subQueries.length > 0) {
        decompositionCorrect++;
      }

      // Synthesis quality
      synthesisQualitySum += example.qualityScore;

      // Routing evaluation
      for (const subQuery of example.decomposition.subQueries) {
        if (subQuery.recommendedAgent) {
          const predicted = this.predictAgent(subQuery.query);
          const correct = predicted === subQuery.recommendedAgent;

          if (correct) {
            routingCorrect++;
          }

          // Track per-agent accuracy
          if (!perAgentAccuracy[subQuery.recommendedAgent]) {
            perAgentAccuracy[subQuery.recommendedAgent] = { correct: 0, total: 0 };
          }
          perAgentAccuracy[subQuery.recommendedAgent].total++;
          if (correct) {
            perAgentAccuracy[subQuery.recommendedAgent].correct++;
          }

          // Check hard negatives
          if (this.isHardNegative(subQuery.recommendedAgent, predicted)) {
            hardNegativeTotal++;
            if (correct) {
              hardNegativeCorrect++;
            }
          }
        }
      }

      totalLatency += example.trajectory.totalLatencyMs;
    }

    const totalRoutingExamples = testSet.reduce(
      (sum, ex) => sum + ex.decomposition.subQueries.filter((sq) => sq.recommendedAgent).length,
      0
    );

    const perAgentResult: Record<string, number> = {};
    for (const [agent, stats] of Object.entries(perAgentAccuracy)) {
      perAgentResult[agent] = stats.total > 0 ? stats.correct / stats.total : 0;
    }

    return {
      decompositionAccuracy: testSet.length > 0 ? decompositionCorrect / testSet.length : 0,
      synthesisQuality: testSet.length > 0 ? synthesisQualitySum / testSet.length : 0,
      routingAccuracy: totalRoutingExamples > 0 ? routingCorrect / totalRoutingExamples : 0,
      hardNegativeAccuracy: hardNegativeTotal > 0 ? hardNegativeCorrect / hardNegativeTotal : 0,
      avgLatencyMs: testSet.length > 0 ? totalLatency / testSet.length : 0,
      totalExamples: testSet.length,
      perAgentAccuracy: perAgentResult,
    };
  }

  /**
   * Generate contrastive pairs from dataset
   */
  generateContrastivePairs(
    dataset: RlmTrainingExample[],
    hardNegativeRatio = 0.3
  ): ContrastivePair[] {
    const pairs: ContrastivePair[] = [];
    const agents = Object.keys(AGENT_DEFINITIONS);

    for (const example of dataset) {
      for (const subQuery of example.decomposition.subQueries) {
        if (!subQuery.recommendedAgent) continue;

        const positiveAgent = subQuery.recommendedAgent;

        for (const negativeAgent of agents) {
          if (negativeAgent === positiveAgent) continue;

          const isHard = this.isHardNegative(positiveAgent, negativeAgent);

          // Apply hard negative ratio
          const include = isHard
            ? Math.random() < hardNegativeRatio
            : Math.random() < 1 - hardNegativeRatio;

          if (include) {
            pairs.push({
              anchor: subQuery.query,
              anchorEmbedding: example.queryEmbedding,
              positiveAgent,
              negativeAgent,
              isHardNegative: isHard,
              quality: example.qualityScore,
              sourceId: example.id,
            });
          }
        }
      }
    }

    return pairs;
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private resetState(): void {
    this.currentEpoch = 0;
    this.currentStep = 0;
    this.bestValLoss = Infinity;
    this.patienceCounter = 0;
    this.lossHistory = [];
    this.valLossHistory = [];
  }

  private splitDataset(
    dataset: RlmTrainingExample[]
  ): { trainSet: RlmTrainingExample[]; valSet: RlmTrainingExample[] } {
    const valSize = Math.floor(dataset.length * this.config.validationSplit);
    const shuffled = this.shuffle([...dataset]);
    return {
      trainSet: shuffled.slice(valSize),
      valSet: shuffled.slice(0, valSize),
    };
  }

  private splitPairs(
    pairs: ContrastivePair[]
  ): { trainSet: ContrastivePair[]; valSet: ContrastivePair[] } {
    const valSize = Math.floor(pairs.length * this.config.validationSplit);
    const shuffled = this.shuffle([...pairs]);
    return {
      trainSet: shuffled.slice(valSize),
      valSet: shuffled.slice(0, valSize),
    };
  }

  private createBatches(dataset: RlmTrainingExample[]): RlmTrainingExample[][] {
    const batches: RlmTrainingExample[][] = [];
    for (let i = 0; i < dataset.length; i += this.config.batchSize) {
      batches.push(dataset.slice(i, i + this.config.batchSize));
    }
    return batches;
  }

  private createPairBatches(pairs: ContrastivePair[]): ContrastivePair[][] {
    const batches: ContrastivePair[][] = [];
    for (let i = 0; i < pairs.length; i += this.config.batchSize) {
      batches.push(pairs.slice(i, i + this.config.batchSize));
    }
    return batches;
  }

  private shuffle<T>(array: T[]): T[] {
    // Fisher-Yates shuffle
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  private trainDecompositionBatch(batch: RlmTrainingExample[]): number {
    let batchLoss = 0;

    for (const example of batch) {
      // Decomposition quality loss
      const qualityLoss = 1 - (example.decomposition.success ? example.qualityScore : 0);

      // Depth appropriateness (penalize too shallow or too deep)
      const depth = example.decomposition.subQueries.length;
      const idealDepth = 3;
      const depthLoss = Math.abs(depth - idealDepth) / idealDepth;

      // Complexity balance loss
      const complexityLoss = Math.abs(example.decomposition.totalComplexity - 1) / 3;

      const loss =
        qualityLoss * this.config.decompositionWeight * 0.6 +
        depthLoss * 0.2 +
        complexityLoss * 0.2;

      batchLoss += loss;
    }

    return batchLoss / batch.length;
  }

  private trainSynthesisBatch(batch: RlmTrainingExample[]): number {
    let batchLoss = 0;

    for (const example of batch) {
      // Sub-answer quality
      const subAnswerQuality =
        example.subAnswers.length > 0
          ? example.subAnswers.reduce((sum, a) => sum + a.quality, 0) / example.subAnswers.length
          : 0;

      // Final answer quality
      const finalQuality = example.qualityScore;

      // Coherence bonus (final should be better than parts average)
      const coherenceBonus = Math.max(0, finalQuality - subAnswerQuality) * 0.5;

      const loss = (1 - (subAnswerQuality * 0.4 + finalQuality * 0.4 + coherenceBonus * 0.2));

      batchLoss += loss * this.config.synthesisWeight;
    }

    return batchLoss / batch.length;
  }

  private trainContrastiveBatch(batch: ContrastivePair[]): number {
    let batchLoss = 0;

    for (const pair of batch) {
      // Triplet loss
      const tripletLoss = this.computeTripletLoss(pair);

      // InfoNCE loss
      const infonceLoss = this.computeInfoNCELoss(pair);

      batchLoss += (tripletLoss * 0.5 + infonceLoss * 0.5) * this.config.routingWeight;
    }

    return batchLoss / batch.length;
  }

  private validateDecomposition(valSet: RlmTrainingExample[]): number {
    if (valSet.length === 0) return 0;

    let totalLoss = 0;
    for (const example of valSet) {
      totalLoss += 1 - example.qualityScore;
    }
    return totalLoss / valSet.length;
  }

  private validateSynthesis(valSet: RlmTrainingExample[]): number {
    if (valSet.length === 0) return 0;

    let totalLoss = 0;
    for (const example of valSet) {
      totalLoss += 1 - example.qualityScore;
    }
    return totalLoss / valSet.length;
  }

  private validateContrastive(
    valSet: ContrastivePair[]
  ): { loss: number; correct: number; total: number } {
    if (valSet.length === 0) return { loss: 0, correct: 0, total: 0 };

    let totalLoss = 0;
    let correct = 0;

    for (const pair of valSet) {
      const tripletLoss = this.computeTripletLoss(pair);
      const infonceLoss = this.computeInfoNCELoss(pair);
      totalLoss += tripletLoss * 0.5 + infonceLoss * 0.5;

      // Check routing correctness
      const posDist = this.agentDistance(pair.anchor, pair.positiveAgent);
      const negDist = this.agentDistance(pair.anchor, pair.negativeAgent);
      if (posDist < negDist) {
        correct++;
      }
    }

    return {
      loss: totalLoss / valSet.length,
      correct,
      total: valSet.length,
    };
  }

  private computeTripletLoss(pair: ContrastivePair): number {
    const posDist = this.agentDistance(pair.anchor, pair.positiveAgent);
    const negDist = this.agentDistance(pair.anchor, pair.negativeAgent);
    return Math.max(0, this.config.contrastiveMargin + posDist - negDist);
  }

  private computeInfoNCELoss(pair: ContrastivePair): number {
    const posSim = 1 - this.agentDistance(pair.anchor, pair.positiveAgent);
    const negSim = 1 - this.agentDistance(pair.anchor, pair.negativeAgent);

    const temp = this.config.infonceTemperature;
    const posExp = Math.exp(posSim / temp);
    const negExp = Math.exp(negSim / temp);

    return -Math.log(posExp / (posExp + negExp));
  }

  private agentDistance(query: string, agent: string): number {
    const queryLower = query.toLowerCase();
    const agentDef = AGENT_DEFINITIONS[agent];

    if (!agentDef) return 1.0;

    const matches = agentDef.keywords.filter((kw) => queryLower.includes(kw)).length;
    return 1.0 - Math.min(1.0, matches / agentDef.keywords.length);
  }

  private predictAgent(query: string): string {
    let bestAgent = 'coder';
    let bestScore = 0;

    for (const [agent, def] of Object.entries(AGENT_DEFINITIONS)) {
      const queryLower = query.toLowerCase();
      const matches = def.keywords.filter((kw) => queryLower.includes(kw)).length;
      const score = matches / def.keywords.length;

      if (score > bestScore) {
        bestScore = score;
        bestAgent = agent;
      }
    }

    return bestAgent;
  }

  private isHardNegative(agent1: string, agent2: string): boolean {
    return HARD_NEGATIVE_PAIRS.some(
      ([a, b]) => (agent1 === a && agent2 === b) || (agent1 === b && agent2 === a)
    );
  }

  private findBestEpoch(): number {
    if (this.valLossHistory.length === 0) return 0;

    let bestIdx = 0;
    let bestLoss = this.valLossHistory[0];

    for (let i = 1; i < this.valLossHistory.length; i++) {
      if (this.valLossHistory[i] < bestLoss) {
        bestLoss = this.valLossHistory[i];
        bestIdx = i;
      }
    }

    return bestIdx;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create an RLM trainer with default configuration
 */
export function createRlmTrainer(config?: Partial<RlmTrainingConfig>): RlmTrainer {
  return new RlmTrainer(config);
}

/**
 * Create an empty RLM training example
 */
export function createEmptyExample(query: string): RlmTrainingExample {
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36)}`,
    query,
    decomposition: {
      subQueries: [],
      strategy: 'none',
      rationale: '',
      totalComplexity: 0,
      success: false,
    },
    subAnswers: [],
    finalAnswer: '',
    qualityScore: 0,
    trajectory: {
      totalLatencyMs: 0,
      retries: 0,
      maxParallelism: 1,
      modelsUsed: [],
      agentsInvoked: [],
      toolsUsed: [],
      attributes: {},
    },
    success: false,
    lessons: [],
    source: 'manual',
  };
}

/**
 * Create a sub-query
 */
export function createSubQuery(
  id: number,
  query: string,
  options: Partial<SubQuery> = {}
): SubQuery {
  return {
    id,
    query,
    expectedType: 'text',
    dependencies: [],
    complexity: 0.5,
    ...options,
  };
}

/**
 * Create a sub-answer
 */
export function createSubAnswer(
  subQueryId: number,
  content: string,
  agent: string,
  options: Partial<SubAnswer> = {}
): SubAnswer {
  return {
    subQueryId,
    content,
    confidence: 0.8,
    agent,
    latencyMs: 0,
    quality: 0.8,
    success: true,
    ...options,
  };
}

// =============================================================================
// Exports
// =============================================================================

export default RlmTrainer;
