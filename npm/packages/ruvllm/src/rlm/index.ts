/**
 * RLM - Retrieval Language Model
 *
 * A recursive retrieval-augmented generation system that combines
 * memory search with intelligent query decomposition and synthesis.
 *
 * @example Basic Usage
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController({
 *   maxDepth: 3,
 *   enableCache: true,
 *   retrievalTopK: 10,
 * });
 *
 * // Add knowledge
 * await rlm.addMemory('TypeScript adds static typing to JavaScript.');
 * await rlm.addMemory('React is a library for building user interfaces.');
 *
 * // Query with retrieval
 * const answer = await rlm.query('Compare TypeScript and JavaScript');
 * console.log(answer.text);
 * console.log('Confidence:', answer.confidence);
 * console.log('Sources:', answer.sources.length);
 * ```
 *
 * @example Streaming
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController();
 *
 * for await (const event of rlm.queryStream('Explain machine learning')) {
 *   if (event.type === 'token') {
 *     process.stdout.write(event.text);
 *   } else {
 *     console.log('\n\nQuality:', event.answer.qualityScore);
 *   }
 * }
 * ```
 *
 * @example With Reflection
 * ```typescript
 * import { RlmController } from '@ruvector/ruvllm';
 *
 * const rlm = new RlmController({
 *   enableReflection: true,
 *   maxReflectionIterations: 2,
 *   minQualityScore: 0.8,
 * });
 *
 * // Answers will be iteratively refined until quality >= 0.8
 * const answer = await rlm.query('Complex technical question...');
 * ```
 *
 * @module rlm
 */

// Export all types
export * from './types';

// Export the controller
export { RlmController } from './controller';

// Export training module
export {
  // Types
  type DecompositionStrategy,
  type SubQuery,
  type QueryDecomposition,
  type SubAnswer,
  type RlmTrajectoryMetadata,
  type RlmTrainingExample,
  type ContrastivePair,
  type RlmTrainingConfig,
  type TrainingResult as RlmTrainingResult,
  type EvaluationResult as RlmEvaluationResult,

  // Constants
  DEFAULT_RLM_CONFIG,
  FAST_RLM_CONFIG,
  THOROUGH_RLM_CONFIG,
  ROUTING_FOCUSED_CONFIG,
  AGENT_DEFINITIONS,
  HARD_NEGATIVE_PAIRS,

  // Classes
  RlmTrainer,

  // Factory functions
  createRlmTrainer,
  createEmptyExample,
  createSubQuery,
  createSubAnswer,
} from './training';
