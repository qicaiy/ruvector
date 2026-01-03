/**
 * @ruvector/edge-net Model Optimizer Tests
 *
 * Comprehensive tests for model quantization and optimization
 */

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import {
    ModelOptimizer,
    QuantizationEngine,
    PruningEngine,
    OnnxOptimizer,
    DistillationEngine,
    BenchmarkEngine,
    TARGET_MODELS,
    QUANTIZATION_CONFIGS,
    PRUNING_STRATEGIES,
} from '../models/model-optimizer.js';
import {
    ComprehensiveBenchmark,
    AccuracyMeter,
    LatencyProfiler,
    MemoryProfiler,
    BENCHMARK_PROFILES,
} from '../models/benchmark.js';

// ============================================
// MODEL OPTIMIZER TESTS
// ============================================

describe('ModelOptimizer', () => {
    let optimizer;

    before(() => {
        optimizer = new ModelOptimizer();
    });

    describe('Configuration', () => {
        it('should have all target models defined', () => {
            const models = optimizer.getTargetModels();
            assert.ok(models['phi-1.5'], 'phi-1.5 should be defined');
            assert.ok(models['qwen-0.5b'], 'qwen-0.5b should be defined');
            assert.ok(models['minilm-l6'], 'minilm-l6 should be defined');
            assert.ok(models['e5-small'], 'e5-small should be defined');
            assert.ok(models['bge-small'], 'bge-small should be defined');
        });

        it('should have correct target sizes', () => {
            const models = optimizer.getTargetModels();
            assert.strictEqual(models['phi-1.5'].originalSize, 280);
            assert.strictEqual(models['phi-1.5'].targetSize, 70);
            assert.strictEqual(models['qwen-0.5b'].originalSize, 430);
            assert.strictEqual(models['qwen-0.5b'].targetSize, 100);
            assert.strictEqual(models['minilm-l6'].originalSize, 22);
            assert.strictEqual(models['minilm-l6'].targetSize, 8);
        });

        it('should get individual model config', () => {
            const config = optimizer.getModelConfig('phi-1.5');
            assert.ok(config, 'Should return config');
            assert.strictEqual(config.type, 'generation');
            assert.ok(config.capabilities.includes('code'), 'Should include code capability');
        });

        it('should return null for unknown model', () => {
            const config = optimizer.getModelConfig('unknown-model');
            assert.strictEqual(config, null);
        });
    });

    describe('Quantization', () => {
        it('should quantize model to INT8', async () => {
            const result = await optimizer.quantize('minilm-l6', 'int8');

            assert.strictEqual(result.model, 'minilm-l6');
            assert.strictEqual(result.method, 'int8');
            assert.strictEqual(result.status, 'completed');
            assert.strictEqual(result.compressionRatio, 4);
            assert.ok(result.quantizedSizeMB < result.originalSizeMB);
        });

        it('should quantize model to INT4', async () => {
            const result = await optimizer.quantize('phi-1.5', 'int4');

            assert.strictEqual(result.method, 'int4');
            assert.strictEqual(result.compressionRatio, 8);
            assert.ok(result.quantizedSizeMB <= 35, 'INT4 should compress to ~35MB for phi-1.5');
        });

        it('should quantize model to FP16', async () => {
            const result = await optimizer.quantize('e5-small', 'fp16');

            assert.strictEqual(result.method, 'fp16');
            assert.strictEqual(result.compressionRatio, 2);
        });

        it('should throw for unknown model', async () => {
            await assert.rejects(
                () => optimizer.quantize('unknown-model', 'int8'),
                /Unknown model/
            );
        });

        it('should throw for unknown quantization method', async () => {
            await assert.rejects(
                () => optimizer.quantize('minilm-l6', 'int2'),
                /Unknown quantization method/
            );
        });

        it('should emit events during quantization', async () => {
            let startEmitted = false;
            let completeEmitted = false;

            optimizer.on('quantize:start', () => { startEmitted = true; });
            optimizer.on('quantize:complete', () => { completeEmitted = true; });

            await optimizer.quantize('bge-small', 'int8');

            assert.ok(startEmitted, 'Should emit start event');
            assert.ok(completeEmitted, 'Should emit complete event');

            optimizer.removeAllListeners();
        });
    });

    describe('Pruning', () => {
        it('should prune model with default settings', async () => {
            const result = await optimizer.prune('minilm-l6');

            assert.strictEqual(result.model, 'minilm-l6');
            assert.strictEqual(result.status, 'completed');
            assert.ok(result.achievedSparsity > 0, 'Should have achieved sparsity');
        });

        it('should prune model with custom sparsity', async () => {
            const result = await optimizer.prune('phi-1.5', { sparsity: 0.7 });

            assert.strictEqual(result.targetSparsity, 0.7);
            assert.ok(result.layerResults.length > 0, 'Should have layer results');
        });

        it('should prune attention heads when requested', async () => {
            const result = await optimizer.prune('minilm-l6', {
                sparsity: 0.5,
                pruneHeads: true,
                headPruneFraction: 0.25,
            });

            assert.ok(result.headPruning, 'Should have head pruning results');
            assert.ok(result.headPruning.prunedHeads > 0, 'Should prune some heads');
        });

        it('should support different sparsity schedules', async () => {
            const uniformResult = await optimizer.prune('e5-small', {
                sparsity: 0.5,
                sparsitySchedule: 'uniform',
            });

            const cubicResult = await optimizer.prune('e5-small', {
                sparsity: 0.5,
                sparsitySchedule: 'cubic',
            });

            // Cubic should have varying sparsity across layers
            const uniformSparsities = uniformResult.layerResults.map(l => l.sparsity);
            const cubicSparsities = cubicResult.layerResults.map(l => l.sparsity);

            // All uniform should be equal
            const uniformUnique = new Set(uniformSparsities);
            assert.strictEqual(uniformUnique.size, 1, 'Uniform should have equal sparsity');

            // Cubic should have varying values
            const cubicUnique = new Set(cubicSparsities);
            assert.ok(cubicUnique.size > 1, 'Cubic should have varying sparsity');
        });
    });

    describe('Knowledge Distillation', () => {
        it('should setup distillation configuration', () => {
            const config = optimizer.setupDistillation('phi-1.5', 'minilm-l6', {
                temperature: 6.0,
                alpha: 0.7,
            });

            assert.strictEqual(config.teacher, 'phi-1.5');
            assert.strictEqual(config.student, 'minilm-l6');
            assert.strictEqual(config.temperature, 6.0);
            assert.strictEqual(config.alpha, 0.7);
            assert.ok(config.trainingConfig, 'Should have training config');
        });

        it('should throw for invalid models', () => {
            assert.throws(
                () => optimizer.setupDistillation('unknown', 'minilm-l6'),
                /must be valid/
            );
        });
    });

    describe('ONNX Optimization', () => {
        it('should apply ONNX optimization passes', async () => {
            const result = await optimizer.optimizeOnnx('minilm-l6');

            assert.strictEqual(result.model, 'minilm-l6');
            assert.ok(result.passes.length > 0, 'Should apply passes');
            assert.ok(result.optimizedGraph, 'Should have optimized graph');
        });
    });

    describe('Export', () => {
        it('should export optimized model', async () => {
            // First quantize
            await optimizer.quantize('minilm-l6', 'int8');

            // Then export
            const result = await optimizer.export('minilm-l6', 'onnx');

            assert.ok(result.path, 'Should have export path');
            assert.ok(result.optimization, 'Should reference optimization');
            assert.strictEqual(result.format, 'onnx');
        });

        it('should indicate if target is met', async () => {
            await optimizer.quantize('minilm-l6', 'int8');
            const result = await optimizer.export('minilm-l6', 'onnx');

            // 22MB / 4 = 5.5MB, target is 8MB, should meet target
            assert.strictEqual(result.meetsTarget, true);
        });
    });

    describe('Full Pipeline', () => {
        it('should run full optimization pipeline', async () => {
            const result = await optimizer.optimizePipeline('minilm-l6', {
                quantizeMethod: 'int8',
                prune: true,
                sparsity: 0.3,
            });

            assert.strictEqual(result.model, 'minilm-l6');
            assert.ok(result.steps.length >= 3, 'Should have multiple steps');
            assert.ok(result.meetsTarget, 'MiniLM-L6 should meet target with INT8');
        });

        it('should run pipeline without pruning', async () => {
            const result = await optimizer.optimizePipeline('e5-small', {
                quantizeMethod: 'int4',
                prune: false,
            });

            const hasPruning = result.steps.some(s => s.step === 'prune');
            assert.strictEqual(hasPruning, false, 'Should not have pruning step');
        });
    });

    describe('Statistics', () => {
        it('should track statistics', async () => {
            const freshOptimizer = new ModelOptimizer();

            await freshOptimizer.quantize('minilm-l6', 'int8');
            await freshOptimizer.prune('minilm-l6');
            await freshOptimizer.export('minilm-l6', 'onnx');

            const stats = freshOptimizer.getStats();

            assert.strictEqual(stats.quantizations, 1);
            assert.strictEqual(stats.prunings, 1);
            assert.strictEqual(stats.exports, 1);
        });

        it('should list models with optimization status', async () => {
            const freshOptimizer = new ModelOptimizer();
            await freshOptimizer.quantize('phi-1.5', 'int4');

            const models = freshOptimizer.listModels();

            const phi = models.find(m => m.key === 'phi-1.5');
            assert.ok(phi.optimized, 'phi-1.5 should be marked as optimized');

            const qwen = models.find(m => m.key === 'qwen-0.5b');
            assert.strictEqual(qwen.optimized, false, 'qwen-0.5b should not be optimized');
        });
    });
});

// ============================================
// QUANTIZATION ENGINE TESTS
// ============================================

describe('QuantizationEngine', () => {
    let engine;

    before(() => {
        engine = new QuantizationEngine();
    });

    describe('Tensor Quantization', () => {
        it('should quantize tensor to INT8', () => {
            const tensor = [0.5, -0.3, 0.8, -0.9, 0.1];
            const result = engine.quantizeTensor(tensor, { bits: 8, symmetric: false });

            // Non-symmetric uses Uint8Array (0-255 range), symmetric uses Int8Array
            assert.ok(result.data instanceof Uint8Array, 'Non-symmetric should return Uint8Array');
            assert.strictEqual(result.originalLength, tensor.length);
            assert.ok(result.params.scale > 0, 'Should have positive scale');
        });

        it('should compute correct quantization parameters', () => {
            const tensor = [-1, 0, 1];
            const params = engine.computeQuantParams(tensor, { bits: 8, symmetric: true });

            assert.strictEqual(params.min, -1);
            assert.strictEqual(params.max, 1);
            assert.strictEqual(params.zeroPoint, 0, 'Symmetric should have zero point = 0');
        });

        it('should dequantize tensor correctly', () => {
            const original = [0.5, -0.3, 0.8];
            const quantized = engine.quantizeTensor(original, { bits: 8, symmetric: false });
            const dequantized = engine.dequantizeTensor(quantized, quantized.params);

            // Check reconstruction error is small
            for (let i = 0; i < original.length; i++) {
                const error = Math.abs(original[i] - dequantized[i]);
                assert.ok(error < 0.1, `Reconstruction error should be small: ${error}`);
            }
        });
    });

    describe('INT4 Block Quantization', () => {
        it('should quantize to INT4 blocks', () => {
            const tensor = new Float32Array(64).map(() => Math.random() - 0.5);
            const result = engine.quantizeInt4Block(tensor, 32);

            assert.ok(result.data instanceof Uint8Array, 'Should return Uint8Array');
            assert.ok(result.scales instanceof Float32Array, 'Should have scales');
            assert.strictEqual(result.scales.length, 2, 'Should have 2 blocks for 64 elements');
            assert.ok(result.compressionRatio > 1, 'Should have compression');
        });
    });
});

// ============================================
// PRUNING ENGINE TESTS
// ============================================

describe('PruningEngine', () => {
    let engine;

    before(() => {
        engine = new PruningEngine();
    });

    describe('Magnitude Pruning', () => {
        it('should prune smallest magnitude weights', () => {
            const tensor = [0.9, 0.1, -0.8, -0.05, 0.7];
            const result = engine.magnitudePrune(tensor, 0.4); // Prune 40%

            assert.strictEqual(result.prunedCount, 2); // 40% of 5 = 2
            assert.ok(result.mask[1] === 0 || result.mask[3] === 0, 'Small values should be pruned');
        });

        it('should preserve specified sparsity', () => {
            const tensor = new Float32Array(1000).map(() => Math.random());
            const sparsity = 0.5;
            const result = engine.magnitudePrune(tensor, sparsity);

            const actualSparsity = result.prunedCount / tensor.length;
            assert.ok(Math.abs(actualSparsity - sparsity) < 0.01, 'Should achieve target sparsity');
        });
    });

    describe('Structured Pruning', () => {
        it('should prune attention heads', () => {
            const numHeads = 12;
            const headDim = 64;
            const weights = new Float32Array(numHeads * headDim).map(() => Math.random());

            const result = engine.structuredPruneHeads(weights, numHeads, 0.25);

            assert.strictEqual(result.prunedHeads, 3, 'Should prune 25% of heads (3)');
            assert.strictEqual(result.remainingHeads.length, 9);
            assert.strictEqual(result.data.length, 9 * headDim);
        });
    });

    describe('Layer-wise Sparsity', () => {
        it('should compute uniform sparsity', () => {
            const sparsity = engine.computeLayerSparsity(5, 12, 0.5, 'uniform');
            assert.strictEqual(sparsity, 0.5);
        });

        it('should compute cubic sparsity', () => {
            const early = engine.computeLayerSparsity(2, 12, 0.5, 'cubic');
            const late = engine.computeLayerSparsity(10, 12, 0.5, 'cubic');

            assert.ok(late > early, 'Later layers should have higher sparsity in cubic');
        });

        it('should preserve first and last layers', () => {
            const first = engine.computeLayerSparsity(0, 12, 0.5, 'first-last-preserved');
            const last = engine.computeLayerSparsity(11, 12, 0.5, 'first-last-preserved');
            const middle = engine.computeLayerSparsity(6, 12, 0.5, 'first-last-preserved');

            assert.ok(first < middle, 'First layer should have lower sparsity');
            assert.ok(last < middle, 'Last layer should have lower sparsity');
        });
    });
});

// ============================================
// ONNX OPTIMIZER TESTS
// ============================================

describe('OnnxOptimizer', () => {
    let optimizer;

    before(() => {
        optimizer = new OnnxOptimizer();
    });

    it('should list available passes', () => {
        const passes = optimizer.getAvailablePasses();

        assert.ok(passes.includes('constant-folding'));
        assert.ok(passes.includes('fuse-attention'));
        assert.ok(passes.includes('memory-optimization'));
    });

    it('should apply all optimization passes', () => {
        const graph = {
            nodes: new Array(50).fill(null),
            attentionHeads: 12,
        };

        const result = optimizer.applyAllPasses(graph);

        assert.ok(result.passes.length > 0, 'Should apply passes');
        assert.ok(result.graph.constantsFolded, 'Should fold constants');
        assert.ok(result.graph.attentionFused, 'Should fuse attention');
    });
});

// ============================================
// DISTILLATION ENGINE TESTS
// ============================================

describe('DistillationEngine', () => {
    let engine;

    before(() => {
        engine = new DistillationEngine();
    });

    it('should configure distillation', () => {
        const config = engine.configure({
            teacher: 'phi-1.5',
            student: 'minilm-l6',
            temperature: 4.0,
            alpha: 0.5,
        });

        assert.strictEqual(config.temperature, 4.0);
        assert.strictEqual(config.alpha, 0.5);
    });

    it('should compute distillation loss', () => {
        engine.configure({ temperature: 4.0, alpha: 0.5 });

        const teacherLogits = [2.0, 1.0, 0.5, 0.1];
        const studentLogits = [1.8, 1.1, 0.4, 0.2];
        const labels = [1, 0, 0, 0];

        const loss = engine.computeLoss(teacherLogits, studentLogits, labels);

        assert.ok(typeof loss.total === 'number', 'Should compute total loss');
        assert.ok(loss.total >= 0, 'Loss should be non-negative');
        assert.ok(typeof loss.distillation === 'number', 'Should have distillation loss');
    });

    it('should get training configuration', () => {
        const config = engine.getTrainingConfig();

        assert.ok(config.epochs > 0, 'Should have epochs');
        assert.ok(config.learningRate > 0, 'Should have learning rate');
        assert.ok(config.batchSize > 0, 'Should have batch size');
    });
});

// ============================================
// BENCHMARK TESTS
// ============================================

describe('ComprehensiveBenchmark', () => {
    let benchmark;

    before(() => {
        benchmark = new ComprehensiveBenchmark();
    });

    after(() => {
        benchmark.reset();
    });

    describe('Accuracy Meter', () => {
        it('should compute MSE', () => {
            const meter = new AccuracyMeter();
            meter.addPrediction([1, 2, 3], [1.1, 2.1, 3.1]);
            meter.addPrediction([4, 5, 6], [4.1, 5.1, 6.1]);

            const metrics = meter.getMetrics();

            assert.ok(metrics.mse > 0, 'MSE should be positive');
            assert.ok(metrics.mse < 0.1, 'MSE should be small for close values');
        });

        it('should compute cosine similarity', () => {
            const meter = new AccuracyMeter();
            meter.addPrediction([1, 0, 0], [1, 0, 0]); // Identical
            meter.addPrediction([0, 1, 0], [0, 1, 0]); // Identical

            const metrics = meter.getMetrics();

            assert.ok(metrics.cosineSimilarity > 0.99, 'Should have high similarity');
        });
    });

    describe('Latency Profiler', () => {
        it('should measure latency', async () => {
            const profiler = new LatencyProfiler();

            profiler.start('test-section');
            await new Promise(resolve => setTimeout(resolve, 10));
            profiler.end('test-section');

            const stats = profiler.getStats('test-section');

            assert.ok(stats.mean >= 10, 'Should measure at least 10ms');
            assert.strictEqual(stats.count, 1);
        });
    });

    describe('Memory Profiler', () => {
        it('should take snapshots', () => {
            const profiler = new MemoryProfiler();

            const snapshot = profiler.snapshot('test');

            assert.ok(snapshot.label === 'test');
            assert.ok(typeof snapshot.heapUsed === 'number');
        });

        it('should track peak memory', () => {
            const profiler = new MemoryProfiler();

            profiler.snapshot('before');
            // Allocate some memory
            const arr = new Array(1000000).fill(0);
            profiler.snapshot('after');

            const summary = profiler.getSummary();
            assert.ok(summary.peakMemoryMB > 0);
        });
    });

    describe('Benchmark Suite', () => {
        it('should run quick benchmark', async () => {
            const result = await benchmark.runSuite('minilm-l6', 'quick');

            assert.strictEqual(result.model, 'minilm-l6');
            assert.ok(result.benchmarks.length > 0);
            assert.ok(result.summary, 'Should have summary');
        });

        it('should generate recommendations', async () => {
            const result = await benchmark.runSuite('e5-small', 'quick');

            const firstBenchmark = result.benchmarks[0];
            assert.ok(firstBenchmark.recommendation, 'Should have recommendation');
            assert.ok(typeof firstBenchmark.recommendation.score === 'number');
        });
    });
});

// ============================================
// INTEGRATION TESTS
// ============================================

describe('Integration', () => {
    it('should optimize phi-1.5 to meet target size', async () => {
        const optimizer = new ModelOptimizer();

        // Quantize to INT4 for maximum compression
        const quantResult = await optimizer.quantize('phi-1.5', 'int4');

        // 280MB / 8 = 35MB, target is 70MB
        assert.ok(quantResult.quantizedSizeMB <= 70, 'Should meet target size');
    });

    it('should optimize qwen-0.5b to meet target size', async () => {
        const optimizer = new ModelOptimizer();

        const quantResult = await optimizer.quantize('qwen-0.5b', 'int4');

        // 430MB / 8 = ~54MB, target is 100MB
        assert.ok(quantResult.quantizedSizeMB <= 100, 'Should meet target size');
    });

    it('should optimize minilm-l6 to meet target size', async () => {
        const optimizer = new ModelOptimizer();

        const quantResult = await optimizer.quantize('minilm-l6', 'int8');

        // 22MB / 4 = 5.5MB, target is 8MB
        assert.ok(quantResult.quantizedSizeMB <= 8, 'Should meet target size');
    });

    it('should run complete optimization workflow', async () => {
        const optimizer = new ModelOptimizer();

        // Full pipeline
        const result = await optimizer.optimizePipeline('bge-small', {
            quantizeMethod: 'int8',
            prune: true,
            sparsity: 0.3,
            benchmark: true,
        });

        assert.ok(result.meetsTarget, 'Should meet target');
        assert.ok(result.steps.length >= 4, 'Should complete all steps');

        // Verify all steps completed
        const stepNames = result.steps.map(s => s.step);
        assert.ok(stepNames.includes('quantize'));
        assert.ok(stepNames.includes('prune'));
        assert.ok(stepNames.includes('onnx-optimize'));
        assert.ok(stepNames.includes('export'));
        assert.ok(stepNames.includes('benchmark'));
    });
});

console.log('Model Optimizer Tests');
