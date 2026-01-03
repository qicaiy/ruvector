#!/usr/bin/env node
/**
 * Edge-Net Plugin System Performance Benchmarks
 *
 * Comprehensive benchmarks measuring:
 * - Plugin loading performance (cold/warm/bundle)
 * - Plugin execution throughput
 * - Sandbox overhead
 * - Memory usage patterns
 *
 * Run: node tests/plugin-benchmark.js
 *
 * @module @ruvector/edge-net/tests/plugin-benchmark
 */

import { performance } from 'perf_hooks';
import { PluginLoader, PluginManager } from '../plugins/plugin-loader.js';
import { CompressionPlugin } from '../plugins/implementations/compression.js';
import { E2EEncryptionPlugin } from '../plugins/implementations/e2e-encryption.js';
import { SwarmIntelligencePlugin } from '../plugins/implementations/swarm-intelligence.js';
import { FederatedLearningPlugin } from '../plugins/implementations/federated-learning.js';
import { ReputationStakingPlugin } from '../plugins/implementations/reputation-staking.js';
import { PLUGIN_CATALOG, PLUGIN_BUNDLES } from '../plugins/plugin-manifest.js';

// ============================================
// BENCHMARK CONFIGURATION
// ============================================

const CONFIG = {
    // Iterations for statistical significance
    warmupIterations: 10,
    benchmarkIterations: 100,

    // Data sizes for throughput tests
    dataSizes: {
        small: 1024,         // 1 KB
        medium: 65536,       // 64 KB
        large: 1048576,      // 1 MB
    },

    // Swarm intelligence settings
    swarm: {
        populationSize: 50,
        dimensions: 10,
        iterations: 100,
    },

    // Federated learning settings
    federated: {
        localDataSize: 1000,
        participants: 5,
        epochs: 5,
    },

    // Reputation staking settings
    staking: {
        operations: 1000,
    },

    // Memory test settings
    memoryOperations: 1000,
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / 1048576).toFixed(2)} MB`;
}

function formatNumber(num) {
    if (num >= 1000000) return `${(num / 1000000).toFixed(2)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(2)}K`;
    return num.toFixed(2);
}

function generateRandomData(size) {
    const buffer = Buffer.alloc(size);
    for (let i = 0; i < size; i++) {
        buffer[i] = Math.floor(Math.random() * 256);
    }
    return buffer;
}

function generateCompressibleData(size) {
    // Data with repeating patterns (compresses well)
    const buffer = Buffer.alloc(size);
    const patterns = [0x00, 0xFF, 0xAA, 0x55];
    for (let i = 0; i < size; i++) {
        // Create runs of repeating bytes
        const runLength = Math.floor(Math.random() * 32) + 4;
        const pattern = patterns[Math.floor(Math.random() * patterns.length)];
        for (let j = 0; j < runLength && i + j < size; j++) {
            buffer[i + j] = pattern;
        }
        i += runLength - 1;
    }
    return buffer;
}

function getMemoryUsage() {
    const usage = process.memoryUsage();
    return {
        heapUsed: usage.heapUsed,
        heapTotal: usage.heapTotal,
        external: usage.external,
        rss: usage.rss,
    };
}

function calculateStats(times) {
    const sorted = [...times].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);
    const mean = sum / sorted.length;
    const variance = sorted.reduce((acc, t) => acc + Math.pow(t - mean, 2), 0) / sorted.length;
    const stdDev = Math.sqrt(variance);

    return {
        mean,
        median: sorted[Math.floor(sorted.length / 2)],
        min: sorted[0],
        max: sorted[sorted.length - 1],
        stdDev,
        p95: sorted[Math.floor(sorted.length * 0.95)],
        p99: sorted[Math.floor(sorted.length * 0.99)],
    };
}

// ============================================
// BENCHMARK RESULTS COLLECTOR
// ============================================

class BenchmarkResults {
    constructor() {
        this.results = new Map();
        this.startTime = Date.now();
    }

    add(category, name, metrics) {
        if (!this.results.has(category)) {
            this.results.set(category, new Map());
        }
        this.results.get(category).set(name, metrics);
    }

    generateReport() {
        const report = {
            timestamp: new Date().toISOString(),
            duration: Date.now() - this.startTime,
            nodeVersion: process.version,
            platform: process.platform,
            arch: process.arch,
            categories: {},
        };

        for (const [category, metrics] of this.results) {
            report.categories[category] = {};
            for (const [name, data] of metrics) {
                report.categories[category][name] = data;
            }
        }

        return report;
    }
}

// ============================================
// PLUGIN LOADING BENCHMARKS
// ============================================

async function benchmarkPluginLoading(results) {
    console.log('\n--- Plugin Loading Performance ---\n');

    // Cold load time (first load, no cache)
    console.log('Testing cold load times...');
    const coldLoadTimes = {};

    for (const pluginId of Object.keys(PLUGIN_CATALOG)) {
        PluginManager.reset();
        const loader = PluginManager.getInstance({ verifySignatures: false });

        global.gc?.(); // Force GC if available

        const start = performance.now();
        try {
            await loader.load(pluginId);
            coldLoadTimes[pluginId] = performance.now() - start;
        } catch (e) {
            coldLoadTimes[pluginId] = null; // Plugin not loadable
        }
    }

    // Filter out null values and calculate stats
    const validColdTimes = Object.values(coldLoadTimes).filter(t => t !== null);
    results.add('Plugin Loading', 'Cold Load Times (ms)', coldLoadTimes);
    results.add('Plugin Loading', 'Cold Load Stats', calculateStats(validColdTimes));

    // Warm load time (already loaded/cached)
    console.log('Testing warm load times...');
    const warmLoadTimes = {};

    for (const pluginId of Object.keys(PLUGIN_CATALOG)) {
        const loader = PluginManager.getInstance();
        const times = [];

        for (let i = 0; i < CONFIG.warmupIterations; i++) {
            const start = performance.now();
            try {
                await loader.load(pluginId); // Should return cached
                times.push(performance.now() - start);
            } catch (e) {
                // Skip
            }
        }

        if (times.length > 0) {
            warmLoadTimes[pluginId] = calculateStats(times);
        }
    }

    results.add('Plugin Loading', 'Warm Load Times (ms)', warmLoadTimes);

    // Bundle load time
    console.log('Testing bundle load times...');
    const bundleLoadTimes = {};

    for (const bundleName of Object.keys(PLUGIN_BUNDLES)) {
        PluginManager.reset();
        const loader = PluginManager.getInstance({ verifySignatures: false });

        global.gc?.();

        const start = performance.now();
        try {
            await loader.loadBundle(bundleName);
            bundleLoadTimes[bundleName] = performance.now() - start;
        } catch (e) {
            bundleLoadTimes[bundleName] = null;
        }
    }

    results.add('Plugin Loading', 'Bundle Load Times (ms)', bundleLoadTimes);

    // Memory footprint per plugin
    console.log('Measuring plugin memory footprint...');
    const memoryFootprints = {};

    for (const pluginId of Object.keys(PLUGIN_CATALOG)) {
        PluginManager.reset();
        global.gc?.();

        const beforeMem = getMemoryUsage();
        const loader = PluginManager.getInstance({ verifySignatures: false });

        try {
            await loader.load(pluginId);
            global.gc?.();

            const afterMem = getMemoryUsage();
            memoryFootprints[pluginId] = afterMem.heapUsed - beforeMem.heapUsed;
        } catch (e) {
            memoryFootprints[pluginId] = null;
        }
    }

    results.add('Plugin Loading', 'Memory Footprint (bytes)', memoryFootprints);

    console.log('  Cold load avg:', calculateStats(validColdTimes).mean.toFixed(3), 'ms');
}

// ============================================
// COMPRESSION PLUGIN BENCHMARKS
// ============================================

async function benchmarkCompression(results) {
    console.log('\n--- CompressionPlugin Performance ---\n');

    const plugin = new CompressionPlugin({ threshold: 0 }); // Disable threshold

    for (const [sizeName, size] of Object.entries(CONFIG.dataSizes)) {
        console.log(`Testing ${sizeName} data (${formatBytes(size)})...`);

        // Test with compressible data
        const compressibleData = generateCompressibleData(size);
        const randomData = generateRandomData(size);

        // Compression throughput (compressible)
        const compressTimes = [];
        const compressedSizes = [];

        for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
            const start = performance.now();
            const result = plugin.compress(compressibleData);
            compressTimes.push(performance.now() - start);
            compressedSizes.push(result.compressedSize || compressibleData.length);
        }

        const compressStats = calculateStats(compressTimes);
        const avgCompressedSize = compressedSizes.reduce((a, b) => a + b, 0) / compressedSizes.length;
        const compressionRatio = avgCompressedSize / size;
        const throughputMBs = (size / 1048576) / (compressStats.mean / 1000);

        results.add('Compression', `${sizeName} Compress (compressible)`, {
            throughputMBs: throughputMBs.toFixed(2),
            avgTimeMs: compressStats.mean.toFixed(3),
            p95Ms: compressStats.p95.toFixed(3),
            compressionRatio: compressionRatio.toFixed(3),
        });

        // Decompression throughput
        const compressedData = plugin.compress(compressibleData);
        const decompressTimes = [];

        for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
            const start = performance.now();
            plugin.decompress(compressedData.data, compressedData.compressed);
            decompressTimes.push(performance.now() - start);
        }

        const decompressStats = calculateStats(decompressTimes);
        const decompressThroughput = (size / 1048576) / (decompressStats.mean / 1000);

        results.add('Compression', `${sizeName} Decompress`, {
            throughputMBs: decompressThroughput.toFixed(2),
            avgTimeMs: decompressStats.mean.toFixed(3),
            p95Ms: decompressStats.p95.toFixed(3),
        });

        // Random data (less compressible)
        const randomCompressTimes = [];
        for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
            const start = performance.now();
            plugin.compress(randomData);
            randomCompressTimes.push(performance.now() - start);
        }

        const randomCompressStats = calculateStats(randomCompressTimes);
        results.add('Compression', `${sizeName} Compress (random)`, {
            throughputMBs: ((size / 1048576) / (randomCompressStats.mean / 1000)).toFixed(2),
            avgTimeMs: randomCompressStats.mean.toFixed(3),
        });
    }

    console.log('  Compression benchmarks complete');
}

// ============================================
// E2E ENCRYPTION PLUGIN BENCHMARKS
// ============================================

async function benchmarkEncryption(results) {
    console.log('\n--- E2EEncryptionPlugin Performance ---\n');

    const plugin = new E2EEncryptionPlugin({ forwardSecrecy: false });
    await plugin.init();

    // Setup sessions
    const peers = ['peer1', 'peer2', 'peer3', 'peer4', 'peer5'];
    for (const peer of peers) {
        await plugin.establishSession(peer, 'dummy-public-key');
    }

    for (const [sizeName, size] of Object.entries(CONFIG.dataSizes)) {
        console.log(`Testing ${sizeName} data (${formatBytes(size)})...`);

        const testData = generateRandomData(size).toString('base64');

        // Encryption ops/sec
        const encryptTimes = [];
        for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
            const peer = peers[i % peers.length];
            const start = performance.now();
            plugin.encrypt(peer, testData);
            encryptTimes.push(performance.now() - start);
        }

        const encryptStats = calculateStats(encryptTimes);
        const encryptOpsPerSec = 1000 / encryptStats.mean;

        results.add('Encryption', `${sizeName} Encrypt`, {
            opsPerSec: formatNumber(encryptOpsPerSec),
            avgTimeMs: encryptStats.mean.toFixed(3),
            p95Ms: encryptStats.p95.toFixed(3),
            throughputMBs: ((size / 1048576) / (encryptStats.mean / 1000)).toFixed(2),
        });

        // Decryption ops/sec
        const encrypted = plugin.encrypt('peer1', testData);
        const decryptTimes = [];

        for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
            const start = performance.now();
            plugin.decrypt('peer1', encrypted);
            decryptTimes.push(performance.now() - start);
        }

        const decryptStats = calculateStats(decryptTimes);
        const decryptOpsPerSec = 1000 / decryptStats.mean;

        results.add('Encryption', `${sizeName} Decrypt`, {
            opsPerSec: formatNumber(decryptOpsPerSec),
            avgTimeMs: decryptStats.mean.toFixed(3),
            p95Ms: decryptStats.p95.toFixed(3),
            throughputMBs: ((size / 1048576) / (decryptStats.mean / 1000)).toFixed(2),
        });
    }

    // Session establishment overhead
    console.log('Testing session establishment...');
    const sessionTimes = [];
    for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
        const start = performance.now();
        await plugin.establishSession(`test-peer-${i}`, 'dummy-key');
        sessionTimes.push(performance.now() - start);
    }

    const sessionStats = calculateStats(sessionTimes);
    results.add('Encryption', 'Session Establishment', {
        opsPerSec: formatNumber(1000 / sessionStats.mean),
        avgTimeMs: sessionStats.mean.toFixed(3),
        p95Ms: sessionStats.p95.toFixed(3),
    });

    await plugin.destroy();
    console.log('  Encryption benchmarks complete');
}

// ============================================
// SWARM INTELLIGENCE PLUGIN BENCHMARKS
// ============================================

async function benchmarkSwarmIntelligence(results) {
    console.log('\n--- SwarmIntelligencePlugin Performance ---\n');

    const algorithms = ['pso', 'ga', 'de', 'aco'];

    for (const algorithm of algorithms) {
        console.log(`Testing ${algorithm.toUpperCase()} algorithm...`);

        const plugin = new SwarmIntelligencePlugin({
            algorithm,
            populationSize: CONFIG.swarm.populationSize,
            iterations: CONFIG.swarm.iterations,
            dimensions: CONFIG.swarm.dimensions,
        });

        // Single step performance
        const stepTimes = [];
        const swarm = plugin.createSwarm('bench-swarm', {
            algorithm,
            dimensions: CONFIG.swarm.dimensions,
            fitnessFunction: (pos) => pos.reduce((sum, x) => sum + x * x, 0),
        });

        for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
            const start = performance.now();
            plugin.step('bench-swarm');
            stepTimes.push(performance.now() - start);
        }

        const stepStats = calculateStats(stepTimes);
        const iterationsPerSec = 1000 / stepStats.mean;

        results.add('Swarm Intelligence', `${algorithm.toUpperCase()} Step`, {
            iterationsPerSec: formatNumber(iterationsPerSec),
            avgTimeMs: stepStats.mean.toFixed(3),
            p95Ms: stepStats.p95.toFixed(3),
            populationSize: CONFIG.swarm.populationSize,
            dimensions: CONFIG.swarm.dimensions,
        });

        // Full optimization run
        const optimizeTimes = [];
        for (let i = 0; i < 10; i++) { // Fewer iterations for full optimization
            const newSwarm = plugin.createSwarm(`bench-opt-${i}`, {
                algorithm,
                dimensions: CONFIG.swarm.dimensions,
            });

            const start = performance.now();
            await plugin.optimize(`bench-opt-${i}`, { iterations: 50 });
            optimizeTimes.push(performance.now() - start);
        }

        const optStats = calculateStats(optimizeTimes);
        results.add('Swarm Intelligence', `${algorithm.toUpperCase()} Full Optimization (50 iter)`, {
            avgTimeMs: optStats.mean.toFixed(2),
            p95Ms: optStats.p95.toFixed(2),
            totalIterationsPerSec: formatNumber((50 * 1000) / optStats.mean),
        });
    }

    // Particle scaling test
    console.log('Testing particle scaling...');
    const scalingSizes = [10, 50, 100, 200, 500];
    const scalingResults = {};

    for (const popSize of scalingSizes) {
        const plugin = new SwarmIntelligencePlugin({
            algorithm: 'pso',
            populationSize: popSize,
            dimensions: CONFIG.swarm.dimensions,
        });

        const swarm = plugin.createSwarm('scale-test');
        const times = [];

        for (let i = 0; i < 50; i++) {
            const start = performance.now();
            plugin.step('scale-test');
            times.push(performance.now() - start);
        }

        const stats = calculateStats(times);
        scalingResults[popSize] = {
            avgTimeMs: stats.mean.toFixed(3),
            iterPerSec: formatNumber(1000 / stats.mean),
        };
    }

    results.add('Swarm Intelligence', 'PSO Population Scaling', scalingResults);
    console.log('  Swarm intelligence benchmarks complete');
}

// ============================================
// FEDERATED LEARNING PLUGIN BENCHMARKS
// ============================================

async function benchmarkFederatedLearning(results) {
    console.log('\n--- FederatedLearningPlugin Performance ---\n');

    const plugin = new FederatedLearningPlugin({
        aggregationStrategy: 'fedavg',
        localEpochs: CONFIG.federated.epochs,
        differentialPrivacy: true,
        minParticipants: 1, // Allow single participant for benchmarking
    });

    // Generate test data
    const localData = Array(CONFIG.federated.localDataSize).fill(null).map(() => ({
        features: Array(10).fill(0).map(() => Math.random()),
        label: Math.random() > 0.5 ? 1 : 0,
    }));

    const globalWeights = Array(10).fill(0).map(() => Math.random());

    // Local training performance
    console.log('Testing local training...');
    const trainTimes = [];

    for (let i = 0; i < 50; i++) {
        const roundId = plugin.startRound(`model-${i}`, globalWeights);

        const start = performance.now();
        await plugin.trainLocal(roundId, localData, {
            participantId: `participant-${i}`,
            epochs: CONFIG.federated.epochs,
        });
        trainTimes.push(performance.now() - start);
    }

    const trainStats = calculateStats(trainTimes);
    results.add('Federated Learning', 'Local Training', {
        avgTimeMs: trainStats.mean.toFixed(2),
        p95Ms: trainStats.p95.toFixed(2),
        samplesProcessed: CONFIG.federated.localDataSize,
        epochs: CONFIG.federated.epochs,
        samplesPerSecond: formatNumber((CONFIG.federated.localDataSize * CONFIG.federated.epochs * 1000) / trainStats.mean),
    });

    // Aggregation performance (FedAvg)
    console.log('Testing aggregation...');
    const aggregationTimes = [];

    for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
        // Simulate multiple participant updates
        const updates = Array(CONFIG.federated.participants).fill(null).map(() => ({
            update: Array(10).fill(0).map(() => Math.random()),
            dataSize: Math.floor(Math.random() * 1000) + 100,
        }));

        const start = performance.now();
        plugin._fedAvg(updates);
        aggregationTimes.push(performance.now() - start);
    }

    const aggStats = calculateStats(aggregationTimes);
    results.add('Federated Learning', 'FedAvg Aggregation', {
        avgTimeMs: aggStats.mean.toFixed(4),
        p95Ms: aggStats.p95.toFixed(4),
        opsPerSec: formatNumber(1000 / aggStats.mean),
        participants: CONFIG.federated.participants,
    });

    // Differential privacy overhead
    console.log('Testing differential privacy overhead...');
    const dpTimes = [];
    const noDpTimes = [];

    for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
        const weights = Array(100).fill(0).map(() => Math.random());

        const start1 = performance.now();
        plugin._addDifferentialPrivacy([...weights]);
        dpTimes.push(performance.now() - start1);

        // Baseline (no DP)
        const start2 = performance.now();
        const _ = weights.map(w => w); // Just copy
        noDpTimes.push(performance.now() - start2);
    }

    const dpStats = calculateStats(dpTimes);
    const noDpStats = calculateStats(noDpTimes);
    results.add('Federated Learning', 'Differential Privacy Overhead', {
        withDpMs: dpStats.mean.toFixed(4),
        withoutDpMs: noDpStats.mean.toFixed(4),
        overheadFactor: (dpStats.mean / noDpStats.mean).toFixed(2),
    });

    console.log('  Federated learning benchmarks complete');
}

// ============================================
// REPUTATION STAKING PLUGIN BENCHMARKS
// ============================================

async function benchmarkReputationStaking(results) {
    console.log('\n--- ReputationStakingPlugin Performance ---\n');

    const plugin = new ReputationStakingPlugin();

    // Mock credit system
    const balances = new Map();
    const creditSystem = {
        getBalance: (nodeId) => balances.get(nodeId) || 0,
        spendCredits: (nodeId, amount) => {
            balances.set(nodeId, (balances.get(nodeId) || 0) - amount);
        },
        earnCredits: (nodeId, amount) => {
            balances.set(nodeId, (balances.get(nodeId) || 0) + amount);
        },
    };

    // Initialize nodes with credits
    for (let i = 0; i < 100; i++) {
        balances.set(`node-${i}`, 10000);
    }

    // Stake operations
    console.log('Testing stake operations...');
    const stakeTimes = [];
    for (let i = 0; i < CONFIG.staking.operations; i++) {
        const nodeId = `node-${i % 100}`;
        const start = performance.now();
        try {
            plugin.stake(nodeId, 100, creditSystem);
        } catch (e) {
            // Ignore insufficient balance
        }
        stakeTimes.push(performance.now() - start);
    }

    const stakeStats = calculateStats(stakeTimes);
    results.add('Reputation Staking', 'Stake Operations', {
        opsPerSec: formatNumber(1000 / stakeStats.mean),
        avgTimeMs: stakeStats.mean.toFixed(4),
        p95Ms: stakeStats.p95.toFixed(4),
    });

    // Slash operations
    console.log('Testing slash operations...');
    const slashTimes = [];
    for (let i = 0; i < CONFIG.staking.operations; i++) {
        const nodeId = `node-${i % 100}`;
        const start = performance.now();
        plugin.slash(nodeId, 'benchmark-test', 0.5);
        slashTimes.push(performance.now() - start);
    }

    const slashStats = calculateStats(slashTimes);
    results.add('Reputation Staking', 'Slash Operations', {
        opsPerSec: formatNumber(1000 / slashStats.mean),
        avgTimeMs: slashStats.mean.toFixed(4),
        p95Ms: slashStats.p95.toFixed(4),
    });

    // Reputation checks
    console.log('Testing reputation lookups...');
    const reputationTimes = [];
    for (let i = 0; i < CONFIG.staking.operations; i++) {
        const nodeId = `node-${i % 100}`;
        const start = performance.now();
        plugin.getReputation(nodeId);
        reputationTimes.push(performance.now() - start);
    }

    const repStats = calculateStats(reputationTimes);
    results.add('Reputation Staking', 'Reputation Lookups', {
        opsPerSec: formatNumber(1000 / repStats.mean),
        avgTimeMs: repStats.mean.toFixed(5),
        p95Ms: repStats.p95.toFixed(5),
    });

    // Leaderboard computation
    console.log('Testing leaderboard computation...');
    const leaderboardTimes = [];
    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        plugin.getLeaderboard(10);
        leaderboardTimes.push(performance.now() - start);
    }

    const lbStats = calculateStats(leaderboardTimes);
    results.add('Reputation Staking', 'Leaderboard (top 10)', {
        opsPerSec: formatNumber(1000 / lbStats.mean),
        avgTimeMs: lbStats.mean.toFixed(3),
        p95Ms: lbStats.p95.toFixed(3),
        stakerCount: plugin.stakes.size,
    });

    // Eligibility checks
    console.log('Testing eligibility checks...');
    const eligibilityTimes = [];
    for (let i = 0; i < CONFIG.staking.operations; i++) {
        const nodeId = `node-${i % 100}`;
        const start = performance.now();
        plugin.isEligible(nodeId, 50, 100);
        eligibilityTimes.push(performance.now() - start);
    }

    const eligStats = calculateStats(eligibilityTimes);
    results.add('Reputation Staking', 'Eligibility Checks', {
        opsPerSec: formatNumber(1000 / eligStats.mean),
        avgTimeMs: eligStats.mean.toFixed(5),
        p95Ms: eligStats.p95.toFixed(5),
    });

    console.log('  Reputation staking benchmarks complete');
}

// ============================================
// SANDBOX OVERHEAD BENCHMARKS
// ============================================

async function benchmarkSandboxOverhead(results) {
    console.log('\n--- Sandbox Overhead ---\n');

    const loader = new PluginLoader({ verifySignatures: false });

    // Capability check overhead
    console.log('Testing capability check overhead...');
    const sandbox = loader._createSandbox(['NETWORK_CONNECT', 'CRYPTO_ENCRYPT']);

    const capCheckTimes = [];
    for (let i = 0; i < CONFIG.benchmarkIterations * 10; i++) {
        const start = performance.now();
        sandbox.hasCapability('NETWORK_CONNECT');
        capCheckTimes.push(performance.now() - start);
    }

    const capStats = calculateStats(capCheckTimes);
    results.add('Sandbox Overhead', 'Capability Check', {
        opsPerSec: formatNumber(1000 / capStats.mean),
        avgTimeNs: (capStats.mean * 1000000).toFixed(0),
        p95Ns: (capStats.p95 * 1000000).toFixed(0),
    });

    // API call overhead vs direct
    console.log('Testing API call overhead...');

    // Direct function call baseline
    const directTimes = [];
    const directFn = (x) => x * 2;
    for (let i = 0; i < CONFIG.benchmarkIterations * 10; i++) {
        const start = performance.now();
        directFn(i);
        directTimes.push(performance.now() - start);
    }

    // Sandboxed API call
    const api = loader._createPluginAPI({ capabilities: ['CRYPTO_ENCRYPT'] }, sandbox);
    const sandboxedTimes = [];
    for (let i = 0; i < CONFIG.benchmarkIterations * 10; i++) {
        const start = performance.now();
        sandbox.hasCapability('CRYPTO_ENCRYPT');
        // Simulate API work
        const _ = i * 2;
        sandboxedTimes.push(performance.now() - start);
    }

    const directStats = calculateStats(directTimes);
    const sandboxedStats = calculateStats(sandboxedTimes);

    results.add('Sandbox Overhead', 'Direct vs Sandboxed Call', {
        directAvgNs: (directStats.mean * 1000000).toFixed(0),
        sandboxedAvgNs: (sandboxedStats.mean * 1000000).toFixed(0),
        overheadFactor: (sandboxedStats.mean / directStats.mean).toFixed(2),
        overheadNs: ((sandboxedStats.mean - directStats.mean) * 1000000).toFixed(0),
    });

    // Capability require (with exception path)
    console.log('Testing capability require overhead...');
    const requireTimes = [];
    for (let i = 0; i < CONFIG.benchmarkIterations * 10; i++) {
        const start = performance.now();
        sandbox.require('NETWORK_CONNECT');
        requireTimes.push(performance.now() - start);
    }

    const requireStats = calculateStats(requireTimes);
    results.add('Sandbox Overhead', 'Capability Require', {
        opsPerSec: formatNumber(1000 / requireStats.mean),
        avgTimeNs: (requireStats.mean * 1000000).toFixed(0),
    });

    console.log('  Sandbox overhead benchmarks complete');
}

// ============================================
// MEMORY USAGE BENCHMARKS
// ============================================

async function benchmarkMemoryUsage(results) {
    console.log('\n--- Memory Usage ---\n');

    // Force GC if available
    if (global.gc) {
        global.gc();
    }

    const baseMemory = getMemoryUsage();

    // Base memory footprint (no plugins)
    results.add('Memory Usage', 'Base Memory', {
        heapUsed: formatBytes(baseMemory.heapUsed),
        heapTotal: formatBytes(baseMemory.heapTotal),
        rss: formatBytes(baseMemory.rss),
    });

    // Per-plugin memory cost
    console.log('Measuring per-plugin memory...');
    const pluginMemory = {};

    const plugins = [
        { name: 'Compression', factory: () => new CompressionPlugin() },
        { name: 'E2E Encryption', factory: () => new E2EEncryptionPlugin() },
        { name: 'Swarm Intelligence', factory: () => new SwarmIntelligencePlugin() },
        { name: 'Federated Learning', factory: () => new FederatedLearningPlugin() },
        { name: 'Reputation Staking', factory: () => new ReputationStakingPlugin() },
    ];

    for (const { name, factory } of plugins) {
        global.gc?.();
        const before = getMemoryUsage();

        const instances = [];
        for (let i = 0; i < 10; i++) {
            instances.push(factory());
        }

        global.gc?.();
        const after = getMemoryUsage();

        pluginMemory[name] = {
            perInstance: formatBytes((after.heapUsed - before.heapUsed) / 10),
            perInstanceBytes: Math.floor((after.heapUsed - before.heapUsed) / 10),
        };
    }

    results.add('Memory Usage', 'Per-Plugin Memory', pluginMemory);

    // Memory after 1000 operations
    console.log('Measuring memory after sustained operations...');

    global.gc?.();
    const beforeOps = getMemoryUsage();

    // Compression operations
    const compression = new CompressionPlugin({ threshold: 0 });
    for (let i = 0; i < CONFIG.memoryOperations; i++) {
        const data = generateCompressibleData(1024);
        const compressed = compression.compress(data);
        compression.decompress(compressed.data, compressed.compressed);
    }

    // Encryption operations
    const encryption = new E2EEncryptionPlugin({ forwardSecrecy: false });
    for (let i = 0; i < CONFIG.memoryOperations; i++) {
        await encryption.establishSession(`peer-${i}`, 'key');
        encryption.encrypt(`peer-${i}`, 'test message');
    }

    // Swarm operations
    const swarm = new SwarmIntelligencePlugin();
    for (let i = 0; i < Math.min(100, CONFIG.memoryOperations); i++) {
        swarm.createSwarm(`swarm-${i}`);
        for (let j = 0; j < 10; j++) {
            swarm.step(`swarm-${i}`);
        }
    }

    global.gc?.();
    const afterOps = getMemoryUsage();

    results.add('Memory Usage', 'After 1000 Operations', {
        heapGrowth: formatBytes(afterOps.heapUsed - beforeOps.heapUsed),
        heapGrowthBytes: afterOps.heapUsed - beforeOps.heapUsed,
        finalHeap: formatBytes(afterOps.heapUsed),
        operationsPerMB: Math.floor(CONFIG.memoryOperations / ((afterOps.heapUsed - beforeOps.heapUsed) / 1048576)),
    });

    // Memory pressure test (create and release)
    console.log('Testing memory release...');

    global.gc?.();
    const beforePressure = getMemoryUsage();

    // Create many instances
    const instances = [];
    for (let i = 0; i < 100; i++) {
        instances.push(new CompressionPlugin());
        instances.push(new E2EEncryptionPlugin());
        instances.push(new SwarmIntelligencePlugin());
    }

    global.gc?.();
    const peakMemory = getMemoryUsage();

    // Release
    instances.length = 0;
    global.gc?.();
    const releasedMemory = getMemoryUsage();

    results.add('Memory Usage', 'Memory Pressure Test', {
        peakGrowth: formatBytes(peakMemory.heapUsed - beforePressure.heapUsed),
        afterRelease: formatBytes(releasedMemory.heapUsed - beforePressure.heapUsed),
        releaseEfficiency: ((1 - (releasedMemory.heapUsed - beforePressure.heapUsed) /
            (peakMemory.heapUsed - beforePressure.heapUsed)) * 100).toFixed(1) + '%',
    });

    console.log('  Memory usage benchmarks complete');
}

// ============================================
// BASELINE COMPARISON
// ============================================

async function benchmarkBaseline(results) {
    console.log('\n--- Baseline Comparison (No Plugins) ---\n');

    // Direct operations without plugin overhead

    // Raw buffer operations (baseline for compression)
    console.log('Testing raw buffer operations...');
    const rawBufferTimes = [];
    for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
        const data = Buffer.alloc(65536);
        const start = performance.now();
        const copy = Buffer.from(data);
        rawBufferTimes.push(performance.now() - start);
    }

    results.add('Baseline', 'Raw Buffer Copy (64KB)', {
        avgTimeMs: calculateStats(rawBufferTimes).mean.toFixed(4),
        throughputMBs: ((65536 / 1048576) / (calculateStats(rawBufferTimes).mean / 1000)).toFixed(2),
    });

    // Raw crypto operations (baseline for encryption)
    console.log('Testing raw crypto operations...');
    const { createCipheriv, randomBytes, createHash } = await import('crypto');

    const cryptoTimes = [];
    const key = randomBytes(32);
    const testData = randomBytes(1024);

    for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
        const iv = randomBytes(12);
        const start = performance.now();
        const cipher = createCipheriv('aes-256-gcm', key, iv);
        cipher.update(testData);
        cipher.final();
        cipher.getAuthTag();
        cryptoTimes.push(performance.now() - start);
    }

    results.add('Baseline', 'Raw AES-256-GCM (1KB)', {
        avgTimeMs: calculateStats(cryptoTimes).mean.toFixed(4),
        opsPerSec: formatNumber(1000 / calculateStats(cryptoTimes).mean),
    });

    // Map operations (baseline for staking)
    console.log('Testing raw Map operations...');
    const map = new Map();
    const mapTimes = [];

    for (let i = 0; i < CONFIG.staking.operations; i++) {
        const start = performance.now();
        map.set(`key-${i}`, { value: i, data: { nested: true } });
        map.get(`key-${i % (i + 1)}`);
        mapTimes.push(performance.now() - start);
    }

    results.add('Baseline', 'Raw Map Operations', {
        avgTimeNs: (calculateStats(mapTimes).mean * 1000000).toFixed(0),
        opsPerSec: formatNumber(1000 / calculateStats(mapTimes).mean),
    });

    // Math operations (baseline for swarm)
    console.log('Testing raw math operations...');
    const mathTimes = [];
    const dimensions = CONFIG.swarm.dimensions;
    const particles = CONFIG.swarm.populationSize;

    for (let i = 0; i < CONFIG.benchmarkIterations; i++) {
        const start = performance.now();
        for (let p = 0; p < particles; p++) {
            let sum = 0;
            for (let d = 0; d < dimensions; d++) {
                sum += Math.random() * Math.random() + Math.sqrt(Math.random());
            }
        }
        mathTimes.push(performance.now() - start);
    }

    results.add('Baseline', 'Raw Math Operations (50 particles x 10 dim)', {
        avgTimeMs: calculateStats(mathTimes).mean.toFixed(3),
        opsPerSec: formatNumber(1000 / calculateStats(mathTimes).mean),
    });

    console.log('  Baseline benchmarks complete');
}

// ============================================
// REPORT GENERATION
// ============================================

function generateTextReport(report) {
    const lines = [];

    lines.push('');
    lines.push('='.repeat(80));
    lines.push('  EDGE-NET PLUGIN SYSTEM PERFORMANCE BENCHMARKS');
    lines.push('='.repeat(80));
    lines.push('');
    lines.push(`Timestamp: ${report.timestamp}`);
    lines.push(`Duration: ${(report.duration / 1000).toFixed(2)}s`);
    lines.push(`Node.js: ${report.nodeVersion}`);
    lines.push(`Platform: ${report.platform} (${report.arch})`);
    lines.push('');

    for (const [category, metrics] of Object.entries(report.categories)) {
        lines.push('-'.repeat(80));
        lines.push(`  ${category}`);
        lines.push('-'.repeat(80));
        lines.push('');

        for (const [name, data] of Object.entries(metrics)) {
            if (typeof data === 'object' && !Array.isArray(data)) {
                lines.push(`  ${name}:`);
                for (const [key, value] of Object.entries(data)) {
                    if (typeof value === 'object') {
                        lines.push(`    ${key}: ${JSON.stringify(value)}`);
                    } else {
                        lines.push(`    ${key}: ${value}`);
                    }
                }
            } else {
                lines.push(`  ${name}: ${JSON.stringify(data)}`);
            }
            lines.push('');
        }
    }

    lines.push('='.repeat(80));
    lines.push('  SUMMARY & RECOMMENDATIONS');
    lines.push('='.repeat(80));
    lines.push('');

    // Extract key metrics for summary
    const compressionThroughput = report.categories['Compression']?.['medium Compress (compressible)']?.throughputMBs || 'N/A';
    const encryptionOps = report.categories['Encryption']?.['small Encrypt']?.opsPerSec || 'N/A';
    const swarmIterSec = report.categories['Swarm Intelligence']?.['PSO Step']?.iterationsPerSec || 'N/A';
    const stakingOps = report.categories['Reputation Staking']?.['Stake Operations']?.opsPerSec || 'N/A';
    const sandboxOverhead = report.categories['Sandbox Overhead']?.['Direct vs Sandboxed Call']?.overheadFactor || 'N/A';

    lines.push('Key Performance Metrics:');
    lines.push(`  - Compression throughput: ${compressionThroughput} MB/s`);
    lines.push(`  - Encryption operations: ${encryptionOps} ops/sec`);
    lines.push(`  - Swarm PSO iterations: ${swarmIterSec} iter/sec`);
    lines.push(`  - Staking operations: ${stakingOps} ops/sec`);
    lines.push(`  - Sandbox overhead factor: ${sandboxOverhead}x`);
    lines.push('');

    lines.push('Optimization Recommendations:');
    lines.push('');

    // Provide context-aware recommendations
    const compressionNum = parseFloat(compressionThroughput) || 0;
    if (compressionNum < 100) {
        lines.push('  [!] Compression: Consider using native WASM LZ4 for better throughput');
    } else {
        lines.push('  [OK] Compression: Throughput is acceptable for most use cases');
    }

    const sandboxNum = parseFloat(sandboxOverhead) || 0;
    if (sandboxNum > 5) {
        lines.push('  [!] Sandbox: High overhead detected - consider caching capability checks');
    } else {
        lines.push('  [OK] Sandbox: Overhead is within acceptable range');
    }

    lines.push('');
    lines.push('For production deployments:');
    lines.push('  1. Enable WASM SIMD for compression/encryption acceleration');
    lines.push('  2. Use worker threads for swarm intelligence computations');
    lines.push('  3. Consider batching staking operations for high-throughput scenarios');
    lines.push('  4. Monitor memory usage with --expose-gc for accurate measurements');
    lines.push('');

    return lines.join('\n');
}

function generateSummaryTable(report) {
    const lines = [];

    lines.push('');
    lines.push('+' + '-'.repeat(78) + '+');
    lines.push('|' + ' '.repeat(25) + 'PERFORMANCE SUMMARY TABLE' + ' '.repeat(28) + '|');
    lines.push('+' + '-'.repeat(78) + '+');
    lines.push('| Plugin/Operation              | Metric              | Value            | Unit     |');
    lines.push('+' + '-'.repeat(78) + '+');

    const tableRows = [
        // Compression
        { plugin: 'CompressionPlugin', op: 'Compress (64KB)', metric: 'Throughput', value: report.categories['Compression']?.['medium Compress (compressible)']?.throughputMBs || 'N/A', unit: 'MB/s' },
        { plugin: 'CompressionPlugin', op: 'Decompress (64KB)', metric: 'Throughput', value: report.categories['Compression']?.['medium Decompress']?.throughputMBs || 'N/A', unit: 'MB/s' },

        // Encryption
        { plugin: 'E2EEncryptionPlugin', op: 'Encrypt (1KB)', metric: 'Operations', value: report.categories['Encryption']?.['small Encrypt']?.opsPerSec || 'N/A', unit: 'ops/sec' },
        { plugin: 'E2EEncryptionPlugin', op: 'Decrypt (1KB)', metric: 'Operations', value: report.categories['Encryption']?.['small Decrypt']?.opsPerSec || 'N/A', unit: 'ops/sec' },
        { plugin: 'E2EEncryptionPlugin', op: 'Session Setup', metric: 'Operations', value: report.categories['Encryption']?.['Session Establishment']?.opsPerSec || 'N/A', unit: 'ops/sec' },

        // Swarm Intelligence
        { plugin: 'SwarmIntelligencePlugin', op: 'PSO Step', metric: 'Iterations', value: report.categories['Swarm Intelligence']?.['PSO Step']?.iterationsPerSec || 'N/A', unit: 'iter/sec' },
        { plugin: 'SwarmIntelligencePlugin', op: 'GA Step', metric: 'Iterations', value: report.categories['Swarm Intelligence']?.['GA Step']?.iterationsPerSec || 'N/A', unit: 'iter/sec' },
        { plugin: 'SwarmIntelligencePlugin', op: 'DE Step', metric: 'Iterations', value: report.categories['Swarm Intelligence']?.['DE Step']?.iterationsPerSec || 'N/A', unit: 'iter/sec' },

        // Federated Learning
        { plugin: 'FederatedLearningPlugin', op: 'Local Training', metric: 'Samples', value: report.categories['Federated Learning']?.['Local Training']?.samplesPerSecond || 'N/A', unit: 'samples/s' },
        { plugin: 'FederatedLearningPlugin', op: 'FedAvg', metric: 'Aggregations', value: report.categories['Federated Learning']?.['FedAvg Aggregation']?.opsPerSec || 'N/A', unit: 'ops/sec' },

        // Reputation Staking
        { plugin: 'ReputationStakingPlugin', op: 'Stake', metric: 'Operations', value: report.categories['Reputation Staking']?.['Stake Operations']?.opsPerSec || 'N/A', unit: 'ops/sec' },
        { plugin: 'ReputationStakingPlugin', op: 'Slash', metric: 'Operations', value: report.categories['Reputation Staking']?.['Slash Operations']?.opsPerSec || 'N/A', unit: 'ops/sec' },
        { plugin: 'ReputationStakingPlugin', op: 'Reputation Lookup', metric: 'Operations', value: report.categories['Reputation Staking']?.['Reputation Lookups']?.opsPerSec || 'N/A', unit: 'ops/sec' },

        // Sandbox
        { plugin: 'Sandbox', op: 'Capability Check', metric: 'Operations', value: report.categories['Sandbox Overhead']?.['Capability Check']?.opsPerSec || 'N/A', unit: 'ops/sec' },
        { plugin: 'Sandbox', op: 'Overhead Factor', metric: 'Factor', value: report.categories['Sandbox Overhead']?.['Direct vs Sandboxed Call']?.overheadFactor || 'N/A', unit: 'x' },
    ];

    for (const row of tableRows) {
        const pluginPad = (row.plugin + ' ' + row.op).substring(0, 30).padEnd(30);
        const metricPad = row.metric.padEnd(20);
        const valuePad = String(row.value).padEnd(17);
        const unitPad = row.unit.padEnd(9);
        lines.push(`| ${pluginPad}| ${metricPad}| ${valuePad}| ${unitPad}|`);
    }

    lines.push('+' + '-'.repeat(78) + '+');
    lines.push('');

    return lines.join('\n');
}

// ============================================
// MAIN BENCHMARK RUNNER
// ============================================

async function runBenchmarks() {
    console.log('');
    console.log('='.repeat(60));
    console.log('  Edge-Net Plugin System Performance Benchmarks');
    console.log('='.repeat(60));
    console.log('');
    console.log('Configuration:');
    console.log(`  - Warmup iterations: ${CONFIG.warmupIterations}`);
    console.log(`  - Benchmark iterations: ${CONFIG.benchmarkIterations}`);
    console.log(`  - Data sizes: ${Object.entries(CONFIG.dataSizes).map(([k, v]) => `${k}=${formatBytes(v)}`).join(', ')}`);
    console.log(`  - GC available: ${typeof global.gc === 'function' ? 'Yes' : 'No (run with --expose-gc for accurate memory)'}`);
    console.log('');

    const results = new BenchmarkResults();

    try {
        // Run all benchmarks
        await benchmarkPluginLoading(results);
        await benchmarkCompression(results);
        await benchmarkEncryption(results);
        await benchmarkSwarmIntelligence(results);
        await benchmarkFederatedLearning(results);
        await benchmarkReputationStaking(results);
        await benchmarkSandboxOverhead(results);
        await benchmarkMemoryUsage(results);
        await benchmarkBaseline(results);

        // Generate reports
        const report = results.generateReport();

        console.log(generateSummaryTable(report));
        console.log(generateTextReport(report));

        // Output JSON for programmatic access
        console.log('\n--- JSON Output ---\n');
        console.log(JSON.stringify(report, null, 2));

        return report;

    } catch (error) {
        console.error('\nBenchmark failed:', error);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run if executed directly
runBenchmarks().then(() => {
    console.log('\nBenchmarks completed successfully.');
    process.exit(0);
}).catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
});

export { runBenchmarks, CONFIG };
