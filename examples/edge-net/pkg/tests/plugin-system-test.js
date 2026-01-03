#!/usr/bin/env node
/**
 * Plugin System Tests
 *
 * Comprehensive testing of the edge-net plugin architecture.
 */

import {
    PLUGIN_CATALOG,
    PLUGIN_BUNDLES,
    PluginCategory,
    PluginTier,
    Capability,
    PluginLoader,
    BasePlugin,
    validateManifest,
    validatePlugin,
    PluginRegistry,
    generatePluginTemplate,
} from '../plugins/index.js';

import { CompressionPlugin } from '../plugins/implementations/compression.js';
import { E2EEncryptionPlugin } from '../plugins/implementations/e2e-encryption.js';
import { FederatedLearningPlugin } from '../plugins/implementations/federated-learning.js';
import { ReputationStakingPlugin } from '../plugins/implementations/reputation-staking.js';
import { SwarmIntelligencePlugin } from '../plugins/implementations/swarm-intelligence.js';

// ============================================
// TEST UTILITIES
// ============================================

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`✅ ${name}`);
        passed++;
    } catch (error) {
        console.log(`❌ ${name}`);
        console.log(`   Error: ${error.message}`);
        failed++;
    }
}

async function testAsync(name, fn) {
    try {
        await fn();
        console.log(`✅ ${name}`);
        passed++;
    } catch (error) {
        console.log(`❌ ${name}`);
        console.log(`   Error: ${error.message}`);
        failed++;
    }
}

function assert(condition, message) {
    if (!condition) throw new Error(message || 'Assertion failed');
}

function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
}

// ============================================
// TESTS
// ============================================

console.log('\n╔════════════════════════════════════════════════════════════════╗');
console.log('║              PLUGIN SYSTEM TESTS                                ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

// --- Catalog Tests ---
console.log('\n--- Plugin Catalog ---\n');

test('Catalog has plugins', () => {
    assert(Object.keys(PLUGIN_CATALOG).length > 0, 'Catalog should have plugins');
});

test('All plugins have required fields', () => {
    for (const [id, plugin] of Object.entries(PLUGIN_CATALOG)) {
        assert(plugin.id === id, `Plugin ${id} ID mismatch`);
        assert(plugin.name, `Plugin ${id} missing name`);
        assert(plugin.version, `Plugin ${id} missing version`);
        assert(plugin.description, `Plugin ${id} missing description`);
        assert(plugin.category, `Plugin ${id} missing category`);
        assert(plugin.tier, `Plugin ${id} missing tier`);
    }
});

test('Plugin categories are valid', () => {
    const validCategories = Object.values(PluginCategory);
    for (const [id, plugin] of Object.entries(PLUGIN_CATALOG)) {
        assert(validCategories.includes(plugin.category),
            `Plugin ${id} has invalid category: ${plugin.category}`);
    }
});

test('Plugin tiers are valid', () => {
    const validTiers = Object.values(PluginTier);
    for (const [id, plugin] of Object.entries(PLUGIN_CATALOG)) {
        assert(validTiers.includes(plugin.tier),
            `Plugin ${id} has invalid tier: ${plugin.tier}`);
    }
});

test('Bundles reference valid plugins', () => {
    for (const [bundleId, bundle] of Object.entries(PLUGIN_BUNDLES)) {
        for (const pluginId of bundle.plugins) {
            assert(PLUGIN_CATALOG[pluginId],
                `Bundle ${bundleId} references missing plugin: ${pluginId}`);
        }
    }
});

// --- Plugin Loader Tests ---
console.log('\n--- Plugin Loader ---\n');

test('Plugin loader initializes', () => {
    const loader = new PluginLoader();
    assert(loader.getCatalog().length > 0, 'Loader should see catalog');
});

test('Loader respects tier restrictions', () => {
    const loader = new PluginLoader({
        allowedTiers: [PluginTier.STABLE],
    });

    const catalog = loader.getCatalog();
    const betaPlugin = catalog.find(p => p.tier === PluginTier.BETA);

    if (betaPlugin) {
        assert(!betaPlugin.isAllowed.allowed, 'Beta plugins should not be allowed');
    }
});

test('Loader respects capability restrictions', () => {
    const loader = new PluginLoader({
        deniedCapabilities: [Capability.SYSTEM_EXEC],
    });

    const catalog = loader.getCatalog();
    const execPlugin = catalog.find(p =>
        p.capabilities?.includes(Capability.SYSTEM_EXEC)
    );

    if (execPlugin) {
        assert(!execPlugin.isAllowed.allowed, 'Exec plugins should not be allowed');
    }
});

// --- Manifest Validation Tests ---
console.log('\n--- Manifest Validation ---\n');

test('Valid manifest passes validation', () => {
    const manifest = {
        id: 'test.valid-plugin',
        name: 'Valid Plugin',
        version: '1.0.0',
        description: 'A valid test plugin',
        category: PluginCategory.CORE,
        tier: PluginTier.STABLE,
        capabilities: [Capability.COMPUTE_WASM],
    };

    const result = validateManifest(manifest);
    assert(result.valid, `Validation should pass: ${result.errors.join(', ')}`);
});

test('Invalid ID fails validation', () => {
    const manifest = {
        id: 'InvalidID',
        name: 'Test',
        version: '1.0.0',
        description: 'Test',
        category: PluginCategory.CORE,
        tier: PluginTier.STABLE,
    };

    const result = validateManifest(manifest);
    assert(!result.valid, 'Invalid ID should fail');
    assert(result.errors.some(e => e.includes('ID')), 'Should mention ID error');
});

test('Missing fields fail validation', () => {
    const manifest = { id: 'test.plugin' };
    const result = validateManifest(manifest);
    assert(!result.valid, 'Missing fields should fail');
    assert(result.errors.length > 0, 'Should have errors');
});

// --- Plugin SDK Tests ---
console.log('\n--- Plugin SDK ---\n');

test('BasePlugin can be extended', () => {
    class TestPlugin extends BasePlugin {
        static manifest = {
            id: 'test.sdk-plugin',
            name: 'SDK Test Plugin',
            version: '1.0.0',
            description: 'Testing SDK',
            category: PluginCategory.CORE,
            tier: PluginTier.EXPERIMENTAL,
            capabilities: [],
        };

        doSomething() {
            return 'worked';
        }
    }

    const plugin = new TestPlugin({ option: 'value' });
    assertEqual(plugin.doSomething(), 'worked', 'Plugin method should work');
    assertEqual(plugin.config.option, 'value', 'Config should be set');
});

test('Plugin registry works', () => {
    const registry = new PluginRegistry();

    class TestPlugin extends BasePlugin {
        static manifest = {
            id: 'test.registry-plugin',
            name: 'Registry Test',
            version: '1.0.0',
            description: 'Testing registry',
            category: PluginCategory.CORE,
            tier: PluginTier.EXPERIMENTAL,
            capabilities: [],
        };
    }

    const result = registry.register(TestPlugin);
    assert(result.id === 'test.registry-plugin', 'Should return ID');
    assert(result.checksum, 'Should generate checksum');
    assert(registry.has('test.registry-plugin'), 'Should be registered');
});

test('Template generator works', () => {
    const template = generatePluginTemplate({
        id: 'my-org.my-plugin',
        name: 'My Plugin',
        category: PluginCategory.AI,
    });

    assert(template.includes('class'), 'Should generate class');
    assert(template.includes('my-org.my-plugin'), 'Should include ID');
    assert(template.includes('BasePlugin'), 'Should extend BasePlugin');
});

// --- Implementation Tests ---
console.log('\n--- Plugin Implementations ---\n');

test('Compression plugin works', () => {
    const comp = new CompressionPlugin({ threshold: 10 });

    // Test with compressible data
    const data = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const result = comp.compress(data);

    assert(result.compressed, 'Should compress');
    assert(result.compressedSize < data.length, 'Should reduce size');

    const decompressed = comp.decompress(result.data, true);
    assertEqual(decompressed.toString(), data, 'Should decompress correctly');
});

test('E2E Encryption plugin works', async () => {
    const crypto = new E2EEncryptionPlugin();

    // Establish session
    await crypto.establishSession('peer-1', 'fake-public-key');
    assert(crypto.hasSession('peer-1'), 'Session should exist');

    // Encrypt/decrypt
    const message = 'Hello, secure world!';
    const encrypted = crypto.encrypt('peer-1', message);

    assert(encrypted.ciphertext, 'Should have ciphertext');
    assert(encrypted.iv, 'Should have IV');
    assert(encrypted.authTag, 'Should have auth tag');

    const decrypted = crypto.decrypt('peer-1', encrypted);
    assertEqual(decrypted, message, 'Should decrypt correctly');
});

test('Federated Learning plugin works', async () => {
    const fl = new FederatedLearningPlugin({
        minParticipants: 2,
        localEpochs: 2,
    });

    // Start round
    const globalWeights = [0, 0, 0, 0, 0];
    const roundId = fl.startRound('test-model', globalWeights);

    assert(roundId, 'Should return round ID');

    // Simulate local training
    const localData = [
        { features: [1, 2, 3, 4, 5] },
        { features: [2, 3, 4, 5, 6] },
    ];

    await fl.trainLocal(roundId, localData, { participantId: 'node-1' });
    await fl.trainLocal(roundId, localData, { participantId: 'node-2' });

    // Check aggregation happened
    const status = fl.getRoundStatus(roundId);
    assertEqual(status.status, 'completed', 'Round should complete');
    assertEqual(status.participants, 2, 'Should have 2 participants');
});

test('Reputation staking plugin works', () => {
    const staking = new ReputationStakingPlugin({ minStake: 5 });

    // Mock credit system
    const credits = {
        balance: 100,
        getBalance: () => credits.balance,
        spendCredits: (_, amount) => { credits.balance -= amount; },
        earnCredits: (_, amount) => { credits.balance += amount; },
    };

    // Stake
    const stake = staking.stake('node-1', 20, credits);
    assertEqual(stake.staked, 20, 'Should stake 20');
    assertEqual(stake.reputation, 100, 'Should start at 100 rep');

    // Record success
    staking.recordSuccess('node-1');
    const newStake = staking.getStake('node-1');
    assertEqual(newStake.successfulTasks, 1, 'Should record success');

    // Slash
    const slashResult = staking.slash('node-1', 'test-misbehavior', 0.5);
    assert(slashResult.slashed > 0, 'Should slash');
    assert(slashResult.newReputation < 100, 'Should reduce reputation');
});

test('Swarm intelligence plugin works', async () => {
    const swarm = new SwarmIntelligencePlugin({
        populationSize: 20,
        iterations: 50,
        dimensions: 5,
    });

    // Create swarm with sphere function (minimize x²)
    swarm.createSwarm('test-swarm', {
        algorithm: 'pso',
        bounds: { min: -10, max: 10 },
        fitnessFunction: (x) => x.reduce((sum, v) => sum + v * v, 0),
    });

    // Run optimization
    const result = await swarm.optimize('test-swarm', {
        iterations: 50,
    });

    assert(result.bestFitness < 1, 'Should find good solution');
    assert(result.iterations === 50, 'Should run 50 iterations');
});

// --- Summary ---
console.log('\n╔════════════════════════════════════════════════════════════════╗');
console.log('║              TEST SUMMARY                                       ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

console.log(`  Passed: ${passed}`);
console.log(`  Failed: ${failed}`);
console.log(`  Total:  ${passed + failed}\n`);

if (failed > 0) {
    console.log('❌ Some tests failed\n');
    process.exit(1);
} else {
    console.log('✅ All tests passed!\n');
    process.exit(0);
}
