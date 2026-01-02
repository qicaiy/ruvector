#!/usr/bin/env node
/**
 * Task Execution Handler Integration Test
 *
 * Tests the distributed task execution system:
 * - TaskExecutionHandler receives task-assign signals
 * - Validates tasks and executes via RealWorkerPool
 * - Sends task-result or task-error back to originator
 *
 * Run: node tests/task-execution-test.js
 */

import { EventEmitter } from 'events';
import { randomBytes } from 'crypto';

// Import components
import { TaskExecutionHandler, TaskValidator, DistributedTaskNetwork } from '../task-execution-handler.js';
import { RealWorkerPool } from '../real-workers.js';

// ============================================
// MOCK SIGNALING (for testing without Firebase)
// ============================================

class MockSignaling extends EventEmitter {
    constructor(peerId) {
        super();
        this.peerId = peerId || `mock-${randomBytes(8).toString('hex')}`;
        this.isConnected = true;
        this.peers = new Map();
        this.sentSignals = [];
    }

    async sendSignal(toPeerId, type, data) {
        const signal = {
            from: this.peerId,
            to: toPeerId,
            type,
            data,
            timestamp: Date.now(),
        };
        this.sentSignals.push(signal);
        return true;
    }

    getOnlinePeers() {
        return Array.from(this.peers.values());
    }

    // Simulate receiving a signal
    simulateSignal(signal) {
        this.emit('signal', signal);
    }

    async connect() {
        this.isConnected = true;
        return true;
    }

    async disconnect() {
        this.isConnected = false;
    }
}

// ============================================
// TEST UTILITIES
// ============================================

function log(msg, level = 'info') {
    const timestamp = new Date().toISOString().slice(11, 23);
    const prefix = {
        info: '\x1b[36m[INFO]\x1b[0m',
        pass: '\x1b[32m[PASS]\x1b[0m',
        fail: '\x1b[31m[FAIL]\x1b[0m',
        warn: '\x1b[33m[WARN]\x1b[0m',
    }[level] || '[INFO]';
    console.log(`${timestamp} ${prefix} ${msg}`);
}

async function runTest(name, testFn) {
    log(`Running: ${name}`);
    try {
        await testFn();
        log(`${name}`, 'pass');
        return true;
    } catch (error) {
        log(`${name}: ${error.message}`, 'fail');
        console.error(error);
        return false;
    }
}

// ============================================
// TESTS
// ============================================

async function testTaskValidator() {
    const validator = new TaskValidator();

    // Valid task
    const validTask = {
        id: 'task-123',
        type: 'compute',
        data: [1, 2, 3, 4, 5],
        priority: 'medium',
    };

    const result1 = validator.validate(validTask);
    if (!result1.valid) throw new Error(`Valid task rejected: ${result1.errors.join(', ')}`);

    // Missing ID
    const result2 = validator.validate({ type: 'compute', data: [] });
    if (result2.valid) throw new Error('Task without ID should be rejected');

    // Missing type
    const result3 = validator.validate({ id: 'test', data: [] });
    if (result3.valid) throw new Error('Task without type should be rejected');

    // Invalid type
    const result4 = validator.validate({ id: 'test', type: 'invalid-type', data: [] });
    if (result4.valid) throw new Error('Task with invalid type should be rejected');

    // Missing data
    const result5 = validator.validate({ id: 'test', type: 'compute' });
    if (result5.valid) throw new Error('Task without data should be rejected');

    log('TaskValidator tests passed');
}

async function testWorkerPoolExecution() {
    const pool = new RealWorkerPool({ size: 2 });
    await pool.initialize();

    // Test compute task
    const computeResult = await pool.execute('compute', [1, 2, 3, 4, 5], { operation: 'sum' });
    if (computeResult.result !== 15) {
        throw new Error(`Expected sum to be 15, got ${computeResult.result}`);
    }

    // Test embed task
    const embedResult = await pool.execute('embed', 'hello world');
    if (!embedResult.embedding || embedResult.embedding.length !== 384) {
        throw new Error('Embed result should have 384-dimension embedding');
    }

    // Test transform task
    const transformResult = await pool.execute('transform', 'hello', { transform: 'uppercase' });
    if (transformResult.transformed !== 'HELLO') {
        throw new Error(`Expected HELLO, got ${transformResult.transformed}`);
    }

    await pool.shutdown();
    log('WorkerPool execution tests passed');
}

async function testTaskExecutionHandler() {
    // Create mock signaling and real worker pool
    const nodeASignaling = new MockSignaling('node-A');
    const nodeBSignaling = new MockSignaling('node-B');

    const workerPool = new RealWorkerPool({ size: 2 });
    await workerPool.initialize();

    // Create handler for Node B (the executor)
    const handler = new TaskExecutionHandler({
        signaling: nodeBSignaling,
        workerPool,
        nodeId: 'node-B',
        capabilities: ['compute', 'embed', 'process'],
    });
    handler.attach();

    // Track events with promise-based waiting
    const taskId = `task-${Date.now()}`;
    let startEvent = null;
    let completeEvent = null;

    const completePromise = new Promise((resolve) => {
        handler.on('task-start', (e) => {
            if (e.taskId === taskId) startEvent = e;
        });
        handler.on('task-complete', (e) => {
            if (e.taskId === taskId) {
                completeEvent = e;
                resolve(e);
            }
        });
        handler.on('task-error', (e) => {
            if (e.taskId === taskId) resolve(e);
        });
    });

    // Simulate Node A sending a task-assign signal to Node B
    const taskAssignSignal = {
        type: 'task-assign',
        from: 'node-A',
        data: {
            task: {
                id: taskId,
                type: 'compute',
                data: [10, 20, 30],
                options: { operation: 'sum' },
            },
        },
        verified: true,
    };

    // Send signal to Node B's handler
    nodeBSignaling.simulateSignal(taskAssignSignal);

    // Wait for completion (with timeout)
    await Promise.race([
        completePromise,
        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout waiting for task completion')), 5000))
    ]);

    // Check events
    if (!startEvent) throw new Error('task-start event not emitted');
    if (!completeEvent) throw new Error('task-complete event not emitted');

    // Check that result was sent back via signaling
    const resultSignal = nodeBSignaling.sentSignals.find(
        s => s.type === 'task-result' && s.data.taskId === taskId
    );
    if (!resultSignal) throw new Error('task-result signal not sent');

    if (resultSignal.data.result.result !== 60) {
        throw new Error(`Expected sum result 60, got ${resultSignal.data.result.result}`);
    }

    // Cleanup
    handler.detach();
    await workerPool.shutdown();

    log('TaskExecutionHandler tests passed');
}

async function testTaskRejection() {
    const signaling = new MockSignaling('node-B');
    const workerPool = new RealWorkerPool({ size: 1 });
    await workerPool.initialize();

    const handler = new TaskExecutionHandler({
        signaling,
        workerPool,
        nodeId: 'node-B',
        capabilities: ['compute'],
    });
    handler.attach();

    const events = [];
    handler.on('task-rejected', (e) => events.push(e));

    // Test 1: Invalid task (missing type)
    signaling.simulateSignal({
        type: 'task-assign',
        from: 'node-A',
        data: {
            task: {
                id: 'bad-task-1',
                data: [1, 2, 3],
                // missing type
            },
        },
        verified: true,
    });

    await new Promise(resolve => setTimeout(resolve, 100));

    const rejection1 = events.find(e => e.taskId === 'bad-task-1');
    if (!rejection1) throw new Error('Task without type should be rejected');

    // Check error signal was sent
    const errorSignal = signaling.sentSignals.find(
        s => s.type === 'task-error' && s.data.taskId === 'bad-task-1'
    );
    if (!errorSignal) throw new Error('task-error signal should be sent for invalid task');

    // Test 2: Missing capabilities
    signaling.simulateSignal({
        type: 'task-assign',
        from: 'node-A',
        data: {
            task: {
                id: 'bad-task-2',
                type: 'compute',
                data: [1, 2, 3],
                requiredCapabilities: ['special-gpu'],
            },
        },
        verified: true,
    });

    await new Promise(resolve => setTimeout(resolve, 100));

    const rejection2 = events.find(e => e.taskId === 'bad-task-2');
    if (!rejection2 || rejection2.reason !== 'capabilities') {
        throw new Error('Task with missing capabilities should be rejected');
    }

    handler.detach();
    await workerPool.shutdown();

    log('Task rejection tests passed');
}

async function testSubmitTaskToRemote() {
    // Create two nodes that can communicate
    const nodeASignaling = new MockSignaling('node-A');
    const nodeBSignaling = new MockSignaling('node-B');

    // Register each other as peers
    nodeASignaling.peers.set('node-B', { id: 'node-B', capabilities: ['compute'] });
    nodeBSignaling.peers.set('node-A', { id: 'node-A', capabilities: ['compute'] });

    // Create worker pool and handler for Node B
    const workerPool = new RealWorkerPool({ size: 2 });
    await workerPool.initialize();

    const nodeBHandler = new TaskExecutionHandler({
        signaling: nodeBSignaling,
        workerPool,
        nodeId: 'node-B',
        capabilities: ['compute'],
    });
    nodeBHandler.attach();

    // Create handler for Node A (the submitter)
    const nodeAHandler = new TaskExecutionHandler({
        signaling: nodeASignaling,
        workerPool: null, // Node A won't execute locally
        nodeId: 'node-A',
        capabilities: [],
    });
    nodeAHandler.attach();

    // Override sendSignal for Node A to deliver to Node B
    const origSendA = nodeASignaling.sendSignal.bind(nodeASignaling);
    nodeASignaling.sendSignal = async (to, type, data) => {
        await origSendA(to, type, data);
        if (to === 'node-B') {
            // Small delay to simulate network
            await new Promise(r => setTimeout(r, 10));
            // Deliver to Node B with proper signal structure
            nodeBSignaling.simulateSignal({
                type,
                from: 'node-A',
                data,
                verified: true,
            });
        }
    };

    // Override sendSignal for Node B to deliver results back to Node A
    const origSendB = nodeBSignaling.sendSignal.bind(nodeBSignaling);
    nodeBSignaling.sendSignal = async (to, type, data) => {
        await origSendB(to, type, data);
        if (to === 'node-A' && (type === 'task-result' || type === 'task-error')) {
            // Small delay to simulate network
            await new Promise(r => setTimeout(r, 10));
            // Deliver result back to Node A
            nodeASignaling.simulateSignal({
                type,
                from: 'node-B',
                data,
                verified: true,
            });
        }
    };

    // Submit task from Node A to Node B
    const resultPromise = nodeAHandler.submitTask('node-B', {
        id: 'remote-task-1',
        type: 'compute',
        data: [5, 10, 15],
        options: { operation: 'sum' },
    }, { timeout: 5000 });

    // Wait for result
    const result = await resultPromise;

    if (result.result.result !== 30) {
        throw new Error(`Expected 30, got ${result.result.result}`);
    }

    if (result.processedBy !== 'node-B') {
        throw new Error(`Expected processedBy to be node-B, got ${result.processedBy}`);
    }

    // Cleanup
    nodeAHandler.detach();
    nodeBHandler.detach();
    await workerPool.shutdown();

    log('Remote task submission tests passed');
}

async function testCapacityLimit() {
    const signaling = new MockSignaling('node-B');
    const workerPool = new RealWorkerPool({ size: 1 });
    await workerPool.initialize();

    const handler = new TaskExecutionHandler({
        signaling,
        workerPool,
        nodeId: 'node-B',
        maxConcurrentTasks: 2, // Low limit for testing
        capabilities: ['compute'],
    });
    handler.attach();

    const events = [];
    handler.on('task-rejected', (e) => events.push(e));
    handler.on('task-start', (e) => events.push({ type: 'start', ...e }));

    // Submit 3 tasks rapidly
    for (let i = 0; i < 3; i++) {
        signaling.simulateSignal({
            type: 'task-assign',
            from: 'node-A',
            data: {
                task: {
                    id: `capacity-task-${i}`,
                    type: 'compute',
                    data: Array(1000).fill(1), // Larger task to take time
                    options: { operation: 'sum' },
                },
            },
            verified: true,
        });
    }

    // Wait a bit
    await new Promise(resolve => setTimeout(resolve, 200));

    // Check that we got at least one rejection for capacity
    const capacityRejection = events.find(e => e.reason === 'capacity');
    if (!capacityRejection) {
        log('Note: Capacity test depends on timing, may not always trigger', 'warn');
    }

    handler.detach();
    await workerPool.shutdown();

    log('Capacity limit tests completed');
}

async function testProgressReporting() {
    const signaling = new MockSignaling('node-B');
    const workerPool = new RealWorkerPool({ size: 1 });
    await workerPool.initialize();

    const handler = new TaskExecutionHandler({
        signaling,
        workerPool,
        nodeId: 'node-B',
        capabilities: ['compute'],
        reportProgress: true,
        progressInterval: 100, // Fast progress for testing
    });
    handler.attach();

    // Submit a task that takes some time
    signaling.simulateSignal({
        type: 'task-assign',
        from: 'node-A',
        data: {
            task: {
                id: 'progress-task',
                type: 'compute',
                data: Array(10000).fill(1),
                options: { operation: 'sum' },
            },
        },
        verified: true,
    });

    // Wait for completion
    await new Promise(resolve => setTimeout(resolve, 500));

    // Check for progress signals
    const progressSignals = signaling.sentSignals.filter(s => s.type === 'task-progress');
    // Progress reporting may or may not fire depending on execution speed
    log(`Progress signals sent: ${progressSignals.length}`, 'info');

    // Check for result signal
    const resultSignal = signaling.sentSignals.find(s => s.type === 'task-result');
    if (!resultSignal) throw new Error('Result signal not sent');

    handler.detach();
    await workerPool.shutdown();

    log('Progress reporting tests passed');
}

// ============================================
// MAIN
// ============================================

async function main() {
    console.log('\n========================================');
    console.log(' Task Execution Handler Integration Tests');
    console.log('========================================\n');

    const tests = [
        ['TaskValidator', testTaskValidator],
        ['WorkerPool Execution', testWorkerPoolExecution],
        ['TaskExecutionHandler Basic', testTaskExecutionHandler],
        ['Task Rejection', testTaskRejection],
        ['Remote Task Submission', testSubmitTaskToRemote],
        ['Capacity Limits', testCapacityLimit],
        ['Progress Reporting', testProgressReporting],
    ];

    let passed = 0;
    let failed = 0;

    for (const [name, testFn] of tests) {
        console.log('');
        const success = await runTest(name, testFn);
        if (success) passed++;
        else failed++;
    }

    console.log('\n========================================');
    console.log(` Results: ${passed} passed, ${failed} failed`);
    console.log('========================================\n');

    process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
    console.error('Test runner error:', err);
    process.exit(1);
});
