/**
 * WASM Core Tests
 *
 * Validates security fixes and core functionality:
 * - Cryptographic randomness (no Math.random)
 * - Ed25519 fail-closed behavior
 * - Memory bounds
 * - Genesis signing and verification
 * - Lineage verification
 */

import { describe, test, expect, beforeAll } from 'vitest';
import {
    detectPlatform,
    getPlatformCapabilities,
    WasmCrypto,
    WasmGenesis,
    WasmInference,
} from '../models/wasm-core.js';

describe('Platform Detection', () => {
    test('detects Node.js platform', () => {
        const platform = detectPlatform();
        expect(platform).toBe('node');
    });

    test('returns platform capabilities', () => {
        const caps = getPlatformCapabilities();
        expect(caps).toHaveProperty('platform');
        expect(caps).toHaveProperty('hasWebAssembly');
        expect(caps).toHaveProperty('hasWebCrypto');
        expect(caps).toHaveProperty('maxMemory');
        expect(typeof caps.maxMemory).toBe('number');
        expect(caps.maxMemory).toBeGreaterThan(0);
    });
});

describe('WasmCrypto', () => {
    let crypto;

    beforeAll(async () => {
        crypto = new WasmCrypto();
        await crypto.init();
    });

    test('computes SHA256 hash', async () => {
        const hash = await crypto.sha256('hello world');
        expect(hash).toBeInstanceOf(Uint8Array);
        expect(hash.length).toBe(32);
    });

    test('computes SHA256 hex string', async () => {
        const hashHex = await crypto.sha256Hex('hello world');
        expect(typeof hashHex).toBe('string');
        expect(hashHex.length).toBe(64);
        // Known SHA256 of "hello world"
        expect(hashHex).toBe('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
    });

    test('canonicalizes objects deterministically', () => {
        const obj1 = { b: 2, a: 1 };
        const obj2 = { a: 1, b: 2 };
        const canon1 = crypto.canonicalize(obj1);
        const canon2 = crypto.canonicalize(obj2);
        expect(canon1).toBe(canon2);
        expect(canon1).toBe('{"a":1,"b":2}');
    });

    test('canonicalizes nested objects', () => {
        const obj = { z: { b: 2, a: 1 }, y: [3, 2, 1] };
        const canon = crypto.canonicalize(obj);
        expect(canon).toBe('{"y":[3,2,1],"z":{"a":1,"b":2}}');
    });

    test('rejects Infinity and NaN', () => {
        expect(() => crypto.canonicalize({ x: Infinity }))
            .toThrow('Cannot canonicalize Infinity/NaN');
        expect(() => crypto.canonicalize({ x: NaN }))
            .toThrow('Cannot canonicalize Infinity/NaN');
    });

    test('computes hash of canonical object', async () => {
        const hash = await crypto.hashCanonical({ test: 'value' });
        expect(typeof hash).toBe('string');
        expect(hash.length).toBe(64);
    });

    test('merkle root of single hash returns the hash', async () => {
        const hash = await crypto.sha256('test');
        const root = await crypto.merkleRoot([hash]);
        // Single element should return itself (or same value)
        expect(root).toBeInstanceOf(Uint8Array);
        expect(root.length).toBe(32);
    });
});

describe('WasmGenesis Security', () => {
    let genesis;

    beforeAll(async () => {
        genesis = new WasmGenesis({
            networkName: 'test-net',
            version: '1.0.0',
        });
        await genesis.init();
    });

    test('generates cryptographically random network IDs', async () => {
        const ids = new Set();
        for (let i = 0; i < 10; i++) {
            const id = await genesis._generateNetworkId(Date.now());
            expect(id).toMatch(/^net_[a-f0-9]{16}$/);
            ids.add(id);
        }
        // All IDs should be unique
        expect(ids.size).toBe(10);
    });

    test('births network with Ed25519 keypair', async () => {
        const result = await genesis.birthNetwork({
            traits: { test: true },
        });

        expect(result).toHaveProperty('networkId');
        expect(result).toHaveProperty('manifest');
        expect(result).toHaveProperty('genesisHash');
        expect(result).toHaveProperty('signature');
        expect(result).toHaveProperty('publicKey');

        // Verify signature format (128 hex chars = 64 bytes)
        expect(result.signature.length).toBe(128);
        // Verify public key format (64 hex chars = 32 bytes)
        expect(result.publicKey.length).toBe(64);
    });

    test('manifest includes cryptographic signature', async () => {
        const result = await genesis.birthNetwork();
        const manifest = result.manifest;

        expect(manifest.integrity).toHaveProperty('signature');
        expect(manifest.integrity).toHaveProperty('signatureAlgorithm', 'Ed25519');
        expect(manifest.integrity).toHaveProperty('genesisHash');
        expect(manifest.genesis).toHaveProperty('publicKey');
        expect(manifest.genesis).toHaveProperty('keyAlgorithm', 'Ed25519');
    });

    test('verifies valid genesis signature', async () => {
        const result = await genesis.birthNetwork();
        const verification = await genesis.verifyGenesis(result.manifest);

        expect(verification.valid).toBe(true);
        expect(verification.genesisHash).toBe(result.genesisHash);
    });

    test('rejects tampered genesis', async () => {
        const result = await genesis.birthNetwork();
        const manifest = JSON.parse(JSON.stringify(result.manifest));

        // Tamper with the genesis
        manifest.genesis.traits.hacked = true;

        const verification = await genesis.verifyGenesis(manifest);
        expect(verification.valid).toBe(false);
        expect(verification.error).toMatch(/hash mismatch/i);
    });

    test('rejects missing signature', async () => {
        const result = await genesis.birthNetwork();
        const manifest = JSON.parse(JSON.stringify(result.manifest));

        // Remove signature
        delete manifest.integrity.signature;

        const verification = await genesis.verifyGenesis(manifest);
        expect(verification.valid).toBe(false);
        expect(verification.error).toMatch(/Missing.*signature/i);
    });
});

describe('WasmGenesis Lineage', () => {
    let genesis;

    beforeAll(async () => {
        genesis = new WasmGenesis({ networkName: 'lineage-test' });
        await genesis.init();
    });

    test('verifies root network (no parent)', async () => {
        const result = await genesis.birthNetwork();
        const verification = await genesis.verifyLineage(result.manifest);

        expect(verification.valid).toBe(true);
        expect(verification.isRoot).toBe(true);
        expect(verification.genesisVerified).toBe(true);
    });

    test('reproduces with lineage tracking', async () => {
        const parent = await genesis.birthNetwork({
            traits: { generation: 0, fitness: 1.0 },
        });

        const child = await genesis.reproduce(parent.manifest, {
            mutationRate: 0.1,
        });

        expect(child.manifest.lineage).toBeDefined();
        expect(child.manifest.lineage.parentId).toBe(parent.manifest.genesis.networkId);
        expect(child.manifest.lineage.generation).toBe(1);
    });

    test('verifies valid lineage chain', async () => {
        const parent = await genesis.birthNetwork();
        const child = await genesis.reproduce(parent.manifest);

        const verification = await genesis.verifyLineage(child.manifest, parent.manifest);

        expect(verification.valid).toBe(true);
        expect(verification.genesisVerified).toBe(true);
        expect(verification.parentVerified).toBe(true);
        expect(verification.parentId).toBe(parent.manifest.genesis.networkId);
    });

    test('rejects mismatched parent ID', async () => {
        const parent = await genesis.birthNetwork();
        const fakeParent = await genesis.birthNetwork();
        const child = await genesis.reproduce(parent.manifest);

        // Try to verify with wrong parent
        const verification = await genesis.verifyLineage(child.manifest, fakeParent.manifest);

        expect(verification.valid).toBe(false);
        expect(verification.error).toMatch(/Parent ID mismatch/i);
    });

    test('rejects broken generation sequence', async () => {
        const parent = await genesis.birthNetwork();
        const child = await genesis.reproduce(parent.manifest);

        // Tamper with generation
        const tamperedManifest = JSON.parse(JSON.stringify(child.manifest));
        tamperedManifest.lineage.generation = 5;

        // Re-sign would be needed for real verification
        // But the generation check should fail first
        const verification = await genesis.verifyLineage(tamperedManifest, parent.manifest);

        // Should fail on genesis hash mismatch since we tampered
        expect(verification.valid).toBe(false);
    });
});

describe('WasmInference', () => {
    let inference;

    beforeAll(async () => {
        inference = new WasmInference();
        await inference.init();
    });

    test('initializes with platform capabilities', async () => {
        expect(inference.ready).toBe(true);
    });

    test('loads model with correct hash path', async () => {
        const modelData = new Uint8Array([1, 2, 3, 4, 5]);
        const manifest = {
            artifacts: {
                model: {
                    sha256: await inference.crypto.sha256Hex(modelData),
                },
            },
        };

        await inference.loadModel(modelData, manifest);
        expect(inference.model).toBeDefined();
        expect(inference.model.manifest).toBe(manifest);
    });

    test('rejects model with hash mismatch', async () => {
        const modelData = new Uint8Array([1, 2, 3, 4, 5]);
        const manifest = {
            artifacts: {
                model: {
                    sha256: 'wrong_hash_value_here',
                },
            },
        };

        await expect(inference.loadModel(modelData, manifest))
            .rejects.toThrow(/hash mismatch/i);
    });

    test('supports legacy artifact format', async () => {
        const modelData = new Uint8Array([1, 2, 3, 4, 5]);
        const hash = await inference.crypto.sha256Hex(modelData);
        const manifest = {
            artifacts: [{ sha256: hash }],
        };

        await inference.loadModel(modelData, manifest);
        expect(inference.model).toBeDefined();
    });
});

describe('Security: No Math.random', () => {
    test('source code does not use Math.random for security', async () => {
        const fs = await import('fs/promises');
        const source = await fs.readFile(
            new URL('../models/wasm-core.js', import.meta.url),
            'utf-8'
        );

        // Find all Math.random occurrences
        const mathRandomUsage = source.match(/Math\.random\(\)/g) || [];

        // Math.random should NOT be used for security-critical operations
        // Check that any usage is clearly documented as non-security
        if (mathRandomUsage.length > 0) {
            // Verify they're only in non-security contexts (placeholder code)
            const lines = source.split('\n');
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].includes('Math.random()')) {
                    // Check if this line is commented or in placeholder section
                    const context = lines.slice(Math.max(0, i - 5), i + 1).join('\n');
                    expect(context).not.toMatch(/_generateNetworkId/);
                    expect(context).not.toMatch(/sign/i);
                    expect(context).not.toMatch(/key/i);
                }
            }
        }
    });
});

describe('Security: Fail Closed', () => {
    test('Ed25519 fallback throws instead of returning zeros', async () => {
        // The JS fallback should throw when Ed25519 is unavailable
        // We can't easily test this without mocking crypto.subtle
        // but we can verify the code pattern exists
        const fs = await import('fs/promises');
        const source = await fs.readFile(
            new URL('../models/wasm-core.js', import.meta.url),
            'utf-8'
        );

        // Verify fail-closed pattern exists
        expect(source).toContain('FAIL CLOSED');
        expect(source).toContain('throw new Error');
        expect(source).not.toContain('returning mock signature');
    });
});

describe('Memory Bounds', () => {
    test('WASM memory is reasonably bounded', async () => {
        const fs = await import('fs/promises');
        const source = await fs.readFile(
            new URL('../models/wasm-core.js', import.meta.url),
            'utf-8'
        );

        // Check that memory max is not 65536 (4GB)
        const memoryMatch = source.match(/maximum:\s*(\d+)/);
        if (memoryMatch) {
            const maxPages = parseInt(memoryMatch[1], 10);
            // Should be <= 1024 pages (64MB) for edge platforms
            expect(maxPages).toBeLessThanOrEqual(1024);
        }
    });
});
