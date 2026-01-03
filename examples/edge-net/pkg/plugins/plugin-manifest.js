/**
 * Edge-Net Plugin Manifest & Catalog System
 *
 * Secure, fast, private plugin architecture for optional features.
 *
 * Features:
 * - Cryptographic verification (Ed25519 signatures)
 * - Lazy loading (load on demand)
 * - Sandboxed execution (WASM isolation)
 * - Zero telemetry (local-first)
 * - Capability-based permissions
 *
 * @module @ruvector/edge-net/plugins
 */

// ============================================
// PLUGIN CATEGORIES
// ============================================

export const PluginCategory = {
    CORE: 'core',           // Essential functionality
    NETWORK: 'network',     // Network topology & routing
    CRYPTO: 'crypto',       // Cryptographic features
    PRIVACY: 'privacy',     // Privacy-enhancing tech
    AI: 'ai',               // AI/ML capabilities
    ECONOMIC: 'economic',   // Incentive mechanisms
    STORAGE: 'storage',     // Data persistence
    EXOTIC: 'exotic',       // Experimental features
};

export const PluginTier = {
    STABLE: 'stable',       // Production-ready
    BETA: 'beta',           // Testing phase
    EXPERIMENTAL: 'experimental', // Use at your own risk
    RESEARCH: 'research',   // Academic/research only
};

// ============================================
// CAPABILITY PERMISSIONS
// ============================================

export const Capability = {
    // Network
    NETWORK_CONNECT: 'network:connect',
    NETWORK_LISTEN: 'network:listen',
    NETWORK_RELAY: 'network:relay',

    // Crypto
    CRYPTO_SIGN: 'crypto:sign',
    CRYPTO_ENCRYPT: 'crypto:encrypt',
    CRYPTO_KEYGEN: 'crypto:keygen',

    // Storage
    STORAGE_READ: 'storage:read',
    STORAGE_WRITE: 'storage:write',
    STORAGE_DELETE: 'storage:delete',

    // Compute
    COMPUTE_CPU: 'compute:cpu',
    COMPUTE_GPU: 'compute:gpu',
    COMPUTE_WASM: 'compute:wasm',

    // System
    SYSTEM_ENV: 'system:env',
    SYSTEM_FS: 'system:fs',
    SYSTEM_EXEC: 'system:exec',
};

// ============================================
// PLUGIN MANIFEST SCHEMA
// ============================================

/**
 * Plugin manifest schema
 * @typedef {Object} PluginManifest
 */
export const PluginManifestSchema = {
    // Required fields
    id: 'string',           // Unique identifier (e.g., "privacy.zk-proofs")
    name: 'string',         // Human-readable name
    version: 'string',      // Semver version
    description: 'string',  // Brief description

    // Classification
    category: 'PluginCategory',
    tier: 'PluginTier',

    // Security
    author: 'string',
    signature: 'string',    // Ed25519 signature of manifest
    checksum: 'string',     // SHA-256 of plugin code
    capabilities: 'Capability[]', // Required permissions

    // Dependencies
    dependencies: 'string[]', // Other plugin IDs
    peerDependencies: 'string[]',
    conflicts: 'string[]',  // Incompatible plugins

    // Loading
    entryPoint: 'string',   // Main module path
    lazyLoad: 'boolean',    // Load on first use
    singleton: 'boolean',   // Single instance only

    // Configuration
    configSchema: 'object', // JSON Schema for config
    defaultConfig: 'object',

    // Metadata
    repository: 'string',
    documentation: 'string',
    license: 'string',
    tags: 'string[]',
};

// ============================================
// OFFICIAL PLUGIN CATALOG
// ============================================

export const PLUGIN_CATALOG = {
    // ========================================
    // TIER 1: PRACTICAL / STABLE
    // ========================================

    'core.webrtc-enhanced': {
        id: 'core.webrtc-enhanced',
        name: 'Enhanced WebRTC',
        version: '1.0.0',
        description: 'Optimized WebRTC with adaptive bitrate, simulcast, and ICE restart',
        category: PluginCategory.CORE,
        tier: PluginTier.STABLE,
        capabilities: [Capability.NETWORK_CONNECT, Capability.NETWORK_RELAY],
        lazyLoad: false,
        configSchema: {
            type: 'object',
            properties: {
                enableSimulcast: { type: 'boolean', default: true },
                adaptiveBitrate: { type: 'boolean', default: true },
                iceRestartThreshold: { type: 'number', default: 3 },
            },
        },
        tags: ['networking', 'p2p', 'video'],
    },

    'core.compression': {
        id: 'core.compression',
        name: 'Data Compression',
        version: '1.0.0',
        description: 'LZ4/Zstd compression for network payloads and storage',
        category: PluginCategory.CORE,
        tier: PluginTier.STABLE,
        capabilities: [Capability.COMPUTE_WASM],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                algorithm: { type: 'string', enum: ['lz4', 'zstd', 'brotli'], default: 'lz4' },
                level: { type: 'number', minimum: 1, maximum: 22, default: 3 },
                threshold: { type: 'number', default: 1024 }, // Min bytes to compress
            },
        },
        tags: ['performance', 'compression'],
    },

    'network.multi-transport': {
        id: 'network.multi-transport',
        name: 'Multi-Transport Layer',
        version: '1.0.0',
        description: 'WebSocket, WebRTC, HTTP/3 QUIC transport with automatic failover',
        category: PluginCategory.NETWORK,
        tier: PluginTier.STABLE,
        capabilities: [Capability.NETWORK_CONNECT, Capability.NETWORK_LISTEN],
        lazyLoad: false,
        configSchema: {
            type: 'object',
            properties: {
                preferredTransport: { type: 'string', enum: ['webrtc', 'websocket', 'quic'], default: 'webrtc' },
                fallbackOrder: { type: 'array', default: ['webrtc', 'websocket', 'quic'] },
                timeoutMs: { type: 'number', default: 5000 },
            },
        },
        tags: ['networking', 'reliability', 'transport'],
    },

    'storage.indexed-db': {
        id: 'storage.indexed-db',
        name: 'IndexedDB Persistence',
        version: '1.0.0',
        description: 'Browser-native storage with encryption at rest',
        category: PluginCategory.STORAGE,
        tier: PluginTier.STABLE,
        capabilities: [Capability.STORAGE_READ, Capability.STORAGE_WRITE, Capability.CRYPTO_ENCRYPT],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                dbName: { type: 'string', default: 'edge-net' },
                encryptionKey: { type: 'string' }, // Optional, derived from PiKey if not set
                maxSizeMB: { type: 'number', default: 100 },
            },
        },
        tags: ['storage', 'persistence', 'encryption'],
    },

    // ========================================
    // TIER 2: PRIVACY / BETA
    // ========================================

    'privacy.e2e-encryption': {
        id: 'privacy.e2e-encryption',
        name: 'End-to-End Encryption',
        version: '1.0.0',
        description: 'X25519 key exchange + ChaCha20-Poly1305 encryption for all messages',
        category: PluginCategory.PRIVACY,
        tier: PluginTier.STABLE,
        capabilities: [Capability.CRYPTO_ENCRYPT, Capability.CRYPTO_KEYGEN],
        lazyLoad: false,
        configSchema: {
            type: 'object',
            properties: {
                keyRotationInterval: { type: 'number', default: 3600000 }, // 1 hour
                forwardSecrecy: { type: 'boolean', default: true },
            },
        },
        tags: ['privacy', 'encryption', 'security'],
    },

    'privacy.onion-routing': {
        id: 'privacy.onion-routing',
        name: 'Onion Routing',
        version: '0.9.0',
        description: 'Tor-style multi-hop routing for anonymity',
        category: PluginCategory.PRIVACY,
        tier: PluginTier.BETA,
        capabilities: [Capability.NETWORK_RELAY, Capability.CRYPTO_ENCRYPT],
        dependencies: ['privacy.e2e-encryption'],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                hopCount: { type: 'number', minimum: 2, maximum: 7, default: 3 },
                circuitLifetime: { type: 'number', default: 600000 }, // 10 minutes
                guardNodes: { type: 'array', default: [] }, // Trusted entry nodes
            },
        },
        tags: ['privacy', 'anonymity', 'tor'],
    },

    'privacy.mixnet': {
        id: 'privacy.mixnet',
        name: 'Mixnet Shuffling',
        version: '0.8.0',
        description: 'Traffic analysis resistance via message batching and shuffling',
        category: PluginCategory.PRIVACY,
        tier: PluginTier.BETA,
        capabilities: [Capability.NETWORK_RELAY, Capability.CRYPTO_ENCRYPT],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                batchSize: { type: 'number', default: 10 },
                batchTimeoutMs: { type: 'number', default: 1000 },
                dummyTraffic: { type: 'boolean', default: true },
            },
        },
        tags: ['privacy', 'anonymity', 'traffic-analysis'],
    },

    // ========================================
    // TIER 3: CRYPTO / EXPERIMENTAL
    // ========================================

    'crypto.zk-proofs': {
        id: 'crypto.zk-proofs',
        name: 'Zero-Knowledge Proofs',
        version: '0.5.0',
        description: 'ZK-SNARKs for verifiable computation without revealing inputs',
        category: PluginCategory.CRYPTO,
        tier: PluginTier.EXPERIMENTAL,
        capabilities: [Capability.COMPUTE_WASM, Capability.CRYPTO_SIGN],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                provingSystem: { type: 'string', enum: ['groth16', 'plonk', 'stark'], default: 'groth16' },
                trustedSetup: { type: 'string' }, // Path to trusted setup params
            },
        },
        tags: ['crypto', 'zk', 'verification'],
    },

    'crypto.homomorphic': {
        id: 'crypto.homomorphic',
        name: 'Homomorphic Encryption',
        version: '0.3.0',
        description: 'TFHE/BFV schemes for computing on encrypted data',
        category: PluginCategory.CRYPTO,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.COMPUTE_WASM, Capability.CRYPTO_ENCRYPT],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                scheme: { type: 'string', enum: ['tfhe', 'bfv', 'ckks'], default: 'tfhe' },
                securityLevel: { type: 'number', enum: [128, 192, 256], default: 128 },
            },
        },
        tags: ['crypto', 'fhe', 'privacy'],
    },

    'crypto.mpc': {
        id: 'crypto.mpc',
        name: 'Multi-Party Computation',
        version: '0.4.0',
        description: 'Secret sharing and secure computation across multiple nodes',
        category: PluginCategory.CRYPTO,
        tier: PluginTier.EXPERIMENTAL,
        capabilities: [Capability.NETWORK_CONNECT, Capability.CRYPTO_ENCRYPT],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                protocol: { type: 'string', enum: ['shamir', 'bgw', 'spdz'], default: 'shamir' },
                threshold: { type: 'number', default: 2 }, // k-of-n threshold
                partyCount: { type: 'number', default: 3 },
            },
        },
        tags: ['crypto', 'mpc', 'privacy'],
    },

    'crypto.threshold-sig': {
        id: 'crypto.threshold-sig',
        name: 'Threshold Signatures',
        version: '0.6.0',
        description: 'N-of-M distributed signing for high-value operations',
        category: PluginCategory.CRYPTO,
        tier: PluginTier.BETA,
        capabilities: [Capability.CRYPTO_SIGN, Capability.CRYPTO_KEYGEN],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                scheme: { type: 'string', enum: ['frost', 'gg20', 'cggmp'], default: 'frost' },
                threshold: { type: 'number', default: 2 },
                parties: { type: 'number', default: 3 },
            },
        },
        tags: ['crypto', 'signatures', 'distributed'],
    },

    // ========================================
    // TIER 4: AI/ML
    // ========================================

    'ai.federated-learning': {
        id: 'ai.federated-learning',
        name: 'Federated Learning',
        version: '0.7.0',
        description: 'Train ML models across nodes without sharing raw data',
        category: PluginCategory.AI,
        tier: PluginTier.BETA,
        capabilities: [Capability.COMPUTE_WASM, Capability.NETWORK_CONNECT],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                aggregationStrategy: { type: 'string', enum: ['fedavg', 'fedprox', 'scaffold'], default: 'fedavg' },
                localEpochs: { type: 'number', default: 5 },
                differentialPrivacy: { type: 'boolean', default: true },
                noiseMultiplier: { type: 'number', default: 1.0 },
            },
        },
        tags: ['ai', 'ml', 'privacy', 'distributed'],
    },

    'ai.model-sharding': {
        id: 'ai.model-sharding',
        name: 'Model Sharding',
        version: '0.5.0',
        description: 'Split large LLMs across multiple nodes for distributed inference',
        category: PluginCategory.AI,
        tier: PluginTier.EXPERIMENTAL,
        capabilities: [Capability.COMPUTE_WASM, Capability.COMPUTE_GPU, Capability.NETWORK_CONNECT],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                shardingStrategy: { type: 'string', enum: ['layer', 'tensor', 'pipeline'], default: 'pipeline' },
                minNodes: { type: 'number', default: 2 },
                cacheKV: { type: 'boolean', default: true },
            },
        },
        tags: ['ai', 'llm', 'distributed', 'inference'],
    },

    'ai.swarm-intelligence': {
        id: 'ai.swarm-intelligence',
        name: 'Swarm Intelligence',
        version: '0.6.0',
        description: 'Ant colony, particle swarm, and genetic algorithms for optimization',
        category: PluginCategory.AI,
        tier: PluginTier.BETA,
        capabilities: [Capability.COMPUTE_CPU, Capability.NETWORK_CONNECT],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                algorithm: { type: 'string', enum: ['aco', 'pso', 'ga', 'de'], default: 'pso' },
                populationSize: { type: 'number', default: 50 },
                iterations: { type: 'number', default: 100 },
            },
        },
        tags: ['ai', 'optimization', 'swarm'],
    },

    // ========================================
    // TIER 5: ECONOMIC
    // ========================================

    'economic.prediction-markets': {
        id: 'economic.prediction-markets',
        name: 'Prediction Markets',
        version: '0.4.0',
        description: 'Bet on task completion, quality, and timing',
        category: PluginCategory.ECONOMIC,
        tier: PluginTier.EXPERIMENTAL,
        capabilities: [Capability.STORAGE_WRITE, Capability.CRYPTO_SIGN],
        dependencies: ['core.credits'],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                marketMaker: { type: 'string', enum: ['cpmm', 'lmsr'], default: 'cpmm' },
                minStake: { type: 'number', default: 1 },
                resolutionPeriod: { type: 'number', default: 86400000 }, // 24 hours
            },
        },
        tags: ['economic', 'prediction', 'incentives'],
    },

    'economic.reputation-staking': {
        id: 'economic.reputation-staking',
        name: 'Reputation Staking',
        version: '0.5.0',
        description: 'Stake credits as collateral, slashed for misbehavior',
        category: PluginCategory.ECONOMIC,
        tier: PluginTier.BETA,
        capabilities: [Capability.STORAGE_WRITE, Capability.CRYPTO_SIGN],
        dependencies: ['core.credits'],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                minStake: { type: 'number', default: 10 },
                slashRate: { type: 'number', default: 0.1 }, // 10% slash
                unbondingPeriod: { type: 'number', default: 604800000 }, // 7 days
            },
        },
        tags: ['economic', 'reputation', 'staking'],
    },

    'economic.compute-amm': {
        id: 'economic.compute-amm',
        name: 'Compute AMM',
        version: '0.3.0',
        description: 'Automated market maker for compute resource pricing',
        category: PluginCategory.ECONOMIC,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.STORAGE_WRITE, Capability.CRYPTO_SIGN],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                curveType: { type: 'string', enum: ['constant-product', 'stable', 'concentrated'], default: 'constant-product' },
                feeRate: { type: 'number', default: 0.003 }, // 0.3%
            },
        },
        tags: ['economic', 'amm', 'defi'],
    },

    // ========================================
    // TIER 6: EXOTIC / RESEARCH
    // ========================================

    'exotic.proof-of-useful-work': {
        id: 'exotic.proof-of-useful-work',
        name: 'Proof of Useful Work',
        version: '0.2.0',
        description: 'Consensus where mining = real computation tasks',
        category: PluginCategory.EXOTIC,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.COMPUTE_CPU, Capability.COMPUTE_WASM, Capability.CRYPTO_SIGN],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                difficultyAdjustment: { type: 'number', default: 2016 }, // blocks
                blockTime: { type: 'number', default: 60000 }, // 1 minute target
                taskVerification: { type: 'string', enum: ['probabilistic', 'deterministic', 'zk'], default: 'probabilistic' },
            },
        },
        tags: ['exotic', 'consensus', 'mining'],
    },

    'exotic.vdf': {
        id: 'exotic.vdf',
        name: 'Verifiable Delay Functions',
        version: '0.3.0',
        description: 'Time-lock puzzles for randomness and fair ordering',
        category: PluginCategory.EXOTIC,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.COMPUTE_WASM],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                construction: { type: 'string', enum: ['wesolowski', 'pietrzak'], default: 'wesolowski' },
                securityParameter: { type: 'number', default: 2048 },
                targetDelay: { type: 'number', default: 10000 }, // 10 seconds
            },
        },
        tags: ['exotic', 'vdf', 'randomness'],
    },

    'exotic.tee-enclave': {
        id: 'exotic.tee-enclave',
        name: 'TEE Enclaves',
        version: '0.2.0',
        description: 'Intel SGX / ARM TrustZone hardware-protected execution',
        category: PluginCategory.EXOTIC,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.COMPUTE_CPU, Capability.SYSTEM_EXEC],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                platform: { type: 'string', enum: ['sgx', 'trustzone', 'sev'], default: 'sgx' },
                attestation: { type: 'boolean', default: true },
            },
        },
        tags: ['exotic', 'tee', 'hardware', 'security'],
    },

    'exotic.quantum-sim': {
        id: 'exotic.quantum-sim',
        name: 'Quantum Circuit Simulator',
        version: '0.1.0',
        description: 'Simulate quantum algorithms on classical hardware',
        category: PluginCategory.EXOTIC,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.COMPUTE_WASM, Capability.COMPUTE_GPU],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                maxQubits: { type: 'number', default: 20 },
                backend: { type: 'string', enum: ['statevector', 'density', 'mps'], default: 'statevector' },
                noise: { type: 'boolean', default: false },
            },
        },
        tags: ['exotic', 'quantum', 'simulation'],
    },

    'exotic.cellular-automata': {
        id: 'exotic.cellular-automata',
        name: 'Cellular Automata',
        version: '0.2.0',
        description: 'Emergent computation via rule-based cell evolution',
        category: PluginCategory.EXOTIC,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.COMPUTE_WASM],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                rule: { type: 'number', default: 110 }, // Rule 110 is Turing complete
                dimensions: { type: 'number', enum: [1, 2, 3], default: 2 },
                gridSize: { type: 'number', default: 256 },
            },
        },
        tags: ['exotic', 'automata', 'emergent'],
    },

    'network.satellite': {
        id: 'network.satellite',
        name: 'Satellite Connectivity',
        version: '0.2.0',
        description: 'Starlink/Iridium fallback for remote or censored regions',
        category: PluginCategory.NETWORK,
        tier: PluginTier.EXPERIMENTAL,
        capabilities: [Capability.NETWORK_CONNECT, Capability.SYSTEM_EXEC],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                provider: { type: 'string', enum: ['starlink', 'iridium', 'viasat'], default: 'starlink' },
                fallbackOnly: { type: 'boolean', default: true },
                maxLatencyMs: { type: 'number', default: 1000 },
            },
        },
        tags: ['network', 'satellite', 'resilience'],
    },

    'network.lora-mesh': {
        id: 'network.lora-mesh',
        name: 'LoRa Mesh Network',
        version: '0.1.0',
        description: 'Low-power radio mesh for IoT and off-grid deployments',
        category: PluginCategory.NETWORK,
        tier: PluginTier.RESEARCH,
        capabilities: [Capability.NETWORK_CONNECT, Capability.SYSTEM_EXEC],
        lazyLoad: true,
        configSchema: {
            type: 'object',
            properties: {
                frequency: { type: 'number', default: 915 }, // MHz (US)
                spreadingFactor: { type: 'number', enum: [7, 8, 9, 10, 11, 12], default: 10 },
                maxHops: { type: 'number', default: 5 },
            },
        },
        tags: ['network', 'lora', 'iot', 'mesh'],
    },
};

// ============================================
// PLUGIN BUNDLES (Pre-configured sets)
// ============================================

export const PLUGIN_BUNDLES = {
    'minimal': {
        name: 'Minimal',
        description: 'Core functionality only',
        plugins: [],
    },

    'standard': {
        name: 'Standard',
        description: 'Recommended for most users',
        plugins: [
            'core.compression',
            'network.multi-transport',
            'privacy.e2e-encryption',
            'storage.indexed-db',
        ],
    },

    'privacy-focused': {
        name: 'Privacy Focused',
        description: 'Maximum privacy and anonymity',
        plugins: [
            'privacy.e2e-encryption',
            'privacy.onion-routing',
            'privacy.mixnet',
            'crypto.zk-proofs',
        ],
    },

    'ai-compute': {
        name: 'AI Compute',
        description: 'Optimized for AI/ML workloads',
        plugins: [
            'core.compression',
            'ai.federated-learning',
            'ai.model-sharding',
            'ai.swarm-intelligence',
        ],
    },

    'resilient': {
        name: 'Resilient',
        description: 'Maximum uptime and connectivity',
        plugins: [
            'network.multi-transport',
            'network.satellite',
            'storage.indexed-db',
            'core.compression',
        ],
    },

    'economic': {
        name: 'Economic',
        description: 'Full incentive mechanisms',
        plugins: [
            'economic.prediction-markets',
            'economic.reputation-staking',
            'economic.compute-amm',
        ],
    },

    'experimental': {
        name: 'Experimental',
        description: 'Bleeding edge features (unstable)',
        plugins: [
            'crypto.homomorphic',
            'crypto.mpc',
            'exotic.proof-of-useful-work',
            'exotic.vdf',
            'exotic.quantum-sim',
        ],
    },
};

export default PLUGIN_CATALOG;
