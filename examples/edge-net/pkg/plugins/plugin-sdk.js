/**
 * Edge-Net Plugin SDK
 *
 * Create custom plugins for the edge-net ecosystem.
 * Provides base classes, utilities, and validation.
 *
 * @module @ruvector/edge-net/plugins/sdk
 */

import { EventEmitter } from 'events';
import { createHash, randomBytes } from 'crypto';
import { Capability, PluginCategory, PluginTier } from './plugin-manifest.js';

// Re-export for plugin authors
export { Capability, PluginCategory, PluginTier };

// ============================================
// BASE PLUGIN CLASS
// ============================================

/**
 * Base class for all custom plugins.
 * Extend this to create your own plugins.
 *
 * @example
 * ```javascript
 * import { BasePlugin, Capability, PluginCategory, PluginTier } from '@ruvector/edge-net/plugins/sdk';
 *
 * export class MyPlugin extends BasePlugin {
 *     static manifest = {
 *         id: 'my-org.my-plugin',
 *         name: 'My Custom Plugin',
 *         version: '1.0.0',
 *         description: 'Does something awesome',
 *         category: PluginCategory.CORE,
 *         tier: PluginTier.BETA,
 *         capabilities: [Capability.COMPUTE_WASM],
 *         configSchema: {
 *             type: 'object',
 *             properties: {
 *                 option1: { type: 'string', default: 'default' },
 *             },
 *         },
 *     };
 *
 *     async onInit() {
 *         console.log('Plugin initialized with config:', this.config);
 *     }
 *
 *     async onDestroy() {
 *         console.log('Plugin destroyed');
 *     }
 *
 *     doSomething() {
 *         return 'Hello from my plugin!';
 *     }
 * }
 * ```
 */
export class BasePlugin extends EventEmitter {
    // Override in subclass
    static manifest = {
        id: 'custom.base-plugin',
        name: 'Base Plugin',
        version: '0.0.0',
        description: 'Base plugin class - extend this',
        category: PluginCategory.CORE,
        tier: PluginTier.EXPERIMENTAL,
        capabilities: [],
        configSchema: { type: 'object', properties: {} },
    };

    constructor(config = {}, context = {}) {
        super();

        this.config = this._mergeConfig(config);
        this.context = context;
        this.api = context.api || {};
        this.sandbox = context.sandbox;
        this.initialized = false;
        this.stats = {
            invocations: 0,
            errors: 0,
            lastUsed: null,
        };
    }

    /**
     * Get plugin manifest
     */
    static getManifest() {
        return this.manifest;
    }

    /**
     * Merge user config with defaults from schema
     */
    _mergeConfig(userConfig) {
        const schema = this.constructor.manifest.configSchema;
        const defaults = {};

        if (schema?.properties) {
            for (const [key, prop] of Object.entries(schema.properties)) {
                if (prop.default !== undefined) {
                    defaults[key] = prop.default;
                }
            }
        }

        return { ...defaults, ...userConfig };
    }

    /**
     * Initialize plugin - override in subclass
     */
    async onInit() {
        // Override in subclass
    }

    /**
     * Destroy plugin - override in subclass
     */
    async onDestroy() {
        // Override in subclass
    }

    /**
     * Called by loader to initialize
     */
    async init() {
        if (this.initialized) return;

        try {
            await this.onInit();
            this.initialized = true;
            this.emit('initialized');
        } catch (error) {
            this.stats.errors++;
            throw error;
        }
    }

    /**
     * Called by loader to destroy
     */
    async destroy() {
        if (!this.initialized) return;

        try {
            await this.onDestroy();
            this.initialized = false;
            this.emit('destroyed');
        } catch (error) {
            this.stats.errors++;
            throw error;
        }
    }

    /**
     * Check if plugin has required capability
     */
    requireCapability(capability) {
        if (!this.sandbox?.hasCapability(capability)) {
            throw new Error(`Plugin ${this.constructor.manifest.id} missing capability: ${capability}`);
        }
    }

    /**
     * Log message with plugin prefix
     */
    log(level, message, data = {}) {
        const prefix = `[${this.constructor.manifest.id}]`;
        console.log(`${prefix} [${level.toUpperCase()}]`, message, data);
    }

    /**
     * Record plugin usage
     */
    _recordUsage() {
        this.stats.invocations++;
        this.stats.lastUsed = Date.now();
    }

    /**
     * Get plugin stats
     */
    getStats() {
        return {
            ...this.stats,
            initialized: this.initialized,
            manifest: this.constructor.manifest,
        };
    }
}

// ============================================
// PLUGIN VALIDATION
// ============================================

/**
 * Validate plugin manifest
 */
export function validateManifest(manifest) {
    const errors = [];

    // Required fields
    const required = ['id', 'name', 'version', 'description', 'category', 'tier'];
    for (const field of required) {
        if (!manifest[field]) {
            errors.push(`Missing required field: ${field}`);
        }
    }

    // ID format: org.plugin-name or category.plugin-name
    if (manifest.id && !/^[a-z0-9-]+\.[a-z0-9-]+$/.test(manifest.id)) {
        errors.push('Invalid ID format. Use: category.plugin-name or org.plugin-name');
    }

    // Version semver
    if (manifest.version && !/^\d+\.\d+\.\d+/.test(manifest.version)) {
        errors.push('Invalid version format. Use semantic versioning: X.Y.Z');
    }

    // Category
    if (manifest.category && !Object.values(PluginCategory).includes(manifest.category)) {
        errors.push(`Invalid category. Use one of: ${Object.values(PluginCategory).join(', ')}`);
    }

    // Tier
    if (manifest.tier && !Object.values(PluginTier).includes(manifest.tier)) {
        errors.push(`Invalid tier. Use one of: ${Object.values(PluginTier).join(', ')}`);
    }

    // Capabilities
    if (manifest.capabilities) {
        const validCaps = Object.values(Capability);
        for (const cap of manifest.capabilities) {
            if (!validCaps.includes(cap)) {
                errors.push(`Invalid capability: ${cap}`);
            }
        }
    }

    return {
        valid: errors.length === 0,
        errors,
    };
}

/**
 * Validate plugin class
 */
export function validatePlugin(PluginClass) {
    const errors = [];

    // Must extend BasePlugin
    if (!(PluginClass.prototype instanceof BasePlugin)) {
        errors.push('Plugin must extend BasePlugin');
    }

    // Must have manifest
    if (!PluginClass.manifest) {
        errors.push('Plugin must have static manifest property');
    } else {
        const manifestValidation = validateManifest(PluginClass.manifest);
        errors.push(...manifestValidation.errors);
    }

    return {
        valid: errors.length === 0,
        errors,
    };
}

// ============================================
// PLUGIN REGISTRY
// ============================================

/**
 * Registry for custom plugins
 */
export class PluginRegistry {
    constructor() {
        this.plugins = new Map();  // id -> PluginClass
        this.metadata = new Map(); // id -> metadata
    }

    /**
     * Register a custom plugin
     */
    register(PluginClass) {
        const validation = validatePlugin(PluginClass);
        if (!validation.valid) {
            throw new Error(`Invalid plugin: ${validation.errors.join(', ')}`);
        }

        const manifest = PluginClass.manifest;
        const id = manifest.id;

        if (this.plugins.has(id)) {
            throw new Error(`Plugin already registered: ${id}`);
        }

        // Generate checksum
        const checksum = createHash('sha256')
            .update(PluginClass.toString())
            .digest('hex');

        this.plugins.set(id, PluginClass);
        this.metadata.set(id, {
            manifest,
            checksum,
            registeredAt: Date.now(),
            source: 'custom',
        });

        return { id, checksum };
    }

    /**
     * Unregister a plugin
     */
    unregister(id) {
        if (!this.plugins.has(id)) {
            return false;
        }

        this.plugins.delete(id);
        this.metadata.delete(id);
        return true;
    }

    /**
     * Get plugin class
     */
    get(id) {
        return this.plugins.get(id);
    }

    /**
     * Check if plugin is registered
     */
    has(id) {
        return this.plugins.has(id);
    }

    /**
     * List all registered plugins
     */
    list() {
        return Array.from(this.metadata.entries()).map(([id, meta]) => ({
            id,
            ...meta.manifest,
            checksum: meta.checksum,
            registeredAt: meta.registeredAt,
        }));
    }

    /**
     * Export plugin for distribution
     */
    export(id) {
        const PluginClass = this.plugins.get(id);
        if (!PluginClass) {
            throw new Error(`Plugin not found: ${id}`);
        }

        const metadata = this.metadata.get(id);

        return {
            manifest: PluginClass.manifest,
            code: PluginClass.toString(),
            checksum: metadata.checksum,
            exportedAt: Date.now(),
        };
    }

    /**
     * Import plugin from exported data
     */
    import(exportedPlugin) {
        // Security: Only import from trusted sources in production
        // This is a simplified implementation
        const { manifest, code, checksum } = exportedPlugin;

        // Verify checksum
        const computedChecksum = createHash('sha256').update(code).digest('hex');
        if (computedChecksum !== checksum) {
            throw new Error('Checksum mismatch - plugin may have been tampered');
        }

        // Parse and register (in production, use proper sandboxing)
        // WARNING: eval is dangerous - use VM2 or similar in production
        console.warn('WARNING: Importing plugins uses eval - only import from trusted sources');

        return { imported: true, manifest };
    }
}

// ============================================
// PLUGIN TEMPLATE GENERATOR
// ============================================

/**
 * Generate plugin template
 */
export function generatePluginTemplate(options = {}) {
    const {
        id = 'my-org.my-plugin',
        name = 'My Plugin',
        description = 'A custom edge-net plugin',
        category = PluginCategory.CORE,
        tier = PluginTier.EXPERIMENTAL,
        capabilities = [],
    } = options;

    return `/**
 * ${name}
 *
 * ${description}
 *
 * @module ${id}
 */

import { BasePlugin, Capability, PluginCategory, PluginTier } from '@ruvector/edge-net/plugins/sdk';

export class ${toPascalCase(id.split('.').pop())}Plugin extends BasePlugin {
    static manifest = {
        id: '${id}',
        name: '${name}',
        version: '1.0.0',
        description: '${description}',
        category: PluginCategory.${Object.keys(PluginCategory).find(k => PluginCategory[k] === category) || 'CORE'},
        tier: PluginTier.${Object.keys(PluginTier).find(k => PluginTier[k] === tier) || 'EXPERIMENTAL'},
        author: 'Your Name',
        capabilities: [${capabilities.map(c => `Capability.${Object.keys(Capability).find(k => Capability[k] === c)}`).join(', ')}],
        dependencies: [],
        configSchema: {
            type: 'object',
            properties: {
                enabled: { type: 'boolean', default: true },
                // Add your config options here
            },
        },
        tags: ['custom'],
    };

    async onInit() {
        this.log('info', 'Initializing...');
        // Your initialization code here
    }

    async onDestroy() {
        this.log('info', 'Destroying...');
        // Your cleanup code here
    }

    // Add your plugin methods here
    exampleMethod() {
        this._recordUsage();
        return 'Hello from ${name}!';
    }
}

export default ${toPascalCase(id.split('.').pop())}Plugin;
`;
}

function toPascalCase(str) {
    return str.split('-').map(s => s.charAt(0).toUpperCase() + s.slice(1)).join('');
}

// ============================================
// SINGLETON REGISTRY
// ============================================

let globalRegistry = null;

export function getRegistry() {
    if (!globalRegistry) {
        globalRegistry = new PluginRegistry();
    }
    return globalRegistry;
}

export default {
    BasePlugin,
    validateManifest,
    validatePlugin,
    PluginRegistry,
    generatePluginTemplate,
    getRegistry,
    Capability,
    PluginCategory,
    PluginTier,
};
