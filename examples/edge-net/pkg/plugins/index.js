/**
 * Edge-Net Plugin System
 *
 * Secure, modular plugin architecture for extending edge-net functionality.
 *
 * Features:
 * - Official plugin catalog (practical to exotic)
 * - Custom plugin SDK for user-created plugins
 * - Capability-based security sandboxing
 * - Lazy loading with verification
 * - Plugin bundles for common use cases
 *
 * @module @ruvector/edge-net/plugins
 *
 * @example
 * ```javascript
 * // Load official plugins
 * import { PluginManager, PLUGIN_CATALOG, PLUGIN_BUNDLES } from '@ruvector/edge-net/plugins';
 *
 * const plugins = PluginManager.getInstance();
 *
 * // View available plugins
 * console.log(plugins.getCatalog());
 *
 * // Load a bundle
 * await plugins.loadBundle('privacy-focused');
 *
 * // Load individual plugin
 * const encryption = await plugins.load('privacy.e2e-encryption');
 *
 * // Create custom plugin
 * import { BasePlugin, generatePluginTemplate, getRegistry } from '@ruvector/edge-net/plugins/sdk';
 *
 * class MyPlugin extends BasePlugin {
 *     static manifest = { ... };
 *     async onInit() { ... }
 * }
 *
 * getRegistry().register(MyPlugin);
 * ```
 */

// Core exports
export { PLUGIN_CATALOG, PLUGIN_BUNDLES, PluginCategory, PluginTier, Capability } from './plugin-manifest.js';
export { PluginLoader, PluginManager } from './plugin-loader.js';
export {
    BasePlugin,
    validateManifest,
    validatePlugin,
    PluginRegistry,
    generatePluginTemplate,
    getRegistry,
} from './plugin-sdk.js';

// Implementation exports (for direct use)
export { CompressionPlugin } from './implementations/compression.js';
export { E2EEncryptionPlugin } from './implementations/e2e-encryption.js';
export { FederatedLearningPlugin } from './implementations/federated-learning.js';
export { ReputationStakingPlugin } from './implementations/reputation-staking.js';
export { SwarmIntelligencePlugin } from './implementations/swarm-intelligence.js';

// Convenience function to get started quickly
import { PluginManager } from './plugin-loader.js';

/**
 * Quick start - get plugin manager with standard bundle
 */
export async function initPlugins(options = {}) {
    const manager = PluginManager.getInstance(options);

    if (options.bundle) {
        await manager.loadBundle(options.bundle);
    }

    return manager;
}

/**
 * List all available plugins with their status
 */
export function listPlugins() {
    const manager = PluginManager.getInstance();
    return manager.getCatalog();
}

export default {
    initPlugins,
    listPlugins,
    PluginManager,
};
