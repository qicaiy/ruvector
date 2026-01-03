#!/usr/bin/env node
/**
 * Edge-Net Plugin CLI
 *
 * Manage plugins from the command line.
 *
 * Commands:
 *   edge-net plugins list              List available plugins
 *   edge-net plugins info <id>         Show plugin details
 *   edge-net plugins enable <id>       Enable a plugin
 *   edge-net plugins disable <id>      Disable a plugin
 *   edge-net plugins bundles           List plugin bundles
 *   edge-net plugins create <name>     Create new plugin from template
 *   edge-net plugins validate <path>   Validate a custom plugin
 *
 * @module @ruvector/edge-net/plugins/cli
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, realpathSync } from 'fs';
import { join, dirname, resolve, isAbsolute } from 'path';
import { fileURLToPath } from 'url';
import {
    PLUGIN_CATALOG,
    PLUGIN_BUNDLES,
    PluginCategory,
    PluginTier,
} from './plugin-manifest.js';
import { PluginManager } from './plugin-loader.js';
import { generatePluginTemplate, validateManifest, getRegistry } from './plugin-sdk.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ============================================
// CLI COMMANDS
// ============================================

const commands = {
    /**
     * List all available plugins
     */
    list: async (args) => {
        const filter = args[0]; // Optional: category filter

        console.log('\n╔════════════════════════════════════════════════════════════════╗');
        console.log('║              EDGE-NET PLUGIN CATALOG                            ║');
        console.log('╚════════════════════════════════════════════════════════════════╝\n');

        const plugins = Object.values(PLUGIN_CATALOG);
        const grouped = {};

        for (const plugin of plugins) {
            if (filter && plugin.category !== filter && plugin.tier !== filter) {
                continue;
            }
            if (!grouped[plugin.category]) {
                grouped[plugin.category] = [];
            }
            grouped[plugin.category].push(plugin);
        }

        for (const [category, categoryPlugins] of Object.entries(grouped)) {
            console.log(`\n┌─ ${category.toUpperCase()} ─────────────────────────────────────────┐`);

            for (const plugin of categoryPlugins) {
                const tier = getTierBadge(plugin.tier);
                const status = getStatusIcon(plugin);
                console.log(`│ ${status} ${plugin.id.padEnd(30)} ${tier}`);
                console.log(`│   ${plugin.description.slice(0, 55)}...`);
            }

            console.log('└────────────────────────────────────────────────────────┘');
        }

        console.log(`\n📦 Total: ${plugins.length} plugins in ${Object.keys(grouped).length} categories\n`);
        console.log('Use: edge-net plugins info <id> for details');
        console.log('Use: edge-net plugins enable <id> to enable a plugin\n');
    },

    /**
     * Show plugin details
     */
    info: async (args) => {
        const pluginId = args[0];
        if (!pluginId) {
            console.error('Usage: edge-net plugins info <plugin-id>');
            process.exit(1);
        }

        const plugin = PLUGIN_CATALOG[pluginId];
        if (!plugin) {
            console.error(`Plugin not found: ${pluginId}`);
            console.log('\nAvailable plugins:');
            Object.keys(PLUGIN_CATALOG).forEach(id => console.log(`  - ${id}`));
            process.exit(1);
        }

        console.log('\n╔════════════════════════════════════════════════════════════════╗');
        console.log(`║  ${plugin.name.padEnd(60)}║`);
        console.log('╚════════════════════════════════════════════════════════════════╝\n');

        console.log(`  ID:          ${plugin.id}`);
        console.log(`  Version:     ${plugin.version}`);
        console.log(`  Category:    ${plugin.category}`);
        console.log(`  Tier:        ${getTierBadge(plugin.tier)}`);
        console.log(`  Description: ${plugin.description}`);

        if (plugin.capabilities?.length > 0) {
            console.log(`\n  Capabilities Required:`);
            plugin.capabilities.forEach(cap => console.log(`    • ${cap}`));
        }

        if (plugin.dependencies?.length > 0) {
            console.log(`\n  Dependencies:`);
            plugin.dependencies.forEach(dep => console.log(`    → ${dep}`));
        }

        if (plugin.tags?.length > 0) {
            console.log(`\n  Tags: ${plugin.tags.join(', ')}`);
        }

        if (plugin.configSchema?.properties) {
            console.log('\n  Configuration Options:');
            for (const [key, prop] of Object.entries(plugin.configSchema.properties)) {
                const def = prop.default !== undefined ? ` (default: ${JSON.stringify(prop.default)})` : '';
                console.log(`    ${key}: ${prop.type}${def}`);
            }
        }

        console.log('\n  Usage:');
        console.log(`    const plugin = await plugins.load('${pluginId}');`);
        console.log('');
    },

    /**
     * List plugin bundles
     */
    bundles: async () => {
        console.log('\n╔════════════════════════════════════════════════════════════════╗');
        console.log('║              PLUGIN BUNDLES                                     ║');
        console.log('╚════════════════════════════════════════════════════════════════╝\n');

        for (const [id, bundle] of Object.entries(PLUGIN_BUNDLES)) {
            console.log(`┌─ ${bundle.name.toUpperCase()} ─────────────────────────────────────────┐`);
            console.log(`│ ID: ${id}`);
            console.log(`│ ${bundle.description}`);
            console.log('│');
            console.log('│ Plugins:');
            if (bundle.plugins.length === 0) {
                console.log('│   (none - minimal bundle)');
            } else {
                bundle.plugins.forEach(p => console.log(`│   • ${p}`));
            }
            console.log('└────────────────────────────────────────────────────────────┘\n');
        }

        console.log('Usage: await plugins.loadBundle("bundle-name")\n');
    },

    /**
     * Create new plugin from template
     */
    create: async (args) => {
        const name = args[0];
        if (!name) {
            console.error('Usage: edge-net plugins create <plugin-name>');
            console.log('\nOptions:');
            console.log('  --category <category>   Plugin category (default: core)');
            console.log('  --tier <tier>          Plugin tier (default: experimental)');
            console.log('  --output <dir>         Output directory (default: ./plugins)');
            process.exit(1);
        }

        // Parse options
        const category = getArg(args, '--category') || 'core';
        const tier = getArg(args, '--tier') || 'experimental';
        const outputDir = getArg(args, '--output') || './plugins';

        const id = `custom.${name.toLowerCase().replace(/\s+/g, '-')}`;
        const pascalName = name.split(/[-\s]+/).map(s =>
            s.charAt(0).toUpperCase() + s.slice(1)
        ).join('');

        console.log(`\n🔧 Creating plugin: ${name}\n`);

        const template = generatePluginTemplate({
            id,
            name,
            description: `Custom edge-net plugin: ${name}`,
            category: PluginCategory[category.toUpperCase()] || PluginCategory.CORE,
            tier: PluginTier[tier.toUpperCase()] || PluginTier.EXPERIMENTAL,
            capabilities: [],
        });

        // Create output directory
        const pluginDir = join(outputDir, name.toLowerCase().replace(/\s+/g, '-'));
        if (!existsSync(pluginDir)) {
            mkdirSync(pluginDir, { recursive: true });
        }

        // Write plugin file
        const pluginFile = join(pluginDir, 'index.js');
        writeFileSync(pluginFile, template);

        // Write package.json
        const packageJson = {
            name: `@edge-net-plugin/${name.toLowerCase().replace(/\s+/g, '-')}`,
            version: '1.0.0',
            type: 'module',
            main: 'index.js',
            description: `Custom edge-net plugin: ${name}`,
            keywords: ['edge-net', 'plugin', category],
            peerDependencies: {
                '@ruvector/edge-net': '^0.4.0',
            },
        };
        writeFileSync(join(pluginDir, 'package.json'), JSON.stringify(packageJson, null, 2));

        // Write README
        const readme = `# ${name}

Custom edge-net plugin.

## Installation

\`\`\`bash
npm install @edge-net-plugin/${name.toLowerCase().replace(/\s+/g, '-')}
\`\`\`

## Usage

\`\`\`javascript
import { getRegistry } from '@ruvector/edge-net/plugins/sdk';
import { ${pascalName}Plugin } from './${name.toLowerCase().replace(/\s+/g, '-')}';

// Register plugin
getRegistry().register(${pascalName}Plugin);

// Use with plugin manager
const plugins = PluginManager.getInstance();
const myPlugin = await plugins.load('${id}');
\`\`\`

## Configuration

See \`configSchema\` in the plugin manifest for available options.

## License

MIT
`;
        writeFileSync(join(pluginDir, 'README.md'), readme);

        console.log(`✅ Created plugin at: ${pluginDir}`);
        console.log('');
        console.log('Files created:');
        console.log(`  📄 ${pluginFile}`);
        console.log(`  📄 ${join(pluginDir, 'package.json')}`);
        console.log(`  📄 ${join(pluginDir, 'README.md')}`);
        console.log('');
        console.log('Next steps:');
        console.log(`  1. Edit ${pluginFile} to add your logic`);
        console.log('  2. Register with: getRegistry().register(YourPlugin)');
        console.log('  3. Load with: plugins.load("' + id + '")');
        console.log('');
    },

    /**
     * Validate a plugin
     */
    validate: async (args) => {
        const pluginPath = args[0];
        if (!pluginPath) {
            console.error('Usage: edge-net plugins validate <plugin-path>');
            process.exit(1);
        }

        console.log(`\n🔍 Validating plugin: ${pluginPath}\n`);

        try {
            // SECURITY: Validate plugin path to prevent arbitrary code execution
            const cwd = process.cwd();
            const allowedDirs = [
                resolve(cwd, 'plugins'),
                resolve(cwd, 'node_modules'),
                resolve(__dirname, 'implementations'),
            ];

            // Resolve the absolute path and follow symlinks
            let absolutePath;
            try {
                absolutePath = isAbsolute(pluginPath)
                    ? realpathSync(pluginPath)
                    : realpathSync(resolve(cwd, pluginPath));
            } catch (e) {
                throw new Error(`Plugin file not found or inaccessible: ${pluginPath}`);
            }

            // Verify path is within allowed directories
            const isAllowed = allowedDirs.some(dir => {
                try {
                    const realDir = realpathSync(dir);
                    return absolutePath.startsWith(realDir + '/') || absolutePath === realDir;
                } catch {
                    return false;
                }
            });

            if (!isAllowed) {
                throw new Error(
                    `Security: Plugin path must be within allowed directories:\n` +
                    allowedDirs.map(d => `  - ${d}`).join('\n')
                );
            }

            // Verify file extension
            if (!absolutePath.endsWith('.js') && !absolutePath.endsWith('.mjs')) {
                throw new Error('Security: Plugin must be a .js or .mjs file');
            }

            const plugin = await import(absolutePath);
            const PluginClass = plugin.default || Object.values(plugin)[0];

            if (!PluginClass) {
                console.error('❌ No plugin class found in module');
                process.exit(1);
            }

            // Validate manifest
            if (!PluginClass.manifest) {
                console.error('❌ Plugin missing static manifest property');
                process.exit(1);
            }

            const validation = validateManifest(PluginClass.manifest);

            if (validation.valid) {
                console.log('✅ Plugin is valid!\n');
                console.log('Manifest:');
                console.log(JSON.stringify(PluginClass.manifest, null, 2));
            } else {
                console.log('❌ Plugin validation failed:\n');
                validation.errors.forEach(err => console.log(`  • ${err}`));
                process.exit(1);
            }
        } catch (error) {
            console.error(`❌ Error loading plugin: ${error.message}`);
            process.exit(1);
        }
    },

    /**
     * Show help
     */
    help: async () => {
        console.log(`
Edge-Net Plugin Management

Usage: edge-net plugins <command> [options]

Commands:
  list [category]        List available plugins (filter by category/tier)
  info <id>             Show detailed plugin information
  bundles               List plugin bundles
  create <name>         Create new plugin from template
  validate <path>       Validate a custom plugin file

Categories:
  core, network, crypto, privacy, ai, economic, storage, exotic

Tiers:
  stable, beta, experimental, research

Examples:
  edge-net plugins list
  edge-net plugins list privacy
  edge-net plugins info crypto.zk-proofs
  edge-net plugins bundles
  edge-net plugins create my-plugin --category ai
  edge-net plugins validate ./my-plugin/index.js
`);
    },
};

// ============================================
// HELPERS
// ============================================

function getTierBadge(tier) {
    const badges = {
        stable: '🟢 STABLE',
        beta: '🟡 BETA',
        experimental: '🟠 EXPERIMENTAL',
        research: '🔴 RESEARCH',
    };
    return badges[tier] || tier;
}

function getStatusIcon(plugin) {
    // In real implementation, check if enabled
    return '○';
}

function getArg(args, flag) {
    const idx = args.indexOf(flag);
    if (idx >= 0 && args[idx + 1]) {
        return args[idx + 1];
    }
    return null;
}

// ============================================
// MAIN
// ============================================

async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'help';
    const commandArgs = args.slice(1);

    if (commands[command]) {
        await commands[command](commandArgs);
    } else {
        console.error(`Unknown command: ${command}`);
        await commands.help();
        process.exit(1);
    }
}

// Run if executed directly
if (process.argv[1]?.includes('cli.js') || process.argv[1]?.includes('plugins')) {
    main().catch(console.error);
}

export { commands };
export default main;
