/**
 * ruvector - High-performance vector database for Node.js
 *
 * This package automatically detects and uses the best available implementation:
 * 1. Native (Rust-based, fastest) - if available for your platform
 * 2. WASM (WebAssembly, universal fallback) - works everywhere
 */

export * from './types';

let implementation: any;
let implementationType: 'native' | 'wasm' = 'wasm';

try {
  // Try to load native module first
  implementation = require('@ruvector/core');
  implementationType = 'native';

  // Verify it's actually working
  if (typeof implementation.VectorDb !== 'function') {
    throw new Error('Native module loaded but VectorDb not found');
  }
} catch (e: any) {
  // Fallback to WASM
  if (process.env.RUVECTOR_DEBUG) {
    console.warn('[ruvector] Native module not available:', e.message);
    console.warn('[ruvector] Falling back to WASM implementation');
  }

  try {
    implementation = require('@ruvector/wasm');
    implementationType = 'wasm';
  } catch (wasmError: any) {
    throw new Error(
      `Failed to load ruvector: Neither native nor WASM implementation available.\n` +
      `Native error: ${e.message}\n` +
      `WASM error: ${wasmError.message}`
    );
  }
}

/**
 * Get the current implementation type
 */
export function getImplementationType(): 'native' | 'wasm' {
  return implementationType;
}

/**
 * Check if native implementation is being used
 */
export function isNative(): boolean {
  return implementationType === 'native';
}

/**
 * Check if WASM implementation is being used
 */
export function isWasm(): boolean {
  return implementationType === 'wasm';
}

/**
 * Get version information
 */
export function getVersion(): { version: string; implementation: string } {
  const pkg = require('../package.json');
  return {
    version: pkg.version,
    implementation: implementationType
  };
}

// Export the VectorDB class (note: native exports VectorDb, we re-export as VectorDB for consistency)
export const VectorDB = implementation.VectorDb;

// Export everything from the implementation
export default implementation;
