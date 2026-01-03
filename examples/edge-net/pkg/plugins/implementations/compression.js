/**
 * Data Compression Plugin
 *
 * LZ4/Zstd compression for network payloads and storage.
 * Uses WASM for high-performance compression.
 *
 * @module @ruvector/edge-net/plugins/compression
 */

// Simple LZ4-like compression (for demo - use actual WASM in production)
export class CompressionPlugin {
    constructor(config = {}) {
        this.config = {
            algorithm: config.algorithm || 'lz4',
            level: config.level || 3,
            threshold: config.threshold || 1024,
        };
        this.stats = {
            compressed: 0,
            decompressed: 0,
            bytesIn: 0,
            bytesOut: 0,
            ratio: 0,
        };
    }

    /**
     * Compress data if above threshold
     */
    compress(data) {
        const input = typeof data === 'string' ? Buffer.from(data) : data;

        if (input.length < this.config.threshold) {
            return { compressed: false, data: input };
        }

        // Simple RLE-based compression (demo)
        const compressed = this._rleCompress(input);

        this.stats.compressed++;
        this.stats.bytesIn += input.length;
        this.stats.bytesOut += compressed.length;
        this.stats.ratio = this.stats.bytesOut / this.stats.bytesIn;

        return {
            compressed: true,
            data: compressed,
            originalSize: input.length,
            compressedSize: compressed.length,
            ratio: compressed.length / input.length,
        };
    }

    /**
     * Decompress data
     */
    decompress(data, wasCompressed = true) {
        if (!wasCompressed) {
            return data;
        }

        const decompressed = this._rleDecompress(data);
        this.stats.decompressed++;

        return decompressed;
    }

    /**
     * Simple RLE compression (for demo)
     */
    _rleCompress(input) {
        const output = [];
        let i = 0;

        while (i < input.length) {
            const byte = input[i];
            let count = 1;

            while (i + count < input.length &&
                   input[i + count] === byte &&
                   count < 255) {
                count++;
            }

            if (count >= 4) {
                output.push(0xFF, count, byte);
            } else {
                for (let j = 0; j < count; j++) {
                    if (byte === 0xFF) {
                        output.push(0xFF, 1, byte);
                    } else {
                        output.push(byte);
                    }
                }
            }

            i += count;
        }

        return Buffer.from(output);
    }

    /**
     * Simple RLE decompression
     */
    _rleDecompress(input) {
        const output = [];
        let i = 0;

        while (i < input.length) {
            if (input[i] === 0xFF && i + 2 < input.length) {
                const count = input[i + 1];
                const byte = input[i + 2];
                for (let j = 0; j < count; j++) {
                    output.push(byte);
                }
                i += 3;
            } else {
                output.push(input[i]);
                i++;
            }
        }

        return Buffer.from(output);
    }

    getStats() {
        return this.stats;
    }
}

export default CompressionPlugin;
