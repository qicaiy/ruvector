# @ruvector/ruvllm v2.3

Self-learning LLM orchestration with SONA adaptive learning, HNSW memory, and SIMD inference for Node.js.

## Installation

```bash
npm install @ruvector/ruvllm
```

## Quick Start

```typescript
import { RuvLLM, RuvLLMConfig } from '@ruvector/ruvllm';

// Initialize with default configuration
const llm = new RuvLLM();

// Or with custom configuration
const llm = new RuvLLM({
  modelPath: './models/ruvltra-small-q4km.gguf',
  sonaEnabled: true,
  flashAttention: true,
  maxTokens: 256,
});

// Generate text
const response = await llm.query('Explain quantum computing');
console.log(response.text);

// Stream generation
for await (const token of llm.stream('Write a haiku about Rust')) {
  process.stdout.write(token);
}
```

## What's New in v2.3

| Feature | Description |
|---------|-------------|
| **RuvLTRA Models** | Purpose-built 0.5B & 3B models for Claude Flow |
| **Task-Specific LoRA** | 5 pre-trained adapters (coder, researcher, security, architect, reviewer) |
| **HuggingFace Hub** | Download/upload models directly |
| **Adapter Merging** | TIES, DARE, SLERP strategies |
| **HNSW Routing** | 150x faster semantic matching |

## CLI Usage

```bash
# Query a model
ruvllm query "What is machine learning?"

# Stream output
ruvllm query --stream "Write a poem"

# Download a model
ruvllm download ruvector/ruvltra-small-q4km

# Benchmark
ruvllm bench ./models/model.gguf
```

## API Reference

### RuvLLM Class

```typescript
class RuvLLM {
  constructor(config?: RuvLLMConfig);

  // Generate text
  query(prompt: string, params?: GenerateParams): Promise<Response>;

  // Stream generation
  stream(prompt: string, params?: GenerateParams): AsyncIterable<string>;

  // Load a model
  loadModel(path: string): Promise<void>;

  // Get SONA learning stats
  sonaStats(): SonaStats | null;

  // Adapt on feedback
  adapt(input: Float32Array, quality: number): void;
}
```

### Configuration

```typescript
interface RuvLLMConfig {
  modelPath?: string;       // Path to GGUF model
  sonaEnabled?: boolean;    // Enable SONA learning (default: true)
  flashAttention?: boolean; // Use Flash Attention 2 (default: true)
  maxTokens?: number;       // Max generation tokens (default: 256)
  temperature?: number;     // Sampling temperature (default: 0.7)
  topP?: number;            // Top-p sampling (default: 0.9)
}
```

### Generate Parameters

```typescript
interface GenerateParams {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
}
```

## SIMD Module

For direct access to optimized SIMD kernels:

```typescript
import { simd } from '@ruvector/ruvllm/simd';

// Dot product
const result = simd.dotProduct(vecA, vecB);

// Matrix multiplication
const output = simd.matmul(matrix, vector);

// Flash Attention
const attended = simd.flashAttention(query, key, value, scale);

// RMS Normalization
simd.rmsNorm(hidden, weights, epsilon);
```

## Performance (M4 Pro)

| Operation | Performance |
|-----------|-------------|
| Inference | 88-135 tok/s |
| Flash Attention | 320µs (seq=2048) |
| HNSW Search | 17-62µs |
| SONA Adapt | <1ms |

## Supported Models

- **RuvLTRA-Small** (494M) - Q4K, Q5K, Q8
- **RuvLTRA-Medium** (3B) - Q4K, Q5K, Q8
- **Qwen 2.5** (0.5B-72B)
- **Llama 3.x** (8B-70B)
- **Mistral** (7B-22B)
- **Phi-3** (3.8B-14B)
- **Gemma-2** (2B-27B)

## Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| macOS | arm64 (M1-M4) | ✅ Full support |
| macOS | x64 | ✅ Supported |
| Linux | x64 | ✅ Supported |
| Linux | arm64 | ✅ Supported |
| Windows | x64 | ✅ Supported |

## Related Packages

- [@ruvector/core](https://www.npmjs.com/package/@ruvector/core) - Vector operations
- [@ruvector/sona](https://www.npmjs.com/package/@ruvector/sona) - SONA learning engine
- [@ruvector/ruvector](https://www.npmjs.com/package/@ruvector/ruvector) - Full Ruvector SDK

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [API Documentation](https://docs.rs/ruvllm)
- [Crate (Rust)](https://crates.io/crates/ruvllm)

## License

MIT OR Apache-2.0
