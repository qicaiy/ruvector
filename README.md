# RuVector

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/ruvnet/ruvector)

**The index is the neural network.** A high-performance vector database with built-in Graph Neural Networks, Neo4j-compatible hypergraph storage, and adaptive tensor compression.

## What is RuVector?

RuVector combines three powerful concepts into one unified system:

1. **Vector Database** — Sub-millisecond HNSW search with 95%+ recall
2. **Graph Neural Network** — The HNSW topology becomes a trainable GNN
3. **Hypergraph Storage** — Neo4j-compatible Cypher queries with N-ary relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                        RuVector Stack                           │
├─────────────────────────────────────────────────────────────────┤
│  Query API     │ Cypher Parser │ Differentiable Search          │
├─────────────────────────────────────────────────────────────────┤
│  GNN Layers    │ Message Passing │ Multi-Head Attention         │
├─────────────────────────────────────────────────────────────────┤
│  HNSW Index    │ Vector Storage │ Tensor Compression (2-32x)    │
├─────────────────────────────────────────────────────────────────┤
│  Rust Core     │ WASM Bindings  │ Node.js (napi-rs)             │
└─────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **GNN on HNSW** | Graph neural network layers that treat the index topology as a trainable graph |
| **Cypher Queries** | Neo4j-compatible query language with `MATCH`, `WHERE`, `RETURN`, `CREATE` |
| **Hyperedges** | N-ary relationships connecting multiple nodes (not just pairs) |
| **Adaptive Compression** | 5-tier tensor compression: f32 → f16 → PQ8 → PQ4 → Binary (2-32x) |
| **Differentiable Search** | Soft attention over candidates with gradient flow for end-to-end training |
| **WASM Support** | Full browser support with WebAssembly bindings |
| **Memory-Mapped Training** | Efficient gradient accumulation on memory-mapped embeddings |

## Quick Start

### Installation

```bash
# Rust
cargo add ruvector-graph

# Node.js
npm install ruvector-gnn-node

# Browser (WASM)
npm install ruvector-gnn-wasm
```

### Basic Usage

**Cypher Queries (Neo4j-compatible):**
```rust
use ruvector_graph::{GraphDB, cypher::CypherExecutor};

let db = GraphDB::new();
let executor = CypherExecutor::new(&db);

// Create nodes and relationships
executor.execute("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})")?;

// Query with pattern matching
let results = executor.execute("MATCH (p:Person)-[:KNOWS]->(friend) RETURN p.name, friend.name")?;
```

**GNN Forward Pass:**
```rust
use ruvector_gnn::{RuvectorLayer, differentiable_search};

// Create GNN layer
let layer = RuvectorLayer::new(128, 256, 4, 0.1); // input, hidden, heads, dropout

// Forward pass with neighbor aggregation
let output = layer.forward(&node_embedding, &neighbor_embeddings, &edge_weights);

// Differentiable search (soft k-NN)
let (indices, weights) = differentiable_search(&query, &candidates, 10, 0.07);
```

**Tensor Compression:**
```rust
use ruvector_gnn::TensorCompress;

let compressor = TensorCompress::new();

// Adaptive compression based on access frequency
let compressed = compressor.compress(&embedding, 0.5)?;  // Warm data → f16
let restored = compressor.decompress(&compressed)?;
```

### Browser Usage (WASM)

```javascript
import init, { JsRuvectorLayer, JsTensorCompress, differentiableSearch } from 'ruvector-gnn-wasm';

await init();

// GNN layer
const layer = new JsRuvectorLayer(128, 256, 4, 0.1);
const output = layer.forward(nodeEmbedding, neighbors, weights);

// Compression
const compressor = new JsTensorCompress();
const compressed = compressor.compress(embedding, 0.5);
```

## Architecture

### Crate Structure

```
crates/
├── ruvector-core/        # Vector database core (HNSW, storage)
├── ruvector-graph/       # Neo4j-compatible hypergraph + Cypher
├── ruvector-gnn/         # GNN layers, compression, training
├── ruvector-gnn-wasm/    # WebAssembly bindings
├── ruvector-gnn-node/    # Node.js bindings (napi-rs)
├── ruvector-graph-wasm/  # Graph WASM bindings
└── ruvector-graph-node/  # Graph Node.js bindings
```

### Compression Tiers

| Tier | Access Freq | Format | Compression | Use Case |
|------|------------|--------|-------------|----------|
| Hot | >80% | f32 | 1x | Active queries |
| Warm | 40-80% | f16 | 2x | Recent data |
| Cool | 10-40% | PQ8 | ~8x | Older data |
| Cold | 1-10% | PQ4 | ~16x | Archived |
| Archive | <1% | Binary | ~32x | Rarely accessed |

### GNN Message Passing

The RuvectorLayer implements attention-based message passing on the HNSW graph:

```
h_new = LayerNorm(h + GRU(h, Attention(W_msg(neighbors), edge_weights)))
```

1. **Message**: Transform neighbor embeddings with learned weights
2. **Aggregate**: Multi-head attention over messages, weighted by edge similarity
3. **Update**: GRU cell combines current state with aggregated messages
4. **Normalize**: Layer normalization with residual connection

## Tutorial

### 1. Creating a Knowledge Graph

```rust
use ruvector_graph::{GraphDB, NodeBuilder, EdgeBuilder};

let db = GraphDB::new();

// Create nodes
let alice = NodeBuilder::new("alice")
    .label("Person")
    .property("name", "Alice")
    .property("age", 30)
    .build();

let knows_rust = NodeBuilder::new("rust")
    .label("Skill")
    .property("name", "Rust")
    .build();

db.create_node(alice)?;
db.create_node(knows_rust)?;

// Create relationship
let edge = EdgeBuilder::new("e1", "alice", "rust")
    .edge_type("KNOWS")
    .property("level", "expert")
    .build();

db.create_edge(edge)?;
```

### 2. Semantic Search with GNN Enhancement

```rust
use ruvector_gnn::{RuvectorLayer, RuvectorQuery, QueryMode};

// Initialize GNN layer
let gnn = RuvectorLayer::new(384, 512, 8, 0.1);

// Build query
let query = RuvectorQuery::neural_search(query_embedding, 10, 2)
    .with_temperature(0.07);

// Search with GNN-enhanced representations
let enhanced = gnn.forward(&query.vector.unwrap(), &neighbor_embs, &weights);
```

### 3. Training with InfoNCE Loss

```rust
use ruvector_gnn::training::{info_nce_loss, sgd_step, TrainConfig};

let config = TrainConfig::default();

// Compute contrastive loss
let loss = info_nce_loss(
    &anchor_embedding,
    &positive_embeddings,
    &negative_embeddings,
    config.temperature
);

// Update embeddings
sgd_step(&mut embedding, &gradient, config.learning_rate);
```

## Documentation

| Topic | Link |
|-------|------|
| **Getting Started** | [docs/guide/GETTING_STARTED.md](./docs/guide/GETTING_STARTED.md) |
| **Cypher Query Language** | [docs/api/CYPHER_REFERENCE.md](./docs/api/CYPHER_REFERENCE.md) |
| **GNN Architecture** | [docs/gnn-layer-implementation.md](./docs/gnn-layer-implementation.md) |
| **Compression Guide** | [docs/optimization/COMPRESSION.md](./docs/optimization/COMPRESSION.md) |
| **WASM Bindings** | [crates/ruvector-gnn-wasm/README.md](./crates/ruvector-gnn-wasm/README.md) |
| **Node.js Bindings** | [crates/ruvector-gnn-node/README.md](./crates/ruvector-gnn-node/README.md) |
| **API Reference** | [docs/api/](./docs/api/) |
| **Performance Tuning** | [docs/optimization/PERFORMANCE_TUNING_GUIDE.md](./docs/optimization/PERFORMANCE_TUNING_GUIDE.md) |

## Performance

```
Query Latency (p50)     <0.5ms     HNSW with SIMD
GNN Forward Pass        ~1ms       Per-node with neighbors
Compression (PQ8)       ~8x        Memory reduction
Recall @ k=10           95%+       High accuracy search
Browser (WASM)          ~2ms       Full functionality
```

## Building from Source

```bash
# Clone
git clone https://github.com/ruvnet/ruvector.git
cd ruvector

# Build all crates
cargo build --release

# Run tests
cargo test --workspace

# Build WASM
cargo build --package ruvector-gnn-wasm --target wasm32-unknown-unknown
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/development/CONTRIBUTING.md).

## License

MIT License - see [LICENSE](./LICENSE).

---

<div align="center">

**Built by [rUv](https://ruv.io)** • [GitHub](https://github.com/ruvnet/ruvector) • [Documentation](./docs/)

*"The index is a sparse neural network whose topology encodes learned similarity."*

</div>
