# RvLite v0.3.0 - Standalone Vector Database

A complete, lightweight vector database compiled to WebAssembly with **22 integrated WASM modules**. Runs anywhere JavaScript runs: browsers, Node.js, Deno, Bun, Cloudflare Workers, Vercel Edge Functions.

## What's New in v0.3.0

- **22 WASM modules** integrated via feature flags (up from 1 in v0.2.0)
- **Feature-gated Cargo.toml** with individual and composite feature sets
- **12 new extension modules**: GNN, attention, delta, learning, math, hyperbolic, nervous system, sparse inference, DAG, router, HNSW, SONA
- **Expanded CLI** with `modules`, `delta-*`, `math-*`, `hyperbolic-*`, `dag-*`, `nervous-*`, `sparse-*`, `router-*`, and `info` commands
- **Updated TypeScript SDK** with typed interfaces for all 22 modules
- **Composite feature sets**: `core-plus`, `ml`, `advanced-search`, `full`

## Architecture

RvLite is a **thin orchestration layer** over battle-tested WASM crates from the RuVector ecosystem:

```
+----------------------------------------------------------+
|  RvLite v0.3.0 (Orchestration Layer)                     |
|  +-- SQL executor (pgvector-compatible)                  |
|  +-- SPARQL executor (RDF triple store)                  |
|  +-- Cypher executor (property graphs)                   |
|  +-- Storage adapter (IndexedDB / filesystem)            |
|  +-- Extensions (feature-gated modules)                  |
+---------------------------+------------------------------+
                            | depends on (100% reuse)
                            v
+----------------------------------------------------------+
|  22 WASM Crates                                          |
+----------------------------------------------------------+
|  Core:      ruvector-core (vectors, SIMD)                |
|  Graph:     ruvector-graph-wasm (Cypher)                 |
|  GNN:       ruvector-gnn-wasm (GCN, GAT, GraphSAGE)     |
|  Attention: ruvector-attention-wasm (39 types)           |
|  HNSW:      micro-hnsw-wasm (neuromorphic, 11.8KB)       |
|  Hyperbolic: ruvector-hyperbolic-hnsw-wasm (Poincare)    |
|  Learning:  ruvector-learning-wasm (MicroLoRA)           |
|  SONA:      ruvector-sona (ReasoningBank, EWC++)         |
|  Math:      ruvector-math-wasm (Wasserstein, manifolds)  |
|  Delta:     ruvector-delta-wasm (incremental updates)    |
|  Sparse:    ruvector-sparse-inference-wasm (PowerInfer)  |
|  Nervous:   ruvector-nervous-system-wasm (SNN, STDP)     |
|  DAG:       ruvector-dag-wasm (workflow orchestration)   |
|  Router:    ruvector-router-wasm (intelligent routing)   |
|  Economy:   ruvector-economy-wasm (token management)     |
|  Exotic:    ruvector-exotic-wasm (exotic distance types) |
|  FPGA:      ruvector-fpga-transformer-wasm               |
|  MinCut:    ruvector-mincut-wasm (graph optimization)    |
|  LLM:       ruvllm-wasm (local GGUF inference)           |
|  Cognitum:  cognitum-gate-kernel (evidence evaluation)   |
+----------------------------------------------------------+
```

## Quick Start

### JavaScript/TypeScript SDK

```typescript
import { RvLite, createRvLite } from 'rvlite';

// Create database (384-dimensional cosine similarity)
const db = await createRvLite({ dimensions: 384 });

// Insert vectors
const id = await db.insert([0.1, 0.2, 0.3, /* ... */], { text: "Hello" });

// Search similar vectors
const results = await db.search([0.1, 0.2, 0.3, /* ... */], 10);

// SQL with pgvector syntax
await db.sql("INSERT INTO vectors (id, vector) VALUES ('v1', '[0.1, 0.2]')");
await db.sql("SELECT id FROM vectors ORDER BY vector <-> '[0.1, 0.2]' LIMIT 5");

// Cypher graph queries
await db.cypher("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})");

// SPARQL RDF queries
await db.addTriple("<http://ex.org/alice>", "<http://ex.org/knows>", "<http://ex.org/bob>");
await db.sparql("SELECT ?s WHERE { ?s <http://ex.org/knows> ?o }");

// Persistence (browser: IndexedDB, Node.js: JSON file)
await db.save();
```

### CLI

```bash
# Install
npm install -g rvlite

# Initialize database
rvlite init --dimensions 384 --metric cosine

# Insert vectors
rvlite insert "[0.1, 0.2, 0.3]" --metadata '{"text": "hello"}'
rvlite embed "Hello world" --insert

# Search
rvlite search "[0.1, 0.2, 0.3]" --top-k 5
rvlite embed-search "similar text" -k 10

# List all 22 modules
rvlite modules

# System info
rvlite info

# Interactive REPL
rvlite repl
```

## Feature Flags

RvLite uses Cargo feature flags so you only pay for what you use:

### Individual Features

| Feature | Crate | Description |
|---------|-------|-------------|
| `graph` | ruvector-graph-wasm | Cypher property graph via native crate |
| `gnn` | ruvector-gnn-wasm | Graph Neural Networks |
| `attention` | ruvector-attention-wasm | 39 attention mechanisms |
| `attention-unified` | ruvector-attention-unified-wasm | Unified attention API |
| `hnsw` | micro-hnsw-wasm | Neuromorphic HNSW (11.8KB) |
| `hyperbolic` | ruvector-hyperbolic-hnsw-wasm | Hyperbolic space search |
| `sona` | ruvector-sona | Self-Optimizing Neural Architecture |
| `learning` | ruvector-learning-wasm | MicroLoRA adaptation |
| `math` | ruvector-math-wasm | Optimal Transport, manifolds |
| `delta` | ruvector-delta-wasm | Incremental vector updates |
| `sparse` | ruvector-sparse-inference-wasm | Sparse inference |
| `nervous` | ruvector-nervous-system-wasm | Bio-inspired SNN |
| `dag` | ruvector-dag-wasm | DAG workflows |
| `router` | ruvector-router-wasm | Intelligent routing |
| `economy` | ruvector-economy-wasm | Token economy |
| `exotic` | ruvector-exotic-wasm | Exotic distance types |
| `fpga` | ruvector-fpga-transformer-wasm | FPGA transformer |
| `mincut` | ruvector-mincut-wasm | Graph min-cut |
| `mincut-transformer` | ruvector-mincut-gated-transformer-wasm | Gated transformer min-cut |
| `tiny-dancer` | ruvector-tiny-dancer-wasm | Lightweight dance framework |
| `llm` | ruvllm-wasm | Local LLM inference |
| `cognitum` | cognitum-gate-kernel | Cognitive gateway |

### Composite Feature Sets

```toml
# Core + Graph + HNSW + SONA
rvlite = { path = "crates/rvlite", features = ["core-plus"] }

# Machine Learning stack
rvlite = { path = "crates/rvlite", features = ["ml"] }

# Advanced search capabilities
rvlite = { path = "crates/rvlite", features = ["advanced-search"] }

# Everything
rvlite = { path = "crates/rvlite", features = ["full"] }
```

| Set | Includes |
|-----|----------|
| `core-plus` | graph, hnsw, sona |
| `ml` | gnn, attention, learning, nervous |
| `advanced-search` | hnsw, hyperbolic, math, sparse |
| `full` | All 22 modules |

## CLI Commands (45+)

### Core Operations

| Command | Description |
|---------|-------------|
| `init` | Initialize a new database |
| `insert` | Insert a vector with metadata |
| `embed` | Generate embedding from text |
| `embed-search` | Embed text and search |
| `batch-insert` | Bulk insert from JSON file |
| `search` | Search for similar vectors |
| `get` | Get a vector by ID |
| `delete` | Delete a vector |
| `stats` | Database statistics |
| `export` | Export database to JSON |
| `import` | Import database from JSON |
| `repl` | Interactive REPL |
| `info` | System information |
| `modules` | List all 22 WASM modules |

### Query Engines

| Command | Description |
|---------|-------------|
| `triple` | Add an RDF triple |
| `triples` | List all triples |

### Delta Operations

| Command | Description |
|---------|-------------|
| `delta-compute` | Compute delta between two vectors |
| `delta-apply` | Apply a delta to a stored vector |

### Math & Geometry

| Command | Description |
|---------|-------------|
| `math-distance` | Compute Wasserstein, KL, JS, Hellinger distances |

### Hyperbolic Space

| Command | Description |
|---------|-------------|
| `hyperbolic-embed` | Project vector to Poincare ball or Lorentz hyperboloid |
| `hyperbolic-search` | Search using hyperbolic distance |
| `hyperbolic-midpoint` | Compute Mobius midpoint |

### DAG Workflows

| Command | Description |
|---------|-------------|
| `dag-create` | Create a new DAG |
| `dag-add-node` | Add a node with dependencies |
| `dag-topo-sort` | Topological sort with parallel levels |
| `dag-list` | List all DAGs |

### Neural & Learning

| Command | Description |
|---------|-------------|
| `nervous-simulate` | Simulate a spiking neural network |
| `sparse-analyze` | Analyze vector sparsity patterns |
| `sona-init` | Initialize SONA learning |
| `sona-learn` | Record learning trajectories |
| `sona-apply` | Apply learned patterns |
| `sona-patterns` | View learned patterns |
| `sona-stats` | SONA statistics |
| `attention` | Run attention mechanisms |
| `attention-train` | Train attention weights |

### Routing

| Command | Description |
|---------|-------------|
| `router-route` | Route query to best target using embeddings |

### Federated Learning

| Command | Description |
|---------|-------------|
| `federated-init` | Initialize federated learning |
| `agent-spawn` | Spawn a learning agent |
| `agent-task` | Assign task to agent |
| `federated-aggregate` | Aggregate agent models |
| `federated-stats` | Federated learning statistics |

### WASM & Benchmarks

| Command | Description |
|---------|-------------|
| `wasm` | Check WASM module status |
| `wasm-info` | Detailed WASM module info |
| `benchmark` | Run vector operation benchmarks |
| `benchmark-wasm` | Run WASM-specific benchmarks |

## TypeScript API

```typescript
import { RvLite, SemanticMemory, createRvLite } from 'rvlite';
import type {
  RvLiteConfig,
  SearchResult,
  QueryResult,
  GnnConfig,
  AttentionConfig,
  DeltaOp,
  LearningConfig,
  MathMetric,
  HyperbolicConfig,
  NervousSystemConfig,
  SparseConfig,
  DagNode,
  RouterConfig,
  SonaConfig,
  HnswConfig,
  ModuleInfo,
  EmbeddingProvider,
} from 'rvlite';

// Core
const db = new RvLite({ dimensions: 384, distanceMetric: 'cosine' });
await db.init();

// Module discovery
const modules: ModuleInfo[] = await db.getModules();  // 22 modules
const features: string[] = await db.getFeatures();
const version: string = db.getVersion();  // "0.3.0"

// Vector operations
const id: string = await db.insert([0.1, 0.2], { text: "hello" });
const results: SearchResult[] = await db.search([0.1, 0.2], 5);
const filtered = await db.searchWithFilter([0.1, 0.2], 5, { type: "doc" });
const entry = await db.get(id);
const deleted: boolean = await db.delete(id);
const count: number = await db.len();
const empty: boolean = await db.isEmpty();

// SQL
const sqlResult: QueryResult = await db.sql("SELECT * FROM vectors LIMIT 10");

// Cypher
const graphResult: QueryResult = await db.cypher("CREATE (n:Node {name: 'test'})");
const stats = await db.cypherStats();
await db.cypherClear();

// SPARQL
await db.addTriple("<http://ex.org/s>", "<http://ex.org/p>", "<http://ex.org/o>");
const sparqlResult: QueryResult = await db.sparql("SELECT ?s WHERE { ?s ?p ?o }");
const tripleCount: number = await db.tripleCount();
await db.clearTriples();

// Persistence
await db.save();
await db.exportJson();
await db.importJson(data);
const loaded = await RvLite.load({ dimensions: 384 });
await RvLite.clearStorage();

// Semantic Memory (higher-level API)
const memory = new SemanticMemory(db);
await memory.store("key1", "content", [0.1, 0.2]);
const memories = await memory.query("search text", [0.1, 0.2], 5);
await memory.addRelation("key1", "RELATED_TO", "key2");
const related = await memory.findRelated("key1", 2);
```

## Build

### WASM (default features only)

```bash
cd crates/rvlite
wasm-pack build --target web --release
```

### WASM with specific features

```bash
# Core + Graph + HNSW + SONA
wasm-pack build --target web --release -- --features core-plus

# ML stack
wasm-pack build --target web --release -- --features ml

# All modules
wasm-pack build --target web --release -- --features full
```

### Node.js

```bash
wasm-pack build --target nodejs --release
```

### NPM package

```bash
cd npm/packages/rvlite
npm run build
```

## Size Budget

**Target**: < 3MB gzipped (default features)

| Component | Estimated Size |
|-----------|---------------|
| ruvector-core (vectors, SIMD) | ~500KB |
| SQL parser | ~200KB |
| SPARQL executor | ~300KB |
| Cypher engine | ~600KB |
| Orchestration | ~100KB |
| **Default total** | **~1.7MB** |

With additional features:

| Feature Set | Additional Size |
|-------------|----------------|
| core-plus (graph, hnsw, sona) | +400KB |
| ml (gnn, attention, learning, nervous) | +800KB |
| advanced-search (hnsw, hyperbolic, math, sparse) | +600KB |
| **full (all 22 modules)** | **~4.5MB** |

## Supported Environments

| Environment | Status |
|------------|--------|
| Chrome/Edge 89+ | Supported |
| Firefox 89+ | Supported |
| Safari 15.2+ | Supported |
| Node.js 18+ | Supported |
| Deno 1.30+ | Supported |
| Bun 1.0+ | Supported |
| Cloudflare Workers | Supported |
| Vercel Edge Functions | Supported |

## Documentation

See `crates/rvlite/docs/` for detailed documentation:

| Document | Description |
|----------|-------------|
| `00_EXISTING_WASM_ANALYSIS.md` | Analysis of existing WASM infrastructure |
| `01_SPECIFICATION.md` | Requirements specification |
| `02_API_SPECIFICATION.md` | TypeScript API design |
| `03_IMPLEMENTATION_ROADMAP.md` | Implementation roadmap |
| `04_REVISED_ARCHITECTURE_MAX_REUSE.md` | Max-reuse architecture |
| `05_ARCHITECTURE_REVIEW_AND_VALIDATION.md` | Architecture validation |

## License

MIT OR Apache-2.0
