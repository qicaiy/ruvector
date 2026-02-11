# ADR-024: Deployment Architecture & Platform Strategy

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-02-11 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | - |
| **Supersedes** | - |
| **Superseded by** | - |
| **Related** | ADR-005 (WASM Runtime), ADR-003 (SIMD Optimization), ADR-001 (Core Architecture) |

## 1. Context

### 1.1 Problem Statement

The RuVector DNA Analyzer must operate across a spectrum of deployment targets -- from clinical HPC clusters processing thousands of whole genomes daily, to a single browser tab running a point-of-care variant caller at a remote field clinic with no internet. The existing codebase already produces native binaries, WASM modules (18+ `*-wasm` crates), and Node.js bindings (6+ `*-node` crates), but there is no unified deployment strategy that specifies how these artifacts compose into platform-specific pipelines, what capabilities each platform gains or loses, or how cross-platform data flows are coordinated.

### 1.2 Decision Drivers

- Genomics data is privacy-sensitive (HIPAA, GDPR, local regulations); compute-local-to-data is a regulatory advantage.
- Nanopore and short-read sequencers are increasingly deployed outside traditional lab settings.
- FPGA-accelerated basecalling is transitioning from proprietary to open toolchains.
- The MCP protocol enables AI assistants to invoke analysis tools, creating a new interaction paradigm for genomic interpretation.
- A single Rust codebase compiled to multiple targets is already the project's core architectural bet; this ADR formalizes the deployment topology around that bet.

## 2. Platform Matrix

The following matrix defines every supported deployment target, mapping each to its use case, the crate graph it activates, and its concrete constraints.

| Platform | Use Case | Crate Surface | SIMD | Memory | Storage | Network | Limitations |
|----------|----------|---------------|------|--------|---------|---------|-------------|
| **Native x86_64** | Clinical labs, HPC, cloud | Full workspace (80 crates) | AVX-512, AVX2, SSE4.2 | Unbounded | mmap + redb | Full | Server required |
| **Native ARM64** | Edge devices, Apple Silicon, Graviton | Full workspace | NEON, SVE/SVE2 | 4-64 GB typical | mmap + redb | Full | Lower single-core clock |
| **Native RISC-V** | Future processors, academic | Full workspace (feature-gated) | V-extension (draft) | Variable | mmap + redb | Full | Early ecosystem, limited SIMD |
| **WASM Browser** | Point-of-care, education, personal genomics | `ruvector-wasm`, `ruvector-delta-wasm`, `ruvector-attention-wasm`, subset modules | SIMD128 (wasm) | ~4 GB (browser limit) | IndexedDB, OPFS | fetch/WebSocket | No filesystem, no threads without SharedArrayBuffer, cold start |
| **WASM Edge** (Cloudflare Workers, Fastly Compute) | Distributed low-latency analysis, API gateway | `ruvector-wasm` (slim), `ruvector-router-wasm` | SIMD128 | 128 MB (worker limit) | None (stateless) / KV store | HTTP | Cold start (~50ms), no long-running processes, 30s CPU time |
| **WASM WASI** (Wasmtime, WasmEdge) | Sandboxed server-side, plugin host | Full workspace via WASI Preview 2 | SIMD128 + host SIMD via imports | Configurable | WASI filesystem | WASI sockets | Host-dependent capabilities |
| **FPGA** (Xilinx Alveo, Intel Agilex) | Real-time basecalling, kmer hashing | `ruvector-fpga-transformer`, custom bitstream | Custom datapath | HBM (4-32 GB) | Host DMA | PCIe Gen4/5 | Fixed function per bitstream, long synthesis |
| **GPU** (CUDA, ROCm, Metal) | Neural network inference, batch variant calling | `ruvector-attention` + GPU backend | Tensor cores, matrix cores | 8-80 GB VRAM | Host-mapped | PCIe / NVLink | Requires driver stack, batch-oriented |

### 2.1 Platform Tier Definitions

**Tier 1 (Primary)**: Native x86_64, Native ARM64, WASM Browser. Full CI, release binaries, performance regression testing on every merge.

**Tier 2 (Supported)**: WASM Edge, WASM WASI, GPU (CUDA). CI on nightly builds, release artifacts published but with "beta" designation.

**Tier 3 (Experimental)**: Native RISC-V, FPGA, GPU (ROCm, Metal). Community-maintained, CI on demand.

## 3. WASM DNA Analyzer: Browser-Based Variant Calling

### 3.1 Architecture

The browser deployment compiles the analysis pipeline into a layered WASM module set, loaded progressively to minimize time-to-first-interaction.

```
Browser Tab
+-------------------------------------------------------------------+
|  Main Thread (UI)                                                  |
|  +-------------------------------------------------------------+  |
|  | React/Svelte App                                             |  |
|  | - File picker (FASTQ/BAM/VCF)                               |  |
|  | - Genome browser visualization                               |  |
|  | - Variant table with filtering                               |  |
|  +-------------------------------------------------------------+  |
|       |  postMessage            |  postMessage                     |
|       v                        v                                   |
|  +------------------+   +------------------+   +-----------------+ |
|  | Worker 0         |   | Worker 1         |   | Worker N        | |
|  | (Coordinator)    |   | (Analysis)       |   | (Analysis)      | |
|  |                  |   |                  |   |                 | |
|  | ruvector-wasm    |   | ruvector-wasm    |   | ruvector-wasm   | |
|  | (core: 2 MB)     |   | (core: 2 MB)     |   | (core: 2 MB)    | |
|  | + router-wasm    |   | + attention-wasm |   | + gnn-wasm      | |
|  |                  |   |                  |   |                 | |
|  +-------+----------+   +-------+----------+   +--------+--------+ |
|          |                       |                       |          |
|          +-----------+-----------+-----------+------------+         |
|                      |                                              |
|                      v                                              |
|  +-------------------------------------------------------------+  |
|  | SharedArrayBuffer: Reference Genome (hg38, ~750 MB quant.)  |  |
|  +-------------------------------------------------------------+  |
|                      |                                              |
|                      v                                              |
|  +-------------------------------------------------------------+  |
|  | IndexedDB / OPFS                                             |  |
|  | - Cached reference genome chunks                             |  |
|  | - User genome data                                           |  |
|  | - Analysis results (VCF)                                     |  |
|  | - Delta checkpoints (ruvector-delta-wasm)                    |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

### 3.2 Progressive Module Loading

Rather than loading the entire analysis suite upfront, modules are fetched on demand based on the user's workflow.

| Load Phase | Module | Compressed Size | Trigger |
|------------|--------|-----------------|---------|
| **Phase 0** (immediate) | `ruvector-wasm` core (VectorDB + HNSW) | ~2 MB | Page load |
| **Phase 1** (on file open) | `ruvector-router-wasm` (variant routing) | ~800 KB | User opens FASTQ/BAM |
| **Phase 2** (on analysis start) | `ruvector-attention-wasm` (neural caller) | ~3 MB | "Run Analysis" button |
| **Phase 3** (on demand) | `ruvector-gnn-wasm` (graph neural net) | ~2.5 MB | Structural variant mode |
| **Phase 4** (on demand) | `ruvector-delta-wasm` (sync engine) | ~600 KB | "Share Results" action |

Total worst-case download: approximately 9 MB compressed. Each module is independently cacheable via Service Worker with content-hash URLs.

### 3.3 Privacy-First Computation Model

All genomic computation occurs within the browser sandbox. The architecture enforces this structurally:

1. **No server-side data path**: The WASM modules operate on `Float32Array` and `Uint8Array` buffers allocated within the WASM linear memory or JavaScript heap. No API endpoint receives raw genomic data.
2. **SharedArrayBuffer for reference genome**: The reference genome (hg38) is downloaded once, stored in IndexedDB, and mapped into a `SharedArrayBuffer` accessible by all Web Workers. This avoids per-worker copies and stays within the ~4 GB browser memory budget.
3. **IndexedDB persistence**: Analysis state, intermediate results, and delta checkpoints persist locally across sessions. The `ruvector-delta-wasm` crate's `DeltaStream` and `JsDeltaWindow` types (shown in the existing codebase at `/home/user/ruvector/crates/ruvector-delta-wasm/src/lib.rs`) provide event-sourced state management with compaction.
4. **Optional encrypted export**: Results can be exported as encrypted VCF files using SubtleCrypto, shared via peer-to-peer WebRTC, or uploaded to a user-chosen endpoint. The system never mandates server contact.

### 3.4 Web Worker Parallelism

Worker count is determined at runtime via `navigator.hardwareConcurrency`. The coordinator (Worker 0) partitions the genome into regions and distributes work:

```
Regions = chromosome_boundaries(reference)
Workers = navigator.hardwareConcurrency - 1  // reserve 1 for UI
Chunks  = distribute(Regions, Workers, strategy=balanced_by_complexity)

for each Worker w:
    w.postMessage({ type: "analyze", regions: Chunks[w], config })

// Results stream back via postMessage as each region completes
// Coordinator merges and deduplicates calls at region boundaries
```

The `ruvector-wasm` crate already supports `Arc<Mutex<CoreVectorDB>>` internally (see `/home/user/ruvector/crates/ruvector-wasm/src/lib.rs`, line 191), which is safe within a single WASM instance. Cross-worker coordination uses `postMessage` with `Transferable` objects for zero-copy buffer passing.

### 3.5 Quantized Reference Genome

A full hg38 reference is approximately 3.1 GB uncompressed. For browser deployment, the reference is quantized and chunked:

- **2-bit encoding**: Each nucleotide (A, C, G, T) is stored in 2 bits, reducing hg38 to ~775 MB.
- **Block compression**: 64 KB blocks with LZ4 decompression in WASM, yielding ~350 MB on-disk in IndexedDB.
- **On-demand decompression**: Only active regions are decompressed into the SharedArrayBuffer working set.
- **Content-addressed chunks**: Each 1 MB chunk is addressed by SHA-256 hash, enabling incremental download and validation.

## 4. Edge Computing for Field Genomics

### 4.1 Deployment Topology: Nanopore + Laptop

```
+--------------------+        USB3        +---------------------------+
| Oxford Nanopore    | ----- FAST5/POD5 ---> | Laptop (ARM64/x86_64) |
| MinION / Flongle   |                    |                           |
+--------------------+                    |  ruvector-cli             |
                                          |  +-- basecaller (FPGA    |
                                          |  |   or CPU fallback)    |
                                          |  +-- variant caller      |
                                          |  +-- annotation engine   |
                                          |  +-- local web UI        |
                                          |      (WASM analyzer)     |
                                          |                           |
                                          |  ruvector-delta-core      |
                                          |  +-- offline journal     |
                                          |  +-- sync queue          |
                                          +----------+----------------+
                                                     |
                                            (when connected)
                                                     |
                                                     v
                                          +---------------------------+
                                          |  Central Lab Server       |
                                          |  ruvector-server (REST)   |
                                          |  ruvector-cluster         |
                                          |  ruvector-replication     |
                                          +---------------------------+
```

### 4.2 Offline-First Architecture

The field deployment operates under the assumption that network connectivity is intermittent or absent. The architecture enforces this through three mechanisms:

**Local-complete pipeline**: The `ruvector-cli` binary includes the full analysis pipeline. No network call is required to progress from raw signal to annotated VCF. The CLI binary is statically linked and self-contained (~50 MB with all features, ~15 MB stripped for ARM64).

**Delta-based synchronization**: When connectivity returns, the `ruvector-delta-core` crate synchronizes results incrementally. Rather than transferring complete VCF files, only `VectorDelta` objects are transmitted. The existing `HybridEncoding` in `ruvector-delta-wasm` (line 202-206) provides efficient serialization. For a typical variant calling session producing 4-5 million variants, delta sync reduces transfer from ~500 MB to ~12 MB by transmitting only changed positions.

**Conflict resolution**: The `ruvector-replication` crate provides vector clocks (`VectorClock`) and last-write-wins (`LastWriteWins`) conflict resolution strategies (see `/home/user/ruvector/crates/ruvector-replication/src/lib.rs`, line 38). When multiple field laptops analyze overlapping regions, the central server reconciles using configurable merge strategies.

### 4.3 Compressed Reference Genomes

For field deployment where storage is constrained:

| Genome | Uncompressed | Quantized (2-bit + index) | With annotations |
|--------|-------------|---------------------------|------------------|
| Human (hg38) | 3.1 GB | 775 MB | 950 MB |
| Malaria (Pf3D7) | 23 MB | 6 MB | 12 MB |
| SARS-CoV-2 | 30 KB | 8 KB | 45 KB |
| Custom panel (targeted) | Variable | Variable | < 100 MB typical |

The quantized human reference at 950 MB with annotations fits comfortably on any modern laptop, eliminating the need for network access to reference data.

## 5. FPGA Pipeline for Basecalling

### 5.1 Architecture

The `ruvector-fpga-transformer` crate (at `/home/user/ruvector/crates/ruvector-fpga-transformer/src/lib.rs`) provides the software interface to FPGA-accelerated inference. The architecture separates the concern of model execution from hardware specifics through the `TransformerBackend` trait:

```
                         +-----------------------------+
                         |  ruvector-fpga-transformer   |
                         |  Engine                      |
                         |  +-- load_artifact()         |
                         |  +-- infer()                 |
                         |  +-- CoherenceGate            |
                         +------+-------+---------+-----+
                                |       |         |
                   +------------+   +---+---+  +--+----------+
                   |                |       |  |              |
            +------v------+ +------v--+ +--v--v------+ +-----v-------+
            | FpgaPcie    | | FpgaDaemon| | NativeSim | | WasmSim     |
            | (pcie feat) | | (daemon)  | | (native)  | | (wasm feat) |
            |             | |           | |           | |             |
            | DMA ring    | | Unix sock | | Pure Rust | | wasm_bindgen|
            | BAR0/BAR1   | | /gRPC     | | simulator | | browser sim |
            +-------------+ +-----------+ +-----------+ +-------------+
                   |              |
                   v              v
            +----------------------------+
            | FPGA Hardware              |
            | +-- Convolution engine     |
            | +-- Multi-head attention   |
            | +-- CTC decoder           |
            | +-- Quantized matmul      |
            | +-- LUT-based softmax     |
            +----------------------------+
```

### 5.2 Basecalling Pipeline Stages

The FPGA implements a fixed-function pipeline for nanopore basecalling:

| Stage | Operation | FPGA Implementation | Throughput Target |
|-------|-----------|--------------------|--------------------|
| 1 | Signal normalization | Streaming mean/variance, INT16 | Line rate |
| 2 | Convolution layers | Systolic array, INT8 weights | 10 TOPS |
| 3 | Multi-head attention | Custom attention kernel with early exit | 5 TOPS |
| 4 | CTC decode | Beam search with hardware prefix tree | 100 Mbases/s |
| 5 | Quality scoring | LUT-based Phred computation | Line rate |

The existing `FixedShape` type (at `/home/user/ruvector/crates/ruvector-fpga-transformer/src/types.rs`, line 58) constrains all dimensions at model-load time, enabling the FPGA synthesis tool to generate optimized datapaths. The `QuantSpec` type carries INT4/INT8 quantization metadata that maps directly to FPGA arithmetic units.

### 5.3 Performance Targets

| Metric | GPU Baseline (A100) | FPGA Target (Alveo U250) | Speedup |
|--------|--------------------|-----------------------------|---------|
| Basecalling throughput | 1 Gbases/s | 10 Gbases/s | 10x |
| Latency per read (1000 bp) | 2 ms | 0.2 ms | 10x |
| Power consumption | 300 W | 75 W | 4x better |
| Batch requirement | 32+ reads | 1 read (streaming) | Real-time capable |

### 5.4 Programmable Pipeline: Model Updates Without Hardware Changes

The `ModelArtifact` system (defined in `/home/user/ruvector/crates/ruvector-fpga-transformer/src/artifact/`) enables model updates without FPGA re-synthesis:

1. **Artifact format**: Signed bundles containing quantized weights, shape metadata, and optional FPGA bitstream.
2. **Weight-only update**: When the model architecture is unchanged, only new `QuantizedTensor` weights are loaded via DMA. The FPGA datapath is reused. Latency: ~200 ms.
3. **Bitstream update**: When architectural changes are needed (new layer types, different attention mechanism), a new bitstream is loaded via partial reconfiguration. Latency: ~2 seconds.
4. **Ed25519 signature verification**: Every artifact is cryptographically signed. The `verify` module in the artifact subsystem validates signatures before any weights reach the FPGA.

### 5.5 Wire Protocol

Communication between host and FPGA uses the binary protocol defined in `/home/user/ruvector/crates/ruvector-fpga-transformer/src/backend/mod.rs`:

- Magic: `0x5256_5846` ("RVXF")
- 24-byte request header (`RequestFrame`) with sequence length, model dimension, vocabulary size, model ID, and flags
- 14-byte response header (`ResponseFrame`) with status, latency, cycles, and gate decision
- CRC32 integrity checking on all frames
- DMA ring buffer with 16 slots of 64 KB each for the PCIe backend (`PcieConfig` defaults)

## 6. Distributed Analysis via ruvector-cluster

### 6.1 Cluster Topology

```
                        +-------------------+
                        |  Load Balancer    |
                        |  (L7 / gRPC-LB)  |
                        +--------+----------+
                                 |
                 +---------------+---------------+
                 |               |               |
          +------v------+ +-----v-------+ +-----v-------+
          | Node 1      | | Node 2      | | Node 3      |
          | (Leader)    | | (Follower)  | | (Follower)  |
          |             | |             | |             |
          | ruvector-   | | ruvector-   | | ruvector-   |
          |  server     | |  server     | |  server     |
          | ruvector-   | | ruvector-   | | ruvector-   |
          |  cluster    | |  cluster    | |  cluster    |
          | ruvector-   | | ruvector-   | | ruvector-   |
          |  raft       | |  raft       | |  raft       |
          |             | |             | |             |
          | Shards:     | | Shards:     | | Shards:     |
          | [0,5,10,..] | | [1,6,11,..] | | [2,7,12,..] |
          +------+------+ +------+------+ +------+------+
                 |               |               |
                 +-------+-------+-------+-------+
                         |               |
                  +------v------+ +------v------+
                  | Replica A   | | Replica B   |
                  | (async)     | | (async)     |
                  +-------------+ +-------------+
```

### 6.2 Genome Sharding Strategy

The `ConsistentHashRing` in `ruvector-cluster` (at `/home/user/ruvector/crates/ruvector-cluster/src/shard.rs`, line 16) uses 150 virtual nodes per physical node for balanced distribution. For genome analysis, sharding follows a domain-aware strategy:

| Sharding Dimension | Strategy | Rationale |
|--------------------|----------|-----------|
| By chromosome | Range-based (chr1-22, X, Y, MT = 25 shards) | Locality for structural variant calling |
| By sample | Hash-based (jump consistent hash) | Even distribution across nodes |
| By analysis stage | Pipeline-based (align -> call -> annotate) | Stage-specific resource allocation |

For a whole-genome cohort study of 1000 samples across a 5-node cluster:

- 64 shards (default `ClusterConfig.shard_count`), replication factor 3
- Each node holds ~38 shard replicas (64 * 3 / 5)
- Chromosome-aware routing ensures all data for a given chromosome on a given sample co-locates

### 6.3 Consensus and State Management

The `ruvector-raft` crate (at `/home/user/ruvector/crates/ruvector-raft/src/lib.rs`) provides Raft consensus for distributed metadata:

- **What Raft manages**: Cluster membership, shard assignments, analysis pipeline state, schema metadata. NOT the genomic data itself.
- **What the data plane manages**: Genomic vectors flow through the `ShardRouter` directly, bypassing consensus for read operations. Writes go through the leader for ordering.
- **Failover**: The `ruvector-replication` crate's `FailoverManager` with `FailoverPolicy` handles primary promotion. Split-brain prevention uses the Raft quorum: a partition with fewer than `min_quorum_size` nodes (default 2) becomes read-only.

### 6.4 REST/gRPC API

The `ruvector-server` crate (at `/home/user/ruvector/crates/ruvector-server/src/lib.rs`) exposes an axum-based REST API on port 6333 (default). For DNA analysis, the API surface extends to:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health`, `/ready` | GET | Liveness and readiness probes |
| `/collections` | CRUD | Manage genome collections (samples, panels) |
| `/collections/{name}/points` | POST | Insert variant vectors |
| `/collections/{name}/search` | POST | Similarity search (find related variants) |
| `/analysis/submit` | POST | Submit analysis job (FASTQ -> VCF pipeline) |
| `/analysis/{id}/status` | GET | Job progress and status |
| `/analysis/{id}/results` | GET | Stream results as they complete |

Compression (`CompressionLayer`) and CORS (`CorsLayer`) are enabled by default.

## 7. MCP Integration: AI-Powered Genomic Interpretation

### 7.1 Architecture

The `mcp-gate` crate (at `/home/user/ruvector/crates/mcp-gate/src/lib.rs`) provides an MCP server that currently exposes coherence gate tools. For the DNA Analyzer, this extends to expose genomic analysis as MCP tools callable by AI assistants:

```
+------------------+       JSON-RPC/stdio       +-------------------+
| AI Assistant     | <========================> | mcp-gate          |
| (Claude, etc.)   |                            |                   |
|                  |    tools/call:              | Tools:            |
| "What variants  |    - permit_action          | - permit_action   |
|  in BRCA1 are   |    - get_receipt            | - get_receipt     |
|  pathogenic?"   |    - replay_decision        | - replay_decision |
|                  |    - search_variants  [NEW] | - search_variants |
|                  |    - annotate_variant [NEW] | - annotate_variant|
|                  |    - run_pipeline    [NEW]  | - run_pipeline    |
+------------------+                            +--------+----------+
                                                         |
                                                         v
                                               +---------+----------+
                                               | ruvector-core      |
                                               | ruvector-server    |
                                               | Analysis Pipeline  |
                                               +--------------------+
```

### 7.2 Genomic MCP Tools

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `search_variants` | Gene name, region, filters | Matching variant vectors with clinical annotations | "Find all ClinVar pathogenic variants in TP53" |
| `annotate_variant` | Chromosome, position, ref, alt | Functional impact, population frequency, clinical significance | "What is the impact of chr17:7674220 G>A?" |
| `run_pipeline` | FASTQ/BAM reference, analysis parameters | Job ID, streaming status | "Analyze this patient's exome against hg38" |
| `compare_samples` | Two sample IDs, region | Delta vectors showing differences | "How do these tumor/normal samples differ in chr9?" |

### 7.3 Coherence Gate for Genomic Decisions

The existing `TileZero` coherence gate (re-exported by `mcp-gate`) provides a safety layer for AI-driven genomic interpretation:

- **permit_action** with `action_type: "clinical_interpretation"` requires higher coherence thresholds than exploratory queries.
- **Witness receipts** create an auditable trail of every AI-assisted interpretation, critical for clinical compliance.
- **Replay capability** allows regulatory review of any AI-generated interpretation by deterministically replaying the decision with its original context.

## 8. Platform-Specific Optimization Strategies

### 8.1 x86_64 Optimizations

- **AVX-512 distance calculations**: The `simsimd` dependency (workspace `Cargo.toml`, line 96) auto-detects and uses the widest SIMD available. For 384-dimensional variant embeddings, AVX-512 processes 16 floats per cycle.
- **Memory-mapped storage**: `memmap2` (line 94) provides zero-copy access to genome indices. For a 64-shard cluster node holding 200 GB of variant data, mmap avoids loading the entire dataset into RAM.
- **Release profile**: `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` (workspace `Cargo.toml`, lines 151-156) produce maximally optimized binaries.

### 8.2 ARM64 Optimizations

- **NEON SIMD**: Automatic fallback from AVX-512 to NEON via `simsimd` runtime detection. NEON processes 4 floats per cycle (128-bit registers).
- **SVE/SVE2** (Graviton3+, Apple M4+): Scalable vector extension with variable-width registers (128-2048 bit). The `ruvector-core` distance functions are written to auto-vectorize under SVE.
- **Static linking**: ARM64 field deployments use `RUSTFLAGS="-C target-feature=+neon"` and static musl linking for a single self-contained binary.

### 8.3 WASM Optimizations

- **SIMD128**: The `detect_simd()` function in `ruvector-wasm` (line 426) detects WASM SIMD support. When available, distance calculations use `v128` operations providing 4x throughput over scalar.
- **Streaming compilation**: WASM modules use `WebAssembly.compileStreaming()` for parallel download and compilation.
- **Memory management**: The `MAX_VECTOR_DIMENSIONS` constant (line 98, set to 65536) prevents allocation bombs. Vector dimensions are validated before any WASM memory allocation.
- **wasm-opt**: All WASM modules pass through Binaryen's `wasm-opt -O3 --enable-simd` in the release build pipeline.

### 8.4 FPGA Optimizations

- **Zero-allocation hot path**: The `Engine::infer()` method (line 170) performs no heap allocations during inference. All buffers are pre-allocated at `load_artifact()` time.
- **INT4/INT8 quantization**: The `QuantSpec` type carries explicit quantization metadata. INT4 weights halve memory bandwidth requirements on the FPGA datapath.
- **LUT-based softmax**: The `LUT_SOFTMAX` flag (protocol flags, line 90) triggers hardware lookup-table softmax, avoiding expensive exponential computation.
- **Early exit**: The `EARLY_EXIT` flag (line 92) enables the coherence gate to terminate inference early when confidence exceeds a threshold, saving cycles.

## 9. Cross-Platform Data Flow

### 9.1 End-to-End Flow: Field to Cloud

```
  Field Laptop          |  Transit          |  Central Lab
  (ARM64, offline)      |  (delta sync)     |  (x86_64 cluster)
                        |                   |
  Nanopore -> basecall  |                   |
  -> align -> call      |                   |
  -> annotate           |                   |
  -> local VCF + deltas |                   |
       |                |                   |
       | (connectivity) |                   |
       +--------------->| ruvector-delta    |
                        | (compressed VectorDelta) |
                        +------------------>| merge via
                        |                   | ruvector-replication
                        |                   | -> cluster-wide
                        |                   |    variant database
                        |                   |
                        |                   | AI assistant queries
                        |                   | via mcp-gate
                        |                   | -> clinical report
```

### 9.2 Data Format Compatibility

All platforms produce and consume the same serialized formats:

| Data Type | Format | Crate | Cross-Platform |
|-----------|--------|-------|---------------|
| Vector embeddings | `Float32Array` / `Vec<f32>` | `ruvector-core` | All platforms |
| Delta updates | `HybridEncoding` (sparse + dense) | `ruvector-delta-core` / `ruvector-delta-wasm` | All platforms |
| Model artifacts | Signed bundle (manifest + weights + bitstream) | `ruvector-fpga-transformer::artifact` | All platforms (FPGA bitstream optional) |
| API payloads | JSON (REST) / Protobuf (gRPC) | `ruvector-server` | All platforms with network |
| MCP messages | JSON-RPC 2.0 over stdio | `mcp-gate` | All platforms with stdio |

## 10. Deployment Topologies

### 10.1 Single Laptop (Field Genomics)

```
Components: ruvector-cli (static binary)
Storage: Local filesystem (mmap)
Network: None required
Capacity: ~10 whole genomes / day (ARM64), ~30 / day (x86_64)
```

### 10.2 Clinical Lab (3-5 Nodes)

```
Components: ruvector-server + ruvector-cluster + ruvector-raft
Storage: NVMe SSDs, mmap
Network: 10 GbE / 25 GbE
Capacity: ~500 whole genomes / day
Fault tolerance: 1 node failure (replication factor 3)
```

### 10.3 Research HPC (50+ Nodes)

```
Components: ruvector-cluster (64+ shards) + FPGA accelerators + GPU nodes
Storage: Parallel filesystem (Lustre/GPFS) + local NVMe
Network: InfiniBand / 100 GbE
Capacity: ~10,000 whole genomes / day
Specialization: FPGA nodes for basecalling, GPU nodes for deep learning, CPU nodes for alignment
```

### 10.4 Global Edge Network

```
Components: WASM Edge workers (Cloudflare/Fastly) + central API
Storage: Edge KV for cached references, central cluster for results
Network: CDN-accelerated
Use case: Low-latency variant lookup API, privacy-preserving query routing
Capacity: 100,000+ queries/second globally
```

### 10.5 Browser-Only (Personal Genomics)

```
Components: ruvector-wasm + ruvector-delta-wasm + ruvector-attention-wasm
Storage: IndexedDB / OPFS
Network: Optional (reference genome download, result sharing)
Capacity: 1 exome / session, panel analysis in < 30 seconds
Privacy: All computation local, no data leaves the browser
```

## 11. Build and Release Strategy

### 11.1 Artifact Matrix

| Target | Build Command | Output | Size |
|--------|--------------|--------|------|
| Linux x86_64 | `cargo build --release` | `ruvector-cli` | ~25 MB |
| Linux ARM64 | `cross build --release --target aarch64-unknown-linux-musl` | `ruvector-cli` | ~20 MB |
| macOS ARM64 | `cargo build --release --target aarch64-apple-darwin` | `ruvector-cli` | ~22 MB |
| WASM (browser) | `wasm-pack build --release --target web` | `*.wasm` + JS glue | ~2 MB (core) |
| WASM (Node) | `wasm-pack build --release --target nodejs` | `*.wasm` + JS glue | ~2.5 MB (core) |
| WASM (edge) | `wasm-pack build --release --target web --features slim` | `*.wasm` | ~800 KB |
| npm (Node.js bindings) | `napi build --release` | `*.node` | ~15 MB |
| Docker | `docker build -t ruvector .` | Container image | ~50 MB (distroless) |
| FPGA bitstream | `make synthesis BOARD=alveo_u250` | `.xclbin` | ~30 MB |

### 11.2 CI Matrix

Every PR runs Tier 1 targets. Nightly builds include Tier 2. Release builds include all tiers.

## 12. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Browser memory limit blocks whole-genome analysis | High | Medium | Streaming analysis with region windowing; progressive result construction |
| WASM SIMD not available in older browsers | Medium | Low | Scalar fallback path; feature detection via `detect_simd()` |
| FPGA synthesis time blocks rapid iteration | High | Medium | NativeSim and WasmSim backends for development; FPGA only for production deployment |
| Edge worker cold start exceeds latency SLA | Medium | Medium | Pre-warming via cron triggers; keep-alive requests; minimal WASM module size |
| Cross-platform delta incompatibility | Low | High | `HybridEncoding` is platform-independent; fuzz-tested across all targets |
| SharedArrayBuffer disabled by browser security policy | Medium | High | Fallback to per-worker copies with memory pressure monitoring; COOP/COEP headers documented |

## 13. Decision

We adopt the multi-platform deployment architecture described above, with the following key commitments:

1. **Single codebase, multiple targets**: The existing pattern of `*-wasm`, `*-node`, and native crates continues. No platform-specific forks.
2. **Progressive capability**: Each platform gets the maximum capability its constraints allow, degrading gracefully.
3. **Privacy by architecture**: Browser and edge deployments are structurally incapable of leaking genomic data to servers.
4. **FPGA as acceleration, not dependency**: The `TransformerBackend` trait ensures every pipeline runs on CPU; FPGA is a 10x acceleration option, not a requirement.
5. **Delta-first synchronization**: All cross-node and cross-platform data exchange uses `ruvector-delta-*` for bandwidth efficiency.
6. **MCP as the AI integration surface**: Genomic analysis is exposed as MCP tools, enabling AI assistants to interpret results within the coherence gate's safety framework.

## 14. Consequences

**Positive**: The architecture enables RuVector DNA Analyzer to serve clinical labs, field researchers, personal genomics users, and AI-powered interpretation pipelines from a single Rust codebase. The progressive loading strategy keeps browser deployments fast. The FPGA pipeline provides a clear path to 10x throughput. The MCP integration positions the system for AI-native genomics workflows.

**Negative**: Maintaining 80+ crates across 7+ targets increases CI complexity and build times. FPGA synthesis remains a bottleneck for hardware iteration. Browser memory limits constrain whole-genome analysis to streaming approaches. The coherence gate adds latency to MCP-mediated interpretations.

**Neutral**: The platform tier system (Tier 1/2/3) acknowledges that not all targets receive equal investment, aligning engineering effort with user demand.
