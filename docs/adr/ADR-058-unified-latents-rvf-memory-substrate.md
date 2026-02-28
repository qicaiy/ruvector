# ADR-058: Unified Latents on RuVector and RVF as an Auditable Latent Memory Substrate

**Status**: Proposed
**Date**: 2026-02-28
**Owner**: rUv
**Deciders**: Architecture Review Board
**Related**: ADR-029 (RVF Canonical Format), ADR-030 (Cognitive Container), ADR-046 (Graph Transformer), ADR-047 (Proof-Gated Mutation), ADR-057 (Federated Transfer Learning)

## Scope

This ADR defines a production architecture for storing, searching, governing, branching, and optionally reconstructing Unified Latents, or UL, inside RuVector and RVF. It uses public RuVector, RVF, and docs.rs material plus the UL paper. UL training itself is external to RuVector. This design packages the trained encoder, prior, and decoder as deployable artifacts and uses RuVector plus RVF for runtime memory, search, governance, and deployment. [1][2][3][4][5]

## Context

Google DeepMind's UL framework jointly regularizes latents with a diffusion prior and decodes them with a diffusion model. The key property for this ADR is that the encoder output noise is linked to the prior's minimum noise level, which gives a tight upper bound on latent bitrate. The paper reports FID 1.4 on ImageNet 512 and FVD 1.3 on Kinetics 600, with fewer training FLOPs than models trained on Stable Diffusion latents. [1]

RuVector and RVF provide the runtime substrate we need. Public documentation describes RuVector as a self learning vector and graph engine with HNSW, GNN reranking, metadata filters, Cypher and SQL surfaces, semantic routing, min cut coherence, MCP integration, and PostgreSQL integration. RVF is a single file substrate that stores vectors, indexes, metadata, graph state, model deltas, witness chains, signatures, a browser runtime, eBPF programs, and a bootable kernel. The public RVF runtime docs expose `RvfStore` for create, ingest, query, delete, compact, and embedding of `KERNEL_SEG` and `EBPF_SEG`, plus metadata, witness, governance, and progressive boot support. [2][3][4][5][6][7][8][9][10][11][12]

## Problem

We want a complete implementation pattern for UL inside RuVector that does all of the following:

1. Stores UL latents for fast retrieval.
2. Preserves enough fidelity for optional reconstruction and generation.
3. Makes every write and sensitive query auditable.
4. Supports branchable experiments and customer specific memory snapshots.
5. Runs on server, browser, edge, or self booting microVM.
6. Uses RuVector specific capabilities, not just plain vector storage.

## Decision

We will implement UL memory in RuVector as a dual lane latent system packaged in RVF.

1. **Index lane**
   Store search optimized UL latent vectors in `VEC_SEG`, indexed by progressive HNSW in `INDEX_SEG`, with quantization codebooks in `QUANT_SEG`. This is the fast retrieval path. [3][6][11]

2. **Archive lane**
   Preserve a higher fidelity latent copy, typically fp16 or fp32 `z0`, for reconstruction. This copy is not the primary ANN surface. It exists for preview, audit, regeneration, and offline evaluation. The practical reason is simple: aggressive quantization is great for recall and memory but can degrade decoder quality. [1][3]

3. **Metadata and filter lane**
   Use `META_SEG` and `META_IDX_SEG` for tenant, modality, UL model version, decoder version, safety labels, bitrate budget, noise level, provenance, and access control tags. RVF runtime supports `MetadataEntry`, `MetadataValue`, metadata filtering, and `FilterExpr` in `QueryOptions`. [3][5][12][13]

4. **Graph lane**
   Use `GRAPH_SEG` to capture asset to asset, frame to clip, near duplicate, semantic cluster, feedback, and workflow edges. Re ranking is done by RuVector's GNN on top of HNSW neighbors. Writes to graph state should be proof gated, following the graph transformer pattern described in the public repo. [2][3][14]

5. **Trust lane**
   Enable automatic witness generation for ingest, delete, compact, and optionally query. Sign segments and lineage with RVF crypto. Use governance policies to constrain write or generation actions. [3][5][8][12][15]

6. **Execution lane**
   Package the same dataset as an RVF cognitive container with:
   1. `WASM_SEG` for browser and offline edge query.
   2. `EBPF_SEG` for hottest vector fast path.
   3. `KERNEL_SEG` for self booting service in Firecracker, TEE, or air gapped deployment. [3][4][9][10]

7. **Learning lane**
   Use RuVector specific learning capabilities after base ANN retrieval:
   1. GNN reranking from usage.
   2. Min cut coherence checks before commit or promotion.
   3. Domain expansion transfer priors for cross domain bootstrapping across image, video, OCR, planning, or other adjacent domains.
   4. Policy kernels and cost curves to choose quality, latency, and generation strategy. [2][3][14][16]

## Why This Decision

This design exploits what UL and RuVector are each good at.

1. UL gives an information disciplined latent with an explicit bitrate handle. [1]
2. RuVector gives vector search, graph structure, self learning reranking, coherence tools, and branchable deployment. [2][3]
3. RVF gives a format that stores data, indexes, models, trust, and runtime in one portable unit. [3][4][5]
4. The public runtime already exposes the right primitives: `RvfStore`, metadata entries, query filters, witness configuration, governance policy, progressive boot, and kernel or eBPF embedding. [5][12][13]

## Architecture

### Logical View

```text
input asset
    |
    v
UL encoder
    |
    +--> z0 archive latent ------------------------------+
    |                                                    |
    +--> z_search quantized latent --> RVF VEC_SEG       |
                                      RVF QUANT_SEG      |
                                      RVF INDEX_SEG      |
                                      RVF META_SEG       |
                                      RVF META_IDX_SEG   |
                                      RVF GRAPH_SEG      |
                                      RVF WITNESS_SEG    |
                                      RVF CRYPTO_SEG     |
                                      RVF HOT_SEG        |
                                      RVF PROFILE_SEG    |
                                      RVF COW segments   |
                                      RVF WASM, EBPF,
                                      RVF KERNEL         |
                                                          |
query image or video frame                                |
    |                                                     |
    v                                                     |
UL encoder                                                |
    |                                                     |
    v                                                     |
RuVector ANN --> GNN rerank --> min cut coherence --> policy gate
    |                                                     |
    +--> retrieval result                                 |
    +--> preview or reconstruct --> UL decoder ----------+
```

### Segment Mapping

| Requirement | RuVector or RVF capability | Primary segment or crate | Purpose |
|---|---|---|---|
| Searchable UL vectors | RVF runtime, progressive HNSW | `VEC_SEG`, `INDEX_SEG`, `rvf-runtime`, `rvf-index` | ANN search with progressive boot [5][6][11] |
| Quantized retrieval | Quantization | `QUANT_SEG`, `rvf-quant` | Reduced memory and fast distance [3][11] |
| High fidelity decode copy | Archive latent lane | `VEC_SEG` or child branch | Reconstruction and audit [1][3] |
| Metadata filters | Metadata entries and filter expressions | `META_SEG`, `META_IDX_SEG`, `MetadataEntry`, `FilterExpr`, `QueryOptions` | Hybrid semantic plus structured filtering [3][5][12][13] |
| Graph reranking | RuVector GNN and graph state | `GRAPH_SEG`, `ruvector-gnn` | Better ranking from context and usage [2][3] |
| Hotset promotion | Temperature tiering | `HOT_SEG` | Keep hottest UL latents near the fast path [3] |
| Audit and provenance | Witness and lineage | `WITNESS_SEG`, `rvf-crypto`, `WITNESS_SEG` support | Query and mutation receipts, lineage [3][8] |
| Signatures and attestation | Segment signing and TEE records | `CRYPTO_SEG`, `rvf-crypto` | Trust and attestation [3][8] |
| Branches and experiments | Copy on write | `COW_MAP_SEG`, `REFCOUNT_SEG`, `MEMBERSHIP_SEG`, `DELTA_SEG` | Cheap tenant or experiment forks [2][3] |
| Browser deployment | WASM runtime | `WASM_SEG`, `@ruvector/rvf-wasm` | Offline search and local preview [2][3] |
| Linux kernel fast path | eBPF | `EBPF_SEG`, `rvf-ebpf` | Hot vector kernel path [3][9] |
| Self booting service | Kernel embedding | `KERNEL_SEG`, `rvf-kernel`, `rvf-launch` | Portable microservice artifact [3][4][10] |
| Query governance | GovernancePolicy | `rvf-runtime::witness` | Restricted, approved, autonomous modes [12][15] |
| Adaptive strategy | Domain expansion and policy kernels | `TRANSFER_PRIOR`, `POLICY_KERNEL`, `COST_CURVE`, `ruvector-domain-expansion` | Cross domain bootstrapping and strategy selection [3][16] |

## Data Model

Each stored UL asset uses the following canonical fields.

### Core Asset Record

| Field | Type | Notes |
|---|---|---|
| `vector_id` | `u64` | Primary vector identity in RVF |
| `asset_id` | string | Stable business identifier |
| `tenant_id` | string | Multi tenant isolation |
| `modality` | enum | image, frame, clip, slide, diagram |
| `source_uri` | string | Original object URI |
| `sha256` | string | Content integrity |
| `encoder_id` | string | UL encoder version |
| `prior_id` | string | UL diffusion prior version |
| `decoder_id` | string | UL decoder version |
| `latent_dim` | integer | Must match RVF store dimension |
| `noise_sigma0` | float | Fixed noise tied to prior minimum noise [1] |
| `bitrate_upper_bound` | float | Stored from UL training or encode side estimator [1] |
| `search_quant` | enum | fp16, int8, int4, binary |
| `safety_class` | enum | public, internal, regulated, restricted |
| `created_at` | timestamp | UTC |
| `branch_id` | string | COW branch lineage |
| `proof_receipt` | string | Graph or write proof identifier |
| `tags` | array | Free labels |

### Edge Record

| Field | Type | Notes |
|---|---|---|
| `src_vector_id` | `u64` | source |
| `dst_vector_id` | `u64` | destination |
| `relation` | enum | near_duplicate, scene_next, same_asset, clicked_after, user_feedback, parent_child |
| `weight` | float | strength |
| `proof_id` | string | mutation proof receipt |
| `created_at` | timestamp | audit |
| `created_by` | string | system or actor |

## Write Path

### Ingest Flow

1. Read asset bytes and normalize.
2. Encode with UL encoder.
3. Produce:
   1. `z0_archive`, a higher fidelity latent for decode.
   2. `z_search`, the quantized latent for ANN.
   3. metadata with UL versioning, noise level, bitrate budget, tenant, and policy tags.
4. Run proof gated validation:
   1. dimension check,
   2. NaN and norm guard,
   3. governance policy check,
   4. coherence gate,
   5. duplicate and near duplicate policy,
   6. optional transfer verification if importing priors from another domain.
5. Call `RvfStore::ingest_batch` for the search lane. [5]
6. Persist archive lane latent in the same file or a linked child branch.
7. Update graph relationships.
8. Append witness receipts and signatures.
9. Promote hot vectors once read frequency crosses threshold.
10. Optionally regenerate or attach `EBPF_SEG` for the hottest subspace.

### Proof Gated Mutation Rule

Every mutation of graph or branch state must satisfy:

```text
state_n
  -> verify invariants
  -> verify policy
  -> verify coherence
  -> verify transfer score if cross domain
  -> append mutation
  -> witness receipt
  -> state_n_plus_1
```

Recommended invariants:

1. latent dimension equals store dimension.
2. bitrate bound is within tenant policy.
3. decoder id exists in registry.
4. branch membership allows visibility.
5. coherence score is above threshold.
6. if transfer prior is applied, target improves and source does not regress.

## Query Path

### Retrieval Only

1. Encode query asset with UL encoder.
2. Search on `z_search` with `RvfStore::query`.
3. Use `QueryOptions` for `ef_search`, metadata `filter`, timeout, and quality preference. [5][13]
4. Apply GNN reranking.
5. Run min cut coherence check.
6. Return IDs, scores, metadata, graph context, and witness receipt.

### Retrieval Plus Preview

1. Run retrieval only path.
2. Load archive latent for top candidates.
3. Send the archive latent to the registered decoder.
4. Return preview bytes or URLs.

### Retrieval Plus Generation

1. Run retrieval plus preview.
2. Only allow decoder or generation tools when `GovernancePolicy` permits it.
3. Cap tool count and cost via `max_tool_calls` and `max_cost_microdollars`.
4. Write a witness receipt for the generation request and output hash. [12][15]

## Search Policy

### Recommended QueryOptions Profile

| Query class | ef_search | quality preference | filter | notes |
|---|---:|---|---|---|
| interactive image search | 64 | balanced | tenant plus modality | low latency |
| compliance audit | 128 | prefer quality | branch plus time range | witness on |
| reconstruction preview | 96 | prefer quality | decoder_id plus safety_class | archive latent required |
| edge offline | 32 | prefer latency | tenant only | use Layer A or B index [11] |

## Governance

Use `WitnessConfig` with all mutation receipts enabled and query audit enabled for regulated or paid reconstruction flows. Public docs show `WitnessConfig` supports flags for ingest, delete, compact, and query auditing. [12]

Use `GovernancePolicy::restricted()` for read only environments, `GovernancePolicy::approved()` for writes with gates, and `GovernancePolicy::autonomous()` only for bounded agent workflows where tool names, cost ceilings, and tool call counts are explicitly set. Public docs show `GovernancePolicy` fields for mode, allowed tools, denied tools, max cost, and max tool calls. [12][15]

Recommended policy defaults:

1. **restricted**
   Retrieval only, no decoder writes, no branch creation.
2. **approved**
   Retrieval, preview, branching, metadata edits, graph updates with proof.
3. **autonomous**
   Same as approved plus bounded generation, hotset promotion, and eBPF refresh.

## Deployment Profiles

### Browser and Offline Edge

Use `WASM_SEG` and `@ruvector/rvlite` or `@ruvector/rvf-wasm` when the decoder is disabled or replaced with preview thumbnails. This gives local search with no backend. [2][3]

### Self Booting Service

Embed a kernel via `RvfStore::embed_kernel` and launch via `rvf-launch`. Public docs show RVF supports a bootable kernel, and `LaunchConfig` exposes `rvf_path`, memory, vcpus, and API port. [4][5][10]

### Linux Hot Path

Compile an eBPF program and embed with `RvfStore::embed_ebpf`. Public docs show `rvf-ebpf` provides compiled programs and constants such as `XDP_DISTANCE`, `SOCKET_FILTER`, and `TC_QUERY_ROUTE`. Use this only for extremely hot and stable lookup shapes. [5][9]

### PostgreSQL Enterprise Mode

For teams that want SQL, mirror the search lane into RuVector Postgres. Public docs describe it as a pgvector replacement with many SQL functions, HNSW support, and self learning capabilities. [2][17]

### Clustered Service

Use RuVector Raft and replication crates for multi node deployment. Public docs show Raft consensus, vector clocks, CRDT style conflict handling, and automatic failover. [2][18]

### Agent Interface

Expose the same RVF artifact through the MCP server for assistants and workflows. Public docs list `@ruvector/rvf-mcp-server` and general MCP integration. [2][3]

## Why Dual Lane Matters

A single latent representation has conflicting objectives.

1. ANN wants compressed, cache friendly vectors.
2. Reconstruction wants fidelity.
3. Audit wants stable lineage and model identity.

Therefore:

1. `z_search` is optimized for lookup.
2. `z_archive` is optimized for fidelity.
3. Both are bound to the same `asset_id`, branch, model versions, and witness chain.

This lets you quantize search aggressively without sacrificing decoder quality.

## RuVector Specific Capabilities Used

This ADR intentionally uses several RuVector and RVF specific capabilities beyond plain ANN.

1. Progressive HNSW boot. `rvf-index` documents Layer A, Layer B, and Layer C loading with increasing recall, so cold starts can answer approximate queries before full load. [11]
2. GNN reranking on top of HNSW neighbors. [2]
3. Min cut coherence as a commit or promotion gate. [2][14]
4. Proof gated mutation from the graph transformer design pattern. [14]
5. COW branching with cheap child artifacts. [2][3]
6. Witness chains and lineage. [2][3][8]
7. Governance policy for tool bounded actions. [12][15]
8. Self booting kernels and browser runtimes in the same file. [3][4]
9. eBPF hot path. [3][5][9]
10. Transfer priors, strategy selection, and cost curves from domain expansion. [16]

## Tradeoff Matrix

| Option | Retrieval latency | Reconstruction fidelity | Governance complexity | Storage cost | Recommended use |
|---|---:|---:|---:|---:|---|
| search lane only | 1 | 0 | 1 | 1 | duplicate search, clustering |
| dual lane without decoder in file | 2 | 2 | 2 | 2 | enterprise retrieval with preview service |
| dual lane with decoder in RVF | 3 | 3 | 3 | 3 | air gapped or portable expert capsule |
| dual lane plus eBPF and kernel | 4 | 3 | 4 | 4 | high QPS appliance style serving |

Scale: 1 is lowest, 4 is highest.

## Operational Checklist

1. Create one RVF store per UL encoder version and dimension.
2. Define a metadata field registry so `field_id` values are stable.
3. Keep archive latents in fp16 or fp32.
4. Quantize only the search lane.
5. Enable witness receipts for all writes.
6. Audit regulated queries.
7. Gate decoder and generation tools with `GovernancePolicy`.
8. Add graph edges only through proof gated code paths.
9. Mirror the hottest classes into eBPF only after observing stable query distributions.
10. Use COW branches for customer, experiment, and compliance freezes.
11. Sign every release artifact.
12. Keep decoder registry immutable per branch.

## Failure Modes and Mitigations

### 1. Latent Version Drift

**Problem**: Queries encoded with `encoder_v2` searched against `encoder_v1` degrade.

**Fix**: One store per encoder version, or explicit alignment branch with witness receipts.

### 2. Quantization Hurts Preview Quality

**Problem**: Decoder receives the compressed search vector.

**Fix**: Always decode from archive lane, never from the aggressively quantized search lane.

### 3. Witness Volume Becomes Too Large

**Problem**: High QPS query audit fills witness chain quickly.

**Fix**: Full query audit only for regulated or paid flows. Use sampled audit for exploratory traffic.

### 4. eBPF Path Becomes Brittle

**Problem**: Kernel fast path assumes fixed dimensions or filters that later change.

**Fix**: Only compile eBPF for stable hot subsets. Rebuild from branch snapshots and keep a fallback user space path.

### 5. Graph Self Learning Corrupts Relevance

**Problem**: Feedback loops strengthen wrong edges.

**Fix**: Require proof receipts, coherence threshold, and rollback on certificate failure.

### 6. Decoder Becomes the Latency Bottleneck

**Problem**: ANN is fast but preview is slow.

**Fix**: Separate retrieval SLA from preview SLA. Cache decoded thumbnails and gate full reconstruction.

## Rollout Plan

### Phase 1: Retrieval Only

1. UL encoder sidecar.
2. RVF search lane.
3. Metadata filters.
4. Witness on ingest.
5. Read only MCP query surface.

### Phase 2: Preview

1. Archive lane.
2. Decoder registry.
3. Governance and query audit.
4. GNN rerank and coherence scoring.

### Phase 3: Portable Expert Capsule

1. Package decoder and service in RVF.
2. Embed kernel.
3. Browser runtime.
4. COW customer branches.

### Phase 4: Hot Path and Self Learning

1. eBPF hotset.
2. Domain expansion priors.
3. Proof gated graph updates.
4. Autonomous policy in bounded environments.

## Acceptance Tests

1. **Duplicate retrieval**: Near duplicate assets appear in top 5 for at least 95 percent of the test set.
2. **Cold boot**: Layer A answers approximate search immediately, then Layer B and Layer C improve recall as the index loads. [11]
3. **Witness integrity**: Recompute chain hashes and verify no receipt mismatch.
4. **Branch fidelity**: Child branch shares parent data and only copies deltas.
5. **Governance**: Decoder and generation calls are denied under restricted mode and allowed under approved mode only when tool and cost budgets pass. [15]
6. **Preview quality**: Reconstruction from archive lane meets your domain threshold.
7. **Fallback safety**: If graph or proof verification fails, retrieval still works from raw ANN without write side mutation.

## Recommended Package Set

### Core Rust

```bash
cargo add rvf-runtime rvf-crypto rvf-index rvf-manifest rvf-import rvf-kernel rvf-ebpf rvf-launch
```

### Node and Browser

```bash
npm install @ruvector/rvf @ruvector/rvf-node @ruvector/rvf-wasm @ruvector/rvf-mcp-server
npm install @ruvector/rvlite
```

### Optional Enterprise and Cluster

```bash
docker pull ruvnet/ruvector-postgres
cargo add ruvector-raft ruvector-replication ruvector-cluster
```

## Reference Implementation

The attached files contain:

1. [`reference_impl_rust.rs`](./unified-latents/reference_impl_rust.rs) — A Rust service side scaffold using documented RVF runtime calls plus thin adapter layers for UL encode and decode.
2. [`reference_impl_typescript.ts`](./unified-latents/reference_impl_typescript.ts) — A Node and edge client scaffold for ingest and query.
3. [`postgres.sql`](./unified-latents/postgres.sql) — A PostgreSQL mirror schema and query examples.

## References

[1] Heek, J., Hoogeboom, E., Mensink, T., and Salimans, T. "Unified Latents (UL): How to train your latents." arXiv, 2026. https://arxiv.org/abs/2602.17270

[2] ruvnet. "RuVector README." GitHub, accessed 2026-02-28. https://github.com/ruvnet/ruvector

[3] ruvnet. "RVF README." GitHub, accessed 2026-02-28. https://github.com/ruvnet/ruvector/blob/main/crates/rvf/README.md

[4] ruvnet. "ADR-030: RVF Cognitive Container." GitHub, 2026-02-14. https://github.com/ruvnet/ruvector/blob/main/docs/adr/ADR-030-rvf-cognitive-container.md

[5] docs.rs. "rvf-runtime crate documentation." accessed 2026-02-28. https://docs.rs/rvf-runtime/latest/rvf_runtime/

[6] docs.rs. "rvf-index crate documentation." accessed 2026-02-28. https://docs.rs/rvf-index/latest/rvf_index/

[7] docs.rs. "rvf-manifest crate documentation." accessed 2026-02-28. https://docs.rs/rvf-manifest/latest/rvf_manifest/

[8] docs.rs. "rvf-crypto crate documentation." accessed 2026-02-28. https://docs.rs/rvf-crypto/latest/rvf_crypto/

[9] docs.rs. "rvf-ebpf crate documentation." accessed 2026-02-28. https://docs.rs/rvf-ebpf/latest/rvf_ebpf/

[10] docs.rs. "rvf-launch LaunchConfig." accessed 2026-02-28. https://docs.rs/rvf-launch/latest/rvf_launch/struct.LaunchConfig.html

[11] docs.rs. "rvf-index progressive indexing." accessed 2026-02-28. https://docs.rs/rvf-index/latest/rvf_index/

[12] docs.rs. "rvf-runtime structs: RvfStore, QueryOptions, MetadataEntry, WitnessConfig, GovernancePolicy." accessed 2026-02-28. https://docs.rs/rvf-runtime/latest/rvf_runtime/

[13] docs.rs. "QueryOptions." accessed 2026-02-28. https://docs.rs/rvf-runtime/latest/rvf_runtime/options/struct.QueryOptions.html

[14] ruvnet. "Graph transformer and proof gated mutation materials." GitHub, accessed 2026-02-28. https://github.com/ruvnet/ruvector

[15] docs.rs. "GovernancePolicy." accessed 2026-02-28. https://docs.rs/rvf-runtime/latest/rvf_runtime/witness/struct.GovernancePolicy.html

[16] ruvnet. "ruvector-domain-expansion README." GitHub, accessed 2026-02-28. https://github.com/ruvnet/ruvector/blob/main/crates/ruvector-domain-expansion/README.md

[17] ruvnet. "RuVector PostgreSQL materials." GitHub, accessed 2026-02-28. https://github.com/ruvnet/ruvector/blob/main/crates/ruvector-postgres/README.md

[18] ruvnet. "Raft and replication materials in RuVector README." GitHub, accessed 2026-02-28. https://github.com/ruvnet/ruvector
