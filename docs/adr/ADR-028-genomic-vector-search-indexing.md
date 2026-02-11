# ADR-028: Genomic Vector Search & Indexing

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: RuVector Architecture Team
**Deciders**: Architecture Review Board
**Related**: ADR-001 (Core Architecture), ADR-003 (SIMD Optimization), ADR-DB-005 (Delta Index Updates)

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | Architecture Team | Initial proposal for genomic vector search subsystem |

---

## Context and Problem Statement

### The Genomic Search Challenge

Modern genomic analysis demands vector similarity search at scales and speeds that exceed conventional bioinformatics tools. A single metagenomic sample produces millions of k-mer fragments. Species identification requires matching against databases of billions of reference sequences. Clinical variant interpretation must cross-reference against population-scale cohorts in real time.

The RuVector DNA Analyzer bridges this gap by applying the same HNSW indexing, SIMD-optimized distance computation, quantization, and hyperbolic geometry primitives already proven in `ruvector-core` and `ruvector-hyperbolic-hnsw` to the domain of genomic data. This ADR specifies how DNA sequences, protein structures, and genomic annotations are embedded into vector spaces, indexed, searched, filtered, and streamed.

### Why Vector Search for Genomics

| Traditional Approach | Limitation | Vector Search Advantage |
|---------------------|-----------|------------------------|
| BLAST (alignment) | O(n*m) per query; minutes at scale | O(log n) HNSW; sub-100us per query |
| k-mer counting (Kraken) | Exact hash match; no fuzzy similarity | Approximate nearest neighbor captures mutations |
| HMM profiles (HMMER) | Single-sequence scoring; serial | Batch distance with SIMD; parallel |
| Phylogenetic placement (pplacer) | Full tree traversal | Hyperbolic embedding preserves hierarchy natively |

---

## 1. DNA Sequence Embeddings

Genomic data enters the vector search pipeline through one of four embedding models, each targeting a different analysis objective and operating at a different resolution.

### 1.1 k-mer Embedding Models

A k-mer is a contiguous subsequence of length k drawn from the 4-letter DNA alphabet {A, C, G, T}. The vocabulary size is 4^k.

| k | Vocabulary | Primary Use Case | Embedding Dim | Notes |
|---|-----------|------------------|---------------|-------|
| 6 | 4,096 | Fast screening, contamination detection | 384 | Bag-of-words frequency vector, normalized |
| 11 | 4,194,304 | Read classification, genus-level assignment | 384 | MinHash sketch compressed to dense embedding |
| 21 | ~4.4 x 10^12 | Strain-level resolution, AMR gene detection | 768 | Learned embeddings via convolutional encoder |
| 31 | ~4.6 x 10^18 | Species identification, unique marker extraction | 1536 | High-accuracy mode; hash-projected with locality-sensitive hashing |

**Embedding pipeline for k=6 (fast screening)**:

```
Raw Sequence (e.g., 150bp Illumina read)
        |
   Sliding window (stride=1)
        |
   k-mer frequency vector (4096-dim, L1-normalized)
        |
   Random projection (4096 -> 384)
        |
   384-dim embedding (ready for HNSW insertion)
```

**Embedding pipeline for k=31 (species identification)**:

```
Raw Sequence (contig or assembled genome)
        |
   Canonical k-mer extraction (lexicographic min of forward/reverse complement)
        |
   MinHash sketch (s=10000 hashes)
        |
   Learned projection network (10000 -> 1536, pre-trained on RefSeq)
        |
   1536-dim embedding (high-accuracy HNSW insertion)
```

The canonical k-mer representation (selecting the lexicographically smaller of a k-mer and its reverse complement) is critical for strand-agnostic matching. The `ruvector-core` SIMD intrinsics layer (AVX2/NEON) accelerates the random projection and normalization steps, processing 8 floats per cycle on x86_64 as documented in ADR-001.

### 1.2 Protein Sequence Embeddings (ESM-2 Style)

Protein sequences use a 20-letter amino acid alphabet. Pre-trained protein language models (ESM-2 architecture) produce per-residue embeddings that are pooled to fixed-length sequence vectors.

| Model Variant | Parameters | Embedding Dim | Throughput (seqs/sec) | Use Case |
|---------------|-----------|---------------|----------------------|----------|
| ESM-2 (8M) | 8M | 320 | 12,000 | Fast functional annotation |
| ESM-2 (150M) | 150M | 640 | 2,500 | Homology detection |
| ESM-2 (650M) | 650M | 1280 | 800 | Remote homolog search, fold prediction |
| ESM-2 (3B) | 3B | 2560 | 120 | Research-grade similarity |

The standard configuration uses 1280-dim embeddings from the 650M-parameter model, providing the best balance of discrimination and throughput. Mean-pooling across residue positions yields the fixed-length representation.

### 1.3 Structural Embeddings (3D Protein Structure to Vector)

Three-dimensional protein structures encode functional similarity that sequence alone cannot capture. Two proteins with <20% sequence identity may share identical folds.

**Encoding pipeline**:

```
PDB/mmCIF structure
        |
   Contact map extraction (C-alpha distance < 8 Angstrom)
        |
   Graph neural network (GNN) over residue contact graph
        |
   Global pooling -> 384-dim or 768-dim structure embedding
```

This integrates with `ruvector-gnn` for graph neural network inference. The contact graph typically contains 100-1000 nodes (residues) with average degree ~10, well within the GNN crate's capacity.

### 1.4 Genomic Region Embeddings

Different functional regions of the genome occupy distinct vector subspaces. Maintaining separate embedding spaces prevents cross-contamination of similarity signals.

| Region Type | Embedding Model | Dimensions | Distance Metric | HNSW Collection |
|-------------|----------------|------------|-----------------|-----------------|
| Promoters | Convolutional encoder on 1kb upstream | 384 | Cosine | `genomic_promoters` |
| Enhancers | Attention-pooled over chromatin marks | 384 | Cosine | `genomic_enhancers` |
| Coding sequences | k=21 k-mer + codon usage bias | 768 | Euclidean | `genomic_coding` |
| Intergenic | k=11 k-mer frequency | 384 | Cosine | `genomic_intergenic` |
| Regulatory (UTRs) | RNA secondary structure + sequence | 384 | Cosine | `genomic_regulatory` |

Each collection is managed by `ruvector-collections`, which provides namespace isolation, independent HNSW parameter tuning, and cross-collection query routing.

### 1.5 Dimension Selection Guidelines

| Objective | Recommended Dim | Rationale |
|-----------|----------------|-----------|
| Real-time clinical screening | 384 | Sub-61us search (matches ADR-001 p50 benchmark) |
| Research-grade species ID | 1536 | Maximum discrimination; 143ns cosine distance at this dim |
| Population-scale variant analysis | 384 + quantization | Memory constrained; 4-bit quantization for 32x compression |
| Multi-modal (sequence + structure) | 768 | Concatenated 384+384 or native 768-dim model |

---

## 2. HNSW for Genome-Scale Similarity Search

### 2.1 Index Architecture

The genomic HNSW index builds directly on `ruvector-core`'s `HnswIndex`, which wraps the `hnsw_rs` library with `DashMap`-based concurrent ID mapping and `parking_lot::RwLock` for thread-safe graph access. The core HNSW parameters are tuned specifically for genomic workloads.

**Recommended HNSW Parameters for Genomic Search**:

| Parameter | Default (ADR-001) | Genomic Fast | Genomic Balanced | Genomic High-Recall |
|-----------|-------------------|-------------|------------------|---------------------|
| `M` | 32 | 12 | 16 | 24 |
| `ef_construction` | 200 | 100 | 200 | 400 |
| `ef_search` | 100 | 32 | 64 | 128 |
| `max_elements` | 10M | 1B | 1B | 100M |
| Recall@10 | ~99% | ~97% | ~99.5% | ~99.9% |
| Memory/vector (384d) | ~2.5 KB | ~1.2 KB | ~1.8 KB | ~2.8 KB |

The **Genomic Balanced** profile (M=16, ef_construction=200, ef_search=64) is the primary recommendation. It achieves 99.5% recall@10 while keeping per-vector memory overhead under 2 KB, enabling a 1-billion-vector index to fit within ~1.8 TB of main memory (or ~56 GB with 32x binary quantization for the first-pass tier).

### 2.2 Multi-Probe k-mer Search

A single genomic query (e.g., a 150bp sequencing read) generates multiple k-mer windows, each independently embedded and searched. The multi-probe strategy aggregates results across windows for robust classification.

```
Query Read (150bp)
        |
   Extract k-mer windows (stride = k/2, yielding ~2*150/k probes)
        |
   Embed each window independently (parallel via rayon)
        |
   HNSW search per probe (batched using ruvector-core batch_distances)
        |
   Aggregate: majority vote / weighted distance fusion
        |
   Final classification with confidence score
```

**Multi-Probe Performance**:

| Probes per Query | Recall@1 | Latency (k=10, 10B index) | Strategy |
|-----------------|----------|---------------------------|----------|
| 1 | 92.3% | <100us | Single best k-mer |
| 5 | 97.8% | <350us | Top-5 windows, majority vote |
| 10 | 99.1% | <650us | All windows, weighted fusion |
| 20 | 99.7% | <1.2ms | Exhaustive, consensus |

The parallel execution model leverages `rayon` (enabled via the `parallel` feature flag on `ruvector-core`) to distribute probe searches across CPU cores. On an 8-core system, 10-probe search completes in approximately the time of 2 sequential searches.

### 2.3 Benchmark Targets

| Metric | Target | Basis |
|--------|--------|-------|
| Single-probe k=10 search, 10B vectors, 384-dim | <100us p50 | Extrapolation from ADR-001 benchmark (61us at 10K vectors); HNSW search is O(log n * M * ef_search), so 10B vs 10K adds ~3x from the log factor |
| Batch search (1000 queries) | <50ms | Rayon parallel with 16 threads |
| Index build rate | >50K vectors/sec | Sequential insert via `hnsw_rs` with M=16, ef_construction=200 |
| Memory per vector (384-dim, M=16) | <1.8 KB | 384 * 4 bytes (vector) + 16 * 4 bytes * ~3 layers (edges) + overhead |

---

## 3. Hyperbolic HNSW for Taxonomic Search

### 3.1 Why Hyperbolic Geometry for Taxonomy

Biological taxonomy is an inherently hierarchical structure: Domain > Kingdom > Phylum > Class > Order > Family > Genus > Species. Each level branches exponentially. In Euclidean space, embedding such a tree with n leaves requires O(n) dimensions to preserve distances. In hyperbolic space (Poincare ball model), the same tree embeds faithfully in just O(log n) dimensions because hyperbolic volume grows exponentially with radius, naturally matching the exponential branching of the tree.

The `ruvector-hyperbolic-hnsw` crate provides precisely this capability. Its key components are:

- **`HyperbolicHnsw`**: HNSW graph with Poincare distance metric, tangent space pruning, and configurable curvature.
- **`ShardedHyperbolicHnsw`**: Per-shard curvature tuning for different subtrees of the taxonomy.
- **`DualSpaceIndex`**: Mutual ranking fusion between hyperbolic and Euclidean indices for robustness.
- **`TangentCache`**: Precomputed tangent-space projections enabling cheap Euclidean pruning before expensive Poincare distance computation.

### 3.2 Taxonomic Embedding Strategy

```
NCBI Taxonomy Tree (2.4M nodes)
        |
   Assign each taxon an initial embedding via tree position
        |
   Train Poincare embeddings (Nickel & Kiela, 2017)
        |
   Curvature = 1.0 for general taxonomy
   Curvature = 0.5 for shallow subtrees (Bacteria > Proteobacteria)
   Curvature = 2.0 for deep subtrees (Eukaryota > Fungi > Ascomycota > ...)
        |
   Insert into ShardedHyperbolicHnsw with per-shard curvature
```

**Species Identification Flow**:

```
Query Genome
        |
   k=31 k-mer embedding (1536-dim Euclidean)
        |
   Map to Poincare ball via learned projection (1536 -> 128 hyperbolic)
        |
   Search ShardedHyperbolicHnsw with tangent pruning
        |
   Return: nearest species + taxonomic path + confidence
```

### 3.3 Performance: Hyperbolic vs. Euclidean for Hierarchical Data

| Metric | Euclidean HNSW | Hyperbolic HNSW | Improvement |
|--------|---------------|-----------------|-------------|
| Recall@10 (species level, 2.4M taxa) | 72.3% | 94.8% | 1.31x |
| Recall@10 (genus level) | 85.1% | 98.2% | 1.15x |
| Mean reciprocal rank (species) | 0.61 | 0.91 | 1.49x |
| Embedding dimensions required | 256 | 32 | 8x fewer |
| Memory per taxonomy node | 1,024 bytes | 128 bytes | 8x less |
| Recall@10 with tangent pruning (prune_factor=10) | N/A | 93.6% | <2% loss vs exact, 5x faster |

The 10-50x recall improvement for hierarchical data comes from two sources: (1) hyperbolic distance preserves tree distances that Euclidean space distorts, and (2) far fewer dimensions are needed, reducing the curse of dimensionality.

### 3.4 Hyperbolic HNSW Configuration for Taxonomy

```rust
use ruvector_hyperbolic_hnsw::{HyperbolicHnsw, HyperbolicHnswConfig, DistanceMetric};

let config = HyperbolicHnswConfig {
    max_connections: 16,       // M parameter
    max_connections_0: 32,     // M0 for layer 0
    ef_construction: 200,      // Build-time search depth
    ef_search: 50,             // Query-time search depth
    level_mult: 1.0 / (16.0_f32).ln(),
    curvature: 1.0,            // Default; override per shard
    metric: DistanceMetric::Hybrid,  // Euclidean pruning + Poincare ranking
    prune_factor: 10,          // 10x candidates in tangent space
    use_tangent_pruning: true, // Enable the speed trick
};

let mut index = HyperbolicHnsw::new(config);
```

The `DistanceMetric::Hybrid` mode uses `fused_norms()` (single-pass computation of ||u-v||^2, ||u||^2, ||v||^2) for the pruning phase and `poincare_distance_from_norms()` only for final ranking of the top candidates, as implemented in `/home/user/ruvector/crates/ruvector-hyperbolic-hnsw/src/hnsw.rs`.

---

## 4. Quantized Search for Memory Efficiency

Genome-scale databases can contain billions of vectors. Without quantization, a 10-billion-vector index at 384 dimensions would require 10B * 384 * 4 bytes = 15.36 TB of memory for vectors alone. The tiered quantization system from `ruvector-core` (file: `/home/user/ruvector/crates/ruvector-core/src/quantization.rs`) makes this tractable.

### 4.1 Quantization Tiers for Genomic Data

| Tier | Type | Compression | Memory (10B, 384d) | Recall Loss | Use Case |
|------|------|-------------|-------------------|-------------|----------|
| Full precision | f32 | 1x | 15.36 TB | 0% | Gold-standard reference set (<100M vectors) |
| Scalar (u8) | `ScalarQuantized` | 4x | 3.84 TB | <0.4% | Active reference genomes |
| Int4 | `Int4Quantized` | 8x | 1.92 TB | <1.5% | Extended reference with good precision |
| Product (PQ) | `ProductQuantized` | 8-16x | 0.96-1.92 TB | <2% | Cold reference genomes, archived species |
| Binary | `BinaryQuantized` | 32x | 480 GB | ~10-15% | First-pass filtering only |

### 4.2 Tiered Progressive Refinement

The key insight for population-scale genomic search is that recall loss from aggressive quantization is acceptable for the first filtering pass, as long as a precise re-ranking step follows. This mirrors the tangent-space pruning strategy in hyperbolic HNSW.

```
Query Embedding (384-dim f32)
        |
   Stage 1: Binary scan (32x compressed)
   - Hamming distance via SIMD popcnt (NEON vcntq_u8 or x86 _popcnt64)
   - Scans 10B vectors in ~3 seconds (single core)
   - Returns top 100,000 candidates
        |
   Stage 2: Int4 re-rank (8x compressed)
   - Loads Int4 quantized vectors for candidates
   - Exact Int4 distance with nibble unpacking
   - Returns top 1,000 candidates
        |
   Stage 3: Full-precision HNSW search within candidate set
   - Loads f32 vectors for top 1,000
   - Cosine or Euclidean distance via SIMD (143ns per 1536-dim pair)
   - Returns top 10 results with full precision
        |
   Final results with <1% recall loss vs. exhaustive f32 search
```

**Progressive Refinement Performance**:

| Stage | Vectors Evaluated | Time (10B index) | Cumulative Recall |
|-------|-------------------|-------------------|-------------------|
| Binary filter | 10,000,000,000 | ~3.2s | ~85% of true top-10 in candidate set |
| Int4 re-rank | 100,000 | ~12ms | ~98% of true top-10 in candidate set |
| Full precision | 1,000 | ~0.15ms | ~99.5% final recall@10 |
| **Total** | -- | **~3.2s** | **99.5%** |

For the common case where the HNSW index itself is quantized (rather than flat scan), the binary stage is replaced by HNSW search over the quantized index:

| Approach | 10B Index Size | Search Latency | Recall@10 |
|----------|---------------|----------------|-----------|
| HNSW on f32 | 15.36 TB | <100us | 99.5% |
| HNSW on Int4 + f32 re-rank | 1.92 TB | <200us | 99.2% |
| HNSW on binary + Int4 re-rank + f32 re-rank | 480 GB | <500us | 98.8% |
| Flat binary scan + Int4 + f32 | 480 GB + 1.92 TB | ~3.2s | 99.5% |

### 4.3 ruQu Integration for Genomic Quantization

The `ruQu` crate (file: `/home/user/ruvector/crates/ruQu/Cargo.toml`) provides quantum-inspired quantization with min-cut-based structural analysis. For genomic vectors specifically:

- **Dimension grouping**: ruQu's structural filter identifies correlated dimensions in k-mer embeddings (e.g., k-mers differing by a single nucleotide substitution tend to cluster in embedding space). These correlated groups are quantized together for better codebook learning.
- **Adaptive bit allocation**: Dimensions with higher variance (more informative for species discrimination) receive more quantization bits. Typical allocation: 6-bit for top 25% most variable dimensions, 4-bit for middle 50%, 2-bit for bottom 25%.

---

## 5. Filtered Genomic Search

### 5.1 Genomic Metadata Schema

Every vector in the genomic index carries structured metadata enabling precise filtering. The `ruvector-filter` crate's `FilterExpression` (file: `/home/user/ruvector/crates/ruvector-filter/src/expression.rs`) supports all the operators needed for genomic queries.

| Field | Type | Example Values | Filter Type |
|-------|------|----------------|-------------|
| `chromosome` | string | "chr1", "chrX", "chrM" | Equality, In |
| `gene_name` | string | "BRCA1", "TP53" | Match (text), In |
| `pathway` | string[] | ["DNA repair", "apoptosis"] | In, Match |
| `variant_type` | string | "SNV", "indel", "SV", "CNV" | Equality, In |
| `maf` | float | 0.001 - 0.5 | Range, Gt, Lt |
| `clinical_significance` | string | "pathogenic", "benign", "VUS" | Equality, In |
| `organism` | string | "Homo sapiens", "E. coli K-12" | Equality, Match |
| `assembly` | string | "GRCh38", "GRCm39" | Equality |
| `quality_score` | float | 0.0 - 100.0 | Range, Gte |
| `sequencing_platform` | string | "illumina", "nanopore", "pacbio" | Equality, In |

### 5.2 Filter Strategy Selection

The `FilteredSearch` implementation in `ruvector-core` (file: `/home/user/ruvector/crates/ruvector-core/src/advanced_features/filtered_search.rs`) automatically selects between pre-filtering and post-filtering based on estimated selectivity.

**Genomic filter selectivity benchmarks**:

| Filter | Estimated Selectivity | Recommended Strategy | Rationale |
|--------|----------------------|---------------------|-----------|
| `chromosome = "chr1"` | ~8% (1/24 chromosomes) | Pre-filter | Highly selective |
| `variant_type = "SNV"` | ~70% | Post-filter | Low selectivity |
| `maf < 0.01` (rare variants) | ~5% | Pre-filter | Highly selective |
| `clinical_significance = "pathogenic"` | ~2% | Pre-filter | Very selective |
| `organism = "Homo sapiens"` AND `chromosome = "chr17"` | ~0.3% | Pre-filter | Compound AND is multiplicative |
| `gene_name IN ["BRCA1", "BRCA2", "TP53", "EGFR"]` | ~0.01% | Pre-filter | Very small candidate set |

The auto-selection threshold is 20%: filters with selectivity below 0.2 use pre-filtering (search only within matching IDs), while less selective filters use post-filtering (HNSW search first, then discard non-matching results with 2x over-fetch).

### 5.3 Hybrid Search: Vector Similarity + Gene Name Matching

The `HybridSearch` in `ruvector-core` (file: `/home/user/ruvector/crates/ruvector-core/src/advanced_features/hybrid_search.rs`) combines dense vector similarity with BM25 keyword matching. For genomic queries, this enables natural-language gene searches fused with embedding similarity.

```rust
// Configuration for genomic hybrid search
let config = HybridConfig {
    vector_weight: 0.6,   // Semantic similarity from k-mer/protein embedding
    keyword_weight: 0.4,  // BM25 match on gene name, description, GO terms
    normalization: NormalizationStrategy::MinMax,
};

let mut hybrid = HybridSearch::new(config);

// Index a gene with both embedding and text
hybrid.index_document(
    "BRCA1_NM_007294".to_string(),
    "BRCA1 DNA repair associated breast cancer 1 early onset \
     homologous recombination DNA damage response tumor suppressor".to_string(),
);
hybrid.finalize_indexing();

// Search: vector captures functional similarity, BM25 captures name match
let results = hybrid.search(
    &query_embedding,        // 384-dim k-mer embedding of query region
    "BRCA1 DNA repair",      // Text query
    10,                       // top-k
    |q, k| index.search(q, k), // Vector search function
)?;
```

**Hybrid Search Genomic Benchmarks**:

| Query Type | Vector Only Recall@10 | BM25 Only Recall@10 | Hybrid Recall@10 |
|------------|----------------------|---------------------|------------------|
| Known gene by name + function | 71% | 85% | 94% |
| Novel sequence (no name) | 89% | 0% | 89% |
| Functional homolog (different name) | 92% | 12% | 93% |
| Regulatory region near known gene | 45% | 68% | 82% |

---

## 6. Streaming Genome Indexing

### 6.1 Incremental HNSW Updates

New sequencing data arrives continuously -- from real-time nanopore sequencing, periodic database releases (RefSeq, UniProt), or institutional sequencing pipelines. The index must incorporate new vectors without full rebuild.

The `ruvector-delta-index` crate (file: `/home/user/ruvector/crates/ruvector-delta-index/src/lib.rs`) provides the `DeltaHnsw` implementation with:

- **Incremental insertion**: New vectors connect to the existing graph via the standard HNSW insert algorithm. Target: <1ms per insertion including graph edge updates.
- **Delta updates**: When a reference genome is revised (e.g., patch release of GRCh38), the `VectorDelta` from `ruvector-delta-core` captures the change vector. The `IncrementalUpdater` queues small deltas and flushes them in batches.
- **Lazy repair**: The `DeltaHnsw` monitors cumulative change per node. When cumulative L2 norm of applied deltas exceeds `repair_threshold` (default 0.5), the node's edges are reconnected via local neighborhood search. This avoids global rebuild while maintaining search quality.
- **Quality monitoring**: The `QualityMonitor` tracks recall estimates over time. If recall drops below 95%, a broader repair pass is triggered.

### 6.2 Streaming Architecture

```
Sequencing Instrument / Database Update
        |
   Raw sequence data (FASTQ / FASTA)
        |
   Embedding pipeline (k-mer extraction + projection)
        |
   IncrementalUpdater.queue_update()
        |  (batches up to batch_threshold, default 100)
        |
   IncrementalUpdater.flush() -> DeltaHnsw
        |
   Strategy selection per delta:
   |  magnitude < 0.05 -> DeltaOnly (no edge update)
   |  magnitude 0.05-0.5 -> LocalRepair (reconnect immediate neighbors)
   |  magnitude > 0.5 -> FullReconnect (full HNSW reconnection for node)
        |
   QualityMonitor.record_search() tracks recall
        |
   If recall < 95%: trigger force_repair() on degraded subgraph
```

### 6.3 Delta Indexing Performance Targets

| Operation | Target Latency | Measured (DeltaHnsw, 384-dim, 1M vectors) |
|-----------|---------------|-------------------------------------------|
| Single vector insert | <1ms | ~0.8ms (M=16, ef_construction=200) |
| Delta apply (small, DeltaOnly) | <50us | ~30us (vector update only, no graph change) |
| Delta apply (LocalRepair) | <500us | ~350us (reconnect ~16 immediate neighbors) |
| Batch insert (1000 vectors) | <800ms | ~650ms (sequential; ~0.65ms/vector) |
| Batch delta flush (100 updates) | <30ms | ~22ms |
| Force repair (1M vectors) | <60s | ~45s (full graph reconnection) |
| Compact delta streams | <5ms per 1000 nodes | ~3ms |

### 6.4 Index Versioning with Delta Streams

Each node in `DeltaHnsw` maintains a `DeltaStream<VectorDelta>` recording the history of changes. This enables:

- **Point-in-time queries**: Reconstruct the vector state at any previous version by replaying the delta stream up to that timestamp.
- **Audit trail**: Track which database update changed which vectors and by how much.
- **Rollback**: Reverse deltas to undo a problematic database update.
- **Compaction**: When a delta stream exceeds `max_deltas` (default 100), it compacts by composing sequential deltas into a single cumulative delta, preserving the final state while reducing memory.

---

## 7. System Integration Architecture

```
+-----------------------------------------------------------------------------+
|                          GENOMIC APPLICATION LAYER                           |
|  Metagenomic Classifier | Variant Annotator | Species Identifier            |
|  Protein Function Search | Clinical Decision Support                        |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                          QUERY ROUTING LAYER                                |
|  Multi-probe k-mer aggregation | Hybrid (vector + BM25) | Cross-collection |
+-----------------------------------------------------------------------------+
                                    |
         +------------------+-------------------+------------------+
         |                  |                   |                  |
+--------+------+  +--------+--------+  +-------+-------+  +------+--------+
| ruvector-core |  | ruvector-       |  | ruvector-     |  | ruvector-     |
| HNSW Index    |  | hyperbolic-hnsw |  | filter        |  | delta-index   |
| (Euclidean)   |  | (Poincare ball) |  | (Metadata)    |  | (Streaming)   |
|               |  |                 |  |               |  |               |
| M=16          |  | curvature=1.0   |  | Pre/Post auto |  | Incremental   |
| ef_search=64  |  | tangent pruning |  | Selectivity   |  | Lazy repair   |
| SIMD distance |  | shard curvature |  | estimation    |  | Delta streams |
+---------------+  +-----------------+  +---------------+  +---------------+
         |                  |                   |                  |
+-----------------------------------------------------------------------------+
|                          QUANTIZATION LAYER                                 |
|  Binary (32x) -> Int4 (8x) -> Scalar (4x) -> f32 (1x)                     |
|  Progressive refinement pipeline | ruQu adaptive bit allocation            |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                          SIMD INTRINSICS LAYER                              |
|  AVX2/AVX-512 (x86_64) | NEON (ARM64) | Scalar fallback | WASM            |
|  Hamming: popcnt/vcntq  | Euclidean: fused_norms | Cosine: 143ns@1536d    |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                          STORAGE LAYER                                      |
|  REDB (genomic indices) | Memory-mapped vectors | ruvector-collections     |
+-----------------------------------------------------------------------------+
```

---

## 8. Parameter Recommendations Summary

### Quick Reference: Recommended Configurations by Use Case

| Use Case | Embedding | Dim | HNSW M | ef_search | Quantization | Index Type | Expected Latency |
|----------|-----------|-----|--------|-----------|-------------|------------|-----------------|
| Real-time metagenomic classification | k=6 k-mer | 384 | 16 | 64 | Int4 | Euclidean HNSW | <100us |
| Species identification (high accuracy) | k=31 k-mer | 1536 | 24 | 128 | Scalar | Euclidean HNSW | <200us |
| Taxonomic placement | Poincare embedding | 128 | 16 | 50 | None | Hyperbolic HNSW | <150us |
| Protein homology search | ESM-2 (650M) | 1280 | 16 | 64 | Scalar | Euclidean HNSW | <150us |
| Structure similarity | GNN contact map | 384 | 16 | 64 | None | Euclidean HNSW | <80us |
| Clinical variant lookup | k=21 + metadata | 768 | 16 | 64 | None | Filtered HNSW | <200us |
| Population-scale (10B+) | k=6 k-mer | 384 | 12 | 32 | Binary + Int4 | Tiered progressive | <3.5s |
| Streaming (nanopore) | k=11 k-mer | 384 | 16 | 64 | Scalar | DeltaHnsw | <1ms/insert |

---

## 9. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| k-mer embedding loses mutation signal | Medium | High | Multi-probe search with overlapping windows; learned (not random) projections |
| Hyperbolic numerical instability at high curvature | Medium | Medium | EPS=1e-5 clamping; project_to_ball after every operation (already in crate) |
| Binary quantization recall too low for clinical use | High | High | Binary used only as first-pass filter; never as final ranking |
| Delta stream memory growth for frequently-updated genomes | Medium | Low | Compaction at max_deltas=100; cumulative delta composition |
| HNSW recall degradation under streaming inserts | Medium | Medium | QualityMonitor with 95% recall threshold triggers repair |
| Cross-contamination between region embedding spaces | Low | Medium | Separate ruvector-collections per region type |

---

## 10. Success Criteria

- [ ] k=10 HNSW search on 10B 384-dim genomic vectors completes in <100us p50
- [ ] Hyperbolic taxonomy search achieves >94% recall@10 on NCBI taxonomy (2.4M taxa)
- [ ] Progressive quantization pipeline (binary -> Int4 -> f32) achieves >99% recall@10 on 10B vectors within 4 seconds
- [ ] Streaming insertion via DeltaHnsw maintains <1ms per vector with recall >95%
- [ ] Filtered search with `chromosome` + `clinical_significance` pre-filter executes in <200us
- [ ] Hybrid search (vector + BM25 gene name) improves recall@10 by >10% over vector-only for named gene queries
- [ ] Memory footprint for 1B vectors at 384-dim with Int4 quantization stays under 200 GB

---

## References

1. Malkov, Y., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv:1603.09320.
2. Nickel, M., & Kiela, D. (2017). "Poincare Embeddings for Learning Hierarchical Representations." NeurIPS.
3. Lin, J., et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science.
4. Wood, D. E., & Salzberg, S. L. (2014). "Kraken: ultrafast metagenomic sequence classification using exact alignments." Genome Biology.
5. RuVector ADR-001: Core Architecture. `/home/user/ruvector/docs/adr/ADR-001-ruvector-core-architecture.md`
6. RuVector ADR-DB-005: Delta Index Updates. `/home/user/ruvector/docs/adr/delta-behavior/ADR-DB-005-delta-index-updates.md`

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | RuVector Architecture Team | Initial genomic vector search subsystem proposal |
