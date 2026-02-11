# ADR-018: RuVector DNA Analyzer - Domain-Driven Design Architecture

**Status**: Proposed
**Date**: 2026-02-11
**Parent**: ADR-001 RuVector Core Architecture, ADR-016 Delta-Behavior DDD Architecture
**Author**: System Architecture Designer

## Abstract

This ADR defines a comprehensive Domain-Driven Design (DDD) architecture for the
"RuVector DNA Analyzer" -- a futuristic DNA analysis engine built on the RuVector
vector database ecosystem. The system encompasses ten bounded contexts spanning raw
signal ingestion through clinical pharmacogenomics, population-scale genomics, pathogen
surveillance, and CRISPR guide engineering. Each context is mapped to existing RuVector
crates and follows the same DDD rigor established by ADR-016 (Delta-Behavior system).

---

## 1. Executive Summary

The RuVector DNA Analyzer models genomic data as high-dimensional vector
representations traversing a pipeline of bounded contexts. By treating each
analysis stage as a distinct subdomain with explicit anti-corruption layers and
published language contracts, the system enables:

- **Streaming basecalling**: Raw nanopore/illumina signals become vector-embedded
  sequences via SONA-powered adaptive neural networks
- **Graph-aware alignment**: Sequences map to population-aware reference graphs,
  not just linear references, using ruvector-graph and ruvector-mincut
- **Incremental variant calling**: Delta-based updates propagate variant discoveries
  through the pipeline without full recomputation
- **Clinical-grade annotation**: Variants flow through annotation and pharmacogenomics
  contexts with traceable provenance chains
- **Pathogen surveillance**: Metagenomic classification leverages
  ruvector-hyperbolic-hnsw for taxonomic tree indexing at scale
- **CRISPR engineering**: Guide RNA design uses ruvector-attention for off-target
  prediction with gated transformer models

---

## 2. Domain Analysis

### 2.1 Strategic Domain Map

```
+===========================================================================+
|                      RUVECTOR DNA ANALYZER SYSTEM                         |
+===========================================================================+
|                                                                           |
|  +-------------------+     +---------------------+     +-----------------+|
|  | 1. Sequence       |     | 2. Alignment &      |     | 3. Variant      ||
|  |    Ingestion      |---->|    Mapping           |---->|    Calling       ||
|  |                   |     |                      |     |                 ||
|  | - Basecallers     |     | - Seed-indexers      |     | - Genotypers    ||
|  | - QC Filters      |     | - Chain-extenders    |     | - SV Callers    ||
|  | - Adaptor Trims   |     | - Graph Aligners     |     | - Phasing       ||
|  +-------------------+     +---------------------+     +--------+--------+|
|                                      |                          |         |
|                                      v                          v         |
|  +-------------------+     +---------------------+     +-----------------+|
|  | 4. Graph Genome   |     | 5. Annotation &     |<----| (Variants)      ||
|  |    Domain         |<----|    Interpretation    |     +-----------------+|
|  |                   |     |                      |                       |
|  | - Ref Graphs      |     | - ClinVar Lookup    |     +-----------------+|
|  | - Min-Cut Parts   |     | - Consequence Pred.  |---->| 6. Epigenomics  ||
|  | - Bubble Chains   |     | - ACMG Classify     |     |                 ||
|  +-------------------+     +---------------------+     | - Methylation   ||
|                                      |                  | - Chromatin     ||
|                                      v                  | - Hi-C / 3D    ||
|  +-------------------+     +---------------------+     +-----------------+|
|  | 7. Pharmaco-      |<----| (Clinical Sig.)     |                       |
|  |    genomics       |     +---------------------+     +-----------------+|
|  |                   |                                  | 9. Pathogen     ||
|  | - PGx Alleles     |     +---------------------+     |    Surveillance ||
|  | - Drug Response   |     | 8. Population        |     |                 ||
|  | - Dosing Models   |     |    Genomics          |     | - Metagenomics  ||
|  +-------------------+     |                      |     | - AMR Detection ||
|                             | - Ancestry           |     | - Outbreak      ||
|  +-------------------+     | - Relatedness        |     +-----------------+|
|  | 10. CRISPR        |     | - GWAS               |                       |
|  |     Engineering   |     +---------------------+                       |
|  |                   |                                                    |
|  | - Guide Design    |                                                    |
|  | - Off-Target Pred |                                                    |
|  | - Edit Scoring    |                                                    |
|  +-------------------+                                                    |
|                                                                           |
+===========================================================================+
```

### 2.2 Core Domain Concepts

| Domain Concept         | Definition                                                           |
|------------------------|----------------------------------------------------------------------|
| **ReadSignal**         | Raw electrical/optical signal from a sequencing instrument            |
| **BasecalledSequence** | Nucleotide string with per-base quality scores (PHRED)               |
| **Alignment**          | Mapping of a sequence to a reference coordinate system               |
| **Variant**            | Deviation from a reference genome (SNV, indel, SV, CNV)              |
| **GenomeGraph**        | Population-aware directed graph representing all known alleles       |
| **Annotation**         | Functional/clinical metadata attached to a variant                   |
| **Epigenome**          | Chromatin state, methylation, and 3D structure overlay               |
| **Pharmacotype**       | Genotype-derived drug metabolism phenotype                           |
| **PopulationAllele**   | Allele frequency and linkage data across cohorts                     |
| **PathogenSignature**  | Taxonomic classification vector for metagenomic reads                |
| **GuideRNA**           | CRISPR spacer sequence with off-target profile                       |

---

## 3. Bounded Context Definitions

### 3.1 Sequence Ingestion Domain

**Purpose**: Convert raw instrument signals into basecalled sequences with quality
metrics. This is the entry point for all genomic data.

#### Ubiquitous Language

| Term               | Definition                                                        |
|--------------------|-------------------------------------------------------------------|
| **ReadSignal**     | Raw analog/digital signal from sequencer (nanopore current, Illumina intensity) |
| **Basecaller**     | Neural network that translates signal to nucleotide sequence       |
| **QualityScore**   | PHRED-scaled confidence for each basecall (Q30 = 1:1000 error)    |
| **AdaptorTrimmer** | Component that removes synthetic adapter sequences                 |
| **ReadGroup**      | Batch of reads from a single library/run/lane                      |
| **FlowCell**       | Physical sequencing unit producing reads                           |
| **SignalChunk**    | Windowed segment of raw signal for streaming basecalling           |

#### Aggregate Root: SequencingRun

```rust
pub mod sequence_ingestion {
    /// Root aggregate for a sequencing run
    pub struct SequencingRun {
        pub id: RunId,
        pub instrument: InstrumentType,
        pub flow_cell: FlowCellId,
        pub read_groups: Vec<ReadGroup>,
        pub status: RunStatus,
        pub metrics: RunMetrics,
        pub started_at: Timestamp,
    }

    // --- Value Objects ---

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RunId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FlowCellId(pub [u8; 16]);

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum InstrumentType {
        NanoporePromethion,
        NanoporeMinion,
        IlluminaNovaSeq,
        IlluminaNextSeq,
        PacBioRevio,
        ElementAviti,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum RunStatus {
        Initializing,
        Sequencing,
        Basecalling,
        Complete,
        Failed,
    }

    pub struct RunMetrics {
        pub total_reads: u64,
        pub total_bases: u64,
        pub mean_quality: f32,
        pub n50_length: u32,
        pub pass_rate: f32,
    }

    // --- Entities ---

    pub struct ReadGroup {
        pub id: ReadGroupId,
        pub sample_id: SampleId,
        pub library_id: LibraryId,
        pub reads: Vec<BasecalledRead>,
    }

    pub struct BasecalledRead {
        pub id: ReadId,
        pub sequence: Vec<u8>,          // ACGT as 0,1,2,3
        pub quality_scores: Vec<u8>,     // PHRED scores
        pub signal_embedding: Vec<f32>,  // SONA-produced embedding
        pub length: u32,
        pub mean_quality: f32,
        pub is_pass: bool,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ReadId(pub u128);
    pub struct ReadGroupId(pub String);
    pub struct SampleId(pub String);
    pub struct LibraryId(pub String);
    pub struct Timestamp(pub u64);

    // --- Invariants ---
    // 1. All reads in a ReadGroup share the same SampleId
    // 2. Quality scores length == sequence length
    // 3. signal_embedding dimensionality is fixed per InstrumentType
    // 4. mean_quality must equal the arithmetic mean of quality_scores
    // 5. is_pass is true iff mean_quality >= instrument pass threshold

    // --- Repository Interface ---
    pub trait SequencingRunRepository: Send + Sync {
        fn save(&self, run: &SequencingRun) -> Result<(), IngestionError>;
        fn find_by_id(&self, id: &RunId) -> Result<Option<SequencingRun>, IngestionError>;
        fn find_by_flow_cell(&self, fc: &FlowCellId) -> Result<Vec<SequencingRun>, IngestionError>;
        fn find_active(&self) -> Result<Vec<SequencingRun>, IngestionError>;
    }

    pub trait ReadRepository: Send + Sync {
        fn store_batch(&self, reads: &[BasecalledRead]) -> Result<u64, IngestionError>;
        fn find_by_id(&self, id: &ReadId) -> Result<Option<BasecalledRead>, IngestionError>;
        fn find_by_quality_range(&self, min_q: f32, max_q: f32) -> Result<Vec<ReadId>, IngestionError>;
        fn count_by_run(&self, run_id: &RunId) -> Result<u64, IngestionError>;
    }

    #[derive(Debug)]
    pub enum IngestionError {
        SignalCorrupted(String),
        BasecallFailed(String),
        QualityBelowThreshold { expected: f32, actual: f32 },
        StorageFull,
        DuplicateRead(ReadId),
    }
}
```

#### Domain Events

| Event                    | Payload                                    | Published When                        |
|--------------------------|--------------------------------------------|---------------------------------------|
| `RunStarted`             | run_id, instrument, flow_cell_id           | New sequencing run begins             |
| `SignalChunkReceived`    | run_id, chunk_index, signal_length         | Raw signal arrives from instrument    |
| `ReadBasecalled`         | read_id, run_id, length, mean_quality      | Single read basecalled                |
| `ReadGroupComplete`      | read_group_id, read_count, pass_rate       | All reads in group basecalled         |
| `RunComplete`            | run_id, total_reads, total_bases, n50      | Entire run finished                   |
| `QualityCheckFailed`     | read_id, reason                            | Read fails QC filters                 |

#### Domain Services

```rust
pub trait BasecallingService: Send + Sync {
    /// Process a raw signal chunk into basecalled reads
    fn basecall(&self, signal: &[f32], config: &BasecallConfig) -> Result<Vec<BasecalledRead>, IngestionError>;
}

pub trait QualityControlService: Send + Sync {
    /// Apply QC filters to reads, returning pass/fail partition
    fn filter(&self, reads: &[BasecalledRead], policy: &QcPolicy) -> (Vec<BasecalledRead>, Vec<BasecalledRead>);
}

pub trait AdaptorTrimmingService: Send + Sync {
    /// Remove adapter sequences from read ends
    fn trim(&self, read: &mut BasecalledRead, adapters: &[Vec<u8>]) -> TrimResult;
}
```

---

### 3.2 Alignment & Mapping Domain

**Purpose**: Map basecalled sequences to positions on a reference genome (linear or
graph-based), producing coordinate-sorted alignments.

#### Ubiquitous Language

| Term              | Definition                                                         |
|-------------------|--------------------------------------------------------------------|
| **Alignment**     | A read mapped to reference coordinates with CIGAR operations       |
| **SeedHit**       | Short exact match between read and reference used to anchor alignment |
| **ChainedAnchors**| Set of collinear seeds forming a candidate alignment region        |
| **CIGAR**         | Compact representation of alignment operations (M/I/D/S/H/N)      |
| **MappingQuality**| PHRED-scaled probability that the mapping position is wrong        |
| **SplitAlignment**| Read spanning a structural breakpoint mapped in multiple segments  |
| **SupplementaryAlignment** | Secondary location for a chimeric/split read              |

#### Aggregate Root: AlignmentBatch

```rust
pub mod alignment_mapping {
    pub struct AlignmentBatch {
        pub id: BatchId,
        pub reference_id: ReferenceId,
        pub alignments: Vec<Alignment>,
        pub unmapped_reads: Vec<ReadId>,
        pub metrics: AlignmentMetrics,
    }

    pub struct Alignment {
        pub read_id: ReadId,
        pub reference_id: ReferenceId,
        pub contig: ContigId,
        pub position: GenomicPosition,
        pub cigar: CigarString,
        pub mapping_quality: u8,
        pub alignment_score: i32,
        pub is_primary: bool,
        pub is_supplementary: bool,
        pub mate: Option<MateInfo>,
        pub tags: AlignmentTags,
    }

    // --- Value Objects ---

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BatchId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ReferenceId(pub u64);

    #[derive(Clone, PartialEq, Eq, Hash)]
    pub struct ContigId(pub String);  // e.g., "chr1", "chrX"

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct GenomicPosition {
        pub contig_index: u32,
        pub offset: u64,         // 0-based position
        pub strand: Strand,
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Strand { Forward, Reverse }

    pub struct CigarString(pub Vec<CigarOp>);

    #[derive(Clone, Copy)]
    pub enum CigarOp {
        Match(u32),
        Insertion(u32),
        Deletion(u32),
        SoftClip(u32),
        HardClip(u32),
        RefSkip(u32),   // spliced alignment
        Mismatch(u32),
    }

    pub struct MateInfo {
        pub contig: ContigId,
        pub position: u64,
        pub insert_size: i64,
    }

    pub struct AlignmentMetrics {
        pub total_reads: u64,
        pub mapped_reads: u64,
        pub mapping_rate: f32,
        pub mean_mapq: f32,
        pub mean_coverage: f32,
        pub duplicate_rate: f32,
    }

    pub struct AlignmentTags(pub Vec<(String, TagValue)>);

    pub enum TagValue {
        Int(i64),
        Float(f32),
        String(String),
        ByteArray(Vec<u8>),
    }

    // --- Invariants ---
    // 1. position.offset + reference_consumed(cigar) <= contig_length
    // 2. mapping_quality in 0..=255
    // 3. Exactly one primary alignment per read per batch
    // 4. Supplementary alignments must reference the same read_id
    // 5. insert_size is defined only for paired-end reads

    // --- Repository Interface ---
    pub trait AlignmentRepository: Send + Sync {
        fn store_batch(&self, batch: &AlignmentBatch) -> Result<(), AlignmentError>;
        fn find_overlapping(&self, region: &GenomicRegion) -> Result<Vec<Alignment>, AlignmentError>;
        fn find_by_read(&self, read_id: &ReadId) -> Result<Vec<Alignment>, AlignmentError>;
        fn coverage_at(&self, position: &GenomicPosition) -> Result<u32, AlignmentError>;
    }

    #[derive(Clone)]
    pub struct GenomicRegion {
        pub contig: ContigId,
        pub start: u64,
        pub end: u64,
    }

    #[derive(Debug)]
    pub enum AlignmentError {
        ReferenceNotFound(ReferenceId),
        IndexCorrupted(String),
        PositionOutOfBounds { position: u64, contig_length: u64 },
        InvalidCigar(String),
    }
}
```

#### Domain Events

| Event                    | Payload                                       | Published When                      |
|--------------------------|-----------------------------------------------|-------------------------------------|
| `AlignmentCompleted`     | batch_id, ref_id, mapped_count, unmapped_count | Batch of reads aligned             |
| `SplitAlignmentDetected` | read_id, segment_count, breakpoints            | Read maps to disjoint regions      |
| `LowMappingQuality`      | read_id, mapq, threshold                      | Read below MAPQ threshold           |
| `CoverageThresholdReached`| region, coverage_depth                        | Region exceeds target coverage      |
| `ChimericReadDetected`   | read_id, contig_a, contig_b                   | Read spans two chromosomes          |

---

### 3.3 Variant Calling Domain

**Purpose**: Identify and genotype genomic variants (SNV, indel, structural variants,
copy number variants) from alignment pileups.

#### Ubiquitous Language

| Term                | Definition                                                       |
|---------------------|------------------------------------------------------------------|
| **Variant**         | Any deviation from the reference: SNV, indel, SV, or CNV         |
| **Genotype**        | Diploid allele assignment (0/0, 0/1, 1/1, etc.)                 |
| **Pileup**          | Stack of aligned reads at a specific position                    |
| **HaplotypeBLock**  | Phased segment of linked alleles on one chromosome copy          |
| **StructuralVariant** | Large-scale rearrangement (>50bp): deletion, duplication, inversion, translocation |
| **VariantQuality**  | PHRED-scaled confidence in the variant call                      |
| **AlleleDepth**     | Read support counts per allele at a site                         |

#### Aggregate Root: VariantCallSet

```rust
pub mod variant_calling {
    pub struct VariantCallSet {
        pub id: CallSetId,
        pub sample_id: SampleId,
        pub reference_id: ReferenceId,
        pub caller: CallerInfo,
        pub variants: Vec<Variant>,
        pub metrics: CallSetMetrics,
    }

    pub struct Variant {
        pub id: VariantId,
        pub position: GenomicPosition,
        pub reference_allele: Vec<u8>,
        pub alternate_alleles: Vec<Vec<u8>>,
        pub variant_type: VariantType,
        pub genotype: Genotype,
        pub quality: f32,
        pub filter: FilterStatus,
        pub allele_depths: Vec<u32>,
        pub total_depth: u32,
        pub strand_bias: f32,
        pub genotype_likelihood: Vec<f32>,  // PL field, PHRED-scaled
        pub effect_embedding: Vec<f32>,     // GNN-predicted effect vector
    }

    // --- Value Objects ---

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CallSetId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VariantId(pub u128);

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum VariantType {
        Snv,
        Insertion,
        Deletion,
        Mnv,                     // multi-nucleotide variant
        DeletionSv,              // structural deletion >50bp
        DuplicationSv,
        InversionSv,
        TranslocationSv,
        CopyNumberGain,
        CopyNumberLoss,
        ComplexSv,
    }

    #[derive(Clone, PartialEq, Eq)]
    pub struct Genotype {
        pub alleles: Vec<u8>,    // indices into ref + alt alleles
        pub phased: bool,        // true = | separator, false = / separator
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum FilterStatus {
        Pass,
        LowQuality,
        LowDepth,
        StrandBias,
        ExcessHeterozygosity,
        MapQualityBias,
        Custom(&'static str),
    }

    pub struct CallerInfo {
        pub name: String,        // e.g., "ruvector-deepvariant"
        pub version: String,
        pub parameters: Vec<(String, String)>,
    }

    pub struct CallSetMetrics {
        pub total_variants: u64,
        pub snv_count: u64,
        pub indel_count: u64,
        pub sv_count: u64,
        pub ti_tv_ratio: f32,    // transition/transversion
        pub het_hom_ratio: f32,
        pub mean_quality: f32,
    }

    // --- Invariants ---
    // 1. reference_allele.len() >= 1 (always anchored)
    // 2. At least one alternate_allele differs from reference_allele
    // 3. genotype.alleles indices are valid into [ref] + alternate_alleles
    // 4. allele_depths.len() == 1 + alternate_alleles.len()
    // 5. total_depth == allele_depths.iter().sum()
    // 6. Ti/Tv ratio for WGS should be ~2.0-2.1 (aggregate invariant)
    // 7. effect_embedding dimensionality matches GNN model config

    // --- Repository Interface ---
    pub trait VariantRepository: Send + Sync {
        fn store_callset(&self, callset: &VariantCallSet) -> Result<(), VariantError>;
        fn find_in_region(&self, region: &GenomicRegion) -> Result<Vec<Variant>, VariantError>;
        fn find_by_type(&self, vtype: VariantType) -> Result<Vec<Variant>, VariantError>;
        fn find_by_quality_range(&self, min_q: f32, max_q: f32) -> Result<Vec<Variant>, VariantError>;
        fn nearest_by_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<Variant>, VariantError>;
    }

    #[derive(Debug)]
    pub enum VariantError {
        InvalidGenotype(String),
        DepthMismatch { expected: usize, actual: usize },
        PositionOutOfBounds,
        DuplicateVariant(VariantId),
        EmbeddingDimensionMismatch { expected: usize, actual: usize },
    }
}
```

#### Domain Events

| Event                   | Payload                                         | Published When                    |
|-------------------------|-------------------------------------------------|-----------------------------------|
| `VariantCalled`         | variant_id, position, type, quality, genotype    | New variant discovered            |
| `GenotypeRefined`       | variant_id, old_gt, new_gt, new_quality          | Genotype updated with more data   |
| `StructuralVariantFound`| variant_id, sv_type, breakpoints, size           | SV detected from split reads      |
| `PhasingCompleted`      | sample_id, block_count, phase_rate               | Haplotype phasing done            |
| `CallSetFinalized`      | callset_id, variant_count, metrics               | Variant calling pipeline done     |

---

### 3.4 Graph Genome Domain

**Purpose**: Maintain population-aware reference graphs that represent all known
alleles as graph structures rather than linear references. Supports min-cut partitioning
for efficient graph traversal.

#### Ubiquitous Language

| Term              | Definition                                                        |
|-------------------|-------------------------------------------------------------------|
| **GenomeGraph**   | Directed graph where nodes are sequence segments, edges are adjacencies |
| **Bubble**        | Subgraph representing allelic variation between two anchor nodes    |
| **Superbubble**   | Nested bubble structure for complex variation                      |
| **Partition**     | Min-cut decomposition of the graph for parallel processing         |
| **PathHaplotype** | A walk through the graph representing one haplotype                |
| **AnchorNode**    | High-confidence invariant node shared by all haplotypes            |

#### Aggregate Root: GenomeGraph

```rust
pub mod graph_genome {
    pub struct GenomeGraph {
        pub id: GraphId,
        pub name: String,            // e.g., "GRCh38-pangenome-v2"
        pub contigs: Vec<ContigGraph>,
        pub population_sources: Vec<PopulationSource>,
        pub statistics: GraphStatistics,
    }

    pub struct ContigGraph {
        pub contig_id: ContigId,
        pub nodes: Vec<SequenceNode>,
        pub edges: Vec<GraphEdge>,
        pub bubbles: Vec<Bubble>,
        pub partitions: Vec<Partition>,
    }

    pub struct SequenceNode {
        pub id: NodeId,
        pub sequence: Vec<u8>,       // nucleotide content
        pub length: u32,
        pub is_reference: bool,      // true if on GRCh38 backbone
        pub allele_frequency: f32,   // population frequency
        pub embedding: Vec<f32>,     // vector representation for ANN search
    }

    pub struct GraphEdge {
        pub from: NodeId,
        pub to: NodeId,
        pub edge_type: EdgeType,
        pub weight: f32,             // traversal frequency
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum EdgeType {
        Reference,           // follows linear reference
        Variant,             // alternative allele path
        Structural,          // structural variant edge
    }

    pub struct Bubble {
        pub id: BubbleId,
        pub source_node: NodeId,     // entry anchor
        pub sink_node: NodeId,       // exit anchor
        pub paths: Vec<Vec<NodeId>>, // allelic paths through bubble
        pub complexity: BubbleComplexity,
    }

    #[derive(Clone, Copy)]
    pub enum BubbleComplexity {
        Simple,              // biallelic SNV/indel
        Multi,               // multiallelic
        Super,               // nested superbubble
    }

    pub struct Partition {
        pub id: PartitionId,
        pub node_set: Vec<NodeId>,
        pub cut_edges: Vec<GraphEdge>,
        pub boundary_nodes: Vec<NodeId>,
        pub min_cut_value: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphId(pub u128);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeId(pub u64);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BubbleId(pub u64);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PartitionId(pub u32);

    pub struct PopulationSource {
        pub name: String,       // e.g., "1000Genomes", "HPRC"
        pub sample_count: u32,
        pub ancestry_groups: Vec<String>,
    }

    pub struct GraphStatistics {
        pub total_nodes: u64,
        pub total_edges: u64,
        pub total_bubbles: u64,
        pub total_partitions: u32,
        pub mean_node_length: f32,
        pub graph_complexity: f32,  // edges/nodes ratio
    }

    // --- Invariants ---
    // 1. Graph is a DAG within each contig (no cycles)
    // 2. Every bubble has exactly one source and one sink
    // 3. All paths through a bubble connect source to sink
    // 4. Partitions are non-overlapping (except boundary nodes)
    // 5. Sum of partition node sets = total node set
    // 6. allele_frequency in [0.0, 1.0]
    // 7. At least one Reference edge path exists per contig (backbone)

    // --- Repository Interface ---
    pub trait GenomeGraphRepository: Send + Sync {
        fn save(&self, graph: &GenomeGraph) -> Result<(), GraphGenomeError>;
        fn find_by_id(&self, id: &GraphId) -> Result<Option<GenomeGraph>, GraphGenomeError>;
        fn subgraph(&self, graph_id: &GraphId, region: &GenomicRegion) -> Result<ContigGraph, GraphGenomeError>;
        fn find_bubbles_in_region(&self, graph_id: &GraphId, region: &GenomicRegion) -> Result<Vec<Bubble>, GraphGenomeError>;
        fn find_partition(&self, graph_id: &GraphId, partition_id: &PartitionId) -> Result<Partition, GraphGenomeError>;
        fn nearest_nodes_by_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<SequenceNode>, GraphGenomeError>;
    }

    #[derive(Debug)]
    pub enum GraphGenomeError {
        CycleDetected(ContigId),
        OrphanedNode(NodeId),
        InvalidBubble { bubble_id: BubbleId, reason: String },
        PartitionOverlap(PartitionId, PartitionId),
        NodeNotFound(NodeId),
    }
}
```

#### Domain Events

| Event                     | Payload                                      | Published When                     |
|---------------------------|----------------------------------------------|------------------------------------|
| `GraphConstructed`        | graph_id, node_count, edge_count             | New pangenome graph built          |
| `BubbleIdentified`        | bubble_id, source, sink, path_count          | Allelic variation site found       |
| `GraphPartitioned`        | graph_id, partition_count, max_cut_value     | Min-cut partitioning complete      |
| `GraphUpdated`            | graph_id, nodes_added, edges_added           | New population data incorporated   |
| `PathHaplotypeResolved`   | graph_id, sample_id, path_nodes              | Sample haplotype traced through graph |

---

### 3.5 Annotation & Interpretation Domain

**Purpose**: Annotate variants with functional consequences, clinical significance,
population frequencies, and ACMG/AMP classification.

#### Aggregate Root: AnnotatedVariant

```rust
pub mod annotation_interpretation {
    pub struct AnnotatedVariant {
        pub variant_id: VariantId,
        pub consequence: VariantConsequence,
        pub clinical: ClinicalAnnotation,
        pub population_freq: PopulationFrequency,
        pub predictions: InSilicoPredictions,
        pub acmg_classification: AcmgClassification,
        pub provenance: AnnotationProvenance,
    }

    pub struct VariantConsequence {
        pub gene: Option<GeneId>,
        pub transcript: Option<TranscriptId>,
        pub consequence_type: ConsequenceType,
        pub hgvs_coding: Option<String>,     // e.g., "c.123A>G"
        pub hgvs_protein: Option<String>,    // e.g., "p.Thr41Ala"
        pub exon_number: Option<u32>,
        pub codon_change: Option<(String, String)>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ConsequenceType {
        Synonymous,
        Missense,
        Nonsense,              // stop gained
        Frameshift,
        SpliceDonor,
        SpliceAcceptor,
        FivePrimeUtr,
        ThreePrimeUtr,
        Intergenic,
        Intronic,
        StartLoss,
        StopLoss,
        InframeDeletion,
        InframeInsertion,
        RegulatoryRegion,
    }

    pub struct ClinicalAnnotation {
        pub clinvar_id: Option<String>,
        pub clinvar_significance: Option<ClinicalSignificance>,
        pub omim_ids: Vec<String>,
        pub disease_associations: Vec<DiseaseAssociation>,
        pub review_status: ClinVarReviewStatus,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ClinicalSignificance {
        Benign,
        LikelyBenign,
        Vus,                   // variant of uncertain significance
        LikelyPathogenic,
        Pathogenic,
        DrugResponse,
        RiskFactor,
        Conflicting,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ClinVarReviewStatus {
        NoAssertion,
        SingleSubmitter,
        MultipleSubmitters,
        ExpertPanel,
        PracticeGuideline,
    }

    pub struct DiseaseAssociation {
        pub disease_name: String,
        pub mondo_id: Option<String>,
        pub inheritance: InheritancePattern,
        pub penetrance: Penetrance,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum InheritancePattern {
        AutosomalDominant,
        AutosomalRecessive,
        XLinked,
        Mitochondrial,
        Complex,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Penetrance { Complete, Incomplete, Unknown }

    pub struct PopulationFrequency {
        pub gnomad_af: Option<f32>,          // global allele frequency
        pub gnomad_af_by_pop: Vec<(String, f32)>,  // per-population AF
        pub topmed_af: Option<f32>,
        pub is_rare: bool,                    // AF < 0.01
    }

    pub struct InSilicoPredictions {
        pub sift_score: Option<f32>,
        pub polyphen_score: Option<f32>,
        pub cadd_phred: Option<f32>,
        pub revel_score: Option<f32>,
        pub alphamissense_score: Option<f32>,
        pub gnn_effect_vector: Vec<f32>,     // RuVector GNN prediction
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum AcmgClassification {
        Benign,
        LikelyBenign,
        Vus,
        LikelyPathogenic,
        Pathogenic,
    }

    pub struct AnnotationProvenance {
        pub databases_queried: Vec<(String, String)>,  // (name, version)
        pub annotation_date: u64,
        pub pipeline_version: String,
    }

    // --- Value Objects ---
    pub struct GeneId(pub String);       // e.g., "ENSG00000141510" (TP53)
    pub struct TranscriptId(pub String); // e.g., "ENST00000269305"

    // --- Repository Interface ---
    pub trait AnnotationRepository: Send + Sync {
        fn annotate(&self, variant_id: &VariantId) -> Result<AnnotatedVariant, AnnotationError>;
        fn find_pathogenic_in_gene(&self, gene: &GeneId) -> Result<Vec<AnnotatedVariant>, AnnotationError>;
        fn search_by_disease(&self, disease: &str) -> Result<Vec<AnnotatedVariant>, AnnotationError>;
        fn nearest_by_effect_vector(&self, vector: &[f32], k: usize) -> Result<Vec<AnnotatedVariant>, AnnotationError>;
    }

    #[derive(Debug)]
    pub enum AnnotationError {
        VariantNotFound(VariantId),
        DatabaseUnavailable(String),
        TranscriptMappingFailed(String),
        ConsequencePredictionFailed(String),
    }
}
```

#### Domain Events

| Event                     | Payload                                        | Published When                    |
|---------------------------|------------------------------------------------|-----------------------------------|
| `VariantAnnotated`        | variant_id, consequence, clinical_sig          | Annotation pipeline completes     |
| `PathogenicVariantFound`  | variant_id, gene, disease, acmg_class          | P/LP variant identified           |
| `NovelVariantDetected`    | variant_id, position, consequence              | Variant absent from all databases |
| `AcmgReclassified`       | variant_id, old_class, new_class, evidence     | Classification changed            |

---

### 3.6 Epigenomics Domain

**Purpose**: Model the epigenetic landscape including DNA methylation, histone
modifications, chromatin accessibility, and 3D genome structure (Hi-C/TADs).

#### Aggregate Root: EpigenomicProfile

```rust
pub mod epigenomics {
    pub struct EpigenomicProfile {
        pub id: ProfileId,
        pub sample_id: SampleId,
        pub cell_type: CellType,
        pub methylation_map: MethylationMap,
        pub chromatin_state: ChromatinStateMap,
        pub hi_c_contacts: Option<ContactMatrix>,
        pub tad_boundaries: Vec<TadBoundary>,
    }

    pub struct MethylationMap {
        pub cpg_sites: Vec<CpGSite>,
        pub global_methylation_level: f32,
        pub differentially_methylated_regions: Vec<DmrRegion>,
    }

    pub struct CpGSite {
        pub position: GenomicPosition,
        pub methylation_ratio: f32,    // 0.0 (unmethylated) to 1.0 (fully methylated)
        pub coverage: u32,
        pub context: MethylationContext,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum MethylationContext { CpG, CHG, CHH }

    pub struct DmrRegion {
        pub region: GenomicRegion,
        pub mean_delta_methylation: f32,
        pub p_value: f64,
        pub associated_gene: Option<GeneId>,
    }

    pub struct ChromatinStateMap {
        pub states: Vec<ChromatinSegment>,
        pub model: ChromHmmModel,
    }

    pub struct ChromatinSegment {
        pub region: GenomicRegion,
        pub state: ChromatinState,
        pub posterior_probability: f32,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ChromatinState {
        ActivePromoter,
        StrongEnhancer,
        WeakEnhancer,
        Transcribed,
        Heterochromatin,
        PoisedPromoter,
        Repressed,
        Quiescent,
    }

    pub struct ContactMatrix {
        pub resolution: u32,           // bin size in bp
        pub matrix_embedding: Vec<f32>,// flattened + compressed via ruvector-core
        pub compartments: Vec<Compartment>,
    }

    pub struct Compartment {
        pub region: GenomicRegion,
        pub compartment_type: CompartmentType,
        pub eigenvector_value: f32,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CompartmentType { A, B }  // A = active, B = inactive

    pub struct TadBoundary {
        pub position: GenomicPosition,
        pub insulation_score: f32,
        pub boundary_strength: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ProfileId(pub u128);
    pub struct CellType(pub String);
    pub struct ChromHmmModel(pub String);

    // --- Repository Interface ---
    pub trait EpigenomicRepository: Send + Sync {
        fn store_profile(&self, profile: &EpigenomicProfile) -> Result<(), EpigenomicError>;
        fn find_methylation_in_region(&self, sample: &SampleId, region: &GenomicRegion) -> Result<Vec<CpGSite>, EpigenomicError>;
        fn find_dmrs(&self, sample_a: &SampleId, sample_b: &SampleId) -> Result<Vec<DmrRegion>, EpigenomicError>;
        fn chromatin_state_at(&self, sample: &SampleId, position: &GenomicPosition) -> Result<ChromatinState, EpigenomicError>;
    }

    #[derive(Debug)]
    pub enum EpigenomicError {
        ProfileNotFound(ProfileId),
        ResolutionMismatch { expected: u32, actual: u32 },
        InsufficientCoverage { site: GenomicPosition, coverage: u32 },
    }
}
```

#### Domain Events

| Event                        | Payload                                   | Published When                   |
|------------------------------|-------------------------------------------|----------------------------------|
| `MethylationProfiled`       | profile_id, sample_id, cpg_count          | Methylation analysis complete    |
| `DmrIdentified`             | region, delta_methylation, gene           | Differentially methylated region |
| `TadBoundaryDisrupted`      | position, variant_id, insulation_change   | Variant disrupts TAD boundary    |
| `ChromatinStateChanged`     | region, old_state, new_state, cell_type   | State transition detected        |

---

### 3.7 Pharmacogenomics Domain

**Purpose**: Translate genotypes into drug response predictions, star-allele calls,
and clinical dosing recommendations.

#### Aggregate Root: PharmacogenomicProfile

```rust
pub mod pharmacogenomics {
    pub struct PharmacogenomicProfile {
        pub id: PgxProfileId,
        pub sample_id: SampleId,
        pub star_alleles: Vec<StarAlleleDiplotype>,
        pub drug_interactions: Vec<DrugGeneInteraction>,
        pub dosing_recommendations: Vec<DosingRecommendation>,
        pub metabolizer_phenotypes: Vec<MetabolizerPhenotype>,
    }

    pub struct StarAlleleDiplotype {
        pub gene: GeneId,
        pub gene_symbol: String,            // e.g., "CYP2D6"
        pub allele_1: StarAllele,
        pub allele_2: StarAllele,
        pub activity_score: f32,
        pub function: AlleleFunction,
    }

    pub struct StarAllele {
        pub name: String,                   // e.g., "*1", "*4", "*17"
        pub defining_variants: Vec<VariantId>,
        pub function: AlleleFunction,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum AlleleFunction {
        NormalFunction,
        DecreasedFunction,
        NoFunction,
        IncreasedFunction,
        UncertainFunction,
    }

    pub struct MetabolizerPhenotype {
        pub gene_symbol: String,
        pub phenotype: MetabolizerStatus,
        pub activity_score: f32,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum MetabolizerStatus {
        UltrarapidMetabolizer,
        RapidMetabolizer,
        NormalMetabolizer,
        IntermediateMetabolizer,
        PoorMetabolizer,
    }

    pub struct DrugGeneInteraction {
        pub drug_name: String,
        pub rxnorm_id: Option<String>,
        pub gene_symbol: String,
        pub evidence_level: EvidenceLevel,
        pub interaction_type: InteractionType,
        pub predicted_response_embedding: Vec<f32>,  // SONA-predicted
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum EvidenceLevel { Level1A, Level1B, Level2A, Level2B, Level3, Level4 }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum InteractionType { Dosing, Efficacy, Toxicity, Contraindication }

    pub struct DosingRecommendation {
        pub drug_name: String,
        pub gene_symbol: String,
        pub phenotype: MetabolizerStatus,
        pub recommendation: String,
        pub source: String,               // e.g., "CPIC", "DPWG"
        pub guideline_version: String,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PgxProfileId(pub u128);

    // --- Repository Interface ---
    pub trait PharmacogenomicRepository: Send + Sync {
        fn store_profile(&self, profile: &PharmacogenomicProfile) -> Result<(), PgxError>;
        fn find_by_sample(&self, sample: &SampleId) -> Result<Option<PharmacogenomicProfile>, PgxError>;
        fn find_interactions_for_drug(&self, drug: &str) -> Result<Vec<DrugGeneInteraction>, PgxError>;
        fn nearest_by_response_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<DrugGeneInteraction>, PgxError>;
    }

    #[derive(Debug)]
    pub enum PgxError {
        GeneNotInPanel(String),
        AlleleNotRecognized { gene: String, allele: String },
        InsufficientCoverage(String),
        GuidelineNotFound(String),
    }
}
```

#### Domain Events

| Event                       | Payload                                      | Published When                  |
|-----------------------------|----------------------------------------------|---------------------------------|
| `StarAllelesCalled`        | sample_id, gene, diplotype, activity_score    | PGx allele calling complete     |
| `DrugInteractionIdentified`| sample_id, drug, gene, interaction_type       | Clinically relevant interaction |
| `DosingAlertGenerated`     | sample_id, drug, recommendation, urgency      | Actionable dosing change        |
| `PoorMetabolizerDetected`  | sample_id, gene, phenotype                    | PM phenotype identified         |

---

### 3.8 Population Genomics Domain

**Purpose**: Analyze cohort-level genomic data for ancestry inference, kinship
estimation, allele frequency calculation, and genome-wide association studies.

#### Aggregate Root: PopulationStudy

```rust
pub mod population_genomics {
    pub struct PopulationStudy {
        pub id: StudyId,
        pub name: String,
        pub cohort: Cohort,
        pub allele_frequencies: AlleleFrequencyTable,
        pub pca_result: Option<PcaResult>,
        pub gwas_results: Vec<GwasResult>,
        pub kinship_matrix: Option<KinshipMatrix>,
    }

    pub struct Cohort {
        pub samples: Vec<SampleId>,
        pub ancestry_composition: Vec<AncestryAssignment>,
        pub sample_count: u32,
    }

    pub struct AncestryAssignment {
        pub sample_id: SampleId,
        pub ancestry_proportions: Vec<(AncestryGroup, f32)>,
        pub principal_components: Vec<f32>,    // top PCs as embedding
    }

    pub struct AncestryGroup(pub String);  // e.g., "EUR", "AFR", "EAS"

    pub struct AlleleFrequencyTable {
        pub variant_count: u64,
        pub entries: Vec<AlleleFrequencyEntry>,
    }

    pub struct AlleleFrequencyEntry {
        pub variant_id: VariantId,
        pub global_af: f32,
        pub population_afs: Vec<(AncestryGroup, f32)>,
        pub hardy_weinberg_p: f64,
    }

    pub struct PcaResult {
        pub eigenvalues: Vec<f64>,
        pub variance_explained: Vec<f64>,
        pub sample_projections: Vec<(SampleId, Vec<f64>)>,
    }

    pub struct GwasResult {
        pub trait_name: String,
        pub variant_id: VariantId,
        pub p_value: f64,
        pub odds_ratio: f64,
        pub beta: f64,
        pub standard_error: f64,
        pub effect_embedding: Vec<f32>,   // vector for similarity search
    }

    pub struct KinshipMatrix {
        pub sample_ids: Vec<SampleId>,
        pub coefficients: Vec<Vec<f32>>,  // symmetric matrix
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StudyId(pub u128);

    // --- Invariants ---
    // 1. ancestry_proportions sum to 1.0 per sample
    // 2. allele_frequency in [0.0, 1.0]
    // 3. hardy_weinberg_p is valid probability
    // 4. kinship_matrix is symmetric and sample_ids.len() == matrix dimension
    // 5. GWAS p-values < 5e-8 are genome-wide significant

    // --- Repository Interface ---
    pub trait PopulationRepository: Send + Sync {
        fn store_study(&self, study: &PopulationStudy) -> Result<(), PopulationError>;
        fn find_allele_freq(&self, variant: &VariantId, population: &AncestryGroup) -> Result<f32, PopulationError>;
        fn find_gwas_hits(&self, trait_name: &str, p_threshold: f64) -> Result<Vec<GwasResult>, PopulationError>;
        fn find_related_samples(&self, sample: &SampleId, kinship_threshold: f32) -> Result<Vec<(SampleId, f32)>, PopulationError>;
        fn nearest_by_ancestry_embedding(&self, pcs: &[f32], k: usize) -> Result<Vec<AncestryAssignment>, PopulationError>;
    }

    #[derive(Debug)]
    pub enum PopulationError {
        SampleNotInCohort(SampleId),
        InsufficientSampleSize { required: u32, actual: u32 },
        HardyWeinbergViolation { variant: VariantId, p_value: f64 },
    }
}
```

#### Domain Events

| Event                      | Payload                                    | Published When                   |
|----------------------------|--------------------------------------------|----------------------------------|
| `AncestryInferred`        | sample_id, ancestry_proportions, pcs        | Ancestry assignment complete     |
| `GwasSignificantHit`      | trait, variant_id, p_value, odds_ratio      | Genome-wide significant signal   |
| `AlleleFrequencyUpdated`  | variant_id, old_af, new_af, population      | New samples shift frequency      |
| `KinshipDetected`         | sample_a, sample_b, coefficient             | Related individuals found        |
| `PopulationStructureShift`| study_id, pc_variance_change                | PCA reveals new clustering       |

---

### 3.9 Pathogen Surveillance Domain

**Purpose**: Classify metagenomic reads to taxonomy, detect antimicrobial resistance
genes, and support real-time pathogen outbreak surveillance.

#### Aggregate Root: SurveillanceSample

```rust
pub mod pathogen_surveillance {
    pub struct SurveillanceSample {
        pub id: SurveillanceSampleId,
        pub sample_id: SampleId,
        pub collection_metadata: CollectionMetadata,
        pub taxonomic_profile: TaxonomicProfile,
        pub amr_detections: Vec<AmrDetection>,
        pub virulence_factors: Vec<VirulenceFactor>,
        pub outbreak_links: Vec<OutbreakLink>,
    }

    pub struct CollectionMetadata {
        pub collection_date: u64,
        pub geographic_location: GeoLocation,
        pub host_species: String,
        pub sample_type: SampleType,
        pub sequencing_platform: String,
    }

    pub struct GeoLocation {
        pub latitude: f64,
        pub longitude: f64,
        pub country: String,
        pub region: Option<String>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum SampleType { Clinical, Environmental, Wastewater, Food, Surveillance }

    pub struct TaxonomicProfile {
        pub classifications: Vec<TaxonomicClassification>,
        pub diversity_index: f64,        // Shannon diversity
        pub dominant_species: Option<TaxonId>,
        pub read_classification_rate: f32,
    }

    pub struct TaxonomicClassification {
        pub taxon_id: TaxonId,
        pub taxon_name: String,
        pub rank: TaxonomicRank,
        pub abundance: f32,              // relative abundance [0.0, 1.0]
        pub read_count: u64,
        pub confidence: f32,
        pub taxonomy_embedding: Vec<f32>, // hyperbolic embedding in taxonomy tree
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TaxonId(pub u64);         // NCBI taxonomy ID

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum TaxonomicRank {
        Superkingdom, Phylum, Class, Order, Family, Genus, Species, Strain,
    }

    pub struct AmrDetection {
        pub gene_name: String,           // e.g., "blaNDM-1", "mecA"
        pub gene_family: String,         // e.g., "carbapenemase", "PBP2a"
        pub drug_class: String,          // e.g., "carbapenems", "methicillin"
        pub identity_percent: f32,
        pub coverage_percent: f32,
        pub contig_id: Option<String>,
        pub mechanism: ResistanceMechanism,
        pub clinical_relevance: ClinicalRelevance,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ResistanceMechanism {
        EnzymaticInactivation,
        TargetModification,
        EffluxPump,
        TargetProtection,
        ReducedPermeability,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum ClinicalRelevance { Critical, High, Moderate, Low }

    pub struct VirulenceFactor {
        pub gene_name: String,
        pub factor_type: String,         // e.g., "toxin", "adhesin", "capsule"
        pub identity_percent: f32,
        pub source_organism: TaxonId,
    }

    pub struct OutbreakLink {
        pub linked_sample: SurveillanceSampleId,
        pub snp_distance: u32,           // core genome SNPs apart
        pub cgmlst_distance: u32,        // cgMLST allelic differences
        pub cluster_id: Option<String>,
        pub link_confidence: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SurveillanceSampleId(pub u128);

    // --- Repository Interface ---
    pub trait SurveillanceRepository: Send + Sync {
        fn store_sample(&self, sample: &SurveillanceSample) -> Result<(), SurveillanceError>;
        fn find_by_taxon(&self, taxon: &TaxonId) -> Result<Vec<SurveillanceSample>, SurveillanceError>;
        fn find_by_amr_gene(&self, gene: &str) -> Result<Vec<SurveillanceSample>, SurveillanceError>;
        fn find_outbreak_cluster(&self, sample: &SurveillanceSampleId, snp_threshold: u32) -> Result<Vec<OutbreakLink>, SurveillanceError>;
        fn nearest_by_taxonomy_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<TaxonomicClassification>, SurveillanceError>;
        fn search_by_geolocation(&self, center: &GeoLocation, radius_km: f64) -> Result<Vec<SurveillanceSample>, SurveillanceError>;
    }

    #[derive(Debug)]
    pub enum SurveillanceError {
        TaxonNotFound(TaxonId),
        ClassificationFailed(String),
        OutbreakLinkageTimeout,
        InsufficientGenomeCoverage { required: f32, actual: f32 },
    }
}
```

#### Domain Events

| Event                       | Payload                                        | Published When                 |
|-----------------------------|------------------------------------------------|--------------------------------|
| `PathogenDetected`         | sample_id, taxon_id, abundance, confidence      | Pathogen above threshold       |
| `AmrGeneDetected`          | sample_id, gene, drug_class, relevance          | Resistance gene found          |
| `OutbreakClusterExpanded`  | cluster_id, new_sample, total_samples           | New sample joins cluster       |
| `NovelResistancePattern`   | sample_id, genes, mechanism                     | Unknown AMR combination        |
| `SurveillanceAlert`        | alert_type, severity, affected_region           | Public health alert triggered  |

---

### 3.10 CRISPR Engineering Domain

**Purpose**: Design guide RNAs for CRISPR-Cas experiments, predict off-target sites,
and score editing efficiency.

#### Aggregate Root: CrisprExperiment

```rust
pub mod crispr_engineering {
    pub struct CrisprExperiment {
        pub id: ExperimentId,
        pub target_gene: GeneId,
        pub target_region: GenomicRegion,
        pub cas_system: CasSystem,
        pub guides: Vec<GuideRna>,
        pub off_target_analysis: OffTargetAnalysis,
        pub editing_predictions: Vec<EditingPrediction>,
    }

    pub struct GuideRna {
        pub id: GuideId,
        pub spacer_sequence: Vec<u8>,        // 20-24nt guide sequence
        pub pam_sequence: Vec<u8>,           // e.g., "NGG" for SpCas9
        pub target_strand: Strand,
        pub genomic_position: GenomicPosition,
        pub on_target_score: f32,            // predicted cutting efficiency
        pub specificity_score: f32,          // 1.0 = perfectly specific
        pub gc_content: f32,
        pub secondary_structure_dg: f32,     // free energy of folding
        pub sequence_embedding: Vec<f32>,    // attention-model embedding
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CasSystem {
        SpCas9,          // S. pyogenes Cas9, PAM = NGG
        SaCas9,          // S. aureus Cas9, PAM = NNGRRT
        Cas12a,          // Cpf1, PAM = TTTV
        Cas13,           // RNA targeting
        BasEditor,       // CBE or ABE
        PrimeEditor,     // PE2/PE3
    }

    pub struct OffTargetAnalysis {
        pub guide_id: GuideId,
        pub off_target_sites: Vec<OffTargetSite>,
        pub aggregate_off_target_score: f32,
        pub search_parameters: OffTargetSearchParams,
    }

    pub struct OffTargetSite {
        pub position: GenomicPosition,
        pub sequence: Vec<u8>,
        pub mismatches: u8,
        pub mismatch_positions: Vec<u8>,
        pub bulges: u8,
        pub cutting_probability: f32,        // model-predicted
        pub in_gene: Option<GeneId>,
        pub in_exon: bool,
        pub site_embedding: Vec<f32>,        // for similarity clustering
    }

    pub struct OffTargetSearchParams {
        pub max_mismatches: u8,
        pub max_bulges: u8,
        pub include_non_canonical_pam: bool,
        pub genome_graph_id: Option<GraphId>,  // search against pangenome
    }

    pub struct EditingPrediction {
        pub guide_id: GuideId,
        pub edit_type: EditType,
        pub predicted_outcome: EditOutcome,
        pub efficiency: f32,
        pub precision: f32,                   // fraction of desired edit among all edits
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum EditType {
        Knockout,            // indel-mediated gene disruption
        KnockIn,             // HDR-mediated insertion
        BaseEdit,            // C>T or A>G without DSB
        PrimeEdit,           // precise edit without DSB or donor
        Deletion,            // defined deletion
        Activation,          // CRISPRa
        Repression,          // CRISPRi
    }

    pub struct EditOutcome {
        pub indel_distribution: Vec<(i32, f32)>,  // (size, probability) negative=del, positive=ins
        pub frameshift_probability: f32,
        pub desired_edit_probability: f32,
    }

    // --- Value Objects ---
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ExperimentId(pub u128);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GuideId(pub u128);

    // --- Invariants ---
    // 1. spacer_sequence.len() matches CasSystem requirements (20 for SpCas9)
    // 2. pam_sequence matches CasSystem PAM motif
    // 3. on_target_score in [0.0, 1.0]
    // 4. specificity_score in [0.0, 1.0]
    // 5. gc_content in [0.0, 1.0] and computed from spacer_sequence
    // 6. off_target_sites sorted by cutting_probability descending
    // 7. indel_distribution probabilities sum to 1.0

    // --- Repository Interface ---
    pub trait CrisprRepository: Send + Sync {
        fn store_experiment(&self, exp: &CrisprExperiment) -> Result<(), CrisprError>;
        fn find_guides_for_gene(&self, gene: &GeneId) -> Result<Vec<GuideRna>, CrisprError>;
        fn find_off_targets_in_region(&self, region: &GenomicRegion) -> Result<Vec<OffTargetSite>, CrisprError>;
        fn rank_guides(&self, guides: &[GuideId], criteria: RankingCriteria) -> Result<Vec<(GuideId, f32)>, CrisprError>;
        fn nearest_by_guide_embedding(&self, embedding: &[f32], k: usize) -> Result<Vec<GuideRna>, CrisprError>;
    }

    pub struct RankingCriteria {
        pub on_target_weight: f32,
        pub specificity_weight: f32,
        pub gc_preference: (f32, f32),      // (min, max) preferred GC range
        pub avoid_genes: Vec<GeneId>,
    }

    #[derive(Debug)]
    pub enum CrisprError {
        NoPamSiteFound(GenomicRegion),
        OffTargetSearchTimeout,
        InvalidSpacerLength { expected: usize, actual: usize },
        GenomeGraphRequired,
    }
}
```

#### Domain Events

| Event                      | Payload                                     | Published When                 |
|----------------------------|---------------------------------------------|--------------------------------|
| `GuideDesigned`           | guide_id, gene, on_target_score, gc         | New guide RNA designed         |
| `OffTargetAnalysisComplete`| guide_id, off_target_count, specificity     | Off-target search finishes     |
| `HighRiskOffTarget`       | guide_id, off_target_position, gene, prob   | Off-target in critical gene    |
| `EditingPredicted`        | guide_id, edit_type, efficiency             | Outcome prediction complete    |
| `GuideRanked`             | experiment_id, top_guide, score             | Guide ranking finalized        |

---

## 4. Context Map: Relationships & Integration Patterns

```
+=====================================================================+
|                      CONTEXT MAP                                     |
+=====================================================================+
|                                                                      |
|  [Sequence Ingestion] ----(Published Language)----> [Alignment]      |
|         |                      FASTQ/CRAM                            |
|         |                                                            |
|  [Alignment] ---------(Published Language)--------> [Variant Calling]|
|         |                      BAM/CRAM                              |
|         |                                                            |
|  [Graph Genome] <===(Shared Kernel)===> [Alignment]                  |
|         |              GenomicCoordinates                            |
|         |              ContigId, GenomicPosition                     |
|         |                                                            |
|  [Variant Calling] ----(Published Language)----> [Annotation]        |
|         |                      VCF                                   |
|         |                                                            |
|  [Annotation] ---------(Conformist)-----------> [Pharmacogenomics]   |
|         |           ClinVar/PharmGKB schema                          |
|         |                                                            |
|  [Annotation] ----(Anti-Corruption Layer)-----> [Epigenomics]        |
|         |           EpigenomeAnnotationAdapter                       |
|         |                                                            |
|  [Variant Calling] --(Anti-Corruption Layer)--> [Population Genomics]|
|         |           PopulationVariantAdapter                         |
|         |                                                            |
|  [Sequence Ingestion] --(Published Language)--> [Pathogen Surveillance]
|         |                  FASTQ                                     |
|         |                                                            |
|  [Graph Genome] ----(Anti-Corruption Layer)---> [CRISPR Engineering] |
|         |           GenomeSearchAdapter                              |
|         |                                                            |
|  [Pathogen Surveillance] --(Customer/Supplier)--> [Population Genomics]
|                           Allele frequency data                      |
|                                                                      |
+=====================================================================+
```

### 4.1 Relationship Details

| Upstream Context        | Downstream Context        | Pattern                  | Shared Artifact              |
|-------------------------|---------------------------|--------------------------|------------------------------|
| Sequence Ingestion      | Alignment & Mapping       | Published Language       | `BasecalledRead` events via FASTQ-like contract |
| Alignment & Mapping     | Variant Calling           | Published Language       | `Alignment` stream via BAM-like contract |
| Graph Genome            | Alignment & Mapping       | Shared Kernel            | `GenomicPosition`, `ContigId`, `GenomicRegion` |
| Variant Calling         | Annotation                | Published Language       | `Variant` events via VCF-like contract |
| Annotation              | Pharmacogenomics          | Conformist               | PGx context conforms to ClinVar/PharmGKB schema |
| Annotation              | Epigenomics               | Anti-Corruption Layer    | `EpigenomeAnnotationAdapter` translates variant effects to chromatin context |
| Variant Calling         | Population Genomics       | Anti-Corruption Layer    | `PopulationVariantAdapter` aggregates individual calls to cohort frequencies |
| Sequence Ingestion      | Pathogen Surveillance     | Published Language       | Raw reads for metagenomic classification |
| Graph Genome            | CRISPR Engineering        | Anti-Corruption Layer    | `GenomeSearchAdapter` provides graph-aware PAM site search |
| Pathogen Surveillance   | Population Genomics       | Customer/Supplier        | Pathogen population frequencies feed allele tables |
| Epigenomics             | Annotation                | Anti-Corruption Layer    | Regulatory annotations enriching variant interpretation |

### 4.2 Anti-Corruption Layer Definitions

```rust
/// ACL: Epigenomics <-> Annotation
pub trait EpigenomeAnnotationAdapter: Send + Sync {
    /// Translate a variant position into its epigenomic context
    fn get_regulatory_context(
        &self,
        position: &GenomicPosition,
        cell_type: &CellType,
    ) -> Result<RegulatoryContext, AdapterError>;
}

pub struct RegulatoryContext {
    pub chromatin_state: ChromatinState,
    pub methylation_level: Option<f32>,
    pub in_enhancer: bool,
    pub in_promoter: bool,
    pub tad_boundary_distance: Option<u64>,
    pub compartment: CompartmentType,
}

/// ACL: Graph Genome <-> CRISPR Engineering
pub trait GenomeSearchAdapter: Send + Sync {
    /// Search for PAM sites across all haplotypes in the pangenome graph
    fn find_pam_sites_in_graph(
        &self,
        graph_id: &GraphId,
        region: &GenomicRegion,
        pam_pattern: &[u8],
    ) -> Result<Vec<PamSiteResult>, AdapterError>;
}

pub struct PamSiteResult {
    pub position: GenomicPosition,
    pub node_id: NodeId,
    pub haplotype_count: u32,        // how many haplotypes contain this site
    pub allele_frequency: f32,
}

/// ACL: Variant Calling <-> Population Genomics
pub trait PopulationVariantAdapter: Send + Sync {
    /// Aggregate individual variant calls into population-level frequencies
    fn aggregate_to_population(
        &self,
        variants: &[Variant],
        cohort: &Cohort,
    ) -> Result<Vec<AlleleFrequencyEntry>, AdapterError>;
}

#[derive(Debug)]
pub enum AdapterError {
    UpstreamUnavailable(String),
    TranslationFailed(String),
    SchemaVersionMismatch { expected: String, actual: String },
}
```

---

## 5. Domain Event Flow

The complete event-driven pipeline flows as follows:

```
   +-----------------------+
   | Instrument Signal     |
   +-----------+-----------+
               |
               v
   +------------------------+     +------------------------+
   | SignalChunkReceived     |     | RunStarted             |
   +------------------------+     +------------------------+
               |
               v
   +------------------------+
   | ReadBasecalled          |---+
   +------------------------+   |
               |                |
               v                v
   +------------------------+  +-----------------------------+
   | AlignmentCompleted      |  | (Pathogen Surveillance)     |
   +------------------------+  | PathogenDetected            |
               |               | AmrGeneDetected             |
               v               | OutbreakClusterExpanded     |
   +------------------------+  +-----------------------------+
   | VariantCalled           |
   | StructuralVariantFound  |
   | PhasingCompleted        |
   +----------+-------------+
              |
     +--------+--------+-------------------+
     v                  v                   v
+----------------+ +------------------+ +---------------------+
| VariantAnnotated| | AlleleFrequency  | | MethylationProfiled |
| PathogenicFound | | Updated          | | TadBoundaryDisrupted|
| AcmgReclassified| | GwasSignificant  | +---------------------+
+--------+-------+ | Hit              |
         |         | AncestryInferred |
         v         +------------------+
+-------------------+
| StarAllelesCalled  |
| DrugInteraction    |
| Identified         |
| DosingAlert        |
| Generated          |
+-------------------+

CRISPR Engineering operates on-demand:
  GraphConstructed + VariantAnnotated --> GuideDesigned
  GuideDesigned --> OffTargetAnalysisComplete
  OffTargetAnalysisComplete --> EditingPredicted --> GuideRanked
```

---

## 6. Mapping to RuVector Crates

Each bounded context maps to specific RuVector infrastructure crates:

```
+===========================================================================+
|  BOUNDED CONTEXT             |  PRIMARY RUVECTOR CRATES                   |
+===========================================================================+
|                              |                                            |
|  1. Sequence Ingestion       |  sona              - adaptive basecalling  |
|                              |  ruvector-core      - read embedding store |
|                              |  ruvector-delta-*   - incremental updates  |
|                              |  ruvector-temporal-tensor - signal windows |
|                              |                                            |
|  2. Alignment & Mapping      |  ruvector-core      - seed index (HNSW)   |
|                              |  ruvector-graph     - graph alignment      |
|                              |  ruvector-mincut    - graph partitioning   |
|                              |  ruvector-dag       - alignment DAG        |
|                              |                                            |
|  3. Variant Calling          |  ruvector-gnn       - variant effect pred. |
|                              |  ruvector-core      - variant embeddings   |
|                              |  ruvector-delta-core- incremental calling  |
|                              |  ruvector-sparse-inference - genotyper     |
|                              |                                            |
|  4. Graph Genome             |  ruvector-graph     - genome graph store   |
|                              |  ruvector-mincut    - min-cut partitioning |
|                              |  ruvector-dag       - variant DAGs         |
|                              |  cognitum-gate-kernel - graph sharding     |
|                              |  ruvector-delta-graph - incremental graphs |
|                              |                                            |
|  5. Annotation               |  ruvector-gnn       - effect prediction   |
|                              |  ruvector-core      - annotation vectors  |
|                              |  ruvector-attention  - consequence pred.  |
|                              |  ruvector-collections - lookup tables     |
|                              |                                            |
|  6. Epigenomics              |  ruvector-temporal-tensor - time-series   |
|                              |  ruvector-core      - methylation vectors |
|                              |  ruvector-graph     - Hi-C contact graphs |
|                              |  ruvector-attention  - 3D structure pred. |
|                              |                                            |
|  7. Pharmacogenomics         |  sona               - drug response pred. |
|                              |  ruvector-core      - PGx embeddings      |
|                              |  ruvector-gnn       - interaction graphs  |
|                              |  ruvector-sparse-inference - allele call  |
|                              |                                            |
|  8. Population Genomics      |  ruvector-core      - PCA embeddings      |
|                              |  ruvector-cluster   - ancestry clustering |
|                              |  ruvector-math      - statistics/PCA      |
|                              |  ruvector-delta-consensus - cohort sync   |
|                              |                                            |
|  9. Pathogen Surveillance    |  ruvector-hyperbolic-hnsw - taxonomy tree |
|                              |  ruvector-core      - pathogen vectors    |
|                              |  ruvector-cluster   - outbreak clustering |
|                              |  ruvector-graph     - transmission graphs |
|                              |  ruvector-raft      - distributed sync    |
|                              |                                            |
|  10. CRISPR Engineering      |  ruvector-attention  - off-target model   |
|                              |  cognitum-gate-kernel - gated seq. attn.  |
|                              |  ruvector-graph     - pangenome search    |
|                              |  ruvector-core      - guide embeddings    |
|                              |  ruvector-mincut    - graph-aware search  |
|                              |                                            |
+===========================================================================+
```

### 6.1 Crate Mapping Rationale

**ruvector-core** serves as the foundational vector storage layer across all ten
contexts. Every entity with an embedding field (reads, variants, guides, taxonomy
nodes, drug responses) stores its vectors through ruvector-core's HNSW index, enabling
sub-millisecond approximate nearest neighbor queries. This is the universal
infrastructure crate.

**sona** (Self-Optimizing Neural Architecture) drives two key functions:
1. *Sequence Ingestion*: Adaptive basecalling with two-tier LoRA fine-tuning. The
   `MicroLoRA` layer adapts per-flowcell, while `BaseLoRA` captures instrument-level
   patterns. EWC++ prevents catastrophic forgetting across runs.
2. *Pharmacogenomics*: Drug response prediction using the `ReasoningBank` to accumulate
   pharmacological evidence and `TrajectoryBuffer` to track patient outcome trajectories.

**ruvector-mincut** powers two contexts:
1. *Graph Genome*: The `SubpolynomialMinCut` algorithm partitions pangenome graphs
   into balanced components for parallel alignment. `HierarchicalDecomposition`
   enables multi-resolution graph traversal.
2. *CRISPR Engineering*: Min-cut analysis identifies structural boundaries in the
   genome graph that affect guide specificity across haplotypes.

**ruvector-gnn** provides Graph Neural Network inference for:
1. *Variant Calling*: Predicting variant effect vectors from local graph topology
   around variant sites. The GNN operates on the alignment pileup graph.
2. *Annotation*: Predicting functional consequences and pathogenicity scores using
   gene interaction networks as input graphs.
3. *Pharmacogenomics*: Modeling drug-gene-variant interaction networks.

**ruvector-attention** with its `ScaledDotProductAttention`, MoE router, and sparse
attention masks serves:
1. *Annotation*: Transformer-based consequence prediction attending to protein sequence
   context windows around variant positions.
2. *Epigenomics*: Attention over Hi-C contact matrices for 3D genome structure
   prediction.
3. *CRISPR Engineering*: Gated attention models for off-target prediction, with the
   guide sequence as query attending to candidate genomic sites as keys.

**ruvector-hyperbolic-hnsw** is purpose-built for the *Pathogen Surveillance* context.
Taxonomic trees are naturally hierarchical, and hyperbolic space embeddings
(Poincare ball model) preserve tree distances with exponentially less distortion than
Euclidean space. The `ShardedHyperbolicHnsw` partitions the taxonomy across curvature
regions, and `DualSpaceIndex` enables both Euclidean sequence-similarity and hyperbolic
taxonomy-distance queries.

**cognitum-gate-kernel** provides the gated graph attention mechanism used in:
1. *Graph Genome*: The `CompactGraph` structure with `ShardEdge` and `VertexEntry`
   maps directly to genome graph shards. The `EvidenceAccumulator` tracks alignment
   evidence across graph bubbles.
2. *CRISPR Engineering*: Gated attention over sequence-PAM interactions.

**ruvector-dag** models dependency structures in:
1. *Alignment*: The `QueryDag` represents multi-seed alignment chains as DAGs.
   `TopologicalIterator` orders chain extensions.
2. *Variant Calling*: Variant dependency DAGs where structural variants may encompass
   smaller variants. `MinCutResult` identifies independent variant blocks.
3. *Graph Genome*: Bubble nesting hierarchies as DAGs.

**ruvector-delta-*** crates enable incremental processing:
1. *Sequence Ingestion*: `ruvector-delta-core` streams basecalling deltas as new
   signal chunks arrive, using `DeltaWindow` for batched processing.
2. *Graph Genome*: `ruvector-delta-graph` propagates graph updates when new population
   data is incorporated without full reconstruction.
3. *Population Genomics*: `ruvector-delta-consensus` synchronizes allele frequency
   updates across distributed cohort nodes via Raft consensus.

**ruvector-temporal-tensor** stores time-series data:
1. *Sequence Ingestion*: Raw signal windows with tiered storage (hot/warm/cold) via
   `TierPolicy` and `BlockMeta`.
2. *Epigenomics*: Temporal methylation profiles tracking changes across cell
   differentiation or treatment time courses.

**ruvector-sparse-inference** provides lightweight neural inference:
1. *Variant Calling*: The `SparseInferenceEngine` runs quantized genotyping models
   with `QuantizedWeights` and `NeuronCache` for efficient per-site inference.
2. *Pharmacogenomics*: Sparse star-allele calling models.

**ruvector-cluster** handles unsupervised grouping:
1. *Population Genomics*: Ancestry clustering from PCA embeddings.
2. *Pathogen Surveillance*: Outbreak cluster detection from SNP distance matrices.

**ruvector-graph** is the general-purpose property graph database used across six
contexts for storing genome graphs, Hi-C contact networks, drug interaction networks,
and transmission graphs. Its `TransactionManager` with `IsolationLevel` support
ensures ACID properties for concurrent pipeline stages.

---

## 7. Deployment Architecture

```
+===========================================================================+
|                   DNA ANALYZER DEPLOYMENT                                 |
+===========================================================================+
|                                                                           |
|  Tier 1: Streaming Layer (Hot Path)                                       |
|  +------+ +----------+ +-----------+ +------------+                       |
|  | Ingest| | Alignment| | Variant   | | Pathogen   |                      |
|  | Worker| | Worker   | | Caller    | | Classifier |                      |
|  +---+---+ +----+-----+ +-----+-----+ +-----+------+                     |
|      |          |              |              |                            |
|      v          v              v              v                            |
|  +-----------------------------------------------------+                 |
|  |  ruvector-delta-core Event Bus (at-least-once)       |                 |
|  +-----------------------------------------------------+                 |
|                                                                           |
|  Tier 2: Analytical Layer (Warm Path)                                     |
|  +----------+ +------------+ +----------+ +-----------+                   |
|  | Annotator| | Population | | Epigenome| | PGx Engine|                   |
|  | Service  | | Aggregator | | Profiler | |           |                   |
|  +----------+ +------------+ +----------+ +-----------+                   |
|                                                                           |
|  Tier 3: Engineering Layer (On-Demand)                                    |
|  +-------------------+                                                    |
|  | CRISPR Designer   |                                                    |
|  | (GPU-accelerated) |                                                    |
|  +-------------------+                                                    |
|                                                                           |
|  Infrastructure:                                                          |
|  +------------------+ +------------------+ +-------------------+          |
|  | ruvector-core    | | ruvector-raft    | | ruvector-postgres |          |
|  | (HNSW Indices)   | | (Consensus)      | | (Durable Store)  |          |
|  +------------------+ +------------------+ +-------------------+          |
|                                                                           |
+===========================================================================+
```

---

## 8. Scalability Considerations

| Concern                   | Strategy                                              | RuVector Crate           |
|---------------------------|-------------------------------------------------------|--------------------------|
| Read throughput           | Sharded ingestion workers, streaming delta windows     | ruvector-delta-core      |
| Alignment parallelism     | Min-cut graph partitions, independent per-partition    | ruvector-mincut          |
| Variant call fan-out      | DAG-based independent variant blocks                   | ruvector-dag             |
| Pangenome graph size      | Hierarchical decomposition, compact graph shards       | cognitum-gate-kernel     |
| Taxonomy search           | Hyperbolic HNSW with curvature-aware sharding          | ruvector-hyperbolic-hnsw |
| Cross-context sync        | Raft consensus for distributed cohort updates           | ruvector-raft            |
| Embedding index growth    | Tiered storage with temporal tensor compression         | ruvector-temporal-tensor |
| Neural inference latency  | Sparse quantized models with neuron caching             | ruvector-sparse-inference|
| Off-target search         | Attention mask sparsification, graph-partitioned search | ruvector-attention       |

---

## 9. Cross-Cutting Concerns

### 9.1 Shared Kernel: Genomic Coordinates

The following types constitute the Shared Kernel used by all bounded contexts:

```rust
/// Shared Kernel - used by ALL bounded contexts
pub mod genomic_coordinates {
    #[derive(Clone, PartialEq, Eq, Hash)]
    pub struct ContigId(pub String);

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct GenomicPosition {
        pub contig_index: u32,
        pub offset: u64,
        pub strand: Strand,
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Strand { Forward, Reverse }

    #[derive(Clone)]
    pub struct GenomicRegion {
        pub contig: ContigId,
        pub start: u64,
        pub end: u64,
    }

    pub struct SampleId(pub String);
    pub struct GeneId(pub String);
}
```

### 9.2 Observability

All domain events carry correlation IDs for distributed tracing:

```rust
pub struct EventEnvelope<T> {
    pub event_id: u128,
    pub correlation_id: u128,         // traces event across contexts
    pub source_context: &'static str, // e.g., "variant_calling"
    pub timestamp: u64,
    pub payload: T,
}
```

### 9.3 Security & Compliance

- All `SampleId` values are pseudonymized; a separate Identity Mapping Service
  (outside the domain model) handles PHI linkage
- Variant data at rest encrypted via ruvector-core's storage layer
- Audit log captures every domain event envelope for HIPAA compliance
- Access control is role-based per bounded context (clinician, researcher, bioinformatician)

---

## 10. Decision Log

| Decision                                                 | Rationale                                                     |
|----------------------------------------------------------|---------------------------------------------------------------|
| 10 bounded contexts (not fewer)                          | Genomics subdomains have genuinely distinct ubiquitous languages; collapsing them creates ambiguity |
| Shared Kernel for genomic coordinates only               | Position/region types are universal; all other types are context-specific to prevent coupling |
| Anti-corruption layers for cross-domain queries          | Epigenomics and Population Genomics have fundamentally different data models from Variant Calling |
| Conformist for Pharmacogenomics consuming Annotation     | PGx standards (CPIC/DPWG) already define the schema; conforming avoids translation overhead |
| Published Language (VCF/BAM-like) for pipeline stages    | Industry-standard formats reduce integration friction |
| ruvector-hyperbolic-hnsw for taxonomy (not flat HNSW)    | Taxonomy is hierarchical; hyperbolic space preserves tree distances with O(log n) dimensions |
| Delta-based incremental updates throughout               | Genomic pipelines process terabytes; full recomputation is prohibitive |
| GNN for variant effect prediction                        | Graph topology around variants carries structural information that MLP/CNN cannot capture |
| Gated attention for CRISPR off-target                    | Sequence-PAM interaction requires position-aware attention with gating for mismatch tolerance |
| SONA for adaptive basecalling                            | Instrument drift requires online adaptation; EWC++ prevents forgetting across runs |

---

## References

- ADR-001: RuVector Core Architecture
- ADR-016: Delta-Behavior System DDD Architecture
- Evans, Eric. *Domain-Driven Design: Tackling Complexity in the Heart of Software*. 2003.
- Poplin et al. "A universal SNP and small-indel variant caller using deep neural networks." *Nature Biotechnology* 36, 983-987 (2018).
- Garrison et al. "Variation graph toolkit improves read mapping by representing genetic variation in the reference." *Nature Biotechnology* 36, 875-879 (2018).
- Rautiainen et al. "Telomere-to-telomere assembly of a complete human genome." *Science* 376, 44-53 (2022).
- Nickel & Kiela. "Poincare Embeddings for Learning Hierarchical Representations." *NeurIPS* 2017.
