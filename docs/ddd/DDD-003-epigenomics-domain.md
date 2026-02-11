# DDD-003: Epigenomics Domain Model

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Related ADR**: ADR-014-coherence-engine, ADR-015-coherence-gated-transformer
**Related DDD**: DDD-001-coherence-gate-domain, DDD-004-crispr-engineering-domain

---

## Overview

This document defines the Domain-Driven Design model for the Epigenomics bounded context within the RuVector DNA Analyzer. The domain covers methylation pattern analysis, chromatin accessibility profiling, histone modification mapping, and 3D genome architecture reconstruction at single-base resolution. It integrates deeply with four RuVector crates: `ruvector-mincut` for TAD boundary detection, `ruvector-attention` for long-range interaction prediction, `ruvector-gnn` for 3D genome functional predictions, and `ruvector-hyperbolic-hnsw` for hierarchical chromatin state search.

---

## Strategic Design

### Domain Vision Statement

> The Epigenomics domain provides multi-layered, single-base-resolution analysis of the regulatory genome -- methylation landscapes, open chromatin, histone marks, and three-dimensional folding -- enabling researchers to move from raw sequencing data to actionable regulatory insights through graph-native algorithms that were previously infeasible at genome scale.

### Core Domain

**Regulatory Landscape Reconstruction** is the core domain. The differentiating capability is:

- Not alignment (that is an upstream infrastructure concern)
- Not variant calling (that belongs to the Genomics bounded context)
- **The novel capability**: Reconstructing the full regulatory state of a genomic locus by integrating methylation, accessibility, histone marks, and 3D contact structure into a single coherent model, powered by graph-cut and attention-based algorithms.

### Supporting Domains

| Domain | Role | Boundary |
|--------|------|----------|
| **Alignment Pipeline** | Produce aligned reads (BAM/CRAM) | Generic, infrastructure |
| **Reference Genome** | Chromosome coordinates, gene annotations | Generic, external |
| **Variant Calling** | SNP/indel context for allele-specific analysis | Separate bounded context |
| **Sequencing QC** | Read quality, bisulfite conversion rates | Generic, infrastructure |

### Generic Subdomains

- Logging and observability
- File I/O (BAM, BED, BigWig, .hic)
- Configuration and parameter management
- Coordinate system transformations (0-based, 1-based, BED intervals)

---

## Ubiquitous Language

### Core Terms

| Term | Definition | Context |
|------|------------|---------|
| **Methylation Level** | Fraction of reads showing methylation at a cytosine (0.0-1.0) | Core metric |
| **CpG Context** | Cytosine followed by guanine; the primary methylation target in mammals | Sequence context |
| **CHG/CHH Context** | Non-CpG methylation contexts (H = A, C, or T); prevalent in plants | Sequence context |
| **Chromatin Accessibility** | Degree to which genomic DNA is physically accessible to transcription factors | Core metric |
| **Peak** | A statistically significant region of enriched signal (accessibility or histone mark) | Statistical entity |
| **Histone Mark** | A covalent post-translational modification on a histone tail (e.g., H3K4me3, H3K27ac) | Modification type |
| **TAD** | Topologically Associating Domain; a self-interacting genomic region in 3D space | 3D structure |
| **Compartment** | A/B classification of chromatin: A = active/open, B = inactive/closed | 3D structure |
| **Contact Map** | Matrix of interaction frequencies between genomic loci from Hi-C data | 3D data |
| **Enhancer** | A distal regulatory element that activates gene expression | Functional annotation |
| **Promoter** | Region immediately upstream of a gene's transcription start site | Functional annotation |

### Algorithmic Terms

| Term | Definition | Context |
|------|------------|---------|
| **Insulation Score** | Local measure of boundary strength between adjacent TADs | TAD detection |
| **Graph Cut** | Partition of the contact map graph that identifies TAD boundaries via min-cut | RuVector integration |
| **Attention Score** | Learned weight for long-range locus-to-locus interactions | Enhancer prediction |
| **Hyperbolic Embedding** | Representation of hierarchical chromatin states in Poincare ball space | State search |

---

## Bounded Contexts

### Context Map

```
+-----------------------------------------------------------------------------+
|                         EPIGENOMICS CONTEXT                                  |
|                           (Core Domain)                                      |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
|  | Methylation   |  | Chromatin     |  | Histone       |  | 3D Genome     |  |
|  | Subcontext    |  | Subcontext    |  | Subcontext    |  | Subcontext    |  |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
+-----------------------------------------------------------------------------+
         |                  |                  |                  |
         | Upstream         | Upstream         | Upstream         | Upstream
         v                  v                  v                  v
+------------------+ +------------------+ +------------------+ +------------------+
|   ALIGNMENT      | |   PEAK CALLING   | |  CHIP-SEQ        | |   HI-C           |
|   PIPELINE       | |   PIPELINE       | |  PIPELINE        | |   PIPELINE       |
|  (Infrastructure)| | (Infrastructure) | | (Infrastructure) | | (Infrastructure) |
+------------------+ +------------------+ +------------------+ +------------------+
         |                                                            |
         | Downstream                                                 | Downstream
         v                                                            v
+------------------+                                         +------------------+
|   CRISPR         |                                         |   VARIANT        |
|   ENGINEERING    |                                         |   CALLING        |
|   CONTEXT        |                                         |   CONTEXT        |
+------------------+                                         +------------------+
```

### Epigenomics Context (Core)

**Responsibility**: Reconstruct the regulatory landscape from multi-omic epigenomic data.

**Key Aggregates**:
- MethylationProfile (Aggregate Root)
- ChromatinLandscape
- HistoneModificationMap
- GenomeTopology

**Anti-Corruption Layers**:
- Alignment ACL (translates BAM records to methylation calls)
- HiC ACL (translates contact matrices to graph structures)
- Annotation ACL (translates gene annotations to promoter/enhancer loci)

---

## Aggregates

### MethylationProfile (Root Aggregate)

The central aggregate for a sample's methylation state across a genomic region.

```
+-----------------------------------------------------------------------+
|                     METHYLATION PROFILE                                |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  profile_id: ProfileId                                                 |
|  sample_id: SampleId                                                   |
|  region: GenomicRegion { chromosome, start, end }                      |
|  assembly: Assembly (e.g., hg38, mm39)                                 |
|  sites: Vec<MethylationSite>                                           |
|  global_methylation: f64                                               |
|  dmr_calls: Vec<DifferentiallyMethylatedRegion>                        |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | MethylationSite (Entity)                                         | |
|  |  site_id: SiteId                                                 | |
|  |  position: GenomicPosition { chromosome: String, offset: u64 }   | |
|  |  methylation_level: f64 (0.0 - 1.0)                             | |
|  |  coverage: u32                                                   | |
|  |  context: MethylationContext { CpG | CHG | CHH }                | |
|  |  strand: Strand { Plus | Minus }                                 | |
|  |  confidence: f64                                                 | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | DifferentiallyMethylatedRegion (Entity)                          | |
|  |  dmr_id: DmrId                                                  | |
|  |  region: GenomicRegion                                           | |
|  |  mean_delta: f64                                                 | |
|  |  p_value: f64                                                    | |
|  |  q_value: f64                                                    | |
|  |  direction: Direction { Hyper | Hypo }                           | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - methylation_level in [0.0, 1.0] for every site                     |
|  - coverage >= 1 for every reported site                               |
|  - sites sorted by genomic position                                    |
|  - All sites fall within the profile's region                          |
|  - q_value uses Benjamini-Hochberg correction                          |
+-----------------------------------------------------------------------+
```

### ChromatinLandscape (Aggregate)

Represents accessible chromatin regions for a sample.

```
+-----------------------------------------------------------------------+
|                    CHROMATIN LANDSCAPE                                  |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  landscape_id: LandscapeId                                             |
|  sample_id: SampleId                                                   |
|  assay_type: AssayType { ATACseq | DNaseseq | FAIREseq }              |
|  regions: Vec<ChromatinRegion>                                         |
|  footprints: Vec<TranscriptionFactorFootprint>                         |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | ChromatinRegion (Entity)                                         | |
|  |  region_id: RegionId                                             | |
|  |  start: u64                                                      | |
|  |  end: u64                                                        | |
|  |  chromosome: String                                              | |
|  |  accessibility_score: f64                                        | |
|  |  peak_summit: u64                                                | |
|  |  p_value: f64                                                    | |
|  |  fold_enrichment: f64                                            | |
|  |  motifs: Vec<MotifHit>                                           | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | MotifHit (Value Object)                                          | |
|  |  motif_id: String                                                | |
|  |  transcription_factor: String                                    | |
|  |  position: u64                                                   | |
|  |  score: f64                                                      | |
|  |  strand: Strand                                                  | |
|  |  p_value: f64                                                    | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - accessibility_score >= 0.0                                          |
|  - end > start for every region                                        |
|  - Regions do not overlap after merging                                 |
|  - peak_summit falls within [start, end]                               |
+-----------------------------------------------------------------------+
```

### HistoneModificationMap (Aggregate)

Tracks histone marks across the genome for a single mark type and sample.

```
+-----------------------------------------------------------------------+
|                  HISTONE MODIFICATION MAP                              |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  map_id: MapId                                                         |
|  sample_id: SampleId                                                   |
|  mark_type: HistoneMarkType                                            |
|       { H3K4me1 | H3K4me3 | H3K27ac | H3K27me3 | H3K36me3 |         |
|         H3K9me3 | H4K20me1 | Custom(String) }                         |
|  modifications: Vec<HistoneModification>                               |
|  broad_domains: Vec<BroadDomain>                                       |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | HistoneModification (Entity)                                     | |
|  |  mod_id: ModId                                                   | |
|  |  position: GenomicPosition                                       | |
|  |  signal_intensity: f64                                           | |
|  |  fold_change: f64                                                | |
|  |  peak_type: PeakType { Narrow | Broad }                         | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | BroadDomain (Entity)                                             | |
|  |  domain_id: DomainId                                             | |
|  |  region: GenomicRegion                                           | |
|  |  mean_signal: f64                                                | |
|  |  chromatin_state: ChromatinState { Active | Poised | Repressed } | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - signal_intensity >= 0.0                                             |
|  - fold_change >= 0.0                                                  |
|  - One map per (sample, mark_type) pair                                |
|  - BroadDomain regions do not overlap                                  |
+-----------------------------------------------------------------------+
```

### GenomeTopology (Aggregate)

Represents the 3D structure of the genome from Hi-C or similar data.

```
+-----------------------------------------------------------------------+
|                     GENOME TOPOLOGY                                    |
|                    (Aggregate Root)                                    |
+-----------------------------------------------------------------------+
|  topology_id: TopologyId                                               |
|  sample_id: SampleId                                                   |
|  resolution: u32 (bin size in bp: 1000, 5000, 10000, ...)             |
|  chromosome: String                                                    |
|  contact_graph: ContactGraph                                           |
|  tads: Vec<TAD>                                                        |
|  compartments: Vec<CompartmentSegment>                                 |
|  interactions: Vec<EnhancerPromoterInteraction>                        |
+-----------------------------------------------------------------------+
|  +------------------------------------------------------------------+ |
|  | ContactGraph (Value Object)                                      | |
|  |  bins: Vec<GenomicBin>                                           | |
|  |  contacts: SparseMatrix<f64> (ICE-normalized)                    | |
|  |  total_contacts: u64                                             | |
|  |                                                                  | |
|  |  fn as_dynamic_graph(&self) -> ruvector_mincut::DynamicGraph     | |
|  |  fn insulation_scores(&self, window: u32) -> Vec<f64>            | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | TAD (Entity)                                                     | |
|  |  tad_id: TadId                                                   | |
|  |  boundary_left: GenomicPosition                                  | |
|  |  boundary_right: GenomicPosition                                 | |
|  |  insulation_score_left: f64                                      | |
|  |  insulation_score_right: f64                                     | |
|  |  intra_tad_contacts: f64                                         | |
|  |  inter_tad_contacts: f64                                         | |
|  |  compartment: Compartment { A | B | Intermediate }              | |
|  |  sub_tads: Vec<TadId>  (hierarchical nesting)                    | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | EnhancerPromoterInteraction (Entity)                             | |
|  |  interaction_id: InteractionId                                   | |
|  |  enhancer_locus: GenomicRegion                                   | |
|  |  promoter_locus: GenomicRegion                                   | |
|  |  target_gene: String                                             | |
|  |  interaction_strength: f64                                       | |
|  |  linear_distance: u64 (bp)                                      | |
|  |  attention_score: f64 (from ruvector-attention)                  | |
|  |  confidence: f64                                                 | |
|  +------------------------------------------------------------------+ |
|  +------------------------------------------------------------------+ |
|  | CompartmentSegment (Value Object)                                | |
|  |  region: GenomicRegion                                           | |
|  |  compartment: Compartment                                        | |
|  |  eigenvector_value: f64 (PC1 of contact matrix)                  | |
|  +------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+
|  Invariants:                                                           |
|  - TAD boundaries are non-overlapping at the same hierarchy level      |
|  - intra_tad_contacts > inter_tad_contacts (TAD definition)           |
|  - Contact matrix is symmetric                                         |
|  - Resolution must be a positive integer divisor of chromosome length  |
|  - interaction_strength in [0.0, 1.0]                                  |
|  - sub_tads are strictly contained within parent TAD boundaries        |
+-----------------------------------------------------------------------+
```

---

## Value Objects

### GenomicPosition

```rust
struct GenomicPosition {
    chromosome: String,
    offset: u64,
}

impl GenomicPosition {
    fn distance_to(&self, other: &GenomicPosition) -> Option<u64>;
    fn is_same_chromosome(&self, other: &GenomicPosition) -> bool;
}
```

### GenomicRegion

```rust
struct GenomicRegion {
    chromosome: String,
    start: u64,
    end: u64,
}

impl GenomicRegion {
    fn length(&self) -> u64 { self.end - self.start }
    fn overlaps(&self, other: &GenomicRegion) -> bool;
    fn contains(&self, pos: &GenomicPosition) -> bool;
    fn midpoint(&self) -> u64 { (self.start + self.end) / 2 }
}
```

### InsulationScore

```rust
struct InsulationScore {
    position: GenomicPosition,
    score: f64,
    window_size: u32,
    is_boundary: bool,
    boundary_strength: f64,
}

impl InsulationScore {
    fn is_tad_boundary(&self, threshold: f64) -> bool {
        self.is_boundary && self.boundary_strength >= threshold
    }
}
```

### ChromatinStateVector

Embedding of chromatin state for hyperbolic search.

```rust
struct ChromatinStateVector {
    methylation: f64,
    accessibility: f64,
    h3k4me3: f64,
    h3k27ac: f64,
    h3k27me3: f64,
    compartment_score: f64,
}

impl ChromatinStateVector {
    fn to_embedding(&self) -> Vec<f32>;
    fn bivalent_score(&self) -> f64 {
        (self.h3k4me3 * self.h3k27me3).sqrt()
    }
}
```

---

## Domain Events

### Methylation Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `MethylationCalled` | Bisulfite/nanopore processing complete | profile_id, site_count, mean_level |
| `DMRIdentified` | Differential analysis complete | dmr_id, region, delta, direction |
| `MethylationDriftDetected` | Temporal comparison | region, old_level, new_level, magnitude |

### Chromatin Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `PeaksCalled` | Peak calling pipeline complete | landscape_id, peak_count, frip_score |
| `FootprintDetected` | Motif analysis complete | tf_name, position, score |
| `AccessibilityChanged` | Differential accessibility | region, fold_change, direction |

### Histone Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `HistoneMarksProfiled` | ChIP-seq processing complete | map_id, mark_type, peak_count |
| `ChromatinStateAssigned` | Multi-mark integration | region, state, confidence |
| `BivalentDomainIdentified` | H3K4me3+H3K27me3 co-occurrence | region, bivalent_score |

### 3D Genome Events

| Event | Trigger | Payload |
|-------|---------|---------|
| `ContactMapBuilt` | Hi-C pipeline complete | topology_id, resolution, bin_count |
| `TADBoundaryDetected` | Min-cut analysis complete | tad_id, boundaries, cut_value |
| `CompartmentSwitched` | A/B compartment change detected | region, old_compartment, new_compartment |
| `EnhancerPromoterLinked` | Interaction prediction complete | enhancer, promoter, strength, gene |
| `TopologyReconstructed` | Full 3D model built | topology_id, tad_count, interaction_count |

---

## Domain Services

### MethylationCaller

Converts aligned bisulfite-seq or nanopore reads to per-site methylation calls.

```rust
trait MethylationCaller {
    /// Call methylation from aligned reads
    async fn call(
        &self,
        alignments: &AlignmentSource,
        region: &GenomicRegion,
        config: &MethylationCallerConfig,
    ) -> Result<MethylationProfile, EpigenomicsError>;

    /// Identify differentially methylated regions between two profiles
    fn call_dmrs(
        &self,
        control: &MethylationProfile,
        treatment: &MethylationProfile,
        config: &DmrConfig,
    ) -> Result<Vec<DifferentiallyMethylatedRegion>, EpigenomicsError>;

    /// Supported calling modes
    fn supported_modes(&self) -> Vec<CallingMode>;
}

struct MethylationCallerConfig {
    min_coverage: u32,
    min_base_quality: u8,
    context_filter: Vec<MethylationContext>,
    calling_mode: CallingMode,  // Bisulfite | Nanopore | EMseq
}
```

### ChromatinPeakCaller

Identifies accessible chromatin regions from ATAC-seq or DNase-seq data.

```rust
trait ChromatinPeakCaller {
    /// Call peaks from accessibility data
    async fn call_peaks(
        &self,
        signal: &SignalTrack,
        control: Option<&SignalTrack>,
        config: &PeakCallerConfig,
    ) -> Result<ChromatinLandscape, EpigenomicsError>;

    /// Identify TF footprints within accessible regions
    fn call_footprints(
        &self,
        landscape: &ChromatinLandscape,
        motif_database: &MotifDatabase,
    ) -> Result<Vec<TranscriptionFactorFootprint>, EpigenomicsError>;
}

struct PeakCallerConfig {
    q_value_threshold: f64,
    min_peak_length: u32,
    max_gap: u32,
    peak_model: PeakModel,  // Narrow | Broad | Mixed
}
```

### TADDetector

Detects TAD boundaries from Hi-C contact maps using `ruvector-mincut`.

```rust
trait TADDetector {
    /// Detect TADs from contact map using graph min-cut
    ///
    /// Algorithm:
    /// 1. Convert contact map to DynamicGraph (bins = vertices, contacts = weighted edges)
    /// 2. Apply ruvector-mincut to find minimum cuts that partition the graph
    /// 3. Boundaries correspond to min-cut edges in the contact graph
    /// 4. Hierarchical TADs via recursive application at multiple resolutions
    async fn detect_tads(
        &self,
        contact_graph: &ContactGraph,
        config: &TadDetectorConfig,
    ) -> Result<Vec<TAD>, EpigenomicsError>;

    /// Detect hierarchical TAD structure using j-tree decomposition
    fn detect_hierarchical(
        &self,
        contact_graph: &ContactGraph,
        resolutions: &[u32],
    ) -> Result<Vec<TAD>, EpigenomicsError>;

    /// Compute insulation scores using graph connectivity
    fn compute_insulation(
        &self,
        contact_graph: &ContactGraph,
        window_sizes: &[u32],
    ) -> Result<Vec<InsulationScore>, EpigenomicsError>;
}

struct TadDetectorConfig {
    min_tad_size: u32,         // Minimum TAD size in bins
    max_tad_size: u32,         // Maximum TAD size in bins
    boundary_strength: f64,     // Min-cut threshold for boundary calls
    hierarchical: bool,         // Enable nested TAD detection
    algorithm: TadAlgorithm,    // MinCut | Insulation | Armatus | Dixon
}

/// TAD detection via ruvector-mincut integration
enum TadAlgorithm {
    /// Use ruvector_mincut::MinCutBuilder for exact boundary detection
    MinCut,
    /// Use ruvector_mincut::SubpolynomialMinCut for dynamic updates
    DynamicMinCut,
    /// Use ruvector_mincut::jtree::JTreeHierarchy for hierarchical detection
    JTreeHierarchical,
    /// Classical insulation score method
    Insulation,
    /// Armatus algorithm
    Armatus,
    /// Dixon et al. directionality index
    Dixon,
}
```

### EnhancerPredictor

Multi-omic integration for enhancer-promoter interaction prediction using `ruvector-attention`.

```rust
trait EnhancerPredictor {
    /// Predict enhancer-promoter interactions from multi-omic features
    ///
    /// Uses ruvector-attention's GraphAttention trait to learn long-range
    /// interaction weights between distal regulatory elements and promoters.
    async fn predict_interactions(
        &self,
        topology: &GenomeTopology,
        chromatin: &ChromatinLandscape,
        histones: &[HistoneModificationMap],
        methylation: &MethylationProfile,
        config: &EnhancerPredictorConfig,
    ) -> Result<Vec<EnhancerPromoterInteraction>, EpigenomicsError>;

    /// Score a single candidate interaction
    fn score_interaction(
        &self,
        enhancer: &ChromatinRegion,
        promoter: &GenomicRegion,
        features: &InteractionFeatures,
    ) -> Result<f64, EpigenomicsError>;
}

struct EnhancerPredictorConfig {
    max_distance: u64,          // Maximum linear distance to consider (e.g., 2 Mb)
    min_contact_score: f64,     // Minimum Hi-C interaction to seed candidates
    attention_heads: usize,     // Number of attention heads for interaction scoring
    attention_dim: usize,       // Attention dimension
    use_geometric: bool,        // Use GeometricAttention in hyperbolic space
    curvature: f32,             // Curvature for geometric attention (negative)
}
```

### ChromatinStateSearchService

Hierarchical chromatin state search using `ruvector-hyperbolic-hnsw`.

```rust
trait ChromatinStateSearchService {
    /// Index chromatin state vectors in hyperbolic space
    ///
    /// Chromatin states have natural hierarchy (e.g., active > promoter > TSS-proximal).
    /// Hyperbolic HNSW preserves this hierarchy during nearest-neighbor search.
    fn index_states(
        &self,
        states: &[(GenomicRegion, ChromatinStateVector)],
        config: &HyperbolicIndexConfig,
    ) -> Result<ChromatinStateIndex, EpigenomicsError>;

    /// Find genomic regions with similar chromatin state
    fn search_similar(
        &self,
        query: &ChromatinStateVector,
        k: usize,
    ) -> Result<Vec<(GenomicRegion, f64)>, EpigenomicsError>;
}

struct HyperbolicIndexConfig {
    curvature: f64,
    use_tangent_pruning: bool,
    prune_factor: usize,
    shard_by_chromosome: bool,
}
```

### Genome3DGraphService

3D genome graph analysis using `ruvector-gnn`.

```rust
trait Genome3DGraphService {
    /// Build a GNN over the 3D genome contact graph
    ///
    /// Nodes = genomic bins with epigenomic features
    /// Edges = Hi-C contacts weighted by interaction frequency
    /// The GNN propagates regulatory signals through 3D proximity.
    fn build_graph(
        &self,
        topology: &GenomeTopology,
        features: &BinFeatureMatrix,
    ) -> Result<GenomeGraph, EpigenomicsError>;

    /// Predict functional annotations using GNN message passing
    fn predict_function(
        &self,
        graph: &GenomeGraph,
        query_bin: usize,
        depth: usize,
    ) -> Result<FunctionalPrediction, EpigenomicsError>;
}
```

---

## Repositories

### MethylationProfileRepository

```rust
trait MethylationProfileRepository {
    async fn store(&self, profile: MethylationProfile) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: ProfileId) -> Option<MethylationProfile>;
    async fn find_by_sample_and_region(
        &self, sample: SampleId, region: &GenomicRegion
    ) -> Option<MethylationProfile>;
    async fn query_sites_in_region(
        &self, profile_id: ProfileId, region: &GenomicRegion
    ) -> Vec<MethylationSite>;
}
```

### GenomeTopologyRepository

```rust
trait GenomeTopologyRepository {
    async fn store(&self, topology: GenomeTopology) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: TopologyId) -> Option<GenomeTopology>;
    async fn find_tads_overlapping(
        &self, topology_id: TopologyId, region: &GenomicRegion
    ) -> Vec<TAD>;
    async fn find_interactions_for_gene(
        &self, topology_id: TopologyId, gene: &str
    ) -> Vec<EnhancerPromoterInteraction>;
}
```

### ChromatinLandscapeRepository

```rust
trait ChromatinLandscapeRepository {
    async fn store(&self, landscape: ChromatinLandscape) -> Result<(), StoreError>;
    async fn find_by_id(&self, id: LandscapeId) -> Option<ChromatinLandscape>;
    async fn find_peaks_in_region(
        &self, landscape_id: LandscapeId, region: &GenomicRegion
    ) -> Vec<ChromatinRegion>;
}
```

---

## Factories

### GenomeTopologyFactory

```rust
impl GenomeTopologyFactory {
    /// Build topology from Hi-C contact matrix
    fn from_hic(
        contact_matrix: &SparseMatrix<f64>,
        chromosome: &str,
        resolution: u32,
        tad_detector: &dyn TADDetector,
        enhancer_predictor: &dyn EnhancerPredictor,
    ) -> Result<GenomeTopology, EpigenomicsError> {
        // 1. Build ContactGraph from sparse matrix
        // 2. Run TAD detection via ruvector-mincut
        // 3. Assign compartments via eigenvector decomposition
        // 4. Predict enhancer-promoter interactions via ruvector-attention
        // 5. Assemble GenomeTopology aggregate
    }
}
```

---

## Integration with RuVector Crates

### ruvector-mincut: TAD Boundary Detection

The contact map is converted to a `DynamicGraph` where genomic bins are vertices and ICE-normalized contact frequencies are edge weights. TAD boundaries correspond to minimum cuts in this graph.

```rust
fn contact_graph_to_dynamic_graph(contact: &ContactGraph) -> DynamicGraph {
    let graph = DynamicGraph::new();
    for (bin_i, bin_j, weight) in contact.iter_contacts() {
        graph.insert_edge(bin_i as u64, bin_j as u64, weight).ok();
    }
    graph
}

fn detect_tad_boundaries(
    contact: &ContactGraph,
    config: &TadDetectorConfig,
) -> Vec<TAD> {
    let graph = contact_graph_to_dynamic_graph(contact);
    let mut mincut = MinCutBuilder::new()
        .exact()
        .with_graph(graph)
        .build()
        .unwrap();

    // Sliding window: find local min-cuts that define TAD boundaries
    // Uses JTreeHierarchy for hierarchical TAD nesting
}
```

### ruvector-attention: Enhancer-Promoter Prediction

Enhancer and promoter loci become nodes in a graph. Edge features encode linear distance, Hi-C contact strength, and shared chromatin marks. `GraphAttention::compute_with_edges` learns which enhancers regulate which promoters.

### ruvector-gnn: 3D Genome Functional Prediction

Each genomic bin is a node with a feature vector (methylation, accessibility, histone signals). Hi-C contacts form the edges. GNN message passing (`RuvectorLayer`) propagates regulatory signals through 3D proximity to predict gene activity, replication timing, and mutation impact.

### ruvector-hyperbolic-hnsw: Chromatin State Search

Chromatin states (combinations of marks, accessibility, methylation) form a natural hierarchy: active/inactive at the top, with finer states (bivalent, poised enhancer, strong enhancer, etc.) below. `HyperbolicHnsw` indexes these states in Poincare ball space for efficient hierarchical nearest-neighbor search across the genome.

---

## Anti-Corruption Layers

### Alignment ACL

```rust
impl AlignmentAcl {
    fn translate_bisulfite_read(&self, read: &BamRecord) -> Result<Vec<MethylationCall>, AclError> {
        // Extract M/C base modifications from BAM MM/ML tags (SAMv1.6)
        // Convert phred-scaled quality to confidence
        // Map to genomic coordinates using CIGAR alignment
    }
}
```

### HiC ACL

```rust
impl HiCAntiCorruptionLayer {
    fn translate_contact_matrix(
        &self, hic_file: &HicFile, chromosome: &str, resolution: u32
    ) -> Result<ContactGraph, AclError> {
        // Read .hic or .cool format
        // Apply ICE normalization
        // Build sparse symmetric contact matrix
        // Convert to ContactGraph value object
    }
}
```

---

## Context Boundaries Summary

| Boundary | Upstream | Downstream | Integration Pattern |
|----------|----------|------------|---------------------|
| Alignment -> Epigenomics | Alignment Pipeline | Epigenomics Context | ACL (MethylationCall) |
| HiC -> Epigenomics | Hi-C Pipeline | Epigenomics Context | ACL (ContactGraph) |
| Epigenomics -> CRISPR | Epigenomics Context | CRISPR Context | Published Language (ChromatinState, MethylationProfile) |
| Epigenomics -> Variant | Epigenomics Context | Variant Calling Context | Domain Events (DMRIdentified) |

---

## References

- DDD-001: Coherence Gate Domain Model
- DDD-002: Syndrome Processing Domain Model
- Evans, Eric. "Domain-Driven Design." Addison-Wesley, 2003.
- Vernon, Vaughn. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
- Rao, S. et al. "A 3D Map of the Human Genome at Kilobase Resolution." Cell, 2014.
- Dixon, J. et al. "Topological Domains in Mammalian Genomes." Nature, 2012.
- Buenrostro, J. et al. "ATAC-seq: A Method for Assaying Chromatin Accessibility Genome-Wide." Current Protocols, 2015.
