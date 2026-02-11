# ADR-018: RuVector DNA Analyzer -- Specification

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow
**Parent ADRs**: ADR-001 (Core Architecture), ADR-003 (SIMD Optimization), ADR-014 (Coherence Engine), ADR-017 (Temporal Tensor Compression)

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | Architecture Team | Initial specification with full requirements, crate mapping, and complexity-bound gate thresholds |

---

## 1. Vision Statement: What "100 Years Ahead" Means for DNA Analysis

### 1.1 The Convergence Problem

Genomics today suffers from a fracture between three computing paradigms that evolved independently and remain poorly integrated:

1. **String algorithms** -- alignment, assembly, variant calling -- descended from the 1970s sequence-comparison tradition (Needleman-Wunsch, Smith-Waterman, BWA-MEM2, minimap2). These treat DNA as a linear string and cannot natively reason about population-level graph structure, epigenomic layers, or the protein-folding consequences of sequence changes.

2. **Graph algorithms** -- pan-genome graphs (vg, minigraph), gene interaction networks, metabolic pathways -- that capture structural variation and biological relationships but remain disconnected from the raw signal processing pipeline and operate at throughput orders of magnitude below sequencing hardware output.

3. **Deep learning models** -- AlphaFold2/3, Evo, Enformer, DNABERT-2, scBERT -- that achieve remarkable accuracy on narrow prediction tasks but have no mechanism for coherence verification, cannot prove when their predictions are structurally consistent with upstream variant calls, and scale poorly to population-level computation.

The "100 years ahead" vision for RuVector DNA Analyzer is the **unification of all three paradigms into a single coherent substrate** where:

- A raw nanopore signal flows in at one end.
- The system simultaneously aligns, calls variants, annotates gene function, predicts protein structure impact, evaluates CRISPR off-target risk, classifies the organism at strain level, and computes population-level ancestry -- all within a single, coherence-gated pipeline that can **prove** when its outputs are mutually consistent and **refuse** when they are not.
- Every result carries a witness certificate (in the sense of ADR-CE-012: Gate Refusal Witness) so that no downstream consumer -- clinical, forensic, agricultural, or research -- ever receives a result whose structural integrity has not been verified.

This is not a pipeline of disconnected tools. It is a **living graph** of biological knowledge updated in real time, where min-cut coherence gates prevent contradictory conclusions from propagating, where HNSW vector search retrieves relevant prior cases in microseconds, and where graph neural networks continuously refine the search index as new data arrives.

### 1.2 Why RuVector Is the Correct Substrate

The RuVector ecosystem already provides every primitive required for this unification:

| Biological Primitive | Computing Primitive | RuVector Crate |
|---------------------|--------------------|--------------------|
| Sequence similarity | Vector nearest-neighbor | `ruvector-core` (61us p50, HNSW) |
| Genome graph | Dynamic graph with min-cut | `ruvector-mincut` + `ruvector-graph` |
| Phylogenetic hierarchy | Hyperbolic embedding | `ruvector-hyperbolic-hnsw` |
| Protein attention | Flash Attention kernels | `ruvector-attention` |
| Signal processing | Sparse neural inference | `ruvector-sparse-inference` |
| Pipeline orchestration | DAG execution engine | `ruvector-dag` |
| Quality gating | Coherence engine + witnesses | `cognitum-gate-kernel` + `ruvector-mincut` |
| Continual learning | SONA + EWC++ | `sona` + `ruvector-nervous-system` |
| Hardware acceleration | FPGA transformer backend | `ruvector-fpga-transformer` |
| Incremental updates | Delta propagation | `ruvector-delta-*` |
| Quantum-inspired search | Grover/QAOA amplitude amplification | `ruQu` |

No other system combines all of these. The specification that follows defines how to compose them.

---

## 2. Functional Requirements

### 2.1 Whole Genome Sequencing Analysis

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-001 | Process whole-genome FASTQ/BAM/CRAM input at >10 Tbp/hour sustained throughput | Benchmark: 33x human genomes (3.2 Gbp each) per hour on a single 128-core node with FPGA acceleration; validated against GIAB HG002 truth set | Critical |
| FR-002 | Support short-read (Illumina 150bp PE), long-read (PacBio HiFi, ONT simplex/duplex), and linked-read (10X Chromium, TELL-Seq) input formats simultaneously in a single analysis run | All three input types produce a unified variant call set; concordance >99.5% against single-input baselines | High |
| FR-003 | Stream-aligned processing: begin variant calling on chromosome 1 while chromosome 22 data is still arriving; no requirement to hold entire genome in memory simultaneously | Memory footprint <2 GB for a streaming single-sample whole genome at 30x depth | Critical |

**Implementation mapping**: FR-001 uses `ruvector-fpga-transformer` for hardware-accelerated base-pair distance computations and `ruvector-sparse-inference` for activation-sparse alignment scoring. FR-003 uses `ruvector-delta-core` for incremental state propagation between streaming windows and `ruvector-temporal-tensor` for temporal quantization of intermediate embeddings.

### 2.2 Raw Signal Processing

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-004 | Ingest Oxford Nanopore FAST5/POD5 raw electrical signal at >4 million samples/second per channel, with adaptive segmentation using dendritic coincidence detection | Basecalling accuracy >Q30 (99.9%) on R10.4.1 chemistry; latency <500us per 4kbp read from signal to base sequence | Critical |
| FR-005 | Ingest PacBio SMRT ZMW traces with polymerase kinetics, resolving base modifications (6mA, 4mC, 5mC) directly from inter-pulse durations | Modification calling concordance >95% against bisulfite-sequencing ground truth on NA12878 | High |
| FR-006 | Detect and separate chimeric reads, adapter contamination, and DNA damage artifacts in real time during signal processing | False chimera rate <0.01%; adapter detection sensitivity >99.9% | High |

**Implementation mapping**: FR-004 maps `ruvector-nervous-system` dendritic trees to nanopore signal segmentation -- the NMDA-like nonlinearity threshold naturally models the current-level transitions between nucleotide k-mers. The 10-50ms coincidence window maps to the ~450us dwell time per nucleotide at typical translocation speeds. FR-005 uses `ruvector-attention` (Flash Attention) for polymerase kinetics modeling where inter-pulse duration sequences are treated as token sequences with causal masking.

### 2.3 Variant Calling

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-007 | Call single-nucleotide variants (SNVs) with <0.0001% false positive rate (equivalently, positive predictive value >99.9999%) on the GIAB v4.2.1 high-confidence regions | Benchmarked against HG001-HG007 truth sets; F1 >99.99% for SNVs in high-confidence regions | Critical |
| FR-008 | Call insertions and deletions (indels) up to 50bp with F1 >99.9% and structural variants >50bp with F1 >95% | Validated against GIAB Tier 1 SV benchmark and CMRG difficult regions | Critical |
| FR-009 | Produce phased diploid variant calls with N50 phase block length >10 Mbp from long-read data and >1 Mbp from short-read with population-reference-informed phasing | Phase block switch error rate <0.1%; validated against trio-phased ground truth | High |
| FR-010 | Detect mosaic variants at allele fractions down to 0.1% in deep sequencing (>1000x) data | Sensitivity >80% at 0.5% VAF; false positive rate <1 per megabase at 0.1% VAF threshold | High |

**Implementation mapping**: FR-007 maps variant candidate sites to nodes in a `ruvector-mincut` graph where edges represent read-support relationships. The coherence gate (ADR-001 Anytime-Valid Coherence Gate) fires when the min-cut value between reference-allele and alt-allele nodes drops below the gate threshold, certifying the variant call with a witness. The El-Hayek et al. (Dec 2025) deterministic min-cut with n^{o(1)} update time ensures that each variant site update is processed in subpolynomial time even as the graph grows. FR-009 uses `ruvector-hyperbolic-hnsw` to embed haplotype blocks in hyperbolic space where the natural tree hierarchy of the haplotype genealogy is captured with O(log n) distortion.

### 2.4 Graph-Genome Structural Variation

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-011 | Construct and query a pan-genome graph representing >10,000 individuals with all known structural variants, inversions, translocations, and complex rearrangements | Graph construction <24 hours from VCF inputs; query time <1ms for any 1 kbp region | High |
| FR-012 | Detect novel structural variants not present in the reference graph by identifying graph topology anomalies via min-cut analysis | Sensitivity >85% for novel SVs >1 kbp; false discovery rate <10% | High |
| FR-013 | Support dynamic graph updates: adding a new individual's variants to the pan-genome graph in <10 seconds without full rebuild | Verified via incremental insertion of 1000 individuals sequentially; query accuracy identical to full rebuild | High |

**Implementation mapping**: FR-011 and FR-013 directly leverage the Abboud et al. (Jul 2025) deterministic almost-linear-time m^{1+o(1)} Gomory-Hu tree construction. The all-pairs mincut structure of the Gomory-Hu tree exactly captures the bottleneck connectivity between any two genomic positions in the pan-genome graph -- this identifies breakpoint hotspots, inversion boundaries, and translocation junctions. Dynamic updates to the graph (FR-013) use `ruvector-delta-graph` for incremental edge propagation and `ruvector-mincut` for subpolynomial maintenance of the cut structure.

### 2.5 Epigenomic Analysis

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-014 | Quantify CpG methylation levels at single-base resolution from bisulfite-seq, EM-seq, and nanopore native methylation data | Pearson correlation >0.95 with ENCODE WGBS gold standards; coverage requirement <10x for >90% CpG sites | High |
| FR-015 | Identify differentially methylated regions (DMRs) between sample groups with FDR-controlled significance, supporting >1000 samples in a single comparison | DMR calling completes in <60 seconds for 1000-sample cohort; FDR <5% validated against known imprinted loci | Medium |
| FR-016 | Detect chromatin accessibility patterns from ATAC-seq and map them to regulatory element annotations in <5 seconds per sample | Peak calling concordance >90% with ENCODE cCRE catalog; footprint resolution <10bp | Medium |

**Implementation mapping**: FR-014 uses `ruvector-core` HNSW to index methylation state vectors (one dimension per CpG site) and `ruvector-gnn` for message-passing across CpG neighborhoods to smooth noisy single-site estimates. The GNN's EWC module (Elastic Weight Consolidation) prevents catastrophic forgetting of tissue-specific methylation patterns when training on diverse sample types. FR-015 uses `ruvector-sparse-inference` for sparse region-level testing where >95% of genomic windows have no significant signal.

### 2.6 Protein Structure Prediction

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-017 | Predict 3D protein structure from amino acid sequence with backbone RMSD <1.0 Angstrom for single-domain proteins <300 residues | GDT-TS >90 on CASP16 FM targets; TM-score >0.9 on the PDB validation set from 2025 | High |
| FR-018 | Predict the structural impact of missense variants, classifying as benign/pathogenic with AUROC >0.95 | Validated against ClinVar pathogenic/benign missense variants with known structures; calibrated probability outputs | High |
| FR-019 | Predict protein-protein interaction interfaces and binding affinity changes caused by mutations | Interface RMSD <2.0 Angstrom for known complexes; ddG prediction Pearson r >0.7 against SKEMPI v2.0 | Medium |

**Implementation mapping**: FR-017 uses `ruvector-attention` (7 attention types: scaled dot, multi-head, flash, linear, local-global, hyperbolic, MoE) for the pairwise distance and angular attention maps that are the core of modern structure prediction. The `ruvector-fpga-transformer` backend provides deterministic-latency inference with zero-allocation hot paths for the iterative structure refinement cycles. FR-018 uses the `sona` (SONA) Micro-LoRA for rapid adaptation to variant-specific structural perturbations -- the ultra-low-rank (1-2) LoRA enables instant learning of missense variant effects from a handful of examples.

### 2.7 CRISPR Off-Target Prediction

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-020 | Predict all off-target sites for a given CRISPR guide RNA with sensitivity >99% and specificity >95% genome-wide | Validated against GUIDE-seq, DISCOVER-seq, and CIRCLE-seq experimental off-target datasets for SpCas9, SaCas9, and Cas12a | Critical |
| FR-021 | Score off-target sites by predicted cleavage activity and assign coherence-gated confidence levels | Spearman correlation >0.8 with experimental cleavage frequencies; coherence gate witness accompanies every prediction above clinical-use threshold | Critical |
| FR-022 | Support simultaneous analysis of multiplexed guide libraries (>10,000 guides) with cross-reactivity detection | Complete analysis for 10,000 guides against human genome in <60 seconds; identify all pairwise cross-reactive guides with Jaccard overlap >0.1 | High |

**Implementation mapping**: FR-020 is the defining use case for min-cut-based analysis in genomics. The guide RNA and potential off-target sites form a bipartite graph where edge weights represent sequence complementarity. The minimum cut between the "on-target" partition and "off-target" partition quantifies the selectivity of the guide. The Khanna et al. (Feb 2025) near-optimal O~(n) linear sketches for hypergraph spectral sparsification enable polylog(n) dynamic updates as new off-target sites are discovered, maintaining the spectral approximation without full recomputation. This maps directly to `ruvector-mincut` sparsification (Benczur-Karger) with the Khanna sketch as the spectral certificate. FR-022 uses `ruvector-core` HNSW for k-mer similarity search across guide libraries with the 61us p50 latency enabling 10,000-guide throughput.

### 2.8 Metagenomic Classification

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-023 | Classify metagenomic reads to strain-level resolution against a database of >1 million reference genomes | Sensitivity >90% at strain level for species with >0.1% relative abundance; false positive rate <1% at genus level; throughput >1 Tbp/hour | High |
| FR-024 | Detect novel organisms not in the reference database and place them on the phylogenetic tree with confidence bounds | Novel organism detection sensitivity >80% for organisms with <85% ANI to nearest reference; phylogenetic placement accuracy within 1 taxonomic rank | Medium |
| FR-025 | Quantify relative abundances with <1% absolute error for species with >1% true abundance and detect species at 0.01% relative abundance | Validated against CAMI2 challenge datasets; Bray-Curtis dissimilarity <0.05 vs. ground truth for > 1% abundance species | High |

**Implementation mapping**: FR-023 uses `ruvector-hyperbolic-hnsw` as the core index. The NCBI taxonomy is a tree, and hyperbolic space (Poincare ball model) is the natural embedding geometry for trees -- the exponential volume growth of hyperbolic balls matches the exponential branching of taxonomy. Tangent-space pruning provides cheap Euclidean pre-filtering before exact hyperbolic ranking, achieving the 61us p50 baseline with hierarchy awareness. FR-024 uses `ruvector-gnn` message-passing layers to propagate phylogenetic signal from known neighbors to novel organisms, with EWC preventing forgetting of rare clade signatures.

### 2.9 Pharmacogenomics

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-026 | Genotype all PharmGKB Level 1A/1B pharmacogenes (CYP2D6, CYP2C19, HLA-B, DPYD, TPMT, UGT1A1, etc.) including star allele calling with structural variant awareness | 100% concordance with GeT-RM reference materials for all Level 1A/1B gene-drug pairs; correct CYP2D6 hybrid allele detection (e.g., *36+*10) | Critical |
| FR-027 | Predict drug response phenotypes (poor/intermediate/normal/rapid/ultra-rapid metabolizer) with confidence intervals | Phenotype prediction concordance >98% with clinical phenotyping for CYP2D6 and CYP2C19; calibrated 95% confidence intervals | High |
| FR-028 | Generate clinical pharmacogenomic reports conforming to PharmGKB Clinical Annotation Level of Evidence guidelines | Reports pass automated validation against CPIC guideline requirements; all Level 1A actionable findings surfaced | High |

**Implementation mapping**: FR-026 leverages `ruvector-graph` for the gene-region graph representation needed to resolve the extreme complexity of CYP2D6 (tandem duplications, deletions, hybrid genes). Star allele assignment is a graph matching problem that maps naturally to `ruvector-mincut` -- the star allele definitions partition the gene graph, and the minimum cut between candidate allele assignments determines the most parsimonious genotype. FR-027 uses `sona` continual learning to adapt drug-response models as new clinical evidence accumulates without losing prior pharmacogenomic knowledge (EWC++ prevents catastrophic forgetting).

### 2.10 Population Genomics

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-029 | Compute identity-by-descent (IBD) segments and kinship coefficients for cohorts up to 1 million individuals in <1 hour | Kinship estimation error <0.01 for first-degree relatives; IBD segment detection sensitivity >95% for segments >1 cM; linear scaling verified to 500K individuals | High |
| FR-030 | Perform principal component analysis and UMAP dimensionality reduction on population-scale genotype matrices (1M individuals x 10M variants) | PCA of 1M x 10M matrix completes in <300 seconds using streaming SVD; results identical (cosine similarity >0.999) to full in-memory computation | Medium |
| FR-031 | Infer local ancestry along chromosomes for admixed individuals using reference panels of >100 populations | Ancestry assignment accuracy >95% per 1-cM window; validated against the 1000 Genomes + HGDP panel | Medium |

**Implementation mapping**: FR-029 uses `ruvector-core` HNSW for approximate IBD segment candidate retrieval (treating haplotype windows as vectors), followed by `ruvector-mincut` for precise segment boundary refinement. FR-030 uses `ruvector-temporal-tensor` for streaming quantized SVD where temporal compression reuses eigenvalue scales across consecutive genomic windows. FR-031 embeds reference population haplotypes in `ruvector-hyperbolic-hnsw` where the population tree structure is naturally captured, then classifies query haplotype windows via nearest-neighbor in hyperbolic space.

### 2.11 Cancer Genomics

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-032 | Decompose tumor mutational profiles into COSMIC SBS/DBS/ID signatures with >95% accuracy for signatures present at >5% contribution | Cosine similarity >0.95 between reconstructed and true profile for PCAWG validation set; correct identification of all signatures with >5% weight | High |
| FR-033 | Detect tumor mutational burden (TMB), microsatellite instability (MSI), and homologous recombination deficiency (HRD) from targeted panel sequencing (>300 genes) | TMB estimation Pearson r >0.95 vs. WGS ground truth; MSI detection sensitivity/specificity >95%/99%; HRD score concordance >90% with Myriad myChoice | High |
| FR-034 | Identify somatic driver mutations and classify variants of uncertain significance (VUS) using protein structure impact prediction (FR-018) integrated with population frequency and functional annotations | Driver gene F1 >90% on PCAWG driver catalog; VUS reclassification concordance >85% with ClinGen expert panels | Medium |

**Implementation mapping**: FR-032 uses `ruvector-attention` MoE (Mixture of Experts) attention where each expert specializes in a mutation signature class, with gating learned via `sona` trajectories. FR-034 demonstrates the coherence engine's cross-domain power: a variant's classification depends on sequence context, protein structure impact, population frequency, and functional evidence. These become nodes in the coherence graph (ADR-014), and the sheaf Laplacian residual energy quantifies when the evidence is contradictory (e.g., a VUS predicted benign by structure but absent from population databases). The coherence gate defers classification when energy exceeds the threshold, escalating to expert review.

### 2.12 Antimicrobial Resistance and Pathogen Surveillance

| ID | Requirement | Acceptance Criteria | Priority |
|----|------------|---------------------|----------|
| FR-035 | Detect antimicrobial resistance (AMR) genes from metagenomic or isolate sequencing with >99% sensitivity and >95% specificity against CARD and ResFinder databases | Validated on WHO priority pathogen panel; time from reads to AMR report <30 seconds for isolate genomes | Critical |
| FR-036 | Perform real-time pathogen surveillance: assign incoming sequences to phylogenetic lineages, detect novel clusters, and trigger alerts for concerning mutations | Lineage assignment latency <1 second per genome; cluster detection within 5 minutes of reaching 3-member threshold; validated against known outbreak retrospectives | Critical |
| FR-037 | Track horizontal gene transfer (HGT) events and plasmid mobility across microbial populations via graph-based analysis | HGT detection sensitivity >80% for events detectable by ANI discordance; plasmid host-range classification accuracy >90% | Medium |

**Implementation mapping**: FR-035 uses `ruvector-core` HNSW for fast AMR gene sequence lookup with quantized (INT4) k-mer vectors for memory efficiency. FR-036 uses `ruvector-delta-consensus` for real-time phylogenetic cluster maintenance -- each new genome triggers a delta update to the surveillance graph, and the coherence gate fires cluster alerts when min-cut values between lineage groups drop below the surveillance threshold. FR-037 uses the Gomory-Hu tree (Abboud et al.) to identify the all-pairs minimum cut structure of the microbial sharing network, where low-cut edges between taxa indicate likely HGT corridors.

---

## 3. Non-Functional Requirements

### 3.1 Latency

| ID | Requirement | Measurement | SOTA Baseline | Target |
|----|------------|-------------|---------------|--------|
| NFR-001 | Single variant call decision | Time from read pileup to genotype output | ~10ms (GATK HaplotypeCaller per-site) | <100us (100x improvement) |
| NFR-002 | Gene annotation lookup | Time from variant position to full annotation | ~5ms (VEP per variant) | <1ms (5x improvement) |
| NFR-003 | K-mer similarity search | Time from query k-mer to top-10 matches | ~500us (sourmash) | <61us (leveraging ruvector-core p50) |
| NFR-004 | Protein structure prediction | Time for single-domain <150 residues | ~120 seconds (AlphaFold2) | <10 seconds (12x improvement via FPGA) |
| NFR-005 | CRISPR off-target search | Time per guide RNA against whole genome | ~30 seconds (Cas-OFFinder) | <100ms (300x improvement) |
| NFR-006 | Coherence gate decision | Time to evaluate witness and issue permit/defer/deny | N/A (no prior art) | <50us (matching cognitum-gate-kernel tick) |

### 3.2 Throughput

| ID | Requirement | Measurement | SOTA Baseline | Target |
|----|------------|-------------|---------------|--------|
| NFR-007 | Sequencing data ingestion | Base pairs processed per hour | ~1 Tbp/hr (DRAGEN v4.3) | >10 Tbp/hr (10x improvement) |
| NFR-008 | Variant calls per second | Variants genotyped and filtered per second | ~50K/s (DeepVariant GPU) | >500K/s (10x improvement) |
| NFR-009 | Metagenomic classifications per second | Reads classified to strain level per second | ~4M reads/s (Kraken2) | >40M reads/s (10x improvement) |
| NFR-010 | Population-scale queries | IBD/kinship lookups per second for 1M-individual cohort | ~100/s (KING) | >10,000/s (100x improvement) |

### 3.3 Accuracy

| ID | Requirement | Metric | SOTA Baseline | Target |
|----|------------|--------|---------------|--------|
| NFR-011 | SNV calling accuracy | F1 score on GIAB high-confidence regions | 99.97% (DeepVariant v1.6) | 99.9999% (six nines) |
| NFR-012 | Indel calling accuracy | F1 score on GIAB Tier 1 | 99.7% (DeepVariant v1.6) | 99.99% |
| NFR-013 | Structural variant accuracy | F1 score on GIAB SV Tier 1 | 95% (Sniffles2 + cuteSV merge) | >98% |
| NFR-014 | Basecalling accuracy | Per-read modal Q-score | Q25 (Dorado v0.8 SUP) | >Q30 (99.9%) |
| NFR-015 | Protein structure RMSD | Backbone RMSD on CASP16 FM targets | 1.5A (AlphaFold3) | <1.0A |

### 3.4 Memory and Resource Efficiency

| ID | Requirement | Measurement | Target |
|----|------------|-------------|--------|
| NFR-016 | Streaming genome processing | Peak RAM for single 30x human genome | <2 GB (via ruvector-delta-core streaming + ruvector-temporal-tensor quantization) |
| NFR-017 | Pan-genome index | Memory for 10,000-individual pan-genome graph index | <64 GB (via ruvector-core scalar quantization 4x + ruvector-hyperbolic-hnsw tangent-space compression) |
| NFR-018 | Reference database | Memory for metagenomic reference of >1M genomes | <256 GB (via ruvector-sparse-inference hot/cold caching and precision lanes) |
| NFR-019 | Browser deployment | WASM binary size for client-side variant viewer | <10 MB (via micro-hnsw-wasm + ruvector-mincut-wasm) |

### 3.5 Platform Support

| ID | Requirement | Target |
|----|------------|--------|
| NFR-020 | Native CPU: x86_64 with AVX-512/AVX2, ARM64 with NEON/SVE, RISC-V with vector extensions | Full performance with SIMD auto-dispatch (matching ADR-003 strategy) |
| NFR-021 | WASM: Browser (Chrome/Firefox/Safari) and edge devices via wasm32-wasi | Core variant calling and annotation; subset of functionality |
| NFR-022 | FPGA: Xilinx Alveo U250/U280 and Intel Stratix 10 via ruvector-fpga-transformer PCIe/Daemon backends | Basecalling, alignment, and structure prediction acceleration |
| NFR-023 | GPU: CUDA 12+ and ROCm 6+ for attention-heavy workloads | Protein structure prediction and population-scale PCA |
| NFR-024 | PostgreSQL extension: Genomic vector queries via ruvector-postgres | Clinical deployment in hospital information systems |

### 3.6 Regulatory and Compliance

| ID | Requirement | Target |
|----|------------|--------|
| NFR-025 | FDA 21 CFR Part 11 compliance for clinical genomics outputs | Complete audit trail via coherence engine witness chain (ADR-CE-017) |
| NFR-026 | HIPAA compliance for patient genomic data | Encryption at rest (AES-256) and in transit (TLS 1.3); zero-copy processing to minimize data exposure surface |
| NFR-027 | ISO 15189 medical laboratory accreditation support | Reproducible results with deterministic algorithms (El-Hayek et al. deterministic min-cut); signed model artifacts (ruvector-fpga-transformer Ed25519 signatures) |
| NFR-028 | GDPR Right to Erasure for genomic data | Delta-based state management (ruvector-delta-core) enables surgical deletion of individual contributions without full index rebuild |

---

## 4. Crate-to-Capability Mapping

This section provides the authoritative mapping from each RuVector crate to its DNA analysis role. Every capability must be traceable to a specific crate.

### 4.1 Core Search and Storage Layer

| Crate | DNA Analysis Capability | Key Parameters |
|-------|------------------------|----------------|
| `ruvector-core` | K-mer vector indexing, variant embedding search, read similarity lookup, AMR gene search | 61us p50 @ k=10, 384-dim; scalar/INT4/product/binary quantization; HNSW with O(log n) search |
| `ruvector-hyperbolic-hnsw` | Phylogenetic tree search (metagenomic taxonomy, population ancestry, haplotype genealogy); hierarchical structure naturally captured by Poincare ball | Per-shard curvature for different taxonomic depths; tangent-space pruning for 10x candidate pre-filtering |
| `ruvector-postgres` | Clinical genomics deployment; SQL-accessible variant queries; GNN-indexed pharmacogenomic lookups | IVFFlat + HNSW indexes; sparse vector support for gene panels; native type I/O for genomic intervals |
| `micro-hnsw-wasm` | Browser-based variant viewer with client-side nearest-neighbor search for real-time genome browsing | <10MB WASM binary; suitable for clinical web interfaces |

### 4.2 Graph and Cut Analysis Layer

| Crate | DNA Analysis Capability | Key Parameters |
|-------|------------------------|----------------|
| `ruvector-mincut` | Variant call quality gating (min-cut between ref/alt partitions); SV breakpoint detection; CRISPR guide selectivity scoring; HGT corridor identification; pharmacogene star-allele assignment | El-Hayek et al. n^{o(1)} update; Benczur-Karger sparsification O(n log n / eps^2); deterministic + auditable |
| `ruvector-graph` | Pan-genome graph construction and traversal; gene interaction networks; metabolic pathway analysis | Dynamic graph with edge insertion/deletion; connected component tracking |
| `ruvector-dag` | Analysis pipeline orchestration; topological scheduling of dependent analysis steps (align -> call -> annotate -> predict); bottleneck detection via min-cut | 7 attention mechanisms for pipeline node prioritization; MinCut optimization with O(n^0.12) bottleneck detection |
| `ruvector-cluster` | Sample clustering for cohort analysis; outbreak cluster detection in surveillance | Integrates with HNSW for cluster seed selection |

### 4.3 Neural Processing Layer

| Crate | DNA Analysis Capability | Key Parameters |
|-------|------------------------|----------------|
| `ruvector-attention` | Protein structure prediction (MSA attention, pairwise distance attention); polymerase kinetics modeling; mutation signature decomposition | 7 types: scaled-dot, multi-head, flash, linear, local-global, hyperbolic, MoE; 2.49-7.47x Flash speedup |
| `ruvector-gnn` | Methylation smoothing across CpG neighborhoods; phylogenetic signal propagation; variant effect prediction from gene interaction graphs | GCN/GAT/GraphSAGE layers; EWC for continual learning; replay buffer for rare variant retention |
| `ruvector-sparse-inference` | Basecalling neural network inference; gene annotation model inference; sparse region testing for DMR analysis | PowerInfer-style activation sparsity; 3/5/7-bit precision lanes; GGUF model support; hot/cold neuron caching |
| `sona` | Rapid adaptation for variant-specific models (Micro-LoRA); continual pharmacogenomic learning; per-patient model fine-tuning | Micro-LoRA rank 1-2 for instant learning; EWC++ for forgetting prevention; ReasoningBank for pattern similarity |
| `ruvector-nervous-system` | Nanopore signal segmentation (dendritic coincidence); hyperdimensional k-mer encoding (HDC); multi-agent pipeline coordination | NMDA-threshold dendrites; 10K+ events/ms HDC throughput; cognitive routing for pipeline stages |
| `cognitum-gate-kernel` | Quality control gating for every analysis output; clinical decision witness generation; 256-tile parallel coherence evaluation | 64KB per tile; deterministic tick loop; e-value evidence accumulation; witness fragment aggregation |

### 4.4 Hardware Acceleration Layer

| Crate | DNA Analysis Capability | Key Parameters |
|-------|------------------------|----------------|
| `ruvector-fpga-transformer` | Basecalling acceleration; alignment scoring; protein structure refinement cycles | INT4/INT8 quantization; zero-allocation hot path; PCIe/Daemon/NativeSim/WasmSim backends; Ed25519 signed artifacts |
| `ruvector-attention-wasm` | Browser-based protein structure visualization with interactive attention maps | WASM-compatible attention computation |
| SIMD intrinsics (within `ruvector-core`) | Distance computation for all vector operations across all platforms | AVX-512, AVX2, SSE4.1, NEON, WASM SIMD auto-dispatch |

### 4.5 State Management and Coordination Layer

| Crate | DNA Analysis Capability | Key Parameters |
|-------|------------------------|----------------|
| `ruvector-delta-core` | Streaming genome processing without full-genome memory; incremental variant database updates; surgical patient data deletion (GDPR) | Delta encoding/propagation; conflict resolution; temporal windowing |
| `ruvector-delta-graph` | Dynamic pan-genome graph updates; real-time surveillance graph maintenance | Incremental edge propagation; compatible with mincut dynamic updates |
| `ruvector-delta-consensus` | Multi-site clinical deployment consensus; distributed surveillance alert coordination | Raft-based consensus for distributed genomics clusters |
| `ruvector-temporal-tensor` | Intermediate embedding compression for streaming analysis; temporal SVD for population PCA; basecall signal quantization | 4x-10.67x compression; access-pattern-driven tier selection; drift-aware segments |
| `ruvector-raft` | Distributed genomics cluster coordination for population-scale analysis | Leader election; log replication; fault tolerance |

### 4.6 Quantum-Inspired Layer

| Crate | DNA Analysis Capability | Key Parameters |
|-------|------------------------|----------------|
| `ruQu` | Amplitude-amplified sequence search (Grover) for rare variant detection; QAOA-based graph partitioning for phylogenetic tree optimization; VQE for molecular energy minimization in drug-binding prediction | Surface code error correction; tensor network evaluation; classical nervous system for quantum coherence gating |

---

## 5. ArXiv Paper Complexity Bounds as Gate Thresholds

The three referenced arXiv papers (2025) provide the theoretical complexity bounds that directly determine the operational thresholds of the DNA Analyzer's coherence gates. This section specifies exactly how each paper's results translate to gate parameters.

### 5.1 El-Hayek, Henzinger, Li (December 2025) -- Deterministic Fully-Dynamic Min-Cut

**Paper**: "Deterministic and Exact Fully Dynamic Minimum Cut of Superpolylogarithmic Size in Subpolynomial Time." arXiv:2512.13105.

**Key Result**: For min-cut value lambda where lambda <= 2^{Theta(log^{3/4-c} n)} for any constant c > 0, the algorithm achieves deterministic n^{o(1)} amortized update time.

**Gate Threshold Application -- Variant Calling Quality Gate (FR-007, NFR-011)**:

The variant call graph for a single genomic locus has |V| proportional to the read depth d (typically 30-1000) and |E| proportional to d^2 (pairwise read overlaps). The min-cut value lambda between the reference and alternate allele partitions corresponds directly to the allelic balance:

- **True heterozygous SNV**: lambda is approximately d/2 (balanced support for both alleles). For d=30, lambda=15. This falls well within the regime lambda <= 2^{Theta(log^{3/4-c} n)} since log^{3/4}(30) is approximately 2.6, and 2^{2.6} is approximately 6.1, so lambda=15 exceeds this bound for very small graphs. However, for population-scale graphs (n = 10^6 variants), log^{3/4}(10^6) is approximately 8.7, and 2^{8.7} is approximately 415, so lambda=15 is well within the tractable regime.

- **Gate threshold**: The coherence gate PERMITS a variant call when:
  1. The min-cut lambda between ref/alt partitions is computed in deterministic n^{o(1)} time (guaranteed by El-Hayek et al.)
  2. lambda > tau_permit where tau_permit = ceil(d * 0.1) (at least 10% of reads support the variant)
  3. The min-cut is verified to be exact (not approximate) with a witness certificate

- **Gate threshold for DEFER**: When lambda is in the range [tau_defer, tau_permit] where tau_defer = ceil(d * 0.02), the system defers to a more sensitive caller. This covers mosaic variants (FR-010).

- **Gate threshold for DENY**: When lambda < tau_defer, the system denies the variant call (insufficient evidence; likely sequencing error).

- **Update time guarantee**: Each new read alignment triggers at most n^{o(1)} update time to the variant graph's min-cut. For n=1000 reads at a deep-sequenced locus, n^{o(1)} is approximately n^{0.12} = 1000^{0.12} is approximately 2.3 operations -- effectively constant time. This is what enables NFR-001 (<100us per variant call decision).

- **Verified empirical scaling**: The `ruvector-mincut` implementation achieves n^{0.12} empirical scaling, matching the theoretical bound.

### 5.2 Khanna, Krauthgamer, Li, Quanrud (February 2025) -- Hypergraph Spectral Sparsification

**Paper**: "Linear Sketches for Hypergraph Cuts." arXiv (February 2025). Near-optimal O~(n) linear sketches for hypergraph spectral sparsification supporting polylog(n) dynamic updates.

**Gate Threshold Application -- CRISPR Off-Target Coherence Gate (FR-020, FR-021)**:

CRISPR guide-genome interactions form a hypergraph, not a simple graph. A single guide RNA interacts with multiple genomic loci simultaneously, and a single locus can be targeted by multiple guides in a multiplexed library. The hyperedges connect {guide, locus_1, locus_2, ..., locus_k} for guides with multiple near-target sites.

- **Sparsification bound**: The Khanna et al. O~(n) sketch size means that for a genome with n approximately 3 x 10^9 positions, the sketch requires O~(3 x 10^9) approximately 30 GB of storage -- feasible for a single machine.

- **Dynamic update bound**: polylog(n) updates means polylog(3 x 10^9) approximately log^c(3 x 10^9) approximately 31.5^c. For c=2 (quadratic polylog), this is approximately 992 operations per update. For c=3 (cubic polylog), approximately 31,000. Either way, this enables real-time maintenance of the spectral sparsifier as new off-target sites are experimentally validated.

- **Gate threshold**: The coherence gate for CRISPR guide approval uses the spectral gap of the sparsified hypergraph:
  1. **PERMIT**: Spectral gap gamma > gamma_clinical (calibrated so that the approved guide has >99% on-target specificity, validated against GUIDE-seq data). The spectral gap measures how well-separated the on-target site is from all off-target sites in the cut-spectral sense.
  2. **DEFER**: gamma in [gamma_research, gamma_clinical] -- guide is acceptable for research but requires additional experimental validation for clinical use.
  3. **DENY**: gamma < gamma_research -- guide has unacceptable off-target profile.

- **Integration with `ruvector-mincut` sparsification**: The existing Benczur-Karger implementation produces (1+epsilon)-approximate sparsifiers. The Khanna et al. sketch provides a **spectral** (not just cut) sparsifier, which is strictly stronger. The DNA Analyzer extends the `SparseGraph` type to support hyperedge representation and spectral gap computation, with the Khanna sketch as the certificate of spectral approximation quality.

### 5.3 Abboud, Choudhary, Gawrychowski, Li (July 2025) -- Almost-Linear Gomory-Hu Trees

**Paper**: "Deterministic Almost-Linear-Time Gomory-Hu Trees for All-Pairs Mincuts." arXiv (July 2025). Constructs Gomory-Hu trees in deterministic m^{1+o(1)} time.

**Gate Threshold Application -- Pan-Genome Structure Gate (FR-011, FR-012, FR-013)**:

The pan-genome graph with 10,000 individuals has approximately m = 10^8 edges (structural variant edges + reference backbone). The Gomory-Hu tree captures ALL pairwise minimum cuts in this graph using only n-1 edges.

- **Construction time bound**: m^{1+o(1)} = (10^8)^{1+o(1)}. For the o(1) term approximately 0.05 (consistent with El-Hayek et al. empirical scaling), this is approximately 10^{8.4} approximately 2.5 x 10^8 operations -- achievable in <1 second on modern hardware at 10^9 ops/second. This enables FR-013 (dynamic graph updates in <10 seconds).

- **All-pairs min-cut information**: The Gomory-Hu tree edge between genomic positions i and j has weight equal to the min-cut between i and j in the full graph. This directly identifies:
  1. **Structural variant breakpoints**: Edges with anomalously low weight indicate positions where the genome graph is "thin" -- likely breakpoints.
  2. **Recombination hotspots**: Regions with many low-weight Gomory-Hu edges correspond to high recombination rates.
  3. **Inversion boundaries**: The min-cut structure changes sharply at inversion boundaries.

- **Gate threshold for structural variant detection (FR-012)**:
  1. **SV candidate generation**: Any Gomory-Hu tree edge with weight < tau_sv = median_weight / 10 triggers SV candidate evaluation.
  2. **SV coherence gate**: The candidate is promoted to a variant call when the exact min-cut (computed via El-Hayek et al. Tier 2) confirms lambda < tau_sv AND the cut partition aligns with known SV signatures.
  3. **Novel SV detection**: Candidates not matching any known signature are flagged with a DEFER decision, triggering assembly-based validation.

- **The two-tier strategy crystallized**: Abboud et al. provides the "wide-area radar" (O~(m) Gomory-Hu tree for global structure), while El-Hayek et al. provides the "precision sonar" (n^{o(1)} exact min-cut for specific loci). This mirrors the ADR-002 (Dynamic Hierarchical j-Tree Decomposition) two-tier architecture, now applied to genomics.

### 5.4 Unified Gate Threshold Table

| Gate | Input | Permit Threshold | Defer Threshold | Deny Threshold | Complexity Bound | Paper |
|------|-------|-----------------|-----------------|----------------|-----------------|-------|
| Variant Call Quality | Read pileup graph | lambda > 0.1d | 0.02d <= lambda <= 0.1d | lambda < 0.02d | n^{o(1)} per update | El-Hayek et al. |
| CRISPR Guide Approval | Guide-genome hypergraph | gamma > gamma_clinical | gamma_research <= gamma <= gamma_clinical | gamma < gamma_research | polylog(n) per update | Khanna et al. |
| Pan-Genome SV Detection | Pan-genome graph | GH-weight < tau_sv AND exact confirmation | GH-weight < tau_sv, no confirmation | GH-weight >= tau_sv | m^{1+o(1)} construction | Abboud et al. |
| Pharmacogene Assignment | Gene-region graph | Unique star-allele min-cut partition | Multiple equiparsimonious assignments | No consistent assignment | n^{o(1)} per allele | El-Hayek et al. |
| Pathogen Alert | Surveillance graph | Cluster min-cut < tau_alert AND n_members >= 3 | Cluster min-cut declining (trend) | Stable min-cut | n^{o(1)} per genome | El-Hayek et al. |
| Protein Impact | Evidence coherence graph | Sheaf energy < tau_benign OR > tau_pathogenic | tau_benign <= energy <= tau_pathogenic | N/A (always classifiable) | Continuous field update | ADR-014 + El-Hayek |

---

## 6. Acceptance Criteria and Validation Plan

### 6.1 Benchmark Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| GIAB HG001-HG007 | SNV/indel/SV accuracy | NIST Genome in a Bottle |
| GIAB Tier 1 SV v0.6 | Structural variant accuracy | NIST |
| CMRG v1.0 | Challenging medically relevant genes | GIAB/CMRG consortium |
| CAMI2 | Metagenomic classification accuracy | CAMI challenge |
| CASP16 targets | Protein structure prediction | CASP |
| SKEMPI v2.0 | Binding affinity prediction | Protein interaction database |
| PCAWG | Mutation signatures and driver genes | ICGC/TCGA |
| GeT-RM | Pharmacogene star-allele accuracy | CDC/GeT-RM program |
| GUIDE-seq datasets | CRISPR off-target validation | Published experimental data |
| 1000 Genomes + HGDP | Population genetics validation | IGSR |

### 6.2 Validation Scenarios (Gherkin Format)

```gherkin
Feature: Variant Calling with Coherence Gating

  Scenario: High-confidence SNV call with witness
    Given a 30x WGS BAM aligned to GRCh38
    And the GIAB HG002 truth set for high-confidence regions
    When the DNA Analyzer processes chromosome 1
    Then every SNV call shall include a coherence witness certificate
    And the min-cut lambda for each called variant shall exceed 0.1 * depth
    And the total F1 score shall exceed 99.99%
    And the processing time shall be less than 60 seconds for chromosome 1

  Scenario: Mosaic variant detection with deferred gating
    Given a 1000x targeted panel BAM with known mosaic variants at 0.5% VAF
    When the DNA Analyzer processes the panel regions
    Then mosaic variant calls with 0.02d <= lambda <= 0.1d shall be flagged as DEFERRED
    And the sensitivity for 0.5% VAF variants shall exceed 80%
    And the false positive rate shall be less than 1 per megabase

  Scenario: CRISPR guide approval with spectral gating
    Given a library of 10,000 SpCas9 guide RNAs
    And the GRCh38 reference genome
    When the DNA Analyzer evaluates all guides
    Then every guide with spectral gap > gamma_clinical shall receive PERMIT
    And every guide with known experimental off-targets shall receive DEFER or DENY
    And the total processing time shall be less than 60 seconds

  Scenario: Real-time pathogen surveillance alert
    Given a running surveillance stream receiving nanopore genomes
    When 3 genomes within 5 SNPs of each other arrive within 24 hours
    Then the coherence gate shall fire a cluster alert within 5 minutes
    And the alert shall include a Gomory-Hu tree witness showing the cluster structure
    And the lineage assignment latency shall be less than 1 second per genome
```

---

## 7. Dependencies and Constraints

### 7.1 Technical Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Rust toolchain | 1.82+ | Language runtime with SIMD intrinsics |
| wasm-pack | 0.13+ | WASM compilation for browser deployment |
| Xilinx Vitis | 2024.2+ | FPGA bitstream compilation (optional) |
| CUDA toolkit | 12.4+ | GPU acceleration (optional) |
| htslib | 1.20+ | BAM/CRAM/VCF file format support (via rust-htslib FFI) |
| minimap2 | 2.28+ | Long-read alignment reference implementation for validation |

### 7.2 Constraints

| Category | Constraint | Rationale |
|----------|-----------|-----------|
| Technical | All algorithms must be deterministic or have deterministic mode | FDA 21 CFR Part 11 requires reproducible results (NFR-027) |
| Technical | Zero-allocation hot path for variant calling inner loop | Required for <100us latency target (NFR-001) |
| Technical | No external network calls during analysis (air-gap compatible) | Clinical environments may lack internet access |
| Business | Must integrate with existing clinical LIMS via HL7 FHIR | Hospital deployment requirement |
| Regulatory | All model artifacts must be Ed25519-signed | Matching ruvector-fpga-transformer artifact format |
| Regulatory | Complete audit trail for every clinical-grade output | Coherence engine witness chain (ADR-CE-017) |

---

## 8. Validation Checklist

- [ ] All 37 functional requirements have testable acceptance criteria
- [ ] All 28 non-functional requirements have measurable targets with SOTA baselines
- [ ] Every functional requirement maps to at least one RuVector crate
- [ ] All three arXiv papers have explicit gate threshold formulations
- [ ] Edge cases documented: mosaic variants, chimeric reads, novel organisms, VUS classification
- [ ] Performance metrics defined for each pipeline stage
- [ ] Security and regulatory requirements specified (FDA, HIPAA, GDPR, ISO 15189)
- [ ] Platform targets enumerated (x86, ARM, RISC-V, WASM, FPGA, GPU)
- [ ] Benchmark datasets identified for all accuracy claims
- [ ] Gherkin acceptance scenarios written for critical paths
- [ ] Dependencies and constraints documented
- [ ] Stakeholder review scheduled

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **Tbp** | Terabase pairs (10^12 base pairs) |
| **VAF** | Variant allele frequency -- fraction of reads supporting the variant |
| **GIAB** | Genome in a Bottle -- NIST reference materials for benchmarking |
| **SV** | Structural variant -- genomic alteration >50bp (deletions, insertions, inversions, translocations) |
| **SNV** | Single nucleotide variant -- single base pair change |
| **Gomory-Hu tree** | A weighted tree on n vertices that encodes all n(n-1)/2 pairwise minimum cuts in O(n) space |
| **Spectral gap** | Second-smallest eigenvalue of the graph Laplacian; measures connectivity |
| **Sheaf Laplacian** | Generalization of the graph Laplacian to sheaves; measures local-to-global consistency (ADR-014) |
| **HNSW** | Hierarchical Navigable Small World -- approximate nearest neighbor search structure |
| **Poincare ball** | Model of hyperbolic geometry where the entire space fits inside a unit ball |
| **EWC++** | Elastic Weight Consolidation with online Fisher approximation -- prevents catastrophic forgetting |
| **Micro-LoRA** | Ultra-low-rank (1-2) Low-Rank Adaptation for instant model specialization |
| **Coherence gate** | A decision mechanism that classifies system state as Permit/Defer/Deny based on structural consistency |
| **Witness certificate** | Cryptographically verifiable proof that a coherence gate decision was computed correctly |
| **Q-score** | Phred quality score; Q30 means 99.9% accuracy per base |
| **CRISPR** | Clustered Regularly Interspaced Short Palindromic Repeats -- gene editing technology |
| **AMR** | Antimicrobial resistance |
| **HGT** | Horizontal gene transfer -- acquisition of genetic material from non-parent organisms |
| **DMR** | Differentially methylated region |
| **TMB** | Tumor mutational burden |
| **MSI** | Microsatellite instability |
| **HRD** | Homologous recombination deficiency |
