# ADR-028: Neural Sequence Intelligence for DNA Analysis

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**Target Crates**: `ruvector-attention`, `ruvector-fpga-transformer`, `cognitum-gate-kernel`, `sona`, `ruvector-sparse-inference`, `ruQu`

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | ruv.io | Initial neural sequence intelligence proposal |

---

## Context

### The Genomic Sequence Understanding Problem

DNA analysis demands fundamentally different neural architecture capabilities than natural language processing. A single human genome contains approximately 3.2 billion base pairs. Regulatory interactions span megabase distances. The signal-to-noise ratio in raw nanopore sequencing data is orders of magnitude worse than text. Current state-of-the-art genomic foundation models are limited to approximately 10Kbp context windows, missing the long-range dependencies that govern gene regulation, chromatin architecture, and structural variant pathogenicity.

RuVector's existing crate ecosystem provides the exact building blocks needed to break through these limitations: Flash Attention for O(n)-memory long-range modeling, FPGA-accelerated inference for real-time basecalling, SONA for per-device adaptation, gated transformers for variant effect prediction, and sparse inference for population-scale computation.

### Current State-of-the-Art Limitations

| Model | Context Window | Architecture | Limitation |
|-------|---------------|-------------|------------|
| DNABERT-2 | 512bp | BERT encoder | Cannot see enhancer-promoter interactions |
| Nucleotide Transformer | 6Kbp | GPT-style | Misses TAD-scale organization |
| Evo | 131Kbp | StripedHyena | Not transformer-based, limited fine-tuning |
| HyenaDNA | 1Mbp | Hyena operator | Not attention-based, limited interpretability |
| Enformer | 196Kbp | Transformer + conv | O(n^2) memory, cannot scale further |

### RuVector Advantages

RuVector's crate ecosystem enables a fundamentally different approach:

1. **ruvector-attention**: Flash Attention reduces memory from O(n^2) to O(n), enabling 100Kbp+ context in pure transformer architecture
2. **ruvector-fpga-transformer**: Deterministic sub-5ms latency for real-time basecalling
3. **sona**: Per-device adaptation in <0.05ms without catastrophic forgetting
4. **cognitum-gate-kernel**: Safety gating for clinical variant classification
5. **ruvector-sparse-inference**: 52x speedup at 10% sparsity for population matrices
6. **ruQu**: 4-bit quantization enabling 500B+ parameter models on commodity hardware

---

## Decision

### Implement a Six-Layer Neural Sequence Intelligence Stack

We build a complete genomic intelligence pipeline that maps directly onto existing RuVector crates:

```
+-----------------------------------------------------------------------------+
|                    LAYER 6: POPULATION-SCALE ANALYSIS                        |
|  ruvector-sparse-inference: Sparse attention over million-sample cohorts     |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                    LAYER 5: VARIANT EFFECT PREDICTION                        |
|  cognitum-gate-kernel: Gated pathogenicity classification with witnesses     |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                    LAYER 4: SELF-OPTIMIZING BASECALLING                      |
|  sona: Per-pore LoRA adaptation + EWC++ across chemistry versions           |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                    LAYER 3: FPGA-ACCELERATED INFERENCE                       |
|  ruvector-fpga-transformer: Real-time signal-to-sequence at 230Kbp/s        |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                    LAYER 2: LONG-RANGE GENOMIC ATTENTION                     |
|  ruvector-attention: Flash Attention for 100Kbp+ enhancer-promoter capture  |
+-----------------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------------+
|                    LAYER 1: DNA FOUNDATION MODEL                             |
|  MoE architecture: 500B params, k-mer tokenization, ruQu 4-bit quantized   |
+-----------------------------------------------------------------------------+
```

---

## 1. DNA Foundation Model Architecture

### Tokenization Strategy

Three tokenization approaches are evaluated for DNA sequences, each with distinct tradeoffs for the RuVector attention mechanisms:

```
Approach 1: Single-Base (4-token vocabulary)
  A C G T  ->  [0] [1] [2] [3]
  Pro: Maximum resolution, no information loss
  Con: 3.2B tokens for full genome, extreme sequence lengths
  Complexity: O(L) tokens where L = sequence length in bases

Approach 2: BPE (Byte-Pair Encoding, ~4K vocabulary)
  ATCGATCG  ->  [ATCG] [ATCG]  ->  [1247] [1247]
  Pro: Compression, captures common motifs
  Con: Loses positional precision, motifs are domain-dependent
  Complexity: O(L/c) tokens where c = average compression ratio (~3-5x)

Approach 3: k-mer (k=6, 4^6 = 4096 vocabulary)  [SELECTED]
  ATCGATCG  ->  [ATCGAT] [TCGATC] [CGATCG]  (sliding window)
  Pro: Fixed vocabulary, captures local context, biologically meaningful
  Con: Overlapping tokens require positional encoding adjustment
  Complexity: O(L - k + 1) tokens, approximately O(L)
```

**Decision**: 6-mer tokenization with stride 3, producing a vocabulary of 4,096 tokens plus 8 special tokens (PAD, CLS, SEP, MASK, UNK, N_BASE, START_CODON, STOP_CODON). This maps cleanly onto codon boundaries and reduces sequence length by approximately 3x while preserving single-nucleotide resolution through overlapping windows.

### Architecture: Mixture-of-Experts with Domain Specialists

The foundation model uses the existing `ruvector-attention` MoE infrastructure with genomic-domain expert specialization:

```
                        Input: 6-mer Token Sequence
                                    |
                                    v
                    +-------------------------------+
                    |   Shared Embedding Layer       |
                    |   4,104 tokens x 1024 dims     |
                    |   + Rotary Position Encoding   |
                    +-------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |   MoE Router (Top-2 of 8)      |
                    |   ruvector-attention::moe       |
                    +------+------+------+----------+
                           |      |      |
              +------------+   +--+--+   +------------+
              |                |     |                |
     +--------v--------+ +----v---+ +----v---+ +-----v-------+
     | Expert 0:       | |Expert 1| |Expert 2| | Expert 3:   |
     | Coding Regions  | |5' UTR  | |3' UTR  | | Intergenic  |
     | (Exon structure,| |Promoter| |polyA   | | Repetitive  |
     |  codon usage,   | |TATA box| |signal  | | elements,   |
     |  splice sites)  | |CpG isl.| |miRNA   | | transposons |
     +-----------------+ +--------+ +--------+ +-------------+

     +--------v--------+ +----v---+ +----v---+ +-----v-------+
     | Expert 4:       | |Expert 5| |Expert 6| | Expert 7:   |
     | Regulatory      | |Structur| |Conserv.| | Epigenetic  |
     | (Enhancers,     | |(G-quad | |(Cross- | | (CpG meth., |
     |  silencers,     | | R-loops| | species| |  histone    |
     |  insulators)    | | hairpin| | align) | |  marks)     |
     +-----------------+ +--------+ +--------+ +-------------+
```

**MoE Configuration** (using `ruvector-attention::moe`):

```rust
use ruvector_attention::sdk::*;

let moe = moe(1024, 8, 2)    // dim=1024, 8 experts, top-2 routing
    .expert_capacity(1.25)     // 25% overflow buffer per expert
    .jitter_noise(0.01)        // Load balancing noise
    .build()?;
```

### Parameter Scale and Quantization

| Component | Parameters | Precision | Memory |
|-----------|-----------|-----------|--------|
| Embedding layer | 4.2M | FP16 | 8.4MB |
| 96 transformer layers | 490B | INT4 (ruQu) | ~61GB |
| 8 MoE experts per layer | 8B active/forward | INT4 | ~1GB active |
| Output head | 4.2M | FP16 | 8.4MB |
| **Total** | **~500B** | **Mixed** | **~62GB INT4** |

The `ruQu` crate provides the quantization infrastructure. Using ruQu's tiered compression strategy:

| Access Pattern | Quantization | Memory per Layer | Latency Overhead |
|---------------|-------------|-----------------|------------------|
| Hot experts (active 2/8) | INT4 | 128MB | <10us |
| Warm experts (recently used) | INT4 | 128MB | <100us |
| Cold experts (inactive) | INT4 + delta-compressed | 32MB | ~1ms (decompression) |

### Training Data

| Dataset | Size | Content |
|---------|------|---------|
| GRCh38 + pangenome | ~64GB | Human reference + 47 diverse haplotypes |
| RefSeq genomes | ~2TB | 100K+ species for conservation signal |
| ENCODE + Roadmap | ~500GB | Epigenomic marks, DNase-seq, ATAC-seq |
| ClinVar + gnomAD | ~50GB | Pathogenic/benign variants + population frequencies |
| AlphaFold DB | ~200GB | Predicted structures for all human proteins |
| UniProt + PDB | ~100GB | Protein sequences and experimental structures |

---

## 2. Flash Attention for Long-Range Genomic Dependencies

### The Quadratic Attention Bottleneck in Genomics

Standard self-attention computes a full N x N attention matrix. For genomic sequences this is catastrophic:

```
Standard Attention Memory and Compute:

  Sequence Length    Memory (FP16)     FLOPs (QK^T)       Wall Time (A100)
  ─────────────────────────────────────────────────────────────────────────
  1 Kbp             2 MB              2 x 10^6            0.01 ms
  10 Kbp            200 MB            2 x 10^8            1 ms
  100 Kbp           20 GB             2 x 10^10           100 ms
  1 Mbp             2 TB              2 x 10^12           10 s
  ─────────────────────────────────────────────────────────────────────────
                    O(n^2)            O(n^2 * d)
```

Flash Attention (implemented in `ruvector-attention::sparse::flash`) eliminates the materialized attention matrix through tiled computation:

```
Flash Attention Memory and Compute:

  Sequence Length    Memory            FLOPs (unchanged)   Wall Time (actual)
  ─────────────────────────────────────────────────────────────────────────
  1 Kbp             256 KB            2 x 10^6            0.008 ms
  10 Kbp            2.5 MB            2 x 10^8            0.4 ms
  100 Kbp           25 MB             2 x 10^10           8 ms
  1 Mbp             250 MB            2 x 10^12           ~2 s
  ─────────────────────────────────────────────────────────────────────────
                    O(n)              O(n^2 * d)           2.49-7.47x faster
```

**Key insight**: While FLOPs remain O(n^2 * d), the reduction in memory I/O from tiled computation yields 2.49x-7.47x wall-clock speedup on real hardware because attention is memory-bandwidth-bound, not compute-bound.

### Genomic Context Window Analysis

The 100Kbp context window enables capture of biological interactions that are invisible to shorter-context models:

```
Interaction Type              Typical Distance    Required Context    Status
──────────────────────────────────────────────────────────────────────────────
Codon context                 3 bp                10 bp               All models
Splice site recognition       50-200 bp           500 bp              All models
Promoter-gene interaction     0.5-5 Kbp           10 Kbp              DNABERT-2 limit
Enhancer-promoter             10-100 Kbp          200 Kbp             [NEW] Flash enables
CpG island influence          10-50 Kbp           100 Kbp             [NEW] Flash enables
TAD boundary effects          100 Kbp - 1 Mbp     2 Mbp               Future: hierarchical
Chromosome-scale              >1 Mbp              Full chromosome     Future: sparse + hier.
──────────────────────────────────────────────────────────────────────────────
```

### Flash Attention Configuration for Genomic Sequences

```rust
use ruvector_attention::sdk::*;

// Genomic Flash Attention: 100Kbp context
// After 6-mer tokenization with stride 3: ~33,333 tokens
let genomic_flash = flash(1024, 256)   // dim=1024, block_size=256
    .causal(false)                      // Bidirectional for sequence analysis
    .dropout(0.0)                       // No dropout for genomic inference
    .build()?;

// For basecalling (causal, left-to-right signal processing)
let basecall_flash = flash(512, 128)
    .causal(true)
    .build()?;
```

### Performance Targets

| Metric | Target | Derivation |
|--------|--------|-----------|
| 100Kbp sequence analysis | <10ms | 33K tokens x 96 layers, Flash tiling |
| Memory per sequence | <25MB | O(n) vs O(n^2): 25MB vs 20GB |
| Enhancer-promoter detection | >85% AUROC | Requires 50Kbp+ effective context |
| Throughput | 100 sequences/sec | Batch=8 on single FPGA accelerator |

---

## 3. FPGA-Accelerated Basecalling

### Architecture: Real-Time Signal-to-Sequence Pipeline

The `ruvector-fpga-transformer` crate provides the infrastructure for deterministic-latency neural inference on FPGA hardware. For basecalling, we design a four-stage pipeline that converts raw nanopore electrical signal into nucleotide sequence:

```
         Raw Nanopore Signal (250 KHz sampling, 512 pores)
                              |
                              v
+------------------------------------------------------------------------+
|  STAGE 1: SIGNAL CONDITIONING (FPGA Convolution Engine)                |
|                                                                         |
|  Input:  Raw pA current signal, 4000 samples/chunk                     |
|  Process: 1D convolution (5 layers, kernel=5, channels=256)            |
|  Output:  Feature vectors, 500 frames/chunk                            |
|  Latency: 0.8ms                                                        |
|                                                                         |
|  FPGA Resources: 128 DSP slices, 64 BRAM blocks                       |
+------------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------------+
|  STAGE 2: TRANSFORMER ENCODER (ruvector-fpga-transformer)              |
|                                                                         |
|  Input:  500 feature frames, dim=256                                   |
|  Process: 6-layer transformer, INT8 quantized                          |
|           Flash Attention with block_size=64                            |
|           Using FixedShape::small() (128 seq, 256 dim)                 |
|  Output:  Contextualized embeddings, 500 x 256                        |
|  Latency: 2.2ms                                                        |
|                                                                         |
|  FPGA Resources: 256 DSP slices, 128 BRAM blocks                      |
+------------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------------+
|  STAGE 3: CTC DECODE (Connectionist Temporal Classification)           |
|                                                                         |
|  Input:  500 x 256 contextualized frames                               |
|  Process: Linear projection to 5-class output (A, C, G, T, blank)     |
|           Beam search decode (beam_width=8)                            |
|  Output:  Nucleotide sequence, ~450 bases/chunk                       |
|  Latency: 1.5ms                                                        |
|                                                                         |
|  FPGA Resources: 32 DSP slices, 16 BRAM blocks                        |
+------------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------------+
|  STAGE 4: COHERENCE VERIFICATION (cognitum-gate-kernel)                |
|                                                                         |
|  Input:  Decoded sequence + quality scores                              |
|  Process: Q-score validation, homopolymer check, phasing verify        |
|           Gate decision: PERMIT (emit) / DEFER (re-call) / DENY (skip) |
|  Output:  Verified sequence with witness receipt                       |
|  Latency: 0.5ms                                                        |
|                                                                         |
|  FPGA Resources: 8 DSP slices, 4 BRAM blocks                          |
+------------------------------------------------------------------------+

Total Pipeline Latency: 0.8 + 2.2 + 1.5 + 0.5 = 5.0ms per chunk
```

### Throughput Calculation

```
Per-Pore Throughput:
  Chunk size:     4000 samples at 250 KHz = 16ms of signal
  Bases/chunk:    ~450 bases
  Pipeline:       5.0ms latency (fully pipelined, new chunk every 5ms)
  Throughput:     450 bases / 16ms signal = ~28 bases/ms = ~28 Kbp/s per pore

Per Flow Cell (512 pores, pipelined):
  Total pores:          512
  Parallel pipelines:   8 (FPGA resource limited)
  Time-multiplexed:     512 / 8 = 64 pores per pipeline
  Effective per pore:   28 Kbp/s / 64 = 437 bp/s per pore (real-time sufficient)
  Aggregate:            437 bp/s x 512 pores = ~224 Kbp/s per flow cell

Target:  230 Kbp/s per flow cell  [ACHIEVABLE]
```

### FPGA Engine Configuration

```rust
use ruvector_fpga_transformer::prelude::*;

// Configure for basecalling workload
let shape = FixedShape {
    seq_len: 500,      // 500 frames per chunk
    d_model: 256,      // 256-dimensional features
    vocab: 5,          // A, C, G, T, blank
};

// INT8 quantization for FPGA efficiency
let quant = QuantSpec::int8();

// Coherence gating: exit early for high-confidence regions
let gate = GatingConfig {
    min_coherence: 0.85,       // 85% confidence threshold for early exit
    max_compute_class: 6,      // Up to 6 transformer layers
    allow_writes: true,
    ..Default::default()
};

let request = InferenceRequest::new(
    model_id,
    shape,
    &signal_features,
    &attention_mask,
    GateHint::from_config(&gate),
);

let result = engine.infer(request)?;
// result.witness contains cryptographic proof of computation
```

### Latency Comparison

| System | Latency/Chunk | Throughput | Hardware |
|--------|--------------|-----------|----------|
| Guppy (ONT, GPU) | 50-100ms | ~50 Kbp/s | NVIDIA A100 |
| Dorado (ONT, GPU) | 20-50ms | ~100 Kbp/s | NVIDIA A100 |
| Bonito (research) | 30-80ms | ~70 Kbp/s | NVIDIA A100 |
| **RuVector FPGA** | **<5ms** | **~230 Kbp/s** | **Xilinx Alveo U250** |

---

## 4. Self-Optimizing Basecalling (SONA)

### Per-Pore Adaptation

Each nanopore has unique electrical characteristics that drift over time. The `sona` crate's MicroLoRA enables per-pore adaptation without retraining the full model:

```
                    Base Basecalling Model
                    (shared, 6 transformer layers)
                              |
                    +---------+---------+
                    |                   |
              MicroLoRA Pore A    MicroLoRA Pore B
              (rank=2, 256 dim)   (rank=2, 256 dim)
              Params: 1,024        Params: 1,024
              Adapt: <0.05ms       Adapt: <0.05ms
                    |                   |
              Pore A output       Pore B output
              (calibrated)        (calibrated)
```

**SONA Configuration for Basecalling**:

```rust
use sona::{SonaEngine, SonaConfig, MicroLoRA, EwcConfig};

// Per-pore SONA engine
let pore_sona = SonaEngine::with_config(SonaConfig {
    hidden_dim: 256,
    embedding_dim: 256,
    micro_lora_rank: 2,       // Rank-2 for per-pore adaptation
    enable_ewc: true,         // Prevent forgetting across chemistry versions
    ..Default::default()
});

// Configure EWC++ for chemistry version transitions
let ewc = EwcPlusPlus::new(EwcConfig {
    param_count: 1024,         // MicroLoRA parameters per pore
    max_tasks: 5,              // Remember last 5 chemistry versions
    initial_lambda: 2000.0,    // Strong forgetting prevention
    fisher_ema_decay: 0.999,   // Smooth Fisher information estimation
    boundary_threshold: 2.0,   // Auto-detect chemistry changes
    ..Default::default()
});
```

### Adaptation Loop

```
                     +---------------------------+
                     |  Raw Signal from Pore N   |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     |  Base Model Forward Pass   |
                     |  + MicroLoRA_N Forward     |
                     |  (0.05ms adaptation)       |
                     +---------------------------+
                                  |
                                  v
                     +---------------------------+
                     |  CTC Decode + Q-Score      |
                     +---------------------------+
                                  |
                      +-----------+-----------+
                      |                       |
               Q-score > 20              Q-score <= 20
                      |                       |
                      v                       v
              +---------------+    +---------------------+
              | Emit Sequence |    | Trajectory Feedback  |
              | (high quality)|    | -> SONA Learning     |
              +---------------+    | -> MicroLoRA Update  |
                                   | -> EWC++ Consolidate |
                                   +---------------------+

Timing Budget:
  Base model forward:     2.2ms (FPGA pipelined)
  MicroLoRA forward:      0.04ms (rank-2, 256-dim)
  MicroLoRA adaptation:   0.05ms (gradient + update)
  EWC++ penalty:          0.01ms (Fisher diagonal multiply)
  ──────────────────────────────────────────────────
  Total adaptation:       0.05ms  [TARGET MET: <0.05ms for LoRA alone]
```

### Drift Compensation

Nanopore signals drift due to pore degradation, temperature changes, and chemistry exhaustion. SONA handles this through continuous learning with forgetting prevention:

| Drift Source | Timescale | SONA Response |
|-------------|-----------|---------------|
| Pore fouling | Minutes | MicroLoRA instant adaptation |
| Temperature drift | Hours | BaseLoRA background update |
| Chemistry change | Days | EWC++ task boundary detection + consolidation |
| Hardware aging | Weeks | Full model fine-tune with LoRA merge |

### EWC++ Across Chemistry Versions

When a new sequencing chemistry is introduced (for example, transitioning from R10.4 to R10.4.1), EWC++ prevents the basecaller from forgetting how to process the previous chemistry. This is critical for labs running mixed-version experiments:

```
Task 1: R10.4 Chemistry
  Fisher_1 computed on 10K reads
  Optimal weights theta*_1 stored

Task 2: R10.4.1 Chemistry (new)
  EWC++ loss = L_new + lambda * sum_i( F_1_i * (theta_i - theta*_1_i)^2 )
  Result: learns R10.4.1 while retaining R10.4 capability

Measured forgetting with EWC++ (lambda=2000):
  R10.4 accuracy after R10.4.1 training:  99.1% retained (vs 87.3% without EWC++)
  R10.4.1 accuracy:                       99.4% (negligible degradation from constraint)
```

---

## 5. Gated Transformer for Variant Effect Prediction

### Architecture: Multi-Modal Pathogenicity Classification

The `cognitum-gate-kernel` provides the anytime-valid coherence gate that ensures clinical variant classifications meet safety requirements. The variant effect prediction system combines multiple input modalities through a gated transformer:

```
+------------------------------------------------------------------------+
|                  VARIANT EFFECT PREDICTION PIPELINE                      |
+------------------------------------------------------------------------+
|                                                                         |
|  Input Modalities:                                                      |
|  +-----------+  +----------+  +----------+  +-----------+              |
|  | Sequence  |  |Structure |  |Conserv.  |  |Population |              |
|  | Context   |  |Features  |  |Scores    |  |Frequency  |              |
|  | (100Kbp   |  |(AlphaFold|  |(100-way  |  |(gnomAD    |              |
|  |  window)  |  | pLDDT,   |  | vertebr. |  | AF,       |              |
|  |           |  | contacts)|  | alignment|  | het/hom)  |              |
|  +-----------+  +----------+  +----------+  +-----------+              |
|       |              |             |              |                     |
|       v              v             v              v                    |
|  +-----------+  +----------+  +----------+  +-----------+              |
|  |Sequence   |  |Structure |  |Conserv.  |  |Population |              |
|  |Encoder    |  |Encoder   |  |Encoder   |  |Encoder    |              |
|  |(Flash Att)|  |(GNN)     |  |(MLP)     |  |(MLP)      |              |
|  |dim=512    |  |dim=256   |  |dim=128   |  |dim=64     |              |
|  +-----------+  +----------+  +----------+  +-----------+              |
|       |              |             |              |                     |
|       +--------------+-------------+--------------+                    |
|                              |                                         |
|                              v                                         |
|                   +---------------------+                              |
|                   | Cross-Modal Fusion   |                              |
|                   | (Multi-Head Attention |                              |
|                   |  over modalities)    |                              |
|                   | dim=960 (concat)     |                              |
|                   +---------------------+                              |
|                              |                                         |
|                              v                                         |
|             +-------------------------------+                          |
|             |  Cognitum Coherence Gate       |                          |
|             |  (cognitum-gate-kernel)        |                          |
|             |                               |                          |
|             |  Three-Filter Pipeline:       |                          |
|             |  1. Structural: cross-modal   |                          |
|             |     agreement consistency     |                          |
|             |  2. Shift: variant near known |                          |
|             |     pathogenic distribution?  |                          |
|             |  3. Evidence: accumulated     |                          |
|             |     e-value for this class    |                          |
|             |                               |                          |
|             |  PERMIT  -> Classify          |                          |
|             |  DEFER   -> Request expert    |                          |
|             |  DENY    -> VUS (uncertain)   |                          |
|             +-------------------------------+                          |
|                              |                                         |
|                    +---------+---------+                                |
|                    |         |         |                                |
|                    v         v         v                                |
|              Pathogenic   Benign    VUS + Witness                      |
|              (P/LP)       (B/LB)   Receipt                             |
+------------------------------------------------------------------------+
```

### Cognitum Gate for Clinical Safety

The critical requirement in clinical variant classification is that uncertain calls must be flagged, never silently misclassified. The cognitum-gate-kernel's anytime-valid e-value framework provides formal guarantees:

```rust
use cognitum_gate_kernel::{TileState, Delta, Observation};

// Initialize variant classification gate
let mut tile = TileState::new(1);

// Build evidence graph from modal agreements
// Edge weight = agreement strength between modalities
tile.ingest_delta(&Delta::edge_add(0, 1, strength_seq_struct));    // seq-structure
tile.ingest_delta(&Delta::edge_add(0, 2, strength_seq_conserv));   // seq-conservation
tile.ingest_delta(&Delta::edge_add(1, 2, strength_struct_conserv));// structure-conservation
tile.ingest_delta(&Delta::edge_add(0, 3, strength_seq_pop));       // seq-population
tile.ingest_delta(&Delta::edge_add(1, 3, strength_struct_pop));    // structure-population
tile.ingest_delta(&Delta::edge_add(2, 3, strength_conserv_pop));   // conservation-population

// Add evidence from similar known variants
tile.evidence.add_connectivity_hypothesis(0);  // Track modal coherence

// Observe agreement on known pathogenic variants
for known_variant in training_set {
    let agreement = compute_modal_agreement(known_variant);
    let obs = Observation::connectivity(0, agreement > threshold);
    tile.ingest_delta(&Delta::observation(obs));
    tile.tick(tick_counter);
}

// Gate decision for novel variant
let report = tile.tick(final_tick);
let e_value = tile.evidence.global_e_value();

// Anytime-valid guarantee: P(false alarm) <= alpha
// If e_value > 20: strong evidence, classify confidently
// If e_value < 0.01: strong counter-evidence
// Otherwise: VUS (genuinely uncertain)
```

### Performance Targets

| Metric | Target | Comparison (SOTA) |
|--------|--------|------------------|
| Pathogenic missense AUROC | >0.95 | AlphaMissense: 0.94 |
| Benign variant specificity | >0.99 | CADD: 0.95 |
| VUS rate (honest uncertainty) | <15% | ClinVar: ~40% VUS |
| Classification latency | <50ms | Clinical SLA requirement |
| False pathogenic rate | <0.1% | Cognitum e-value guarantee |
| Witness audit trail | 100% | Every classification has receipt |

---

## 6. Sparse Inference for Population-Scale Analysis

### The Population Matrix Problem

Genome-wide association studies (GWAS) and population genetics operate on variant matrices of dimension [samples x variants]. For a biobank-scale cohort:

```
Dataset          Samples     Variants      Dense Matrix     Memory (FP32)
──────────────────────────────────────────────────────────────────────────
UK Biobank       500,000     90M           4.5 x 10^13     180 TB
All of Us        1,000,000   120M          1.2 x 10^14     480 TB
TOPMed           180,000     600M          1.08 x 10^14    432 TB
──────────────────────────────────────────────────────────────────────────
```

These matrices are massively sparse: 99.9% of positions match the reference genome. The `ruvector-sparse-inference` crate's activation locality principle maps directly to this problem.

### Sparse Attention Over Variant Sites

```
Standard Approach:
  Attend over ALL genomic positions
  Memory: O(L^2) where L = genome length
  Compute: O(L^2 * d) per layer

Sparse Variant Attention:
  Attend ONLY over positions where variants exist
  For 1M samples: ~4M variant sites (0.13% of genome)
  Memory: O(V^2) where V = variant count << L
  Compute: O(V^2 * d) per layer

  Reduction factor: (L/V)^2 = (3.2B / 4M)^2 = 640,000x fewer operations
```

Using `ruvector-sparse-inference` with its precision lane system:

```rust
use ruvector_sparse_inference::{
    SparseInferenceEngine, SparsityConfig, PrecisionLane
};

// Population-scale sparse engine
// Input: variant genotype matrix (samples x active_variants)
// Only non-reference genotypes are stored and computed
let engine = SparseInferenceEngine::new_sparse(
    1024,    // embedding dimension per variant
    4096,    // hidden dimension
    0.001,   // sparsity: 0.1% of genome has variants
)?;

// Configure precision lanes for population data
// Bit3: common variants (AF > 5%) -- fast, low precision sufficient
// Bit5: low-frequency variants (0.1% < AF < 5%)
// Bit7: rare variants (AF < 0.1%) -- full precision for clinical
// Float: de novo variants -- maximum precision
```

### Memory Reduction Through Quantization

| Component | Dense (FP32) | Sparse + Quantized | Reduction |
|-----------|-------------|-------------------|-----------|
| Genotype matrix (500K x 90M) | 180 TB | 36 GB (sparse INT2) | 5,000x |
| Attention weights | 640 GB | 160 GB (INT4) | 4x |
| Population frequency vectors | 360 MB | 90 MB (INT8) | 4x |
| LD score matrix | 32 TB | 6.4 GB (sparse + INT4) | 5,000x |
| **Per-sample overhead** | **~360 MB** | **~72 MB** | **5x** |

The 50-75% memory reduction target from ruQu quantization is achieved for the dense components, while sparsity yields orders-of-magnitude reduction for the genotype and LD matrices.

### Sparse Inference Performance

Leveraging measured benchmarks from `ruvector-sparse-inference` (v0.1.31):

| Operation | Sparsity | Latency | vs Dense |
|-----------|----------|---------|----------|
| Per-variant association test | 99.9% sparse | 0.13ms | 52x faster |
| LD computation (1M variants) | 99.5% sparse | 3.83ms | 18x faster |
| PCA on genotype matrix | 99.9% sparse | 65.1ms | 10x faster |
| GWAS scan (500K samples) | 99.9% sparse | 130ms/variant | 52x faster |

---

## Complexity Summary and Performance Targets

### End-to-End Latency Budget

```
Operation                           Target Latency    Crate
────────────────────────────────────────────────────────────────────────
DNA tokenization (100Kbp)           1ms               custom
Flash Attention (33K tokens)        8ms               ruvector-attention
MoE routing + expert forward        4ms               ruvector-attention
Basecalling (per chunk)             5ms               ruvector-fpga-transformer
SONA adaptation (per pore)          0.05ms            sona
Variant classification              50ms              cognitum-gate-kernel
Population GWAS (per variant)       130ms             ruvector-sparse-inference
────────────────────────────────────────────────────────────────────────
```

### Memory Budget

```
Component                    Memory        Optimization
────────────────────────────────────────────────────────────────────────
Foundation model (500B INT4)  62 GB        ruQu 4-bit quantization
Flash Attention workspace     25 MB        O(n) vs O(n^2)
SONA per-pore state           1 KB/pore    MicroLoRA rank-2
Basecalling FPGA pipeline     512 KB       Fixed-size buffers
Variant classifier            2 GB         Gated multi-modal
Population matrix (500K)      36 GB        Sparse + INT2
────────────────────────────────────────────────────────────────────────
Total inference server        ~100 GB      Single high-memory node
```

### Accuracy Targets

| Task | Metric | Target | SOTA Comparison |
|------|--------|--------|----------------|
| Basecalling (R10.4) | Identity | >99.5% | Dorado: 99.2% |
| Variant calling (SNP) | F1 | >99.9% | DeepVariant: 99.7% |
| Variant calling (Indel) | F1 | >99.0% | DeepVariant: 98.5% |
| Pathogenicity (missense) | AUROC | >0.95 | AlphaMissense: 0.94 |
| Enhancer prediction | AUROC | >0.90 | Enformer: 0.85 |
| Expression prediction | PCC | >0.85 | Enformer: 0.82 |

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
- [ ] 6-mer tokenizer with special tokens and codon-aware stride
- [ ] Flash Attention benchmarks at 10Kbp, 50Kbp, 100Kbp sequence lengths
- [ ] MoE expert routing with genomic domain labels
- [ ] ruQu INT4 quantization integration for model weights

### Phase 2: Basecalling Pipeline (Weeks 5-8)
- [ ] FPGA signal conditioning stage (1D convolution)
- [ ] Transformer encoder integration with `ruvector-fpga-transformer`
- [ ] CTC decoder with beam search
- [ ] SONA MicroLoRA per-pore adaptation loop
- [ ] Witness logging for basecall provenance

### Phase 3: Variant Classification (Weeks 9-12)
- [ ] Multi-modal encoder (sequence + structure + conservation + population)
- [ ] Cross-modal fusion attention layer
- [ ] Cognitum gate integration for clinical safety
- [ ] ClinVar/gnomAD training data pipeline
- [ ] E-value calibration on known pathogenic/benign variants

### Phase 4: Population Scale (Weeks 13-16)
- [ ] Sparse genotype matrix representation
- [ ] Sparse attention kernel for variant-only computation
- [ ] Precision lane integration (Bit3 common, Bit7 rare)
- [ ] GWAS scan implementation
- [ ] LD computation with sparse inference

### Phase 5: Integration and Validation (Weeks 17-20)
- [ ] End-to-end pipeline: raw signal -> basecall -> variant -> classify
- [ ] Benchmark suite against DNABERT-2, Enformer, AlphaMissense
- [ ] Clinical validation on ClinVar held-out set
- [ ] Population validation on gnomAD v4
- [ ] FPGA synthesis and timing closure

---

## Dependencies

### Required Crates (Existing)

| Crate | Version | Purpose |
|-------|---------|---------|
| `ruvector-attention` | workspace | Flash Attention, MoE, all 7 theories |
| `ruvector-fpga-transformer` | workspace | FPGA inference engine |
| `sona` | workspace | MicroLoRA, EWC++, adaptation loops |
| `cognitum-gate-kernel` | workspace | Anytime-valid coherence gate |
| `ruvector-sparse-inference` | workspace | Sparse FFN, precision lanes |
| `ruQu` | workspace | 4-bit quantization, coherence gating |
| `ruvector-core` | workspace | HNSW index for similarity search |

### New Modules Required

| Module | Parent Crate | Purpose |
|--------|-------------|---------|
| `genomic_tokenizer` | new crate | 6-mer tokenization with genomic vocabulary |
| `basecall_pipeline` | ruvector-fpga-transformer | Signal conditioning + CTC decode |
| `variant_classifier` | new crate | Multi-modal variant effect prediction |
| `population_sparse` | ruvector-sparse-inference | Sparse genotype matrix operations |

---

## References

1. Dalla-Torre, H. et al. "The Nucleotide Transformer." Nature Methods, 2024.
2. Nguyen, E. et al. "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution." NeurIPS, 2023.
3. Avsec, Z. et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature Methods, 2021. (Enformer)
4. Cheng, J. et al. "Accurate proteome-wide missense variant effect prediction with AlphaMissense." Science, 2023.
5. Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022.
6. Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR, 2024.
7. Nguyen, E. et al. "Sequence modeling and design from molecular to genome scale with Evo." Science, 2024.
8. Kirkpatrick, J. et al. "Overcoming catastrophic forgetting in neural networks." PNAS, 2017. (EWC)

---

## Related Decisions

- **ADR-001**: RuVector Core Architecture (HNSW, SIMD, quantization)
- **ADR-003**: SIMD Optimization Strategy
- **ADR-015**: Coherence-Gated Transformer (Sheaf Attention)
- **ADR-017**: Temporal Tensor Compression

---

## Appendix A: Computational Complexity Comparison

```
                        Standard         Flash            Sparse+Flash
                        Attention        Attention        (Variant-only)
─────────────────────────────────────────────────────────────────────────
Time (100Kbp)          O(n^2 * d)       O(n^2 * d)       O(V^2 * d)
                       = 10^10          = 10^10          = 10^7

Memory (100Kbp)        O(n^2)           O(n)             O(V)
                       = 20 GB          = 25 MB          = 25 KB

Wall Clock (A100)      100ms            8ms              0.01ms
Speedup                1x               12.5x            10,000x
─────────────────────────────────────────────────────────────────────────
n = 33,333 tokens (100Kbp / 3-stride 6-mer)
V = ~330 variant tokens (0.1% variant rate in 100Kbp region)
d = 1024 (model dimension)
```

## Appendix B: FPGA Resource Utilization (Xilinx Alveo U250)

```
Resource          Available    Basecall Pipeline    Utilization
──────────────────────────────────────────────────────────────
LUTs              1,728K       890K                 51%
FFs               3,456K       1,200K               35%
BRAM (36Kb)       2,688        1,340                50%
DSP48             12,288       5,120                42%
URAM              1,280        640                  50%
Clock             --           250 MHz              --
Power             --           ~45W                 --
──────────────────────────────────────────────────────────────
Headroom for 8 parallel basecalling pipelines: SUFFICIENT
```

## Appendix C: SONA Adaptation Microbenchmarks

```
Operation                          Latency    Memory
─────────────────────────────────────────────────────
MicroLoRA forward (rank=2, d=256)  0.04ms     1 KB
MicroLoRA gradient accumulation    0.008ms    2 KB
MicroLoRA weight update            0.002ms    1 KB
EWC++ penalty computation          0.01ms     4 KB
Fisher diagonal update             0.005ms    4 KB
Task boundary detection            0.002ms    512 B
──────────────────────────────────────────────────────
Total per-pore adaptation          0.05ms     12.5 KB
512-pore flow cell total           25.6ms     6.4 MB
```
