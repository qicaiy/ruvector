# RuvDLLM: High-Speed Diffusion LLM Self-Learning Framework

## Executive Summary

**RuvDLLM** is a diffusion language model extension for **ruvLLM**, leveraging the existing SIMD infrastructure, SONA learning loops, MicroLoRA adapters, and RuVector integration. It adds diffusion-specific capabilities while reusing proven components.

### Building on Existing Infrastructure

| Existing Component | Location | Reused For |
|-------------------|----------|------------|
| **MicroLoRA/BaseLoRA** | `sona/lora.rs` | Foundation for TALoRA |
| **SIMD Ops** | `simd_inference.rs` | Dot products, softmax, RMSNorm |
| **Q4 Quantization** | `simd_inference.rs` | `Q4Weights` for diffusion model |
| **SONA Loops** | `sona/loops/` | Instant/background/deep learning |
| **Federated Learning** | `crates/sona/training/federated.rs` | `FederatedCoordinator`, `EphemeralAgent` |
| **HNSW Index** | `crates/ruvector-core/index/hnsw.rs` | Pattern retrieval |
| **Flash Attention** | `crates/ruvector-attention/sparse/flash.rs` | Memory-efficient attention |
| **EWC** | `crates/ruvector-gnn/ewc.rs` | Elastic Weight Consolidation |
| **SimSIMD** | Cargo.toml | Similarity computations |
| **Candle** | Cargo.toml | Model loading (optional) |

### What Already Exists vs What's New

```
EXISTING (in ruvector ecosystem)              NEW (diffusion-specific)
────────────────────────────────              ────────────────────────
✓ MicroLoRA (rank 1-2, AVX2)                  ★ TALoRA timestep routing
✓ FederatedCoordinator                        ★ DAF aggregation strategies
✓ EphemeralAgent                              ★ Diffusion-aware privacy tiers
✓ HnswIndex (cosine, euclidean)               ★ DGR uncertainty-guided retrieval
✓ FlashAttention                              ★ MDLM/BD3LM samplers
✓ Q4Weights quantization                      ★ Noise scheduler
✓ SONA learning loops                         ★ Bidirectional attention for diffusion
✓ EWC (Elastic Weight Consolidation)          ★ QLoRA conversion pipeline
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RuvDLLM Framework                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NOVEL CONTRIBUTIONS                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  • Timestep-Aware LoRA (TALoRA) - Different adapters per denoise   │   │
│  │  • Denoising-Guided Retrieval (DGR) - Uncertainty drives retrieval │   │
│  │  • Diffusion-Aware Federation (DAF) - Schedule-aligned aggregation │   │
│  │  • First Rust implementation of full stack                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CORE CAPABILITIES                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  • AR→Diffusion QLoRA conversion (from any LLaMA/Qwen model)       │   │
│  │  • Real-time MicroLoRA adaptation (<1ms overhead)                  │   │
│  │  • Federated learning with hybrid privacy tiers                    │   │
│  │  • CPU SIMD + GPU acceleration                                     │   │
│  │  • RuVector integration for pattern storage                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Goals

### Performance Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Inference latency (7B, 8 steps) | <100ms | <50ms |
| Adaptation overhead | <1ms | <0.5ms |
| Tokens/second (CPU SIMD) | 200+ | 500+ |
| Tokens/second (GPU) | 1000+ | 2000+ |
| Memory (7B Q4 + adapters) | <6GB | <4GB |
| Pattern retrieval (HNSW) | <0.1ms | <0.05ms |

### Originality Goals

1. **Timestep-Aware LoRA (TALoRA)**: Novel contribution - no existing work applies different LoRA adapters at different diffusion timesteps for text generation
2. **Denoising-Guided Retrieval (DGR)**: Novel contribution - using model uncertainty during denoising to dynamically retrieve adapters
3. **Diffusion-Aware Federation (DAF)**: Novel contribution - aggregating federated updates with awareness of noise schedule semantics

### Security & Privacy Goals

| Tier | Data Scope | Protection |
|------|------------|------------|
| Private | User-only | E2E encrypted, never leaves device |
| Group | Team | Group key encryption |
| Tenant | Organization | Org-wide access control |
| Public | Global | Differential privacy (ε=1.0), k-anonymity (k=5) |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RuvDLLM Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query ──► Embed ──► RuVector Search ──► TALoRA Selection ──► Diffusion    │
│              │              │                   │                  │         │
│              │              │                   │                  │         │
│              ▼              ▼                   ▼                  ▼         │
│         ┌────────┐    ┌─────────┐        ┌──────────┐      ┌──────────┐    │
│         │ SIMD   │    │ HnswIdx │        │ Timestep │      │ MDLM/    │    │
│         │ Embed  │    │ (exist) │        │ Router   │      │ BD3LM    │    │
│         └────────┘    └─────────┘        └──────────┘      └──────────┘    │
│                                                                   │         │
│                                                                   ▼         │
│                                                            ┌──────────┐    │
│                                                            │ Response │    │
│                                                            └────┬─────┘    │
│                                                                 │          │
│         ┌───────────────────────────────────────────────────────┘          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SONA Learning Loops (EXISTING)                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Loop A (Instant): Record trajectory, update MicroLoRA bank         │   │
│  │  Loop B (Background): Cluster patterns, train BaseLoRA (CPU SIMD)   │   │
│  │  Loop C (Deep): Consolidate, EWC++, federate if consented           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
examples/ruvLLM/
├── src/
│   ├── sona/                      # EXISTING: SONA self-learning
│   │   ├── lora.rs                # ✓ MicroLoRA, BaseLoRA, LoRAEngine
│   │   ├── engine.rs              # ✓ SONA orchestration
│   │   ├── ewc.rs                 # ✓ Elastic Weight Consolidation
│   │   ├── reasoning_bank.rs      # ✓ Pattern storage
│   │   └── loops/                 # ✓ Instant/Background/Coordinator
│   │
│   ├── simd_inference.rs          # EXISTING: SIMD operations
│   │   # ✓ SimdOps (dot_product_avx2, softmax_avx2, rms_norm_avx2)
│   │   # ✓ Q4Weights quantization
│   │   # ✓ TransformerLayer, KvCache
│   │
│   ├── diffusion/                 # NEW: Diffusion extensions
│   │   ├── mod.rs                 # Module exports
│   │   ├── model.rs               # Diffusion model (uses Q4Weights)
│   │   ├── sampler.rs             # MDLM/BD3LM samplers
│   │   ├── scheduler.rs           # Noise schedules
│   │   ├── qlora_convert.rs       # AR→Diffusion conversion
│   │   ├── talora.rs              # TALoRA (wraps MicroLoRA) [NOVEL]
│   │   └── dgr.rs                 # DGR (uses HnswIndex) [NOVEL]
│   │
│   ├── federation/                # EXTEND: Build on existing
│   │   ├── mod.rs
│   │   ├── daf.rs                 # DAF (extends FederatedCoordinator)
│   │   └── privacy_tiers.rs       # Privacy tier extensions
│   │
│   └── gpu/                       # NEW: GPU acceleration
│       ├── mod.rs
│       ├── cuda.rs                # CUDA (via cudarc)
│       └── metal.rs               # Metal (via candle)
│
├── Cargo.toml                     # EXISTING: Already has most deps
│
└── docs/diffusion/                # This documentation

crates/
├── ruvector-core/                 # EXISTING: Reuse
│   └── src/index/hnsw.rs          # ✓ HnswIndex for pattern retrieval
│
├── ruvector-attention/            # EXISTING: Reuse
│   └── src/sparse/flash.rs        # ✓ FlashAttention
│
├── ruvector-gnn/                  # EXISTING: Reuse
│   └── src/ewc.rs                 # ✓ EWC implementation
│
└── sona/                          # EXISTING: Reuse & Extend
    └── src/training/federated.rs  # ✓ FederatedCoordinator, EphemeralAgent
```

### Dependency on Existing Code

```rust
// TALoRA wraps existing MicroLoRA
use crate::sona::lora::{MicroLoRA, BaseLoRA, LoRAEngine};

// Uses existing SIMD infrastructure
use crate::simd_inference::{SimdOps, Q4Weights, TransformerLayer};

// Integrates with existing SONA loops
use crate::sona::loops::{InstantLoop, BackgroundLoop};

// Uses existing HNSW index
use ruvector_core::index::HnswIndex;

// Uses existing flash attention
use ruvector_attention::sparse::FlashAttention;

// Extends existing federated learning
use ruvector_sona::training::federated::{FederatedCoordinator, EphemeralAgent};

// Uses existing EWC
use ruvector_gnn::ewc::Ewc;
```

### TALoRA: Extending MicroLoRA

```rust
/// TALoRA wraps existing MicroLoRA with timestep awareness
pub struct TALoRA {
    /// Reuse existing MicroLoRA for each bank
    banks: [Vec<MicroLoRA>; 3],  // Uses crate::sona::lora::MicroLoRA
    /// HNSW indices for retrieval (uses ruvector-core)
    indices: [HnswIndex; 3],     // Uses ruvector_core::index::HnswIndex
    /// Timestep boundaries
    boundaries: [u32; 2],
}

impl TALoRA {
    pub fn retrieve(&self, query: &[f32], timestep: u32, k: usize) -> Vec<&MicroLoRA> {
        let bank_idx = self.get_bank_index(timestep);
        // Use existing HnswIndex.search()
        let results = self.indices[bank_idx].search(query, k).unwrap();
        results.iter().map(|r| &self.banks[bank_idx][r.id.parse().unwrap()]).collect()
    }
}
```

### DAF: Extending FederatedCoordinator

```rust
/// DAF extends existing FederatedCoordinator with timestep awareness
pub struct DAFCoordinator {
    /// Base coordinator (from crates/sona)
    base: FederatedCoordinator,  // Uses ruvector_sona::training::federated
    /// Per-timestep-group aggregation strategies
    strategies: [AggregationStrategy; 3],
}

impl DAFCoordinator {
    /// Override aggregate to add timestep-aware logic
    pub fn aggregate_with_daf(&mut self, export: AgentExport) -> AggregationResult {
        // First use base coordinator's logic
        let base_result = self.base.aggregate(export.clone());

        // Then apply DAF-specific aggregation per timestep group
        // ...
        base_result
    }
}
```

## Key Differentiators

### vs. dLLM (Python)
| Aspect | dLLM | RuvDLLM |
|--------|------|---------|
| Language | Python | Rust |
| Inference | PyTorch | Native SIMD/GPU |
| Real-time adaptation | No | Yes (MicroLoRA) |
| Federation | No | Yes (existing + DAF) |
| Privacy tiers | No | Yes |
| Memory | ~4GB overhead | ~500MB overhead |

### vs. DiffuLLaMA
| Aspect | DiffuLLaMA | RuvDLLM |
|--------|------------|---------|
| Adaptation | Static | Dynamic (TALoRA) |
| Retrieval | None | DGR (uses HnswIndex) |
| Federation | None | DAF (extends existing) |
| Deployment | GPU required | CPU viable |

## Development Phases

### Phase 1: Foundation
- [ ] Diffusion sampler (MDLM) using existing Q4Weights
- [ ] Noise scheduler
- [ ] Bidirectional attention (adapt existing FlashAttention)
- [ ] Basic QLoRA conversion

### Phase 2: Novel Contributions
- [ ] TALoRA (wrap MicroLoRA with timestep routing)
- [ ] DGR (uncertainty → HnswIndex retrieval)
- [ ] Integration tests with existing SONA loops

### Phase 3: Federation Extension
- [ ] DAF (extend FederatedCoordinator)
- [ ] Privacy tiers (extend EphemeralAgent)
- [ ] Gossip sync for diffusion updates

### Phase 4: Optimization
- [ ] GPU kernels (CUDA/Metal)
- [ ] Benchmark suite
- [ ] Production hardening

## Success Criteria

1. **Performance**: Meet latency and throughput targets
2. **Novelty**: TALoRA, DGR, and DAF working and benchmarked
3. **Integration**: Clean extension of existing ruvector components
4. **Security**: Pass security audit for federation
5. **Usability**: Clean API, good documentation
6. **Compatibility**: Works with existing ruvLLM/ruvector ecosystem

## References

- [DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) - AR→Diffusion conversion
- [dLLM](https://github.com/ZHZisZZ/dllm) - Diffusion LLM framework
- [RAMoLE](https://arxiv.org/abs/2406.16989) - Retrieval-augmented LoRA
- [FedEx-LoRA](https://arxiv.org/abs/2410.09432) - Federated LoRA
- [C-LoRA](https://jamessealesmith.github.io/continual-diffusion/) - Continual LoRA

---

**Next**: [01-ARCHITECTURE.md](./01-ARCHITECTURE.md) - Detailed system architecture
