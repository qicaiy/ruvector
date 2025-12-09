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
| **RuVector Core** | `ruvector-core` | HNSW index for pattern storage |
| **SimSIMD** | Cargo.toml | Similarity computations |
| **Candle** | Cargo.toml | Model loading (optional) |

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
│         │ SIMD   │    │ HNSW    │        │ Timestep │      │ MDLM/    │    │
│         │ Embed  │    │ Index   │        │ Router   │      │ BD3LM    │    │
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
│  │                    SONA Learning Loops                               │   │
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
│   │   └── dgr.rs                 # DGR (uses RuVector) [NOVEL]
│   │
│   ├── federation/                # NEW: Federated learning
│   │   ├── mod.rs
│   │   ├── daf.rs                 # DAF protocol [NOVEL]
│   │   ├── gossip.rs              # Gossip protocol
│   │   └── privacy.rs             # Privacy tiers
│   │
│   └── gpu/                       # NEW: GPU acceleration
│       ├── mod.rs
│       ├── cuda.rs                # CUDA (via cudarc)
│       └── metal.rs               # Metal (via candle)
│
├── Cargo.toml                     # EXISTING: Already has most deps
│   # ✓ ruvector-core, ruvector-gnn, ruvector-attention
│   # ✓ simsimd, candle-*, tokio
│   # + Add: quinn, cudarc (optional)
│
└── docs/diffusion/                # This documentation
```

### Dependency on Existing Code

```rust
// TALoRA wraps existing MicroLoRA
use crate::sona::lora::{MicroLoRA, BaseLoRA, LoRAEngine};

// Uses existing SIMD infrastructure
use crate::simd_inference::{SimdOps, Q4Weights};

// Integrates with existing SONA loops
use crate::sona::loops::{InstantLoop, BackgroundLoop};

// Uses RuVector for pattern storage
use ruvector_core::HnswIndex;
```

## Key Differentiators

### vs. dLLM (Python)
| Aspect | dLLM | RuvDLLM |
|--------|------|---------|
| Language | Python | Rust |
| Inference | PyTorch | Native SIMD/GPU |
| Real-time adaptation | No | Yes (MicroLoRA) |
| Federation | No | Yes (DAF) |
| Privacy tiers | No | Yes |
| Memory | ~4GB overhead | ~500MB overhead |

### vs. DiffuLLaMA
| Aspect | DiffuLLaMA | RuvDLLM |
|--------|------------|---------|
| Adaptation | Static | Dynamic (TALoRA) |
| Retrieval | None | DGR |
| Federation | None | DAF |
| Deployment | GPU required | CPU viable |

### vs. RAMoLE/LoraRetriever
| Aspect | RAMoLE | RuvDLLM |
|--------|--------|---------|
| Model type | Autoregressive | Diffusion |
| Timestep awareness | N/A | TALoRA |
| Uncertainty guidance | No | DGR |
| Implementation | Python | Rust |

## Development Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Core diffusion model with Q4 quantization
- [ ] MDLM sampler with SIMD optimization
- [ ] Basic QLoRA conversion pipeline

### Phase 2: Novel Contributions (Weeks 3-4)
- [ ] TALoRA implementation
- [ ] DGR implementation
- [ ] Integration with RuVector

### Phase 3: Federation (Weeks 5-6)
- [ ] DAF protocol
- [ ] Privacy tiers
- [ ] Gossip sync

### Phase 4: Optimization (Weeks 7-8)
- [ ] GPU kernels (CUDA/Metal)
- [ ] Benchmark suite
- [ ] Production hardening

## Success Criteria

1. **Performance**: Meet latency and throughput targets
2. **Novelty**: TALoRA, DGR, and DAF working and benchmarked
3. **Security**: Pass security audit for federation
4. **Usability**: Clean API, good documentation
5. **Compatibility**: Works with existing ruvLLM ecosystem

## References

- [DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) - AR→Diffusion conversion
- [dLLM](https://github.com/ZHZisZZ/dllm) - Diffusion LLM framework
- [RAMoLE](https://arxiv.org/abs/2406.16989) - Retrieval-augmented LoRA
- [FedEx-LoRA](https://arxiv.org/abs/2410.09432) - Federated LoRA
- [C-LoRA](https://jamessealesmith.github.io/continual-diffusion/) - Continual LoRA

---

**Next**: [01-ARCHITECTURE.md](./01-ARCHITECTURE.md) - Detailed system architecture
