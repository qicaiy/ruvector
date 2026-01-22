# ADR-014: Recursive Language Model (RLM) Integration

**Status:** Proposed
**Date:** 2026-01-21
**Decision Makers:** Ruvector Architecture Team
**Technical Area:** Agentic Orchestration / Recursive Reasoning / LLM Integration

---

## Context and Problem Statement

Recursive Language Models (RLMs) represent an emerging paradigm where a language model can call itself or sub-models recursively to handle complex queries and virtually unbounded context. Instead of processing a single large prompt in one shot (which hits context length limits and causes "context rot"), an RLM decomposes tasks iteratively, treating external information as part of the environment the model can query on-demand.

This ADR proposes integrating RLM capabilities into the existing `ruvllm` Rust crate and `@ruvector/ruvllm` npm package, creating a unified system that combines:

1. **ruvllm** - Agentic orchestration layer for task routing and recursive control
2. **ruvector** - External memory module for semantic search and retrieval (RAG)
3. **mistral.rs** - Rust-native inference engine for quantized LLM execution

### Current State

The existing `ruvllm` crate (v2.3.0) already provides:
- **Claude Flow Integration**: Agent routing, task classification, model selection
- **HNSW Router**: 150x faster pattern search via semantic similarity
- **SONA Learning**: Three-tier temporal learning loops (instant/hourly/weekly)
- **ReasoningBank**: Trajectory recording, pattern storage, verdict analysis
- **Quality Scoring**: Coherence validation, diversity analysis, schema validators
- **Reflection System**: Self-correction via IoE confidence checking

The npm package provides TypeScript bindings for:
- Session management with multi-turn context
- SONA adaptive learning coordination
- Federated learning with ephemeral agents
- LoRA adapter hot-swapping
- Streaming inference

### Gap Analysis

| Capability | Current Status | RLM Requirement |
|------------|----------------|-----------------|
| Query decomposition | Partial (TaskClassifier) | Full recursive planning |
| Sub-model orchestration | Single model routing | Multi-model coordination |
| Recursive depth control | None | Max depth + memoization |
| Answer synthesis | Basic merging | LLM-driven composition |
| Reflection loops | ConfidenceChecker | Multi-pass refinement |
| Token budgeting | KV cache management | Cross-call budget tracking |
| External tool calls | MCP integration | Environment trait abstraction |

---

## Decision Drivers

### Performance Requirements
- **Orchestration latency**: <5ms per recursive step
- **Memory lookup**: <2ms HNSW retrieval per query
- **Token budget tracking**: Real-time across recursive calls
- **Cache hit ratio**: >80% for repeated sub-queries

### Scalability Requirements
- **Recursion depth**: Support 1-10 levels with configurable limits
- **Concurrent sessions**: 1000+ with isolated recursion stacks
- **Sub-query parallelism**: Fan-out up to 4 concurrent sub-queries
- **Pattern capacity**: 1M+ indexed patterns in ruvector

### Portability Requirements
- **WASM support**: Full RLM logic runnable in browser
- **Cross-platform**: x86_64, ARM64, WASM32 targets
- **Minimal dependencies**: Core logic in pure Rust

---

## Considered Options

### Option A: External RLM Orchestrator

Build a separate `ruvrlm` crate that wraps `ruvllm` and `ruvector`:

```
User Query --> ruvrlm (orchestrator) --> ruvllm (inference)
                     |                        ^
                     +--> ruvector (memory) --+
```

**Pros:**
- Clean separation of concerns
- Independent versioning
- Minimal changes to existing crates

**Cons:**
- Additional dependency to manage
- Cross-crate coordination overhead
- Duplicate type definitions

### Option B: Integrated RLM Module in ruvllm

Extend `ruvllm` with new modules implementing RLM traits:

```
ruvllm/
├── src/
│   ├── rlm/              # NEW: RLM core
│   │   ├── mod.rs
│   │   ├── environment.rs    # RlmEnv trait
│   │   ├── controller.rs     # Recursive controller
│   │   ├── decomposer.rs     # Query decomposition
│   │   ├── synthesizer.rs    # Answer composition
│   │   └── reflection.rs     # Multi-pass refinement
│   ├── backends/
│   │   └── mistral_backend.rs  # NEW: mistral.rs integration
│   ├── ruvector_integration.rs # Enhanced for RLM
│   └── ...
```

**Pros:**
- Leverages existing infrastructure (SONA, ReasoningBank, HnswRouter)
- Single versioning and release cycle
- Type sharing without duplication
- Natural integration with Claude Flow agents

**Cons:**
- Increases crate size
- Risk of coupling concerns

### Option C: Trait-Based Abstraction Layer

Define RLM traits in a lightweight `ruvllm-traits` crate, implement in `ruvllm`:

```
ruvllm-traits/           # Minimal trait definitions
ruvllm/                  # Full implementation
@ruvector/ruvllm/        # npm bindings
```

**Pros:**
- Maximum flexibility for alternative implementations
- Minimal core dependency
- Enables ecosystem growth

**Cons:**
- Multiple crates to maintain
- Indirection overhead

---

## Decision Outcome

**Chosen Option: Option B - Integrated RLM Module in ruvllm**

The RLM system will be implemented as a new `rlm` module within the existing `ruvllm` crate, leveraging:
- `ReasoningBank` for trajectory storage and pattern retrieval
- `HnswRouter` for semantic similarity in query decomposition
- `ClaudeFlowAgent` coordination for multi-model routing
- `QualityScoringEngine` for answer validation
- `ReflectiveAgent` for multi-pass refinement

### Rationale

1. **Reuse**: 80%+ of RLM requirements are already implemented (routing, memory, reflection)
2. **Cohesion**: RLM is the natural evolution of Claude Flow agent orchestration
3. **Performance**: In-crate integration avoids serialization overhead
4. **Developer Experience**: Single import for full RLM capabilities

---

## Technical Specifications

### Core Traits

#### LlmBackend Trait

```rust
/// Backend abstraction for any LLM engine (mistral.rs, candle, API)
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Backend identifier
    fn id(&self) -> &str;

    /// Model information
    fn model_info(&self) -> &ModelInfo;

    /// Maximum context length in tokens
    fn max_context(&self) -> usize;

    /// Estimate token count for text
    fn estimate_tokens(&self, text: &str) -> usize;

    /// Generate completion
    async fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<GenerationOutput>;

    /// Generate with streaming
    async fn generate_stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<impl Stream<Item = StreamToken>>;

    /// Embed text to vector (optional, for self-embedding models)
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Err(RlmError::EmbeddingNotSupported)
    }
}

/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_sequences: Vec<String>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

/// Generation output
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub tokens_used: usize,
    pub finish_reason: FinishReason,
    pub logprobs: Option<Vec<f32>>,
}
```

#### RlmEnvironment Trait

```rust
/// RLM Environment - the sandbox in which recursive reasoning operates
pub trait RlmEnvironment: Send + Sync {
    /// Associated backend type
    type Backend: LlmBackend;

    /// Associated memory store type
    type Memory: MemoryStore;

    /// Get the LLM backend
    fn backend(&self) -> &Self::Backend;

    /// Get the memory store (ruvector)
    fn memory(&self) -> &Self::Memory;

    /// Retrieve relevant context for a query
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<MemorySpan>>;

    /// Store a new memory span
    async fn store_memory(
        &self,
        text: &str,
        metadata: MemoryMetadata,
    ) -> Result<MemoryId>;

    /// Decompose a complex query into sub-queries
    async fn decompose_query(
        &self,
        query: &str,
        context: &QueryContext,
    ) -> Result<QueryDecomposition>;

    /// Synthesize partial answers into a coherent response
    async fn synthesize_answers(
        &self,
        original_query: &str,
        sub_answers: &[SubAnswer],
    ) -> Result<String>;

    /// Main recursive query answering
    async fn answer_query(
        &self,
        query: &str,
        depth: usize,
        config: &RlmConfig,
    ) -> Result<RlmAnswer>;
}

/// Memory span retrieved from ruvector
#[derive(Debug, Clone)]
pub struct MemorySpan {
    pub id: MemoryId,
    pub text: String,
    pub embedding: Vec<f32>,
    pub similarity_score: f32,
    pub source: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Query decomposition result
#[derive(Debug, Clone)]
pub struct QueryDecomposition {
    pub strategy: DecompositionStrategy,
    pub sub_queries: Vec<SubQuery>,
    pub dependencies: Vec<(usize, usize)>,  // (from, to) dependency edges
}

#[derive(Debug, Clone)]
pub enum DecompositionStrategy {
    /// No decomposition needed - answer directly
    Direct,
    /// Split by conjunction ("and", "or")
    Conjunction(Vec<String>),
    /// Split by aspect (what, why, how)
    Aspect(Vec<String>),
    /// Sequential multi-step reasoning
    Sequential(Vec<String>),
    /// Parallel independent sub-questions
    Parallel(Vec<String>),
}
```

### RLM Controller

```rust
/// Recursive Language Model Controller
pub struct RlmController<E: RlmEnvironment> {
    /// The RLM environment
    env: E,

    /// Configuration
    config: RlmConfig,

    /// Query cache for memoization
    cache: DashMap<String, CachedAnswer>,

    /// Active recursion depth tracking
    depth_tracker: AtomicUsize,

    /// Token budget tracker
    budget_tracker: TokenBudgetTracker,

    /// ReasoningBank for trajectory recording
    reasoning_bank: ReasoningBank,

    /// Quality scorer for answer validation
    quality_scorer: QualityScoringEngine,
}

/// RLM Configuration
#[derive(Debug, Clone)]
pub struct RlmConfig {
    /// Maximum recursion depth (default: 5)
    pub max_depth: usize,

    /// Maximum sub-queries per level (default: 4)
    pub max_sub_queries: usize,

    /// Token budget for entire query chain
    pub token_budget: usize,

    /// Enable memoization cache
    pub enable_cache: bool,

    /// Cache TTL in seconds
    pub cache_ttl: u64,

    /// Number of context chunks to retrieve
    pub retrieval_top_k: usize,

    /// Minimum quality score to accept answer
    pub min_quality_score: f32,

    /// Enable reflection loops
    pub enable_reflection: bool,

    /// Maximum reflection iterations
    pub max_reflection_iterations: usize,

    /// Parallelism for independent sub-queries
    pub parallel_sub_queries: bool,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_sub_queries: 4,
            token_budget: 16000,
            enable_cache: true,
            cache_ttl: 3600,
            retrieval_top_k: 5,
            min_quality_score: 0.7,
            enable_reflection: true,
            max_reflection_iterations: 2,
            parallel_sub_queries: true,
        }
    }
}

impl<E: RlmEnvironment> RlmController<E> {
    /// Process a query through the RLM pipeline
    pub async fn process(&self, query: &str) -> Result<RlmAnswer> {
        // Record trajectory start
        let trajectory_id = self.reasoning_bank.start_trajectory(query);

        // Check cache first
        if self.config.enable_cache {
            if let Some(cached) = self.cache.get(query) {
                if !cached.is_expired() {
                    self.reasoning_bank.record_cache_hit(trajectory_id);
                    return Ok(cached.answer.clone());
                }
            }
        }

        // Execute recursive answering
        let answer = self.env.answer_query(query, 0, &self.config).await?;

        // Validate quality
        let quality = self.quality_scorer.score(&answer)?;
        if quality.overall < self.config.min_quality_score {
            // Trigger reflection
            if self.config.enable_reflection {
                return self.reflect_and_improve(query, &answer, quality).await;
            }
        }

        // Cache successful answer
        if self.config.enable_cache {
            self.cache.insert(query.to_string(), CachedAnswer::new(answer.clone()));
        }

        // Record trajectory completion
        self.reasoning_bank.complete_trajectory(trajectory_id, &answer)?;

        Ok(answer)
    }

    /// Reflection loop for improving low-quality answers
    async fn reflect_and_improve(
        &self,
        query: &str,
        initial_answer: &RlmAnswer,
        quality: QualityMetrics,
    ) -> Result<RlmAnswer> {
        let mut current = initial_answer.clone();
        let mut current_quality = quality;

        for iteration in 0..self.config.max_reflection_iterations {
            // Generate critique
            let critique = self.generate_critique(&current, &current_quality).await?;

            // Attempt improvement
            let improved = self.improve_answer(query, &current, &critique).await?;
            let new_quality = self.quality_scorer.score(&improved)?;

            if new_quality.overall >= self.config.min_quality_score {
                return Ok(improved);
            }

            if new_quality.overall > current_quality.overall {
                current = improved;
                current_quality = new_quality;
            }
        }

        // Return best attempt
        Ok(current)
    }
}
```

### Architecture Diagram

```
+============================================================================+
|                     RECURSIVE LANGUAGE MODEL ARCHITECTURE                   |
+============================================================================+
|                                                                             |
|   User Query                                                                |
|       |                                                                     |
|       v                                                                     |
|   +-------------------+                                                     |
|   | RLM CONTROLLER    |                                                     |
|   | - Cache check     |                                                     |
|   | - Budget tracking |                                                     |
|   | - Trajectory rec  |                                                     |
|   +--------+----------+                                                     |
|            |                                                                |
|            v                                                                |
|   +-------------------+     +-------------------+     +-----------------+   |
|   | RLM ENVIRONMENT   |<--->| RUVECTOR MEMORY   |<--->| HNSW INDEX     |   |
|   |                   |     | - Semantic search |     | - 150x faster  |   |
|   | - retrieve()      |     | - Pattern store   |     | - Graph edges  |   |
|   | - decompose()     |     | - Session state   |     |                |   |
|   | - synthesize()    |     +-------------------+     +-----------------+   |
|   | - answer_query()  |                                                     |
|   +--------+----------+                                                     |
|            |                                                                |
|   +--------+------------------------------------------+                     |
|   |        |                                          |                     |
|   |        v  (if complex)                            v  (if simple)        |
|   |   +----+----+                               +-----+-----+               |
|   |   | DECOMP  |                               | DIRECT    |               |
|   |   | OSER    |                               | ANSWER    |               |
|   |   +---------+                               +-----------+               |
|   |        |                                                                |
|   |        v                                                                |
|   |   +---------+  +---------+  +---------+                                |
|   |   |SubQuery1|  |SubQuery2|  |SubQuery3|  (parallel or sequential)      |
|   |   +----+----+  +----+----+  +----+----+                                |
|   |        |            |            |                                      |
|   |        v            v            v                                      |
|   |   +----+------------+------------+----+                                |
|   |   |         RECURSIVE CALLS           |  (depth + 1)                   |
|   |   |     answer_query(sub, depth+1)    |                                |
|   |   +-----------------------------------+                                 |
|   |                    |                                                    |
|   |                    v                                                    |
|   |   +----------------+----------------+                                   |
|   |   |         SYNTHESIZER             |                                   |
|   |   | - Merge sub-answers             |                                   |
|   |   | - LLM-driven composition        |                                   |
|   |   | - Coherence check               |                                   |
|   |   +---------------------------------+                                   |
|   |                    |                                                    |
|   +--------------------+                                                    |
|                        |                                                    |
|                        v                                                    |
|   +--------------------+--------------------+                               |
|   |          QUALITY SCORING                |                               |
|   | - Coherence validation                  |                               |
|   | - Factual grounding check               |                               |
|   | - Completeness assessment               |                               |
|   +-------------------+---------------------+                               |
|                       |                                                     |
|          +------------+------------+                                        |
|          |                         |                                        |
|          v  (quality < threshold)  v  (quality >= threshold)               |
|   +------+------+           +------+------+                                |
|   | REFLECTION  |           | FINAL       |                                |
|   | LOOP        |           | ANSWER      |                                |
|   | - Critique  |           +-------------+                                |
|   | - Improve   |                                                          |
|   +-------------+                                                          |
|                                                                             |
+============================================================================+
|                           LLM BACKEND LAYER                                 |
+============================================================================+
|                                                                             |
|   +-------------------+     +-------------------+     +-----------------+   |
|   | MISTRAL.RS        |     | CANDLE BACKEND    |     | API BACKEND    |   |
|   | - GGUF loading    |     | - Rust-native     |     | - Claude API   |   |
|   | - Quantization    |     | - Metal/CUDA      |     | - OpenAI API   |   |
|   | - Paged attention |     | - WASM support    |     | - Fallback     |   |
|   | - FlashAttention  |     |                   |     |                |   |
|   +-------------------+     +-------------------+     +-----------------+   |
|                                                                             |
+============================================================================+
```

### npm Package Integration

```typescript
// npm/packages/ruvllm/src/rlm/index.ts

/**
 * Recursive Language Model for unbounded context reasoning
 */
export interface RlmConfig {
  /** Maximum recursion depth (default: 5) */
  maxDepth?: number;
  /** Token budget for entire chain */
  tokenBudget?: number;
  /** Enable query caching */
  enableCache?: boolean;
  /** Retrieval top-k for memory */
  retrievalTopK?: number;
  /** Enable reflection loops */
  enableReflection?: boolean;
}

/**
 * RLM Answer with provenance
 */
export interface RlmAnswer {
  text: string;
  confidence: number;
  sources: MemorySpan[];
  subQueries?: SubQuery[];
  qualityScore: number;
  tokenUsage: TokenUsage;
  trajectory: TrajectoryId;
}

/**
 * Recursive Language Model Controller
 */
export class RlmController {
  constructor(config?: RlmConfig);

  /**
   * Process a query with recursive decomposition
   */
  async query(input: string): Promise<RlmAnswer>;

  /**
   * Stream response tokens
   */
  async queryStream(input: string): AsyncIterable<StreamToken>;

  /**
   * Add memory to the knowledge base
   */
  async addMemory(text: string, metadata?: Record<string, unknown>): Promise<MemoryId>;

  /**
   * Search memory semantically
   */
  async searchMemory(query: string, topK?: number): Promise<MemorySpan[]>;
}

/**
 * RLM Environment for custom integrations
 */
export abstract class RlmEnvironment {
  abstract retrieve(query: string, topK: number): Promise<MemorySpan[]>;
  abstract decompose(query: string): Promise<QueryDecomposition>;
  abstract synthesize(query: string, subAnswers: SubAnswer[]): Promise<string>;
  abstract generate(prompt: string, params: GenerationParams): Promise<GenerationOutput>;
}
```

### Multi-Platform Build Matrix

```toml
# Cargo.toml additions for RLM

[features]
# RLM core (always included)
rlm-core = []

# RLM with mistral.rs backend
rlm-mistral = ["rlm-core", "mistralrs", "mistralrs-core"]

# RLM with candle backend (default)
rlm-candle = ["rlm-core", "candle"]

# RLM optimized for Metal (macOS)
rlm-metal = ["rlm-candle", "metal"]

# RLM optimized for CUDA (NVIDIA)
rlm-cuda = ["rlm-candle", "cuda"]

# RLM for WASM (browser/edge)
rlm-wasm = ["rlm-core", "wasm"]

# Full RLM with all backends
rlm-full = ["rlm-candle", "rlm-mistral"]
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Orchestration per step | <5ms | Cache hit path |
| Memory retrieval | <2ms | HNSW top-5 |
| Token estimation | <0.1ms | Tokenizer lookup |
| Sub-query fanout | 4 parallel | Configurable |
| Max recursion depth | 10 levels | Default: 5 |
| Cache hit ratio | >80% | Session-scoped |
| Quality threshold | 0.7 | Triggers reflection |

---

## Domain-Driven Design

See **DDD-001-recursive-language-model.md** for bounded context definitions:

- **Orchestration Context**: Query lifecycle, recursion control, caching
- **Memory Context**: Retrieval, storage, semantic search
- **Inference Context**: LLM backends, generation, token budgets
- **Quality Context**: Scoring, reflection, validation
- **Learning Context**: SONA integration, trajectory recording

---

## Consequences

### Positive Consequences

1. **Unbounded context**: Handle arbitrarily long queries via decomposition
2. **Grounded generation**: Every answer backed by retrieved facts
3. **Self-improving**: SONA learns from successful query patterns
4. **Multi-backend**: Support local (mistral.rs) and API (Claude) inference
5. **Portable**: Full RLM runs in browser via WASM

### Negative Consequences

1. **Latency**: Recursive calls add overhead vs. single-shot
2. **Complexity**: More failure modes to handle
3. **Token cost**: Multiple LLM calls per query

### Mitigation Strategies

| Risk | Mitigation |
|------|------------|
| Latency | Aggressive caching, parallel sub-queries |
| Complexity | Comprehensive error types, observability |
| Token cost | Budget tracking, early termination |
| Infinite recursion | Hard depth limit, cycle detection |

---

## Implementation Roadmap

| Phase | Components | Timeline |
|-------|------------|----------|
| 1. Core Traits | `LlmBackend`, `RlmEnvironment` traits | Week 1 |
| 2. Controller | `RlmController` with caching | Week 2 |
| 3. Decomposition | Query analysis and splitting | Week 3 |
| 4. Synthesis | Answer merging with quality | Week 4 |
| 5. Reflection | Multi-pass improvement loop | Week 5 |
| 6. npm Bindings | TypeScript API and streaming | Week 6 |
| 7. Benchmarks | Performance validation suite | Week 7 |

---

## Related Decisions

- **ADR-001**: Ruvector Core Architecture (HNSW, Graph Store)
- **ADR-002**: RuvLLM Integration with Ruvector
- **ADR-008**: Mistral.rs Integration
- **DDD-001**: RLM Domain-Driven Design (this ADR's companion)

---

## References

1. Zhang, A. et al. "Recursive Language Models." arXiv:2512.24601 (2025)
2. RuvLLM Architecture: `/crates/ruvllm/src/lib.rs`
3. mistral.rs: https://github.com/EricLBuehler/mistral.rs
4. Ruvector: https://github.com/ruvnet/ruvector
5. "Reflexion" agent pattern for multi-pass self-refinement

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-21 | Ruvector Architecture Team | Initial proposal |
