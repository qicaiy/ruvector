# DDD-001: Recursive Language Model Domain Design

**Status:** Proposed
**Date:** 2026-01-21
**Technical Area:** Domain-Driven Design / RLM Architecture
**Related ADR:** ADR-014-recursive-language-model-integration

---

## Executive Summary

This document defines the Domain-Driven Design (DDD) architecture for integrating Recursive Language Models (RLM) into the ruvllm ecosystem. The design identifies six bounded contexts, their aggregate roots, domain events, and integration patterns with existing components including RuvLTRA, SONA, and ReasoningBank.

---

## Strategic Design

### Domain Vision Statement

> **Enable unbounded context reasoning through recursive task decomposition, semantic memory retrieval, and continuous self-improvement, while maintaining sub-5ms orchestration latency and cross-platform portability.**

### Problem Space Analysis

```
+===========================================================================+
|                        RLM PROBLEM SPACE                                   |
+===========================================================================+
|                                                                            |
|  CORE SUBDOMAIN: Recursive Reasoning                                       |
|  +---------------------------------------------------------------------+  |
|  | - Query decomposition strategies                                     |  |
|  | - Recursive depth management                                         |  |
|  | - Answer synthesis and merging                                       |  |
|  | - Memoization and caching                                            |  |
|  +---------------------------------------------------------------------+  |
|                                                                            |
|  SUPPORTING SUBDOMAIN: Semantic Memory                                     |
|  +---------------------------------------------------------------------+  |
|  | - Vector storage and retrieval (ruvector)                            |  |
|  | - Pattern matching via HNSW                                          |  |
|  | - Session state persistence                                          |  |
|  | - Knowledge graph relationships                                      |  |
|  +---------------------------------------------------------------------+  |
|                                                                            |
|  SUPPORTING SUBDOMAIN: LLM Inference                                       |
|  +---------------------------------------------------------------------+  |
|  | - RuvLTRA model execution (Qwen 0.5B, ANE-optimized)                  |  |
|  | - Backend abstraction (Candle, mistral.rs, APIs)                     |  |
|  | - Token budget management                                            |  |
|  | - KV cache lifecycle                                                 |  |
|  +---------------------------------------------------------------------+  |
|                                                                            |
|  SUPPORTING SUBDOMAIN: Quality Assurance                                   |
|  +---------------------------------------------------------------------+  |
|  | - Answer validation and scoring                                      |  |
|  | - Multi-pass reflection loops                                        |  |
|  | - Coherence and factual grounding                                    |  |
|  +---------------------------------------------------------------------+  |
|                                                                            |
|  GENERIC SUBDOMAIN: Continuous Learning                                    |
|  +---------------------------------------------------------------------+  |
|  | - SONA three-tier learning (instant/hourly/weekly)                   |  |
|  | - Trajectory recording (ReasoningBank)                               |  |
|  | - EWC++ catastrophic forgetting prevention                           |  |
|  | - MicroLoRA adaptation                                               |  |
|  +---------------------------------------------------------------------+  |
|                                                                            |
|  GENERIC SUBDOMAIN: Observability                                          |
|  +---------------------------------------------------------------------+  |
|  | - Witness logging and audit trails                                   |  |
|  | - Performance metrics collection                                     |  |
|  | - Cost tracking (API calls, tokens)                                  |  |
|  +---------------------------------------------------------------------+  |
|                                                                            |
+===========================================================================+
```

---

## Bounded Contexts

### Context Map

```
+===========================================================================+
|                          CONTEXT MAP                                       |
+===========================================================================+
|                                                                            |
|   +---------------------+        ACL          +---------------------+     |
|   | ORCHESTRATION       |<------------------->| MEMORY              |     |
|   | CONTEXT             |                     | CONTEXT             |     |
|   | (RlmController)     |                     | (ruvector)          |     |
|   +----------+----------+                     +----------+----------+     |
|              |                                           |                 |
|              | U                                         | U               |
|              |                                           |                 |
|   +----------v----------+        OHS          +----------v----------+     |
|   | INFERENCE           |<------------------->| LEARNING            |     |
|   | CONTEXT             |                     | CONTEXT             |     |
|   | (RuvLTRA, backends) |                     | (SONA, Reasoning)   |     |
|   +----------+----------+                     +---------------------+     |
|              |                                                             |
|              | PL                                                          |
|              |                                                             |
|   +----------v----------+                     +---------------------+     |
|   | QUALITY             |        CF           | OBSERVABILITY       |     |
|   | CONTEXT             |<------------------->| CONTEXT             |     |
|   | (scoring, reflect)  |                     | (witness, metrics)  |     |
|   +---------------------+                     +---------------------+     |
|                                                                            |
|   Legend: U=Upstream, ACL=Anticorruption Layer, OHS=Open Host Service,    |
|           PL=Published Language, CF=Conformist                            |
|                                                                            |
+===========================================================================+
```

---

## Bounded Context 1: Orchestration Context

### Purpose
Manages the recursive query lifecycle, including decomposition, synthesis, and caching.

### Aggregate Roots

#### QuerySession Aggregate

```rust
/// Query session - the root aggregate for RLM orchestration
pub struct QuerySession {
    // Identity
    id: SessionId,
    user_id: Option<UserId>,

    // State
    state: SessionState,
    created_at: DateTime<Utc>,
    last_active: DateTime<Utc>,

    // Query stack (for recursion tracking)
    query_stack: Vec<ActiveQuery>,
    max_depth: usize,

    // Cache
    memoization_cache: HashMap<QueryHash, CachedAnswer>,

    // Budget
    token_budget: TokenBudget,
    tokens_consumed: usize,
}

impl QuerySession {
    /// Submit a top-level query
    pub fn submit_query(&mut self, query: Query) -> Result<QueryId, DomainError> {
        self.validate_can_submit()?;
        let query_id = QueryId::generate();
        let active = ActiveQuery::new(query_id, query, 0);
        self.query_stack.push(active);
        self.domain_events.push(QuerySubmitted {
            session_id: self.id,
            query_id,
            depth: 0,
        });
        Ok(query_id)
    }

    /// Push a sub-query onto the recursion stack
    pub fn push_subquery(&mut self, parent_id: QueryId, subquery: Query) -> Result<QueryId, DomainError> {
        let current_depth = self.current_depth();
        if current_depth >= self.max_depth {
            return Err(DomainError::MaxRecursionDepthExceeded);
        }

        let query_id = QueryId::generate();
        let active = ActiveQuery::new(query_id, subquery, current_depth + 1)
            .with_parent(parent_id);
        self.query_stack.push(active);

        self.domain_events.push(SubQueryPushed {
            session_id: self.id,
            parent_id,
            query_id,
            depth: current_depth + 1,
        });
        Ok(query_id)
    }

    /// Complete a query with an answer
    pub fn complete_query(&mut self, query_id: QueryId, answer: Answer) -> Result<(), DomainError> {
        let idx = self.find_query_index(query_id)?;
        let completed = self.query_stack.remove(idx);

        // Cache the answer
        let hash = QueryHash::from(&completed.query);
        self.memoization_cache.insert(hash, CachedAnswer::new(answer.clone()));

        // Update token consumption
        self.tokens_consumed += answer.tokens_used;

        self.domain_events.push(QueryCompleted {
            session_id: self.id,
            query_id,
            answer_quality: answer.quality_score,
            tokens_used: answer.tokens_used,
        });

        Ok(())
    }

    /// Check cache for existing answer
    pub fn check_cache(&self, query: &Query) -> Option<&CachedAnswer> {
        let hash = QueryHash::from(query);
        self.memoization_cache.get(&hash)
            .filter(|c| !c.is_expired())
    }
}

/// Active query in the recursion stack
pub struct ActiveQuery {
    id: QueryId,
    query: Query,
    depth: usize,
    parent_id: Option<QueryId>,
    started_at: DateTime<Utc>,
    sub_answers: Vec<SubAnswer>,
}
```

#### Query Value Objects

```rust
/// Immutable query value object
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Query {
    text: String,
    context: QueryContext,
    constraints: QueryConstraints,
}

impl Query {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            context: QueryContext::default(),
            constraints: QueryConstraints::default(),
        }
    }

    pub fn with_context(mut self, context: QueryContext) -> Self {
        self.context = context;
        self
    }

    /// Check if query is decomposable
    pub fn is_complex(&self) -> bool {
        // Heuristics for complexity
        self.text.contains(" and ")
            || self.text.contains(" or ")
            || self.text.len() > 200
            || self.text.matches('?').count() > 1
    }
}

/// Query context for grounding
pub struct QueryContext {
    /// Session history (previous Q&A)
    history: Vec<ConversationTurn>,
    /// Retrieved memory spans
    retrieved_spans: Vec<MemorySpanRef>,
    /// User preferences
    preferences: UserPreferences,
}

/// Query constraints
pub struct QueryConstraints {
    /// Maximum response length
    max_tokens: Option<usize>,
    /// Required sources
    required_sources: Vec<SourceId>,
    /// Output format
    format: OutputFormat,
}
```

### Domain Events

| Event | Payload | Consumers |
|-------|---------|-----------|
| `QuerySubmitted` | session_id, query_id, depth | Learning, Observability |
| `SubQueryPushed` | session_id, parent_id, query_id, depth | Learning |
| `QueryCompleted` | session_id, query_id, quality, tokens | Learning, Observability |
| `QueryDecomposed` | query_id, strategy, sub_query_ids | Learning |
| `AnswerSynthesized` | query_id, sub_answer_ids, final_quality | Quality |
| `CacheHit` | session_id, query_hash | Observability |
| `BudgetExhausted` | session_id, limit, consumed | Observability |

### Repository Interface

```rust
#[async_trait]
pub trait QuerySessionRepository {
    async fn save(&self, session: &QuerySession) -> Result<()>;
    async fn find_by_id(&self, id: &SessionId) -> Result<Option<QuerySession>>;
    async fn find_by_user(&self, user_id: &UserId, limit: usize) -> Result<Vec<QuerySession>>;
    async fn delete(&self, id: &SessionId) -> Result<bool>;
}
```

---

## Bounded Context 2: Memory Context

### Purpose
Provides semantic search and storage via ruvector integration.

### Aggregate Roots

#### MemoryStore Aggregate

```rust
/// Memory store managing vector indices
pub struct MemoryStore {
    id: StoreId,
    config: MemoryStoreConfig,

    // Indices (ruvector-backed)
    pattern_index: HnswIndex,
    session_index: HnswIndex,
    witness_index: HnswIndex,

    // Statistics
    total_entries: usize,
    last_compaction: Option<DateTime<Utc>>,
}

impl MemoryStore {
    /// Store a new memory span with embedding
    pub fn store(&mut self, span: MemorySpan) -> Result<MemoryId, DomainError> {
        self.validate_embedding(&span.embedding)?;

        let id = MemoryId::generate();
        self.pattern_index.insert(id, &span.embedding, span.metadata.clone())?;
        self.total_entries += 1;

        self.domain_events.push(MemoryStored {
            store_id: self.id,
            memory_id: id,
            namespace: span.namespace,
        });

        Ok(id)
    }

    /// Retrieve similar spans
    pub fn retrieve(&self, query_embedding: &[f32], top_k: usize) -> Vec<MemorySpan> {
        self.pattern_index
            .search(query_embedding, top_k)
            .into_iter()
            .map(|result| MemorySpan {
                id: result.id,
                text: result.metadata.get("text").cloned().unwrap_or_default(),
                embedding: result.vector,
                similarity_score: result.score,
                source: result.metadata.get("source").cloned(),
                metadata: result.metadata,
            })
            .collect()
    }

    /// Compact and optimize indices
    pub fn compact(&mut self) -> Result<CompactionResult, DomainError> {
        let before = self.total_entries;
        // Trigger HNSW graph optimization
        self.pattern_index.optimize()?;

        self.last_compaction = Some(Utc::now());
        self.domain_events.push(StoreCompacted {
            store_id: self.id,
            entries_before: before,
            entries_after: self.total_entries,
        });

        Ok(CompactionResult {
            entries_removed: before - self.total_entries,
            duration: Duration::from_secs(1), // Placeholder
        })
    }
}

/// Memory span value object
#[derive(Clone)]
pub struct MemorySpan {
    pub id: MemoryId,
    pub text: String,
    pub embedding: Vec<f32>,
    pub similarity_score: f32,
    pub source: Option<String>,
    pub namespace: Namespace,
    pub metadata: HashMap<String, String>,
}

/// Namespaces for organizing memory
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Namespace {
    /// Learned patterns from successful tasks
    Patterns,
    /// Session conversation history
    Sessions,
    /// Audit/witness entries
    Witness,
    /// User-provided knowledge base
    Knowledge,
    /// Temporary working memory
    Scratch,
}
```

### Domain Services

```rust
/// Embedding service (anti-corruption layer to external models)
pub trait EmbeddingService: Send + Sync {
    /// Generate embedding for text
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Batch embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Embedding dimension
    fn dimension(&self) -> usize;
}

/// RuvLTRA-based embedding service
pub struct RuvLtraEmbeddingService {
    model: Arc<RuvLtraModel>,
    tokenizer: Arc<RuvTokenizer>,
    dimension: usize,
}

impl EmbeddingService for RuvLtraEmbeddingService {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Use RuvLTRA's hidden states as embeddings
        let tokens = self.tokenizer.encode(text)?;
        let hidden_states = self.model.forward(&tokens, &self.positions(&tokens), None)?;

        // Mean pooling over sequence
        let embedding = self.mean_pool(&hidden_states, self.dimension);
        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dimension // 896 for RuvLTRA Qwen 0.5B
    }
}
```

---

## Bounded Context 3: Inference Context

### Purpose
Executes LLM generation via RuvLTRA and other backends.

### Aggregate Roots

#### InferenceEngine Aggregate

```rust
/// Inference engine managing LLM backends
pub struct InferenceEngine {
    id: EngineId,

    // Primary backend (RuvLTRA)
    primary_backend: Box<dyn LlmBackend>,

    // Fallback backends
    fallbacks: Vec<Box<dyn LlmBackend>>,

    // Active generations
    active_generations: HashMap<GenerationId, ActiveGeneration>,

    // Configuration
    config: InferenceConfig,

    // Statistics
    stats: InferenceStats,
}

impl InferenceEngine {
    /// Create with RuvLTRA as primary backend
    pub fn with_ruvltra(config: RuvLtraConfig) -> Result<Self, DomainError> {
        let model = RuvLtraModel::new(&config)?;
        let backend = RuvLtraBackend::new(model);

        Ok(Self {
            id: EngineId::generate(),
            primary_backend: Box::new(backend),
            fallbacks: vec![],
            active_generations: HashMap::new(),
            config: InferenceConfig::default(),
            stats: InferenceStats::default(),
        })
    }

    /// Generate completion
    pub async fn generate(&mut self, request: GenerationRequest) -> Result<Generation, DomainError> {
        let gen_id = GenerationId::generate();

        // Check token budget
        let estimated_tokens = self.estimate_tokens(&request.prompt);
        if estimated_tokens > request.max_tokens {
            return Err(DomainError::TokenBudgetExceeded {
                estimated: estimated_tokens,
                budget: request.max_tokens,
            });
        }

        // Track active generation
        self.active_generations.insert(gen_id, ActiveGeneration::new(&request));

        // Attempt primary backend
        let result = self.primary_backend.generate(&request.prompt, &request.params).await;

        let generation = match result {
            Ok(output) => Generation {
                id: gen_id,
                text: output.text,
                tokens_used: output.tokens_used,
                finish_reason: output.finish_reason,
                backend: self.primary_backend.id().to_string(),
                latency: output.latency,
            },
            Err(e) => {
                // Try fallbacks
                for fallback in &self.fallbacks {
                    if let Ok(output) = fallback.generate(&request.prompt, &request.params).await {
                        return Ok(Generation {
                            id: gen_id,
                            text: output.text,
                            tokens_used: output.tokens_used,
                            finish_reason: output.finish_reason,
                            backend: fallback.id().to_string(),
                            latency: output.latency,
                        });
                    }
                }
                return Err(DomainError::GenerationFailed(e.to_string()));
            }
        };

        // Update stats
        self.stats.total_generations += 1;
        self.stats.total_tokens += generation.tokens_used;

        // Remove from active
        self.active_generations.remove(&gen_id);

        self.domain_events.push(GenerationCompleted {
            engine_id: self.id,
            generation_id: gen_id,
            tokens: generation.tokens_used,
            latency_ms: generation.latency.as_millis() as u64,
        });

        Ok(generation)
    }
}

/// RuvLTRA backend implementation
pub struct RuvLtraBackend {
    model: Arc<RwLock<RuvLtraModel>>,
    tokenizer: Arc<RuvTokenizer>,
    kv_caches: HashMap<SessionId, KvCache>,
    config: RuvLtraConfig,
}

impl LlmBackend for RuvLtraBackend {
    fn id(&self) -> &str {
        "ruvltra"
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model.read().info().into()
    }

    fn max_context(&self) -> usize {
        self.config.max_position_embeddings
    }

    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<GenerationOutput> {
        let start = std::time::Instant::now();

        // Tokenize
        let input_ids = self.tokenizer.encode(prompt)?;
        let positions: Vec<usize> = (0..input_ids.len()).collect();

        // Generate tokens auto-regressively
        let mut generated = Vec::new();
        let mut kv_cache = Vec::new();

        for _ in 0..params.max_tokens {
            let logits = self.model.read().forward(&input_ids, &positions, Some(&mut kv_cache))?;

            // Sample next token
            let next_token = self.sample(&logits, params)?;

            if next_token == self.tokenizer.eos_token_id() {
                break;
            }

            generated.push(next_token);
        }

        // Decode output
        let text = self.tokenizer.decode(&generated)?;

        Ok(GenerationOutput {
            text,
            tokens_used: generated.len(),
            finish_reason: FinishReason::Stop,
            latency: start.elapsed(),
            logprobs: None,
        })
    }
}
```

### Value Objects

```rust
/// Generation request (immutable)
#[derive(Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub params: GenerationParams,
    pub max_tokens: usize,
    pub session_id: Option<SessionId>,
    pub request_id: RequestId,
}

/// Generation parameters
#[derive(Clone)]
pub struct GenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub stop_sequences: Vec<String>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            stop_sequences: vec![],
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}
```

---

## Bounded Context 4: Quality Context

### Purpose
Validates and improves answer quality through scoring and reflection.

### Aggregate Roots

#### QualityAssessment Aggregate

```rust
/// Quality assessment for an answer
pub struct QualityAssessment {
    id: AssessmentId,
    answer_id: AnswerId,

    // Scores
    overall_score: f32,
    dimension_scores: DimensionScores,

    // Issues found
    issues: Vec<QualityIssue>,

    // Reflection history
    reflection_attempts: Vec<ReflectionAttempt>,

    // Metadata
    created_at: DateTime<Utc>,
    assessor: Assessor,
}

impl QualityAssessment {
    /// Assess an answer
    pub fn assess(answer: &Answer, context: &AssessmentContext) -> Self {
        let mut assessment = Self::new(answer.id);

        // Score each dimension
        assessment.dimension_scores = DimensionScores {
            coherence: Self::score_coherence(answer),
            completeness: Self::score_completeness(answer, context),
            factual_grounding: Self::score_grounding(answer, &context.retrieved_spans),
            relevance: Self::score_relevance(answer, &context.original_query),
            consistency: Self::score_consistency(answer, &context.sub_answers),
        };

        // Calculate overall
        assessment.overall_score = assessment.dimension_scores.weighted_average();

        // Identify issues
        assessment.issues = Self::identify_issues(&assessment.dimension_scores, answer);

        assessment.domain_events.push(AnswerAssessed {
            assessment_id: assessment.id,
            answer_id: answer.id,
            overall_score: assessment.overall_score,
            issues_count: assessment.issues.len(),
        });

        assessment
    }

    /// Record a reflection attempt
    pub fn record_reflection(&mut self, attempt: ReflectionAttempt) {
        self.reflection_attempts.push(attempt.clone());

        self.domain_events.push(ReflectionRecorded {
            assessment_id: self.id,
            attempt_number: self.reflection_attempts.len(),
            improved: attempt.improved_score > self.overall_score,
        });

        if attempt.improved_score > self.overall_score {
            self.overall_score = attempt.improved_score;
        }
    }

    /// Check if quality meets threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.overall_score >= threshold
    }

    /// Get improvement suggestions
    pub fn suggestions(&self) -> Vec<ImprovementSuggestion> {
        self.issues
            .iter()
            .flat_map(|issue| issue.suggestions())
            .collect()
    }
}

/// Quality dimension scores
#[derive(Clone)]
pub struct DimensionScores {
    /// Logical flow and structure
    pub coherence: f32,
    /// Coverage of query aspects
    pub completeness: f32,
    /// Backed by retrieved facts
    pub factual_grounding: f32,
    /// Addresses the actual query
    pub relevance: f32,
    /// Consistent with sub-answers
    pub consistency: f32,
}

impl DimensionScores {
    pub fn weighted_average(&self) -> f32 {
        let weights = [0.2, 0.25, 0.25, 0.2, 0.1]; // Configurable
        let scores = [
            self.coherence,
            self.completeness,
            self.factual_grounding,
            self.relevance,
            self.consistency,
        ];

        scores.iter().zip(weights.iter())
            .map(|(s, w)| s * w)
            .sum()
    }
}

/// Quality issue found in answer
#[derive(Clone)]
pub struct QualityIssue {
    pub dimension: QualityDimension,
    pub severity: Severity,
    pub description: String,
    pub location: Option<TextSpan>,
}

impl QualityIssue {
    pub fn suggestions(&self) -> Vec<ImprovementSuggestion> {
        match self.dimension {
            QualityDimension::Coherence => vec![
                ImprovementSuggestion::AddTransitions,
                ImprovementSuggestion::ReorganizeParagraphs,
            ],
            QualityDimension::Completeness => vec![
                ImprovementSuggestion::AddMissingAspect(self.description.clone()),
            ],
            QualityDimension::FactualGrounding => vec![
                ImprovementSuggestion::AddCitation,
                ImprovementSuggestion::VerifyFact(self.description.clone()),
            ],
            QualityDimension::Relevance => vec![
                ImprovementSuggestion::FocusOnQuery,
            ],
            QualityDimension::Consistency => vec![
                ImprovementSuggestion::ResolveContradiction(self.description.clone()),
            ],
        }
    }
}
```

### Domain Services

```rust
/// Reflection service for answer improvement
pub struct ReflectionService {
    inference_engine: Arc<InferenceEngine>,
    max_iterations: usize,
    quality_threshold: f32,
}

impl ReflectionService {
    /// Reflect and improve an answer
    pub async fn reflect_and_improve(
        &self,
        answer: Answer,
        assessment: &QualityAssessment,
        context: &ReflectionContext,
    ) -> Result<ImprovedAnswer, DomainError> {
        let mut current = answer;
        let mut current_score = assessment.overall_score;
        let mut attempts = Vec::new();

        for iteration in 0..self.max_iterations {
            if current_score >= self.quality_threshold {
                break;
            }

            // Generate critique prompt
            let critique_prompt = self.build_critique_prompt(&current, assessment);
            let critique = self.inference_engine.generate(GenerationRequest {
                prompt: critique_prompt,
                params: GenerationParams::default(),
                max_tokens: 500,
                ..Default::default()
            }).await?;

            // Generate improved answer
            let improvement_prompt = self.build_improvement_prompt(
                &context.original_query,
                &current,
                &critique.text,
            );
            let improved = self.inference_engine.generate(GenerationRequest {
                prompt: improvement_prompt,
                params: GenerationParams::default(),
                max_tokens: context.max_tokens,
                ..Default::default()
            }).await?;

            // Re-assess
            let new_assessment = QualityAssessment::assess(
                &Answer::from(improved.clone()),
                &context.into(),
            );

            attempts.push(ReflectionAttempt {
                iteration,
                critique: critique.text,
                improved_answer: improved.text.clone(),
                improved_score: new_assessment.overall_score,
            });

            if new_assessment.overall_score > current_score {
                current = Answer::from(improved);
                current_score = new_assessment.overall_score;
            }
        }

        Ok(ImprovedAnswer {
            answer: current,
            final_score: current_score,
            reflection_attempts: attempts,
        })
    }
}
```

---

## Bounded Context 5: Learning Context

### Purpose
Manages continuous improvement through SONA integration and trajectory learning.

### Aggregate Roots

#### LearningCoordinator Aggregate

```rust
/// Coordinates SONA learning across the system
pub struct LearningCoordinator {
    id: CoordinatorId,

    // SONA integration
    sona: Arc<RwLock<SonaIntegration>>,

    // ReasoningBank for trajectory storage
    reasoning_bank: Arc<RwLock<ReasoningBank>>,

    // Active trajectories
    active_trajectories: HashMap<TrajectoryId, ActiveTrajectory>,

    // Learning statistics
    stats: LearningStats,
}

impl LearningCoordinator {
    /// Start recording a new trajectory
    pub fn start_trajectory(&mut self, query: &Query) -> TrajectoryId {
        let id = TrajectoryId::generate();
        let trajectory = ActiveTrajectory::new(id, query.clone());
        self.active_trajectories.insert(id, trajectory);

        self.domain_events.push(TrajectoryStarted {
            coordinator_id: self.id,
            trajectory_id: id,
        });

        id
    }

    /// Record a step in a trajectory
    pub fn record_step(&mut self, trajectory_id: TrajectoryId, step: TrajectoryStep) -> Result<(), DomainError> {
        let trajectory = self.active_trajectories
            .get_mut(&trajectory_id)
            .ok_or(DomainError::TrajectoryNotFound)?;

        trajectory.add_step(step);
        Ok(())
    }

    /// Complete a trajectory with outcome
    pub fn complete_trajectory(
        &mut self,
        trajectory_id: TrajectoryId,
        outcome: TrajectoryOutcome,
    ) -> Result<(), DomainError> {
        let trajectory = self.active_trajectories
            .remove(&trajectory_id)
            .ok_or(DomainError::TrajectoryNotFound)?;

        // Store in ReasoningBank
        let completed = trajectory.complete(outcome.clone());
        self.reasoning_bank.write().store_trajectory(completed.clone())?;

        // Trigger SONA learning based on outcome
        if outcome.is_successful() {
            self.sona.write().instant_learn(&completed)?;
        }

        self.stats.total_trajectories += 1;
        if outcome.is_successful() {
            self.stats.successful_trajectories += 1;
        }

        self.domain_events.push(TrajectoryCompleted {
            coordinator_id: self.id,
            trajectory_id,
            successful: outcome.is_successful(),
            quality_score: outcome.quality_score,
        });

        Ok(())
    }

    /// Trigger background learning loop
    pub async fn run_background_loop(&mut self) -> Result<LearningResult, DomainError> {
        // Query ReasoningBank for recent high-quality trajectories
        let trajectories = self.reasoning_bank.read()
            .query_recent_successful(Duration::from_secs(3600))?;

        if trajectories.is_empty() {
            return Ok(LearningResult::NoData);
        }

        // Train SONA on accumulated data
        let result = self.sona.write().background_learn(&trajectories)?;

        self.domain_events.push(BackgroundLearningCompleted {
            coordinator_id: self.id,
            trajectories_processed: trajectories.len(),
            improvement: result.improvement_score,
        });

        Ok(LearningResult::Completed(result))
    }
}

/// Active trajectory being recorded
pub struct ActiveTrajectory {
    id: TrajectoryId,
    query: Query,
    steps: Vec<TrajectoryStep>,
    started_at: DateTime<Utc>,
}

/// Trajectory step
#[derive(Clone)]
pub struct TrajectoryStep {
    pub step_type: StepType,
    pub input: String,
    pub output: String,
    pub duration: Duration,
    pub tokens_used: usize,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone)]
pub enum StepType {
    Retrieval,
    Decomposition,
    SubQuery,
    Generation,
    Synthesis,
    Reflection,
}
```

---

## Bounded Context 6: Observability Context

### Purpose
Provides audit logging, metrics collection, and operational visibility.

### Aggregate Roots

```rust
/// Witness log for audit trails
pub struct WitnessLog {
    id: LogId,

    // Entries (ruvector-backed)
    entries: HnswIndex,

    // Write buffer for batching
    write_buffer: Vec<WitnessEntry>,
    buffer_limit: usize,

    // Statistics
    total_entries: u64,
}

impl WitnessLog {
    /// Record a witness entry (async batched)
    pub fn record(&mut self, entry: WitnessEntry) {
        self.write_buffer.push(entry.clone());

        self.domain_events.push(WitnessRecorded {
            log_id: self.id,
            entry_type: entry.entry_type,
            session_id: entry.session_id,
        });

        // Flush if buffer is full
        if self.write_buffer.len() >= self.buffer_limit {
            self.flush();
        }
    }

    /// Flush write buffer
    pub fn flush(&mut self) {
        for entry in self.write_buffer.drain(..) {
            let embedding = self.compute_embedding(&entry);
            let _ = self.entries.insert(
                entry.id,
                &embedding,
                entry.to_metadata(),
            );
        }
        self.total_entries += self.write_buffer.len() as u64;
    }

    /// Search witness entries semantically
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Vec<WitnessEntry> {
        self.entries
            .search(query_embedding, limit)
            .into_iter()
            .filter_map(|r| WitnessEntry::from_metadata(&r.metadata))
            .collect()
    }
}

/// Witness entry value object
#[derive(Clone)]
pub struct WitnessEntry {
    pub id: EntryId,
    pub session_id: SessionId,
    pub entry_type: WitnessType,
    pub timestamp: DateTime<Utc>,
    pub latency: LatencyBreakdown,
    pub routing_decision: Option<RoutingDecision>,
    pub quality_score: Option<f32>,
    pub error: Option<ErrorInfo>,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone)]
pub enum WitnessType {
    QueryStart,
    RetrievalComplete,
    GenerationComplete,
    QueryComplete,
    Error,
    CacheHit,
}
```

---

## Integration Patterns

### Anti-Corruption Layer: Memory Context

```rust
/// ACL translating between Orchestration and Memory contexts
pub struct MemoryAcl {
    memory_store: Arc<MemoryStore>,
    embedding_service: Arc<dyn EmbeddingService>,
}

impl MemoryAcl {
    /// Translate query to memory retrieval
    pub async fn retrieve_for_query(&self, query: &Query) -> Vec<MemorySpan> {
        let embedding = self.embedding_service.embed(&query.text)?;
        self.memory_store.retrieve(&embedding, 5)
    }

    /// Store answer as new memory
    pub async fn store_answer(&self, query: &Query, answer: &Answer) -> MemoryId {
        let embedding = self.embedding_service.embed(&format!("{} {}", query.text, answer.text))?;
        let span = MemorySpan {
            id: MemoryId::generate(),
            text: answer.text.clone(),
            embedding,
            similarity_score: 1.0,
            source: Some("generated".to_string()),
            namespace: Namespace::Patterns,
            metadata: [
                ("query".to_string(), query.text.clone()),
                ("quality".to_string(), answer.quality_score.to_string()),
            ].into_iter().collect(),
        };
        self.memory_store.store(span)
    }
}
```

### Event Flow

```
+===========================================================================+
|                          EVENT FLOW                                        |
+===========================================================================+
|                                                                            |
|  ORCHESTRATION          MEMORY           INFERENCE        LEARNING         |
|  -----------           ------           ---------        --------          |
|      |                    |                 |               |              |
|  QuerySubmitted           |                 |               |              |
|      |                    |                 |               |              |
|      +---(retrieve)------>|                 |               |              |
|      |                    |                 |               |              |
|      |<---(spans)---------+                 |               |              |
|      |                    |                 |               |              |
|  QueryDecomposed          |                 |               |              |
|      |                    |                 |               |              |
|      +----(generate)------|---------------->|               |              |
|      |                    |                 |               |              |
|      |<---(generation)----|----------------+                |              |
|      |                    |                 |               |              |
|  SubQueryCompleted        |                 |               |              |
|      |                    |                 |               |              |
|  AnswerSynthesized        |                 |               |              |
|      |                    |                 |               |              |
|      +--(store pattern)-->|                 |               |              |
|      |                    |                 |               |              |
|  QueryCompleted           |                 |               |              |
|      |                    |                 |               |              |
|      +---(trajectory)-----|-----------------|-------------->|              |
|      |                    |                 |               |              |
|                           |                 |         TrajectoryCompleted  |
|                           |                 |               |              |
|                           |<--(SONA learn)--|---------------+              |
|                           |                 |               |              |
+===========================================================================+
```

---

## npm Package Domain Layer

```typescript
// npm/packages/ruvllm/src/rlm/domain/index.ts

/**
 * Query session aggregate (TypeScript port)
 */
export class QuerySession {
  private readonly id: string;
  private readonly queryStack: ActiveQuery[] = [];
  private readonly cache: Map<string, CachedAnswer> = new Map();
  private tokensConsumed = 0;

  constructor(config: SessionConfig) {
    this.id = crypto.randomUUID();
  }

  submitQuery(query: Query): QueryId {
    const queryId = crypto.randomUUID();
    this.queryStack.push({
      id: queryId,
      query,
      depth: 0,
      subAnswers: [],
    });
    return queryId;
  }

  checkCache(query: Query): CachedAnswer | null {
    const hash = this.hashQuery(query);
    const cached = this.cache.get(hash);
    if (cached && !this.isExpired(cached)) {
      return cached;
    }
    return null;
  }

  private hashQuery(query: Query): string {
    return crypto.createHash('sha256')
      .update(query.text)
      .digest('hex');
  }
}

/**
 * Memory store domain service
 */
export class MemoryStore {
  constructor(private readonly index: HnswIndex) {}

  async store(span: MemorySpan): Promise<MemoryId> {
    const id = crypto.randomUUID();
    await this.index.insert(id, span.embedding, span.metadata);
    return id;
  }

  async retrieve(queryEmbedding: Float32Array, topK: number): Promise<MemorySpan[]> {
    const results = await this.index.search(queryEmbedding, topK);
    return results.map(r => ({
      id: r.id,
      text: r.metadata.text,
      embedding: r.vector,
      similarityScore: r.score,
      source: r.metadata.source,
      metadata: r.metadata,
    }));
  }
}

/**
 * Quality assessment value object
 */
export interface QualityAssessment {
  readonly overallScore: number;
  readonly dimensionScores: DimensionScores;
  readonly issues: QualityIssue[];
  readonly meetsThreshold: (threshold: number) => boolean;
}

export interface DimensionScores {
  coherence: number;
  completeness: number;
  factualGrounding: number;
  relevance: number;
  consistency: number;
}
```

---

## RuvLTRA Integration

### RuvLTRA as Primary Inference Backend

```rust
/// RuvLTRA integration for RLM
pub struct RuvLtraRlmBackend {
    model: Arc<RwLock<RuvLtraModel>>,
    config: RuvLtraConfig,
    ane_dispatcher: AneDispatcher,
}

impl RuvLtraRlmBackend {
    /// Create from RuvLTRA config
    pub fn new(config: RuvLtraConfig) -> Result<Self, DomainError> {
        let model = RuvLtraModel::new(&config)?;
        let ane_dispatcher = AneDispatcher::new(config.ane_optimization);

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            config,
            ane_dispatcher,
        })
    }

    /// Enable SONA pretraining for continuous learning
    pub fn enable_sona(&self) -> Result<(), DomainError> {
        self.model.write().enable_sona_pretraining()
    }

    /// Get embedding from hidden states
    pub fn embed(&self, text: &str, tokenizer: &RuvTokenizer) -> Result<Vec<f32>, DomainError> {
        let tokens = tokenizer.encode(text)?;
        let positions: Vec<usize> = (0..tokens.len()).collect();

        // Forward through model to get hidden states
        let logits = self.model.read().forward(&tokens, &positions, None)?;

        // Extract final hidden state (before LM head)
        // For embeddings, we use mean pooling over the last layer
        let hidden_size = self.config.hidden_size;
        let seq_len = tokens.len();

        // Mean pool over sequence dimension
        let mut embedding = vec![0.0; hidden_size];
        for t in 0..seq_len {
            for h in 0..hidden_size {
                // Note: This is simplified - actual implementation would
                // access intermediate hidden states, not logits
                embedding[h] += logits[t * self.config.vocab_size + h % self.config.vocab_size];
            }
        }
        for h in 0..hidden_size {
            embedding[h] /= seq_len as f32;
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut embedding {
            *x /= norm;
        }

        Ok(embedding)
    }
}

impl LlmBackend for RuvLtraRlmBackend {
    fn id(&self) -> &str {
        "ruvltra"
    }

    fn max_context(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough estimate: ~4 chars per token for Qwen tokenizer
        text.len() / 4
    }

    async fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<GenerationOutput> {
        // Implementation in InferenceEngine section
        todo!()
    }
}
```

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **Query** | A user question or task to be answered by the RLM system |
| **Sub-Query** | A decomposed part of a complex query processed recursively |
| **Memory Span** | A text chunk with embedding stored in ruvector |
| **Trajectory** | A recorded sequence of steps for learning from experience |
| **Quality Score** | A 0-1 measure of answer quality across dimensions |
| **Reflection** | The process of critiquing and improving an answer |
| **RuvLTRA** | The primary ANE-optimized LLM backend (Qwen 0.5B) |
| **SONA** | Self-Optimizing Neural Architecture for continuous learning |
| **HNSW** | Hierarchical Navigable Small World graphs for fast similarity search |
| **Memoization** | Caching of sub-query answers to avoid redundant computation |
| **Token Budget** | Maximum tokens allocated for a query chain |
| **ANE** | Apple Neural Engine for accelerated inference |

---

## References

- ADR-014: Recursive Language Model Integration
- ADR-002: RuvLLM Integration with Ruvector
- `/crates/ruvllm/src/models/ruvltra.rs` - RuvLTRA model implementation
- `/crates/ruvllm/src/reasoning_bank/mod.rs` - ReasoningBank implementation
- `/crates/ruvllm/src/sona/mod.rs` - SONA learning integration

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-21 | Ruvector Architecture Team | Initial DDD specification |
