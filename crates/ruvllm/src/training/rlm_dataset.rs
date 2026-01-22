//! # RLM (Recursive Learning Machine) Dataset Module
//!
//! Provides training data structures for RLM task routing and decomposition,
//! including query decomposition, sub-answer synthesis, and trajectory metadata.
//!
//! ## Overview
//!
//! The RLM training pipeline focuses on two key capabilities:
//!
//! 1. **Query Decomposition**: Breaking complex queries into manageable sub-queries
//! 2. **Answer Synthesis**: Combining sub-answers into coherent final responses
//!
//! ## Dataset Structure
//!
//! ```text
//! RlmTrainingExample
//! ├── query: String              # Original complex query
//! ├── decomposition              # How the query was broken down
//! │   ├── sub_queries: Vec       # Individual sub-queries
//! │   ├── dependencies           # DAG of sub-query dependencies
//! │   └── strategy               # Decomposition strategy used
//! ├── sub_answers: Vec           # Answers to each sub-query
//! ├── final_answer: String       # Synthesized final answer
//! ├── quality_score: f32         # Overall quality (0.0-1.0)
//! └── trajectory                 # Execution metadata
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::training::rlm_dataset::{RlmDataset, RlmDatasetConfig};
//! use ruvllm::reasoning_bank::ReasoningBank;
//!
//! // Create dataset from ReasoningBank patterns
//! let bank = ReasoningBank::new(config)?;
//! let dataset = RlmDataset::from_reasoning_bank(&bank, 0.7)?;
//!
//! // Generate contrastive pairs for routing
//! let pairs = dataset.generate_contrastive_pairs();
//! println!("Generated {} contrastive pairs", pairs.len());
//!
//! // Create training batches
//! let batches = dataset.to_training_batch(32);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::Result;
use crate::reasoning_bank::{
    Pattern, ReasoningBank, Trajectory, TrajectoryMetadata as BankTrajectoryMetadata, Verdict,
};

// =============================================================================
// Query Decomposition Types
// =============================================================================

/// Strategy for decomposing a query
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Sequential steps that must be executed in order
    Sequential,
    /// Independent sub-queries that can run in parallel
    Parallel,
    /// Tree structure with hierarchical dependencies
    Hierarchical,
    /// Complex DAG with arbitrary dependencies
    DagBased,
    /// Iterative refinement (query -> result -> refined query)
    Iterative,
    /// No decomposition needed (simple query)
    None,
}

impl Default for DecompositionStrategy {
    fn default() -> Self {
        Self::Sequential
    }
}

impl DecompositionStrategy {
    /// Get complexity weight for this strategy
    pub fn complexity_weight(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Sequential => 1.5,
            Self::Parallel => 1.5,
            Self::Hierarchical => 2.0,
            Self::Iterative => 2.5,
            Self::DagBased => 3.0,
        }
    }
}

/// A sub-query in the decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuery {
    /// Unique identifier within the decomposition
    pub id: u32,
    /// The sub-query text
    pub query: String,
    /// Expected output type (e.g., "code", "analysis", "data")
    pub expected_type: String,
    /// Dependencies (IDs of sub-queries that must complete first)
    pub dependencies: Vec<u32>,
    /// Recommended agent type for this sub-query
    pub recommended_agent: Option<String>,
    /// Estimated complexity (0.0-1.0)
    pub complexity: f32,
    /// Optional context from parent query
    pub context: Option<String>,
}

impl SubQuery {
    /// Create a new sub-query
    pub fn new(id: u32, query: String) -> Self {
        Self {
            id,
            query,
            expected_type: "text".to_string(),
            dependencies: Vec::new(),
            recommended_agent: None,
            complexity: 0.5,
            context: None,
        }
    }

    /// Builder: Set expected output type
    pub fn with_type(mut self, expected_type: &str) -> Self {
        self.expected_type = expected_type.to_string();
        self
    }

    /// Builder: Add dependency
    pub fn with_dependency(mut self, dep_id: u32) -> Self {
        if !self.dependencies.contains(&dep_id) {
            self.dependencies.push(dep_id);
        }
        self
    }

    /// Builder: Set recommended agent
    pub fn with_agent(mut self, agent: &str) -> Self {
        self.recommended_agent = Some(agent.to_string());
        self
    }

    /// Builder: Set complexity
    pub fn with_complexity(mut self, complexity: f32) -> Self {
        self.complexity = complexity.clamp(0.0, 1.0);
        self
    }

    /// Check if this sub-query has no dependencies
    pub fn is_root(&self) -> bool {
        self.dependencies.is_empty()
    }
}

/// Decomposition of a complex query into sub-queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDecomposition {
    /// Sub-queries in execution order
    pub sub_queries: Vec<SubQuery>,
    /// Decomposition strategy used
    pub strategy: DecompositionStrategy,
    /// Reasoning for this decomposition
    pub rationale: String,
    /// Total estimated complexity
    pub total_complexity: f32,
    /// Whether decomposition was successful
    pub success: bool,
    /// Error message if decomposition failed
    pub error: Option<String>,
}

impl QueryDecomposition {
    /// Create a new decomposition
    pub fn new(strategy: DecompositionStrategy) -> Self {
        Self {
            sub_queries: Vec::new(),
            strategy,
            rationale: String::new(),
            total_complexity: 0.0,
            success: true,
            error: None,
        }
    }

    /// Create a failed decomposition
    pub fn failed(error: &str) -> Self {
        Self {
            sub_queries: Vec::new(),
            strategy: DecompositionStrategy::None,
            rationale: String::new(),
            total_complexity: 0.0,
            success: false,
            error: Some(error.to_string()),
        }
    }

    /// Add a sub-query
    pub fn add_sub_query(&mut self, sub_query: SubQuery) {
        self.total_complexity += sub_query.complexity * self.strategy.complexity_weight();
        self.sub_queries.push(sub_query);
    }

    /// Get sub-queries that can run next (dependencies satisfied)
    pub fn get_ready_queries(&self, completed: &[u32]) -> Vec<&SubQuery> {
        self.sub_queries
            .iter()
            .filter(|sq| {
                !completed.contains(&sq.id)
                    && sq.dependencies.iter().all(|dep| completed.contains(dep))
            })
            .collect()
    }

    /// Validate the decomposition DAG (no cycles)
    pub fn validate(&self) -> Result<()> {
        // Topological sort to detect cycles
        let mut in_degree: HashMap<u32, usize> = HashMap::new();
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();

        for sq in &self.sub_queries {
            in_degree.entry(sq.id).or_insert(0);
            for &dep in &sq.dependencies {
                *in_degree.entry(sq.id).or_insert(0) += 1;
                adj.entry(dep).or_default().push(sq.id);
            }
        }

        let mut queue: Vec<u32> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited = 0;
        while let Some(node) = queue.pop() {
            visited += 1;
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }

        if visited != self.sub_queries.len() {
            return Err(crate::error::RuvLLMError::InvalidOperation(
                "Decomposition contains a cycle".to_string(),
            ));
        }

        Ok(())
    }
}

// =============================================================================
// Sub-Answer Types
// =============================================================================

/// Answer to a sub-query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAnswer {
    /// ID of the sub-query this answers
    pub sub_query_id: u32,
    /// The answer content
    pub content: String,
    /// Confidence in this answer (0.0-1.0)
    pub confidence: f32,
    /// Agent that produced this answer
    pub agent: String,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Quality score (0.0-1.0)
    pub quality: f32,
    /// Whether this answer was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Intermediate reasoning/chain-of-thought
    pub reasoning: Option<String>,
}

impl SubAnswer {
    /// Create a successful sub-answer
    pub fn success(sub_query_id: u32, content: String, agent: &str) -> Self {
        Self {
            sub_query_id,
            content,
            confidence: 0.8,
            agent: agent.to_string(),
            latency_ms: 0,
            quality: 0.8,
            success: true,
            error: None,
            reasoning: None,
        }
    }

    /// Create a failed sub-answer
    pub fn failure(sub_query_id: u32, error: &str, agent: &str) -> Self {
        Self {
            sub_query_id,
            content: String::new(),
            confidence: 0.0,
            agent: agent.to_string(),
            latency_ms: 0,
            quality: 0.0,
            success: false,
            error: Some(error.to_string()),
            reasoning: None,
        }
    }

    /// Builder: Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Builder: Set quality
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Builder: Set latency
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Builder: Set reasoning
    pub fn with_reasoning(mut self, reasoning: &str) -> Self {
        self.reasoning = Some(reasoning.to_string());
        self
    }
}

// =============================================================================
// Trajectory Metadata
// =============================================================================

/// Metadata about the RLM execution trajectory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RlmTrajectoryMetadata {
    /// Session ID
    pub session_id: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// Total latency in milliseconds
    pub total_latency_ms: u64,
    /// Number of retries
    pub retries: u32,
    /// Maximum parallel branches executed
    pub max_parallelism: u32,
    /// Models used during execution
    pub models_used: Vec<String>,
    /// Agents invoked
    pub agents_invoked: Vec<String>,
    /// Tools used
    pub tools_used: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

impl RlmTrajectoryMetadata {
    /// Create from ReasoningBank trajectory metadata
    pub fn from_bank_metadata(meta: &BankTrajectoryMetadata) -> Self {
        Self {
            session_id: meta.session_id.clone(),
            user_id: meta.user_id.clone(),
            total_latency_ms: 0,
            retries: 0,
            max_parallelism: 1,
            models_used: meta.models_used.clone(),
            agents_invoked: Vec::new(),
            tools_used: meta.tools_invoked.clone(),
            attributes: meta.attributes.clone(),
        }
    }
}

// =============================================================================
// RLM Training Example
// =============================================================================

/// A complete RLM training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmTrainingExample {
    /// Unique identifier
    pub id: Uuid,
    /// Original complex query
    pub query: String,
    /// Query embedding (optional)
    pub query_embedding: Option<Vec<f32>>,
    /// How the query was decomposed
    pub decomposition: QueryDecomposition,
    /// Answers to each sub-query
    pub sub_answers: Vec<SubAnswer>,
    /// Final synthesized answer
    pub final_answer: String,
    /// Final answer embedding (optional)
    pub final_embedding: Option<Vec<f32>>,
    /// Overall quality score (0.0-1.0)
    pub quality_score: f32,
    /// Execution trajectory metadata
    pub trajectory: RlmTrajectoryMetadata,
    /// Whether this example was successful
    pub success: bool,
    /// Lessons learned from this example
    pub lessons: Vec<String>,
    /// Source of this example (e.g., "reasoning_bank", "synthetic", "human")
    pub source: String,
}

impl RlmTrainingExample {
    /// Create a new RLM training example
    pub fn new(query: String, decomposition: QueryDecomposition) -> Self {
        Self {
            id: Uuid::new_v4(),
            query,
            query_embedding: None,
            decomposition,
            sub_answers: Vec::new(),
            final_answer: String::new(),
            final_embedding: None,
            quality_score: 0.0,
            trajectory: RlmTrajectoryMetadata::default(),
            success: false,
            lessons: Vec::new(),
            source: "manual".to_string(),
        }
    }

    /// Create from a ReasoningBank trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        // Extract decomposition from trajectory steps
        let mut decomposition = QueryDecomposition::new(DecompositionStrategy::Sequential);

        for (i, step) in trajectory.steps.iter().enumerate() {
            let sub_query =
                SubQuery::new(i as u32, step.action.clone()).with_complexity(step.confidence);
            decomposition.add_sub_query(sub_query);
        }

        // Extract sub-answers
        let sub_answers: Vec<SubAnswer> = trajectory
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| {
                let mut answer = SubAnswer::success(
                    i as u32,
                    step.rationale.clone(),
                    step.metadata
                        .as_ref()
                        .and_then(|m| m.model.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("unknown"),
                )
                .with_confidence(step.confidence)
                .with_latency(step.latency_ms);

                if !step.outcome.is_success() {
                    answer.success = false;
                    answer.quality = step.outcome.quality_score();
                }

                answer
            })
            .collect();

        let quality_score = trajectory.quality;
        let success = trajectory.is_success();

        Self {
            id: trajectory.uuid,
            query: String::new(), // Would need to extract from context
            query_embedding: Some(trajectory.query_embedding.clone()),
            decomposition,
            sub_answers,
            final_answer: String::new(),
            final_embedding: trajectory.response_embedding.clone(),
            quality_score,
            trajectory: RlmTrajectoryMetadata::from_bank_metadata(&trajectory.metadata),
            success,
            lessons: trajectory.lessons.clone(),
            source: "reasoning_bank".to_string(),
        }
    }

    /// Add a sub-answer
    pub fn add_sub_answer(&mut self, answer: SubAnswer) {
        self.sub_answers.push(answer);
    }

    /// Set the final answer
    pub fn set_final_answer(&mut self, answer: String) {
        self.final_answer = answer;
    }

    /// Compute quality from sub-answers
    pub fn compute_quality(&mut self) {
        if self.sub_answers.is_empty() {
            self.quality_score = 0.0;
            return;
        }

        // Weighted average based on sub-query complexity
        let mut total_weight = 0.0;
        let mut weighted_quality = 0.0;

        for (i, answer) in self.sub_answers.iter().enumerate() {
            let weight = if i < self.decomposition.sub_queries.len() {
                self.decomposition.sub_queries[i].complexity
            } else {
                1.0
            };
            weighted_quality += answer.quality * weight;
            total_weight += weight;
        }

        self.quality_score = if total_weight > 0.0 {
            weighted_quality / total_weight
        } else {
            0.0
        };

        self.success = self.quality_score >= 0.5 && self.sub_answers.iter().all(|a| a.success);
    }

    /// Get agent routing targets (for contrastive learning)
    pub fn get_agent_targets(&self) -> Vec<(String, String)> {
        self.decomposition
            .sub_queries
            .iter()
            .filter_map(|sq| {
                sq.recommended_agent
                    .as_ref()
                    .map(|agent| (sq.query.clone(), agent.clone()))
            })
            .collect()
    }
}

// =============================================================================
// Contrastive Pair for Routing
// =============================================================================

/// A contrastive pair for agent routing training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastivePair {
    /// Anchor query
    pub anchor: String,
    /// Anchor embedding (optional)
    pub anchor_embedding: Option<Vec<f32>>,
    /// Positive agent (correct routing)
    pub positive_agent: String,
    /// Negative agent (incorrect routing)
    pub negative_agent: String,
    /// Whether this is a hard negative
    pub is_hard_negative: bool,
    /// Quality score of the anchor example
    pub quality: f32,
    /// Source example ID
    pub source_id: Uuid,
}

// =============================================================================
// RLM Dataset Configuration
// =============================================================================

/// Configuration for RLM dataset generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmDatasetConfig {
    /// Minimum quality threshold for including examples
    pub min_quality: f32,
    /// Maximum examples to include
    pub max_examples: usize,
    /// Include only successful examples
    pub success_only: bool,
    /// Hard negative mining ratio
    pub hard_negative_ratio: f32,
    /// Embedding dimension (for validation)
    pub embedding_dim: usize,
    /// Augmentation enabled
    pub augmentation_enabled: bool,
    /// Random seed for shuffling
    pub seed: u64,
}

impl Default for RlmDatasetConfig {
    fn default() -> Self {
        Self {
            min_quality: 0.5,
            max_examples: 100_000,
            success_only: false,
            hard_negative_ratio: 0.3,
            embedding_dim: 768,
            augmentation_enabled: true,
            seed: 42,
        }
    }
}

// =============================================================================
// Training Batch
// =============================================================================

/// A batch of training examples
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Batch index
    pub batch_idx: usize,
    /// Examples in this batch
    pub examples: Vec<RlmTrainingExample>,
    /// Contrastive pairs in this batch
    pub contrastive_pairs: Vec<ContrastivePair>,
}

impl TrainingBatch {
    /// Get batch size
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get average quality
    pub fn avg_quality(&self) -> f32 {
        if self.examples.is_empty() {
            return 0.0;
        }
        self.examples.iter().map(|e| e.quality_score).sum::<f32>() / self.examples.len() as f32
    }
}

// =============================================================================
// RLM Dataset
// =============================================================================

/// RLM dataset for training decomposition and synthesis
pub struct RlmDataset {
    /// Training examples
    examples: Vec<RlmTrainingExample>,
    /// Configuration
    config: RlmDatasetConfig,
    /// Dataset statistics
    stats: RlmDatasetStats,
}

/// Statistics for the RLM dataset
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RlmDatasetStats {
    /// Total examples
    pub total_examples: usize,
    /// Successful examples
    pub successful_examples: usize,
    /// Average quality score
    pub avg_quality: f32,
    /// Average decomposition depth
    pub avg_decomposition_depth: f32,
    /// Strategy distribution
    pub strategy_distribution: HashMap<String, usize>,
    /// Agent distribution
    pub agent_distribution: HashMap<String, usize>,
}

impl RlmDataset {
    /// Create a new empty dataset
    pub fn new(config: RlmDatasetConfig) -> Self {
        Self {
            examples: Vec::new(),
            config,
            stats: RlmDatasetStats::default(),
        }
    }

    /// Create dataset from ReasoningBank
    ///
    /// Extracts patterns from the bank and converts them to RLM training examples.
    pub fn from_reasoning_bank(bank: &ReasoningBank, min_quality: f32) -> Result<Self> {
        let patterns = bank.export_patterns()?;
        Self::from_patterns(&patterns, min_quality)
    }

    /// Create dataset from pre-exported patterns
    ///
    /// Note: This creates synthetic RLM examples from stored patterns.
    /// The patterns are converted to simple decompositions with one sub-query each.
    pub fn from_patterns(patterns: &[Pattern], min_quality: f32) -> Result<Self> {
        let mut config = RlmDatasetConfig::default();
        config.min_quality = min_quality;

        let mut dataset = Self::new(config);

        // Convert patterns to RLM examples
        for pattern in patterns {
            if pattern.avg_quality >= min_quality {
                // Create a synthetic decomposition from the pattern
                let mut decomposition = QueryDecomposition::new(DecompositionStrategy::Sequential);

                // Get an example action if available
                let action = pattern
                    .example_actions
                    .first()
                    .cloned()
                    .unwrap_or_else(|| format!("Pattern-{}", pattern.id));

                // Add a single sub-query for the pattern
                let sub_query = SubQuery::new(0, action.clone())
                    .with_type("analysis")
                    .with_complexity(pattern.confidence);
                decomposition.add_sub_query(sub_query);

                let mut example = RlmTrainingExample::new(action, decomposition);
                example.query_embedding = Some(pattern.embedding.clone());
                example.quality_score = pattern.avg_quality;
                example.success = pattern.success_count > 0;
                example.source = "pattern_store".to_string();
                example.lessons = pattern.lessons.clone();

                dataset.add_example(example);
            }
        }

        dataset.compute_stats();
        Ok(dataset)
    }

    /// Add an example to the dataset
    pub fn add_example(&mut self, example: RlmTrainingExample) {
        if example.quality_score >= self.config.min_quality
            && self.examples.len() < self.config.max_examples
        {
            if !self.config.success_only || example.success {
                self.examples.push(example);
            }
        }
    }

    /// Generate contrastive pairs for agent routing
    pub fn generate_contrastive_pairs(&self) -> Vec<ContrastivePair> {
        let mut pairs = Vec::new();
        let agents: Vec<&str> = vec![
            "coder",
            "researcher",
            "reviewer",
            "tester",
            "architect",
            "security-architect",
            "debugger",
            "documenter",
            "refactorer",
            "optimizer",
            "devops",
            "api-docs",
            "planner",
        ];

        for example in &self.examples {
            let agent_targets = example.get_agent_targets();

            for (query, positive_agent) in agent_targets {
                // Find negative agents
                for &negative_agent in &agents {
                    if negative_agent != positive_agent {
                        // Determine if it's a hard negative
                        let is_hard = Self::is_hard_negative(&positive_agent, negative_agent);

                        // Only include if it meets the hard negative ratio requirement
                        let include = if is_hard {
                            rand::random::<f32>() < self.config.hard_negative_ratio
                        } else {
                            rand::random::<f32>() < (1.0 - self.config.hard_negative_ratio)
                        };

                        if include {
                            pairs.push(ContrastivePair {
                                anchor: query.clone(),
                                anchor_embedding: example.query_embedding.clone(),
                                positive_agent: positive_agent.clone(),
                                negative_agent: negative_agent.to_string(),
                                is_hard_negative: is_hard,
                                quality: example.quality_score,
                                source_id: example.id,
                            });
                        }
                    }
                }
            }
        }

        pairs
    }

    /// Check if a negative agent is a hard negative for the positive
    fn is_hard_negative(positive: &str, negative: &str) -> bool {
        // Define confusable agent pairs
        let confusable_pairs = [
            ("coder", "debugger"),
            ("coder", "refactorer"),
            ("researcher", "reviewer"),
            ("tester", "reviewer"),
            ("architect", "planner"),
            ("documenter", "api-docs"),
            ("optimizer", "debugger"),
            ("devops", "architect"),
            ("security-architect", "reviewer"),
        ];

        confusable_pairs
            .iter()
            .any(|(a, b)| (positive == *a && negative == *b) || (positive == *b && negative == *a))
    }

    /// Create training batches
    pub fn to_training_batch(&self, batch_size: usize) -> Vec<TrainingBatch> {
        let mut batches = Vec::new();
        let contrastive_pairs = self.generate_contrastive_pairs();

        // Shuffle examples
        let mut examples = self.examples.clone();
        Self::shuffle(&mut examples, self.config.seed);

        // Create batches
        for (batch_idx, chunk) in examples.chunks(batch_size).enumerate() {
            let batch_ids: std::collections::HashSet<Uuid> = chunk.iter().map(|e| e.id).collect();

            // Get contrastive pairs for this batch
            let batch_pairs: Vec<ContrastivePair> = contrastive_pairs
                .iter()
                .filter(|p| batch_ids.contains(&p.source_id))
                .cloned()
                .collect();

            batches.push(TrainingBatch {
                batch_idx,
                examples: chunk.to_vec(),
                contrastive_pairs: batch_pairs,
            });
        }

        batches
    }

    /// Shuffle examples with seed
    fn shuffle(examples: &mut [RlmTrainingExample], seed: u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut rng_state = hasher.finish();

        for i in (1..examples.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            examples.swap(i, j);
        }
    }

    /// Compute dataset statistics
    fn compute_stats(&mut self) {
        self.stats.total_examples = self.examples.len();
        self.stats.successful_examples = self.examples.iter().filter(|e| e.success).count();

        if !self.examples.is_empty() {
            self.stats.avg_quality = self.examples.iter().map(|e| e.quality_score).sum::<f32>()
                / self.examples.len() as f32;

            self.stats.avg_decomposition_depth = self
                .examples
                .iter()
                .map(|e| e.decomposition.sub_queries.len() as f32)
                .sum::<f32>()
                / self.examples.len() as f32;
        }

        // Strategy distribution
        self.stats.strategy_distribution.clear();
        for example in &self.examples {
            let strategy = format!("{:?}", example.decomposition.strategy);
            *self
                .stats
                .strategy_distribution
                .entry(strategy)
                .or_insert(0) += 1;
        }

        // Agent distribution
        self.stats.agent_distribution.clear();
        for example in &self.examples {
            for answer in &example.sub_answers {
                *self
                    .stats
                    .agent_distribution
                    .entry(answer.agent.clone())
                    .or_insert(0) += 1;
            }
        }
    }

    /// Get dataset statistics
    pub fn stats(&self) -> &RlmDatasetStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &RlmDatasetConfig {
        &self.config
    }

    /// Get number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get all examples
    pub fn examples(&self) -> &[RlmTrainingExample] {
        &self.examples
    }

    /// Filter examples by quality
    pub fn filter_by_quality(&self, min_quality: f32) -> Vec<&RlmTrainingExample> {
        self.examples
            .iter()
            .filter(|e| e.quality_score >= min_quality)
            .collect()
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.examples).map_err(|e| {
            crate::error::RuvLLMError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            ))
        })
    }

    /// Import from JSON
    pub fn from_json(json: &str, config: RlmDatasetConfig) -> Result<Self> {
        let examples: Vec<RlmTrainingExample> = serde_json::from_str(json).map_err(|e| {
            crate::error::RuvLLMError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            ))
        })?;

        let mut dataset = Self::new(config);
        for example in examples {
            dataset.add_example(example);
        }
        dataset.compute_stats();

        Ok(dataset)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_query_creation() {
        let sq = SubQuery::new(0, "Analyze the codebase".to_string())
            .with_type("analysis")
            .with_agent("researcher")
            .with_complexity(0.7);

        assert_eq!(sq.id, 0);
        assert_eq!(sq.expected_type, "analysis");
        assert_eq!(sq.recommended_agent, Some("researcher".to_string()));
        assert!((sq.complexity - 0.7).abs() < 0.001);
        assert!(sq.is_root());
    }

    #[test]
    fn test_query_decomposition() {
        let mut decomp = QueryDecomposition::new(DecompositionStrategy::Sequential);

        decomp.add_sub_query(SubQuery::new(0, "Step 1".to_string()).with_complexity(0.5));
        decomp.add_sub_query(
            SubQuery::new(1, "Step 2".to_string())
                .with_dependency(0)
                .with_complexity(0.5),
        );

        assert_eq!(decomp.sub_queries.len(), 2);
        assert!(decomp.validate().is_ok());

        let ready = decomp.get_ready_queries(&[]);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, 0);

        let ready_after = decomp.get_ready_queries(&[0]);
        assert_eq!(ready_after.len(), 1);
        assert_eq!(ready_after[0].id, 1);
    }

    #[test]
    fn test_decomposition_cycle_detection() {
        let mut decomp = QueryDecomposition::new(DecompositionStrategy::DagBased);

        // Create a cycle: 0 -> 1 -> 2 -> 0
        decomp.add_sub_query(SubQuery::new(0, "A".to_string()).with_dependency(2));
        decomp.add_sub_query(SubQuery::new(1, "B".to_string()).with_dependency(0));
        decomp.add_sub_query(SubQuery::new(2, "C".to_string()).with_dependency(1));

        assert!(decomp.validate().is_err());
    }

    #[test]
    fn test_sub_answer_creation() {
        let answer = SubAnswer::success(0, "Result".to_string(), "coder")
            .with_confidence(0.9)
            .with_quality(0.85)
            .with_latency(150);

        assert_eq!(answer.sub_query_id, 0);
        assert!(answer.success);
        assert!((answer.confidence - 0.9).abs() < 0.001);
        assert!((answer.quality - 0.85).abs() < 0.001);
        assert_eq!(answer.latency_ms, 150);
    }

    #[test]
    fn test_rlm_training_example() {
        let decomp = QueryDecomposition::new(DecompositionStrategy::Sequential);
        let mut example = RlmTrainingExample::new("Complex query".to_string(), decomp);

        example.add_sub_answer(
            SubAnswer::success(0, "Answer 1".to_string(), "coder").with_quality(0.8),
        );

        example.add_sub_answer(
            SubAnswer::success(1, "Answer 2".to_string(), "tester").with_quality(0.9),
        );

        example.compute_quality();
        assert!(example.quality_score > 0.0);
    }

    #[test]
    fn test_rlm_dataset() {
        let config = RlmDatasetConfig {
            min_quality: 0.3,
            ..Default::default()
        };
        let mut dataset = RlmDataset::new(config);

        let mut example1 = RlmTrainingExample::new(
            "Query 1".to_string(),
            QueryDecomposition::new(DecompositionStrategy::Sequential),
        );
        example1.quality_score = 0.8;
        example1.success = true;

        let mut example2 = RlmTrainingExample::new(
            "Query 2".to_string(),
            QueryDecomposition::new(DecompositionStrategy::Parallel),
        );
        example2.quality_score = 0.6;
        example2.success = true;

        dataset.add_example(example1);
        dataset.add_example(example2);

        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_training_batches() {
        let config = RlmDatasetConfig::default();
        let mut dataset = RlmDataset::new(config);

        for i in 0..10 {
            let mut example = RlmTrainingExample::new(
                format!("Query {}", i),
                QueryDecomposition::new(DecompositionStrategy::Sequential),
            );
            example.quality_score = 0.8;
            example.success = true;
            dataset.add_example(example);
        }

        let batches = dataset.to_training_batch(3);
        assert_eq!(batches.len(), 4); // ceil(10/3) = 4
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn test_hard_negative_detection() {
        assert!(RlmDataset::is_hard_negative("coder", "debugger"));
        assert!(RlmDataset::is_hard_negative("debugger", "coder"));
        assert!(!RlmDataset::is_hard_negative("coder", "documenter"));
    }
}
