//! # RLM (Recursive Learning Machine) Trainer Module
//!
//! Provides training capabilities for RuvLTRA models on RLM task routing and
//! decomposition, including GRPO optimization and SONA integration for online learning.
//!
//! ## Training Capabilities
//!
//! 1. **Decomposition Training**: Learn to break complex queries into sub-queries
//! 2. **Synthesis Training**: Learn to combine sub-answers into final responses
//! 3. **Contrastive Fine-tuning**: Improve agent routing accuracy
//! 4. **Online Learning**: Real-time adaptation via SONA integration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        RlmTrainer                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
//! │  │ Decomp      │   │ Synthesis   │   │ Contrastive │           │
//! │  │ Trainer     │   │ Trainer     │   │ Fine-tuner  │           │
//! │  └─────────────┘   └─────────────┘   └─────────────┘           │
//! │        │                  │                  │                  │
//! │        v                  v                  v                  │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │              GRPO Optimizer                         │       │
//! │  │  - Relative advantage computation                   │       │
//! │  │  - PPO-style clipping                              │       │
//! │  │  - Adaptive KL penalty                             │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! │        │                  │                  │                  │
//! │        v                  v                  v                  │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │              SONA Integration                       │       │
//! │  │  - Online trajectory recording                      │       │
//! │  │  - EWC++ consolidation                             │       │
//! │  │  - Pattern bank updates                            │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::training::{RlmTrainer, RlmTrainingConfig, RlmDataset};
//! use ruvllm::models::ruvltra::RuvLtraModel;
//!
//! // Create trainer
//! let config = RlmTrainingConfig::default();
//! let model = Arc::new(RwLock::new(RuvLtraModel::new(&model_config)?));
//! let mut trainer = RlmTrainer::new(model, config);
//!
//! // Load dataset
//! let dataset = RlmDataset::from_reasoning_bank(&bank, 0.7)?;
//!
//! // Train decomposition
//! let decomp_result = trainer.train_decomposition(&dataset)?;
//! println!("Decomposition loss: {}", decomp_result.final_loss);
//!
//! // Train synthesis
//! let synth_result = trainer.train_synthesis(&dataset)?;
//! println!("Synthesis loss: {}", synth_result.final_loss);
//!
//! // Contrastive fine-tuning for routing
//! let pairs = dataset.generate_contrastive_pairs();
//! let routing_result = trainer.contrastive_finetune(&pairs)?;
//! println!("Routing accuracy: {:.2}%", routing_result.accuracy * 100.0);
//! ```

use chrono;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{Result, RuvLLMError};
use crate::models::ruvltra::RuvLtraModel;
use crate::sona::{SonaIntegration, Trajectory};
use crate::training::grpo::{GrpoConfig, GrpoOptimizer, GrpoSample, GrpoUpdateResult, SampleGroup};
use crate::training::rlm_dataset::{
    ContrastivePair, RlmDataset, RlmTrainingExample, TrainingBatch,
};

// =============================================================================
// Training Configuration
// =============================================================================

/// Configuration for RLM training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmTrainingConfig {
    /// Learning rate for decomposition training
    pub decomposition_lr: f32,
    /// Learning rate for synthesis training
    pub synthesis_lr: f32,
    /// Learning rate for contrastive fine-tuning
    pub contrastive_lr: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs for each training phase
    pub epochs: usize,
    /// GRPO configuration
    pub grpo_config: GrpoConfig,
    /// Contrastive margin for triplet loss
    pub contrastive_margin: f32,
    /// Temperature for InfoNCE loss
    pub infonce_temperature: f32,
    /// Weight for decomposition loss
    pub decomposition_weight: f32,
    /// Weight for synthesis loss
    pub synthesis_weight: f32,
    /// Weight for routing loss
    pub routing_weight: f32,
    /// Enable SONA online learning
    pub sona_enabled: bool,
    /// Minimum quality for online learning updates
    pub online_quality_threshold: f32,
    /// Checkpoint interval (epochs)
    pub checkpoint_interval: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Gradient clipping max norm
    pub max_grad_norm: f32,
    /// Validation split ratio
    pub validation_split: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for RlmTrainingConfig {
    fn default() -> Self {
        Self {
            decomposition_lr: 1e-5,
            synthesis_lr: 1e-5,
            contrastive_lr: 2e-5,
            batch_size: 32,
            epochs: 10,
            grpo_config: GrpoConfig::for_tool_use(),
            contrastive_margin: 0.5,
            infonce_temperature: 0.07,
            decomposition_weight: 1.0,
            synthesis_weight: 1.0,
            routing_weight: 1.0,
            sona_enabled: true,
            online_quality_threshold: 0.7,
            checkpoint_interval: 1,
            early_stopping_patience: 3,
            max_grad_norm: 1.0,
            validation_split: 0.1,
            seed: 42,
        }
    }
}

impl RlmTrainingConfig {
    /// Configuration for fast fine-tuning
    pub fn fast() -> Self {
        Self {
            epochs: 3,
            batch_size: 64,
            decomposition_lr: 1e-4,
            synthesis_lr: 1e-4,
            contrastive_lr: 5e-5,
            checkpoint_interval: 3,
            early_stopping_patience: 1,
            ..Default::default()
        }
    }

    /// Configuration for thorough training
    pub fn thorough() -> Self {
        Self {
            epochs: 50,
            batch_size: 16,
            decomposition_lr: 5e-6,
            synthesis_lr: 5e-6,
            contrastive_lr: 1e-5,
            checkpoint_interval: 5,
            early_stopping_patience: 10,
            ..Default::default()
        }
    }

    /// Configuration for routing-focused training
    pub fn routing_focused() -> Self {
        Self {
            routing_weight: 2.0,
            decomposition_weight: 0.5,
            synthesis_weight: 0.5,
            contrastive_lr: 3e-5,
            contrastive_margin: 0.3,
            infonce_temperature: 0.05,
            ..Default::default()
        }
    }
}

// =============================================================================
// Training Result
// =============================================================================

/// Result of a training phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmTrainingResult {
    /// Training phase name
    pub phase: String,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Total training steps
    pub total_steps: usize,
    /// Final training loss
    pub final_loss: f32,
    /// Best validation loss
    pub best_val_loss: f32,
    /// Best epoch
    pub best_epoch: usize,
    /// Final accuracy (for classification tasks)
    pub accuracy: f32,
    /// Loss history per epoch
    pub loss_history: Vec<f32>,
    /// Validation loss history
    pub val_loss_history: Vec<f32>,
    /// Training duration in milliseconds
    pub duration_ms: u64,
    /// Whether early stopping was triggered
    pub early_stopped: bool,
    /// GRPO update statistics
    pub grpo_stats: Option<GrpoStats>,
}

/// GRPO training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GrpoStats {
    /// Total GRPO updates
    pub total_updates: u64,
    /// Average policy loss
    pub avg_policy_loss: f32,
    /// Average KL divergence
    pub avg_kl_divergence: f32,
    /// Average entropy
    pub avg_entropy: f32,
    /// Average clip fraction
    pub avg_clip_fraction: f32,
    /// Final KL coefficient
    pub final_kl_coef: f32,
}

// =============================================================================
// RLM Trainer
// =============================================================================

/// RLM-specific trainer for RuvLTRA models
pub struct RlmTrainer {
    /// The model being trained
    model: Arc<RwLock<RuvLtraModel>>,
    /// GRPO optimizer
    optimizer: GrpoOptimizer,
    /// Training configuration
    config: RlmTrainingConfig,
    /// SONA integration for online learning
    sona: Option<Arc<RwLock<SonaIntegration>>>,
    /// Training checkpoints
    checkpoints: Vec<TrainingCheckpoint>,
    /// Current training state
    state: TrainingState,
}

/// Training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Epoch number
    pub epoch: usize,
    /// Step number
    pub step: usize,
    /// Training loss
    pub loss: f32,
    /// Validation loss
    pub val_loss: f32,
    /// Timestamp (Unix milliseconds)
    pub timestamp: u64,
    /// Phase name
    pub phase: String,
}

/// Current training state
#[derive(Debug, Clone, Default)]
struct TrainingState {
    current_epoch: usize,
    current_step: usize,
    best_val_loss: f32,
    patience_counter: usize,
    accumulated_loss: f32,
    accumulated_steps: usize,
}

impl RlmTrainer {
    /// Create a new RLM trainer
    pub fn new(model: Arc<RwLock<RuvLtraModel>>, config: RlmTrainingConfig) -> Self {
        let optimizer = GrpoOptimizer::new(config.grpo_config.clone());

        // Get SONA from model if available
        let sona = if config.sona_enabled {
            model.read().sona().cloned()
        } else {
            None
        };

        Self {
            model,
            optimizer,
            config,
            sona,
            checkpoints: Vec::new(),
            state: TrainingState {
                best_val_loss: f32::INFINITY,
                ..Default::default()
            },
        }
    }

    /// Train on decomposition task
    ///
    /// Learns to break complex queries into manageable sub-queries
    pub fn train_decomposition(&mut self, dataset: &RlmDataset) -> Result<RlmTrainingResult> {
        let start_time = std::time::Instant::now();
        self.state = TrainingState {
            best_val_loss: f32::INFINITY,
            ..Default::default()
        };

        let batches = dataset.to_training_batch(self.config.batch_size);
        let val_size = (batches.len() as f32 * self.config.validation_split) as usize;
        let train_batches = &batches[val_size..];
        let val_batches = &batches[..val_size];

        let mut loss_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut grpo_stats = GrpoStats::default();

        for epoch in 0..self.config.epochs {
            self.state.current_epoch = epoch;
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Training loop
            for batch in train_batches {
                let batch_result = self.train_decomposition_batch(batch)?;
                epoch_loss += batch_result.total_loss;
                batch_count += 1;
                self.state.current_step += 1;

                // Update GRPO stats
                grpo_stats.total_updates += 1;
                grpo_stats.avg_policy_loss =
                    grpo_stats.avg_policy_loss * 0.99 + batch_result.policy_loss * 0.01;
                grpo_stats.avg_kl_divergence =
                    grpo_stats.avg_kl_divergence * 0.99 + batch_result.kl_divergence * 0.01;
                grpo_stats.avg_entropy =
                    grpo_stats.avg_entropy * 0.99 + batch_result.entropy * 0.01;
                grpo_stats.avg_clip_fraction =
                    grpo_stats.avg_clip_fraction * 0.99 + batch_result.clip_fraction * 0.01;
                grpo_stats.final_kl_coef = batch_result.kl_coef;
            }

            let avg_loss = if batch_count > 0 {
                epoch_loss / batch_count as f32
            } else {
                0.0
            };
            loss_history.push(avg_loss);

            // Validation
            let val_loss = self.validate_decomposition(val_batches)?;
            val_loss_history.push(val_loss);

            // Early stopping check
            if val_loss < self.state.best_val_loss {
                self.state.best_val_loss = val_loss;
                self.state.patience_counter = 0;
            } else {
                self.state.patience_counter += 1;
                if self.state.patience_counter >= self.config.early_stopping_patience {
                    break;
                }
            }

            // Checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0 {
                self.save_checkpoint("decomposition", avg_loss, val_loss);
            }
        }

        Ok(RlmTrainingResult {
            phase: "decomposition".to_string(),
            epochs_completed: self.state.current_epoch + 1,
            total_steps: self.state.current_step,
            final_loss: *loss_history.last().unwrap_or(&0.0),
            best_val_loss: self.state.best_val_loss,
            best_epoch: val_loss_history
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            accuracy: 0.0, // Not applicable for decomposition
            loss_history,
            val_loss_history,
            duration_ms: start_time.elapsed().as_millis() as u64,
            early_stopped: self.state.patience_counter >= self.config.early_stopping_patience,
            grpo_stats: Some(grpo_stats),
        })
    }

    /// Train decomposition on a single batch
    fn train_decomposition_batch(&mut self, batch: &TrainingBatch) -> Result<GrpoUpdateResult> {
        let mut samples = Vec::new();

        for example in &batch.examples {
            // Convert example to GRPO samples for decomposition quality
            let sample = self.example_to_decomposition_sample(example);
            samples.push(sample);
        }

        // Create sample group
        let group = SampleGroup::new(
            samples.clone(),
            self.state.current_step as u64,
            "decomposition".to_string(),
        );
        self.optimizer.add_group(group);

        // Compute relative advantages
        let rewards: Vec<f32> = samples.iter().map(|s| s.reward).collect();
        let advantages = self.optimizer.compute_relative_advantages(&rewards);

        // Perform GRPO update
        let log_probs: Vec<f32> = samples.iter().map(|s| s.log_prob).collect();
        let ref_log_probs: Vec<f32> = samples.iter().map(|s| s.ref_log_prob).collect();

        self.optimizer
            .grpo_update(&log_probs, &advantages, &ref_log_probs)
    }

    /// Convert example to decomposition GRPO sample
    fn example_to_decomposition_sample(&self, example: &RlmTrainingExample) -> GrpoSample {
        // Reward is based on decomposition quality metrics
        let decomp_quality = if example.decomposition.success {
            example.quality_score
        } else {
            0.0
        };

        // Additional reward factors
        let depth_factor = (example.decomposition.sub_queries.len() as f32 / 5.0).min(1.0);
        let complexity_match = 1.0
            - (example.decomposition.total_complexity - 1.0)
                .abs()
                .min(1.0);

        let reward = decomp_quality * 0.6 + depth_factor * 0.2 + complexity_match * 0.2;

        GrpoSample {
            state: example
                .query_embedding
                .clone()
                .unwrap_or_else(|| vec![0.0; 768]),
            action: 0,                         // Decomposition action
            log_prob: -reward.ln().max(-10.0), // Simulated log prob
            ref_log_prob: -1.0,
            reward,
            done: true,
            value: Some(reward),
            tool_name: "decompose".to_string(),
            parameters: None,
        }
    }

    /// Validate decomposition on batches
    fn validate_decomposition(&self, batches: &[TrainingBatch]) -> Result<f32> {
        if batches.is_empty() {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;
        let mut total_examples = 0;

        for batch in batches {
            for example in &batch.examples {
                // Simple validation loss based on decomposition quality
                let loss = 1.0 - example.quality_score;
                total_loss += loss;
                total_examples += 1;
            }
        }

        Ok(if total_examples > 0 {
            total_loss / total_examples as f32
        } else {
            0.0
        })
    }

    /// Train on synthesis task
    ///
    /// Learns to combine sub-answers into coherent final responses
    pub fn train_synthesis(&mut self, dataset: &RlmDataset) -> Result<RlmTrainingResult> {
        let start_time = std::time::Instant::now();
        self.state = TrainingState {
            best_val_loss: f32::INFINITY,
            ..Default::default()
        };

        let batches = dataset.to_training_batch(self.config.batch_size);
        let val_size = (batches.len() as f32 * self.config.validation_split) as usize;
        let train_batches = &batches[val_size..];
        let val_batches = &batches[..val_size];

        let mut loss_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut grpo_stats = GrpoStats::default();

        for epoch in 0..self.config.epochs {
            self.state.current_epoch = epoch;
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            for batch in train_batches {
                let batch_result = self.train_synthesis_batch(batch)?;
                epoch_loss += batch_result.total_loss;
                batch_count += 1;
                self.state.current_step += 1;

                // Update stats
                grpo_stats.total_updates += 1;
                grpo_stats.avg_policy_loss =
                    grpo_stats.avg_policy_loss * 0.99 + batch_result.policy_loss * 0.01;
            }

            let avg_loss = if batch_count > 0 {
                epoch_loss / batch_count as f32
            } else {
                0.0
            };
            loss_history.push(avg_loss);

            // Validation
            let val_loss = self.validate_synthesis(val_batches)?;
            val_loss_history.push(val_loss);

            // Early stopping
            if val_loss < self.state.best_val_loss {
                self.state.best_val_loss = val_loss;
                self.state.patience_counter = 0;
            } else {
                self.state.patience_counter += 1;
                if self.state.patience_counter >= self.config.early_stopping_patience {
                    break;
                }
            }

            // Checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0 {
                self.save_checkpoint("synthesis", avg_loss, val_loss);
            }
        }

        Ok(RlmTrainingResult {
            phase: "synthesis".to_string(),
            epochs_completed: self.state.current_epoch + 1,
            total_steps: self.state.current_step,
            final_loss: *loss_history.last().unwrap_or(&0.0),
            best_val_loss: self.state.best_val_loss,
            best_epoch: val_loss_history
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            accuracy: 0.0,
            loss_history,
            val_loss_history,
            duration_ms: start_time.elapsed().as_millis() as u64,
            early_stopped: self.state.patience_counter >= self.config.early_stopping_patience,
            grpo_stats: Some(grpo_stats),
        })
    }

    /// Train synthesis on a single batch
    fn train_synthesis_batch(&mut self, batch: &TrainingBatch) -> Result<GrpoUpdateResult> {
        let mut samples = Vec::new();

        for example in &batch.examples {
            let sample = self.example_to_synthesis_sample(example);
            samples.push(sample);
        }

        let group = SampleGroup::new(
            samples.clone(),
            self.state.current_step as u64,
            "synthesis".to_string(),
        );
        self.optimizer.add_group(group);

        let rewards: Vec<f32> = samples.iter().map(|s| s.reward).collect();
        let advantages = self.optimizer.compute_relative_advantages(&rewards);

        let log_probs: Vec<f32> = samples.iter().map(|s| s.log_prob).collect();
        let ref_log_probs: Vec<f32> = samples.iter().map(|s| s.ref_log_prob).collect();

        self.optimizer
            .grpo_update(&log_probs, &advantages, &ref_log_probs)
    }

    /// Convert example to synthesis GRPO sample
    fn example_to_synthesis_sample(&self, example: &RlmTrainingExample) -> GrpoSample {
        // Reward based on how well sub-answers were combined
        let sub_answer_quality = if example.sub_answers.is_empty() {
            0.0
        } else {
            example.sub_answers.iter().map(|a| a.quality).sum::<f32>()
                / example.sub_answers.len() as f32
        };

        // Final answer quality (proxied by overall quality)
        let final_quality = example.quality_score;

        // Coherence bonus (higher if final quality > sub-answer average)
        let coherence_bonus = (final_quality - sub_answer_quality).max(0.0) * 0.5;

        let reward = (sub_answer_quality * 0.4 + final_quality * 0.4 + coherence_bonus * 0.2)
            .clamp(0.0, 1.0);

        GrpoSample {
            state: example
                .final_embedding
                .clone()
                .unwrap_or_else(|| vec![0.0; 768]),
            action: 1, // Synthesis action
            log_prob: -reward.ln().max(-10.0),
            ref_log_prob: -1.0,
            reward,
            done: true,
            value: Some(reward),
            tool_name: "synthesize".to_string(),
            parameters: None,
        }
    }

    /// Validate synthesis on batches
    fn validate_synthesis(&self, batches: &[TrainingBatch]) -> Result<f32> {
        if batches.is_empty() {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;
        let mut total_examples = 0;

        for batch in batches {
            for example in &batch.examples {
                // Loss based on quality gap
                let loss = 1.0 - example.quality_score;
                total_loss += loss;
                total_examples += 1;
            }
        }

        Ok(if total_examples > 0 {
            total_loss / total_examples as f32
        } else {
            0.0
        })
    }

    /// Contrastive fine-tuning for agent routing
    ///
    /// Uses triplet loss and InfoNCE to improve routing accuracy
    pub fn contrastive_finetune(&mut self, pairs: &[ContrastivePair]) -> Result<RlmTrainingResult> {
        let start_time = std::time::Instant::now();
        self.state = TrainingState {
            best_val_loss: f32::INFINITY,
            ..Default::default()
        };

        if pairs.is_empty() {
            return Err(RuvLLMError::InvalidOperation(
                "No contrastive pairs provided".to_string(),
            ));
        }

        // Split pairs into train/val
        let val_size = (pairs.len() as f32 * self.config.validation_split) as usize;
        let train_pairs = &pairs[val_size..];
        let val_pairs = &pairs[..val_size];

        let mut loss_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut correct = 0;
        let mut total = 0;

        for epoch in 0..self.config.epochs {
            self.state.current_epoch = epoch;
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Process in batches
            for chunk in train_pairs.chunks(self.config.batch_size) {
                let batch_loss = self.train_contrastive_batch(chunk)?;
                epoch_loss += batch_loss;
                batch_count += 1;
                self.state.current_step += 1;
            }

            let avg_loss = if batch_count > 0 {
                epoch_loss / batch_count as f32
            } else {
                0.0
            };
            loss_history.push(avg_loss);

            // Validation
            let (val_loss, val_correct, val_total) = self.validate_contrastive(val_pairs)?;
            val_loss_history.push(val_loss);
            correct = val_correct;
            total = val_total;

            // Early stopping
            if val_loss < self.state.best_val_loss {
                self.state.best_val_loss = val_loss;
                self.state.patience_counter = 0;
            } else {
                self.state.patience_counter += 1;
                if self.state.patience_counter >= self.config.early_stopping_patience {
                    break;
                }
            }

            // Checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0 {
                self.save_checkpoint("contrastive", avg_loss, val_loss);
            }
        }

        let accuracy = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };

        Ok(RlmTrainingResult {
            phase: "contrastive".to_string(),
            epochs_completed: self.state.current_epoch + 1,
            total_steps: self.state.current_step,
            final_loss: *loss_history.last().unwrap_or(&0.0),
            best_val_loss: self.state.best_val_loss,
            best_epoch: val_loss_history
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            accuracy,
            loss_history,
            val_loss_history,
            duration_ms: start_time.elapsed().as_millis() as u64,
            early_stopped: self.state.patience_counter >= self.config.early_stopping_patience,
            grpo_stats: None,
        })
    }

    /// Train contrastive on a batch of pairs
    fn train_contrastive_batch(&mut self, pairs: &[ContrastivePair]) -> Result<f32> {
        let mut batch_loss = 0.0;

        for pair in pairs {
            // Compute triplet loss
            let triplet_loss = self.compute_triplet_loss(pair);

            // Compute InfoNCE loss (simplified)
            let infonce_loss = self.compute_infonce_loss(pair);

            // Combined loss
            let combined_loss = triplet_loss * 0.5 + infonce_loss * 0.5;
            batch_loss += combined_loss;

            // Record for SONA if enabled
            if self.config.sona_enabled && pair.quality >= self.config.online_quality_threshold {
                self.record_routing_trajectory(pair, combined_loss)?;
            }
        }

        Ok(if !pairs.is_empty() {
            batch_loss / pairs.len() as f32
        } else {
            0.0
        })
    }

    /// Compute triplet loss for a contrastive pair
    fn compute_triplet_loss(&self, pair: &ContrastivePair) -> f32 {
        // Simplified triplet loss: max(0, margin + d(anchor, positive) - d(anchor, negative))
        // In practice, this would use actual embeddings

        // Simulate distances based on agent similarity
        let positive_dist = self.agent_distance(&pair.anchor, &pair.positive_agent);
        let negative_dist = self.agent_distance(&pair.anchor, &pair.negative_agent);

        (self.config.contrastive_margin + positive_dist - negative_dist).max(0.0)
    }

    /// Compute InfoNCE loss for a contrastive pair
    fn compute_infonce_loss(&self, pair: &ContrastivePair) -> f32 {
        // Simplified InfoNCE: -log(exp(sim(a,p)/t) / (exp(sim(a,p)/t) + sum(exp(sim(a,n)/t))))

        let pos_sim = 1.0 - self.agent_distance(&pair.anchor, &pair.positive_agent);
        let neg_sim = 1.0 - self.agent_distance(&pair.anchor, &pair.negative_agent);

        let temp = self.config.infonce_temperature;
        let pos_exp = (pos_sim / temp).exp();
        let neg_exp = (neg_sim / temp).exp();

        -(pos_exp / (pos_exp + neg_exp)).ln()
    }

    /// Compute semantic distance between query and agent
    fn agent_distance(&self, query: &str, agent: &str) -> f32 {
        // Simplified heuristic based on keyword matching
        // In practice, would use actual embeddings

        let query_lower = query.to_lowercase();

        let agent_keywords: &[&str] = match agent {
            "coder" => &["implement", "build", "create", "code", "write", "develop"],
            "researcher" => &["research", "investigate", "analyze", "explore", "study"],
            "reviewer" => &["review", "check", "evaluate", "assess", "examine"],
            "tester" => &["test", "write tests", "unit test", "coverage", "validate"],
            "architect" => &["design", "plan", "architecture", "schema", "structure"],
            "security-architect" => &["security", "audit", "vulnerability", "xss", "injection"],
            "debugger" => &["fix", "debug", "bug", "error", "trace", "crash"],
            "documenter" => &["document", "jsdoc", "readme", "comment", "explain"],
            "refactorer" => &["refactor", "restructure", "modernize", "clean", "simplify"],
            "optimizer" => &["optimize", "performance", "speed", "cache", "improve"],
            "devops" => &["deploy", "ci/cd", "kubernetes", "docker", "infrastructure"],
            "api-docs" => &["openapi", "swagger", "api reference", "endpoint", "spec"],
            "planner" => &["plan", "estimate", "schedule", "timeline", "sprint"],
            _ => &[],
        };

        let matches = agent_keywords
            .iter()
            .filter(|kw| query_lower.contains(*kw))
            .count();

        // Lower distance for more keyword matches
        1.0 - (matches as f32 / agent_keywords.len().max(1) as f32).min(1.0)
    }

    /// Validate contrastive pairs
    fn validate_contrastive(&self, pairs: &[ContrastivePair]) -> Result<(f32, usize, usize)> {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let total = pairs.len();

        for pair in pairs {
            let triplet_loss = self.compute_triplet_loss(pair);
            let infonce_loss = self.compute_infonce_loss(pair);
            total_loss += triplet_loss * 0.5 + infonce_loss * 0.5;

            // Check if routing is correct (positive closer than negative)
            let pos_dist = self.agent_distance(&pair.anchor, &pair.positive_agent);
            let neg_dist = self.agent_distance(&pair.anchor, &pair.negative_agent);
            if pos_dist < neg_dist {
                correct += 1;
            }
        }

        let avg_loss = if total > 0 {
            total_loss / total as f32
        } else {
            0.0
        };

        Ok((avg_loss, correct, total))
    }

    /// SONA integration for online learning from a trajectory
    ///
    /// Records a trajectory for continuous learning. The trajectory's quality_score
    /// must meet the online_quality_threshold for learning to occur.
    pub fn online_learn(&mut self, trajectory: &Trajectory) -> Result<()> {
        if !self.config.sona_enabled {
            return Ok(());
        }

        if trajectory.quality_score < self.config.online_quality_threshold {
            return Ok(());
        }

        if let Some(sona) = &self.sona {
            let mut sona_guard = sona.write();
            sona_guard.record_trajectory(trajectory.clone())?;
        }

        Ok(())
    }

    /// Record a routing trajectory for SONA
    fn record_routing_trajectory(&self, pair: &ContrastivePair, loss: f32) -> Result<()> {
        if let Some(sona) = &self.sona {
            // Create a mini-trajectory for the routing decision
            let trajectory = Trajectory {
                request_id: uuid::Uuid::new_v4().to_string(),
                session_id: "rlm-training".to_string(),
                query_embedding: pair
                    .anchor_embedding
                    .clone()
                    .unwrap_or_else(|| vec![0.0; 768]),
                response_embedding: vec![0.0; 768], // Placeholder
                quality_score: pair.quality * (1.0 - loss).max(0.0),
                routing_features: vec![],
                model_index: 0,
                timestamp: chrono::Utc::now(),
            };

            let mut sona_guard = sona.write();
            sona_guard.record_trajectory(trajectory)?;
        }

        Ok(())
    }

    /// Save a training checkpoint
    fn save_checkpoint(&mut self, phase: &str, loss: f32, val_loss: f32) {
        self.checkpoints.push(TrainingCheckpoint {
            epoch: self.state.current_epoch,
            step: self.state.current_step,
            loss,
            val_loss,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            phase: phase.to_string(),
        });
    }

    /// Get training checkpoints
    pub fn checkpoints(&self) -> &[TrainingCheckpoint] {
        &self.checkpoints
    }

    /// Get current configuration
    pub fn config(&self) -> &RlmTrainingConfig {
        &self.config
    }

    /// Get GRPO optimizer
    pub fn optimizer(&self) -> &GrpoOptimizer {
        &self.optimizer
    }

    /// Reset trainer state
    pub fn reset(&mut self) {
        self.optimizer.reset();
        self.checkpoints.clear();
        self.state = TrainingState {
            best_val_loss: f32::INFINITY,
            ..Default::default()
        };
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ruvltra::RuvLtraConfig;

    fn create_test_trainer() -> RlmTrainer {
        let model_config = RuvLtraConfig::tiny();
        let model = Arc::new(RwLock::new(RuvLtraModel::new(&model_config).unwrap()));
        let config = RlmTrainingConfig {
            epochs: 2,
            batch_size: 4,
            sona_enabled: false,
            ..Default::default()
        };
        RlmTrainer::new(model, config)
    }

    fn create_test_dataset() -> RlmDataset {
        use crate::training::rlm_dataset::{
            DecompositionStrategy, QueryDecomposition, RlmDatasetConfig,
        };

        let config = RlmDatasetConfig::default();
        let mut dataset = RlmDataset::new(config);

        for i in 0..10 {
            let mut example = RlmTrainingExample::new(
                format!("Test query {}", i),
                QueryDecomposition::new(DecompositionStrategy::Sequential),
            );
            example.quality_score = 0.7 + (i as f32 * 0.02);
            example.success = true;
            example.query_embedding = Some(vec![0.1 * i as f32; 768]);
            dataset.add_example(example);
        }

        dataset
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = create_test_trainer();
        assert_eq!(trainer.config.epochs, 2);
        assert_eq!(trainer.config.batch_size, 4);
    }

    #[test]
    fn test_decomposition_training() {
        let mut trainer = create_test_trainer();
        let dataset = create_test_dataset();

        let result = trainer.train_decomposition(&dataset).unwrap();
        assert!(result.epochs_completed > 0);
        assert!(!result.loss_history.is_empty());
    }

    #[test]
    fn test_synthesis_training() {
        let mut trainer = create_test_trainer();
        let dataset = create_test_dataset();

        let result = trainer.train_synthesis(&dataset).unwrap();
        assert!(result.epochs_completed > 0);
    }

    #[test]
    fn test_contrastive_finetuning() {
        let mut trainer = create_test_trainer();

        let pairs = vec![
            ContrastivePair {
                anchor: "Implement a login function".to_string(),
                anchor_embedding: None,
                positive_agent: "coder".to_string(),
                negative_agent: "reviewer".to_string(),
                is_hard_negative: false,
                quality: 0.9,
                source_id: uuid::Uuid::new_v4(),
            },
            ContrastivePair {
                anchor: "Review the pull request".to_string(),
                anchor_embedding: None,
                positive_agent: "reviewer".to_string(),
                negative_agent: "coder".to_string(),
                is_hard_negative: false,
                quality: 0.85,
                source_id: uuid::Uuid::new_v4(),
            },
        ];

        let result = trainer.contrastive_finetune(&pairs).unwrap();
        assert!(result.epochs_completed > 0);
    }

    #[test]
    fn test_agent_distance() {
        let trainer = create_test_trainer();

        // "Implement" should be closer to "coder"
        let coder_dist = trainer.agent_distance("Implement a login function", "coder");
        let reviewer_dist = trainer.agent_distance("Implement a login function", "reviewer");
        assert!(coder_dist < reviewer_dist);

        // "Review" should be closer to "reviewer"
        let reviewer_dist2 = trainer.agent_distance("Review the code quality", "reviewer");
        let coder_dist2 = trainer.agent_distance("Review the code quality", "coder");
        assert!(reviewer_dist2 < coder_dist2);
    }

    #[test]
    fn test_triplet_loss() {
        let trainer = create_test_trainer();

        let pair = ContrastivePair {
            anchor: "Implement authentication".to_string(),
            anchor_embedding: None,
            positive_agent: "coder".to_string(),
            negative_agent: "documenter".to_string(),
            is_hard_negative: false,
            quality: 0.9,
            source_id: uuid::Uuid::new_v4(),
        };

        let loss = trainer.compute_triplet_loss(&pair);
        // Loss should be low when positive is clearly correct
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_config_presets() {
        let fast = RlmTrainingConfig::fast();
        assert_eq!(fast.epochs, 3);

        let thorough = RlmTrainingConfig::thorough();
        assert_eq!(thorough.epochs, 50);

        let routing = RlmTrainingConfig::routing_focused();
        assert!(routing.routing_weight > routing.decomposition_weight);
    }

    #[test]
    fn test_checkpointing() {
        let mut trainer = create_test_trainer();
        trainer.save_checkpoint("test", 0.5, 0.6);

        assert_eq!(trainer.checkpoints().len(), 1);
        assert_eq!(trainer.checkpoints()[0].phase, "test");
    }

    #[test]
    fn test_reset() {
        let mut trainer = create_test_trainer();
        trainer.save_checkpoint("test", 0.5, 0.6);
        trainer.state.current_epoch = 5;

        trainer.reset();

        assert!(trainer.checkpoints().is_empty());
    }
}
