//! Quality Scoring
//!
//! Quality assessment for RLM answers.

use serde::{Deserialize, Serialize};

/// Configuration for quality scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorerConfig {
    /// Weight for answer completeness.
    pub completeness_weight: f32,

    /// Weight for answer coherence.
    pub coherence_weight: f32,

    /// Weight for answer relevance.
    pub relevance_weight: f32,

    /// Weight for confidence signals.
    pub confidence_weight: f32,

    /// Minimum acceptable length for answers.
    pub min_answer_length: usize,

    /// Maximum answer length before penalty.
    pub max_answer_length: usize,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            completeness_weight: 0.3,
            coherence_weight: 0.25,
            relevance_weight: 0.3,
            confidence_weight: 0.15,
            min_answer_length: 10,
            max_answer_length: 2000,
        }
    }
}

/// Quality scorer for RLM answers.
#[derive(Debug, Clone)]
pub struct QualityScorer {
    /// Configuration.
    config: ScorerConfig,
}

impl QualityScorer {
    /// Create a new quality scorer with default configuration.
    pub fn new() -> Self {
        Self {
            config: ScorerConfig::default(),
        }
    }

    /// Create a new quality scorer with custom configuration.
    pub fn with_config(config: ScorerConfig) -> Self {
        Self { config }
    }

    /// Score the quality of an answer.
    pub fn score(&self, context: &ScoringContext) -> QualityScore {
        let completeness = self.score_completeness(context);
        let coherence = self.score_coherence(context);
        let relevance = self.score_relevance(context);
        let confidence = self.score_confidence(context);

        let overall = completeness * self.config.completeness_weight
            + coherence * self.config.coherence_weight
            + relevance * self.config.relevance_weight
            + confidence * self.config.confidence_weight;

        QualityScore {
            overall,
            completeness,
            coherence,
            relevance,
            confidence,
            details: self.generate_details(context, overall),
        }
    }

    /// Score answer completeness.
    fn score_completeness(&self, context: &ScoringContext) -> f32 {
        let answer_len = context.answer.len();

        if answer_len < self.config.min_answer_length {
            return 0.2;
        }

        if answer_len > self.config.max_answer_length {
            // Slight penalty for very long answers
            return 0.8 - (answer_len - self.config.max_answer_length) as f32 * 0.0001;
        }

        // Ideal length range
        let ideal_min = self.config.min_answer_length * 5;
        let ideal_max = self.config.max_answer_length / 2;

        if answer_len >= ideal_min && answer_len <= ideal_max {
            1.0
        } else if answer_len < ideal_min {
            0.5 + 0.5 * (answer_len as f32 / ideal_min as f32)
        } else {
            0.9
        }
    }

    /// Score answer coherence.
    fn score_coherence(&self, context: &ScoringContext) -> f32 {
        // Simple heuristics for coherence
        let answer = &context.answer;
        let mut score = 0.7f32;

        // Check for sentence structure
        let sentences: Vec<&str> = answer.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if sentences.len() >= 2 {
            score += 0.1;
        }

        // Check for capitalization patterns
        let starts_capitalized = answer.chars().next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);

        if starts_capitalized {
            score += 0.05;
        }

        // Check for proper ending
        if answer.ends_with('.') || answer.ends_with('!') || answer.ends_with('?') {
            score += 0.05;
        }

        // Penalize repetition
        let words: Vec<&str> = answer.split_whitespace().collect();
        if words.len() > 10 {
            let unique_words: std::collections::HashSet<&str> = words.iter().copied().collect();
            let repetition_ratio = unique_words.len() as f32 / words.len() as f32;
            if repetition_ratio < 0.5 {
                score -= 0.2;
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Score answer relevance to the query.
    fn score_relevance(&self, context: &ScoringContext) -> f32 {
        let query_words: std::collections::HashSet<String> = context.query
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2) // Skip short words
            .map(String::from)
            .collect();

        let answer_lower = context.answer.to_lowercase();

        if query_words.is_empty() {
            return 0.5;
        }

        let matched_words = query_words.iter()
            .filter(|w| answer_lower.contains(w.as_str()))
            .count();

        let coverage = matched_words as f32 / query_words.len() as f32;

        // Base relevance from word coverage
        let mut score = 0.3 + 0.5 * coverage;

        // Bonus for semantic signals
        let semantic_signals = ["because", "therefore", "however", "specifically", "for example"];
        for signal in semantic_signals {
            if answer_lower.contains(signal) {
                score += 0.05;
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Score confidence signals in the answer.
    fn score_confidence(&self, context: &ScoringContext) -> f32 {
        let answer_lower = context.answer.to_lowercase();
        let mut score = 0.7f32;

        // Negative confidence signals
        let uncertainty_phrases = [
            "i'm not sure", "i don't know", "possibly", "maybe",
            "it's unclear", "uncertain", "might be", "could be"
        ];

        for phrase in uncertainty_phrases {
            if answer_lower.contains(phrase) {
                score -= 0.15;
            }
        }

        // Positive confidence signals
        let confidence_phrases = [
            "is defined as", "specifically", "in particular",
            "the answer is", "research shows", "according to"
        ];

        for phrase in confidence_phrases {
            if answer_lower.contains(phrase) {
                score += 0.1;
            }
        }

        // Use provided confidence if available
        if let Some(provided_conf) = context.provided_confidence {
            score = 0.5 * score + 0.5 * provided_conf;
        }

        score.clamp(0.0, 1.0)
    }

    /// Generate quality assessment details.
    fn generate_details(&self, context: &ScoringContext, overall: f32) -> QualityDetails {
        let mut issues = Vec::new();
        let mut strengths = Vec::new();

        // Check length
        if context.answer.len() < self.config.min_answer_length {
            issues.push("Answer is too short".to_string());
        } else if context.answer.len() > self.config.max_answer_length {
            issues.push("Answer may be unnecessarily verbose".to_string());
        } else {
            strengths.push("Answer has appropriate length".to_string());
        }

        // Check query word coverage
        let query_words: Vec<&str> = context.query.split_whitespace().collect();
        let answer_lower = context.answer.to_lowercase();
        let covered = query_words.iter()
            .filter(|w| answer_lower.contains(&w.to_lowercase()))
            .count();

        if covered < query_words.len() / 2 {
            issues.push("Answer may not fully address the query".to_string());
        } else {
            strengths.push("Answer addresses key query terms".to_string());
        }

        // Overall assessment
        let quality_level = if overall >= 0.8 {
            QualityLevel::High
        } else if overall >= 0.6 {
            QualityLevel::Good
        } else if overall >= 0.4 {
            QualityLevel::Acceptable
        } else {
            QualityLevel::Low
        };

        QualityDetails {
            quality_level,
            issues,
            strengths,
            suggestion: if overall < 0.6 {
                Some("Consider providing more detailed and specific information".to_string())
            } else {
                None
            },
        }
    }
}

impl Default for QualityScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for quality scoring.
#[derive(Debug, Clone)]
pub struct ScoringContext {
    /// The original query.
    pub query: String,

    /// The generated answer.
    pub answer: String,

    /// Provided confidence score (optional).
    pub provided_confidence: Option<f32>,

    /// Sub-query answers (if aggregated).
    pub sub_answers: Vec<String>,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

impl ScoringContext {
    /// Create a new scoring context.
    pub fn new(query: &str, answer: &str) -> Self {
        Self {
            query: query.to_string(),
            answer: answer.to_string(),
            provided_confidence: None,
            sub_answers: Vec::new(),
            processing_time_ms: 0,
        }
    }

    /// Set provided confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.provided_confidence = Some(confidence);
        self
    }

    /// Add sub-answers.
    pub fn with_sub_answers(mut self, sub_answers: Vec<String>) -> Self {
        self.sub_answers = sub_answers;
        self
    }

    /// Set processing time.
    pub fn with_processing_time(mut self, time_ms: u64) -> Self {
        self.processing_time_ms = time_ms;
        self
    }
}

/// Quality score result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    /// Overall quality score (0.0 - 1.0).
    pub overall: f32,

    /// Completeness score.
    pub completeness: f32,

    /// Coherence score.
    pub coherence: f32,

    /// Relevance score.
    pub relevance: f32,

    /// Confidence score.
    pub confidence: f32,

    /// Detailed assessment.
    pub details: QualityDetails,
}

impl QualityScore {
    /// Check if the score meets a quality threshold.
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.overall >= threshold
    }

    /// Get the quality level.
    pub fn level(&self) -> QualityLevel {
        self.details.quality_level
    }
}

/// Detailed quality assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDetails {
    /// Quality level category.
    pub quality_level: QualityLevel,

    /// List of identified issues.
    pub issues: Vec<String>,

    /// List of identified strengths.
    pub strengths: Vec<String>,

    /// Improvement suggestion (if applicable).
    pub suggestion: Option<String>,
}

/// Quality level categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    /// High quality (>= 0.8).
    High,

    /// Good quality (>= 0.6).
    Good,

    /// Acceptable quality (>= 0.4).
    Acceptable,

    /// Low quality (< 0.4).
    Low,
}

impl QualityLevel {
    /// Get the minimum threshold for this level.
    pub fn min_threshold(&self) -> f32 {
        match self {
            Self::High => 0.8,
            Self::Good => 0.6,
            Self::Acceptable => 0.4,
            Self::Low => 0.0,
        }
    }

    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::High => "High quality - meets all criteria",
            Self::Good => "Good quality - meets most criteria",
            Self::Acceptable => "Acceptable - meets minimum requirements",
            Self::Low => "Low quality - needs improvement",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scorer_creation() {
        let scorer = QualityScorer::new();
        assert_eq!(scorer.config.completeness_weight, 0.3);
    }

    #[test]
    fn test_scoring_good_answer() {
        let scorer = QualityScorer::new();
        let context = ScoringContext::new(
            "What is artificial intelligence?",
            "Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. AI is used in various applications including natural language processing, computer vision, and robotics."
        );

        let score = scorer.score(&context);
        assert!(score.overall >= 0.6);
        assert_eq!(score.details.quality_level, QualityLevel::Good);
    }

    #[test]
    fn test_scoring_short_answer() {
        let scorer = QualityScorer::new();
        let context = ScoringContext::new(
            "What is AI?",
            "AI."
        );

        let score = scorer.score(&context);
        assert!(score.overall < 0.6);
        assert!(score.completeness < 0.5);
    }

    #[test]
    fn test_scoring_with_confidence() {
        let scorer = QualityScorer::new();
        let context = ScoringContext::new(
            "What is machine learning?",
            "Machine learning is a subset of AI that enables systems to learn from data."
        ).with_confidence(0.9);

        let score = scorer.score(&context);
        assert!(score.confidence > 0.7);
    }

    #[test]
    fn test_quality_levels() {
        assert_eq!(QualityLevel::High.min_threshold(), 0.8);
        assert_eq!(QualityLevel::Good.min_threshold(), 0.6);
        assert_eq!(QualityLevel::Acceptable.min_threshold(), 0.4);
        assert_eq!(QualityLevel::Low.min_threshold(), 0.0);
    }

    #[test]
    fn test_quality_score_threshold() {
        let score = QualityScore {
            overall: 0.75,
            completeness: 0.8,
            coherence: 0.7,
            relevance: 0.75,
            confidence: 0.7,
            details: QualityDetails {
                quality_level: QualityLevel::Good,
                issues: vec![],
                strengths: vec![],
                suggestion: None,
            },
        };

        assert!(score.meets_threshold(0.6));
        assert!(score.meets_threshold(0.75));
        assert!(!score.meets_threshold(0.8));
    }

    #[test]
    fn test_uncertainty_detection() {
        let scorer = QualityScorer::new();

        let uncertain = ScoringContext::new(
            "What is quantum computing?",
            "I'm not sure, but it might be related to quantum physics."
        );
        let score_uncertain = scorer.score(&uncertain);

        let confident = ScoringContext::new(
            "What is quantum computing?",
            "Quantum computing is specifically defined as a type of computation that harnesses quantum mechanical phenomena."
        );
        let score_confident = scorer.score(&confident);

        assert!(score_confident.confidence > score_uncertain.confidence);
    }

    #[test]
    fn test_relevance_scoring() {
        let scorer = QualityScorer::new();

        // Relevant answer
        let relevant = ScoringContext::new(
            "What is Python programming?",
            "Python is a high-level programming language known for its simplicity and readability."
        );
        let relevant_score = scorer.score(&relevant);

        // Irrelevant answer
        let irrelevant = ScoringContext::new(
            "What is Python programming?",
            "The weather today is sunny with a high of 75 degrees."
        );
        let irrelevant_score = scorer.score(&irrelevant);

        assert!(relevant_score.relevance > irrelevant_score.relevance);
    }
}
