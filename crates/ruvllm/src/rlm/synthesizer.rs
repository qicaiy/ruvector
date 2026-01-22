//! Answer Synthesis
//!
//! Synthesizes partial answers from sub-queries into coherent final responses.
//! Optimized for minimal string allocations and efficient aggregation.

use super::config::AggregationStrategy;
#[allow(unused_imports)]
use super::decomposer::{DecompositionStrategy, SubQuery};
#[allow(unused_imports)]
use super::types::{RlmAnswer, SubAnswer, TokenUsage};
use crate::error::Result;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::borrow::Cow;

/// Answer synthesizer for combining sub-query results.
#[derive(Debug, Clone)]
pub struct AnswerSynthesizer {
    /// Configuration reference.
    strategy: AggregationStrategy,
}

impl AnswerSynthesizer {
    /// Create a new answer synthesizer with the given strategy.
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self { strategy }
    }

    /// Create a synthesizer with default strategy.
    pub fn default_strategy() -> Self {
        Self::new(AggregationStrategy::WeightedMerge)
    }

    /// Synthesize sub-answers into a final answer.
    pub fn synthesize(
        &self,
        original_query: &str,
        decomposition_strategy: &DecompositionStrategy,
        sub_answers: Vec<SubAnswer>,
    ) -> Result<SynthesisResult> {
        if sub_answers.is_empty() {
            return Ok(SynthesisResult {
                text: "Unable to generate an answer.".to_string(),
                confidence: 0.0,
                token_usage: TokenUsage::default(),
                sub_answers_used: 0,
            });
        }

        // Single answer - no synthesis needed
        if sub_answers.len() == 1 {
            let answer = &sub_answers[0];
            return Ok(SynthesisResult {
                text: answer.answer.clone(),
                confidence: answer.confidence,
                token_usage: TokenUsage {
                    output_tokens: answer.tokens_used,
                    ..Default::default()
                },
                sub_answers_used: 1,
            });
        }

        // Apply synthesis strategy
        match self.strategy {
            AggregationStrategy::Concatenate => {
                self.synthesize_concatenate(original_query, decomposition_strategy, &sub_answers)
            }
            AggregationStrategy::WeightedMerge => {
                self.synthesize_weighted_merge(original_query, decomposition_strategy, &sub_answers)
            }
            AggregationStrategy::Summarize => {
                self.synthesize_summarize(original_query, decomposition_strategy, &sub_answers)
            }
            AggregationStrategy::BestAnswer => self.synthesize_best_answer(&sub_answers),
            AggregationStrategy::Consensus => self.synthesize_consensus(&sub_answers),
        }
    }

    /// Synthesize by concatenating answers.
    /// Optimized: pre-calculates capacity and uses efficient string building.
    fn synthesize_concatenate(
        &self,
        _original_query: &str,
        decomposition_strategy: &DecompositionStrategy,
        sub_answers: &[SubAnswer],
    ) -> Result<SynthesisResult> {
        // Calculate total text length for capacity hint
        let total_len: usize = sub_answers.iter().map(|a| a.answer.len() + 20).sum();
        let mut text = String::with_capacity(total_len);

        let num_answers = sub_answers.len();

        match decomposition_strategy {
            DecompositionStrategy::Sequential(_) => {
                // Numbered steps for sequential
                for (i, answer) in sub_answers.iter().enumerate() {
                    if i > 0 {
                        text.push_str("\n\n");
                    }
                    // Avoid format! by building string manually
                    text.push_str(&(i + 1).to_string());
                    text.push_str(". ");
                    text.push_str(&answer.answer);
                }
            }
            DecompositionStrategy::Comparison(_) => {
                // Comparison format
                text.push_str("Comparison:");
                for answer in sub_answers {
                    text.push_str("\n\n**");
                    text.push_str(&answer.sub_query);
                    text.push_str("**\n");
                    text.push_str(&answer.answer);
                }
            }
            _ => {
                // Default concatenation with line breaks
                for (i, answer) in sub_answers.iter().enumerate() {
                    if i > 0 {
                        text.push_str("\n\n");
                    }
                    text.push_str(&answer.answer);
                }
            }
        }

        let avg_confidence =
            sub_answers.iter().map(|a| a.confidence).sum::<f32>() / num_answers as f32;

        let total_tokens: usize = sub_answers.iter().map(|a| a.tokens_used).sum();

        Ok(SynthesisResult {
            text,
            confidence: avg_confidence,
            token_usage: TokenUsage {
                synthesis_tokens: 0, // Simple concat uses no synthesis tokens
                output_tokens: total_tokens,
                ..Default::default()
            },
            sub_answers_used: num_answers,
        })
    }

    /// Synthesize with weighted merge based on confidence.
    /// Optimized: uses sort_unstable and pre-allocated string building.
    fn synthesize_weighted_merge(
        &self,
        original_query: &str,
        decomposition_strategy: &DecompositionStrategy,
        sub_answers: &[SubAnswer],
    ) -> Result<SynthesisResult> {
        let num_answers = sub_answers.len();

        // Sort by confidence (descending) - use unstable sort for better perf
        let mut sorted_answers: Vec<_> = sub_answers.iter().collect();
        sorted_answers.sort_unstable_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pre-calculate capacity
        let truncated_query = truncate_query_fast(original_query);
        let total_len: usize = sub_answers
            .iter()
            .map(|a| a.answer.len() + 30)
            .sum::<usize>()
            + 100;
        let mut text = String::with_capacity(total_len);

        match decomposition_strategy {
            DecompositionStrategy::Conjunction(_) => {
                // Merge conjunction parts
                text.push_str("Regarding your question about \"");
                text.push_str(&truncated_query);
                text.push_str("\":");
                for answer in sorted_answers.iter() {
                    if answer.confidence >= 0.5 {
                        text.push_str("\n\n- ");
                        text.push_str(&answer.answer);
                    }
                }
            }
            DecompositionStrategy::Comparison(items) => {
                // Structured comparison
                text.push_str("Here's a comparison:");
                for (i, answer) in sorted_answers.iter().enumerate() {
                    if i < items.len() {
                        text.push_str("\n\n**");
                        text.push_str(
                            items
                                .get(i)
                                .map(|s| s.as_str())
                                .unwrap_or(&answer.sub_query),
                        );
                        text.push_str("**\n");
                        text.push_str(&answer.answer);
                    }
                }
            }
            DecompositionStrategy::Sequential(_) => {
                // Ordered steps
                text.push_str("To address \"");
                text.push_str(&truncated_query);
                text.push_str("\":");
                for (i, answer) in sorted_answers.iter().enumerate() {
                    text.push_str("\n\nStep ");
                    text.push_str(&(i + 1).to_string());
                    text.push_str(": ");
                    text.push_str(&answer.answer);
                }
            }
            _ => {
                // Default weighted merge
                text.push_str("In response to \"");
                text.push_str(&truncated_query);
                text.push_str("\":");
                for (i, answer) in sorted_answers.iter().take(3).enumerate() {
                    if i > 0 {
                        text.push_str("\n\n");
                    } else {
                        text.push_str("\n\n");
                    }
                    text.push_str(&answer.answer);
                }
            }
        }

        // Calculate weighted confidence using harmonic-like weights
        // Optimized: pre-compute reciprocals
        let mut total_weight = 0.0f32;
        let mut weighted_sum = 0.0f32;
        for (i, answer) in sorted_answers.iter().enumerate() {
            let weight = 1.0 / (i as f32 + 1.0);
            total_weight += weight;
            weighted_sum += answer.confidence * weight;
        }
        let weighted_confidence = weighted_sum / total_weight;

        let total_tokens: usize = sub_answers.iter().map(|a| a.tokens_used).sum();

        Ok(SynthesisResult {
            text,
            confidence: weighted_confidence,
            token_usage: TokenUsage {
                synthesis_tokens: 50, // Estimate for merge overhead
                output_tokens: total_tokens,
                ..Default::default()
            },
            sub_answers_used: num_answers,
        })
    }

    /// Synthesize by summarizing all answers (placeholder - would use LLM).
    fn synthesize_summarize(
        &self,
        original_query: &str,
        _decomposition_strategy: &DecompositionStrategy,
        sub_answers: &[SubAnswer],
    ) -> Result<SynthesisResult> {
        // In production, this would call an LLM to summarize
        // For now, we'll create a structured summary

        let mut parts = Vec::new();
        parts.push(format!(
            "Summary for \"{}\":",
            truncate_query(original_query)
        ));

        // Extract key points from each answer
        for answer in sub_answers {
            // Take first sentence as key point
            let first_sentence = answer
                .answer
                .split(&['.', '!', '?'][..])
                .next()
                .unwrap_or(&answer.answer)
                .trim();

            if !first_sentence.is_empty() {
                parts.push(format!("- {}", first_sentence));
            }
        }

        let avg_confidence =
            sub_answers.iter().map(|a| a.confidence).sum::<f32>() / sub_answers.len() as f32;

        Ok(SynthesisResult {
            text: parts.join("\n"),
            confidence: avg_confidence * 0.9, // Slight penalty for summarization
            token_usage: TokenUsage {
                synthesis_tokens: 100, // Estimate for summarization
                output_tokens: sub_answers.iter().map(|a| a.tokens_used).sum(),
                ..Default::default()
            },
            sub_answers_used: sub_answers.len(),
        })
    }

    /// Synthesize by selecting the best answer.
    fn synthesize_best_answer(&self, sub_answers: &[SubAnswer]) -> Result<SynthesisResult> {
        let best = sub_answers
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        Ok(SynthesisResult {
            text: best.answer.clone(),
            confidence: best.confidence,
            token_usage: TokenUsage {
                output_tokens: best.tokens_used,
                ..Default::default()
            },
            sub_answers_used: 1,
        })
    }

    /// Synthesize using consensus across answers.
    fn synthesize_consensus(&self, sub_answers: &[SubAnswer]) -> Result<SynthesisResult> {
        // Filter to high-confidence answers
        let confident_answers: Vec<_> =
            sub_answers.iter().filter(|a| a.confidence >= 0.7).collect();

        if confident_answers.is_empty() {
            // Fall back to best answer
            return self.synthesize_best_answer(sub_answers);
        }

        // Combine confident answers
        let mut parts = Vec::new();
        for answer in &confident_answers {
            parts.push(answer.answer.clone());
        }

        let avg_confidence = confident_answers.iter().map(|a| a.confidence).sum::<f32>()
            / confident_answers.len() as f32;

        // Boost confidence for consensus
        let consensus_confidence = (avg_confidence * 1.1).min(1.0);

        Ok(SynthesisResult {
            text: parts.join("\n\n"),
            confidence: consensus_confidence,
            token_usage: TokenUsage {
                output_tokens: confident_answers.iter().map(|a| a.tokens_used).sum(),
                ..Default::default()
            },
            sub_answers_used: confident_answers.len(),
        })
    }
}

impl Default for AnswerSynthesizer {
    fn default() -> Self {
        Self::default_strategy()
    }
}

/// Result of answer synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    /// Synthesized answer text.
    pub text: String,
    /// Confidence in the synthesized answer.
    pub confidence: f32,
    /// Token usage for synthesis.
    pub token_usage: TokenUsage,
    /// Number of sub-answers used.
    pub sub_answers_used: usize,
}

/// Truncate query for display (returns Cow to avoid allocation for short strings).
#[inline]
fn truncate_query_fast(query: &str) -> Cow<'_, str> {
    if query.len() <= 50 {
        Cow::Borrowed(query)
    } else {
        // Ensure we don't split in the middle of a UTF-8 character
        let end = query
            .char_indices()
            .take_while(|(i, _)| *i < 47)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(47.min(query.len()));
        Cow::Owned(format!("{}...", &query[..end]))
    }
}

/// Truncate query for display (legacy API).
fn truncate_query(query: &str) -> String {
    truncate_query_fast(query).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_answer(query: &str, answer: &str, confidence: f32) -> SubAnswer {
        SubAnswer {
            sub_query: query.to_string(),
            answer: answer.to_string(),
            confidence,
            sources: Vec::new(),
            tokens_used: answer.len() / 4,
            latency_ms: 100,
            depth: 0,
        }
    }

    #[test]
    fn test_synthesizer_single_answer() {
        let synthesizer = AnswerSynthesizer::default();
        let answers = vec![create_test_answer(
            "What is AI?",
            "AI is artificial intelligence.",
            0.9,
        )];

        let result = synthesizer
            .synthesize("What is AI?", &DecompositionStrategy::Direct, answers)
            .unwrap();

        assert_eq!(result.text, "AI is artificial intelligence.");
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.sub_answers_used, 1);
    }

    #[test]
    fn test_synthesizer_concatenate() {
        let synthesizer = AnswerSynthesizer::new(AggregationStrategy::Concatenate);
        let answers = vec![
            create_test_answer("What is ML?", "ML is machine learning.", 0.9),
            create_test_answer("What is DL?", "DL is deep learning.", 0.85),
        ];

        let result = synthesizer
            .synthesize(
                "What is ML and DL?",
                &DecompositionStrategy::Conjunction(vec!["ML".to_string(), "DL".to_string()]),
                answers,
            )
            .unwrap();

        assert!(result.text.contains("machine learning"));
        assert!(result.text.contains("deep learning"));
        assert_eq!(result.sub_answers_used, 2);
    }

    #[test]
    fn test_synthesizer_best_answer() {
        let synthesizer = AnswerSynthesizer::new(AggregationStrategy::BestAnswer);
        let answers = vec![
            create_test_answer("q1", "Low confidence answer", 0.5),
            create_test_answer("q2", "High confidence answer", 0.95),
            create_test_answer("q3", "Medium confidence answer", 0.7),
        ];

        let result = synthesizer
            .synthesize("test query", &DecompositionStrategy::Direct, answers)
            .unwrap();

        assert_eq!(result.text, "High confidence answer");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.sub_answers_used, 1);
    }

    #[test]
    fn test_synthesizer_consensus() {
        let synthesizer = AnswerSynthesizer::new(AggregationStrategy::Consensus);
        let answers = vec![
            create_test_answer("q1", "Answer 1", 0.8),
            create_test_answer("q2", "Answer 2", 0.75),
            create_test_answer("q3", "Low confidence", 0.4),
        ];

        let result = synthesizer
            .synthesize("test query", &DecompositionStrategy::Direct, answers)
            .unwrap();

        // Should only include the two confident answers
        assert!(result.text.contains("Answer 1"));
        assert!(result.text.contains("Answer 2"));
        assert!(!result.text.contains("Low confidence"));
        assert_eq!(result.sub_answers_used, 2);
    }

    #[test]
    fn test_synthesizer_empty_answers() {
        let synthesizer = AnswerSynthesizer::default();
        let result = synthesizer
            .synthesize("test query", &DecompositionStrategy::Direct, Vec::new())
            .unwrap();

        assert!(result.text.contains("Unable to generate"));
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_truncate_query() {
        assert_eq!(truncate_query("short"), "short");
        assert_eq!(
            truncate_query("this is a very long query that should be truncated because it exceeds fifty characters"),
            "this is a very long query that should be trunca..."
        );
    }
}
