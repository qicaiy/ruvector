//! Answer Refiner
//!
//! Refines the answer embedding using the updated latent state.

use super::TrmConfig;

/// Answer refinement network
///
/// Takes the current answer embedding and latent state,
/// produces a refined answer through a small MLP with residual connection.
pub struct AnswerRefiner {
    /// First linear layer: (embedding_dim + hidden_dim) -> embedding_dim * 2
    w1: Vec<f32>,
    b1: Vec<f32>,

    /// Second linear layer: embedding_dim * 2 -> embedding_dim
    w2: Vec<f32>,
    b2: Vec<f32>,

    embedding_dim: usize,
    hidden_dim: usize,
    residual_scale: f32,

    // Buffers
    combined_buffer: Vec<f32>,
    hidden_buffer: Vec<f32>,
    delta_buffer: Vec<f32>,
}

impl AnswerRefiner {
    /// Create a new answer refiner
    pub fn new(config: &TrmConfig) -> Self {
        let embedding_dim = config.embedding_dim;
        let hidden_dim = config.hidden_dim;
        let combined_dim = embedding_dim + hidden_dim;

        Self {
            w1: Self::init_weights(combined_dim, embedding_dim * 2),
            b1: vec![0.0; embedding_dim * 2],
            w2: Self::init_weights(embedding_dim * 2, embedding_dim),
            b2: vec![0.0; embedding_dim],
            embedding_dim,
            hidden_dim,
            residual_scale: config.residual_scale,
            combined_buffer: vec![0.0; combined_dim],
            hidden_buffer: vec![0.0; embedding_dim * 2],
            delta_buffer: vec![0.0; embedding_dim],
        }
    }

    /// Xavier/Glorot initialization
    fn init_weights(in_dim: usize, out_dim: usize) -> Vec<f32> {
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let mut weights = Vec::with_capacity(in_dim * out_dim);

        for i in 0..in_dim * out_dim {
            let x = ((i as f64 * 0.6180339887498949) % 1.0) as f32;
            weights.push((x - 0.5) * 2.0 * std);
        }

        weights
    }

    /// Linear transformation
    fn linear(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32], in_dim: usize, out_dim: usize) {
        output.copy_from_slice(bias);

        for i in 0..in_dim {
            let input_val = input[i];
            if input_val.abs() < 1e-8 {
                continue;
            }
            let w_row = &weights[i * out_dim..(i + 1) * out_dim];
            for (j, &w) in w_row.iter().enumerate() {
                output[j] += input_val * w;
            }
        }
    }

    /// GELU activation
    fn gelu_inplace(data: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;

        for x in data.iter_mut() {
            let x3 = *x * *x * *x;
            let inner = SQRT_2_OVER_PI * (*x + COEFF * x3);
            let tanh_val = inner.tanh();
            *x = *x * 0.5 * (1.0 + tanh_val);
        }
    }

    /// Refine the answer using the latent state
    ///
    /// The answer is refined by:
    /// 1. Combining answer (mean-pooled) with latent
    /// 2. Passing through 2-layer MLP with GELU
    /// 3. Adding scaled residual to original answer
    pub fn refine(
        &self,
        _question: &[f32],  // Reserved for future use
        latent: &[f32],
        answer: &mut [f32],
    ) {
        // Clone buffers for mutation
        let mut combined_buffer = self.combined_buffer.clone();
        let mut hidden_buffer = self.hidden_buffer.clone();
        let mut delta_buffer = self.delta_buffer.clone();

        // Mean pool the answer to embedding_dim
        let answer_pooled = self.mean_pool(answer, self.embedding_dim);

        // Combine answer with latent
        combined_buffer[..self.embedding_dim].copy_from_slice(&answer_pooled);

        let latent_len = latent.len().min(self.hidden_dim);
        combined_buffer[self.embedding_dim..self.embedding_dim + latent_len]
            .copy_from_slice(&latent[..latent_len]);

        // First linear + GELU
        let combined_dim = self.embedding_dim + self.hidden_dim;
        Self::linear(
            &combined_buffer[..combined_dim],
            &self.w1,
            &self.b1,
            &mut hidden_buffer,
            combined_dim,
            self.embedding_dim * 2,
        );
        Self::gelu_inplace(&mut hidden_buffer);

        // Second linear
        Self::linear(
            &hidden_buffer,
            &self.w2,
            &self.b2,
            &mut delta_buffer,
            self.embedding_dim * 2,
            self.embedding_dim,
        );

        // Apply residual update to each token position in answer
        let num_tokens = answer.len() / self.embedding_dim;
        if num_tokens == 0 {
            // Answer is smaller than embedding_dim, apply directly
            for (i, delta) in delta_buffer.iter().enumerate() {
                if i < answer.len() {
                    answer[i] += self.residual_scale * delta;
                }
            }
        } else {
            // Apply delta to each token position
            for token in 0..num_tokens {
                let start = token * self.embedding_dim;
                for (i, delta) in delta_buffer.iter().enumerate() {
                    if start + i < answer.len() {
                        answer[start + i] += self.residual_scale * delta;
                    }
                }
            }
        }
    }

    /// Mean pooling to target dimension
    fn mean_pool(&self, input: &[f32], target_dim: usize) -> Vec<f32> {
        if input.len() == target_dim {
            return input.to_vec();
        }

        let mut pooled = vec![0.0; target_dim];
        let num_chunks = (input.len() + target_dim - 1) / target_dim;

        for chunk in input.chunks(target_dim) {
            for (i, &v) in chunk.iter().enumerate() {
                pooled[i] += v / num_chunks as f32;
            }
        }

        pooled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> TrmConfig {
        TrmConfig {
            hidden_dim: 64,
            embedding_dim: 64,
            residual_scale: 0.1,
            ..TrmConfig::default()
        }
    }

    #[test]
    fn test_refiner_creation() {
        let config = make_config();
        let refiner = AnswerRefiner::new(&config);
        assert_eq!(refiner.embedding_dim, 64);
    }

    #[test]
    fn test_refiner_preserves_dimensions() {
        let config = make_config();
        let refiner = AnswerRefiner::new(&config);

        let question = vec![0.1; 64];
        let latent = vec![0.5; 64];
        let mut answer = vec![1.0; 64 * 10]; // 10 tokens

        let original_len = answer.len();
        refiner.refine(&question, &latent, &mut answer);

        assert_eq!(answer.len(), original_len);
    }

    #[test]
    fn test_refiner_modifies_answer() {
        let config = make_config();
        let refiner = AnswerRefiner::new(&config);

        let question = vec![0.5; 64];
        let latent = vec![0.5; 64];
        let mut answer = vec![1.0; 64];
        let original = answer.clone();

        refiner.refine(&question, &latent, &mut answer);

        // Answer should be modified
        let diff: f32 = answer.iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 0.0, "Answer should be modified");
    }

    #[test]
    fn test_refiner_residual_scale() {
        let config = TrmConfig {
            hidden_dim: 64,
            embedding_dim: 64,
            residual_scale: 0.1,
            ..TrmConfig::default()
        };
        let refiner = AnswerRefiner::new(&config);

        // With zero latent, changes should be minimal
        let question = vec![0.0; 64];
        let latent = vec![0.0; 64];
        let mut answer = vec![1.0; 64];

        refiner.refine(&question, &latent, &mut answer);

        // Changes should be bounded by residual scale
        let max_diff = answer.iter().map(|x| (x - 1.0).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1.0, "Residual should be bounded, max_diff={}", max_diff);
    }

    #[test]
    fn test_mean_pool() {
        let config = make_config();
        let refiner = AnswerRefiner::new(&config);

        // Exact size
        let input = vec![1.0; 64];
        let pooled = refiner.mean_pool(&input, 64);
        assert_eq!(pooled.len(), 64);
        assert!((pooled[0] - 1.0).abs() < 1e-6);

        // Larger input (2x)
        let input = vec![2.0; 128];
        let pooled = refiner.mean_pool(&input, 64);
        assert_eq!(pooled.len(), 64);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
    }
}
