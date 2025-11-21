//! FastGRNN model implementation
//!
//! Lightweight Gated Recurrent Neural Network optimized for inference

use crate::error::{Result, TinyDancerError};
use ndarray::{Array1, Array2};
use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::serialize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// FastGRNN model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastGRNNConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Gate non-linearity parameter
    pub nu: f32,
    /// Hidden non-linearity parameter
    pub zeta: f32,
    /// Rank constraint for low-rank factorization
    pub rank: Option<usize>,
}

impl Default for FastGRNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 5, // 5 features from feature engineering
            hidden_dim: 8,
            output_dim: 1,
            nu: 1.0,
            zeta: 1.0,
            rank: Some(4),
        }
    }
}

/// FastGRNN model for neural routing
pub struct FastGRNN {
    config: FastGRNNConfig,
    /// Weight matrix for reset gate (U_r)
    w_reset: Array2<f32>,
    /// Weight matrix for update gate (U_u)
    w_update: Array2<f32>,
    /// Weight matrix for candidate (U_c)
    w_candidate: Array2<f32>,
    /// Recurrent weight matrix (W)
    w_recurrent: Array2<f32>,
    /// Output weight matrix
    w_output: Array2<f32>,
    /// Bias for reset gate
    b_reset: Array1<f32>,
    /// Bias for update gate
    b_update: Array1<f32>,
    /// Bias for candidate
    b_candidate: Array1<f32>,
    /// Bias for output
    b_output: Array1<f32>,
    /// Whether the model is quantized
    quantized: bool,
}

impl FastGRNN {
    /// Create a new FastGRNN model with the given configuration
    pub fn new(config: FastGRNNConfig) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let w_reset = Array2::from_shape_fn(
            (config.hidden_dim, config.input_dim),
            |_| rng.gen_range(-0.1..0.1),
        );
        let w_update = Array2::from_shape_fn(
            (config.hidden_dim, config.input_dim),
            |_| rng.gen_range(-0.1..0.1),
        );
        let w_candidate = Array2::from_shape_fn(
            (config.hidden_dim, config.input_dim),
            |_| rng.gen_range(-0.1..0.1),
        );
        let w_recurrent = Array2::from_shape_fn(
            (config.hidden_dim, config.hidden_dim),
            |_| rng.gen_range(-0.1..0.1),
        );
        let w_output = Array2::from_shape_fn(
            (config.output_dim, config.hidden_dim),
            |_| rng.gen_range(-0.1..0.1),
        );

        let b_reset = Array1::zeros(config.hidden_dim);
        let b_update = Array1::zeros(config.hidden_dim);
        let b_candidate = Array1::zeros(config.hidden_dim);
        let b_output = Array1::zeros(config.output_dim);

        Ok(Self {
            config,
            w_reset,
            w_update,
            w_candidate,
            w_recurrent,
            w_output,
            b_reset,
            b_update,
            b_candidate,
            b_output,
            quantized: false,
        })
    }

    /// Load model from a file (safetensors format)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        const MODEL_VERSION: &str = "1.0";

        let buffer = fs::read(path.as_ref())?;

        // First, read metadata from the buffer
        let (_, metadata_struct) = SafeTensors::read_metadata(&buffer)?;

        // Extract custom metadata HashMap
        let metadata = metadata_struct.metadata()
            .as_ref()
            .ok_or_else(|| TinyDancerError::SafetensorsError(
                "Missing metadata in safetensors file".to_string()
            ))?;

        // Check version compatibility
        let version = metadata.get("version")
            .ok_or_else(|| TinyDancerError::SafetensorsError(
                "Missing version in metadata".to_string()
            ))?;

        if version != MODEL_VERSION {
            return Err(TinyDancerError::SafetensorsError(
                format!("Incompatible model version: expected {}, got {}", MODEL_VERSION, version)
            ));
        }

        // Load config from metadata
        let config_json = metadata.get("config")
            .ok_or_else(|| TinyDancerError::SafetensorsError(
                "Missing config in metadata".to_string()
            ))?;
        let config: FastGRNNConfig = serde_json::from_str(config_json)?;

        // Load quantization flag
        let quantized = metadata.get("quantized")
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        // Now deserialize the tensors
        let tensors = SafeTensors::deserialize(&buffer)?;

        // Helper function to load tensor as Array2
        let load_array2 = |name: &str, expected_shape: [usize; 2]| -> Result<Array2<f32>> {
            let tensor = tensors.tensor(name)
                .map_err(|e| TinyDancerError::SafetensorsError(
                    format!("Failed to load tensor '{}': {}", name, e)
                ))?;

            // Validate shape
            if tensor.shape().len() != 2 {
                return Err(TinyDancerError::SafetensorsError(
                    format!("Tensor '{}' has invalid rank: expected 2, got {}", name, tensor.shape().len())
                ));
            }

            if tensor.shape()[0] != expected_shape[0] || tensor.shape()[1] != expected_shape[1] {
                return Err(TinyDancerError::SafetensorsError(
                    format!("Tensor '{}' has invalid shape: expected {:?}, got {:?}",
                        name, expected_shape, tensor.shape())
                ));
            }

            // Convert to f32 slice
            let data = tensor.data();
            let float_slice = bytemuck::cast_slice::<u8, f32>(data);

            Ok(Array2::from_shape_vec(
                (expected_shape[0], expected_shape[1]),
                float_slice.to_vec()
            ).map_err(|e| TinyDancerError::SafetensorsError(
                format!("Failed to create array from tensor '{}': {}", name, e)
            ))?)
        };

        // Helper function to load tensor as Array1
        let load_array1 = |name: &str, expected_len: usize| -> Result<Array1<f32>> {
            let tensor = tensors.tensor(name)
                .map_err(|e| TinyDancerError::SafetensorsError(
                    format!("Failed to load tensor '{}': {}", name, e)
                ))?;

            // Validate shape
            if tensor.shape().len() != 1 {
                return Err(TinyDancerError::SafetensorsError(
                    format!("Tensor '{}' has invalid rank: expected 1, got {}", name, tensor.shape().len())
                ));
            }

            if tensor.shape()[0] != expected_len {
                return Err(TinyDancerError::SafetensorsError(
                    format!("Tensor '{}' has invalid length: expected {}, got {}",
                        name, expected_len, tensor.shape()[0])
                ));
            }

            // Convert to f32 slice
            let data = tensor.data();
            let float_slice = bytemuck::cast_slice::<u8, f32>(data);

            Ok(Array1::from_vec(float_slice.to_vec()))
        };

        // Load all weight matrices with dimension validation
        let w_reset = load_array2("w_reset", [config.hidden_dim, config.input_dim])?;
        let w_update = load_array2("w_update", [config.hidden_dim, config.input_dim])?;
        let w_candidate = load_array2("w_candidate", [config.hidden_dim, config.input_dim])?;
        let w_recurrent = load_array2("w_recurrent", [config.hidden_dim, config.hidden_dim])?;
        let w_output = load_array2("w_output", [config.output_dim, config.hidden_dim])?;

        // Load all biases
        let b_reset = load_array1("b_reset", config.hidden_dim)?;
        let b_update = load_array1("b_update", config.hidden_dim)?;
        let b_candidate = load_array1("b_candidate", config.hidden_dim)?;
        let b_output = load_array1("b_output", config.output_dim)?;

        Ok(Self {
            config,
            w_reset,
            w_update,
            w_candidate,
            w_recurrent,
            w_output,
            b_reset,
            b_update,
            b_candidate,
            b_output,
            quantized,
        })
    }

    /// Save model to a file (safetensors format)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        const MODEL_VERSION: &str = "1.0";

        // Create tensor views directly (avoiding lifetime issues with closures)
        let w_reset_view = {
            let shape = vec![self.w_reset.nrows(), self.w_reset.ncols()];
            let data = bytemuck::cast_slice::<f32, u8>(self.w_reset.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let w_update_view = {
            let shape = vec![self.w_update.nrows(), self.w_update.ncols()];
            let data = bytemuck::cast_slice::<f32, u8>(self.w_update.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let w_candidate_view = {
            let shape = vec![self.w_candidate.nrows(), self.w_candidate.ncols()];
            let data = bytemuck::cast_slice::<f32, u8>(self.w_candidate.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let w_recurrent_view = {
            let shape = vec![self.w_recurrent.nrows(), self.w_recurrent.ncols()];
            let data = bytemuck::cast_slice::<f32, u8>(self.w_recurrent.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let w_output_view = {
            let shape = vec![self.w_output.nrows(), self.w_output.ncols()];
            let data = bytemuck::cast_slice::<f32, u8>(self.w_output.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let b_reset_view = {
            let shape = vec![self.b_reset.len()];
            let data = bytemuck::cast_slice::<f32, u8>(self.b_reset.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let b_update_view = {
            let shape = vec![self.b_update.len()];
            let data = bytemuck::cast_slice::<f32, u8>(self.b_update.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let b_candidate_view = {
            let shape = vec![self.b_candidate.len()];
            let data = bytemuck::cast_slice::<f32, u8>(self.b_candidate.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        let b_output_view = {
            let shape = vec![self.b_output.len()];
            let data = bytemuck::cast_slice::<f32, u8>(self.b_output.as_slice().unwrap());
            TensorView::new(safetensors::Dtype::F32, shape, data)
                .expect("Failed to create tensor view")
        };

        // Create tensor map
        let tensors: Vec<(&str, TensorView)> = vec![
            ("w_reset", w_reset_view),
            ("w_update", w_update_view),
            ("w_candidate", w_candidate_view),
            ("w_recurrent", w_recurrent_view),
            ("w_output", w_output_view),
            ("b_reset", b_reset_view),
            ("b_update", b_update_view),
            ("b_candidate", b_candidate_view),
            ("b_output", b_output_view),
        ];

        // Create metadata with config and version
        let mut metadata = HashMap::new();
        metadata.insert("version".to_string(), MODEL_VERSION.to_string());
        metadata.insert("config".to_string(), serde_json::to_string(&self.config)?);
        metadata.insert("quantized".to_string(), self.quantized.to_string());
        metadata.insert("description".to_string(),
            "FastGRNN model for neural routing in Tiny Dancer".to_string());

        // Serialize to bytes (pass iterator, not reference)
        let bytes = serialize(tensors, &Some(metadata))?;

        // Write to file
        fs::write(path.as_ref(), bytes)?;

        Ok(())
    }

    /// Forward pass through the FastGRNN model
    ///
    /// # Arguments
    /// * `input` - Input vector (sequence of features)
    /// * `initial_hidden` - Optional initial hidden state
    ///
    /// # Returns
    /// Output score (typically between 0.0 and 1.0 after sigmoid)
    pub fn forward(&self, input: &[f32], initial_hidden: Option<&[f32]>) -> Result<f32> {
        if input.len() != self.config.input_dim {
            return Err(TinyDancerError::InvalidInput(format!(
                "Expected input dimension {}, got {}",
                self.config.input_dim,
                input.len()
            )));
        }

        let x = Array1::from_vec(input.to_vec());
        let mut h = if let Some(hidden) = initial_hidden {
            Array1::from_vec(hidden.to_vec())
        } else {
            Array1::zeros(self.config.hidden_dim)
        };

        // FastGRNN cell computation
        // r_t = sigmoid(W_r * x_t + b_r)
        let r = sigmoid(&(self.w_reset.dot(&x) + &self.b_reset), self.config.nu);

        // u_t = sigmoid(W_u * x_t + b_u)
        let u = sigmoid(&(self.w_update.dot(&x) + &self.b_update), self.config.nu);

        // c_t = tanh(W_c * x_t + W * (r_t ⊙ h_{t-1}) + b_c)
        let c = tanh(
            &(self.w_candidate.dot(&x) + self.w_recurrent.dot(&(&r * &h)) + &self.b_candidate),
            self.config.zeta,
        );

        // h_t = u_t ⊙ h_{t-1} + (1 - u_t) ⊙ c_t
        h = &u * &h + &((Array1::<f32>::ones(u.len()) - &u) * &c);

        // Output: y = W_out * h_t + b_out
        let output = self.w_output.dot(&h) + &self.b_output;

        // Apply sigmoid to get probability
        Ok(sigmoid_scalar(output[0]))
    }

    /// Batch inference for multiple inputs
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        inputs
            .iter()
            .map(|input| self.forward(input, None))
            .collect()
    }

    /// Quantize the model to INT8
    pub fn quantize(&mut self) -> Result<()> {
        // TODO: Implement INT8 quantization
        self.quantized = true;
        Ok(())
    }

    /// Apply magnitude-based pruning
    pub fn prune(&mut self, sparsity: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TinyDancerError::InvalidInput(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        // TODO: Implement magnitude-based pruning
        Ok(())
    }

    /// Get model size in bytes
    pub fn size_bytes(&self) -> usize {
        let params = self.w_reset.len()
            + self.w_update.len()
            + self.w_candidate.len()
            + self.w_recurrent.len()
            + self.w_output.len()
            + self.b_reset.len()
            + self.b_update.len()
            + self.b_candidate.len()
            + self.b_output.len();

        params * if self.quantized { 1 } else { 4 } // 1 byte for INT8, 4 bytes for f32
    }

    /// Get configuration
    pub fn config(&self) -> &FastGRNNConfig {
        &self.config
    }
}

/// Sigmoid activation with scaling parameter
fn sigmoid(x: &Array1<f32>, scale: f32) -> Array1<f32> {
    x.mapv(|v| sigmoid_scalar(v * scale))
}

/// Scalar sigmoid
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation with scaling parameter
fn tanh(x: &Array1<f32>, scale: f32) -> Array1<f32> {
    x.mapv(|v| (v * scale).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastgrnn_creation() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config).unwrap();
        assert!(model.size_bytes() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let config = FastGRNNConfig {
            input_dim: 10,
            hidden_dim: 8,
            output_dim: 1,
            ..Default::default()
        };
        let model = FastGRNN::new(config).unwrap();
        let input = vec![0.5; 10];
        let output = model.forward(&input, None).unwrap();
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_batch_inference() {
        let config = FastGRNNConfig {
            input_dim: 10,
            ..Default::default()
        };
        let model = FastGRNN::new(config).unwrap();
        let inputs = vec![vec![0.5; 10], vec![0.3; 10], vec![0.8; 10]];
        let outputs = model.forward_batch(&inputs).unwrap();
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_save_load_roundtrip() {
        use tempfile::NamedTempFile;

        // Create a model with specific configuration
        let config = FastGRNNConfig {
            input_dim: 8,
            hidden_dim: 12,
            output_dim: 2,
            nu: 1.5,
            zeta: 0.8,
            rank: Some(6),
        };
        let original_model = FastGRNN::new(config).unwrap();

        // Save to temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        original_model.save(path).unwrap();

        // Load from file
        let loaded_model = FastGRNN::load(path).unwrap();

        // Verify configuration matches
        assert_eq!(loaded_model.config.input_dim, original_model.config.input_dim);
        assert_eq!(loaded_model.config.hidden_dim, original_model.config.hidden_dim);
        assert_eq!(loaded_model.config.output_dim, original_model.config.output_dim);
        assert_eq!(loaded_model.config.nu, original_model.config.nu);
        assert_eq!(loaded_model.config.zeta, original_model.config.zeta);
        assert_eq!(loaded_model.config.rank, original_model.config.rank);
        assert_eq!(loaded_model.quantized, original_model.quantized);

        // Verify weights match (sample check)
        assert_eq!(loaded_model.w_reset.shape(), original_model.w_reset.shape());
        assert_eq!(loaded_model.w_update.shape(), original_model.w_update.shape());
        assert_eq!(loaded_model.w_candidate.shape(), original_model.w_candidate.shape());
        assert_eq!(loaded_model.w_recurrent.shape(), original_model.w_recurrent.shape());
        assert_eq!(loaded_model.w_output.shape(), original_model.w_output.shape());

        // Verify forward pass produces same results
        let input = vec![0.5; 8];
        let original_output = original_model.forward(&input, None).unwrap();
        let loaded_output = loaded_model.forward(&input, None).unwrap();
        assert!((original_output - loaded_output).abs() < 1e-6);
    }

    #[test]
    fn test_config_preservation() {
        use tempfile::NamedTempFile;

        let config = FastGRNNConfig {
            input_dim: 15,
            hidden_dim: 20,
            output_dim: 3,
            nu: 2.0,
            zeta: 1.2,
            rank: Some(10),
        };
        let model = FastGRNN::new(config.clone()).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        model.save(temp_file.path()).unwrap();

        let loaded_model = FastGRNN::load(temp_file.path()).unwrap();

        assert_eq!(loaded_model.config.input_dim, config.input_dim);
        assert_eq!(loaded_model.config.hidden_dim, config.hidden_dim);
        assert_eq!(loaded_model.config.output_dim, config.output_dim);
        assert_eq!(loaded_model.config.nu, config.nu);
        assert_eq!(loaded_model.config.zeta, config.zeta);
        assert_eq!(loaded_model.config.rank, config.rank);
    }

    #[test]
    fn test_dimension_validation() {
        use tempfile::NamedTempFile;

        // Create and save a model
        let config = FastGRNNConfig {
            input_dim: 10,
            hidden_dim: 8,
            output_dim: 1,
            ..Default::default()
        };
        let model = FastGRNN::new(config).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        model.save(temp_file.path()).unwrap();

        // Load should succeed with correct dimensions
        let loaded_model = FastGRNN::load(temp_file.path()).unwrap();

        // Test forward pass with correct input dimension
        let input = vec![0.5; 10];
        assert!(loaded_model.forward(&input, None).is_ok());

        // Test forward pass with incorrect input dimension should fail
        let wrong_input = vec![0.5; 5];
        assert!(loaded_model.forward(&wrong_input, None).is_err());
    }

    #[test]
    fn test_version_compatibility() {
        use tempfile::NamedTempFile;

        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        model.save(temp_file.path()).unwrap();

        // Loading should succeed with current version
        let result = FastGRNN::load(temp_file.path());
        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

        // Verify the loaded model has correct version metadata
        let loaded = result.unwrap();
        assert_eq!(loaded.config.input_dim, model.config.input_dim);
    }

    #[test]
    fn test_quantized_flag_persistence() {
        use tempfile::NamedTempFile;

        let config = FastGRNNConfig::default();
        let mut model = FastGRNN::new(config).unwrap();

        // Mark as quantized
        model.quantize().unwrap();
        assert!(model.quantized);

        let temp_file = NamedTempFile::new().unwrap();
        model.save(temp_file.path()).unwrap();

        let loaded_model = FastGRNN::load(temp_file.path()).unwrap();
        assert!(loaded_model.quantized, "Quantized flag not preserved");
    }

    #[test]
    fn test_multiple_save_load_cycles() {
        use tempfile::NamedTempFile;

        let config = FastGRNNConfig {
            input_dim: 5,
            hidden_dim: 6,
            output_dim: 1,
            ..Default::default()
        };
        let original_model = FastGRNN::new(config).unwrap();

        let input = vec![0.3; 5];
        let original_output = original_model.forward(&input, None).unwrap();

        // Save and load multiple times
        let temp_file1 = NamedTempFile::new().unwrap();
        original_model.save(temp_file1.path()).unwrap();
        let model1 = FastGRNN::load(temp_file1.path()).unwrap();

        let temp_file2 = NamedTempFile::new().unwrap();
        model1.save(temp_file2.path()).unwrap();
        let model2 = FastGRNN::load(temp_file2.path()).unwrap();

        let temp_file3 = NamedTempFile::new().unwrap();
        model2.save(temp_file3.path()).unwrap();
        let model3 = FastGRNN::load(temp_file3.path()).unwrap();

        // Output should remain consistent
        let final_output = model3.forward(&input, None).unwrap();
        assert!((original_output - final_output).abs() < 1e-6);
    }
}
