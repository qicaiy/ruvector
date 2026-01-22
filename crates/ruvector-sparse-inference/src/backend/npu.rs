//! NPU (Neural Processing Unit) backend for hardware-accelerated inference
//!
//! This module provides NPU acceleration when available.
//! Currently a stub - will be implemented when NPU hardware support is added.

use super::Backend;
use crate::config::ActivationType;
use ndarray::Array2;

/// Check if NPU hardware is available
pub fn is_available() -> bool {
    false
}

/// NPU backend for neural network acceleration
pub struct NpuBackend;

impl NpuBackend {
    /// Create a new NPU backend
    pub fn new() -> Self {
        Self
    }
}

impl Default for NpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for NpuBackend {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn sparse_matmul(&self, matrix: &Array2<f32>, input: &[f32], rows: &[usize]) -> Vec<f32> {
        let mut output = vec![0.0; rows.len()];
        for (i, &row) in rows.iter().enumerate() {
            output[i] = matrix
                .row(row)
                .iter()
                .zip(input.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        output
    }

    fn sparse_matmul_accumulate(
        &self,
        matrix: &Array2<f32>,
        input: &[f32],
        cols: &[usize],
        output: &mut [f32],
    ) {
        for &col in cols {
            for (i, val) in matrix.column(col).iter().enumerate() {
                output[i] += val * input[col];
            }
        }
    }

    fn activation(&self, data: &mut [f32], activation_type: ActivationType) {
        match activation_type {
            ActivationType::ReLU => {
                for x in data.iter_mut() {
                    *x = x.max(0.0);
                }
            }
            ActivationType::Sigmoid => {
                for x in data.iter_mut() {
                    *x = 1.0 / (1.0 + (-*x).exp());
                }
            }
            ActivationType::Tanh => {
                for x in data.iter_mut() {
                    *x = x.tanh();
                }
            }
            ActivationType::None => {}
        }
    }

    fn add(&self, a: &mut [f32], b: &[f32]) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += y;
        }
    }

    fn axpy(&self, a: &mut [f32], b: &[f32], scalar: f32) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += y * scalar;
        }
    }

    fn name(&self) -> &'static str {
        "NPU (stub)"
    }

    fn simd_width(&self) -> usize {
        1
    }
}
