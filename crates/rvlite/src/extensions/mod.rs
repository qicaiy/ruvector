//! RvLite Extensions - Feature-gated WASM module integrations
//!
//! Each submodule wraps a specific WASM crate and exposes it through
//! the unified RvLite API via wasm_bindgen.

#[cfg(feature = "gnn")]
pub mod gnn;

#[cfg(feature = "attention")]
pub mod attention;

#[cfg(feature = "delta")]
pub mod delta;

#[cfg(feature = "learning")]
pub mod learning;

#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "hyperbolic")]
pub mod hyperbolic;

#[cfg(feature = "nervous")]
pub mod nervous;

#[cfg(feature = "sparse")]
pub mod sparse;

#[cfg(feature = "dag")]
pub mod dag;

#[cfg(feature = "router")]
pub mod router;

#[cfg(feature = "hnsw")]
pub mod hnsw;

#[cfg(feature = "sona")]
pub mod sona;
