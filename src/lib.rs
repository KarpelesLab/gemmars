//! gemmars: CPU-first inference for Gemma 4 and Qwen 3.5/3.8 in pure Rust, with optional
//! runtime-detected accelerators.

pub mod audio;
pub mod chat;
pub mod config;
#[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
pub mod cuda;
pub mod drafter;
pub mod gpu;
#[cfg(feature = "hf")]
pub mod hf;
pub mod kernels;
#[cfg(feature = "media")]
pub mod media;
#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
pub mod metal;
pub mod mm_config;
pub mod model;
pub mod platform;
pub mod pool;
pub mod prof;
pub mod quant;
pub mod qwen;
pub mod safetensors;
pub mod sampler;
pub mod vision;
