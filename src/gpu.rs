//! Accelerator backends: a trait the model's linear layers dispatch through, plus a global
//! registry chosen at start-up from the platform detection. Every backend is optional at run time;
//! the CPU kernels remain the reference path and the fallback.

use crate::quant::{Q4Matrix, QMatrix};
use std::any::Any;
use std::sync::OnceLock;

/// A weight matrix in one of the supported storage formats (`[n_out, n_in]`).
pub enum WeightRef<'a> {
    Bf16(&'a [u16]),
    Q8(&'a QMatrix),
    Q4(&'a Q4Matrix),
}

impl WeightRef<'_> {
    pub fn bytes(&self) -> usize {
        match self {
            WeightRef::Bf16(w) => w.len() * 2,
            WeightRef::Q8(m) => m.bytes(),
            WeightRef::Q4(m) => m.bytes(),
        }
    }
}

/// Device-side state for one weight matrix (buffers / handles), owned by the layer.
pub type DeviceWeight = Box<dyn Any + Send + Sync>;

/// One matmul of a batch sharing the same input `x`: `y[t][n_out] = x[t][n_in] * W^T`.
pub struct Job<'a> {
    pub w: WeightRef<'a>,
    pub dev: &'a DeviceWeight,
    pub n_out: usize,
    pub y: &'a mut [f32],
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    /// Whether this backend should run a matmul of these dimensions. Dispatch has a fixed cost
    /// (~0.15 ms on Metal), so small ops and single-token decode stay on the CPU.
    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool;
    /// Prepares device-side state for a weight matrix (zero-copy where the API allows).
    fn upload(&self, w: &WeightRef, n_out: usize, n_in: usize) -> Option<DeviceWeight>;
    /// Runs a batch of matmuls over one shared input in a single submission. Returns false if
    /// nothing was computed (the caller then runs the whole batch on the CPU).
    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool;
    /// Cumulative timing summary for `--profile`.
    fn stats(&self) -> Option<String> {
        None
    }
}

static ACTIVE: OnceLock<Option<Box<dyn Backend>>> = OnceLock::new();

/// Installs the backend for this process (call once, before loading models).
pub fn install(b: Option<Box<dyn Backend>>) {
    let _ = ACTIVE.set(b);
}

pub fn active() -> Option<&'static dyn Backend> {
    ACTIVE.get().and_then(|b| b.as_deref())
}

/// Picks and initialises a backend for the detected platform and the user's preference.
pub fn init(plat: &crate::platform::Platform, pref: crate::platform::DevicePref) -> Option<&'static dyn Backend> {
    use crate::platform::DevicePref;
    let chosen: Option<Box<dyn Backend>> = match plat.pick(pref) {
        None => None,
        Some(g) => {
            let r: Result<Box<dyn Backend>, String> = match g.kind {
                #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
                crate::platform::GpuKind::Metal => crate::metal::MetalBackend::new().map(|b| Box::new(b) as Box<dyn Backend>).map_err(|e| e.to_string()),
                #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
                crate::platform::GpuKind::Cuda => crate::cuda::CudaBackend::new(g.index).map(|b| Box::new(b) as Box<dyn Backend>).map_err(|e| e.to_string()),
                _ => Err(format!("{} compute is not implemented in this build", g.kind.as_str())),
            };
            match r {
                Ok(b) => Some(b),
                Err(e) => {
                    if pref != DevicePref::Auto || !e.contains("not implemented") {
                        eprintln!("[{}: {e}; using CPU]", g.kind.as_str());
                    }
                    None
                }
            }
        }
    };
    install(chosen);
    active()
}
