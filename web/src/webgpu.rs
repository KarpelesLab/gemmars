//! The WebGPU backend as seen from the module: weights are uploaded through `gpu_upload`, and a
//! batch of prefill matmuls over one input runs through `gpu_matmul`, both implemented by the host
//! in JavaScript (see `web/gemmars.js`, which also holds the WGSL shaders). The host suspends the
//! module during the asynchronous readback (JSPI), so from here the calls look synchronous.

use gemmars::gpu::{Backend, DeviceWeight, Job, WeightRef};
use std::sync::atomic::{AtomicU64, Ordering};

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Non-zero when the host has a WebGPU device and can suspend the module (JSPI).
    fn gpu_available() -> i32;
    /// Uploads a weight matrix. `kind`: 0 bf16, 1 q8, 2 q4. `scales` is null for bf16.
    /// Returns a handle, or -1 when the host could not allocate it (the matrix stays on the CPU).
    fn gpu_upload(kind: u32, data: *const u8, data_len: usize, scales: *const u8, scales_len: usize, n_out: u32, n_in: u32) -> i32;
    /// Runs `n_jobs` matmuls over `x` ([t, n_in] f32); each job is a `JobDesc`. Returns 0 on success.
    fn gpu_matmul(x: *const f32, t: u32, n_in: u32, jobs: *const JobDesc, n_jobs: u32) -> i32;
}

/// One matmul for the host: output `y` is [t, n_out] f32 in module memory.
#[repr(C)]
pub struct JobDesc {
    handle: u32,
    n_out: u32,
    y: u32,
    _pad: u32,
}

struct Handle(i32);

pub struct WebGpuBackend {
    calls: AtomicU64,
    flops: AtomicU64,
    micros: AtomicU64,
}

const MIN_T: usize = 16;
/// The round trip through the host (submit, wait for the readback promise, copy back) is around a
/// millisecond; the single-threaded CPU in the browser does ~40 GFLOP/s, so 1e8 FLOPs is the break-even.
const MIN_FLOPS: f64 = 1e8;

pub fn install() {
    if unsafe { gpu_available() } != 0 {
        gemmars::gpu::install(Some(Box::new(WebGpuBackend { calls: AtomicU64::new(0), flops: AtomicU64::new(0), micros: AtomicU64::new(0) })));
    } else {
        crate::log("[webgpu: not available on this host (no adapter or no JSPI); using CPU]");
    }
}

impl Backend for WebGpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }
    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool {
        t >= MIN_T && n_out % 64 == 0 && n_in % 32 == 0 && 2.0 * t as f64 * n_out as f64 * n_in as f64 >= MIN_FLOPS
    }
    fn upload(&self, w: &WeightRef, n_out: usize, n_in: usize) -> Option<DeviceWeight> {
        let h = unsafe {
            match w {
                WeightRef::Bf16(w) => gpu_upload(0, w.as_ptr() as *const u8, w.len() * 2, std::ptr::null(), 0, n_out as u32, n_in as u32),
                WeightRef::Q8(m) => gpu_upload(1, m.q.as_ptr() as *const u8, m.q.len(), m.scales.as_ptr() as *const u8, m.scales.len() * 4, n_out as u32, n_in as u32),
                WeightRef::Q4(m) => gpu_upload(2, m.q.as_ptr(), m.q.len(), m.scales.as_ptr() as *const u8, m.scales.len() * 2, n_out as u32, n_in as u32),
            }
        };
        if h < 0 {
            return None;
        }
        Some(Box::new(Handle(h)))
    }
    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool {
        let t0 = gemmars::clock::Instant::now();
        let descs: Vec<JobDesc> = jobs
            .iter_mut()
            .map(|j| JobDesc { handle: j.dev.downcast_ref::<Handle>().map(|h| h.0 as u32).unwrap_or(u32::MAX), n_out: j.n_out as u32, y: j.y.as_mut_ptr() as u32, _pad: 0 })
            .collect();
        if descs.iter().any(|d| d.handle == u32::MAX) {
            return false;
        }
        let rc = unsafe { gpu_matmul(x.as_ptr(), t as u32, n_in as u32, descs.as_ptr(), descs.len() as u32) };
        if rc != 0 {
            crate::log(&format!("[webgpu: matmul failed ({rc}); falling back to CPU]"));
            return false;
        }
        let flops: f64 = jobs.iter().map(|j| 2.0 * t as f64 * j.n_out as f64 * n_in as f64).sum();
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.flops.fetch_add(flops as u64, Ordering::Relaxed);
        self.micros.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        true
    }
    fn stats(&self) -> Option<String> {
        let calls = self.calls.load(Ordering::Relaxed);
        if calls == 0 {
            return None;
        }
        let secs = self.micros.load(Ordering::Relaxed) as f64 / 1e6;
        Some(format!("{calls} batches, {:.2} GFLOP in {:.3}s = {:.0} GFLOP/s incl. host round trips", self.flops.load(Ordering::Relaxed) as f64 / 1e9, secs, self.flops.load(Ordering::Relaxed) as f64 / 1e9 / secs))
    }
}
