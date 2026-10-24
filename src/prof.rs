//! Lightweight always-on profiler: accumulated wall time per category.

use crate::clock::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Cat {
    Matmul = 0,
    Attention,
    Norm,
    Elementwise,
    Embed,
    Head,
    Other,
}

const NAMES: [&str; 7] = ["matmul", "attention", "norm", "elementwise", "embed+ple", "lm_head", "drafter"];
static NANOS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static CALLS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static BYTES: AtomicU64 = AtomicU64::new(0);

pub struct Timer(Cat, Instant);

#[inline]
pub fn start(c: Cat) -> Timer {
    Timer(c, Instant::now())
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        NANOS[self.0 as usize].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS[self.0 as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Record bytes of weights streamed (for bandwidth reporting).
#[inline]
pub fn add_bytes(n: usize) {
    BYTES.fetch_add(n as u64, Ordering::Relaxed);
}

pub fn reset() {
    for a in NANOS.iter().chain(CALLS.iter()) {
        a.store(0, Ordering::Relaxed);
    }
    BYTES.store(0, Ordering::Relaxed);
}

pub fn report(total_secs: f64, tokens: usize) {
    eprintln!(
        "[profile over {tokens} forward tokens, {:.1} ms/token]",
        total_secs * 1e3 / tokens.max(1) as f64
    );
    if let Some(s) = crate::gpu::active().and_then(|b| b.stats()) {
        eprintln!("  gpu: {s}");
    }
    let mut acc = 0f64;
    for i in 0..NAMES.len() {
        let s = NANOS[i].load(Ordering::Relaxed) as f64 * 1e-9;
        let c = CALLS[i].load(Ordering::Relaxed);
        if c == 0 {
            continue;
        }
        acc += s;
        eprintln!(
            "  {:<12} {:>8.1} ms/tok  {:>6} calls/tok  {:>7.1} us/call",
            NAMES[i],
            s * 1e3 / tokens.max(1) as f64,
            c / tokens.max(1) as u64,
            s * 1e6 / c as f64
        );
    }
    let bytes = BYTES.load(Ordering::Relaxed) as f64;
    let mm = NANOS[Cat::Matmul as usize].load(Ordering::Relaxed) as f64 * 1e-9;
    eprintln!(
        "  weights streamed {:.2} GB/tok, matmul bandwidth {:.0} GB/s, accounted {:.0}% of wall",
        bytes / tokens.max(1) as f64 / 1e9,
        bytes / mm.max(1e-9) / 1e9,
        acc / total_secs.max(1e-9) * 100.0
    );
}
