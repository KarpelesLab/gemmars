//! Int8 block quantization (Q8: blocks of 32 weights sharing one f32 scale).

use crate::kernels::bf16_to_f32;
use crate::pool;

pub const QBLOCK: usize = 32;

/// A quantized `[n_out, n_in]` matrix: int8 values plus one scale per 32-element block.
pub struct QMatrix {
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
    pub n_out: usize,
    pub n_in: usize,
}

impl QMatrix {
    pub fn blocks_per_row(&self) -> usize {
        self.n_in / QBLOCK
    }
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// Quantizes a bf16 row-major matrix. Symmetric round-to-nearest, scale = absmax / 127 per block.
pub fn quantize_bf16(w: &[u16], n_out: usize, n_in: usize) -> QMatrix {
    assert_eq!(w.len(), n_out * n_in);
    assert!(n_in % QBLOCK == 0, "n_in must be a multiple of {QBLOCK}");
    let bpr = n_in / QBLOCK;
    let mut q = vec![0i8; n_out * n_in];
    let mut scales = vec![0f32; n_out * bpr];
    let qp = SendPtr(q.as_mut_ptr());
    let sp = SendPtr(scales.as_mut_ptr());
    // Rows in chunks so each task is a few hundred KiB.
    let rows_per_task = (256 * 1024 / (n_in * 2)).max(1);
    let ntasks = n_out.div_ceil(rows_per_task);
    pool::run(ntasks, &|task| {
        let r0 = task * rows_per_task;
        let r1 = (r0 + rows_per_task).min(n_out);
        for r in r0..r1 {
            let row = &w[r * n_in..(r + 1) * n_in];
            // SAFETY: rows are disjoint between tasks.
            let qrow = unsafe { std::slice::from_raw_parts_mut(qp.get().add(r * n_in), n_in) };
            let srow = unsafe { std::slice::from_raw_parts_mut(sp.get().add(r * bpr), bpr) };
            for b in 0..bpr {
                let blk = &row[b * QBLOCK..(b + 1) * QBLOCK];
                let mut amax = 0f32;
                for &v in blk {
                    amax = amax.max(bf16_to_f32(v).abs());
                }
                let scale = amax / 127.0;
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                srow[b] = scale;
                for (i, &v) in blk.iter().enumerate() {
                    qrow[b * QBLOCK + i] = (bf16_to_f32(v) * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    });
    QMatrix { q, scales, n_out, n_in }
}

/// Quantized activations `[t, n_in]`: int8 values and one scale per block.
#[allow(dead_code)]
pub struct QAct {
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
    pub t: usize,
    pub n_in: usize,
}

pub fn quantize_act(x: &[f32], t: usize, n_in: usize) -> QAct {
    assert_eq!(x.len(), t * n_in);
    assert!(n_in % QBLOCK == 0);
    let bpr = n_in / QBLOCK;
    let mut q = vec![0i8; t * n_in];
    let mut scales = vec![0f32; t * bpr];
    for ti in 0..t {
        let row = &x[ti * n_in..(ti + 1) * n_in];
        for b in 0..bpr {
            let blk = &row[b * QBLOCK..(b + 1) * QBLOCK];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            scales[ti * bpr + b] = amax / 127.0;
            for (i, &v) in blk.iter().enumerate() {
                q[ti * n_in + b * QBLOCK + i] = (v * inv).round() as i8;
            }
        }
    }
    QAct { q, scales, t, n_in }
}

/// Dequantizes rows `r0..r1` of `m` into `out` (f32, row-major, `(r1-r0) * n_in`).
pub fn dequantize_rows(m: &QMatrix, r0: usize, r1: usize, out: &mut [f32]) {
    let n_in = m.n_in;
    let bpr = m.blocks_per_row();
    for (i, r) in (r0..r1).enumerate() {
        let qrow = &m.q[r * n_in..(r + 1) * n_in];
        let srow = &m.scales[r * bpr..(r + 1) * bpr];
        let orow = &mut out[i * n_in..(i + 1) * n_in];
        for b in 0..bpr {
            let s = srow[b];
            for j in 0..QBLOCK {
                orow[b * QBLOCK + j] = qrow[b * QBLOCK + j] as f32 * s;
            }
        }
    }
}
