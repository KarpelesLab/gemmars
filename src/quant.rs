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
    /// Sum of the int8 values of each block (used by the Q4 kernel to fold the nibble offset).
    pub sums: Vec<i32>,
    pub t: usize,
    pub n_in: usize,
}

pub fn quantize_act(x: &[f32], t: usize, n_in: usize) -> QAct {
    assert_eq!(x.len(), t * n_in);
    assert!(n_in % QBLOCK == 0);
    let bpr = n_in / QBLOCK;
    let mut q = vec![0i8; t * n_in];
    let mut scales = vec![0f32; t * bpr];
    let mut sums = vec![0i32; t * bpr];
    for ti in 0..t {
        let row = &x[ti * n_in..(ti + 1) * n_in];
        for b in 0..bpr {
            let blk = &row[b * QBLOCK..(b + 1) * QBLOCK];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            scales[ti * bpr + b] = amax / 127.0;
            let mut s = 0i32;
            for (i, &v) in blk.iter().enumerate() {
                let qi = (v * inv).round() as i8;
                q[ti * n_in + b * QBLOCK + i] = qi;
                s += qi as i32;
            }
            sums[ti * bpr + b] = s;
        }
    }
    QAct { q, scales, sums, t, n_in }
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

// ---------------------------------------------------------------------------
// Q4: GGUF Q4_0 layout: 4-bit symmetric blocks of 32 (two nibbles per byte, f16 scale) = 4.5 bits/weight.
// Byte j of a block holds element j in its low nibble and element j+16 in its high nibble;
// stored value = clamp(round(v / d), -8, 7) + 8 with d = (signed absmax) / -8.
// ---------------------------------------------------------------------------

pub struct Q4Matrix {
    pub q: Vec<u8>,       // [n_out, n_in/2]
    pub scales: Vec<u16>, // f16, [n_out, n_in/32]
    pub n_out: usize,
    pub n_in: usize,
}

impl Q4Matrix {
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 2
    }
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
#[inline]
fn f32_to_f16(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

pub fn quantize_q4(w: &[u16], n_out: usize, n_in: usize) -> Q4Matrix {
    assert_eq!(w.len(), n_out * n_in);
    assert!(n_in % QBLOCK == 0);
    let bpr = n_in / QBLOCK;
    let mut q = vec![0u8; n_out * n_in / 2];
    let mut scales = vec![0u16; n_out * bpr];
    let qp = SendPtr(q.as_mut_ptr());
    let sp = SendPtr(scales.as_mut_ptr());
    let rows_per_task = (256 * 1024 / (n_in * 2)).max(1);
    let ntasks = n_out.div_ceil(rows_per_task);
    pool::run(ntasks, &|task| {
        let r0 = task * rows_per_task;
        let r1 = (r0 + rows_per_task).min(n_out);
        let mut blk = [0f32; QBLOCK];
        for r in r0..r1 {
            let row = &w[r * n_in..(r + 1) * n_in];
            // SAFETY: rows are disjoint between tasks.
            let qrow = unsafe { std::slice::from_raw_parts_mut(qp.get().add(r * n_in / 2), n_in / 2) };
            let srow = unsafe { std::slice::from_raw_parts_mut(sp.get().add(r * bpr), bpr) };
            for b in 0..bpr {
                let mut vmax = 0f32;
                for (j, &v) in row[b * QBLOCK..(b + 1) * QBLOCK].iter().enumerate() {
                    let f = bf16_to_f32(v);
                    blk[j] = f;
                    if f.abs() > vmax.abs() {
                        vmax = f;
                    }
                }
                let d = f16_to_f32(f32_to_f16(vmax / -8.0));
                srow[b] = f32_to_f16(d);
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let out = &mut qrow[b * 16..(b + 1) * 16];
                for j in 0..16 {
                    let lo = ((blk[j] * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                    let hi = ((blk[j + 16] * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
                    out[j] = lo | (hi << 4);
                }
            }
        }
    });
    Q4Matrix { q, scales, n_out, n_in }
}

/// Dequantizes rows `r0..r1` of `m` into `out` (f32 row-major).
pub fn dequantize_rows_q4(m: &Q4Matrix, r0: usize, r1: usize, out: &mut [f32]) {
    let n_in = m.n_in;
    let bpr = n_in / QBLOCK;
    for (i, r) in (r0..r1).enumerate() {
        let qrow = &m.q[r * n_in / 2..(r + 1) * n_in / 2];
        let srow = &m.scales[r * bpr..(r + 1) * bpr];
        let orow = &mut out[i * n_in..(i + 1) * n_in];
        for b in 0..bpr {
            let d = f16_to_f32(srow[b]);
            for j in 0..16 {
                let byte = qrow[b * 16 + j];
                orow[b * QBLOCK + j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
                orow[b * QBLOCK + j + 16] = ((byte >> 4) as i32 - 8) as f32 * d;
            }
        }
    }
}
