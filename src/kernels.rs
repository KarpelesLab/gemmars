//! CPU compute kernels: bf16 weights, f32 activations, multi-threaded via the spin pool.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::pool;
use crate::prof::{self, Cat};
use crate::quant::{quantize_act, dequantize_rows, QMatrix, QBLOCK};
use std::cell::RefCell;

#[inline(always)]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub fn bf16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = bf16_to_f32(s);
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct JobRef<'a> {
    w: &'a [u16],
    n_out: usize,
    y: SendPtr,
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// y[t][o] = sum_i w[o][i] * x[t][i]
///
/// `w` is `[n_out, n_in]` bf16 row-major (PyTorch Linear layout),
/// `x` is `[t, n_in]` f32, `y` is `[t, n_out]` f32.
#[allow(dead_code)]
pub fn matmul(w: &[u16], n_out: usize, n_in: usize, x: &[f32], t: usize, y: &mut [f32]) {
    matmul_multi(&mut [(w, n_out, y)], n_in, x, t);
}

/// Several matmuls sharing the same input `x`, executed as one parallel region.
/// Each job is `(w [n_out, n_in], n_out, y [t, n_out])`.
pub fn matmul_multi(jobs: &mut [(&[u16], usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
    assert_eq!(x.len(), t * n_in, "input shape mismatch");
    if t == 0 {
        return;
    }
    let mut total_bytes = 0usize;
    let refs: Vec<JobRef> = jobs
        .iter_mut()
        .map(|(w, n_out, y)| {
            assert_eq!(w.len(), *n_out * n_in, "weight shape mismatch");
            assert_eq!(y.len(), t * *n_out, "output shape mismatch");
            total_bytes += w.len() * 2;
            JobRef { w, n_out: *n_out, y: SendPtr(y.as_mut_ptr()) }
        })
        .collect();
    let _timer = prof::start(Cat::Matmul);
    prof::add_bytes(total_bytes);

    // Block size: ~total/(8*threads) bytes of weights, but at least 128 KiB, in multiples of 4 rows.
    let nthreads = pool::threads();
    let target = (total_bytes / (nthreads * 8)).max(128 * 1024);
    let rows_per_block = (target / (n_in * 2)).max(16) / 16 * 16;
    let mut blocks: Vec<(u32, u32, u32)> = Vec::new();
    for (j, job) in refs.iter().enumerate() {
        let mut r = 0;
        while r < job.n_out {
            let r1 = (r + rows_per_block).min(job.n_out);
            blocks.push((j as u32, r as u32, r1 as u32));
            r = r1;
        }
    }
    // Prefill: pack activations as [tile][k][8 tokens] so the GEMM kernel can broadcast weights by lane.
    let xt = if t >= 8 { Some(pack_xt(x, t, n_in)) } else { None };
    pool::run(blocks.len(), &|b| {
        let (j, r0, r1) = blocks[b];
        let job = refs[j as usize];
        unsafe { matmul_block(job.w, job.n_out, n_in, x, xt.as_deref(), t, r0 as usize, r1 as usize, job.y.get()) }
    });
}

/// Tokens per prefill GEMM tile (8 rows x 8 tokens = 16 accumulators; 12 tokens measured slower, register pressure).
pub const TN: usize = 8;

/// Packs `x` ([t, n_in]) into token tiles of TN: `xt[tile][k][c] = x[tile*TN + c][k]`, zero padded.
fn pack_xt(x: &[f32], t: usize, n_in: usize) -> Vec<f32> {
    let tiles = t.div_ceil(TN);
    let mut xt = vec![0f32; tiles * n_in * TN];
    let p = SendPtr(xt.as_mut_ptr());
    pool::run(tiles, &|tile| {
        // SAFETY: each tile writes a disjoint range of `xt`.
        let dst = unsafe { std::slice::from_raw_parts_mut(p.get().add(tile * n_in * TN), n_in * TN) };
        for c in 0..TN {
            let tok = tile * TN + c;
            if tok >= t {
                break;
            }
            let src = &x[tok * n_in..(tok + 1) * n_in];
            for k in 0..n_in {
                dst[k * TN + c] = src[k];
            }
        }
    });
    xt
}

#[cfg(target_arch = "aarch64")]
pub mod simd {
    use std::arch::aarch64::*;

    pub trait Load8: Copy {
        unsafe fn load8(p: *const Self) -> (float32x4_t, float32x4_t);
    }
    impl Load8 for u16 {
        #[inline(always)]
        unsafe fn load8(p: *const u16) -> (float32x4_t, float32x4_t) {
            let w = vld1q_u16(p);
            let z = vdupq_n_u16(0);
            (
                vreinterpretq_f32_u16(vzip1q_u16(z, w)),
                vreinterpretq_f32_u16(vzip2q_u16(z, w)),
            )
        }
    }
    impl Load8 for f32 {
        #[inline(always)]
        unsafe fn load8(p: *const f32) -> (float32x4_t, float32x4_t) {
            (vld1q_f32(p), vld1q_f32(p.add(4)))
        }
    }

    /// Dot products of R weight rows against TC activation rows.
    /// Each iteration consumes 8*G elements per row; A independent accumulators per output
    /// hide FMA latency. `n` must be a multiple of 8*G.
    #[inline(always)]
    pub unsafe fn kernel<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
        w: *const W,
        ws: usize,
        x: *const f32,
        xs: usize,
        n: usize,
    ) -> [[f32; TC]; R] {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [[[zero; A]; TC]; R];
        let mut i = 0;
        while i < n {
            let mut wv = [[(zero, zero); G]; R];
            for r in 0..R {
                for g in 0..G {
                    wv[r][g] = W::load8(w.add(r * ws + i + 8 * g));
                }
            }
            for c in 0..TC {
                for g in 0..G {
                    let xl = vld1q_f32(x.add(c * xs + i + 8 * g));
                    let xh = vld1q_f32(x.add(c * xs + i + 8 * g + 4));
                    for r in 0..R {
                        acc[r][c][(2 * g) % A] = vfmaq_f32(acc[r][c][(2 * g) % A], wv[r][g].0, xl);
                        acc[r][c][(2 * g + 1) % A] = vfmaq_f32(acc[r][c][(2 * g + 1) % A], wv[r][g].1, xh);
                    }
                }
            }
            i += 8 * G;
        }
        let mut out = [[0f32; TC]; R];
        for r in 0..R {
            for c in 0..TC {
                let mut s = acc[r][c][0];
                for a in 1..A {
                    s = vaddq_f32(s, acc[r][c][a]);
                }
                out[r][c] = vaddvq_f32(s);
            }
        }
        out
    }

    /// Runs the kernel over every R-row group in `r0..r1` (the count must be a multiple of R)
    /// for tokens `c0..c0+TC`, storing to `y[(c0+c)*n_out + r]`. Kept out of line so each tile
    /// shape gets its own register allocation.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_tiles<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
        w: *const W,
        n_in: usize,
        x: *const f32,
        y: *mut f32,
        n_out: usize,
        r0: usize,
        r1: usize,
        c0: usize,
    ) {
        let mut r = r0;
        while r + R <= r1 {
            let o = kernel::<W, R, TC, G, A>(w.add(r * n_in), n_in, x.add(c0 * n_in), n_in, n_in);
            for rr in 0..R {
                for c in 0..TC {
                    *y.add((c0 + c) * n_out + r + rr) = o[rr][c];
                }
            }
            r += R;
        }
    }

    /// Loads 4 consecutive weights as f32.
    pub trait LoadW4: Copy {
        unsafe fn load4(p: *const Self) -> float32x4_t;
    }
    impl LoadW4 for u16 {
        #[inline(always)]
        unsafe fn load4(p: *const u16) -> float32x4_t {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(p)))
        }
    }
    impl LoadW4 for f32 {
        #[inline(always)]
        unsafe fn load4(p: *const f32) -> float32x4_t {
            vld1q_f32(p)
        }
    }

    /// Prefill GEMM micro-kernel: 8 weight rows (row-major, starting at `w`) times TN=8 tokens
    /// from a packed panel `xt` ([n_in][8]). Weights are loaded 4 at a time and broadcast by lane;
    /// accumulators are vectors over tokens. Stores rows `rb..rb+8` for tokens `c0..c0+c_valid`.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_8x8<W: LoadW4>(
        w: *const W,
        n_in: usize,
        xt: *const f32,
        y: *mut f32,
        n_out: usize,
        rb: usize,
        c0: usize,
        c_valid: usize,
    ) {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [[zero; 2]; 8];
        let mut k = 0;
        while k < n_in {
            let xp = xt.add(k * 8);
            let x0a = vld1q_f32(xp);
            let x0b = vld1q_f32(xp.add(4));
            let x1a = vld1q_f32(xp.add(8));
            let x1b = vld1q_f32(xp.add(12));
            let x2a = vld1q_f32(xp.add(16));
            let x2b = vld1q_f32(xp.add(20));
            let x3a = vld1q_f32(xp.add(24));
            let x3b = vld1q_f32(xp.add(28));
            for r in 0..8 {
                let wq = W::load4(w.add(r * n_in + k));
                acc[r][0] = vfmaq_laneq_f32::<0>(acc[r][0], x0a, wq);
                acc[r][1] = vfmaq_laneq_f32::<0>(acc[r][1], x0b, wq);
                acc[r][0] = vfmaq_laneq_f32::<1>(acc[r][0], x1a, wq);
                acc[r][1] = vfmaq_laneq_f32::<1>(acc[r][1], x1b, wq);
                acc[r][0] = vfmaq_laneq_f32::<2>(acc[r][0], x2a, wq);
                acc[r][1] = vfmaq_laneq_f32::<2>(acc[r][1], x2b, wq);
                acc[r][0] = vfmaq_laneq_f32::<3>(acc[r][0], x3a, wq);
                acc[r][1] = vfmaq_laneq_f32::<3>(acc[r][1], x3b, wq);
            }
            k += 4;
        }
        let mut tmp = [0f32; 8];
        for r in 0..8 {
            vst1q_f32(tmp.as_mut_ptr(), acc[r][0]);
            vst1q_f32(tmp.as_mut_ptr().add(4), acc[r][1]);
            for c in 0..c_valid {
                *y.add((c0 + c) * n_out + rb + r) = tmp[c];
            }
        }
    }

    /// Q8 dot products of R quantized rows (int8 + per-32 scales) with one quantized activation row.
    /// Four blocks (128 weights) per step: SDOT into int32, pairwise-add into one vector of block
    /// sums, convert, and FMA with the product of weight and activation scales.
    #[inline(never)]
    pub unsafe fn q8_dot_rows<const R: usize>(
        q: *const i8,
        s: *const f32,
        n_in: usize,
        xq: *const i8,
        xs: *const f32,
    ) -> [f32; R] {
        let bpr = n_in / 32;
        let zero = vdupq_n_s32(0);
        let mut accf = [vdupq_n_f32(0.0); R];
        let mut b = 0;
        while b + 4 <= bpr {
            let k = b * 32;
            let x0 = vld1q_s8(xq.add(k));
            let x1 = vld1q_s8(xq.add(k + 16));
            let x2 = vld1q_s8(xq.add(k + 32));
            let x3 = vld1q_s8(xq.add(k + 48));
            let x4 = vld1q_s8(xq.add(k + 64));
            let x5 = vld1q_s8(xq.add(k + 80));
            let x6 = vld1q_s8(xq.add(k + 96));
            let x7 = vld1q_s8(xq.add(k + 112));
            let sxv = vld1q_f32(xs.add(b));
            for r in 0..R {
                let wp = q.add(r * n_in + k);
                let a0 = vdotq_s32(vdotq_s32(zero, vld1q_s8(wp), x0), vld1q_s8(wp.add(16)), x1);
                let a1 = vdotq_s32(vdotq_s32(zero, vld1q_s8(wp.add(32)), x2), vld1q_s8(wp.add(48)), x3);
                let a2 = vdotq_s32(vdotq_s32(zero, vld1q_s8(wp.add(64)), x4), vld1q_s8(wp.add(80)), x5);
                let a3 = vdotq_s32(vdotq_s32(zero, vld1q_s8(wp.add(96)), x6), vld1q_s8(wp.add(112)), x7);
                let sums = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
                let swv = vld1q_f32(s.add(r * bpr + b));
                accf[r] = vfmaq_f32(accf[r], vcvtq_f32_s32(sums), vmulq_f32(swv, sxv));
            }
            b += 4;
        }
        let mut out = [0f32; R];
        for r in 0..R {
            out[r] = vaddvq_f32(accf[r]);
        }
        while b < bpr {
            let k = b * 32;
            let x0 = vld1q_s8(xq.add(k));
            let x1 = vld1q_s8(xq.add(k + 16));
            let sx = *xs.add(b);
            for r in 0..R {
                let wp = q.add(r * n_in + k);
                let a = vdotq_s32(vdotq_s32(zero, vld1q_s8(wp), x0), vld1q_s8(wp.add(16)), x1);
                out[r] += vaddvq_s32(a) as f32 * *s.add(r * bpr + b) * sx;
            }
            b += 1;
        }
        out
    }

    /// Q8 decode for one token: rows `r0..r1` of `m` against activation row (`xq`, `xs`), into `y[r]`.
    pub unsafe fn q8_rows_t1(m: &super::QMatrix, xq: *const i8, xs: *const f32, y: *mut f32, r0: usize, r1: usize) {
        let n_in = m.n_in;
        let bpr = n_in / 32;
        let mut r = r0;
        while r + 2 <= r1 {
            let o = q8_dot_rows::<2>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
            *y.add(r) = o[0];
            *y.add(r + 1) = o[1];
            r += 2;
        }
        if r < r1 {
            let o = q8_dot_rows::<1>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
            *y.add(r) = o[0];
        }
    }

    /// Rows `r0..r1` (any count) for `tc` tokens (1..=4) starting at `c0`. Requires `n_in % 32 == 0`.
    ///
    /// Decode is memory-bound, and the prefetcher likes few long streams per core, so tiles
    /// use at most 2 weight rows with many accumulators (measured ~2x faster than 4+ rows).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block_tc<W: Load8>(
        w: *const W,
        n_in: usize,
        x: *const f32,
        y: *mut f32,
        n_out: usize,
        r0: usize,
        r1: usize,
        c0: usize,
        tc: usize,
    ) {
        macro_rules! go {
            ($TC:expr, $G:expr, $A:expr) => {{
                let m = r0 + (r1 - r0) / 2 * 2;
                run_tiles::<W, 2, $TC, $G, $A>(w, n_in, x, y, n_out, r0, m, c0);
                run_tiles::<W, 1, $TC, $G, $A>(w, n_in, x, y, n_out, m, r1, c0);
            }};
        }
        match tc {
            1 => go!(1, 4, 8),
            2 => go!(2, 2, 2),
            3 => go!(3, 2, 2),
            4 => go!(4, 2, 2),
            _ => unreachable!("bad tc {tc}"),
        }
    }
}

/// Computes rows r0..r1 of the output for all t tokens.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_block(
    w: &[u16],
    n_out: usize,
    n_in: usize,
    x: &[f32],
    xt: Option<&[f32]>,
    t: usize,
    r0: usize,
    r1: usize,
    y: *mut f32,
) {
    use simd::{block_tc, gemm_8x8};
    if n_in % 32 != 0 {
        return matmul_block_scalar(w, n_out, n_in, x, t, r0, r1, y);
    }
    let wp = w.as_ptr();
    let xp = x.as_ptr();
    let mut r = r0;
    if let Some(xt) = xt {
        // Prefill: 8x8 outer-product tiles; weights stay in L2 across token tiles.
        let tiles = t.div_ceil(TN);
        while r + 8 <= r1 {
            for tile in 0..tiles {
                let c0 = tile * TN;
                gemm_8x8::<u16>(wp.add(r * n_in), n_in, xt.as_ptr().add(tile * n_in * TN), y, n_out, r, c0, (t - c0).min(TN));
            }
            r += 8;
        }
    }
    // Decode (and leftover rows): streaming dot-product tiles over up to 4 tokens at a time.
    if r < r1 {
        let mut c = 0;
        while c < t {
            let tc = (t - c).min(4);
            block_tc::<u16>(wp, n_in, xp, y, n_out, r, r1, c, tc);
            c += tc;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_block(
    w: &[u16],
    n_out: usize,
    n_in: usize,
    x: &[f32],
    _xt: Option<&[f32]>,
    t: usize,
    r0: usize,
    r1: usize,
    y: *mut f32,
) {
    matmul_block_scalar(w, n_out, n_in, x, t, r0, r1, y)
}

#[allow(clippy::too_many_arguments)]
unsafe fn matmul_block_scalar(
    w: &[u16],
    n_out: usize,
    n_in: usize,
    x: &[f32],
    t: usize,
    r0: usize,
    r1: usize,
    y: *mut f32,
) {
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        if s.len() < n_in {
            s.resize(n_in, 0.0);
        }
        for r in r0..r1 {
            bf16_to_f32_slice(&w[r * n_in..(r + 1) * n_in], &mut s[..n_in]);
            for c in 0..t {
                let v = dot(&s[..n_in], &x[c * n_in..(c + 1) * n_in]);
                *y.add(c * n_out + r) = v;
            }
        }
    });
}

#[derive(Clone, Copy)]
struct QJobRef<'a> {
    m: &'a QMatrix,
    y: SendPtr,
}

/// Several Q8 matmuls sharing the same f32 input `x` ([t, n_in]); each job is `(matrix, y [t, n_out])`.
pub fn matmul_q8_multi(jobs: &mut [(&QMatrix, &mut [f32])], x: &[f32], t: usize) {
    if t == 0 || jobs.is_empty() {
        return;
    }
    let n_in = jobs[0].0.n_in;
    assert_eq!(x.len(), t * n_in, "input shape mismatch");
    let mut total_bytes = 0usize;
    let refs: Vec<QJobRef> = jobs
        .iter_mut()
        .map(|(m, y)| {
            assert_eq!(m.n_in, n_in, "all jobs must share n_in");
            assert_eq!(y.len(), t * m.n_out, "output shape mismatch");
            total_bytes += m.bytes();
            QJobRef { m, y: SendPtr(y.as_mut_ptr()) }
        })
        .collect();
    let _timer = prof::start(Cat::Matmul);
    prof::add_bytes(total_bytes);

    let nthreads = pool::threads();
    let bytes_per_row = n_in + 4 * n_in / QBLOCK;
    let target = (total_bytes / (nthreads * 8)).max(128 * 1024);
    let rows_per_block = (target / bytes_per_row).max(16) / 16 * 16;
    let mut blocks: Vec<(u32, u32, u32)> = Vec::new();
    for (j, job) in refs.iter().enumerate() {
        let mut r = 0;
        while r < job.m.n_out {
            let r1 = (r + rows_per_block).min(job.m.n_out);
            blocks.push((j as u32, r as u32, r1 as u32));
            r = r1;
        }
    }
    if t >= 8 {
        let xt = pack_xt(x, t, n_in);
        pool::run(blocks.len(), &|b| {
            let (j, r0, r1) = blocks[b];
            let job = refs[j as usize];
            unsafe { matmul_q8_block_prefill(job.m, x, &xt, t, r0 as usize, r1 as usize, job.y.get()) }
        });
    } else {
        let xa = quantize_act(x, t, n_in);
        pool::run(blocks.len(), &|b| {
            let (j, r0, r1) = blocks[b];
            let job = refs[j as usize];
            unsafe { matmul_q8_block_decode(job.m, &xa, t, r0 as usize, r1 as usize, job.y.get()) }
        });
    }
}

/// Prefill with Q8 weights: dequantize 8 rows at a time into a thread-local f32 scratch and run
/// the f32 GEMM tiles over all tokens.
unsafe fn matmul_q8_block_prefill(m: &QMatrix, x: &[f32], xt: &[f32], t: usize, r0: usize, r1: usize, y: *mut f32) {
    let n_in = m.n_in;
    let n_out = m.n_out;
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        if s.len() < 8 * n_in {
            s.resize(8 * n_in, 0.0);
        }
        let mut r = r0;
        while r < r1 {
            let rows = (r1 - r).min(8);
            dequantize_rows(m, r, r + rows, &mut s[..rows * n_in]);
            #[cfg(target_arch = "aarch64")]
            {
                let sp = s.as_ptr();
                if rows == 8 {
                    let tiles = t.div_ceil(TN);
                    for tile in 0..tiles {
                        let c0 = tile * TN;
                        simd::gemm_8x8::<f32>(sp, n_in, xt.as_ptr().add(tile * n_in * TN), y, n_out, r, c0, (t - c0).min(TN));
                    }
                } else if n_in % 32 == 0 {
                    let mut c = 0;
                    while c < t {
                        let tc = (t - c).min(4);
                        simd::block_tc::<f32>(sp, n_in, x.as_ptr(), y.add(r), n_out, 0, rows, c, tc);
                        c += tc;
                    }
                } else {
                    scalar_rows_f32(&s[..rows * n_in], n_in, x, t, y.add(r), n_out, rows);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let _ = xt;
                scalar_rows_f32(&s[..rows * n_in], n_in, x, t, y.add(r), n_out, rows);
            }
            r += rows;
        }
    });
}

/// Reference path: f32 rows `w` ([rows, n_in]) times all tokens, stored at `y[c*n_out + i]`.
unsafe fn scalar_rows_f32(w: &[f32], n_in: usize, x: &[f32], t: usize, y: *mut f32, n_out: usize, rows: usize) {
    for i in 0..rows {
        for c in 0..t {
            *y.add(c * n_out + i) = dot(&w[i * n_in..(i + 1) * n_in], &x[c * n_in..(c + 1) * n_in]);
        }
    }
}

unsafe fn matmul_q8_block_decode(m: &QMatrix, xa: &crate::quant::QAct, t: usize, r0: usize, r1: usize, y: *mut f32) {
    let n_in = m.n_in;
    let bpr = n_in / QBLOCK;
    for c in 0..t {
        let xq = xa.q.as_ptr().add(c * n_in);
        let xs = xa.scales.as_ptr().add(c * bpr);
        let yc = y.add(c * m.n_out);
        #[cfg(target_arch = "aarch64")]
        {
            simd::q8_rows_t1(m, xq, xs, yc, r0, r1);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for r in r0..r1 {
                let mut acc = 0f32;
                for b in 0..bpr {
                    let mut s = 0i32;
                    for j in 0..QBLOCK {
                        s += m.q[r * n_in + b * QBLOCK + j] as i32 * *xq.add(b * QBLOCK + j) as i32;
                    }
                    acc += s as f32 * m.scales[r * bpr + b] * *xs.add(b);
                }
                *yc.add(r) = acc;
            }
        }
    }
}

/// f32 dot product with 8 independent accumulators (auto-vectorizes).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for i in 0..chunks {
        let a8 = &a[i * 8..i * 8 + 8];
        let b8 = &b[i * 8..i * 8 + 8];
        for j in 0..8 {
            acc[j] += a8[j] * b8[j];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// y += alpha * x
#[inline]
pub fn axpy(y: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// RMSNorm: out = x * rsqrt(mean(x^2) + eps) * w   (w optional, bf16).
pub fn rmsnorm(x: &[f32], w: Option<&[u16]>, eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let r = ss.powf(-0.5);
    match w {
        Some(w) => {
            for i in 0..n {
                out[i] = x[i] * r * bf16_to_f32(w[i]);
            }
        }
        None => {
            for i in 0..n {
                out[i] = x[i] * r;
            }
        }
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: Option<&[u16]>, eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let r = ss.powf(-0.5);
    match w {
        Some(w) => {
            for i in 0..n {
                x[i] *= r * bf16_to_f32(w[i]);
            }
        }
        None => {
            for v in x.iter_mut() {
                *v *= r;
            }
        }
    }
}

/// Row-wise RMSNorm over a `[t, n]` matrix, in parallel when there are many rows.
pub fn rmsnorm_rows(x: &[f32], n: usize, w: Option<&[u16]>, eps: f32, out: &mut [f32]) {
    let _timer = prof::start(Cat::Norm);
    let par = x.len() > 4 * n;
    for_chunks(out, n, par, |i, o| rmsnorm(&x[i * n..(i + 1) * n], w, eps, o));
}

pub fn rmsnorm_rows_inplace(x: &mut [f32], n: usize, w: Option<&[u16]>, eps: f32) {
    let _timer = prof::start(Cat::Norm);
    let par = x.len() > 4 * n;
    for_chunks(x, n, par, |_, xi| rmsnorm_inplace(xi, w, eps));
}

/// Runs `f(index, chunk)` over fixed-size chunks, in parallel only when asked.
pub fn for_chunks<F>(x: &mut [f32], size: usize, par: bool, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let n = x.len();
    let nchunks = n.div_ceil(size.max(1));
    if !par || nchunks <= 1 {
        for (i, c) in x.chunks_mut(size).enumerate() {
            f(i, c);
        }
        return;
    }
    let p = SendPtr(x.as_mut_ptr());
    pool::run(nchunks, &|i| {
        let start = i * size;
        let len = size.min(n - start);
        // SAFETY: chunks are disjoint and `x` outlives the call.
        let c = unsafe { std::slice::from_raw_parts_mut(p.get().add(start), len) };
        f(i, c);
    });
}

/// Fast vectorizable exp (relative error ~1e-7).
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let fx = (x * std::f32::consts::LOG2_E).round();
    let r = (x - fx * 0.693_359_375) - fx * -2.121_944_4e-4;
    let p = 1.0
        + r * (1.0
            + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_888_9)))));
    f32::from_bits(((fx as i32 + 127) << 23) as u32) * p
}

/// GELU with tanh approximation (`gelu_pytorch_tanh`): x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)).
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    const K2: f32 = 2.0 * 0.797_884_560_802_865_4;
    let u = K2 * (x + 0.044_715 * x * x * x);
    x / (1.0 + fast_exp(-u))
}

const PAR_MIN: usize = 8192;

fn par_chunk(len: usize) -> usize {
    (len.div_ceil(pool::threads())).max(1024).next_multiple_of(16)
}

/// a[i] = gelu(a[i]) * b[i]
pub fn gelu_mul(a: &mut [f32], b: &[f32]) {
    let _timer = prof::start(Cat::Elementwise);
    let par = a.len() >= PAR_MIN;
    let chunk = if par { par_chunk(a.len()) } else { a.len() };
    for_chunks(a, chunk, par, |i, c| {
        for (ai, &bi) in c.iter_mut().zip(&b[i * chunk..]) {
            *ai = gelu_tanh(*ai) * bi;
        }
    });
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    let _timer = prof::start(Cat::Elementwise);
    let par = a.len() >= PAR_MIN;
    let chunk = if par { par_chunk(a.len()) } else { a.len() };
    for_chunks(a, chunk, par, |i, c| {
        for (ai, &bi) in c.iter_mut().zip(&b[i * chunk..]) {
            *ai += bi;
        }
    });
}

pub fn scale_inplace(a: &mut [f32], s: f32) {
    for v in a.iter_mut() {
        *v *= s;
    }
}

/// Inverse frequencies for RoPE on a head of `dim`, rotating `proportion` of it
/// (zeros for the un-rotated part, matching HF "proportional" RoPE).
pub fn rope_inv_freq(dim: usize, theta: f64, proportion: f64) -> Vec<f32> {
    let half = dim / 2;
    let rot = ((proportion * dim as f64) as usize) / 2;
    (0..half)
        .map(|i| {
            if i < rot {
                (1.0 / theta.powf((2 * i) as f64 / dim as f64)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// cos/sin tables `[positions.len(), dim/2]` for the given inverse frequencies.
pub fn rope_tables(positions: &[usize], inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0f32; positions.len() * half];
    let mut sin = vec![0f32; positions.len() * half];
    for (pi, &p) in positions.iter().enumerate() {
        for (i, &f) in inv_freq.iter().enumerate() {
            // Match the reference: angle computed in f32.
            let a = p as f32 * f;
            cos[pi * half + i] = a.cos();
            sin[pi * half + i] = a.sin();
        }
    }
    (cos, sin)
}

/// Applies rotate-half RoPE to one head vector `v` of length `2*cos.len()`.
#[inline]
pub fn apply_rope(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    debug_assert_eq!(v.len(), 2 * half);
    let (a, b) = v.split_at_mut(half);
    for i in 0..half {
        let x1 = a[i];
        let x2 = b[i];
        a[i] = x1 * cos[i] - x2 * sin[i];
        b[i] = x2 * cos[i] + x1 * sin[i];
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = fast_exp(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}
