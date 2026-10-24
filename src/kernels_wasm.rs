//! wasm32 kernels using the fixed-width 128-bit SIMD proposal (`simd128`). The feature is a
//! compile-time choice on wasm (the host either accepts the module or rejects it), so this module
//! is only built with `-C target-feature=+simd128`; without it `kernels.rs` uses the scalar paths.
//! Same function shapes as the NEON and AVX2 modules so the call sites are shared. Vectors hold 4
//! lanes, so "8 elements" is two vectors per step. Q8/Q4 use the i16 widening multiply plus the
//! i32x4 dot-product instruction (`i32x4_dot_i16x8`), which sums adjacent i16 pairs for free.

use crate::quant::f16_to_f32;
use std::arch::wasm32::*;

pub trait Load8: Copy {
    /// 8 consecutive weights as two f32x4 vectors.
    unsafe fn load8(p: *const Self) -> (v128, v128);
    fn to_f32(self) -> f32;
}
impl Load8 for u16 {
    #[inline(always)]
    unsafe fn load8(p: *const u16) -> (v128, v128) {
        let h = v128_load(p as *const v128);
        let lo = u32x4_shl(u32x4_extend_low_u16x8(h), 16);
        let hi = u32x4_shl(u32x4_extend_high_u16x8(h), 16);
        (lo, hi)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
}
impl Load8 for f32 {
    #[inline(always)]
    unsafe fn load8(p: *const f32) -> (v128, v128) {
        (v128_load(p as *const v128), v128_load(p.add(4) as *const v128))
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

/// Loads one weight broadcast to all lanes (prefill GEMM).
pub trait LoadW4: Copy + Load8 {
    unsafe fn bcast(p: *const Self) -> v128;
}
impl LoadW4 for u16 {
    #[inline(always)]
    unsafe fn bcast(p: *const u16) -> v128 {
        f32x4_splat(f32::from_bits((*p as u32) << 16))
    }
}
impl LoadW4 for f32 {
    #[inline(always)]
    unsafe fn bcast(p: *const f32) -> v128 {
        f32x4_splat(*p)
    }
}

#[inline(always)]
fn hsum(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v)
}

#[inline(always)]
fn hsum_i32(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v) + i32x4_extract_lane::<1>(v) + i32x4_extract_lane::<2>(v) + i32x4_extract_lane::<3>(v)
}

#[inline(always)]
fn fma(a: v128, b: v128, c: v128) -> v128 {
    // No fused multiply-add in simd128 (relaxed-simd has one); mul + add.
    f32x4_add(f32x4_mul(a, b), c)
}

// ---- f32/bf16 dot-product tiles ------------------------------------------------------------------

/// Dot products of R weight rows against TC activation rows; each step consumes 8*G elements per
/// row with A independent accumulator pairs per output. `n` must be a multiple of 8*G.
#[inline(always)]
pub unsafe fn kernel<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
    w: *const W,
    ws: usize,
    x: *const f32,
    xs: usize,
    n: usize,
) -> [[f32; TC]; R] {
    let zero = f32x4_splat(0.0);
    let mut acc = [[[(zero, zero); A]; TC]; R];
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
                let xv = f32::load8(x.add(c * xs + i + 8 * g));
                for r in 0..R {
                    let a = &mut acc[r][c][g % A];
                    a.0 = fma(wv[r][g].0, xv.0, a.0);
                    a.1 = fma(wv[r][g].1, xv.1, a.1);
                }
            }
        }
        i += 8 * G;
    }
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            let mut s = f32x4_add(acc[r][c][0].0, acc[r][c][0].1);
            for a in 1..A {
                s = f32x4_add(s, f32x4_add(acc[r][c][a].0, acc[r][c][a].1));
            }
            out[r][c] = hsum(s);
        }
    }
    out
}

/// Runs the kernel over every R-row group in `r0..r1` for tokens `c0..c0+TC`, storing to
/// `y[(c0+c)*n_out + r]`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn run_tiles<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
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

/// Rows `r0..r1` (any count) for `tc` tokens (1..=4) starting at `c0`. Requires `n_in % 32 == 0`.
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
    // wasm engines map v128 locals onto 16-32 host registers; keep the live set small.
    macro_rules! go {
        ($TC:expr, $G:expr, $A:expr) => {{
            let m = r0 + (r1 - r0) / 2 * 2;
            run_tiles::<W, 2, $TC, $G, $A>(w, n_in, x, y, n_out, r0, m, c0);
            run_tiles::<W, 1, $TC, $G, $A>(w, n_in, x, y, n_out, m, r1, c0);
        }};
    }
    match tc {
        1 => go!(1, 4, 2),
        2 => go!(2, 2, 2),
        3 => go!(3, 2, 1),
        4 => go!(4, 2, 1),
        _ => unreachable!("bad tc {tc}"),
    }
}

/// Prefill GEMM micro-kernel: 8 weight rows times 8 tokens from a packed panel `xt` ([n_in][8]);
/// weights are broadcast, accumulators are two vectors (8 tokens) per row.
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
    let zero = f32x4_splat(0.0);
    let mut acc = [(zero, zero); 8];
    let mut k = 0;
    while k < n_in {
        let (x0, x1) = f32::load8(xt.add(k * 8));
        for r in 0..8 {
            let wv = W::bcast(w.add(r * n_in + k));
            acc[r].0 = fma(wv, x0, acc[r].0);
            acc[r].1 = fma(wv, x1, acc[r].1);
        }
        k += 1;
    }
    let mut tmp = [0f32; 8];
    for r in 0..8 {
        v128_store(tmp.as_mut_ptr() as *mut v128, acc[r].0);
        v128_store(tmp.as_mut_ptr().add(4) as *mut v128, acc[r].1);
        for c in 0..c_valid {
            *y.add((c0 + c) * n_out + rb + r) = tmp[c];
        }
    }
}

// ---- Q8 -----------------------------------------------------------------------------------------

/// Signed i8 x i8 dot product of two 16-byte vectors, as four i32 lane partial sums.
#[inline(always)]
fn dot_i8x16(a: v128, b: v128) -> v128 {
    let lo = i16x8_mul(i16x8_extend_low_i8x16(a), i16x8_extend_low_i8x16(b));
    let hi = i16x8_mul(i16x8_extend_high_i8x16(a), i16x8_extend_high_i8x16(b));
    // |a*b| <= 127*128 so the i16 products fit; the pairwise-add to i32 cannot overflow.
    i32x4_add(i32x4_extadd_pairwise_i16x8(lo), i32x4_extadd_pairwise_i16x8(hi))
}

/// Q8: R weight rows against TC quantized activation rows (token stride `n_in`).
#[inline(never)]
pub unsafe fn q8_dot_rows_tc<const R: usize, const TC: usize>(
    q: *const i8,
    s: *const f32,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let mut accf = [[f32x4_splat(0.0); TC]; R];
    for b in 0..bpr {
        let k = b * 32;
        let mut xv = [(i32x4_splat(0), i32x4_splat(0)); TC];
        for c in 0..TC {
            let p = xq.add(c * n_in + k);
            xv[c] = (v128_load(p as *const v128), v128_load(p.add(16) as *const v128));
        }
        for r in 0..R {
            let p = q.add(r * n_in + k);
            let w0 = v128_load(p as *const v128);
            let w1 = v128_load(p.add(16) as *const v128);
            let d = *s.add(r * bpr + b);
            for c in 0..TC {
                let p32 = i32x4_add(dot_i8x16(w0, xv[c].0), dot_i8x16(w1, xv[c].1));
                let scale = f32x4_splat(d * *xs.add(c * bpr + b));
                accf[r][c] = fma(f32x4_convert_i32x4(p32), scale, accf[r][c]);
            }
        }
    }
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            out[r][c] = hsum(accf[r][c]);
        }
    }
    out
}

/// Q8 decode for one token: rows `r0..r1` of `m` against activation row (`xq`, `xs`), into `y[r]`.
pub unsafe fn q8_rows_t1(m: &crate::quant::QMatrix, xq: *const i8, xs: *const f32, y: *mut f32, r0: usize, r1: usize) {
    let n_in = m.n_in;
    let bpr = n_in / 32;
    let mut r = r0;
    while r + 2 <= r1 {
        let o = q8_dot_rows_tc::<2, 1>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
        *y.add(r) = o[0][0];
        *y.add(r + 1) = o[1][0];
        r += 2;
    }
    if r < r1 {
        let o = q8_dot_rows_tc::<1, 1>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
        *y.add(r) = o[0][0];
    }
}

// ---- Q4_0 ---------------------------------------------------------------------------------------

/// Q4_0: R weight rows against TC quantized activation rows. Nibbles are used unsigned (0..15) and
/// the +8 offset is folded in via the block sums of x.
#[inline(never)]
pub unsafe fn q4_dot_rows_tc<const R: usize, const TC: usize>(
    q: *const u8,
    s: *const u16,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
    xsums: *const i32,
) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let half_row = n_in / 2;
    let m0f = u8x16_splat(0x0F);
    let mut accf = [[f32x4_splat(0.0); TC]; R];
    let mut corr = [[0f32; TC]; R];
    for b in 0..bpr {
        let k = b * 32;
        let mut xv = [(i32x4_splat(0), i32x4_splat(0)); TC];
        for c in 0..TC {
            let p = xq.add(c * n_in + k);
            xv[c] = (v128_load(p as *const v128), v128_load(p.add(16) as *const v128));
        }
        for r in 0..R {
            let bytes = v128_load(q.add(r * half_row + b * 16) as *const v128);
            let lo = v128_and(bytes, m0f); // elements 0..15
            let hi = u8x16_shr(bytes, 4); // elements 16..31 (shift clears the high nibble)
            let d = f16_to_f32(*s.add(r * bpr + b));
            for c in 0..TC {
                // Nibbles are 0..15 and x is i8, so the i16 products fit comfortably.
                let p32 = i32x4_add(dot_i8x16(lo, xv[c].0), dot_i8x16(hi, xv[c].1));
                let sc = d * *xs.add(c * bpr + b);
                accf[r][c] = fma(f32x4_convert_i32x4(p32), f32x4_splat(sc), accf[r][c]);
                corr[r][c] += sc * *xsums.add(c * bpr + b) as f32;
            }
        }
    }
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            out[r][c] = hsum(accf[r][c]) - 8.0 * corr[r][c];
        }
    }
    out
}

/// Q4 decode for one token: rows `r0..r1` of `m` against activation row, into `y[r]`.
pub unsafe fn q4_rows_t1(m: &crate::quant::Q4Matrix, xq: *const i8, xs: *const f32, xsums: *const i32, y: *mut f32, r0: usize, r1: usize) {
    let n_in = m.n_in;
    let bpr = n_in / 32;
    let mut r = r0;
    while r + 2 <= r1 {
        let o = q4_dot_rows_tc::<2, 1>(m.q.as_ptr().add(r * n_in / 2), m.scales.as_ptr().add(r * bpr), n_in, xq, xs, xsums);
        *y.add(r) = o[0][0];
        *y.add(r + 1) = o[1][0];
        r += 2;
    }
    if r < r1 {
        let o = q4_dot_rows_tc::<1, 1>(m.q.as_ptr().add(r * n_in / 2), m.scales.as_ptr().add(r * bpr), n_in, xq, xs, xsums);
        *y.add(r) = o[0][0];
    }
}

#[allow(dead_code)]
fn _unused(v: v128) -> i32 {
    hsum_i32(v)
}
