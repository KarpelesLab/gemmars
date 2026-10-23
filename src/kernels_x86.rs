//! x86-64 kernels: AVX2 + FMA, chosen at run time (`is_x86_feature_detected!`), with scalar
//! fallbacks in the same shapes so the call sites in `kernels.rs` are shared with the NEON module.
//! bf16 weights widen with a 16-bit shift; Q8 uses the `maddubs` sign trick (|w| <= 127 so the
//! int16 pair sums cannot saturate); Q4 nibbles go in unsigned with the +8 offset folded through
//! the activation block sums, like the NEON path.

use crate::quant::f16_to_f32;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU8, Ordering};

static LEVEL: AtomicU8 = AtomicU8::new(0);

/// True when AVX2 and FMA are available (cached).
#[inline]
pub fn avx2() -> bool {
    match LEVEL.load(Ordering::Relaxed) {
        0 => {
            let ok = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            LEVEL.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
            ok
        }
        l => l == 2,
    }
}

pub trait Load8: Copy {
    /// 8 consecutive weights as f32 lanes.
    unsafe fn load8(p: *const Self) -> __m256;
    fn to_f32(self) -> f32;
}
impl Load8 for u16 {
    #[inline(always)]
    unsafe fn load8(p: *const u16) -> __m256 {
        let h = _mm_loadu_si128(p as *const __m128i);
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
}
impl Load8 for f32 {
    #[inline(always)]
    unsafe fn load8(p: *const f32) -> __m256 {
        _mm256_loadu_ps(p)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

/// Loads one weight broadcast to all lanes (prefill GEMM).
pub trait LoadW4: Copy + Load8 {
    unsafe fn bcast(p: *const Self) -> __m256;
}
impl LoadW4 for u16 {
    #[inline(always)]
    unsafe fn bcast(p: *const u16) -> __m256 {
        _mm256_set1_ps(f32::from_bits((*p as u32) << 16))
    }
}
impl LoadW4 for f32 {
    #[inline(always)]
    unsafe fn bcast(p: *const f32) -> __m256 {
        _mm256_broadcast_ss(&*p)
    }
}

#[inline(always)]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

// ---- f32/bf16 dot-product tiles ------------------------------------------------------------------

/// Dot products of R weight rows against TC activation rows; each step consumes 8*G elements per
/// row with A independent accumulators per output. `n` must be a multiple of 8*G.
#[inline(always)]
pub unsafe fn kernel<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(w: *const W, ws: usize, x: *const f32, xs: usize, n: usize) -> [[f32; TC]; R] {
    if avx2() {
        kernel_avx2::<W, R, TC, G, A>(w, ws, x, xs, n)
    } else {
        kernel_scalar::<W, R, TC>(w, ws, x, xs, n)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(w: *const W, ws: usize, x: *const f32, xs: usize, n: usize) -> [[f32; TC]; R] {
    let zero = _mm256_setzero_ps();
    let mut acc = [[[zero; A]; TC]; R];
    let mut i = 0;
    while i < n {
        let mut wv = [[zero; G]; R];
        for r in 0..R {
            for g in 0..G {
                wv[r][g] = W::load8(w.add(r * ws + i + 8 * g));
            }
        }
        for c in 0..TC {
            for g in 0..G {
                let xv = _mm256_loadu_ps(x.add(c * xs + i + 8 * g));
                for r in 0..R {
                    acc[r][c][g % A] = _mm256_fmadd_ps(wv[r][g], xv, acc[r][c][g % A]);
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
                s = _mm256_add_ps(s, acc[r][c][a]);
            }
            out[r][c] = hsum(s);
        }
    }
    out
}

unsafe fn kernel_scalar<W: Load8, const R: usize, const TC: usize>(w: *const W, ws: usize, x: *const f32, xs: usize, n: usize) -> [[f32; TC]; R] {
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            let mut acc = [0f32; 8];
            for i in 0..n {
                acc[i % 8] += (*w.add(r * ws + i)).to_f32() * *x.add(c * xs + i);
            }
            out[r][c] = acc.iter().sum();
        }
    }
    out
}

/// Runs the kernel over every R-row group in `r0..r1` for tokens `c0..c0+TC`, storing to
/// `y[(c0+c)*n_out + r]`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn run_tiles<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(w: *const W, n_in: usize, x: *const f32, y: *mut f32, n_out: usize, r0: usize, r1: usize, c0: usize) {
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
pub unsafe fn block_tc<W: Load8>(w: *const W, n_in: usize, x: *const f32, y: *mut f32, n_out: usize, r0: usize, r1: usize, c0: usize, tc: usize) {
    // 16 ymm registers: keep accumulators + weight vectors under that.
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
/// weights are broadcast, accumulators are vectors over tokens.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_8x8<W: LoadW4>(w: *const W, n_in: usize, xt: *const f32, y: *mut f32, n_out: usize, rb: usize, c0: usize, c_valid: usize) {
    if avx2() {
        gemm_8x8_avx2(w, n_in, xt, y, n_out, rb, c0, c_valid)
    } else {
        for r in 0..8 {
            for c in 0..c_valid {
                let mut acc = 0f32;
                for k in 0..n_in {
                    acc += (*w.add(r * n_in + k)).to_f32() * *xt.add(k * 8 + c);
                }
                *y.add((c0 + c) * n_out + rb + r) = acc;
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_8x8_avx2<W: LoadW4>(w: *const W, n_in: usize, xt: *const f32, y: *mut f32, n_out: usize, rb: usize, c0: usize, c_valid: usize) {
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut k = 0;
    while k < n_in {
        let xv = _mm256_loadu_ps(xt.add(k * 8));
        for r in 0..8 {
            acc[r] = _mm256_fmadd_ps(W::bcast(w.add(r * n_in + k)), xv, acc[r]);
        }
        k += 1;
    }
    let mut tmp = [0f32; 8];
    for r in 0..8 {
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[r]);
        for c in 0..c_valid {
            *y.add((c0 + c) * n_out + rb + r) = tmp[c];
        }
    }
}

// ---- Q8 -----------------------------------------------------------------------------------------

/// Q8: R weight rows against TC quantized activation rows (token stride `n_in`).
#[inline(never)]
pub unsafe fn q8_dot_rows_tc<const R: usize, const TC: usize>(q: *const i8, s: *const f32, n_in: usize, xq: *const i8, xs: *const f32) -> [[f32; TC]; R] {
    if avx2() {
        q8_dot_rows_tc_avx2::<R, TC>(q, s, n_in, xq, xs)
    } else {
        let bpr = n_in / 32;
        let mut out = [[0f32; TC]; R];
        for r in 0..R {
            for c in 0..TC {
                let mut acc = 0f32;
                for b in 0..bpr {
                    let mut sum = 0i32;
                    for j in 0..32 {
                        sum += *q.add(r * n_in + b * 32 + j) as i32 * *xq.add(c * n_in + b * 32 + j) as i32;
                    }
                    acc += sum as f32 * *s.add(r * bpr + b) * *xs.add(c * bpr + b);
                }
                out[r][c] = acc;
            }
        }
        out
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q8_dot_rows_tc_avx2<const R: usize, const TC: usize>(q: *const i8, s: *const f32, n_in: usize, xq: *const i8, xs: *const f32) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let ones = _mm256_set1_epi16(1);
    let mut accf = [[_mm256_setzero_ps(); TC]; R];
    for b in 0..bpr {
        let k = b * 32;
        let mut xv = [_mm256_setzero_si256(); TC];
        for c in 0..TC {
            xv[c] = _mm256_loadu_si256(xq.add(c * n_in + k) as *const __m256i);
        }
        for r in 0..R {
            let wv = _mm256_loadu_si256(q.add(r * n_in + k) as *const __m256i);
            let wabs = _mm256_sign_epi8(wv, wv);
            let d = *s.add(r * bpr + b);
            for c in 0..TC {
                let xsgn = _mm256_sign_epi8(xv[c], wv);
                let p16 = _mm256_maddubs_epi16(wabs, xsgn);
                let p32 = _mm256_madd_epi16(p16, ones);
                let scale = _mm256_set1_ps(d * *xs.add(c * bpr + b));
                accf[r][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, accf[r][c]);
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
pub unsafe fn q4_dot_rows_tc<const R: usize, const TC: usize>(q: *const u8, s: *const u16, n_in: usize, xq: *const i8, xs: *const f32, xsums: *const i32) -> [[f32; TC]; R] {
    if avx2() {
        q4_dot_rows_tc_avx2::<R, TC>(q, s, n_in, xq, xs, xsums)
    } else {
        let bpr = n_in / 32;
        let half_row = n_in / 2;
        let mut out = [[0f32; TC]; R];
        for r in 0..R {
            for c in 0..TC {
                let mut acc = 0f32;
                for b in 0..bpr {
                    let blk = q.add(r * half_row + b * 16);
                    let mut sum = 0i32;
                    for j in 0..16 {
                        let v = *blk.add(j);
                        sum += ((v & 15) as i32 - 8) * *xq.add(c * n_in + b * 32 + j) as i32;
                        sum += ((v >> 4) as i32 - 8) * *xq.add(c * n_in + b * 32 + j + 16) as i32;
                    }
                    acc += sum as f32 * f16_to_f32(*s.add(r * bpr + b)) * *xs.add(c * bpr + b);
                }
                out[r][c] = acc;
            }
        }
        let _ = xsums;
        out
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q4_dot_rows_tc_avx2<const R: usize, const TC: usize>(q: *const u8, s: *const u16, n_in: usize, xq: *const i8, xs: *const f32, xsums: *const i32) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let half_row = n_in / 2;
    let ones = _mm256_set1_epi16(1);
    let m0f = _mm_set1_epi8(0x0F);
    let mut accf = [[_mm256_setzero_ps(); TC]; R];
    let mut corr = [[0f32; TC]; R];
    for b in 0..bpr {
        let k = b * 32;
        let mut xv = [_mm256_setzero_si256(); TC];
        for c in 0..TC {
            xv[c] = _mm256_loadu_si256(xq.add(c * n_in + k) as *const __m256i);
        }
        for r in 0..R {
            let bytes = _mm_loadu_si128(q.add(r * half_row + b * 16) as *const __m128i);
            let lo = _mm_and_si128(bytes, m0f);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(bytes), m0f);
            let wv = _mm256_set_m128i(hi, lo); // elements 0..15 = low nibbles, 16..31 = high nibbles
            let d = f16_to_f32(*s.add(r * bpr + b));
            for c in 0..TC {
                let p16 = _mm256_maddubs_epi16(wv, xv[c]);
                let p32 = _mm256_madd_epi16(p16, ones);
                let sc = d * *xs.add(c * bpr + b);
                accf[r][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(sc), accf[r][c]);
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
