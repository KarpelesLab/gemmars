//! Memory bandwidth and matmul-kernel microbenchmark.
//! usage: membench [GB]   (default 4 GB buffer)

#[path = "../kernels.rs"]
#[allow(dead_code)]
mod kernels;
#[path = "../prof.rs"]
#[allow(dead_code)]
mod prof;
#[path = "../pool.rs"]
#[allow(dead_code)]
mod pool;
#[path = "../quant.rs"]
#[allow(dead_code)]
mod quant;

use rayon::prelude::*;
use std::time::Instant;

fn bench_read(buf: &[u64], threads: usize) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let bytes = buf.len() * 8;
    let chunk = (buf.len() / (threads * 16)).max(1 << 16);
    let mut best = 0f64;
    for _ in 0..3 {
        let t = Instant::now();
        let s: u64 = pool.install(|| buf.par_chunks(chunk).map(|c| c.iter().fold(0u64, |a, &b| a ^ b)).sum());
        std::hint::black_box(s);
        let gbs = bytes as f64 / t.elapsed().as_secs_f64() / 1e9;
        best = best.max(gbs);
    }
    best
}

fn bench_matmul(w: &[u16], n_out: usize, n_in: usize, t: usize, threads: usize) -> f64 {
    pool::init(threads);
    let x = vec![0.5f32; t * n_in];
    let mut y = vec![0f32; t * n_out];
    let mut best = 0f64;
    for _ in 0..3 {
        let ti = Instant::now();
        kernels::matmul(w, n_out, n_in, &x, t, &mut y);
        let gbs = (w.len() * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9;
        best = best.max(gbs);
    }
    std::hint::black_box(&y);
    best
}

fn main() {
    let gb: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let n = gb << 27; // u64 count
    println!("filling {gb} GB buffer...");
    let buf: Vec<u64> = (0..n as u64).into_par_iter().map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let threads = [1usize, 2, 4, 6, 8, 12, 18];
    println!("\nplain read bandwidth (GB/s):");
    for &th in &threads {
        println!("  {th:>2} threads: {:>6.1}", bench_read(&buf, th));
    }
    // Reinterpret the buffer as a bf16 weight matrix [n_out, n_in].
    let n_in = 4096;
    let w: &[u16] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len() * 4) };
    let n_out = w.len() / n_in;
    let n_out = n_out.min(262_144);
    let w = &w[..n_out * n_in];
    println!("\nbf16 matmul [{n_out} x {n_in}], t=1 (GB/s of weights):");
    for &th in &threads {
        println!("  {th:>2} threads: {:>6.1}", bench_matmul(w, n_out, n_in, 1, th));
    }
    println!("\nbf16 matmul, t=4:");
    for &th in &[6usize, 12, 18] {
        println!("  {th:>2} threads: {:>6.1}", bench_matmul(w, n_out, n_in, 4, th));
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::env::var("VARIANTS").is_err() { println!("(set VARIANTS=1 for single-thread kernel variants)"); } else {
        println!("\nsingle-thread kernel variants, t=1 (GB/s):");
        let rows = n_out;
        {
            pool::init(1);
            let x = vec![0.5f32; n_in];
            let mut y = vec![0f32; n_out];
            let mut best = 0f64;
            for _ in 0..3 {
                let ti = Instant::now();
                kernels::matmul(w, n_out, n_in, &x, 1, &mut y);
                best = best.max((w.len() * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9);
            }
            println!("  {:<22} {:>6.1}", "matmul() 1 thread", best);
        }
        let x = vec![0.5f32; n_in];
        macro_rules! var {
            ($name:expr, $R:expr, $TC:expr, $G:expr, $A:expr) => {{
                let mut best = 0f64;
                let mut sink = 0f32;
                for _ in 0..3 {
                    let ti = Instant::now();
                    let mut r = 0;
                    while r + $R <= rows {
                        let o = unsafe {
                            kernels::simd::kernel::<u16, $R, $TC, $G, $A>(w.as_ptr().add(r * n_in), n_in, x.as_ptr(), n_in, n_in)
                        };
                        for rr in 0..$R {
                            for c in 0..$TC {
                                sink += o[rr][c];
                            }
                        }
                        r += $R;
                    }
                    best = best.max((rows * n_in * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9);
                }
                std::hint::black_box(sink);
                println!("  {:<22} {:>6.1}", $name, best);
            }};
        }
        var!("R4 G1 A2 (current)", 4, 1, 1, 2);
        var!("R4 G1 A1", 4, 1, 1, 1);
        var!("R4 G2 A4", 4, 1, 2, 4);
        var!("R2 G2 A4", 2, 1, 2, 4);
        var!("R2 G4 A8", 2, 1, 4, 8);
        var!("R1 G4 A8", 1, 1, 4, 8);
        var!("R8 G1 A1", 8, 1, 1, 1);
        var!("R8 G1 A2", 8, 1, 1, 2);
        var!("R8 G2 A2", 8, 1, 2, 2);
        var!("R12 G1 A1", 12, 1, 1, 1);
        var!("R16 G1 A1", 16, 1, 1, 1);
        var!("R6 G1 A2", 6, 1, 1, 2);
        var!("R8 TC2 G1 A1", 8, 2, 1, 1);
        var!("R4 TC2 G1 A2 (cur)", 4, 2, 1, 2);
        var!("R4 TC4 G1 A1 (cur)", 4, 4, 1, 1);
        var!("R8 TC4 G1 A1", 8, 4, 1, 1);
        var!("R1 G2 A4", 1, 1, 2, 4);
        var!("R1 G4 A8", 1, 1, 4, 8);
        var!("R1 G8 A16", 1, 1, 8, 16);
        var!("R2 G2 A4", 2, 1, 2, 4);
        var!("R2 G4 A8", 2, 1, 4, 8);
        // AND/SHL conversion variant, single row, A accumulators.
        {
            use std::arch::aarch64::*;
            let half = n_in / 2;
            let xe: Vec<f32> = (0..half).map(|j| x[2 * j]).collect();
            let xo: Vec<f32> = (0..half).map(|j| x[2 * j + 1]).collect();
            macro_rules! andshl {
                ($name:expr, $A:expr) => {{
                    let mut best = 0f64;
                    let mut sink = 0f32;
                    for _ in 0..3 {
                        let ti = Instant::now();
                        for r in 0..rows {
                            unsafe {
                                let wp = w.as_ptr().add(r * n_in) as *const u32;
                                let mask = vdupq_n_u32(0xFFFF_0000);
                                let mut acc = [vdupq_n_f32(0.0); $A];
                                let mut i = 0;
                                while i < n_in {
                                    for g in 0..$A / 2 {
                                        let w32 = vld1q_u32(wp.add((i + 8 * g) / 2));
                                        let hi = vreinterpretq_f32_u32(vandq_u32(w32, mask));
                                        let lo = vreinterpretq_f32_u32(vshlq_n_u32::<16>(w32));
                                        acc[2 * g] = vfmaq_f32(acc[2 * g], lo, vld1q_f32(xe.as_ptr().add((i + 8 * g) / 2)));
                                        acc[2 * g + 1] = vfmaq_f32(acc[2 * g + 1], hi, vld1q_f32(xo.as_ptr().add((i + 8 * g) / 2)));
                                    }
                                    i += 4 * $A;
                                }
                                let mut sum = acc[0];
                                for a in 1..$A {
                                    sum = vaddq_f32(sum, acc[a]);
                                }
                                sink += vaddvq_f32(sum);
                            }
                        }
                        best = best.max((rows * n_in * 2) as f64 / ti.elapsed().as_secs_f64() / 1e9);
                    }
                    std::hint::black_box(sink);
                    println!("  {:<22} {:>6.1}", $name, best);
                }};
            }
            andshl!("ANDSHL R1 A4", 4);
            andshl!("ANDSHL R1 A8", 8);
            andshl!("ANDSHL R1 A16", 16);
        }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Prefill GEMM kernel compute ceiling: single thread, operands resident in L1 / L2.
        for (name, rows, k, toks) in [("L1: 8 rows x 512 k x 8 tok", 8usize, 512usize, 8usize), ("L2: 8 rows x 4096 k x 64 tok", 8, 4096, 64), ("L2: 64 rows x 2560 k x 64 tok", 64, 2560, 64)] {
            let wq: Vec<u16> = (0..rows * k).map(|i| 0x3f80 ^ (i as u16 & 0x3f)).collect();
            let xt: Vec<f32> = vec![0.5f32; toks * k]; // packed [tile][k][8]
            let mut y = vec![0f32; toks * rows];
            let flops_per_pass = 2.0 * rows as f64 * k as f64 * toks as f64;
            let passes = (2e10 / flops_per_pass).max(1.0) as usize;
            let t0 = Instant::now();
            for _ in 0..passes {
                let mut r = 0;
                while r + 8 <= rows {
                    for tile in 0..toks / 8 {
                        unsafe {
                            kernels::simd::gemm_8x8::<u16>(wq.as_ptr().add(r * k), k, xt.as_ptr().add(tile * k * 8), y.as_mut_ptr(), rows, r, tile * 8, 8);
                        }
                    }
                    r += 8;
                }
            }
            std::hint::black_box(&y);
            println!("gemm_8x8 1 thread {name}: {:.0} GFLOP/s", flops_per_pass * passes as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Quantized decode kernels, single thread, over ~1 GB of weights.
        let n_in = 4096usize;
        let rows = 131072usize; // q8: 512 MB; q4: 256 MB
        let wb: Vec<u16> = (0..rows * n_in).map(|i| 0x3f80 ^ (i as u16 & 0x7f)).collect();
        pool::init(1);
        let q8 = quant::quantize_bf16(&wb, rows, n_in);
        let q4 = quant::quantize_q4(&wb, rows, n_in);
        let x: Vec<f32> = (0..4 * n_in).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let xa = quant::quantize_act(&x, 4, n_in);
        let bpr = n_in / 32;
        println!("\nquantized decode kernels, 1 thread (GB/s of weight bytes | weight-equivalent bf16 GB/s):");
        macro_rules! bench {
            ($name:expr, $bytes:expr, $body:expr) => {{
                let mut best = 0f64;
                let mut sink = 0f32;
                for _ in 0..3 {
                    let t0 = Instant::now();
                    sink += $body;
                    best = best.max($bytes as f64 / t0.elapsed().as_secs_f64() / 1e9);
                }
                std::hint::black_box(sink);
                println!("  {:<24} {:>6.1} | {:>6.1}", $name, best, best * (rows * n_in * 2) as f64 / $bytes as f64);
            }};
        }
        unsafe {
            bench!("q8 tc=4 (R2)", q8.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q8_dot_rows_tc::<2, 4>(q8.q.as_ptr().add(r * n_in), q8.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr());
                    s += o[0][0] + o[0][1] + o[0][2] + o[0][3] + o[1][0] + o[1][1] + o[1][2] + o[1][3];
                    r += 2;
                }
                s
            });
            bench!("q4 tc=4 (R2)", q4.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q4_dot_rows_tc::<2, 4>(q4.q.as_ptr().add(r * n_in / 2), q4.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr(), xa.sums.as_ptr());
                    s += o[0][0] + o[0][1] + o[0][2] + o[0][3] + o[1][0] + o[1][1] + o[1][2] + o[1][3];
                    r += 2;
                }
                s
            });
            bench!("q8 t=1 (R2)", q8.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q8_dot_rows::<2>(q8.q.as_ptr().add(r * n_in), q8.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr());
                    s += o[0] + o[1];
                    r += 2;
                }
                s
            });
            for tc in [2usize, 4] {
                let name = format!("q8 tc={tc} (R1)");
                bench!(name, q8.bytes() , {
                    let mut s = 0f32;
                    for r in 0..rows {
                        let o: [f32; 4] = if tc == 4 {
                            kernels::simd::q8_dot_rows_tc::<1, 4>(q8.q.as_ptr().add(r * n_in), q8.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr())[0]
                        } else {
                            let o2 = kernels::simd::q8_dot_rows_tc::<1, 2>(q8.q.as_ptr().add(r * n_in), q8.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr())[0];
                            [o2[0], o2[1], 0.0, 0.0]
                        };
                        s += o[0] + o[1] + o[2] + o[3];
                    }
                    s
                });
            }
            bench!("q4 t=1 (R2)", q4.bytes(), {
                let mut s = 0f32;
                let mut r = 0;
                while r + 2 <= rows {
                    let o = kernels::simd::q4_dot_rows::<2>(q4.q.as_ptr().add(r * n_in / 2), q4.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr(), xa.sums.as_ptr());
                    s += o[0] + o[1];
                    r += 2;
                }
                s
            });
            for tc in [2usize, 4] {
                let name = format!("q4 tc={tc} (R1)");
                bench!(name, q4.bytes(), {
                    let mut s = 0f32;
                    for r in 0..rows {
                        let o: [f32; 4] = if tc == 4 {
                            kernels::simd::q4_dot_rows_tc::<1, 4>(q4.q.as_ptr().add(r * n_in / 2), q4.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr(), xa.sums.as_ptr())[0]
                        } else {
                            let o2 = kernels::simd::q4_dot_rows_tc::<1, 2>(q4.q.as_ptr().add(r * n_in / 2), q4.scales.as_ptr().add(r * bpr), n_in, xa.q.as_ptr(), xa.scales.as_ptr(), xa.sums.as_ptr())[0];
                            [o2[0], o2[1], 0.0, 0.0]
                        };
                        s += o[0] + o[1] + o[2] + o[3];
                    }
                    s
                });
            }
        }
    }
    println!("\nbf16 matmul, t=32 (f32 conversion path), GFLOP/s:");
    for &th in &[6usize, 12, 18] {
        let gbs = bench_matmul(w, n_out, n_in, 32, th);
        println!("  {th:>2} threads: {:>6.1} GB/s = {:>7.0} GFLOP/s", gbs, gbs / 2.0 * 2.0 * 32.0);
    }
}
