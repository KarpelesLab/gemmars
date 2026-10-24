//! Metal compute backend (Apple Silicon), reached through the Objective-C runtime with `dlsym`:
//! no Metal framework is linked and nothing is needed at build time. Shaders are compiled from
//! source at start-up. Weights live in page-aligned host memory (quantized matrices and whole
//! safetensors blobs) and are wrapped as shared `MTLBuffer`s without copying (unified memory), so
//! "uploading" a model costs nothing.
//!
//! Only prefill-sized matmuls run here. Decode is memory-bound and the CPU already streams weights
//! at ~280 GB/s; with ~0.15 ms per Metal submission, offloading single-token matmuls one by one is
//! a loss (the GPU's 450 GB/s would need every op of the forward pass on the device). Prefill is
//! compute-bound and the GEMM below reaches ~11 TFLOP/s against ~1.3 on the CPU.
//!
//! GEMM: a threadgroup of 128 threads (2x2 SIMD-groups) computes a 64 (out) x 64 (tokens) tile.
//! Per 32-wide K block the weights are dequantized to f32 in threadgroup memory; x fragments are
//! read straight from device memory with transposed 8x8 loads; `simdgroup_float8x8` FMAs do the
//! math. Accumulators must be named variables: arrays of `simdgroup_float8x8` are spilled to
//! memory by the shader compiler and run 12x slower.

use crate::ggml::GgmlType;
use crate::gpu::{Backend, DeviceWeight, Job, WeightRef};
use crate::platform::gpu::metal::{Id, Objc, Sel};
use crate::quant::PAGE;
use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, CString, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const TILE: usize = 64; // out rows and tokens per threadgroup
const MIN_T: usize = 16; // smaller token counts stay on the CPU
const MIN_FLOPS: f64 = 3e8; // ~0.15 ms submission cost vs ~1 TFLOP/s on the CPU

const SHADER_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct P { uint n_out; uint n_in; uint w_off; uint s_off; };
#define LD 36   // threadgroup row stride (floats): 32 + padding against bank conflicts

// ---- Dequantizers: 32 consecutive k values of one row, as 8 float4s -------------------------
struct Q4W { device const uchar* q; device const half* s; uint bpr; uint half_row; };
struct Q8W { device const char* q; device const float* s; uint bpr; uint n_in; };
struct BfW { device const ushort* w; uint n_in; };

inline float4 nib_lo(uint u) { return float4(float(int(u & 15) - 8), float(int((u >> 8) & 15) - 8), float(int((u >> 16) & 15) - 8), float(int((u >> 24) & 15) - 8)); }
inline float4 nib_hi(uint u) { return float4(float(int((u >> 4) & 15) - 8), float(int((u >> 12) & 15) - 8), float(int((u >> 20) & 15) - 8), float(int((u >> 28) & 15) - 8)); }
inline void load(Q4W w, uint row, uint b, thread float4* out) {
    float d = float(w.s[(ulong)row * w.bpr + b]);
    uint4 v = *(device const uint4*)(w.q + (ulong)row * w.half_row + b * 16);
    out[0] = nib_lo(v.x) * d; out[1] = nib_lo(v.y) * d; out[2] = nib_lo(v.z) * d; out[3] = nib_lo(v.w) * d;
    out[4] = nib_hi(v.x) * d; out[5] = nib_hi(v.y) * d; out[6] = nib_hi(v.z) * d; out[7] = nib_hi(v.w) * d;
}
// Affine Q4 (MLX): a bf16 offset per group of `gk` blocks is added to every element.
struct Q4A { Q4W b; device const ushort* o; uint gpr; uint gk; };
inline void load(Q4A w, uint row, uint b, thread float4* out) {
    load(w.b, row, b, out);
    float m = as_type<float>(uint(w.o[(ulong)row * w.gpr + b / w.gk]) << 16);
    for (uint i = 0; i < 8; i++) out[i] += m;
}
inline void load(Q8W w, uint row, uint b, thread float4* out) {
    float d = w.s[(ulong)row * w.bpr + b];
    device const char4* p = (device const char4*)(w.q + (ulong)row * w.n_in + b * 32);
    for (uint i = 0; i < 8; i++) out[i] = float4(p[i]) * d;
}
inline float4 bf4(packed_ushort4 u4) {
    ushort4 u = ushort4(u4);
    return float4(as_type<float>(uint(u.x) << 16), as_type<float>(uint(u.y) << 16), as_type<float>(uint(u.z) << 16), as_type<float>(uint(u.w) << 16));
}
// f32 "weights": attention keys and transposed values.
struct F32W { device const float* w; uint n_in; };
inline void load(F32W w, uint row, uint b, thread float4* out) {
    device const float4* p = (device const float4*)(w.w + (ulong)row * w.n_in + b * 32);
    for (uint i = 0; i < 8; i++) out[i] = p[i];
}
inline void load(BfW w, uint row, uint b, thread float4* out) {
    // bf16 tensors inside a safetensors blob are only 2-byte aligned: packed loads.
    device const packed_ushort4* p = (device const packed_ushort4*)(w.w + (ulong)row * w.n_in + b * 32);
    for (uint i = 0; i < 8; i++) out[i] = bf4(p[i]);
}

// ---- GEMM ------------------------------------------------------------------------------------
#define MMA(I, J) simdgroup_multiply_accumulate(c##I##J, a##I, b##J, c##I##J)
#define ST(I, J) simdgroup_store(c##I##J, y + (ulong)(t0 + sc + J * 8) * p.n_out + o0 + sr + I * 8, p.n_out, ulong2(0, 0), true)
template <typename W>
inline void gemm_body(W w, device const float* x, device float* y, constant P& p, uint2 tg, uint tid, uint sg, threadgroup float* Ws) {
    uint o0 = tg.x * 64, t0 = tg.y * 64;
    uint nblk = p.n_in / 32;
    simdgroup_float8x8 c00(0.f), c01(0.f), c02(0.f), c03(0.f), c10(0.f), c11(0.f), c12(0.f), c13(0.f),
                       c20(0.f), c21(0.f), c22(0.f), c23(0.f), c30(0.f), c31(0.f), c32(0.f), c33(0.f);
    uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    uint drow = tid / 2, dhalf = tid % 2;   // each thread dequantizes half a block of one row
    device const float* xs = x + (ulong)(t0 + sc) * p.n_in;
    for (uint b = 0; b < nblk; b++) {
        float4 wv[8];
        load(w, o0 + drow, b, wv);
        threadgroup float4* wd = (threadgroup float4*)(Ws + drow * LD + dhalf * 16);
        wd[0] = wv[dhalf * 4]; wd[1] = wv[dhalf * 4 + 1]; wd[2] = wv[dhalf * 4 + 2]; wd[3] = wv[dhalf * 4 + 3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1, b2, b3;
            simdgroup_load(a0, Ws + (sr + 0) * LD + k, LD);
            simdgroup_load(a1, Ws + (sr + 8) * LD + k, LD);
            simdgroup_load(a2, Ws + (sr + 16) * LD + k, LD);
            simdgroup_load(a3, Ws + (sr + 24) * LD + k, LD);
            simdgroup_load(b0, xs + (ulong)0 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            simdgroup_load(b1, xs + (ulong)8 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            simdgroup_load(b2, xs + (ulong)16 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            simdgroup_load(b3, xs + (ulong)24 * p.n_in + b * 32 + k, p.n_in, ulong2(0, 0), true);
            MMA(0, 0); MMA(0, 1); MMA(0, 2); MMA(0, 3); MMA(1, 0); MMA(1, 1); MMA(1, 2); MMA(1, 3);
            MMA(2, 0); MMA(2, 1); MMA(2, 2); MMA(2, 3); MMA(3, 0); MMA(3, 1); MMA(3, 2); MMA(3, 3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // c_ij is [out 8][tok 8]; y is [tok][n_out], so store transposed.
    ST(0, 0); ST(0, 1); ST(0, 2); ST(0, 3); ST(1, 0); ST(1, 1); ST(1, 2); ST(1, 3);
    ST(2, 0); ST(2, 1); ST(2, 2); ST(2, 3); ST(3, 0); ST(3, 1); ST(3, 2); ST(3, 3);
}
#define GEMM(NAME, WT, MAKE) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], \
                 device const float* x [[buffer(2)]], device float* y [[buffer(3)]], constant P& p [[buffer(4)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float Ws[64 * LD]; \
    WT w = MAKE; gemm_body<WT>(w, x, y, p, tg, tid, sg, Ws); }
// ---- Convolution as an implicit GEMM -------------------------------------------------------------
// K runs over (ky, kx, ci) with cin % 32 == 0, so each 32-wide block of K is one tap's contiguous
// channel slice. Every threadgroup gathers its 64 pixels x 32 channels into threadgroup memory
// (zeros outside the image), then runs the same simdgroup MMAs as the GEMM.
// mode: 0 plain, 1 group norm (stats per group of cpg channels), 2 layer norm (stats per input
// pixel); st holds (mean, 1/std); gb holds gamma[cin] then beta[cin].
struct CP { uint n_out; uint n_in; uint w_off; uint s_off; uint h; uint w; uint cin; uint k; uint stride; uint ow; uint npix; uint pad; uint mode; uint silu; uint cpg; uint _p; };
inline float4 norm4(float4 v, uint c, uint pin, constant CP& p, device const float2* st, device const float* gb) {
    float o[4] = { v.x, v.y, v.z, v.w };
    for (uint i = 0; i < 4; i++) {
        uint ch = c + i;
        float2 m = p.mode == 1 ? st[ch / p.cpg] : st[pin];
        float y = (o[i] - m.x) * m.y * gb[ch] + gb[p.cin + ch];
        o[i] = p.silu != 0 ? y / (1.0f + exp(-y)) : y;
    }
    return float4(o[0], o[1], o[2], o[3]);
}
template <typename W>
inline void conv_body(W wt, device const float* x, device float* y, constant CP& p, device const float2* st, device const float* gb, uint2 tg, uint tid, uint sg, threadgroup float* Ws, threadgroup float* Xs) {
    uint o0 = tg.x * 64, t0 = tg.y * 64;
    uint nblk = p.n_in / 32;
    simdgroup_float8x8 c00(0.f), c01(0.f), c02(0.f), c03(0.f), c10(0.f), c11(0.f), c12(0.f), c13(0.f),
                       c20(0.f), c21(0.f), c22(0.f), c23(0.f), c30(0.f), c31(0.f), c32(0.f), c33(0.f);
    uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    uint drow = tid / 2, dhalf = tid % 2;
    uint pix = t0 + drow;
    int oy = int(pix / p.ow), ox = int(pix % p.ow);
    for (uint b = 0; b < nblk; b++) {
        float4 wv[8];
        load(wt, o0 + drow, b, wv);
        threadgroup float4* wd = (threadgroup float4*)(Ws + drow * LD + dhalf * 16);
        wd[0] = wv[dhalf * 4]; wd[1] = wv[dhalf * 4 + 1]; wd[2] = wv[dhalf * 4 + 2]; wd[3] = wv[dhalf * 4 + 3];
        uint kk = b * 32, tap = kk / p.cin, ci0 = kk - tap * p.cin;
        int ky = int(tap / p.k), kx = int(tap % p.k);
        int iy = oy * int(p.stride) + ky - int(p.pad), ix = ox * int(p.stride) + kx - int(p.pad);
        threadgroup float4* xd = (threadgroup float4*)(Xs + drow * LD + dhalf * 16);
        if (pix < p.npix && iy >= 0 && ix >= 0 && iy < int(p.h) && ix < int(p.w)) {
            device const float4* src = (device const float4*)(x + ((ulong)iy * p.w + ix) * p.cin + ci0 + dhalf * 16);
            float4 v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
            if (p.mode != 0) {
                uint c0 = ci0 + dhalf * 16, pin = uint(iy) * p.w + uint(ix);
                v0 = norm4(v0, c0, pin, p, st, gb); v1 = norm4(v1, c0 + 4, pin, p, st, gb);
                v2 = norm4(v2, c0 + 8, pin, p, st, gb); v3 = norm4(v3, c0 + 12, pin, p, st, gb);
            }
            xd[0] = v0; xd[1] = v1; xd[2] = v2; xd[3] = v3;
        } else {
            xd[0] = 0.f; xd[1] = 0.f; xd[2] = 0.f; xd[3] = 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1, b2, b3;
            simdgroup_load(a0, Ws + (sr + 0) * LD + k, LD);
            simdgroup_load(a1, Ws + (sr + 8) * LD + k, LD);
            simdgroup_load(a2, Ws + (sr + 16) * LD + k, LD);
            simdgroup_load(a3, Ws + (sr + 24) * LD + k, LD);
            simdgroup_load(b0, Xs + (sc + 0) * LD + k, LD, ulong2(0, 0), true);
            simdgroup_load(b1, Xs + (sc + 8) * LD + k, LD, ulong2(0, 0), true);
            simdgroup_load(b2, Xs + (sc + 16) * LD + k, LD, ulong2(0, 0), true);
            simdgroup_load(b3, Xs + (sc + 24) * LD + k, LD, ulong2(0, 0), true);
            MMA(0, 0); MMA(0, 1); MMA(0, 2); MMA(0, 3); MMA(1, 0); MMA(1, 1); MMA(1, 2); MMA(1, 3);
            MMA(2, 0); MMA(2, 1); MMA(2, 2); MMA(2, 3); MMA(3, 0); MMA(3, 1); MMA(3, 2); MMA(3, 3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ST(0, 0); ST(0, 1); ST(0, 2); ST(0, 3); ST(1, 0); ST(1, 1); ST(1, 2); ST(1, 3);
    ST(2, 0); ST(2, 1); ST(2, 2); ST(2, 3); ST(3, 0); ST(3, 1); ST(3, 2); ST(3, 3);
}
#define CONV(NAME, WT, MAKE) \
kernel void NAME(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]], \
                 device const float* x [[buffer(2)]], device float* y [[buffer(3)]], constant CP& p [[buffer(4)]], \
                 device const float2* st [[buffer(5)]], device const float* gb [[buffer(6)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float Ws[64 * LD]; threadgroup float Xs[64 * LD]; \
    WT w = MAKE; conv_body<WT>(w, x, y, p, st, gb, tg, tid, sg, Ws, Xs); }

#define MK_Q4 Q4W{ (device const uchar*)qb + p.w_off, (device const half*)sb + p.s_off, p.n_in / 32, p.n_in / 2 }
#define MK_Q8 Q8W{ (device const char*)qb + p.w_off, (device const float*)sb + p.s_off, p.n_in / 32, p.n_in }
#define MK_BF BfW{ (device const ushort*)qb + p.w_off, p.n_in }
#define MK_F32 F32W{ (device const float*)qb + p.w_off, p.n_in }
GEMM(gemm_bf, BfW, MK_BF) GEMM(gemm_q8, Q8W, MK_Q8) GEMM(gemm_q4, Q4W, MK_Q4) GEMM(gemm_f32, F32W, MK_F32)
struct OP { uint o_off; uint gpr; uint gk; uint _p; };
kernel void gemm_q4a(device const void* qb [[buffer(0)]], device const void* sb [[buffer(1)]],
                     device const float* x [[buffer(2)]], device float* y [[buffer(3)]], constant P& p [[buffer(4)]],
                     device const ushort* ob [[buffer(5)]], constant OP& op [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Ws[64 * LD];
    Q4A w = { MK_Q4, ob + op.o_off, op.gpr, op.gk }; gemm_body<Q4A>(w, x, y, p, tg, tid, sg, Ws); }
CONV(conv_bf, BfW, MK_BF) CONV(conv_q8, Q8W, MK_Q8) CONV(conv_q4, Q4W, MK_Q4)

// ---- Normalization statistics and softmax --------------------------------------------------------
// Two passes (mean, then the mean squared deviation) for accuracy.
struct GS { uint n; uint c; uint groups; float eps; };
kernel void group_stats(device const float* x [[buffer(0)]], device float2* st [[buffer(1)]], constant GS& p [[buffer(2)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    uint cpg = p.c / p.groups, total = p.n * cpg;
    float s = 0.f;
    for (uint i = tid; i < total; i += 256) s += x[(ulong)(i / cpg) * p.c + g * cpg + i % cpg];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = (red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7]) / float(total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.f;
    for (uint i = tid; i < total; i += 256) { float d = x[(ulong)(i / cpg) * p.c + g * cpg + i % cpg] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        st[g] = float2(mean, rsqrt(t / float(total) + p.eps));
    }
}
struct RS { uint c; float eps; };
kernel void row_stats(device const float* x [[buffer(0)]], device float2* st [[buffer(1)]], constant RS& p [[buffer(2)]],
                      uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    device const float* row = x + (ulong)r * p.c;
    float s = 0.f;
    for (uint i = lane; i < p.c; i += 32) s += row[i];
    float mean = simd_sum(s) / float(p.c);
    float v = 0.f;
    for (uint i = lane; i < p.c; i += 32) { float d = row[i] - mean; v += d * d; }
    v = simd_sum(v) / float(p.c);
    if (lane == 0) st[r] = float2(mean, rsqrt(v + p.eps));
}
// Row softmax of scaled scores in place; columns from `cols` (or after the row, if causal) to `ld`
// become 0 so the following product ignores the padding.
struct SM { uint cols; uint ld; float scale; uint causal; };
kernel void softmax_rows(device float* s [[buffer(0)]], constant SM& p [[buffer(1)]],
                         uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    device float* row = s + (ulong)r * p.ld;
    uint lim = p.causal != 0 ? min(r + 1, p.cols) : p.cols;
    float m = -INFINITY;
    for (uint i = lane; i < lim; i += 32) m = max(m, row[i] * p.scale);
    m = simd_max(m);
    float sum = 0.f;
    for (uint i = lane; i < lim; i += 32) { float e = exp(row[i] * p.scale - m); row[i] = e; sum += e; }
    float inv = 1.0f / simd_sum(sum);
    for (uint i = lane; i < p.ld; i += 32) row[i] = i < lim ? row[i] * inv : 0.f;
}
"#;

/// GGUF (ggml) block formats for the GEMM: one dequantizer per type producing the 32 values of
/// K block `b` of a row (a sub-block of a 256-element super-block for the K and i-quants). The
/// weights are the raw tensor bytes; `P::s_off` carries the row length in bytes. The i-quant grids
/// (`GGUF_TABLES`) are appended from `ggml_tables.rs` when the library is built.
const GGUF_SHADER_SRC: &str = r#"
struct GgW { device const uchar* q; uint rb; };
#define MK_GG(T) T{ { (device const uchar*)qb + p.w_off, p.s_off } }
inline float h2f(device const uchar* p) { return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8)))); }
inline uint rd16(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8); }
inline uint rd32(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }
inline uchar4 ld4(device const uchar* p) { return uchar4(*(device const packed_uchar4*)p); }
inline float4 f4(uchar4 v) { return float4(v); }
inline float4 f4(char4 v) { return float4(v); }
inline device const uchar* sblk(GgW w, uint row, uint b, uint tsz) { return w.q + (ulong)row * w.rb + (ulong)(b / 8) * tsz; }
// 8 values of an i-quant grid entry (bytes, lowest first) with sign bits `s` (bit j: value j negative).
inline void grid8(ulong g, uint s, float db, thread float4* o) {
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = db * float((g >> (8 * j)) & 255) * ((s >> j) & 1 ? -1.f : 1.f);
    o[0] = float4(v[0], v[1], v[2], v[3]); o[1] = float4(v[4], v[5], v[6], v[7]);
}
inline void grid4x2(uint g1, uint g2, uint s, float db, thread float4* o) {
    grid8(ulong(g1) | (ulong(g2) << 32), s, db, o);
}
inline uint ksign(uint i) { return i | ((popcount(i) & 1) << 7); }

struct Q4_0W { GgW g; };
inline void load(Q4_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + b * 18;
    float d = h2f(B);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 2 + 4 * i);
        out[i] = (f4(v & uchar(15)) - 8.f) * d;
        out[i + 4] = (f4(v >> uchar(4)) - 8.f) * d;
    }
}
struct IQ4_NLW { GgW g; };
inline float4 kv4(uchar4 v) { return float4(kvalues_iq4nl[v.x], kvalues_iq4nl[v.y], kvalues_iq4nl[v.z], kvalues_iq4nl[v.w]); }
inline void load(IQ4_NLW w, uint row, uint b, thread float4* out) {
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + b * 18;
    float d = h2f(B);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 2 + 4 * i);
        out[i] = kv4(v & uchar(15)) * d;
        out[i + 4] = kv4(v >> uchar(4)) * d;
    }
}
struct Q8_0W { GgW g; };
inline void load(Q8_0W w, uint row, uint b, thread float4* out) {
    device const uchar* B = w.g.q + (ulong)row * w.g.rb + b * 34;
    float d = h2f(B);
    for (uint i = 0; i < 8; i++) out[i] = f4(as_type<char4>(ld4(B + 2 + 4 * i))) * d;
}
inline void scale_min_k4(uint j, device const uchar* s, thread float& sc, thread float& mn) {
    if (j < 4) { sc = float(s[j] & 63); mn = float(s[j + 4] & 63); }
    else { sc = float((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4)); mn = float((s[j + 4] >> 4) | ((s[j] >> 6) << 4)); }
}
struct Q4_KW { GgW g; };
inline void load(Q4_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 144);
    uint j = b % 8;
    float sc, mn; scale_min_k4(j, B + 4, sc, mn);
    float d1 = h2f(B) * sc, m1 = h2f(B + 2) * mn;
    device const uchar* q = B + 16 + 32 * (j / 2);
    uchar sh = uchar(4 * (j % 2));
    for (uint i = 0; i < 8; i++) out[i] = f4((ld4(q + 4 * i) >> sh) & uchar(15)) * d1 - m1;
}
struct Q5_KW { GgW g; };
inline void load(Q5_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 176);
    uint j = b % 8;
    float sc, mn; scale_min_k4(j, B + 4, sc, mn);
    float d1 = h2f(B) * sc, m1 = h2f(B + 2) * mn;
    device const uchar* q = B + 48 + 32 * (j / 2);
    uchar sh = uchar(4 * (j % 2));
    for (uint i = 0; i < 8; i++) {
        uchar4 hb = ((ld4(B + 16 + 4 * i) >> uchar(j)) & uchar(1)) << uchar(4);
        out[i] = f4(((ld4(q + 4 * i) >> sh) & uchar(15)) | hb) * d1 - m1;
    }
}
struct Q6_KW { GgW g; };
inline void load(Q6_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 210);
    uint j = b % 8, n = j / 4, jj = j % 4;
    float d = h2f(B + 208);
    device const uchar* ql = B + 64 * n + 32 * (jj & 1);
    device const uchar* qh = B + 128 + 32 * n;
    uchar lsh = uchar(4 * (jj >> 1)), hsh = uchar(2 * jj);
    for (uint i = 0; i < 8; i++) {
        float dl = d * float(as_type<char>(B[192 + 8 * n + 2 * jj + i / 4]));
        uchar4 q = ((ld4(ql + 4 * i) >> lsh) & uchar(15)) | (((ld4(qh + 4 * i) >> hsh) & uchar(3)) << uchar(4));
        out[i] = (f4(q) - 32.f) * dl;
    }
}
struct Q3_KW { GgW g; };
inline void load(Q3_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 110);
    uint j = b % 8, n = j / 4, jj = j % 4;
    float d = h2f(B + 108);
    uchar sh = uchar(2 * jj), m = uchar(1 << (4 * n + jj));
    for (uint h = 0; h < 2; h++) {
        uint k = 8 * n + 2 * jj + h;
        uint lo = k < 8 ? (B[96 + k] & 15) : (B[96 + k - 8] >> 4);
        uint hi = (B[104 + k % 4] >> (2 * (k / 4))) & 3;
        float dl = d * float(int(lo | (hi << 4)) - 32);
        for (uint i = 0; i < 4; i++) {
            uint l = 16 * h + 4 * i;
            uchar4 q = (ld4(B + 32 + 32 * n + l) >> sh) & uchar(3);
            uchar4 hm = ld4(B + l) & m;
            out[4 * h + i] = (f4(q) - select(float4(4.f), float4(0.f), hm != uchar4(0))) * dl;
        }
    }
}
struct Q2_KW { GgW g; };
inline void load(Q2_KW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 84);
    uint j = b % 8, n = j / 4, jj = j % 4;
    float d = h2f(B + 80), dmin = h2f(B + 82);
    uchar sh = uchar(2 * jj);
    for (uint h = 0; h < 2; h++) {
        uchar s = B[8 * n + 2 * jj + h];
        float dl = d * float(s & 15), ml = dmin * float(s >> 4);
        for (uint i = 0; i < 4; i++) out[4 * h + i] = f4((ld4(B + 16 + 32 * n + 16 * h + 4 * i) >> sh) & uchar(3)) * dl - ml;
    }
}
struct IQ4_XSW { GgW g; };
inline void load(IQ4_XSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 136);
    uint j = b % 8;
    int ls = int(((B[4 + j / 2] >> (4 * (j % 2))) & 15) | (((rd16(B + 2) >> (2 * j)) & 3) << 4)) - 32;
    float dl = h2f(B) * float(ls);
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(B + 8 + 16 * j + 4 * i);
        out[i] = kv4(v & uchar(15)) * dl;
        out[i + 4] = kv4(v >> uchar(4)) * dl;
    }
}
struct IQ2_XXSW { GgW g; };
inline void load(IQ2_XXSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 66);
    uint j = b % 8;
    uint a0 = rd32(B + 2 + 8 * j), a1 = rd32(B + 6 + 8 * j);
    float db = h2f(B) * (0.5f + float(a1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; l++) grid8(iq2xxs_grid[(a0 >> (8 * l)) & 255], ksign((a1 >> (7 * l)) & 127), db, out + 2 * l);
}
struct IQ2_XSW { GgW g; };
inline void load(IQ2_XSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 74);
    uint j = b % 8;
    float d = h2f(B);
    uchar s = B[66 + j];
    for (uint l = 0; l < 4; l++) {
        uint q = rd16(B + 2 + 2 * (4 * j + l));
        float db = d * (0.5f + float(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
        grid8(iq2xs_grid[q & 511], ksign(q >> 9), db, out + 2 * l);
    }
}
struct IQ2_SW { GgW g; };
inline void load(IQ2_SW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 82);
    uint j = b % 8;
    float d = h2f(B);
    uchar s = B[74 + j];
    uint qh = B[66 + j];
    for (uint l = 0; l < 4; l++) {
        float db = d * (0.5f + float(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
        uint idx = uint(B[2 + 4 * j + l]) | ((qh << (8 - 2 * l)) & 0x300);
        grid8(iq2s_grid[idx], B[34 + 4 * j + l], db, out + 2 * l);
    }
}
struct IQ3_XXSW { GgW g; };
inline void load(IQ3_XXSW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 98);
    uint j = b % 8;
    uint aux = rd32(B + 66 + 4 * j);
    float db = h2f(B) * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; l++)
        grid4x2(iq3xxs_grid[B[2 + 8 * j + 2 * l]], iq3xxs_grid[B[3 + 8 * j + 2 * l]], ksign((aux >> (7 * l)) & 127), db, out + 2 * l);
}
struct IQ3_SW { GgW g; };
inline void load(IQ3_SW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 110);
    uint j = b % 8;
    float db = h2f(B) * float(1 + 2 * ((B[106 + j / 2] >> (4 * (j % 2))) & 15));
    uint qh = B[66 + j];
    for (uint l = 0; l < 4; l++) {
        uint i1 = uint(B[2 + 8 * j + 2 * l]) | ((qh << (8 - 2 * l)) & 256);
        uint i2 = uint(B[3 + 8 * j + 2 * l]) | ((qh << (7 - 2 * l)) & 256);
        grid4x2(iq3s_grid[i1], iq3s_grid[i2], B[74 + 4 * j + l], db, out + 2 * l);
    }
}
// Ternary grids: values -1, 0, 1 as signed bytes, plus +-1/8.
inline void grid1(ulong g, float dl, float delta, thread float4* o) {
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = dl * (float(as_type<char>(uchar((g >> (8 * j)) & 255))) + delta);
    o[0] = float4(v[0], v[1], v[2], v[3]); o[1] = float4(v[4], v[5], v[6], v[7]);
}
struct IQ1_SW { GgW g; };
inline void load(IQ1_SW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 50);
    uint j = b % 8;
    uint qh = rd16(B + 34 + 2 * j);
    float dl = h2f(B) * float(2 * ((qh >> 12) & 7) + 1);
    float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    for (uint l = 0; l < 4; l++) grid1(iq1s_grid[uint(B[2 + 4 * j + l]) | (((qh >> (3 * l)) & 7) << 8)], dl, delta, out + 2 * l);
}
struct IQ1_MW { GgW g; };
inline void load(IQ1_MW w, uint row, uint b, thread float4* out) {
    device const uchar* B = sblk(w.g, row, b, 56);
    uint j = b % 8;
    uint s0 = rd16(B + 48), s1 = rd16(B + 50), s2 = rd16(B + 52), s3 = rd16(B + 54);
    float d = float(as_type<half>(ushort((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000))));
    uint sv = rd16(B + 48 + 2 * (j / 2)) >> (6 * (j % 2));
    for (uint l = 0; l < 4; l++) {
        uint qh = B[32 + 2 * j + l / 2];
        uint idx = uint(B[4 * j + l]) | ((qh << (l % 2 ? 4 : 8)) & 0x700);
        float dl = d * float(2 * ((sv >> (l < 2 ? 0 : 3)) & 7) + 1);
        float delta = (qh & (l % 2 ? 0x80 : 0x08)) ? -0.125f : 0.125f;
        grid1(iq1s_grid[idx], dl, delta, out + 2 * l);
    }
}
// Q4_1 / Q5_0 / Q5_1: blocks of 32 with an f16 min (`MIN`) and/or a fifth bit per element (`HB`).
inline void load_q45(device const uchar* B, bool MIN, bool HB, thread float4* out) {
    float d = h2f(B), m = MIN ? h2f(B + 2) : 0.f;
    uint qh = HB ? rd32(B + (MIN ? 4 : 2)) : 0;
    device const uchar* q = B + 2 + (MIN ? 2 : 0) + (HB ? 4 : 0);
    float off = (HB && !MIN) ? 16.f : 0.f;
    for (uint i = 0; i < 4; i++) {
        uchar4 v = ld4(q + 4 * i);
        float4 lo = f4(v & uchar(15)), hi = f4(v >> uchar(4));
        if (HB) {
            uint e = 4 * i;
            lo += float4(float((qh >> e) & 1), float((qh >> (e + 1)) & 1), float((qh >> (e + 2)) & 1), float((qh >> (e + 3)) & 1)) * 16.f;
            hi += float4(float((qh >> (e + 16)) & 1), float((qh >> (e + 17)) & 1), float((qh >> (e + 18)) & 1), float((qh >> (e + 19)) & 1)) * 16.f;
        }
        out[i] = (lo - off) * d + m;
        out[i + 4] = (hi - off) * d + m;
    }
}
struct Q4_1W { GgW g; };
inline void load(Q4_1W w, uint row, uint b, thread float4* out) { load_q45(w.g.q + (ulong)row * w.g.rb + b * 20, true, false, out); }
struct Q5_0W { GgW g; };
inline void load(Q5_0W w, uint row, uint b, thread float4* out) { load_q45(w.g.q + (ulong)row * w.g.rb + b * 22, false, true, out); }
struct Q5_1W { GgW g; };
inline void load(Q5_1W w, uint row, uint b, thread float4* out) { load_q45(w.g.q + (ulong)row * w.g.rb + b * 24, true, true, out); }
struct F16W { GgW g; };
inline void load(F16W w, uint row, uint b, thread float4* out) {
    device const packed_half4* p = (device const packed_half4*)(w.g.q + (ulong)row * w.g.rb + b * 64);
    for (uint i = 0; i < 8; i++) out[i] = float4(half4(p[i]));
}
struct F32W2 { GgW g; };
inline void load(F32W2 w, uint row, uint b, thread float4* out) {
    device const packed_float4* p = (device const packed_float4*)(w.g.q + (ulong)row * w.g.rb + b * 128);
    for (uint i = 0; i < 8; i++) out[i] = float4(p[i]);
}
#define GG(T) GEMM(gemm_##T, T##W, MK_GG(T##W))
GEMM(gemm_F32, F32W2, MK_GG(F32W2)) GG(F16) GG(Q4_1) GG(Q5_0) GG(Q5_1)
GG(Q4_0) GG(IQ4_NL) GG(Q8_0) GG(Q4_K) GG(Q5_K) GG(Q6_K) GG(Q3_K) GG(Q2_K) GG(IQ4_XS)
GG(IQ2_XXS) GG(IQ2_XS) GG(IQ2_S) GG(IQ3_XXS) GG(IQ3_S) GG(IQ1_S) GG(IQ1_M)
"#;

/// GGUF types with a Metal GEMM (`gemm_<TYPE>` in `GGUF_SHADER_SRC`).
const GGUF_GPU_TYPES: [GgmlType; 21] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q4_0,
    GgmlType::IQ4_NL,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q3_K,
    GgmlType::Q2_K,
    GgmlType::IQ4_XS,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
];

/// The i-quant grids as MSL constant arrays.
fn gguf_shader_tables() -> String {
    use crate::ggml::tables::*;
    use std::fmt::Write;
    let mut s = String::new();
    let arr = |s: &mut String, ty: &str, name: &str, vals: &mut dyn Iterator<Item = String>| {
        let _ = write!(s, "constant {ty} {name}[] = {{");
        for v in vals {
            let _ = write!(s, "{v},");
        }
        s.push_str("};\n");
    };
    arr(
        &mut s,
        "float",
        "kvalues_iq4nl",
        &mut KVALUES_IQ4NL.iter().map(|v| format!("{v}.f")),
    );
    for (name, t) in [
        ("iq2xxs_grid", &IQ2XXS_GRID[..]),
        ("iq2xs_grid", &IQ2XS_GRID[..]),
        ("iq2s_grid", &IQ2S_GRID[..]),
        ("iq1s_grid", &IQ1S_GRID[..]),
    ] {
        arr(&mut s, "ulong", name, &mut t.iter().map(|v| format!("{v:#x}ul")));
    }
    for (name, t) in [("iq3xxs_grid", &IQ3XXS_GRID[..]), ("iq3s_grid", &IQ3S_GRID[..])] {
        arr(&mut s, "uint", name, &mut t.iter().map(|v| format!("{v:#x}u")));
    }
    s
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    n_out: u32,
    n_in: u32,
    w_off: u32,
    s_off: u32,
    h: u32,
    w: u32,
    cin: u32,
    k: u32,
    stride: u32,
    ow: u32,
    npix: u32,
    pad: u32,
    mode: u32,
    silu: u32,
    cpg: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n_out: u32,
    n_in: u32,
    w_off: u32,
    s_off: u32,
}

#[repr(C)]
struct MtlSize {
    w: usize,
    h: usize,
    d: usize,
}

/// A device buffer plus the byte offset of the wrapped data inside it.
struct Buf {
    id: Id,
    offset_bytes: usize,
}

struct MetalWeight {
    data: Buf,
    scales: Option<Buf>,
    /// Affine Q4: the offsets buffer, offsets per row, and blocks of 32 per offset group.
    offsets: Option<(Buf, u32, u32)>,
}

#[repr(C)]
struct OffsetParams {
    o_off: u32,
    gpr: u32,
    gk: u32,
    _p: u32,
}
unsafe impl Send for MetalWeight {}
unsafe impl Sync for MetalWeight {}

struct Staging {
    x: Id,
    x_cap: usize,
    y: Id,
    y_cap: usize,
    /// Normalization statistics (float2 per group or row) and gamma/beta for fused norms.
    stats: Grow,
    gb: Grow,
    /// Attention operands: per-head padded Q, K, transposed V, the scores of one head, outputs.
    aq: Grow,
    ak: Grow,
    av: Grow,
    asc: Grow,
    ao: Grow,
}

/// A device buffer that only ever grows.
struct Grow {
    id: Id,
    cap: usize,
}

impl Grow {
    const fn new() -> Self {
        Self {
            id: std::ptr::null_mut(),
            cap: 0,
        }
    }
}

#[derive(Default)]
struct Stats {
    calls: AtomicU64,
    copy_in_ns: AtomicU64,
    gpu_ns: AtomicU64,
    copy_out_ns: AtomicU64,
}

pub struct MetalBackend {
    stats: Stats,
    o: Objc,
    _foundation: dlsym::Library,
    device: Id,
    queue: Id,
    /// GEMM pipelines indexed by format: bf16, q8, q4.
    pipes: [Id; 3],
    /// GEMM over affine Q4 weights.
    pipe_q4a: Id,
    /// Implicit-GEMM convolution pipelines, same order.
    conv_pipes: [Id; 3],
    gemm_f32: Id,
    /// GEMM pipelines for the GGUF block formats.
    gguf_pipes: Vec<(GgmlType, Id)>,
    group_stats: Id,
    row_stats: Id,
    softmax: Id,
    staging: Mutex<Staging>,
    sels: Sels,
}
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

struct Sels {
    command_buffer: Sel,
    compute_encoder: Sel,
    set_pipeline: Sel,
    set_buffer: Sel,
    set_bytes: Sel,
    dispatch: Sel,
    end_encoding: Sel,
    commit: Sel,
    wait: Sel,
    contents: Sel,
    new_buffer_nocopy: Sel,
    new_buffer_bytes: Sel,
    new_buffer_len: Sel,
    status: Sel,
    release: Sel,
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
}

type Msg0 = unsafe extern "C" fn(Id, Sel) -> Id;
type Msg1 = unsafe extern "C" fn(Id, Sel, Id) -> Id;
type MsgU = unsafe extern "C" fn(Id, Sel) -> u64;
type MsgPtr = unsafe extern "C" fn(Id, Sel) -> *mut c_void;
type MsgNewLib = unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id;
type MsgNewPipe = unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id;
type MsgNewBufNoCopy = unsafe extern "C" fn(Id, Sel, *mut c_void, usize, usize, Id) -> Id;
type MsgNewBufBytes = unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize) -> Id;
type MsgNewBufLen = unsafe extern "C" fn(Id, Sel, usize, usize) -> Id;
type MsgSetBuffer = unsafe extern "C" fn(Id, Sel, Id, usize, usize);
type MsgSetBytes = unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize);
type MsgSetPipe = unsafe extern "C" fn(Id, Sel, Id);
type MsgDispatch = unsafe extern "C" fn(Id, Sel, *const MtlSize, *const MtlSize);
type MsgVoid = unsafe extern "C" fn(Id, Sel);

trait IntoInnerUnsafe<T> {
    fn into_inner_unsafe(self) -> T;
}
impl<T: dlsym::FnPtr> IntoInnerUnsafe<T> for dlsym::Symbol<'_, T> {
    fn into_inner_unsafe(self) -> T {
        unsafe { self.into_inner() }
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let o = Objc::open().map_err(|e| anyhow!(e))?;
        let foundation =
            unsafe { dlsym::Library::open("/System/Library/Frameworks/Foundation.framework/Foundation") }.map_err(|e| anyhow!("{e}"))?;
        let device = o.default_device();
        if device.is_null() {
            bail!("no Metal device");
        }
        let msg0: Msg0 = o.msg_send();
        let queue = unsafe { msg0(device, o.sel(c"newCommandQueue")) };
        if queue.is_null() {
            bail!("newCommandQueue failed");
        }
        let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id = unsafe { o.objc.get(c"objc_getClass") }
            .map_err(|e| anyhow!("{e}"))?
            .into_inner_unsafe();
        let nsstring_cls = unsafe { get_class(c"NSString".as_ptr()) };
        let msg_str: unsafe extern "C" fn(Id, Sel, *const std::ffi::c_char) -> Id = o.msg_send();
        let error_text = |err: Id| -> String {
            if err.is_null() {
                return String::new();
            }
            // SAFETY: `localizedDescription` returns an NSString owned by the NSError.
            unsafe { o.nsstring_to_string(msg0(err, o.sel(c"localizedDescription"))) }
        };
        // Compile the shader library from source.
        let src = CString::new(format!("{SHADER_SRC}{}{GGUF_SHADER_SRC}", gguf_shader_tables()))?;
        let ns_src = unsafe { msg_str(nsstring_cls, o.sel(c"stringWithUTF8String:"), src.as_ptr()) };
        let new_lib: MsgNewLib = o.msg_send();
        let mut err: Id = std::ptr::null_mut();
        let lib = unsafe {
            new_lib(
                device,
                o.sel(c"newLibraryWithSource:options:error:"),
                ns_src,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        if lib.is_null() {
            bail!("Metal shader compilation failed: {}", error_text(err));
        }
        let pipeline = |name: &CStr| -> Result<Id> {
            let ns_name = unsafe { msg_str(nsstring_cls, o.sel(c"stringWithUTF8String:"), name.as_ptr()) };
            let msg_fn: Msg1 = o.msg_send();
            let f = unsafe { msg_fn(lib, o.sel(c"newFunctionWithName:"), ns_name) };
            if f.is_null() {
                bail!("kernel {name:?} not found");
            }
            let new_pipe: MsgNewPipe = o.msg_send();
            let mut err: Id = std::ptr::null_mut();
            let p = unsafe { new_pipe(device, o.sel(c"newComputePipelineStateWithFunction:error:"), f, &mut err) };
            if p.is_null() {
                bail!("pipeline for {name:?} failed: {}", error_text(err));
            }
            Ok(p)
        };
        let pipes = [pipeline(c"gemm_bf")?, pipeline(c"gemm_q8")?, pipeline(c"gemm_q4")?];
        let pipe_q4a = pipeline(c"gemm_q4a")?;
        let conv_pipes = [pipeline(c"conv_bf")?, pipeline(c"conv_q8")?, pipeline(c"conv_q4")?];
        let gemm_f32 = pipeline(c"gemm_f32")?;
        let gguf_pipes = GGUF_GPU_TYPES
            .iter()
            .map(|&k| Ok((k, pipeline(&CString::new(format!("gemm_{}", k.name()))?)?)))
            .collect::<Result<Vec<_>>>()?;
        let group_stats = pipeline(c"group_stats")?;
        let row_stats = pipeline(c"row_stats")?;
        let softmax = pipeline(c"softmax_rows")?;
        let pool_push = unsafe { o.objc.get::<unsafe extern "C" fn() -> *mut c_void>(c"objc_autoreleasePoolPush") }
            .map_err(|e| anyhow!("{e}"))?
            .into_inner_unsafe();
        let pool_pop = unsafe { o.objc.get::<unsafe extern "C" fn(*mut c_void)>(c"objc_autoreleasePoolPop") }
            .map_err(|e| anyhow!("{e}"))?
            .into_inner_unsafe();
        let sels = Sels {
            command_buffer: o.sel(c"commandBuffer"),
            compute_encoder: o.sel(c"computeCommandEncoder"),
            set_pipeline: o.sel(c"setComputePipelineState:"),
            set_buffer: o.sel(c"setBuffer:offset:atIndex:"),
            set_bytes: o.sel(c"setBytes:length:atIndex:"),
            dispatch: o.sel(c"dispatchThreadgroups:threadsPerThreadgroup:"),
            end_encoding: o.sel(c"endEncoding"),
            commit: o.sel(c"commit"),
            wait: o.sel(c"waitUntilCompleted"),
            contents: o.sel(c"contents"),
            new_buffer_nocopy: o.sel(c"newBufferWithBytesNoCopy:length:options:deallocator:"),
            new_buffer_bytes: o.sel(c"newBufferWithBytes:length:options:"),
            new_buffer_len: o.sel(c"newBufferWithLength:options:"),
            status: o.sel(c"status"),
            release: o.sel(c"release"),
            pool_push,
            pool_pop,
        };
        Ok(Self {
            stats: Stats::default(),
            o,
            _foundation: foundation,
            device,
            queue,
            pipes,
            pipe_q4a,
            conv_pipes,
            gemm_f32,
            gguf_pipes,
            group_stats,
            row_stats,
            softmax,
            staging: Mutex::new(Staging {
                x: std::ptr::null_mut(),
                x_cap: 0,
                y: std::ptr::null_mut(),
                y_cap: 0,
                stats: Grow::new(),
                gb: Grow::new(),
                aq: Grow::new(),
                ak: Grow::new(),
                av: Grow::new(),
                asc: Grow::new(),
                ao: Grow::new(),
            }),
            sels,
        })
    }

    /// Wraps host memory as a shared buffer without copying (the enclosing pages are mapped; the
    /// caller guarantees they belong to one allocation); falls back to copying. The iOS
    /// Simulator's Metal driver aborts on wrapped host memory (it hands buffers to the host's GPU
    /// process as shared memory), so it always copies there.
    fn wrap(&self, ptr: *const u8, len: usize) -> Option<Buf> {
        let start = (ptr as usize) & !(PAGE - 1);
        let end = (ptr as usize + len).div_ceil(PAGE) * PAGE;
        let nocopy: MsgNewBufNoCopy = self.o.msg_send();
        let id = if cfg!(target_abi = "sim") {
            std::ptr::null_mut()
        } else {
            unsafe {
                nocopy(
                    self.device,
                    self.sels.new_buffer_nocopy,
                    start as *mut c_void,
                    end - start,
                    0,
                    std::ptr::null_mut(),
                )
            }
        };
        if !id.is_null() {
            return Some(Buf {
                id,
                offset_bytes: ptr as usize - start,
            });
        }
        let copy: MsgNewBufBytes = self.o.msg_send();
        let id = unsafe { copy(self.device, self.sels.new_buffer_bytes, ptr as *const c_void, len, 0) };
        if id.is_null() {
            return None;
        }
        Some(Buf { id, offset_bytes: 0 })
    }

    fn gguf_pipe(&self, kind: GgmlType) -> Option<Id> {
        self.gguf_pipes.iter().find(|(k, _)| *k == kind).map(|(_, p)| *p)
    }

    /// Makes `g` at least `bytes` long (replaced buffers are released).
    fn grow(&self, g: &mut Grow, bytes: usize) -> bool {
        if g.cap >= bytes && !g.id.is_null() {
            return true;
        }
        let new_len: MsgNewBufLen = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let cap = bytes.max(16).div_ceil(PAGE) * PAGE * 2;
        let id = unsafe { new_len(self.device, self.sels.new_buffer_len, cap, 0) };
        if id.is_null() {
            return false;
        }
        if !g.id.is_null() {
            unsafe { void(g.id, self.sels.release) };
        }
        g.id = id;
        g.cap = cap;
        true
    }

    fn ensure_staging(&self, st: &mut Staging, x_bytes: usize, y_bytes: usize) -> bool {
        let new_len: MsgNewBufLen = self.o.msg_send();
        if st.x_cap < x_bytes {
            let cap = x_bytes.div_ceil(PAGE) * PAGE * 2;
            st.x = unsafe { new_len(self.device, self.sels.new_buffer_len, cap, 0) };
            st.x_cap = if st.x.is_null() { 0 } else { cap };
        }
        if st.y_cap < y_bytes {
            let cap = y_bytes.div_ceil(PAGE) * PAGE * 2;
            st.y = unsafe { new_len(self.device, self.sels.new_buffer_len, cap, 0) };
            st.y_cap = if st.y.is_null() { 0 } else { cap };
        }
        !st.x.is_null() && !st.y.is_null()
    }
}

fn format_index(w: &WeightRef) -> (usize, usize, usize) {
    // (pipeline index, bytes per weight element, bytes per scale)
    match w {
        WeightRef::Bf16(_) => (0, 2, 1),
        WeightRef::Q8(_) => (1, 1, 4),
        WeightRef::Q4(_) => (2, 1, 2),
        WeightRef::Gguf(_) => (usize::MAX, 1, 1),
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn stats(&self) -> Option<String> {
        let n = self.stats.calls.load(Ordering::Relaxed);
        if n == 0 {
            return None;
        }
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
        Some(format!(
            "{n} submissions: copy-in {:.1} ms, gpu {:.1} ms, copy-out {:.1} ms",
            ms(&self.stats.copy_in_ns),
            ms(&self.stats.gpu_ns),
            ms(&self.stats.copy_out_ns)
        ))
    }

    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool {
        t >= MIN_T && n_out.is_multiple_of(TILE) && n_in.is_multiple_of(32) && 2.0 * (t * n_out * n_in) as f64 >= MIN_FLOPS
    }

    fn upload(&self, w: &WeightRef, _n_out: usize, _n_in: usize) -> Option<DeviceWeight> {
        let mw = match w {
            WeightRef::Gguf(m) => {
                self.gguf_pipe(m.kind)?;
                MetalWeight {
                    data: self.wrap(m.data.as_ptr(), m.data.len())?,
                    scales: None,
                    offsets: None,
                }
            }
            WeightRef::Bf16(w) => MetalWeight {
                data: self.wrap(w.as_ptr() as *const u8, w.len() * 2)?,
                scales: None,
                offsets: None,
            },
            WeightRef::Q8(m) => MetalWeight {
                data: self.wrap(m.q.as_ptr() as *const u8, m.q.len())?,
                scales: Some(self.wrap(m.scales.as_ptr() as *const u8, m.scales.len() * 4)?),
                offsets: None,
            },
            WeightRef::Q4(m) => MetalWeight {
                data: self.wrap(m.q.as_ptr(), m.q.len())?,
                scales: Some(self.wrap(m.scales.as_ptr() as *const u8, m.scales.len() * 2)?),
                offsets: match &m.offsets {
                    Some(o) => Some((
                        self.wrap(o.v.as_ptr() as *const u8, o.v.len() * 2)?,
                        (m.n_in / o.group) as u32,
                        (o.group / 32) as u32,
                    )),
                    None => None,
                },
            },
        };
        Some(Box::new(mw))
    }

    fn conv2d(&self, x: &[f32], g: &crate::gpu::ConvGeom, norm: Option<&crate::gpu::NormSpec>, job: &mut Job) -> bool {
        use crate::gpu::NormKind;
        if matches!(job.w, WeightRef::Gguf(_)) {
            return false;
        }
        let n_in = g.k * g.k * g.cin;
        let t = g.oh * g.ow;
        let npx = g.h * g.w;
        if !g.cin.is_multiple_of(32) || !job.n_out.is_multiple_of(TILE) || x.len() < npx * g.cin || job.y.len() < t * job.n_out {
            return false;
        }
        if let Some(n) = norm
            && (n.gamma.len() != g.cin
                || n.beta.len() != g.cin
                || matches!(n.kind, NormKind::Group(gr) if gr == 0 || !g.cin.is_multiple_of(gr)))
        {
            return false;
        }
        let Some(mw) = job.dev.downcast_ref::<MetalWeight>() else {
            return false;
        };
        if mw.offsets.is_some() {
            return false; // no affine convolution kernel
        }
        let t_pad = t.div_ceil(TILE) * TILE;
        let mut st = self.staging.lock().unwrap();
        if !self.ensure_staging(&mut st, npx * g.cin * 4, t_pad * job.n_out * 4) {
            return false;
        }
        let n_stats = match norm.map(|n| n.kind) {
            Some(NormKind::Group(gr)) => gr,
            Some(NormKind::Layer) => npx,
            None => 1,
        };
        let st = &mut *st;
        if !self.grow(&mut st.stats, n_stats * 8) || !self.grow(&mut st.gb, 2 * g.cin * 4) {
            return false;
        }
        let contents: MsgPtr = self.o.msg_send();
        let t0 = Instant::now();
        unsafe {
            let xp = contents(st.x, self.sels.contents) as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), xp, npx * g.cin);
            if let Some(n) = norm {
                let gp = contents(st.gb.id, self.sels.contents) as *mut f32;
                std::ptr::copy_nonoverlapping(n.gamma.as_ptr(), gp, g.cin);
                std::ptr::copy_nonoverlapping(n.beta.as_ptr(), gp.add(g.cin), g.cin);
            }
        }
        let msg0: Msg0 = self.o.msg_send();
        let set_pipe: MsgSetPipe = self.o.msg_send();
        let set_buf: MsgSetBuffer = self.o.msg_send();
        let set_bytes: MsgSetBytes = self.o.msg_send();
        let dispatch: MsgDispatch = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let status: MsgU = self.o.msg_send();
        unsafe {
            let t1 = Instant::now();
            let pool = (self.sels.pool_push)();
            let cb = msg0(self.queue, self.sels.command_buffer);
            let enc = msg0(cb, self.sels.compute_encoder);
            // Statistics first; dispatches in one encoder run in order.
            match norm {
                Some(
                    n @ crate::gpu::NormSpec {
                        kind: NormKind::Group(gr), ..
                    },
                ) => {
                    #[repr(C)]
                    struct Gs {
                        n: u32,
                        c: u32,
                        groups: u32,
                        eps: f32,
                    }
                    let p = Gs {
                        n: npx as u32,
                        c: g.cin as u32,
                        groups: *gr as u32,
                        eps: n.eps,
                    };
                    set_pipe(enc, self.sels.set_pipeline, self.group_stats);
                    set_buf(enc, self.sels.set_buffer, st.x, 0, 0);
                    set_buf(enc, self.sels.set_buffer, st.stats.id, 0, 1);
                    set_bytes(
                        enc,
                        self.sels.set_bytes,
                        &p as *const Gs as *const c_void,
                        std::mem::size_of::<Gs>(),
                        2,
                    );
                    dispatch(
                        enc,
                        self.sels.dispatch,
                        &MtlSize { w: *gr, h: 1, d: 1 },
                        &MtlSize { w: 256, h: 1, d: 1 },
                    );
                }
                Some(n) => {
                    #[repr(C)]
                    struct Rs {
                        c: u32,
                        eps: f32,
                    }
                    let p = Rs {
                        c: g.cin as u32,
                        eps: n.eps,
                    };
                    set_pipe(enc, self.sels.set_pipeline, self.row_stats);
                    set_buf(enc, self.sels.set_buffer, st.x, 0, 0);
                    set_buf(enc, self.sels.set_buffer, st.stats.id, 0, 1);
                    set_bytes(
                        enc,
                        self.sels.set_bytes,
                        &p as *const Rs as *const c_void,
                        std::mem::size_of::<Rs>(),
                        2,
                    );
                    dispatch(
                        enc,
                        self.sels.dispatch,
                        &MtlSize { w: npx, h: 1, d: 1 },
                        &MtlSize { w: 32, h: 1, d: 1 },
                    );
                }
                None => {}
            }
            let (fi, elem_bytes, scale_bytes) = format_index(&job.w);
            let params = ConvParams {
                n_out: job.n_out as u32,
                n_in: n_in as u32,
                w_off: (mw.data.offset_bytes / elem_bytes) as u32,
                s_off: mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0) as u32,
                h: g.h as u32,
                w: g.w as u32,
                cin: g.cin as u32,
                k: g.k as u32,
                stride: g.stride as u32,
                ow: g.ow as u32,
                npix: t as u32,
                pad: (g.k / 2) as u32,
                mode: match norm.map(|n| n.kind) {
                    None => 0,
                    Some(NormKind::Group(_)) => 1,
                    Some(NormKind::Layer) => 2,
                },
                silu: norm.is_some_and(|n| n.silu) as u32,
                cpg: match norm.map(|n| n.kind) {
                    Some(NormKind::Group(gr)) => (g.cin / gr) as u32,
                    _ => 1,
                },
                _p: 0,
            };
            set_pipe(enc, self.sels.set_pipeline, self.conv_pipes[fi]);
            set_buf(enc, self.sels.set_buffer, mw.data.id, 0, 0);
            set_buf(
                enc,
                self.sels.set_buffer,
                mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id),
                0,
                1,
            );
            set_buf(enc, self.sels.set_buffer, st.x, 0, 2);
            set_buf(enc, self.sels.set_buffer, st.y, 0, 3);
            set_bytes(
                enc,
                self.sels.set_bytes,
                &params as *const ConvParams as *const c_void,
                std::mem::size_of::<ConvParams>(),
                4,
            );
            set_buf(enc, self.sels.set_buffer, st.stats.id, 0, 5);
            set_buf(enc, self.sels.set_buffer, st.gb.id, 0, 6);
            let groups = MtlSize {
                w: job.n_out / TILE,
                h: t_pad / TILE,
                d: 1,
            };
            let threads = MtlSize { w: 128, h: 1, d: 1 };
            dispatch(enc, self.sels.dispatch, &groups, &threads);
            void(enc, self.sels.end_encoding);
            void(cb, self.sels.commit);
            void(cb, self.sels.wait);
            let ok = status(cb, self.sels.status) == 4;
            let t2 = Instant::now();
            if ok {
                let yp = contents(st.y, self.sels.contents) as *const f32;
                std::ptr::copy_nonoverlapping(yp, job.y.as_mut_ptr(), t * job.n_out);
            }
            (self.sels.pool_pop)(pool);
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ok
        }
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tq: usize,
        tk: usize,
        heads: usize,
        dh: usize,
        causal: bool,
        out: &mut [f32],
    ) -> bool {
        let d = heads * dh;
        if q.len() < tq * d || k.len() < tk * d || v.len() < tk * d || out.len() < tq * d || tq == 0 || tk == 0 {
            return false;
        }
        // GEMM tiling: out rows (keys, value dims) in multiples of 64, K in multiples of 32.
        let (dhp, dho, tkp, tqp) = (
            dh.next_multiple_of(32),
            dh.next_multiple_of(64),
            tk.next_multiple_of(64),
            tq.next_multiple_of(64),
        );
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        let ok = self.grow(&mut st.aq, heads * tqp * dhp * 4)
            && self.grow(&mut st.ak, heads * tkp * dhp * 4)
            && self.grow(&mut st.av, heads * dho * tkp * 4)
            && self.grow(&mut st.asc, tqp * tkp * 4)
            && self.grow(&mut st.ao, heads * tqp * dho * 4);
        if !ok {
            return false;
        }
        let contents: MsgPtr = self.o.msg_send();
        let t0 = Instant::now();
        unsafe {
            // Per-head operands, zero padded (padding contributes nothing to any product).
            let qp = contents(st.aq.id, self.sels.contents) as *mut f32;
            let kp = contents(st.ak.id, self.sels.contents) as *mut f32;
            let vp = contents(st.av.id, self.sels.contents) as *mut f32;
            std::ptr::write_bytes(qp, 0, heads * tqp * dhp);
            std::ptr::write_bytes(kp, 0, heads * tkp * dhp);
            std::ptr::write_bytes(vp, 0, heads * dho * tkp);
            for h in 0..heads {
                for i in 0..tq {
                    std::ptr::copy_nonoverlapping(q.as_ptr().add(i * d + h * dh), qp.add((h * tqp + i) * dhp), dh);
                }
                for j in 0..tk {
                    std::ptr::copy_nonoverlapping(k.as_ptr().add(j * d + h * dh), kp.add((h * tkp + j) * dhp), dh);
                    for c in 0..dh {
                        *vp.add((h * dho + c) * tkp + j) = v[j * d + h * dh + c];
                    }
                }
            }
        }
        let msg0: Msg0 = self.o.msg_send();
        let set_pipe: MsgSetPipe = self.o.msg_send();
        let set_buf: MsgSetBuffer = self.o.msg_send();
        let set_bytes: MsgSetBytes = self.o.msg_send();
        let dispatch: MsgDispatch = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let status: MsgU = self.o.msg_send();
        unsafe {
            let t1 = Instant::now();
            let pool = (self.sels.pool_push)();
            let cb = msg0(self.queue, self.sels.command_buffer);
            let enc = msg0(cb, self.sels.compute_encoder);
            #[repr(C)]
            struct Sm {
                cols: u32,
                ld: u32,
                scale: f32,
                causal: u32,
            }
            let sm = Sm {
                cols: tk as u32,
                ld: tkp as u32,
                scale: 1.0 / (dh as f32).sqrt(),
                causal: causal as u32,
            };
            for h in 0..heads {
                // Scores: keys as the "weights" ([tkp, dhp]), queries as the input.
                let p = Params {
                    n_out: tkp as u32,
                    n_in: dhp as u32,
                    w_off: 0,
                    s_off: 0,
                };
                set_pipe(enc, self.sels.set_pipeline, self.gemm_f32);
                set_buf(enc, self.sels.set_buffer, st.ak.id, h * tkp * dhp * 4, 0);
                set_buf(enc, self.sels.set_buffer, st.ak.id, 0, 1);
                set_buf(enc, self.sels.set_buffer, st.aq.id, h * tqp * dhp * 4, 2);
                set_buf(enc, self.sels.set_buffer, st.asc.id, 0, 3);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &p as *const Params as *const c_void,
                    std::mem::size_of::<Params>(),
                    4,
                );
                dispatch(
                    enc,
                    self.sels.dispatch,
                    &MtlSize {
                        w: tkp / TILE,
                        h: tqp / TILE,
                        d: 1,
                    },
                    &MtlSize { w: 128, h: 1, d: 1 },
                );
                set_pipe(enc, self.sels.set_pipeline, self.softmax);
                set_buf(enc, self.sels.set_buffer, st.asc.id, 0, 0);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &sm as *const Sm as *const c_void,
                    std::mem::size_of::<Sm>(),
                    1,
                );
                dispatch(
                    enc,
                    self.sels.dispatch,
                    &MtlSize { w: tqp, h: 1, d: 1 },
                    &MtlSize { w: 32, h: 1, d: 1 },
                );
                // Output: transposed values as the "weights" ([dho, tkp]), probabilities as the input.
                let p = Params {
                    n_out: dho as u32,
                    n_in: tkp as u32,
                    w_off: 0,
                    s_off: 0,
                };
                set_pipe(enc, self.sels.set_pipeline, self.gemm_f32);
                set_buf(enc, self.sels.set_buffer, st.av.id, h * dho * tkp * 4, 0);
                set_buf(enc, self.sels.set_buffer, st.av.id, 0, 1);
                set_buf(enc, self.sels.set_buffer, st.asc.id, 0, 2);
                set_buf(enc, self.sels.set_buffer, st.ao.id, h * tqp * dho * 4, 3);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &p as *const Params as *const c_void,
                    std::mem::size_of::<Params>(),
                    4,
                );
                dispatch(
                    enc,
                    self.sels.dispatch,
                    &MtlSize {
                        w: dho / TILE,
                        h: tqp / TILE,
                        d: 1,
                    },
                    &MtlSize { w: 128, h: 1, d: 1 },
                );
            }
            void(enc, self.sels.end_encoding);
            void(cb, self.sels.commit);
            void(cb, self.sels.wait);
            let ok = status(cb, self.sels.status) == 4;
            let t2 = Instant::now();
            if ok {
                let op = contents(st.ao.id, self.sels.contents) as *const f32;
                for h in 0..heads {
                    for i in 0..tq {
                        std::ptr::copy_nonoverlapping(op.add((h * tqp + i) * dho), out.as_mut_ptr().add(i * d + h * dh), dh);
                    }
                }
            }
            (self.sels.pool_pop)(pool);
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ok
        }
    }

    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool {
        if jobs.is_empty() || !n_in.is_multiple_of(32) || x.len() < t * n_in {
            return false;
        }
        let t_pad = t.div_ceil(TILE) * TILE;
        let mut y_total = 0;
        for j in jobs.iter() {
            if j.n_out % TILE != 0 || j.y.len() < t * j.n_out || j.dev.downcast_ref::<MetalWeight>().is_none() {
                return false;
            }
            y_total += t_pad * j.n_out;
        }
        let mut st = self.staging.lock().unwrap();
        if !self.ensure_staging(&mut st, t_pad * n_in * 4, y_total * 4) {
            return false;
        }
        let contents: MsgPtr = self.o.msg_send();
        let t0 = Instant::now();
        unsafe {
            let xp = contents(st.x, self.sels.contents) as *mut f32;
            std::ptr::copy_nonoverlapping(x.as_ptr(), xp, t * n_in);
            if t_pad > t {
                std::ptr::write_bytes(xp.add(t * n_in), 0, (t_pad - t) * n_in);
            }
        }
        let msg0: Msg0 = self.o.msg_send();
        let set_pipe: MsgSetPipe = self.o.msg_send();
        let set_buf: MsgSetBuffer = self.o.msg_send();
        let set_bytes: MsgSetBytes = self.o.msg_send();
        let dispatch: MsgDispatch = self.o.msg_send();
        let void: MsgVoid = self.o.msg_send();
        let status: MsgU = self.o.msg_send();
        unsafe {
            let t1 = Instant::now();
            let pool = (self.sels.pool_push)();
            let cb = msg0(self.queue, self.sels.command_buffer);
            let enc = msg0(cb, self.sels.compute_encoder);
            set_buf(enc, self.sels.set_buffer, st.x, 0, 2);
            let mut y_off = 0usize;
            for j in jobs.iter() {
                let mw = j.dev.downcast_ref::<MetalWeight>().unwrap();
                let (fi, elem_bytes, scale_bytes) = format_index(&j.w);
                // GGUF: byte offset of the data, and the row length in bytes in place of s_off.
                let (s_off, pipe) = match &j.w {
                    WeightRef::Gguf(m) => (m.row_bytes(), self.gguf_pipe(m.kind).unwrap()),
                    _ => (
                        mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0),
                        if mw.offsets.is_some() { self.pipe_q4a } else { self.pipes[fi] },
                    ),
                };
                let params = Params {
                    n_out: j.n_out as u32,
                    n_in: n_in as u32,
                    w_off: (mw.data.offset_bytes / elem_bytes) as u32,
                    s_off: s_off as u32,
                };
                set_pipe(enc, self.sels.set_pipeline, pipe);
                if let Some((ob, gpr, gk)) = &mw.offsets {
                    let op = OffsetParams {
                        o_off: (ob.offset_bytes / 2) as u32,
                        gpr: *gpr,
                        gk: *gk,
                        _p: 0,
                    };
                    set_buf(enc, self.sels.set_buffer, ob.id, 0, 5);
                    set_bytes(
                        enc,
                        self.sels.set_bytes,
                        &op as *const OffsetParams as *const c_void,
                        std::mem::size_of::<OffsetParams>(),
                        6,
                    );
                }
                set_buf(enc, self.sels.set_buffer, mw.data.id, 0, 0);
                set_buf(
                    enc,
                    self.sels.set_buffer,
                    mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id),
                    0,
                    1,
                );
                set_buf(enc, self.sels.set_buffer, st.y, y_off * 4, 3);
                set_bytes(
                    enc,
                    self.sels.set_bytes,
                    &params as *const Params as *const c_void,
                    std::mem::size_of::<Params>(),
                    4,
                );
                let groups = MtlSize {
                    w: j.n_out / TILE,
                    h: t_pad / TILE,
                    d: 1,
                };
                let threads = MtlSize { w: 128, h: 1, d: 1 };
                dispatch(enc, self.sels.dispatch, &groups, &threads);
                y_off += t_pad * j.n_out;
            }
            void(enc, self.sels.end_encoding);
            void(cb, self.sels.commit);
            void(cb, self.sels.wait);
            let ok = status(cb, self.sels.status) == 4; // MTLCommandBufferStatusCompleted
            let t2 = Instant::now();
            if ok {
                let yp = contents(st.y, self.sels.contents) as *const f32;
                let mut y_off = 0usize;
                for j in jobs.iter_mut() {
                    std::ptr::copy_nonoverlapping(yp.add(y_off), j.y.as_mut_ptr(), t * j.n_out);
                    y_off += t_pad * j.n_out;
                }
            }
            (self.sels.pool_pop)(pool);
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ok
        }
    }
}
