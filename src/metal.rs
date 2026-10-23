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

use crate::gpu::{Backend, DeviceWeight, Job, WeightRef};
use crate::platform::gpu::metal::{Id, Objc, Sel};
use crate::quant::PAGE;
use anyhow::{anyhow, bail, Result};
use std::ffi::{c_void, CStr, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
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
inline void load(Q8W w, uint row, uint b, thread float4* out) {
    float d = w.s[(ulong)row * w.bpr + b];
    device const char4* p = (device const char4*)(w.q + (ulong)row * w.n_in + b * 32);
    for (uint i = 0; i < 8; i++) out[i] = float4(p[i]) * d;
}
inline float4 bf4(packed_ushort4 u4) {
    ushort4 u = ushort4(u4);
    return float4(as_type<float>(uint(u.x) << 16), as_type<float>(uint(u.y) << 16), as_type<float>(uint(u.z) << 16), as_type<float>(uint(u.w) << 16));
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
#define MK_Q4 Q4W{ (device const uchar*)qb + p.w_off, (device const half*)sb + p.s_off, p.n_in / 32, p.n_in / 2 }
#define MK_Q8 Q8W{ (device const char*)qb + p.w_off, (device const float*)sb + p.s_off, p.n_in / 32, p.n_in }
#define MK_BF BfW{ (device const ushort*)qb + p.w_off, p.n_in }
GEMM(gemm_bf, BfW, MK_BF) GEMM(gemm_q8, Q8W, MK_Q8) GEMM(gemm_q4, Q4W, MK_Q4)
"#;

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
}
unsafe impl Send for MetalWeight {}
unsafe impl Sync for MetalWeight {}

struct Staging {
    x: Id,
    x_cap: usize,
    y: Id,
    y_cap: usize,
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
        let foundation = unsafe { dlsym::Library::open("/System/Library/Frameworks/Foundation.framework/Foundation") }.map_err(|e| anyhow!("{e}"))?;
        let device = o.default_device();
        if device.is_null() {
            bail!("no Metal device");
        }
        let msg0: Msg0 = o.msg_send();
        let queue = unsafe { msg0(device, o.sel(c"newCommandQueue")) };
        if queue.is_null() {
            bail!("newCommandQueue failed");
        }
        let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id = unsafe { o.objc.get(c"objc_getClass") }.map_err(|e| anyhow!("{e}"))?.into_inner_unsafe();
        let nsstring_cls = unsafe { get_class(c"NSString".as_ptr()) };
        let msg_str: unsafe extern "C" fn(Id, Sel, *const std::ffi::c_char) -> Id = o.msg_send();
        let error_text = |err: Id| -> String {
            if err.is_null() {
                return String::new();
            }
            o.nsstring_to_string(unsafe { msg0(err, o.sel(c"localizedDescription")) })
        };
        // Compile the shader library from source.
        let src = CString::new(SHADER_SRC)?;
        let ns_src = unsafe { msg_str(nsstring_cls, o.sel(c"stringWithUTF8String:"), src.as_ptr()) };
        let new_lib: MsgNewLib = o.msg_send();
        let mut err: Id = std::ptr::null_mut();
        let lib = unsafe { new_lib(device, o.sel(c"newLibraryWithSource:options:error:"), ns_src, std::ptr::null_mut(), &mut err) };
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
        let pool_push = unsafe { o.objc.get::<unsafe extern "C" fn() -> *mut c_void>(c"objc_autoreleasePoolPush") }.map_err(|e| anyhow!("{e}"))?.into_inner_unsafe();
        let pool_pop = unsafe { o.objc.get::<unsafe extern "C" fn(*mut c_void)>(c"objc_autoreleasePoolPop") }.map_err(|e| anyhow!("{e}"))?.into_inner_unsafe();
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
            staging: Mutex::new(Staging { x: std::ptr::null_mut(), x_cap: 0, y: std::ptr::null_mut(), y_cap: 0 }),
            sels,
        })
    }

    /// Wraps host memory as a shared buffer without copying (the enclosing pages are mapped; the
    /// caller guarantees they belong to one allocation); falls back to copying.
    fn wrap(&self, ptr: *const u8, len: usize) -> Option<Buf> {
        let start = (ptr as usize) & !(PAGE - 1);
        let end = (ptr as usize + len).div_ceil(PAGE) * PAGE;
        let nocopy: MsgNewBufNoCopy = self.o.msg_send();
        let id = unsafe { nocopy(self.device, self.sels.new_buffer_nocopy, start as *mut c_void, end - start, 0, std::ptr::null_mut()) };
        if !id.is_null() {
            return Some(Buf { id, offset_bytes: ptr as usize - start });
        }
        let copy: MsgNewBufBytes = self.o.msg_send();
        let id = unsafe { copy(self.device, self.sels.new_buffer_bytes, ptr as *const c_void, len, 0) };
        if id.is_null() {
            return None;
        }
        Some(Buf { id, offset_bytes: 0 })
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
        t >= MIN_T && n_out % TILE == 0 && n_in % 32 == 0 && 2.0 * (t * n_out * n_in) as f64 >= MIN_FLOPS
    }

    fn upload(&self, w: &WeightRef, _n_out: usize, _n_in: usize) -> Option<DeviceWeight> {
        let mw = match w {
            WeightRef::Bf16(w) => MetalWeight { data: self.wrap(w.as_ptr() as *const u8, w.len() * 2)?, scales: None },
            WeightRef::Q8(m) => MetalWeight {
                data: self.wrap(m.q.as_ptr() as *const u8, m.q.len())?,
                scales: Some(self.wrap(m.scales.as_ptr() as *const u8, m.scales.len() * 4)?),
            },
            WeightRef::Q4(m) => MetalWeight {
                data: self.wrap(m.q.as_ptr() as *const u8, m.q.len())?,
                scales: Some(self.wrap(m.scales.as_ptr() as *const u8, m.scales.len() * 2)?),
            },
        };
        Some(Box::new(mw))
    }

    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool {
        if jobs.is_empty() || n_in % 32 != 0 || x.len() < t * n_in {
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
                let params = Params {
                    n_out: j.n_out as u32,
                    n_in: n_in as u32,
                    w_off: (mw.data.offset_bytes / elem_bytes) as u32,
                    s_off: mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0) as u32,
                };
                set_pipe(enc, self.sels.set_pipeline, self.pipes[fi]);
                set_buf(enc, self.sels.set_buffer, mw.data.id, 0, 0);
                set_buf(enc, self.sels.set_buffer, mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id), 0, 1);
                set_buf(enc, self.sels.set_buffer, st.y, y_off * 4, 3);
                set_bytes(enc, self.sels.set_bytes, &params as *const Params as *const c_void, std::mem::size_of::<Params>(), 4);
                let groups = MtlSize { w: j.n_out / TILE, h: t_pad / TILE, d: 1 };
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
