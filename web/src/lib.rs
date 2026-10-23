//! gemmars in the browser (and Deno): a plain C-ABI WebAssembly module, no bindgen. The host
//! registers the model files in the module's virtual filesystem, then drives a chat session token
//! by token. GPU work goes the other way: the module imports `gpu_*` functions that the host
//! implements with WebGPU, and the host wraps them with JSPI (`WebAssembly.Suspending`) so the
//! synchronous `Backend::matmul` call can wait for the asynchronous buffer readback.
//!
//! ABI: strings are (ptr, len) UTF-8 in module memory; buffers for the host are allocated with
//! `web_alloc` and must be freed with `web_free`. Functions returning `i32` use `< 0` for errors
//! (the message is available through `web_last_error`).

mod webgpu;

use anyhow::{anyhow, bail, Result};
use gemmars::chat::{ChatFormat, GemmaChat, QwenChat};
use gemmars::model::{Model, Quant, Session};
use gemmars::qwen::{QwenModel, QwenSession};
use gemmars::sampler::Sampler;
use gemmars::{config, fsx, quant::PageVec, safetensors};
use std::cell::RefCell;
use std::path::Path;
use tokenizers::Tokenizer;

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn host_log(ptr: *const u8, len: usize);
    fn host_random(ptr: *mut u8, len: usize);
}

/// `getrandom` custom backend (selected with `--cfg getrandom_backend="custom"` in
/// `.cargo/config.toml`): the tokenizer's hash maps seed from it.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(dest: *mut u8, len: usize) -> Result<(), getrandom::Error> {
    unsafe { host_random(dest, len) };
    Ok(())
}

pub(crate) fn log(s: &str) {
    unsafe { host_log(s.as_ptr(), s.len()) }
}

enum Lm {
    Gemma(Model<'static>, Session),
    Qwen(QwenModel<'static>, QwenSession),
}

impl Lm {
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        match self {
            Lm::Gemma(m, s) => m.forward(s, tokens),
            Lm::Qwen(m, s) => m.forward(s, tokens),
        }
    }
    fn reset(&mut self) {
        match self {
            Lm::Gemma(m, s) => *s = m.new_session(),
            Lm::Qwen(m, s) => *s = m.new_session(),
        }
    }
    fn fmt(&self) -> &'static dyn ChatFormat {
        match self {
            Lm::Gemma(..) => &GemmaChat,
            Lm::Qwen(..) => &QwenChat,
        }
    }
}

struct Engine {
    lm: Lm,
    tok: Tokenizer,
    eos: Vec<u32>,
    sampler: Sampler,
    prefill_chunk: usize,
    /// Logits for the next token (after a prefill or a step), empty when the turn has ended.
    logits: Vec<f32>,
    /// Tokens generated in the current turn.
    out: Vec<u32>,
    turns: usize,
    max_tokens: usize,
    think: bool,
    /// Scratch for strings handed to the host (kept alive until the next call).
    text: String,
}

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(e: &anyhow::Error) -> i32 {
    let msg = format!("{e:#}");
    log(&format!("[error] {msg}"));
    LAST_ERROR.with(|l| *l.borrow_mut() = msg);
    -1
}

unsafe fn str_arg<'a>(ptr: *const u8, len: usize) -> &'a str {
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }).unwrap_or("")
}

/// Panics are reported to the host console (the module itself aborts).
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| log(&format!("[panic] {info}"))));
}

// ---- memory -------------------------------------------------------------------------------------

/// Allocates `len` bytes (page-aligned, so the buffer can later be registered as a file).
#[unsafe(no_mangle)]
pub extern "C" fn web_alloc(len: usize) -> *mut u8 {
    let v = PageVec::<u64>::zeroed(len.div_ceil(8).max(1));
    let p = v.as_ptr() as *mut u8;
    std::mem::forget(v);
    p
}

/// Frees a buffer from `web_alloc` that was not handed to `web_put_file`.
#[unsafe(no_mangle)]
pub extern "C" fn web_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { PageVec::<u64>::from_raw(ptr as *mut u64, len.div_ceil(8).max(1)) });
    }
}

/// Registers the buffer (from `web_alloc`, `len` valid bytes) as file `path` in the virtual
/// filesystem; the module takes ownership.
#[unsafe(no_mangle)]
pub extern "C" fn web_put_file(path: *const u8, path_len: usize, ptr: *mut u8, len: usize) {
    let path = unsafe { str_arg(path, path_len) }.to_string();
    let v = unsafe { PageVec::<u64>::from_raw(ptr as *mut u64, len.div_ceil(8).max(1)) };
    fsx::put(&path, v, len);
}

/// Registers file `path` as kept by the host (`len` bytes, read through `host_read(id, ...)`):
/// used for lookup-table shards (`*-tables.safetensors`) so they do not count against the 4 GiB
/// of module memory.
#[unsafe(no_mangle)]
pub extern "C" fn web_put_host_file(path: *const u8, path_len: usize, id: u32, len: f64) -> i32 {
    let path = unsafe { str_arg(path, path_len) }.to_string();
    if len > usize::MAX as f64 {
        return set_error(&anyhow!("{path}: {len} bytes is too large for a host file in wasm32"));
    }
    fsx::put_host(&path, fsx::HostFile { id, len: len as usize });
    0
}

// ---- model --------------------------------------------------------------------------------------

/// Loads the model in directory `dir` of the virtual filesystem. `quant`: 0 bf16, 1 q8, 2 q4.
/// `gpu`: non-zero to use the host's WebGPU implementation for prefill matmuls.
#[unsafe(no_mangle)]
pub extern "C" fn web_load(dir: *const u8, dir_len: usize, quant: u32, gpu: u32, prefill_chunk: u32) -> i32 {
    install_panic_hook();
    let dir = unsafe { str_arg(dir, dir_len) }.to_string();
    match load(Path::new(&dir), quant, gpu != 0, prefill_chunk as usize) {
        Ok(e) => {
            ENGINE.with(|g| *g.borrow_mut() = Some(e));
            0
        }
        Err(e) => set_error(&e),
    }
}

fn load(dir: &Path, quant: u32, gpu: bool, prefill_chunk: usize) -> Result<Engine> {
    let quant = match quant {
        0 => Quant::Bf16,
        1 => Quant::Q8,
        2 => Quant::Q4,
        q => bail!("unknown quant {q}"),
    };
    if gpu {
        webgpu::install();
    }
    let t0 = gemmars::clock::Instant::now();
    let model_type = config::model_type(dir)?;
    let eos = config::load_eos_ids(dir);
    let tok = Tokenizer::from_bytes(fsx::read(&dir.join("tokenizer.json"))?).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let files: &'static safetensors::ModelFiles = Box::leak(Box::new(safetensors::ModelFiles::open(dir, true)?));
    let lm = if model_type.starts_with("qwen3") {
        let cfg = config::QwenConfig::load(dir)?;
        let m = QwenModel::load(cfg, files, quant)?;
        let s = m.new_session();
        Lm::Qwen(m, s)
    } else {
        let cfg = config::TextConfig::load(dir)?;
        let m = Model::load(cfg, files, quant)?;
        let s = m.new_session();
        Lm::Gemma(m, s)
    };
    let (layers, hidden, streamed, stored) = match &lm {
        Lm::Gemma(m, _) => (m.cfg.num_hidden_layers, m.cfg.hidden_size, m.streamed_bytes(), m.quant),
        Lm::Qwen(m, _) => (m.cfg.num_hidden_layers, m.cfg.hidden_size, m.streamed_bytes(), m.quant),
    };
    let _ = quant;
    log(&format!(
        "[loaded {model_type}: {layers} layers, hidden {hidden}, {:.2} GB checkpoint, {stored:?} weights{} streaming {:.2} GB/token, device {}, {:.2}s]",
        files.total_bytes() as f64 / 1e9,
        if Quant::stored(files).is_some() { " (pre-quantized)" } else { "" },
        streamed as f64 / 1e9,
        gemmars::gpu::active().map(|b| b.name().to_string()).unwrap_or_else(|| "cpu".into()),
        t0.elapsed().as_secs_f64()
    ));
    Ok(Engine {
        lm,
        tok,
        eos,
        sampler: Sampler::new(0.0, 0, 1.0, 0),
        // Each GPU batch costs a host round trip, so larger chunks amortize better there.
        prefill_chunk: if prefill_chunk != 0 { prefill_chunk } else if gemmars::gpu::active().is_some() { 512 } else { 256 },
        logits: Vec::new(),
        out: Vec::new(),
        turns: 0,
        max_tokens: 0,
        think: false,
        text: String::new(),
    })
}

/// Sampling settings for the following turns.
#[unsafe(no_mangle)]
pub extern "C" fn web_set_sampling(temperature: f32, top_k: u32, top_p: f32, seed: u64) {
    ENGINE.with(|g| {
        if let Some(e) = g.borrow_mut().as_mut() {
            e.sampler = Sampler::new(temperature, top_k as usize, top_p, seed);
        }
    });
}

/// Starts a new conversation (drops the KV cache).
#[unsafe(no_mangle)]
pub extern "C" fn web_reset() {
    ENGINE.with(|g| {
        if let Some(e) = g.borrow_mut().as_mut() {
            e.lm.reset();
            e.turns = 0;
            e.logits.clear();
            e.out.clear();
        }
    });
}

/// Adds a user turn (and, on the first turn, the optional system prompt), runs the prefill and
/// returns the number of prompt tokens processed. Then call `web_next` until it returns -1.
#[unsafe(no_mangle)]
pub extern "C" fn web_chat_turn(user: *const u8, user_len: usize, system: *const u8, system_len: usize, think: u32, max_tokens: u32) -> i32 {
    let user = unsafe { str_arg(user, user_len) }.to_string();
    let system = if system.is_null() || system_len == 0 { None } else { Some(unsafe { str_arg(system, system_len) }.to_string()) };
    ENGINE.with(|g| {
        let mut g = g.borrow_mut();
        let Some(e) = g.as_mut() else {
            return set_error(&anyhow!("no model loaded"));
        };
        match e.chat_turn(&user, system.as_deref(), think != 0, max_tokens as usize) {
            Ok(n) => n as i32,
            Err(err) => set_error(&err),
        }
    })
}

impl Engine {
    fn chat_turn(&mut self, user: &str, system: Option<&str>, think: bool, max_tokens: usize) -> Result<usize> {
        let fmt = self.lm.fmt();
        let mut text = String::new();
        if self.turns == 0 {
            text.push_str(&fmt.conversation_start(system, think));
        } else if !self.out.is_empty() {
            // The previous reply ended (with or without its stop token); close the model turn.
            text.push_str(&fmt.end_of_model_turn());
        }
        text.push_str(&fmt.user_turn(user, think));
        let ids = self.tok.encode(text.as_str(), false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec();
        self.out.clear();
        self.logits.clear();
        for c in ids.chunks(self.prefill_chunk) {
            self.logits = self.lm.forward(c);
        }
        self.turns += 1;
        self.max_tokens = max_tokens;
        self.think = think;
        Ok(ids.len())
    }

    /// Samples the next token and runs it through the model; None when the turn is over.
    fn next(&mut self) -> Option<u32> {
        if self.logits.is_empty() || self.out.len() >= self.max_tokens {
            self.logits.clear();
            return None;
        }
        let id = self.sampler.sample(&self.logits);
        if self.eos.contains(&id) {
            // Keep it in `out` so the next turn knows the model turn is closed, but do not emit it.
            self.out.push(id);
            self.logits.clear();
            return None;
        }
        self.out.push(id);
        self.logits = self.lm.forward(&[id]);
        Some(id)
    }

    fn decoded(&self) -> String {
        let shown: Vec<u32> = self.out.iter().copied().filter(|t| !self.eos.contains(t)).collect();
        self.tok.decode(&shown, true).unwrap_or_default()
    }
}

/// Generates one token: returns its id, or -1 when the reply is complete (stop token or limit).
#[unsafe(no_mangle)]
pub extern "C" fn web_next() -> i32 {
    ENGINE.with(|g| g.borrow_mut().as_mut().and_then(|e| e.next()).map(|t| t as i32).unwrap_or(-1))
}

/// The reply so far, decoded (valid until the next call into the module); returns the pointer,
/// the length goes to `*len`.
#[unsafe(no_mangle)]
pub extern "C" fn web_text(len: *mut usize) -> *const u8 {
    ENGINE.with(|g| {
        let mut g = g.borrow_mut();
        match g.as_mut() {
            Some(e) => {
                e.text = e.decoded();
                unsafe { *len = e.text.len() };
                e.text.as_ptr()
            }
            None => {
                unsafe { *len = 0 };
                std::ptr::null()
            }
        }
    })
}

/// Last error message (pointer; length to `*len`).
#[unsafe(no_mangle)]
pub extern "C" fn web_last_error(len: *mut usize) -> *const u8 {
    LAST_ERROR.with(|l| {
        let l = l.borrow();
        unsafe { *len = l.len() };
        l.as_ptr()
    })
}

/// Number of tokens generated in the current turn.
#[unsafe(no_mangle)]
pub extern "C" fn web_generated() -> u32 {
    ENGINE.with(|g| g.borrow().as_ref().map(|e| e.out.iter().filter(|t| !e.eos.contains(t)).count() as u32).unwrap_or(0))
}

/// GPU statistics line (pointer; length to `*len`), empty without a GPU backend.
#[unsafe(no_mangle)]
pub extern "C" fn web_gpu_stats(len: *mut usize) -> *const u8 {
    ENGINE.with(|g| {
        let mut g = g.borrow_mut();
        match g.as_mut() {
            Some(e) => {
                e.text = gemmars::gpu::active().and_then(|b| b.stats()).unwrap_or_default();
                unsafe { *len = e.text.len() };
                e.text.as_ptr()
            }
            None => {
                unsafe { *len = 0 };
                std::ptr::null()
            }
        }
    })
}
