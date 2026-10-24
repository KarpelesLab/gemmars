//! Minimal zero-copy safetensors reader (single file or sharded directory).

use anyhow::{bail, Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
use crate::pool;
use crate::quant::PageVec;
use crate::quant::{Q4Matrix, QBuf, QKind, QMatrix};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte range inside the file.
    pub start: usize,
    pub end: usize,
}

enum Storage {
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(Mmap),
    /// Heap copy in a page-aligned, page-multiple allocation (so accelerators can map it in place).
    Heap(PageVec<u64>, usize),
    /// Kept by the host (browser), read by range; tensors in it are only usable as row tables.
    Host(crate::fsx::HostFile),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Mmap(m) => &m[..],
            Storage::Heap(v, n) => unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, *n) },
            Storage::Host(_) => &[],
        }
    }
}

impl std::ops::Deref for Storage {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.bytes()
    }
}

pub struct SafeTensors {
    mmap: Storage,
    tensors: HashMap<String, TensorInfo>,
    metadata: HashMap<String, String>,
}

/// Reads a whole file into memory using parallel positional reads (sequential where the
/// platform has no positional read).
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) fn read_file_parallel(file: &File, len: usize) -> Result<PageVec<u64>> {
    let mut buf = PageVec::<u64>::zeroed(len.div_ceil(8));
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
    #[cfg(unix)]
    const CHUNK: usize = 32 << 20;
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        use std::sync::Mutex;
        let err: Mutex<Option<std::io::Error>> = Mutex::new(None);
        let p = SendBytes(bytes.as_mut_ptr());
        let n = len.div_ceil(CHUNK);
        pool::run(n, &|i| {
            let start = i * CHUNK;
            let end = (start + CHUNK).min(len);
            // SAFETY: chunks are disjoint.
            let chunk = unsafe { std::slice::from_raw_parts_mut(p.get().add(start), end - start) };
            if let Err(e) = file.read_exact_at(chunk, start as u64) {
                *err.lock().unwrap() = Some(e);
            }
        });
        if let Some(e) = err.into_inner().unwrap() {
            return Err(e).context("read weights");
        }
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file;
        f.seek(SeekFrom::Start(0))?;
        f.read_exact(bytes).context("read weights")?;
    }
    Ok(buf)
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct SendBytes(*mut u8);
#[cfg(unix)]
unsafe impl Send for SendBytes {}
#[cfg(unix)]
unsafe impl Sync for SendBytes {}
#[cfg(unix)]
impl SendBytes {
    fn get(self) -> *mut u8 {
        self.0
    }
}

impl SafeTensors {
    pub fn open(path: &Path, in_memory: bool) -> Result<Self> {
        if let Some(hf) = crate::fsx::host_file(path) {
            let mut n8 = [0u8; 8];
            crate::fsx::read_host(hf, 0, &mut n8)?;
            let n = u64::from_le_bytes(n8) as usize;
            if 8 + n > hf.len {
                bail!("{}: header length {} exceeds file", path.display(), n);
            }
            let mut head = vec![0u8; n];
            crate::fsx::read_host(hf, 8, &mut head)?;
            return Self::from_header(path, Storage::Host(hf), &head, 8 + n, hf.len);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let mmap = if in_memory {
            let (v, len) = crate::fsx::read_pages(path)?;
            Storage::Heap(v, len)
        } else {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            Storage::Mmap(unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?)
        };
        #[cfg(target_arch = "wasm32")]
        let mmap = {
            let _ = in_memory;
            let (v, len) = crate::fsx::read_pages(path)?;
            Storage::Heap(v, len)
        };
        if mmap.len() < 8 {
            bail!("{}: file too small", path.display());
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        if 8 + n > mmap.len() {
            bail!("{}: header length {} exceeds file", path.display(), n);
        }
        let len = mmap.len();
        let head = mmap[8..8 + n].to_vec();
        Self::from_header(path, mmap, &head, 8 + n, len)
    }

    fn from_header(path: &Path, mmap: Storage, head: &[u8], data_start: usize, file_len: usize) -> Result<Self> {
        let header: serde_json::Value = serde_json::from_slice(head).with_context(|| format!("{}: parse header", path.display()))?;
        let obj = header.as_object().context("header is not an object")?;
        let mut tensors = HashMap::with_capacity(obj.len());
        let mut metadata = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, v) in m {
                        if let Some(v) = v.as_str() {
                            metadata.insert(k.clone(), v.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = v["dtype"].as_str().context("dtype")?.to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let offs = v["data_offsets"].as_array().context("data_offsets")?;
            let start = data_start + offs[0].as_u64().unwrap() as usize;
            let end = data_start + offs[1].as_u64().unwrap() as usize;
            if end > file_len || start > end {
                bail!("tensor {name} out of bounds");
            }
            tensors.insert(name.clone(), TensorInfo { dtype, shape, start, end });
        }
        Ok(Self { mmap, tensors, metadata })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Raw bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        if let Storage::Host(_) = self.mmap {
            bail!("tensor {name} is kept by the host and can only be read by rows");
        }
        Ok(&self.mmap[t.start..t.end])
    }

    /// Returns a bf16 tensor as a slice of raw u16 bit patterns.
    pub fn bf16(&self, name: &str) -> Result<&[u16]> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        if let Storage::Host(_) = self.mmap {
            bail!("tensor {name} is kept by the host and can only be read by rows");
        }
        if t.dtype != "BF16" {
            bail!("tensor {name} has dtype {}, expected BF16", t.dtype);
        }
        let b = &self.mmap[t.start..t.end];
        if b.as_ptr() as usize % 2 != 0 {
            bail!("tensor {name} is not 2-byte aligned");
        }
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) })
    }

    pub fn len(&self) -> usize {
        match &self.mmap {
            Storage::Host(h) => h.len,
            m => m.len(),
        }
    }

    /// Touch every page so the weights are resident before the first forward pass.
    pub fn preload(&self) {
        #[cfg(unix)]
        if let Storage::Mmap(m) = &self.mmap {
            let _ = m.advise(memmap2::Advice::WillNeed);
        }
        const CHUNK: usize = 64 << 20;
        let bytes: &[u8] = &self.mmap;
        let n = bytes.len().div_ceil(CHUNK);
        let total = std::sync::atomic::AtomicU64::new(0);
        pool::run(n, &|i| {
            let c = &bytes[i * CHUNK..((i + 1) * CHUNK).min(bytes.len())];
            let s: u64 = c.iter().step_by(4096).map(|&b| b as u64).sum();
            total.fetch_add(s, std::sync::atomic::Ordering::Relaxed);
        });
        std::hint::black_box(total);
    }
}

/// A model directory: either `model.safetensors` or an index plus shards.
pub struct ModelFiles {
    shards: Vec<SafeTensors>,
    index: HashMap<String, usize>,
    /// Tensors the model loader read, and the format it quantized each to (None: used as stored).
    /// `write_quantized` replays this to produce a pre-quantized checkpoint.
    touched: Mutex<BTreeMap<String, Option<QKind>>>,
    /// How each tensor was used: (as a matmul weight, as a lookup table).
    uses: Mutex<HashMap<String, (bool, bool)>>,
}

/// Maximum shard size written by `write_quantized`.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
const SHARD: usize = 1 << 30;

/// One tensor of a checkpoint being written.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
struct Planned {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: usize,
    /// (source tensor, quantization, scales part); None for padding.
    src: Option<(String, Option<QKind>, bool)>,
}

/// A weight matrix stored pre-quantized in the checkpoint, borrowed in place.
pub enum Stored {
    Q8(QMatrix),
    Q4(Q4Matrix),
}

impl ModelFiles {
    pub fn open(dir: &Path, in_memory: bool) -> Result<Self> {
        let single = dir.join("model.safetensors");
        let idx = dir.join("model.safetensors.index.json");
        let mut files: Vec<PathBuf> = Vec::new();
        if crate::fsx::exists(&single) {
            files.push(single);
        } else if crate::fsx::exists(&idx) {
            let v: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&idx)?)?;
            let map = v["weight_map"].as_object().context("weight_map")?;
            let mut names: Vec<String> = map.values().map(|x| x.as_str().unwrap().to_string()).collect();
            names.sort();
            names.dedup();
            files.extend(names.into_iter().map(|n| dir.join(n)));
        } else {
            bail!("no model.safetensors or index found in {}", dir.display());
        }
        let mut shards = Vec::with_capacity(files.len());
        for p in &files {
            shards.push(SafeTensors::open(p, in_memory)?);
        }
        let mut index = HashMap::new();
        for (i, s) in shards.iter().enumerate() {
            for n in s.names() {
                index.insert(n.clone(), i);
            }
        }
        Ok(Self { shards, index, touched: Mutex::new(BTreeMap::new()), uses: Mutex::new(HashMap::new()) })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).and_then(|&i| self.shards[i].info(name))
    }

    pub fn bf16(&self, name: &str) -> Result<&[u16]> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.touch(name);
        self.shards[i].bf16(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.touch(name);
        self.raw(name)
    }

    fn raw(&self, name: &str) -> Result<&[u8]> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.shards[i].bytes(name)
    }

    fn touch(&self, name: &str) {
        self.touched.lock().unwrap().entry(name.to_string()).or_insert(None);
    }

    /// Records whether `name` was used as a matmul weight or as a lookup table (`write_quantized`
    /// puts lookup-only tensors in their own shard).
    pub fn note_use(&self, name: &str, lin: bool, table: bool) {
        let mut u = self.uses.lock().unwrap();
        let e = u.entry(name.to_string()).or_insert((false, false));
        e.0 |= lin;
        e.1 |= table;
    }

    /// Records that the loader quantized tensor `name` to `kind` (see `write_quantized`).
    pub fn mark_quant(&self, name: &str, kind: QKind) {
        self.touched.lock().unwrap().insert(name.to_string(), Some(kind));
    }

    /// For a tensor in a host-kept shard: the file and the tensor's absolute byte offset.
    pub fn host_tensor(&self, name: &str) -> Option<(crate::fsx::HostFile, u64)> {
        let i = *self.index.get(name)?;
        let sh = &self.shards[i];
        match &sh.mmap {
            Storage::Host(h) => sh.tensors.get(name).map(|t| (*h, t.start as u64)),
            _ => None,
        }
    }

    /// A header metadata value (from the first shard that has it).
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.shards.iter().find_map(|s| s.metadata.get(key).map(|v| v.as_str()))
    }

    /// The pre-quantized form of matrix `name` (`[n_out, n_in]`, or a table of `n_out` rows), if
    /// the checkpoint has one: tensors `{name}.q8` + `{name}.q8s` or `{name}.q4` + `{name}.q4s`.
    ///
    /// # Safety
    /// The returned matrix borrows this checkpoint's memory without a lifetime; the caller must
    /// keep `self` alive (and not move its storage) for as long as the matrix is used. The model
    /// loaders guarantee this by holding a `&'a ModelFiles` for the model's lifetime.
    pub unsafe fn quantized(&self, name: &str, n_out: usize, n_in: usize) -> Result<Option<Stored>> {
        let check = |t: &str, dtype: &str, shape: &[usize]| -> Result<&[u8]> {
            let info = self.info(t).with_context(|| format!("missing tensor {t}"))?;
            if info.dtype != dtype || info.shape != shape {
                bail!("tensor {t} is {} {:?}, expected {dtype} {shape:?}", info.dtype, info.shape);
            }
            let b = self.raw(t)?;
            let align = match dtype {
                "F32" => 4,
                "F16" => 2,
                _ => 1,
            };
            if b.as_ptr() as usize % align != 0 {
                bail!("tensor {t} is not {align}-byte aligned");
            }
            Ok(b)
        };
        if n_in % crate::quant::QBLOCK != 0 {
            return Ok(None);
        }
        let bpr = n_in / crate::quant::QBLOCK;
        let q8 = format!("{name}.q8");
        let q4 = format!("{name}.q4");
        if self.has(&q8) {
            let q = check(&q8, "I8", &[n_out, n_in])?;
            let sc = check(&format!("{name}.q8s"), "F32", &[n_out, bpr])?;
            self.touched.lock().unwrap().insert(name.to_string(), Some(QKind::Q8));
            let q = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const i8, q.len()) };
            let sc = unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const f32, sc.len() / 4) };
            return Ok(Some(Stored::Q8(QMatrix { q: unsafe { QBuf::borrowed(q) }, scales: unsafe { QBuf::borrowed(sc) }, n_out, n_in })));
        }
        if self.has(&q4) {
            let q = check(&q4, "U8", &[n_out, n_in / 2])?;
            let sc = check(&format!("{name}.q4s"), "F16", &[n_out, bpr])?;
            self.touched.lock().unwrap().insert(name.to_string(), Some(QKind::Q4));
            let sc = unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const u16, sc.len() / 2) };
            return Ok(Some(Stored::Q4(Q4Matrix { q: unsafe { QBuf::borrowed(q) }, scales: unsafe { QBuf::borrowed(sc) }, n_out, n_in })));
        }
        Ok(None)
    }

    /// Writes a checkpoint holding exactly the tensors the loader read: quantized ones as
    /// `{name}.q8|q4` + scales, the rest as stored. Tensors only ever read as lookup tables (never
    /// as a matmul weight) go to a second shard, `model-tables.safetensors`, which a browser host
    /// can keep outside wasm memory and read row by row. Large tensors start on 16 KiB boundaries
    /// so accelerators can map them in place. Returns the files written with their sizes.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[allow(clippy::type_complexity)]
    pub fn write_quantized(&self, dir: &Path, metadata: &[(&str, &str)]) -> Result<Vec<(PathBuf, u64)>> {
        use crate::quant::QBLOCK;
        let touched = self.touched.lock().unwrap().clone();
        let uses = self.uses.lock().unwrap().clone();
        let mut weights: Vec<Planned> = Vec::new();
        let mut tables: Vec<Planned> = Vec::new();
        for (name, kind) in &touched {
            let info = self.info(name).with_context(|| format!("missing tensor {name}"))?.clone();
            let (lin, table) = uses.get(name).copied().unwrap_or((false, false));
            let dst = if table && !lin { &mut tables } else { &mut weights };
            match kind {
                None => {
                    let dtype: &'static str = match info.dtype.as_str() {
                        "BF16" => "BF16",
                        "F32" => "F32",
                        "F16" => "F16",
                        "I8" => "I8",
                        "U8" => "U8",
                        other => bail!("tensor {name}: dtype {other} not supported by the writer"),
                    };
                    dst.push(Planned { name: name.clone(), dtype, shape: info.shape.clone(), bytes: info.end - info.start, src: Some((name.clone(), None, false)) });
                }
                Some(k) => {
                    if info.dtype != "BF16" {
                        bail!("tensor {name}: only BF16 tensors can be quantized (is {})", info.dtype);
                    }
                    let n_out = info.shape[0];
                    let n_in: usize = info.shape[1..].iter().product();
                    if n_in % QBLOCK != 0 {
                        bail!("tensor {name}: row length {n_in} is not a multiple of {QBLOCK}");
                    }
                    let bpr = n_in / QBLOCK;
                    let (qd, qshape, qbytes, sd, sbytes) = match k {
                        QKind::Q8 => ("I8", vec![n_out, n_in], n_out * n_in, "F32", n_out * bpr * 4),
                        QKind::Q4 => ("U8", vec![n_out, n_in / 2], n_out * n_in / 2, "F16", n_out * bpr * 2),
                    };
                    dst.push(Planned { name: format!("{name}.{}", k.suffix()), dtype: qd, shape: qshape, bytes: qbytes, src: Some((name.clone(), Some(*k), false)) });
                    dst.push(Planned { name: format!("{name}.{}s", k.suffix()), dtype: sd, shape: vec![n_out, bpr], bytes: sbytes, src: Some((name.clone(), Some(*k), true)) });
                }
            }
        }
        // Clear outputs of an earlier run so the loader does not pick a stale layout.
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().to_string();
            if n == "model.safetensors" || n == "model.safetensors.index.json" || ((n.starts_with("model-weights") || n.starts_with("model-tables")) && n.ends_with(".safetensors")) {
                std::fs::remove_file(e.path()).with_context(|| format!("remove {}", e.path().display()))?;
            }
        }
        // Shards of at most SHARD bytes: one allocation per file must stay below 2 GiB in wasm32
        // (isize::MAX), and smaller files download and resume better.
        let split = |plan: Vec<Planned>| -> Vec<Vec<Planned>> {
            let mut out: Vec<Vec<Planned>> = vec![Vec::new()];
            let mut size = 0usize;
            for o in plan {
                let est = o.bytes.next_multiple_of(16384);
                if size > 0 && size + est > SHARD {
                    out.push(Vec::new());
                    size = 0;
                }
                size += est;
                out.last_mut().unwrap().push(o);
            }
            out.retain(|s| !s.is_empty());
            out
        };
        let mut files: Vec<(String, Vec<Planned>)> = Vec::new();
        let weight_shards = split(weights);
        let table_shards = split(tables);
        if table_shards.is_empty() && weight_shards.len() == 1 {
            files.push(("model.safetensors".into(), weight_shards.into_iter().next().unwrap()));
        } else {
            let (nw, nt) = (weight_shards.len(), table_shards.len());
            for (i, p) in weight_shards.into_iter().enumerate() {
                files.push((format!("model-weights-{:05}-of-{:05}.safetensors", i + 1, nw), p));
            }
            for (i, p) in table_shards.into_iter().enumerate() {
                files.push((format!("model-tables-{:05}-of-{:05}.safetensors", i + 1, nt), p));
            }
        }
        let mut written = Vec::new();
        let mut map = serde_json::Map::new();
        let mut total = 0u64;
        let n_files = files.len();
        for (file, plan) in files {
            let p = dir.join(&file);
            let (size, names) = self.write_file(&p, plan, metadata)?;
            for n in names {
                map.insert(n, serde_json::Value::String(file.clone()));
            }
            total += size;
            written.push((p, size));
        }
        if n_files > 1 {
            let index = serde_json::json!({ "metadata": { "total_size": total }, "weight_map": map });
            std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec_pretty(&index)?)?;
        }
        Ok(written)
    }

    /// Writes one safetensors file from a plan; returns its size and the tensor names (no pads).
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn write_file(&self, out: &Path, plan: Vec<Planned>, metadata: &[(&str, &str)]) -> Result<(u64, Vec<String>)> {
        use crate::quant::{quantize_bf16, quantize_q4};
        use std::io::Write;
        const BIG: usize = 1 << 20;
        const PAGE_ALIGN: usize = 16384;
        // Lay out with alignment padding (explicit U8 pad tensors keep the buffer hole-free).
        let mut layout: Vec<Planned> = Vec::new();
        let mut off = 0usize;
        for (i, o) in plan.into_iter().enumerate() {
            let align = if o.bytes >= BIG { PAGE_ALIGN } else { 64 };
            let pad = off.next_multiple_of(align) - off;
            if pad > 0 {
                layout.push(Planned { name: format!("__pad.{i}"), dtype: "U8", shape: vec![pad], bytes: pad, src: None });
                off += pad;
            }
            off += o.bytes;
            layout.push(o);
        }
        let mut header = serde_json::Map::new();
        let mut meta = serde_json::Map::new();
        for (k, v) in metadata {
            meta.insert(k.to_string(), serde_json::Value::String(v.to_string()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(meta));
        let mut off = 0usize;
        for o in &layout {
            header.insert(o.name.clone(), serde_json::json!({ "dtype": o.dtype, "shape": o.shape, "data_offsets": [off, off + o.bytes] }));
            off += o.bytes;
        }
        let mut hjson = serde_json::to_vec(&header)?;
        // Pad the header with spaces (allowed by the format) so the data starts page-aligned.
        let target = (8 + hjson.len()).next_multiple_of(PAGE_ALIGN) - 8;
        hjson.resize(target, b' ');

        let tmp = out.with_extension("safetensors.partial");
        let mut f = std::io::BufWriter::with_capacity(8 << 20, File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?);
        f.write_all(&(hjson.len() as u64).to_le_bytes())?;
        f.write_all(&hjson)?;
        let mut cache: Option<(String, Vec<u8>, Vec<u8>)> = None; // last quantized tensor: (name, q, scales)
        for o in &layout {
            match &o.src {
                None => f.write_all(&vec![0u8; o.bytes])?,
                Some((name, None, _)) => f.write_all(self.raw(name)?)?,
                Some((name, Some(k), scales)) => {
                    if cache.as_ref().map(|c| &c.0) != Some(name) {
                        let info = self.info(name).unwrap();
                        let n_out = info.shape[0];
                        let n_in: usize = info.shape[1..].iter().product();
                        let b = self.raw(name)?;
                        let w = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) };
                        let (q, s) = match k {
                            QKind::Q8 => {
                                let m = quantize_bf16(w, n_out, n_in);
                                (as_bytes(&m.q).to_vec(), as_bytes(&m.scales).to_vec())
                            }
                            QKind::Q4 => {
                                let m = quantize_q4(w, n_out, n_in);
                                (as_bytes(&m.q).to_vec(), as_bytes(&m.scales).to_vec())
                            }
                        };
                        cache = Some((name.clone(), q, s));
                    }
                    let c = cache.as_ref().unwrap();
                    let data = if *scales { &c.2 } else { &c.1 };
                    debug_assert_eq!(data.len(), o.bytes);
                    f.write_all(data)?;
                }
            }
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&tmp, out)?;
        let names = layout.into_iter().filter(|o| o.src.is_some()).map(|o| o.name).collect();
        Ok((std::fs::metadata(out)?.len(), names))
    }

    pub fn total_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    pub fn preload(&self) {
        for s in &self.shards {
            s.preload();
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::quant::{quantize_bf16, quantize_q4};

    /// Writes a minimal bf16 safetensors file with the given 2-D tensors.
    fn write_bf16(path: &Path, tensors: &[(&str, usize, usize, Vec<u16>)]) {
        let mut header = serde_json::Map::new();
        let mut off = 0;
        for (n, r, c, _) in tensors {
            header.insert(n.to_string(), serde_json::json!({"dtype": "BF16", "shape": [r, c], "data_offsets": [off, off + r * c * 2]}));
            off += r * c * 2;
        }
        let mut h = serde_json::to_vec(&header).unwrap();
        h.resize(h.len().next_multiple_of(8), b' '); // as the format requires
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        for (.., d) in tensors {
            out.extend(d.iter().flat_map(|v| v.to_le_bytes()));
        }
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn pre_quantized_round_trip() {
        let dir = std::env::temp_dir().join(format!("oxinfer-qtest-{}", std::process::id()));
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let (r, c) = (48, 96);
        let mk = |seed: u32| -> Vec<u16> { (0..r * c).map(|i| ((((i as u32).wrapping_mul(2654435761) ^ seed) >> 20) as f32 / 4096.0 - 0.5).to_bits()).map(|b| (b >> 16) as u16).collect() };
        let (a, b, t, n) = (mk(1), mk(2), mk(3), mk(4));
        write_bf16(&dir.join("model.safetensors"), &[("a", r, c, a.clone()), ("b", r, c, b.clone()), ("tab", r, c, t.clone()), ("norm", 1, c, n[..c].to_vec())]);
        let files = ModelFiles::open(&dir, true).unwrap();
        // What a loader does: two matmul weights (q8, q4), a lookup-only table (q4), a raw tensor.
        files.bf16("a").unwrap();
        files.mark_quant("a", QKind::Q8);
        files.note_use("a", true, false);
        files.bf16("b").unwrap();
        files.mark_quant("b", QKind::Q4);
        files.note_use("b", true, false);
        files.bf16("tab").unwrap();
        files.mark_quant("tab", QKind::Q4);
        files.note_use("tab", false, true);
        files.bf16("norm").unwrap();
        let written = files.write_quantized(&out, &[("oxinfer_quant", "q4")]).unwrap();
        assert_eq!(written.len(), 2, "weights and tables shards: {written:?}");

        let back = ModelFiles::open(&out, false).unwrap();
        assert_eq!(back.metadata("oxinfer_quant"), Some("q4"));
        assert_eq!(back.bf16("norm").unwrap(), &n[..c]);
        let (ea, eb, et) = (quantize_bf16(&a, r, c), quantize_q4(&b, r, c), quantize_q4(&t, r, c));
        match unsafe { back.quantized("a", r, c) }.unwrap() {
            Some(Stored::Q8(m)) => assert!(m.q[..] == ea.q[..] && m.scales[..] == ea.scales[..] && m.q.is_borrowed()),
            _ => panic!("a should be stored as q8"),
        }
        match unsafe { back.quantized("b", r, c) }.unwrap() {
            Some(Stored::Q4(m)) => assert!(m.q[..] == eb.q[..] && m.scales[..] == eb.scales[..]),
            _ => panic!("b should be stored as q4"),
        }
        match unsafe { back.quantized("tab", r, c) }.unwrap() {
            Some(Stored::Q4(m)) => assert!(m.q[..] == et.q[..] && m.scales[..] == et.scales[..]),
            _ => panic!("tab should be stored as q4"),
        }
        // Large tensors are page-aligned (none here); every tensor is at least 64-byte aligned.
        for name in ["a.q8", "a.q8s", "b.q4", "b.q4s", "tab.q4"] {
            assert_eq!(back.raw(name).unwrap().as_ptr() as usize % 64, 0, "{name} alignment");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
