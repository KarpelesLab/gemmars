//! Minimal zero-copy safetensors reader (single file or sharded directory).

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
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
    Mmap(Mmap),
    /// Heap copy, kept as u64 for 8-byte alignment.
    Heap(Vec<u64>, usize),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            Storage::Mmap(m) => &m[..],
            Storage::Heap(v, n) => unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, *n) },
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
}

/// Reads a whole file into memory using parallel positional reads.
fn read_file_parallel(file: &File, len: usize) -> Result<Vec<u64>> {
    use std::os::unix::fs::FileExt;
    let mut buf: Vec<u64> = vec![0u64; len.div_ceil(8)];
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
    const CHUNK: usize = 32 << 20;
    bytes
        .par_chunks_mut(CHUNK)
        .enumerate()
        .try_for_each(|(i, chunk)| file.read_exact_at(chunk, (i * CHUNK) as u64))
        .context("read weights")?;
    Ok(buf)
}

impl SafeTensors {
    pub fn open(path: &Path, in_memory: bool) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = if in_memory {
            let len = file.metadata()?.len() as usize;
            Storage::Heap(read_file_parallel(&file, len)?, len)
        } else {
            Storage::Mmap(unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?)
        };
        if mmap.len() < 8 {
            bail!("{}: file too small", path.display());
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        if 8 + n > mmap.len() {
            bail!("{}: header length {} exceeds file", path.display(), n);
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + n])
            .with_context(|| format!("{}: parse header", path.display()))?;
        let data_start = 8 + n;
        let obj = header.as_object().context("header is not an object")?;
        let mut tensors = HashMap::with_capacity(obj.len());
        for (name, v) in obj {
            if name == "__metadata__" {
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
            if end > mmap.len() || start > end {
                bail!("tensor {name} out of bounds");
            }
            tensors.insert(name.clone(), TensorInfo { dtype, shape, start, end });
        }
        Ok(Self { mmap, tensors })
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
        Ok(&self.mmap[t.start..t.end])
    }

    /// Returns a bf16 tensor as a slice of raw u16 bit patterns.
    pub fn bf16(&self, name: &str) -> Result<&[u16]> {
        let t = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
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
        self.mmap.len()
    }

    /// Touch every page so the weights are resident before the first forward pass.
    pub fn preload(&self) {
        if let Storage::Mmap(m) = &self.mmap {
            let _ = m.advise(memmap2::Advice::WillNeed);
        }
        const CHUNK: usize = 64 << 20;
        let total: u64 = self
            .mmap
            .par_chunks(CHUNK)
            .map(|c| c.iter().step_by(4096).map(|&b| b as u64).sum::<u64>())
            .sum();
        std::hint::black_box(total);
    }
}

/// A model directory: either `model.safetensors` or an index plus shards.
pub struct ModelFiles {
    shards: Vec<SafeTensors>,
    index: HashMap<String, usize>,
}

impl ModelFiles {
    pub fn open(dir: &Path, in_memory: bool) -> Result<Self> {
        let single = dir.join("model.safetensors");
        let idx = dir.join("model.safetensors.index.json");
        let mut files: Vec<PathBuf> = Vec::new();
        if single.exists() {
            files.push(single);
        } else if idx.exists() {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx)?)?;
            let map = v["weight_map"].as_object().context("weight_map")?;
            let mut names: Vec<String> = map.values().map(|x| x.as_str().unwrap().to_string()).collect();
            names.sort();
            names.dedup();
            files.extend(names.into_iter().map(|n| dir.join(n)));
        } else {
            bail!("no model.safetensors or index found in {}", dir.display());
        }
        let shards: Vec<SafeTensors> = files
            .par_iter()
            .map(|p| SafeTensors::open(p, in_memory))
            .collect::<Result<Vec<_>>>()?;
        let mut index = HashMap::new();
        for (i, s) in shards.iter().enumerate() {
            for n in s.names() {
                index.insert(n.clone(), i);
            }
        }
        Ok(Self { shards, index })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).and_then(|&i| self.shards[i].info(name))
    }

    pub fn bf16(&self, name: &str) -> Result<&[u16]> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.shards[i].bf16(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.shards[i].bytes(name)
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
