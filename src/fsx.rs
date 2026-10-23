//! File access behind one small interface so the model loaders work both on a real filesystem and
//! in a browser, where there is none: on `wasm32-unknown-unknown` the host (JavaScript) registers
//! files in an in-memory table (`vfs_put`) and the loaders read them from there. Large files are
//! stored page-aligned and handed out by move, so a checkpoint is never copied inside the module.

use crate::quant::PageVec;
use anyhow::{Context, Result};
use std::path::Path;

/// A file kept by the host instead of in module memory, read by byte range (browser only: lets a
/// model use lookup tables larger than what fits in the 4 GiB wasm address space).
#[derive(Clone, Copy, Debug)]
pub struct HostFile {
    pub id: u32,
    pub len: usize,
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod imp {
    use super::*;
    pub fn read(path: &Path) -> Result<Vec<u8>> {
        std::fs::read(path).with_context(|| format!("read {}", path.display()))
    }
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }
    pub fn read_pages(path: &Path) -> Result<(PageVec<u64>, usize)> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let len = file.metadata()?.len() as usize;
        Ok((crate::safetensors::read_file_parallel(&file, len)?, len))
    }
    pub fn host_file(_path: &Path) -> Option<HostFile> {
        None
    }
    pub fn read_host(_f: HostFile, _offset: u64, _buf: &mut [u8]) -> Result<()> {
        anyhow::bail!("host-backed files exist only in the browser build")
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod imp {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        static FILES: RefCell<HashMap<String, (PageVec<u64>, usize)>> = RefCell::new(HashMap::new());
        static HOST: RefCell<HashMap<String, HostFile>> = RefCell::new(HashMap::new());
    }

    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        /// Copies `len` bytes at `offset` of host file `id` into `ptr`; returns 0 on success.
        fn host_read(id: u32, offset: f64, ptr: *mut u8, len: usize) -> i32;
    }

    /// Registers a file the host keeps (see `HostFile`).
    pub fn put_host(path: &str, f: HostFile) {
        HOST.with(|h| h.borrow_mut().insert(key(Path::new(path)), f));
    }
    pub fn host_file(path: &Path) -> Option<HostFile> {
        HOST.with(|h| h.borrow().get(&key(path)).copied())
    }
    pub fn read_host(f: HostFile, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset as usize + buf.len() > f.len {
            anyhow::bail!("host read past the end of file {}", f.id);
        }
        let rc = unsafe { host_read(f.id, offset as f64, buf.as_mut_ptr(), buf.len()) };
        if rc != 0 {
            anyhow::bail!("host read of file {} failed ({rc})", f.id);
        }
        Ok(())
    }

    fn key(path: &Path) -> String {
        // Normalise "a//b", "./a" and a trailing slash away; everything is relative to one root.
        let mut parts: Vec<&str> = Vec::new();
        for c in path.components() {
            match c {
                std::path::Component::Normal(s) => parts.push(s.to_str().unwrap_or("")),
                std::path::Component::ParentDir => {
                    parts.pop();
                }
                _ => {}
            }
        }
        parts.join("/")
    }

    /// Registers (or replaces) a file. `len` is the number of valid bytes in `data`.
    pub fn put(path: &str, data: PageVec<u64>, len: usize) {
        FILES.with(|f| f.borrow_mut().insert(key(Path::new(path)), (data, len)));
    }
    pub fn read(path: &Path) -> Result<Vec<u8>> {
        FILES.with(|f| {
            let f = f.borrow();
            let (v, n) = f.get(&key(path)).with_context(|| format!("read {}: not provided by the host", path.display()))?;
            Ok(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, *n) }.to_vec())
        })
    }
    pub fn exists(path: &Path) -> bool {
        FILES.with(|f| f.borrow().contains_key(&key(path))) || host_file(path).is_some()
    }
    /// Takes a file out of the table (large checkpoints are consumed, not copied).
    pub fn read_pages(path: &Path) -> Result<(PageVec<u64>, usize)> {
        FILES.with(|f| f.borrow_mut().remove(&key(path)).with_context(|| format!("open {}: not provided by the host", path.display())))
    }
}

pub use imp::*;
