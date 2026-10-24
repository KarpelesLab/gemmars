//! Weight streaming for models larger than RAM: the weights are read in place from memory-mapped
//! checkpoint files, and a reader thread pulls the byte ranges of the next layers into the page
//! cache while the current one computes.
//!
//! The model is cut into *units* (a decoder layer, the LM head, the MTP layer, ...) listed in
//! execution order; the forward passes bracket each unit with [`StreamHook::enter`]. Every pass
//! reads the units cyclically, so a plain LRU page cache smaller than the model gets no hits at
//! all (each page is evicted before its next use). Instead, given a page-cache budget, a fixed
//! *resident* set of units is kept in memory (those used most per pass first: draft layers and
//! heads, then evenly spaced layers), and every other unit is dropped from the cache right after
//! use, so the cache holds the resident set plus the read-ahead window and nothing else. On
//! Darwin a clean file page leaves the cache only through `msync(MS_INVALIDATE)` (`madvise`
//! `MADV_DONTNEED` / `MADV_FREE` leave file pages cached), and a read-ahead with
//! `fcntl(F_RDADVISE)` or `pread` runs at full flash speed (~13 GB/s on an M5 Max) while
//! demand faults from the compute threads manage 2-3 GB/s, so the reader thread does the reads
//! and then touches the pages so that compute threads only take minor faults.
//!
//! For phone emulation on a desktop, `flash_bytes_per_sec` throttles the bytes the reader
//! actually fetches from storage (pages that were not cached, sampled with `mincore`), and
//! [`StreamStats`] reports bytes read versus cache hits and the time the forward passes waited
//! for IO.
//!
//! The streamer keeps raw addresses into the mapped files: drop it (with the models holding it)
//! before the `ModelFiles` it was built from.

use crate::clock::Instant;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// How resident units are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentMode {
    /// As clean file pages in the page cache (not charged as app memory on iOS; the OS may still
    /// evict them under pressure, after which they are read again on use).
    Cache,
    /// Page cache pages pinned with `mlock`.
    Lock,
    /// Copied into anonymous memory mapped over the same addresses (dirty memory: charged to the
    /// app, never evicted).
    Copy,
}

/// How the reader thread fetches a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoMethod {
    /// `fcntl(F_RDADVISE)` on Darwin, `madvise(MADV_WILLNEED)` elsewhere, then touch the pages.
    Advise,
    /// `pread` into a scratch buffer (fills the page cache), then touch the pages.
    Pread,
    /// Only touch the pages (demand faults on the reader thread).
    Touch,
}

/// Streaming settings (see the module docs).
#[derive(Clone, Debug)]
pub struct StreamOptions {
    /// Page-cache bytes the model may occupy. The resident set gets this minus `prefetch_bytes`,
    /// and streamed units are evicted after use. None: no budget (the OS cache decides; units are
    /// only read ahead).
    pub cache_budget: Option<u64>,
    /// Size of the resident set, overriding the one derived from `cache_budget` (also enables
    /// eviction of the streamed units).
    pub resident_bytes: Option<u64>,
    /// Read-ahead window: units up to this many bytes past the current one are being fetched
    /// (at least the next one).
    pub prefetch_bytes: u64,
    pub resident_mode: ResidentMode,
    /// Drop streamed units from the page cache after use (only with a budget or resident size).
    pub evict: bool,
    /// Drop every cached page of the files when streaming starts (a cold cache, as after a
    /// reboot); only with a budget or resident size.
    pub cold_start: bool,
    /// Emulated storage bandwidth: the reader sleeps so that pages that were not cached arrive
    /// no faster than this. None: as fast as the device reads.
    pub flash_bytes_per_sec: Option<f64>,
    pub io: IoMethod,
    /// Reader request size.
    pub chunk_bytes: usize,
    /// Reader threads (requests in flight).
    pub io_threads: usize,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            cache_budget: None,
            resident_bytes: None,
            prefetch_bytes: 768 << 20,
            resident_mode: ResidentMode::Cache,
            evict: true,
            cold_start: true,
            flash_bytes_per_sec: None,
            io: if cfg!(any(target_os = "macos", target_os = "ios")) {
                IoMethod::Advise
            } else {
                IoMethod::Pread
            },
            chunk_bytes: 8 << 20,
            io_threads: 2,
        }
    }
}

impl StreamOptions {
    /// Fills `flash_bytes_per_sec` from `OXINFER_FLASH_MBPS` (MB/s) when it is set.
    pub fn with_env(mut self) -> Self {
        if let Some(v) = std::env::var("OXINFER_FLASH_MBPS").ok().and_then(|v| v.parse::<f64>().ok()) {
            self.flash_bytes_per_sec = Some(v * 1e6);
        }
        self
    }

    fn managed(&self) -> bool {
        self.cache_budget.is_some() || self.resident_bytes.is_some()
    }
}

/// One unit of a model: a name, the memory its weights occupy (`(address, length)` spans inside
/// mapped files), and how many times a decoding round uses it (residency priority).
#[derive(Clone, Debug)]
pub struct UnitSpec {
    pub name: String,
    pub ranges: Vec<(usize, usize)>,
    pub uses: f64,
}

/// Cumulative counters; subtract two snapshots for one step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamStats {
    /// Bytes the reader found uncached (fetched from storage).
    pub bytes_read: u64,
    /// Bytes the reader found already cached.
    pub bytes_cached: u64,
    /// Bytes dropped from the page cache after use.
    pub bytes_evicted: u64,
    /// Time forward passes waited for a unit to arrive.
    pub stall: Duration,
    /// Times a forward pass had to wait.
    pub stalls: u64,
    /// Reader busy time (including throttling).
    pub io_busy: Duration,
    /// Part of `io_busy` spent sleeping for the bandwidth throttle.
    pub throttled: Duration,
    /// Streamed (non-resident) unit uses.
    pub units: u64,
    /// Resident unit uses.
    pub resident_uses: u64,
}

impl std::ops::Sub for StreamStats {
    type Output = StreamStats;
    fn sub(self, o: StreamStats) -> StreamStats {
        StreamStats {
            bytes_read: self.bytes_read - o.bytes_read,
            bytes_cached: self.bytes_cached - o.bytes_cached,
            bytes_evicted: self.bytes_evicted - o.bytes_evicted,
            stall: self.stall.saturating_sub(o.stall),
            stalls: self.stalls - o.stalls,
            io_busy: self.io_busy.saturating_sub(o.io_busy),
            throttled: self.throttled.saturating_sub(o.throttled),
            units: self.units - o.units,
            resident_uses: self.resident_uses - o.resident_uses,
        }
    }
}

/// What the streamer decided at start-up.
#[derive(Clone, Debug, Default)]
pub struct StreamPlan {
    /// `(name, bytes, resident)` per unit, in execution order.
    pub units: Vec<(String, u64, bool)>,
    pub resident_bytes: u64,
    /// Bytes of the units that are streamed (read once per pass that uses them all).
    pub streamed_bytes: u64,
    /// Resident units that could not be pinned (`ResidentMode::Lock`) or copied.
    pub resident_failures: usize,
    /// Weight bytes outside the mapped files (copies in RAM): 0 when every weight is read in place.
    pub unmapped_bytes: u64,
    /// Time spent evicting the cache and loading the resident set.
    pub setup: Duration,
}

/// A page-aligned piece of a unit: its address, length and place in a file.
#[derive(Clone, Copy, Debug)]
struct Span {
    file: usize,
    addr: usize,
    len: usize,
    off: u64,
}

struct Unit {
    /// Page-aligned outward (what to read).
    spans: Vec<Span>,
    /// Page-aligned inward (what to evict: pages shared with neighbours stay).
    inner: Vec<Span>,
    bytes: u64,
    resident: bool,
}

struct State {
    /// Position (in `order`) of the unit in use or expected next.
    cur: usize,
    loaded: Vec<bool>,
    in_use: Option<usize>,
    stop: bool,
}

#[derive(Default)]
struct Counters {
    bytes_read: AtomicU64,
    bytes_cached: AtomicU64,
    bytes_evicted: AtomicU64,
    stall_ns: AtomicU64,
    stalls: AtomicU64,
    io_ns: AtomicU64,
    throttle_ns: AtomicU64,
    units: AtomicU64,
    resident_uses: AtomicU64,
}

struct Inner {
    opts: StreamOptions,
    files: Vec<std::fs::File>,
    /// `(address, length)` of each file's mapping.
    maps: Vec<(usize, usize)>,
    units: Vec<Unit>,
    /// Streamed units in execution order, and each unit's position in it.
    order: Vec<usize>,
    pos: Vec<Option<usize>>,
    managed: bool,
    page: usize,
    /// Emulated storage bandwidth in bytes/s as f64 bits, 0 for none.
    rate: AtomicU64,
    throttle: Mutex<Throttle>,
    st: Mutex<State>,
    cv: Condvar,
    c: Counters,
}

/// The streaming engine shared by the models it serves (see the module docs).
pub struct Streamer {
    inner: Arc<Inner>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    plan: StreamPlan,
}

/// A model's link to a streamer: unit `base + i` is the model's unit `i`.
#[derive(Clone)]
pub struct StreamHook {
    s: Arc<Streamer>,
    base: usize,
}

/// Marks a unit as in use until dropped.
pub struct UnitGuard<'a>(Option<(&'a Streamer, usize)>);

impl Drop for UnitGuard<'_> {
    fn drop(&mut self) {
        if let Some((s, u)) = self.0 {
            s.end(u);
        }
    }
}

impl StreamHook {
    pub fn new(s: Arc<Streamer>, base: usize) -> Self {
        Self { s, base }
    }
    /// Waits until the model's unit `i` is in memory; it stays in use until the guard drops.
    pub fn enter(&self, i: usize) -> UnitGuard<'_> {
        let u = self.base + i;
        self.s.begin(u);
        UnitGuard(Some((&self.s, u)))
    }
}

/// `hook.enter(i)` if there is a hook.
pub(crate) fn enter(hook: &Option<StreamHook>, i: usize) -> UnitGuard<'_> {
    match hook {
        Some(h) => h.enter(i),
        None => UnitGuard(None),
    }
}

impl Streamer {
    /// Starts streaming `units` (in execution order) of models whose weights borrow `files`
    /// (all of them memory-mapped).
    pub fn new(files: &[&ModelFiles], units: Vec<UnitSpec>, opts: StreamOptions) -> Result<Arc<Self>> {
        let t0 = Instant::now();
        let page = sys::page_size();
        let mut maps = Vec::new();
        let mut handles = Vec::new();
        for f in files {
            let ms = f
                .mappings()
                .context("streaming needs memory-mapped weights (open the files with mmap)")?;
            for m in ms {
                maps.push((m.bytes.as_ptr() as usize, m.bytes.len()));
                handles.push(m.file.try_clone().context("duplicate file handle")?);
            }
        }
        let locate = |a: usize| maps.iter().position(|&(b, l)| a >= b && a < b + l);
        let mut plan = StreamPlan::default();
        let mut built = Vec::with_capacity(units.len());
        for u in &units {
            let mut rs: Vec<(usize, usize)> = u.ranges.iter().copied().filter(|r| r.1 > 0).collect();
            rs.sort_unstable();
            // Coalesce: tensors of a unit sit next to each other (small gaps hold norms).
            let mut merged: Vec<(usize, usize)> = Vec::new();
            for (a, l) in rs {
                match merged.last_mut() {
                    Some(m) if a <= m.0 + m.1 + (1 << 20) && locate(a) == locate(m.0) => m.1 = m.1.max(a + l - m.0),
                    _ => merged.push((a, l)),
                }
            }
            let (mut spans, mut inner) = (Vec::new(), Vec::new());
            let mut bytes = 0u64;
            for (a, l) in merged {
                let Some(fi) = locate(a) else {
                    plan.unmapped_bytes += l as u64;
                    continue;
                };
                let (base, mlen) = maps[fi];
                let s = a & !(page - 1);
                let e = (a + l).next_multiple_of(page).min(base + mlen);
                spans.push(Span {
                    file: fi,
                    addr: s,
                    len: e - s,
                    off: (s - base) as u64,
                });
                let (is, ie) = (a.next_multiple_of(page), (a + l) & !(page - 1));
                if ie > is {
                    inner.push(Span {
                        file: fi,
                        addr: is,
                        len: ie - is,
                        off: (is - base) as u64,
                    });
                }
                bytes += (e - s) as u64;
            }
            built.push(Unit {
                spans,
                inner,
                bytes,
                resident: false,
            });
        }

        let managed = opts.managed();
        if managed {
            let budget = opts
                .resident_bytes
                .unwrap_or_else(|| opts.cache_budget.unwrap().saturating_sub(opts.prefetch_bytes));
            for i in choose_resident(&units, &built, budget) {
                built[i].resident = true;
            }
        }
        let order: Vec<usize> = (0..built.len()).filter(|&i| !built[i].resident).collect();
        let mut pos = vec![None; built.len()];
        for (p, &u) in order.iter().enumerate() {
            pos[u] = Some(p);
        }
        for (u, b) in units.iter().zip(&built) {
            plan.units.push((u.name.clone(), b.bytes, b.resident));
            if b.resident {
                plan.resident_bytes += b.bytes;
            } else {
                plan.streamed_bytes += b.bytes;
            }
        }
        let n = order.len();
        let rate = AtomicU64::new(opts.flash_bytes_per_sec.map_or(0, f64::to_bits));
        let inner = Arc::new(Inner {
            rate,
            throttle: Mutex::new(Throttle::new()),
            opts,
            files: handles,
            maps,
            units: built,
            order,
            pos,
            managed,
            page,
            st: Mutex::new(State {
                cur: 0,
                loaded: vec![false; n],
                in_use: None,
                stop: false,
            }),
            cv: Condvar::new(),
            c: Counters::default(),
        });
        if managed {
            plan.resident_failures = inner.setup();
        }
        // Start counting from here: the set-up reads are not part of any pass.
        inner.reset_counters();
        plan.setup = t0.elapsed();
        let th = {
            let inner = inner.clone();
            std::thread::Builder::new()
                .name("oxinfer-stream".into())
                .spawn(move || inner.io_loop())
                .context("start the weight streaming thread")?
        };
        Ok(Arc::new(Self {
            inner,
            thread: Mutex::new(Some(th)),
            plan,
        }))
    }

    pub fn plan(&self) -> &StreamPlan {
        &self.plan
    }

    pub fn options(&self) -> &StreamOptions {
        &self.inner.opts
    }

    pub fn stats(&self) -> StreamStats {
        let c = &self.inner.c;
        StreamStats {
            bytes_read: c.bytes_read.load(Relaxed),
            bytes_cached: c.bytes_cached.load(Relaxed),
            bytes_evicted: c.bytes_evicted.load(Relaxed),
            stall: Duration::from_nanos(c.stall_ns.load(Relaxed)),
            stalls: c.stalls.load(Relaxed),
            io_busy: Duration::from_nanos(c.io_ns.load(Relaxed)),
            throttled: Duration::from_nanos(c.throttle_ns.load(Relaxed)),
            units: c.units.load(Relaxed),
            resident_uses: c.resident_uses.load(Relaxed),
        }
    }

    /// Bytes of the mapped files currently in the page cache (`mincore` over every mapping;
    /// None where the platform cannot tell).
    pub fn cached_bytes(&self) -> Option<u64> {
        let mut n = 0;
        for &(a, l) in &self.inner.maps {
            let a0 = a & !(self.inner.page - 1);
            n += sys::resident_bytes(a0, (a + l - a0).next_multiple_of(self.inner.page), self.inner.page)?;
        }
        Some(n)
    }

    /// Changes the emulated storage bandwidth while running.
    pub fn set_flash_bytes_per_sec(&self, rate: Option<f64>) {
        self.inner.rate.store(rate.map_or(0, f64::to_bits), Relaxed);
    }

    fn begin(&self, u: usize) {
        let inner = &self.inner;
        let Some(p) = inner.pos.get(u).copied().flatten() else {
            inner.c.resident_uses.fetch_add(1, Relaxed);
            return;
        };
        inner.c.units.fetch_add(1, Relaxed);
        let mut st = inner.st.lock().unwrap();
        if st.cur != p {
            st.cur = p;
            inner.cv.notify_all();
        }
        if !st.loaded[p] {
            let t = Instant::now();
            inner.c.stalls.fetch_add(1, Relaxed);
            while !st.loaded[p] {
                st = inner.cv.wait(st).unwrap();
            }
            inner.c.stall_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
        st.in_use = Some(p);
    }

    fn end(&self, u: usize) {
        let inner = &self.inner;
        let Some(p) = inner.pos.get(u).copied().flatten() else {
            return;
        };
        let mut st = inner.st.lock().unwrap();
        st.in_use = None;
        st.loaded[p] = false;
        if inner.managed && inner.opts.evict {
            inner.evict(inner.order[p]);
        }
        st.cur = (p + 1) % inner.order.len();
        inner.cv.notify_all();
    }
}

impl Streamer {
    /// Stops the reader thread (idempotent; dropping the last handle does it too). Call it before
    /// the mapped files go away when other handles may outlive them.
    pub fn shutdown(&self) {
        self.inner.st.lock().unwrap().stop = true;
        self.inner.cv.notify_all();
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The resident set within `budget` bytes: units used several times per round first (most used
/// first), then the once-per-pass units spread evenly over the pass so the reader can work ahead
/// while they compute.
fn choose_resident(specs: &[UnitSpec], units: &[Unit], budget: u64) -> Vec<usize> {
    let mut left = budget;
    let mut chosen = Vec::new();
    let mut multi: Vec<usize> = (0..units.len()).filter(|&i| specs[i].uses > 1.0).collect();
    multi.sort_by(|&a, &b| specs[b].uses.total_cmp(&specs[a].uses));
    for i in multi {
        if units[i].bytes <= left {
            left -= units[i].bytes;
            chosen.push(i);
        }
    }
    let once: Vec<usize> = (0..units.len()).filter(|&i| specs[i].uses <= 1.0 && units[i].bytes > 0).collect();
    if once.is_empty() {
        return chosen;
    }
    let avg = once.iter().map(|&i| units[i].bytes).sum::<u64>() / once.len() as u64;
    let fit = ((left / avg.max(1)) as usize).min(once.len());
    let mut rest = Vec::new();
    for (k, &i) in once.iter().enumerate() {
        // Every (n / fit)-th unit (Bresenham), if it fits.
        let pick = (k + 1) * fit / once.len() > k * fit / once.len();
        if pick && units[i].bytes <= left {
            left -= units[i].bytes;
            chosen.push(i);
        } else {
            rest.push(i);
        }
    }
    for i in rest {
        if units[i].bytes <= left {
            left -= units[i].bytes;
            chosen.push(i);
        }
    }
    chosen.sort_unstable();
    chosen
}

impl Inner {
    fn reset_counters(&self) {
        let c = &self.c;
        for a in [
            &c.bytes_read,
            &c.bytes_cached,
            &c.bytes_evicted,
            &c.stall_ns,
            &c.stalls,
            &c.io_ns,
            &c.throttle_ns,
            &c.units,
            &c.resident_uses,
        ] {
            a.store(0, Relaxed);
        }
    }

    /// Optionally drops the whole files from the cache, then loads (and pins or copies) the
    /// resident units. Returns how many could not be pinned or copied.
    fn setup(&self) -> usize {
        if self.opts.cold_start {
            for (fi, &(a, l)) in self.maps.iter().enumerate() {
                let a0 = a & !(self.page - 1);
                let span = Span {
                    file: fi,
                    addr: a0,
                    len: (a + l - a0) & !(self.page - 1),
                    off: 0,
                };
                sys::evict(&self.files[fi], span);
            }
        }
        // The resident set loads at full speed (it is read once, before any pass).
        let mut failures = 0;
        for u in self.units.iter().filter(|u| u.resident) {
            match self.opts.resident_mode {
                ResidentMode::Cache => self.load(u, false),
                ResidentMode::Lock => {
                    self.load(u, false);
                    failures += u.spans.iter().filter(|s| !sys::lock(**s)).count().min(1);
                }
                ResidentMode::Copy => {
                    // Only whole pages can be remapped; the shared edge pages stay file-backed.
                    self.load(u, false);
                    failures += u.inner.iter().filter(|s| !sys::copy_in(&self.files[s.file], **s)).count().min(1);
                }
            }
        }
        failures
    }

    fn evict(&self, u: usize) {
        let mut n = 0;
        for s in &self.units[u].inner {
            if sys::evict(&self.files[s.file], *s) {
                n += s.len as u64;
            }
        }
        self.c.bytes_evicted.fetch_add(n, Relaxed);
    }

    /// Brings unit `u` into the page cache (and this process's page tables) with `io_threads`
    /// readers, counting what was already cached and throttling (if `throttle`) what was not.
    fn load(&self, u: &Unit, throttle: bool) {
        let chunk = self.opts.chunk_bytes.max(self.page).next_multiple_of(self.page);
        let mut chunks = Vec::new();
        for s in &u.spans {
            let mut o = 0;
            while o < s.len {
                chunks.push(Span {
                    addr: s.addr + o,
                    off: s.off + o as u64,
                    len: chunk.min(s.len - o),
                    file: s.file,
                });
                o += chunk;
            }
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let work = || {
            crate::pool::raise_qos();
            let mut scratch = Vec::new();
            while let Some(&c) = chunks.get(next.fetch_add(1, Relaxed)) {
                // A full `mincore` costs ~15 ms per GB on Darwin: first look at three pages, and
                // count page by page only when one of them is missing. Unknown counts as a miss.
                let probe = [c.addr, c.addr + ((c.len / 2) & !(self.page - 1)), c.addr + c.len - self.page];
                let have = if probe
                    .iter()
                    .all(|&a| sys::resident_bytes(a, self.page, self.page) == Some(self.page as u64))
                {
                    c.len as u64
                } else {
                    sys::resident_bytes(c.addr, c.len, self.page).unwrap_or(0)
                };
                let miss = c.len as u64 - have;
                self.c.bytes_cached.fetch_add(have, Relaxed);
                self.c.bytes_read.fetch_add(miss, Relaxed);
                if miss == 0 {
                    // Cached pages map at minor-fault cost wherever they are first used.
                    continue;
                }
                match self.opts.io {
                    IoMethod::Advise => sys::advise(&self.files[c.file], c),
                    IoMethod::Pread => sys::pread(&self.files[c.file], c, &mut scratch),
                    IoMethod::Touch => {}
                }
                if std::env::var_os("OXINFER_STREAM_NOTOUCH").is_none() {
                    touch(c.addr, c.len, self.page);
                }
                let rate = f64::from_bits(self.rate.load(Relaxed));
                if throttle && rate > 0.0 {
                    let wait = self.throttle.lock().unwrap().charge(miss, rate);
                    if !wait.is_zero() {
                        std::thread::sleep(wait);
                        self.c.throttle_ns.fetch_add(wait.as_nanos() as u64, Relaxed);
                    }
                }
            }
        };
        let n = self.opts.io_threads.clamp(1, chunks.len().max(1));
        if n == 1 {
            work();
        } else {
            std::thread::scope(|sc| {
                for _ in 1..n {
                    sc.spawn(work);
                }
                work();
            });
        }
    }

    fn io_loop(&self) {
        // As urgent as the compute threads, which wait for it.
        crate::pool::raise_qos();
        let n = self.order.len();
        if n == 0 {
            return;
        }
        loop {
            let job = {
                let mut st = self.st.lock().unwrap();
                loop {
                    if st.stop {
                        return;
                    }
                    // The window: the current unit and those after it, up to `prefetch_bytes`.
                    let mut want = Vec::new();
                    let mut acc = 0u64;
                    for k in 0..n {
                        let p = (st.cur + k) % n;
                        let b = self.units[self.order[p]].bytes;
                        if k > 1 && acc + b > self.opts.prefetch_bytes {
                            break;
                        }
                        acc += b;
                        want.push(p);
                    }
                    if self.managed && self.opts.evict {
                        // Units read ahead but skipped (the pass took another path) leave the cache.
                        for p in 0..n {
                            if st.loaded[p] && st.in_use != Some(p) && !want.contains(&p) {
                                st.loaded[p] = false;
                                self.evict(self.order[p]);
                            }
                        }
                    }
                    if let Some(&p) = want.iter().find(|&&p| !st.loaded[p] && st.in_use != Some(p)) {
                        break p;
                    }
                    st = self.cv.wait(st).unwrap();
                }
            };
            let t = Instant::now();
            self.load(&self.units[self.order[job]], true);
            self.c.io_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            let mut st = self.st.lock().unwrap();
            st.loaded[job] = true;
            self.cv.notify_all();
        }
    }
}

/// Emulated storage: each fetched byte occupies the device for `1 / rate` seconds.
struct Throttle {
    t0: Instant,
    /// When the emulated device becomes free, in seconds since `t0`.
    free_at: f64,
}

impl Throttle {
    fn new() -> Self {
        Self {
            t0: Instant::now(),
            free_at: 0.0,
        }
    }
    /// Accounts `bytes` fetched just now at `rate` bytes/s; returns how long to wait until the
    /// emulated device would have delivered them.
    fn charge(&mut self, bytes: u64, rate: f64) -> Duration {
        let now = self.t0.elapsed().as_secs_f64();
        self.free_at = self.free_at.max(now) + bytes as f64 / rate;
        Duration::from_secs_f64((self.free_at - now).max(0.0))
    }
}

/// Reads one byte per page so the pages are mapped into this process.
fn touch(addr: usize, len: usize, page: usize) {
    let mut s = 0u8;
    let mut a = addr;
    while a < addr + len {
        // SAFETY: the span lies inside a live read-only mapping (see `Streamer`).
        s = s.wrapping_add(unsafe { std::ptr::read_volatile(a as *const u8) });
        a += page;
    }
    std::hint::black_box(s);
}

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android"))]
mod sys {
    use super::Span;
    use std::ffi::c_void;
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
        fn getpagesize() -> i32;
        fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> i32;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        fn msync(addr: *mut c_void, len: usize, flags: i32) -> i32;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        fn mlock(addr: *const c_void, len: usize) -> i32;
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        fn posix_fadvise(fd: i32, off: i64, len: i64, advice: i32) -> i32;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const MS_INVALIDATE: i32 = 2;
    const MADV_WILLNEED: i32 = 3;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const MADV_DONTNEED: i32 = 4;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 2;
    const MAP_FIXED: i32 = 0x10;
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const MAP_ANON: i32 = 0x1000;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const MAP_ANON: i32 = 0x20;
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const F_RDADVISE: i32 = 44;

    pub fn page_size() -> usize {
        (unsafe { getpagesize() }).max(4096) as usize
    }

    /// Bytes of `[addr, addr + len)` (page-aligned) in memory.
    pub fn resident_bytes(addr: usize, len: usize, page: usize) -> Option<u64> {
        let mut v = vec![0u8; len.div_ceil(page)];
        if unsafe { mincore(addr as *mut c_void, len, v.as_mut_ptr()) } != 0 {
            return None;
        }
        Some(v.iter().filter(|&&b| b & 1 != 0).count() as u64 * page as u64)
    }

    /// Drops clean pages from the page cache. On Darwin `msync(MS_INVALIDATE)` does (and
    /// `madvise` does not); on Linux the mapping is zapped and the file range advised away.
    pub fn evict(_f: &std::fs::File, s: Span) -> bool {
        let p = s.addr as *mut c_void;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe {
            madvise(p, s.len, MADV_DONTNEED);
            posix_fadvise(_f.as_raw_fd(), s.off as i64, s.len as i64, 4) == 0
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        unsafe {
            msync(p, s.len, MS_INVALIDATE) == 0
        }
    }

    /// Asks the kernel to read a range ahead (synchronous on Darwin for `F_RDADVISE`).
    pub fn advise(_f: &std::fs::File, s: Span) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            #[repr(C)]
            struct Radvisory {
                off: i64,
                count: i32,
            }
            let mut o = 0;
            while o < s.len {
                let n = (s.len - o).min(1 << 30);
                let ra = Radvisory {
                    off: (s.off + o as u64) as i64,
                    count: n as i32,
                };
                if unsafe { fcntl(_f.as_raw_fd(), F_RDADVISE, &ra as *const Radvisory) } != 0 {
                    unsafe { madvise((s.addr + o) as *mut c_void, n, MADV_WILLNEED) };
                }
                o += n;
            }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe {
            madvise(s.addr as *mut c_void, s.len, MADV_WILLNEED);
        }
    }

    pub fn pread(f: &std::fs::File, s: Span, scratch: &mut Vec<u8>) {
        use std::os::unix::fs::FileExt;
        scratch.resize(s.len, 0);
        let _ = f.read_exact_at(&mut scratch[..s.len], s.off);
    }

    pub fn lock(s: Span) -> bool {
        unsafe { mlock(s.addr as *const c_void, s.len) == 0 }
    }

    /// Replaces the mapped pages `s` with anonymous memory holding the same bytes.
    pub fn copy_in(f: &std::fs::File, s: Span) -> bool {
        use std::os::unix::fs::FileExt;
        let p = s.addr as *mut c_void;
        unsafe {
            let m = mmap(p, s.len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED | MAP_ANON, -1, 0);
            if m != p {
                return false;
            }
            let buf = std::slice::from_raw_parts_mut(s.addr as *mut u8, s.len);
            let ok = f.read_exact_at(buf, s.off).is_ok();
            mprotect(p, s.len, PROT_READ);
            ok
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android")))]
mod sys {
    use super::Span;

    pub fn page_size() -> usize {
        4096
    }
    pub fn resident_bytes(_addr: usize, _len: usize, _page: usize) -> Option<u64> {
        None
    }
    pub fn evict(_f: &std::fs::File, _s: Span) -> bool {
        false
    }
    pub fn advise(_f: &std::fs::File, _s: Span) {}
    pub fn pread(_f: &std::fs::File, _s: Span, _scratch: &mut Vec<u8>) {}
    pub fn lock(_s: Span) -> bool {
        false
    }
    pub fn copy_in(_f: &std::fs::File, _s: Span) -> bool {
        false
    }
}

/// Streams a Qwen model and optionally its MTP layer, whose weights borrow `files` (all
/// memory-mapped; the main checkpoint and any MTP sidecar). Units run in the order of a
/// speculative round: the MTP layer (and trimmed draft head), the layers, the LM head.
/// `draft_tokens` (the most drafts per round) sets the residency priority of the draft units.
/// Detach (`set_stream(None)`) or drop the models before dropping `files`.
pub fn attach_qwen(
    model: &mut crate::qwen::QwenModel<'_>,
    mtp: Option<&mut crate::qwen_mtp::QwenMtp<'_>>,
    files: &[&ModelFiles],
    draft_tokens: usize,
    opts: StreamOptions,
) -> Result<Arc<Streamer>> {
    let mut units = Vec::new();
    let shared_head = mtp.as_ref().is_some_and(|m| m.draft_head().is_none());
    if let Some(m) = &mtp {
        units.extend(m.stream_units(draft_tokens));
    }
    let base = units.len();
    let mut mu = model.stream_units();
    if shared_head {
        // Drafts read the LM head too.
        mu.last_mut().unwrap().uses += draft_tokens as f64;
    }
    units.extend(mu);
    let s = Streamer::new(files, units, opts)?;
    if let Some(m) = mtp {
        m.set_stream(Some(StreamHook::new(s.clone(), 0)));
    }
    model.set_stream(Some(StreamHook::new(s.clone(), base)));
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(uses: f64) -> UnitSpec {
        UnitSpec {
            name: String::new(),
            ranges: Vec::new(),
            uses,
        }
    }
    fn unit(bytes: u64) -> Unit {
        Unit {
            spans: Vec::new(),
            inner: Vec::new(),
            bytes,
            resident: false,
        }
    }

    #[test]
    fn resident_set_prefers_reused_units_then_spreads() {
        // A draft layer used 4 times per round, then 8 layers of 10.
        let mut specs = vec![spec(4.0)];
        let mut units = vec![unit(5)];
        for _ in 0..8 {
            specs.push(spec(1.0));
            units.push(unit(10));
        }
        assert_eq!(choose_resident(&specs, &units, 4), Vec::<usize>::new());
        assert_eq!(choose_resident(&specs, &units, 5), vec![0]);
        // 40 left: four of the eight layers, every other one.
        assert_eq!(choose_resident(&specs, &units, 45), vec![0, 2, 4, 6, 8]);
        assert_eq!(choose_resident(&specs, &units, 1000).len(), 9);
    }

    #[test]
    fn throttle_paces_bytes() {
        let mut th = Throttle::new();
        let t = Instant::now();
        for _ in 0..4 {
            std::thread::sleep(th.charge(5_000_000, 1e9));
        }
        let e = t.elapsed().as_secs_f64();
        assert!((0.018..0.2).contains(&e), "{e}");
    }
}
