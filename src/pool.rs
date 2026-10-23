//! Minimal spin-waiting thread pool for low-latency parallel-for.
//!
//! Workers spin on a generation counter while work is flowing, so dispatching a job
//! costs about a microsecond. After a quiet period they back off to parking.
//! A region ends as soon as every *claimed* block is done; a worker that never
//! joined the region (e.g. because it was preempted) does not hold up the barrier.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Arc;
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

type JobFn<'a> = &'a (dyn Fn(usize) + Sync);

/// One parallel-for, living on the caller's stack for the duration of `run`.
struct Region {
    f: JobFn<'static>,
    n: usize,
    next: AtomicUsize,
}

struct Shared {
    generation: AtomicU64,
    region: AtomicPtr<Region>,
    active: AtomicUsize,
    sleepers: AtomicUsize,
    exit: AtomicBool,
}

pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<Thread>,
    workers: usize,
}

#[cfg(target_os = "macos")]
fn raise_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}
#[cfg(not(target_os = "macos"))]
fn raise_qos() {}

/// Claims and runs blocks of the current region, if any. Returns false if no region was published.
#[inline]
fn participate(shared: &Shared) -> bool {
    shared.active.fetch_add(1, SeqCst);
    let rp = shared.region.load(SeqCst);
    if rp.is_null() {
        shared.active.fetch_sub(1, SeqCst);
        return false;
    }
    // SAFETY: the publisher keeps the region alive until `active` drops to zero after
    // it has cleared the pointer, and we incremented `active` before loading it.
    let region = unsafe { &*rp };
    loop {
        let i = region.next.fetch_add(1, Relaxed);
        if i >= region.n {
            break;
        }
        (region.f)(i);
    }
    shared.active.fetch_sub(1, SeqCst);
    true
}

fn worker(shared: Arc<Shared>) {
    raise_qos();
    let mut seen = 0u64;
    loop {
        // Wait for a new generation: spin, then yield, then park with growing timeouts.
        let mut spins = 0u32;
        let mut t0: Option<Instant> = None;
        let mut park = Duration::from_micros(100);
        loop {
            if shared.exit.load(Acquire) {
                return;
            }
            let g = shared.generation.load(Acquire);
            if g != seen {
                seen = g;
                break;
            }
            spins += 1;
            if spins < 4096 {
                std::hint::spin_loop();
                continue;
            }
            let now = Instant::now();
            let start = *t0.get_or_insert(now);
            if now.duration_since(start) < Duration::from_millis(3) {
                thread::yield_now();
                continue;
            }
            shared.sleepers.fetch_add(1, SeqCst);
            if shared.generation.load(SeqCst) == seen && !shared.exit.load(SeqCst) {
                thread::park_timeout(park);
                park = (park * 2).min(Duration::from_millis(10));
            }
            shared.sleepers.fetch_sub(1, SeqCst);
        }
        participate(&shared);
    }
}

impl Pool {
    /// A pool using `threads` threads in total (the caller counts as one).
    pub fn new(threads: usize) -> Self {
        raise_qos();
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            region: AtomicPtr::new(std::ptr::null_mut()),
            active: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            exit: AtomicBool::new(false),
        });
        // Spawning can fail where the platform has no threads (wasm32 outside wasi-threads, where
        // `spawn` returns Unsupported); whatever was spawned is the pool, the rest runs inline.
        let mut handles: Vec<thread::Thread> = Vec::with_capacity(threads.max(1) - 1);
        for i in 0..threads.max(1) - 1 {
            let s = shared.clone();
            match thread::Builder::new().name(format!("gemmars-{i}")).spawn(move || worker(s)) {
                Ok(h) => handles.push(h.thread().clone()),
                Err(_) => break,
            }
        }
        let workers = handles.len();
        Self { shared, handles, workers }
    }

    pub fn threads(&self) -> usize {
        self.workers + 1
    }

    /// Runs `f(i)` for every `i in 0..n` across the pool, returning when all are done.
    /// Must not be called from inside a job.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if n == 1 || self.workers == 0 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let s = &self.shared;
        // SAFETY: the region outlives every access: the pointer is cleared and `active`
        // observed zero before this function returns.
        let region = Region { f: unsafe { std::mem::transmute::<JobFn<'_>, JobFn<'static>>(f) }, n, next: AtomicUsize::new(0) };
        s.region.store(&region as *const Region as *mut Region, SeqCst);
        s.generation.fetch_add(1, SeqCst);
        if s.sleepers.load(SeqCst) > 0 {
            for h in &self.handles {
                h.unpark();
            }
        }
        loop {
            let i = region.next.fetch_add(1, Relaxed);
            if i >= n {
                break;
            }
            f(i);
        }
        s.region.store(std::ptr::null_mut(), SeqCst);
        while s.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }

    pub fn shutdown(&self) {
        self.shared.exit.store(true, SeqCst);
        for h in &self.handles {
            h.unpark();
        }
    }
}

static GLOBAL: AtomicPtr<Pool> = AtomicPtr::new(std::ptr::null_mut());

/// Installs a global pool with `threads` threads (replacing any previous one).
pub fn init(threads: usize) {
    let p = Box::into_raw(Box::new(Pool::new(threads)));
    let old = GLOBAL.swap(p, SeqCst);
    if !old.is_null() {
        // Leaked on purpose: in-flight references may still exist.
        unsafe { (*old).shutdown() };
    }
}

pub fn get() -> &'static Pool {
    let p = GLOBAL.load(Acquire);
    if p.is_null() {
        init(thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        return get();
    }
    unsafe { &*p }
}

#[inline]
pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
    get().run(n, f)
}

pub fn threads() -> usize {
    get().threads()
}
