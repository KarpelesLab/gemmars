//! Monotonic time. `std::time::Instant` is unavailable on `wasm32-unknown-unknown` (it panics), so
//! there the host supplies a millisecond clock through the `host_now_ms` import.

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use std::time::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use web::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod web {
    use std::time::Duration;

    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        fn host_now_ms() -> f64;
    }

    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        pub fn now() -> Self {
            Instant(unsafe { host_now_ms() })
        }
        pub fn elapsed(&self) -> Duration {
            Self::now().duration_since(*self)
        }
        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_secs_f64(((self.0 - earlier.0) / 1e3).max(0.0))
        }
    }
}
