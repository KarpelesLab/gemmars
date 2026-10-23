//! Emits `has_simd` when a hand-written SIMD kernel module exists for the target: NEON on
//! aarch64, AVX2/scalar-dispatch on x86-64, and simd128 on wasm32 when that target feature is
//! enabled (`-C target-feature=+simd128`). Everything else uses the scalar paths in `kernels.rs`.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_simd)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let simd = match arch.as_str() {
        "aarch64" | "x86_64" => true,
        "wasm32" => features.split(',').any(|f| f == "simd128"),
        _ => false,
    };
    if simd {
        println!("cargo:rustc-cfg=has_simd");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
