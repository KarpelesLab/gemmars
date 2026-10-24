//! Greedy chat completion with an ANEMLL Core ML model on the Apple Neural Engine, through the
//! library API (what `oxinfer --model <dir>` does, minus streaming and sampling).
//!
//! ```sh
//! cargo run --release --example anemll -- ~/.cache/oxinfer/hub/anemll--anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4 "Why is the sky blue?"
//! ```

#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
fn main() -> anyhow::Result<()> {
    use oxinfer::anemll::{AnemllModel, Meta, Options};
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("usage: anemll MODEL_DIR [PROMPT]"));
    let prompt = args.next().unwrap_or_else(|| "Write a haiku about the sea.".into());

    let meta = Meta::load(&dir)?;
    let tok = meta.tokenizer()?;
    let fmt = meta.chat_format()?;
    let stop = meta.stop_tokens(&tok, fmt);
    let mut model = AnemllModel::load(&dir, Options::default())?;

    let text = fmt.first_turn(None, false, &[], &prompt);
    let ids = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let mut next = model.prefill(&ids)?;
    let mut out = Vec::new();
    while out.len() < 256 && model.remaining() > 0 {
        let id = next.argmax();
        if stop.contains(&id) {
            break;
        }
        out.push(id);
        next = model.step(id)?;
    }
    println!("{}", tok.decode(&out, true).map_err(|e| anyhow::anyhow!("{e}"))?);
    Ok(())
}

#[cfg(not(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64")))]
fn main() {
    eprintln!("ANEMLL Core ML models need Apple Silicon macOS / iOS and the `gpu` feature");
}
