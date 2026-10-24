//! Stable Diffusion with a LoRA through the library:
//! `cargo run --release --example sd_lora -- SD_DIR LORA[:SCALE] PROMPT OUT.png`

use oxinfer::sd::lora::LoraSpec;
use oxinfer::sd::{Params, Pipeline};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [model, lora, prompt, out] = a.as_slice() else {
        anyhow::bail!("usage: sd_lora SD_DIR LORA[:SCALE] PROMPT OUT.png");
    };
    let mut spec = LoraSpec::parse(lora)?;
    spec.path = oxinfer::hf::resolve_file(&spec.path.to_string_lossy(), None)?;
    let pipe = Pipeline::load_with_loras(&oxinfer::hf::resolve(model, None)?, &[spec], &mut |l| eprintln!("{l}"))?;
    let size = pipe.default_size();
    let params = Params {
        prompt: prompt.clone(),
        negative: String::new(),
        steps: 4,
        guidance: 1.0,
        seed: 1,
        width: size,
        height: size,
        sampler: oxinfer::sd::sched::Sampler::Euler,
    };
    let (img, _) = pipe.generate(&params, &mut |i, n| eprint!("\r{i}/{n}"))?;
    std::fs::write(out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
    Ok(())
}
