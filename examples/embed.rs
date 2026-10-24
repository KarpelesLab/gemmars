//! Embeddings through the library: `cargo run --release --example embed -- MODEL TEXT...`
//! (MODEL is a local directory or a repo id such as sentence-transformers/all-MiniLM-L6-v2).

use oxinfer::embed::{Embedder, cosine};
use oxinfer::model::Quant;
use oxinfer::safetensors::ModelFiles;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let spec = args.next().expect("usage: embed MODEL TEXT...");
    let texts: Vec<String> = args.collect();
    oxinfer::hf::set_hub(oxinfer::hf::Hub::Auto);
    let dir = oxinfer::hf::resolve(&spec, None)?;
    let files = ModelFiles::open(&dir, false)?;
    let emb = Embedder::load(&dir, &files, Quant::Bf16)?;
    let vecs = texts
        .iter()
        .map(|t| emb.embed(t, None).map(|e| e.dense))
        .collect::<anyhow::Result<Vec<_>>>()?;
    for a in &vecs {
        println!(
            "{}",
            vecs.iter().map(|b| format!("{:7.4}", cosine(a, b))).collect::<Vec<_>>().join(" ")
        );
    }
    Ok(())
}
