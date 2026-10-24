//! A chat through the high-level `engine` API, with streaming and stats (an empty line quits):
//! `cargo run --release --example chat -- MODEL [key=value ...]`
//! (MODEL is a directory, a .gguf file or a repo id; options are `EngineOptions` keys, plus
//! `temp=F` and `think=1`).

use oxinfer::engine::{Engine, EngineOptions, Flow, SamplingParams};
use std::io::{BufRead, Write};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: chat MODEL [key=value ...]");
    let mut opts = EngineOptions::default();
    let mut sampling = SamplingParams::default();
    let mut think = false;
    for kv in args {
        let (k, v) = kv.split_once('=').expect("options are key=value");
        match k {
            "temp" => sampling.temperature = v.parse()?,
            "think" => think = v == "1",
            _ => opts.set(k, v)?,
        }
    }
    let mut engine = Engine::load(&model, &opts)?;
    engine.set_sampling(sampling);
    engine.config_mut().think = think;
    let i = engine.info();
    eprintln!(
        "[{} ({:?}, {}, {:?} weights, {:?} speculative) on {} with {} threads, loaded in {:.2}s]",
        i.model_type, i.family, i.format, i.quant, i.speculative, i.device, i.threads, i.load_secs
    );
    let stdin = std::io::stdin();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        let reply = engine.send(line.trim_end(), &mut |d| {
            // Thinking dimmed, as the CLI shows it.
            let mut out = std::io::stdout().lock();
            let _ = if d.thinking {
                write!(out, "\x1b[2m{}\x1b[0m", d.text)
            } else {
                write!(out, "{}", d.text)
            };
            let _ = out.flush();
            Flow::Continue
        })?;
        let s = &reply.stats;
        println!();
        eprintln!(
            "[{:?} | prefill {} tok {:.1} tok/s | decode {} tok {:.1} tok/s | {:.2} tok/forward, {:.0}% drafts accepted | context {}]",
            reply.finish,
            s.prompt_tokens,
            s.prefill_tok_s(),
            s.generated_tokens,
            s.decode_tok_s(),
            s.tokens_per_forward(),
            100.0 * s.acceptance(),
            s.context_tokens
        );
    }
    Ok(())
}
