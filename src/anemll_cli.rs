//! `oxinfer --model <ANEMLL model>`: generation with an ANEMLL Core ML model on the Apple Neural
//! Engine (see `oxinfer::anemll`), with the same prompt formatting, sampling and streaming as the
//! native paths.

use crate::{Args, Printer, report, system_prompt};
use anyhow::{Context, Result, bail};
use oxinfer::anemll::{AnemllModel, Meta, Next, Options, PrefillMode};
use oxinfer::chat::{self, ChatFormat};
use oxinfer::coreml::ComputeUnits;
use oxinfer::sampler::Sampler;
use std::io::{BufRead, Read};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

/// The chat format matching the model's template (`tokenizer_config.json`).
fn chat_format(meta: &Meta) -> Result<&'static dyn ChatFormat> {
    let cfg: serde_json::Value = std::fs::read(meta.tokenizer_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let jinja = std::fs::read_to_string(meta.tokenizer_dir.join("chat_template.jinja")).ok();
    let tpl = jinja.as_deref().or(cfg["chat_template"].as_str()).unwrap_or("");
    let arch = format!("{} {}", meta.architecture, meta.model_prefix).to_lowercase();
    Ok(if tpl.contains("<|start_header_id|>") {
        &chat::Llama3Chat
    } else if tpl.contains("<start_of_turn>") || arch.contains("gemma3") {
        &chat::Gemma3Chat
    } else if tpl.contains("You are Qwen, created by Alibaba Cloud") || arch.contains("qwen2") {
        &chat::Qwen2Chat
    } else if tpl.contains("<|im_start|>") || arch.contains("qwen") {
        chat::QwenChat::for_template(Some(tpl))
    } else {
        bail!("no chat format for this model ({}); use --raw", meta.architecture)
    })
}

/// Stop tokens as ANEMLL's `chat.py` collects them: the tokenizer's EOS token, then
/// `<|endoftext|>`, `<end_of_turn>` and `<|eot_id|>` when the vocabulary has them, plus the chat
/// format's end-of-turn token.
fn stop_ids(meta: &Meta, tok: &Tokenizer, fmt: &dyn ChatFormat) -> Vec<u32> {
    let cfg: serde_json::Value = std::fs::read(meta.tokenizer_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let eos = &cfg["eos_token"];
    let mut ids = Vec::new();
    let names = [
        eos.as_str().or(eos["content"].as_str()).unwrap_or(""),
        "<|endoftext|>",
        "<end_of_turn>",
        "<|eot_id|>",
    ];
    for n in names {
        if let Some(id) = tok.token_to_id(n).filter(|_| !n.is_empty())
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    }
    chat::with_turn_end(fmt, tok, ids)
}

fn units(args: &Args) -> Result<ComputeUnits> {
    if let Some(u) = &args.coreml_units {
        return ComputeUnits::parse(u).with_context(|| format!("unknown --coreml-units {u} (ane|all|gpu|cpu)"));
    }
    Ok(match args.device {
        oxinfer::platform::DevicePref::Cpu => ComputeUnits::CpuOnly,
        oxinfer::platform::DevicePref::Gpu(_) => ComputeUnits::CpuAndGpu,
        oxinfer::platform::DevicePref::Auto => ComputeUnits::CpuAndNeuralEngine,
    })
}

/// Streams the reply. Returns (generated ids including the stop token, steps, seconds).
#[allow(clippy::too_many_arguments)]
fn generate(
    m: &mut AnemllModel,
    tok: &Tokenizer,
    mut next: Next,
    sampler: &mut Sampler,
    eos: &[u32],
    max_tokens: usize,
    fmt: &dyn ChatFormat,
    think: bool,
) -> Result<(Vec<u32>, usize, f64)> {
    let t0 = Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut printer = Printer::new(fmt, think, tok);
    let mut steps = 0usize;
    // Benchmarks and reference comparisons generate a fixed number of tokens.
    let eos: &[u32] = if std::env::var_os("OXINFER_IGNORE_EOS").is_some() {
        &[]
    } else {
        eos
    };
    while out.len() < max_tokens {
        let id = match &next {
            Next::Logits(l) => sampler.sample(&l[..l.len().min(tok.get_vocab_size(true).max(1))]),
            Next::Argmax { token, .. } => *token,
        };
        out.push(id);
        if eos.contains(&id) {
            break;
        }
        printer.on_token(id);
        printer.flush_text(tok, &out, false);
        if out.len() >= max_tokens {
            break;
        }
        if m.remaining() == 0 {
            printer.flush_text(tok, &out, true);
            eprintln!("\n[context full: the Core ML model holds {} tokens]", m.context_length());
            return Ok((out, steps, t0.elapsed().as_secs_f64()));
        }
        next = m.step(id)?;
        steps += 1;
    }
    let shown = if out.last().is_some_and(|e| eos.contains(e)) {
        &out[..out.len() - 1]
    } else {
        &out[..]
    };
    printer.flush_text(tok, shown, true);
    println!();
    Ok((out, steps, t0.elapsed().as_secs_f64()))
}

/// Whether `dir` holds an ANEMLL model.
pub fn detect(dir: &Path) -> bool {
    oxinfer::anemll::is_anemll(dir)
}

pub fn run(args: &Args) -> Result<()> {
    let meta = Meta::load(&args.model)?;
    let tok = Tokenizer::from_file(meta.tokenizer_dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let fmt = chat_format(&meta)?;
    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };
    let first_prompt = |prompt: &str| {
        if args.raw {
            prompt.to_string()
        } else {
            fmt.first_turn(system_prompt(args).as_deref(), args.think, &[], prompt)
        }
    };
    let prompt = if args.chat {
        None
    } else {
        Some(match &args.prompt {
            Some(p) => p.clone(),
            None => {
                let mut s = String::new();
                std::io::stdin().lock().read_to_string(&mut s)?;
                s
            }
        })
    };
    if args.print_prompt {
        print!("{}", first_prompt(prompt.as_deref().unwrap_or("")));
        return Ok(());
    }
    if args.tools {
        bail!("--tools is not supported with Core ML models");
    }
    let opts = Options {
        units: units(args)?,
        prefill: match args.coreml_prefill.as_deref() {
            None | Some("auto") => PrefillMode::Auto,
            Some("batch") => PrefillMode::Batched,
            Some("single") => PrefillMode::Single,
            Some(o) => bail!("unknown --coreml-prefill {o} (auto|batch|single)"),
        },
    };
    eprintln!(
        "[Core ML: {} ({}{}, context {}, batch {}), compute units {}]",
        if meta.name.is_empty() {
            args.model.display().to_string()
        } else {
            meta.name.clone()
        },
        if meta.monolithic {
            "monolithic".to_string()
        } else {
            format!("{} chunk(s)", meta.num_chunks)
        },
        if meta.argmax_in_model { ", argmax in model" } else { "" },
        meta.context_length,
        meta.batch_size,
        opts.units.as_str()
    );
    let mut m = AnemllModel::load(&args.model, opts)?;
    eprintln!(
        "[loaded in {:.1}s; {} prefill{}]",
        m.stats.load_secs,
        if m.options.prefill == PrefillMode::Batched {
            "batched"
        } else {
            "token-by-token (no batch-prefill flag in meta.yaml; --coreml-prefill batch to force)"
        },
        if m.rotation() { "; sliding-window rotation" } else { "" }
    );
    if args.profile {
        eprint!("{}", m.describe());
    }
    if !m.has_logits() {
        if args.dump_logits.is_some() {
            bail!("this model computes the argmax inside the network (argmax_in_model): no logits to dump");
        }
        if args.temperature > 0.0 {
            eprintln!("[argmax_in_model: the model returns only its top token, so generation is greedy]");
        }
    }
    let eos = stop_ids(&meta, &tok, fmt);
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);

    if args.chat {
        let mut first = true;
        let stdin = std::io::stdin();
        loop {
            eprint!("> ");
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim_end_matches(['\n', '\r']);
            if line.trim().is_empty() {
                continue;
            }
            let text = if first {
                first = false;
                first_prompt(line)
            } else {
                fmt.end_of_model_turn() + &fmt.user_turn(line, args.think)
            };
            let ids = encode(&text)?;
            if args.show_tokens {
                eprintln!("[prompt ids: {ids:?}]");
            }
            let t = Instant::now();
            let next = m.prefill(&ids)?;
            let prefill_secs = t.elapsed().as_secs_f64();
            let (out, n, secs) = generate(&mut m, &tok, next, &mut sampler, &eos, args.max_tokens, fmt, args.think)?;
            if args.show_tokens {
                eprintln!("[output ids: {out:?}]");
            }
            report(ids.len(), prefill_secs, n, secs);
            if m.remaining() == 0 {
                eprintln!("[context full; start a new session]");
                break;
            }
        }
        return Ok(());
    }

    let text = first_prompt(prompt.as_deref().unwrap_or(""));
    let ids = encode(&text)?;
    if args.show_tokens {
        eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
    }
    if ids.is_empty() {
        bail!("empty prompt");
    }
    if ids.len() > m.context_length() {
        bail!(
            "prompt is {} tokens but this Core ML model has a fixed context of {} tokens",
            ids.len(),
            m.context_length()
        );
    }
    let t = Instant::now();
    let next = m.prefill(&ids)?;
    let prefill_secs = t.elapsed().as_secs_f64();
    if let Some(path) = &args.dump_logits {
        let Next::Logits(logits) = &next else { unreachable!() };
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top = idx[..10]
            .iter()
            .map(|&i| format!("{i}:{:.3}", logits[i]))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("[wrote {} logits to {} | top10 {top}]", logits.len(), path.display());
        eprintln!("[prefill {} tok in {:.2}s]", ids.len(), prefill_secs);
        return Ok(());
    }
    let (out, n, secs) = generate(&mut m, &tok, next, &mut sampler, &eos, args.max_tokens, fmt, args.think)?;
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    report(ids.len(), prefill_secs, n, secs);
    if args.profile {
        eprintln!("[{} Core ML predictions]", m.stats.predictions);
    }
    Ok(())
}
