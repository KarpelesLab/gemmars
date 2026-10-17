mod chat;
mod config;
mod kernels;
mod model;
mod pool;
mod prof;
mod safetensors;
mod sampler;

use anyhow::{bail, Context, Result};
use model::{Model, Session};
use sampler::Sampler;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

struct Args {
    model: PathBuf,
    prompt: Option<String>,
    system: Option<String>,
    think: bool,
    raw: bool,
    chat: bool,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    threads: Option<usize>,
    preload: bool,
    mmap: bool,
    profile: bool,
    dump_logits: Option<PathBuf>,
    show_tokens: bool,
    prefill_chunk: usize,
}

fn usage() -> ! {
    eprintln!(
        "usage: gemmars --model DIR [options]\n\
         \n\
         options:\n\
         \x20 --prompt TEXT        single prompt (default: read from stdin)\n\
         \x20 --chat               interactive multi-turn chat\n\
         \x20 --system TEXT        system prompt\n\
         \x20 --think              enable thinking mode\n\
         \x20 --raw                no chat template (prompt is fed as-is, with <bos>)\n\
         \x20 --max-tokens N       max new tokens (default 1024)\n\
         \x20 --temp F             temperature, 0 = greedy (default 0)\n\
         \x20 --top-k N            top-k (default 64)\n\
         \x20 --top-p F            top-p (default 0.95)\n\
         \x20 --seed N             RNG seed\n\
         \x20 --threads N          worker threads (default: all cores)\n\
         \x20 --mmap               map weights from disk instead of copying into RAM\n\
         \x20 --preload            with --mmap, touch all weight pages before starting\n\
         \x20 --profile            print per-operation timing breakdown\n\
         \x20 --prefill-chunk N    tokens per prefill chunk (default 256)\n\
         \x20 --dump-logits FILE   write last-prompt-token logits (f32 LE) and exit\n\
         \x20 --show-tokens        print token ids"
    );
    std::process::exit(2)
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::new(),
        prompt: None,
        system: None,
        think: false,
        raw: false,
        chat: false,
        max_tokens: 1024,
        temperature: 0.0,
        top_k: 64,
        top_p: 0.95,
        seed: 42,
        threads: None,
        preload: false,
        mmap: false,
        profile: false,
        dump_logits: None,
        show_tokens: false,
        prefill_chunk: 256,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().with_context(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" | "-m" => a.model = PathBuf::from(val()?),
            "--prompt" | "-p" => a.prompt = Some(val()?),
            "--system" => a.system = Some(val()?),
            "--think" => a.think = true,
            "--raw" => a.raw = true,
            "--chat" => a.chat = true,
            "--max-tokens" | "-n" => a.max_tokens = val()?.parse()?,
            "--temp" | "-t" => a.temperature = val()?.parse()?,
            "--top-k" => a.top_k = val()?.parse()?,
            "--top-p" => a.top_p = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--threads" | "-j" => a.threads = Some(val()?.parse()?),
            "--preload" => a.preload = true,
            "--mmap" => a.mmap = true,
            "--profile" => a.profile = true,
            "--prefill-chunk" => a.prefill_chunk = val()?.parse()?,
            "--dump-logits" => a.dump_logits = Some(PathBuf::from(val()?)),
            "--show-tokens" => a.show_tokens = true,
            "-h" | "--help" => usage(),
            _ => {
                eprintln!("unknown argument {k}");
                usage()
            }
        }
    }
    if a.model.as_os_str().is_empty() {
        usage();
    }
    Ok(a)
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn prefill(model: &Model, sess: &mut Session, ids: &[u32], chunk: usize) -> Vec<f32> {
    let mut logits = Vec::new();
    for c in ids.chunks(chunk.max(1)) {
        logits = model.forward(sess, c);
    }
    logits
}

/// Streams the model's reply to stdout. Returns (generated ids including stop token, forwards, seconds).
fn generate(
    model: &Model,
    sess: &mut Session,
    tok: &Tokenizer,
    mut logits: Vec<f32>,
    sampler: &mut Sampler,
    eos: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, usize, f64) {
    let t0 = Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let stdout = std::io::stdout();
    let mut n_forward = 0usize;
    const SOC: u32 = 100; // <|channel>  (start of thinking)
    const EOC: u32 = 101; // <channel|>  (end of thinking)
    for _ in 0..max_tokens {
        let id = sampler.sample(&logits);
        out.push(id);
        if eos.contains(&id) {
            break;
        }
        if id == SOC || id == EOC {
            // Thinking trace is shown dimmed; the special tokens themselves are not decoded.
            let mut lock = stdout.lock();
            let _ = lock.write_all(if id == SOC { b"\x1b[2m" } else { b"\x1b[0m\n" });
            let _ = lock.flush();
        }
        // Incremental decode: print the stable prefix of the decoded text.
        if let Ok(text) = tok.decode(&out, true) {
            let stable = if text.ends_with('\u{FFFD}') { text.len().saturating_sub(3) } else { text.len() };
            if stable > printed && text.is_char_boundary(stable) {
                let mut lock = stdout.lock();
                let _ = lock.write_all(text[printed..stable].as_bytes());
                let _ = lock.flush();
                printed = stable;
            }
        }
        logits = model.forward(sess, &[id]);
        n_forward += 1;
    }
    if let Ok(text) = tok.decode(&out, true) {
        if text.len() > printed {
            print!("{}", &text[printed..]);
        }
    }
    println!();
    (out, n_forward, t0.elapsed().as_secs_f64())
}

fn report(prefill_tokens: usize, prefill_secs: f64, decode_tokens: usize, decode_secs: f64) {
    eprintln!(
        "[prefill {} tok in {:.2}s = {:.1} tok/s | decode {} tok in {:.2}s = {:.1} tok/s]",
        prefill_tokens,
        prefill_secs,
        prefill_tokens as f64 / prefill_secs.max(1e-9),
        decode_tokens,
        decode_secs,
        decode_tokens as f64 / decode_secs.max(1e-9),
    );
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let threads = args.threads.unwrap_or_else(default_threads);
    pool::init(threads);
    run(args)
}

fn run(args: Args) -> Result<()> {
    let t0 = Instant::now();
    let cfg = config::TextConfig::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let model = Model::load(cfg, &files)?;
    let tok = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    eprintln!(
        "[loaded {} layers, hidden {}, {:.1} GB of weights ({}), {} threads, {:.2}s]",
        model.cfg.num_hidden_layers,
        model.cfg.hidden_size,
        files.total_bytes() as f64 / 1e9,
        if args.mmap { "mmap" } else { "in RAM" },
        pool::threads(),
        t0.elapsed().as_secs_f64()
    );
    if args.preload && args.mmap {
        let t = Instant::now();
        files.preload();
        eprintln!("[preloaded weights in {:.2}s]", t.elapsed().as_secs_f64());
    }

    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };

    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let mut sess = model.new_session();

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
            let mut text = String::new();
            if first {
                text.push_str(&chat::conversation_start(args.system.as_deref(), args.think));
                first = false;
            } else {
                // The previous stop token was sampled but never fed to the model.
                text.push_str(&chat::end_of_model_turn());
            }
            text.push_str(&chat::user_turn(line));
            let ids = encode(&text)?;
            if args.show_tokens {
                eprintln!("[prompt ids: {ids:?}]");
            }
            let t = Instant::now();
            let logits = prefill(&model, &mut sess, &ids, args.prefill_chunk);
            let prefill_secs = t.elapsed().as_secs_f64();
            prof::reset();
            let (out, n, decode_secs) = generate(&model, &mut sess, &tok, logits, &mut sampler, &eos, args.max_tokens);
            if args.show_tokens {
                eprintln!("[output ids: {out:?}]");
            }
            report(ids.len(), prefill_secs, n, decode_secs);
            if args.profile {
                prof::report(decode_secs, n);
            }
        }
        return Ok(());
    }

    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut s = String::new();
            std::io::stdin().lock().read_to_string(&mut s)?;
            s
        }
    };
    let text = if args.raw {
        format!("<bos>{prompt}")
    } else {
        let mut t = chat::conversation_start(args.system.as_deref(), args.think);
        t.push_str(&chat::user_turn(&prompt));
        t
    };
    let ids = encode(&text)?;
    if args.show_tokens {
        eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
    }
    if ids.is_empty() {
        bail!("empty prompt");
    }

    let t = Instant::now();
    let logits = prefill(&model, &mut sess, &ids, args.prefill_chunk);
    let prefill_secs = t.elapsed().as_secs_f64();
    if args.profile {
        eprintln!("-- prefill --");
        prof::report(prefill_secs, ids.len());
    }

    if let Some(path) = &args.dump_logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        let top = {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            idx.truncate(10);
            idx.iter().map(|&i| format!("{i}:{:.3}", logits[i])).collect::<Vec<_>>().join(" ")
        };
        eprintln!("[wrote {} logits to {} | top10 {top}]", logits.len(), path.display());
        eprintln!("[prefill {} tok in {:.2}s]", ids.len(), prefill_secs);
        return Ok(());
    }

    prof::reset();
    let (out, n, decode_secs) = generate(&model, &mut sess, &tok, logits, &mut sampler, &eos, args.max_tokens);
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    report(ids.len(), prefill_secs, n, decode_secs);
    if args.profile {
        eprintln!("-- decode --");
        prof::report(decode_secs, n);
    }
    Ok(())
}
