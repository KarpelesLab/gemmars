mod chat;
mod config;
mod drafter;
mod kernels;
mod model;
mod pool;
mod prof;
mod quant;
mod safetensors;
mod sampler;

use anyhow::{bail, Context, Result};
use model::{Model, Quant, Session};
use sampler::argmax;
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
    quant: Quant,
    profile: bool,
    dump_logits: Option<PathBuf>,
    show_tokens: bool,
    prefill_chunk: usize,
    draft: Option<PathBuf>,
    draft_tokens: usize,
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
         \x20 --quant bf16|q8|q4   weight precision for streamed matmuls (default bf16)\n\
         \x20 --preload            with --mmap, touch all weight pages before starting\n\
         \x20 --profile            print per-operation timing breakdown\n\
         \x20 --prefill-chunk N    tokens per prefill chunk (default 256)\n\
         \x20 --draft DIR          speculative decoding with a Gemma 4 assistant (drafter) model; greedy only\n\
         \x20 --draft-tokens N     max draft tokens per round (default 6; adaptive from 3)\n\
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
        quant: Quant::Bf16,
        profile: false,
        dump_logits: None,
        show_tokens: false,
        prefill_chunk: 256,
        draft: None,
        draft_tokens: 6,
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
            "--quant" => {
                a.quant = match val()?.as_str() {
                    "bf16" => Quant::Bf16,
                    "q8" | "int8" => Quant::Q8,
                    "q4" | "q4_0" => Quant::Q4,
                    other => bail!("unknown quantization {other} (use bf16, q8 or q4)"),
                }
            }
            "--profile" => a.profile = true,
            "--prefill-chunk" => a.prefill_chunk = val()?.parse()?,
            "--draft" => a.draft = Some(PathBuf::from(val()?)),
            "--draft-tokens" => a.draft_tokens = val()?.parse()?,
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

/// Incremental stdout printer: prints the stable decoded prefix of the token list.
struct Printer {
    printed: usize,
}

impl Printer {
    const SOC: u32 = 100; // <|channel>  (start of thinking)
    const EOC: u32 = 101; // <channel|>  (end of thinking)

    fn new() -> Self {
        Self { printed: 0 }
    }

    fn on_token(&mut self, id: u32) {
        if id == Self::SOC || id == Self::EOC {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_all(if id == Self::SOC { b"\x1b[2m" } else { b"\x1b[0m\n" });
            let _ = lock.flush();
        }
    }

    fn flush_text(&mut self, tok: &Tokenizer, out: &[u32], final_: bool) {
        if let Ok(text) = tok.decode(out, true) {
            let stable = if !final_ && text.ends_with('\u{FFFD}') { text.len().saturating_sub(3) } else { text.len() };
            if stable > self.printed && text.is_char_boundary(stable) {
                let mut lock = std::io::stdout().lock();
                let _ = lock.write_all(text[self.printed..stable].as_bytes());
                let _ = lock.flush();
                self.printed = stable;
            }
        }
    }
}

struct SpecStats {
    forwards: usize,
    drafted: usize,
    accepted: usize,
    /// Histogram of accepted-draft counts per round.
    per_pos: [usize; 16],
}

/// Greedy speculative decoding with the Gemma 4 drafter. Produces exactly the plain greedy
/// sequence. `hidden_last` is the final-normed hidden state of the last prompt token.
#[allow(clippy::too_many_arguments)]
fn generate_speculative(
    model: &Model,
    drafter: &drafter::Drafter,
    sess: &mut Session,
    tok: &Tokenizer,
    hidden_last: Vec<f32>,
    k: usize,
    eos: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, SpecStats, f64) {
    let t0 = Instant::now();
    let h = model.cfg.hidden_size;
    let (ls, lg) = model.cfg.shared_kv_layers();
    let mut stats = SpecStats { forwards: 0, drafted: 0, accepted: 0, per_pos: [0; 16] };
    let mut out: Vec<u32> = Vec::new();
    let mut printer = Printer::new();

    // First token from the prompt's logits (the prefill forward already counted by the caller).
    let logits = model.logits(&hidden_last, 1);
    let mut last = argmax(&logits);
    let mut h_prev = hidden_last; // hidden state of the token that produced `last`
    out.push(last);
    printer.on_token(last);
    printer.flush_text(tok, &out, false);

    // Adaptive draft length: grow after a fully accepted round, shrink after a fully rejected one.
    let (k_min, k_max) = (2usize.min(k.max(1)), k.max(1));
    let mut k_cur = k_min.max(3).min(k_max);
    'outer: while !eos.contains(&last) && out.len() < max_tokens {
        // Draft k tokens from the drafter, chaining its projected hidden state.
        let pos = sess.pos; // index of `last` (not yet in the cache)
        let k = k_cur.min(max_tokens - out.len());
        let mut drafts = Vec::with_capacity(k);
        let mut hid = h_prev.clone();
        let mut cur = last;
        for _ in 0..k.max(1) {
            let _t = prof::start(prof::Cat::Other);
            let emb = model.embed_scaled(cur);
            let (d, nh) = drafter.step(&emb, &hid, pos, sess.cache(ls), sess.cache(lg));
            drafts.push(d);
            hid = nh;
            cur = d;
        }
        stats.drafted += drafts.len();
        if stats.forwards == 0 && std::env::var("GEMMARS_DEBUG_DRAFT").is_ok() {
            eprintln!("[debug: pos {pos} last {last} drafts {drafts:?}]");
        }

        // Verify: one main forward over [last, drafts...].
        let mut seq = Vec::with_capacity(drafts.len() + 1);
        seq.push(last);
        seq.extend_from_slice(&drafts);
        let check = std::env::var("GEMMARS_CHECK_SPEC").is_ok();
        let sess_before = if check { Some(sess.clone()) } else { None };
        let hs = model.forward_hidden(sess, &seq);
        let logits = model.logits(&hs, seq.len());
        stats.forwards += 1;
        let v = model.cfg.vocab_size;
        if let Some(mut s2) = sess_before {
            // Debug: recompute every position through single-token forwards and compare argmaxes.
            for (i, &tk) in seq.iter().enumerate() {
                let l1 = model.forward(&mut s2, &[tk]);
                let lk = &logits[i * v..(i + 1) * v];
                let (a1, ak) = (argmax(&l1), argmax(lk));
                if a1 != ak {
                    eprintln!(
                        "[check: pos {} token {}: t=1 argmax {} ({:.4}) vs t={} argmax {} ({:.4}); cross: t1[{}]={:.4} tk[{}]={:.4}]",
                        pos + i, tk, a1, l1[a1 as usize], seq.len(), ak, lk[ak as usize], ak, l1[ak as usize], a1, lk[a1 as usize]
                    );
                }
            }
        }
        let mut n = 0;
        while n < drafts.len() && argmax(&logits[n * v..(n + 1) * v]) == drafts[n] {
            n += 1;
        }
        stats.accepted += n;
        if n < stats.per_pos.len() {
            stats.per_pos[n] += 1;
        }
        if n == drafts.len() {
            k_cur = (k_cur + 1).min(k_max);
        } else if n == 0 {
            k_cur = (k_cur - 1).max(k_min);
        }
        if stats.forwards == 1 && std::env::var("GEMMARS_DEBUG_DRAFT").is_ok() {
            let truth: Vec<u32> = (0..seq.len()).map(|i| argmax(&logits[i * v..(i + 1) * v])).collect();
            eprintln!("[debug: main argmax after each of {seq:?} = {truth:?}; accepted {n}]");
        }
        for &d in &drafts[..n] {
            if out.len() >= max_tokens {
                sess.truncate(pos + n + 1);
                break 'outer;
            }
            out.push(d);
            printer.on_token(d);
            if eos.contains(&d) {
                sess.truncate(pos + n);
                break 'outer;
            }
        }
        let bonus = argmax(&logits[n * v..(n + 1) * v]);
        // Keep only [last, drafts[..n]] in the cache; `bonus` is processed by the next round.
        sess.truncate(pos + n + 1);
        h_prev = hs[n * h..(n + 1) * h].to_vec();
        last = bonus;
        if out.len() >= max_tokens {
            break;
        }
        out.push(bonus);
        printer.on_token(bonus);
        printer.flush_text(tok, &out, false);
    }
    if let Some(&e) = out.last() {
        if eos.contains(&e) {
            out.pop();
            printer.flush_text(tok, &out, true);
            out.push(e);
        } else {
            printer.flush_text(tok, &out, true);
        }
    }
    println!();
    (out, stats, t0.elapsed().as_secs_f64())
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
    let model = Model::load(cfg, &files, args.quant)?;
    let tok = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    eprintln!(
        "[loaded {} layers, hidden {}, {:.1} GB checkpoint ({}), {:?} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
        model.cfg.num_hidden_layers,
        model.cfg.hidden_size,
        files.total_bytes() as f64 / 1e9,
        if args.mmap { "mmap" } else { "in RAM" },
        model.quant,
        model.streamed_bytes() as f64 / 1e9,
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

    let draft_files = match &args.draft {
        Some(d) => Some(safetensors::ModelFiles::open(d, true)?),
        None => None,
    };
    let drafter = match (&args.draft, &draft_files) {
        (Some(d), Some(f)) => {
            let dr = drafter::Drafter::load(d, f)?;
            if dr.backbone_hidden != model.cfg.hidden_size {
                bail!("drafter backbone size {} does not match model hidden size {}", dr.backbone_hidden, model.cfg.hidden_size);
            }
            if args.temperature > 0.0 {
                eprintln!("[note: speculative decoding is greedy; --temp ignored]");
            }
            eprintln!("[drafter loaded: {} layers, hidden {}, {} draft tokens/round]", dr.cfg.num_hidden_layers, dr.cfg.hidden_size, args.draft_tokens);
            Some(dr)
        }
        _ => None,
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
            if let Some(dr) = &drafter {
                let mut hs = Vec::new();
                for c in ids.chunks(args.prefill_chunk.max(1)) {
                    hs = model.forward_hidden(&mut sess, c);
                }
                let h = model.cfg.hidden_size;
                let hidden_last = hs[hs.len() - h..].to_vec();
                let prefill_secs = t.elapsed().as_secs_f64();
                let (out, st, secs) =
                    generate_speculative(&model, dr, &mut sess, &tok, hidden_last, args.draft_tokens, &eos, args.max_tokens);
                if args.show_tokens {
                    eprintln!("[output ids: {out:?}]");
                }
                report(ids.len(), prefill_secs, out.len(), secs);
                eprintln!("[speculative: {:.2} tok/forward, drafts accepted {}%]", out.len() as f64 / st.forwards.max(1) as f64, 100 * st.accepted / st.drafted.max(1));
                continue;
            }
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

    if let Some(dr) = &drafter {
        let t = Instant::now();
        let mut hs = Vec::new();
        for c in ids.chunks(args.prefill_chunk.max(1)) {
            hs = model.forward_hidden(&mut sess, c);
        }
        let h = model.cfg.hidden_size;
        let hidden_last = hs[hs.len() - h..].to_vec();
        let prefill_secs = t.elapsed().as_secs_f64();
        prof::reset();
        let (out, st, secs) = generate_speculative(&model, dr, &mut sess, &tok, hidden_last, args.draft_tokens, &eos, args.max_tokens);
        if args.show_tokens {
            eprintln!("[output ids: {out:?}]");
        }
        let n = out.len();
        report(ids.len(), prefill_secs, n, secs);
        eprintln!(
            "[speculative: {} main forwards for {} tokens = {:.2} tok/forward; drafts accepted {}/{} = {:.0}%]",
            st.forwards,
            n,
            n as f64 / st.forwards.max(1) as f64,
            st.accepted,
            st.drafted,
            100.0 * st.accepted as f64 / st.drafted.max(1) as f64
        );
        eprintln!("[accepted-per-round histogram: {:?}]", &st.per_pos[..=args.draft_tokens.min(15)]);
        if args.profile {
            eprintln!("-- decode --");
            prof::report(secs, st.forwards);
        }
        return Ok(());
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
