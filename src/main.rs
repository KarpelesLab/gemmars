use anyhow::{Context, Result, bail};
use oxinfer::chat::{self, ChatFormat};
#[cfg(feature = "hf")]
use oxinfer::hf;
#[cfg(feature = "media")]
use oxinfer::media;
use oxinfer::model::{Model, Quant, Session};
use oxinfer::sampler::{Sampler, argmax};
#[cfg(feature = "media")]
use oxinfer::{audio, mm_config, vision};
use oxinfer::{config, drafter, platform, pool, prof, qwen, safetensors};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

struct Args {
    model: PathBuf,
    prompt: Option<String>,
    system: Option<String>,
    no_system: bool,
    tools: bool,
    print_prompt: bool,
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
    prefill_chunk: Option<usize>,
    draft: Option<PathBuf>,
    draft_tokens: usize,
    images: Vec<PathBuf>,
    audios: Vec<PathBuf>,
    dump_mm: Option<PathBuf>,
    max_soft_tokens: usize,
    cache_dir: Option<PathBuf>,
    download_only: bool,
    save_quantized: Option<PathBuf>,
    // Image generation (diffusers folders).
    output: Option<PathBuf>,
    steps: Option<usize>,
    guidance: Option<f32>,
    negative: String,
    size: Option<(usize, usize)>,
    sampler: Option<String>,
    loras: Vec<oxinfer::sd::lora::LoraSpec>,
    device: platform::DevicePref,
    devices: bool,
    embed: bool,
    prompt_name: Option<String>,
    similarity: bool,
    sparse: bool,
    colbert: bool,
    dim: Option<usize>,
}

fn usage() -> ! {
    eprintln!(
        "usage: oxinfer --model DIR|org/name [options]\n\
         \n\
         A HuggingFace repo id is fetched into the cache (~/.cache/oxinfer/hub or $OXINFER_CACHE)\n\
         on first use; downloads resume. Set HF_TOKEN for gated repos.\n\
         \n\
         options:\n\
         \x20 --prompt TEXT        single prompt (default: read from stdin)\n\
         \x20 --chat               interactive multi-turn chat\n\
         \x20 --system TEXT        system prompt (default: a short one with the local date and time)\n\
         \x20 --no-system          no system prompt\n\
         \x20 --tools              let the model call built-in tools (date and time, calculator)\n\
         \x20 --print-prompt       print the formatted prompt and exit\n\
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
         \x20 --prefill-chunk N    tokens per prefill chunk (default 256 on CPU, 1024 with a GPU backend)\n\
         \x20 --draft DIR          speculative decoding with a Gemma 4 assistant (drafter) model; greedy only\n\
         \x20 --draft-tokens N     max draft tokens per round (default 6; adaptive from 3)\n\
         \x20 --image FILE         attach an image (repeatable; PNG/JPEG/WebP/GIF/BMP via oxideav)\n\
         \x20 --audio FILE         attach an audio clip (repeatable; WAV/MP3/FLAC/AAC/OGG via oxideav)\n\
         \x20 --max-soft-tokens N  image soft-token budget: 70, 140, 280 (default), 560 or 1120\n\
         \x20 --dump-mm PREFIX     write multimodal soft tokens / patches as f32 files (for reference checks)\n\
         \x20 --cache-dir DIR      where fetched models are cached\n\
         \x20 --download-only      fetch the model (and drafter) and exit\n\
         \x20 --save-quantized DIR write a text-only checkpoint with the weights already quantized to\n\
         \x20                      --quant (loads zero-copy with no quantization pass; for browsers, mmap)\n\
         \n\
         image generation (a diffusers folder: Stable Diffusion 1.x / 2.x, SD-Turbo):\n\
         \x20 -o, --output FILE     PNG to write (default image.png)\n\
         \x20 --steps N            denoising steps (default 1 for turbo models, else 25)\n\
         \x20 --guidance F         classifier-free guidance (default 0 for turbo, else 7.5)\n\
         \x20 --negative TEXT      negative prompt (with guidance > 1)\n\
         \x20 --size WxH           image size, multiples of 64 (default the model's)\n\
         \x20 --sampler NAME       euler or euler_a (default euler_a)\n\
         \x20 --seed N             noise seed (default 42)\n\
         \x20 --lora FILE[:SCALE]  merge a LoRA (kohya or diffusers safetensors; repeatable; scale 1);\n\
         \x20                      FILE may be org/repo/file.safetensors on Hugging Face\n\
         \x20 --devices            list detected CPU features and GPUs, then exit\n\
         \x20 --device D           auto (default) | cpu | cuda | metal | vulkan\n\
         \x20 --dump-logits FILE   write last-prompt-token logits (f32 LE) and exit\n\
         \n\
         embeddings (sentence-transformers models are detected; --embed forces it):\n\
         \x20 --embed              embed the prompt, or each stdin line: one JSON array per line\n\
         \x20 --prompt-name NAME   the model's prompt to prepend (e.g. query, document)\n\
         \x20 --dim N              truncate to N dimensions (Matryoshka) and renormalize\n\
         \x20 --similarity         print the cosine similarity matrix instead\n\
         \x20 --sparse, --colbert  BGE-M3 lexical weights / multi-vector embeddings instead\n\
         \x20 --show-tokens        print token ids"
    );
    std::process::exit(2)
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::new(),
        prompt: None,
        system: None,
        no_system: false,
        tools: false,
        print_prompt: false,
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
        prefill_chunk: None,
        draft: None,
        draft_tokens: 6,
        images: Vec::new(),
        audios: Vec::new(),
        dump_mm: None,
        max_soft_tokens: 0,
        cache_dir: None,
        download_only: false,
        save_quantized: None,
        output: None,
        steps: None,
        guidance: None,
        negative: String::new(),
        size: None,
        sampler: None,
        loras: Vec::new(),
        device: platform::DevicePref::Auto,
        devices: false,
        embed: false,
        prompt_name: None,
        similarity: false,
        sparse: false,
        colbert: false,
        dim: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().with_context(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" | "-m" => a.model = PathBuf::from(val()?),
            "--prompt" | "-p" => a.prompt = Some(val()?),
            "--system" => a.system = Some(val()?),
            "--no-system" => a.no_system = true,
            "--tools" => a.tools = true,
            "--print-prompt" => a.print_prompt = true,
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
            "--prefill-chunk" => a.prefill_chunk = Some(val()?.parse()?),
            "--draft" => a.draft = Some(PathBuf::from(val()?)),
            "--draft-tokens" => a.draft_tokens = val()?.parse()?,
            "--image" => a.images.push(PathBuf::from(val()?)),
            "--audio" => a.audios.push(PathBuf::from(val()?)),
            "--dump-mm" => a.dump_mm = Some(PathBuf::from(val()?)),
            "--max-soft-tokens" => a.max_soft_tokens = val()?.parse()?,
            "--cache-dir" => a.cache_dir = Some(PathBuf::from(val()?)),
            "--download-only" => a.download_only = true,
            "--save-quantized" => {
                a.save_quantized = Some(PathBuf::from(val()?));
                a.mmap = true;
            }
            "--devices" => a.devices = true,
            "--output" | "-o" => a.output = Some(PathBuf::from(val()?)),
            "--steps" => a.steps = Some(val()?.parse()?),
            "--guidance" | "--cfg" => a.guidance = Some(val()?.parse()?),
            "--negative" => a.negative = val()?,
            "--sampler" => a.sampler = Some(val()?),
            "--lora" => a.loras.push(oxinfer::sd::lora::LoraSpec::parse(&val()?)?),
            "--size" => {
                let v = val()?;
                let (w, h) = v
                    .split_once(['x', 'X'])
                    .with_context(|| format!("--size expects WIDTHxHEIGHT, got {v}"))?;
                a.size = Some((w.parse()?, h.parse()?));
            }
            "--device" => {
                let v = val()?;
                a.device = platform::DevicePref::parse(&v).with_context(|| format!("unknown device {v} (auto|cpu|cuda|metal|vulkan)"))?;
            }
            "--dump-logits" => a.dump_logits = Some(PathBuf::from(val()?)),
            "--embed" => a.embed = true,
            "--prompt-name" => a.prompt_name = Some(val()?),
            "--similarity" => a.similarity = true,
            "--sparse" => a.sparse = true,
            "--colbert" => a.colbert = true,
            "--dim" => a.dim = Some(val()?.parse()?),
            "--show-tokens" => a.show_tokens = true,
            "-h" | "--help" => usage(),
            _ => {
                eprintln!("unknown argument {k}");
                usage()
            }
        }
    }
    if a.model.as_os_str().is_empty() && !a.devices {
        usage();
    }
    Ok(a)
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Soft tokens for one attachment: placeholder token id and `[n, hidden]` embeddings.
struct MmFeature {
    token_id: u32,
    emb: Vec<f32>,
    n: usize,
}

/// Decodes and encodes all attachments. Returns the placeholder text to prepend to the user
/// message and the features in the same order.
#[cfg(not(feature = "media"))]
fn prepare_multimodal(_args: &Args, _files: &safetensors::ModelFiles, _text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    bail!("built without the `media` feature: image and audio inputs are unavailable")
}

#[cfg(feature = "media")]
fn prepare_multimodal(args: &Args, files: &safetensors::ModelFiles, text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    let mmc = mm_config::MmConfig::load(&args.model)?;
    let media = media::Media::new();
    let mut text = String::new();
    let mut feats = Vec::new();
    if !args.images.is_empty() {
        let vcfg = mmc.vision.clone().context("checkpoint has no vision tower")?;
        let t = Instant::now();
        let tower = vision::VisionTower::load(vcfg, files, text_hidden)?;
        let budget = if args.max_soft_tokens > 0 {
            args.max_soft_tokens
        } else {
            mmc.vision_soft_tokens_per_image
        };
        if !vision::SUPPORTED_SOFT_TOKENS.contains(&budget) {
            bail!("--max-soft-tokens must be one of {:?}", vision::SUPPORTED_SOFT_TOKENS);
        }
        for (i, path) in args.images.iter().enumerate() {
            let img = media.load_image(path)?;
            let patches = vision::preprocess(&img.data, img.width, img.height, &tower.cfg, budget)?;
            let emb = tower.encode(&patches);
            let n = emb.len() / text_hidden;
            eprintln!(
                "[image {}: {}x{} -> {}x{} patches -> {} soft tokens, {:.2}s]",
                path.display(),
                img.width,
                img.height,
                patches.pw,
                patches.ph,
                n,
                t.elapsed().as_secs_f64()
            );
            if let (Some(prefix), 0) = (&args.dump_mm, i) {
                let b: Vec<u8> = patches.patches.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{}_pixels.bin", prefix.display()), b)?;
                let b: Vec<u8> = emb.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{}_image.bin", prefix.display()), b)?;
            }
            text.push_str("<|image>");
            for _ in 0..n {
                text.push_str("<|image|>");
            }
            text.push_str("<image|>");
            feats.push(MmFeature {
                token_id: mmc.image_token_id,
                emb,
                n,
            });
        }
    }
    if !args.audios.is_empty() {
        let acfg = mmc.audio.clone().context("checkpoint has no audio tower")?;
        let t = Instant::now();
        let tower = audio::AudioTower::load(acfg, files, text_hidden)?;
        for (i, path) in args.audios.iter().enumerate() {
            let mut pcm = media.load_audio(path, audio::SAMPLE_RATE)?;
            // The reference feature extractor pads the waveform to a multiple of 128 samples.
            let padded = pcm.len().div_ceil(128) * 128;
            pcm.resize(padded, 0.0);
            let frames = audio::num_mel_frames(pcm.len());
            if frames == 0 {
                bail!("{}: audio too short", path.display());
            }
            let mel = audio::mel_features(&pcm);
            let emb = tower.encode(&mel, frames);
            let n = emb.len() / text_hidden;
            eprintln!(
                "[audio {}: {:.2}s -> {} mel frames -> {} soft tokens, {:.2}s]",
                path.display(),
                pcm.len() as f64 / audio::SAMPLE_RATE as f64,
                frames,
                n,
                t.elapsed().as_secs_f64()
            );
            if let (Some(prefix), 0) = (&args.dump_mm, i) {
                let b: Vec<u8> = mel.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{}_mel.bin", prefix.display()), b)?;
                let b: Vec<u8> = emb.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{}_audio.bin", prefix.display()), b)?;
            }
            text.push_str("<|audio>");
            for _ in 0..n {
                text.push_str("<|audio|>");
            }
            text.push_str("<audio|>");
            feats.push(MmFeature {
                token_id: mmc.audio_token_id,
                emb,
                n,
            });
        }
    }
    Ok((text, feats))
}

/// Maps placeholder token positions in `ids` to soft-token embeddings, in attachment order.
fn mm_overrides(ids: &[u32], feats: &[MmFeature], hidden: usize) -> Result<Vec<(usize, Vec<f32>)>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for f in feats {
        let mut k = 0;
        while k < f.n {
            let pos = ids[cursor..]
                .iter()
                .position(|&t| t == f.token_id)
                .context("fewer placeholder tokens than soft tokens")?
                + cursor;
            out.push((pos, f.emb[k * hidden..(k + 1) * hidden].to_vec()));
            cursor = pos + 1;
            k += 1;
        }
    }
    Ok(out)
}

/// Prefill in chunks with multimodal overrides; returns the hidden states of the last chunk.
fn prefill_mm<M: TextLm>(model: &M, sess: &mut M::Sess, ids: &[u32], chunk: usize, overrides: &[(usize, Vec<f32>)]) -> Vec<f32> {
    let mut hs = Vec::new();
    let mut start = 0;
    for c in ids.chunks(chunk.max(1)) {
        let end = start + c.len();
        let local: Vec<(usize, &[f32])> = overrides
            .iter()
            .filter(|(p, _)| *p >= start && *p < end)
            .map(|(p, e)| (p - start, e.as_slice()))
            .collect();
        hs = model.forward_hidden_mm(sess, c, &local);
        start = end;
    }
    hs
}

#[allow(dead_code)]
fn prefill(model: &Model, sess: &mut Session, ids: &[u32], chunk: usize) -> Vec<f32> {
    let mut logits = Vec::new();
    for c in ids.chunks(chunk.max(1)) {
        logits = model.forward(sess, c);
    }
    logits
}

/// Common surface of the model families (Gemma 4, Qwen 3.5/3.8) used by the generation loops.
trait TextLm {
    type Sess;
    fn new_session(&self) -> Self::Sess;
    fn hidden_size(&self) -> usize;
    fn forward_hidden_mm(&self, sess: &mut Self::Sess, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32>;
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32>;
    fn forward(&self, sess: &mut Self::Sess, tokens: &[u32]) -> Vec<f32> {
        let h = self.hidden_size();
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }
}

impl TextLm for Model<'_> {
    type Sess = Session;
    fn new_session(&self) -> Session {
        Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn forward_hidden_mm(&self, sess: &mut Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        Model::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Model::logits(self, hidden, t)
    }
}

impl TextLm for qwen::QwenModel<'_> {
    type Sess = qwen::QwenSession;
    fn new_session(&self) -> qwen::QwenSession {
        qwen::QwenModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn forward_hidden_mm(&self, sess: &mut qwen::QwenSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        qwen::QwenModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        qwen::QwenModel::logits(self, hidden, t)
    }
}

/// Incremental stdout printer: prints the stable decoded prefix of the token list.
struct Printer {
    printed: usize,
    soc: u32,
    eoc: u32,
    /// Token that opens a tool call: nothing from it on is shown (the host runs the call).
    call: Option<u32>,
}

impl Printer {
    fn new(fmt: &dyn ChatFormat, think: bool, tok: &Tokenizer) -> Self {
        let (soc, eoc) = fmt.think_tokens();
        let id = |t: &str| if t.is_empty() { None } else { tok.token_to_id(t) };
        let (soc, eoc) = (id(soc).unwrap_or(u32::MAX), id(eoc).unwrap_or(u32::MAX));
        let call = id(fmt.tool_call_marker());
        if fmt.starts_in_thinking(think) {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_all(b"\x1b[2m");
            let _ = lock.flush();
        }
        Self {
            printed: 0,
            soc,
            eoc,
            call,
        }
    }

    fn on_token(&mut self, id: u32) {
        if id == self.soc || id == self.eoc {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_all(if id == self.soc { b"\x1b[2m" } else { b"\x1b[0m\n" });
            let _ = lock.flush();
        }
    }

    fn flush_text(&mut self, tok: &Tokenizer, out: &[u32], final_: bool) {
        // Thinking delimiters are rendered as dimming, never as text (some tokenizers keep them).
        let visible = match self.call.and_then(|c| out.iter().position(|&t| t == c)) {
            Some(i) => &out[..i],
            None => out,
        };
        let shown: Vec<u32> = visible.iter().copied().filter(|&t| t != self.soc && t != self.eoc).collect();
        if let Ok(text) = tok.decode(&shown, true) {
            let stable = if !final_ && text.ends_with('\u{FFFD}') {
                text.len().saturating_sub(3)
            } else {
                text.len()
            };
            if stable > self.printed && text.is_char_boundary(stable) {
                let mut lock = std::io::stdout().lock();
                let _ = lock.write_all(&text.as_bytes()[self.printed..stable]);
                let _ = lock.flush();
                self.printed = stable;
            }
        }
        if final_ {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_all(b"\x1b[0m");
            let _ = lock.flush();
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
    let mut stats = SpecStats {
        forwards: 0,
        drafted: 0,
        accepted: 0,
        per_pos: [0; 16],
    };
    let mut out: Vec<u32> = Vec::new();
    let mut printer = Printer::new(&chat::GemmaChat, false, tok);

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
        if stats.forwards == 0 && std::env::var("OXINFER_DEBUG_DRAFT").is_ok() {
            eprintln!("[debug: pos {pos} last {last} drafts {drafts:?}]");
        }

        // Verify: one main forward over [last, drafts...].
        let mut seq = Vec::with_capacity(drafts.len() + 1);
        seq.push(last);
        seq.extend_from_slice(&drafts);
        let check = std::env::var("OXINFER_CHECK_SPEC").is_ok();
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
                        pos + i,
                        tk,
                        a1,
                        l1[a1 as usize],
                        seq.len(),
                        ak,
                        lk[ak as usize],
                        ak,
                        l1[ak as usize],
                        a1,
                        lk[a1 as usize]
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
        if stats.forwards == 1 && std::env::var("OXINFER_DEBUG_DRAFT").is_ok() {
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
#[allow(clippy::too_many_arguments)]
fn generate<M: TextLm>(
    model: &M,
    sess: &mut M::Sess,
    tok: &Tokenizer,
    mut logits: Vec<f32>,
    sampler: &mut Sampler,
    eos: &[u32],
    max_tokens: usize,
    fmt: &dyn ChatFormat,
    think: bool,
) -> (Vec<u32>, usize, f64) {
    let t0 = Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut printer = Printer::new(fmt, think, tok);
    let mut n_forward = 0usize;
    for _ in 0..max_tokens {
        let id = sampler.sample(&logits);
        out.push(id);
        if eos.contains(&id) {
            break;
        }
        printer.on_token(id);
        printer.flush_text(tok, &out, false);
        logits = model.forward(sess, &[id]);
        n_forward += 1;
    }
    let shown = if out.last().is_some_and(|e| eos.contains(e)) {
        &out[..out.len() - 1]
    } else {
        &out[..]
    };
    printer.flush_text(tok, shown, true);
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

/// Resolves a model spec to a directory: a local path, or (with the `hf` feature) a repo id.
fn resolve_model(spec: &str, cache_dir: Option<&std::path::Path>) -> Result<PathBuf> {
    #[cfg(feature = "hf")]
    {
        hf::resolve(spec, cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = cache_dir;
        let p = PathBuf::from(spec);
        if p.is_dir() {
            Ok(p)
        } else {
            bail!("{spec}: not a directory (built without the `hf` feature, so repo ids cannot be fetched)")
        }
    }
}

/// Resolves a file spec: a local path, or (with the `hf` feature) `org/repo/path/in/repo`.
fn resolve_file(spec: &str, cache_dir: Option<&std::path::Path>) -> Result<PathBuf> {
    #[cfg(feature = "hf")]
    {
        hf::resolve_file(spec, cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = cache_dir;
        let p = PathBuf::from(spec);
        if p.is_file() {
            Ok(p)
        } else {
            bail!("{spec}: no such file (built without the `hf` feature, so repo files cannot be fetched)")
        }
    }
}

/// Writes the tensors the loader used, quantized as loaded, plus the config and tokenizer files.
fn save_quantized(src: &Path, out: &Path, files: &safetensors::ModelFiles, quant: Quant) -> Result<()> {
    if out.exists() && std::fs::canonicalize(out)? == std::fs::canonicalize(src)? {
        bail!("refusing to write into the source directory {}", src.display());
    }
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let t0 = Instant::now();
    let table = oxinfer::model::table_quant(quant);
    let source = src.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let written = files.write_quantized(
        out,
        &[
            ("format", "pt"),
            ("oxinfer_quant", quant.as_str()),
            ("oxinfer_table_quant", table.as_str()),
            ("oxinfer_source", &source),
        ],
    )?;
    let mut copied = Vec::new();
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        let keep = (name.ends_with(".json") && name != "model.safetensors.index.json") || name.ends_with(".jinja");
        if keep && e.file_type()?.is_file() {
            std::fs::copy(e.path(), out.join(&name))?;
            copied.push(name);
        }
    }
    copied.sort();
    for (p, size) in &written {
        eprintln!("[wrote {} ({:.2} GB)]", p.display(), *size as f64 / 1e9);
    }
    eprintln!(
        "[{} weights, tables {}; copied {}; {:.1}s]",
        quant.as_str(),
        table.as_str(),
        copied.join(", "),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = parse_args()?;
    if args.devices {
        print!("{}", platform::detect().report());
        return Ok(());
    }
    // Resolve repo ids to cached local directories.
    let spec = args.model.to_string_lossy().to_string();
    args.model = resolve_model(&spec, args.cache_dir.as_deref())?;
    if let Some(d) = &args.draft {
        let spec = d.to_string_lossy().to_string();
        args.draft = Some(resolve_model(&spec, args.cache_dir.as_deref())?);
    }
    for l in &mut args.loras {
        l.path = resolve_file(&l.path.to_string_lossy(), args.cache_dir.as_deref())?;
    }
    if args.download_only {
        eprintln!("[model at {}]", args.model.display());
        return Ok(());
    }
    let threads = args.threads.unwrap_or_else(default_threads);
    pool::init(threads);
    run(args)
}

fn run(mut args: Args) -> Result<()> {
    let plat = platform::detect();
    eprintln!("[platform: {}]", plat.summary());
    let gpu = oxinfer::gpu::init(&plat, args.device);
    match gpu {
        Some(b) => eprintln!("[device: {} (prefill matmuls on the GPU, everything else on the CPU)]", b.name()),
        None => eprintln!("[device: cpu]"),
    }
    // Bigger prefill chunks keep the GPU busier (577 vs 431 tok/s on E4B q4 at 1024 vs 256);
    // on the CPU 256 keeps the activations in cache.
    args.prefill_chunk = Some(args.prefill_chunk.unwrap_or(if gpu.is_some() { 1024 } else { 256 }));
    if oxinfer::sd::is_diffusers(&args.model) {
        return run_image(&args);
    }
    if args.embed || oxinfer::embed::is_embedding_model(&args.model) {
        return run_embed(&args);
    }
    let model_type = config::model_type(&args.model)?;
    match model_type.as_str() {
        "gemma4" | "gemma4_text" | "gemma3" | "gemma3_text" => run_gemma(args),
        "qwen3" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_6" | "qwen3_8" => run_qwen(args),
        other => bail!("unsupported model_type {other}"),
    }
}

/// The system prompt in effect: `--system`, else the default one (unless `--no-system`, a raw
/// prompt, or a logits dump, which must match reference renderings without one).
fn system_prompt(args: &Args) -> Option<String> {
    if args.system.is_some() {
        return args.system.clone();
    }
    if args.no_system || args.raw || args.dump_logits.is_some() {
        return None;
    }
    Some(oxinfer::tools::default_system_prompt(args.tools))
}

fn tool_specs(args: &Args) -> Vec<oxinfer::tools::ToolSpec> {
    if args.tools { oxinfer::tools::builtin_specs() } else { Vec::new() }
}

/// After a reply, runs the built-in tools it called, feeds the results back and lets the model
/// continue, for up to 5 rounds. Returns the tokens generated after the last round.
#[allow(clippy::too_many_arguments)]
fn continue_with_tools<M: TextLm>(
    args: &Args,
    model: &M,
    sess: &mut M::Sess,
    tok: &Tokenizer,
    eos: &[u32],
    fmt: &dyn ChatFormat,
    sampler: &mut Sampler,
    mut out: Vec<u32>,
) -> Result<Vec<u32>> {
    if !args.tools {
        return Ok(out);
    }
    for _ in 0..5 {
        let raw = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        let calls = fmt.parse_tool_calls(&raw);
        if calls.is_empty() {
            break;
        }
        let results: Vec<oxinfer::tools::ToolResult> = calls
            .iter()
            .map(|c| {
                let content = oxinfer::tools::run_builtin(c);
                eprintln!("[tool {}({}) -> {content}]", c.name, serde_json::Value::Object(c.arguments.clone()));
                oxinfer::tools::ToolResult {
                    name: c.name.clone(),
                    content,
                }
            })
            .collect();
        let text = fmt.tool_results(&results, args.think);
        let ids = tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let t0 = Instant::now();
        let hs = prefill_mm(model, sess, &ids, args.prefill_chunk.unwrap_or(256), &[]);
        let h = model.hidden_size();
        let logits = model.logits(&hs[hs.len() - h..], 1);
        let prefill_secs = t0.elapsed().as_secs_f64();
        let (o, n, secs) = generate(model, sess, tok, logits, sampler, eos, args.max_tokens, fmt, args.think);
        report(ids.len(), prefill_secs, n, secs);
        out = o;
    }
    Ok(out)
}

/// Text-to-image with a diffusers Stable Diffusion folder.
fn run_image(args: &Args) -> Result<()> {
    use oxinfer::sd::{Params, Pipeline, sched::Sampler};
    let t0 = Instant::now();
    let pipe = Pipeline::load_with_loras(&args.model, &args.loras, &mut |line| eprintln!("{line}"))?;
    eprintln!(
        "[loaded diffusion pipeline in {:.2}s{}]",
        t0.elapsed().as_secs_f64(),
        if pipe.turbo { ", turbo" } else { "" }
    );
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut s = String::new();
            std::io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        }
    };
    let side = pipe.default_size();
    let (width, height) = args.size.unwrap_or((side, side));
    let sampler = match args.sampler.as_deref().unwrap_or("euler_a") {
        "euler" => Sampler::Euler,
        "euler_a" | "euler_ancestral" => Sampler::EulerAncestral,
        other => bail!("unknown sampler {other} (use euler or euler_a)"),
    };
    let params = Params {
        prompt,
        negative: args.negative.clone(),
        steps: args.steps.unwrap_or(if pipe.turbo { 1 } else { 25 }),
        guidance: args.guidance.unwrap_or(if pipe.turbo { 0.0 } else { 7.5 }),
        seed: args.seed,
        width,
        height,
        sampler,
    };
    eprintln!(
        "[{}x{}, {} steps, guidance {}, {:?}, seed {}]",
        width, height, params.steps, params.guidance, sampler, params.seed
    );
    let (img, tm) = pipe.generate(&params, &mut |i, n| eprint!("\r[step {i}/{n}]"))?;
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[text {:.2}s | denoise {:.2}s ({} UNet passes, {:.2}s each) | decode {:.2}s | wrote {}]",
        tm.text,
        tm.denoise,
        tm.unet_calls,
        tm.denoise / tm.unet_calls.max(1) as f64,
        tm.decode,
        out.display()
    );
    if args.profile {
        eprint!("[time by operation]\n{}", oxinfer::sd::ops::timing::report());
        if let Some(s) = oxinfer::gpu::active().and_then(|b| b.stats()) {
            eprintln!("  gpu: {s}");
        }
    }
    if oxinfer::sd::ops::gpu_check::on() {
        eprintln!("[gpu check: {}]", oxinfer::sd::ops::gpu_check::summary());
    }
    Ok(())
}

/// Plain generation (single prompt or chat) for any model family, without attachments or drafter.
fn run_text<M: TextLm>(args: &Args, model: &M, tok: &Tokenizer, eos: &[u32], fmt: &dyn ChatFormat) -> Result<()> {
    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let mut sess = model.new_session();
    let h = model.hidden_size();
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
                fmt.first_turn(system_prompt(args).as_deref(), args.think, &tool_specs(args), line)
            } else {
                fmt.end_of_model_turn() + &fmt.user_turn(line, args.think)
            };
            let ids = encode(&text)?;
            if args.show_tokens {
                eprintln!("[prompt ids: {ids:?}]");
            }
            let t = Instant::now();
            let hs = prefill_mm(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &[]);
            let logits = model.logits(&hs[hs.len() - h..], 1);
            let prefill_secs = t.elapsed().as_secs_f64();
            prof::reset();
            let (out, n, decode_secs) = generate(model, &mut sess, tok, logits, &mut sampler, eos, args.max_tokens, fmt, args.think);
            if args.show_tokens {
                eprintln!("[output ids: {out:?}]");
            }
            report(ids.len(), prefill_secs, n, decode_secs);
            if args.profile {
                prof::report(decode_secs, n);
            }
            continue_with_tools(args, model, &mut sess, tok, eos, fmt, &mut sampler, out)?;
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
        prompt
    } else {
        fmt.first_turn(system_prompt(args).as_deref(), args.think, &tool_specs(args), &prompt)
    };
    if args.print_prompt {
        print!("{text}");
        return Ok(());
    }
    let ids = encode(&text)?;
    if args.show_tokens {
        eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
    }
    if ids.is_empty() {
        bail!("empty prompt");
    }
    let t = Instant::now();
    let hs = prefill_mm(model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &[]);
    let logits = model.logits(&hs[hs.len() - h..], 1);
    let prefill_secs = t.elapsed().as_secs_f64();
    if args.profile {
        eprintln!("-- prefill --");
        prof::report(prefill_secs, ids.len());
    }
    if let Some(path) = &args.dump_logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let top = idx[..10]
            .iter()
            .map(|&i| format!("{i}:{:.3}", logits[i]))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("[wrote {} logits to {} | top10 {top}]", logits.len(), path.display());
        eprintln!("[prefill {} tok in {:.2}s]", ids.len(), prefill_secs);
        return Ok(());
    }
    prof::reset();
    let (out, n, decode_secs) = generate(model, &mut sess, tok, logits, &mut sampler, eos, args.max_tokens, fmt, args.think);
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    report(ids.len(), prefill_secs, n, decode_secs);
    if args.profile {
        eprintln!("-- decode --");
        prof::report(decode_secs, n);
    }
    continue_with_tools(args, model, &mut sess, tok, eos, fmt, &mut sampler, out)?;
    Ok(())
}

/// Embeddings for the prompt or each stdin line.
fn run_embed(args: &Args) -> Result<()> {
    let t0 = Instant::now();
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let emb = oxinfer::embed::Embedder::load(&args.model, &files, args.quant)?;
    eprintln!(
        "[loaded embedding model: dim {}, {:?} pooling{}{}, max {} tokens, prompts [{}], {:.2}s]",
        emb.dim(),
        emb.pooling,
        if emb.normalize { ", normalized" } else { "" },
        match (emb.has_sparse(), emb.has_colbert()) {
            (true, true) => ", sparse + colbert heads",
            (true, false) => ", sparse head",
            (false, true) => ", colbert head",
            _ => "",
        },
        emb.max_len,
        emb.prompts.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(", "),
        t0.elapsed().as_secs_f64()
    );
    if (args.sparse && !emb.has_sparse()) || (args.colbert && !emb.has_colbert()) {
        bail!("this model has no {} head", if args.sparse { "sparse" } else { "colbert" });
    }
    let texts: Vec<String> = match &args.prompt {
        Some(p) => vec![p.clone()],
        None => std::io::stdin()
            .lock()
            .lines()
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|l| !l.trim().is_empty())
            .collect(),
    };
    let fmt = |v: &[f32]| format!("[{}]", v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","));
    let t1 = Instant::now();
    let mut vecs = Vec::with_capacity(texts.len());
    for text in &texts {
        let mut e = emb.embed(text, args.prompt_name.as_deref())?;
        if let Some(d) = args.dim {
            e.truncate(d).context("--dim")?;
        }
        if args.similarity {
            vecs.push(e.dense);
            continue;
        }
        let line = if args.sparse {
            let m: serde_json::Map<String, serde_json::Value> = emb
                .sparse_tokens(&e)
                .unwrap()
                .into_iter()
                .map(|(k, w)| (k, serde_json::Value::from(w as f64)))
                .collect();
            serde_json::Value::Object(m).to_string()
        } else if args.colbert {
            format!("[{}]", e.colbert.unwrap().iter().map(|v| fmt(v)).collect::<Vec<_>>().join(","))
        } else {
            fmt(&e.dense)
        };
        println!("{line}");
    }
    if args.similarity {
        for a in &vecs {
            println!(
                "{}",
                vecs.iter()
                    .map(|b| format!("{:7.4}", oxinfer::embed::cosine(a, b)))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    eprintln!("[{} texts in {:.2}s]", texts.len(), t1.elapsed().as_secs_f64());
    Ok(())
}

fn run_qwen(args: Args) -> Result<()> {
    let t0 = Instant::now();
    if args.draft.is_some() || !args.images.is_empty() || !args.audios.is_empty() {
        bail!("speculative decoding and attachments are not supported for Qwen models yet");
    }
    let cfg = config::QwenConfig::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let model = qwen::QwenModel::load(cfg, &files, args.quant)?;
    let tok = Tokenizer::from_file(args.model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let fmt = chat::QwenChat::for_model(&args.model);
    let eos = chat::with_turn_end(fmt, &tok, eos);
    eprintln!(
        "[loaded Qwen: {} layers, hidden {}, {:.1} GB checkpoint ({}), {:?} weights streaming {:.2} GB/token, {} threads, {:.2}s]",
        model.cfg.num_hidden_layers,
        model.cfg.hidden_size,
        files.total_bytes() as f64 / 1e9,
        if args.mmap { "mmap" } else { "in RAM" },
        model.quant,
        model.streamed_bytes() as f64 / 1e9,
        pool::threads(),
        t0.elapsed().as_secs_f64()
    );
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
    run_text(&args, &model, &tok, &eos, fmt)
}

fn run_gemma(args: Args) -> Result<()> {
    let t0 = Instant::now();
    let cfg = config::TextConfig::load(&args.model)?;
    let eos = config::load_eos_ids(&args.model);
    let files = safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let model = Model::load(cfg, &files, args.quant)?;
    let tok = Tokenizer::from_file(args.model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    if model.cfg.gemma3 {
        if args.draft.is_some() || !args.images.is_empty() || !args.audios.is_empty() {
            bail!("speculative decoding and attachments are not supported for Gemma 3 yet");
        }
        if let Some(out) = &args.save_quantized {
            return save_quantized(&args.model, out, &files, model.quant);
        }
        let eos = chat::with_turn_end(&chat::Gemma3Chat, &tok, eos);
        return run_text(&args, &model, &tok, &eos, &chat::Gemma3Chat);
    }
    let eos = chat::with_turn_end(&chat::GemmaChat, &tok, eos);
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
    if let Some(out) = &args.save_quantized {
        return save_quantized(&args.model, out, &files, model.quant);
    }
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
                bail!(
                    "drafter backbone size {} does not match model hidden size {}",
                    dr.backbone_hidden,
                    model.cfg.hidden_size
                );
            }
            if args.temperature > 0.0 {
                eprintln!("[note: speculative decoding is greedy; --temp ignored]");
            }
            eprintln!(
                "[drafter loaded: {} layers, hidden {}, {} draft tokens/round]",
                dr.cfg.num_hidden_layers, dr.cfg.hidden_size, args.draft_tokens
            );
            Some(dr)
        }
        _ => None,
    };

    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let mut sess = model.new_session();

    if args.chat {
        let mut first = true;
        let (mm_text, mm_feats) = if !args.images.is_empty() || !args.audios.is_empty() {
            prepare_multimodal(&args, &files, model.cfg.hidden_size)?
        } else {
            (String::new(), Vec::new())
        };
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
            let mut overrides = Vec::new();
            if first {
                text.push_str(&chat::GemmaChat.conversation_start_tools(system_prompt(&args).as_deref(), args.think, &tool_specs(&args)));
                text.push_str(&chat::user_turn(&format!("{mm_text}{line}")));
                first = false;
            } else {
                // The previous stop token was sampled but never fed to the model.
                text.push_str(&chat::end_of_model_turn());
                text.push_str(&chat::user_turn(line));
            }
            let ids = encode(&text)?;
            if !mm_feats.is_empty() && !text.is_empty() && text.contains("<|image|>") || text.contains("<|audio|>") {
                overrides = mm_overrides(&ids, &mm_feats, model.cfg.hidden_size)?;
            }
            if args.show_tokens {
                eprintln!("[prompt ids: {ids:?}]");
            }
            let t = Instant::now();
            if let Some(dr) = &drafter {
                let hs = prefill_mm(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
                let h = model.cfg.hidden_size;
                let hidden_last = hs[hs.len() - h..].to_vec();
                let prefill_secs = t.elapsed().as_secs_f64();
                let (out, st, secs) =
                    generate_speculative(&model, dr, &mut sess, &tok, hidden_last, args.draft_tokens, &eos, args.max_tokens);
                if args.show_tokens {
                    eprintln!("[output ids: {out:?}]");
                }
                report(ids.len(), prefill_secs, out.len(), secs);
                eprintln!(
                    "[speculative: {:.2} tok/forward, drafts accepted {}%]",
                    out.len() as f64 / st.forwards.max(1) as f64,
                    100 * st.accepted / st.drafted.max(1)
                );
                continue;
            }
            let hs = prefill_mm(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
            let logits = model.logits(&hs[hs.len() - model.cfg.hidden_size..], 1);
            let prefill_secs = t.elapsed().as_secs_f64();
            prof::reset();
            let (out, n, decode_secs) = generate(
                &model,
                &mut sess,
                &tok,
                logits,
                &mut sampler,
                &eos,
                args.max_tokens,
                &chat::GemmaChat,
                args.think,
            );
            if args.show_tokens {
                eprintln!("[output ids: {out:?}]");
            }
            report(ids.len(), prefill_secs, n, decode_secs);
            if args.profile {
                prof::report(decode_secs, n);
            }
            continue_with_tools(&args, &model, &mut sess, &tok, &eos, &chat::GemmaChat, &mut sampler, out)?;
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
    let (mm_text, mm_feats) = if !args.images.is_empty() || !args.audios.is_empty() {
        prepare_multimodal(&args, &files, model.cfg.hidden_size)?
    } else {
        (String::new(), Vec::new())
    };
    let prompt = format!("{mm_text}{prompt}");
    let text = if args.raw {
        format!("<bos>{prompt}")
    } else {
        let mut t = chat::GemmaChat.conversation_start_tools(system_prompt(&args).as_deref(), args.think, &tool_specs(&args));
        t.push_str(&chat::user_turn(&prompt));
        t
    };
    if args.print_prompt {
        print!("{text}");
        return Ok(());
    }
    let ids = encode(&text)?;
    let overrides = mm_overrides(&ids, &mm_feats, model.cfg.hidden_size)?;
    if args.show_tokens {
        eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
    }
    if ids.is_empty() {
        bail!("empty prompt");
    }

    if let Some(dr) = &drafter {
        let t = Instant::now();
        let hs = prefill_mm(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
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
    let hs = prefill_mm(&model, &mut sess, &ids, args.prefill_chunk.unwrap_or(256), &overrides);
    let logits = model.logits(&hs[hs.len() - model.cfg.hidden_size..], 1);
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
    let (out, n, decode_secs) = generate(
        &model,
        &mut sess,
        &tok,
        logits,
        &mut sampler,
        &eos,
        args.max_tokens,
        &chat::GemmaChat,
        args.think,
    );
    if args.show_tokens {
        eprintln!("[output ids: {out:?}]");
    }
    report(ids.len(), prefill_secs, n, decode_secs);
    if args.profile {
        eprintln!("-- decode --");
        prof::report(decode_secs, n);
    }
    continue_with_tools(&args, &model, &mut sess, &tok, &eos, &chat::GemmaChat, &mut sampler, out)?;
    Ok(())
}
