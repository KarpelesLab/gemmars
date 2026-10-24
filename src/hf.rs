//! Fetching and caching HuggingFace model repositories by name, over rsurl (pure-Rust HTTP).
//!
//! `--model google/gemma-4-E4B-it` resolves to `$OXINFER_CACHE/google--gemma-4-E4B-it` (default
//! `~/.cache/oxinfer/hub`), downloading the config, tokenizer, template and safetensors files that
//! are missing or incomplete. Downloads are resumable. Gated repos need `HF_TOKEN`.

use anyhow::{Context, Result, anyhow, bail};
use rsurl::{DownloadOptions, Request};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn cache_root() -> PathBuf {
    if let Ok(p) = std::env::var("OXINFER_CACHE") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("oxinfer").join("hub")
}

/// A local directory is used as is; otherwise `spec` must be a repo id (`org/name`), which is
/// fetched into the cache (if needed) and its directory returned.
pub fn resolve(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|s| s.is_empty() || s.starts_with('.')) || spec.starts_with('/') {
        bail!("{spec}: not an existing directory and not a HuggingFace repo id of the form org/name");
    }
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_else(cache_root);
    let dir = root.join(spec.replace('/', "--"));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    fetch_repo(spec, &dir)?;
    Ok(dir)
}

fn token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .filter(|t| !t.is_empty())
}

fn request(url: &str) -> Result<Request> {
    let mut r = Request::get(url)
        .map_err(|e| anyhow!("{url}: {e}"))?
        .follow_redirects(true)
        .max_redirs(10)
        .header("User-Agent", "oxinfer/0.1")
        // The resumable downloader stores raw bytes; refuse compressed transfer encodings.
        .header("Accept-Encoding", "identity");
    if let Some(t) = token() {
        r = r.header("Authorization", &format!("Bearer {t}"));
    }
    Ok(r)
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Deserialize)]
struct RepoInfo {
    #[serde(default)]
    siblings: Vec<Sibling>,
}

const SMALL_FILES: [&str; 8] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
];

fn fetch_repo(repo: &str, dir: &Path) -> Result<()> {
    let api = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let resp = request(&api)?.send().map_err(|e| anyhow!("{api}: {e}"))?;
    match resp.status {
        200 => {}
        401 | 403 => bail!(
            "{repo}: access denied (HTTP {}). If the repo is gated, accept its license on huggingface.co and set HF_TOKEN.",
            resp.status
        ),
        404 => bail!("{repo}: repository not found on huggingface.co"),
        s => bail!("{api}: HTTP {s}"),
    }
    let info: RepoInfo = serde_json::from_slice(&resp.body).with_context(|| format!("parse {api}"))?;
    let sizes: HashMap<String, Option<u64>> = info.siblings.into_iter().map(|s| (s.rfilename, s.size)).collect();

    let mut files: Vec<String> = SMALL_FILES
        .iter()
        .filter(|f| sizes.contains_key(**f))
        .map(|f| f.to_string())
        .collect();
    if sizes.contains_key("model.safetensors") {
        files.push("model.safetensors".into());
    } else if sizes.contains_key("model.safetensors.index.json") {
        fetch_file(
            repo,
            dir,
            "model.safetensors.index.json",
            sizes["model.safetensors.index.json"],
            0,
            0,
        )?;
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        files.extend(shards);
    } else {
        bail!("{repo}: no safetensors weights found");
    }
    let count = files.len();
    for (i, f) in files.iter().enumerate() {
        fetch_file(repo, dir, f, sizes.get(f).copied().flatten(), i + 1, count)?;
    }
    Ok(())
}

fn human_gb(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.2} GB", b as f64 / 1e9)
    } else {
        format!("{:.1} MB", b as f64 / 1e6)
    }
}

/// Renders a progress line: `[=====>     ]  42.0%  3.21/7.64 GB  31 MB/s  eta 2m10s`.
fn progress_line(label: &str, done: u64, total: Option<u64>, secs: f64) -> String {
    let rate = done as f64 / secs.max(1e-3);
    let speed = format!("{:5.0} MB/s", rate / 1e6);
    match total {
        Some(t) if t > 0 => {
            let frac = (done as f64 / t as f64).clamp(0.0, 1.0);
            let width = 30usize;
            let filled = (frac * width as f64) as usize;
            let bar: String = (0..width)
                .map(|i| {
                    if i < filled {
                        '='
                    } else if i == filled {
                        '>'
                    } else {
                        ' '
                    }
                })
                .collect();
            let eta = if rate > 0.0 { ((t - done) as f64 / rate) as u64 } else { 0 };
            // The rate is noise for the first second; show no estimate yet.
            let eta_s = if secs < 1.0 {
                "--".to_string()
            } else if eta >= 3600 {
                format!("{}h{:02}m", eta / 3600, (eta % 3600) / 60)
            } else if eta >= 60 {
                format!("{}m{:02}s", eta / 60, eta % 60)
            } else {
                format!("{eta}s")
            };
            format!(
                "  {label} [{bar}] {:5.1}%  {}/{}  {speed}  eta {eta_s}",
                frac * 100.0,
                human_gb(done),
                human_gb(t)
            )
        }
        _ => format!("  {label} {}  {speed}", human_gb(done)),
    }
}

fn fetch_file(repo: &str, dir: &Path, name: &str, expected: Option<u64>, index: usize, count: usize) -> Result<()> {
    use std::io::IsTerminal;
    let path = dir.join(name);
    if let Ok(md) = std::fs::metadata(&path)
        && md.len() > 0
        && expected.is_none_or(|s| s == md.len())
    {
        return Ok(());
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{name}");
    let t0 = Instant::now();
    let label = format!("[{index}/{count}] {name}");
    let tty = std::io::stderr().is_terminal();
    let mut last = Instant::now();
    let mut last_pct = -1i64;
    let plabel = label.clone();
    let opts = DownloadOptions {
        progress: Some(Box::new(move |done: u64, total: Option<u64>| {
            let secs = t0.elapsed().as_secs_f64();
            if tty {
                if last.elapsed().as_millis() >= 200 {
                    last = Instant::now();
                    eprint!("\r\x1b[K{}", progress_line(&plabel, done, total, secs));
                }
            } else {
                // Not a terminal: one log line per 5% (or every 30 s when the size is unknown).
                let pct = total.filter(|&t| t > 0).map(|t| (done * 100 / t) as i64 / 5 * 5);
                let due = match pct {
                    Some(p) => p != last_pct,
                    None => last.elapsed().as_secs() >= 30,
                };
                if due {
                    last = Instant::now();
                    last_pct = pct.unwrap_or(-1);
                    eprintln!("{}", progress_line(&plabel, done, total, secs));
                }
            }
        })),
        ..Default::default()
    };
    eprintln!(
        "[fetching {repo}/{name}{}]",
        expected.map(|s| format!(" ({})", human_gb(s))).unwrap_or_default()
    );
    let out = request(&url)?
        .download_resumable(&path, opts)
        .map_err(|e| anyhow!("download {url}: {e}"))?;
    let secs = t0.elapsed().as_secs_f64().max(1e-3);
    if tty {
        eprint!("\r\x1b[K");
    }
    eprintln!(
        "  {label}: done, {} in {:.0}s ({:.0} MB/s){}",
        human_gb(out.bytes_written + out.resumed_from),
        secs,
        out.bytes_written as f64 / secs / 1e6,
        if out.resumed_from > 0 {
            format!(", resumed from {}", human_gb(out.resumed_from))
        } else {
            String::new()
        }
    );
    if let (Some(exp), Ok(md)) = (expected, std::fs::metadata(&path))
        && md.len() != exp
    {
        bail!("{}: size {} after download, expected {exp}", path.display(), md.len());
    }
    Ok(())
}
