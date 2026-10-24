//! Image and audio attachments (Gemma 4): decoding through oxideav (the `media` feature),
//! encoding with the vision / audio towers into soft tokens, and mapping them onto the
//! placeholder tokens of the prompt.

#[cfg(feature = "media")]
use crate::clock::Instant;
use crate::safetensors::ModelFiles;
#[cfg(feature = "media")]
use crate::{audio, media, mm_config, vision};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Attachments for one prompt.
pub struct MmRequest<'a> {
    /// The model directory (for `config.json` / processor configs).
    pub model: &'a Path,
    pub images: &'a [PathBuf],
    pub audios: &'a [PathBuf],
    /// Image soft-token budget (0: the model's default).
    pub max_soft_tokens: usize,
    /// Write the first image's / clip's inputs and soft tokens as f32 files with this prefix.
    pub dump_mm: Option<&'a Path>,
}

/// Soft tokens for one attachment: placeholder token id and `[n, hidden]` embeddings.
pub struct MmFeature {
    /// The placeholder token the embeddings replace.
    pub token_id: u32,
    /// `[n, hidden]` soft-token embeddings.
    pub emb: Vec<f32>,
    pub n: usize,
}

/// Decodes and encodes all attachments. Returns the placeholder text to prepend to the user
/// message and the features in the same order.
#[cfg(not(feature = "media"))]
pub fn prepare(_req: &MmRequest, _files: &ModelFiles, _text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    bail!("built without the `media` feature: image and audio inputs are unavailable")
}

#[cfg(feature = "media")]
pub fn prepare(args: &MmRequest, files: &ModelFiles, text_hidden: usize) -> Result<(String, Vec<MmFeature>)> {
    let mmc = mm_config::MmConfig::load(args.model)?;
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
            if let (Some(prefix), 0) = (args.dump_mm, i) {
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
            if let (Some(prefix), 0) = (args.dump_mm, i) {
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
pub fn overrides(ids: &[u32], feats: &[MmFeature], hidden: usize) -> Result<Vec<(usize, Vec<f32>)>> {
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
