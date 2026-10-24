//! MPNet encoders (all-mpnet-base-v2 and friends): a BERT-style post-LayerNorm encoder with
//! RoBERTa-style absolute positions plus a T5-style relative position bias, one table of
//! `[buckets, heads]` shared by every layer and added to the attention scores.

use crate::bert::{Dense, LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{Quant, Table, linear_multi};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_eps() -> f64 {
    1e-12
}
fn d_pad() -> usize {
    1
}
fn d_buckets() -> usize {
    32
}

/// Farthest distance with its own log-spaced bucket; beyond it everything shares the last one.
const MAX_DISTANCE: usize = 128;

#[derive(Debug, Clone, Deserialize)]
pub struct MpnetConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "d_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "d_pad")]
    pub pad_token_id: usize,
    #[serde(default = "d_buckets")]
    pub relative_attention_num_buckets: usize,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

impl MpnetConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: MpnetConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse MPNet config")?;
        if let Some(a) = &cfg.hidden_act
            && a != "gelu"
        {
            bail!("hidden_act {a} is not supported (only gelu)");
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            bail!(
                "hidden_size {} is not a multiple of {} heads",
                cfg.hidden_size,
                cfg.num_attention_heads
            );
        }
        Ok(cfg)
    }
}

/// Bidirectional T5 bucket of query position `i` attending to key position `j`: the first half of
/// the buckets is for keys at or before the query, the second for keys after it. Within each half,
/// distances below `n/4` get their own bucket and the rest are log-spaced up to `MAX_DISTANCE`.
fn relative_bucket(i: usize, j: usize, num_buckets: usize) -> usize {
    let half = num_buckets / 2;
    let (base, n) = if j > i { (half, j - i) } else { (0, i - j) };
    let exact = half / 2;
    if n < exact {
        return base + n;
    }
    // f32 like the reference, so the bucket edges land on the same distances.
    let log = (n as f32 / exact as f32).ln() / (MAX_DISTANCE as f32 / exact as f32).ln() * (half - exact) as f32;
    base + (exact + log as usize).min(half - 1)
}

struct MpnetLayer<'a> {
    q: Dense<'a>,
    k: Dense<'a>,
    v: Dense<'a>,
    o: Dense<'a>,
    ln_attn: LayerNorm,
    up: Dense<'a>,
    down: Dense<'a>,
    ln_out: LayerNorm,
}

pub struct MpnetModel<'a> {
    pub cfg: MpnetConfig,
    pub quant: Quant,
    word: Table<'a>,
    pos: Table<'a>,
    ln_embed: LayerNorm,
    /// `[buckets, heads]`, shared by all layers.
    rel_bias: Vec<f32>,
    layers: Vec<MpnetLayer<'a>>,
}

impl<'a> MpnetModel<'a> {
    pub fn load(cfg: MpnetConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let prefix = ["", "mpnet.", "model."]
            .into_iter()
            .find(|p| files.has(&format!("{p}embeddings.word_embeddings.weight")))
            .context("cannot find embeddings.word_embeddings in checkpoint")?;
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let n = |s: &str| format!("{prefix}{s}");
        let word = Table::load(files, &n("embeddings.word_embeddings.weight"), cfg.vocab_size, h, Quant::Bf16)?;
        let pos = Table::load(
            files,
            &n("embeddings.position_embeddings.weight"),
            cfg.max_position_embeddings,
            h,
            Quant::Bf16,
        )?;
        let ln_embed = LayerNorm::load(files, &n("embeddings.LayerNorm"), h)?;
        let rel_bias = files.f32(&n("encoder.relative_attention_bias.weight"))?;
        if rel_bias.len() != cfg.relative_attention_num_buckets * cfg.num_attention_heads {
            bail!(
                "relative_attention_bias: expected {} x {} values",
                cfg.relative_attention_num_buckets,
                cfg.num_attention_heads
            );
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = |s: &str| n(&format!("encoder.layer.{l}.{s}"));
            let d = |s: &str, o: usize, i: usize| Dense::load(files, &p(s), o, i, quant);
            layers.push(MpnetLayer {
                q: d("attention.attn.q", h, h)?,
                k: d("attention.attn.k", h, h)?,
                v: d("attention.attn.v", h, h)?,
                o: d("attention.attn.o", h, h)?,
                ln_attn: LayerNorm::load(files, &p("attention.LayerNorm"), h)?,
                up: d("intermediate.dense", inter, h)?,
                down: d("output.dense", h, inter)?,
                ln_out: LayerNorm::load(files, &p("output.LayerNorm"), h)?,
            });
        }
        Ok(Self {
            cfg,
            quant,
            word,
            pos,
            ln_embed,
            rel_bias,
            layers,
        })
    }

    /// Last hidden states `[t, hidden]` for one sequence (special tokens included).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, nh) = (cfg.hidden_size, cfg.num_attention_heads);
        let hd = h / nh;
        // Positions start after the padding index, as in RoBERTa.
        let off = cfg.pad_token_id + 1;
        let t = tokens.len().min(cfg.max_position_embeddings - off);
        let tokens = &tokens[..t];
        let eps = cfg.layer_norm_eps as f32;
        let mut x = vec![0f32; t * h];
        let mut pe = vec![0f32; h];
        for (i, row) in x.chunks_mut(h).enumerate() {
            self.word.row(tokens[i] as usize, row);
            self.pos.row(off + i, &mut pe);
            add_inplace(row, &pe);
        }
        self.ln_embed.apply_rows(&mut x, eps);

        // Bias `[head, i, j]`, computed once for all layers.
        let nb = cfg.relative_attention_num_buckets;
        let mut bias = vec![0f32; nh * t * t];
        for i in 0..t {
            for j in 0..t {
                let b = relative_bucket(i, j, nb);
                for hh in 0..nh {
                    bias[(hh * t + i) * t + j] = self.rel_bias[b * nh + hh];
                }
            }
        }

        let scale = (hd as f32).powf(-0.5);
        let (mut q, mut k, mut v) = (vec![0f32; t * h], vec![0f32; t * h], vec![0f32; t * h]);
        let mut att = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        let mut u = vec![0f32; t * cfg.intermediate_size];
        for layer in &self.layers {
            linear_multi(
                &mut [
                    (&layer.q.w, h, &mut q[..]),
                    (&layer.k.w, h, &mut k[..]),
                    (&layer.v.w, h, &mut v[..]),
                ],
                h,
                &x,
                t,
            );
            layer.q.add_bias(&mut q);
            layer.k.add_bias(&mut k);
            layer.v.add_bias(&mut v);
            scale_inplace(&mut q, scale);
            attend(&q, &k, &v, &bias, nh, hd, t, &mut att);
            layer.o.forward(&att, t, &mut o);
            add_inplace(&mut x, &o);
            layer.ln_attn.apply_rows(&mut x, eps);
            layer.up.forward(&x, t, &mut u);
            let par = u.len() > 8192;
            for_chunks(&mut u, 4096, par, |_, c| {
                for e in c.iter_mut() {
                    *e = gelu_erf(*e);
                }
            });
            layer.down.forward(&u, t, &mut o);
            add_inplace(&mut x, &o);
            layer.ln_out.apply_rows(&mut x, eps);
        }
        x
    }
}

/// Bidirectional attention with an additive score bias `[head, t, t]`. `q` is pre-scaled; `q`,
/// `k`, `v` and `out` are `[t, heads * hd]`. Parallel over query tokens.
#[allow(clippy::too_many_arguments)]
fn attend(q: &[f32], k: &[f32], v: &[f32], bias: &[f32], nh: usize, hd: usize, t: usize, out: &mut [f32]) {
    let h = nh * hd;
    for_chunks(out, h, t > 1, |i, orow| {
        let mut s = vec![0f32; t];
        orow.fill(0.0);
        for hh in 0..nh {
            let qh = &q[i * h + hh * hd..][..hd];
            let b = &bias[(hh * t + i) * t..][..t];
            for (j, sj) in s.iter_mut().enumerate() {
                *sj = dot(qh, &k[j * h + hh * hd..][..hd]) + b[j];
            }
            softmax_inplace(&mut s);
            let oh = &mut orow[hh * hd..][..hd];
            for (j, &p) in s.iter().enumerate() {
                axpy(oh, p, &v[j * h + hh * hd..][..hd]);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::relative_bucket;

    #[test]
    fn buckets_match_reference() {
        // (j - i, bucket) where the reference's bucket changes, from transformers'
        // `MPNetEncoder.relative_position_bucket` over -520..=520.
        let edges = [
            (-520, 15),
            (-90, 14),
            (-63, 13),
            (-45, 12),
            (-31, 11),
            (-22, 10),
            (-15, 9),
            (-11, 8),
            (-7, 7),
            (-6, 6),
            (-5, 5),
            (-4, 4),
            (-3, 3),
            (-2, 2),
            (-1, 1),
            (0, 0),
            (1, 17),
            (2, 18),
            (3, 19),
            (4, 20),
            (5, 21),
            (6, 22),
            (7, 23),
            (8, 24),
            (12, 25),
            (16, 26),
            (23, 27),
            (32, 28),
            (46, 29),
            (64, 30),
            (91, 31),
        ];
        let mut want = 0;
        let mut e = edges.iter().peekable();
        for r in -520i64..=520 {
            if let Some(&&(at, b)) = e.peek()
                && at == r
            {
                want = b;
                e.next();
            }
            let (i, j) = if r < 0 { ((-r) as usize, 0) } else { (0, r as usize) };
            assert_eq!(relative_bucket(i, j, 32), want, "relative position {r}");
        }
    }
}
