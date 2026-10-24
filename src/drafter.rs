//! Gemma 4 "assistant" drafter for speculative decoding: a 4-layer, 256-wide decoder whose
//! layers all cross-attend to the main model's shared KV caches. Input is the concatenation of
//! the main model's (scaled) embedding of the last token and the main model's hidden state of
//! the token before it; output is an argmax draft token (over a centroid-selected vocabulary
//! subset) and a projected hidden state to feed the next draft step.

use crate::config::TextConfig;
use crate::kernels::*;
use crate::model::{LayerCache, attention};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

struct DLayer<'a> {
    sliding: bool,
    hd: usize,
    q: &'a [u16],
    o: &'a [u16],
    q_norm: &'a [u16],
    ln_in: &'a [u16],
    ln_post_attn: &'a [u16],
    ln_pre_ffn: &'a [u16],
    ln_post_ffn: &'a [u16],
    gate: &'a [u16],
    up: &'a [u16],
    down: &'a [u16],
    layer_scalar: f32,
}

pub struct Drafter<'a> {
    pub cfg: TextConfig,
    pub backbone_hidden: usize,
    pre_proj: &'a [u16],  // [H, 2*B]
    post_proj: &'a [u16], // [B, H]
    embed: &'a [u16],     // [V, H] (tied LM head)
    centroids: &'a [u16], // [C, H]
    token_ordering: Vec<u32>,
    num_centroids: usize,
    top_k_centroids: usize,
    layers: Vec<DLayer<'a>>,
    final_norm: &'a [u16],
    inv_freq_sliding: Vec<f32>,
    inv_freq_full: Vec<f32>,
}

impl<'a> Drafter<'a> {
    pub fn load(dir: &Path, files: &'a ModelFiles) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let backbone_hidden = raw["backbone_hidden_size"].as_u64().context("backbone_hidden_size")? as usize;
        let num_centroids = raw["num_centroids"].as_u64().unwrap_or(2048) as usize;
        let top_k_centroids = raw["centroid_intermediate_top_k"].as_u64().unwrap_or(32) as usize;
        if raw["use_ordered_embeddings"].as_bool() != Some(true) {
            bail!("drafter without ordered embeddings is not supported");
        }
        let cfg = TextConfig::load(dir)?;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let v = cfg.vocab_size;

        let get = |name: &str, shape: &[usize]| -> Result<&'a [u16]> {
            let info = files.info(name).with_context(|| format!("missing tensor {name}"))?;
            if info.shape != shape {
                bail!("tensor {name} has shape {:?}, expected {:?}", info.shape, shape);
            }
            files.bf16(name)
        };
        let token_ordering: Vec<u32> = {
            let info = files.info("masked_embedding.token_ordering").context("token_ordering")?;
            if info.dtype != "I64" || info.shape != [v] {
                bail!("unexpected token_ordering dtype/shape");
            }
            let bytes = files.bytes("masked_embedding.token_ordering")?;
            bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c) as u32).collect()
        };
        let mut layers = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}.");
            let g = |n: &str, s: &[usize]| get(&format!("{p}{n}"), s);
            let hd = cfg.layer_head_dim(l);
            let layer_scalar = g("layer_scalar", &[1]).map(|s| bf16_to_f32(s[0])).unwrap_or(1.0);
            layers.push(DLayer {
                sliding: cfg.is_sliding(l),
                hd,
                q: g("self_attn.q_proj.weight", &[nh * hd, h])?,
                o: g("self_attn.o_proj.weight", &[h, nh * hd])?,
                q_norm: g("self_attn.q_norm.weight", &[hd])?,
                ln_in: g("input_layernorm.weight", &[h])?,
                ln_post_attn: g("post_attention_layernorm.weight", &[h])?,
                ln_pre_ffn: g("pre_feedforward_layernorm.weight", &[h])?,
                ln_post_ffn: g("post_feedforward_layernorm.weight", &[h])?,
                gate: g("mlp.gate_proj.weight", &[cfg.intermediate_size, h])?,
                up: g("mlp.up_proj.weight", &[cfg.intermediate_size, h])?,
                down: g("mlp.down_proj.weight", &[h, cfg.intermediate_size])?,
                layer_scalar,
            });
        }
        let rs = cfg.rope_spec(true);
        let rf = cfg.rope_spec(false);
        Ok(Self {
            backbone_hidden,
            pre_proj: get("pre_projection.weight", &[h, 2 * backbone_hidden])?,
            post_proj: get("post_projection.weight", &[backbone_hidden, h])?,
            embed: get("model.embed_tokens.weight", &[v, h])?,
            centroids: get("masked_embedding.centroids.weight", &[num_centroids, h])?,
            token_ordering,
            num_centroids,
            top_k_centroids,
            layers,
            final_norm: get("model.norm.weight", &[h])?,
            inv_freq_sliding: rope_inv_freq(cfg.head_dim, rs.theta, rs.proportion),
            inv_freq_full: rope_inv_freq(cfg.global_head_dim, rf.theta, rf.proportion),
            cfg,
        })
    }

    /// One draft step. `tok_emb`: the main model's scaled embedding of the last token; `hidden`:
    /// the main-side hidden state of the token before it; `pos`: the last token's position;
    /// `cache_s`/`cache_g`: the main model's shared sliding/global KV caches (positions `< pos`
    /// are attended). Returns the argmax draft token and the hidden state for the next step.
    pub fn step(&self, tok_emb: &[f32], hidden: &[f32], pos: usize, cache_s: &LayerCache, cache_g: &LayerCache) -> (u32, Vec<f32>) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let eps = cfg.rms_norm_eps as f32;
        let b = self.backbone_hidden;
        debug_assert_eq!(tok_emb.len(), b);
        debug_assert_eq!(hidden.len(), b);

        let mut input = Vec::with_capacity(2 * b);
        input.extend_from_slice(tok_emb);
        input.extend_from_slice(hidden);
        let mut x = vec![0f32; h];
        matmul(self.pre_proj, h, 2 * b, &input, 1, &mut x);

        let (cos_s, sin_s) = rope_tables(&[pos], &self.inv_freq_sliding);
        let (cos_f, sin_f) = rope_tables(&[pos], &self.inv_freq_full);

        let mut hn = vec![0f32; h];
        for layer in &self.layers {
            let hd = layer.hd;
            let (cos, sin, cache, window) = if layer.sliding {
                (&cos_s, &sin_s, cache_s, Some(cfg.sliding_window))
            } else {
                (&cos_f, &sin_f, cache_g, None)
            };
            rmsnorm(&x, Some(layer.ln_in), eps, &mut hn);
            let mut q = vec![0f32; nh * hd];
            matmul(layer.q, nh * hd, h, &hn, 1, &mut q);
            for qh in q.chunks_mut(hd) {
                rmsnorm_inplace(qh, Some(layer.q_norm), eps);
                apply_rope(qh, cos, sin);
            }
            let mut att = vec![0f32; nh * hd];
            // The query sits at `pos` but only positions `< pos` are cached; `attention` treats the
            // query's own slot as cached, so anchor it at `pos - 1` to cover exactly [pos-window, pos).
            debug_assert_eq!(cache.len, pos, "drafter expects the cache to hold exactly the tokens before `pos`");
            attention(&q, cache, nh, 1, pos - 1, window, &mut att);
            let mut o = vec![0f32; h];
            matmul(layer.o, h, nh * hd, &att, 1, &mut o);
            rmsnorm_inplace(&mut o, Some(layer.ln_post_attn), eps);
            add_inplace(&mut x, &o);

            rmsnorm(&x, Some(layer.ln_pre_ffn), eps, &mut hn);
            let inter = cfg.intermediate_size;
            let mut g = vec![0f32; inter];
            let mut u = vec![0f32; inter];
            matmul_multi(&mut [(layer.gate, inter, &mut g[..]), (layer.up, inter, &mut u[..])], h, &hn, 1);
            gelu_mul(&mut g, &u);
            matmul(layer.down, h, inter, &g, 1, &mut o);
            rmsnorm_inplace(&mut o, Some(layer.ln_post_ffn), eps);
            add_inplace(&mut x, &o);
            if layer.layer_scalar != 1.0 {
                scale_inplace(&mut x, layer.layer_scalar);
            }
        }
        rmsnorm_inplace(&mut x, Some(self.final_norm), eps);

        let mut next_hidden = vec![0f32; b];
        matmul(self.post_proj, b, h, &x, 1, &mut next_hidden);

        // Centroid-masked LM head: top-k centroids, then exact logits over their token clusters.
        let mut cl = vec![0f32; self.num_centroids];
        matmul(self.centroids, self.num_centroids, h, &x, 1, &mut cl);
        let mut idx: Vec<usize> = (0..self.num_centroids).collect();
        let k = self.top_k_centroids.min(self.num_centroids);
        idx.select_nth_unstable_by(k - 1, |&a, &b| cl[b].partial_cmp(&cl[a]).unwrap_or(std::cmp::Ordering::Equal));
        let per = cfg.vocab_size / self.num_centroids;
        let sel = &idx[..k];
        let mut scores = vec![0f32; k * per];
        let embed = self.embed;
        let ordering = &self.token_ordering;
        let xr = &x;
        for_chunks(&mut scores, per, true, |i, chunk| {
            let c = sel[i];
            for (j, &tok) in ordering[c * per..(c + 1) * per].iter().enumerate() {
                let row = &embed[tok as usize * h..(tok as usize + 1) * h];
                chunk[j] = dot_bf16(row, xr);
            }
        });
        let mut best = (f32::NEG_INFINITY, 0u32);
        for (i, &c) in sel.iter().enumerate() {
            for j in 0..per {
                let sc = scores[i * per + j];
                if sc > best.0 {
                    best = (sc, ordering[c * per + j]);
                }
            }
        }
        (best.1, next_hidden)
    }
}
