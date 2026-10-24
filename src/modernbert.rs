//! ModernBERT encoders (modernbert-embed, gte-modernbert): pre-LayerNorm blocks without biases,
//! a fused QKV projection with rotary positions, global attention every few layers and a
//! bidirectional sliding window in the others (each with its own RoPE theta), a GeGLU MLP and a
//! final LayerNorm. There are no position embeddings.

use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, attention_masked, linear, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_eps() -> f64 {
    1e-5
}
fn d_every() -> usize {
    3
}
fn d_local() -> usize {
    128
}
fn d_global_theta() -> f64 {
    160_000.0
}
fn d_act() -> String {
    "gelu".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModernBertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "d_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "d_act")]
    pub hidden_activation: String,
    /// Every n-th layer (from 0) attends globally; the others use the local window.
    #[serde(default = "d_every")]
    pub global_attn_every_n_layers: usize,
    /// Total local window: a token sees keys at most `local_attention / 2` positions away.
    #[serde(default = "d_local")]
    pub local_attention: usize,
    #[serde(default = "d_global_theta")]
    pub global_rope_theta: f64,
    /// None: local layers use the global theta.
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    /// Newer configs: `full_attention` / `sliding_attention` per layer.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Newer configs: `{full_attention: {rope_theta}, sliding_attention: {rope_theta}}`.
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

impl ModernBertConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut cfg: ModernBertConfig =
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse ModernBERT config")?;
        if cfg.attention_bias || cfg.mlp_bias {
            bail!("ModernBERT with attention_bias or mlp_bias is not supported");
        }
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) || cfg.global_attn_every_n_layers == 0 {
            bail!("ModernBERT config: bad head count or global_attn_every_n_layers");
        }
        act_fn(&cfg.hidden_activation)?;
        if let Some(rp) = &cfg.rope_parameters {
            if let Some(t) = rp["full_attention"]["rope_theta"].as_f64() {
                cfg.global_rope_theta = t;
            }
            if let Some(t) = rp["sliding_attention"]["rope_theta"].as_f64() {
                cfg.local_rope_theta = Some(t);
            }
        }
        if let Some(lt) = &cfg.layer_types
            && lt.len() != cfg.num_hidden_layers
        {
            bail!("ModernBERT config: {} layer_types for {} layers", lt.len(), cfg.num_hidden_layers);
        }
        Ok(cfg)
    }
    /// Whether layer `l` uses the sliding window.
    fn is_local(&self, l: usize) -> bool {
        match &self.layer_types {
            Some(lt) => lt[l] == "sliding_attention",
            None => !l.is_multiple_of(self.global_attn_every_n_layers),
        }
    }
}

/// Whether `model_type` is ModernBERT.
pub fn is_modernbert(model_type: &str) -> bool {
    model_type == "modernbert"
}

/// `hidden_activation`: exact (erf) GELU for `gelu`, the tanh approximation for its aliases.
fn act_fn(name: &str) -> Result<fn(f32) -> f32> {
    Ok(match name {
        "gelu" => crate::bert::gelu_erf,
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => gelu_tanh,
        a => bail!("hidden_activation {a} is not supported"),
    })
}

/// LayerNorm with an optional bias (`norm_bias`).
struct LayerNorm {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
}

impl LayerNorm {
    fn load(files: &ModelFiles, p: &str, h: usize, bias: bool) -> Result<Self> {
        let w = files.f32(&format!("{p}.weight"))?;
        let b = if bias { Some(files.f32(&format!("{p}.bias"))?) } else { None };
        if w.len() != h || b.as_ref().is_some_and(|b| b.len() != h) {
            bail!("{p}: expected {h} values");
        }
        Ok(Self { w, b })
    }
    fn apply_rows(&self, x: &[f32], y: &mut [f32], eps: f32) {
        let h = self.w.len();
        y.copy_from_slice(x);
        for_chunks(y, h, y.len() > 4 * h, |_, r| {
            let mean = r.iter().sum::<f32>() / h as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
            let inv = (var + eps).sqrt().recip();
            for (i, (v, w)) in r.iter_mut().zip(&self.w).enumerate() {
                *v = (*v - mean) * inv * w + self.b.as_ref().map_or(0.0, |b| b[i]);
            }
        });
    }
}

struct ModernBertLayer<'a> {
    /// None in layer 0 (the embedding norm already ran).
    attn_norm: Option<LayerNorm>,
    wqkv: Lin<'a>,
    wo: Lin<'a>,
    mlp_norm: LayerNorm,
    /// `[2 * inter, h]`: the GeGLU input rows, then the gate rows.
    wi: Lin<'a>,
    mlp_wo: Lin<'a>,
    local: bool,
}

pub struct ModernBertModel<'a> {
    pub cfg: ModernBertConfig,
    pub quant: Quant,
    word: Table<'a>,
    emb_norm: LayerNorm,
    layers: Vec<ModernBertLayer<'a>>,
    final_norm: LayerNorm,
    act: fn(f32) -> f32,
}

impl<'a> ModernBertModel<'a> {
    pub fn load(cfg: ModernBertConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let prefix = ["", "model."]
            .into_iter()
            .find(|p| files.has(&format!("{p}embeddings.tok_embeddings.weight")))
            .context("cannot find embeddings.tok_embeddings in checkpoint")?;
        let (h, inter, nb) = (cfg.hidden_size, cfg.intermediate_size, cfg.norm_bias);
        let n = |s: &str| format!("{prefix}{s}");
        let word = Table::load(files, &n("embeddings.tok_embeddings.weight"), cfg.vocab_size, h, Quant::Bf16)?;
        let emb_norm = LayerNorm::load(files, &n("embeddings.norm"), h, nb)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = |s: &str| n(&format!("layers.{l}.{s}"));
            let lin = |s: &str, o: usize, i: usize| load_lin(files, &p(s), o, i, quant);
            layers.push(ModernBertLayer {
                attn_norm: if l == 0 {
                    None
                } else {
                    Some(LayerNorm::load(files, &p("attn_norm"), h, nb)?)
                },
                wqkv: lin("attn.Wqkv.weight", 3 * h, h)?,
                wo: lin("attn.Wo.weight", h, h)?,
                mlp_norm: LayerNorm::load(files, &p("mlp_norm"), h, nb)?,
                wi: lin("mlp.Wi.weight", 2 * inter, h)?,
                mlp_wo: lin("mlp.Wo.weight", h, inter)?,
                local: cfg.is_local(l),
            });
        }
        let final_norm = LayerNorm::load(files, &n("final_norm"), h, nb)?;
        let act = act_fn(&cfg.hidden_activation)?;
        Ok(Self {
            cfg,
            quant,
            word,
            emb_norm,
            layers,
            final_norm,
            act,
        })
    }

    /// Last hidden states `[t, hidden]` for one sequence (special tokens included).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, nh, inter) = (cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size);
        let hd = h / nh;
        let t = tokens.len().min(cfg.max_position_embeddings);
        let tokens = &tokens[..t];
        let eps = cfg.norm_eps as f32;
        let mut e = vec![0f32; t * h];
        for (i, row) in e.chunks_mut(h).enumerate() {
            self.word.row(tokens[i] as usize, row);
        }
        let mut x = vec![0f32; t * h];
        self.emb_norm.apply_rows(&e, &mut x, eps);

        // RoPE tables for the global and local thetas.
        let pos: Vec<usize> = (0..t).collect();
        let global = rope_tables(&pos, &rope_inv_freq(hd, cfg.global_rope_theta, 1.0));
        let local = rope_tables(&pos, &rope_inv_freq(hd, cfg.local_rope_theta.unwrap_or(cfg.global_rope_theta), 1.0));
        // The reference masks keys more than `local_attention / 2` away (inclusive bound);
        // `attention_masked` keeps |q - k| < window.
        let window = cfg.local_attention / 2 + 1;
        let half = hd / 2;
        let scale = (hd as f32).powf(-0.5);

        let mut xn = vec![0f32; t * h];
        let mut qkv = vec![0f32; t * 3 * h];
        let (mut q, mut k, mut v) = (vec![0f32; t * h], vec![0f32; t * h], vec![0f32; t * h]);
        let mut att = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        let mut u = vec![0f32; t * 2 * inter];
        let mut g = vec![0f32; t * inter];
        for layer in &self.layers {
            let input = match &layer.attn_norm {
                Some(ln) => {
                    ln.apply_rows(&x, &mut xn, eps);
                    &xn
                }
                None => &x,
            };
            linear(&layer.wqkv, 3 * h, h, input, t, &mut qkv);
            let (cos, sin) = if layer.local { &local } else { &global };
            for ti in 0..t {
                let src = &qkv[ti * 3 * h..(ti + 1) * 3 * h];
                let r = ti * h..(ti + 1) * h;
                q[r.clone()].copy_from_slice(&src[..h]);
                k[r.clone()].copy_from_slice(&src[h..2 * h]);
                v[r].copy_from_slice(&src[2 * h..]);
                let (c, s) = (&cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                for hh in 0..nh {
                    let off = ti * h + hh * hd;
                    apply_rope(&mut q[off..off + hd], c, s);
                    apply_rope(&mut k[off..off + hd], c, s);
                }
            }
            scale_inplace(&mut q, scale);
            let mut cache = LayerCache::new(nh, hd);
            cache.push(&k, &v, t);
            attention_masked(&q, &cache, nh, t, 0, layer.local.then_some(window), false, &mut att);
            linear(&layer.wo, h, h, &att, t, &mut o);
            add_inplace(&mut x, &o);

            layer.mlp_norm.apply_rows(&x, &mut xn, eps);
            linear(&layer.wi, 2 * inter, h, &xn, t, &mut u);
            let act = self.act;
            for_chunks(&mut g, inter, t > 4, |i, gr| {
                let (a, gate) = u[i * 2 * inter..(i + 1) * 2 * inter].split_at(inter);
                for ((y, &a), &b) in gr.iter_mut().zip(a).zip(gate) {
                    *y = act(a) * b;
                }
            });
            linear(&layer.mlp_wo, h, inter, &g, t, &mut o);
            add_inplace(&mut x, &o);
        }
        self.final_norm.apply_rows(&x, &mut e, eps);
        e
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn local_layers_follow_global_attn_every_n_layers() {
        let cfg: super::ModernBertConfig = serde_json::from_str(
            r#"{"hidden_size": 8, "num_hidden_layers": 7, "num_attention_heads": 2, "intermediate_size": 8,
                "vocab_size": 4, "max_position_embeddings": 16}"#,
        )
        .unwrap();
        let local: Vec<bool> = (0..7).map(|l| cfg.is_local(l)).collect();
        assert_eq!(local, [false, true, true, false, true, true, false]);
    }
}
