//! Gemma 4 text-model configuration, parsed from HuggingFace `config.json`.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

fn d256() -> usize {
    256
}
fn d512() -> usize {
    512
}
fn d_eps() -> f64 {
    1e-6
}
fn d_maxpos() -> usize {
    131_072
}
fn d_vocab() -> usize {
    262_144
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "d256")]
    pub head_dim: usize,
    #[serde(default = "d512")]
    pub global_head_dim: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size_per_layer_input: usize,
    #[serde(default = "d256")]
    pub hidden_size_per_layer_input: usize,
    #[serde(default = "d512")]
    pub sliding_window: usize,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, RopeParams>>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default = "d_maxpos")]
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct RopeSpec {
    pub theta: f64,
    /// Fraction of the head dim that is rotated (1.0 = full RoPE).
    pub proportion: f64,
}

impl TextConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )?;
        let text = raw.get("text_config").cloned().unwrap_or(raw);
        let mut cfg: TextConfig = serde_json::from_value(text).context("parse text config")?;
        cfg.finalize()?;
        Ok(cfg)
    }

    fn finalize(&mut self) -> Result<()> {
        if self.layer_types.is_none() {
            // Default 5:1 pattern.
            self.layer_types = Some(
                (0..self.num_hidden_layers)
                    .map(|i| if (i + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" }.to_string())
                    .collect(),
            );
        }
        let lt = self.layer_types.as_mut().unwrap();
        if lt.len() != self.num_hidden_layers {
            bail!("layer_types has {} entries, expected {}", lt.len(), self.num_hidden_layers);
        }
        if let Some(last) = lt.last_mut() {
            *last = "full_attention".to_string();
        }
        if self.enable_moe_block {
            bail!("MoE models (26B-A4B) are not supported yet");
        }
        Ok(())
    }

    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types.as_ref().unwrap()[layer] == "sliding_attention"
    }

    pub fn layer_head_dim(&self, layer: usize) -> usize {
        if self.is_sliding(layer) { self.head_dim } else { self.global_head_dim }
    }

    pub fn layer_n_kv(&self, layer: usize) -> usize {
        if !self.is_sliding(layer) && self.attention_k_eq_v {
            if let Some(n) = self.num_global_key_value_heads {
                return n;
            }
        }
        self.num_key_value_heads
    }

    /// Global layers reuse K as V when `attention_k_eq_v` is set.
    pub fn layer_k_eq_v(&self, layer: usize) -> bool {
        !self.is_sliding(layer) && self.attention_k_eq_v
    }

    pub fn first_kv_shared_layer(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers
    }

    pub fn is_kv_shared(&self, layer: usize) -> bool {
        self.num_kv_shared_layers > 0 && layer >= self.first_kv_shared_layer()
    }

    /// Index of the layer whose KV cache this layer reads from.
    pub fn kv_source(&self, layer: usize) -> usize {
        if !self.is_kv_shared(layer) {
            return layer;
        }
        let first = self.first_kv_shared_layer();
        let sliding = self.is_sliding(layer);
        (0..first)
            .rev()
            .find(|&l| self.is_sliding(l) == sliding)
            .expect("no non-shared layer of matching type")
    }

    /// Layers whose K/V are the model's "shared KV states" (consumed by KV-sharing layers and
    /// by the drafter): the last non-shared layer of each type, as (sliding, global).
    pub fn shared_kv_layers(&self) -> (usize, usize) {
        let first = self.first_kv_shared_layer();
        let s = (0..first).rev().find(|&l| self.is_sliding(l)).expect("no sliding layer");
        let g = (0..first).rev().find(|&l| !self.is_sliding(l)).expect("no global layer");
        (s, g)
    }

    pub fn layer_intermediate(&self, layer: usize) -> usize {
        if self.use_double_wide_mlp && self.is_kv_shared(layer) {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }

    pub fn rope_spec(&self, sliding: bool) -> RopeSpec {
        let key = if sliding { "sliding_attention" } else { "full_attention" };
        match self.rope_parameters.as_ref().and_then(|m| m.get(key)) {
            Some(p) => RopeSpec {
                theta: p.rope_theta,
                proportion: if p.rope_type.as_deref() == Some("proportional") {
                    p.partial_rotary_factor.unwrap_or(1.0)
                } else {
                    1.0
                },
            },
            None => {
                if sliding {
                    RopeSpec { theta: 10_000.0, proportion: 1.0 }
                } else {
                    RopeSpec { theta: 1_000_000.0, proportion: 0.25 }
                }
            }
        }
    }

    pub fn has_ple(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }
}

/// Stop tokens from `generation_config.json` (falls back to `<eos>` and `<turn|>`).
pub fn load_eos_ids(dir: &Path) -> Vec<u32> {
    let path = dir.join("generation_config.json");
    if let Ok(b) = std::fs::read(&path) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&b) {
            match &v["eos_token_id"] {
                serde_json::Value::Number(n) => return vec![n.as_u64().unwrap() as u32],
                serde_json::Value::Array(a) => {
                    return a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect();
                }
                _ => {}
            }
        }
    }
    vec![1, 106]
}

/// `model_type` from the root of config.json (e.g. "gemma4", "qwen3_5").
pub fn model_type(dir: &Path) -> Result<String> {
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    raw.get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| raw.get("text_config").and_then(|t| t.get("model_type")).and_then(|v| v.as_str()).map(|s| s.to_string()))
        .context("config.json has no model_type")
}

fn d4() -> usize {
    4
}
fn d128() -> usize {
    128
}
fn d16() -> usize {
    16
}
fn d32() -> usize {
    32
}
fn d_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenRope {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

/// Qwen3.5 / Qwen3.8 dense hybrid text config (Gated DeltaNet + gated full attention).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct QwenConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "d256")]
    pub head_dim: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "d4")]
    pub full_attention_interval: usize,
    #[serde(default = "d4")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "d128")]
    pub linear_key_head_dim: usize,
    #[serde(default = "d128")]
    pub linear_value_head_dim: usize,
    #[serde(default = "d16")]
    pub linear_num_key_heads: usize,
    #[serde(default = "d32")]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub rope_parameters: Option<QwenRope>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "d_true")]
    pub attn_output_gate: bool,
}

impl QwenConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let text = raw.get("text_config").cloned().unwrap_or(raw);
        let mut cfg: QwenConfig = serde_json::from_value(text).context("parse qwen text config")?;
        if cfg.layer_types.is_none() {
            let k = cfg.full_attention_interval.max(1);
            cfg.layer_types = Some(
                (0..cfg.num_hidden_layers)
                    .map(|i| if (i + 1) % k == 0 { "full_attention" } else { "linear_attention" }.to_string())
                    .collect(),
            );
        }
        if cfg.layer_types.as_ref().unwrap().len() != cfg.num_hidden_layers {
            bail!("layer_types length mismatch");
        }
        Ok(cfg)
    }
    pub fn is_full(&self, l: usize) -> bool {
        self.layer_types.as_ref().unwrap()[l] == "full_attention"
    }
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref().map(|r| r.rope_theta).unwrap_or(10_000_000.0)
    }
    /// Number of rotated dimensions per head (partial RoPE).
    pub fn rotary_dim(&self) -> usize {
        let f = self.rope_parameters.as_ref().and_then(|r| r.partial_rotary_factor).or(self.partial_rotary_factor).unwrap_or(1.0);
        (self.head_dim as f64 * f) as usize
    }
}
