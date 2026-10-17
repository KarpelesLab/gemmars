//! Gemma 4 text decoder: embeddings, per-layer embeddings (PLE), sliding/global
//! attention with KV sharing, gated-GELU MLP, tied softcapped LM head.

use crate::config::TextConfig;
use crate::kernels::*;
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{bail, Context, Result};

struct PleLayer<'a> {
    gate: &'a [u16], // [P, H]
    proj: &'a [u16], // [H, P]
    norm: &'a [u16], // [H]
}

struct PleGlobal<'a> {
    embed: &'a [u16], // [V, L*P]
    proj: &'a [u16],  // [L*P, H]
    norm: &'a [u16],  // [P]
}

pub struct Layer<'a> {
    pub sliding: bool,
    pub hd: usize,
    pub n_kv: usize,
    pub kv_src: usize,
    pub own_kv: bool,
    pub k_eq_v: bool,
    pub inter: usize,
    q: &'a [u16],
    k: Option<&'a [u16]>,
    v: Option<&'a [u16]>,
    o: &'a [u16],
    q_norm: &'a [u16],
    k_norm: Option<&'a [u16]>,
    ln_in: &'a [u16],
    ln_post_attn: &'a [u16],
    ln_pre_ffn: &'a [u16],
    ln_post_ffn: &'a [u16],
    gate: &'a [u16],
    up: &'a [u16],
    down: &'a [u16],
    ple: Option<PleLayer<'a>>,
    layer_scalar: f32,
}

pub struct Model<'a> {
    pub cfg: TextConfig,
    embed: &'a [u16], // [V, H], also the LM head
    ple: Option<PleGlobal<'a>>,
    pub layers: Vec<Layer<'a>>,
    final_norm: &'a [u16],
    inv_freq_sliding: Vec<f32>,
    inv_freq_full: Vec<f32>,
}

pub struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    pub n_kv: usize,
    pub hd: usize,
    pub len: usize,
}

impl LayerCache {
    fn new(n_kv: usize, hd: usize) -> Self {
        Self { k: Vec::new(), v: Vec::new(), n_kv, hd, len: 0 }
    }
    fn push(&mut self, k: &[f32], v: &[f32], t: usize) {
        debug_assert_eq!(k.len(), t * self.n_kv * self.hd);
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
        self.len += t;
    }
    #[inline(always)]
    fn k_at(&self, pos: usize, kvh: usize) -> &[f32] {
        let o = (pos * self.n_kv + kvh) * self.hd;
        &self.k[o..o + self.hd]
    }
    #[inline(always)]
    fn v_at(&self, pos: usize, kvh: usize) -> &[f32] {
        let o = (pos * self.n_kv + kvh) * self.hd;
        &self.v[o..o + self.hd]
    }
    #[allow(dead_code)]
    fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.k.truncate(len * self.n_kv * self.hd);
            self.v.truncate(len * self.n_kv * self.hd);
            self.len = len;
        }
    }
}

/// Mutable decoding state: KV caches and the current position.
#[allow(dead_code)]
pub struct Session {
    caches: Vec<Option<LayerCache>>,
    pub pos: usize,
}

#[allow(dead_code)]
impl Session {
    /// Drops cached positions at or beyond `pos` (for rewinding a conversation).
    pub fn truncate(&mut self, pos: usize) {
        for c in self.caches.iter_mut().flatten() {
            c.truncate(pos);
        }
        self.pos = self.pos.min(pos);
    }
}

impl<'a> Model<'a> {
    pub fn load(cfg: TextConfig, files: &'a ModelFiles) -> Result<Self> {
        let prefix = if files.has("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if files.has("model.embed_tokens.weight") {
            "model."
        } else {
            bail!("cannot find embed_tokens in checkpoint");
        };
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nl = cfg.num_hidden_layers;
        let p = cfg.hidden_size_per_layer_input;

        let get = |name: &str, shape: &[usize]| -> Result<&'a [u16]> {
            let full = format!("{prefix}{name}");
            let info = files.info(&full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape != shape {
                bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
            }
            files.bf16(&full)
        };

        let embed = get("embed_tokens.weight", &[cfg.vocab_size, h])?;
        let final_norm = get("norm.weight", &[h])?;
        let ple = if cfg.has_ple() {
            Some(PleGlobal {
                embed: get("embed_tokens_per_layer.weight", &[cfg.vocab_size_per_layer_input, nl * p])?,
                proj: get("per_layer_model_projection.weight", &[nl * p, h])?,
                norm: get("per_layer_projection_norm.weight", &[p])?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let lp = format!("layers.{l}.");
            let g = |n: &str, s: &[usize]| get(&format!("{lp}{n}"), s);
            let sliding = cfg.is_sliding(l);
            let hd = cfg.layer_head_dim(l);
            let n_kv = cfg.layer_n_kv(l);
            let own_kv = !cfg.is_kv_shared(l);
            let k_eq_v = cfg.layer_k_eq_v(l);
            let inter = cfg.layer_intermediate(l);
            let (k, v, k_norm) = if own_kv {
                (
                    Some(g("self_attn.k_proj.weight", &[n_kv * hd, h])?),
                    if k_eq_v { None } else { Some(g("self_attn.v_proj.weight", &[n_kv * hd, h])?) },
                    Some(g("self_attn.k_norm.weight", &[hd])?),
                )
            } else {
                (None, None, None)
            };
            let ple_l = if cfg.has_ple() {
                Some(PleLayer {
                    gate: g("per_layer_input_gate.weight", &[p, h])?,
                    proj: g("per_layer_projection.weight", &[h, p])?,
                    norm: g("post_per_layer_input_norm.weight", &[h])?,
                })
            } else {
                None
            };
            let layer_scalar = match g("layer_scalar", &[1]) {
                Ok(s) => bf16_to_f32(s[0]),
                Err(_) => 1.0,
            };
            layers.push(Layer {
                sliding,
                hd,
                n_kv,
                kv_src: cfg.kv_source(l),
                own_kv,
                k_eq_v,
                inter,
                q: g("self_attn.q_proj.weight", &[nh * hd, h])?,
                k,
                v,
                o: g("self_attn.o_proj.weight", &[h, nh * hd])?,
                q_norm: g("self_attn.q_norm.weight", &[hd])?,
                k_norm,
                ln_in: g("input_layernorm.weight", &[h])?,
                ln_post_attn: g("post_attention_layernorm.weight", &[h])?,
                ln_pre_ffn: g("pre_feedforward_layernorm.weight", &[h])?,
                ln_post_ffn: g("post_feedforward_layernorm.weight", &[h])?,
                gate: g("mlp.gate_proj.weight", &[inter, h])?,
                up: g("mlp.up_proj.weight", &[inter, h])?,
                down: g("mlp.down_proj.weight", &[h, inter])?,
                ple: ple_l,
                layer_scalar,
            });
        }

        let rs = cfg.rope_spec(true);
        let rf = cfg.rope_spec(false);
        let inv_freq_sliding = rope_inv_freq(cfg.head_dim, rs.theta, rs.proportion);
        let inv_freq_full = rope_inv_freq(cfg.global_head_dim, rf.theta, rf.proportion);

        Ok(Self { cfg, embed, ple, layers, final_norm, inv_freq_sliding, inv_freq_full })
    }

    pub fn new_session(&self) -> Session {
        let caches = self
            .layers
            .iter()
            .map(|l| if l.own_kv { Some(LayerCache::new(l.n_kv, l.hd)) } else { None })
            .collect();
        Session { caches, pos: 0 }
    }

    /// Runs `tokens` through the model starting at `sess.pos`, appending to the KV
    /// cache, and returns the (softcapped) logits for the last token.
    pub fn forward(&self, sess: &mut Session, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nl = cfg.num_hidden_layers;
        let start = sess.pos;
        let eps = cfg.rms_norm_eps as f32;

        let par = t > 1;
        let timer = prof::start(Cat::Embed);
        // Token embeddings, scaled by sqrt(hidden).
        let mut x = vec![0f32; t * h];
        let scale = (h as f32).sqrt();
        for_chunks(&mut x, h, par, |ti, row| {
            let tok = tokens[ti] as usize;
            let src = &self.embed[tok * h..(tok + 1) * h];
            for (r, &s) in row.iter_mut().zip(src) {
                *r = bf16_to_f32(s) * scale;
            }
        });

        // Per-layer embeddings: [t, nl, p]
        let pli: Option<Vec<f32>> = self.ple.as_ref().map(|ple| {
            let p = cfg.hidden_size_per_layer_input;
            let mut proj = vec![0f32; t * nl * p];
            matmul(ple.proj, nl * p, h, &x, t, &mut proj);
            let s_proj = (h as f32).powf(-0.5);
            let s_emb = (p as f32).sqrt();
            let s_out = std::f32::consts::FRAC_1_SQRT_2;
            for_chunks(&mut proj, nl * p, par, |ti, row| {
                let tok = tokens[ti] as usize;
                let emb = &ple.embed[tok * nl * p..(tok + 1) * nl * p];
                for l in 0..nl {
                    let chunk = &mut row[l * p..(l + 1) * p];
                    scale_inplace(chunk, s_proj);
                    rmsnorm_inplace(chunk, Some(ple.norm), eps);
                    for (j, c) in chunk.iter_mut().enumerate() {
                        *c = (*c + bf16_to_f32(emb[l * p + j]) * s_emb) * s_out;
                    }
                }
            });
            proj
        });

        let positions: Vec<usize> = (start..start + t).collect();
        let rope_s = rope_tables(&positions, &self.inv_freq_sliding);
        let rope_f = rope_tables(&positions, &self.inv_freq_full);
        drop(timer);

        for (l, layer) in self.layers.iter().enumerate() {
            let rope = if layer.sliding { &rope_s } else { &rope_f };
            self.layer_forward(l, layer, &mut x, t, start, pli.as_deref(), sess, rope);
        }

        // Final norm + LM head on the last token only.
        let _timer = prof::start(Cat::Head);
        let mut last = vec![0f32; h];
        rmsnorm(&x[(t - 1) * h..], Some(self.final_norm), eps, &mut last);
        let mut logits = vec![0f32; cfg.vocab_size];
        matmul(self.embed, cfg.vocab_size, h, &last, 1, &mut logits);
        if let Some(cap) = cfg.final_logit_softcapping {
            let cap = cap as f32;
            for_chunks(&mut logits, 8192, true, |_, c| {
                for v in c.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
            });
        }
        sess.pos += t;
        logits
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        l: usize,
        layer: &Layer,
        x: &mut [f32],
        t: usize,
        start: usize,
        pli: Option<&[f32]>,
        sess: &mut Session,
        rope: &(Vec<f32>, Vec<f32>),
    ) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let hd = layer.hd;
        let n_kv = layer.n_kv;
        let eps = cfg.rms_norm_eps as f32;
        let half = hd / 2;
        let (cos, sin) = rope;
        let par = t > 1;

        // ---- Attention ----
        let mut hn = vec![0f32; t * h];
        rmsnorm_rows(x, h, Some(layer.ln_in), eps, &mut hn);

        // Q (and K, V when this layer owns its cache) share the input: one fused parallel region.
        let mut q = vec![0f32; t * nh * hd];
        let mut k = vec![0f32; if layer.own_kv { t * n_kv * hd } else { 0 }];
        let mut v = vec![0f32; if layer.own_kv && !layer.k_eq_v { t * n_kv * hd } else { 0 }];
        {
            let mut jobs: Vec<(&[u16], usize, &mut [f32])> = vec![(layer.q, nh * hd, &mut q[..])];
            if layer.own_kv {
                jobs.push((layer.k.unwrap(), n_kv * hd, &mut k[..]));
                if !layer.k_eq_v {
                    jobs.push((layer.v.unwrap(), n_kv * hd, &mut v[..]));
                }
            }
            matmul_multi(&mut jobs, h, &hn, t);
        }
        for_chunks(&mut q, hd, par, |i, qh| {
            let ti = i / nh;
            rmsnorm_inplace(qh, Some(layer.q_norm), eps);
            apply_rope(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
        });

        if layer.own_kv {
            if layer.k_eq_v {
                v = k.clone();
            }
            for_chunks(&mut v, hd, par, |_, vh| rmsnorm_inplace(vh, None, eps));
            let k_norm = layer.k_norm.unwrap();
            for_chunks(&mut k, hd, par, |i, kh| {
                let ti = i / n_kv;
                rmsnorm_inplace(kh, Some(k_norm), eps);
                apply_rope(kh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            });
            sess.caches[l].as_mut().unwrap().push(&k, &v, t);
        }

        let cache = sess.caches[layer.kv_src].as_ref().expect("kv source cache");
        let window = if layer.sliding { Some(cfg.sliding_window) } else { None };
        let mut att = vec![0f32; t * nh * hd];
        attention(&q, cache, nh, t, start, window, &mut att);

        let mut o = vec![0f32; t * h];
        matmul(layer.o, h, nh * hd, &att, t, &mut o);
        rmsnorm_rows_inplace(&mut o, h, Some(layer.ln_post_attn), eps);
        add_inplace(x, &o);

        // ---- MLP ----
        rmsnorm_rows(x, h, Some(layer.ln_pre_ffn), eps, &mut hn);
        let inter = layer.inter;
        let mut g = vec![0f32; t * inter];
        let mut u = vec![0f32; t * inter];
        matmul_multi(&mut [(layer.gate, inter, &mut g[..]), (layer.up, inter, &mut u[..])], h, &hn, t);
        gelu_mul(&mut g, &u);
        let mut d = o; // reuse buffer [t, h]
        matmul(layer.down, h, inter, &g, t, &mut d);
        rmsnorm_rows_inplace(&mut d, h, Some(layer.ln_post_ffn), eps);
        add_inplace(x, &d);

        // ---- Per-layer embedding injection ----
        if let (Some(pw), Some(pli)) = (&layer.ple, pli) {
            let p = cfg.hidden_size_per_layer_input;
            let nl = cfg.num_hidden_layers;
            let mut gp = vec![0f32; t * p];
            matmul(pw.gate, p, h, x, t, &mut gp);
            for_chunks(&mut gp, p, par, |ti, row| {
                let inp = &pli[(ti * nl + l) * p..(ti * nl + l + 1) * p];
                for (r, &i) in row.iter_mut().zip(inp) {
                    *r = gelu_tanh(*r) * i;
                }
            });
            let mut pr = d;
            matmul(pw.proj, h, p, &gp, t, &mut pr);
            rmsnorm_rows_inplace(&mut pr, h, Some(pw.norm), eps);
            add_inplace(x, &pr);
        }

        if layer.layer_scalar != 1.0 {
            let s = layer.layer_scalar;
            for_chunks(x, 4096, par, |_, c| scale_inplace(c, s));
        }
    }
}

/// Causal (optionally sliding-window) GQA attention over the cache.
/// `q`: [t, nh, hd] already normed and rotated. `out`: [t, nh, hd].
/// Attention scaling is 1.0 (Gemma 4 relies on QK-norm).
///
/// Work is split per (token, kv head) so the `rep` query heads sharing a KV head read
/// each key/value once. When that gives too few tasks (short decode), it splits per head.
fn attention(q: &[f32], cache: &LayerCache, nh: usize, t: usize, start: usize, window: Option<usize>, out: &mut [f32]) {
    let _timer = prof::start(Cat::Attention);
    let hd = cache.hd;
    let n_kv = cache.n_kv;
    let rep = nh / n_kv;
    let par = t > 1 || cache.len > 256;
    let threads = crate::pool::threads();
    let group = if t * n_kv >= threads { rep } else { 1 };
    let groups_per_kv = rep / group;
    let ntasks = t * n_kv * groups_per_kv;
    let op = SendPtr(out.as_mut_ptr());
    let run = |task: usize| {
        let ti = task / (n_kv * groups_per_kv);
        let rem = task % (n_kv * groups_per_kv);
        let kvh = rem / groups_per_kv;
        let h0 = kvh * rep + (rem % groups_per_kv) * group;
        let qpos = start + ti;
        let lo = window.map(|w| (qpos + 1).saturating_sub(w)).unwrap_or(0);
        let hi = qpos + 1;
        debug_assert!(hi <= cache.len);
        let n = hi - lo;
        let qbase = (ti * nh + h0) * hd;
        let mut scores = vec![0f32; group * n];
        for j in 0..n {
            let k = cache.k_at(lo + j, kvh);
            for h in 0..group {
                scores[h * n + j] = dot(&q[qbase + h * hd..qbase + (h + 1) * hd], k);
            }
        }
        for h in 0..group {
            softmax_inplace(&mut scores[h * n..(h + 1) * n]);
        }
        // SAFETY: tasks write disjoint [group*hd] ranges of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(qbase), group * hd) };
        o.fill(0.0);
        for j in 0..n {
            let v = cache.v_at(lo + j, kvh);
            for h in 0..group {
                axpy(&mut o[h * hd..(h + 1) * hd], scores[h * n + j], v);
            }
        }
    };
    if par {
        crate::pool::run(ntasks, &run);
    } else {
        for i in 0..ntasks {
            run(i);
        }
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
