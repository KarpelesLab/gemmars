//! Gemma 4 text decoder: embeddings, per-layer embeddings (PLE), sliding/global
//! attention with KV sharing, gated-GELU MLP, tied softcapped LM head.

use crate::config::TextConfig;
use crate::kernels::*;
use crate::prof::{self, Cat};
use crate::quant::{quantize_bf16, quantize_q4, Q4Matrix, QMatrix};
use crate::safetensors::ModelFiles;
use anyhow::{bail, Context, Result};

/// Weight precision for the streamed (linear) weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quant {
    Bf16,
    Q8,
    /// Q4_0 layer weights; the LM head and PLE projection stay Q8.
    Q4,
}

/// A linear layer's weight `[n_out, n_in]`, either borrowed bf16 or quantized int8.
pub enum Lin<'a> {
    Bf16(&'a [u16]),
    Q8(QMatrix),
    Q4(Q4Matrix),
}

impl Lin<'_> {
    pub fn bytes(&self) -> usize {
        match self {
            Lin::Bf16(w) => w.len() * 2,
            Lin::Q8(m) => m.bytes(),
            Lin::Q4(m) => m.bytes(),
        }
    }
}

/// Fused matmuls over one shared input: all jobs must have the same precision.
fn linear_multi(jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
    if jobs.is_empty() {
        return;
    }
    if jobs.iter().all(|(l, _, _)| matches!(l, Lin::Bf16(_))) {
        let mut v: Vec<(&[u16], usize, &mut [f32])> = jobs
            .iter_mut()
            .map(|(l, n, y)| match l {
                Lin::Bf16(w) => (*w, *n, &mut **y),
                _ => unreachable!(),
            })
            .collect();
        matmul_multi(&mut v, n_in, x, t);
    } else if jobs.iter().all(|(l, _, _)| matches!(l, Lin::Q8(_))) {
        let mut v: Vec<(&QMatrix, &mut [f32])> = jobs
            .iter_mut()
            .map(|(l, n, y)| match l {
                Lin::Q8(m) => {
                    debug_assert_eq!(m.n_out, *n);
                    (m, &mut **y)
                }
                _ => unreachable!(),
            })
            .collect();
        matmul_quant_multi(&mut v, x, t);
    } else if jobs.iter().all(|(l, _, _)| matches!(l, Lin::Q4(_))) {
        let mut v: Vec<(&Q4Matrix, &mut [f32])> = jobs
            .iter_mut()
            .map(|(l, n, y)| match l {
                Lin::Q4(m) => {
                    debug_assert_eq!(m.n_out, *n);
                    (m, &mut **y)
                }
                _ => unreachable!(),
            })
            .collect();
        matmul_quant_multi(&mut v, x, t);
    } else {
        panic!("mixed precision in fused matmul");
    }
}

fn linear(l: &Lin, n_out: usize, n_in: usize, x: &[f32], t: usize, y: &mut [f32]) {
    linear_multi(&mut [(l, n_out, y)], n_in, x, t);
}

struct PleLayer<'a> {
    gate: Lin<'a>,   // [P, H]
    proj: Lin<'a>,   // [H, P]
    norm: &'a [u16], // [H]
}

struct PleGlobal<'a> {
    embed: &'a [u16], // [V, L*P]
    proj: Lin<'a>,    // [L*P, H]
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
    q: Lin<'a>,
    k: Option<Lin<'a>>,
    v: Option<Lin<'a>>,
    o: Lin<'a>,
    q_norm: &'a [u16],
    k_norm: Option<&'a [u16]>,
    ln_in: &'a [u16],
    ln_post_attn: &'a [u16],
    ln_pre_ffn: &'a [u16],
    ln_post_ffn: &'a [u16],
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    ple: Option<PleLayer<'a>>,
    layer_scalar: f32,
}

pub struct Model<'a> {
    pub cfg: TextConfig,
    pub quant: Quant,
    embed: &'a [u16], // [V, H] for lookups
    head: Lin<'a>,    // tied LM head (same values as `embed`)
    ple: Option<PleGlobal<'a>>,
    pub layers: Vec<Layer<'a>>,
    final_norm: &'a [u16],
    inv_freq_sliding: Vec<f32>,
    inv_freq_full: Vec<f32>,
}

#[derive(Clone)]
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
    pub fn k_at(&self, pos: usize, kvh: usize) -> &[f32] {
        let o = (pos * self.n_kv + kvh) * self.hd;
        &self.k[o..o + self.hd]
    }
    #[inline(always)]
    pub fn v_at(&self, pos: usize, kvh: usize) -> &[f32] {
        let o = (pos * self.n_kv + kvh) * self.hd;
        &self.v[o..o + self.hd]
    }
    fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.k.truncate(len * self.n_kv * self.hd);
            self.v.truncate(len * self.n_kv * self.hd);
            self.len = len;
        }
    }
}

/// Mutable decoding state: KV caches and the current position.
#[derive(Clone)]
pub struct Session {
    caches: Vec<Option<LayerCache>>,
    pub pos: usize,
}

impl Session {
    /// Drops cached positions at or beyond `pos` (rewinding after rejected speculative tokens).
    pub fn truncate(&mut self, pos: usize) {
        for c in self.caches.iter_mut().flatten() {
            c.truncate(pos);
        }
        self.pos = self.pos.min(pos);
    }

    /// The KV cache owned by `layer` (which must not be a KV-sharing layer).
    pub fn cache(&self, layer: usize) -> &LayerCache {
        self.caches[layer].as_ref().expect("layer has no own KV cache")
    }
}

impl<'a> Model<'a> {
    pub fn load(cfg: TextConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
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
        // A streamed linear weight: borrowed bf16 or quantized on load.
        let lin_with = |name: &str, n_out: usize, n_in: usize, q: Quant| -> Result<Lin<'a>> {
            let w = get(name, &[n_out, n_in])?;
            Ok(match q {
                Quant::Bf16 => Lin::Bf16(w),
                Quant::Q8 => Lin::Q8(quantize_bf16(w, n_out, n_in)),
                Quant::Q4 => Lin::Q4(quantize_q4(w, n_out, n_in)),
            })
        };
        let lin = |name: &str, n_out: usize, n_in: usize| lin_with(name, n_out, n_in, quant);
        // The PLE projection feeds every layer directly: keep it at Q8 under Q4. The LM head follows
        // GEMMARS_HEAD_Q8=1 (Q8) or defaults to the model precision.
        let hi_quant = if quant == Quant::Q4 { Quant::Q8 } else { quant };
        let head_quant = if quant == Quant::Q4 && std::env::var("GEMMARS_HEAD_Q8").is_ok() { Quant::Q8 } else { quant };

        let embed = get("embed_tokens.weight", &[cfg.vocab_size, h])?;
        let head = lin_with("embed_tokens.weight", cfg.vocab_size, h, head_quant)?;
        let final_norm = get("norm.weight", &[h])?;
        let ple = if cfg.has_ple() {
            Some(PleGlobal {
                embed: get("embed_tokens_per_layer.weight", &[cfg.vocab_size_per_layer_input, nl * p])?,
                proj: lin_with("per_layer_model_projection.weight", nl * p, h, hi_quant)?,
                norm: get("per_layer_projection_norm.weight", &[p])?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let lp = format!("layers.{l}.");
            let g = |n: &str, s: &[usize]| get(&format!("{lp}{n}"), s);
            let gl = |n: &str, n_out: usize, n_in: usize| lin(&format!("{lp}{n}"), n_out, n_in);
            let sliding = cfg.is_sliding(l);
            let hd = cfg.layer_head_dim(l);
            let n_kv = cfg.layer_n_kv(l);
            let own_kv = !cfg.is_kv_shared(l);
            let k_eq_v = cfg.layer_k_eq_v(l);
            let inter = cfg.layer_intermediate(l);
            let (k, v, k_norm) = if own_kv {
                (
                    Some(gl("self_attn.k_proj.weight", n_kv * hd, h)?),
                    if k_eq_v { None } else { Some(gl("self_attn.v_proj.weight", n_kv * hd, h)?) },
                    Some(g("self_attn.k_norm.weight", &[hd])?),
                )
            } else {
                (None, None, None)
            };
            let ple_l = if cfg.has_ple() {
                Some(PleLayer {
                    gate: gl("per_layer_input_gate.weight", p, h)?,
                    proj: gl("per_layer_projection.weight", h, p)?,
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
                q: gl("self_attn.q_proj.weight", nh * hd, h)?,
                k,
                v,
                o: gl("self_attn.o_proj.weight", h, nh * hd)?,
                q_norm: g("self_attn.q_norm.weight", &[hd])?,
                k_norm,
                ln_in: g("input_layernorm.weight", &[h])?,
                ln_post_attn: g("post_attention_layernorm.weight", &[h])?,
                ln_pre_ffn: g("pre_feedforward_layernorm.weight", &[h])?,
                ln_post_ffn: g("post_feedforward_layernorm.weight", &[h])?,
                gate: gl("mlp.gate_proj.weight", inter, h)?,
                up: gl("mlp.up_proj.weight", inter, h)?,
                down: gl("mlp.down_proj.weight", h, inter)?,
                ple: ple_l,
                layer_scalar,
            });
        }

        let rs = cfg.rope_spec(true);
        let rf = cfg.rope_spec(false);
        let inv_freq_sliding = rope_inv_freq(cfg.head_dim, rs.theta, rs.proportion);
        let inv_freq_full = rope_inv_freq(cfg.global_head_dim, rf.theta, rf.proportion);

        Ok(Self { cfg, quant, embed, head, ple, layers, final_norm, inv_freq_sliding, inv_freq_full })
    }

    /// Bytes of linear weights streamed per decoded token.
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.head.bytes();
        if let Some(p) = &self.ple {
            n += p.proj.bytes();
        }
        for l in &self.layers {
            n += l.q.bytes() + l.o.bytes() + l.gate.bytes() + l.up.bytes() + l.down.bytes();
            n += l.k.as_ref().map_or(0, |x| x.bytes()) + l.v.as_ref().map_or(0, |x| x.bytes());
            if let Some(p) = &l.ple {
                n += p.gate.bytes() + p.proj.bytes();
            }
        }
        n
    }

    pub fn new_session(&self) -> Session {
        let caches = self
            .layers
            .iter()
            .map(|l| if l.own_kv { Some(LayerCache::new(l.n_kv, l.hd)) } else { None })
            .collect();
        Session { caches, pos: 0 }
    }

    /// The main model's input embedding of `tok`, scaled by sqrt(hidden) (as fed to the drafter).
    pub fn embed_scaled(&self, tok: u32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let scale = (h as f32).sqrt();
        self.embed[tok as usize * h..(tok as usize + 1) * h].iter().map(|&s| bf16_to_f32(s) * scale).collect()
    }

    /// Softcapped logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * cfg.vocab_size];
        linear(&self.head, cfg.vocab_size, cfg.hidden_size, hidden, t, &mut logits);
        if let Some(cap) = cfg.final_logit_softcapping {
            let cap = cap as f32;
            for_chunks(&mut logits, 8192, true, |_, c| {
                for v in c.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
            });
        }
        logits
    }

    /// Runs `tokens` through the model starting at `sess.pos`, appending to the KV
    /// cache, and returns the (softcapped) logits for the last token.
    pub fn forward(&self, sess: &mut Session, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden(sess, tokens);
        self.logits(&hs[(t - 1) * h..], 1)
    }

    /// Runs `tokens` through the decoder and returns the final-normed hidden states `[t, hidden]`.
    pub fn forward_hidden(&self, sess: &mut Session, tokens: &[u32]) -> Vec<f32> {
        self.forward_hidden_mm(sess, tokens, &[])
    }

    /// Like `forward_hidden`, but positions listed in `mm` (relative to this call, sorted) take
    /// the given embedding (a multimodal soft token in text space) instead of the token
    /// embedding; their per-layer-embedding lookup uses the pad token, as in the reference.
    pub fn forward_hidden_mm(&self, sess: &mut Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
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
        // Multimodal soft tokens replace the placeholder embeddings; PLE uses the pad token there.
        let mut ple_tokens: Vec<u32> = tokens.to_vec();
        for &(i, emb) in mm {
            debug_assert!(i < t && emb.len() == h);
            x[i * h..(i + 1) * h].copy_from_slice(emb);
            ple_tokens[i] = 0;
        }
        let tokens = &ple_tokens[..];

        // Per-layer embeddings: [t, nl, p]
        let pli: Option<Vec<f32>> = self.ple.as_ref().map(|ple| {
            let p = cfg.hidden_size_per_layer_input;
            let mut proj = vec![0f32; t * nl * p];
            linear(&ple.proj, nl * p, h, &x, t, &mut proj);
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

        // Final norm.
        rmsnorm_rows_inplace(&mut x, h, Some(self.final_norm), eps);
        sess.pos += t;
        x
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
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&layer.q, nh * hd, &mut q[..])];
            if layer.own_kv {
                jobs.push((layer.k.as_ref().unwrap(), n_kv * hd, &mut k[..]));
                if !layer.k_eq_v {
                    jobs.push((layer.v.as_ref().unwrap(), n_kv * hd, &mut v[..]));
                }
            }
            linear_multi(&mut jobs, h, &hn, t);
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
        linear(&layer.o, h, nh * hd, &att, t, &mut o);
        rmsnorm_rows_inplace(&mut o, h, Some(layer.ln_post_attn), eps);
        add_inplace(x, &o);

        // ---- MLP ----
        rmsnorm_rows(x, h, Some(layer.ln_pre_ffn), eps, &mut hn);
        let inter = layer.inter;
        let mut g = vec![0f32; t * inter];
        let mut u = vec![0f32; t * inter];
        linear_multi(&mut [(&layer.gate, inter, &mut g[..]), (&layer.up, inter, &mut u[..])], h, &hn, t);
        gelu_mul(&mut g, &u);
        let mut d = o; // reuse buffer [t, h]
        linear(&layer.down, h, inter, &g, t, &mut d);
        rmsnorm_rows_inplace(&mut d, h, Some(layer.ln_post_ffn), eps);
        add_inplace(x, &d);

        // ---- Per-layer embedding injection ----
        if let (Some(pw), Some(pli)) = (&layer.ple, pli) {
            let p = cfg.hidden_size_per_layer_input;
            let nl = cfg.num_hidden_layers;
            let mut gp = vec![0f32; t * p];
            linear(&pw.gate, p, h, x, t, &mut gp);
            for_chunks(&mut gp, p, par, |ti, row| {
                let inp = &pli[(ti * nl + l) * p..(ti * nl + l + 1) * p];
                for (r, &i) in row.iter_mut().zip(inp) {
                    *r = gelu_tanh(*r) * i;
                }
            });
            let mut pr = d;
            linear(&pw.proj, h, p, &gp, t, &mut pr);
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
pub(crate) fn attention(q: &[f32], cache: &LayerCache, nh: usize, t: usize, start: usize, window: Option<usize>, out: &mut [f32]) {
    let _timer = prof::start(Cat::Attention);
    let hd = cache.hd;
    let n_kv = cache.n_kv;
    let rep = nh / n_kv;
    let threads = crate::pool::threads();
    let par = t > 1 || cache.len > 32;
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
