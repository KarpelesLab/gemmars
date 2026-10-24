//! Qwen3.5 / Qwen3.8 dense hybrid text model: Gated DeltaNet (linear attention with a causal
//! short conv and a delta-rule recurrent state) interleaved with gated full attention, SiLU MLPs
//! and zero-centered RMSNorm; plain Qwen3 (full attention only) runs on the same code. Text only
//! (the vision encoder and MTP head are not loaded).

use crate::config::QwenConfig;
use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, attention, linear, linear_multi, load_lin, table_quant};
use crate::pool;
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

struct Gdn<'a> {
    qkv: Lin<'a>,         // [2*key_dim + value_dim, H]
    z: Lin<'a>,           // [value_dim, H]
    b: Lin<'a>,           // [n_v, H]
    a: Lin<'a>,           // [n_v, H]
    conv_w: Vec<f32>,     // [conv_dim, K]
    decay_coef: Vec<f32>, // -exp(A_log) per v head
    dt_bias: Vec<f32>,
    norm_w: Vec<f32>, // [hd_v], plain weight (gated norm)
    out: Lin<'a>,     // [H, value_dim]
}

struct Attn<'a> {
    q: Lin<'a>, // [nh*hd*2, H] (query and gate interleaved per head)
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
    q_norm: Vec<f32>, // 1 + w (or w, see `zero_centered_norm`)
    k_norm: Vec<f32>,
}

// One per layer, built once: the size difference is irrelevant.
#[allow(clippy::large_enum_variant)]
enum Mixer<'a> {
    Gdn(Gdn<'a>),
    Attn(Attn<'a>),
}

struct QLayer<'a> {
    mixer: Mixer<'a>,
    ln_in: Vec<f32>,   // 1 + w
    ln_post: Vec<f32>, // 1 + w
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
}

pub struct QwenModel<'a> {
    pub cfg: QwenConfig,
    pub quant: Quant,
    embed: Table<'a>,
    layers: Vec<QLayer<'a>>,
    final_norm: Vec<f32>,
    lm_head: Lin<'a>,
    inv_freq: Vec<f32>,
}

/// Recurrent state of one Gated DeltaNet layer.
pub struct GdnState {
    s: Vec<f32>,    // [n_v, hd_k, hd_v]
    conv: Vec<f32>, // [K-1, conv_dim]: the last K-1 raw conv inputs (oldest first)
}

pub struct QwenSession {
    kv: Vec<Option<LayerCache>>,
    gdn: Vec<Option<GdnState>>,
    pub pos: usize,
}

fn norm_weight(w: &[u16], zero_centered: bool) -> Vec<f32> {
    let o = if zero_centered { 1.0 } else { 0.0 };
    w.iter().map(|&v| o + bf16_to_f32(v)).collect()
}

impl<'a> QwenModel<'a> {
    pub fn load(cfg: QwenConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let has = |n: &str| files.has(n) || files.has(&format!("{n}.q8")) || files.has(&format!("{n}.q4"));
        let prefix = if has("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if has("model.embed_tokens.weight") {
            "model."
        } else {
            bail!("cannot find embed_tokens in checkpoint");
        };
        let h = cfg.hidden_size;
        let get = |full: &str, shape: &[usize]| -> Result<&'a [u16]> {
            let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape != shape {
                bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
            }
            files.bf16(full)
        };
        // Small parameter vectors are stored as BF16 in some checkpoints and F32 in others.
        let getf = |full: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape != shape {
                bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
            }
            let b = files.bytes(full)?;
            Ok(match info.dtype.as_str() {
                "BF16" => b.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect(),
                "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
                other => bail!("tensor {full} has dtype {other}, expected BF16 or F32"),
            })
        };
        let lin = |full: &str, n_out: usize, n_in: usize| load_lin(files, full, n_out, n_in, quant);
        let one_plus = |w: &[u16]| norm_weight(w, cfg.zero_centered_norm);

        let tied = cfg.tie_word_embeddings || !has("lm_head.weight");
        // Lookups use the head's precision when tied (one copy in a pre-quantized file), else bf16.
        let embed = Table::load(
            files,
            &format!("{prefix}embed_tokens.weight"),
            cfg.vocab_size,
            h,
            if tied { quant } else { table_quant(quant) },
        )?;
        let lm_head = if tied {
            lin(&format!("{prefix}embed_tokens.weight"), cfg.vocab_size, h)?
        } else {
            lin("lm_head.weight", cfg.vocab_size, h)?
        };
        let final_norm = one_plus(get(&format!("{prefix}norm.weight"), &[h])?);

        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (nk, nv, hdk, hdv) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let kc = cfg.linear_conv_kernel_dim;
        let inter = cfg.intermediate_size;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}layers.{l}.");
            let mixer = if cfg.is_full(l) {
                let ap = format!("{lp}self_attn.");
                let qw = 2 * nh * hd; // q and gate
                Mixer::Attn(Attn {
                    q: lin(&format!("{ap}q_proj.weight"), if cfg.attn_output_gate { qw } else { nh * hd }, h)?,
                    k: lin(&format!("{ap}k_proj.weight"), nkv * hd, h)?,
                    v: lin(&format!("{ap}v_proj.weight"), nkv * hd, h)?,
                    o: lin(&format!("{ap}o_proj.weight"), h, nh * hd)?,
                    q_norm: one_plus(get(&format!("{ap}q_norm.weight"), &[hd])?),
                    k_norm: one_plus(get(&format!("{ap}k_norm.weight"), &[hd])?),
                })
            } else {
                let gp = format!("{lp}linear_attn.");
                let conv_w = getf(&format!("{gp}conv1d.weight"), &[conv_dim, 1, kc])?;
                let a_log = getf(&format!("{gp}A_log"), &[nv])?;
                Mixer::Gdn(Gdn {
                    qkv: lin(&format!("{gp}in_proj_qkv.weight"), conv_dim, h)?,
                    z: lin(&format!("{gp}in_proj_z.weight"), value_dim, h)?,
                    b: lin(&format!("{gp}in_proj_b.weight"), nv, h)?,
                    a: lin(&format!("{gp}in_proj_a.weight"), nv, h)?,
                    conv_w,
                    decay_coef: a_log.iter().map(|&v| -v.exp()).collect(),
                    dt_bias: getf(&format!("{gp}dt_bias"), &[nv])?,
                    norm_w: getf(&format!("{gp}norm.weight"), &[hdv])?,
                    out: lin(&format!("{gp}out_proj.weight"), h, value_dim)?,
                })
            };
            layers.push(QLayer {
                mixer,
                ln_in: one_plus(get(&format!("{lp}input_layernorm.weight"), &[h])?),
                ln_post: one_plus(get(&format!("{lp}post_attention_layernorm.weight"), &[h])?),
                gate: lin(&format!("{lp}mlp.gate_proj.weight"), inter, h)?,
                up: lin(&format!("{lp}mlp.up_proj.weight"), inter, h)?,
                down: lin(&format!("{lp}mlp.down_proj.weight"), h, inter)?,
            });
        }
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_theta();
        let inv_freq: Vec<f32> = (0..rot / 2)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / rot as f64)) as f32)
            .collect();
        Ok(Self {
            cfg,
            quant,
            embed,
            layers,
            final_norm,
            lm_head,
            inv_freq,
        })
    }

    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        for l in &self.layers {
            n += l.gate.bytes() + l.up.bytes() + l.down.bytes();
            n += match &l.mixer {
                Mixer::Gdn(g) => g.qkv.bytes() + g.z.bytes() + g.b.bytes() + g.a.bytes() + g.out.bytes(),
                Mixer::Attn(a) => a.q.bytes() + a.k.bytes() + a.v.bytes() + a.o.bytes(),
            };
        }
        n
    }

    pub fn new_session(&self) -> QwenSession {
        let cfg = &self.cfg;
        let conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let mut kv = Vec::new();
        let mut gdn = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            if cfg.is_full(l) {
                kv.push(Some(LayerCache::new(cfg.num_key_value_heads, cfg.head_dim)));
                gdn.push(None);
            } else {
                kv.push(None);
                gdn.push(Some(GdnState {
                    s: vec![0f32; cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim],
                    conv: vec![0f32; (cfg.linear_conv_kernel_dim - 1) * conv_dim],
                }));
            }
        }
        QwenSession { kv, gdn, pos: 0 }
    }

    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * self.cfg.vocab_size];
        linear(&self.lm_head, self.cfg.vocab_size, self.cfg.hidden_size, hidden, t, &mut logits);
        logits
    }

    pub fn forward(&self, sess: &mut QwenSession, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[(t - 1) * self.cfg.hidden_size..], 1)
    }

    /// Final-normed hidden states `[t, hidden]` for `tokens` appended at `sess.pos`.
    pub fn forward_hidden_mm(&self, sess: &mut QwenSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let start = sess.pos;
        let par = t > 1;

        let timer = prof::start(Cat::Embed);
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, par, |ti, row| {
            let tok = tokens[ti] as usize;
            self.embed.row(tok, row);
        });
        for &(i, emb) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv_freq);
        drop(timer);

        let mut hn = vec![0f32; t * h];
        let inter = cfg.intermediate_size;
        let mut g = vec![0f32; t * inter];
        let mut u = vec![0f32; t * inter];
        let mut y = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            rmsnorm_f32w_rows(&x, h, &layer.ln_in, eps, &mut hn);
            match &layer.mixer {
                Mixer::Gdn(gd) => self.gdn_forward(gd, sess.gdn[l].as_mut().unwrap(), &hn, t, &mut y),
                Mixer::Attn(at) => self.attn_forward(at, sess.kv[l].as_mut().unwrap(), &hn, t, start, &cos, &sin, &mut y),
            }
            add_inplace(&mut x, &y);
            rmsnorm_f32w_rows(&x, h, &layer.ln_post, eps, &mut hn);
            linear_multi(&mut [(&layer.gate, inter, &mut g[..]), (&layer.up, inter, &mut u[..])], h, &hn, t);
            silu_mul(&mut g, &u);
            linear(&layer.down, h, inter, &g, t, &mut y);
            add_inplace(&mut x, &y);
        }
        rmsnorm_f32w_rows(&x.clone(), h, &self.final_norm, eps, &mut x);
        sess.pos += t;
        x
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_forward(&self, at: &Attn, cache: &mut LayerCache, hn: &[f32], t: usize, start: usize, cos: &[f32], sin: &[f32], y: &mut [f32]) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let eps = cfg.rms_norm_eps as f32;
        let half = self.inv_freq.len();
        let gated = cfg.attn_output_gate;
        let qw = if gated { 2 * nh * hd } else { nh * hd };
        let mut qg = vec![0f32; t * qw];
        let mut k = vec![0f32; t * nkv * hd];
        let mut v = vec![0f32; t * nkv * hd];
        linear_multi(
            &mut [
                (&at.q, qw, &mut qg[..]),
                (&at.k, nkv * hd, &mut k[..]),
                (&at.v, nkv * hd, &mut v[..]),
            ],
            h,
            hn,
            t,
        );

        // Split query / gate, norm, partial RoPE, fold in the 1/sqrt(hd) scaling.
        let scale = (hd as f32).powf(-0.5);
        let mut q = vec![0f32; t * nh * hd];
        let mut gate = vec![0f32; t * nh * hd];
        let par = t > 1;
        {
            let qp = SendPtrF(q.as_mut_ptr());
            let gp = SendPtrF(gate.as_mut_ptr());
            let run = |i: usize| {
                let (ti, hh) = (i / nh, i % nh);
                // SAFETY: disjoint (token, head) slices.
                let qo = unsafe { std::slice::from_raw_parts_mut(qp.get().add(i * hd), hd) };
                let src = if gated {
                    &qg[ti * qw + hh * 2 * hd..ti * qw + hh * 2 * hd + hd]
                } else {
                    &qg[ti * qw + hh * hd..ti * qw + (hh + 1) * hd]
                };
                rmsnorm_f32w(src, &at.q_norm, eps, qo);
                apply_rope(
                    &mut qo[..2 * half],
                    &cos[ti * half..(ti + 1) * half],
                    &sin[ti * half..(ti + 1) * half],
                );
                for v in qo.iter_mut() {
                    *v *= scale;
                }
                if gated {
                    let go = unsafe { std::slice::from_raw_parts_mut(gp.get().add(i * hd), hd) };
                    go.copy_from_slice(&qg[ti * qw + hh * 2 * hd + hd..ti * qw + (hh + 1) * 2 * hd]);
                }
            };
            if par {
                pool::run(t * nh, &run);
            } else {
                for i in 0..t * nh {
                    run(i);
                }
            }
        }
        let mut kn = vec![0f32; t * nkv * hd];
        for_chunks(&mut kn, hd, par, |i, ko| {
            let ti = i / nkv;
            rmsnorm_f32w(&k[i * hd..(i + 1) * hd], &at.k_norm, eps, ko);
            apply_rope(
                &mut ko[..2 * half],
                &cos[ti * half..(ti + 1) * half],
                &sin[ti * half..(ti + 1) * half],
            );
        });
        cache.push(&kn, &v, t);
        let mut att = vec![0f32; t * nh * hd];
        attention(&q, cache, nh, t, start, None, &mut att);
        if gated {
            for (a, &g) in att.iter_mut().zip(&gate) {
                *a *= 1.0 / (1.0 + fast_exp(-g));
            }
        }
        linear(&at.o, h, nh * hd, &att, t, y);
    }

    fn gdn_forward(&self, gd: &Gdn, st: &mut GdnState, hn: &[f32], t: usize, y: &mut [f32]) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let (nk, nv, hdk, hdv) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let kc = cfg.linear_conv_kernel_dim;
        let rep = nv / nk;

        let mut qkv = vec![0f32; t * conv_dim];
        let mut z = vec![0f32; t * value_dim];
        let mut b = vec![0f32; t * nv];
        let mut a = vec![0f32; t * nv];
        linear_multi(
            &mut [
                (&gd.qkv, conv_dim, &mut qkv[..]),
                (&gd.z, value_dim, &mut z[..]),
                (&gd.b, nv, &mut b[..]),
                (&gd.a, nv, &mut a[..]),
            ],
            h,
            hn,
            t,
        );

        // Causal depthwise conv over time (kernel kc, left context from the state), then SiLU.
        let _timer = prof::start(Cat::Attention);
        let mut c = vec![0f32; t * conv_dim];
        {
            let st_conv = &st.conv;
            let src = |i: isize, ch: usize| -> f32 {
                if i >= 0 {
                    qkv[i as usize * conv_dim + ch]
                } else {
                    st_conv[((kc as isize - 1 + i) as usize) * conv_dim + ch]
                }
            };
            for_chunks(&mut c, conv_dim, t > 1, |i, row| {
                for ch in 0..conv_dim {
                    let mut s = 0f32;
                    for j in 0..kc {
                        s += gd.conv_w[ch * kc + j] * src(i as isize - (kc as isize - 1) + j as isize, ch);
                    }
                    row[ch] = s / (1.0 + fast_exp(-s));
                }
            });
        }
        // Update conv state with the last kc-1 raw inputs.
        {
            let keep = kc - 1;
            let mut new_state = vec![0f32; keep * conv_dim];
            for slot in 0..keep {
                // slot 0 is the oldest kept input: time index t - keep + slot
                let ti = t as isize - keep as isize + slot as isize;
                if ti >= 0 {
                    new_state[slot * conv_dim..(slot + 1) * conv_dim]
                        .copy_from_slice(&qkv[ti as usize * conv_dim..(ti as usize + 1) * conv_dim]);
                } else {
                    let old = (keep as isize + ti) as usize;
                    new_state[slot * conv_dim..(slot + 1) * conv_dim].copy_from_slice(&st.conv[old * conv_dim..(old + 1) * conv_dim]);
                }
            }
            st.conv = new_state;
        }

        // Gated delta rule, recurrent over time, parallel over value heads.
        let mut out = vec![0f32; t * value_dim];
        let op = SendPtrF(out.as_mut_ptr());
        let sp = SendPtrF(st.s.as_mut_ptr());
        let qscale = (hdk as f32).powf(-0.5);
        pool::run(nv, &|hv| {
            let hk = hv / rep;
            // SAFETY: each head owns its own state slab and output columns.
            let s = unsafe { std::slice::from_raw_parts_mut(sp.get().add(hv * hdk * hdv), hdk * hdv) };
            let mut qn = vec![0f32; hdk];
            let mut kn = vec![0f32; hdk];
            let mut kvmem = vec![0f32; hdv];
            let mut delta = vec![0f32; hdv];
            for i in 0..t {
                let row = &c[i * conv_dim..(i + 1) * conv_dim];
                let qv = &row[hk * hdk..(hk + 1) * hdk];
                let kv = &row[key_dim + hk * hdk..key_dim + (hk + 1) * hdk];
                let vv = &row[2 * key_dim + hv * hdv..2 * key_dim + (hv + 1) * hdv];
                let qn_inv = (qv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
                let kn_inv = (kv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
                for d in 0..hdk {
                    qn[d] = qv[d] * qn_inv * qscale;
                    kn[d] = kv[d] * kn_inv;
                }
                let beta = 1.0 / (1.0 + fast_exp(-b[i * nv + hv]));
                let ga = a[i * nv + hv] + gd.dt_bias[hv];
                let softplus = if ga > 20.0 { ga } else { (1.0 + ga.exp()).ln() };
                let decay = (gd.decay_coef[hv] * softplus).exp();
                // S *= decay; kvmem = k^T S
                kvmem.fill(0.0);
                for dk in 0..hdk {
                    let srow = &mut s[dk * hdv..(dk + 1) * hdv];
                    let kd = kn[dk];
                    for j in 0..hdv {
                        srow[j] *= decay;
                        kvmem[j] += srow[j] * kd;
                    }
                }
                for j in 0..hdv {
                    delta[j] = (vv[j] - kvmem[j]) * beta;
                }
                // S += k (x) delta; out = q^T S
                let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(i * value_dim + hv * hdv), hdv) };
                o.fill(0.0);
                for dk in 0..hdk {
                    let srow = &mut s[dk * hdv..(dk + 1) * hdv];
                    let (kd, qd) = (kn[dk], qn[dk]);
                    for j in 0..hdv {
                        srow[j] += kd * delta[j];
                        o[j] += srow[j] * qd;
                    }
                }
            }
        });
        drop(_timer);

        // Gated RMSNorm per head: norm(o) * w * silu(z)
        for_chunks(&mut out, hdv, t * nv > 64, |i, o| {
            let ss: f32 = o.iter().map(|v| v * v).sum::<f32>() / hdv as f32 + eps;
            let r = ss.powf(-0.5);
            let zz = &z[i * hdv..(i + 1) * hdv];
            for j in 0..hdv {
                let g = zz[j];
                o[j] = o[j] * r * gd.norm_w[j] * (g / (1.0 + fast_exp(-g)));
            }
        });
        linear(&gd.out, h, value_dim, &out, t, y);
    }
}

#[derive(Clone, Copy)]
struct SendPtrF(*mut f32);
unsafe impl Send for SendPtrF {}
unsafe impl Sync for SendPtrF {}
impl SendPtrF {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
