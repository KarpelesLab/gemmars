//! The VAE decoder (diffusers `AutoencoderKL.decode`): latents back to RGB at 8x the resolution.

use super::ops::{Conv, GroupNorm, Linear, Pre, Tensors, attention, upsample2x};
use super::unet::Resnet;
use crate::kernels::add_inplace;
use anyhow::Result;
use std::path::Path;

/// The single-head self-attention in the decoder's middle block.
struct MidAttn {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

struct Up {
    resnets: Vec<Resnet>,
    up: Option<Conv>,
}

pub struct VaeDecoder {
    post_quant: Option<Linear>,
    conv_in: Conv,
    mid1: Resnet,
    attn: MidAttn,
    mid2: Resnet,
    ups: Vec<Up>,
    norm_out: GroupNorm,
    conv_out: Conv,
    pub scaling_factor: f32,
    pub shift_factor: f32,
    pub latent_channels: usize,
}

impl VaeDecoder {
    pub fn load(dir: &Path, weights: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let t = Tensors::open(weights)?;
        let eps = 1e-6;
        let p = "decoder";
        let mut ups = Vec::new();
        while t.has(&format!("{p}.up_blocks.{}.resnets.0.conv1.weight", ups.len())) {
            let b = format!("{p}.up_blocks.{}", ups.len());
            let mut resnets = Vec::new();
            while t.has(&format!("{b}.resnets.{}.conv1.weight", resnets.len())) {
                resnets.push(Resnet::load(&t, &format!("{b}.resnets.{}", resnets.len()), eps)?);
            }
            let up = if t.has(&format!("{b}.upsamplers.0.conv.weight")) {
                Some(Conv::load(&t, &format!("{b}.upsamplers.0.conv"))?)
            } else {
                None
            };
            ups.push(Up { resnets, up });
        }
        let a = format!("{p}.mid_block.attentions.0");
        let conv_in = Conv::load(&t, &format!("{p}.conv_in"))?;
        Ok(Self {
            post_quant: if t.has("post_quant_conv.weight") {
                Some(Linear::load(&t, "post_quant_conv")?)
            } else {
                None
            },
            latent_channels: conv_in.cin,
            conv_in,
            mid1: Resnet::load(&t, &format!("{p}.mid_block.resnets.0"), eps)?,
            attn: MidAttn {
                norm: GroupNorm::load(&t, &format!("{a}.group_norm"), 32, eps)?,
                q: Linear::load(&t, &format!("{a}.to_q"))?,
                k: Linear::load(&t, &format!("{a}.to_k"))?,
                v: Linear::load(&t, &format!("{a}.to_v"))?,
                o: Linear::load(&t, &format!("{a}.to_out.0"))?,
            },
            mid2: Resnet::load(&t, &format!("{p}.mid_block.resnets.1"), eps)?,
            ups,
            norm_out: GroupNorm::load(&t, &format!("{p}.conv_norm_out"), 32, eps)?,
            conv_out: Conv::load(&t, &format!("{p}.conv_out"))?,
            // Older configs (SD 1.x) omit it; diffusers' default is the SD 1/2 VAE's 0.18215.
            scaling_factor: cfg["scaling_factor"].as_f64().unwrap_or(0.18215) as f32,
            shift_factor: cfg["shift_factor"].as_f64().unwrap_or(0.0) as f32,
        })
    }

    /// `latent` (`[h*w, latent_channels]`, as produced by the denoiser) to RGB in [-1, 1]
    /// (`[8h * 8w, 3]`); returns `(pixels, height, width)`.
    pub fn decode(&self, latent: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let mut z: Vec<f32> = latent.iter().map(|&v| v / self.scaling_factor + self.shift_factor).collect();
        if let Some(pq) = &self.post_quant {
            z = pq.forward(&z, h * w);
        }
        let (mut x, mut h, mut w) = self.conv_in.forward(&z, h, w, 1);
        x = self.mid1.forward(&x, h, w, None);
        let n = h * w;
        let pre = Pre::Group(&self.attn.norm, false);
        let (q, k, v) = if self.attn.q.on_device(n) {
            (
                self.attn.q.forward_pre(&x, n, pre),
                self.attn.k.forward_pre(&x, n, pre),
                self.attn.v.forward_pre(&x, n, pre),
            )
        } else {
            let y = pre.apply(&x, n);
            (self.attn.q.forward(&y, n), self.attn.k.forward(&y, n), self.attn.v.forward(&y, n))
        };
        let c = self.attn.q.n_out;
        let a = attention(&q, &k, &v, n, n, 1, c, false);
        add_inplace(&mut x, &self.attn.o.forward(&a, n));
        x = self.mid2.forward(&x, h, w, None);
        let mut c = self.conv_in.cout;
        for b in &self.ups {
            for r in &b.resnets {
                x = r.forward(&x, h, w, None);
                c = r.cout;
            }
            if let Some(u) = &b.up {
                let up = upsample2x(&x, h, w, c);
                (h, w) = (h * 2, w * 2);
                x = u.forward(&up, h, w, 1).0;
            }
        }
        (self.conv_out.forward_pre(&x, h, w, 1, Pre::Group(&self.norm_out, true)).0, h, w)
    }
}
