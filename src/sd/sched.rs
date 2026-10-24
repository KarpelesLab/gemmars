//! Euler samplers (diffusers `EulerDiscreteScheduler` / `EulerAncestralDiscreteScheduler`) over the
//! model's training noise schedule from `scheduler/scheduler_config.json`.

use super::ops::Rng;
use anyhow::{Result, bail};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    Euler,
    EulerAncestral,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prediction {
    Epsilon,
    V,
}

pub struct Schedule {
    /// sigma for each training timestep (index = timestep).
    train_sigmas: Vec<f64>,
    spacing: String,
    steps_offset: usize,
    pub prediction: Prediction,
}

impl Schedule {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(path)?)?;
        let n = cfg["num_train_timesteps"].as_u64().unwrap_or(1000) as usize;
        let (b0, b1) = (
            cfg["beta_start"].as_f64().unwrap_or(0.00085),
            cfg["beta_end"].as_f64().unwrap_or(0.012),
        );
        let betas: Vec<f64> = match cfg["beta_schedule"].as_str().unwrap_or("scaled_linear") {
            "scaled_linear" => (0..n)
                .map(|i| (b0.sqrt() + (b1.sqrt() - b0.sqrt()) * i as f64 / (n - 1) as f64).powi(2))
                .collect(),
            "linear" => (0..n).map(|i| b0 + (b1 - b0) * i as f64 / (n - 1) as f64).collect(),
            other => bail!("beta schedule {other} is not supported"),
        };
        let mut ac = 1.0;
        let train_sigmas = betas
            .iter()
            .map(|b| {
                ac *= 1.0 - b;
                ((1.0 - ac) / ac).sqrt()
            })
            .collect();
        let prediction = match cfg["prediction_type"].as_str().unwrap_or("epsilon") {
            "epsilon" => Prediction::Epsilon,
            "v_prediction" => Prediction::V,
            other => bail!("prediction type {other} is not supported"),
        };
        Ok(Self {
            train_sigmas,
            // diffusers' Euler schedulers default to "linspace" when the config predates the key.
            spacing: cfg["timestep_spacing"].as_str().unwrap_or("linspace").to_string(),
            steps_offset: cfg["steps_offset"].as_u64().unwrap_or(0) as usize,
            prediction,
        })
    }

    /// Timesteps and sigmas (one extra trailing 0) for `steps` inference steps, and the noise
    /// scale of the initial latent.
    pub fn plan(&self, steps: usize) -> (Vec<f64>, Vec<f64>, f64) {
        let n = self.train_sigmas.len();
        let ts: Vec<f64> = match self.spacing.as_str() {
            "linspace" => (0..steps)
                .map(|i| (n - 1) as f64 * (steps - 1 - i) as f64 / (steps.max(2) - 1) as f64)
                .collect(),
            "trailing" => {
                let ratio = n as f64 / steps as f64;
                (0..steps).map(|i| (n as f64 - i as f64 * ratio).round() - 1.0).collect()
            }
            _ => {
                let ratio = n / steps;
                (0..steps).rev().map(|i| (i * ratio + self.steps_offset) as f64).collect()
            }
        };
        // Linear interpolation of the training sigmas at (possibly fractional) timesteps.
        let interp = |t: f64| {
            let i = t.floor().clamp(0.0, (n - 1) as f64) as usize;
            let f = t - i as f64;
            if i + 1 < n {
                self.train_sigmas[i] * (1.0 - f) + self.train_sigmas[i + 1] * f
            } else {
                self.train_sigmas[n - 1]
            }
        };
        let mut sigmas: Vec<f64> = ts.iter().map(|&t| interp(t)).collect();
        let max = sigmas.iter().cloned().fold(0.0, f64::max);
        sigmas.push(0.0);
        let init = if self.spacing == "linspace" || self.spacing == "trailing" {
            max
        } else {
            (max * max + 1.0).sqrt()
        };
        (ts, sigmas, init)
    }

    /// Input scaling for the model at noise level `sigma`.
    pub fn scale_input(sigma: f64) -> f32 {
        (1.0 / (sigma * sigma + 1.0).sqrt()) as f32
    }

    /// One Euler step from `sigma` to `next`, given the model output for the scaled input.
    pub fn step(&self, x: &mut [f32], model_out: &[f32], sigma: f64, next: f64, sampler: Sampler, rng: &mut Rng) {
        let s2 = sigma * sigma;
        let (sigma_to, up) = match sampler {
            Sampler::Euler => (next, 0.0),
            Sampler::EulerAncestral => {
                let up = (next * next * (s2 - next * next) / s2).max(0.0).sqrt();
                ((next * next - up * up).max(0.0).sqrt(), up)
            }
        };
        let dt = (sigma_to - sigma) as f32;
        for (xi, &m) in x.iter_mut().zip(model_out) {
            let x0 = match self.prediction {
                Prediction::Epsilon => *xi - sigma as f32 * m,
                Prediction::V => (m as f64 * (-sigma / (s2 + 1.0).sqrt()) + *xi as f64 / (s2 + 1.0)) as f32,
            };
            let d = (*xi - x0) / sigma as f32;
            *xi += d * dt;
        }
        if up > 0.0 {
            for xi in x.iter_mut() {
                *xi += rng.normal() * up as f32;
            }
        }
    }
}
