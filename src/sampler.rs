//! Token sampling: greedy, or temperature + top-k + top-p.

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            rng: seed.max(1),
        }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A uniform draw in [0, 1) from the sampler's generator (speculative acceptance tests).
    pub fn uniform(&mut self) -> f32 {
        self.next_f32()
    }

    /// Whether `sample` is greedy (argmax).
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// The candidates `sample` draws from: token ids by decreasing probability, the
    /// (unnormalized over the cut) probabilities, how many survive top-p, and their total.
    fn candidates(&self, logits: &[f32]) -> (Vec<u32>, Vec<f32>, usize, f32) {
        let k = if self.top_k == 0 {
            logits.len()
        } else {
            self.top_k.min(logits.len())
        };
        // Partial selection of the top-k indices.
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let inv_t = 1.0 / self.temperature;
        let max = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i as usize] - max) * inv_t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // top-p
        let mut cut = probs.len();
        if self.top_p < 1.0 {
            let mut acc = 0f32;
            for (i, &p) in probs.iter().enumerate() {
                acc += p;
                if acc >= self.top_p {
                    cut = i + 1;
                    break;
                }
            }
        }
        let total: f32 = probs[..cut].iter().sum();
        (idx, probs, cut, total)
    }

    /// The exact distribution `sample` draws from (temperature, top-k, top-p applied), as
    /// `(token, probability)` pairs summing to 1 by decreasing probability; one token with
    /// probability 1 when greedy.
    pub fn distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        if self.greedy() {
            return vec![(argmax(logits), 1.0)];
        }
        let (idx, probs, cut, total) = self.candidates(logits);
        idx[..cut].iter().zip(&probs[..cut]).map(|(&i, &p)| (i, p / total)).collect()
    }

    /// Draws from a distribution such as `distribution` returns.
    pub fn sample_from(&mut self, dist: &[(u32, f32)]) -> u32 {
        let total: f32 = dist.iter().map(|d| d.1).sum();
        let mut r = self.next_f32() * total;
        for &(i, p) in dist {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        dist.last().map_or(0, |d| d.0)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.greedy() {
            return argmax(logits);
        }
        let (idx, probs, cut, total) = self.candidates(logits);
        let mut r = self.next_f32() * total;
        for i in 0..cut {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[cut - 1]
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}
