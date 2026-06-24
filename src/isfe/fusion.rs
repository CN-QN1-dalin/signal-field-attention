/// Intent Fusion — 三通道意图融合

#[derive(Debug, Clone)]
pub struct IntentFusion {
    dim: usize,
}

impl IntentFusion {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// enhancement = ring_mean + 0.5 * ema + 0.5 * semantic
    pub fn fuse(&self, ring_mean: &[f32], ema: &[f32], semantic: &[f32]) -> Vec<f32> {
        assert_eq!(ring_mean.len(), self.dim);
        assert_eq!(ema.len(), self.dim);
        assert_eq!(semantic.len(), self.dim);
        let mut enhancement = vec![0.0; self.dim];
        for i in 0..self.dim {
            enhancement[i] = ring_mean[i] + 0.5 * ema[i] + 0.5 * semantic[i];
        }
        enhancement
    }

    /// 计算余弦相似度 (映射到 [0, 1])
    pub fn validate(&self, enhancement: &[f32], expected: &[f32]) -> f32 {
        let dot: f32 = enhancement.iter().zip(expected.iter()).map(|(a, b)| a * b).sum();
        let norm_enh: f32 = enhancement.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_exp: f32 = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_enh > 1e-10 && norm_exp > 1e-10 {
            (dot / (norm_enh * norm_exp) + 1.0) / 2.0
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion() {
        let fusion = IntentFusion::new(3);
        let ring = vec![1.0, 2.0, 3.0];
        let ema = vec![4.0, 5.0, 6.0];
        let semantic = vec![7.0, 8.0, 9.0];
        let enhancement = fusion.fuse(&ring, &ema, &semantic);
        assert!((enhancement[0] - 6.5).abs() < 1e-5);
        assert!((enhancement[1] - 8.5).abs() < 1e-5);
        assert!((enhancement[2] - 10.5).abs() < 1e-5);
    }
}
