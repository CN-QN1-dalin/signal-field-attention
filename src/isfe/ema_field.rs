/// Intent EMA Field — 长期意图趋势

#[derive(Debug, Clone)]
pub struct IntentEMAField {
    ema: Vec<f32>,
    gamma: f32,
    dim: usize,
}

impl IntentEMAField {
    /// 创建新的 EMA Field
    pub fn new(dim: usize, gamma: f32) -> Self {
        Self {
            ema: vec![0.0; dim],
            gamma,
            dim,
        }
    }

    /// 更新 EMA: EMA[t] = γ·EMA[t-1] + (1-γ)·intent[t]
    pub fn update(&mut self, intent: Vec<f32>) {
        assert_eq!(intent.len(), self.dim, "Intent dimension mismatch");
        for i in 0..self.dim {
            self.ema[i] = self.gamma * self.ema[i] + (1.0 - self.gamma) * intent[i];
        }
    }

    /// 获取当前 EMA 值
    pub fn get_value(&self) -> &Vec<f32> {
        &self.ema
    }

    /// 是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.ema.iter().any(|&x| x != 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_update() {
        let mut ema = IntentEMAField::new(3, 0.98);
        assert!(!ema.is_initialized());
        ema.update(vec![1.0, 2.0, 3.0]);
        assert!(ema.is_initialized());
        let value = ema.get_value();
        assert!((value[0] - 0.02).abs() < 1e-5);
        assert!((value[1] - 0.04).abs() < 1e-5);
        assert!((value[2] - 0.06).abs() < 1e-5);
        ema.update(vec![4.0, 5.0, 6.0]);
        let value = ema.get_value();
        assert!((value[0] - 0.0996).abs() < 1e-4);
        assert!((value[1] - 0.1392).abs() < 1e-4);
        assert!((value[2] - 0.1788).abs() < 1e-4);
    }
}
