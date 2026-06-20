//! 北岳: Soma Convergence — O(1)增量推理
//!
//! 论文 Section 2.5:
//!   H(t) = Σ_{m=1}^{k} A_m · cos(ω_m · t + φ_m)
//!
//! 每个谐振模式独立 O(1) 更新

/// 谐振模式 (A_m, φ_m, ω_m)
#[derive(Debug, Clone)]
pub struct ResonanceMode {
    pub amplitude: f32,  // A_m
    pub phase: f32,      // φ_m
    pub frequency: f32,  // ω_m
}

impl ResonanceMode {
    /// A_m · cos(ω_m · t + φ_m)
    #[inline]
    pub fn evaluate(&self, t: f32) -> f32 {
        self.amplitude * (self.frequency * t + self.phase).cos()
    }

    /// O(1) 增量更新
    pub fn update(&mut self, observation: f32, lr: f32, gamma: f32) {
        self.amplitude = gamma * self.amplitude + (1.0 - gamma) * observation.abs();
        let obs_phase = observation.signum() * self.frequency;
        self.phase = gamma * self.phase + (1.0 - gamma) * obs_phase * lr;
    }
}

/// Soma Convergence: O(1) 增量推理
pub struct SomaConvergence {
    modes: Vec<ResonanceMode>,
    num_modes: usize,
    gamma: f32,
    step: usize,
    head_dim: usize,
    projections: Vec<Vec<f32>>,  // [head_dim][num_modes]
}

impl SomaConvergence {
    pub fn new(num_modes: usize, head_dim: usize, gamma: f32) -> Self {
        let modes: Vec<ResonanceMode> = (0..num_modes)
            .map(|i| ResonanceMode {
                amplitude: 0.0,
                phase: (i as f32 * 0.618 * std::f32::consts::PI * 2.0).fract(), // 黄金比例分散
                frequency: (i as f32 + 1.0) * 0.1,
            })
            .collect();
        // 简单投影初始化
        let projections = (0..head_dim)
            .map(|_| (0..num_modes).map(|i| (i as f32 * 0.01 + 0.001) % 0.02 - 0.01).collect())
            .collect();
        Self { modes, num_modes, gamma, step: 0, head_dim, projections }
    }

    /// O(1) 解码步
    pub fn decode_step(&mut self, observation: &[f32]) -> Vec<f32> {
        let t = self.step as f32;
        let obs_energy: f32 = observation.iter().map(|&x| x * x).sum::<f32>().sqrt();
        for mode in &mut self.modes {
            mode.update(obs_energy, 0.01, self.gamma);
        }
        // 投影模式值
        let mode_vals: Vec<f32> = self.modes.iter().map(|m| m.evaluate(t)).collect();
        let output: Vec<f32> = self.projections.iter()
            .map(|proj| proj.iter().zip(mode_vals.iter()).map(|(&p, &m)| p * m).sum())
            .collect();
        self.step += 1;
        output
    }

    /// 固定内存 (与序列长度无关)
    pub fn memory_bytes(&self) -> usize {
        (self.num_modes * 3 + self.head_dim * self.num_modes) * 4
    }

    pub fn reset(&mut self) {
        for mode in &mut self.modes { mode.amplitude = 0.0; }
        self.step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_evaluate() {
        let mode = ResonanceMode { amplitude: 1.0, phase: 0.0, frequency: 1.0 };
        assert!((mode.evaluate(0.0) - 1.0).abs() < 1e-5);
        assert!((mode.evaluate(std::f32::consts::PI) - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_convergence_decode() {
        let mut conv = SomaConvergence::new(16, 64, 0.98);
        let obs = vec![0.1f32; 64];
        let out = conv.decode_step(&obs);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_mode_update() {
        let mut mode = ResonanceMode { amplitude: 0.0, phase: 0.0, frequency: 1.0 };
        mode.update(1.0, 0.01, 0.98);
        assert!((mode.amplitude - 0.02).abs() < 1e-5);
    }
}
