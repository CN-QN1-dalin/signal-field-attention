//! Dalin Soma: Signal Field Attention Inference Acceleration Framework
//!
//! 五岳架构 (Five Sacred Peaks):
//! - 东岳 Soma Engine     — 信号场推理加速 (O(1) memory)
//! - 南岳 Soma LingYa     — 参数高效微调 (50% less params than LoRA)
//! - 西岳 Soma Native     — 零设计神经网络 (SomaBlock)
//! - 北岳 Soma Convergence — O(1)增量推理 (resonance modes)
//! - 中岳 Soma Heritage   — 蒸馏训练 (progressive replacement)
//!
//! Zero external dependencies — pure Rust, no ndarray/candle/metal-rs

pub mod config;
pub mod math;
pub mod engine;
pub mod lingya;
pub mod native;
pub mod convergence;
pub mod heritage;

pub use config::SomaConfig;
pub use engine::SomaEngine;
