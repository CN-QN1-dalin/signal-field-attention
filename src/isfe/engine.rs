//! Intent Signal Field Engine — 主引擎

use crate::ring_buffer::IntentRingBuffer;
use crate::ema_field::IntentEMAField;
use crate::semantic_pool::IntentSemanticPool;
use crate::fusion::IntentFusion;

#[derive(Debug)]
pub struct IntentSignalFieldEngine {
    ring_buffer: IntentRingBuffer,
    ema_field: IntentEMAField,
    semantic_pool: IntentSemanticPool,
    fusion: IntentFusion,
    dim: usize,
}

#[derive(Debug, Clone)]
pub struct EngineResult {
    pub user_intent: Vec<f32>,
    pub ring_mean: Vec<f32>,
    pub ema: Vec<f32>,
    pub semantic: Vec<f32>,
    pub enhancement: Vec<f32>,
    pub confidence: f32,
}

impl IntentSignalFieldEngine {
    pub fn new(dim: usize, ring_capacity: usize, gamma: f32,
               semantic_slots: usize, temperature: f32) -> Self {
        Self {
            ring_buffer: IntentRingBuffer::new(dim, ring_capacity),
            ema_field: IntentEMAField::new(dim, gamma),
            semantic_pool: IntentSemanticPool::new(semantic_slots, dim, temperature),
            fusion: IntentFusion::new(dim),
            dim,
        }
    }

    pub fn process_dialogue(&mut self, user_input: &str, ai_response: &str) -> EngineResult {
        let user_intent = self.embed_intent(user_input);
        self.ring_buffer.push(user_intent.clone());
        self.ema_field.update(user_intent.clone());
        self.semantic_pool.add_intent(user_intent.clone());

        let ring_mean = self.ring_buffer.get_mean();
        let ema = self.ema_field.get_value().clone();
        let semantic = self.semantic_pool.query(&user_intent);
        let enhancement = self.fusion.fuse(&ring_mean, &ema, &semantic);
        let confidence = self.calculate_confidence(&enhancement, &user_intent);

        EngineResult {
            user_intent,
            ring_mean,
            ema,
            semantic,
            enhancement,
            confidence,
        }
    }

    fn embed_intent(&self, text: &str) -> Vec<f32> {
        let mut intent = vec![0.0; self.dim];
        for (i, c) in text.chars().enumerate() {
            if i < self.dim {
                intent[i] = c as f32 / 255.0;
            }
        }
        let norm: f32 = intent.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in intent.iter_mut() { *x /= norm; }
        }
        intent
    }

    fn calculate_confidence(&self, enhancement: &[f32], intent: &[f32]) -> f32 {
        let dot: f32 = enhancement.iter().zip(intent.iter()).map(|(a, b)| a * b).sum();
        let norm_enh: f32 = enhancement.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_intent: f32 = intent.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_enh > 1e-10 && norm_intent > 1e-10 {
            (dot / (norm_enh * norm_intent) + 1.0) / 2.0
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_process_dialogue() {
        let mut engine = IntentSignalFieldEngine::new(10, 4, 0.98, 8, 0.07);
        let result = engine.process_dialogue("你好", "你好，有什么可以帮助你的？");
        assert_eq!(result.user_intent.len(), 10);
        assert_eq!(result.enhancement.len(), 10);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
}
