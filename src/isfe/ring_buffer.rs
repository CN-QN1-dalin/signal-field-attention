/// Intent RingBuffer — 短期意图记忆

#[derive(Debug, Clone)]
pub struct IntentRingBuffer {
    buffer: Vec<Vec<f32>>,
    head: usize,
    size: usize,
    capacity: usize,
    dim: usize,
}

impl IntentRingBuffer {
    /// 创建新的 RingBuffer
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            buffer: vec![vec![0.0; dim]; capacity],
            head: 0,
            size: 0,
            capacity,
            dim,
        }
    }

    /// 添加意图
    pub fn push(&mut self, intent: Vec<f32>) {
        assert_eq!(intent.len(), self.dim, "Intent dimension mismatch");
        self.buffer[self.head] = intent;
        self.head = (self.head + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// 获取平均意图
    pub fn get_mean(&self) -> Vec<f32> {
        if self.size == 0 {
            return vec![0.0; self.dim];
        }
        let mut mean = vec![0.0; self.dim];
        for i in 0..self.size {
            for j in 0..self.dim {
                mean[j] += self.buffer[i][j];
            }
        }
        for j in 0..self.dim {
            mean[j] /= self.size as f32;
        }
        mean
    }

    /// 获取缓冲区大小
    pub fn size(&self) -> usize {
        self.size
    }

    /// 是否已满
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_pop() {
        let mut rb = IntentRingBuffer::new(3, 4);
        rb.push(vec![1.0, 2.0, 3.0]);
        rb.push(vec![4.0, 5.0, 6.0]);
        let mean = rb.get_mean();
        assert!((mean[0] - 2.5).abs() < 1e-5);
        assert!((mean[1] - 3.5).abs() < 1e-5);
        assert!((mean[2] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut rb = IntentRingBuffer::new(3, 2);
        rb.push(vec![1.0, 2.0, 3.0]);
        rb.push(vec![4.0, 5.0, 6.0]);
        rb.push(vec![7.0, 8.0, 9.0]);
        assert!(rb.is_full());
        assert_eq!(rb.size(), 2);
        let mean = rb.get_mean();
        assert!((mean[0] - 5.5).abs() < 1e-5);
        assert!((mean[1] - 6.5).abs() < 1e-5);
        assert!((mean[2] - 7.5).abs() < 1e-5);
    }
}
