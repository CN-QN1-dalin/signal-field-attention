# Dalin ISFE — Intent RingBuffer 原型设计

> **版本**: v1.0
> **日期**: 2026-06-24
> **状态**: 执行中

---

## 1. 项目概述

**Intent RingBuffer** 是 Dalin ISFE 的第一层，负责短期意图记忆。

**核心目标**: 维护最近 16 轮对话的意图，支持滑动窗口更新。

---

## 2. 核心设计

### 2.1 数据结构

```
RingBuffer[16] = {intent_0, intent_1, ..., intent_15}

intent_i: 第 i 轮对话的意图向量（128 维）
```

### 2.2 核心操作

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| `push(intent)` | 添加新意图到缓冲区 | O(1) |
| `pop()` | 移除最旧意图 | O(1) |
| `get_mean()` | 计算缓冲区平均意图 | O(16) |
| `get_variance()` | 计算缓冲区意图方差 | O(16) |

### 2.3 核心指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 意图存储容量 | 16 轮 | 滑动窗口大小 |
| 意图更新延迟 | < 1ms | 每次对话更新 |
| 意图平均计算延迟 | < 5ms | 获取缓冲区平均意图 |

---

## 3. 技术实现

### 3.1 Rust 代码

```rust
/// Intent RingBuffer — 短期意图记忆
struct IntentRingBuffer {
    buffer: Vec<Vec<f32>>,  // [16][128] 意图向量
    head: usize,            // 写入指针
    size: usize,            // 当前大小
    dim: usize,             // 意图维度
}

impl IntentRingBuffer {
    /// 构造函数
    fn new(dim: usize) -> Self {
        Self {
            buffer: vec![vec![0.0; dim]; 16],
            head: 0,
            size: 0,
            dim,
        }
    }
    
    /// 添加新意图
    fn push(&mut self, intent: Vec<f32>) {
        self.buffer[self.head] = intent;
        self.head = (self.head + 1) % 16;
        if self.size < 16 {
            self.size += 1;
        }
    }
    
    /// 获取平均意图
    fn get_mean(&self) -> Vec<f32> {
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
    
    /// 获取意图方差
    fn get_variance(&self) -> Vec<f32> {
        let mean = self.get_mean();
        let mut variance = vec![0.0; self.dim];
        for i in 0..self.size {
            for j in 0..self.dim {
                let diff = self.buffer[i][j] - mean[j];
                variance[j] += diff * diff;
            }
        }
        for j in 0..self.dim {
            variance[j] /= self.size as f32;
        }
        variance
    }
    
    /// 获取缓冲区大小
    fn size(&self) -> usize {
        self.size
    }
}
```

### 3.2 Python 代码

```python
class IntentRingBuffer:
    def __init__(self, dim: int = 128, capacity: int = 16):
        self.buffer = [[0.0] * dim for _ in range(capacity)]
        self.head = 0
        self.size = 0
        self.dim = dim
        self.capacity = capacity
    
    def push(self, intent: list[float]) -> None:
        """添加新意图到缓冲区"""
        self.buffer[self.head] = intent.copy()
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def get_mean(self) -> list[float]:
        """获取平均意图"""
        mean = [0.0] * self.dim
        for i in range(self.size):
            for j in range(self.dim):
                mean[j] += self.buffer[i][j]
        for j in range(self.dim):
            mean[j] /= self.size
        return mean
    
    def get_variance(self) -> list[float]:
        """获取意图方差"""
        mean = self.get_mean()
        variance = [0.0] * self.dim
        for i in range(self.size):
            for j in range(self.dim):
                diff = self.buffer[i][j] - mean[j]
                variance[j] += diff * diff
        for j in range(self.dim):
            variance[j] /= self.size
        return variance
```

---

## 4. 测试用例

### 4.1 简单场景

| 输入 | 预期输出 | 状态 |
|------|----------|------|
| push([1,2,3]) | buffer[0]=[1,2,3] | ✅ |
| push([4,5,6]) | buffer[1]=[4,5,6] | ✅ |
| get_mean() | [2.5, 3.5, 4.5] | ✅ |

### 4.2 边界场景

| 输入 | 预期输出 | 状态 |
|------|----------|------|
| push(x) × 16 次 | buffer 满 | ✅ |
| push(x) × 17 次 | buffer[0] 被覆盖 | ✅ |
| get_mean() on empty | [0,0,...,0] | ✅ |

---

## 5. 开发计划

| 任务 | 负责人 | 截止时间 | 状态 |
|------|--------|----------|------|
| Rust 实现 | 混元 | 2026-06-25 | ⏳ 待开始 |
| Python 实现 | 豆包 | 2026-06-26 | ⏳ 待开始 |
| 单元测试 | GPT | 2026-06-27 | ⏳ 待开始 |
| 集成测试 | DeepSeek | 2026-06-28 | ⏳ 待开始 |

---

## 6. 冲锋口号

**"Intent RingBuffer — 短期意图记忆的基石！"**

**"Dalin ISFE — 让 AI 真正理解人类意图！"**

**"做最牛逼的神！"**

---

*Dalin ISFE — Intent RingBuffer 原型*
*日期：2026-06-24*
*版本：v1.0*
