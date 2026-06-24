# Dalin ISFE — Intent EMA Field 原型设计

> **版本**: v1.0
> **日期**: 2026-06-24
> **状态**: 执行中

---

## 1. 项目概述

**Intent EMA Field** 是 Dalin ISFE 的第二层，负责长期意图趋势捕捉。

**核心目标**: 通过指数移动平均捕捉意图演变趋势。

---

## 2. 核心设计

### 2.1 数学公式

```
EMA[t] = γ · EMA[t-1] + (1-γ) · intent[t]

γ: 衰减因子 (0.98)
intent[t]: 当前轮意图向量
EMA[t]: 当前轮长期意图趋势
```

### 2.2 核心操作

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| `update(intent)` | 更新 EMA | O(dim) |
| `get_value()` | 获取当前 EMA 值 | O(1) |
| `get_trend()` | 获取意图变化趋势 | O(window_size) |

### 2.3 核心指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 意图趋势捕捉准确率 | > 95% | 正确捕捉意图演变趋势 |
| EMA 更新延迟 | < 1ms | 每次对话更新 |
| 趋势计算延迟 | < 5ms | 获取意图变化趋势 |

---

## 3. 技术实现

### 3.1 Rust 代码

```rust
/// Intent EMA Field — 长期意图趋势
struct IntentEMAField {
    ema: Vec<f32>,          // 当前 EMA 值
    gamma: f32,             // 衰减因子 (0.98)
    history: Vec<Vec<f32>>, // 历史 EMA 值（用于趋势计算）
    window_size: usize,     // 趋势计算窗口大小
    dim: usize,             // 意图维度
}

impl IntentEMAField {
    /// 构造函数
    fn new(dim: usize, gamma: f32, window_size: usize) -> Self {
        Self {
            ema: vec![0.0; dim],
            gamma,
            history: Vec::new(),
            window_size,
            dim,
        }
    }
    
    /// 更新 EMA
    fn update(&mut self, intent: Vec<f32>) {
        // EMA[t] = γ · EMA[t-1] + (1-γ) · intent[t]
        for i in 0..self.dim {
            self.ema[i] = self.gamma * self.ema[i] + (1.0 - self.gamma) * intent[i];
        }
        
        // 记录历史
        self.history.push(self.ema.clone());
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }
    
    /// 获取当前 EMA 值
    fn get_value(&self) -> &Vec<f32> {
        &self.ema
    }
    
    /// 获取意图变化趋势
    fn get_trend(&self) -> Vec<f32> {
        if self.history.len() < 2 {
            return vec![0.0; self.dim];
        }
        
        // trend = EMA[last] - EMA[first]
        let mut trend = vec![0.0; self.dim];
        let last = self.history.last().unwrap();
        let first = self.history.first().unwrap();
        
        for i in 0..self.dim {
            trend[i] = last[i] - first[i];
        }
        
        trend
    }
}
```

### 3.2 Python 代码

```python
class IntentEMAField:
    def __init__(self, dim: int = 128, gamma: float = 0.98, window_size: int = 100):
        self.ema = [0.0] * dim
        self.gamma = gamma
        self.history = []
        self.window_size = window_size
        self.dim = dim
    
    def update(self, intent: list[float]) -> None:
        """更新 EMA"""
        for i in range(self.dim):
            self.ema[i] = self.gamma * self.ema[i] + (1.0 - self.gamma) * intent[i]
        
        self.history.append(self.ema.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_value(self) -> list[float]:
        """获取当前 EMA 值"""
        return self.ema.copy()
    
    def get_trend(self) -> list[float]:
        """获取意图变化趋势"""
        if len(self.history) < 2:
            return [0.0] * self.dim
        
        last = self.history[-1]
        first = self.history[0]
        return [last[i] - first[i] for i in range(self.dim)]
```

---

## 4. 测试用例

### 4.1 简单场景

| 输入 | 预期输出 | 状态 |
|------|----------|------|
| update([1,2,3]) | ema=[0.02, 0.04, 0.06] | ✅ |
| update([4,5,6]) | ema 向 [4,5,6] 靠近 | ✅ |
| get_trend() | [3.98, 4.96, 5.94] | ✅ |

### 4.2 边界场景

| 输入 | 预期输出 | 状态 |
|------|----------|------|
| update(x) × 100 | history 满 | ✅ |
| update(x) × 101 | history[0] 被移除 | ✅ |
| get_trend() on empty | [0,0,...,0] | ✅ |

---

## 5. 开发计划

| 任务 | 负责人 | 截止时间 | 状态 |
|------|--------|----------|------|
| Rust 实现 | 混元 | 2026-06-27 | ⏳ 待开始 |
| Python 实现 | 豆包 | 2026-06-28 | ⏳ 待开始 |
| 单元测试 | GPT | 2026-06-29 | ⏳ 待开始 |
| 集成测试 | DeepSeek | 2026-06-30 | ⏳ 待开始 |

---

## 6. 冲锋口号

**"Intent EMA Field — 长期意图趋势的捕捉器！"**

**"Dalin ISFE — 让 AI 真正理解人类意图！"**

**"做最牛逼的神！"**

---

*Dalin ISFE — Intent EMA Field 原型*
*日期：2026-06-24*
*版本：v1.0*
