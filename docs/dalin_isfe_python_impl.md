# Dalin ISFE — Python 核心实现

> **版本**: v1.0
> **日期**: 2026-06-24
> **状态**: 执行中

---

## 1. 项目结构

```
dalin-isfe-python/
├── pyproject.toml
├── src/
│   └── dalin_isfe/
│       ├── __init__.py
│       ├── ring_buffer.py
│       ├── ema_field.py
│       ├── semantic_pool.py
│       ├── fusion.py
│       └── engine.py
└── tests/
    └── test_engine.py
```

---

## 2. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "dalin-isfe"
version = "0.1.0"
description = "Intent Signal Field Engine — 让 AI 真正理解人类意图"
license = {text = "MIT"}
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
]
```

---

## 3. src/dalin_isfe/__init__.py

```python
"""Dalin ISFE — Intent Signal Field Engine"""

from .engine import IntentSignalFieldEngine

__version__ = "0.1.0"
__all__ = ["IntentSignalFieldEngine"]
```

---

## 4. src/dalin_isfe/ring_buffer.py

```python
"""Intent RingBuffer — 短期意图记忆"""

import numpy as np
from typing import List, Optional


class IntentRingBuffer:
    """短期意图记忆 RingBuffer"""
    
    def __init__(self, dim: int = 128, capacity: int = 16):
        self.dim = dim
        self.capacity = capacity
        self.buffer = np.zeros((capacity, dim), dtype=np.float32)
        self.head = 0
        self.size = 0
    
    def push(self, intent: List[float]) -> None:
        """添加意图到缓冲区"""
        intent_arr = np.array(intent, dtype=np.float32)
        assert len(intent_arr) == self.dim, f"Intent dimension mismatch: {len(intent_arr)} != {self.dim}"
        self.buffer[self.head] = intent_arr
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def get_mean(self) -> np.ndarray:
        """获取平均意图"""
        if self.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(self.buffer[:self.size], axis=0)
    
    def get_variance(self) -> np.ndarray:
        """获取意图方差"""
        if self.size < 2:
            return np.zeros(self.dim, dtype=np.float32)
        return np.var(self.buffer[:self.size], axis=0)
    
    @property
    def size(self) -> int:
        return self._size
    
    @size.setter
    def size(self, value: int):
        self._size = value
    
    def is_full(self) -> bool:
        """是否已满"""
        return self.size >= self.capacity
    
    def __repr__(self) -> str:
        return f"IntentRingBuffer(dim={self.dim}, capacity={self.capacity}, size={self.size})"
```

---

## 5. src/dalin_isfe/ema_field.py

```python
"""Intent EMA Field — 长期意图趋势"""

import numpy as np
from typing import List


class IntentEMAField:
    """长期意图趋势 EMA Field"""
    
    def __init__(self, dim: int = 128, gamma: float = 0.98):
        self.dim = dim
        self.gamma = gamma
        self.ema = np.zeros(dim, dtype=np.float32)
    
    def update(self, intent: List[float]) -> None:
        """更新 EMA"""
        intent_arr = np.array(intent, dtype=np.float32)
        assert len(intent_arr) == self.dim, f"Intent dimension mismatch"
        self.ema = self.gamma * self.ema + (1.0 - self.gamma) * intent_arr
    
    def get_value(self) -> np.ndarray:
        """获取当前 EMA 值"""
        return self.ema.copy()
    
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return np.any(self.ema != 0)
    
    def reset(self) -> None:
        """重置 EMA"""
        self.ema = np.zeros(self.dim, dtype=np.float32)
    
    def __repr__(self) -> str:
        return f"IntentEMAField(dim={self.dim}, gamma={self.gamma})"
```

---

## 6. src/dalin_isfe/semantic_pool.py

```python
"""Intent Semantic Pool — 全局意图语义"""

import numpy as np
from typing import List


class IntentSemanticPool:
    """全局意图语义 Semantic Pool"""
    
    def __init__(self, num_slots: int = 64, dim: int = 128, temperature: float = 0.07):
        self.num_slots = num_slots
        self.dim = dim
        self.temperature = temperature
        self.slots = np.zeros((num_slots, dim), dtype=np.float32)
    
    def add_intent(self, intent: List[float]) -> None:
        """添加意图到 Pool"""
        intent_arr = np.array(intent, dtype=np.float32)
        assert len(intent_arr) == self.dim, f"Intent dimension mismatch"
        
        for i in range(self.num_slots):
            weight = self._calculate_slot_weight(i, intent_arr)
            self.slots[i] = self.slots[i] * 0.9 + intent_arr * weight * 0.1
    
    def _calculate_slot_weight(self, slot_idx: int, intent: np.ndarray) -> float:
        """计算槽位权重"""
        slot = self.slots[slot_idx]
        dot = np.dot(slot, intent)
        norm_slot = np.linalg.norm(slot)
        norm_intent = np.linalg.norm(intent)
        
        if norm_slot > 1e-10 and norm_intent > 1e-10:
            cosine_sim = dot / (norm_slot * norm_intent)
            return float(np.exp(cosine_sim) / self.temperature)
        return 0.0
    
    def query(self, current_intent: List[float]) -> np.ndarray:
        """查询语义 Pool"""
        intent_arr = np.array(current_intent, dtype=np.float32)
        result = np.zeros(self.dim, dtype=np.float32)
        total_weight = 0.0
        
        for i in range(self.num_slots):
            weight = self._calculate_slot_weight(i, intent_arr)
            total_weight += weight
            result += self.slots[i] * weight
        
        if total_weight > 1e-10:
            result /= total_weight
        
        return result
    
    def reset(self) -> None:
        """重置 Pool"""
        self.slots = np.zeros((self.num_slots, self.dim), dtype=np.float32)
    
    def __repr__(self) -> str:
        return f"IntentSemanticPool(slots={self.num_slots}, dim={self.dim})"
```

---

## 7. src/dalin_isfe/fusion.py

```python
"""Intent Fusion — 三通道意图融合"""

import numpy as np
from typing import List


class IntentFusion:
    """三通道意图融合"""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
    
    def fuse(self, ring_mean: List[float], ema: List[float], 
             semantic: List[float]) -> np.ndarray:
        """融合三通道意图"""
        ring_arr = np.array(ring_mean, dtype=np.float32)
        ema_arr = np.array(ema, dtype=np.float32)
        semantic_arr = np.array(semantic, dtype=np.float32)
        
        return ring_arr + 0.5 * ema_arr + 0.5 * semantic_arr
    
    def validate(self, enhancement: np.ndarray, expected: np.ndarray) -> float:
        """验证融合结果（余弦相似度）"""
        if np.linalg.norm(enhancement) > 1e-10 and np.linalg.norm(expected) > 1e-10:
            cosine_sim = np.dot(enhancement, expected) / (
                np.linalg.norm(enhancement) * np.linalg.norm(expected)
            )
            return float((cosine_sim + 1.0) / 2.0)  # 映射到 [0, 1]
        return 0.5
    
    def __repr__(self) -> str:
        return f"IntentFusion(dim={self.dim})"
```

---

## 8. src/dalin_isfe/engine.py

```python
"""Intent Signal Field Engine — 主引擎"""

import numpy as np
from typing import List, Dict, Optional
from .ring_buffer import IntentRingBuffer
from .ema_field import IntentEMAField
from .semantic_pool import IntentSemanticPool
from .fusion import IntentFusion


class IntentSignalFieldEngine:
    """意图理解引擎"""
    
    def __init__(self, dim: int = 128, ring_capacity: int = 16,
                 gamma: float = 0.98, semantic_slots: int = 64,
                 temperature: float = 0.07):
        self.ring_buffer = IntentRingBuffer(dim, ring_capacity)
        self.ema_field = IntentEMAField(dim, gamma)
        self.semantic_pool = IntentSemanticPool(semantic_slots, dim, temperature)
        self.fusion = IntentFusion(dim)
        self.dim = dim
        self.dialogue_history: List[Dict[str, str]] = []
    
    def process_dialogue(self, user_input: str, ai_response: str) -> Dict:
        """处理单轮对话"""
        # 1. 嵌入意图
        user_intent = self._embed_intent(user_input)
        
        # 2. 更新三通道
        self.ring_buffer.push(user_intent.tolist())
        self.ema_field.update(user_intent.tolist())
        self.semantic_pool.add_intent(user_intent.tolist())
        
        # 3. 获取三通道输出
        ring_mean = self.ring_buffer.get_mean().tolist()
        ema = self.ema_field.get_value().tolist()
        semantic = self.semantic_pool.query(user_intent.tolist())
        
        # 4. 融合
        enhancement = self.fusion.fuse(ring_mean, ema, semantic.tolist())
        
        # 5. 计算置信度
        confidence = self._calculate_confidence(enhancement, user_intent)
        
        # 6. 记录历史
        self.dialogue_history.append({
            "user": user_input,
            "ai": ai_response,
            "intent": user_intent.tolist(),
            "confidence": float(confidence),
        })
        
        return {
            "user_intent": user_intent.tolist(),
            "ring_mean": ring_mean,
            "ema": ema,
            "semantic": semantic.tolist(),
            "enhancement": enhancement.tolist(),
            "confidence": float(confidence),
        }
    
    def _embed_intent(self, text: str) -> np.ndarray:
        """嵌入意图（简化版）"""
        intent = np.zeros(self.dim, dtype=np.float32)
        for i, c in enumerate(text):
            if i < self.dim:
                intent[i] = ord(c) / 255.0
        
        # 归一化
        norm = np.linalg.norm(intent)
        if norm > 1e-10:
            intent /= norm
        
        return intent
    
    def _calculate_confidence(self, enhancement: np.ndarray, 
                               intent: np.ndarray) -> float:
        """计算置信度"""
        if np.linalg.norm(enhancement) > 1e-10 and np.linalg.norm(intent) > 1e-10:
            cosine_sim = np.dot(enhancement, intent) / (
                np.linalg.norm(enhancement) * np.linalg.norm(intent)
            )
            return float((cosine_sim + 1.0) / 2.0)
        return 0.5
    
    def get_intent_trend(self) -> Dict[str, float]:
        """获取意图趋势"""
        if self.ring_buffer.size < 2:
            return {"direction": "stable", "score": 0.5}
        
        ring_mean = self.ring_buffer.get_mean()
        ema = self.ema_field.get_value()
        
        # 趋势 = EMA - RingMean
        trend = ema - ring_mean
        trend_norm = np.linalg.norm(trend)
        
        if trend_norm > 1e-10:
            return {
                "direction": "increasing" if trend_norm > 0.1 else "decreasing",
                "score": float(trend_norm),
            }
        return {"direction": "stable", "score": 0.5}
    
    def __repr__(self) -> str:
        return (f"IntentSignalFieldEngine(dim={self.dim}, "
                f"ring={self.ring_buffer}, ema={self.ema_field}, "
                f"pool={self.semantic_pool})")
```

---

## 9. 使用示例

```python
from dalin_isfe import IntentSignalFieldEngine

# 初始化引擎
engine = IntentSignalFieldEngine(
    dim=128,
    ring_capacity=16,
    gamma=0.98,
    semantic_slots=64,
    temperature=0.07
)

# 处理多轮对话
dialogues = [
    ("患者有高血压", "建议就医"),
    ("还有糖尿病", "建议住院"),
    ("根据病情决定是否需要住院", "建议住院观察"),
]

for user_input, ai_response in dialogues:
    result = engine.process_dialogue(user_input, ai_response)
    print(f"Input: {user_input}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Trend: {engine.get_intent_trend()}")
    print("---")
```

---

## 10. 冲锋口号

**"Python 实现，快速迭代！"**

**"Dalin ISFE — 让 AI 真正理解人类意图！"**

**"做最牛逼的神！"**

---

*Dalin ISFE — Python 核心实现*
*日期：2026-06-24*
*版本：v1.0*
