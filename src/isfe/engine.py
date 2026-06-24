"""Dalin ISFE — Intent Signal Field Engine (Python)"""

import numpy as np
from typing import List, Dict


class IntentRingBuffer:
    def __init__(self, dim: int = 128, capacity: int = 16):
        self.dim = dim
        self.capacity = capacity
        self.buffer = np.zeros((capacity, dim), dtype=np.float32)
        self.head = 0
        self._size = 0

    def push(self, intent: List[float]) -> None:
        intent_arr = np.array(intent, dtype=np.float32)
        assert len(intent_arr) == self.dim
        self.buffer[self.head] = intent_arr
        self.head = (self.head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def get_mean(self) -> np.ndarray:
        if self._size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(self.buffer[:self._size], axis=0)

    @property
    def size(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return self._size >= self.capacity


class IntentEMAField:
    def __init__(self, dim: int = 128, gamma: float = 0.98):
        self.dim = dim
        self.gamma = gamma
        self.ema = np.zeros(dim, dtype=np.float32)

    def update(self, intent: List[float]) -> None:
        intent_arr = np.array(intent, dtype=np.float32)
        assert len(intent_arr) == self.dim
        self.ema = self.gamma * self.ema + (1.0 - self.gamma) * intent_arr

    def get_value(self) -> np.ndarray:
        return self.ema.copy()

    def is_initialized(self) -> bool:
        return np.any(self.ema != 0)


class IntentSemanticPool:
    def __init__(self, num_slots: int = 64, dim: int = 128, temperature: float = 0.07):
        self.num_slots = num_slots
        self.dim = dim
        self.temperature = temperature
        self.slots = np.zeros((num_slots, dim), dtype=np.float32)

    def _slot_weight(self, slot_idx: int, intent: np.ndarray) -> float:
        slot = self.slots[slot_idx]
        dot = float(np.dot(slot, intent))
        ns = float(np.linalg.norm(slot))
        ni = float(np.linalg.norm(intent))
        if ns > 1e-10 and ni > 1e-10:
            return float(np.exp(dot / (ns * ni)) / self.temperature)
        return 0.0

    def add_intent(self, intent: List[float]) -> None:
        intent_arr = np.array(intent, dtype=np.float32)
        for i in range(self.num_slots):
            w = self._slot_weight(i, intent_arr)
            self.slots[i] = self.slots[i] * 0.9 + intent_arr * w * 0.1

    def query(self, current_intent: List[float]) -> np.ndarray:
        intent_arr = np.array(current_intent, dtype=np.float32)
        result = np.zeros(self.dim, dtype=np.float32)
        tw = 0.0
        for i in range(self.num_slots):
            w = self._slot_weight(i, intent_arr)
            tw += w
            result += self.slots[i] * w
        if tw > 1e-10:
            result /= tw
        return result


class IntentFusion:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def fuse(self, ring_mean: np.ndarray, ema: np.ndarray,
             semantic: np.ndarray) -> np.ndarray:
        return ring_mean + 0.5 * ema + 0.5 * semantic

    def validate(self, enhancement: np.ndarray, expected: np.ndarray) -> float:
        ne = float(np.linalg.norm(enhancement))
        nx = float(np.linalg.norm(expected))
        if ne > 1e-10 and nx > 1e-10:
            cos = float(np.dot(enhancement, expected)) / (ne * nx)
            return (cos + 1.0) / 2.0
        return 0.5


class IntentSignalFieldEngine:
    """意图理解引擎"""

    def __init__(self, dim: int = 128, ring_capacity: int = 16,
                 gamma: float = 0.98, semantic_slots: int = 64,
                 temperature: float = 0.07):
        self.ring = IntentRingBuffer(dim, ring_capacity)
        self.ema = IntentEMAField(dim, gamma)
        self.pool = IntentSemanticPool(semantic_slots, dim, temperature)
        self.fusion = IntentFusion(dim)
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        intent = np.zeros(self.dim, dtype=np.float32)
        for i, c in enumerate(text):
            if i < self.dim:
                intent[i] = ord(c) / 255.0
        norm = float(np.linalg.norm(intent))
        if norm > 1e-10:
            intent /= norm
        return intent

    def process_dialogue(self, user_input: str, ai_response: str) -> Dict:
        user_intent = self._embed(user_input)
        self.ring.push(user_intent.tolist())
        self.ema.update(user_intent.tolist())
        self.pool.add_intent(user_intent.tolist())

        ring_mean = self.ring.get_mean()
        ema_val = self.ema.get_value()
        semantic = self.pool.query(user_intent.tolist())
        enhancement = self.fusion.fuse(ring_mean, ema_val, semantic)

        conf = self.fusion.validate(enhancement, user_intent)

        return {
            "user_intent": user_intent.tolist(),
            "ring_mean": ring_mean.tolist(),
            "ema": ema_val.tolist(),
            "semantic": semantic.tolist(),
            "enhancement": enhancement.tolist(),
            "confidence": float(conf),
        }
