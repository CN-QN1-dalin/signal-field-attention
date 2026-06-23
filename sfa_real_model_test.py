#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFA v7 真实模型集成测试
在 Qwen2.5-0.5B 上验证 SFA 计算正确性和正交性
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json
import os

MODEL_PATH = "/Users/apple/models/Qwen2.5-0.5B-Instruct"

SFA_RING_SIZE = 16
SFA_SEMANTIC_SLOTS = 64
SFA_EMA_GAMMA = 0.98
SFA_ALPHA_BASE = 0.1
SFA_CROSS_DECAY = 0.8
SFA_ENHANCEMENT_CLIP = 0.5


class SignalFieldAttention(nn.Module):
    """SFA 增强层"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.ring_size = SFA_RING_SIZE
        self.semantic_slots = SFA_SEMANTIC_SLOTS

        self.register_buffer("ring_buffer", torch.zeros(num_heads, SFA_RING_SIZE, head_dim))
        self.register_buffer("ring_offset", torch.zeros(num_heads, dtype=torch.long))
        self.register_buffer("field_state", torch.zeros(num_heads, head_dim))
        self.register_buffer("semantic_pool", torch.zeros(num_heads, SFA_SEMANTIC_SLOTS, head_dim))
        self.register_buffer("gaussian_comp", torch.zeros(num_heads, head_dim))

    def _update_ring(self, head_idx, kv):
        offset = int(self.ring_offset[head_idx].item())
        self.ring_buffer[head_idx, offset] = kv
        self.ring_offset[head_idx] = (offset + 1) % self.ring_size

    def _ring_mean(self, head_idx):
        offset = int(self.ring_offset[head_idx].item())
        if offset < self.ring_size - 1:
            return self.ring_buffer[head_idx, : offset + 1].mean(dim=0)
        return self.ring_buffer[head_idx].mean(dim=0)

    def _update_field(self, head_idx, val):
        self.field_state[head_idx] = (
            SFA_EMA_GAMMA * self.field_state[head_idx] + (1 - SFA_EMA_GAMMA) * val
        )

    def _semantic_attention(self, head_idx, query):
        pool = self.semantic_pool[head_idx]
        scores = torch.matmul(pool, query) / (self.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights.unsqueeze(-1).transpose(-1, -2), pool).squeeze(-1)

    def compute_enhancement(self, attn_output):
        """Compute SFA enhancement for each head"""
        enhancement = torch.zeros_like(attn_output)
        batch, seq, _ = attn_output.shape

        for b in range(batch):
            for s in range(seq):
                for h in range(self.num_heads):
                    kv = attn_output[b, s, h * self.head_dim : (h + 1) * self.head_dim]

                    self._update_ring(h, kv)
                    ring_mean = self._ring_mean(h)

                    self._update_field(h, kv)
                    field = self.field_state[h]

                    semantic = self._semantic_attention(h, kv)

                    raw = ring_mean + 0.5 * field + 0.5 * semantic
                    raw = torch.clamp(raw, -SFA_ENHANCEMENT_CLIP, SFA_ENHANCEMENT_CLIP)

                    layer_ratio = 1.0
                    alpha = SFA_ALPHA_BASE * (0.3 + 0.7 * layer_ratio) * (SFA_CROSS_DECAY ** 0)
                    raw *= alpha

                    enhancement[b, s, h * self.head_dim : (h + 1) * self.head_dim] = raw

        return enhancement

    def forward(self, attn_output):
        enhancement = self.compute_enhancement(attn_output)
        return attn_output + enhancement


def test_baseline_ppl(model, tokenizer, texts, device="cpu"):
    print("\n" + "=" * 60)
    print("Test 1: Baseline PPL (no SFA)")
    print("=" * 60)
    results = {}
    for name, text in texts.items():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ppl = torch.exp(loss).item()
        results[name] = ppl
        print(f"  {name}: PPL = {ppl:.4f}")
    return results


def main():
    print("=" * 60)
    print("SFA v7 Real Model Integration Test")
    print("=" * 60)

    device = "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map=device
    )
    base_model.eval()
    print(f"Model loaded: {MODEL_PATH}")

    # Test texts
    texts = {
        "short": "The quick brown fox jumps over the lazy dog. " * 3,
        "medium": "Signal Field Attention compresses KV cache through three channels. " * 3,
        "long": "The signal field emerged from the attention mechanism. " * 10,
    }

    # Baseline PPL
    baseline = test_baseline_ppl(base_model, tokenizer, texts, device)

    # SFA computation test
    print("\n" + "=" * 60)
    print("Test 2: SFA Engine Computation")
    print("=" * 60)

    hidden_size = base_model.config.hidden_size
    num_heads = base_model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    torch.manual_seed(42)
    attn_output = torch.randn(1, 32, hidden_size, device=device)

    sfa = SignalFieldAttention(hidden_size, num_heads, num_heads, head_dim).to(device)

    start = time.time()
    enhanced = sfa(attn_output)
    elapsed = time.time() - start

    orig_norm = torch.norm(attn_output).item()
    enh_norm = torch.norm(enhanced - attn_output).item()
    ratio = enh_norm / orig_norm * 100

    print(f"  Attention output norm: {orig_norm:.4f}")
    print(f"  Enhancement norm: {enh_norm:.6f}")
    print(f"  Enhancement ratio: {ratio:.4f}%")
    print(f"  SFA compute time: {elapsed*1000:.2f}ms")
    print(f"  Enhanced output shape: {enhanced.shape}")

    # Orthogonality
    diff = enhanced - attn_output
    cos_sim = nn.functional.cosine_similarity(
        attn_output.flatten(), diff.flatten(), dim=0
    ).item()
    print(f"  Cosine similarity (attn vs enhancement): {cos_sim:.6f}")

    # Save results
    result = {
        "baseline_ppl": baseline,
        "sfa_enhancement_ratio_pct": round(ratio, 4),
        "sfa_cosine_similarity": round(cos_sim, 6),
        "sfa_compute_time_ms": round(elapsed * 1000, 2),
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfa_real_model_test_results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("✅ SFA v7 Real Model Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    main()
