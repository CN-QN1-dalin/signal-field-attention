#!/usr/bin/env python3
"""
Soma Benchmark Suite v3 - 标准基准测试框架
==========================================
公平对比 SFA vs Causal Standard Attention（共享权重）

关键设计决策：
- SFA 是因果注意力（causal），每个 token 只 attention 到历史
- Standard Attention 必须加因果掩码（mask = tril - eye）
- t=0 时 SFA ring_buffer 为空 → 输出 zeros
- t=0 时 Causal SA mask 全 -inf → softmax uniform → 非零
- 这是设计预期差异，不影响整体正确性

测试项：
1. 正确性验证：相同权重下 SFA vs Causal Attention 输出误差
2. 推理速度：Prefill 和 Decode 阶段耗时
3. 内存压缩：KV Cache 内存占用
4. 困惑度：在真实文本上的 PPL

作者：贾大林
日期：2026-06-15
"""

import mlx.core as mx
import numpy as np
import time
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "01_soma_engine"))

from soma_engine import (
    SignalFieldLayer, EngineConfig, create_soma_engine, RingKVBuffer
)


@dataclass
class BenchmarkConfig:
    model_path: str = "/Users/apple/Models/Qwen2.5-0.5B-Instruct"
    k: int = 16
    gamma: float = 0.98
    alpha: float = 0.1
    seed: int = 42
    seq_lengths: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096])


def random_qkv_weights(dims: int, num_heads: int, num_kv_heads: int, seed: int = 42):
    """生成一致的 QKV/Output 权重用于公平对比"""
    np.random.seed(seed)
    scale_qkv = np.sqrt(2.0 / (dims + 3 * dims))
    qkv_w = np.random.randn(dims, 3 * dims).astype(np.float32) * scale_qkv
    scale_out = np.sqrt(2.0 / (dims + dims))
    out_w = np.random.randn(dims, dims).astype(np.float32) * scale_out
    return mx.array(qkv_w), mx.array(out_w)


class CausalStandardAttention:
    """因果标准 Attention 层（与 SFA 对齐）
    
    使用因果掩码：每个 token 只 attention 到历史（不含自己）。
    mask = tril - eye，即下三角去掉对角线。
    
    注意：t=0 时 mask 全为 -inf，softmax 产生 uniform 分布，
    这与 SFA 中 t=0 ring_buffer 为空输出 zeros 不同。
    这是设计预期，不影响整体正确性。
    """
    def __init__(self, dims: int, num_heads: int, qkv_w: mx.array, out_w: mx.array):
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.qkv_weight = qkv_w
        self.out_weight = out_w

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq, d = x.shape
        x_flat = x.reshape(batch * seq, d)
        qkv = x_flat @ self.qkv_weight
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (0, 1, 3, 2, 4))
        q = qkv[:, :, :, 0, :]
        k = qkv[:, :, :, 1, :]
        v = qkv[:, :, :, 2, :]

        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))

        scores = (q_t @ mx.transpose(k_t, (0, 1, 3, 2))) * self.scale
        # 因果掩码：排除自身和未来
        mask = mx.tril(mx.ones((seq, seq))) - mx.eye(seq)
        scores = mx.where(mask > 0, scores, -1e10)
        w = mx.softmax(scores, axis=-1)
        attn = mx.transpose(w @ v_t, (0, 2, 1, 3)).reshape(batch, seq, d)
        return attn @ self.out_weight


def build_soma_layer(dims: int, num_heads: int, num_kv_heads: int,
                      qkv_w: mx.array, out_w: mx.array,
                      k: int = 16, gamma: float = 0.98, alpha: float = 0.1) -> SignalFieldLayer:
    """构建带固定权重的 Soma Engine"""
    engine = create_soma_engine(dims=dims, num_heads=num_heads,
                                 num_kv_heads=num_kv_heads, k=k,
                                 gamma=gamma, alpha=alpha)
    engine.qkv_weight = qkv_w
    engine.out_weight = out_w
    return engine


def cosine_sim(a: mx.array, b: mx.array) -> float:
    dot = float(mx.sum(a * b))
    na = float(mx.linalg.norm(a))
    nb = float(mx.linalg.norm(b))
    return dot / (na * nb + 1e-10)


def correctness_benchmark(cfg: BenchmarkConfig):
    """Test 1: 正确性验证 - 相同权重下 SFA vs Causal Attention"""
    print("\n" + "=" * 60)
    print("🔬 Test 1: Correctness (Shared Weights, Causal Mask)")
    print("=" * 60)
    print("\n  注：t=0 时 SFA ring_buffer 为空 → zeros")
    print("     Causal SA mask 全 -inf → uniform → 非零")
    print("     这是设计预期差异，不影响整体正确性。")

    dims = 128
    heads = 4
    kv_heads = 2

    qkv_w, out_w = random_qkv_weights(dims, heads, kv_heads, cfg.seed)

    std_attn = CausalStandardAttention(dims, heads, qkv_w, out_w)
    soma = build_soma_layer(dims, heads, kv_heads, qkv_w, out_w,
                            k=cfg.k, gamma=cfg.gamma, alpha=cfg.alpha)
    # 关闭衰减和远场以验证近场通道的正确性
    soma.decay_table.table = mx.ones(cfg.k * 16)  # 足够大的表
    soma.alpha = 0.0

    results = {}
    print(f"\n  Config: dims={dims}, heads={heads}, k={cfg.k}, gamma={cfg.gamma}, alpha={cfg.alpha}")
    print(f"  {'SeqLen':>8} | {'MeanErr':>12} | {'MaxErr':>12} | {'Sim':>12} | {'Sim(skip t0)':>14} | {'Status'}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}")

    for seq_len in [16, 32, 64, 128, 256, 512, 1024]:
        x = mx.random.normal((1, seq_len, dims))

        out_std = std_attn(x)
        out_soma = soma.prefill(x, full_mode=True)[0]
        mx.eval(out_std, out_soma)

        # 整体相似度
        sim_all = cosine_sim(out_std, out_soma)
        mean_err = float(mx.mean(mx.abs(out_std - out_soma)))
        max_err = float(mx.max(mx.abs(out_std - out_soma)))

        # 跳过 t=0 的相似度
        if seq_len > 1:
            sim_skip_t0 = cosine_sim(out_std[:, 1:, :], out_soma[:, 1:, :])
        else:
            sim_skip_t0 = sim_all

        status = "✅ PASS" if sim_skip_t0 > 0.9999 else "⚠️ WARN" if sim_skip_t0 > 0.99 else "❌ FAIL"
        print(f"  {seq_len:>8} | {mean_err:>12.8f} | {max_err:>12.6f} | {sim_all:>12.6f} | {sim_skip_t0:>14.8f} | {status}")

        results[f"seq_{seq_len}"] = {
            "mean_error": round(mean_err, 10),
            "max_error": round(max_err, 6),
            "similarity_all": round(sim_all, 6),
            "similarity_skip_t0": round(sim_skip_t0, 6)
        }

    return results


def speed_benchmark(cfg: BenchmarkConfig):
    """Test 2: 推理速度对比"""
    print("\n" + "=" * 60)
    print("⚡ Test 2: Inference Speed")
    print("=" * 60)

    dims = 896
    heads = 14
    kv_heads = 2

    qkv_w, out_w = random_qkv_weights(dims, heads, kv_heads, cfg.seed)

    std_attn = CausalStandardAttention(dims, heads, qkv_w, out_w)
    soma = build_soma_layer(dims, heads, kv_heads, qkv_w, out_w,
                            k=cfg.k, gamma=cfg.gamma, alpha=cfg.alpha)

    results = {}
    print(f"\n  Config: dims={dims}, heads={heads}, k={cfg.k}")
    print(f"  {'SeqLen':>8} | {'Std Prefill':>14} | {'Soma Prefill':>14} | {'Speedup':>10} | {'Decode':>12}")
    print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}-+-{'-'*12}")

    for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
        x = mx.random.normal((1, seq_len, dims))

        # Warmup
        _ = std_attn(x); mx.eval(_)
        _ = soma.prefill(x); mx.eval(_)

        # Standard Attention Prefill
        t0 = time.time()
        for _ in range(10):
            out = std_attn(x); mx.eval(out)
        t_std = (time.time() - t0) / 10

        # Soma Prefill
        t0 = time.time()
        for _ in range(10):
            out = soma.prefill(x); mx.eval(out)
        t_soma = (time.time() - t0) / 10

        # Soma Decode (O(1))
        x_dec = mx.random.normal((1, 1, dims))
        _, fs, rb = soma.prefill(x)
        t0 = time.time()
        for _ in range(100):
            out, _, _ = soma.decode_step(x_dec, fs, rb); mx.eval(out)
        t_decode = (time.time() - t0) / 100

        speedup = t_std / t_soma if t_soma > 0 else 0
        print(f"  {seq_len:>8} | {t_std*1000:>10.1f}ms | {t_soma*1000:>10.1f}ms | {speedup:>8.2f}x | {t_decode*1000:>9.4f}ms")

        results[f"seq_{seq_len}"] = {
            "std_prefill_ms": round(t_std * 1000, 2),
            "soma_prefill_ms": round(t_soma * 1000, 2),
            "speedup": round(speedup, 2),
            "decode_ms_token": round(t_decode * 1000, 4)
        }

    return results


def memory_benchmark(cfg: BenchmarkConfig):
    """Test 3: 内存压缩"""
    print("\n" + "=" * 60)
    print("💾 Test 3: Memory Compression")
    print("=" * 60)

    dims = 896
    heads = 14
    head_dim = dims // heads
    k = cfg.k

    soma_mem = 2 * k * heads * head_dim * 4 + heads * head_dim * 4

    results = {}
    print(f"\n  Soma 固定内存: {soma_mem/1024:.1f} KB")
    print(f"  {'SeqLen':>10} | {'Std Attention':>14} | {'Soma':>10} | {'Compression':>14}")
    print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*10}-+-{'-'*14}")

    for seq_len in [128, 512, 1024, 4096, 16384, 65536]:
        std_mem = 2 * seq_len * heads * head_dim * 4
        comp = std_mem / soma_mem if soma_mem > 0 else float('inf')

        print(f"  {seq_len:>10} | {std_mem/1024:>10.1f} KB | {soma_mem/1024:>6.1f} KB | {comp:>10.1f}x")

        results[f"len_{seq_len}"] = {
            "standard_kb": round(std_mem / 1024, 1),
            "soma_kb": round(soma_mem / 1024, 1),
            "compression_ratio": round(comp, 1)
        }

    return results


def ppl_benchmark(cfg: BenchmarkConfig):
    """Test 4: 困惑度测试"""
    print("\n" + "=" * 60)
    print("📊 Test 4: Perplexity (PPL)")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ⚠️ transformers 未安装，跳过 PPL 测试")
        return {}

    model_path = cfg.model_path
    if not os.path.exists(model_path):
        print(f"  ⚠️ 模型不存在: {model_path}，跳过 PPL 测试")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    model.eval()

    sample_texts = [
        "The Transformer architecture has revolutionized natural language processing.",
        "深度学习模型在处理自然语言任务时表现出色。",
        "Self-attention allows the model to focus on relevant parts of the input.",
        "大语言模型正在改变我们与技术互动的方式。",
        "Knowledge distillation transfers knowledge from large to small models.",
    ] * 10

    results = {}
    import torch

    for seq_len in [64, 256, 1024]:
        print(f"\n  Seq len={seq_len}:")

        total_logprob = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in sample_texts:
                enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
                input_ids = enc["input_ids"]
                if input_ids.shape[1] < 10:
                    continue
                for start in range(0, input_ids.shape[1] - 5, 5):
                    chunk = input_ids[:, start:start+seq_len]
                    if chunk.shape[1] < 10:
                        continue
                    outputs = model(chunk)
                    logits = outputs.logits[:, :-1, :]
                    targets = chunk[:, 1:]
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="sum"
                    )
                    total_logprob += float(loss.item())
                    total_tokens += targets.numel()

        ppl = float(np.exp(total_logprob / max(total_tokens, 1)))
        print(f"    HF Model PPL: {ppl:.4f} ({total_tokens} tokens)")

        results[f"seq_{seq_len}"] = {
            "hf_ppl": round(ppl, 4),
            "tokens_tested": total_tokens
        }

    return results


def save_results(all_results, cfg):
    output = BASE_DIR / "benchmark_results.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Results saved to: {output}")


def run_all():
    print("=" * 60)
    print("🏆 Soma Benchmark Suite v3")
    print("公平对比：SFA vs Causal Standard Attention（共享权重）")
    print("=" * 60)

    cfg = BenchmarkConfig()
    all_results = {}

    all_results["correctness"] = correctness_benchmark(cfg)
    all_results["speed"] = speed_benchmark(cfg)
    all_results["memory"] = memory_benchmark(cfg)
    all_results["ppl"] = ppl_benchmark(cfg)

    save_results(all_results, cfg)

    print("\n" + "=" * 60)
    print("✅ All benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
