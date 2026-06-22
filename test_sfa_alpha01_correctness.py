#!/usr/bin/env python3
"""
============================================================
🎯 SFA v7 α=0.1 完整正确性测试
============================================================
测试内容:
  a. PPL 对比：baseline vs SFA-enhanced (α=0.1)
  b. 输出一致性：cosine similarity + Pearson correlation
  c. 正交性验证：SFA enhancement 向量与原始 attention 输出的夹角
  d. 稳定性：不同序列长度下的 PPL 变化
  e. 内存占用：SFA 状态额外消耗的内存

硬件: Apple Silicon M1 Pro, 16GB RAM (Metal/CPU)
模型: Qwen2.5-0.5B-Instruct (或可用的最小模型)
α: 0.1 (全 SFA 增强系数)
============================================================
"""

import os
import sys
import json
import math
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 配置
# ============================================================
@dataclass
class TestConfig:
    model_name: str = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")
    alpha: float = 0.1
    device: str = "cpu"  # MPS cross_entropy has known issues; use CPU for correctness tests
    dtype: str = "float16"
    seq_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    seed: int = 42
    num_test_sequences: int = 5
    
    # SFA 参数
    ring_size: int = 16
    semantic_slots: int = 64
    ema_gamma: float = 0.98
    cross_decay: float = 0.7
    enhancement_clip: float = 0.01


# ============================================================
# SFA v7 三通道增强引擎（Python 参考实现）
# ============================================================
class SFATripleChannelEngine:
    """
    SFA v7 三通道增强引擎的 Python 参考实现
    
    三个通道:
    1. RingBuffer (短期): 最近 N 个 token 的滑动平均
    2. EMA Field (长期): 指数移动平均的 attention 输出
    3. Semantic Pool (全局): 注意力聚合的语义槽
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.alpha = config.alpha
        self.ring_size = config.ring_size
        self.semantic_slots = config.semantic_slots
        self.ema_gamma = config.ema_gamma
        self.cross_decay = config.cross_decay
        self.enhancement_clip = config.enhancement_clip
        
        # SFA 状态
        self.ring_buffers: Optional[List[torch.Tensor]] = None
        self.field_states: Optional[List[torch.Tensor]] = None
        self.semantic_pool: Optional[torch.Tensor] = None
        self.ring_offsets: Optional[List[int]] = None
        self.initialized = False
    
    def initialize(self, n_layers: int, hidden_size: int):
        """初始化 SFA 状态"""
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.ring_buffers = [
            torch.zeros(self.ring_size, hidden_size, device=self.config.device)
            for _ in range(n_layers)
        ]
        self.field_states = [
            torch.zeros(hidden_size, device=self.config.device)
            for _ in range(n_layers)
        ]
        self.semantic_pool = torch.zeros(
            self.semantic_slots, hidden_size, device=self.config.device
        )
        self.ring_offsets = [0] * n_layers
        self.initialized = True
    
    def reset(self):
        """重置所有序列状态"""
        if not self.initialized:
            return
        for i in range(self.n_layers):
            self.ring_buffers[i].zero_()
            self.field_states[i].zero_()
            self.ring_offsets[i] = 0
        self.semantic_pool.zero_()
    
    def compute_enhancement(self, layer_idx: int, attn_output: torch.Tensor) -> torch.Tensor:
        """
        计算单层的 SFA 增强信号
        
        Args:
            layer_idx: 层索引
            attn_output: attention 输出 [batch, seq_len, hidden]
                         取最后一个 token 的 mean 作为当前状态近似
        
        Returns:
            enhancement: [hidden] SFA 增强向量
        """
        if not self.initialized:
            return torch.zeros(self.hidden_size, device=self.config.device)
        
        # 取 attn_output 的 mean 作为当前状态近似
        current_state = attn_output.mean(dim=(0, 1))  # [hidden]
        
        # === 通道 1: Ring Buffer ===
        ring = self.ring_buffers[layer_idx]
        offset = self.ring_offsets[layer_idx]
        ring[offset] = current_state
        self.ring_offsets[layer_idx] = (offset + 1) % self.ring_size
        
        # 计算 ring mean
        valid_entries = min(offset + 1, self.ring_size) if offset < self.ring_size else self.ring_size
        ring_mean = ring[:valid_entries].mean(dim=0)  # [hidden]
        
        # === 通道 2: EMA Field ===
        field = self.field_states[layer_idx]
        new_field = self.ema_gamma * field + (1 - self.ema_gamma) * current_state
        self.field_states[layer_idx] = new_field
        field = new_field
        
        # === 通道 3: Semantic Pool (简化版) ===
        # 使用 field 作为 query 计算语义注意力
        # 简化：直接使用 field 本身作为 semantic signal
        semantic_signal = field
        
        # === 融合 ===
        enhancement = ring_mean + 0.5 * (field + semantic_signal)
        
        # Clip
        enhancement = torch.clamp(enhancement, -self.enhancement_clip, self.enhancement_clip)
        
        # Scale by alpha with layer decay
        layer_ratio = layer_idx / max(self.n_layers - 1, 1)
        alpha_eff = self.alpha * (0.3 + layer_ratio * 0.7) * (self.cross_decay ** layer_idx)
        enhancement *= alpha_eff
        
        return enhancement
    
    def get_memory_usage(self) -> Dict[str, int]:
        """计算 SFA 额外占用的内存（字节）"""
        hs = self.hidden_size
        nl = self.n_layers
        rs = self.ring_size
        ss = self.semantic_slots
        
        ring_mem = nl * rs * hs * 2  # float16
        field_mem = nl * hs * 2
        semantic_mem = ss * hs * 2
        
        return {
            "ring_buffers_bytes": ring_mem,
            "field_states_bytes": field_mem,
            "semantic_pool_bytes": semantic_mem,
            "total_bytes": ring_mem + field_mem + semantic_mem,
            "total_mb": (ring_mem + field_mem + semantic_mem) / (1024 * 1024),
        }


# ============================================================
# 测试辅助函数
# ============================================================
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个向量的 cosine similarity"""
    a_norm = a / (a.norm() + 1e-8)
    b_norm = b / (b.norm() + 1e-8)
    return float((a_norm * b_norm).mean().item())


def pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算 Pearson 相关系数"""
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
    if denominator < 1e-8:
        return 0.0
    return float(numerator / denominator)


def compute_ppl(model, tokenizer, input_ids: torch.Tensor, device: str) -> Tuple[float, torch.Tensor]:
    """
    计算困惑度 (Perplexity)
    
    Returns:
        ppl: 困惑度值
        logits: 原始模型的 logits
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), use_cache=False)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # Shift so that tokens predict next shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 排除 padding token
        valid_mask = shift_labels != -100
        valid_losses = losses[valid_mask.view(-1)]
        
        if valid_losses.numel() == 0:
            return float('inf'), logits
        
        perplexity = float(torch.exp(valid_losses.mean()).item())
    
    return perplexity, logits


def apply_sfa_enhancement(
    model, tokenizer, input_ids: torch.Tensor, sfa_engine: SFATripleChannelEngine,
    device: str, config: TestConfig
) -> Tuple[float, torch.Tensor]:
    """
    应用 SFA 增强后的困惑度计算
    
    通过在每个 decoder layer 的 attention output 之后注入 SFA 增强信号
    """
    enhanced_logits_list = []
    
    with torch.no_grad():
        # 获取 hidden size 和 n_layers
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        
        # 初始化 SFA 引擎
        sfa_engine.initialize(n_layers, hidden_size)
        
        # 前向传播，逐层注入 SFA 增强
        outputs = model.model.embed_tokens(input_ids.to(device))
        
        past_key_values = None
        
        for layer_idx, decoder_layer in enumerate(model.model.layers):
            # 标准 decoder layer 前向
            layer_outputs = decoder_layer(
                inputs_embeds=outputs,
                attention_mask=None,
                position_ids=None,
                past_key_value=past_key_values,
                use_cache=False,
                output_attentions=False,
            )
            
            # 获取 attention output (layer_outputs[0] 是 hidden state)
            attn_output = layer_outputs[0]
            
            # 计算 SFA 增强
            enhancement = sfa_engine.compute_enhancement(layer_idx, attn_output)
            
            # 注入增强（广播到 [batch, seq_len, hidden]）
            enhancement_broadcast = enhancement.unsqueeze(0).unsqueeze(0)
            outputs = attn_output + enhancement_broadcast
            
            # 保存最后一层的输出用于 PPL 计算
            if layer_idx == n_layers - 1:
                enhanced_logits_list.append(outputs)
        
        # 通过 LM head 得到 logits
        if enhanced_logits_list:
            last_hidden = enhanced_logits_list[-1]
            logits = model.lm_head(last_hidden)
        else:
            logits = model.lm_head(outputs)
    
    # 计算增强后的 PPL
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    valid_mask = shift_labels != -100
    valid_losses = losses[valid_mask.view(-1)]
    
    if valid_losses.numel() == 0:
        return float('inf'), logits
    
    enhanced_ppl = float(torch.exp(valid_losses.mean()).item())
    
    return enhanced_ppl, logits


# ============================================================
# 主测试流程
# ============================================================
def run_tests():
    """运行所有测试"""
    config = TestConfig()
    results = {
        "config": asdict(config),
        "tests": {},
        "summary": {},
    }
    
    print("=" * 60)
    print("🎯 SFA v7 α=0.1 完整正确性测试")
    print("=" * 60)
    print(f"模型: {config.model_name}")
    print(f"设备: {config.device}")
    print(f"α: {config.alpha}")
    print(f"序列长度: {config.seq_lengths}")
    print()
    
    # 加载模型和 tokenizer
    print("⏳ 加载模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.device == "mps" else torch.float32,
        device_map=config.device if config.device != "cpu" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print(f"✅ 模型加载完成 ({config.device})")
    print()
    
    # 生成测试文本
    test_texts = [
        "The quick brown fox jumps over the lazy dog. " * 4,
        "Machine learning is a subset of artificial intelligence. " * 4,
        "Signal Field Attention provides orthogonal information "
        "channels to standard attention mechanisms. " * 4,
        "The Dalin Soma project aims to revolutionize inference "
        "acceleration through algorithmic innovation. " * 4,
        "Apple Silicon offers excellent performance for "
        "Metal-accelerated machine learning workloads. " * 4,
    ]
    
    # ============================================================
    # 测试 a: PPL 对比 (baseline vs SFA-enhanced)
    # ============================================================
    print("=" * 60)
    print("📊 测试 a: PPL 对比 (baseline vs SFA-enhanced)")
    print("=" * 60)
    
    ppl_results = {}
    for seq_len in config.seq_lengths:
        print(f"\n  序列长度: {seq_len}")
        
        # 截断测试文本
        text = test_texts[0][:seq_len]
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        
        # Baseline PPL
        baseline_ppl, _ = compute_ppl(model, tokenizer, input_ids, config.device)
        print(f"    Baseline PPL: {baseline_ppl:.4f}")
        
        # SFA-enhanced PPL
        sfa_engine = SFATripleChannelEngine(config)
        enhanced_ppl, _ = apply_sfa_enhancement(
            model, tokenizer, input_ids, sfa_engine, config.device, config
        )
        print(f"    SFA-enhanced PPL (α={config.alpha}): {enhanced_ppl:.4f}")
        
        # 计算改善百分比
        if baseline_ppl > 0:
            improvement = (baseline_ppl - enhanced_ppl) / baseline_ppl * 100
        else:
            improvement = 0.0
        print(f"    PPL 改善: {improvement:+.2f}%")
        
        ppl_results[seq_len] = {
            "baseline_ppl": baseline_ppl,
            "enhanced_ppl": enhanced_ppl,
            "improvement_pct": improvement,
        }
    
    results["tests"]["ppl_comparison"] = ppl_results
    print("\n✅ PPL 对比测试完成")
    print()
    
    # ============================================================
    # 测试 b: 输出一致性 (Cosine Similarity + Pearson)
    # ============================================================
    print("=" * 60)
    print("📊 测试 b: 输出一致性 (Cosine Similarity + Pearson)")
    print("=" * 60)
    
    consistency_results = {}
    for seq_len in [64, 128, 256]:
        print(f"\n  序列长度: {seq_len}")
        
        text = test_texts[0][:seq_len]
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        
        # Baseline logits
        _, baseline_logits = compute_ppl(model, tokenizer, input_ids, config.device)
        
        # SFA-enhanced logits
        sfa_engine = SFATripleChannelEngine(config)
        enhanced_ppl, enhanced_logits = apply_sfa_enhancement(
            model, tokenizer, input_ids, sfa_engine, config.device, config
        )
        
        # 计算 cosine similarity
        cos_sim = cosine_similarity(baseline_logits.flatten(), enhanced_logits.flatten())
        pearson = pearson_correlation(baseline_logits.flatten(), enhanced_logits.flatten())
        
        print(f"    Cosine Similarity: {cos_sim:.6f}")
        print(f"    Pearson Correlation: {pearson:.6f}")
        
        consistency_results[seq_len] = {
            "cosine_similarity": cos_sim,
            "pearson_correlation": pearson,
        }
    
    results["tests"]["consistency"] = consistency_results
    print("\n✅ 输出一致性测试完成")
    print()
    
    # ============================================================
    # 测试 c: 正交性验证
    # ============================================================
    print("=" * 60)
    print("📊 测试 c: 正交性验证")
    print("=" * 60)
    
    orthogonality_results = []
    for seq_len in [64, 128]:
        print(f"\n  序列长度: {seq_len}")
        
        text = test_texts[0][:seq_len]
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        
        sfa_engine = SFATripleChannelEngine(config)
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        sfa_engine.initialize(n_layers, hidden_size)
        
        # 逐层计算 enhancement 与 attention output 的 cosine similarity
        with torch.no_grad():
            outputs = model.model.embed_tokens(input_ids.to(config.device))
            
            for layer_idx, decoder_layer in enumerate(model.model.layers):
                layer_outputs = decoder_layer(
                    inputs_embeds=outputs,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    use_cache=False,
                )
                
                attn_output = layer_outputs[0]
                enhancement = sfa_engine.compute_enhancement(layer_idx, attn_output)
                
                # 计算 enhancement 与 attn_output 的 cosine similarity
                # 取最后一个 token
                last_token_attn = attn_output[0, -1, :]  # [hidden]
                cos_sim = cosine_similarity(enhancement, last_token_attn)
                
                if layer_idx < 3 or layer_idx == n_layers - 1:  # 只打印前3层和最后一层
                    print(f"    Layer {layer_idx}: Cosine(enhancement, attn) = {cos_sim:.6f}")
                
                orthogonality_results.append({
                    "layer": layer_idx,
                    "cosine_similarity": cos_sim,
                })
    
    # 统计
    avg_cos = np.mean([r["cosine_similarity"] for r in orthogonality_results])
    min_cos = np.min([r["cosine_similarity"] for r in orthogonality_results])
    max_cos = np.max([r["cosine_similarity"] for r in orthogonality_results])
    print(f"\n  平均 Cosine Similarity: {avg_cos:.6f}")
    print(f"  范围: [{min_cos:.6f}, {max_cos:.6f}]")
    print(f"  接近 0 = 正交 ✅")
    
    results["tests"]["orthogonality"] = {
        "per_layer": orthogonality_results,
        "avg_cosine": avg_cos,
        "min_cosine": min_cos,
        "max_cosine": max_cos,
    }
    print("\n✅ 正交性验证测试完成")
    print()
    
    # ============================================================
    # 测试 d: 稳定性（不同序列长度下的 PPL 变化）
    # ============================================================
    print("=" * 60)
    print("📊 测试 d: 稳定性 — 不同序列长度下的 PPL 变化")
    print("=" * 60)
    
    stability_results = {}
    for seq_len in config.seq_lengths:
        print(f"\n  序列长度: {seq_len}")
        
        text = test_texts[0][:seq_len]
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        
        baseline_ppl, _ = compute_ppl(model, tokenizer, input_ids, config.device)
        
        sfa_engine = SFATripleChannelEngine(config)
        enhanced_ppl, _ = apply_sfa_enhancement(
            model, tokenizer, input_ids, sfa_engine, config.device, config
        )
        
        delta = enhanced_ppl - baseline_ppl
        print(f"    Baseline: {baseline_ppl:.4f}")
        print(f"    Enhanced: {enhanced_ppl:.4f}")
        print(f"    Delta: {delta:+.4f}")
        
        stability_results[seq_len] = {
            "baseline_ppl": baseline_ppl,
            "enhanced_ppl": enhanced_ppl,
            "delta": delta,
        }
    
    results["tests"]["stability"] = stability_results
    print("\n✅ 稳定性测试完成")
    print()
    
    # ============================================================
    # 测试 e: 内存占用
    # ============================================================
    print("=" * 60)
    print("📊 测试 e: 内存占用")
    print("=" * 60)
    
    sfa_engine = SFATripleChannelEngine(config)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    sfa_engine.initialize(n_layers, hidden_size)
    
    mem_usage = sfa_engine.get_memory_usage()
    print(f"  Hidden Size: {hidden_size}")
    print(f"  N Layers: {n_layers}")
    print(f"  Ring Buffer: {mem_usage['ring_buffers_bytes'] / 1024:.1f} KB")
    print(f"  Field States: {mem_usage['field_states_bytes'] / 1024:.1f} KB")
    print(f"  Semantic Pool: {mem_usage['semantic_pool_bytes'] / 1024:.1f} KB")
    print(f"  Total: {mem_usage['total_mb']:.2f} MB")
    
    results["tests"]["memory_usage"] = mem_usage
    print("\n✅ 内存占用测试完成")
    print()
    
    # ============================================================
    # 总结
    # ============================================================
    print("=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    
    # PPL 改善
    avg_improvement = np.mean([
        v["improvement_pct"] for v in ppl_results.values()
    ])
    print(f"\n  平均 PPL 改善: {avg_improvement:+.2f}%")
    
    # 正交性
    avg_orth = np.mean([r["cosine_similarity"] for r in orthogonality_results])
    print(f"  平均 Enhancement-Attention Cosine: {avg_orth:.6f}")
    print(f"  {'✅ 正交性良好' if abs(avg_orth) < 0.1 else '⚠️ 正交性不足'}")
    
    # 一致性
    avg_consistency = np.mean([
        v["cosine_similarity"] for v in consistency_results.values()
    ])
    print(f"  平均输出一致性 (Cosine): {avg_consistency:.6f}")
    print(f"  {'✅ 一致性可接受' if avg_consistency > 0.99 else '⚠️ 一致性较低'}")
    
    # 内存
    print(f"  SFA 额外内存: {mem_usage['total_mb']:.2f} MB")
    
    results["summary"] = {
        "avg_ppl_improvement_pct": float(avg_improvement),
        "avg_orthogonality_cosine": float(avg_orth),
        "avg_output_consistency": float(avg_consistency),
        "sfa_memory_mb": mem_usage["total_mb"],
        "overall": "PASS" if avg_improvement < 0 and avg_orth < 0.1 and avg_consistency > 0.99 else "NEEDS_REVIEW",
    }
    
    print(f"\n  总体评估: {results['summary']['overall']}")
    print()
    
    # 保存结果
    output_path = "test_results_sfa_alpha01_correctness.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📄 测试结果已保存到: {output_path}")
    
    return results


if __name__ == "__main__":
    try:
        results = run_tests()
        print("\n🎉 所有测试完成！")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)
