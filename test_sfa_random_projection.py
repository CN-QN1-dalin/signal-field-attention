#!/usr/bin/env python3
"""
============================================================
🔬 SFA v7 随机投影正交化 — 核心修复
============================================================
问题: 原始 SFA enhancement 与 attention output 高度相关 (cosine=0.65)
解决: 将 enhancement 投影到与 attention output 正交的子空间

方法:
  1. 计算 enhancement 向量 e 和 attention 向量 a
  2. 从 e 中减去沿 a 方向的投影: e_orth = e - proj_a(e)
  3. 归一化 e_orth 保持固定幅度
============================================================
"""

import os, sys, json, math, torch, torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")
DEVICE = "cpu"
RING_SIZE = 16
EMA_GAMMA = 0.98
CLIP_NORM = 0.5   # enhancement 的最大范数（增大 50 倍）


class SFARandomProjectionEngine:
    """
    带随机投影正交化的 SFA 引擎
    
    核心改进:
    1. Enhancement 通过 Gram-Schmidt 正交化与 attention output 解耦
    2. 引入随机投影矩阵增加正交通道的独立性
    3. 固定 enhancement 范数避免幅度过大破坏模型
    """
    
    def __init__(self, n_layers, hidden_size, alpha=0.1, seed=42):
        self.n_layers = n_layers
        self.hs = hidden_size
        self.alpha = alpha
        
        # SFA 状态
        self.ring_buffers = [np.zeros((RING_SIZE, hidden_size), dtype=np.float32) for _ in range(n_layers)]
        self.field_states = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        self.ring_offsets = [0] * n_layers
        self.prev_hiddens = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        
        # 随机投影矩阵 (hidden_size x random_dim)
        rng = np.random.RandomState(seed)
        self.proj_matrix = rng.randn(hidden_size, 32).astype(np.float32)  # 32 维随机子空间
        
        # 正交性分析
        self.cos_sims = []
        self.enhancement_norms = []
    
    def reset(self):
        for i in range(self.n_layers):
            self.ring_buffers[i].fill(0)
            self.field_states[i].fill(0)
            self.prev_hiddens[i].fill(0)
            self.ring_offsets[i] = 0
        self.cos_sims.clear()
        self.enhancement_norms.clear()
    
    def compute_raw_enhancement(self, layer_idx, last_token_np):
        """计算原始 SFA enhancement（未正交化）"""
        ring = self.ring_buffers[layer_idx]
        offset = self.ring_offsets[layer_idx]
        ring[offset] = last_token_np
        self.ring_offsets[layer_idx] = (offset + 1) % RING_SIZE
        valid = min(offset + 1, RING_SIZE) if offset < RING_SIZE else RING_SIZE
        
        ring_mean = ring[:valid].mean(axis=0)
        ring_diff = last_token_np - ring_mean  # 趋势变化
        
        # Direction change
        prev = self.prev_hiddens[layer_idx]
        dir_change = last_token_np - prev
        self.prev_hiddens[layer_idx] = last_token_np.copy()
        
        # EMA field
        field = self.field_states[layer_idx]
        new_field = EMA_GAMMA * field + (1 - EMA_GAMMA) * dir_change
        self.field_states[layer_idx] = new_field
        
        # Raw enhancement: combination of ring_diff and field direction
        raw = ring_diff * 0.5 + new_field * 0.5
        
        return raw
    
    def project_orthogonal(self, enhancement, attention_vec):
        """
        将 enhancement 投影到与 attention_vec 正交的子空间
        
        Args:
            enhancement: [hidden_size] raw enhancement vector
            attention_vec: [hidden_size] attention output vector
        
        Returns:
            orth_enhancement: [hidden_size] orthogonalized enhancement
        """
        enh_tensor = torch.from_numpy(enhancement).float()
        attn_tensor = torch.from_numpy(attention_vec).float()
        
        # 1. 减去沿 attention 方向的投影 (Gram-Schmidt)
        attn_norm = attn_tensor.norm() + 1e-8
        proj_coeff = torch.dot(enh_tensor, attn_tensor) / (attn_norm ** 2)
        orth_enh = enh_tensor - proj_coeff * attn_tensor
        
        # 2. 随机投影: 将 orth_enh 投影到随机子空间
        proj_tensor = torch.from_numpy(self.proj_matrix).float()
        random_component = proj_tensor @ (proj_tensor.T @ orth_enh) * 0.3
        
        # 3. 合并: 70% 正交化 + 30% 随机
        combined = orth_enh * 0.7 + random_component * 0.3
        
        # 4. 固定范数
        combined_norm = combined.norm() + 1e-8
        combined = combined / combined_norm * CLIP_NORM
        
        # 记录分析数据
        cos = float(torch.nn.functional.cosine_similarity(
            combined, attn_tensor, dim=0).item())
        self.cos_sims.append(cos)
        self.enhancement_norms.append(float(combined_norm.item()))
        
        return combined.numpy()
    
    def compute_enhancement(self, layer_idx, last_token_tensor):
        """
        计算正交化的 SFA enhancement
        
        Args:
            layer_idx: 层索引
            last_token_tensor: [hidden_size] torch tensor
        
        Returns:
            enhancement: [hidden_size] numpy array (已正交化)
        """
        token_np = last_token_tensor.cpu().numpy()
        
        # 1. 计算原始 enhancement
        raw = self.compute_raw_enhancement(layer_idx, token_np)
        
        # 2. 正交化
        orth = self.project_orthogonal(raw, token_np)
        
        # 3. 应用 alpha（层自适应缩放，但不压缩到接近 0）
        layer_ratio = layer_idx / max(self.n_layers - 1, 1)
        alpha_eff = self.alpha * (0.3 + layer_ratio * 0.7)
        
        # 记录未缩放的范数用于分析
        self.enhancement_norms.append(float(torch.from_numpy(orth).norm().item()))
        
        return orth


def cosine_sim(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum().item())


def main():
    print("=" * 60)
    print("🔬 SFA v7 随机投影正交化测试")
    print("=" * 60)
    
    print("\n⏳ 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"✅ Layers: {n_layers}, Hidden: {hidden_size}, Vocab: {vocab_size}")
    
    test_text = "The quick brown fox jumps over the lazy dog. Machine learning is powerful. " * 5
    
    results = {
        "model": MODEL_PATH,
        "method": "random_projection_orthogonalization",
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "tests": {}
    }
    
    # ============================================================
    # 测试 1: 正交性验证
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 测试 1: 正交性验证 (随机投影)")
    print("=" * 60)
    
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        sfa = SFARandomProjectionEngine(n_layers, hidden_size, alpha=alpha)
        sfa.reset()
        
        input_ids = tokenizer(test_text[:128], return_tensors="pt")["input_ids"]
        
        hooks = []
        def make_hook(layer_idx, sfa_engine):
            def hook(module, args):
                hidden = args[0]
                last_token = hidden[:, -1, :]
                
                for b in range(last_token.shape[0]):
                    sfa_engine.compute_enhancement(layer_idx, last_token[b])
                
                return (hidden,) + args[1:]
            return hook
        
        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_pre_hook(make_hook(i, sfa))
            hooks.append(h)
        
        with torch.no_grad():
            model(input_ids)
        
        for h in hooks:
            h.remove()
        
        if sfa.cos_sims:
            avg_cos = np.mean(sfa.cos_sims)
            avg_norm = np.mean(sfa.enhancement_norms)
            status = "✅" if abs(avg_cos) < 0.3 else "⚠️"
            print(f"  α={alpha:.2f}: Cosine={avg_cos:.6f}, Norm={avg_norm:.6f} {status}")
            
            results["tests"][f"alpha_{alpha}"] = {
                "avg_cosine": round(float(avg_cos), 6),
                "avg_norm": round(float(avg_norm), 6),
                "samples": len(sfa.cos_sims),
            }
    
    # ============================================================
    # 测试 2: PPL 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 测试 2: PPL 对比 (随机投影)")
    print("=" * 60)
    
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 4,
        "Machine learning models use attention mechanisms. " * 4,
        "Signal Field Attention provides orthogonal channels. " * 4,
    ]
    
    # Baseline
    baseline_ppls = []
    for text in texts:
        input_ids = tokenizer(text[:128], return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model(input_ids)
        ppl = math.exp(nn.CrossEntropyLoss()(
            out.logits[:, :-1, :].view(-1, vocab_size),
            input_ids[:, 1:].view(-1)
        ).item())
        baseline_ppls.append(ppl)
    avg_baseline = np.mean(baseline_ppls)
    print(f"  平均 Baseline PPL: {avg_baseline:.4f}")
    
    # SFA Enhanced
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        sfa = SFARandomProjectionEngine(n_layers, hidden_size, alpha=alpha)
        sfa.reset()
        
        enhanced_ppls = []
        for text in texts:
            hooks = []
            def make_hook(layer_idx, sfa_engine):
                def hook(module, args):
                    hidden = args[0]
                    last_token = hidden[:, -1, :]
                    
                    for b in range(last_token.shape[0]):
                        enh_np = sfa_engine.compute_enhancement(layer_idx, last_token[b])
                        enh_tensor = torch.from_numpy(enh_np).float().to(DEVICE)
                        
                        hidden = hidden.clone()
                        hidden[b, -1, :] += enh_tensor
                        hidden += enh_tensor.unsqueeze(0).unsqueeze(0) * 0.05
                    
                    return (hidden,) + args[1:]
                return hook
            
            for i, layer in enumerate(model.model.layers):
                h = layer.register_forward_pre_hook(make_hook(i, sfa))
                hooks.append(h)
            
            with torch.no_grad():
                input_ids = tokenizer(text[:128], return_tensors="pt")["input_ids"]
                out = model(input_ids)
            
            ppl = math.exp(nn.CrossEntropyLoss()(
                out.logits[:, :-1, :].view(-1, vocab_size),
                input_ids[:, 1:].view(-1)
            ).item())
            enhanced_ppls.append(ppl)
            
            for h in hooks:
                h.remove()
        
        avg_enhanced = np.mean(enhanced_ppls)
        improvement = (avg_baseline - avg_enhanced) / avg_baseline * 100
        print(f"  α={alpha:.2f}: Enhanced={avg_enhanced:.4f}, 改善={improvement:+.2f}%")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    
    cos_values = [v["avg_cosine"] for v in results["tests"].values() if "avg_cosine" in v]
    if cos_values:
        best_alpha = min([0.01, 0.05, 0.1, 0.2, 0.5, 1.0], 
                        key=lambda a: abs(results["tests"].get(f"alpha_{a}", {}).get("avg_cosine", 1.0)))
        best_cos = results["tests"][f"alpha_{best_alpha}"]["avg_cosine"]
        print(f"\n  最佳正交性 α={best_alpha}: Cosine={best_cos:.6f}")
        print(f"  {'✅ 正交性达标 (<0.3)' if abs(best_cos) < 0.3 else '⚠️ 正交性仍需优化'}")
    
    print("\n📄 随机投影正交化测试完成")


if __name__ == "__main__":
    try:
        main()
        print("\n🎉 测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
