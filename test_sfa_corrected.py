#!/usr/bin/env python3
"""
============================================================
🔬 SFA v7 修正版 — 真正的正交增强
============================================================
问题诊断:
  1. 原实现中 ring_mean ≈ field（两者都是 attention 输出的平滑）
  2. 因此 enhancement 与 attention output 高度相关 (cosine=0.65)
  
修正方案:
  - Ring buffer 用 DIFFERENCE (当前 - ring_mean) 而非 raw mean
  - Field 用 DIRECTION 而非 magnitude
  - 降低 cross_decay 或使用 constant alpha
============================================================
"""

import os, sys, json, math, torch, torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")
DEVICE = "cpu"
RING_SIZE = 16
EMA_GAMMA = 0.98
CLIP = 0.01


class SFACorrectedEngine:
    """
    修正版 SFA 引擎
    
    关键改进:
    1. Ring buffer 存储的是 DIFFERENCE (current - ring_mean)，捕捉趋势变化
    2. Field 存储的是 DIRECTION (normalized difference)，捕捉方向
    3. Enhancement = ring_diff + field_direction，确保正交性
    """
    
    def __init__(self, n_layers, hidden_size, alpha=0.1):
        self.n_layers = n_layers
        self.hs = hidden_size
        self.alpha = alpha
        
        # Ring buffer: 存储最近的 hidden states
        self.ring_buffers = [np.zeros((RING_SIZE, hidden_size), dtype=np.float32) for _ in range(n_layers)]
        self.ring_offsets = [0] * n_layers
        
        # EMA field: 存储 attention 输出的方向变化
        self.field_dirs = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        self.prev_hidden = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        
        # 用于正交性分析
        self.enhancement_vectors = []
        self.attn_vectors = []
    
    def reset(self):
        for i in range(self.n_layers):
            self.ring_buffers[i].fill(0)
            self.field_dirs[i].fill(0)
            self.prev_hidden[i].fill(0)
        self.enhancement_vectors.clear()
        self.attn_vectors.clear()
    
    def compute_enhancement(self, layer_idx, last_token_tensor):
        """
        计算正交增强信号
        
        last_token_tensor: [hidden_size] torch tensor
        Returns: enhancement vector [hidden_size]
        """
        hs = self.hs
        idx = layer_idx
        
        token_np = last_token_tensor.cpu().numpy()
        
        # === Ring Difference ===
        ring = self.ring_buffers[idx]
        offset = self.ring_offsets[idx]
        ring[offset] = token_np
        self.ring_offsets[idx] = (offset + 1) % RING_SIZE
        valid = min(offset + 1, RING_SIZE) if offset < RING_SIZE else RING_SIZE
        
        ring_mean = ring[:valid].mean(axis=0)
        ring_diff = token_np - ring_mean  # 捕捉趋势变化
        
        # === Direction Change ===
        prev = self.prev_hidden[idx]
        dir_change = token_np - prev
        self.prev_hidden[idx] = token_np.copy()
        
        # Normalize direction
        dir_norm = np.linalg.norm(dir_change) + 1e-8
        dir_normalized = dir_change / dir_norm
        
        # EMA smooth the direction
        field_dir = self.field_dirs[idx]
        new_field = EMA_GAMMA * field_dir + (1 - EMA_GAMMA) * dir_normalized
        self.field_dirs[idx] = new_field
        
        # === Orthogonal Enhancement ===
        # Enhancement = weighted combination of ring_diff and direction
        # Both are DIFFERENCES, not raw values → orthogonal to absolute position
        enhancement = ring_diff * 0.5 + new_field * 0.5
        
        # Clip by norm, not by element-wise
        enh_norm = np.linalg.norm(enhancement) + 1e-8
        enh_scaled = enhancement / enh_norm * CLIP  # Fixed magnitude
        
        # Apply alpha
        enh_scaled *= self.alpha
        
        # Record for orthogonality analysis
        enh_tensor = torch.from_numpy(enh_scaled).float()
        self.enhancement_vectors.append(enh_tensor)
        self.attn_vectors.append(last_token_tensor)
        
        return enh_tensor


def cosine_sim(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum().item())


def main():
    print("=" * 60)
    print("🔬 SFA v7 修正版 — 正交增强测试")
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
    
    # ============================================================
    # 测试 1: 正交性验证
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 测试 1: 正交性验证 (修正版 SFA)")
    print("=" * 60)
    
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5]:
        sfa = SFACorrectedEngine(n_layers, hidden_size, alpha=alpha)
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
        
        if sfa.enhancement_vectors:
            cos_sims = [cosine_sim(e, a) for e, a in zip(sfa.enhancement_vectors, sfa.attn_vectors)]
            avg_cos = np.mean(cos_sims)
            print(f"  α={alpha:.2f}: Avg Cosine = {avg_cos:.6f} {'✅' if abs(avg_cos) < 0.3 else '⚠️'}")
    
    # ============================================================
    # 测试 2: PPL 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 测试 2: PPL 对比 (修正版 SFA)")
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
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5]:
        sfa = SFACorrectedEngine(n_layers, hidden_size, alpha=alpha)
        sfa.reset()
        
        enhanced_ppls = []
        for text in texts:
            hooks = []
            def make_hook(layer_idx, sfa_engine):
                def hook(module, args):
                    hidden = args[0]
                    last_token = hidden[:, -1, :]
                    
                    for b in range(last_token.shape[0]):
                        enh = sfa_engine.compute_enhancement(layer_idx, last_token[b])
                        hidden = hidden.clone()
                        hidden[b, -1, :] += enh
                        hidden += enh.unsqueeze(0).unsqueeze(0) * 0.05
                    
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
        print(f"  α={alpha:.2f}: Enhanced PPL={avg_enhanced:.4f}, 改善={improvement:+.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ 修正版测试完成")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
        print("\n🎉 完成！")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
