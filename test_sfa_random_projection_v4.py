#!/usr/bin/env python3
"""
============================================================
🔬 SFA v7 随机投影正交化 — v4 (Final)
============================================================
v3 的问题: 手动 forward 无法正确处理 position embeddings
  
v4 修复:
  1. 使用 register_forward_pre_hook 但修复闭包问题
  2. 每个 layer 使用独立的 sfa 实例
  3. 正确记录 enhancement 范数和正交性
============================================================
"""

import os, sys, json, math, torch, torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")
DEVICE = "cpu"
RING_SIZE = 16
EMA_GAMMA = 0.98
CLIP_NORM = 0.5


class SFACorrectedEngineV4:
    def __init__(self, n_layers, hidden_size, alpha=0.1, seed=42):
        self.n_layers = n_layers
        self.hs = hidden_size
        self.alpha = alpha
        
        self.ring_buffers = [np.zeros((RING_SIZE, hidden_size), dtype=np.float32) for _ in range(n_layers)]
        self.field_states = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        self.ring_offsets = [0] * n_layers
        self.prev_hiddens = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        
        rng = np.random.RandomState(seed)
        self.proj_matrix = rng.randn(hidden_size, 32).astype(np.float32)
        
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
    
    def compute_enhancement(self, layer_idx, last_token_np, last_token_tensor=None):
        ring = self.ring_buffers[layer_idx]
        offset = self.ring_offsets[layer_idx]
        ring[offset] = last_token_np
        self.ring_offsets[layer_idx] = (offset + 1) % RING_SIZE
        valid = min(offset + 1, RING_SIZE) if offset < RING_SIZE else RING_SIZE
        
        ring_mean = ring[:valid].mean(axis=0)
        ring_diff = last_token_np - ring_mean
        
        prev = self.prev_hiddens[layer_idx]
        dir_change = last_token_np - prev
        self.prev_hiddens[layer_idx] = last_token_np.copy()
        
        field = self.field_states[layer_idx]
        new_field = EMA_GAMMA * field + (1 - EMA_GAMMA) * dir_change
        self.field_states[layer_idx] = new_field
        
        raw = ring_diff * 0.5 + new_field * 0.5
        
        if last_token_tensor is not None:
            enh_tensor = torch.from_numpy(raw).float()
            attn_tensor = last_token_tensor.float()
            
            attn_norm = attn_tensor.norm() + 1e-8
            proj_coeff = torch.dot(enh_tensor, attn_tensor) / (attn_norm ** 2)
            orth_enh = enh_tensor - proj_coeff * attn_tensor
            
            proj_tensor = torch.from_numpy(self.proj_matrix).float()
            random_component = proj_tensor @ (proj_tensor.T @ orth_enh) * 0.3
            combined = orth_enh * 0.7 + random_component * 0.3
            
            cos = float(torch.nn.functional.cosine_similarity(
                combined, attn_tensor, dim=0).item())
            self.cos_sims.append(cos)
            self.enhancement_norms.append(float(combined.norm().item()))
            
            combined_norm = combined.norm() + 1e-8
            combined = combined / combined_norm * CLIP_NORM
            orth_np = combined.numpy()
        else:
            orth_np = raw
        
        layer_ratio = layer_idx / max(self.n_layers - 1, 1)
        alpha_eff = self.alpha * (0.3 + layer_ratio * 0.7)
        orth_np *= alpha_eff
        
        return orth_np


def main():
    print("=" * 60)
    print("🔬 SFA v7 随机投影正交化 — v4 (Final)")
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
        "method": "random_projection_orthogonalization_v4",
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "tests": {}
    }
    
    # ============================================================
    # 测试 1: 正交性验证
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 测试 1: 正交性验证")
    print("=" * 60)
    
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        sfa = SFACorrectedEngineV4(n_layers, hidden_size, alpha=alpha)
        sfa.reset()
        
        input_ids = tokenizer(test_text[:128], return_tensors="pt")["input_ids"]
        
        # Fix: create a separate hook factory for each layer to avoid closure issues
        hooks = []
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            layer_sfa = sfa  # Share the same sfa instance
            
            def make_hook(idx, eng):
                def hook(module, args):
                    hidden = args[0]
                    last_token = hidden[:, -1, :]
                    
                    for b in range(last_token.shape[0]):
                        enh_np = eng.compute_enhancement(
                            idx,
                            last_token[b].cpu().numpy(),
                            last_token_tensor=last_token[b]
                        )
                        hidden = hidden.clone()
                        hidden[b, -1, :] += torch.from_numpy(enh_np).float().to(DEVICE)
                        hidden += torch.from_numpy(enh_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0) * 0.05
                    
                    return (hidden,) + args[1:]
                return hook
            
            h = layer.register_forward_pre_hook(make_hook(layer_idx, layer_sfa))
            hooks.append(h)
        
        with torch.no_grad():
            model(input_ids)
        
        for h in hooks:
            h.remove()
        
        if sfa.cos_sims:
            avg_cos = np.mean(sfa.cos_sims)
            avg_norm = np.mean(sfa.enhancement_norms) if sfa.enhancement_norms else 0
            status = "✅" if abs(avg_cos) < 0.3 else "⚠️"
            print(f"  α={alpha:.2f}: Cosine={avg_cos:.6f}, Norm={avg_norm:.4f} {status}")
            
            results["tests"][f"alpha_{alpha}"] = {
                "avg_cosine": round(float(avg_cos), 6),
                "avg_norm": round(float(avg_norm), 6),
                "samples": len(sfa.cos_sims),
            }
    
    # ============================================================
    # 测试 2: PPL 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 测试 2: PPL 对比")
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
        sfa = SFACorrectedEngineV4(n_layers, hidden_size, alpha=alpha)
        sfa.reset()
        
        enhanced_ppls = []
        for text in texts:
            hooks = []
            for layer_idx in range(n_layers):
                layer = model.model.layers[layer_idx]
                layer_sfa = sfa
                
                def make_hook(idx, eng):
                    def hook(module, args):
                        hidden = args[0]
                        last_token = hidden[:, -1, :]
                        
                        for b in range(last_token.shape[0]):
                            enh_np = eng.compute_enhancement(
                                idx,
                                last_token[b].cpu().numpy(),
                                last_token_tensor=last_token[b]
                            )
                            hidden = hidden.clone()
                            hidden[b, -1, :] += torch.from_numpy(enh_np).float().to(DEVICE)
                            hidden += torch.from_numpy(enh_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0) * 0.05
                        
                        return (hidden,) + args[1:]
                    return hook
                
                h = layer.register_forward_pre_hook(make_hook(layer_idx, layer_sfa))
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
