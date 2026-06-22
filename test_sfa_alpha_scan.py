#!/usr/bin/env python3
"""
============================================================
🔬 SFA α 超参数扫描测试
============================================================
目的: 找到最佳的 SFA enhancement 系数 α，
使得 PPL 改善最大且正交性最优。

测试 α 值: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
============================================================
"""

import os, sys, json, math, torch, torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")
ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
DEVICE = "cpu"
SEQ_LEN = 128  # 固定序列长度
RING_SIZE = 16
EMA_GAMMA = 0.98
CROSS_DECAY = 0.7
CLIP = 0.01


class SFAEngine:
    def __init__(self, n_layers, hidden_size, alpha):
        self.n_layers = n_layers
        self.hs = hidden_size
        self.alpha = alpha
        self.ring_buffers = [np.zeros((RING_SIZE, hidden_size), dtype=np.float32) for _ in range(n_layers)]
        self.field_states = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        self.ring_offsets = [0] * n_layers
    
    def reset(self):
        for i in range(self.n_layers):
            self.ring_buffers[i].fill(0)
            self.field_states[i].fill(0)
            self.ring_offsets[i] = 0
    
    def compute_enhancement(self, layer_idx, last_token_np):
        ring = self.ring_buffers[layer_idx]
        offset = self.ring_offsets[layer_idx]
        ring[offset] = last_token_np
        self.ring_offsets[layer_idx] = (offset + 1) % RING_SIZE
        valid = min(offset + 1, RING_SIZE) if offset < RING_SIZE else RING_SIZE
        
        ring_mean = ring[:valid].mean(axis=0)
        field = self.field_states[layer_idx]
        new_field = EMA_GAMMA * field + (1 - EMA_GAMMA) * last_token_np
        self.field_states[layer_idx] = new_field
        
        enhancement = ring_mean + new_field
        enhancement = np.clip(enhancement, -CLIP, CLIP)
        
        layer_ratio = layer_idx / max(self.n_layers - 1, 1)
        alpha_eff = self.alpha * (0.3 + layer_ratio * 0.7) * (CROSS_DECAY ** layer_idx)
        enhancement *= alpha_eff
        
        return enhancement


def main():
    print("=" * 60)
    print("🔬 SFA α 超参数扫描")
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
    
    # 测试文本
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 4,
        "Machine learning models use attention mechanisms to process sequences. " * 4,
        "Signal Field Attention provides an orthogonal channel to standard attention. " * 4,
    ]
    
    results = {"model": MODEL_PATH, "n_layers": n_layers, "hidden_size": hidden_size, "alphas": {}}
    
    # Baseline PPL (平均)
    print("\n📊 计算 Baseline PPL...")
    baseline_ppls = []
    for text in texts:
        input_ids = tokenizer(text[:SEQ_LEN], return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model(input_ids)
        ppl = math.exp(nn.CrossEntropyLoss()(
            out.logits[:, :-1, :].view(-1, vocab_size),
            input_ids[:, 1:].view(-1)
        ).item())
        baseline_ppls.append(ppl)
    avg_baseline = np.mean(baseline_ppls)
    print(f"  平均 Baseline PPL: {avg_baseline:.4f}")
    
    # Test each α
    print("\n" + "=" * 60)
    print("🔬 α 扫描结果")
    print("=" * 60)
    
    for alpha in ALPHAS:
        print(f"\n  α = {alpha}")
        
        enhanced_ppls = []
        ortho_cosines = []
        
        for text_idx, text in enumerate(texts):
            sfa = SFAEngine(n_layers, hidden_size, alpha=alpha)
            sfa.reset()
            
            # Register pre_hooks
            hooks = []
            def make_hook(layer_idx, sfa_engine, collect_cos=False):
                def hook(module, args):
                    hidden = args[0]
                    last_token = hidden[:, -1, :]
                    
                    for b in range(last_token.shape[0]):
                        enh_np = sfa_engine.compute_enhancement(layer_idx, last_token[b].cpu().numpy())
                        enh_tensor = torch.from_numpy(enh_np).float().to(DEVICE)
                        
                        if collect_cos:
                            cos = float(torch.nn.functional.cosine_similarity(
                                enh_tensor, last_token[b], dim=0).item())
                            ortho_cosines.append(cos)
                        
                        hidden = hidden.clone()
                        hidden[b, -1, :] += enh_tensor
                        hidden += enh_tensor.unsqueeze(0).unsqueeze(0) * 0.05
                    
                    return (hidden,) + args[1:]
                return hook
            
            for i, layer in enumerate(model.model.layers):
                h = layer.register_forward_pre_hook(make_hook(i, sfa, collect_cos=(text_idx == 0)))
                hooks.append(h)
            
            with torch.no_grad():
                input_ids = tokenizer(text[:SEQ_LEN], return_tensors="pt")["input_ids"]
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
        
        print(f"    Avg Enhanced PPL: {avg_enhanced:.4f}")
        print(f"    PPL 改善: {improvement:+.2f}%")
        
        if ortho_cosines:
            avg_ortho = np.mean(ortho_cosines)
            print(f"    Avg Cosine(enhancement, attn): {avg_ortho:.6f}")
        
        results["alphas"][str(alpha)] = {
            "avg_enhanced_ppl": round(avg_enhanced, 4),
            "improvement_pct": round(improvement, 2),
            "baseline_ppl": round(avg_baseline, 4),
            "per_text_ppl": [round(p, 4) for p in enhanced_ppls],
        }
        if ortho_cosines:
            results["alphas"][str(alpha)]["avg_orthogonality_cosine"] = round(float(avg_ortho), 6)
    
    # 找出最佳 α
    best_alpha = min(ALPHAS, key=lambda a: results["alphas"][str(a)]["improvement_pct"])
    print(f"\n{'='*60}")
    print(f"🏆 最佳 α = {best_alpha} (PPL 改善: {results['alphas'][str(best_alpha)]['improvement_pct']:+.2f}%)")
    print(f"{'='*60}")
    
    # 保存
    with open("test_sfa_alpha_scan.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 结果: test_sfa_alpha_scan.json")


if __name__ == "__main__":
    try:
        main()
        print("\n🎉 扫描完成！")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
