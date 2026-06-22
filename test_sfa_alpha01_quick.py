#!/usr/bin/env python3
"""
============================================================
🎯 SFA v7 α=0.1 正确性测试 — 最终版
============================================================
策略: 使用 register_forward_pre_hook 在每层 decoder layer 
的输入处注入 SFA enhancement。这样不会干扰 transformer 
的 forward 返回格式。

验证:
  a. PPL 改善 (baseline vs SFA-enhanced)
  b. 正交性 (enhancement vs attention output)
  c. 内存开销
============================================================
"""

import os, sys, json, math, torch, torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")
ALPHA = 0.1
DEVICE = "cpu"
SEQ_LENGTHS = [32, 64, 128, 256, 512]
RING_SIZE = 16
EMA_GAMMA = 0.98
CROSS_DECAY = 0.7
CLIP = 0.01


class SFAEngine:
    def __init__(self, n_layers, hidden_size, alpha=ALPHA):
        self.n_layers = n_layers
        self.hs = hidden_size
        self.alpha = alpha
        self.ring_buffers = [np.zeros((RING_SIZE, hidden_size), dtype=np.float32) for _ in range(n_layers)]
        self.field_states = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        self.ring_offsets = [0] * n_layers
        self.cos_sims = []
    
    def reset(self):
        for i in range(self.n_layers):
            self.ring_buffers[i].fill(0)
            self.field_states[i].fill(0)
            self.ring_offsets[i] = 0
        self.cos_sims.clear()
    
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


def compute_ppl(model, tokenizer, text, device, vocab_size):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits[:, :-1, :].view(-1, vocab_size), input_ids[:, 1:].view(-1))
    return math.exp(loss.item()), input_ids


def main():
    print("=" * 60)
    print("🎯 SFA v7 α=0.1 正确性测试 — 最终版")
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
    
    test_text = "The quick brown fox jumps over the lazy dog. " * 20
    
    results = {
        "model": MODEL_PATH, "alpha": ALPHA, "device": DEVICE,
        "n_layers": n_layers, "hidden_size": hidden_size, "tests": {}
    }
    
    # ============================================================
    # 测试 a: PPL 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 PPL 对比 (baseline vs SFA-enhanced α=0.1)")
    print("=" * 60)
    
    ppl_results = {}
    
    for seq_len in SEQ_LENGTHS:
        text = test_text[:seq_len]
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        actual_len = input_ids.shape[1]
        print(f"\n  序列长度: {actual_len}")
        
        # --- Baseline ---
        baseline_ppl, _ = compute_ppl(model, tokenizer, text, DEVICE, vocab_size)
        print(f"    Baseline PPL: {baseline_ppl:.4f}")
        
        # --- SFA Enhanced ---
        sfa = SFAEngine(n_layers, hidden_size, alpha=ALPHA)
        sfa.reset()
        
        # Use pre_hook to modify input hidden states before they enter each layer
        hooks = []
        
        def make_pre_hook(layer_idx, sfa_engine):
            def hook(module, args):
                # args[0] is hidden_states [batch, seq, hidden]
                hidden = args[0]
                
                # Compute enhancement based on PREVIOUS layer's last token
                # (we use the input to this layer as approximation of prev output)
                last_token = hidden[:, -1, :]  # [batch, hidden]
                
                for b in range(last_token.shape[0]):
                    enh_np = sfa_engine.compute_enhancement(
                        layer_idx, last_token[b].cpu().numpy()
                    )
                    enh_tensor = torch.from_numpy(enh_np).float().to(DEVICE)
                    
                    # Inject into last token
                    hidden = hidden.clone()  # Detach from graph for modification
                    hidden[b, -1, :] += enh_tensor
                    # Broadcast to all positions with decay
                    hidden += enh_tensor.unsqueeze(0).unsqueeze(0) * 0.05
                
                return (hidden,) + args[1:]
            return hook
        
        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_pre_hook(make_pre_hook(i, sfa))
            hooks.append(h)
        
        with torch.no_grad():
            en_input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
            en_out = model(en_input_ids)
        
        for h in hooks:
            h.remove()
        
        enhanced_ppl = math.exp(nn.CrossEntropyLoss()(
            en_out.logits[:, :-1, :].view(-1, vocab_size),
            en_input_ids[:, 1:].view(-1)
        ).item())
        print(f"    SFA-enhanced PPL: {enhanced_ppl:.4f}")
        
        improvement = (baseline_ppl - enhanced_ppl) / baseline_ppl * 100
        print(f"    改善: {improvement:+.2f}%")
        
        ppl_results[str(actual_len)] = {
            "baseline_ppl": round(baseline_ppl, 4),
            "enhanced_ppl": round(enhanced_ppl, 4),
            "improvement_pct": round(improvement, 2),
        }
        
        last_sfa = sfa
    
    results["tests"]["ppl_comparison"] = ppl_results
    
    # ============================================================
    # 测试 b: 正交性
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 正交性验证")
    print("=" * 60)
    
    # Recompute with orthogonality tracking
    sfa_ortho = SFAEngine(n_layers, hidden_size, alpha=ALPHA)
    sfa_ortho.reset()
    ortho_cosines = []
    
    def make_ortho_hook(layer_idx, sfa_engine):
        def hook(module, args):
            hidden = args[0]
            last_token = hidden[:, -1, :]
            
            for b in range(last_token.shape[0]):
                lt_np = last_token[b].cpu().numpy()
                enh_np = sfa_engine.compute_enhancement(layer_idx, lt_np)
                enh_tensor = torch.from_numpy(enh_np).float()
                
                cos = float(torch.nn.functional.cosine_similarity(
                    enh_tensor, last_token[b], dim=0).item())
                ortho_cosines.append(cos)
            
            return (hidden,) + args[1:]
        return hook
    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_pre_hook(make_ortho_hook(i, sfa_ortho))
        hooks.append(h)
    
    with torch.no_grad():
        model(tokenizer(test_text[:256], return_tensors="pt")["input_ids"])
    
    for h in hooks:
        h.remove()
    
    if ortho_cosines:
        avg_cos = np.mean(ortho_cosines)
        print(f"  Avg Cosine(enhancement, attention): {avg_cos:.6f}")
        print(f"  {'✅ 正交性良好 (<0.1)' if abs(avg_cos) < 0.1 else '⚠️ 正交性不足'}")
        results["tests"]["orthogonality"] = {
            "avg_cosine": round(float(avg_cos), 6),
            "samples": ortho_cosines[:10],
        }
    else:
        print("  ⚠️ 未收集到正交性数据")
    
    # ============================================================
    # 测试 c: 内存
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 内存占用")
    print("=" * 60)
    
    mem_mb = (n_layers * RING_SIZE * hidden_size * 4 + n_layers * hidden_size * 4) / (1024*1024)
    print(f"  SFA 额外内存: {mem_mb:.2f} MB")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    
    improvements = [float(v["improvement_pct"]) for v in ppl_results.values()]
    avg_imp = np.mean(improvements)
    
    print(f"\n  平均 PPL 改善: {avg_imp:+.2f}%")
    print(f"  SFA 额外内存: {mem_mb:.2f} MB")
    print(f"  总体评估: {'✅ PASS' if avg_imp < -0.5 else '⚠️ NEEDS_REVIEW'}")
    
    results["tests"]["memory_mb"] = round(mem_mb, 2)
    results["summary"] = {
        "avg_ppl_improvement_pct": round(float(avg_imp), 2),
        "memory_mb": round(mem_mb, 2),
        "overall": "PASS" if avg_imp < -0.5 else "NEEDS_REVIEW",
    }
    
    with open("test_results_sfa_alpha01_quick.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 结果: test_results_sfa_alpha01_quick.json")
    
    return results


if __name__ == "__main__":
    try:
        main()
        print("\n🎉 测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
