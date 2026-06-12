#!/usr/bin/env python3
"""
Signal Field 0.5B信号场蒸馏训练 + PPL验证
T Signal Field 0.5B Signal Field Distillation Training with PPL Validation

Model: Qwen2.5-0.5B-Instruct (fp16/bfloat16)
Framework: Pure MLX - ZERO PyTorch
Target: Distill signal field attention to match original attention output
Goal: PPL degradation < 10% after replacement

Strategy: Layer-level distillation
  1. Get hidden states before target layer (frozen)
  2. Teacher: original attention output
  3. Student: signal field output
  4. MSE loss → mx.grad → update sf_basis only
  5. After training, compute full model PPL

Iron rules:
  - Pure MLX, ZERO numpy/pytorch in computation path
  - log_softmax subtract max first
  - take_along_axis not mx.take
  - Signal field = plain Python class + __call__
  - File/variable names all English
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, List, Dict, Any

import mlx.core as mx
import mlx.nn as nn

# ============================================================================
# Model Config
# ============================================================================

MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

HIDDEN_SIZE = 896
NUM_ATTENTION_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
NUM_LAYERS = 24

DEFAULT_K = 16
DEFAULT_GAMMA = 0.98
DEFAULT_LR = 1e-3
DEFAULT_TRAIN_STEPS = 500
PRINT_FREQ = 50

CHINESE_TEXTS = [
    "今天天气真好，阳光明媚，适合出去散步。公园里有很多人在锻炼身体，有的在跑步，有的在打太极。",
    "人工智能技术正在快速发展，机器学习和深度学习已经应用在很多领域。从自然语言处理到计算机视觉，从医疗诊断到自动驾驶，AI正在改变我们的生活。",
    "春天的花朵盛开，空气中弥漫着花香的味道，蝴蝶在花丛中飞舞，蜜蜂忙碌地采蜜。这是大自然最美的季节。",
    "量子计算是未来科技的重要方向，它利用量子叠加和纠缠的原理，能够在某些问题上实现指数级的加速。",
    "中国的高铁技术世界领先，复兴号列车时速可达350公里，从北京到上海只需4个多小时，极大方便了人们的出行。",
]

ENGLISH_TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The transformer architecture has revolutionized natural language processing and computer vision.",
    "Deep neural networks with multiple layers can learn hierarchical representations of data.",
]


# ============================================================================
# Signal Field Attention (Pure Python Class + __call__)
# ============================================================================

class SignalFieldAttentionGQA:
    """
    Signal Field Attention - Ring Buffer Based Attention Replacement
    
    Core: Replace O(n^2) attention with O(k) signal field interaction
    - Ring buffer: keep last k positions as signal field bases
    - Decay: older positions weighted less by gamma^position
    - GQA: support grouped query attention
    - Trainable: sf_basis is the only trainable parameter
    
    MUST be plain Python class with __call__ for MLX model replacement.
    MLX nn.Module calls __call__ which calls forward.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        k: int = 16,
        gamma: float = 0.98,
    ):
        self.dims = dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k = k
        self.gamma = gamma
        self.n_rep = num_heads // num_kv_heads

        # Projection layers (borrowed from original attention, NOT trained)
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rope = None

        # Signal field basis (TRAINABLE) - [num_kv_heads, k, head_dim]
        scale = 0.02
        self.sf_basis = mx.random.normal(
            [num_kv_heads, k, head_dim], dtype=mx.float32
        ) * scale

        # Decay mask: [k], newest=1.0, oldest=gamma^(k-1)
        positions = mx.arange(k - 1, -1, -1, dtype=mx.float32)
        self._decay_mask = gamma ** positions

    def set_weights(self, attn_layer):
        """Copy weights from original attention layer (frozen, not trained)"""
        self.q_proj = attn_layer.q_proj
        self.k_proj = attn_layer.k_proj
        self.v_proj = attn_layer.v_proj
        self.o_proj = attn_layer.o_proj
        if hasattr(attn_layer, 'rope'):
            self.rope = attn_layer.rope

    def forward(self, x, mask=None, cache=None):
        """
        Signal field forward pass
        
        Args:
            x: Input [batch, seq_len, dims]
            mask: Causal mask (used for compatibility, signal field uses ring buffer)
            cache: KV cache (not used in training, reserved for inference)
        
        Returns:
            Output [batch, seq_len, dims]
        """
        B, L, _ = x.shape

        # Q/K/V projection (same as original attention)
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape: [B, L, H, D] -> [B, H, L, D]
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE (positional encoding from original model)
        if self.rope is not None:
            offset = 0
            if cache is not None and hasattr(cache, 'offset'):
                offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        # GQA expansion: repeat K/V heads to match Q heads
        if self.n_rep > 1:
            keys = mx.repeat(keys, self.n_rep, axis=1)
            values = mx.repeat(values, self.n_rep, axis=1)

        # =================================================================
        # Signal Field Core - THE INNOVATION
        # =================================================================

        # Ring buffer: take last k positions as signal field interaction points
        if L <= self.k:
            # Short sequence: use all positions
            sf_keys = keys
            sf_values = values
            effective_k = L
        else:
            # Long sequence: ring buffer truncation
            sf_keys = keys[:, :, -self.k:, :]
            sf_values = values[:, :, -self.k:, :]
            effective_k = self.k

        # Query-SignalField interaction: [B, H, L, D] x [B, H, D, k] -> [B, H, L, k]
        scale = 1.0 / (self.head_dim ** 0.5)
        sf_scores = (queries * scale) @ sf_keys.transpose(0, 1, 3, 2)

        # Apply temporal decay (older positions decay more)
        if L > self.k and effective_k == self.k:
            sf_scores = sf_scores * self._decay_mask.reshape(1, 1, 1, -1)

        # Softmax normalization
        sf_weights = mx.softmax(sf_scores, axis=-1)

        # Weighted aggregation: [B, H, L, k] x [B, H, k, D] -> [B, H, L, D]
        output = sf_weights @ sf_values

        # Add signal field basis interaction (trainable component)
        # sf_basis: [num_kv_heads, k, head_dim] -> expand for GQA
        if self.n_rep > 1:
            sf_basis_expanded = mx.repeat(self.sf_basis, self.n_rep, axis=0)
        else:
            sf_basis_expanded = self.sf_basis
        # sf_basis_expanded: [num_heads, k, head_dim]

        # Query-basis interaction for residual signal
        q_basis = (queries * scale) @ sf_basis_expanded.transpose(0, 2, 1)  # [B, H, L, k]
        q_basis_weights = mx.softmax(q_basis, axis=-1)
        basis_output = q_basis_weights @ sf_basis_expanded  # [B, H, L, D]

        # Combine: attention on ring buffer + signal field basis residual
        output = output + 0.1 * basis_output

        # Reshape back: [B, H, L, D] -> [B, L, dims]
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)

        return output

    def __call__(self, x, mask=None, cache=None):
        """Required for MLX model replacement - MLX calls __call__ not forward"""
        return self.forward(x, mask, cache)


# ============================================================================
# Model Loading (Pure MLX)
# ============================================================================

def load_model_and_tokenizer(model_path: str = MODEL_PATH):
    """Load Qwen model using mlx_lm (pure MLX, zero PyTorch)"""
    from mlx_lm import load as mlx_load
    
    print(f"Loading model: {model_path}")
    model, tokenizer = mlx_load(model_path)
    print(f"Model loaded successfully!")
    
    # Verify model structure
    print(f"  Layers: {len(model.model.layers)}")
    layer0 = model.model.layers[0].self_attn
    print(f"  Layer 0 attn type: {type(layer0).__name__}")
    print(f"  Has q_proj: {hasattr(layer0, 'q_proj')}")
    print(f"  Has rope: {hasattr(layer0, 'rope')}")
    
    return model, tokenizer


# ============================================================================
# PPL Computation (Iron Rules)
# ============================================================================

def compute_ppl(model, tokenizer, text, max_length=512):
    """
    Compute Perplexity with numerical stability
    
    IRON RULES:
    1. log_softmax must subtract max first (prevent NaN)
    2. Use take_along_axis (not mx.take)
    """
    # Tokenize
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float('inf')
    
    tokens = tokens[:max_length]
    input_ids = mx.array([tokens])
    
    # Forward
    logits = model(input_ids)
    
    # Shift: predict next token
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # IRON RULE: log_softmax subtract max first
    max_logits = mx.max(shift_logits, axis=-1, keepdims=True)
    shifted = shift_logits - max_logits
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    
    # IRON RULE: use take_along_axis not mx.take
    expanded_labels = shift_labels[:, :, None]
    token_log_probs = mx.take_along_axis(log_probs, expanded_labels, axis=-1).squeeze(-1)
    
    # Mask padding (token_id=0)
    mask = shift_labels != 0
    n_tokens = mx.sum(mask)
    
    if float(n_tokens) == 0:
        return float('inf')
    
    nll = -mx.sum(token_log_probs * mask) / n_tokens
    ppl = float(mx.exp(nll))
    
    return ppl


# ============================================================================
# Hidden State Extraction
# ============================================================================

def get_hidden_states_before_layer(model, input_ids, layer_idx):
    """
    Get hidden states before target layer (frozen, no gradient needed)
    
    This allows us to do layer-level distillation without backprop through
    the entire model.
    """
    h = model.model.embed_tokens(input_ids)
    
    # Run through layers before target
    for i in range(layer_idx):
        h = model.model.layers[i](h)
    
    return h


def get_attention_output(attn_layer, hidden_states, mask=None):
    """
    Get attention layer output given hidden states
    
    For Qwen2.5: DecoderLayer calls self_attn(input_layernorm(x), mask, cache)
    So we need to apply input_layernorm first.
    """
    return attn_layer(hidden_states, mask=mask)


# ============================================================================
# Layer-Level Distillation Training
# ============================================================================

def distill_layer(
    model,
    tokenizer,
    layer_idx: int = 0,
    k: int = DEFAULT_K,
    gamma: float = DEFAULT_GAMMA,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    lr: float = DEFAULT_LR,
    print_freq: int = PRINT_FREQ,
):
    """
    Distill signal field for a single layer
    
    Strategy:
    1. Get hidden states before layer (frozen)
    2. Teacher: original attention output on those hidden states
    3. Student: signal field output on those hidden states
    4. MSE loss → mx.grad → update sf_basis
    5. After training, replace layer and compute full model PPL
    """
    print(f"\n{'='*60}")
    print(f"Signal Field 0.5B信号场蒸馏训练")
    print(f"{'='*60}")
    print(f"Layer: {layer_idx}")
    print(f"Config: k={k}, gamma={gamma}, lr={lr}, steps={train_steps}")
    print(f"MLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    
    results = {
        'layer_idx': layer_idx,
        'k': k,
        'gamma': gamma,
        'lr': lr,
        'train_steps': train_steps,
    }
    
    # Prepare training data
    all_texts = CHINESE_TEXTS + ENGLISH_TEXTS
    train_tokens = []
    for text in all_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 10:
            train_tokens.append(toks[:256])
    
    test_text = CHINESE_TEXTS[0]
    
    # Step 1: Baseline PPL
    print(f"\n[1/4] Computing Baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, test_text)
    results['baseline_ppl'] = baseline_ppl
    print(f"   Baseline PPL: {baseline_ppl:.4f}")
    
    # Step 2: Setup signal field
    print(f"\n[2/4] Setting up SignalField for layer {layer_idx}...")
    
    original_attn = model.model.layers[layer_idx].self_attn
    target_layer = model.model.layers[layer_idx]
    
    sf_attn = SignalFieldAttentionGQA(
        dims=HIDDEN_SIZE,
        num_heads=NUM_ATTENTION_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        k=k,
        gamma=gamma,
    )
    sf_attn.set_weights(original_attn)
    print(f"   sf_basis shape: {sf_attn.sf_basis.shape}")
    print(f"   sf_basis dtype: {sf_attn.sf_basis.dtype}")
    
    # Step 3: Training
    print(f"\n[3/4] Training (layer-level distillation)...")
    print("-" * 60)
    
    steps_log = []
    step_times = []
    
    # Pre-compute teacher outputs and hidden states for training
    print("   Preparing training data (teacher outputs)...")
    train_data = []
    for tokens in train_tokens:
        input_ids = mx.array([tokens])
        
        # Get hidden states before target layer
        with mx.no_grad():
            h = model.model.embed_tokens(input_ids)
            for i in range(layer_idx):
                h = model.model.layers[i](h)
            
            # Apply input_layernorm (Qwen2.5 does this before attention)
            normed_h = target_layer.input_layernorm(h)
            
            # Teacher output
            teacher_out = original_attn(normed_h)
        
        train_data.append({
            'hidden_states': h,
            'normed_hidden': normed_h,
            'teacher_output': teacher_out,
        })
    print(f"   Training sequences: {len(train_data)}")
    
    # Define loss function for gradient computation
    def loss_fn(sf_basis):
        sf_attn.sf_basis = sf_basis
        
        total_loss = mx.array(0.0)
        for data in train_data:
            # Student forward
            student_out = sf_attn(data['normed_hidden'])
            # MSE loss
            diff = student_out - data['teacher_output']
            total_loss = total_loss + mx.mean(diff ** 2)
        
        return total_loss / len(train_data)
    
    # Training loop
    for step in range(1, train_steps + 1):
        step_start = time.time()
        
        # Compute gradient
        loss_val, grad = mx.value_and_grad(loss_fn)(sf_attn.sf_basis)
        loss_val = float(loss_val)
        
        # Update sf_basis (SGD)
        sf_attn.sf_basis = sf_attn.sf_basis - lr * grad
        
        # Evaluate
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        if step % print_freq == 0 or step == train_steps:
            # Replace and compute PPL
            model.model.layers[layer_idx].self_attn = sf_attn
            current_ppl = compute_ppl(model, tokenizer, test_text)
            degr = ((current_ppl - baseline_ppl) / baseline_ppl) * 100
            
            # Restore original
            model.model.layers[layer_idx].self_attn = original_attn
            
            if abs(degr) < 5:
                status = "✅"
            elif abs(degr) < 10:
                status = "⚠️"
            else:
                status = "❌"
            
            steps_log.append({
                'step': step,
                'loss': loss_val,
                'ppl': current_ppl,
                'degradation': degr,
                'step_time': step_time,
            })
            
            print(
                f"Step {step:4d}/{train_steps} | "
                f"loss={loss_val:.6f} | "
                f"PPL={current_ppl:.2f} ({degr:+.1f}%) | "
                f"{step_time:.2f}s | {status}"
            )
    
    results['steps'] = steps_log
    results['avg_step_time'] = sum(step_times) / len(step_times) if step_times else 0
    
    # Step 4: Final evaluation with trained signal field
    print(f"\n[4/4] Final Evaluation...")
    
    # Replace with trained signal field
    model.model.layers[layer_idx].self_attn = sf_attn
    final_ppl = compute_ppl(model, tokenizer, test_text)
    degr = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
    
    # Test on all texts
    all_ppls = {}
    for i, text in enumerate(CHINESE_TEXTS + ENGLISH_TEXTS):
        ppl = compute_ppl(model, tokenizer, text)
        label = f"text_{i}"
        all_ppls[label] = ppl
    
    results['final_ppl'] = final_ppl
    results['degradation'] = degr
    results['all_ppls'] = all_ppls
    
    print(f"\n{'='*60}")
    print(f"训练完成")
    print(f"{'='*60}")
    print(f"Baseline:    {baseline_ppl:.4f}")
    print(f"SignalField: {final_ppl:.4f}")
    print(f"Degradation: {degr:+.2f}%")
    
    if degr < 0:
        print(f"🎉 PPL提升 {-degr:.2f}%!")
    elif degr < 5:
        print(f"✅ 退化 < 5%，优秀！")
    elif degr < 10:
        print(f"⚠️ 退化 < 10%，可用")
    else:
        print(f"❌ 退化 > 10%，需要继续训练")
    print(f"{'='*60}")
    
    # Restore original attention
    model.model.layers[layer_idx].self_attn = original_attn
    
    return results


# ============================================================================
# Multi-Layer Distillation
# ============================================================================

def distill_multi_layer(
    model,
    tokenizer,
    layer_indices: List[int],
    k: int = DEFAULT_K,
    gamma: float = DEFAULT_GAMMA,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    lr: float = DEFAULT_LR,
    print_freq: int = PRINT_FREQ,
):
    """
    Distill signal field for multiple layers (sequential)
    
    Strategy: Train each layer independently, then evaluate all together
    """
    print(f"\n{'='*60}")
    print(f"Signal Field 0.5B多层信号场蒸馏")
    print(f"{'='*60}")
    print(f"Layers: {layer_indices}")
    
    results = {
        'layer_indices': layer_indices,
        'k': k,
        'gamma': gamma,
        'lr': lr,
        'train_steps': train_steps,
        'per_layer': {},
    }
    
    all_texts = CHINESE_TEXTS + ENGLISH_TEXTS
    test_text = CHINESE_TEXTS[0]
    
    # Baseline
    print(f"\nBaseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, test_text)
    results['baseline_ppl'] = baseline_ppl
    print(f"   Baseline: {baseline_ppl:.4f}")
    
    # Train each layer
    trained_sf_attns = {}
    for layer_idx in layer_indices:
        print(f"\n--- Training Layer {layer_idx} ---")
        layer_result = distill_layer(
            model, tokenizer,
            layer_idx=layer_idx,
            k=k, gamma=gamma,
            train_steps=train_steps,
            lr=lr,
            print_freq=print_freq,
        )
        results['per_layer'][layer_idx] = layer_result
        
        # Save trained sf_attn for final eval
        trained_sf_attns[layer_idx] = {
            'sf_basis': layer_result.get('sf_basis', None),
        }
    
    # Final evaluation with ALL layers replaced
    print(f"\n{'='*60}")
    print(f"Final: ALL layers replaced evaluation")
    print(f"{'='*60}")
    
    # Replace all layers
    original_attns = {}
    for layer_idx in layer_indices:
        original_attns[layer_idx] = model.model.layers[layer_idx].self_attn
        
        sf_attn = SignalFieldAttentionGQA(
            dims=HIDDEN_SIZE,
            num_heads=NUM_ATTENTION_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            k=k,
            gamma=gamma,
        )
        sf_attn.set_weights(original_attns[layer_idx])
        
        # Load trained basis if available
        per_layer = results['per_layer'].get(layer_idx, {})
        if 'sf_basis' in per_layer and per_layer['sf_basis'] is not None:
            sf_attn.sf_basis = mx.array(per_layer['sf_basis'])
        
        model.model.layers[layer_idx].self_attn = sf_attn
    
    final_ppl = compute_ppl(model, tokenizer, test_text)
    degr = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
    
    results['final_ppl'] = final_ppl
    results['degradation'] = degr
    
    print(f"Baseline: {baseline_ppl:.4f}")
    print(f"SignalField ({len(layer_indices)} layers): {final_ppl:.4f}")
    print(f"Degradation: {degr:+.2f}%")
    
    # Restore all
    for layer_idx in layer_indices:
        model.model.layers[layer_idx].self_attn = original_attns[layer_idx]
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Signal Field 0.5B信号场蒸馏训练")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--layer', type=int, default=0, help="Single layer index")
    parser.add_argument('--layers', type=str, default=None, help="Multi-layer: 0,1,2")
    parser.add_argument('--k', type=int, default=DEFAULT_K, help="Ring buffer size")
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help="Decay factor")
    parser.add_argument('--steps', type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--print_freq', type=int, default=PRINT_FREQ)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        ts = time.strftime('%Y%m%d_%H%M%S')
        if args.layers:
            lstr = args.layers.replace(',', '-')
            output_path = os.path.join(output_dir, f'distill_{lstr}_{ts}.json')
        else:
            output_path = os.path.join(output_dir, f'distill_layer{args.layer}_{ts}.json')
    
    # Load model
    print(f"MLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Train
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(',')]
        results = distill_multi_layer(
            model, tokenizer, layer_indices,
            k=args.k, gamma=args.gamma,
            train_steps=args.steps, lr=args.lr,
            print_freq=args.print_freq,
        )
    else:
        results = distill_layer(
            model, tokenizer, args.layer,
            k=args.k, gamma=args.gamma,
            train_steps=args.steps, lr=args.lr,
            print_freq=args.print_freq,
        )
    
    # Save results (convert mx arrays to lists for JSON)
    def convert_for_json(obj):
        if isinstance(obj, mx.array):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    json_results = convert_for_json(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
