#!/usr/bin/env python3
"""
Signal Field 0.5B信号场蒸馏训练 v2
T Signal Field 0.5B Signal Field Distillation Training v2

Key changes from v1:
1. Learned compression (not ring buffer) - compress ALL K/V into k vectors
2. Full model NLL loss (not layer MSE) - directly optimizes PPL
3. Trainable: compress_queries + decay, frozen: model weights

Model: Qwen2.5-0.5B-Instruct (fp16/bfloat16)
Framework: Pure MLX
Goal: PPL degradation < 10% after signal field replacement

Iron rules:
  - Pure MLX, ZERO numpy/pytorch
  - log_softmax subtract max first
  - take_along_axis not mx.take
  - Signal field = plain Python class + __call__
  - MLX has NO mx.no_grad() - gradients only via mx.grad()
"""

import os
import sys
import json
import time
import argparse

import mlx.core as mx
import mlx.nn as nn

# ============================================================================
# Config
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
DEFAULT_TRAIN_STEPS = 300
PRINT_FREQ = 25
MAX_SEQ_LEN = 64

CHINESE_TEXTS = [
    "今天天气真好，阳光明媚，适合出去散步。公园里有很多人在锻炼身体。",
    "人工智能技术正在快速发展，机器学习已经应用在很多领域。",
    "春天的花朵盛开，空气中弥漫着花香的味道，蝴蝶在花丛中飞舞。",
    "量子计算是未来科技的重要方向，利用量子叠加实现指数级加速。",
    "中国的高铁技术世界领先，复兴号列车时速可达350公里。",
]

ENGLISH_TEXTS = [
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "The transformer architecture has revolutionized natural language processing.",
    "Deep neural networks learn hierarchical representations of data.",
]


# ============================================================================
# Signal Field Attention v2 - Learned Compression
# ============================================================================

class SignalFieldAttentionGQA:
    """
    Signal Field Attention v2: Learned Compression
    
    Core innovation: Compress ALL K/V into k vectors using learnable queries,
    then compute attention on compressed representation.
    
    v1 problem: Ring buffer truncation loses information, sf_basis can't compensate.
    v2 fix: Learnable compression queries select WHICH information to keep.
    
    This is equivalent to cross-attention compression (Perceiver-style),
    framed as signal field interaction.
    
    Trainable parameters:
    - compress_queries: [num_kv_heads, k, head_dim] - what to compress
    - decay_log: [k] - temporal decay (log space for stability)
    
    Frozen (from original attention):
    - q_proj, k_proj, v_proj, o_proj, rope
    """

    def __init__(self, dims, num_heads, num_kv_heads, head_dim, k=16, gamma=0.98):
        self.dims = dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k = k
        self.gamma = gamma
        self.n_rep = num_heads // num_kv_heads

        # Frozen: from original attention
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rope = None

        # Trainable: compression queries [kv_heads, k, head_dim]
        self.compress_queries = mx.random.normal(
            [num_kv_heads, k, head_dim], dtype=mx.float32
        ) * 0.02

        # Trainable: decay [k] in log space
        self.decay_log = mx.log(mx.full([k], gamma, dtype=mx.float32))

    def set_weights(self, attn_layer):
        """Copy frozen weights from original attention"""
        self.q_proj = attn_layer.q_proj
        self.k_proj = attn_layer.k_proj
        self.v_proj = attn_layer.v_proj
        self.o_proj = attn_layer.o_proj
        if hasattr(attn_layer, 'rope'):
            self.rope = attn_layer.rope

    def get_flat_params(self):
        """Get trainable params as flat array for gradient computation"""
        return mx.concatenate([
            self.compress_queries.flatten(),
            self.decay_log.flatten(),
        ])

    def set_flat_params(self, flat):
        """Set trainable params from flat array"""
        n_cq = self.compress_queries.size
        self.compress_queries = flat[:n_cq].reshape(self.compress_queries.shape)
        self.decay_log = flat[n_cq:].reshape(self.decay_log.shape)

    def forward(self, x, mask=None, cache=None):
        """
        Signal field forward pass
        
        1. Q/K/V projections (frozen)
        2. Learnable compression: compress ALL K/V -> k vectors
        3. Query-compressed attention with decay
        4. Output projection
        """
        B, L, _ = x.shape
        scale = 1.0 / (self.head_dim ** 0.5)

        # Q/K/V projection (frozen, from original attention)
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape: [B, L, H, D] -> [B, H, L, D]
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE (positional encoding)
        if self.rope is not None:
            offset = 0
            if cache is not None and hasattr(cache, 'offset'):
                offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        # =================================================================
        # Signal Field Core: Learned Compression
        # =================================================================

        # Step 1: Compress ALL K/V into k vectors
        # compress_queries: [kv_heads, k, D] -> [1, kv_heads, k, D] (broadcast over B)
        cq = self.compress_queries.reshape(1, self.num_kv_heads, self.k, self.head_dim)

        # Compression scores: [1, kv_heads, k, D] @ [B, kv_heads, D, L] -> [B, kv_heads, k, L]
        compress_scores = mx.matmul(cq * scale, keys.transpose(0, 1, 3, 2))
        compress_weights = mx.softmax(compress_scores, axis=-1)  # [B, kv, k, L]

        # Compressed K/V: [B, kv_heads, k, D]
        sf_keys = compress_weights @ keys
        sf_values = compress_weights @ values

        # Step 2: GQA expansion for query heads
        if self.n_rep > 1:
            sf_keys = mx.repeat(sf_keys, self.n_rep, axis=1)
            sf_values = mx.repeat(sf_values, self.n_rep, axis=1)

        # Step 3: Query-compressed attention
        # [B, H, L, D] @ [B, H, D, k] -> [B, H, L, k]
        sf_scores = (queries * scale) @ sf_keys.transpose(0, 1, 3, 2)

        # Step 4: Apply learnable decay
        decay = mx.exp(self.decay_log)  # [k]
        sf_scores = sf_scores * decay.reshape(1, 1, 1, -1)

        # Softmax + weighted sum
        sf_weights = mx.softmax(sf_scores, axis=-1)  # [B, H, L, k]
        output = sf_weights @ sf_values  # [B, H, L, D]

        # Reshape + output projection
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)

        return output

    def __call__(self, x, mask=None, cache=None):
        return self.forward(x, mask, cache)


# ============================================================================
# PPL & NLL Computation (Iron Rules)
# ============================================================================

def compute_nll(logits, input_ids):
    """
    Compute NLL loss for gradient optimization
    
    IRON RULES:
    1. log_softmax subtract max first (prevent NaN)
    2. take_along_axis not mx.take
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # IRON: subtract max first
    max_logits = mx.max(shift_logits, axis=-1, keepdims=True)
    shifted = shift_logits - max_logits
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp

    # IRON: take_along_axis
    expanded_labels = shift_labels[:, :, None]
    token_log_probs = mx.take_along_axis(log_probs, expanded_labels, axis=-1).squeeze(-1)

    mask = shift_labels != 0
    n_tokens = mx.sum(mask)
    if float(n_tokens) == 0:
        return mx.array(0.0)

    return -mx.sum(token_log_probs * mask) / n_tokens


def compute_ppl(model, tokenizer, text, max_length=512):
    """Compute PPL with iron rules"""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float('inf')
    tokens = tokens[:max_length]
    input_ids = mx.array([tokens])

    logits = model(input_ids)
    nll = compute_nll(logits, input_ids)
    return float(mx.exp(nll))


# ============================================================================
# Distillation Training
# ============================================================================

def distill_layer(
    model, tokenizer,
    layer_idx=0, k=DEFAULT_K, gamma=DEFAULT_GAMMA,
    train_steps=DEFAULT_TRAIN_STEPS, lr=DEFAULT_LR,
    print_freq=PRINT_FREQ, max_seq_len=MAX_SEQ_LEN,
):
    """
    Distill signal field for a single layer using full model NLL loss
    
    Strategy:
    - Replace one attention layer with signal field
    - Forward through ENTIRE model
    - NLL loss directly optimizes PPL
    - Gradient only updates signal field params (compress_queries, decay)
    - Model weights are frozen (not in grad function args)
    """
    print(f"\n{'='*60}")
    print(f"Signal Field 0.5B信号场蒸馏训练 v2")
    print(f"{'='*60}")
    print(f"Layer: {layer_idx}")
    print(f"Config: k={k}, gamma={gamma}, lr={lr}, steps={train_steps}")
    print(f"Max seq len: {max_seq_len}")
    print(f"MLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")

    results = {
        'layer_idx': layer_idx,
        'k': k, 'gamma': gamma, 'lr': lr,
        'train_steps': train_steps,
        'max_seq_len': max_seq_len,
    }

    # Prepare training data
    all_texts = CHINESE_TEXTS + ENGLISH_TEXTS
    train_inputs = []
    for text in all_texts:
        tokens = tokenizer.encode(text)[:max_seq_len]
        if len(tokens) > 5:
            train_inputs.append(mx.array([tokens]))

    test_text = CHINESE_TEXTS[0]
    print(f"Training sequences: {len(train_inputs)}")

    # Baseline PPL
    print(f"\n[1/3] Computing Baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, test_text)
    results['baseline_ppl'] = baseline_ppl
    print(f"   Baseline PPL: {baseline_ppl:.4f}")

    # Setup signal field
    print(f"\n[2/3] Setting up SignalField for layer {layer_idx}...")
    original_attn = model.model.layers[layer_idx].self_attn

    sf_attn = SignalFieldAttentionGQA(
        dims=HIDDEN_SIZE,
        num_heads=NUM_ATTENTION_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        k=k, gamma=gamma,
    )
    sf_attn.set_weights(original_attn)
    
    n_params = sf_attn.get_flat_params().size
    print(f"   Trainable params: {n_params} ({n_params * 4 / 1024:.1f} KB)")
    print(f"   compress_queries: {sf_attn.compress_queries.shape}")
    print(f"   decay_log: {sf_attn.decay_log.shape}")

    # Untrained PPL
    model.model.layers[layer_idx].self_attn = sf_attn
    untrained_ppl = compute_ppl(model, tokenizer, test_text)
    print(f"   Untrained PPL: {untrained_ppl:.4f}")
    model.model.layers[layer_idx].self_attn = original_attn

    # Training loop
    print(f"\n[3/3] Training (full model NLL)...")
    print("-" * 60)

    steps_log = []
    step_times = []

    for step in range(1, train_steps + 1):
        step_start = time.time()

        # Pick training sample (cycle through)
        input_ids = train_inputs[(step - 1) % len(train_inputs)]

        # Loss function: takes flat_params, returns NLL
        def loss_fn(flat_params):
            sf_attn.set_flat_params(flat_params)
            model.model.layers[layer_idx].self_attn = sf_attn
            logits = model(input_ids)
            return compute_nll(logits, input_ids)

        # Compute gradient
        params = sf_attn.get_flat_params()
        loss_val, grad = mx.value_and_grad(loss_fn)(params)
        loss_val = float(loss_val)

        # SGD update
        sf_attn.set_flat_params(params - lr * grad)

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Evaluate
        if step % print_freq == 0 or step == train_steps:
            model.model.layers[layer_idx].self_attn = sf_attn
            current_ppl = compute_ppl(model, tokenizer, test_text)
            model.model.layers[layer_idx].self_attn = original_attn

            degr = ((current_ppl - baseline_ppl) / baseline_ppl) * 100

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
                f"NLL={loss_val:.4f} | "
                f"PPL={current_ppl:.2f} ({degr:+.1f}%) | "
                f"{step_time:.1f}s | {status}"
            )

    results['steps'] = steps_log
    results['avg_step_time'] = sum(step_times) / len(step_times) if step_times else 0

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"训练完成")
    print(f"{'='*60}")

    model.model.layers[layer_idx].self_attn = sf_attn
    final_ppl = compute_ppl(model, tokenizer, test_text)
    
    # Test on all texts
    all_ppls = {}
    for i, text in enumerate(CHINESE_TEXTS + ENGLISH_TEXTS):
        ppl = compute_ppl(model, tokenizer, text)
        label = f"text_{i}"
        all_ppls[label] = ppl

    model.model.layers[layer_idx].self_attn = original_attn

    degr = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
    results['final_ppl'] = final_ppl
    results['degradation'] = degr
    results['all_ppls'] = all_ppls
    results['untrained_ppl'] = untrained_ppl

    print(f"Baseline:    {baseline_ppl:.4f}")
    print(f"Untrained:   {untrained_ppl:.4f}")
    print(f"SignalField: {final_ppl:.4f}")
    print(f"Degradation: {degr:+.2f}%")

    if degr < 0:
        print(f"🎉 PPL提升 {-degr:.2f}%!")
    elif degr < 5:
        print(f"✅ 退化 < 5%，优秀！")
    elif degr < 10:
        print(f"⚠️ 退化 < 10%，可用")
    else:
        print(f"❌ 退化 > 10%，需要调参或更多训练")
    print(f"{'='*60}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Signal Field 0.5B信号场蒸馏训练 v2")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--steps', type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--print_freq', type=int, default=PRINT_FREQ)
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        ts = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'distill_v2_layer{args.layer}_{ts}.json')

    # Load model
    from mlx_lm import load as mlx_load

    print(f"MLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    model, tokenizer = mlx_load(args.model_path)
    print(f"Model loaded: {len(model.model.layers)} layers")

    # Train
    results = distill_layer(
        model, tokenizer,
        layer_idx=args.layer,
        k=args.k, gamma=args.gamma,
        train_steps=args.steps, lr=args.lr,
        print_freq=args.print_freq,
        max_seq_len=args.max_seq_len,
    )

    # Save results
    def to_json(obj):
        if isinstance(obj, mx.array):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json(v) for v in obj]
        elif isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        return str(obj)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(to_json(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")

    return results


if __name__ == "__main__":
    main()
