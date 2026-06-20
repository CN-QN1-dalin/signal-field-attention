"""
Signal Field7B信号场PPL验证 v10
验证信号场替换Qwen2.5-7B Attention后的PPL变化

核心设计：
1. 不提取QuantizedLinear权重（4bit不支持提取）
2. 不做梯度训练（4bit不支持backward）
3. 只做forward计算PPL
4. Monkey-patch方式替换Attention计算（保留RoPE等原始机制）

Qwen2.5-7B使用GQA:
- Q: 28 heads, 3584 dim
- K/V: 4 heads, 512 dim (128*4)
- head_dim: 128
- RoPE: 在Q,K投影后、attention前应用
"""

import json
import time
import sys
import os

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("需要MLX环境（Mac M1 Pro）")
    print("运行: pip install mlx mlx-lm")
    exit(0)

try:
    from mlx_lm import load
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    print("mlx_lm not available")
    print("运行: pip install mlx-lm")
    exit(0)

# Import signal field components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taicu_sf_v2 import RingKVBuffer


# ============================================================
# Model Config (Qwen2.5-7B)
# ============================================================
MODEL_PATH = "/Users/apple/models/Qwen2.5-7B-Instruct-4bit/"

# Qwen2.5-7B dimensions
QWEN_CONFIG = {
    "hidden_size": 3584,
    "num_attention_heads": 28,     # Q heads
    "num_key_value_heads": 4,      # K/V heads (GQA)
    "head_dim": 128,               # 3584 / 28 = 128
    "num_hidden_layers": 28,
}


# ============================================================
# PPL Calculation (Core Function)
# ============================================================
def compute_ppl(logits: mx.array, labels: mx.array) -> float:
    """
    Compute perplexity from logits and labels.
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]  # [batch, seq-1, vocab]
    shift_labels = labels[:, 1:]       # [batch, seq-1]
    
    # Log softmax (numerically stable: subtract max first)
    max_logits = mx.max(shift_logits, axis=-1, keepdims=True)
    shifted = shift_logits - max_logits
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    log_softmax = shifted - log_sum_exp  # [batch, seq-1, vocab]
    
    # Gather log probs for correct tokens using take_along_axis
    # shift_labels: [batch, seq-1] -> need [batch, seq-1, 1] for take_along_axis
    expanded_labels = shift_labels[:, :, None]  # [batch, seq-1, 1]
    token_log_probs = mx.take_along_axis(log_softmax, expanded_labels, axis=-1)  # [batch, seq-1, 1]
    token_log_probs = token_log_probs.squeeze(-1)  # [batch, seq-1]
    
    # Mean negative log likelihood
    nll = -mx.mean(token_log_probs)
    mx.eval(nll)
    
    # PPL = exp(nll)
    ppl = float(mx.exp(nll))
    
    return ppl


def compute_ppl_from_text(model, tokenizer, text: str, max_length: int = 512) -> float:
    """Compute PPL for a single text."""
    # Tokenize
    input_ids = mx.array(tokenizer.encode(text))[None, :]  # [1, seq]
    
    # Truncate if needed
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]
    
    # Forward pass
    logits = model(input_ids)
    mx.eval(logits)
    
    # Compute PPL
    ppl = compute_ppl(logits, input_ids)
    
    return ppl


# ============================================================
# Signal Field Attention (GQA + RoPE aware)
# ============================================================
class SignalFieldAttentionGQA:
    """
    Signal Field attention for GQA models (like Qwen2.5-7B).
    
    Key features:
    - Supports GQA: Q has n_heads, K/V have n_kv_heads
    - Preserves RoPE from original attention module
    - Ring buffer for near-field attention
    - Field state for far-field attention
    
    Implements __call__ so it's callable when replacing layer.self_attn.
    """
    
    def __init__(self, original_attn, k: int = 16, 
                 gamma: float = 0.98, alpha: float = 0.1):
        """
        Args:
            original_attn: Original attention module (has rope, q_proj, etc.)
            k: Ring buffer size (near field window)
            gamma: Field state decay
            alpha: Far field weight
        """
        self.original_attn = original_attn
        self.k = k
        self.gamma = gamma
        self.alpha = alpha
        
        # Infer config from model or use defaults
        self.n_heads = QWEN_CONFIG['num_attention_heads']
        self.n_kv_heads = QWEN_CONFIG['num_key_value_heads']
        self.head_dim = QWEN_CONFIG['head_dim']
        self.hidden_size = QWEN_CONFIG['hidden_size']
        
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        # Ring buffer and field state (for KV heads)
        self.ring_buffer = None
        self.field_state = None
        
        # Store rope reference for RoPE application
        self._rope = getattr(original_attn, 'rope', None)
    
    def __call__(self, x: mx.array, mask=None, cache=None):
        """Make this object callable like an attention module."""
        return self.forward(x, mask, cache)
    
    def forward(self, x: mx.array, mask=None, cache=None):
        """
        Forward pass with signal field attention.
        
        Args:
            x: [batch, seq, hidden_size] - input hidden states
            mask: attention mask (unused)
            cache: kv cache (ignored - we use ring buffer)
            
        Returns:
            mx.array: [batch, seq, hidden_size] - output hidden states
        """
        B, S, D = x.shape
        
        # Step 1: QKV projection using original projections
        q = self.original_attn.q_proj(x)  # [B, S, n_heads*hd]
        k = self.original_attn.k_proj(x)  # [B, S, n_kv_heads*hd]
        v = self.original_attn.v_proj(x)  # [B, S, n_kv_heads*hd]
        
        # Step 2: Reshape for attention
        # Q: [B, S, n_heads, hd]
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        # K/V: [B, S, n_kv_heads, hd]
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim)
        
        # Step 3: Transpose to [B, H, S, hd] format
        q = mx.transpose(q, axes=(0, 2, 1, 3))  # [B, n_heads, S, hd]
        k = mx.transpose(k, axes=(0, 2, 1, 3))  # [B, n_kv_heads, S, hd]
        v = mx.transpose(v, axes=(0, 2, 1, 3))  # [B, n_kv_heads, S, hd]
        
        # Step 4: Apply RoPE (IMPORTANT - from original implementation)
        if self._rope is not None:
            q = self._rope(q)  # Apply to Q
            k = self._rope(k)  # Apply to K
        
        # Step 5: Expand K/V to match Q heads for attention computation
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = mx.concatenate([k] * repeat_factor, axis=1)  # [B, n_heads, S, hd]
            v = mx.concatenate([v] * repeat_factor, axis=1)
        
        # Step 6: Initialize structures for signal field
        self.ring_buffer = RingKVBuffer(self.k, self.n_heads, self.head_dim)
        self.field_state = mx.zeros((self.n_heads, self.head_dim), dtype=mx.float32)
        
        outputs = []
        
        # Step 7: Token-by-token attention with ring buffer
        for t in range(S):
            q_t = q[:, :, t, :]  # [B, n_heads, hd]
            
            # Read ring buffer for history
            keys_ring, values_ring = self.ring_buffer.read()
            
            if keys_ring is not None:
                # keys_ring: [k, n_heads, hd] - from ring buffer
                # We need: [n_heads, k, hd] for matmul
                k_hist = mx.transpose(keys_ring, axes=(1, 0, 2))
                v_hist = mx.transpose(values_ring, axes=(1, 0, 2))
            else:
                k_hist = mx.zeros((self.n_heads, 0, self.head_dim), dtype=mx.float32)
                v_hist = mx.zeros((self.n_heads, 0, self.head_dim), dtype=mx.float32)
            
            # Near field: standard dot-product attention with decay
            if k_hist.shape[1] > 0:
                # q_t: [B, n_heads, hd] -> [B, n_heads, 1, hd]
                # k_hist: [n_heads, k, hd] -> [1, n_heads, k, hd]
                q_exp = q_t[:, :, None, :]
                k_exp = k_hist[None, :, :, :]
                v_exp = v_hist[None, :, :, :]
                
                # Scores
                scores = mx.matmul(q_exp, mx.transpose(k_exp, axes=(0, 1, 3, 2)))
                scores = scores * self.scale
                
                # Gaussian decay
                decay_len = min(k_hist.shape[1], self.k)
                indices = mx.arange(decay_len, dtype=mx.float32)
                decay = mx.exp(-indices * indices / (2 * 4.0))  # sigma=2
                decay = decay / (mx.sum(decay) + 1e-10)
                scores = scores * decay[None, None, None, :]
                
                # Softmax and weighted sum
                weights = mx.softmax(scores, axis=-1)
                local_attn = mx.squeeze(mx.matmul(weights, v_exp), axis=2)  # [B, n_heads, hd]
            else:
                local_attn = mx.zeros((B, self.n_heads, self.head_dim), dtype=mx.float32)
            
            # Far field: field state contribution
            far = self.alpha * self.field_state[None, :, :]  # [B, n_heads, hd]
            attn = local_attn + far
            
            outputs.append(attn)
            
            # Update ring buffer with current KV
            k_t = k[:, :, t, :]  # [B, n_heads, hd]
            v_t = v[:, :, t, :]
            self.ring_buffer.write(k_t[0], v_t[0])  # Write batch 0's KV
            
            # Update field state
            k_t_mean = mx.mean(k_t, axis=0)  # [n_heads, hd]
            self.field_state = self.gamma * self.field_state + (1 - self.gamma) * k_t_mean
        
        # Step 8: Stack and reshape outputs
        # outputs: list of [B, n_heads, hd] -> [B, S, n_heads, hd]
        out = mx.stack(outputs, axis=2)
        # Transpose: [B, S, n_heads, hd] -> [B, n_heads, S, hd] -> [B, S, n_heads*hd]
        out = mx.transpose(out, axes=(0, 2, 1, 3))
        out = out.reshape(B, S, self.n_heads * self.head_dim)
        
        # Step 9: Output projection
        out = self.original_attn.o_proj(out)
        
        return out


# ============================================================
# Layer Replacement
# ============================================================
def replace_attention_with_sf(model, layer_idx: int, k: int = 16):
    """Replace attention in a specific layer with signal field version."""
    layer = model.model.layers[layer_idx]
    original_attn = layer.self_attn
    
    # Verify we have rope
    if not hasattr(original_attn, 'rope'):
        print(f"  Warning: Layer {layer_idx} attention has no rope attribute")
    
    # Create signal field wrapper
    sf_attn = SignalFieldAttentionGQA(original_attn, k=k)
    
    # CRITICAL: Replace the entire self_attn object on the layer
    # MLX nn.Module uses __call__ which may bypass a patched forward
    # So we must replace the object itself
    layer.self_attn = sf_attn
    
    print(f"  Layer {layer_idx}: Attention -> Signal Field (k={k}, GQA+RoPE)")


def replace_multiple_layers(model, num_layers: int, k: int = 16):
    """Replace attention in multiple layers."""
    for i in range(num_layers):
        replace_attention_with_sf(model, i, k)


# ============================================================
# Experiments
# ============================================================
def experiment_baseline(model, tokenizer, texts: list) -> dict:
    """Baseline PPL with original model."""
    print("\n" + "=" * 60)
    print("Experiment 1: Baseline PPL (original attention)")
    print("=" * 60)
    
    results = []
    
    for i, text in enumerate(texts):
        preview = text[:60].replace('\n', ' ')
        print(f"\n  Text {i+1}: {preview}...")
        
        start = time.time()
        ppl = compute_ppl_from_text(model, tokenizer, text)
        elapsed = time.time() - start
        
        print(f"    PPL: {ppl:.4f}, Time: {elapsed:.2f}s")
        
        results.append({
            'text_idx': i,
            'text_preview': text[:100],
            'ppl': ppl,
            'time_sec': elapsed
        })
    
    avg_ppl = sum(r['ppl'] for r in results) / len(results)
    print(f"\n  Average PPL: {avg_ppl:.4f}")
    
    return {
        'experiment': 'baseline',
        'avg_ppl': avg_ppl,
        'details': results
    }


def experiment_signal_field(model, tokenizer, texts: list, 
                           num_layers: int = 1, k: int = 16) -> dict:
    """PPL with signal field attention."""
    print("\n" + "=" * 60)
    print(f"Experiment 2: Signal Field PPL")
    print(f"  Layers: 0 to {num_layers-1}")
    print(f"  Ring buffer size (k): {k}")
    print("=" * 60)
    
    # Replace attention layers
    print("\n  Replacing attention layers...")
    replace_multiple_layers(model, num_layers, k)
    
    results = []
    
    for i, text in enumerate(texts):
        preview = text[:60].replace('\n', ' ')
        print(f"\n  Text {i+1}: {preview}...")
        
        start = time.time()
        ppl = compute_ppl_from_text(model, tokenizer, text)
        elapsed = time.time() - start
        
        print(f"    PPL: {ppl:.4f}, Time: {elapsed:.2f}s")
        
        results.append({
            'text_idx': i,
            'text_preview': text[:100],
            'ppl': ppl,
            'time_sec': elapsed
        })
    
    avg_ppl = sum(r['ppl'] for r in results) / len(results)
    print(f"\n  Average PPL: {avg_ppl:.4f}")
    
    return {
        'experiment': f'signal_field_L{num_layers}_k{k}',
        'avg_ppl': avg_ppl,
        'details': results
    }


def experiment_k_comparison(model, tokenizer, texts: list) -> dict:
    """
    Compare PPL for different k values.
    Note: For fair comparison, we reload model for each k value.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: K Value Comparison")
    print("=" * 60)
    
    results = []
    
    for k in [8, 16, 32, 64]:
        print(f"\n  Testing k={k}...")
        
        # Reload model for fair comparison
        try:
            model, tokenizer = load(MODEL_PATH)
        except Exception as e:
            print(f"    Failed to reload model: {e}")
            continue
        
        # Replace with this k
        replace_attention_with_sf(model, 0, k=k)
        
        ppls = []
        for text in texts:
            ppl = compute_ppl_from_text(model, tokenizer, text)
            ppls.append(ppl)
        
        avg_ppl = sum(ppls) / len(ppls)
        print(f"    avg PPL: {avg_ppl:.4f}")
        
        results.append({
            'k': k,
            'avg_ppl': avg_ppl
        })
    
    return {
        'experiment': 'k_comparison',
        'results': results
    }


# ============================================================
# Test Texts
# ============================================================
TEST_TEXTS = [
    # 中文 - 日常
    "今天天气真好，阳光明媚，适合出去散步。公园里有很多人在锻炼身体。",
    # 中文 - 科技
    "人工智能技术正在快速发展，机器学习和深度学习已经应用在很多领域。",
    # 中文 - 文学
    "春天的花朵盛开，空气中弥漫着花香的味道，蝴蝶在花丛中飞舞。",
    # 英文 - 技术
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    # 英文 - 日常
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
]


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Signal Field7B信号场PPL验证 v10")
    print("验证信号场替换后模型PPL变化")
    print("=" * 70)
    
    # System info
    print(f"\nMLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    
    try:
        model, tokenizer = load(MODEL_PATH)
        print("Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        # TokenizerWrapper doesn't support len(), use vocab_size or get_vocab
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            vocab_size = getattr(tokenizer, 'vocab_size', None) or len(getattr(tokenizer, '_tokenizer', tokenizer).get_vocab())
        print(f"  Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nPlease ensure:")
        print("  1. Model path exists:", MODEL_PATH)
        print("  2. MLX and mlx-lm are installed")
        return None
    
    # Check model structure
    print("\nModel structure:")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"  Total layers: {len(model.model.layers)}")
        
        # Check layer 0 attention
        if hasattr(model.model.layers[0], 'self_attn'):
            attn = model.model.layers[0].self_attn
            print(f"  Layer 0 attention type: {type(attn).__name__}")
            print(f"  Has rope: {hasattr(attn, 'rope')}")
            print(f"  Has q_proj: {hasattr(attn, 'q_proj')}")
            print(f"  Has k_proj: {hasattr(attn, 'k_proj')}")
            print(f"  Has v_proj: {hasattr(attn, 'v_proj')}")
            print(f"  Has o_proj: {hasattr(attn, 'o_proj')}")
    
    # Config summary
    print("\nQwen2.5-7B Config:")
    print(f"  hidden_size: {QWEN_CONFIG['hidden_size']}")
    print(f"  num_attention_heads (Q): {QWEN_CONFIG['num_attention_heads']}")
    print(f"  num_key_value_heads (K/V): {QWEN_CONFIG['num_key_value_heads']}")
    print(f"  head_dim: {QWEN_CONFIG['head_dim']}")
    
    # ============================================================
    # Run Experiments
    # ============================================================
    all_results = {
        'config': {
            'model_path': MODEL_PATH,
            'hidden_size': QWEN_CONFIG['hidden_size'],
            'num_attention_heads': QWEN_CONFIG['num_attention_heads'],
            'num_key_value_heads': QWEN_CONFIG['num_key_value_heads'],
            'head_dim': QWEN_CONFIG['head_dim'],
            'num_layers': QWEN_CONFIG['num_hidden_layers'],
            'mlx_version': mx.__version__,
        },
        'test_texts': [t[:50] for t in TEST_TEXTS],
        'experiments': []
    }
    
    # Experiment 1: Baseline
    print("\n" + "=" * 70)
    print("Starting experiments...")
    print("=" * 70)
    
    exp1 = experiment_baseline(model, tokenizer, TEST_TEXTS)
    all_results['experiments'].append(exp1)
    baseline_ppl = exp1['avg_ppl']
    
    # Experiment 2: Signal Field (k=16)
    exp2 = experiment_signal_field(model, tokenizer, TEST_TEXTS, num_layers=1, k=16)
    all_results['experiments'].append(exp2)
    sf_ppl = exp2['avg_ppl']
    
    # Degradation analysis
    print("\n" + "=" * 60)
    print("PPL Analysis")
    print("=" * 60)
    degradation = (sf_ppl - baseline_ppl) / baseline_ppl * 100
    print(f"  Baseline PPL:      {baseline_ppl:.4f}")
    print(f"  Signal Field PPL: {sf_ppl:.4f}")
    print(f"  Change:           {degradation:+.2f}%")
    
    if degradation > 0:
        print("\n  Note: Higher PPL = worse language modeling capability")
        print("        This is expected when changing attention mechanism")
    else:
        print("\n  Surprise: Signal Field improves PPL!")
    
    # Save results
    output_file = "taicu_7b_ppl_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
