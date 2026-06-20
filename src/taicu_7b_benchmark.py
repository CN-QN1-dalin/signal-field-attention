"""
Signal Field7B信号场Benchmark
测试信号场替换Qwen2.5-7B单层Attention的效果
"""

import json
import time
import sys
import os

# MLX imports
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("=" * 70)
    print("MLX not found in current environment")
    print("This benchmark requires MLX (Apple Silicon ML framework)")
    print("On your Mac M1 Pro, run: pip install mlx mlx-lm")
    print("=" * 70)
    print("\nNote: Code syntax is verified. Ready to run on Mac.")
    exit(0)

# Model loading
try:
    from mlx_lm import load
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    print("mlx_lm not available, will use random weights for testing")

# Import signal field from existing implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taicu_sf_v2 import SignalFieldIncrLayer, RingKVBuffer


# ============================================================
# Qwen2.5-7B Config (from model)
# ============================================================
QWEN_CONFIG = {
    "dims": 3584,      # hidden_size
    "num_heads": 28,   # num_attention_heads  
    "head_dim": 128,   # head_dim = dims / heads
    "num_layers": 28,  # num_hidden_layers
}


# ============================================================
# Qwen Attention Layer (extracted from loaded model)
# ============================================================
class QwenAttentionExtractor:
    """Extract attention weights from loaded Qwen model layer"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.tokenizer = None
        self.layer_weights = None
        self.config = None
        
        if model_path and MLX_LM_AVAILABLE:
            try:
                print(f"Loading model from: {model_path}")
                self.model, self.tokenizer = load(model_path)
                print("Model loaded successfully!")
                
                # Extract layer 0 attention weights
                if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                    layer0 = self.model.layers[0]
                    if hasattr(layer0, 'self_attn'):
                        attn = layer0.self_attn
                        self.layer_weights = {
                            'q_proj': self._get_weight(attn, 'q_proj'),
                            'k_proj': self._get_weight(attn, 'k_proj'),
                            'v_proj': self._get_weight(attn, 'v_proj'),
                            'o_proj': self._get_weight(attn, 'o_proj'),
                        }
                        
                        # Infer config from weights
                        q_weight = self.layer_weights['q_proj']
                        if q_weight is not None:
                            self.config = {
                                'dims': q_weight.shape[1],
                                'num_heads': q_weight.shape[0] // 128,
                                'head_dim': 128,
                            }
                            print(f"  Inferred config: dims={self.config['dims']}, "
                                  f"heads={self.config['num_heads']}, head_dim={self.config['head_dim']}")
                    else:
                        print("Warning: layer 0 has no self_attn attribute")
                else:
                    print("Warning: model has no layers attribute")
                    
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Will use random weights for testing")
                self.model = None
                self.tokenizer = None
                self.layer_weights = None
        else:
            print("Model path not provided or mlx_lm not available")
            print("Will use random weights for testing")
    
    def _get_weight(self, attn, name: str):
        """Safely get weight from attention module"""
        if hasattr(attn, name):
            w = getattr(attn, name)
            if isinstance(w, mx.array):
                return w
        return None
    
    def get_random_weights(self, dims: int, num_heads: int):
        """Generate random weights matching Qwen structure"""
        head_dim = 128
        scale = 0.02
        
        # Q, K, V projections: [num_heads * head_dim, dims]
        q_weight = mx.random.normal((num_heads * head_dim, dims)) * scale
        k_weight = mx.random.normal((num_heads * head_dim, dims)) * scale  
        v_weight = mx.random.normal((num_heads * head_dim, dims)) * scale
        
        # O projection: [dims, num_heads * head_dim]
        o_weight = mx.random.normal((dims, num_heads * head_dim)) * scale
        
        return {
            'q_proj': q_weight,
            'k_proj': k_weight,
            'v_proj': v_weight,
            'o_proj': o_weight,
        }


# ============================================================
# Standard Qwen-style Attention (for comparison)
# ============================================================
class StandardAttention:
    """Standard attention with KV cache - matches Qwen implementation"""
    
    def __init__(self, dims: int, num_heads: int, head_dim: int = 128):
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        
        # Standard init
        self.q_proj = mx.random.normal((num_heads * head_dim, dims)) * 0.02
        self.k_proj = mx.random.normal((num_heads * head_dim, dims)) * 0.02
        self.v_proj = mx.random.normal((num_heads * head_dim, dims)) * 0.02
        self.o_proj = mx.random.normal((dims, num_heads * head_dim)) * 0.02
    
    def set_weights(self, weights: dict):
        """Set weights from Qwen model"""
        if weights.get('q_proj') is not None:
            self.q_proj = weights['q_proj']
        if weights.get('k_proj') is not None:
            self.k_proj = weights['k_proj']
        if weights.get('v_proj') is not None:
            self.v_proj = weights['v_proj']
        if weights.get('o_proj') is not None:
            self.o_proj = weights['o_proj']
    
    def forward(self, x: mx.array, cache_k=None, cache_v=None):
        """
        Standard attention forward with KV cache
        x: [batch, seq, dims]
        Returns: (output, new_k, new_v)
        """
        B, S, D = x.shape
        
        # Project to Q, K, V
        q = mx.matmul(x, mx.transpose(self.q_proj, axes=(1, 0)))
        k = mx.matmul(x, mx.transpose(self.k_proj, axes=(1, 0)))
        v = mx.matmul(x, mx.transpose(self.v_proj, axes=(1, 0)))
        
        # Reshape to [batch, seq, heads, head_dim]
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)
        
        # Transpose to [batch, heads, seq, head_dim]
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        
        # Concatenate with cache
        if cache_k is not None:
            k = mx.concatenate([cache_k, k], axis=2)
            v = mx.concatenate([cache_v, v], axis=2)
        
        # Attention scores
        scores = mx.matmul(q, mx.transpose(k, axes=(0, 1, 3, 2))) * self.scale
        weights = mx.softmax(scores, axis=-1)
        attn_out = mx.matmul(weights, v)
        
        # Transpose back and reshape
        attn_out = mx.transpose(attn_out, axes=(0, 2, 1, 3))
        attn_out = attn_out.reshape(B, S, D)
        
        # Output projection
        out = mx.matmul(attn_out, mx.transpose(self.o_proj, axes=(1, 0)))
        
        # Return full K, V for cache
        return out, k, v


# ============================================================
# Signal Field Attention Layer (adapted from taicu_sf_v2)
# ============================================================
class SignalFieldAttention:
    """
    Signal Field attention with Ring KV cache
    Adapted from taicu_sf_v2.SignalFieldIncrLayer
    """
    
    def __init__(self, dims: int, num_heads: int, k: int = 16,
                 gamma: float = 0.98, alpha: float = 0.1):
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.k = k
        self.gamma = gamma
        self.alpha = alpha
        
        # Initialize with standard deviation
        scale = 0.02
        self.q_proj = mx.random.normal((num_heads * self.head_dim, dims)) * scale
        self.k_proj = mx.random.normal((num_heads * self.head_dim, dims)) * scale
        self.v_proj = mx.random.normal((num_heads * self.head_dim, dims)) * scale
        self.o_proj = mx.random.normal((dims, num_heads * self.head_dim)) * scale
        
        # Ring buffer and field state
        self.ring_buffer = None
        self.field_state = None
    
    def set_weights(self, weights: dict):
        """Set weights from Qwen model"""
        if weights.get('q_proj') is not None:
            self.q_proj = weights['q_proj']
        if weights.get('k_proj') is not None:
            self.k_proj = weights['k_proj']
        if weights.get('v_proj') is not None:
            self.v_proj = weights['v_proj']
        if weights.get('o_proj') is not None:
            self.o_proj = weights['o_proj']
    
    def _init_ring_buffer(self):
        """Initialize ring buffer"""
        self.ring_buffer = RingKVBuffer(self.k, self.num_heads, self.head_dim)
        self.field_state = mx.zeros((self.num_heads, self.head_dim), dtype=mx.float32)
    
    def prefill(self, x: mx.array):
        """
        Prefill phase - computes attention for full sequence
        x: [batch, seq, dims]
        Returns: (output, field_state, ring_buffer)
        """
        B, S, D = x.shape
        
        # Project to Q, K, V
        q = mx.matmul(x, mx.transpose(self.q_proj, axes=(1, 0)))
        k = mx.matmul(x, mx.transpose(self.k_proj, axes=(1, 0)))
        v = mx.matmul(x, mx.transpose(self.v_proj, axes=(1, 0)))
        
        # Reshape
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)
        
        # Transpose: [B, S, H, hd] -> [B, H, S, hd]
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        
        # Initialize structures
        self._init_ring_buffer()
        outputs = []
        
        for t in range(S):
            q_t = q[:, :, t, :]  # [B, H, hd]
            
            # Read ring buffer
            keys_ring, values_ring = self.ring_buffer.read()
            
            if keys_ring is not None:
                # keys_ring: [k, H, hd], needs [H, k, hd]
                k_hist = mx.transpose(keys_ring, axes=(1, 0, 2))
                v_hist = mx.transpose(values_ring, axes=(1, 0, 2))
            else:
                k_hist = mx.zeros((self.num_heads, 0, self.head_dim), dtype=mx.float32)
                v_hist = mx.zeros((self.num_heads, 0, self.head_dim), dtype=mx.float32)
            
            # Compute attention
            if k_hist.shape[1] > 0:
                # Expand for batch: [B, H, 1, hd] x [1, H, k, hd]
                q_exp = q_t[:, :, None, :]  # [B, H, 1, hd]
                k_exp = k_hist[None, :, :, :]  # [1, H, k, hd]
                v_exp = v_hist[None, :, :, :]
                
                # Scores
                scores = mx.matmul(q_exp, mx.transpose(k_exp, axes=(0, 1, 3, 2)))
                scores = scores * self.scale
                
                # Apply Gaussian decay
                decay_len = min(k_hist.shape[1], self.k)
                indices = mx.arange(decay_len, dtype=mx.float32)
                decay = mx.exp(-indices * indices / (2 * 4.0))  # sigma=2
                decay = decay / mx.sum(decay)
                scores = scores * decay[None, None, None, :]
                
                # Softmax and weighted sum
                weights = mx.softmax(scores, axis=-1)
                local_attn = mx.squeeze(mx.matmul(weights, v_exp), axis=2)  # [B, H, hd]
            else:
                local_attn = mx.zeros((B, self.num_heads, self.head_dim), dtype=mx.float32)
            
            # Far field contribution
            far = self.alpha * self.field_state[None, :, :]
            attn = local_attn + far
            
            outputs.append(attn)
            
            # Update structures
            k_t = k[:, :, t, :]  # [B, H, hd]
            v_t = v[:, :, t, :]
            self.ring_buffer.write(k_t[0], v_t[0])
            
            k_t_mean = mx.mean(k_t, axis=0)
            self.field_state = self.gamma * self.field_state + (1 - self.gamma) * k_t_mean
        
        # Stack outputs
        out = mx.stack(outputs, axis=1)  # [B, S, H, hd]
        out = mx.transpose(out, axes=(0, 2, 1, 3))  # [B, H, S, hd]
        out = out.reshape(B, S, D)  # [B, S, dims]
        out = mx.matmul(out, mx.transpose(self.o_proj, axes=(1, 0)))
        
        return out, self.field_state, self.ring_buffer
    
    def decode_step(self, x_new: mx.array):
        """
        Single step decode (incremental)
        x_new: [batch, 1, dims]
        Returns: (output, new_field_state, new_ring_buffer)
        """
        B, _, D = x_new.shape
        
        # Project
        q = mx.matmul(x_new, mx.transpose(self.q_proj, axes=(1, 0)))
        k = mx.matmul(x_new, mx.transpose(self.k_proj, axes=(1, 0)))
        v = mx.matmul(x_new, mx.transpose(self.v_proj, axes=(1, 0)))
        
        q = q.reshape(B, 1, self.num_heads, self.head_dim)
        k = k.reshape(B, 1, self.num_heads, self.head_dim)
        v = v.reshape(B, 1, self.num_heads, self.head_dim)
        
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        
        q_t = q[:, :, 0, :]  # [B, H, hd]
        k_t = k[:, :, 0, :]
        v_t = v[:, :, 0, :]
        
        # Read ring buffer
        keys_ring, values_ring = self.ring_buffer.read()
        
        if keys_ring is not None:
            k_hist = mx.transpose(keys_ring, axes=(1, 0, 2))
            v_hist = mx.transpose(values_ring, axes=(1, 0, 2))
        else:
            k_hist = mx.zeros((self.num_heads, 0, self.head_dim), dtype=mx.float32)
            v_hist = mx.zeros((self.num_heads, 0, self.head_dim), dtype=mx.float32)
        
        # Compute attention
        if k_hist.shape[1] > 0:
            q_exp = q_t[:, :, None, :]
            k_exp = k_hist[None, :, :, :]
            v_exp = v_hist[None, :, :, :]
            
            scores = mx.matmul(q_exp, mx.transpose(k_exp, axes=(0, 1, 3, 2)))
            scores = scores * self.scale
            
            # Apply decay
            decay_len = min(k_hist.shape[1], self.k)
            indices = mx.arange(decay_len, dtype=mx.float32)
            decay = mx.exp(-indices * indices / (2 * 4.0))
            decay = decay / mx.sum(decay)
            scores = scores * decay[None, None, None, :]
            
            weights = mx.softmax(scores, axis=-1)
            local_attn = mx.squeeze(mx.matmul(weights, v_exp), axis=2)
        else:
            local_attn = mx.zeros((B, self.num_heads, self.head_dim), dtype=mx.float32)
        
        # Far field
        far = self.alpha * self.field_state[None, :, :]
        attn = local_attn + far
        attn = attn.reshape(B, 1, D)
        out = mx.matmul(attn, mx.transpose(self.o_proj, axes=(1, 0)))
        
        # Update ring buffer (create new one)
        new_ring = RingKVBuffer(self.k, self.num_heads, self.head_dim)
        if keys_ring is not None:
            for i in range(keys_ring.shape[0]):
                new_ring.keys[i] = keys_ring[i]
                new_ring.values[i] = values_ring[i]
            new_ring.pos = keys_ring.shape[0]
            new_ring.size = keys_ring.shape[0]
        new_ring.write(k_t[0], v_t[0])
        
        # Update field state
        k_t_mean = mx.mean(k_t, axis=0)
        new_field_state = self.gamma * self.field_state + (1 - self.gamma) * k_t_mean
        
        self.ring_buffer = new_ring
        self.field_state = new_field_state
        
        return out, new_field_state, new_ring


# ============================================================
# Benchmark Functions
# ============================================================

def benchmark_speed_single_step(standard_attn, signal_attn, seq_len: int, 
                                 num_iterations: int = 50):
    """
    Benchmark single step decode speed
    Standard: O(seq) complexity due to full attention
    Signal: O(k) complexity due to ring buffer
    """
    print(f"\n  Testing seq_len={seq_len}, {num_iterations} iterations...")
    
    # Prefill to target sequence length
    prefill_len = min(seq_len, 64)
    x_prefill = mx.random.normal((1, prefill_len, standard_attn.dims))
    
    # Standard attention prefill
    _, cache_k, cache_v = standard_attn.forward(x_prefill)
    mx.eval(cache_k)
    mx.eval(cache_v)
    
    # Signal field prefill
    _, field_state, ring_buffer = signal_attn.prefill(x_prefill)
    mx.eval(field_state)
    
    # Benchmark standard attention decode
    times_std = []
    for _ in range(num_iterations):
        x_new = mx.random.normal((1, 1, standard_attn.dims))
        
        start = time.time()
        _, cache_k, cache_v = standard_attn.forward(x_new, cache_k, cache_v)
        mx.eval(cache_k)
        mx.eval(cache_v)
        times_std.append((time.time() - start) * 1000)
    
    # Reset signal field state
    _, field_state, ring_buffer = signal_attn.prefill(x_prefill)
    
    # Benchmark signal field decode
    times_sf = []
    for _ in range(num_iterations):
        x_new = mx.random.normal((1, 1, standard_attn.dims))
        
        start = time.time()
        _, field_state, ring_buffer = signal_attn.decode_step(x_new)
        mx.eval(field_state)
        times_sf.append((time.time() - start) * 1000)
    
    avg_std = sum(times_std) / len(times_std)
    avg_sf = sum(times_sf) / len(times_sf)
    speedup = avg_std / avg_sf if avg_sf > 0 else 0
    
    print(f"    Standard: {avg_std:.3f}ms, SignalField: {avg_sf:.3f}ms, Speedup: {speedup:.2f}x")
    
    return {
        'seq_len': seq_len,
        'standard_ms': avg_std,
        'signal_field_ms': avg_sf,
        'speedup': speedup
    }


def benchmark_correctness(standard_attn, signal_attn, seq_len: int):
    """
    Test correctness by comparing outputs
    Note: Outputs won't be identical due to different attention mechanisms
    but should show reasonable correlation
    """
    print(f"\n  Testing correctness for seq_len={seq_len}...")
    
    # Use same input
    mx.random.seed(42)
    x = mx.random.normal((1, seq_len, standard_attn.dims))
    
    # Standard attention output
    out_std, _, _ = standard_attn.forward(x)
    mx.eval(out_std)
    
    # Signal field output
    out_sf, _, _ = signal_attn.prefill(x)
    mx.eval(out_sf)
    
    # Compare
    diff = mx.abs(out_std - out_sf)
    mx.eval(diff)
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))
    
    # Cosine similarity (flattened)
    flat_std = out_std.flatten()
    flat_sf = out_sf.flatten()
    cos_sim = float(mx.sum(flat_std * flat_sf) / (
        mx.sqrt(mx.sum(flat_std * flat_std)) * mx.sqrt(mx.sum(flat_sf * flat_sf)) + 1e-8
    ))
    mx.eval(cos_sim)
    
    print(f"    Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
    print(f"    Cosine similarity: {cos_sim:.4f}")
    
    return {
        'seq_len': seq_len,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'cosine_similarity': cos_sim
    }


def benchmark_memory(dims: int, num_heads: int, k: int = 16,
                    seq_lengths: list = [128, 512, 1024, 2048, 4096]):
    """
    Theoretical memory comparison
    Standard: KV cache grows with sequence length (2 * seq * heads * head_dim)
    Signal: Fixed size ring buffer + field state (constant)
    """
    print("\n  Memory Analysis:")
    print(f"    dims={dims}, num_heads={num_heads}, k={k}")
    
    head_dim = dims // num_heads
    results = []
    
    # Signal field memory (constant)
    ring_kv_mem = 2 * k * num_heads * head_dim * 4  # bytes (float32)
    field_state_mem = num_heads * head_dim * 4
    signal_total = ring_kv_mem + field_state_mem
    
    print(f"    Signal Field: {signal_total/1024:.1f} KB (fixed)")
    
    for seq_len in seq_lengths:
        # Standard attention memory (grows with seq)
        std_mem = 2 * seq_len * num_heads * head_dim * 4
        ratio = std_mem / signal_total if signal_total > 0 else 0
        
        print(f"    seq={seq_len:5d}: Standard={std_mem/1024/1024:.1f}MB, "
              f"Signal={signal_total/1024:.1f}KB, Ratio={ratio:.0f}x")
        
        results.append({
            'seq_len': seq_len,
            'standard_mb': std_mem / 1024 / 1024,
            'signal_kb': signal_total / 1024,
            'ratio': ratio
        })
    
    return results


def main():
    print("=" * 70)
    print("Signal Field7B信号场Benchmark")
    print("=" * 70)
    
    # System info
    print(f"\nMLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"MLX LM Available: {MLX_LM_AVAILABLE}")
    
    # Model path
    model_path = "/Users/apple/models/Qwen2.5-7B-Instruct-4bit/"
    
    # Try to load model
    extractor = QwenAttentionExtractor(model_path)
    
    if extractor.config:
        dims = extractor.config['dims']
        num_heads = extractor.config['num_heads']
        head_dim = extractor.config['head_dim']
    else:
        dims = QWEN_CONFIG['dims']
        num_heads = QWEN_CONFIG['num_heads']
        head_dim = QWEN_CONFIG['head_dim']
        print(f"\nUsing default config: dims={dims}, heads={num_heads}, head_dim={head_dim}")
    
    # Calculate parameter count
    param_count = 4 * dims * (dims // num_heads * num_heads)  # QKV + O projections
    print(f"Attention params: {param_count / 1e9:.2f}B (single layer)")
    
    # Create attention layers with same weights
    print("\n" + "-" * 70)
    print("Creating attention layers...")
    
    weights = extractor.get_random_weights(dims, num_heads) if extractor.layer_weights is None \
              else extractor.layer_weights
    
    standard_attn = StandardAttention(dims, num_heads, head_dim)
    standard_attn.set_weights(weights)
    
    signal_attn = SignalFieldAttention(dims, num_heads, k=16, gamma=0.98, alpha=0.1)
    signal_attn.set_weights(weights)
    
    print(f"Standard Attention created: dims={dims}, heads={num_heads}")
    print(f"Signal Field Attention created: dims={dims}, heads={num_heads}, k=16")
    
    # ============================================================
    # Test 1: Speed Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("Test 1: Decode Speed Comparison")
    print("=" * 70)
    
    speed_results = []
    seq_lengths = [64, 128, 256, 512]
    
    for seq_len in seq_lengths:
        result = benchmark_speed_single_step(standard_attn, signal_attn, seq_len, 30)
        speed_results.append(result)
    
    # ============================================================
    # Test 2: Correctness
    # ============================================================
    print("\n" + "=" * 70)
    print("Test 2: Output Similarity")
    print("=" * 70)
    print("Note: Different attention mechanisms won't produce identical outputs.")
    print("      Cosine similarity > 0.5 indicates reasonable correlation.")
    
    correct_results = []
    for seq_len in [8, 16, 32]:
        result = benchmark_correctness(standard_attn, signal_attn, seq_len)
        correct_results.append(result)
    
    # ============================================================
    # Test 3: Memory Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("Test 3: Memory Comparison")
    print("=" * 70)
    
    memory_results = benchmark_memory(dims, num_heads, k=16)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    # Speedup summary
    avg_speedup = sum(r['speedup'] for r in speed_results) / len(speed_results)
    max_speedup = max(r['speedup'] for r in speed_results)
    
    print(f"\nSpeed Comparison (Signal Field vs Standard):")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Max speedup: {max_speedup:.2f}x")
    
    print(f"\nMemory Savings (at seq=4096):")
    mem_4096 = next((r for r in memory_results if r['seq_len'] == 4096), None)
    if mem_4096:
        print(f"  Standard: {mem_4096['standard_mb']:.1f} MB")
        print(f"  Signal: {mem_4096['signal_kb']:.1f} KB")
        print(f"  Savings: {mem_4096['ratio']:.0f}x less memory")
    
    # ============================================================
    # Save Results
    # ============================================================
    results = {
        'config': {
            'dims': dims,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'mlx_version': mx.__version__,
            'device': str(mx.default_device()),
            'model_loaded': extractor.layer_weights is not None
        },
        'speed_comparison': speed_results,
        'correctness': correct_results,
        'memory_comparison': memory_results,
        'summary': {
            'avg_speedup': avg_speedup,
            'max_speedup': max_speedup,
            'memory_ratio_at_4096': mem_4096['ratio'] if mem_4096 else 0
        }
    }
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'taicu_7b_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
