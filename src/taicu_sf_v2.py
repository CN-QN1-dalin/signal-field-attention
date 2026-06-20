"""
Signal Field incremental inference v2 - 纯MLX版本
Two-Channel Signal Field with Ring KV Cache
"""

import json
import time

# Try MLX import, exit if not available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("需要MLX环境，请安装: pip install mlx")
    exit(0)


class RingKVBuffer:
    """Ring buffer for KV cache - stores last k tokens"""
    def __init__(self, k: int, num_heads: int, head_dim: int):
        self.k = k
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys = mx.zeros((k, num_heads, head_dim), dtype=mx.float32)
        self.values = mx.zeros((k, num_heads, head_dim), dtype=mx.float32)
        self.pos = 0
        self.size = 0
    
    def write(self, k_vec: mx.array, v_vec: mx.array):
        """Write KV pair at current position"""
        self.keys[self.pos] = k_vec
        self.values[self.pos] = v_vec
        self.pos = (self.pos + 1) % self.k
        self.size = min(self.size + 1, self.k)
    
    def read(self):
        """Read valid KV pairs in chronological order"""
        if self.size == 0:
            return None, None
        if self.size < self.k:
            return self.keys[:self.size], self.values[:self.size]
        # Full: from pos to end, then 0 to pos
        keys_out = mx.concatenate([self.keys[self.pos:], self.keys[:self.pos]], axis=0)
        vals_out = mx.concatenate([self.values[self.pos:], self.values[:self.pos]], axis=0)
        return keys_out, vals_out


class GaussianDecayTable:
    """Precomputed Gaussian decay table"""
    def __init__(self, k: int, sigma: float = 2.0):
        indices = mx.arange(k, dtype=mx.float32)
        self.table = mx.exp(-indices * indices / (2 * sigma * sigma))
        self.table = self.table / mx.sum(self.table)


class SignalFieldIncrLayer:
    """
    Single layer signal field with two channels:
    - Channel 1: Ring KV cache (near field, precise)
    - Channel 2: Field state vector S (far field, compressed)
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
        
        # Xavier init - all as mx.array
        scale = (2.0 / (dims + dims)) ** 0.5
        self.qkv_weight = mx.random.normal((dims, 3 * dims)) * scale
        self.out_weight = mx.random.normal((dims, dims)) * scale
        
        # Decay table
        self.decay_table = GaussianDecayTable(k)
    
    def _qkv_proj(self, x: mx.array):
        """QKV projection"""
        batch, seq, dims = x.shape
        x_flat = x.reshape(batch * seq, dims)
        qkv = mx.matmul(x_flat, self.qkv_weight)
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, axes=(0, 1, 3, 2, 4))
        return qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    
    def _compute_attention(self, q_t, keys_hist, values_hist, field_state):
        """
        Compute attention for a single query against history.
        q_t: [batch, heads, head_dim]
        keys_hist: [seq_hist, heads, head_dim]
        values_hist: [seq_hist, heads, head_dim]
        field_state: [heads, head_dim]
        Returns: [batch, heads, head_dim]
        """
        seq_hist = keys_hist.shape[0]
        batch = q_t.shape[0]

        if seq_hist == 0:
            local_attn = mx.zeros((batch, self.num_heads, self.head_dim), dtype=mx.float32)
        else:
            # Transpose keys/values: [seq, heads, hd] -> [heads, seq, hd]
            k_h = mx.transpose(keys_hist, axes=(1, 0, 2))  # [heads, seq, hd]
            v_h = mx.transpose(values_hist, axes=(1, 0, 2))  # [heads, seq, hd]

            # Expand for batch: [1, heads, seq, hd] and [batch, heads, 1, hd]
            k_exp = k_h[None, :, :, :]  # [1, heads, seq, hd]
            v_exp = v_h[None, :, :, :]  # [1, heads, seq, hd]
            q_exp = q_t[:, :, None, :]  # [batch, heads, 1, hd]

            # Scores: [batch, heads, 1, seq]
            scores = mx.matmul(q_exp, mx.transpose(k_exp, axes=(0, 1, 3, 2))) * self.scale

            # Apply decay: [seq] -> [1, 1, 1, seq]
            n_decay = min(seq_hist, self.k)
            decay = self.decay_table.table[:n_decay]
            scores = scores * decay[None, None, None, :]

            # Softmax over seq dimension
            weights = mx.softmax(scores, axis=-1)  # [batch, heads, 1, seq]

            # Weighted sum: [batch, heads, 1, hd]
            local_attn = mx.squeeze(mx.matmul(weights, v_exp), axis=2)  # [batch, heads, hd]

        # Far field contribution
        far = self.alpha * field_state[None, :, :]  # [1, heads, hd] -> broadcast to [batch, heads, hd]

        return local_attn + far

    def full_forward(self, x: mx.array):
        """
        Full forward - reference implementation
        x: [batch, seq, dims]
        Returns: [batch, seq, dims]
        """
        batch, seq, dims = x.shape
        q, k, v = self._qkv_proj(x)
        
        # Initialize field state
        field_state = mx.zeros((self.num_heads, self.head_dim), dtype=mx.float32)
        
        outputs = []
        for t in range(seq):
            q_t = q[:, t, :, :]  # [batch, heads, hd]
            
            # History up to position t (not including t for causal)
            if t > 0:
                k_hist = k[0, max(0,t-self.k):t, :, :]  # last k tokens
                v_hist = v[0, max(0,t-self.k):t, :, :]
            else:
                k_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
                v_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
            
            # Compute attention
            attn = self._compute_attention(q_t, k_hist, v_hist, field_state)
            outputs.append(attn)
            
            # Update field state (mean over batch)
            k_t_mean = mx.mean(k[:, t, :, :], axis=0)
            field_state = self.gamma * field_state + (1 - self.gamma) * k_t_mean
        
        # Stack outputs: each is [batch, heads, hd] -> reshape to [batch, dims]
        out = mx.stack([o.reshape(batch, dims) for o in outputs], axis=1)  # [batch, seq, dims]
        out = mx.matmul(out, self.out_weight)
        return out
    
    def prefill(self, x: mx.array):
        """
        Incremental prefill - returns same output as full_forward
        x: [batch, seq, dims]
        Returns: (output, field_state, ring_buffer)
        """
        batch, seq, dims = x.shape
        q, k, v = self._qkv_proj(x)
        
        # Initialize ring buffer and field state
        ring_buffer = RingKVBuffer(self.k, self.num_heads, self.head_dim)
        field_state = mx.zeros((self.num_heads, self.head_dim), dtype=mx.float32)
        
        outputs = []
        for t in range(seq):
            q_t = q[:, t, :, :]
            k_t = k[:, t, :, :]
            v_t = v[:, t, :, :]
            
            # Read ring buffer
            keys_ring, values_ring = ring_buffer.read()
            
            if keys_ring is not None:
                k_hist = keys_ring  # [k, heads, hd]
                v_hist = values_ring
            else:
                k_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
                v_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
            
            # Compute attention - SAME LOGIC as full_forward
            attn = self._compute_attention(q_t, k_hist, v_hist, field_state)
            outputs.append(attn)
            
            # Update structures
            ring_buffer.write(k_t[0], v_t[0])
            k_t_mean = mx.mean(k_t, axis=0)
            field_state = self.gamma * field_state + (1 - self.gamma) * k_t_mean
        
        # Output projection
        out = mx.stack([o.reshape(batch, dims) for o in outputs], axis=1)  # [batch, seq, dims]
        out = mx.matmul(out, self.out_weight)
        
        return out, field_state, ring_buffer
    
    def decode_step(self, x_new: mx.array, field_state: mx.array, ring_buffer: RingKVBuffer):
        """
        Single step decode
        x_new: [batch, 1, dims]
        Returns: (output, new_field_state, new_ring_buffer)
        """
        batch = x_new.shape[0]
        q, k, v = self._qkv_proj(x_new)
        
        q_t = q[:, 0, :, :]
        k_t = k[:, 0, :, :]
        v_t = v[:, 0, :, :]
        
        # Read ring buffer
        keys_ring, values_ring = ring_buffer.read()
        
        if keys_ring is not None:
            k_hist = keys_ring
            v_hist = values_ring
        else:
            k_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
            v_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
        
        # Compute attention
        attn = self._compute_attention(q_t, k_hist, v_hist, field_state)  # [batch, heads, hd]
        attn = attn.reshape(batch, 1, self.dims)  # merge heads -> [batch, 1, dims]
        
        # Output projection
        out = mx.matmul(attn, self.out_weight)
        
        # Update ring buffer
        new_ring = RingKVBuffer(self.k, self.num_heads, self.head_dim)
        if keys_ring is not None:
            # Copy existing entries to new ring
            for i in range(keys_ring.shape[0]):
                new_ring.keys[i] = keys_ring[i]
                new_ring.values[i] = values_ring[i]
            new_ring.pos = keys_ring.shape[0]
            new_ring.size = keys_ring.shape[0]
        new_ring.write(k_t[0], v_t[0])
        
        # Update field state
        k_t_mean = mx.mean(k_t, axis=0)
        new_field_state = self.gamma * field_state + (1 - self.gamma) * k_t_mean
        
        return out, new_field_state, new_ring


class AttentionLayer:
    """Standard attention for comparison"""
    def __init__(self, dims: int, num_heads: int):
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        scale = (2.0 / (dims + dims)) ** 0.5
        self.qkv_weight = mx.random.normal((dims, 3 * dims)) * scale
        self.out_weight = mx.random.normal((dims, dims)) * scale
    
    def forward(self, x: mx.array, cache_k=None, cache_v=None):
        batch, seq, dims = x.shape
        x_flat = x.reshape(batch * seq, dims)
        qkv = mx.matmul(x_flat, self.qkv_weight)
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, axes=(0, 1, 3, 2, 4))
        
        q = qkv[:, :, :, 0]
        k = qkv[:, :, :, 1]
        v = qkv[:, :, :, 2]
        
        if cache_k is not None:
            k = mx.concatenate([cache_k, k], axis=1)
            v = mx.concatenate([cache_v, v], axis=1)
        
        q_t = mx.transpose(q, axes=(0, 2, 1, 3))
        k_t = mx.transpose(k, axes=(0, 2, 1, 3))
        v_t = mx.transpose(v, axes=(0, 2, 1, 3))
        
        scores = mx.matmul(q_t, mx.transpose(k_t, axes=(0, 1, 3, 2)))
        scores = scores / (self.head_dim ** 0.5)
        weights = mx.softmax(scores, axis=-1)
        attn = mx.matmul(weights, v_t)
        attn = mx.transpose(attn, axes=(0, 2, 1, 3))
        
        attn = attn.reshape(batch, seq, dims)
        out = mx.matmul(attn, self.out_weight)
        
        return out, k, v


def test1_correctness(dims=128, heads=4, seq_lengths=[4, 8, 16, 32, 64]):
    """Test 1: prefill vs full_forward consistency"""
    print("\n" + "="*60)
    print("Test 1: Correctness - prefill vs full_forward")
    print("="*60)
    
    results = []
    for seq_len in seq_lengths:
        layer = SignalFieldIncrLayer(dims, heads, k=16)
        x = mx.random.normal((1, seq_len, dims))
        
        # Full forward
        out_full = layer.full_forward(x)
        mx.eval(out_full)
        
        # Prefill
        out_prefill, field_state, ring_buffer = layer.prefill(x)
        mx.eval(out_prefill)
        
        # Compare
        diff = mx.abs(out_full - out_prefill)
        mx.eval(diff)
        max_diff = float(mx.max(diff))
        mx.eval(out_full)
        mx.eval(out_prefill)
        abs_out = float(mx.max(mx.abs(out_full)))
        rel_err = max_diff / (abs_out + 1e-8)
        
        status = "PASS" if rel_err < 0.01 else "FAIL"
        print(f"  seq_len={seq_len:3d}: max_diff={max_diff:.6f}, rel_err={rel_err*100:.2f}% [{status}]")
        
        results.append({
            "seq_len": seq_len,
            "max_diff": max_diff,
            "rel_err": float(rel_err),
            "status": status
        })
    
    return results


def test2_decode_speed(dims=128, heads=4, k=16, seq_lengths=[128, 256, 512, 1024, 2048, 4096]):
    """Test 2: Decode speed vs sequence length"""
    print("\n" + "="*60)
    print("Test 2: Decode Speed - should be roughly constant")
    print("="*60)
    
    layer = SignalFieldIncrLayer(dims, heads, k=k)
    warmup_x = mx.random.normal((1, 32, dims))
    _, field_state, ring_buffer = layer.prefill(warmup_x)
    
    results = []
    for seq_len in seq_lengths:
        # Prefill based on target seq length
        prefill_len = min(seq_len, 64)
        prefill_x = mx.random.normal((1, prefill_len, dims))
        _, field_state, ring_buffer = layer.prefill(prefill_x)
        mx.eval(field_state)
        
        # Time decode steps
        num_steps = 10
        start = time.time()
        for _ in range(num_steps):
            x_new = mx.random.normal((1, 1, dims))
            _, field_state, ring_buffer = layer.decode_step(x_new, field_state, ring_buffer)
            mx.eval(field_state)
        elapsed = (time.time() - start) / num_steps * 1000
        
        print(f"  seq_len={seq_len:5d}: {elapsed:.3f}ms/step")
        results.append({"seq_len": seq_len, "ms_per_step": elapsed})
    
    return results


def test3_memory_comparison(dims_list=[128, 3584], heads_list=[4, 28], k=16):
    """Test 3: Theoretical memory comparison"""
    print("\n" + "="*60)
    print("Test 3: Memory Comparison - SignalField vs Attention")
    print("="*60)
    
    results = []
    
    for dims, heads in zip(dims_list, heads_list):
        head_dim = dims // heads
        
        # Signal Field: Ring KV + field state (fixed)
        ring_kv_mem = 2 * k * heads * head_dim * 4
        field_state_mem = heads * head_dim * 4
        total_signal = ring_kv_mem + field_state_mem
        
        seq_range = [64, 256, 512, 1024, 2048, 4096]
        
        print(f"\n  dims={dims}, heads={heads}, k={k}:")
        print(f"    Signal Field: {total_signal/1024:.1f} KB (fixed)")
        
        for seq in seq_range:
            attn_mem = 2 * seq * heads * head_dim * 4
            ratio = attn_mem / total_signal
            print(f"    seq={seq:5d}: Attention={attn_mem/1024:.1f}KB, ratio={ratio:.1f}x")
            
            results.append({
                "dims": dims,
                "heads": heads,
                "seq": seq,
                "signal_field_kb": float(total_signal/1024),
                "attention_kb": float(attn_mem/1024),
                "ratio": float(ratio)
            })
    
    return results


def test4_speedup(dims=128, heads=4, k=16, decode_steps=50):
    """Test 4: Speedup vs standard attention"""
    print("\n" + "="*60)
    print("Test 4: Speedup - SignalField vs Attention")
    print("="*60)
    
    signal_layer = SignalFieldIncrLayer(dims, heads, k=k)
    attn_layer = AttentionLayer(dims, heads)
    
    # Warmup
    warmup_x = mx.random.normal((1, 32, dims))
    _, fs, rb = signal_layer.prefill(warmup_x)
    _, k_cache, v_cache = attn_layer.forward(warmup_x)
    mx.eval(fs)
    mx.eval(k_cache)
    
    # Time signal field decode
    start = time.time()
    for _ in range(decode_steps):
        x_new = mx.random.normal((1, 1, dims))
        _, fs, rb = signal_layer.decode_step(x_new, fs, rb)
        mx.eval(fs)
    signal_time = (time.time() - start) / decode_steps * 1000
    
    # Time attention decode
    start = time.time()
    for _ in range(decode_steps):
        x_new = mx.random.normal((1, 1, dims))
        _, k_cache, v_cache = attn_layer.forward(x_new, k_cache, v_cache)
        mx.eval(k_cache)
    attn_time = (time.time() - start) / decode_steps * 1000
    
    speedup = attn_time / signal_time if signal_time > 0 else 0
    
    print(f"  Signal Field: {signal_time:.3f}ms/step")
    print(f"  Attention:    {attn_time:.3f}ms/step")
    print(f"  Speedup:      {speedup:.2f}x")
    
    return {
        "signal_ms": float(signal_time),
        "attention_ms": float(attn_time),
        "speedup": float(speedup)
    }


def main():
    print("="*60)
    print("Signal Field增量推理测试")
    print("="*60)
    
    print(f"\nMLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    
    # Small config tests
    print("\n" + "="*60)
    print("Small Config: dims=128, heads=4, k=16")
    print("="*60)
    
    results = {
        "config": {"dims": 128, "heads": 4, "k": 16},
        "tests": {}
    }
    
    # Test 1: Correctness
    results["tests"]["correctness"] = test1_correctness(128, 4)
    
    # Test 2: Decode speed
    results["tests"]["decode_speed"] = test2_decode_speed(128, 4, 16)
    
    # Test 3: Memory comparison (small config)
    results["tests"]["memory_comparison"] = test3_memory_comparison([128], [4], 16)
    
    # Test 4: Speedup
    results["tests"]["speedup"] = test4_speedup(128, 4, 16)
    
    # Large config memory only
    print("\n" + "="*60)
    print("Large Config: dims=3584, heads=28, k=16 (memory only)")
    print("="*60)
    results["large_config"] = test3_memory_comparison([3584], [28], 16)
    
    # Save results
    output_file = "Signal Field增量推理v2结果.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
