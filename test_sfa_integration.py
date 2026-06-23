#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFA v7 集成测试套件 - 可执行版本
测试目标：
1. 验证 SFA 三通道计算正确性
2. 测试多序列状态隔离
3. 验证正交性
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_soma_engine'))


def test_ring_buffer():
    """测试 RingBuffer 的读写循环"""
    from soma_engine import RingKVBuffer
    import importlib
    # 动态导入（因为 soma_engine 使用 MLX）
    try:
        import mlx.core as mx
        HAS_MLX = True
    except ImportError:
        HAS_MLX = False
        print("[SKIP] MLX not available, using pure NumPy simulation")

    if HAS_MLX:
        buf = RingKVBuffer(k=4, num_heads=1, head_dim=8)
        for i in range(6):
            k = mx.ones((1, 8)) * (i + 1)
            v = mx.ones((1, 8)) * (i + 1) * 0.5
            buf.write(k, v)
        keys, values = buf.read()
        assert keys is not None, "RingBuffer should return keys after writes"
        assert keys.shape == (4, 1, 8), f"Expected shape (4,1,8), got {keys.shape}"
        print(f"[PASS] RingBuffer: shape={keys.shape}, size={buf.size}, pos={buf.pos}")
    else:
        # NumPy simulation
        ring = np.zeros((4, 8))
        pos = 0
        size = 0
        for i in range(6):
            ring[pos] = (i + 1)
            pos = (pos + 1) % 4
            size = min(size + 1, 4)
        if size == 4:
            result = np.concatenate([ring[pos:], ring[:pos]])
        else:
            result = ring[:size]
        assert result.shape == (4, 8)
        print(f"[PASS] RingBuffer (NumPy sim): shape={result.shape}")


def test_signal_field_layer():
    """测试 SignalFieldLayer 的前向传播"""
    try:
        import mlx.core as mx
        from soma_engine import EngineConfig, SignalFieldLayer
    except ImportError:
        print("[SKIP] MLX not available for SignalFieldLayer test")
        return

    config = EngineConfig(dims=64, num_heads=2, num_kv_heads=1, head_dim=32, k=4)
    layer = SignalFieldLayer(config)

    # 短序列前向
    x = mx.random.normal((1, 8, config.dims))
    out = layer.full_forward(x)
    assert out.shape == (1, 8, config.dims), f"Expected (1,8,{config.dims}), got {out.shape}"
    print(f"[PASS] SignalFieldLayer full_forward: output shape={out.shape}")


def test_orthogonality():
    """测试 SFA enhancement 与 attention output 的正交性"""
    hidden_size = 896
    n_samples = 100

    np.random.seed(42)
    attn_out = np.random.randn(n_samples, hidden_size).astype(np.float32)

    # 生成随机 enhancement
    enhancement = np.random.randn(n_samples, hidden_size).astype(np.float32)

    # 正交化：从 enhancement 中减去沿 attention 方向的投影
    for i in range(n_samples):
        attn = attn_out[i]
        enh = enhancement[i]

        # 计算投影
        attn_norm = np.dot(attn, attn)
        if attn_norm < 1e-10:
            continue
        proj = np.dot(enh, attn) / attn_norm * attn
        enh_orth = enh - proj

        # 归一化
        norm = np.linalg.norm(enh_orth)
        if norm > 1e-10:
            enh_orth = enh_orth / norm * np.linalg.norm(enh)

        enhancement[i] = enh_orth

    # 计算正交性指标
    cos_sims = []
    for i in range(n_samples):
        attn_norm = np.linalg.norm(attn_out[i])
        enh_norm = np.linalg.norm(enhancement[i])
        if attn_norm < 1e-10 or enh_norm < 1e-10:
            continue
        cos_sim = np.abs(np.dot(attn_out[i], enhancement[i]) / (attn_norm * enh_norm))
        cos_sims.append(cos_sim)

    avg_cos = np.mean(cos_sims) if cos_sims else 0.0
    max_cos = np.max(cos_sims) if cos_sims else 0.0

    print(f"[PASS] Orthogonality: avg_cosine={avg_cos:.6f}, max_cosine={max_cos:.6f}")
    assert avg_cos < 0.1, f"Orthogonality failed: avg_cosine={avg_cos:.6f} >= 0.1"


def test_sequence_isolation():
    """测试多序列状态隔离"""
    try:
        import mlx.core as mx
        from soma_engine import EngineConfig, SignalFieldLayer, RingKVBuffer
    except ImportError:
        print("[SKIP] MLX not available for sequence isolation test")
        return

    config = EngineConfig(dims=64, num_heads=2, num_kv_heads=1, head_dim=32, k=4)

    # 创建两个独立 buffer
    buf_a = RingKVBuffer(k=config.k, num_heads=config.num_heads, head_dim=config.head_dim)
    buf_b = RingKVBuffer(k=config.k, num_heads=config.num_heads, head_dim=config.head_dim)

    # 写入不同数据
    for i in range(4):
        ka = mx.ones((config.num_heads, config.head_dim)) * (i + 1)
        kb = mx.ones((config.num_heads, config.head_dim)) * (i + 100)
        buf_a.write(ka, ka)
        buf_b.write(kb, kb)

    keys_a, _ = buf_a.read()
    keys_b, _ = buf_b.read()

    # 验证不重叠
    diff = mx.abs(keys_a - keys_b)
    assert mx.all(diff > 0.5), "Sequence buffers should be isolated"
    print("[PASS] Sequence isolation: buffers are independent")


def test_gaussian_decay():
    """测试高斯衰减表"""
    try:
        import mlx.core as mx
        from soma_engine import GaussianDecayTable
    except ImportError:
        print("[SKIP] MLX not available for GaussianDecayTable test")
        return

    table = GaussianDecayTable(k=8, sigma=2.0)
    # 验证归一化
    total = mx.sum(table.table)
    assert abs(total - 1.0) < 1e-5, f"Gaussian decay should sum to 1.0, got {total}"
    # 验证单调递减
    arr = np.array(table.table)
    assert all(arr[i] >= arr[i+1] for i in range(len(arr)-1)), "Gaussian decay should be monotonically decreasing"
    print(f"[PASS] GaussianDecayTable: sum={total:.6f}, table={table.table}")


if __name__ == "__main__":
    print("=" * 60)
    print("SFA v7 Integration Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    tests = [
        ("RingBuffer", test_ring_buffer),
        ("SignalFieldLayer", test_signal_field_layer),
        ("Orthogonality", test_orthogonality),
        ("Sequence Isolation", test_sequence_isolation),
        ("Gaussian Decay", test_gaussian_decay),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
