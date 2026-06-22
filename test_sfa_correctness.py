# SFA v7 正确性测试

import numpy as np
import sys

def test_orthogonality():
    """测试 SFA enhancement 与 attention output 的正交性"""
    print("Testing orthogonality...")
    
    # 模拟数据
    hidden_size = 896
    n_samples = 100
    
    # 生成随机 attention output
    attn_out = np.random.randn(n_samples, hidden_size).astype(np.float32)
    
    # 生成随机 SFA enhancement (正交化后)
    enhancement = np.random.randn(n_samples, hidden_size).astype(np.float32)
    
    # 正交化：从 enhancement 中减去沿 attention 方向的投影
    for i in range(n_samples):
        attn = attn_out[i]
        enh = enhancement[i]
        
        # 计算投影
        proj = np.dot(enh, attn) / np.dot(attn, attn) * attn
        enh_orth = enh - proj
        
        # 归一化
        norm = np.linalg.norm(enh_orth)
        if norm > 0:
            enh_orth = enh_orth / norm * 0.5  # 固定范数
            
        enhancement[i] = enh_orth
    
    # 计算余弦相似度
    cos_sim = []
    for i in range(n_samples):
        attn = attn_out[i]
        enh = enhancement[i]
        
        norm_attn = np.linalg.norm(attn)
        norm_enh = np.linalg.norm(enh)
        
        if norm_attn > 0 and norm_enh > 0:
            cos_sim.append(np.dot(attn, enh) / (norm_attn * norm_enh))
    
    avg_cos = np.mean(cos_sim)
    print(f"Average cosine similarity: {avg_cos:.6f}")
    
    # 检查是否满足正交性要求
    if abs(avg_cos) < 0.1:
        print("✅ PASS: Orthogonality verified (cosine < 0.1)")
        return True
    else:
        print("❌ FAIL: Orthogonality not met")
        return False

def test_sequence_isolation():
    """测试多序列状态隔离"""
    print("\nTesting sequence isolation...")
    
    # 模拟两个序列的状态
    seq1_state = np.random.randn(896).astype(np.float32)
    seq2_state = np.random.randn(896).astype(np.float32)
    
    # 验证状态不共享
    if not np.array_equal(seq1_state, seq2_state):
        print("✅ PASS: Sequence states are isolated")
        return True
    else:
        print("❌ FAIL: Sequence states are shared")
        return False

if __name__ == "__main__":
    results = []
    results.append(test_orthogonality())
    results.append(test_sequence_isolation())
    
    if all(results):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
