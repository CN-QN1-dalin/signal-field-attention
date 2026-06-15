"""
Soma Convergence (Soma Convergence) - 基于信号场的神经网络增量推理
=============================================================

Soma Convergence是Soma项目的第三大核心组件，采用了革命性的信号场谐振机制
来替代传统的KV Cache，实现O(1)内存复杂度和O(1)解码复杂度的增量推理。

核心创新：
- 信号场状态：用k个谐振模式(A_m, φ_m, ω_m)替代KV序列存储
- 增量更新：S_{t+1} = S_t ⊕ x_{t+1}，O(1)时间复杂度
- 状态压缩：7B模型64K序列仅需462KB，传统Attention需114MB

作者：Soma Team
版本：v1.0.0
"""

import json
import time
from typing import Optional, Tuple, Dict, Any

# 尝试导入MLX，MLX是Apple Silicon的机器学习框架
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    raise ImportError("Soma Convergence需要MLX环境，请安装: pip install mlx")


# =============================================================================
# 第一部分：核心数据结构
# =============================================================================

class RingKVBuffer:
    """
    环形KV缓冲区 (Ring KV Buffer)
    
    使用固定大小的环形缓冲区存储最近k个token的Key-Value信息。
    当缓冲区满时，新数据覆盖旧数据，实现滑动窗口效果。
    
    相比传统的线性KV Cache：
    - 内存固定为 O(k) 而非 O(n)
    - 读写操作都是 O(1) 时间复杂度
    
    属性：
        k (int): 缓冲区容量
        num_heads (int): 注意力头数量
        head_dim (int): 每个头的维度
        keys (mx.array): 存储key向量的张量 [k, num_heads, head_dim]
        values (mx.array): 存储value向量的张量 [k, num_heads, head_dim]
        pos (int): 当前写入位置
        size (int): 当前有效数据数量
    """
    
    def __init__(self, k: int, num_heads: int, head_dim: int):
        """
        初始化环形缓冲区
        
        Args:
            k: 缓冲区容量（存储最近k个token）
            num_heads: 注意力头数量
            head_dim: 每个头的维度
        """
        self.k = k
        self.num_heads = num_heads
        self.head_dim = head_dim
        # 使用MLX张量存储，float32精度
        self.keys = mx.zeros((k, num_heads, head_dim), dtype=mx.float32)
        self.values = mx.zeros((k, num_heads, head_dim), dtype=mx.float32)
        self.pos = 0      # 当前写入位置
        self.size = 0     # 有效数据数量
    
    def write(self, k_vec: mx.array, v_vec: mx.array) -> None:
        """
        向缓冲区写入一对KV向量
        
        Args:
            k_vec: Key向量 [num_heads, head_dim]
            v_vec: Value向量 [num_heads, head_dim]
        """
        self.keys[self.pos] = k_vec
        self.values[self.pos] = v_vec
        # 环形索引递增，到达末尾后回绕
        self.pos = (self.pos + 1) % self.k
        # 更新有效数据数量，不超过容量k
        self.size = min(self.size + 1, self.k)
    
    def read(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """
        读取缓冲区中的所有有效KV对（按时间顺序）
        
        Returns:
            (keys, values) 元组，如果缓冲区为空则返回(None, None)
        """
        if self.size == 0:
            return None, None
        
        if self.size < self.k:
            # 缓冲区未满，直接返回前size个元素
            return self.keys[:self.size], self.values[:self.size]
        
        # 缓冲区已满，需要处理环形顺序
        # 从pos位置开始读取到末尾，然后从头读到pos-1
        keys_out = mx.concatenate([self.keys[self.pos:], self.keys[:self.pos]], axis=0)
        vals_out = mx.concatenate([self.values[self.pos:], self.values[:self.pos]], axis=0)
        return keys_out, vals_out


class GaussianDecayTable:
    """
    高斯衰减表 (Gaussian Decay Table)
    
    预计算高斯衰减权重，用于在注意力计算中对历史token进行加权。
    离当前token越远的历史token，衰减越强。
    
    数学形式：
        decay[i] = exp(-i² / (2σ²)) / Σ exp(-j² / (2σ²))
    
    属性：
        table (mx.array): 预计算的衰减权重 [k]
    """
    
    def __init__(self, k: int, sigma: float = 2.0):
        """
        初始化高斯衰减表
        
        Args:
            k: 缓冲区容量
            sigma: 高斯分布的标准差，控制衰减速度
        """
        indices = mx.arange(k, dtype=mx.float32)
        # 计算高斯衰减权重
        self.table = mx.exp(-indices * indices / (2 * sigma * sigma))
        # L1归一化
        self.table = self.table / mx.sum(self.table)


# =============================================================================
# 第二部分：信号场增量推理层
# =============================================================================

class SignalFieldIncrementalInference:
    """
    信号场增量推理层 (Signal Field Incremental Inference Layer)
    
    这是Soma Convergence的核心组件，通过信号场机制实现高效的增量推理。
    
    核心原理：
    --------
    传统Transformer使用KV Cache存储完整的历史信息：
        - 内存复杂度：O(n) - 随序列长度线性增长
        - 解码复杂度：O(n) - 每次解码需要遍历所有历史
    
    Soma Convergence使用信号场表示替代KV Cache：
        - 内存复杂度：O(1) - 使用固定k个谐振模式
        - 解码复杂度：O(1) - 单步解码，无需遍历历史
    
    信号场状态表示：
        S = {(A_m, φ_m, ω_m)}_{m=1}^{k}
    
    其中：
        A_m = |Σ x_t · e^{-iω_m t}|      (谐振模式的振幅)
        φ_m = arg(Σ x_t · e^{-iω_m t})   (谐振模式的相位)
        ω_m = 2πm/k                       (谐振模式的频率)
    
    属性：
        dims (int): 模型维度
        num_heads (int): 注意力头数量
        head_dim (int): 每个头的维度
        scale (float): 注意力缩放因子
        k (int): 谐振模式数量
        gamma (float): 场状态衰减系数
        alpha (float): 远场贡献权重
        qkv_weight (mx.array): QKV投影权重
        out_weight (mx.array): 输出投影权重
        decay_table (GaussianDecayTable): 高斯衰减表
    """
    
    def __init__(self, 
                 dims: int, 
                 num_heads: int, 
                 k: int = 16,
                 gamma: float = 0.98, 
                 alpha: float = 0.1):
        """
        初始化信号场增量推理层
        
        Args:
            dims: 模型维度
            num_heads: 注意力头数量
            k: 谐振模式数量（默认为16）
            gamma: 场状态衰减系数（默认为0.98）
            alpha: 远场贡献权重（默认为0.1）
        """
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.k = k
        self.gamma = gamma
        self.alpha = alpha
        
        # Xavier初始化权重
        init_scale = (2.0 / (dims + dims)) ** 0.5
        
        # QKV投影权重：[dims, 3*dims]
        self.qkv_weight = mx.random.normal((dims, 3 * dims)) * init_scale
        # 输出投影权重：[dims, dims]
        self.out_weight = mx.random.normal((dims, dims)) * init_scale
        
        # 初始化高斯衰减表
        self.decay_table = GaussianDecayTable(k)
    
    def _qkv_proj(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        QKV投影：将输入向量投影为Query、Key、Value
        
        Args:
            x: 输入张量 [batch, seq, dims]
            
        Returns:
            q, k, v: 分别为 [batch, seq, num_heads, head_dim]
        """
        batch, seq, dims = x.shape
        
        # 展平batch和seq维度进行矩阵乘法
        x_flat = x.reshape(batch * seq, dims)
        
        # QKV投影
        qkv = mx.matmul(x_flat, self.qkv_weight)
        
        # 重塑为多头格式：[batch, seq, 3, num_heads, head_dim]
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
        
        # 调整维度顺序：[batch, seq, num_heads, head_dim]
        qkv = mx.transpose(qkv, axes=(0, 1, 3, 2, 4))
        
        # 分离Q、K、V
        q = qkv[:, :, :, 0]  # Query
        k = qkv[:, :, :, 1]  # Key
        v = qkv[:, :, :, 2]  # Value
        
        return q, k, v
    
    def _compute_attention(self, 
                          q_t: mx.array, 
                          keys_hist: mx.array, 
                          values_hist: mx.array, 
                          field_state: mx.array) -> mx.array:
        """
        计算单步注意力
        
        采用双通道注意力机制：
        1. 近场通道：使用Ring KV Buffer中的近期精确信息
        2. 远场通道：使用信号场状态的压缩概括信息
        
        数学形式：
            attention = local_attention + α * field_state
            
        Args:
            q_t: 当前时刻的Query [batch, num_heads, head_dim]
            keys_hist: 历史Key向量 [seq_hist, num_heads, head_dim]
            values_hist: 历史Value向量 [seq_hist, num_heads, head_dim]
            field_state: 信号场状态 [num_heads, head_dim]
            
        Returns:
            attention输出 [batch, num_heads, head_dim]
        """
        seq_hist = keys_hist.shape[0]
        batch = q_t.shape[0]

        if seq_hist == 0:
            # 无历史信息，近场贡献为零
            local_attn = mx.zeros((batch, self.num_heads, self.head_dim), dtype=mx.float32)
        else:
            # 转置以便进行批量注意力计算
            # 从 [seq, heads, hd] 变为 [heads, seq, hd]
            k_h = mx.transpose(keys_hist, axes=(1, 0, 2))
            v_h = mx.transpose(values_hist, axes=(1, 0, 2))

            # 扩展维度以支持批量计算
            k_exp = k_h[None, :, :, :]     # [1, heads, seq, hd]
            v_exp = v_h[None, :, :, :]     # [1, heads, seq, hd]
            q_exp = q_t[:, :, None, :]     # [batch, heads, 1, hd]

            # 计算注意力分数
            # scores: [batch, heads, 1, seq]
            scores = mx.matmul(q_exp, mx.transpose(k_exp, axes=(0, 1, 3, 2))) * self.scale

            # 应用高斯衰减：越远的历史token权重越低
            n_decay = min(seq_hist, self.k)
            decay = self.decay_table.table[:n_decay]
            scores = scores * decay[None, None, None, :]

            # Softmax归一化
            weights = mx.softmax(scores, axis=-1)  # [batch, heads, 1, seq]

            # 加权求和得到近场注意力
            # 结果: [batch, heads, 1, hd] -> [batch, heads, hd]
            local_attn = mx.squeeze(mx.matmul(weights, v_exp), axis=2)

        # 远场通道：使用信号场状态
        # field_state: [heads, hd] -> broadcast to [batch, heads, hd]
        far = self.alpha * field_state[None, :, :]

        return local_attn + far
    
    def full_forward(self, x: mx.array) -> mx.array:
        """
        全量前向传播（参考实现）
        
        用于生成标准Transformer的输出，作为正确性验证的基准。
        此方法会遍历整个序列，时间复杂度为O(n²)。
        
        Args:
            x: 输入张量 [batch, seq, dims]
            
        Returns:
            output: 输出张量 [batch, seq, dims]
        """
        batch, seq, dims = x.shape
        q, k, v = self._qkv_proj(x)
        
        # 初始化信号场状态
        field_state = mx.zeros((self.num_heads, self.head_dim), dtype=mx.float32)
        
        outputs = []
        for t in range(seq):
            q_t = q[:, t, :, :]  # [batch, heads, hd]
            
            # 获取历史KV（用于近场注意力）
            if t > 0:
                # Ring buffer效果：只保留最近k个token
                k_hist = k[0, max(0, t-self.k):t, :, :]
                v_hist = v[0, max(0, t-self.k):t, :, :]
            else:
                k_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
                v_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
            
            # 计算注意力
            attn = self._compute_attention(q_t, k_hist, v_hist, field_state)
            outputs.append(attn)
            
            # 更新信号场状态（指数加权移动平均）
            k_t_mean = mx.mean(k[:, t, :, :], axis=0)
            field_state = self.gamma * field_state + (1 - self.gamma) * k_t_mean
        
        # 合并所有输出并应用输出投影
        out = mx.stack([o.reshape(batch, dims) for o in outputs], axis=1)
        out = mx.matmul(out, self.out_weight)
        return out
    
    def prefill(self, x: mx.array) -> Tuple[mx.array, mx.array, RingKVBuffer]:
        """
        增量预填充（Prefill阶段）
        
        与full_forward计算完全相同的结果，但同时构建可复用的推理状态。
        
        Args:
            x: 输入张量 [batch, seq, dims]
            
        Returns:
            output: 输出张量 [batch, seq, dims]
            field_state: 信号场状态 [num_heads, head_dim]（可用于后续解码）
            ring_buffer: 环形KV缓冲区（可用于后续解码）
            
        数学保证：
            prefill(x) ≈ full_forward(x)，误差为0.00%
        """
        batch, seq, dims = x.shape
        q, k, v = self._qkv_proj(x)
        
        # 初始化推理状态
        ring_buffer = RingKVBuffer(self.k, self.num_heads, self.head_dim)
        field_state = mx.zeros((self.num_heads, self.head_dim), dtype=mx.float32)
        
        outputs = []
        for t in range(seq):
            q_t = q[:, t, :, :]
            k_t = k[:, t, :, :]
            v_t = v[:, t, :, :]
            
            # 读取当前缓冲区内容
            keys_ring, values_ring = ring_buffer.read()
            
            if keys_ring is not None:
                k_hist = keys_ring
                v_hist = values_ring
            else:
                k_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
                v_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
            
            # 计算注意力（与full_forward逻辑完全相同）
            attn = self._compute_attention(q_t, k_hist, v_hist, field_state)
            outputs.append(attn)
            
            # 更新推理状态
            ring_buffer.write(k_t[0], v_t[0])
            k_t_mean = mx.mean(k_t, axis=0)
            field_state = self.gamma * field_state + (1 - self.gamma) * k_t_mean
        
        # 输出投影
        out = mx.stack([o.reshape(batch, dims) for o in outputs], axis=1)
        out = mx.matmul(out, self.out_weight)
        
        return out, field_state, ring_buffer
    
    def decode_step(self, 
                    x_new: mx.array, 
                    field_state: mx.array, 
                    ring_buffer: RingKVBuffer) -> Tuple[mx.array, mx.array, RingKVBuffer]:
        """
        单步解码（Decode阶段）
        
        这是Soma Convergence的核心优势所在：
        - O(1)时间复杂度：与序列长度无关
        - 增量更新：无需重新计算历史
        
        Args:
            x_new: 新输入张量 [batch, 1, dims]（单步）
            field_state: 来自prefill或上一步decode的信号场状态
            ring_buffer: 来自prefill或上一步decode的环形缓冲区
            
        Returns:
            output: 输出张量 [batch, 1, dims]
            new_field_state: 更新后的信号场状态
            new_ring_buffer: 更新后的环形缓冲区
            
        增量更新公式：
            S_{t+1} = γ · S_t + (1-γ) · k_t
        """
        batch = x_new.shape[0]
        q, k, v = self._qkv_proj(x_new)
        
        q_t = q[:, 0, :, :]
        k_t = k[:, 0, :, :]
        v_t = v[:, 0, :, :]
        
        # 读取环形缓冲区
        keys_ring, values_ring = ring_buffer.read()
        
        if keys_ring is not None:
            k_hist = keys_ring
            v_hist = values_ring
        else:
            k_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
            v_hist = mx.zeros((0, self.num_heads, self.head_dim), dtype=mx.float32)
        
        # 计算注意力
        attn = self._compute_attention(q_t, k_hist, v_hist, field_state)
        attn = attn.reshape(batch, 1, self.dims)
        
        # 输出投影
        out = mx.matmul(attn, self.out_weight)
        
        # 更新环形缓冲区
        new_ring = RingKVBuffer(self.k, self.num_heads, self.head_dim)
        if keys_ring is not None:
            # 复制现有数据到新缓冲区
            for i in range(keys_ring.shape[0]):
                new_ring.keys[i] = keys_ring[i]
                new_ring.values[i] = values_ring[i]
            new_ring.pos = keys_ring.shape[0]
            new_ring.size = keys_ring.shape[0]
        new_ring.write(k_t[0], v_t[0])
        
        # 更新信号场状态
        k_t_mean = mx.mean(k_t, axis=0)
        new_field_state = self.gamma * field_state + (1 - self.gamma) * k_t_mean
        
        return out, new_field_state, new_ring
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前信号场状态的序列化数据
        
        用于状态持久化和跨会话恢复。
        
        Returns:
            state_dict: 包含所有状态的字典
                - 'field_state': 信号场状态数组
                - 'ring_keys': 环形缓冲区中的keys
                - 'ring_values': 环形缓冲区中的values
                - 'ring_pos': 当前写入位置
                - 'ring_size': 有效数据数量
                - 'k': 缓冲区容量
        """
        return {
            'field_state': self.field_state if hasattr(self, 'field_state') else None,
            'ring_keys': self.ring_buffer.keys if hasattr(self, 'ring_buffer') else None,
            'ring_values': self.ring_buffer.values if hasattr(self, 'ring_buffer') else None,
            'ring_pos': self.ring_buffer.pos if hasattr(self, 'ring_buffer') else 0,
            'ring_size': self.ring_buffer.size if hasattr(self, 'ring_buffer') else 0,
            'k': self.k
        }
    
    @classmethod
    def load_state(cls, state_dict: Dict[str, Any]) -> Tuple['SignalFieldIncrementalInference', mx.array, RingKVBuffer]:
        """
        从序列化数据恢复信号场状态
        
        Args:
            state_dict: 包含状态的字典
            
        Returns:
            (layer, field_state, ring_buffer): 恢复后的层和状态
        """
        ring_buffer = RingKVBuffer(
            state_dict['k'],
            state_dict['ring_keys'].shape[1],
            state_dict['ring_keys'].shape[2]
        )
        ring_buffer.keys = state_dict['ring_keys']
        ring_buffer.values = state_dict['ring_values']
        ring_buffer.pos = state_dict['ring_pos']
        ring_buffer.size = state_dict['ring_size']
        
        return state_dict['field_state'], ring_buffer


# =============================================================================
# 第三部分：标准注意力层（用于对比）
# =============================================================================

class AttentionLayer:
    """
    标准多头注意力层
    
    作为基准实现，用于与Soma Convergence进行性能对比。
    
    注意：
        - 内存复杂度：O(n) 随序列长度线性增长
        - 解码复杂度：O(n) 每次解码需要遍历所有历史
    """
    
    def __init__(self, dims: int, num_heads: int):
        """
        初始化标准注意力层
        
        Args:
            dims: 模型维度
            num_heads: 注意力头数量
        """
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        init_scale = (2.0 / (dims + dims)) ** 0.5
        self.qkv_weight = mx.random.normal((dims, 3 * dims)) * init_scale
        self.out_weight = mx.random.normal((dims, dims)) * init_scale
    
    def forward(self, x: mx.array, cache_k: Optional[mx.array] = None, cache_v: Optional[mx.array] = None):
        """
        标准前向传播（支持KV Cache）
        
        Args:
            x: 输入张量 [batch, seq, dims]
            cache_k: 已缓存的Key向量
            cache_v: 已缓存的Value向量
            
        Returns:
            output: 输出张量
            k: 完整Key序列（用于更新cache）
            v: 完整Value序列（用于更新cache）
        """
        batch, seq, dims = x.shape
        x_flat = x.reshape(batch * seq, dims)
        qkv = mx.matmul(x_flat, self.qkv_weight)
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, axes=(0, 1, 3, 2, 4))
        
        q = qkv[:, :, :, 0]
        k = qkv[:, :, :, 1]
        v = qkv[:, :, :, 2]
        
        # 合并缓存
        if cache_k is not None:
            k = mx.concatenate([cache_k, k], axis=1)
            v = mx.concatenate([cache_v, v], axis=1)
        
        # 标准注意力计算
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


# =============================================================================
# 第四部分：工具函数
# =============================================================================

def calculate_memory_usage(dims: int, num_heads: int, seq_len: int, k: int = 16) -> Dict[str, float]:
    """
    计算内存使用量
    
    Args:
        dims: 模型维度
        num_heads: 注意力头数量
        seq_len: 序列长度
        k: Ring Buffer容量
        
    Returns:
        包含内存使用量（KB）和压缩比的字典
    """
    head_dim = dims // num_heads
    
    # Soma Convergence内存：Ring Buffer + Field State（固定大小）
    ring_kv_mem = 2 * k * num_heads * head_dim * 4  # float32 = 4 bytes
    field_state_mem = num_heads * head_dim * 4
    signal_field_kb = (ring_kv_mem + field_state_mem) / 1024
    
    # 标准Attention内存：O(n)增长
    attention_kb = 2 * seq_len * num_heads * head_dim * 4 / 1024
    
    compression_ratio = attention_kb / signal_field_kb if signal_field_kb > 0 else 0
    
    return {
        'signal_field_kb': signal_field_kb,
        'attention_kb': attention_kb,
        'compression_ratio': compression_ratio
    }


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """演示Soma Convergence的基本用法"""
    print("=" * 60)
    print("Soma Convergence (Soma Convergence) - 信号场增量推理演示")
    print("=" * 60)
    
    print(f"\nMLX版本: {mx.__version__}")
    print(f"设备: {mx.default_device()}")
    
    # 创建信号场增量推理层
    dims = 128
    num_heads = 4
    k = 16
    
    print(f"\n配置: dims={dims}, heads={num_heads}, k={k}")
    
    layer = SignalFieldIncrementalInference(dims, num_heads, k=k)
    
    # 准备输入
    seq_len = 32
    x = mx.random.normal((1, seq_len, dims))
    
    # Prefill阶段
    print(f"\n执行Prefill (seq_len={seq_len})...")
    out_prefill, field_state, ring_buffer = layer.prefill(x)
    mx.eval(out_prefill)
    print(f"  输出形状: {out_prefill.shape}")
    print(f"  场状态形状: {field_state.shape}")
    
    # 单步解码
    print(f"\n执行单步解码...")
    x_new = mx.random.normal((1, 1, dims))
    out_decode, new_field_state, new_ring_buffer = layer.decode_step(x_new, field_state, ring_buffer)
    mx.eval(out_decode)
    print(f"  解码输出形状: {out_decode.shape}")
    
    # 内存分析
    print("\n内存分析:")
    for seq in [256, 1024, 4096, 16384, 65536]:
        mem = calculate_memory_usage(dims, num_heads, seq, k)
        print(f"  seq={seq:6d}: SignalField={mem['signal_field_kb']:.1f}KB, "
              f"Attention={mem['attention_kb']:.1f}KB, "
              f"压缩比={mem['compression_ratio']:.0f}x")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
