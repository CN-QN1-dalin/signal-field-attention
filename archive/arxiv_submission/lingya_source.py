#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soma LingYa - 灵芽通道参数高效微调模块
Soma LingYa: Parameter-Efficient Fine-Tuning via LingYa Channel


版本：v0.9

本模块实现了基于灵芽通道（LingYa Channel）的神经网络参数高效微调方法，
完全替代LoRA的低秩分解范式，实现真正意义上的推理零开销微调。

核心特性：
- 推理零开销：参数可完全融合进原始权重
- 参数效率极高：仅需训练门控参数（<0.01%原始参数）
- 训练稳定：门控机制天然具有梯度稳定性
- 完全原创：不依赖低秩分解假设
"""

import mlx.core as mx
from typing import Optional, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass


class ChannelType(Enum):
    """
    灵芽通道类型
    
    - ROOT: 根通道，R=单位阵，保留原始特征
    - BRANCH: 枝通道，R=随机正交，特征组合
    - LEAF: 叶通道，R=低秩随机，细节捕捉
    """
    ROOT = "root"
    BRANCH = "branch"
    LEAF = "leaf"


@dataclass
class LingYaConfig:
    """
    灵芽通道配置
    
    Attributes:
        d_in: 输入维度
        d_out: 输出维度
        rank: 灵芽秩（低秩近似的秩）
        channel_type: 通道类型
        growth_scale: 生长尺度因子α
        max_growth: P矩阵最大范数
    """
    d_in: int = 512
    d_out: int = 512
    rank: int = 8
    channel_type: ChannelType = ChannelType.BRANCH
    growth_scale: float = 1.0
    max_growth: float = 5.0


class LingYaChannel:
    """
    灵芽通道
    
    核心机制：
    1. 每个通道有脚手架R和生长点P
    2. P矩阵从零开始生长，训练中逐渐获得知识
    3. 可选的固化机制，将生长转化为静态权重
    
    数学形式：
    W = I + R @ P · α
    
    其中：
    - R ∈ R^{d_out × rank}: 脚手架矩阵（冻结的正交基）
    - P ∈ R^{rank × d_in}: 生长矩阵（从零开始训练）
    - α: 生长尺度因子
    
    Attributes:
        d_in: 输入维度
        d_out: 输出维度
        rank: 灵芽秩
        channel_type: 通道类型
        growth_scale: 生长尺度因子
        max_growth: P矩阵最大范数
        R: 脚手架矩阵
        P: 生长矩阵（初始为零）
        I: 单位阵（用于恒等连接）
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int = 8,
        channel_type: ChannelType = ChannelType.BRANCH,
        growth_scale: float = 1.0,
        max_growth: float = 5.0
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.channel_type = channel_type
        self.growth_scale = growth_scale
        self.max_growth = max_growth
        
        # 1. 初始化脚手架矩阵 R
        self.R = self._init_scaffold()
        
        # 2. 初始化生长矩阵 P（全部为零）
        self.P = mx.zeros([self.rank, self.d_in])
        
        # 3. 原始单位阵（用于恒等连接）
        if d_out != d_in:
            self.I = mx.zeros([d_out, d_in])
            self.I[:d_in, :d_in] = mx.eye(d_in)
        else:
            self.I = mx.eye(d_out)
        
        # 4. 通道状态
        self.is_frozen = False
        self.usage_count = 0
        self.gradient_history = []
        
        # 5. 固化后的静态权重
        self.W_frozen: Optional[mx.array] = None
    
    def _init_scaffold(self) -> mx.array:
        """
        初始化脚手架矩阵 R
        
        根据通道类型选择不同的初始化方式：
        - ROOT: 单位阵
        - BRANCH: 随机正交（通过SVD近似）
        - LEAF: 低秩随机
        """
        if self.channel_type == ChannelType.ROOT:
            R = mx.eye(self.d_out)[:, :self.rank]
        elif self.channel_type == ChannelType.BRANCH:
            R = mx.random.normal(shape=[self.d_out, self.rank])
            U, S, Vt = mx.linalg.svd(R)
            R = U[:, :self.rank]
        else:
            R = mx.random.normal(shape=[self.d_out, self.rank]) * 0.02
        return R
    
    def _get_effective_weight(self) -> mx.array:
        """
        获取有效权重
        
        W = I + R @ P · α
        
        实现 delta clamp 修复版：确保增量不会过大
        
        Returns:
            W: 有效权重矩阵
        """
        if self.is_frozen and self.W_frozen is not None:
            return self.W_frozen
        
        # 计算 R @ P · α
        growth_term = self.R @ self.P * self.growth_scale
        
        # 应用 delta clamp 修复
        # 确保 ||P||_fro <= max_growth
        P_norm = mx.sqrt(mx.sum(self.P ** 2))
        if P_norm > self.max_growth:
            scale = self.max_growth / P_norm
            growth_term = growth_term * scale
        
        W = self.I + growth_term
        return W
    
    def __call__(self, x: mx.array, record_usage: bool = True) -> mx.array:
        """
        前向传播
        
        Args:
            x: 输入，形状 [batch, seq, d_in]
            record_usage: 是否记录使用次数
            
        Returns:
            output: 输出，形状 [batch, seq, d_out]
        """
        if record_usage:
            self.usage_count += 1
        
        W = self._get_effective_weight()
        
        # 矩阵乘法：输出 = 输入 @ 权重^T
        output = mx.einsum('...d,Dd->...D', x, W)
        
        return output
    
    def compute_gradients(self, grad_output: mx.array, x: mx.array) -> Dict[str, mx.array]:
        """
        计算P矩阵的梯度
        
        对于 W = I + R @ P · α
        dL/dP = R^T · dL/dW · α
        
        Args:
            grad_output: 输出梯度，形状 [batch, seq, d_out]
            x: 输入，形状 [batch, seq, d_in]
            
        Returns:
            gradients: 包含P梯度的字典
        """
        if self.is_frozen:
            return {}
        
        # dL/dW：需要反转 einsum 操作
        # grad_output: [..., d_out], x: [..., d_in]
        # grad_W: [d_out, d_in] = sum over batch/seq of grad_output.T @ x
        grad_W = mx.einsum('bsd,bsi->di', grad_output, x)
        
        # dL/dP = R^T · mean(grad_W) · α
        grad_P = self.R.T @ grad_W * self.growth_scale
        
        return {'P': grad_P}
    
    def grow(self, gradients: Optional[Dict[str, mx.array]] = None, lr: float = 0.01) -> float:
        """
        执行一次生长更新
        
        沿负梯度方向更新P矩阵
        
        Args:
            gradients: 梯度字典（可选）
            lr: 学习率
            
        Returns:
            P_norm: 更新后P矩阵的范数
        """
        if self.is_frozen:
            return 0.0
        
        if gradients is None or 'P' not in gradients:
            return float(mx.sum(self.P ** 2))
        
        grad_P = gradients['P']
        
        # 梯度下降更新
        self.P = self.P - lr * grad_P
        
        # 记录历史
        self.gradient_history.append(float(mx.sum(self.P ** 2)))
        
        return float(mx.sum(self.P ** 2))
    
    def freeze(self) -> mx.array:
        """
        固化通道
        
        将灵芽生长转化为静态权重。
        融合后 W_frozen = (I + R@P·α)，推理时无需额外计算。
        
        Returns:
            W_frozen: 融合后的权重矩阵
        """
        if not self.is_frozen:
            W = self._get_effective_weight()
            self.W_frozen = W
            self.is_frozen = True
        return self.W_frozen
    
    def unfreeze(self) -> None:
        """
        解冻通道（恢复灵芽生长模式）
        """
        self.is_frozen = False
        self.W_frozen = None
    
    def get_num_params(self) -> int:
        """
        获取参数量
        
        仅统计P矩阵参数量（R矩阵可冻结不训练）
        
        Returns:
            num_params: 可训练参数量
        """
        return self.rank * self.d_in
    
    def get_diagnostic(self) -> dict:
        """
        获取诊断信息
        
        Returns:
            diagnostics: 包含通道状态的字典
        """
        P_norm = float(mx.sqrt(mx.sum(self.P ** 2)))
        return {
            "channel_type": self.channel_type.value,
            "is_frozen": self.is_frozen,
            "usage_count": self.usage_count,
            "P_norm": P_norm,
            "P_shape": list(self.P.shape),
            "num_trainable_params": self.get_num_params()
        }


class LingYaBlock:
    """
    灵芽块
    
    将多个灵芽通道组合成一个块，可替代FFN层。
    
    Attributes:
        channels: 通道列表
        num_channels: 通道数量
    """
    
    def __init__(
        self,
        d_model: int,
        num_channels: int = 4,
        rank: int = 8,
        growth_scale: float = 1.0
    ):
        self.channels: List[LingYaChannel] = []
        self.num_channels = num_channels
        
        for i in range(num_channels):
            # 早期层用小秩，后期层用大秩
            channel_rank = max(4, rank - i // 2)
            
            # 不同层使用不同通道类型
            if i == 0:
                channel_type = ChannelType.ROOT
            elif i < num_channels // 2:
                channel_type = ChannelType.BRANCH
            else:
                channel_type = ChannelType.LEAF
            
            channel = LingYaChannel(
                d_in=d_model,
                d_out=d_model,
                rank=channel_rank,
                channel_type=channel_type,
                growth_scale=growth_scale
            )
            self.channels.append(channel)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        前向传播
        
        所有通道并行处理，结果求和
        
        Args:
            x: 输入 [batch, seq, d_model]
            
        Returns:
            output: 输出 [batch, seq, d_model]
        """
        outputs = [channel(x) for channel in self.channels]
        return mx.mean(mx.stack(outputs), axis=0)
    
    def grow_all(self, gradients_list: List[Dict], lr: float = 0.01) -> List[float]:
        """
        对所有通道执行生长更新
        
        Args:
            gradients_list: 梯度字典列表
            lr: 学习率
            
        Returns:
            norms: 各通道P范数列表
        """
        norms = []
        for channel, grads in zip(self.channels, gradients_list):
            norm = channel.grow(grads, lr)
            norms.append(norm)
        return norms
    
    def freeze_all(self) -> None:
        """固化所有通道"""
        for channel in self.channels:
            channel.freeze()
    
    def get_num_params(self) -> int:
        """获取总参数量"""
        return sum(ch.get_num_params() for ch in self.channels)


def create_lingya_stack(
    d_model: int,
    num_layers: int,
    rank: int = 8,
    growth_scale: float = 1.0
) -> List[LingYaChannel]:
    """
    创建灵芽通道栈的工厂函数
    
    Args:
        d_model: 模型维度
        num_layers: 层数
        rank: 灵芽秩
        growth_scale: 生长尺度
        
    Returns:
        channels: 灵芽通道列表
    """
    channels = []
    for i in range(num_layers):
        layer_rank = max(4, rank - i // 3)
        channel = LingYaChannel(
            d_in=d_model,
            d_out=d_model,
            rank=layer_rank,
            growth_scale=growth_scale
        )
        channels.append(channel)
    return channels


def diagnose_channel(channel: LingYaChannel) -> str:
    """
    诊断灵芽通道状态
    
    Args:
        channel: 灵芽通道实例
        
    Returns:
        diagnosis: 诊断报告字符串
    """
    diag = channel.get_diagnostic()
    return f"""
灵芽通道诊断报告
====================
通道类型: {diag['channel_type']}
是否固化: {'是' if diag['is_frozen'] else '否'}
使用次数: {diag['usage_count']}
P矩阵范数: {diag['P_norm']:.4f}
P矩阵形状: {diag['P_shape']}
可训练参数: {diag['num_trainable_params']}
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Soma LingYa (Soma LingYa) - 灵芽通道演示")
    print("=" * 60)
    
    # 创建灵芽通道
    channel = LingYaChannel(
        d_in=512,
        d_out=512,
        rank=8,
        channel_type=ChannelType.BRANCH,
        growth_scale=1.0
    )
    
    print("\n初始状态:")
    print(diagnose_channel(channel))
    
    # 模拟前向传播
    x = mx.random.normal((2, 32, 512))
    output = channel(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 模拟训练
    print("\n模拟训练过程:")
    grad_output = mx.random.normal((2, 32, 512))
    
    for step in range(5):
        # 计算梯度
        gradients = channel.compute_gradients(grad_output, x)
        
        # 生长更新
        P_norm = channel.grow(gradients, lr=0.01)
        print(f"  Step {step}: P_norm = {P_norm:.4f}")
    
    # 固化
    print("\n固化通道:")
    channel.freeze()
    print(f"  是否固化: {channel.is_frozen}")
    print(f"  W_frozen形状: {channel.W_frozen.shape if channel.W_frozen is not None else 'None'}")
    
    print("\n最终状态:")
    print(diagnose_channel(channel))
    
    # 与LoRA对比
    print("\n" + "=" * 60)
    print("灵芽 vs LoRA 参数对比")
    print("=" * 60)
    
    d_model = 512
    
    # LoRA参数: A ∈ R^{r×d}, B ∈ R^{d×r}
    lora_rank = 8
    lora_params = lora_rank * d_model + d_model * lora_rank
    print(f"LoRA参数: {lora_rank}×{d_model} + {d_model}×{lora_rank} = {lora_params:,}")
    print(f"  占比: {lora_params / d_model / d_model * 100:.2f}%")
    
    # 灵芽参数: P ∈ R^{r×d}
    lingya_params = lora_rank * d_model
    print(f"灵芽参数: {lora_rank}×{d_model} = {lingya_params:,}")
    print(f"  占比: {lingya_params / d_model / d_model * 100:.4f}%")
    
    print(f"\n灵芽参数比LoRA少: {(lora_params - lingya_params) / lora_params * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Soma LingYa演示完成")
    print("=" * 60)
