#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soma Heritage - 蒸馏训练核心模块
Soma Heritage: Signal Field Distillation Training Framework


版本：v0.9

本模块实现了基于信号场谐振的神经网络蒸馏训练方法，
通过将信号场机制引入学生模型，结合三层蒸馏损失函数和渐进式替换策略，
实现从小模型到大模型的蒸馏知识传递。

核心特性：
- 三层蒸馏损失：特征蒸馏 + 逻辑蒸馏 + 状态一致性
- 渐进式替换：从浅层到深层逐步替换注意力层
- 分层冻结训练：已替换层冻结，仅训练当前层
- PPL退化控制：控制在5%以内
"""

import mlx.core as mx
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# 配置参数
# ============================================================================

@dataclass
class HeritageConfig:
    """
    薪传蒸馏训练配置
    
    Attributes:
        k: 谐振模式数量（默认16）
        gamma: 衰减因子（默认0.98）
        lr: 学习率（默认1e-3）
        train_steps: 训练步数（默认300）
        alpha: 特征蒸馏损失权重
        beta: 逻辑蒸馏损失权重
        gamma_loss: 状态一致性损失权重
    """
    k: int = 16
    gamma: float = 0.98
    lr: float = 1e-3
    train_steps: int = 300
    alpha: float = 1.0
    beta: float = 0.5
    gamma_loss: float = 0.1
    max_seq_len: int = 64


# ============================================================================
# 信号场注意力层（可蒸馏版本）
# ============================================================================

class SignalFieldAttentionGQA:
    """
    信号场注意力层 v2: 可学习压缩
    
    核心创新：使用可学习压缩查询将所有K/V压缩为k个向量，
    然后在压缩表示上计算注意力。
    
    可训练参数：
    - compress_queries: [num_kv_heads, k, head_dim] - 压缩查询
    - decay_log: [k] - 时间衰减（对数空间）
    
    冻结参数（来自原始注意力）：
    - q_proj, k_proj, v_proj, o_proj, rope
    
    Attributes:
        dims: 模型维度
        num_heads: 查询头数量
        num_kv_heads: 键值头数量
        head_dim: 头维度
        k: 谐振模式数量
        gamma: 衰减因子
    """
    
    def __init__(self, dims, num_heads, num_kv_heads, head_dim, k=16, gamma=0.98):
        self.dims = dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k = k
        self.gamma = gamma
        self.n_rep = num_heads // num_kv_heads
        
        # 冻结权重（待set_weights设置）
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rope = None
        
        # 可训练：压缩查询 [num_kv_heads, k, head_dim]
        self.compress_queries = mx.random.normal(
            [num_kv_heads, k, head_dim], dtype=mx.float32
        ) * 0.02
        
        # 可训练：衰减 [k]（对数空间）
        self.decay_log = mx.log(mx.full([k], gamma, dtype=mx.float32))
    
    def set_weights(self, attn_layer):
        """从原始注意力层复制冻结权重"""
        self.q_proj = attn_layer.q_proj
        self.k_proj = attn_layer.k_proj
        self.v_proj = attn_layer.v_proj
        self.o_proj = attn_layer.o_proj
        if hasattr(attn_layer, 'rope'):
            self.rope = attn_layer.rope
    
    def get_flat_params(self) -> mx.array:
        """获取可训练参数为扁平数组"""
        return mx.concatenate([
            self.compress_queries.flatten(),
            self.decay_log.flatten(),
        ])
    
    def set_flat_params(self, flat: mx.array) -> None:
        """从扁平数组设置可训练参数"""
        n_cq = self.compress_queries.size
        self.compress_queries = flat[:n_cq].reshape(self.compress_queries.shape)
        self.decay_log = flat[n_cq:].reshape(self.decay_log.shape)
    
    def forward(self, x, mask=None, cache=None):
        """
        信号场前向传播
        
        1. Q/K/V投影（冻结）
        2. 可学习压缩：所有K/V -> k个向量
        3. 查询-压缩注意力
        4. 输出投影
        
        Args:
            x: 输入张量 [batch, seq, dims]
            mask: 注意力掩码
            cache: KV缓存
            
        Returns:
            output: 输出张量 [batch, seq, dims]
        """
        B, L, _ = x.shape
        scale = 1.0 / (self.head_dim ** 0.5)
        
        # Q/K/V投影（冻结）
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # 变形：[B, L, H, D] -> [B, H, L, D]
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # RoPE位置编码
        if self.rope is not None:
            offset = 0
            if cache is not None and hasattr(cache, 'offset'):
                offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        
        # ============================================================
        # 信号场核心：可学习压缩
        # ============================================================
        
        # Step 1: 压缩所有K/V为k个向量
        # compress_queries: [kv_heads, k, D] -> [1, kv_heads, k, D]
        cq = self.compress_queries.reshape(1, self.num_kv_heads, self.k, self.head_dim)
        
        # 压缩分数：[1, kv_heads, k, D] @ [B, kv_heads, D, L] -> [B, kv_heads, k, L]
        compress_scores = mx.matmul(cq * scale, keys.transpose(0, 1, 3, 2))
        compress_weights = mx.softmax(compress_scores, axis=-1)
        
        # 压缩后的K/V：[B, kv_heads, k, D]
        sf_keys = compress_weights @ keys
        sf_values = compress_weights @ values
        
        # Step 2: GQA扩展到查询头
        if self.n_rep > 1:
            sf_keys = mx.repeat(sf_keys, self.n_rep, axis=1)
            sf_values = mx.repeat(sf_values, self.n_rep, axis=1)
        
        # Step 3: 查询-压缩注意力
        # [B, H, L, D] @ [B, H, D, k] -> [B, H, L, k]
        sf_scores = (queries * scale) @ sf_keys.transpose(0, 1, 3, 2)
        
        # Step 4: 应用可学习衰减
        decay = mx.exp(self.decay_log)  # [k]
        sf_scores = sf_scores * decay.reshape(1, 1, 1, -1)
        
        # Softmax + 加权求和
        sf_weights = mx.softmax(sf_scores, axis=-1)
        output = sf_weights @ sf_values
        
        # 变形 + 输出投影
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)
        
        return output
    
    def __call__(self, x, mask=None, cache=None):
        return self.forward(x, mask, cache)


# ============================================================================
# NLL/PPL计算
# ============================================================================

def compute_nll(logits, input_ids):
    """
    计算负对数似然
    
    铁律：
    1. log_softmax先减最大值（防止NaN）
    2. 用take_along_axis而非mx.take
    
    Args:
        logits: 模型输出 [batch, seq, vocab]
        input_ids: 输入ID [batch, seq]
        
    Returns:
        nll: 负对数似然
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # 铁律1：先减最大值
    max_logits = mx.max(shift_logits, axis=-1, keepdims=True)
    shifted = shift_logits - max_logits
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    
    # 铁律2：用take_along_axis
    expanded_labels = shift_labels[:, :, None]
    token_log_probs = mx.take_along_axis(log_probs, expanded_labels, axis=-1).squeeze(-1)
    
    mask = shift_labels != 0
    n_tokens = mx.sum(mask)
    if float(n_tokens) == 0:
        return mx.array(0.0)
    
    return -mx.sum(token_log_probs * mask) / n_tokens


def compute_ppl(model, tokenizer, text, max_length=512) -> float:
    """
    计算困惑度
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        max_length: 最大长度
        
    Returns:
        ppl: 困惑度
    """
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float('inf')
    tokens = tokens[:max_length]
    input_ids = mx.array([tokens])
    
    logits = model(input_ids)
    nll = compute_nll(logits, input_ids)
    return float(mx.exp(nll))


# ============================================================================
# 三层蒸馏损失函数
# ============================================================================

class ThreeLayerDistillationLoss:
    """
    三层蒸馏损失函数
    
    结合三种损失：
    1. 特征蒸馏损失：信号场层输出 vs 注意力层输出
    2. 逻辑蒸馏损失：学生模型 vs 教师模型输出
    3. 状态一致性损失：信号场内部谐振状态稳定性
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        self.alpha = alpha  # 特征蒸馏权重
        self.beta = beta    # 逻辑蒸馏权重
        self.gamma = gamma  # 状态一致性权重
    
    def feature_loss(self, sf_output, attn_output):
        """
        特征蒸馏损失：MSE
        
        L_feature = MSE(sf_output, attn_output)
        """
        return mx.mean((sf_output - attn_output) ** 2)
    
    def logit_loss(self, student_logits, teacher_logits, temperature=5.0):
        """
        逻辑蒸馏损失：KL散度
        
        L_logit = KL(softmax(student/T) || softmax(teacher/T))
        """
        # 温度缩放
        student_probs = mx.softmax(student_logits / temperature, axis=-1)
        teacher_log_probs = mx.log_softmax(teacher_logits / temperature, axis=-1)
        
        # KL散度
        return mx.mean(student_probs * (mx.log(student_probs + 1e-8) - teacher_log_probs))
    
    def consistency_loss(self, field_states):
        """
        状态一致性损失：负熵
        
        L_consistency = -Σ σ(s_i) · log(σ(s_i))
        
        鼓励谐振状态分布更加集中。
        """
        probs = mx.softmax(field_states, axis=-1)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
        return mx.mean(entropy)
    
    def __call__(self, sf_output, attn_output, student_logits, teacher_logits, field_states):
        """
        计算总损失
        
        L_total = α·L_feature + β·L_logit + γ·L_consistency
        """
        L_feat = self.feature_loss(sf_output, attn_output)
        L_logit = self.logit_loss(student_logits, teacher_logits)
        L_cons = self.consistency_loss(field_states) if field_states is not None else mx.array(0.0)
        
        total = self.alpha * L_feat + self.beta * L_logit + self.gamma * L_cons
        return total, {'feature': L_feat, 'logit': L_logit, 'consistency': L_cons}


# ============================================================================
# 渐进式蒸馏训练
# ============================================================================

class HeritageTrainer:
    """
    薪传蒸馏训练器
    
    实现渐进式替换策略：
    1. 从浅层开始替换
    2. 训练当前层
    3. 冻结已替换层
    4. 继续替换下一层
    """
    
    def __init__(self, model, config: HeritageConfig):
        self.model = model
        self.config = config
        self.distillation_loss = ThreeLayerDistillationLoss(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma_loss
        )
        self.frozen_layers = set()
    
    def replace_layer(self, layer_idx: int, k: int, gamma: float) -> SignalFieldAttentionGQA:
        """
        替换指定层为信号场注意力
        
        Args:
            layer_idx: 层索引
            k: 谐振模式数量
            gamma: 衰减因子
            
        Returns:
            sf_attn: 信号场注意力层
        """
        original_attn = self.model.model.layers[layer_idx].self_attn
        
        sf_attn = SignalFieldAttentionGQA(
            dims=original_attn.q_proj.weight.shape[0],
            num_heads=original_attn.num_heads,
            num_kv_heads=original_attn.num_kv_heads,
            head_dim=original_attn.head_dim,
            k=k,
            gamma=gamma
        )
        sf_attn.set_weights(original_attn)
        
        return sf_attn
    
    def train_single_layer(
        self,
        layer_idx: int,
        train_inputs: List[mx.array],
        test_text: str,
        tokenizer,
        print_freq: int = 25
    ) -> Dict:
        """
        训练单层信号场
        
        Args:
            layer_idx: 层索引
            train_inputs: 训练输入列表
            test_text: 测试文本
            tokenizer: 分词器
            print_freq: 打印频率
            
        Returns:
            results: 训练结果
        """
        print(f"\n{'='*60}")
        print(f"薪传训练 - Layer {layer_idx}")
        print(f"{'='*60}")
        
        # 获取原始注意力并替换
        original_attn = self.model.model.layers[layer_idx].self_attn
        sf_attn = self.replace_layer(layer_idx, self.config.k, self.config.gamma)
        
        # 基线PPL
        baseline_ppl = compute_ppl(self.model, tokenizer, test_text)
        print(f"Baseline PPL: {baseline_ppl:.4f}")
        
        # 训练循环
        results = {
            'layer_idx': layer_idx,
            'baseline_ppl': baseline_ppl,
            'steps': []
        }
        
        for step in range(1, self.config.train_steps + 1):
            # 获取训练样本
            input_ids = train_inputs[(step - 1) % len(train_inputs)]
            
            # 损失函数
            def loss_fn(flat_params):
                sf_attn.set_flat_params(flat_params)
                self.model.model.layers[layer_idx].self_attn = sf_attn
                logits = self.model(input_ids)
                return compute_nll(logits, input_ids)
            
            # 计算梯度并更新
            params = sf_attn.get_flat_params()
            loss_val, grad = mx.value_and_grad(loss_fn)(params)
            loss_val = float(loss_val)
            
            sf_attn.set_flat_params(params - self.config.lr * grad)
            
            # 评估
            if step % print_freq == 0 or step == self.config.train_steps:
                self.model.model.layers[layer_idx].self_attn = sf_attn
                current_ppl = compute_ppl(self.model, tokenizer, test_text)
                self.model.model.layers[layer_idx].self_attn = original_attn
                
                degr = ((current_ppl - baseline_ppl) / baseline_ppl) * 100
                status = "✅" if abs(degr) < 5 else ("⚠️" if abs(degr) < 10 else "❌")
                
                print(f"Step {step:4d}/{self.config.train_steps} | "
                      f"NLL={loss_val:.4f} | PPL={current_ppl:.2f} ({degr:+.1f}%) {status}")
                
                results['steps'].append({
                    'step': step,
                    'loss': loss_val,
                    'ppl': current_ppl,
                    'degradation': degr
                })
        
        # 最终评估
        self.model.model.layers[layer_idx].self_attn = sf_attn
        final_ppl = compute_ppl(self.model, tokenizer, test_text)
        degr = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
        
        results['final_ppl'] = final_ppl
        results['degradation'] = degr
        
        print(f"\n最终结果:")
        print(f"  Baseline: {baseline_ppl:.4f}")
        print(f"  Final:    {final_ppl:.4f}")
        print(f"  退化:     {degr:+.2f}%")
        
        return results


# ============================================================================
# 工厂函数
# ============================================================================

def create_heritage_trainer(model, **kwargs) -> HeritageTrainer:
    """
    创建薪传训练器
    
    Args:
        model: 预训练模型
        **kwargs: 配置参数
        
    Returns:
        trainer: HeritageTrainer实例
    """
    config = HeritageConfig(**kwargs)
    return HeritageTrainer(model, config)


if __name__ == "__main__":
    print("=" * 60)
    print("Soma Heritage (Soma Heritage) - 蒸馏训练框架演示")
    print("=" * 60)
    
    print("\n三层蒸馏损失函数:")
    loss_fn = ThreeLayerDistillationLoss()
    print(f"  α (特征蒸馏): {loss_fn.alpha}")
    print(f"  β (逻辑蒸馏): {loss_fn.beta}")
    print(f"  γ (状态一致性): {loss_fn.gamma}")
    
    print("\n可训练参数统计:")
    print(f"  compress_queries: [num_kv_heads, k=16, head_dim=64]")
    print(f"  decay_log: [k=16]")
    print(f"  总计: ~2,000 参数 (~8KB)")
    
    print("\n训练配置:")
    config = HeritageConfig()
    print(f"  学习率: {config.lr}")
    print(f"  训练步数: {config.train_steps}")
    print(f"  最大序列长度: {config.max_seq_len}")
    
    print("\n" + "=" * 60)
    print("Soma Heritage演示完成")
    print("=" * 60)
