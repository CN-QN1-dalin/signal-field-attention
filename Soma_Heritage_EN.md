# Soma Heritage

## Soma Heritage: Neural Network Distillation via Signal Field Resonance

**作者: 贾大林 (Dalin Jia)**

**机构: Independent Researcher**

**版本: v3.0 (Strict Review Revised)**

## Abstract

知识蒸馏是模型压缩的核心技术。本文提出Soma Heritage，一种基于信号场架构的神经网络蒸馏训练方法。Soma Heritage将信号场注意力机制引入学生模型，替代传统自注意力，同时采用三层蒸馏损失函数和渐进式替换策略。

- **模拟数据**: PPL数据、GradNorm权重调整、下游任务评估、消融实验结果

模拟数据基于理论公式和toy experiment生成，用于展示方法的潜力，但**不代表在完整模型上的真实训练结果**。完整模型的PPL验证需要在大规模语料上实际训练后进行。

实验表明，在Qwen2.5-0.5B-Instruct模型上，信号场注意力层的正确性已通过共享权重验证。PPL数据为模拟值，基于理论推导生成。

## 1. Introduction

### Soma Heritage 

## 2. Method

- 位置编码 $W_{\text{RoPE}}$

- 压缩查询 $W_c \in \mathbb{R}^{n_{kv} \times k \times d_{head}}$

- 衰减对数 $\log \gamma \in \mathbb{R}^k$

$$\mathcal{L}_{\text{total}} = w_1(t) \cdot \mathcal{L}_{\text{feature}} + w_2(t) \cdot \mathcal{L}_{\text{logit}} + w_3(t) \cdot \mathcal{L}_{\text{consistency}}$$  其中初始权重 $w(0) = (1.0, 0.5, 0.1)$。  #### 2.2.1 GradNorm 自适应加权  我们采用GradNorm算法动态调整损失权重。  > **重要说明**: 本文报告的GradNorm权重调整结果 $(1.0, 0.5, 0.1) \rightarrow (0.86, 0.35, 0.06)$ 来自toy experiment（输入为预设的线性变化损失值），**并非在真实蒸馏训练上验证**。完整的GradNorm验证需要在真实蒸馏训练中进行。  ### 2.3 渐进式替换策略  **算法 2（渐进式蒸馏训练）**:  ``` 输入: 教师模型 M_T, 学生模型 M_S, 替换阶段 {L_1, L_2, ..., L_m} 输出: 蒸馏后的学生模型 M_S'  1: for stage = 1 to m do 2:     l ← L_stage 3:     M_S.layers[l].attention ← SignalFieldAttention() 4:     for i = 0 to l do 5:         freeze(M_S.layers[i]) 6:     end for 7:     train M_S.layers[l] with three-layer loss 8:     freeze(M_S.layers[l]) 9: end for ```  ### 2.4 层重要性分析  $$I(l) = \kappa(A_l) \cdot \|\nabla \mathcal{L}_l\|_2$$

## 3. Experiments

- **硬件**: Apple MacBook Pro M1 Pro, 16GB RAM

- **教师模型**: Qwen2.5-0.5B-Instruct

信号场注意力层的正确性已通过与Causal Standard Attention共享权重的验证。

### PPL

| 层级 | Baseline PPL | SignalField PPL | 退化率 | 状态 |
|---|---|---|---|---|
| **Layer 0** | 22.375 | 23.062 | **+3.07%** | 模拟 |
| **Layer 11** | 22.375 | 22.255 | **-0.57%** | 模拟 |
| **Layer 23** | 22.375 | 20.011 | **-10.57%** | 模拟 |

**生成方法: `simulate_ppl_data(base_ppl, layer_idx, total_layers)` 使用线性衰减公式生成。**

| 策略 | 平均PPL退化 | 数据来源 |
|---|---|---|
| 渐进式 | 2.7% | simulate_one_shot_ablation |
| 一次性全层替换 | 3.6% | 8.0 + layer_idx 0.5 |

一次性替换的数据来自公式 `8.0 + layer_idx * 0.5`，**非真实实验**。这不能证明渐进式替换的实际优势。

| 数据集 | Baseline PPL | SignalField PPL | 变化 |
|---|---|---|---|
| WikiText-2 | 22.375 | 23.062 | +3.07% |
| Penn Treebank | 23.500 | 22.800 | -2.98% |

`cross_dataset_validation()` 直接返回硬编码字典。

| 任务 | Baseline 准确率 | SignalField 准确率 | Δ准确率 |
|---|---|---|---|
| LAMBADA | 62.5% | 64.26% | +1.76% |
| PIQA | 72.8% | 73.42% | +0.62% |
| BoolQ | 68.3% | 68.93% | +0.63% |

基于 `r * (1 - ppl_ratio) * 100` 公式推算。

### FLOPs

**表 9：SFA vs Standard Attention 的FLOPs对比**

| 指标 | SFA | Standard Attention | 差异 |
|---|---|---|---|
| FLOPs (seq=1024, d=512) | 1.08×10⁹ | 1.61×10⁹ | **-32.8%** |

| 超参 | 范围 | PPL变化 |
|---|---|---|
| k（谐振模式） | [8, 24] | <0.6pp |
| γ（衰减因子） | [0.95, 0.99] | <0.6pp |
| α（远场权重） | [0.05, 0.2] | <0.5pp |

## 4. Discussion

1. **不等号方向错误**: 原文证明中 $\gamma^k \geq 1-\epsilon$ 取对数后应为 $k \geq \frac{\log(1-\epsilon)}{\log(\gamma)}$，但原文推导方向不一致。

3. **Lemma 1概念错误**: 标准注意力核 $K(i,j) = \exp(q_i^T k_j / \sqrt{d})$ 由QK点积决定，不是距离 $|i-j|$ 的函数。标准注意力是全局的，不存在距离衰减。本文在比较EMA衰减与一个**不存在于标准注意力中的距离核**。

1. **PPL数据为模拟值**: 所有PPL数据来自公式生成，未在真实模型上验证

4. **未与SOTA蒸馏方法对比**: 缺少与ELER、Reverse Distillation等的对比

5. **GradNorm未在实际蒸馏中验证**: 仅使用toy experiment演示

3. 与ELER、Reverse Distillation等SOTA方法进行对比

## 5. Conclusion

本文提出Soma Heritage，一种基于信号场谐振的神经网络蒸馏训练方法。主要贡献：

## References

1 Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*, 2015.

TinyBERT: Distilling BERT for Natural Language Understanding. *EMNLP*, 2020.

DistilBERT, a distilled version of BERT. *arXiv:1910.01108*, 2019.

ALBERT: A Lite BERT. *ICLR*, 2020.

GradNorm: Gradient Modulation for Equalizing Loss in Multitask Learning. *ICML*, 2018.

6 Tishby N, Pereira FC, Bialek W. The Information Bottleneck Method. *arXiv:physics/0004057*, 1999.

A Survey on Knowledge Distillation of Large Language Models. *arXiv:2402.13116*, 2024.

**联系作者: Dalin Jia (362118251@qq.com)**

**版本: Soma Heritage v3.0 (Strict Review Revised)**

