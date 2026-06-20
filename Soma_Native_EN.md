# Soma Native

## Soma Native Architecture: A Signal Field-Native Neural Network Design

**作者: 贾大林 (Dalin Jia)**

**机构: Independent Researcher**

**版本: v3.0 (Strict Review Revised)**

## Abstract

本文提出Soma Native Architecture，一个从零设计的、完全基于信号场机制的原生神经网络架构。Soma Native用统一场块替代Transformer的注意力层和前馈网络层，采用双通道信号场机制实现O(k·n)计算复杂度和O(k)内存复杂度。

- **真实实验**: 正确性验证（Sim=1.0）、内存计算、FLOPs理论分析

- **模拟/理论数据**: 延迟估算、7B模型PPL（标注为TBD）、Homeostasis/GrowthTemporal实验

**关键词: Soma Native, 信号场, Transformer替代, O(k·n)复杂度**

## 1. Introduction

### Transformer

| 方案 | 计算复杂度 | 内存复杂度 | 增量更新 | 架构完整性 |
|---|---|---|---|---|
| FlashAttention | O(n²) | O(n) | ✗ | 局部优化 |
| Mamba (SSM) | O(n) | O(1) | ✓ | 局部替换 |
| Linear Attention | O(n) | O(n) | ✗ | 局部替换 |
| RetNet | O(n) | O(1) | ✓ | 局部替换 |
| H2O | O(n) | O(√n) | ✗ | 局部替换 |
| **Soma Native** | **O(k·n)** | **O(k)** | **✓** | **完整架构** |

### Soma Native

1. **统一场块**: 一个Soma Block同时替代Attention + FFN + LayerNorm

2. **双通道信号场**: 近场Ring Buffer+ 远场EMA State

## 2. Method

$$S(x, t) = \sum_{\tau < t} \gamma^{t-\tau} \cdot a(x, \tau)$$  **定义 2（双通道信号场）**:  $$\text{SF}(q, K, V) = \text{Attention}_{\text{near}}(q, K_{\text{ring}}, V_{\text{ring}}) + \alpha \cdot \text{Attention}_{\text{far}}(q, S_K, S_V)$$

### Unified Field Block

$$\text{SomaBlock}(x) = x + \text{Homeostasis}_2\left(\text{LingYaBlock}\left(\text{Homeostsis}_1\left(x + \text{SignalFieldLayer}(x)\right)\right)\right)$$  ### 2.3 稳态调节（Homeostasis）  $$\text{Homeostasis}(x)_i = x_i \cdot \rho_i$$

Homeostasis和GrowthTemporal是Soma Native架构的原创组件，但目前**没有实验数据验证其有效性**。它们的理论设计在本文中有描述，但实际效果需要在完整模型训练中验证。

## 3. Experiments

- **硬件**: Apple MacBook Pro M1 Pro, 16GB RAM

- **模型规模**: Small (256D, 6L, 4H), Medium (512D, 12L, 8H)

| 序列长度 | Soma Native内存 | Transformer内存 | 压缩比 |
|---|---|---|---|
| 512 | 462 KB | 14 MB | **31x** |
| 1,024 | 462 KB | 28 MB | **62x** |
| 2,048 | 462 KB | 56 MB | **123x** |
| 4,096 | 462 KB | 112 MB | **248x** |
| 8,192 | 462 KB | 224 MB | **496x** |
| 16,384 | 462 KB | 448 MB | **992x** |
| 65,536 | 462 KB | 896 MB | **1,986x** |

上述Transformer内存基于full attention（无GQA），float16精度。如果使用GQA（kv_heads=4），内存为上述值的1/7。

| 维度 $d$ | Transformer Attention | Soma SignalField | 节省 |
|---|---|---|---|
| 128 | ~65K | ~49K | **24%** |
| 256 | ~262K | ~197K | **25%** |
| 512 | ~1.05M | ~786K | **25%** |
| 768 | ~2.36M | ~1.77M | **25%** |

### FLOPs 

**表 7：SFA vs Standard Attention 的FLOPs对比**

| 指标 | SFA | Standard Attention | 差异 |
|---|---|---|---|
| 总FLOPs | 1.08×10⁹ | 1.61×10⁹ | **-32.8%** |
| 独特FLOPs (注意力) | 8.9×10⁶ | 5.4×10⁸ | **-98.4%** |

| 序列长度 | Standard Attention | SFA融合后 | 优势 |
|---|---|---|---|
| 64 | 12.5 μs | 11.8 μs | -5.6% |
| 1,024 | 45.0 μs | 11.8 μs | **3.8×** |
| 8,192 | 360.0 μs | 11.8 μs | **30.5×** |
| 65,536 | 2,880.0 μs | 11.8 μs | **244×** |

延迟数据来自理论公式估算，**非实测**。MLX原型实测显示Soma的prefill阶段比标准Attention慢（见Soma Engine论文）。

| 指标 | Soma Native | Transformer | 提升 |
|---|---|---|---|
| 推理内存 | 462 KB × 28 | 114 MB | **248x** |
| 单步解码 | TBD | TBD | — |
| PPL (验证集) | **TBD** | 6.66 | — |

## 4. Discussion

### Mamba

| 特性 | Mamba SSM | Soma Native |
|---|---|---|
| 信息交互机制 | 状态空间 | 信号场（近场+远场） |
| 计算复杂度 | O(n·d) | O(k·n·d) |
| 内存复杂度 | O(d) | O(k·d) |
| 增量更新 | ✓ | ✓ |
| 全局注意力 | ✗ | ✓（通过远场通道） |

1. **Homeostasis和GrowthTemporal无实验验证**: 这两个组件仅有理论设计，没有实际训练数据

4. **硬件适配**: 当前基于MLX，需要针对GPU/TPU进行算子优化

5. **7B模型PPL未验证**: 表9标注为TBD

1. 在完整模型上训练Soma Native，验证Homeostasis和GrowthTemporal的有效性

4. 在C++/Metal上实现完整Soma Native架构

## 5. Conclusion

本文提出Soma Native Architecture，一个从零设计的、完全基于信号场机制的原生神经网络架构。主要贡献：

1. **统一场块设计**: 一个Soma Block同时替代Transformer的Attention和FFN

2. **双通道信号场**: 近场Ring Buffer + 远场EMA State

3. **原生组件设计**: Homeostasis、GrowthTemporal、LingYaBlock

**重要声明: 本文的延迟数据为理论估算，Homeostasis和GrowthTemporal无实验验证，7B模型PPL标注为TBD。完整验证需要在实际模型训练中进行。**

## References

Attention Is All You Need. *NeurIPS*, 2017.

Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*, 2023.

Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML*, 2020.

FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*, 2022.

RetNet: Retentive Network: A Successor to Transformer for Large Language Models. *arXiv:2307.08621*, 2023.

6 Zhang B, Sennrich E. Root Mean Square Layer Normalization. *NeurIPS*, 2019.

RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*, 2021.

**联系作者: Dalin Jia (362118251@qq.com)**

**版本: Soma Native v3.0 (Strict Review Revised)**

