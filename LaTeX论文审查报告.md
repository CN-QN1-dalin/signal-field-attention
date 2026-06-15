# LaTeX 论文审查报告

> **审查对象**: SOMA-SFA LaTeX 初稿  
> **审查日期**: 2026-06-15  
> **审查基准**: 五篇学术论文 + 测试对比数据 + 实验记录

---

## 🔴 严重问题（必须修正）

### 1. 对比基线不实
- **LaTeX 声称**: "We compare our method with vanilla Transformer, StreamingLLM, SnapKV, and Mamba."
- **实际**: 我们**没有**对 StreamingLLM 和 SnapKV 做实验对比
- **风险**: 审稿人要求复现对比实验 → 无法提供数据 → 直接拒稿
- **修正**: 移除 "StreamingLLM, SnapKV"，改为 "vanilla Transformer and Mamba"

### 2. 模型名称不符
- **LaTeX 声称**: "LLaMA-7B and Qwen-7B"
- **实际**: 仅测试了 **Qwen2.5-7B-Instruct (4-bit)**
- **风险**: "LLaMA-7B 的结果"不存在，审稿人可能质疑数据真实性
- **修正**: 改为 "Qwen2.5-7B-Instruct (4-bit quantized)"

### 3. "248× KV memory compression" 缺少上下文
- **LaTeX 声称**: "up to 248× KV memory compression"
- **实际**: 这是 **7B 模型、64K 序列** 下的特定数据点
- **风险**: 没有说明模型规模和序列长度，"248×" 显得夸张
- **修正**: 加上条件 "on a 7B model with 64K sequence"

---

## 🟡 重要问题（建议修正）

### 4. 作者名和机构
- **LaTeX**: `\author{Dalin}`，无机构
- **建议**: 使用正式署名（贾大林 / Soma Team）+ 机构（Soma Labs）
- **原因**: 顶会论文需要完整作者信息

### 5. 硬件配置缺失
- **LaTeX**: 未提及硬件
- **建议**: 在 "Experimental Setup" 中补充 "Apple M1 Pro, 16GB RAM, MLX 0.31.2"
- **原因**: 4.16x 加速的 MLX 原型数据必须说明硬件

### 6. "without pre-training" 说法需核实
- **LaTeX 声称**: "Without pre-training, architectural modification, or token discarding"
- **实际**: SFA 确实不需要预训练（即插即用），但 LingYa 微调模块**需要训练**
- **修正**: 改为 "without full model pre-training" 或分别说明

### 7. 参考文献不完整
- **LaTeX 仅 5 条参考文献**，而实际论文中引用了 9+ 条
- **建议**: 合并五篇论文的全部参考文献为一个完整的 Bibliography
- **最低要求**: 补充 FlashAttention、LoRA、GQA、LinFormer、Performer、RetNet 等关键论文

---

## 🟢 正确部分（无需修正）

- ✅ 248× 压缩比（7B, 64K 序列）—— 数据准确
- ✅ 4.16× 加速（标注为 C++/Metal 部署目标）—— 数据准确
- ✅ Cosine Similarity > 0.9999999（t=1+）—— 数据准确
- ✅ 8.1KB 参数开销 —— 数据准确
- ✅ 462KB 固定内存 —— 数据准确
- ✅ 核心创新描述（近远场双通道、信号场建模）—— 与论文一致
- ✅ SOMA 五个模块描述 —— 与五篇论文一致
- ✅ 即插即用、兼容 Transformer —— 正确

---

## 📝 建议修正后的 "Experiments" 章节

```latex
\section{Experiments}
\subsection{Experimental Setup}
We conduct experiments on \textbf{Qwen2.5-7B-Instruct (4-bit quantized)} baseline models, 
running on Apple M1 Pro with 16GB RAM using MLX 0.31.2. 
We compare our method with vanilla Transformer and Mamba baselines.

\subsection{Main Results}
Experimental results show that SOMA-SFA maintains nearly consistent PPL and semantic 
similarity with the original model. The maximum KV cache compression ratio reaches 
\textbf{$248\times$ on a 7B model with 64K sequence} ($462$KB vs $114$MB), 
and the overall inference speed achieves \textbf{$4.16\times$ acceleration target} 
(with C++/Metal deployment).

In continuous dialogue tests, the memory remains constant without overflow.

\subsection{Ablation Study}
Ablation experiments verify the effectiveness of each core component...
```

---

## 🎯 上传前行动清单

| # | 项目 | 状态 |
|---|------|------|
| 1 | 修正对比基线（移除 StreamingLLM, SnapKV） | 🔴 必须 |
| 2 | 修正模型名称为 Qwen2.5-7B-Instruct (4-bit) | 🔴 必须 |
| 3 | 补充压缩比条件（7B, 64K） | 🔴 必须 |
| 4 | 补充作者信息和机构 | 🟡 建议 |
| 5 | 补充硬件配置 | 🟡 建议 |
| 6 | 修正 "without pre-training" 表述 | 🟡 建议 |
| 7 | 扩展参考文献至 15+ 条 | 🟡 建议 |
| 8 | 编译 LaTeX 检查公式和交叉引用 | 🟡 建议 |
| 9 | 生成 PDF 做最终通读 | 🟢 建议 |
