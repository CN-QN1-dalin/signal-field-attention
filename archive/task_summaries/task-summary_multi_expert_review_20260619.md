# Task Summary: 太初五岳多专家全景审视

## Objective
用5位专家身份（架构师/数学家/实验科学家/编译器工程师/学术出版人）全面审视太初五岳项目，对照QN1、Dalinsoma及所有实验记录，找出系统性问题。

## 核心发现

### 1. 架构师视角：三层代码孤立
- SOMA X大脑（预测编码）/ MLX原型（SFA Engine）/ llama.cpp（骨架）无任何代码共享
- 40+文件含20+废弃迭代版本，无统一管理
- 终极融合三通道设计矛盾：RingBuffer保留信息 vs GuiYuan压缩信息 vs Nova理解信息

### 2. 数学家视角：证明存在方向错误
- Heritage定理1：不等号方向反了（log(γ)<0时未反转）
- Heritage Lemma1：比较不存在的"距离核"（标准注意力是全局的，无距离衰减）
- 远场EMA通道：α·F_t近似softmax(QK^T)V的数学基础薄弱

### 3. 实验科学家视角：50%+数据为模拟
- Heritage蒸馏：硬编码字典返回
- LingYa延迟：无实验条件标注
- 最新实测：终极融合α=0.2时PPL改善-0.74%（最佳），α≥1.0时爆炸

### 4. 编译器工程师视角：llama.cpp停留在骨架
- P0 bug：field_state名字匹配脆弱、单序列假设、跨设备拷贝缺失
- P0 bug：n_sfa_layers误用hparams.n_swa
- P0 bug：seq_cp/seq_rm索引比较反转
- SFA核心（RingBuffer/EMA场状态/远场融合）均未实现

### 5. 学术出版人视角：叙事不一致
- 5篇论文分别说SFA是"替代注意力/蒸馏方法/原生架构/收敛机制/PEFT方法"
- "Soma Labs"机构名称对独立研究者不诚实
- 投稿策略：先投Juejin积累反馈，再投arXiv/Workshop

## 优先级路线图
Phase 0 (1-2天): 删除废弃代码/标注模拟数据/修正数学错误
Phase 1 (1周): WikiText-2完整PPL测试/α=0.1 Cosine验证/修复field_state
Phase 2 (2-4周): 修复蒸馏框架/验证可学习enhancement
Phase 3 (1-2周): 合并论文/先投Juejin

## File
- `MULTI_EXPERT_REVIEW_2026-06-19.md` — 全景审视报告
