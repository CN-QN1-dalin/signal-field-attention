# Dalin Soma — 五岳推理加速框架 (Rust)

**QN1 幻化引擎 · 大林素玛**  
**Signal Field Attention (SFA) Inference Acceleration Framework**

## 概述

Dalin Soma 是基于信号场注意力（SFA）的零侵入推理加速框架的纯 Rust 实现。  
五岳架构覆盖从注意力加速到蒸馏训练的完整链路，**零外部依赖**，论文-代码严格一致。

## 五岳架构

| 模块 | 代号 | 功能 | 论文对应 |
|------|------|------|----------|
| 东岳 | Soma Engine | 信号场双通道注意力 (Ring Buffer + EMA Field) | §1.2, §2.2.2, §3.3 |
| 南岳 | Soma LingYa | 参数高效微调 (脚手架+生长矩阵, 比LoRA省50%) | §2.3 |
| 西岳 | Soma Native | 零设计神经网络 (SomaBlock + Homeostasis) | §2.4 |
| 北岳 | Soma Convergence | O(1)增量推理 (谐振模式) | §2.5 |
| 中岳 | Soma Heritage | 蒸馏训练 (S型层分配, 渐进替换) | §2.6 |

## 核心特性

- **零外部依赖**: 纯 Rust，无需 ndarray/candle/metal-rs
- **论文-代码一致**: 每个模块标注对应论文章节/表格
- **完整测试覆盖**: 32 个单元测试，覆盖所有核心模块
- **GQA 支持**: 天然兼容分组查询注意力
- **O(1) 解码**: 每步恒定延迟，与序列长度无关

## 快速开始

```bash
# 编译
cargo build --release

# 运行演示 (输出所有论文核心数据)
cargo run --release

# 运行测试
cargo test
```

## 配置参数

### Qwen2.5-0.5B
- dims=896, heads=14, kv_heads=2, head_dim=64
- window_size=16, γ=0.98, α=0.1
- 每层额外参数: 2,064 (~8.0 KB)
- 4096 序列压缩比: ~1,687× (GQA)

### Qwen2.5-7B
- dims=3584, heads=28, kv_heads=4, head_dim=128
- 每层额外参数: 8,208 (~32.0 KB)
- 28 层总计: ~229 KB (占 7B 模型 0.003%)

## 性能数据

| 指标 | 数据 |
|------|------|
| Decode 延迟 | 0.41ms/步 (恒定, CV<2%) |
| LingYa vs LoRA | 节省 50% 参数 |
| 华岳层分配 | 71% 替换 + 29% 保留 |
| 内存压缩 | 与序列长度无关 |

## 项目结构

```
dalin-soma-rust/
├── Cargo.toml          # MIT 协议
├── README.md
├── src/
│   ├── lib.rs           # 五岳模块导出
│   ├── main.rs          # 演示程序
│   ├── config.rs        # 模型配置 + 压缩比计算
│   ├── math.rs          # softmax/EMA/cosine/衰减
│   ├── engine.rs        # 东岳: SFA Engine
│   ├── lingya.rs        # 南岳: LingYa PEFT
│   ├── native.rs        # 西岳: SomaBlock
│   ├── convergence.rs   # 北岳: 谐振推理
│   └── heritage.rs      # 中岳: 蒸馏训练
```

## 许可证

MIT License — 欢迎使用和修改

## 联系方式

- **GitHub**: github.com/CN-QN1-dalin/signal-field-attention
- **团队**: QN1 幻化引擎团队
