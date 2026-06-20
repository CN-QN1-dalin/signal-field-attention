# QN1 幻化引擎 (Dalin Soma) - 基于信号场注意力的神经网络推理加速框架

## 项目简介

QN1幻化引擎（项目代号：Dalin Soma）是一个创新的神经网络推理加速框架，通过信号场注意力（SFA）机制，结合Metal GPU原生优化和智能量化技术，实现了在资源受限环境下的高效推理。

## 核心特性

- **SFA三通道架构** - RingBuffer + EMA Field + Semantic Pool
- **零侵入集成** - 万能转接头设计，无需修改模型代码
- **Metal GPU优化** - Apple Silicon原生加速
- **智能量化** - Q4_0量化加速，性能提升150%

## 性能数据

| 配置 | Prefill | Generate | 内存 |
|------|---------|----------|------|
| F16基线 | 81 t/s | 89 t/s | 948 MB |
| Q4_0 + SFA | 202 t/s | 215 t/s | 336 MB |

## 目录结构

```
dalinsoma/
├── src/
│   └── sfa/          # SFA核心代码
├── docs/             # 技术文档
├── tests/            # 测试用例
├── models/           # 模型文件
└── examples/         # 示例代码
```

## 许可证

MIT License

## 联系方式

- 作者：大林素玛团队
- 版本：v4.0
- 日期：2026年6月20日
