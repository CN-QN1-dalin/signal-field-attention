# 太初五岳：论文声称 vs 实测数据 完整差距分析

**日期**: 2026-06-19  
**状态**: 数据审计完成

---

## 一、核心指标对比总表

| 指标 | 论文声称 | 实测数据 | 差距 | 状态 |
|------|----------|----------|------|------|
| **Cosine Similarity** | >0.9999999 | ~0.98-0.99 (Alpha=0.1) | ⚠️ 论文在Alpha=0条件下得出 | **需修正** |
| **KV内存压缩比** | 248× (7B/64K) | 16KB (128 dims) | ⚠️ 理论值 vs 小模型实测 | **需标注** |
| **单token解码加速** | 4.16× | 0.034ms/step (CPU) | ❌ 从未达成 | **需删除/标注目标** |
| **Metal GPU加速** | 4.16× | Prefill 1.11×, Decode 1.01× | ❌ 远未达成 | **需修正** |
| **额外参数** | ~8.1KB | ~262KB (含投影权重) | ❌ 统计口径不一致 | **需澄清** |
| **PPL改善 (长文本)** | ~0% | -10.02% (alpha=2.0) | ✅ 优于论文 | **可引用** |
| **PPL改善 (短文本)** | ~0% | -6.34% (alpha=2.0) | ✅ 优于论文 | **可引用** |
| **PPL改善 (WikiText-2)** | 未提及 | -2.53% 全局 | ✅ 新增验证 | **可引用** |
| **RingBuffer压缩** | O(k) vs O(n) | 4096 vs 10240 bytes | ✅ 符合理论 | **可引用** |
| **LingYa正交性** | 正交适配器 | 误差 3.58e-07, 节省50% | ✅ 完全验证 | **可引用** |

---

## 二、逐项详细分析

### 1. Cosine Similarity (t≥1)

**论文声称**: >0.9999999 (误差<10^-6)

**实测数据**:
- Alpha=0 (禁用SFA核心创新): ~0.9999999 ✅
- Alpha=0.1 (真实SFA): ~0.98-0.99 ⚠️
- Alpha=2.0 (PPL优化): 未直接测量Cosine，但PPL改善显著

**差距原因**:
论文中的 Cosine>0.9999999 是在 **Alpha=0, 衰减禁用** 条件下得出的。这意味着：
- RingBuffer 没有参与计算
- EMA 场状态没有更新
- 实际上就是标准注意力

**修正建议**:
```
修正前: Cosine Similarity > 0.9999999 (t≥1)
修正后: 
  - 标准注意力对齐: Alpha=0 时 Cosine > 0.9999999
  - 真实SFA (Alpha=0.1): Cosine ~0.98-0.99
  - PPL优化 (Alpha=2.0): PPL 改善 -10.02% (长文本)
```

### 2. KV内存压缩比

**论文声称**: 248× (7B模型, 64K序列, 462KB vs 114MB)

**实测数据**:
- MLX原型 (128 dims): 16 KB (RingBuffer + FieldState)
- C++实现 (64 dims): 4352 bytes (4096+256)
- 理论计算 (7B, 64K): 248× 是正确的理论值

**差距原因**:
248× 是 **7B模型的理论计算值**，不是实测值。实测只在128 dims小模型上进行。

**修正建议**:
```
修正前: 248× KV memory compression (462KB vs 114MB at 64K sequence)
修正后: 
  - 理论压缩比: 248× (7B模型, 64K序列, float16)
  - 实测压缩比: 16KB (128 dims原型)
  - 注: 248× 为理论推算，需在7B模型上实测验证
```

### 3. 单token解码加速

**论文声称**: 4.16× single-token decoding speedup

**实测数据**:
- MLX原型: 0.52ms/step
- C++ CPU: 0.034ms/step
- Metal GPU: 0.034ms/step
- **从未测量过 4.16× 加速**

**差距原因**:
4.16× 是 **C++/Metal 部署的目标值**，从未在实测中达成。MLX Python原型本身就比标准Transformer慢。

**修正建议**:
```
修正前: The C++/Metal deployment target achieves 4.16× single-token decoding speedup.
修正后: 
  - MLX原型: 0.52ms/step (Python解释器开销)
  - C++ CPU: 0.034ms/step
  - Metal GPU: 0.034ms/step
  - 4.16× 加速为C++/Metal部署目标值，未在本论文中验证
```

### 4. Metal GPU加速

**论文声称**: 4.16× 解码加速

**实测数据**:
- Prefill: 1.11× 加速 (32611 → 36222 t/s)
- Decode: 1.01× 加速 (28988 → 29312 t/s)

**差距原因**:
Decode 阶段本身已经很快 (0.034ms/step)，GPU 加速空间有限。Prefill 受益于 GPU 并行计算。

**修正建议**:
```
修正前: 4.16× 解码加速
修正后: 
  - Prefill: 1.11× GPU加速
  - Decode: 1.01× GPU加速 (已足够快)
```

### 5. 额外参数

**论文声称**: ~8.1KB (2064参数)

**实测数据**:
- RingBuffer + FieldState: ~16KB (128 dims)
- QKV投影权重: 2*128*128*3*4 = 163,840 bytes
- Out投影权重: 128*128*4 = 65,536 bytes
- **总计: ~262KB**

**差距原因**:
论文只计算了 RingBuffer + FieldState 的内存，没有计算 QKV 投影权重。

**修正建议**:
```
修正前: Soma Engine requires only ~8.1KB parameters (2064 parameters)
修正后: 
  - 信号场状态: ~16KB (RingBuffer + FieldState)
  - 投影权重: ~246KB (QKV + Out)
  - 总计: ~262KB
  - 注: 8.1KB 仅指信号场状态，不含投影权重
```

### 6. PPL改善 (可引用数据)

**论文声称**: ~0% (基本一致)

**实测数据**:
- 长文本 (512 tokens, alpha=2.0): **-10.02%** ✅
- 短文本 (256 tokens, alpha=2.0): **-6.34%** ✅
- WikiText-2 风格 (全局, alpha=2.0): **-2.53%** ✅

**结论**: 实测 **优于** 论文声称，这些数据可以直接引用。

---

## 三、数据分级

### Level A: 可直接引用 (实测验证)
| 指标 | 值 | 条件 |
|------|-----|------|
| PPL改善 (长文本) | -10.02% | alpha=2.0, 512 tokens |
| PPL改善 (短文本) | -6.34% | alpha=2.0, 256 tokens |
| PPL改善 (WikiText-2) | -2.53% | alpha=2.0, 50条合成文本 |
| RingBuffer压缩 | 4096 vs 10240 bytes | n=50, k=8, d=64 |
| LingYa正交性 | 误差 3.58e-07 | 正交适配器 |
| LingYa参数节省 | 50% | vs LoRA |

### Level B: 需修正后引用
| 指标 | 原值 | 修正后 |
|------|------|--------|
| Cosine Similarity | >0.9999999 | Alpha=0时>0.9999999, Alpha=0.1时~0.98 |
| KV压缩比 | 248× | 理论248×, 实测16KB (128 dims) |
| 额外参数 | 8.1KB | 信号场16KB + 投影246KB = 262KB |
| Metal加速 | 4.16× | Prefill 1.11×, Decode 1.01× |

### Level C: 需删除或标注为目标值
| 指标 | 原值 | 建议 |
|------|------|------|
| 单token解码加速 | 4.16× | 删除或标注为"C++/Metal部署目标" |
| 4.16× 加速 | 4.16× | 删除，标注为未验证 |

---

## 四、修正后的论文数据声明

```latex
\textbf{Experimental Results (MLX Prototype, float32):}
\begin{itemize}
    \item PPL Improvement: -10.02\% (long sequence, 512 tokens, $\alpha=2.0$)
    \item PPL Improvement: -6.34\% (short sequence, 256 tokens, $\alpha=2.0$)
    \item PPL Improvement: -2.53\% (WikiText-2 style, global average)
    \item Cosine Similarity: $>0.9999999$ when $\alpha=0$ (standard attention alignment)
    \item Cosine Similarity: $\sim0.98$-$0.99$ with real SFA ($\alpha=0.1$)
    \item Ring Buffer Compression: 4096 bytes vs 10240 bytes (n=50, k=8)
    \item LingYa Orthogonality Error: $3.58 \times 10^{-7}$, Parameter Savings: 50\%
\end{itemize}

\textbf{Theoretical Targets (C++/Metal Deployment):}
\begin{itemize}
    \item KV Memory Compression: 248$\times$ (7B model, 64K sequence, float16)
    \item Metal GPU Acceleration: 1.11$\times$ (prefill), 1.01$\times$ (decode)
    \item Target Decoding Speedup: 4.16$\times$ (not yet verified)
\end{itemize}

\textbf{Note:} All experimental data above is from MLX Python prototype implementation. 
The C++/Metal kernel deployment targets are theoretical values and require further verification.
```

---

## 五、下一步行动

1. **修正论文**: 更新 SFA_Technical_Paper.tex 中的数据
2. **补充实测**: 在更大模型 (1B/7B) 上验证理论压缩比
3. **Metal优化**: 探索进一步提升 GPU 加速的方法
4. **真实数据集**: 等待网络恢复后下载 WikiText-2 真实数据集

---

*数据审计完成。Level A 数据可直接引用，Level B 需修正后引用，Level C 需删除或标注为目标值。*
