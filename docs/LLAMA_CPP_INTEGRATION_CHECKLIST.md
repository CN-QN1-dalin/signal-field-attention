# LLaMA.cpp SFA 对接前检查清单

> 目标：确保所有 SFA 代码可以无缝集成到 llama.cpp 最新代码库（commit 80452d6, master branch）

## P0: 硬编码路径修复

- [ ] `src/sfa/sfa_engine.h` — `#include "/tmp/llama.cpp/ggml/include/ggml.h"` → 改为相对路径 `"ggml.h"`
- [ ] `src/sfa/sfa_adapter.h` — `#include "/tmp/llama.cpp/ggml/include/ggml.h"` → 改为相对路径 `"ggml.h"`

## P0: 常量不一致修复

- [ ] `src/sfa/sfa_lockfree.h` 有**过时常量**：
  - `ENHANCEMENT_CLIP = 0.01f` → 应为 `0.5f`
  - `CROSS_DECAY = 0.7f` → 应为 `0.8f`
  - `ALPHA_BASE = 2.0f` → 应为 `0.1f`
- [ ] `sfa_llama_cpp.h::sfa_init()` 默认 `alpha_base=2.0f` → 应为 `0.1f`

## P0: ggml API 兼容性

- [ ] `src/sfa/sfa_llama_bridge.cpp` 使用 `ggml_new_i32(ctx, value)` — **此函数未在 ggml.h 公开声明**
  - 替代方案：使用 `ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1)` + 手动赋值
- [ ] `src/sfa/sfa_llama_cpp.cpp` 同样使用 `ggml_new_i32` — 需要修复
- [ ] 确认所有使用的 ggml API 在 llama.cpp 80452d6 版本中存在且签名匹配

## P1: 编译系统集成

- [ ] 创建 `src/sfa/CMakeLists.txt` 或将 SFA 源码纳入 llama.cpp 的 ggml/src 或 src/
- [ ] 确认 `llama-model.h` 和 `llama-graph.h` 的 include 路径正确
- [ ] 确认 `ggml-backend.h` 的 include 路径正确

## P1: 生命周期钩子映射

- [ ] 确定 SFA 钩子在 llama.cpp 中的插入点：
  - `llama_model_load()` → `sfa_llama_init()` ✅ 已有
  - `llama_decode()` 前 → `sfa_llama_seq_start()` ✅ 已有
  - `llama_seq_cp()` → `sfa_llama_seq_copy()` ✅ 已有
  - `llama_seq_rm()` → `sfa_llama_seq_remove()` ✅ 已有
  - `llama_free()` → `sfa_llama_free()` ✅ 已有
- [ ] 确认 `llama-graph.h` 提供了 `llama_graph` 回调机制（用于 attention 后注入）

## P1: 架构适配

- [ ] `sfa_llama_bridge.cpp` 中 `ggml_mean` 对 `[batch, seq_len, hidden]` 的行为需要验证
  - ggml_mean 沿 axis 0 求均值 → 得到 `[1, hidden]`，但我们需要 `[hidden]` 或 broadcast
- [ ] `build_sfa_enhance_for_layer` 中使用 `std::memcpy(tensor->data, ...)` 可能不安全
  - ggml 张量的 data 指针在图构建阶段可能未分配
  - 应使用 `ggml_dup_tensor` + `ggml_set_f32` 或创建 view tensor

## P2: Metal 内核编译

- [ ] 安装/修复 Xcode Command Line Tools：`xcode-select --install`
- [ ] 编译测试：`xcrun -sdk macosx metal -c src/sfa/sfa_kernel.metal -o /tmp/sfa_kernel.metallib`
- [ ] 确认 `sfa_kernel.metal` 中的 `sfa_enhance_compute` kernel 参数与 ggml tensor 布局匹配

## P2: 文档同步

- [ ] `TECHNICAL_REPORT.md` 中所有常量引用应与校准后值一致
- [ ] `INTEGRATION_GUIDE.md` 应包含完整的编译和集成步骤
- [ ] 更新 `OPEN_SOURCE.md` Quick Start 指向正确的集成方式

## P3: 清理

- [ ] 删除 `01_soma_engine/__pycache__/`
- [ ] 清理根目录临时文件
- [ ] 确认 `.gitignore` 不包含 `src/sfa/`（不应忽略 SFA 源码）
