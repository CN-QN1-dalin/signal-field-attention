#pragma once

// ============================================================
// 🎯 SFA v7 → llama.cpp 集成桥接器 - 头文件
// ============================================================
// 功能：提供 llama.cpp 生命周期钩子，将 SFA v7 注入推理管线
// 状态：P0 Bug 已修复 (2026-06-22)
// ============================================================

#include "sfa_llama_cpp.h"
#include "ggml.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * llama.cpp 集成入口点
 * 在 llama_load_from_file() 之后调用
 * 
 * @param model 已加载的 llama_model 指针
 */
void sfa_llama_init(struct llama_model *model);

/**
 * 在 llama_decode() 之前调用（每个序列开始）
 * 
 * @param seq_id 当前序列 ID
 */
void sfa_llama_seq_start(int seq_id);

/**
 * 序列复制时调用 (Bug 3 修复)
 * 
 * @param dst_seq_id 目标序列 ID
 * @param src_seq_id 源序列 ID
 */
void sfa_llama_seq_copy(int dst_seq_id, int src_seq_id);

/**
 * 序列删除时调用 (Bug 3 修复)
 * 
 * @param seq_id 要删除的序列 ID
 */
void sfa_llama_seq_remove(int seq_id);

/**
 * 释放 llama_context 时调用
 */
void sfa_llama_free();

/**
 * 在 attention 层之后构建 SFA 增强
 * 返回增强后的 attention tensor
 * 
 * @param ctx0 ggml 临时上下文
 * @param attn_out 当前层的 attention 输出 [batch, seq_len, hidden]
 * @param layer_idx 当前层索引
 * @param seq_id 当前序列 ID
 * @param gf ggml 计算图
 * @return 增强后的 attention 输出 tensor
 */
ggml_tensor * sfa_llama_build_enhance(
    ggml_context *ctx0,
    ggml_tensor * attn_out,
    int layer_idx,
    int seq_id,
    const ggml_cgraph *gf);

/**
 * 启用/禁用 SFA 增强
 * 
 * @param on true=启用, false=禁用
 */
void sfa_llama_enable(bool on);

/**
 * 设置 alpha 参数（控制增强强度）
 * 
 * @param alpha alpha_base 值
 */
void sfa_llama_set_alpha(float alpha);

#ifdef __cplusplus
}
#endif
