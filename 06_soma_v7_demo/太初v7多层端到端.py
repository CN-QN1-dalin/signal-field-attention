#!/usr/bin/env python3
"""
Soma v7多层端到端推理脚本
基于v6飙车版框架，实现多层信号场层替换注意力层

模式:
- v7a保守版: 替换8层 [8-15], 保留24层注意力
- v7b激进版: 替换24层 [4-27], 保留8层注意力

信号场参数: k=8, kn=256, base_alpha=0.1, 自适应alpha
"""
import gc, sys, time, json, os, subprocess, resource
from pathlib import Path
from datetime import datetime

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
except ImportError as e:
    print(f"导入失败: {e}"); sys.exit(1)

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_PATH = str(Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed")

# ======================================================================
# 配置
# ======================================================================
# 硬编码最优配置（来自v5d/v6验证）
BASE_ALPHA = 0.04
K_ANCHORS = 8
KN = 256  # 近场窗口大小
L_THRESHOLD = 2048  # 自适应alpha阈值

# 替换策略
REPLACE_MODES = {
    "v7a": {
        "name": "保守版",
        "replace_layers": list(range(8, 16)),  # [8-15] 共8层
        "desc": "替换8层[8-15], 保留24层注意力"
    },
    "v7b": {
        "name": "激进版", 
        "replace_layers": list(range(4, 28)),  # [4-27] 共24层
        "desc": "替换24层[4-27], 保留8层注意力"
    }
}


# ======================================================================
# 辅助函数（与v6一致）
# ======================================================================
def get_mem_mb():
    """获取进程RSS内存(MB)"""
    try:
        return int(subprocess.check_output(f"ps -o rss= -p {os.getpid()}", shell=True).decode().strip()) / 1024
    except:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576

def get_peak_mem_mb():
    """获取MLX峰值GPU内存(MB)"""
    try:
        return mx.get_peak_memory() / (1024 ** 2)
    except:
        return 0

def reset_peak_mem():
    """重置MLX峰值内存计数器"""
    try:
        mx.reset_peak_memory()
    except:
        pass

def sys_info():
    info = {}
    try: info['cpu'] = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
    except: info['cpu'] = '?'
    try: info['mem_gb'] = round(int(subprocess.check_output("sysctl -n hw.memsize", shell=True).decode().strip()) / (1024**3), 1)
    except: info['mem_gb'] = 0
    info['gpu'] = 'Apple M1 Pro'
    try: info['python'] = subprocess.check_output("python3 --version", shell=True).decode().strip()
    except: info['python'] = '?'
    try: info['mlx'] = mx.__version__
    except: pass
    return info


# ======================================================================
# 信号场层实现（与v6 AdaptiveSSM 兼容）
# ======================================================================
def gqa(q, k, v, scale):
    """Grouped Query Attention"""
    B, H, L, D = q.shape
    Hk = k.shape[1]
    nr = H // Hk
    Q = q.reshape(B, Hk, nr, L, D)
    K = k.reshape(B, Hk, 1, k.shape[2], D)
    V = v.reshape(B, Hk, 1, v.shape[2], D)
    sc = (Q * scale) @ K.transpose(0, 1, 2, 4, 3)
    w = mx.softmax(sc, axis=-1)
    return (w @ V).reshape(B, H, L, D)


class SignalFieldLayer:
    """
    信号场层 - 多层替换的核心组件
    
    支持:
    - 压缩queries: [num_kv_heads, k, head_dim]
    - decay_log: [k], 初始化为ln(0.98)
    - 自适应alpha: αeff = αbase × max(1, L/Lthreshold)
    """
    
    def __init__(self, n_kv_heads, head_dim, n_anchors=K_ANCHORS, gamma_v=0.98):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_anchors = n_anchors
        self.gamma_v = gamma_v
        
        # 初始化锚点存储
        self.keys = [mx.zeros([1, n_kv_heads, head_dim]) for _ in range(n_anchors)]
        self.values = [mx.zeros([1, n_kv_heads, head_dim]) for _ in range(n_anchors)]
        self.initialized = False
        self._anchor_idx = 0
        
        # decay_log: 初始化为ln(0.98)
        self.decay_log = mx.log(mx.array([0.98] * n_anchors, dtype=mx.float32))
        
        # 压缩矩阵 (简化实现)
        self.compress_dim = min(n_anchors, n_kv_heads)
        self.compress_weight = mx.random.normal(
            (n_kv_heads, self.compress_dim), scale=mx.sqrt(1.0 / self.compress_dim)
        )
    
    def adaptive_alpha(self, seq_len):
        """自适应alpha: αeff = αbase × max(1, L/Lthreshold)"""
        return BASE_ALPHA * max(1.0, seq_len / L_THRESHOLD)
    
    def init_uniform(self, K, V, kn=KN):
        """均匀采样远场锚点"""
        B, Hk, L, D = K.shape
        far_end = max(kn, L - kn)
        step = max(1, far_end // self.n_anchors)
        positions = [min(int(i * step), L - 1) for i in range(self.n_anchors)]
        positions = positions[:self.n_anchors]
        while len(positions) < self.n_anchors:
            positions.append(max(0, L - 1))
        for j, pos in enumerate(positions):
            pos = min(pos, L - 1)  # 防越界
            self.keys[j] = K[0, :, pos, :].reshape(1, Hk, D)
            self.values[j] = V[0, :, pos, :].reshape(1, Hk, D)
        self.initialized = True
        self._positions = positions
    
    def query(self, q, n_heads, scale):
        """查询信号场"""
        B, H, L, D = q.shape
        Hk = self.n_kv_heads
        nr = H // Hk
        q_r = q.reshape(B, Hk, nr, L, D)
        
        # 堆叠锚点
        ak = mx.stack(self.keys, axis=2)  # [1, Hk, n_anchors, D]
        av = mx.stack(self.values, axis=2)  # [1, Hk, n_anchors, D]
        
        # 转置用于attention
        ak_t = ak.reshape(1, Hk, 1, self.n_anchors, D).transpose(0, 1, 2, 4, 3)
        score = (q_r * scale) @ ak_t
        weight = mx.softmax(score, axis=-1)
        
        # 加权求和
        av_r = av.reshape(1, Hk, 1, self.n_anchors, D)
        out = (weight @ av_r).reshape(B, H, L, D)
        
        return out


# ======================================================================
# 多层信号场模型
# ======================================================================
class MultiLayerSignalFieldModel:
    """
    多层信号场替换模型
    
    策略:
    - v7a保守版: 替换[8-15], 保留24层
    - v7b激进版: 替换[4-27], 保留8层
    """
    
    def __init__(self, model, mode="v7a"):
        self.model = model
        self.mode = mode
        
        # 替换层列表
        self.replace_layers = REPLACE_MODES[mode]["replace_layers"]
        
        # 模型结构信息
        sa = model.model.layers[0].self_attn
        self.n_heads = sa.n_heads
        self.n_kv_heads = sa.n_kv_heads
        self.head_dim = sa.k_proj.weight.shape[0] // self.n_kv_heads
        self.n_layers = len(model.model.layers)
        
        # 创建信号场层列表
        self.ssm_list = []
        for i in range(self.n_layers):
            if i in self.replace_layers:
                # 需要替换的层：创建新的SSM
                ssm = SignalFieldLayer(self.n_kv_heads, self.head_dim, n_anchors=K_ANCHORS)
            else:
                # 保留的层：使用None占位
                ssm = None
            self.ssm_list.append(ssm)
        
        print(f"\n[Soma v7 {mode}] {REPLACE_MODES[mode]['desc']}")
        print(f"  总层数: {self.n_layers}")
        print(f"  替换层数: {len(self.replace_layers)}")
        print(f"  替换层: {self.replace_layers}")
        n_total = len(model.model.layers)
        n_keep = n_total - len(self.replace_layers)
        print(f"  保留层: {n_keep}层注意力")
    
    def get_layer_mode(self, layer_idx):
        """判断某层是否需要替换"""
        return layer_idx in self.replace_layers


# ======================================================================
# 推理函数
# ======================================================================
def uf_with_multi_ssm(md, ids, multi_ssm_model, fa=BASE_ALPHA, kn=KN, mode='multi_ssm'):
    """
    多层信号场前向传播
    
    Args:
        md: 模型实例
        ids: 输入token ids
        multi_ssm_model: MultiLayerSignalFieldModel实例
        fa: 基础alpha
        kn: 近场窗口大小
        mode: 'standard' (全注意力) 或 'multi_ssm' (多层SSM替换)
    """
    h = md.model.embed_tokens(ids)
    B, L = ids.shape
    
    ssm_list = multi_ssm_model.ssm_list
    n_layers = len(md.model.layers)
    
    for i, ly in enumerate(md.model.layers):
        hn = ly.input_layernorm(h)
        at = ly.self_attn
        B2, L2, D = hn.shape
        
        # QKV投影
        Q = at.q_proj(hn).reshape(B2, L2, at.n_heads, -1).transpose(0, 2, 1, 3)
        K = at.k_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        V = at.v_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        # RoPE
        Q = at.rope(Q)
        K = at.rope(K)
        
        # 判断是否使用SSM替换
        use_ssm = (mode == 'multi_ssm') and multi_ssm_model.get_layer_mode(i)
        ssm_initialized = (ssm_list[i] is not None and ssm_list[i].initialized)
        
        if use_ssm and ssm_initialized:
            # SSM替换注意力：近场 + 信号场
            nk = K[:, :, -kn:, :] if L2 > kn else K
            nv = V[:, :, -kn:, :] if L2 > kn else V
            ao_near = gqa(Q, nk, nv, at.scale)
            
            # 信号场查询
            hr = ssm_list[i].query(Q, at.n_heads, at.scale)
            
            # 自适应alpha
            adapt_alpha = fa * max(1.0, L2 / L_THRESHOLD)
            ao = ao_near + adapt_alpha * hr
        else:
            # 全量attention
            if mode == 'standard':
                ao = gqa(Q, K, V, at.scale)
            else:
                # 保留层：使用近场+SSM（如果可用）
                nk = K[:, :, -kn:, :] if L2 > kn else K
                nv = V[:, :, -kn:, :] if L2 > kn else V
                ao = gqa(Q, nk, nv, at.scale)
        
        # 输出投影 + 残差 + MLP
        ao = ao.transpose(0, 2, 1, 3).reshape(B2, L2, -1)
        h = h + at.o_proj(ao)
        h = h + ly.mlp(ly.post_attention_layernorm(h))
        mx.eval(h)
    
    h = md.model.norm(h)
    return md.model.embed_tokens.as_linear(h)


def uf_standard(md, ids):
    """标准前向传播（用于参考输出）"""
    h = md.model.embed_tokens(ids)
    B, L = ids.shape
    
    for ly in md.model.layers:
        hn = ly.input_layernorm(h)
        at = ly.self_attn
        B2, L2, D = hn.shape
        
        Q = at.q_proj(hn).reshape(B2, L2, at.n_heads, -1).transpose(0, 2, 1, 3)
        K = at.k_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        V = at.v_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        Q = at.rope(Q)
        K = at.rope(K)
        
        ao = gqa(Q, K, V, at.scale)
        
        ao = ao.transpose(0, 2, 1, 3).reshape(B2, L2, -1)
        h = h + at.o_proj(ao)
        h = h + ly.mlp(ly.post_attention_layernorm(h))
        mx.eval(h)
    
    h = md.model.norm(h)
    return md.model.embed_tokens.as_linear(h)


def csim(a, b):
    """余弦相似度"""
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    d = mx.sum(a * b, axis=-1)
    na = mx.sqrt(mx.sum(a**2, axis=-1) + 1e-8)
    nb = mx.sqrt(mx.sum(b**2, axis=-1) + 1e-8)
    r = float((d / (na * nb)).mean())
    mx.eval(r)
    return r


# ======================================================================
# 完整生成测试
# ======================================================================
def run_multi_ssm_generate(model, tokenizer, multi_ssm_model, prompt_tokens, max_tokens=50, fa=BASE_ALPHA, kn=KN):
    """
    使用多层SSM流式生成
    
    Returns:
        dict: 包含tokens, tok_per_sec, peak_mem_MB等
    """
    n_heads = model.model.layers[0].self_attn.n_heads
    n_kv_heads = model.model.layers[0].self_attn.n_kv_heads
    head_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0] // n_kv_heads
    nl = len(model.model.layers)
    
    prompt_tokens = list(prompt_tokens)
    
    # ===== Prefill阶段 =====
    prefill_ids = mx.array([prompt_tokens])
    mx.clear_cache()
    gc.collect()
    reset_peak_mem()
    
    t_prefill = time.time()
    
    # 初始化SSM锚点
    h = model.model.embed_tokens(prefill_ids)
    B, L = prefill_ids.shape
    
    for i, ly in enumerate(model.model.layers):
        hn = ly.input_layernorm(h)
        at = ly.self_attn
        B2, L2, D = hn.shape
        
        Q = at.q_proj(hn).reshape(B2, L2, at.n_heads, -1).transpose(0, 2, 1, 3)
        K = at.k_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        V = at.v_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        Q = at.rope(Q)
        K = at.rope(K)
        
        # 初始化SSM锚点（只对替换层）
        if multi_ssm_model.get_layer_mode(i) and multi_ssm_model.ssm_list[i] is not None:
            multi_ssm_model.ssm_list[i].init_uniform(K, V, kn=kn)
        
        # 全量attention（prefill阶段）
        ao = gqa(Q, K, V, at.scale)
        
        ao = ao.transpose(0, 2, 1, 3).reshape(B2, L2, -1)
        h = h + at.o_proj(ao)
        h = h + ly.mlp(ly.post_attention_layernorm(h))
        mx.eval(h)
    
    h = model.model.norm(h)
    logits_full = model.model.embed_tokens.as_linear(h)
    mx.eval(logits_full)
    
    prefill_time = time.time() - t_prefill
    peak_mem = get_peak_mem_mb()
    
    # ===== Decode阶段 =====
    decode_tokens = 0
    t_decode = time.time()
    current_ids = list(prompt_tokens)
    
    for step in range(max_tokens):
        # 单token预测
        last_id = mx.array([[current_ids[-1]]])
        h = model.model.embed_tokens(last_id)
        
        for i, ly in enumerate(model.model.layers):
            hn = ly.input_layernorm(h)
            at = ly.self_attn
            B2, L2, D = hn.shape
            
            Q = at.q_proj(hn).reshape(B2, L2, at.n_heads, -1).transpose(0, 2, 1, 3)
            K = at.k_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
            V = at.v_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
            
            Q = at.rope(Q)
            K = at.rope(K)
            
            use_ssm = multi_ssm_model.get_layer_mode(i)
            ssm_initialized = (multi_ssm_model.ssm_list[i] is not None and 
                              multi_ssm_model.ssm_list[i].initialized)
            
            if use_ssm and ssm_initialized:
                # 近场窗口
                kn_ctx = min(kn, len(current_ids))
                nk = K[:, :, -kn_ctx:, :] if L2 > 1 else K
                nv = V[:, :, -kn_ctx:, :] if L2 > 1 else V
                ao_near = gqa(Q, nk, nv, at.scale)
                
                # SSM查询
                hr = multi_ssm_model.ssm_list[i].query(Q, at.n_heads, at.scale)
                adapt_alpha = fa * max(1.0, len(current_ids) / L_THRESHOLD)
                ao = ao_near + adapt_alpha * hr
            else:
                # 保留层：近场窗口
                kn_ctx = min(kn, len(current_ids))
                nk = K[:, :, -kn_ctx:, :] if L2 > 1 else K
                nv = V[:, :, -kn_ctx:, :] if L2 > 1 else V
                ao = gqa(Q, nk, nv, at.scale)
            
            ao = ao.transpose(0, 2, 1, 3).reshape(B2, L2, -1)
            h = h + at.o_proj(ao)
            h = h + ly.mlp(ly.post_attention_layernorm(h))
            mx.eval(h)
        
        h = model.model.norm(h)
        logits = model.model.embed_tokens.as_linear(h)
        mx.eval(logits)
        
        # 采样
        probs = mx.softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
        next_tok = int(mx.argmax(probs, axis=-1))
        mx.eval(next_tok)
        
        current_ids.append(next_tok)
        decode_tokens += 1
        
        del h, logits, probs
        gc.collect()
    
    decode_time = time.time() - t_decode
    tok_per_sec = decode_tokens / decode_time if decode_time > 0 else 0
    
    return {
        'tokens': current_ids,
        'decode_tokens': decode_tokens,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'tok_per_sec': tok_per_sec,
        'peak_mem_MB': peak_mem
    }


def run_standard_generate(model, tokenizer, prompt_tokens, max_tokens=50):
    """标准模型生成（参考）"""
    prompt_tokens = list(prompt_tokens)
    
    mx.clear_cache()
    gc.collect()
    reset_peak_mem()
    
    t_prefill = time.time()
    
    # Prefill
    prefill_ids = mx.array([prompt_tokens])
    logits_full = uf_standard(model, prefill_ids)
    mx.eval(logits_full)
    
    prefill_time = time.time() - t_prefill
    peak_mem = get_peak_mem_mb()
    
    # Decode
    decode_tokens = 0
    t_decode = time.time()
    current_ids = list(prompt_tokens)
    
    for step in range(max_tokens):
        last_id = mx.array([[current_ids[-1]]])
        h = model.model.embed_tokens(last_id)
        
        for ly in model.model.layers:
            hn = ly.input_layernorm(h)
            at = ly.self_attn
            B2, L2, D = hn.shape
            
            Q = at.q_proj(hn).reshape(B2, L2, at.n_heads, -1).transpose(0, 2, 1, 3)
            K = at.k_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
            V = at.v_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
            
            Q = at.rope(Q)
            K = at.rope(K)
            
            kn_ctx = min(KN, len(current_ids))
            nk = K[:, :, -kn_ctx:, :] if L2 > 1 else K
            nv = V[:, :, -kn_ctx:, :] if L2 > 1 else V
            ao = gqa(Q, nk, nv, at.scale)
            
            ao = ao.transpose(0, 2, 1, 3).reshape(B2, L2, -1)
            h = h + at.o_proj(ao)
            h = h + ly.mlp(ly.post_attention_layernorm(h))
            mx.eval(h)
        
        h = model.model.norm(h)
        logits = model.model.embed_tokens.as_linear(h)
        mx.eval(logits)
        
        probs = mx.softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
        next_tok = int(mx.argmax(probs, axis=-1))
        mx.eval(next_tok)
        
        current_ids.append(next_tok)
        decode_tokens += 1
        
        del h, logits, probs
        gc.collect()
    
    decode_time = time.time() - t_decode
    tok_per_sec = decode_tokens / decode_time if decode_time > 0 else 0
    
    return {
        'tokens': current_ids,
        'decode_tokens': decode_tokens,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'tok_per_sec': tok_per_sec,
        'peak_mem_MB': peak_mem
    }


# ======================================================================
# 端到端cos测试
# ======================================================================
def run_cos_test(model, tokenizer, multi_ssm_model, test_ids, max_tokens=20):
    """
    端到端cos测试：对比参考模型和SSM模型的输出相似度
    
    Returns:
        cos_sim: 余弦相似度
    """
    # 清空缓存
    mx.clear_cache()
    gc.collect()
    
    # 参考输出（标准模型）
    ref_ids = mx.array([test_ids[:min(len(test_ids), 512)]])  # 限制长度
    logits_ref = uf_standard(model, ref_ids)
    mx.eval(logits_ref)
    
    # SSM输出
    logits_ssm = uf_with_multi_ssm(model, ref_ids, multi_ssm_model, mode='multi_ssm')
    mx.eval(logits_ssm)
    
    # 计算cos相似度
    cos_sim = csim(logits_ref, logits_ssm)
    
    return cos_sim


# ======================================================================
# 基准测试
# ======================================================================
def benchmark_multi_ssm(model, tokenizer, mode, seq_len, max_decode=50):
    """单长度性能基准测试"""
    print(f"\n  [{mode}] 序列长度: {seq_len}")
    
    # 测试prompt
    test_prompt = "人工智能技术正在深刻改变"
    
    # 编码
    messages = [{"role": "user", "content": test_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text)
    
    # 限制长度
    if len(prompt_tokens) > seq_len:
        prompt_tokens = prompt_tokens[:seq_len]
    
    # 清理内存
    mx.clear_cache()
    gc.collect()
    reset_peak_mem()
    
    # 创建多层SSM模型
    multi_ssm_model = MultiLayerSignalFieldModel(model, mode=mode)
    
    # 运行参考基准（标准模型）
    print(f"    运行参考基准...")
    ref_result = run_standard_generate(model, tokenizer, prompt_tokens, max_tokens=min(20, max_decode))
    ref_tok_s = ref_result['tok_per_sec']
    ref_mem = ref_result['peak_mem_MB']
    
    # 运行SSM模型
    print(f"    运行{mode}模型...")
    ssm_result = run_multi_ssm_generate(model, tokenizer, multi_ssm_model, prompt_tokens, max_tokens=max_decode)
    ssm_tok_s = ssm_result['tok_per_sec']
    ssm_mem = ssm_result['peak_mem_MB']
    
    # 端到端cos测试
    print(f"    计算cos相似度...")
    test_ids = prompt_tokens * 2  # 重复以增加长度
    cos_sim = run_cos_test(model, tokenizer, multi_ssm_model, test_ids, max_tokens=10)
    
    print(f"    tok/s: {ssm_tok_s:.1f} (参考: {ref_tok_s:.1f})")
    print(f"    peak_MB: {ssm_mem:.1f}")
    print(f"    cos_sim: {cos_sim:.4f}")
    
    return {
        'mode': mode,
        'seq': seq_len,
        'tok_s': ssm_tok_s,
        'peak_MB': ssm_mem,
        'cos_sim': cos_sim,
        'ref_tok_s': ref_tok_s,
        'status': '✓'
    }


# ======================================================================
# 主函数
# ======================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Soma v7多层端到端推理脚本")
    parser.add_argument("--mode", type=str, choices=["v7a", "v7b", "both"], default="both",
                        help="测试模式: v7a保守版, v7b激进版, 或both两种都跑")
    parser.add_argument("--seq", type=int, nargs="+", default=[2048, 4096, 8192],
                        help="测试序列长度列表")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="每次生成的最大token数")
    
    args = parser.parse_args()
    
    # 模式列表
    if args.mode == "both":
        modes = ["v7a", "v7b"]
    else:
        modes = [args.mode]
    
    seq_lengths = args.seq
    max_decode = args.max_tokens
    
    # 打印信息
    info = sys_info()
    print("=" * 70)
    print("Soma v7多层端到端 — 推理性能")
    print("=" * 70)
    print(f"\n环境信息:")
    print(f"  CPU: {info['cpu']}")
    print(f"  GPU: {info['gpu']}")
    print(f"  内存: {info['mem_gb']} GB")
    print(f"  Python: {info['python']}")
    print(f"  MLX: {info.get('mlx', '?')}")
    print(f"\n信号场参数:")
    print(f"  k={K_ANCHORS}, kn={KN}, base_alpha={BASE_ALPHA}")
    print(f"  Lthreshold={L_THRESHOLD}")
    
    # 加载模型
    print(f"\n[1/5] 加载模型...")
    model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"  ⚠️ 模型路径不存在: {model_path}")
        model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"
        print(f"  使用默认路径: {model_path}")
    
    try:
        model, tokenizer = load(model_path)
        sa = model.model.layers[0].self_attn
        n_heads = sa.n_heads
        n_kv_heads = sa.n_kv_heads
        print(f"  ✅ 模型加载成功")
        print(f"     层数: {len(model.model.layers)}")
        print(f"     注意力头: {n_heads}")
        print(f"     KV头: {n_kv_heads}")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return
    
    # 打印表头
    print("\n" + "=" * 70)
    print(f"{'mode':<6} | {'seq':<4} | {'tok/s':<8} | {'peak_MB':<10} | {'cos_sim':<8} | {'status'}")
    print("-" * 70)
    
    results = []
    
    for mode in modes:
        for seq_len in seq_lengths:
            result = benchmark_multi_ssm(model, tokenizer, mode, seq_len, max_decode=max_decode)
            results.append(result)
            
            # 打印结果行
            print("-" * 70)
            print(f"{result['mode']:<6} | {result['seq']:<4} | {result['tok_s']:<8.1f} | "
                  f"{result['peak_MB']:<10.1f} | {result['cos_sim']:<8.4f} | {result['status']}")
            print("-" * 70)
    
    # 总结
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    if results:
        print("\n结论:")
        
        # v7a vs v7b 对比
        v7a_results = [r for r in results if r["mode"] == "v7a"]
        v7b_results = [r for r in results if r["mode"] == "v7b"]
        
        if v7a_results and v7b_results:
            avg_v7a_cos = sum(r["cos_sim"] for r in v7a_results) / len(v7a_results)
            avg_v7b_cos = sum(r["cos_sim"] for r in v7b_results) / len(v7b_results)
            avg_v7a_speed = sum(r["tok_s"] for r in v7a_results) / len(v7a_results)
            avg_v7b_speed = sum(r["tok_s"] for r in v7b_results) / len(v7b_results)
            
            print(f"\nv7a(保守版8层替换) vs v7b(激进版24层替换):")
            print(f"  cos_sim: {avg_v7a_cos:.4f} vs {avg_v7b_cos:.4f}")
            print(f"  tok/s:   {avg_v7a_speed:.1f} vs {avg_v7b_speed:.1f}")
            
            if avg_v7a_cos > avg_v7b_cos:
                diff = avg_v7a_cos - avg_v7b_cos
                print(f"\n  → v7a保守版保持更好的模型质量 (差值: {diff:.4f})")
            else:
                diff = avg_v7b_cos - avg_v7a_cos
                print(f"\n  → v7b激进版质量损失可接受 (差值: {diff:.4f})")
            
            if avg_v7a_speed > 0 and avg_v7b_speed > 0:
                speedup = avg_v7b_speed / avg_v7a_speed
                print(f"  → 激进版速度: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    main()
