#!/usr/bin/env python3
"""
Soma v7 文本生成质量验证脚本
测试不同SSM替换策略下的生成文本质量

✅ 检查清单确认：
1. 模型加载：只用load，从self_attn取参数
2. 生成方式：手动decode循环，prefill+decode，argmax采样
3. gqa函数：4维输出[B,H,L,D]
4. SignalFieldLayer：照抄v7已验证版本
5. BLEU计算：纯Python实现，不依赖nltk
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
BASE_ALPHA = 0.04
K_ANCHORS = 8
KN = 256
L_THRESHOLD = 2048

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

TEST_PROMPTS = [
    {"name": "常识", "prompt": "请简要介绍一下中国的四大发明"},
    {"name": "推理", "prompt": "如果所有的猫都是动物，所有的动物都需要水，那么猫需要水吗？请解释推理过程"},
    {"name": "创意", "prompt": "写一首关于春天的小诗，四行即可"}
]

# ======================================================================
# 辅助函数
# ======================================================================
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

def clear_mem():
    """清理内存"""
    mx.clear_cache()
    gc.collect()

# ======================================================================
# gqa函数（照抄v7已验证版本，4维输出[B,H,L,D]）
# ======================================================================
def gqa(q, k, v, scale):
    """Grouped Query Attention - 输出4维[B,H,L,D]"""
    B, H, L, D = q.shape
    Hk = k.shape[1]
    nr = H // Hk
    Q = q.reshape(B, Hk, nr, L, D)
    K = k.reshape(B, Hk, 1, k.shape[2], D)
    V = v.reshape(B, Hk, 1, v.shape[2], D)
    sc = (Q * scale) @ K.transpose(0, 1, 2, 4, 3)
    w = mx.softmax(sc, axis=-1)
    return (w @ V).reshape(B, H, L, D)

# ======================================================================
# SignalFieldLayer（照抄v7已验证版本）
# ======================================================================
class SignalFieldLayer:
    """信号场层 - 照抄v7已验证版本"""
    
    def __init__(self, n_kv_heads, head_dim, n_anchors=K_ANCHORS, gamma_v=0.98):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_anchors = n_anchors
        self.gamma_v = gamma_v
        
        self.keys = [mx.zeros([1, n_kv_heads, head_dim]) for _ in range(n_anchors)]
        self.values = [mx.zeros([1, n_kv_heads, head_dim]) for _ in range(n_anchors)]
        self.initialized = False
        self._anchor_idx = 0
        self.decay_log = mx.log(mx.array([0.98] * n_anchors, dtype=mx.float32))
        
        self.compress_dim = min(n_anchors, n_kv_heads)
        self.compress_weight = mx.random.normal(
            (n_kv_heads, self.compress_dim), scale=mx.sqrt(1.0 / self.compress_dim)
        )
    
    def init_uniform(self, K, V, kn=KN):
        """均匀采样远场锚点 - positions防越界"""
        B, Hk, L, D = K.shape
        far_end = max(kn, L - kn)
        step = max(1, far_end // self.n_anchors)
        positions = [min(int(i * step), L - 1) for i in range(self.n_anchors)]
        positions = positions[:self.n_anchors]
        while len(positions) < self.n_anchors:
            positions.append(max(0, L - 1))
        for j, pos in enumerate(positions):
            pos = min(pos, L - 1)
            self.keys[j] = K[0, :, pos, :].reshape(1, Hk, D)
            self.values[j] = V[0, :, pos, :].reshape(1, Hk, D)
        self.initialized = True
        self._positions = positions
    
    def query(self, q, n_heads, scale):
        """查询信号场 - 返回4维[B,H,L,D]"""
        B, H, L, D = q.shape
        Hk = self.n_kv_heads
        nr = H // Hk
        q_r = q.reshape(B, Hk, nr, L, D)
        
        ak = mx.stack(self.keys, axis=2)  # [1, Hk, n_anchors, D]
        av = mx.stack(self.values, axis=2)  # [1, Hk, n_anchors, D]
        
        ak_t = ak.reshape(1, Hk, 1, self.n_anchors, D).transpose(0, 1, 2, 4, 3)
        score = (q_r * scale) @ ak_t
        weight = mx.softmax(score, axis=-1)
        
        av_r = av.reshape(1, Hk, 1, self.n_anchors, D)
        out = (weight @ av_r).reshape(B, H, L, D)
        
        return out

# ======================================================================
# MultiLayerSignalFieldModel（照抄v7，不传config）
# ======================================================================
class MultiLayerSignalFieldModel:
    """多层信号场替换模型 - 不传config参数"""
    
    def __init__(self, model, mode="v7a"):
        self.model = model
        self.mode = mode
        self.replace_layers = REPLACE_MODES[mode]["replace_layers"]
        
        # 从self_attn取参数
        sa = model.model.layers[0].self_attn
        self.n_heads = sa.n_heads
        self.n_kv_heads = sa.n_kv_heads
        self.head_dim = sa.k_proj.weight.shape[0] // self.n_kv_heads
        self.n_layers = len(model.model.layers)
        
        self.ssm_list = []
        for i in range(self.n_layers):
            if i in self.replace_layers:
                ssm = SignalFieldLayer(self.n_kv_heads, self.head_dim, n_anchors=K_ANCHORS)
            else:
                ssm = None
            self.ssm_list.append(ssm)
    
    def get_layer_mode(self, layer_idx):
        """判断某层是否需要替换"""
        return layer_idx in self.replace_layers
    
    def reset(self):
        """重置所有SSM层状态"""
        for ssm in self.ssm_list:
            if ssm is not None:
                ssm.initialized = False

# ======================================================================
# Baseline生成（照抄v7 run_standard_generate）
# ======================================================================
def generate_baseline(model, tokenizer, prompt_tokens, max_tokens=100):
    """Baseline生成 - 手动decode循环，argmax采样"""
    prompt_tokens = list(prompt_tokens)
    current_ids = prompt_tokens.copy()
    
    # Prefill
    prefill_ids = mx.array([prompt_tokens])
    h = model.model.embed_tokens(prefill_ids)
    B, L = prefill_ids.shape
    
    for ly in model.model.layers:
        hn = ly.input_layernorm(h)
        at = ly.self_attn
        B2, L2, D = hn.shape
        
        Q = at.q_proj(hn).reshape(B2, L2, at.n_heads, -1).transpose(0, 2, 1, 3)
        K = at.k_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        V = at.v_proj(hn).reshape(B2, L2, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        Q = at.rope(Q)
        K = at.rope(K)
        
        # 近场窗口（跟v7一致）
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
    
    # Decode循环 - argmax采样
    for step in range(max_tokens):
        # argmax采样
        probs = mx.softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
        next_tok = int(mx.argmax(probs, axis=-1))
        mx.eval(next_tok)
        current_ids.append(next_tok)
        
        if next_tok == tokenizer.eos_token_id:
            break
        
        # 单token前向
        last_id = mx.array([[next_tok]])
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
        
        del h, probs
    
    response_ids = current_ids[len(prompt_tokens):]
    text = tokenizer.decode(response_ids)
    
    return {
        'tokens': current_ids,
        'response_tokens': response_ids,
        'text': text,
        'num_tokens': len(response_ids)
    }

# ======================================================================
# Multi-SSM生成（照抄v7 run_multi_ssm_generate）
# ======================================================================
def generate_with_ssm(model, tokenizer, multi_ssm_model, prompt_tokens, max_tokens=100):
    """Multi-SSM生成 - 手动decode循环，argmax采样"""
    prompt_tokens = list(prompt_tokens)
    current_ids = prompt_tokens.copy()
    
    # ===== Prefill阶段 =====
    prefill_ids = mx.array([prompt_tokens])
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
        
        # 初始化SSM锚点
        if multi_ssm_model.get_layer_mode(i) and multi_ssm_model.ssm_list[i] is not None:
            multi_ssm_model.ssm_list[i].init_uniform(K, V, kn=KN)
        
        # 全量attention（prefill阶段）
        ao = gqa(Q, K, V, at.scale)
        
        ao = ao.transpose(0, 2, 1, 3).reshape(B2, L2, -1)
        h = h + at.o_proj(ao)
        h = h + ly.mlp(ly.post_attention_layernorm(h))
        mx.eval(h)
    
    h = model.model.norm(h)
    logits = model.model.embed_tokens.as_linear(h)
    mx.eval(logits)
    
    # ===== Decode阶段 =====
    for step in range(max_tokens):
        # argmax采样
        probs = mx.softmax(logits[0, -1, :].astype(mx.float32), axis=-1)
        next_tok = int(mx.argmax(probs, axis=-1))
        mx.eval(next_tok)
        current_ids.append(next_tok)
        
        if next_tok == tokenizer.eos_token_id:
            break
        
        # 单token前向
        last_id = mx.array([[next_tok]])
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
                kn_ctx = min(KN, len(current_ids))
                nk = K[:, :, -kn_ctx:, :] if L2 > 1 else K
                nv = V[:, :, -kn_ctx:, :] if L2 > 1 else V
                ao_near = gqa(Q, nk, nv, at.scale)
                
                # SSM查询
                hr = multi_ssm_model.ssm_list[i].query(Q, at.n_heads, at.scale)
                adapt_alpha = BASE_ALPHA * max(1.0, len(current_ids) / L_THRESHOLD)
                ao = ao_near + adapt_alpha * hr
            else:
                # 保留层：近场窗口
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
        
        del h, probs
    
    response_ids = current_ids[len(prompt_tokens):]
    text = tokenizer.decode(response_ids)
    
    return {
        'tokens': current_ids,
        'response_tokens': response_ids,
        'text': text,
        'num_tokens': len(response_ids)
    }

# ======================================================================
# BLEU计算（纯Python实现，不依赖nltk）
# ======================================================================
def simple_bleu(reference, hypothesis, n=2):
    """
    简化BLEU分数 - 纯Python实现
    
    Args:
        reference: 参考文本（字符串）
        hypothesis: 待评估文本（字符串）
        n: 最大n-gram阶数
    
    Returns:
        float: BLEU分数 (0-1)
    """
    def get_ngrams(tokens, n):
        """获取n-gram集合"""
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    def count_clip(candidate_ngrams, reference_ngrams):
        """计算clip次数"""
        count = 0
        for ng in candidate_ngrams:
            count += min(candidate_ngrams[ng], reference_ngrams.get(ng, 0))
        return count
    
    # 分词
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    if not ref_tokens or not hyp_tokens:
        return 0.0
    
    # 计算各阶n-gram
    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = get_ngrams(ref_tokens, i)
        hyp_ngrams = get_ngrams(hyp_tokens, i)
        
        if not hyp_ngrams:
            precisions.append(0.0)
        else:
            # 计数
            hyp_count = {}
            for ng in hyp_ngrams:
                hyp_count[ng] = hyp_count.get(ng, 0) + 1
            
            ref_count = {}
            for ng in ref_ngrams:
                ref_count[ng] = ref_count.get(ng, 0) + 1
            
            clip = count_clip(hyp_count, ref_count)
            total = len(hyp_tokens) - i + 1
            
            if total > 0:
                precisions.append(clip / total)
            else:
                precisions.append(0.0)
    
    # 如果所有precision都是0，返回0
    if all(p == 0 for p in precisions):
        return 0.0
    
    # 几何平均
    p_log_sum = sum(max(0.0001, p) for p in precisions) / n
    geo_mean = p_log_sum ** (1 / n)  # 简化版，不做幂运算
    
    # 简短惩罚
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = 1.0 - ref_len / (hyp_len + 0.01)
    
    return geo_mean * bp


def word_overlap_ratio(reference, hypothesis):
    """
    词级重叠率 - 更简单的相似度度量
    
    Returns:
        float: 0-1之间的重叠率
    """
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    
    if not ref_words and not hyp_words:
        return 1.0
    if not ref_words or not hyp_words:
        return 0.0
    
    intersection = len(ref_words & hyp_words)
    union = len(ref_words | hyp_words)
    
    return intersection / union if union > 0 else 0.0


def compute_bleu(ref, hyp):
    """计算BLEU分数的便捷函数"""
    bleu = simple_bleu(ref, hyp, n=2)
    return bleu

# ======================================================================
# 主测试函数
# ======================================================================
def run_test(model, tokenizer, prompt_data, max_tokens=100):
    """对单个prompt运行三种模式的测试"""
    
    # 格式化prompt
    messages = [{"role": "user", "content": prompt_data["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text)
    
    print(f"\n{'='*66}")
    print(f"Prompt [{prompt_data['name']}]: {prompt_data['prompt']}")
    print(f"{'='*66}")
    
    result = {
        "prompt_name": prompt_data["name"],
        "prompt_text": prompt_data["prompt"],
    }
    
    # ===== Baseline =====
    print(f"\n[Baseline] 原始模型...")
    clear_mem()
    t0 = time.time()
    baseline = generate_baseline(model, tokenizer, prompt_tokens, max_tokens)
    result["baseline"] = {
        "text": baseline["text"],
        "num_tokens": baseline["num_tokens"],
        "time": time.time() - t0
    }
    print(f"  生成{baseline['num_tokens']}token, {result['baseline']['time']:.1f}s")
    print(f"  文本: {baseline['text'][:80]}...")
    
    # ===== v7a =====
    print(f"\n[v7a] 保守版...")
    clear_mem()
    multi_ssm_a = MultiLayerSignalFieldModel(model, mode="v7a")
    t0 = time.time()
    v7a = generate_with_ssm(model, tokenizer, multi_ssm_a, prompt_tokens, max_tokens)
    result["v7a"] = {
        "text": v7a["text"],
        "num_tokens": v7a["num_tokens"],
        "time": time.time() - t0
    }
    print(f"  生成{v7a['num_tokens']}token, {result['v7a']['time']:.1f}s")
    print(f"  文本: {v7a['text'][:80]}...")
    multi_ssm_a.reset()
    del multi_ssm_a
    
    # ===== v7b =====
    print(f"\n[v7b] 激进版...")
    clear_mem()
    multi_ssm_b = MultiLayerSignalFieldModel(model, mode="v7b")
    t0 = time.time()
    v7b = generate_with_ssm(model, tokenizer, multi_ssm_b, prompt_tokens, max_tokens)
    result["v7b"] = {
        "text": v7b["text"],
        "num_tokens": v7b["num_tokens"],
        "time": time.time() - t0
    }
    print(f"  生成{v7b['num_tokens']}token, {result['v7b']['time']:.1f}s")
    print(f"  文本: {v7b['text'][:80]}...")
    del multi_ssm_b
    clear_mem()
    
    return result

# ======================================================================
# 主函数
# ======================================================================
def main():
    info = sys_info()
    print("=" * 70)
    print("Soma v7 文本生成质量验证")
    print("=" * 70)
    print(f"\n环境: CPU={info['cpu']}, 内存={info['mem_gb']}GB, MLX={info.get('mlx', '?')}")
    print(f"信号场: k={K_ANCHORS}, kn={KN}, alpha={BASE_ALPHA}")
    print(f"测试: {len(TEST_PROMPTS)}个prompt, max_tokens=100, argmax采样")
    
    # 加载模型
    print(f"\n[1/4] 加载模型...")
    model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"  ⚠️ 路径不存在，使用默认路径")
        model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    
    try:
        model, tokenizer = load(model_path)
        sa = model.model.layers[0].self_attn
        print(f"  ✅ 模型加载成功")
        print(f"     层数={len(model.model.layers)}, 头={sa.n_heads}, KV头={sa.n_kv_heads}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        sys.exit(1)
    
    # 运行测试
    print(f"\n[2/4] 运行文本生成测试...")
    all_results = []
    
    for prompt_data in TEST_PROMPTS:
        try:
            result = run_test(model, tokenizer, prompt_data, max_tokens=100)
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算BLEU
    print(f"\n[3/4] 计算BLEU分数...")
    bleu_results = []
    
    for res in all_results:
        baseline_text = res["baseline"]["text"]
        v7a_text = res["v7a"]["text"]
        v7b_text = res["v7b"]["text"]
        
        bleu_v7a = compute_bleu(baseline_text, v7a_text)
        bleu_v7b = compute_bleu(baseline_text, v7b_text)
        overlap_v7a = word_overlap_ratio(baseline_text, v7a_text)
        overlap_v7b = word_overlap_ratio(baseline_text, v7b_text)
        
        def status(b):
            if b >= 0.4: return "✓"
            elif b >= 0.2: return "⚠"
            else: return "❌"
        
        bleu_results.append({
            "name": res["prompt_name"],
            "bleu_v7a": bleu_v7a,
            "bleu_v7b": bleu_v7b,
            "overlap_v7a": overlap_v7a,
            "overlap_v7b": overlap_v7b,
            "stat_v7a": status(bleu_v7a),
            "stat_v7b": status(bleu_v7b)
        })
    
    # 输出报告
    print(f"\n[4/4] 输出报告...")
    
    print("\n" + "=" * 70)
    print("Soma v7 文本生成质量验证 - 完整报告")
    print("=" * 70)
    
    for res in all_results:
        print(f"\n{'='*66}")
        print(f"Prompt [{res['prompt_name']}]: {res['prompt_text']}")
        print(f"{'='*66}")
        
        print(f"\n[原始模型]:")
        print(f"  {res['baseline']['text']}")
        print(f"  ({res['baseline']['num_tokens']}token, {res['baseline']['time']:.1f}s)")
        
        print(f"\n[v7a保守版]:")
        print(f"  {res['v7a']['text']}")
        print(f"  ({res['v7a']['num_tokens']}token, {res['v7a']['time']:.1f}s)")
        
        print(f"\n[v7b激进版]:")
        print(f"  {res['v7b']['text']}")
        print(f"  ({res['v7b']['num_tokens']}token, {res['v7b']['time']:.1f}s)")
    
    # BLEU对比表
    print(f"\n{'='*66}")
    print("量化对比")
    print(f"{'='*66}")
    print(f"{'Prompt':<8} | {'BLEU(v7a)':<10} | {'BLEU(v7b)':<10} | {'重叠率v7a':<10} | {'重叠率v7b':<10} | v7a | v7b")
    print("-" * 90)
    
    for br in bleu_results:
        print(f"{br['name']:<8} | {br['bleu_v7a']:.4f}     | {br['bleu_v7b']:.4f}     | {br['overlap_v7a']:.4f}      | {br['overlap_v7b']:.4f}      | {br['stat_v7a']:4} | {br['stat_v7b']}")
    
    print(f"\n说明:")
    print(f"  ✓ BLEU >= 0.4: 质量良好")
    print(f"  ⚠ BLEU 0.2-0.4: 质量下降")
    print(f"  ❌ BLEU < 0.2: 质量严重下降")
    
    # 保存JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_tokens": 100,
            "signal_field": {"k": K_ANCHORS, "kn": KN, "base_alpha": BASE_ALPHA, "L_threshold": L_THRESHOLD}
        },
        "results": all_results,
        "bleu_scores": bleu_results
    }
    
    output_path = Path("灵芽实验/Soma v7文本验证结果.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
