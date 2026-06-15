#!/usr/bin/env python3
"""Qwen2.5-14B-Instruct-4bit 性能基准测试 - Mac M1 Pro"""
import time, os, psutil
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-14B-Instruct-4bit/snapshots/dad510143ae5cdb1466778bde3161460f9e9f604")

PROMPTS = {
    "短prompt": "请用中文简要介绍量子计算的基本原理。",
    "中prompt": "请详细解释Transformer架构中自注意力机制的工作原理，包括Q、K、V矩阵的计算过程，多头注意力的作用，以及位置编码的必要性。请用中文回答，不少于200字。",
    "长prompt": "在人工智能领域，大语言模型（LLM）的发展经历了从GPT-1到GPT-4的演进过程。请详细分析以下几个问题：1）模型规模增长带来的能力涌现现象；2）指令微调（Instruction Tuning）和人类反馈强化学习（RLHF）对模型对齐的重要性；3）当前大模型面临的幻觉问题及其可能的解决方案；4）多模态大模型的技术路线比较；5）大模型在科学研究中的应用前景。请用中文逐一回答，每个问题至少100字。"
}

MAX_TOKENS = 200

def get_system_info():
    mem = psutil.virtual_memory()
    return {
        "rss_gb": psutil.Process(os.getpid()).memory_info().rss / 1024**3,
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "mem_total_gb": mem.total / 1024**3,
        "mem_used_pct": mem.percent,
    }

# 新版API用sampler对象
sampler = make_sampler(temp=0.7, top_p=0.9)

print("=" * 60)
print("Qwen2.5-14B-Instruct-4bit 性能报告")
print("=" * 60)
print(f"设备: Apple M1 Pro 16GB")
print(f"框架: MLX")
print()

# 加载模型
print("【加载模型】")
t0 = time.time()
model, tokenizer = load(MODEL_PATH)
load_time = time.time() - t0
gpu_mem = mx.get_active_memory() / 1024**3
print(f"- 加载时间: {load_time:.1f}s")
print(f"- GPU内存: {gpu_mem:.2f}GB")
print()

results = []

for name, prompt_text in PROMPTS.items():
    print(f"【测试: {name}】")
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_tokens = len(tokenizer.encode(text))
    
    t0 = time.time()
    response = generate(model, tokenizer, prompt=text, max_tokens=MAX_TOKENS, sampler=sampler, verbose=False)
    total_time = time.time() - t0
    
    output_text = response if isinstance(response, str) else str(response)
    gen_text = output_text[len(text):] if output_text.startswith(text) else output_text
    output_tokens = len(tokenizer.encode(gen_text))
    
    speed = output_tokens / total_time if total_time > 0 else 0
    
    sys_info = get_system_info()
    gpu_peak = mx.get_peak_memory() / 1024**3
    
    print(f"- 输入长度: {input_tokens} tokens")
    print(f"- 输出长度: {output_tokens} tokens")
    print(f"- 总耗时: {total_time:.2f}s")
    print(f"- 生成速度: {speed:.2f} tokens/sec")
    print(f"- 内存(RSS): {sys_info['rss_gb']:.2f}GB / {sys_info['mem_total_gb']:.1f}GB ({sys_info['mem_used_pct']}%)")
    print(f"- GPU内存峰值: {gpu_peak:.2f}GB")
    print(f"- CPU: {sys_info['cpu_pct']}%")
    print(f"- 输出预览: {gen_text[:80]}...")
    print()
    
    results.append({
        "name": name, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_time": total_time, "speed": speed, "rss_gb": sys_info['rss_gb'],
        "gpu_peak": gpu_peak, "cpu_pct": sys_info['cpu_pct']
    })

# 汇总
print("=" * 60)
print("汇总对比")
print("=" * 60)
print(f"{'测试':<12} {'输入':>6} {'输出':>6} {'速度(tok/s)':>12} {'RSS(GB)':>8} {'GPU峰值(GB)':>11} {'CPU%':>5}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<12} {r['input_tokens']:>6} {r['output_tokens']:>6} {r['speed']:>12.2f} {r['rss_gb']:>8.2f} {r['gpu_peak']:>11.2f} {r['cpu_pct']:>5}")
print("-" * 60)

avg_speed = sum(r['speed'] for r in results) / len(results)
max_gpu = max(r['gpu_peak'] for r in results)
max_rss = max(r['rss_gb'] for r in results)
h100_speed = 100
pct_h100 = avg_speed / h100_speed * 100

print(f"\n平均速度: {avg_speed:.2f} tokens/sec")
print(f"内存峰值: {max_rss:.2f}GB / 16GB ({max_rss/16*100:.0f}%)")
print(f"GPU峰值: {max_gpu:.2f}GB")
print(f"\nH100 14B推理: ~{h100_speed} tokens/sec")
print(f"咱的速度是H100的: {pct_h100:.0f}%")
print("=" * 60)
