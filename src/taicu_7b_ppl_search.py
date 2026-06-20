"""
Signal Field7B信号场PPL参数搜索 v10

遍历 k/alpha/gamma 参数组合，找PPL最低的组合。

搜索策略：
1. 粗搜索：先搜k和alpha，gamma固定0.98
2. 细搜索：基于粗搜索结果，对top参数进行gamma搜索
3. 每组参数取多段文本PPL平均值

优化：加载一次模型，每组参数替换→计算→恢复
"""

import json
import time
import sys
import os
from typing import List, Tuple, Optional

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("需要MLX环境（Mac M1 Pro）")
    exit(0)

try:
    from mlx_lm import load
except ImportError:
    print("mlx_lm not available")
    exit(0)

# Import from verify module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taicu_7b_ppl_verify import (
    compute_ppl, 
    compute_ppl_from_text,
    SignalFieldAttentionGQA, 
    QWEN_CONFIG
)


# ============================================================
# Config
# ============================================================
MODEL_PATH = "/Users/apple/models/Qwen2.5-7B-Instruct-4bit/"

# Search parameter ranges
K_VALUES = [4, 8, 16, 32, 64, 128, 256]
ALPHA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
GAMMA_VALUES = [0.9, 0.95, 0.98, 0.99, 1.0]

# Short texts for fast search (128 token limit)
SEARCH_TEXTS = [
    "人工智能技术正在快速发展，机器学习和深度学习已经应用在很多领域。",
    "The quick brown fox jumps over the lazy dog.",
    "春天的花朵盛开，空气中弥漫着花香的味道。",
]

# Number of layers to replace with signal field
NUM_LAYERS = 1  # Start with layer 0 only for speed


# ============================================================
# Helper Functions
# ============================================================
def compute_avg_ppl(model, tokenizer, texts: List[str], max_length: int = 128) -> float:
    """Compute average PPL across multiple texts."""
    ppls = []
    for text in texts:
        try:
            ppl = compute_ppl_from_text(model, tokenizer, text, max_length=max_length)
            ppls.append(ppl)
        except Exception as e:
            print(f"    Warning: PPL computation failed: {e}")
            continue
    return sum(ppls) / len(ppls) if ppls else float('inf')


def save_and_restore_attention(model, layer_idx: int, original_attn):
    """Context manager for saving and restoring attention layer."""
    layer = model.model.layers[layer_idx]
    original = layer.self_attn
    layer.self_attn = original_attn
    yield
    layer.self_attn = original


# ============================================================
# Single Layer Parameter Search
# ============================================================
def search_single_layer(
    model, 
    tokenizer, 
    layer_idx: int = 0,
    k_values: List[int] = K_VALUES,
    alpha_values: List[int] = ALPHA_VALUES,
    gamma_values: List[int] = GAMMA_VALUES,
    baseline_ppl: Optional[float] = None
) -> List[dict]:
    """
    Search parameters for a single layer.
    
    Returns:
        List of dicts with k, alpha, gamma, avg_ppl, rank
    """
    results = []
    total_combinations = len(k_values) * len(alpha_values) * len(gamma_values)
    print(f"\n{'='*60}")
    print(f"Parameter Search: Layer {layer_idx}")
    print(f"  k values: {k_values}")
    print(f"  alpha values: {alpha_values}")
    print(f"  gamma values: {gamma_values}")
    print(f"  Total combinations: {total_combinations}")
    print(f"{'='*60}\n")
    
    # Get original attention module
    layer = model.model.layers[layer_idx]
    original_attn = layer.self_attn
    
    # Verify we have rope
    if not hasattr(original_attn, 'rope'):
        print(f"  Warning: Layer {layer_idx} attention has no rope attribute")
    
    count = 0
    start_time = time.time()
    
    for k in k_values:
        for alpha in alpha_values:
            for gamma in gamma_values:
                count += 1
                
                # Check for OOM risk with large k
                if k > 128:
                    print(f"  [{count}/{total_combinations}] k={k}, alpha={alpha}, gamma={gamma} ... ", end="", flush=True)
                else:
                    print(f"  [{count}/{total_combinations}] k={k}, alpha={alpha}, gamma={gamma} ... ", end="", flush=True)
                
                try:
                    # Create signal field attention with this config
                    sf_attn = SignalFieldAttentionGQA(original_attn, k=k, gamma=gamma, alpha=alpha)
                    
                    # Replace attention
                    layer.self_attn = sf_attn
                    
                    # Compute PPL
                    ppl_start = time.time()
                    avg_ppl = compute_avg_ppl(model, tokenizer, SEARCH_TEXTS, max_length=128)
                    ppl_time = time.time() - ppl_start
                    
                    # Restore original attention
                    layer.self_attn = original_attn
                    
                    # Calculate degradation
                    if baseline_ppl:
                        degradation_pct = (avg_ppl - baseline_ppl) / baseline_ppl * 100
                    else:
                        degradation_pct = None
                    
                    results.append({
                        'k': k,
                        'alpha': alpha,
                        'gamma': gamma,
                        'avg_ppl': avg_ppl,
                        'degradation_pct': degradation_pct,
                        'ppl_time_sec': ppl_time,
                        'layer': layer_idx
                    })
                    
                    print(f"PPL={avg_ppl:.4f} ({ppl_time:.1f}s)")
                    
                    # Clear cache periodically
                    if count % 5 == 0:
                        mx.clear_cache()
                        
                except Exception as e:
                    print(f"FAILED: {e}")
                    # Make sure we restore
                    layer.self_attn = original_attn
                    mx.clear_cache()
                    
                    results.append({
                        'k': k,
                        'alpha': alpha,
                        'gamma': gamma,
                        'avg_ppl': float('inf'),
                        'error': str(e),
                        'layer': layer_idx
                    })
                    continue
    
    total_time = time.time() - start_time
    
    # Sort by PPL (lower is better)
    results.sort(key=lambda x: x['avg_ppl'])
    
    # Add rank
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    print(f"\nSearch completed in {total_time/60:.1f} minutes")
    
    return results


# ============================================================
# Multi-Layer Search (Refinement)
# ============================================================
def refine_top_results(
    model,
    tokenizer,
    top_results: List[dict],
    gamma_values: List[float],
    num_layers: int = 3
) -> List[dict]:
    """
    Refine top results by searching gamma and testing multiple layers.
    """
    print(f"\n{'='*60}")
    print(f"Refining Top {len(top_results)} Results with Multiple Layers")
    print(f"{'='*60}\n")
    
    refined_results = []
    
    for i, top in enumerate(top_results[:5]):  # Top 5 only
        k = top['k']
        alpha = top['alpha']
        
        print(f"\n--- Refining #{i+1}: k={k}, alpha={alpha} ---")
        
        for gamma in gamma_values:
            print(f"  gamma={gamma}, layers={num_layers} ... ", end="", flush=True)
            
            try:
                # Replace multiple layers
                for layer_idx in range(num_layers):
                    layer = model.model.layers[layer_idx]
                    original_attn = layer.self_attn
                    sf_attn = SignalFieldAttentionGQA(original_attn, k=k, gamma=gamma, alpha=alpha)
                    layer.self_attn = sf_attn
                
                # Compute PPL
                avg_ppl = compute_avg_ppl(model, tokenizer, SEARCH_TEXTS, max_length=128)
                
                # Restore
                for layer_idx in range(num_layers):
                    layer = model.model.layers[layer_idx]
                    layer.self_attn = original_attn
                
                print(f"PPL={avg_ppl:.4f}")
                
                refined_results.append({
                    'k': k,
                    'alpha': alpha,
                    'gamma': gamma,
                    'num_layers': num_layers,
                    'avg_ppl': avg_ppl,
                    'rank': i * len(gamma_values) + gamma_values.index(gamma) + 1
                })
                
                mx.clear_cache()
                
            except Exception as e:
                print(f"FAILED: {e}")
                mx.clear_cache()
                continue
    
    refined_results.sort(key=lambda x: x['avg_ppl'])
    return refined_results


# ============================================================
# Baseline PPL
# ============================================================
def get_baseline_ppl(model, tokenizer, texts: List[str]) -> float:
    """Get baseline PPL without signal field."""
    print(f"\n{'='*60}")
    print("Computing Baseline PPL (original attention)")
    print(f"{'='*60}")
    
    avg_ppl = compute_avg_ppl(model, tokenizer, texts, max_length=128)
    print(f"Baseline PPL: {avg_ppl:.4f}")
    
    return avg_ppl


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Signal Field7B信号场PPL参数搜索 v10")
    print("=" * 70)
    
    # System info
    print(f"\nMLX Version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"Num layers to replace: {NUM_LAYERS}")
    
    # Load model (only once)
    print(f"\nLoading model: {MODEL_PATH}")
    
    try:
        model, tokenizer = load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    # Get baseline PPL
    baseline_ppl = get_baseline_ppl(model, tokenizer, SEARCH_TEXTS)
    
    # ============================================================
    # Phase 1: Coarse Search (k and alpha, gamma=0.98 fixed)
    # ============================================================
    print(f"\n{'='*70}")
    print("Phase 1: Coarse Search (gamma=0.98 fixed)")
    print(f"{'='*70}")
    
    coarse_results = search_single_layer(
        model, tokenizer,
        layer_idx=0,
        k_values=K_VALUES,
        alpha_values=ALPHA_VALUES,
        gamma_values=[0.98],  # Fixed
        baseline_ppl=baseline_ppl
    )
    
    # Print top 10
    print(f"\n{'='*60}")
    print("Top 10 Results (Coarse Search)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'k':<5} {'alpha':<8} {'gamma':<8} {'PPL':<12} {'Degradation':<12}")
    print("-" * 60)
    
    for r in coarse_results[:10]:
        deg_str = f"{r['degradation_pct']:+.2f}%" if r.get('degradation_pct') else "N/A"
        print(f"{r['rank']:<5} {r['k']:<5} {r['alpha']:<8} {r['gamma']:<8} {r['avg_ppl']:<12.4f} {deg_str:<12}")
    
    # ============================================================
    # Phase 2: Fine Search (full gamma sweep on top k/alpha)
    # ============================================================
    print(f"\n{'='*70}")
    print("Phase 2: Fine Search (full gamma sweep on top k/alpha)")
    print(f"{'='*70}")
    
    # Get top 3 k/alpha combinations
    top_k_alpha = []
    seen = set()
    for r in coarse_results:
        key = (r['k'], r['alpha'])
        if key not in seen:
            seen.add(key)
            top_k_alpha.append(r)
            if len(top_k_alpha) >= 3:
                break
    
    fine_results = []
    for top in top_k_alpha:
        k, alpha = top['k'], top['alpha']
        print(f"\n--- Fine tuning: k={k}, alpha={alpha} ---")
        
        for gamma in GAMMA_VALUES:
            print(f"  gamma={gamma} ... ", end="", flush=True)
            
            try:
                layer = model.model.layers[0]
                original_attn = layer.self_attn
                sf_attn = SignalFieldAttentionGQA(original_attn, k=k, gamma=gamma, alpha=alpha)
                layer.self_attn = sf_attn
                
                avg_ppl = compute_avg_ppl(model, tokenizer, SEARCH_TEXTS, max_length=128)
                
                layer.self_attn = original_attn
                
                degradation_pct = (avg_ppl - baseline_ppl) / baseline_ppl * 100
                print(f"PPL={avg_ppl:.4f} ({degradation_pct:+.2f}%)")
                
                fine_results.append({
                    'k': k,
                    'alpha': alpha,
                    'gamma': gamma,
                    'avg_ppl': avg_ppl,
                    'degradation_pct': degradation_pct
                })
                
                mx.clear_cache()
                
            except Exception as e:
                print(f"FAILED: {e}")
                mx.clear_cache()
                continue
    
    # Sort fine results
    fine_results.sort(key=lambda x: x['avg_ppl'])
    
    # ============================================================
    # Final Output
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS: Top 10 Parameter Combinations")
    print(f"{'='*70}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Rank':<5} {'k':<5} {'alpha':<8} {'gamma':<8} {'PPL':<12} {'Degradation':<12}")
    print("-" * 60)
    
    for i, r in enumerate(fine_results[:10]):
        deg_str = f"{r['degradation_pct']:+.2f}%"
        print(f"{i+1:<5} {r['k']:<5} {r['alpha']:<8} {r['gamma']:<8} {r['avg_ppl']:<12.4f} {deg_str:<12}")
    
    # ============================================================
    # Save Results
    # ============================================================
    output = {
        'search_config': {
            'model_path': MODEL_PATH,
            'k_values': K_VALUES,
            'alpha_values': ALPHA_VALUES,
            'gamma_values': GAMMA_VALUES,
            'num_layers': NUM_LAYERS,
            'search_texts': SEARCH_TEXTS,
            'mlx_version': mx.__version__
        },
        'baseline_ppl': baseline_ppl,
        'coarse_search_results': coarse_results[:20],
        'fine_search_results': fine_results[:20],
        'final_top_10': [
            {**r, 'rank': i+1} for i, r in enumerate(fine_results[:10])
        ]
    }
    
    output_file = "taicu_7b_ppl_search_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return output


if __name__ == "__main__":
    main()
