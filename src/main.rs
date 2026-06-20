//! Dalin Soma — Signal Field Attention Inference Acceleration Framework
//!
//! 五岳推理加速框架 (Rust Implementation)
//! 零外部依赖, 纯 Rust
//! 论文-代码严格一致

use dalin_soma::config::SomaConfig;
use dalin_soma::engine::{SomaEngine, benchmark_decode_latency};
use dalin_soma::lingya::SomaLingYa;
use dalin_soma::heritage::{SomaHeritage, DistillWeights, ScheduleType, SigmoidDistribution};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       Dalin Soma — Signal Field Attention (Rust)        ║");
    println!("║       五岳推理加速框架 · 论文-代码严格一致                ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let cfg_05b = SomaConfig::qwen_0_5b();
    let cfg_7b = SomaConfig::qwen_7b();

    // ─── 配置参数 (论文 Section 3.1) ────
    println!("═══ 配置参数 (论文 Section 3.1) ═══");
    println!("0.5B: dims={} heads={} kv_heads={} head_dim={} k={} γ={} α={}",
        cfg_05b.dims, cfg_05b.num_heads, cfg_05b.kv_heads,
        cfg_05b.head_dim, cfg_05b.window_size, cfg_05b.gamma, cfg_05b.alpha);
    println!(" 7B: dims={} heads={} kv_heads={} head_dim={} k={} γ={} α={}",
        cfg_7b.dims, cfg_7b.num_heads, cfg_7b.kv_heads,
        cfg_7b.head_dim, cfg_7b.window_size, cfg_7b.gamma, cfg_7b.alpha);
    println!();

    // ─── 参数效率 (论文 Section 2.2.3 & 3.5) ────
    println!("═══ 参数效率 (论文 Section 2.2.3) ═══");
    println!("0.5B: extra_params/layer = {} ≈ {:.1} KB",
        cfg_05b.extra_params_per_layer(),
        cfg_05b.extra_params_per_layer() as f64 * 4.0 / 1024.0);
    println!(" 7B: extra_params/layer = {} ≈ {:.1} KB",
        cfg_7b.extra_params_per_layer(),
        cfg_7b.extra_params_per_layer() as f64 * 4.0 / 1024.0);
    let total_7b = cfg_7b.extra_params_per_layer() * cfg_7b.num_layers;
    println!(" 7B: total signal field params = {} ≈ {:.1} KB (占比 {:.1e})",
        total_7b, total_7b as f64 * 4.0 / 1024.0, total_7b as f64 / 7e9);
    println!();

    // ─── 内存压缩 (论文 Table 5 & 6) ────
    println!("═══ 内存压缩 (论文 Table 5 & 6) ═══");
    let seqs = [128, 512, 1024, 4096, 16384, 65536];
    println!("0.5B Model (kv_heads={} GQA):", cfg_05b.kv_heads);
    println!("{:>8} {:>12} {:>12} {:>10}", "SeqLen", "StdAttn(KB)", "Soma(KB)", "Ratio");
    for &seq in &seqs {
        println!("{:>8} {:>12.1} {:>12.1} {:>10.1}x",
            seq, cfg_05b.standard_kv_memory_kb(seq), cfg_05b.soma_kv_memory_kb(),
            cfg_05b.compression_ratio(seq));
    }
    println!();
    println!("7B @ 64K: Soma={:.0}KB vs Std(f16)={:.0}MB → {:.0}x, Std(f32)={:.0}MB → {:.0}x",
        cfg_7b.soma_kv_memory_kb(),
        4.0 * 65536.0 * 128.0 * 2.0 * 2.0 / 1024.0 / 1024.0,
        4.0 * 65536.0 * 128.0 * 2.0 * 2.0 / 1024.0 / cfg_7b.soma_kv_memory_kb(),
        4.0 * 65536.0 * 128.0 * 2.0 * 4.0 / 1024.0 / 1024.0,
        4.0 * 65536.0 * 128.0 * 2.0 * 4.0 / 1024.0 / cfg_7b.soma_kv_memory_kb());
    println!();

    // ─── 正确性验证 (论文 Table 2) ────
    println!("═══ 正确性验证 (论文 Table 2: α=0 纯近场) ═══");
    let mut cfg_verify = SomaConfig::qwen_0_5b();
    cfg_verify.alpha = 0.0; // 论文3.2: "远场通道权重设置为α=0.0"
    let mut engine_verify = SomaEngine::new(&cfg_verify);
    let seq_lengths = [16, 32, 64, 128, 256, 512, 1024];
    println!("{:>8} {:>10} {:>10} {:>12} {:>12} {:>4}", 
        "SeqLen", "AvgErr", "MaxErr", "AvgSim", "MinSim", "OK");
    for &seq_len in &seq_lengths {
        engine_verify.reset();
        let results = engine_verify.verify_correctness(seq_len);
        for r in &results {
            println!("{}", r);
        }
    }
    println!();

    // ─── Decode O(1) (论文 Table 4) ────
    println!("═══ Decode O(1) 验证 (论文 Table 4) ═══");
    let prefill_lengths = [128, 256, 512, 1024, 2048, 4096];
    let results = benchmark_decode_latency(&cfg_05b, &prefill_lengths, 20);
    println!("{:>8} {:>12} {:>12} {:>8}", "Prefill", "Avg(ms/step)", "Std(ms)", "CV(%)");
    for r in &results {
        println!("{}", r);
    }
    println!();

    // ─── LingYa vs LoRA (论文 Section 2.3) ────
    println!("═══ LingYa vs LoRA (论文 Section 2.3) ═══");
    let ly = SomaLingYa::new(896, 896, 8, 0.1);
    println!("LingYa params: {} | LoRA params: {} | Savings: {:.1}%",
        ly.trainable_params(), ly.lora_params(), ly.savings_vs_lora());
    println!();

    // ─── 华岳 S型层分配 (v7: 71%替换 + 29%保留) ────
    println!("═══ 华岳 S型层分配 (v7: 71%+29%) ═══");
    let sig_dist = SigmoidDistribution::v7_default();
    println!("28层替换概率分布 (Sigmoid):");
    println!("{:>6} {:>8} {:>6} {:>6}", "Layer", "Prob", "Type", "Depth");
    let mask = SigmoidDistribution::v7_28layer_mask();
    for l in 0..28 {
        let prob = sig_dist.replace_probability(l, 28);
        let layer_type = if mask[l] { "SFA" } else { "ATTN" };
        let depth = l as f32 / 27.0;
        println!("{:>6} {:>8.3} {:>6} {:>6.2}", l, prob, layer_type, depth);
    }
    let attn_count = mask.iter().filter(|&&x| !x).count();
    let sfa_count = mask.iter().filter(|&&x| x).count();
    println!("保留: {}/{} ({:.0}%) | 替换: {}/{} ({:.0}%)",
        attn_count, 28, attn_count as f32 / 28.0 * 100.0,
        sfa_count, 28, sfa_count as f32 / 28.0 * 100.0);
    println!();

    // ─── Heritage 渐进替换 (论文 Section 2.6) ────
    println!("═══ Heritage 渐进替换 (论文 Section 2.6) ═══");
    println!("{:>8} {:>8} {:>8} {:>8}", "Step", "Ratio", "Retain", "Replace");
    for &step in &[0, 1000, 2000, 5000, 7000, 9000, 10000] {
        let mut heritage = SomaHeritage::new(28, 10000, DistillWeights::default(),
            ScheduleType::Sigmoid { steepness: 10.0 });
        for _ in 0..step { heritage.advance(); }
        let (retain, replace) = heritage.layer_stats();
        println!("{:>8} {:>8.3} {:>8} {:>8}", step, heritage.current_ratio(), retain, replace);
    }
    println!();

    println!("═══ 五岳就位 · 论文-代码一致 · 可开源 ═══");
}
