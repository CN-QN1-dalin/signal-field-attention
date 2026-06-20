#!/usr/bin/env python3
"""
Signal Field Attention 8 Experiment Suites — 总测试入口

运行: python3 run_all.py
"""

import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    ("01-signal-field", "信号场 v5d — 零训练注意力替换", "signal_field.py"),
    ("02-huayue", "华岳 — 零训练混合架构", "huayue.py"),
    ("03-guiyuan", "归元v2 — SSM KV压缩", "guiyuan.py"),
    ("04-lingya", "灵芽 — 正交基微调", "lingya.py"),
    ("05-ring-buffer", "RingBuffer — O(1) KV Cache", "ring_buffer.py"),
    ("06-rca", "RCA — 频域注意力", "rca.py"),
    ("07-metal-kernel", "Metal — 直接GPU内核", "metal_kernel.py"),
    ("08-ultra", "Ultra — 通用模型部署", "ultra.py"),
]


def run_experiment(exp_dir: str, exp_name: str, script: str) -> bool:
    """运行单个实验"""
    print(f"\n{'=' * 60}")
    print(f"📦 实验: {exp_name}")
    print(f"   目录: {exp_dir}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [sys.executable, f"{exp_dir}/{script}"],
            capture_output=False,
            text=True,
            cwd=".",
            timeout=300,  # 5分钟超时
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"\n  ⚠️ 实验超时 (300s)")
        return False
    except Exception as e:
        print(f"\n  ⚠️ 实验异常: {e}")
        return False


def main():
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "scripts"

    # 创建scripts目录
    scripts_dir.mkdir(exist_ok=True)

    all_passed = True
    results = []

    for exp_dir, exp_name, script in EXPERIMENTS:
        exp_path = base_dir / exp_dir
        if not exp_path.exists():
            print(f"\n⚠️  实验目录不存在: {exp_dir}")
            all_passed = False
            results.append((exp_name, False))
            continue

        passed = run_experiment(str(exp_path), exp_name, script)
        results.append((exp_name, passed))
        if not passed:
            all_passed = False

    # 汇总
    print("\n" + "=" * 60)
    print("📊 实验汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"\n总计: {passed_count}/{total} 通过")

    if all_passed:
        print("\n🎉 所有实验通过！")
        print("\n下一步:")
        print("  1. 运行 ./scripts/01-*.sh 生成benchmark报告")
        print("  2. 运行 ./scripts/02-*.sh 生成精度报告")
        print("  3. 运行 ./scripts/03-*.sh 生成论文数据")
    else:
        print("\n⚠️  部分实验失败，请检查日志")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
