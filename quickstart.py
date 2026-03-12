#!/usr/bin/env python3
"""
快速开始脚本

一键运行完整实验流程的示例脚本。

使用方法:
    python quickstart.py --stage all
    
    # 或单独运行某一步
    python quickstart.py --stage data
    python quickstart.py --stage sft
    python quickstart.py --stage dpo
    python quickstart.py --stage eval
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ 失败: {description}")
        return False
    
    print(f"\n✅ 完成: {description}")
    return True


def stage_data():
    """步骤 1-2: 准备数据和 Benchmark"""
    steps = [
        ([sys.executable, "data/scripts/build_sft_data.py"], "构造 SFT 数据"),
        ([sys.executable, "data/scripts/build_dpo_data.py"], "构造 DPO 数据"),
        ([sys.executable, "data/scripts/prepare_benchmark.py"], "准备 Benchmark"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            return False
    
    return True


def stage_sft():
    """步骤 3: SFT 训练"""
    return run_command(
        [sys.executable, "training/sft_train.py", "--config", "configs/sft_config.yaml"],
        "SFT 训练"
    )


def stage_dpo():
    """步骤 4: DPO 训练"""
    return run_command(
        [sys.executable, "training/dpo_train.py", "--config", "configs/dpo_config.yaml"],
        "DPO 训练"
    )


def stage_eval():
    """步骤 5-7: 评估全流程"""
    steps = [
        ([sys.executable, "evaluation/run_benchmark.py", "--config", "configs/eval_config.yaml"],
         "运行 Benchmark"),
        ([sys.executable, "evaluation/gpt4_judge.py", "--input", "results/raw_outputs/"],
         "GPT-4 评判"),
        ([sys.executable, "evaluation/eval_metrics.py", "--input", "results/judgments/"],
         "计算评估指标"),
        ([sys.executable, "evaluation/visualize.py", "--report", "results/reports/evaluation_report.json"],
         "生成可视化图表"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Quick Start")
    parser.add_argument(
        '--stage',
        choices=['all', 'data', 'sft', 'dpo', 'eval'],
        default='all',
        help='运行阶段'
    )
    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='如果输出已存在则跳过'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("医疗场景模型安全性对齐训练 - 快速开始")
    print("="*60)
    
    stages = {
        'data': stage_data,
        'sft': stage_sft,
        'dpo': stage_dpo,
        'eval': stage_eval,
    }
    
    if args.stage == 'all':
        # 按顺序运行所有阶段
        for name, func in stages.items():
            if not func():
                print(f"\n❌ 在阶段 '{name}' 失败，停止执行")
                return 1
        
        print("\n" + "="*60)
        print("🎉 所有阶段完成！")
        print("="*60)
        print("\n结果位置:")
        print("  - 模型: models/{base,sft,dpo}/")
        print("  - 评估结果: results/reports/evaluation_report.json")
        print("  - 可视化: results/reports/*.png")
        print("\n查看 Jupyter Notebook: notebooks/analysis.ipynb")
        
    else:
        # 运行指定阶段
        if not stages[args.stage]():
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
