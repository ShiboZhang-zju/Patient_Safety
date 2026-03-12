#!/usr/bin/env python3
"""
结果可视化脚本

生成图表展示评估结果。
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_report(report_path: Path) -> dict:
    """加载评估报告"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_dimension_comparison(report: dict, output_dir: Path):
    """绘制各维度对比图"""
    scores = report['dimension_scores']
    models = list(scores.keys())
    dimensions = ['safety', 'accuracy', 'helpfulness', 'harmlessness', 'medical_correctness', 'overall']
    
    # 准备数据
    data = []
    for model in models:
        for dim in dimensions:
            data.append({
                'Model': model,
                'Dimension': dim,
                'Score': scores[model].get(dim, 0)
            })
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(dimensions))
    width = 0.25
    
    for i, model in enumerate(models):
        values = [scores[model].get(dim, 0) for dim in dimensions]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison by Dimension')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dimension_comparison.png', dpi=300)
    print(f"保存: {output_dir / 'dimension_comparison.png'}")


def plot_safety_violations(report: dict, output_dir: Path):
    """绘制安全违规率对比图"""
    violations = report['safety_violations']
    models = list(violations.keys())
    rates = [violations[m] * 100 for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['green' if r < 10 else 'orange' if r < 30 else 'red' for r in rates]
    bars = ax.bar(models, rates, color=colors)
    
    ax.set_ylabel('Safety Violation Rate (%)')
    ax.set_title('Safety Violation Rate by Model')
    ax.set_ylim(0, max(rates) * 1.2)
    
    # 添加数值标签
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'safety_violations.png', dpi=300)
    print(f"保存: {output_dir / 'safety_violations.png'}")


def plot_radar_chart(report: dict, output_dir: Path):
    """绘制雷达图"""
    from math import pi
    
    scores = report['dimension_scores']
    models = list(scores.keys())
    dimensions = ['safety', 'accuracy', 'helpfulness', 'harmlessness', 'medical_correctness']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # 计算角度
    angles = [n / float(len(dimensions)) * 2 * pi for n in range(len(dimensions))]
    angles += angles[:1]
    
    # 绘制每个模型
    for model in models:
        values = [scores[model].get(dim, 0) for dim in dimensions]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.set_title('Model Performance Radar', y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png', dpi=300)
    print(f"保存: {output_dir / 'radar_chart.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Results")
    parser.add_argument('--report', type=str, required=True, help='评估报告路径')
    parser.add_argument('--output', type=str, help='输出目录')
    args = parser.parse_args()
    
    report_path = Path(args.report)
    output_dir = Path(args.output) if args.output else report_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    # 加载报告
    report = load_report(report_path)
    
    # 设置样式
    sns.set_style("whitegrid")
    
    # 生成图表
    print("\n生成图表...")
    plot_dimension_comparison(report, output_dir)
    plot_safety_violations(report, output_dir)
    plot_radar_chart(report, output_dir)
    
    print("\n" + "=" * 60)
    print("可视化完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
