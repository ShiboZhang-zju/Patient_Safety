#!/usr/bin/env python3
"""
Benchmark 运行脚本

并发运行 Base / SFT / DPO 三个模型，生成回答。

使用方法:
    python evaluation/run_benchmark.py --config configs/eval_config.yaml
    
    或
    
    python evaluation/run_benchmark.py \
        --models base sft dpo \
        --benchmark evaluation/benchmarks/medical_safety.json \
        --output results/raw_outputs/
"""

import os
import sys
import yaml
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import concurrent.futures

sys.path.append(str(Path(__file__).parent.parent))
from training.utils import load_model_for_inference, generate_response, format_chatml_prompt


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_benchmark(benchmark_path: str) -> List[Dict]:
    """加载 benchmark 数据"""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式
    if 'questions' in data:
        return data['questions']
    return data


def run_model_on_benchmark(
    model_name: str,
    model_config: Dict,
    benchmark: List[Dict],
    generation_config: Dict
) -> List[Dict]:
    """
    运行单个模型在 benchmark 上
    
    Args:
        model_name: 模型标识名
        model_config: 模型配置
        benchmark: benchmark 问题列表
        generation_config: 生成参数
    
    Returns:
        带有回答的结果列表
    """
    print(f"\n加载模型: {model_name}")
    
    # 加载模型
    model, tokenizer = load_model_for_inference(
        model_path=model_config['path'],
        use_lora=model_config.get('use_lora', False),
        adapter_path=model_config.get('adapter_path')
    )
    
    results = []
    
    print(f"生成回答: {model_name}")
    for item in tqdm(benchmark, desc=model_name):
        question = item['question']
        
        # 格式化 prompt
        prompt = format_chatml_prompt(question)
        
        # 生成回答
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=generation_config.get('max_new_tokens', 512),
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.9),
            repetition_penalty=generation_config.get('repetition_penalty', 1.1)
        )
        
        results.append({
            'id': item.get('id', 'unknown'),
            'question': question,
            'category': item.get('category', 'unknown'),
            'model': model_name,
            'response': response,
            'metadata': {
                'expected_behavior': item.get('expected_behavior', ''),
                'keywords': item.get('keywords', []),
                'difficulty': item.get('difficulty', 'unknown')
            }
        })
    
    # 清理 GPU 内存
    del model
    torch.cuda.empty_cache()
    
    return results


def run_all_models(config: dict) -> Dict[str, List[Dict]]:
    """运行所有模型"""
    models_config = config['models']
    benchmark_path = config['benchmark']['data_path']
    generation_config = config['generation']
    
    # 加载 benchmark
    print(f"加载 benchmark: {benchmark_path}")
    benchmark = load_benchmark(benchmark_path)
    
    # 限制样本数（如果需要）
    if config['benchmark'].get('num_samples', -1) > 0:
        benchmark = benchmark[:config['benchmark']['num_samples']]
    
    print(f"共 {len(benchmark)} 个问题")
    
    # 运行每个模型
    all_results = {}
    
    for model_name, model_config in models_config.items():
        print(f"\n{'='*60}")
        print(f"运行模型: {model_name}")
        print(f"{'='*60}")
        
        results = run_model_on_benchmark(
            model_name=model_name,
            model_config=model_config,
            benchmark=benchmark,
            generation_config=generation_config
        )
        
        all_results[model_name] = results
    
    return all_results


def save_results(results: Dict[str, List[Dict]], output_dir: Path):
    """保存结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个模型单独保存
    for model_name, model_results in results.items():
        output_path = output_dir / f"{model_name}_responses.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)
        print(f"  保存: {output_path}")
    
    # 保存汇总
    summary = {
        'models': list(results.keys()),
        'num_questions': len(next(iter(results.values()))),
        'results_by_model': {
            name: len(res) for name, res in results.items()
        }
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n汇总保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Benchmark")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='指定要运行的模型 (覆盖配置文件)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录 (覆盖配置文件)'
    )
    args = parser.parse_args()
    
    # 加载配置
    print("=" * 60)
    print("Benchmark 评估")
    print("=" * 60)
    
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.models:
        # 过滤只保留指定的模型
        config['models'] = {
            k: v for k, v in config['models'].items()
            if k in args.models
        }
    
    output_dir = args.output or config['output']['raw_outputs_dir']
    
    # 打印信息
    print(f"\n评估配置:")
    print(f"  Benchmark: {config['benchmark']['data_path']}")
    print(f"  模型: {list(config['models'].keys())}")
    print(f"  输出目录: {output_dir}")
    
    # 运行
    print("\n开始评估...")
    results = run_all_models(config)
    
    # 保存结果
    print("\n保存结果...")
    save_results(results, output_dir)
    
    print("\n" + "=" * 60)
    print("Benchmark 运行完成")
    print("=" * 60)
    print(f"\n下一步: 运行 GPT-4 评判")
    print(f"  python evaluation/gpt4_judge.py --input {output_dir}")


if __name__ == "__main__":
    main()
