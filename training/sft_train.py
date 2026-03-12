#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) 训练脚本

使用 TRL (Transformers Reinforcement Learning) 库进行训练。
支持 LoRA 高效微调。

关于你的问题：
1. SFT 默认会改变原始参数（全参数训练）
2. 使用 LoRA 可以只训练 Adapter，冻结原参数
3. 本项目建议使用 LoRA 节省显存

使用方法:
    python training/sft_train.py --config configs/sft_config.yaml
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
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict):
    """
    加载模型和 tokenizer
    
    回答你的问题:
    - 默认加载会修改所有参数
    - 配合 LoRA 使用可以冻结原参数，只训练 Adapter
    """
    model_config = config['model']
    
    print(f"加载模型: {model_config['base_model']}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['base_model'],
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 设置 pad token（Qwen 系列通常有 eos_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 应用 LoRA (如果需要)
    if model_config.get('use_lora', False):
        print("应用 LoRA 配置...")
        lora_config = config['lora']
        
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()  # 打印可训练参数比例
    
    return model, tokenizer


def prepare_dataset(config: dict, tokenizer):
    """准备数据集"""
    data_config = config['data']
    
    # 加载数据
    print(f"加载训练数据: {data_config['train_file']}")
    
    # 使用 datasets 库加载 JSON 数据
    dataset = load_dataset('json', data_files={
        'train': data_config['train_file'],
        'eval': data_config['eval_file']
    })
    
    # 格式化函数 (根据 template 类型)
    def format_prompt(example):
        """格式化 prompt 为 ChatML 格式 (Qwen 默认)"""
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']
        
        if input_text:
            prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        # 完整文本（用于训练）
        full_text = prompt + output + "<|im_end|>"
        
        return {
            'text': full_text,
            'prompt': prompt,
            'completion': output
        }
    
    # 应用格式化
    dataset = dataset.map(format_prompt)
    
    return dataset


def train(config: dict, model, tokenizer, dataset):
    """执行训练"""
    training_config = config['training']
    
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
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        fp16=training_config['fp16'],
        seed=training_config['seed'],
        report_to=training_config.get('report_to', ['tensorboard']),
        remove_unused_columns=False,
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        args=training_args,
        data_collator=data_collator,
        max_seq_length=config['data']['max_seq_length'],
        dataset_text_field='text',
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存最终模型
    print(f"\n保存模型到: {training_config['output_dir']}")
    trainer.save_model(training_config['output_dir'])
    tokenizer.save_pretrained(training_config['output_dir'])
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/sft_config.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    # 加载配置
    print("=" * 60)
    print("SFT 训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    
    config = load_config(args.config)
    
    # 打印关键配置
    print("\n关键配置:")
    print(f"  基础模型: {config['model']['base_model']}")
    print(f"  使用 LoRA: {config['model'].get('use_lora', False)}")
    print(f"  学习率: {config['training']['learning_rate']}")
    print(f"  训练轮数: {config['training']['num_train_epochs']}")
    print(f"  输出目录: {config['training']['output_dir']}")
    
    # 加载模型
    print("\n[1/3] 加载模型和 tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 准备数据
    print("\n[2/3] 准备数据集...")
    dataset = prepare_dataset(config, tokenizer)
    print(f"  训练集: {len(dataset['train'])} 条")
    print(f"  验证集: {len(dataset['eval'])} 条")
    
    # 训练
    print("\n[3/3] 开始训练...")
    trainer = train(config, model, tokenizer, dataset)
    
    print("\n" + "=" * 60)
    print("SFT 训练完成")
    print("=" * 60)
    
    # 说明
    print("\n关于 SFT 的重要说明:")
    print("1. 如果使用了 LoRA，原模型参数被冻结，只保存了 Adapter")
    print("2. 推理时需要合并 Adapter 或动态加载")
    print("3. SFT 模型将作为 DPO 的基础模型")


if __name__ == "__main__":
    main()
