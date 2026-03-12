#!/usr/bin/env python3
"""
训练工具函数

包含模型加载、数据预处理、训练辅助函数等
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_for_inference(
    model_path: str,
    use_lora: bool = False,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False
):
    """
    加载模型用于推理
    
    Args:
        model_path: 基础模型路径
        use_lora: 是否使用 LoRA
        adapter_path: LoRA adapter 路径
        load_in_8bit: 是否 8bit 量化加载
    
    Returns:
        model, tokenizer
    """
    print(f"加载模型: {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    load_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16,
        'device_map': 'auto'
    }
    
    if load_in_8bit:
        load_kwargs['load_in_8bit'] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs
    )
    
    # 加载 LoRA Adapter
    if use_lora and adapter_path:
        print(f"加载 LoRA Adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # 可选：合并 adapter 以加速推理
        # model = model.merge_and_unload()
    
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> str:
    """
    生成回答
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_new_tokens: 最大生成长度
        temperature: 温度
        top_p: nucleus sampling 参数
        repetition_penalty: 重复惩罚
    
    Returns:
        生成的文本
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text


def format_chatml_prompt(instruction: str, input_text: str = "") -> str:
    """
    格式化为 ChatML 格式 (Qwen 默认格式)
    
    Args:
        instruction: 指令
        input_text: 可选输入
    
    Returns:
        格式化后的 prompt
    """
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}的事情稍后再说。<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}的事情稍后再说。<|im_start|>assistant\n"
    
    return prompt


def count_model_parameters(model) -> Dict[str, int]:
    """
    统计模型参数
    
    Returns:
        {
            'total': 总参数,
            'trainable': 可训练参数,
            'frozen': 冻结参数
        }
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_percent': 100 * trainable / total if total > 0 else 0
    }


def save_training_info(output_dir: str, info: Dict[str, Any]):
    """保存训练信息"""
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    info_path = output_dir / "training_info.json"
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"训练信息已保存: {info_path}")
