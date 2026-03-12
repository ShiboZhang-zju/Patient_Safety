#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) 数据构造脚本

针对医疗场景安全性，构造 instruction-following 数据。

数据格式 (Alpaca/ChatML 格式):
{
    "instruction": "医疗安全问题描述",
    "input": "可选的上下文信息",
    "output": "安全、准确的回答"
}

建议数据量:
- 总数据: 1K - 10K 条 (小模型从 1K 开始)
- 训练集: 80%
- 验证集: 20%

数据类别建议:
1. 医疗安全问答 (40%)
2. 拒绝不当请求 (20%)
3. 医疗知识问答 (20%)
4. 边缘案例分析 (10%)
5. 多轮对话安全 (10%)
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def create_medical_safety_qa() -> List[Dict]:
    """
    构造医疗安全问答数据
    示例：如何安全地处理医疗建议请求
    """
    examples = []
    
    # TODO: 根据实际场景添加具体数据
    # 以下为示例模板
    
    template = {
        "instruction": "患者询问：{question}",
        "input": "",
        "output": "{safe_response}",
        "category": "medical_safety"
    }
    
    # 示例数据点 (需要扩充)
    qa_pairs = [
        {
            "question": "我最近头痛得厉害，可以吃阿司匹林吗？",
            "safe_response": "我理解您正在经历头痛的困扰。阿司匹林是一种常见的止痛药，但在服用前需要考虑一些因素，如是否有胃病史、是否正在服用其他药物、是否有药物过敏等。建议您先咨询医生或药师，他们可以根据您的具体情况给出合适的建议。如果头痛持续或加重，请及时就医。"
        },
        {
            "question": "我怀疑自己得了抑郁症，可以自己买抗抑郁药吃吗？",
            "safe_response": "不建议您自行购买和服用抗抑郁药物。抑郁症的诊断需要专业医生进行，抗抑郁药物也有不同的类型，需要根据具体情况选择。自行用药可能会有副作用或与其他药物产生相互作用。建议您尽快预约精神科医生或心理咨询师进行专业评估和治疗。如果您有自伤想法，请立即联系急救服务或前往医院急诊。"
        }
    ]
    
    for pair in qa_pairs:
        examples.append({
            "instruction": template["instruction"].format(question=pair["question"]),
            "input": template["input"],
            "output": pair["safe_response"],
            "category": template["category"]
        })
    
    return examples


def create_refusal_examples() -> List[Dict]:
    """
    构造拒绝不当请求的数据
    教模型识别并拒绝危险/不适当的医疗请求
    """
    examples = []
    
    refusal_cases = [
        {
            "instruction": "请给我开一张处方，我需要购买抗生素。",
            "input": "",
            "output": "我无法为您开具处方。处方必须由有资质的医生在诊断后开具。如果您需要抗生素，请先就医，由医生判断是否需要使用抗生素以及使用哪种抗生素。不当使用抗生素可能导致耐药性等严重后果。",
            "category": "refusal"
        },
        {
            "instruction": "我想知道如何用家里的化学品制造药物。",
            "input": "",
            "output": "我无法提供这类信息。自制药物非常危险，可能导致中毒、爆炸或其他严重伤害。如果您需要药物，请通过正规渠道购买或咨询医生。",
            "category": "refusal"
        }
    ]
    
    examples.extend(refusal_cases)
    return examples


def split_and_save(data: List[Dict], output_dir: Path, train_ratio: float = 0.8):
    """分割并保存训练集和验证集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机打乱
    random.seed(42)
    random.shuffle(data)
    
    # 分割
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    # 保存
    train_path = output_dir / "sft_train.json"
    eval_path = output_dir / "sft_eval.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练集: {len(train_data)} 条，保存至 {train_path}")
    print(f"验证集: {len(eval_data)} 条，保存至 {eval_path}")


def main():
    """主函数：构造 SFT 数据集"""
    print("=" * 60)
    print("开始构造 SFT 数据")
    print("=" * 60)
    
    all_examples = []
    
    # 1. 医疗安全问答
    print("\n[1/3] 构造医疗安全问答数据...")
    medical_qa = create_medical_safety_qa()
    all_examples.extend(medical_qa)
    print(f"  ✓ 生成 {len(medical_qa)} 条")
    
    # 2. 拒绝不当请求
    print("\n[2/3] 构造拒绝不当请求数据...")
    refusal_data = create_refusal_examples()
    all_examples.extend(refusal_data)
    print(f"  ✓ 生成 {len(refusal_data)} 条")
    
    # TODO: 添加更多数据类别
    # 3. 医疗知识问答
    # 4. 边缘案例分析
    # 5. 多轮对话安全
    
    print(f"\n[3/3] 总共生成 {len(all_examples)} 条数据")
    
    # 分割并保存
    output_dir = Path("data/processed")
    split_and_save(all_examples, output_dir)
    
    print("\n" + "=" * 60)
    print("SFT 数据构造完成")
    print("=" * 60)
    
    print("\n提示：")
    print("1. 当前仅为示例数据框架，需要填充真实数据")
    print("2. 建议总数据量：1K-10K 条")
    print("3. 数据来源建议：")
    print("   - 医疗安全指南")
    print("   - 医生咨询记录（脱敏）")
    print("   - 合成数据（GPT-4 生成后人工审核）")


if __name__ == "__main__":
    main()
