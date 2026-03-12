#!/usr/bin/env python3
"""
Run local inference with a base model (e.g. Qwen3-0.6B) on:
- the SFT validation split (chat-style messages), or
- arbitrary JSONL benchmark files,
and save the model outputs for later comparison with SFT/DPO 等版本。

Example usages
--------------

1) 在 SFT 验证集上跑 base 模型：

    python data/scripts/run_local_infer.py \\
      --model-path /Users/xieyun/models/Qwen3-0.6B \\
      --input data/processed/sft_train_v1.val.jsonl \\
      --output data/processed/base_qwen3_0.6b_sft_val_outputs.jsonl \\
      --input-format sft

2) 在 benchmark 上跑 base 模型（假设每行有字段 `prompt` 或 `input`）：

    python data/scripts/run_local_infer.py \\
      --model-path /Users/xieyun/models/Qwen3-0.6B \\
      --input evaluation/benchmarks/patientsafetybench_eval.jsonl \\
      --output data/processed/base_qwen3_0.6b_benchmark_outputs.jsonl \\
      --input-format plain

设计要点
--------
- 只依赖 transformers / torch，不调用任何远程 API。
- 兼容 Qwen3-0.6B 这类 chat 模型：优先使用 tokenizer.apply_chat_template 构建 prompt。
- 对 SFT 格式（messages 数组：system / user / assistant），只把 system+user 喂给模型，
  把 assistant 作为参考，用来对比生成质量。
- 对 benchmark/普通 JSONL：尝试用 `prompt` / `input` / `question` 字段构造文本输入。
- 支持 max-samples / batch-size / max-new-tokens / temperature 等参数。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path, max_samples: int | None = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data.append(obj)
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


def build_prompt_from_sft_messages(
    sample: Dict[str, Any],
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool = True,
) -> str:
    """
    From an SFT sample with messages[system, user, assistant], build a chat prompt
    containing only system+user, leaving assistant for evaluation.
    Prefer tokenizer.apply_chat_template when available.
    """
    messages = sample.get("messages") or []
    chat: List[Dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        if role == "assistant":
            # do not include target assistant text in the input prompt
            continue
        if role in ("system", "user"):
            chat.append({"role": role, "content": m.get("content") or ""})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback: simple concatenation
    parts = []
    for m in chat:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def build_prompt_from_plain(sample: Dict[str, Any]) -> str:
    """
    For non-SFT formats, try to construct a prompt from common fields.
    Priority: 'prompt' -> 'instruction' + 'input' -> 'input' -> 'question'.
    """
    if "prompt" in sample and isinstance(sample["prompt"], str):
        return sample["prompt"]
    instr = sample.get("instruction", "")
    inp = sample.get("input", "")
    if instr or inp:
        if instr and inp:
            return instr.rstrip() + "\n\n" + inp.lstrip()
        return instr or inp
    if "question" in sample and isinstance(sample["question"], str):
        return sample["question"]
    # Fallback: best-effort string of the sample
    return json.dumps(sample, ensure_ascii=False)


def prepare_prompts(
    samples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    input_format: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Build (prompt, meta) pairs from samples.
    meta 至少包含 'id'（如果原 sample 有的话）。
    """
    prompts: List[Tuple[str, Dict[str, Any]]] = []
    for s in samples:
        _id = s.get("id")
        meta = {"id": _id} if _id is not None else {}
        if input_format == "sft":
            prompt = build_prompt_from_sft_messages(s, tokenizer)
            # also携带参考 answer 便于后续对比
            msgs = s.get("messages") or []
            ref_answer = ""
            for m in msgs:
                if (m.get("role") or "").strip().lower() == "assistant":
                    ref_answer = m.get("content") or ""
                    break
            meta["ref_answer"] = ref_answer
        else:
            prompt = build_prompt_from_plain(s)
        prompts.append((prompt, meta))
    return prompts


def batched(iterable, batch_size: int):
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_inference(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    input_format: str = "sft",
    max_samples: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str | None = None,
) -> None:
    # Load data
    samples = load_jsonl(input_path, max_samples=max_samples)
    total = len(samples)
    print(f"Loaded {total} samples from {input_path}")

    # Load model and tokenizer
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device is None else None,
    )
    if device is not None:
        model.to(device)
    model.eval()

    prompts_with_meta = prepare_prompts(samples, tokenizer, input_format=input_format)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for batch in batched(prompts_with_meta, batch_size):
            texts = [p for p, _ in batch]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if device is not None:
                enc = {k: v.to(device) for k, v in enc.items()}
            else:
                enc = {k: v.to(model.device) for k, v in enc.items()}

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the generated part
            input_lens = enc["input_ids"].shape[1]
            gen_texts = tokenizer.batch_decode(
                gen[:, input_lens:], skip_special_tokens=True
            )

            for (_, meta), gen_text in zip(batch, gen_texts):
                record: Dict[str, Any] = dict(meta)
                record["model_output"] = gen_text.strip()
                record["model_path"] = str(model_path)
                record["input_format"] = input_format
                record["max_new_tokens"] = max_new_tokens
                record["temperature"] = temperature
                record["top_p"] = top_p
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote outputs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local inference with a base model on SFT val or benchmark JSONL."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to local model (e.g., /Users/xieyun/models/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL path (SFT val or benchmark)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path to save model generations",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["sft", "plain"],
        default="sft",
        help="Input format: 'sft' for messages[system,user,assistant], 'plain' for generic benchmark JSONL.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of samples to run (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. 'cuda', 'mps', or 'cpu'. Default: auto/device_map='auto'.",
    )
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        input_path=args.input,
        output_path=args.output,
        input_format=args.input_format,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )


if __name__ == "__main__":
    main()

