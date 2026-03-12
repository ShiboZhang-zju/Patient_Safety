#!/usr/bin/env python3
"""
SFT data generation: read seed, expand via teacher API, validate, and write.

Flow:
  1. Load seed JSONL and select samples to expand
  2. For each seed: get category, prompt template, random variation_hint
  3. Build teacher input (instruction + system prompt + one seed example)
  4. Call teacher API → parse JSON → validate → rewrite id → write
  5. Retry on failure; flush to file periodically

Usage:
  python generate_sft.py --seed medical_sft_100_english.jsonl --out data/processed/generated_round1.jsonl
  python generate_sft.py --seed seed.jsonl --out out.jsonl --max-samples 20 --n-per-seed 3 --flush-every 5
  Failed runs are logged to <out_stem>.errors.jsonl (seed_id, category, error, raw_response).

  Alibaba Cloud (Qwen as teacher):
    export DASHSCOPE_API_KEY=sk-xxx
    python generate_sft.py --provider aliyun --seed seed.jsonl --out out.jsonl
    # Optional: DASHSCOPE_BASE_URL (default Beijing), --model qwen-turbo | qwen-plus | qwen-max
"""

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env from project root if present (so env vars in .env.example / .env are applied)
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parent.parent.parent  # data/scripts -> project root
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

# Allow importing prompts from same directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from prompts import (
    CATEGORIES,
    SYSTEM_PROMPT,
    get_full_generation_instruction,
    VARIATION_HINTS,
)

VALID_CATEGORIES = set(CATEGORIES.keys())

# Default base URL for Alibaba DashScope (Qwen) OpenAI-compatible API
DASHSCOPE_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# General-class weak filter: if assistant contains these (case-insensitive), flag for review
GENERAL_FLAG_TRIGGERS = [
    "likely", "diagnosis", "viral", "orthostatic", "reflux",
    "esophageal sphincter", "calcium carbonate", "ibuprofen", "acetaminophen",
    "gas-x", "simethicone",
]


def flag_general_review(assistant_content: str) -> list[str]:
    """Return list of trigger substrings found in assistant (general class). Empty = no flag."""
    content_lower = (assistant_content or "").lower()
    return [t for t in GENERAL_FLAG_TRIGGERS if t in content_lower]


def load_seed(path: Path) -> list[dict]:
    """Load seed samples from JSONL."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skip line {i}: {e}", file=sys.stderr)
    return samples


def get_user_content(messages: list) -> str:
    for m in messages or []:
        if isinstance(m, dict) and (m.get("role") or "").strip().lower() == "user":
            return (m.get("content") or "").strip()
    return ""


def get_assistant_content(messages: list) -> str:
    for m in messages or []:
        if isinstance(m, dict) and (m.get("role") or "").strip().lower() == "assistant":
            return (m.get("content") or "").strip()
    return ""


def build_teacher_user_message(seed: dict, category: str, variation_hint: str) -> str:
    """Build the user message sent to the teacher model: instruction + system prompt + one seed example."""
    instruction = get_full_generation_instruction(
        category,
        include_output_constraints=True,
        include_diversity=True,
        variation_hint=variation_hint,
    )
    messages = seed.get("messages") or []
    user_text = get_user_content(messages)
    assistant_text = get_assistant_content(messages)
    seed_snippet = f"User: {user_text}\nAssistant: {assistant_text}"

    return (
        instruction
        + "\n\n---\nSystem prompt to copy exactly into messages[0].content:\n"
        + SYSTEM_PROMPT
        + "\n\n---\nOne seed example (create a new sample that is meaningfully different in wording, details, and scenario):\n"
        + seed_snippet
    )


def call_teacher_api(
    user_content: str,
    *,
    api_key: str,
    base_url: str | None,
    model: str,
) -> str:
    """Call OpenAI-compatible chat API; return assistant content."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package required for API calls: pip install openai") from None

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    messages = [
        {"role": "system", "content": "You generate exactly one SFT sample in JSON format per request. Output only the JSON object, no other text."},
        {"role": "user", "content": user_content},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    content = (resp.choices[0].message.content or "").strip()
    return content


def extract_json_from_text(text: str) -> str:
    """Strip markdown code fences or extract first { ... } block."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


def validate_sample(obj: dict, expected_category: str) -> tuple[bool, str]:
    """Validate parsed object: structure, category match, non-empty user/assistant, assistant min length. Returns (ok, error_msg)."""
    if not isinstance(obj, dict):
        return False, "not a dict"

    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return False, "missing or invalid messages"
    if len(msgs) != 3:
        return False, f"messages length must be 3, got {len(msgs)}"

    roles = [(m.get("role") or "").strip().lower() for m in msgs if isinstance(m, dict)]
    if roles != ["system", "user", "assistant"]:
        return False, f"roles must be [system, user, assistant], got {roles}"

    system_content = (msgs[0].get("content") or "").strip() if isinstance(msgs[0], dict) else ""
    if system_content != SYSTEM_PROMPT:
        return False, "messages[0].content must exactly equal SYSTEM_PROMPT"

    cat = (obj.get("category") or "").strip().lower()
    if cat != expected_category:
        return False, f"category mismatch: expected {expected_category}, got {cat}"

    user_content = (msgs[1].get("content") or "").strip() if isinstance(msgs[1], dict) else ""
    assistant_content = (msgs[2].get("content") or "").strip() if isinstance(msgs[2], dict) else ""

    if not user_content:
        return False, "empty user content"
    if not assistant_content:
        return False, "empty assistant content"
    if len(assistant_content) < 80:
        return False, "assistant content too short"

    return True, ""


def rewrite_id(sample: dict, new_id: str) -> dict:
    """Return copy with id set to new_id."""
    return {**sample, "id": new_id}


def generate_one(
    seed: dict,
    category: str,
    variation_hint: str,
    next_id: str,
    api_key: str,
    base_url: str | None,
    model: str,
) -> tuple[dict | None, str, str]:
    """
    Generate one new sample from seed. Returns (sample_dict, error_msg, raw_response).
    On success error_msg is empty and raw_response is empty. On failure raw_response is the API reply for logging.
    """
    user_msg = build_teacher_user_message(seed, category, variation_hint)
    try:
        full_raw = call_teacher_api(user_msg, api_key=api_key, base_url=base_url, model=model)
    except Exception as e:
        return None, str(e), ""
    raw = extract_json_from_text(full_raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSON decode: {e}", full_raw
    ok, err = validate_sample(obj, category)
    if not ok:
        return None, f"validation: {err}", full_raw
    return rewrite_id(obj, next_id), "", ""


def generate_with_retries(
    seed: dict,
    category: str,
    variation_hint: str,
    next_id: str,
    api_key: str,
    base_url: str | None,
    model: str,
    max_retries: int,
) -> tuple[dict | None, str, str]:
    """
    Wrapper for threaded use: perform up to max_retries attempts.
    Returns (sample_dict_or_None, error_msg, raw_response_for_logging).
    """
    last_err = ""
    last_raw = ""
    for _ in range(max_retries):
        sample, err, raw = generate_one(
            seed,
            category,
            variation_hint,
            next_id,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        last_err, last_raw = err, raw
        if sample is not None:
            return sample, "", ""
    return None, last_err, last_raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand seed SFT data via teacher API (optionally multiple samples per seed)."
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("medical_sft_100_english.jsonl"),
        help="Path to seed JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/generated_round1.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max number of seed samples to expand (0 = all)",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Start index for generated ids (sft_000001, ...)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Flush output file every N samples (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per sample on API or validation failure (default: 3)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (default: OPENAI_API_KEY or DASHSCOPE_API_KEY when --provider aliyun)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="API base URL (default: OPENAI_BASE_URL; with --provider aliyun: DASHSCOPE_BASE_URL or DashScope Beijing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Teacher model (or set SFT_MODEL in .env; default: gpt-4o / qwen-plus by provider)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "aliyun"],
        default=os.environ.get("SFT_PROVIDER", "openai"),
        help="API provider: openai (default) or aliyun (DashScope/Qwen). Can set SFT_PROVIDER env.",
    )
    parser.add_argument(
        "--seed-rng",
        type=int,
        default=None,
        help="Random seed for variation_hint selection (different per sample via index+seed)",
    )
    parser.add_argument(
        "--n-per-seed",
        type=int,
        default=1,
        help="Number of new samples to generate per seed (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent teacher API calls (default: 4)",
    )
    parser.add_argument(
        "--flag-general",
        action="store_true",
        help="Write general-class samples that hit review triggers to <out_stem>.flagged.jsonl",
    )
    args = parser.parse_args()

    # Resolve api_key, base_url, model by provider (env SFT_MODEL overrides default when not passed)
    if args.provider == "aliyun":
        args.api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        args.base_url = args.base_url or os.environ.get("DASHSCOPE_BASE_URL", "") or DASHSCOPE_DEFAULT_BASE
        args.model = args.model or os.environ.get("SFT_MODEL", "") or "qwen-plus"
    else:
        args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "") or None
        args.model = args.model or os.environ.get("SFT_MODEL", "") or "gpt-4o"

    seed_path = args.seed
    if not seed_path.is_absolute():
        seed_path = Path.cwd() / seed_path
    if not seed_path.exists():
        print(f"Error: seed file not found: {seed_path}", file=sys.stderr)
        sys.exit(1)

    if not args.api_key:
        print(
            "Error: set API key (OPENAI_API_KEY or --api-key; for Alibaba use DASHSCOPE_API_KEY or --api-key).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Provider: {args.provider} | Model: {args.model} | Base URL: {args.base_url or '(default)'}")

    samples = load_seed(seed_path)
    if not samples:
        print("Error: no samples loaded from seed.", file=sys.stderr)
        sys.exit(1)

    # Select samples to expand
    to_expand = samples
    if args.max_samples > 0:
        to_expand = samples[: args.max_samples]
    n_total = len(to_expand)

    # RNG for variation hints: each sample gets a different hint; pass --seed-rng for reproducibility
    rng = random.Random(args.seed_rng) if args.seed_rng is not None else random.Random()

    # Build job list with preassigned IDs so we can run jobs concurrently
    jobs: list[dict] = []
    next_id_val = args.start_id
    n_per_seed = max(1, args.n_per_seed)
    for i, seed in enumerate(to_expand):
        category = (seed.get("category") or "general").strip().lower()
        if category not in VALID_CATEGORIES:
            category = "general"
        for k in range(n_per_seed):
            variation_hint = rng.choice(VARIATION_HINTS)
            next_id = f"sft_{next_id_val:06d}"
            jobs.append(
                {
                    "seed_index": i,
                    "k": k,
                    "seed": seed,
                    "category": category,
                    "variation_hint": variation_hint,
                    "next_id": next_id,
                }
            )
            next_id_val += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    errors_path = args.out.parent / (args.out.stem + ".errors.jsonl")
    flag_path = args.out.parent / (args.out.stem + ".flagged.jsonl")

    written = 0
    failed = 0
    skipped_dup = 0
    seen_users: set[str] = set()

    class _NoOpWriter:
        def write(self, _s: str) -> None: ...
        def flush(self) -> None: ...

    flag_file = open(flag_path, "w", encoding="utf-8") if args.flag_general else _NoOpWriter()

    try:
        with open(args.out, "w", encoding="utf-8") as f, open(errors_path, "w", encoding="utf-8") as err_f:
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
                future_to_job = {}
                for job in jobs:
                    fut = executor.submit(
                        generate_with_retries,
                        job["seed"],
                        job["category"],
                        job["variation_hint"],
                        job["next_id"],
                        args.api_key,
                        args.base_url or None,
                        args.model,
                        args.max_retries,
                    )
                    future_to_job[fut] = job

                for fut in as_completed(future_to_job):
                    job = future_to_job[fut]
                    i = job["seed_index"]
                    k = job["k"]
                    category = job["category"]
                    seed = job["seed"]

                    try:
                        sample, err_msg, raw_response = fut.result()
                    except Exception as e:
                        sample, err_msg, raw_response = None, str(e), ""

                    if sample is None:
                        failed += 1
                        print(
                            f"Skip seed {i + 1} gen {k + 1} (id {seed.get('id', '?')}) after {args.max_retries} retries: {err_msg}",
                            file=sys.stderr,
                        )
                        err_f.write(
                            json.dumps(
                                {
                                    "seed_id": seed.get("id"),
                                    "category": category,
                                    "error": err_msg,
                                    "raw_response": (raw_response or "")[:2000],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        err_f.flush()
                        continue

                    user_text = get_user_content(sample["messages"])
                    if user_text in seen_users:
                        skipped_dup += 1
                        continue
                    seen_users.add(user_text)

                    # General-class weak filter: flag for review if assistant contains trigger phrases
                    flagged_triggers: list[str] = []
                    if category == "general":
                        assistant_text = get_assistant_content(sample["messages"])
                        flagged_triggers = flag_general_review(assistant_text)
                    if flagged_triggers:
                        print(
                            f"  [flag] {sample['id']} general review: {flagged_triggers}",
                            file=sys.stderr,
                        )
                        if args.flag_general:
                            flag_file.write(
                                json.dumps(
                                    {
                                        "id": sample["id"],
                                        "category": category,
                                        "triggers": flagged_triggers,
                                        "sample": sample,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            flag_file.flush()

                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
                    if written % args.flush_every == 0:
                        f.flush()
                    print(
                        f"[seed {i + 1}/{n_total} gen {k + 1}/{n_per_seed}] wrote {sample['id']} ({category})"
                    )
    finally:
        if args.flag_general and hasattr(flag_file, "close"):
            flag_file.close()

    print(f"Done: {written} generated, {failed} failed, {skipped_dup} skipped (duplicate user) -> {args.out}")
    if failed:
        print(f"Errors logged -> {errors_path}")


if __name__ == "__main__":
    main()
