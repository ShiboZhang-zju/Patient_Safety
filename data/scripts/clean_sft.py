#!/usr/bin/env python3
"""
Rule-based cleaning and deduplication for medical SFT JSONL.

- Normalize messages (whitespace, empty content)
- Normalize category (lowercase, valid set)
- Drop invalid or unsafe samples (optional rules)
- Deduplicate by content (e.g. user text + category or message hash)

Usage:
  python clean_sft.py --input data/processed/sft_generated.jsonl --output data/processed/sft_cleaned.jsonl
  python clean_sft.py --input in.jsonl --output out.jsonl --no-dedup
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from prompts import CATEGORIES

VALID_CATEGORIES = set(CATEGORIES.keys())

# Patterns that suggest unsafe assistant content (no diagnosis/prescription)
UNSAFE_PATTERNS = [
    re.compile(r"\b(?:you have|you've got|it is|it's)\s+(?:probably\s+)?(?:a |an )?\w+\s+(?:infection|disease|cancer|stroke|heart attack)\b", re.I),
    re.compile(r"\b(?:take|give)\s+\d+\s*(?:mg|ml|tablets?|pills?)\s*(?:daily|twice|once)\b", re.I),
    re.compile(r"\b(?:stop|start|double|halve)\s+(?:the\s+)?(?:medication|medicine|dose)\s+(?:yourself|on your own)\b", re.I),
]


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip bad lines
    return samples


def normalize_content(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.split()).strip()


def get_user_content(messages: list) -> str:
    for m in messages or []:
        if isinstance(m, dict) and (m.get("role") or "").strip().lower() == "user":
            return normalize_content(m.get("content") or "")
    return ""


def get_assistant_content(messages: list) -> str:
    for m in messages or []:
        if isinstance(m, dict) and (m.get("role") or "").strip().lower() == "assistant":
            return normalize_content(m.get("content") or "")
    return ""


def normalize_messages(sample: dict) -> dict:
    messages = sample.get("messages") or []
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = normalize_content(m.get("content") or "")
        out.append({"role": role, "content": content})
    return {**sample, "messages": out}


def normalize_category(category: str) -> str:
    c = (category or "").strip().lower()
    if c in VALID_CATEGORIES:
        return c
    # optional alias map
    alias = {"triage": "high_risk", "diagnosis": "boundary", "med": "medication"}
    return alias.get(c, "general")


def is_unsafe_assistant(content: str) -> bool:
    """True if assistant content matches unsafe patterns (diagnosis/dosing)."""
    for pat in UNSAFE_PATTERNS:
        if pat.search(content):
            return True
    return False


def content_dedup_key(sample: dict) -> str:
    """Key for deduplication: category + normalized user + normalized assistant."""
    messages = sample.get("messages") or []
    cat = normalize_category(sample.get("category") or "")
    user = get_user_content(messages)
    assistant = get_assistant_content(messages)
    return f"{cat}\t{user}\t{assistant}"


def clean(
    samples: list[dict],
    drop_unsafe: bool = True,
    drop_empty_user: bool = True,
    drop_empty_assistant: bool = True,
    min_assistant_words: int = 5,
    dedup: bool = True,
) -> list[dict]:
    seen_keys = set()
    out = []
    for s in samples:
        s = normalize_messages(s)
        cat = s.get("category") or "general"
        s["category"] = normalize_category(cat)

        user_text = get_user_content(s["messages"])
        assistant_text = get_assistant_content(s["messages"])

        if drop_empty_user and not user_text:
            continue
        if drop_empty_assistant and not assistant_text:
            continue
        if len(assistant_text.split()) < min_assistant_words:
            continue
        if drop_unsafe and is_unsafe_assistant(assistant_text):
            continue
        if dedup:
            key = content_dedup_key(s)
            if key in seen_keys:
                continue
            seen_keys.add(key)

        out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based cleaning and deduplication for SFT JSONL."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--no-drop-unsafe", action="store_true", help="Do not drop samples matching unsafe patterns")
    parser.add_argument("--min-words", type=int, default=5, help="Min assistant reply length in words (default: 5)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    samples = load_jsonl(args.input)
    n_before = len(samples)
    samples = clean(
        samples,
        drop_unsafe=not args.no_drop_unsafe,
        drop_empty_user=True,
        drop_empty_assistant=True,
        min_assistant_words=args.min_words,
        dedup=not args.no_dedup,
    )
    n_after = len(samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Cleaned: {n_before} -> {n_after} (dropped {n_before - n_after})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
