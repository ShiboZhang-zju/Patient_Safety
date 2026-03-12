#!/usr/bin/env python3
"""
Random sampling by category for manual review of SFT data.

Reads JSONL, samples n per category (or total n distributed by category ratio),
and writes a review JSONL + optional per-category files.

Usage:
  python sample_review.py --input data/processed/sft_cleaned.jsonl --output data/review/sft_review.jsonl --per-category 5
  python sample_review.py --input in.jsonl --output out.jsonl --total 50 --by-ratio
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from prompts import CATEGORIES

VALID_CATEGORIES = set(CATEGORIES.keys())


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return samples


def group_by_category(samples: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for s in samples:
        cat = (s.get("category") or "general").strip().lower()
        if cat not in VALID_CATEGORIES:
            cat = "general"
        groups[cat].append(s)
    return dict(groups)


def sample_per_category(
    groups: dict[str, list[dict]],
    per_category: int,
    seed: int | None = 42,
) -> list[dict]:
    if seed is not None:
        random.seed(seed)
    out = []
    for cat in sorted(groups.keys()):
        pool = groups[cat]
        n = min(per_category, len(pool))
        out.extend(random.sample(pool, n))
    random.shuffle(out)
    return out


def sample_total_by_ratio(
    groups: dict[str, list[dict]],
    total: int,
    seed: int | None = 42,
) -> list[dict]:
    """Sample total items, keeping category proportions roughly similar."""
    if seed is not None:
        random.seed(seed)
    total_available = sum(len(v) for v in groups.values())
    if total_available == 0:
        return []
    out = []
    for cat in sorted(groups.keys()):
        pool = groups[cat]
        ratio = len(pool) / total_available
        n = max(1, min(len(pool), round(total * ratio)))
        out.extend(random.sample(pool, n))
    random.shuffle(out)
    return out[:total]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample SFT data by category for manual review."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL path for review set")
    parser.add_argument(
        "--per-category",
        type=int,
        default=0,
        help="Sample this many per category (if set, --total is ignored)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=50,
        help="Total samples when using --by-ratio (default: 50)",
    )
    parser.add_argument(
        "--by-ratio",
        action="store_true",
        help="Sample by category ratio up to --total",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--split-by-category",
        action="store_true",
        help="Also write one file per category under output dir (e.g. review_general.jsonl)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    samples = load_jsonl(args.input)
    groups = group_by_category(samples)

    if args.per_category > 0:
        selected = sample_per_category(groups, args.per_category, seed=args.seed)
    elif args.by_ratio:
        selected = sample_total_by_ratio(groups, args.total, seed=args.seed)
    else:
        selected = sample_total_by_ratio(groups, args.total, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Sampled {len(selected)} samples -> {args.output}")
    for cat in sorted(groups.keys()):
        n_in_sample = sum(1 for s in selected if (s.get("category") or "").strip().lower() == cat)
        print(f"  {cat}: {n_in_sample} (of {len(groups[cat])} total)")

    if args.split_by_category:
        for cat in sorted(groups.keys()):
            subset = [s for s in selected if (s.get("category") or "").strip().lower() == cat]
            if not subset:
                continue
            out_cat = args.output.parent / f"review_{cat}.jsonl"
            with open(out_cat, "w", encoding="utf-8") as f:
                for s in subset:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"  Wrote {out_cat} ({len(subset)} samples)")


if __name__ == "__main__":
    main()
