#!/usr/bin/env python3
"""
Split SFT pool into stratified train / val sets with cluster-aware grouping.

Goals:
- Input is a JSONL dataset like data/processed/sft_train_v1.jsonl
- Each sample has: id, category, messages[3] (system/user/assistant)
- We want:
  - Train ~2700, Val ~200–250 (total ~2937)
  - Stratified by category: general / medication / boundary / high_risk
  - All samples sharing the same (user, assistant) pair must go to the same split
  - Preferably keep the same user entirely in one split to reduce leakage

Usage example:

  python split_dataset.py \\
    --input data/processed/sft_train_v1.jsonl \\
    --train-out data/processed/sft_train_v1.train.jsonl \\
    --val-out data/processed/sft_train_v1.val.jsonl

By default it will:
- Target validation size ~8% of total (can override via --val-ratio or --val-size)
- Use the following approximate per-category val targets (can be overridden):
    general:    107
    medication:  52
    boundary:    36
    high_risk:   40
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_VAL_TARGETS = {
    "general": 107,
    "medication": 52,
    "boundary": 36,
    "high_risk": 40,
}


def load_dataset(path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
    return samples


def extract_user_assistant(sample: Dict) -> Tuple[str, str]:
    msgs = sample.get("messages") or []
    user = ""
    assistant = ""
    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        if role == "user":
            user = (m.get("content") or "").strip()
        elif role == "assistant":
            assistant = (m.get("content") or "").strip()
    return user, assistant


def build_clusters(samples: List[Dict]) -> Dict[str, List[int]]:
    """
    Build clusters by (user, assistant) pair so that identical QA pairs
    are always in the same cluster. We deliberately do NOT fully merge
    by user to avoid creating very large clusters that break balancing.

    Returns a mapping: cluster_id -> list of indices in `samples`.
    """
    pair_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx, s in enumerate(samples):
        user, assistant = extract_user_assistant(s)
        pair = (user, assistant)
        pair_to_indices[pair].append(idx)

    cluster_map: Dict[str, List[int]] = {}
    for cid, (_, idx_list) in enumerate(pair_to_indices.items()):
        cluster_map[f"c{cid}"] = idx_list
    return cluster_map


def cluster_category_counts(
    clusters: Dict[str, List[int]], samples: List[Dict]
) -> Dict[str, Counter]:
    """
    For each cluster, count category counts, so we can use it for stratified selection.
    Returns: cluster_id -> Counter({category: count})
    """
    result: Dict[str, Counter] = {}
    for cid, idx_list in clusters.items():
        ctr = Counter()
        for i in idx_list:
            cat = (samples[i].get("category") or "unknown").strip().lower()
            ctr[cat] += 1
        result[cid] = ctr
    return result


def stratified_split(
    samples: List[Dict],
    clusters: Dict[str, List[int]],
    val_targets: Dict[str, int],
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Greedy stratified split at cluster level.
    We randomly shuffle clusters, then assign clusters to val set while
    trying not to exceed per-category val_targets.

    Returns:
      train_indices, val_indices (lists of sample indices)
    """
    random.seed(seed)
    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids)

    cluster_cats = cluster_category_counts(clusters, samples)

    val_indices: List[int] = []
    train_indices: List[int] = []
    current_val_counts = Counter()

    # First pass: greedy assignment with soft cap (1.1x target) to avoid early overshoot
    soft_factor = 1.1
    for cid in cluster_ids:
        idx_list = clusters[cid]
        cats = cluster_cats[cid]

        ok_for_val = True
        for cat, cnt in cats.items():
            target = val_targets.get(cat, 0)
            if target <= 0:
                continue
            if current_val_counts[cat] + cnt > target * soft_factor:
                ok_for_val = False
                break

        if ok_for_val:
            val_indices.extend(idx_list)
            current_val_counts.update(cats)
        else:
            train_indices.extend(idx_list)

    # Second pass: top up categories that are still under target by moving
    # some clusters from train -> val, prioritizing clusters that best fill deficits.
    train_set = set(train_indices)
    remaining_cids = [cid for cid in cluster_ids if any(i in train_set for i in clusters[cid])]

    def deficits() -> Dict[str, int]:
        return {cat: max(0, tgt - current_val_counts[cat]) for cat, tgt in val_targets.items()}

    while True:
        deficit = deficits()
        # Stop if essentially no deficit left
        if all(v <= 0 for v in deficit.values()):
            break

        # Score remaining clusters by how much they help fill deficits
        best_score = 0
        best_cid = None
        for cid in remaining_cids:
            cats = cluster_cats[cid]
            score = sum(min(deficit.get(cat, 0), cnt) for cat, cnt in cats.items())
            if score > best_score:
                best_score = score
                best_cid = cid

        if not best_cid or best_score == 0:
            # No remaining cluster can meaningfully reduce deficit
            break

        # Move best cluster from train to val
        idx_list = clusters[best_cid]
        for i in idx_list:
            if i in train_set:
                train_set.remove(i)
                val_indices.append(i)
        # rebuild train_indices from train_set at the end of the loop
        current_val_counts.update(cluster_cats[best_cid])
        remaining_cids.remove(best_cid)

    # Rebuild train_indices from the final train_set (anything not in val_indices)
    val_set = set(val_indices)
    final_train = sorted(train_set - val_set)
    final_val = sorted(val_set)
    return final_train, final_val


def write_jsonl(path: Path, samples: List[Dict], indices: List[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in indices:
            f.write(json.dumps(samples[i], ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split SFT dataset into stratified train/val with cluster-aware grouping."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL path (e.g., data/processed/sft_train_v1.jsonl)",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        required=True,
        help="Output JSONL path for train split",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        required=True,
        help="Output JSONL path for val split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.08,
        help="Approximate validation ratio if val-size not specified (default: 0.08)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=0,
        help="Optional fixed validation size; overrides val-ratio if > 0",
    )
    args = parser.parse_args()

    samples = load_dataset(args.input)
    total = len(samples)
    print(f"Loaded {total} samples from {args.input}")

    # Compute default per-category targets based on DEFAULT_VAL_TARGETS scaled if needed
    cat_counts = Counter((s.get("category") or "unknown").strip().lower() for s in samples)
    print("Category counts in input:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")

    # Total desired val size
    desired_val = args.val_size or int(round(total * args.val_ratio))
    # Compute per-category val targets directly from current distribution
    # so that val set matches the empirical proportions of the dataset.
    val_targets = {
        cat: int(round(cnt / total * desired_val)) for cat, cnt in cat_counts.items()
    }
    print("\nValidation targets (approximate, per category):")
    for cat, tgt in sorted(val_targets.items()):
        print(f"  {cat}: {tgt}")
    print(f"Total target val size (approx): {sum(val_targets.values())}")

    # Build clusters and split
    clusters = build_clusters(samples)
    print(f"\nBuilt {len(clusters)} clusters for splitting")

    train_idx, val_idx = stratified_split(
        samples,
        clusters,
        val_targets=val_targets,
        seed=args.seed,
    )

    print(f"\nFinal split sizes: train={len(train_idx)}, val={len(val_idx)}, total={len(train_idx)+len(val_idx)}")

    # Sanity check category distribution
    def count_by_cat(indices: List[int]) -> Counter:
        ctr = Counter()
        for i in indices:
            cat = (samples[i].get("category") or "unknown").strip().lower()
            ctr[cat] += 1
        return ctr

    train_cats = count_by_cat(train_idx)
    val_cats = count_by_cat(val_idx)

    print("\nTrain category counts:")
    for cat, cnt in sorted(train_cats.items()):
        print(f"  {cat}: {cnt}")
    print("\nVal category counts:")
    for cat, cnt in sorted(val_cats.items()):
        print(f"  {cat}: {cnt}")

    # Write splits
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.train_out, samples, train_idx)
    write_jsonl(args.val_out, samples, val_idx)

    print(f"\nWrote train split to {args.train_out}")
    print(f"Wrote val split to {args.val_out}")


if __name__ == "__main__":
    main()

