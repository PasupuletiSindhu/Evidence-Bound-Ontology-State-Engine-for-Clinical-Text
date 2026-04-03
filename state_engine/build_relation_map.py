#!/usr/bin/env python3
"""Build a learned label->canonical relation map from training data."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def build_relation_map(
    examples: List[dict],
    predicted_key: str = "predicted_label",
    gold_key: str = "gold_relation",
    min_top_ratio: float = 0.0,
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    label_to_rel_counts = defaultdict(Counter)

    for ex in examples:
        if not isinstance(ex, dict):
            continue
        label = str(ex.get(predicted_key, "")).strip().upper().replace(" ", "_")
        gold = str(ex.get(gold_key, "")).strip().lower().replace(" ", "_")
        if not label or not gold:
            continue
        label_to_rel_counts[label][gold] += 1

    label_map: Dict[str, str] = {}
    stats: Dict[str, dict] = {}
    for label, counter in label_to_rel_counts.items():
        most_common_rel, top_count = counter.most_common(1)[0]
        total = sum(counter.values())
        top_ratio = (top_count / total) if total else 0.0
        mapped_rel = most_common_rel if top_ratio >= min_top_ratio else "unknown"
        label_map[label] = mapped_rel
        stats[label] = {
            "counts": dict(counter),
            "top_relation": most_common_rel,
            "top_count": top_count,
            "total": total,
            "top_ratio": top_ratio,
            "mapped_to": mapped_rel,
        }

    return label_map, stats


def _load_examples(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("examples", "data", "train", "records"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    raise ValueError("Unsupported input JSON format; expected list or dict with examples/data/train/records.")


def main():
    p = argparse.ArgumentParser(description="Learn relation label map from training data.")
    p.add_argument(
        "--input",
        type=str,
        default="results/relation_train.json",
        help="Training pairs JSON (e.g. from scripts/build_relation_train.py).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/relation_map.json",
        help="Output label -> canonical relation map.",
    )
    p.add_argument("--predicted_key", type=str, default="predicted_label")
    p.add_argument("--gold_key", type=str, default="gold_relation")
    p.add_argument(
        "--min_top_ratio",
        type=float,
        default=0.0,
        help="Optional confidence filter; map to 'unknown' when top frequency ratio is below this threshold.",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    examples = _load_examples(in_path)
    label_map, stats = build_relation_map(
        examples,
        predicted_key=args.predicted_key,
        gold_key=args.gold_key,
        min_top_ratio=args.min_top_ratio,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved learned mapping to {out_path}")
    print(f"Saved mapping stats to {stats_path}")


if __name__ == "__main__":
    main()
