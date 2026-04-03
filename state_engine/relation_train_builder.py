"""Build (predicted_label, gold_relation) rows from paraphrase extractor JSON for relation_map training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from state_engine.relations import infer_relation_from_text, parse_fact_spans


def infer_gold_relation_from_fact(fact: str) -> str:
    """Gold relation from fact text: regex-backed parse when possible, else clusterer."""
    fact = (fact or "").strip()
    parsed = parse_fact_spans(fact)
    if parsed:
        return str(parsed[1]).strip().lower().replace(" ", "_").replace("-", "_")
    r = infer_relation_from_text(fact)
    return str(r).strip().lower().replace(" ", "_").replace("-", "_")


def collect_relation_train_examples(
    paraphrase_payload: Dict[str, Any],
    baseline_keys: Optional[Sequence[str]] = None,
) -> List[dict]:
    """
    For each set and each triple across baselines, emit
    {"predicted_label": label, "gold_relation": inferred_relation}.
    """
    keys = tuple(baseline_keys) if baseline_keys else ("baseline_1", "baseline_2", "baseline_3")
    examples: List[dict] = []
    for s in paraphrase_payload.get("sets") or []:
        fact = str(s.get("fact", "")).strip()
        gold_relation = infer_gold_relation_from_fact(fact)
        for bkey in keys:
            block = s.get(bkey) or {}
            for para_block in block.get("triples_per_paraphrase") or []:
                for t in para_block or []:
                    if not isinstance(t, (list, tuple)) or len(t) < 3:
                        continue
                    label = str(t[1]).strip()
                    if not label:
                        continue
                    examples.append(
                        {
                            "predicted_label": label,
                            "gold_relation": gold_relation,
                        }
                    )
    return examples


def write_relation_train_json(
    paraphrase_path: Path,
    output_path: Path,
    baseline_keys: Optional[Sequence[str]] = None,
) -> List[dict]:
    payload = json.loads(Path(paraphrase_path).read_text(encoding="utf-8"))
    examples = collect_relation_train_examples(payload, baseline_keys=baseline_keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return examples
