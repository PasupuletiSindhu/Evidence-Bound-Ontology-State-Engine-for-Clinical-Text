"""
Learn extractor label -> canonical relation from paraphrase JSON.

Supervision uses **embedding clusters over relation phrases** (verb-centric cues, not full facts);
each set gets cluster→relation via majority cue vote; extractor labels aggregate against that relation.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from state_engine.relations import CANONICAL_RELATIONS, relation_lookup_key, infer_relation_from_text

logger = logging.getLogger(__name__)

_STOP_KEYS = frozenset(
    {
        "AND",
        "ARE",
        "AT",
        "FOR",
        "IN",
        "IS",
        "OF",
        "ON",
        "OR",
        "THE",
        "TO",
    }
)


def _plausible_learnt_label(key: str, total_occurrences: int) -> bool:
    """Drop URI fragments, quoted garbage, and ultra-rare junk tokens from the saved map."""
    if re.fullmatch(r"LABEL_[0-9]+", key):
        return True
    if total_occurrences < 2:
        return False
    if any(ch in key for ch in "<>\"'`"):
        return False
    if len(key) > 56:
        return False
    if key in _STOP_KEYS:
        return False
    return True


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_extractor_label_stats(
    paraphrase_sets: List[dict],
    fact_to_relation: Dict[str, str],
    baseline_keys: Iterable[str],
) -> Tuple[Dict[str, Counter[str]], Dict[str, Any]]:
    """
    For each extractor label key, count co-occurrences with fact-inferred relations.
    """
    label_relation_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    per_set_debug: List[dict] = []

    for i, s in enumerate(paraphrase_sets or []):
        fact = str(s.get("fact", "")).strip()
        gold = fact_to_relation.get(fact) or fact_to_relation.get(str(i))
        if not gold:
            logger.warning("relation_map_builder: missing inferred relation for set %s fact=%r", i, fact[:80])
            continue

        for bkey in baseline_keys:
            block = s.get(bkey) or {}
            for para_block in block.get("triples_per_paraphrase") or []:
                for t in para_block or []:
                    if not isinstance(t, (list, tuple)) or len(t) < 2:
                        continue
                    raw_label = str(t[1]).strip()
                    if not raw_label:
                        continue
                    lk = relation_lookup_key(raw_label)
                    label_relation_counts[lk][gold] += 1

        per_set_debug.append({"set_id": i, "fact": fact, "inferred_relation": gold})

    meta = {
        "num_sets": len(paraphrase_sets or []),
        "labeled_sets": len(per_set_debug),
    }
    return label_relation_counts, {"per_set": per_set_debug, "meta": meta}


def build_argmax_mapping(
    label_relation_counts: Dict[str, Counter[str]],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    label -> most frequent inferred relation; ties broken by lexicographically smallest relation.
    Omits ``related_to`` as a mapped target (weak fallback, not a label dictionary entry).
    """
    mapping: Dict[str, str] = {}
    tie_detail: List[dict] = []
    for label in sorted(label_relation_counts.keys()):
        ctr = label_relation_counts[label]
        if not ctr:
            continue
        max_ct = max(ctr.values())
        candidates = sorted(
            [
                r
                for r, c in ctr.items()
                if c == max_ct and r in CANONICAL_RELATIONS and r not in ("unknown", "related_to")
            ]
        )
        if not candidates:
            continue
        chosen = candidates[0]
        if len(candidates) > 1:
            tie_detail.append(
                {
                    "label": label,
                    "tie": candidates,
                    "counts": {r: ctr[r] for r in candidates},
                    "chosen": chosen,
                }
            )
        tot = sum(ctr.values())
        if not _plausible_learnt_label(label, tot):
            continue
        mapping[label] = chosen

    return mapping, {"tie_breaks": tie_detail}


def build_fact_to_relation_from_facts(paraphrase_sets: List[dict]) -> Dict[str, str]:
    """Map each fact (and set index) -> relation inferred from fact text."""
    out: Dict[str, str] = {}
    for i, s in enumerate(paraphrase_sets or []):
        fact = str(s.get("fact", "")).strip()
        r = infer_relation_from_text(fact)
        out[fact] = r
        out[str(i)] = r
    return out


def build_and_save_relation_map(
    *,
    paraphrase_path: Path,
    map_out: Path,
    build_log_out: Optional[Path] = None,
    baseline_keys: Sequence[str] = ("baseline_1", "baseline_2", "baseline_3"),
    cluster_assignments: Optional[List[Dict[str, Any]]] = None,
    cluster_summaries: Optional[List[Dict[str, Any]]] = None,
    label_cluster_trace: Optional[List[Dict[str, Any]]] = None,
    relation_cluster_map_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build label->relation JSON from paraphrase extractor outputs and save artifacts.
    """
    para = _load_json(paraphrase_path)
    sets = para.get("sets") or []

    fact_to_rel = build_fact_to_relation_from_facts(sets)
    label_counts, dbg = collect_extractor_label_stats(sets, fact_to_rel, baseline_keys)
    mapping, tie_info = build_argmax_mapping(label_counts)

    logger.info(
        "relation_map_builder: %d labels -> relation (from %d paraphrase sets; embedding cluster supervision)",
        len(mapping),
        len(sets),
    )
    if tie_info["tie_breaks"]:
        logger.info(
            "relation_map_builder: %d labels had tied argmax counts (resolved lexicographically)",
            len(tie_info["tie_breaks"]),
        )
        for row in tie_info["tie_breaks"][:15]:
            logger.debug("tie-break label=%s chosen=%s %s", row["label"], row["chosen"], row["counts"])

    uncovered = sorted(
        k for k, ctr in label_counts.items() if k not in mapping and ctr.total() > 0
    )
    if uncovered:
        logger.info(
            "relation_map_builder: %d extractor label keys not exported (filtered/low-support/ambiguous)",
            len(uncovered),
        )

    map_out.parent.mkdir(parents=True, exist_ok=True)
    map_out.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("relation_map_builder: wrote %s (%d entries)", map_out, len(mapping))

    if build_log_out:
        log_payload = {
            "source_paraphrase": str(paraphrase_path),
            "supervision": "Supervised relation centroids (cue→canonical per fact; nearest centroid for general cues) or KMeans fallback",
            "relation_cluster_map": relation_cluster_map_path,
            "baselines_used": list(baseline_keys),
            "runtime_disambiguation": (
                "normalize_relation_label() uses infer_relation_from_text(fact) "
                "(relation-phrase embed + nearest centroid vs REL_MAP for shared LABEL_n)."
            ),
            **dbg["meta"],
            "fact_inferred_relations": {
                row["fact"]: row["inferred_relation"] for row in dbg["per_set"] if row.get("fact")
            },
            "cluster_fact_assignments": cluster_assignments,
            "cluster_summaries": cluster_summaries,
            "label_cluster_relation_trace_count": len(label_cluster_trace or []),
            "label_cluster_relation_trace_head": (label_cluster_trace or [])[:200],
            "label_relation_counts": {k: dict(v) for k, v in sorted(label_counts.items())},
            "mapping": mapping,
            "tie_breaks": tie_info["tie_breaks"],
            "labels_seen_without_mapping": sorted(set(label_counts.keys()) - set(mapping.keys())),
        }
        build_log_out.parent.mkdir(parents=True, exist_ok=True)
        build_log_out.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
        logger.info("relation_map_builder: wrote build log %s", build_log_out)

    return mapping
