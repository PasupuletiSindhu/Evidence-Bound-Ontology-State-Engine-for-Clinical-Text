"""
Align NER+RE baseline outputs with the same graph-QA schema as baseline_3 (Qwen).

BC5CDR relation models emit labels like CID / NR / LABEL_*; QA + relation_aliases expect
canonical names (causes, treats, …). This module maps and orients triples at extraction time
so :mod:`baselines.qa_eval` stays unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

Triple4 = Tuple[str, str, str, str]


def _clean_entity_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _is_negative_relation(rel: str) -> bool:
    u = (rel or "").strip().upper().replace(" ", "_")
    if not u:
        return True
    if u in frozenset(
        {
            "NR",
            "NEGATIVE",
            "NO_RELATION",
            "NONE",
            "O",
            "FALSE",
            "NEG",
            "0",
        }
    ):
        return True
    if u.startswith("NOT_"):
        return True
    low = (rel or "").strip().lower()
    if "no relation" in low or "not related" in low:
        return True
    # Common HF binary: first id = negative
    if u == "LABEL_0":
        return True
    return False


def map_relation_to_qa_schema(rel: str) -> Optional[str]:
    """
    Map BERT relation head output to a canonical relation string for graph QA.

    Returns None to drop the edge (negative / unknown).
    """
    if _is_negative_relation(rel):
        return None
    u = (rel or "").strip().upper().replace(" ", "_")
    lo = (rel or "").strip().lower()

    # BC5CDR chemical–disease (induces / causes disease)
    if u == "CID" or u.endswith("_CID") or "CID" in u:
        return "causes"
    if u in ("POSITIVE", "TRUE", "1", "YES", "RELATION", "POS"):
        return "causes"
    if u == "LABEL_1":
        return "causes"
    if u.startswith("LABEL_"):
        return None

    # Already canonical or covered by qa relation_aliases downstream
    fixed = lo.replace(" ", "_").replace("-", "_")
    if fixed:
        return fixed
    return None


def _entity_type_bucket(label: str) -> str:
    """Coarse type for orientation: chemical, disease, or other."""
    lab = (label or "").strip().lower()
    if "chemical" in lab or lab in ("chem", "drug"):
        return "chemical"
    if "disease" in lab or "disorder" in lab or lab in ("dis", "disease"):
        return "disease"
    return "other"


def orient_subj_obj_for_qa_schema(
    subj: str,
    obj: str,
    head_meta: Dict[str, Any],
    tail_meta: Dict[str, Any],
    canonical_rel: str,
) -> Tuple[str, str]:
    """
    For chemical–disease style graphs, ensure (drug/chemical, relation, disease) when
    canonical_rel is ``causes`` (BC5CDR CID semantics).
    """
    if canonical_rel != "causes":
        return subj, obj
    hb = _entity_type_bucket(str(head_meta.get("label") or ""))
    tb = _entity_type_bucket(str(tail_meta.get("label") or ""))
    if hb == "disease" and tb == "chemical":
        return obj, subj
    return subj, obj


def swap_entity_indices_for_qa(
    head_i: int,
    tail_i: int,
    entities: List[Dict[str, Any]],
    canonical_rel: str,
) -> Tuple[int, int]:
    """Swap head/tail indices when disease–chemical order is reversed (for CUI lookup)."""
    if canonical_rel != "causes":
        return head_i, tail_i
    h_ent = entities[head_i] if 0 <= head_i < len(entities) else {}
    t_ent = entities[tail_i] if 0 <= tail_i < len(entities) else {}
    hb = _entity_type_bucket(str(h_ent.get("label") or ""))
    tb = _entity_type_bucket(str(t_ent.get("label") or ""))
    if hb == "disease" and tb == "chemical":
        return tail_i, head_i
    return head_i, tail_i


def normalize_baseline_triple_row(
    subj: str,
    relation: str,
    obj: str,
    polarity: str,
    head_meta: Dict[str, Any],
    tail_meta: Dict[str, Any],
) -> Optional[Triple4]:
    """
    Produce one (s, r, o, pol) row aligned to QA graph schema, or None to skip.
    """
    canon = map_relation_to_qa_schema(relation)
    if canon is None:
        return None
    s = _clean_entity_text(subj)
    o = _clean_entity_text(obj)
    if not s or not o:
        return None
    s, o = orient_subj_obj_for_qa_schema(s, o, head_meta, tail_meta, canon)
    return (s, canon, o, polarity or "positive")


def finalize_baseline_triples(
    rows: List[Triple4],
    *,
    dedupe: bool = True,
) -> List[Triple4]:
    """Optional dedupe identical (s, r, o, pol)."""
    if not dedupe:
        return list(rows)
    seen = set()
    out: List[Triple4] = []
    for t in rows:
        k = (t[0], t[1], t[2], t[3])
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out
