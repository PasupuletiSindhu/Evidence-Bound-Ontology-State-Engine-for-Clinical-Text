"""
Per-scope semantic entity clustering after ontology alignment.

Merges entities when cosine similarity >= merge_threshold (default 0.88). Optional
lexical containment merge is off by default (embedding-first).

distinct_threshold .. merge_threshold: near-duplicate band (logged, no merge unless
containment forces merge).

Deterministic: existing nodes are scanned in lexicographic order; argmax similarity,
ties broken by lexicographically smallest canonical representative.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from state_engine.embeddings import (
    cosine_similarity_vectors,
    get_embedding_or_none,
    is_embedding_backend_available,
)
from state_engine.semantic_canonicalizer import normalize_surface

MergeMeta = Dict[str, object]
logger = logging.getLogger(__name__)

_HEADWORD_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "for",
    "with",
    "and",
    "or",
    "by",
}


def _headword_key(text: str) -> str:
    """
    Lightweight 'headword' heuristic for entity merge.

    We avoid heavy NLP here; instead we extract a stable token that captures the core
    concept for common clinical phrases.
    """
    t = normalize_surface(text or "")
    if not t:
        return ""
    toks = [w for w in t.split() if w and w not in _HEADWORD_STOP]
    if not toks:
        return ""
    # High-value concept anchor: if the phrase mentions bleeding, treat 'bleeding' as head.
    if "bleeding" in toks:
        return "bleeding"
    # Otherwise, use the last token as a rough head.
    return toks[-1]


def lexical_containment_merge(a: str, b: str, *, min_len: int = 4) -> bool:
    """True if one normalized surface is a substring of the other (length >= min_len)."""
    na = normalize_surface(a or "")
    nb = normalize_surface(b or "")
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) < min_len or len(nb) < min_len:
        return False
    return na in nb or nb in na


@dataclass
class EmbeddingEntityRegistry:
    """
    Accumulates canonical entity strings for one logical scope (e.g. one fact set's
    incremental run) and maps new aligned strings onto prior representatives.
    """

    merge_threshold: float = 0.88
    distinct_threshold: float = 0.65
    uncertain_band: Tuple[float, float] = (0.65, 0.88)
    use_lexical_containment: bool = False
    use_headword_merge: bool = True

    _representatives: List[str] = field(default_factory=list)
    _embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    _surface_counts: Dict[str, Counter[str]] = field(default_factory=dict)
    _surface_recency: Dict[str, List[str]] = field(default_factory=dict)
    merge_provenance: List[Dict] = field(default_factory=list)
    near_duplicate_log: List[Dict] = field(default_factory=list)

    def _ensure_embedding(self, canonical: str) -> Optional[np.ndarray]:
        c = (canonical or "").strip()
        if not c:
            return None
        if c in self._embeddings:
            return self._embeddings[c]
        emb = get_embedding_or_none(c)
        if emb is not None:
            self._embeddings[c] = emb
        return emb

    def _similarity_to_canonical(self, entity: str, canonical: str) -> float:
        """Similarity in [0,1]; string-normalization equality => 1.0."""
        ne = normalize_surface(entity or "")
        nc = normalize_surface(canonical or "")
        if ne and ne == nc:
            return 1.0
        if (entity or "").strip() == (canonical or "").strip():
            return 1.0
        ea = self._ensure_embedding(entity)
        ec = self._ensure_embedding(canonical)
        if ea is None or ec is None:
            return 0.0
        return cosine_similarity_vectors(ea, ec)

    def canonicalize_entity_with_existing(
        self, entity: str, existing_nodes: List[str]
    ) -> Tuple[str, MergeMeta]:
        """
        Map ``entity`` onto an existing representative if similarity >= merge_threshold,
        else add as new. Returns (canonical_string, metadata).
        """
        e = (entity or "").strip()
        if not e:
            return e, {"action": "empty"}

        meta: MergeMeta = {"action": "new", "similarity": None, "merged_into": None}

        if not existing_nodes:
            self._add_new_canonical(e)
            return e, meta

        best: Optional[str] = None
        best_sim = -1.0
        for c in sorted(existing_nodes):
            sim = self._similarity_to_canonical(e, c)
            if best is None or sim > best_sim or (sim == best_sim and c < best):
                best_sim = sim
                best = c

        meta["similarity"] = float(best_sim)

        force_contain = bool(
            self.use_lexical_containment
            and best is not None
            and lexical_containment_merge(e, best)
        )
        force_head = bool(
            self.use_headword_merge
            and best is not None
            and _headword_key(e)
            and _headword_key(e) == _headword_key(best)
            # don't allow headword-only merges for very dissimilar items
            and best_sim >= float(self.distinct_threshold)
        )
        if best is not None and (best_sim >= self.merge_threshold or force_contain or force_head):
            self.merge_provenance.append(
                {
                    "incoming": e,
                    "canonical": best,
                    "similarity": float(best_sim),
                    "decision": (
                        "merge_containment"
                        if force_contain and best_sim < self.merge_threshold
                        else ("merge_headword" if force_head and best_sim < self.merge_threshold else "merge")
                    ),
                }
            )
            meta["action"] = "merge"
            meta["merged_into"] = best
            if force_contain and best_sim < self.merge_threshold:
                meta["merge_reason"] = "lexical_containment"
            if force_head and best_sim < self.merge_threshold:
                meta["merge_reason"] = "headword"
            logger.debug(
                "registry merge incoming=%r -> %r sim=%.4f decision=%s",
                e,
                best,
                float(best_sim),
                meta.get("merge_reason") or "threshold",
            )
            return best, meta

        lo, hi = self.uncertain_band
        if best is not None and lo <= best_sim < hi:
            self.near_duplicate_log.append(
                {
                    "incoming": e,
                    "nearest": best,
                    "similarity": float(best_sim),
                    "decision": "uncertain_no_merge",
                }
            )
            meta["action"] = "near_duplicate_distinct"
            meta["nearest"] = best

        self._add_new_canonical(e)
        return e, meta

    def canonicalize_entity(self, entity: str) -> str:
        """Use internal representative list as ``existing_nodes``."""
        canon, _ = self.canonicalize_entity_with_existing(entity, list(self._representatives))
        return canon

    def _add_new_canonical(self, e: str) -> None:
        if e not in self._representatives:
            self._representatives.append(e)
        self._ensure_embedding(e)

    def record_surface(self, canonical: str, surface: str) -> None:
        c = (canonical or "").strip()
        t = (surface or "").strip()
        if not c or not t:
            return
        if c not in self._surface_counts:
            self._surface_counts[c] = Counter()
        self._surface_counts[c][t] += 1
        if c not in self._surface_recency:
            self._surface_recency[c] = []
        self._surface_recency[c].append(t)

    def preferred_surface(self, canonical: str) -> str:
        """
        QA/display string: highest count, tie-break lexicographic asc,
        then most recently seen.
        """
        c = (canonical or "").strip()
        ctr = self._surface_counts.get(c)
        if not ctr:
            return c
        max_count = max(ctr.values())
        candidates = sorted([s for s, n in ctr.items() if n == max_count])
        if not candidates:
            return c
        if len(candidates) == 1:
            return candidates[0]
        rec = self._surface_recency.get(c, [])
        for s in reversed(rec):
            if s in candidates:
                return s
        return candidates[0]

    def surface_evidence_counts(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in self._surface_counts.items()}

    def surface_recency_lists(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._surface_recency.items()}

    def export_mapping(self) -> Dict:
        return {
            "canonical_to_surfaces": self.surface_evidence_counts(),
            "preferred_surface": {
                k: self.preferred_surface(k) for k in self._surface_counts
            },
            "merge_provenance": list(self.merge_provenance),
            "near_duplicate_log": list(self.near_duplicate_log),
            "representatives": list(self._representatives),
            "embedding_backend": is_embedding_backend_available(),
        }


