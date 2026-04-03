"""Map canonical graph nodes to surface strings for QA lookup."""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set, Tuple

Triple = Tuple[str, str, str]

# canonical_node -> surface variants seen in extraction / mapper
_ALIASES: Dict[str, Set[str]] = defaultdict(set)

# Global QA display: merge counts across all sets for preferred surface selection
_GLOBAL_SURFACE_COUNTS: Dict[str, Counter[str]] = defaultdict(Counter)
_GLOBAL_SURFACE_RECENCY: Dict[str, List[str]] = defaultdict(list)


def clear_entity_aliases() -> None:
    _ALIASES.clear()
    _GLOBAL_SURFACE_COUNTS.clear()
    _GLOBAL_SURFACE_RECENCY.clear()


def register_entity_alias(canonical: str, surface: str) -> None:
    c = (canonical or "").strip()
    t = (surface or "").strip()
    if not c or not t:
        return
    _ALIASES[c].add(t)
    _ALIASES[c].add(t.lower())


def ingest_semantic_registry_surfaces(registry) -> None:
    """Accumulate per-set clustering surfaces into global QA preference state."""
    for canon, ctr in registry.surface_evidence_counts().items():
        for surf, n in ctr.items():
            _GLOBAL_SURFACE_COUNTS[canon][surf] += int(n)
    for canon, seq in registry.surface_recency_lists().items():
        _GLOBAL_SURFACE_RECENCY[canon].extend(seq)
        pref = registry.preferred_surface(canon)
        if pref:
            register_entity_alias(canon, pref)
        for surf in registry.surface_evidence_counts().get(canon, {}):
            register_entity_alias(canon, surf)


def global_preferred_surface(canonical: str) -> str:
    """Most common surface for QA; tie lexicographic; then most recent in global recency."""
    c = (canonical or "").strip()
    ctr = _GLOBAL_SURFACE_COUNTS.get(c)
    if not ctr:
        return c
    max_count = max(ctr.values())
    candidates = sorted([s for s, n in ctr.items() if n == max_count])
    if len(candidates) == 1:
        return candidates[0]
    rec = _GLOBAL_SURFACE_RECENCY.get(c, [])
    for s in reversed(rec):
        if s in candidates:
            return s
    return candidates[0]


def register_mapper_inverse(mapper) -> None:
    """Register text->id mappings so QA can query with gold surface strings."""
    if mapper is None:
        return
    vocab = getattr(mapper, "_map", None)
    if not isinstance(vocab, dict):
        return
    for key, cui in vocab.items():
        cid = str(cui).strip()
        if not cid:
            continue
        register_entity_alias(cid, cid)
        register_entity_alias(cid, str(key).strip())


def expand_triples_for_qa(triples: Iterable[Triple]) -> List[Triple]:
    """
    Duplicate edges under every known surface alias for (subject, object),
    using the same string normalization as baselines.qa_eval.build_graph_index.
    Also emits (preferred_surface(s), r, preferred_surface(o)) when preferred differs.
    """
    try:
        from baselines.qa_eval import _normalize_entity as qn
    except Exception:

        def qn(x):
            s = (x or "").strip()
            if s and s[0].isdigit():
                for i, c in enumerate(s):
                    if not c.isdigit() and c not in ".:":
                        s = s[i:].lstrip(".: ")
                        break
            return s.lower()

    out_set: Set[Triple] = set()
    for row in triples:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        s, r, o = str(row[0]).strip(), str(row[1]).strip().lower().replace(" ", "_"), str(row[2]).strip()
        if not s or not r or not o:
            continue
        subj_vars = {qn(s)} | {qn(a) for a in _ALIASES.get(s, ())}
        obj_vars = {qn(o)} | {qn(a) for a in _ALIASES.get(o, ())}
        ps = global_preferred_surface(s)
        po = global_preferred_surface(o)
        if ps and ps != s:
            subj_vars.add(qn(ps))
        if po and po != o:
            obj_vars.add(qn(po))
        for sv in subj_vars:
            for ov in obj_vars:
                if sv and ov:
                    out_set.add((sv, r, ov))
    return sorted(out_set)
