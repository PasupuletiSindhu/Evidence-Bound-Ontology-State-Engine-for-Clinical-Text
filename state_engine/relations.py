"""Relation label normalization for extractor outputs (learned map + canonical schema)."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from state_engine.semantic_canonicalizer import (
    had_agentive_leading_scaffolding,
    strip_agentive_surface_prefix,
)

CANONICAL_RELATIONS = frozenset(
    {
        "causes",
        "treats",
        "interacts_with",
        "prevents",
        "reduces",
        "increases",
        "metabolized_by",
        "related_to",
        "unknown",
    }
)

# Fixed set for embedding-based disambiguation (no related_to / unknown).
SEMANTIC_CANONICAL_RELATIONS: List[str] = [
    "causes",
    "treats",
    "interacts_with",
    "prevents",
    "reduces",
    "increases",
    "metabolized_by",
]

SEMANTIC_SIMILARITY_THRESHOLD = 0.75

_logger = logging.getLogger(__name__)

_CANONICAL_REL_EMBED: Optional[np.ndarray] = None

REL_MAP: Dict[str, str] = {}


def relation_lookup_key(rel: str) -> str:
    """Normalize extractor label for REL_MAP lookup (uppercase, underscores)."""
    return str(rel or "").strip().upper().replace(" ", "_")


def load_relation_map(path: str = "relation_map.json") -> Dict[str, str]:
    global REL_MAP
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        REL_MAP = {}
        return REL_MAP
    except Exception:
        REL_MAP = {}
        return REL_MAP

    if not isinstance(data, dict):
        REL_MAP = {}
        return REL_MAP

    cleaned: Dict[str, str] = {}
    for k, v in data.items():
        sk = str(k).strip()
        if sk.startswith("_"):
            continue
        lk = sk.upper().replace(" ", "_")
        lv = str(v).strip().lower().replace(" ", "_")
        if lk and lv and lv in CANONICAL_RELATIONS and lv not in ("unknown", "related_to"):
            cleaned[lk] = lv
    REL_MAP = cleaned
    return REL_MAP


def _norm_fact_gold(fact_gold_relation: Optional[str]) -> Optional[str]:
    if not fact_gold_relation:
        return None
    fg = str(fact_gold_relation).strip().lower().replace(" ", "_").replace("-", "_")
    if fg in CANONICAL_RELATIONS and fg not in ("unknown", "related_to"):
        return fg
    return None


def extract_relation_phrase(fact_text: str) -> str:
    """
    Strip entity-heavy context; return a short relational cue for embedding.
    Order: longer / specific patterns before broad ones. Fallback: full fact lowercased.
    """
    t = (fact_text or "").strip()
    if not t:
        return ""
    low = t.lower()
    if re.search(r"\bis\s+used\s+to\s+treat\b", low):
        return "treats"
    if re.search(r"\binteracts\s+with\b", low):
        return "interacts_with"
    if re.search(r"\bis\s+metabolized\s+by\b", low) or re.search(r"\bis\s+metabolised\s+by\b", low):
        return "metabolized_by"
    if re.search(r"\bincreases\s+the\s+risk\s+of\b", low):
        return "increases"
    if re.search(r"\bleads\s+to\b", low):
        return "causes"
    if re.search(r"\bcauses\b", low):
        return "causes"
    if re.search(r"\bprevent(?:s|ing)?\b", low):
        return "prevents"
    if (
        re.search(r"\breliev(?:es|e|ing)?\b", low)
        or re.search(r"\blower(?:s|ing)?\b", low)
        or re.search(r"\breduce(?:s|d|ing)?\b", low)
    ):
        return "reduces"
    if re.search(r"\btreats\b", low) or re.search(r"\btreat(?:s|ed|ing)?\b", low):
        return "treats"
    if re.search(r"\bincreases\b", low) or re.search(r"\braises\b", low):
        return "increases"
    return low


def _normalize_phrase_for_embedding(phrase: str) -> str:
    t = (phrase or "").strip().lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _canonical_relation_embeddings() -> np.ndarray:
    global _CANONICAL_REL_EMBED
    if _CANONICAL_REL_EMBED is not None:
        return _CANONICAL_REL_EMBED
    from state_engine.embeddings import embed_texts

    m = embed_texts(list(SEMANTIC_CANONICAL_RELATIONS))
    if m.shape[0] == len(SEMANTIC_CANONICAL_RELATIONS) and not np.allclose(m, 0.0):
        _CANONICAL_REL_EMBED = m
    return m


def resolve_relation(label: str, fact_text: str, label_map: Dict[str, str]) -> str:
    """
    Prefer semantic match of relation phrase (from fact_text) vs canonical relation embeddings;
    else ``label_map.get(relation_lookup_key(label))`` (learned map fallback).
    """
    key = relation_lookup_key(label)
    phrase = extract_relation_phrase(fact_text)
    phrase_embed = _normalize_phrase_for_embedding(phrase)

    best_rel: Optional[str] = None
    best_score = 0.0

    if phrase_embed:
        try:
            from state_engine.embeddings import embed_texts

            canon = _canonical_relation_embeddings()
            pv = embed_texts([phrase_embed])[0]
            if canon.shape[0] > 0 and not np.allclose(pv, 0.0):
                sims = canon @ pv
                j = int(np.argmax(sims))
                best_rel = SEMANTIC_CANONICAL_RELATIONS[j]
                best_score = float(sims[j])
        except Exception as exc:  # pragma: no cover - defensive
            _logger.debug("resolve_relation embedding error: %s", exc)

    if best_rel is not None and best_score >= SEMANTIC_SIMILARITY_THRESHOLD:
        _logger.debug(
            "resolve_relation path=semantic_override label_key=%r phrase=%r -> %r sim=%.4f",
            key,
            phrase,
            best_rel,
            best_score,
        )
        return best_rel

    fb = label_map.get(key)
    if fb:
        _logger.debug(
            "resolve_relation path=label_map_fallback label_key=%r phrase=%r -> %r (best_semantic=%r sim=%.4f)",
            key,
            phrase,
            fb,
            best_rel,
            best_score,
        )
        return fb

    _logger.debug(
        "resolve_relation path=no_map label_key=%r phrase=%r -> unknown (best_semantic=%r sim=%.4f)",
        key,
        phrase,
        best_rel,
        best_score,
    )
    return "unknown"


def infer_relation_from_text(fact_text: str) -> str:
    """
    Relation for a seed fact: **cue-first** canonical mapping from the relation phrase; only for
    ``general relation`` cues do we use the cluster map (nearest relation centroid). Shared with
    ``normalize_relation_label(..., fact_gold_relation=...)`` for extractor disambiguation.
    """
    from state_engine.relation_clusterer import infer_fact_relation

    return infer_fact_relation(fact_text)


def normalize_relation_label(
    rel: str,
    *,
    fact_gold_relation: Optional[str] = None,
    fact_text: Optional[str] = None,
) -> str:
    """
    Map raw extractor relation string to a canonical relation.

    Uses REL_MAP (learned from fact-text supervision) and literal canonical names.
    When ``fact_text`` is set, :func:`resolve_relation` runs (embeddings + map fallback)
    before the plain map path. When the corpus map disagrees with the current seed fact
    (shared ``LABEL_n``), ``fact_gold_relation`` (typically :func:`infer_relation_from_text`
    on the fact) wins.
    """
    rel_raw = rel
    fg = _norm_fact_gold(fact_gold_relation)

    key = relation_lookup_key(rel)
    mapped = REL_MAP.get(key) if key in REL_MAP else None

    rl = str(rel or "").strip().lower().replace(" ", "_").replace("-", "_")
    literal = rl if rl in CANONICAL_RELATIONS and rl != "unknown" else None

    out: str
    via: str

    if literal is not None:
        if literal == "related_to" and fg is not None:
            out, via = fg, "literal_related_to+fact_gold"
        elif fg is not None and literal != fg:
            out, via = fg, "literal_vs_fact_gold"
        else:
            out, via = literal, "literal_canonical"
    else:
        ft = (fact_text or "").strip()
        if ft:
            r_ctx = resolve_relation(rel, ft, REL_MAP)
            if r_ctx != "unknown":
                if fg is not None and r_ctx != fg:
                    out, via = fg, "resolve_relation+fact_gold_override"
                else:
                    out, via = r_ctx, "resolve_relation"
            elif mapped is not None:
                if fg is not None and mapped != fg:
                    out, via = fg, "rel_map+fact_gold_override"
                else:
                    out, via = mapped, "rel_map"
            elif fg is not None:
                out, via = fg, "fact_gold_only"
            else:
                out, via = "unknown", "none"
        elif mapped is not None:
            if fg is not None and mapped != fg:
                out, via = fg, "rel_map_no_fact_text+fact_gold_override"
            else:
                out, via = mapped, "rel_map_no_fact_text"
        elif fg is not None:
            out, via = fg, "fact_gold_only_no_fact_text"
        else:
            out, via = "unknown", "none"

    _logger.debug(
        "normalize_relation_label raw=%r -> %r (via=%s; map_hit=%s fact_gold=%s)",
        rel_raw,
        out,
        via,
        mapped is not None,
        fg,
    )
    return out


def _relation_suggests_forward_agent_slot(rl: str) -> bool:
    """True when the usual reading is (agent/cause/drug, relation, outcome/other)."""
    x = (rl or "").lower().replace(" ", "_").replace("-", "_")
    if not x:
        return False
    if "by" in x and any(
        p in x
        for p in (
            "caused_by",
            "treated_by",
            "prevented_by",
            "increased_by",
            "reduced_by",
            "metabolized_by",
        )
    ):
        return False
    return any(
        k in x
        for k in (
            "cause",
            "treat",
            "prevent",
            "increase",
            "reduce",
            "interact",
            "metaboliz",
            "associat",
        )
    )


def normalize_passive_entity_slots(subj: str, rel: str, obj: str) -> Tuple[str, str, str]:
    """
    Fix passive-voice leakage into entity strings and wrong slot assignment.

    If agentive scaffolding (e.g. ``caused by``) appears only on the object span for a
    forward causal relation, swap endpoints then strip. Always strip remaining
    leading scaffolding from both spans.
    """
    s = (subj or "").strip()
    r = (rel or "").strip()
    o = (obj or "").strip()
    rl = r.lower().replace(" ", "_").replace("-", "_")
    forward = _relation_suggests_forward_agent_slot(rl)
    hs, ho = had_agentive_leading_scaffolding(s), had_agentive_leading_scaffolding(o)
    if forward and ho and not hs:
        s, o = o, s
    s = strip_agentive_surface_prefix(s)
    o = strip_agentive_surface_prefix(o)
    return s, r, o


def preprocess_extractor_triple(subj: str, rel: str, obj: str) -> Tuple[str, str, str]:
    """
    Structural fixes for passive / inverse phrasing (swap endpoints only; no label-ID table).
    """
    s = (subj or "").strip()
    r = (rel or "").strip()
    o = (obj or "").strip()
    rl = r.lower().replace(" ", "_").replace("-", "_")

    if "caused_by" in rl or rl.endswith("_cause_of") or rl == "cause_of":
        s, r, o = o, r, s
    elif "treated_by" in rl or "treats_by" in rl:
        s, r, o = o, r, s
    elif "prevented_by" in rl or "prevents_by" in rl:
        s, r, o = o, r, s
    elif "increased_by" in rl or "increases_by" in rl:
        s, r, o = o, r, s
    elif "reduced_by" in rl or "reduces_by" in rl:
        s, r, o = o, r, s
    elif rl in {"metabolizes", "metabolize"} or rl.startswith("metabolizes_"):
        s, r, o = o, r, s
    return normalize_passive_entity_slots(s, r, o)


def _strip_leading_article(phrase: str) -> str:
    p = (phrase or "").strip()
    low = p.lower()
    for art in ("the ", "a ", "an "):
        if low.startswith(art):
            return p[len(art) :].strip()
    return p


def _tail_obj(m: re.Match) -> str:
    return _strip_leading_article(m.group(2).strip())


_RX = lambda p: re.compile(p, re.IGNORECASE | re.DOTALL)


# Ordered: specific patterns before broad ones. ``relation`` aligns with qa_100 graph schema.
_FACT_PARSE_SPECS: Tuple[dict, ...] = (
    {"pattern": _RX(r"^(.+?)\s+is\s+metabolized\s+by\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "metabolized_by"},
    {"pattern": _RX(r"^(.+?)\s+is\s+used\s+to\s+treat\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "treats"},
    {"pattern": _RX(r"^(.+?)\s+can\s+cause\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "causes"},
    {"pattern": _RX(r"^(.+?)\s+increases\s+the\s+risk\s+of\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "increases"},
    {"pattern": _RX(r"^(.+?)\s+can\s+reduce\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "reduces"},
    {"pattern": _RX(r"^(.+?)\s+interacts\s+with\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "interacts_with"},
    {"pattern": _RX(r"^(.+?)\s+regulates\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "causes"},
    {"pattern": _RX(r"^(.+?)\s+relieves\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "reduces"},
    {"pattern": _RX(r"^(.+?)\s+suppress(?:es)?\s+the\s+immune\s+system\s*\.\s*$"), "kind": "immune", "relation": "causes"},
    {"pattern": _RX(r"^(.+?)\s+causes\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "causes"},
    {"pattern": _RX(r"^(.+?)\s+increases\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "increases"},
    {"pattern": _RX(r"^(.+?)\s+prevents\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "prevents"},
    {"pattern": _RX(r"^(.+?)\s+prevent\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "prevents"},
    {"pattern": _RX(r"^(.+?)\s+lower(?:s)?\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "reduces"},
    {"pattern": _RX(r"^(.+?)\s+reduce(?:s)?\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "reduces"},
    {"pattern": _RX(r"^(.+?)\s+treats\s+(.+?)\s*\.\s*$"), "kind": "tail", "relation": "treats"},
)


def _match_fact_spec(fact: str) -> Optional[Tuple[str, str, str]]:
    t = (fact or "").strip()
    if not t:
        return None

    m0 = re.match(
        r"^(.+?)\s+is\s+a\s+diuretic\s*\.\s*$",
        t,
        re.IGNORECASE | re.DOTALL,
    )
    if m0:
        return (m0.group(1).strip(), "causes", "diuretic")

    for spec in _FACT_PARSE_SPECS:
        m = spec["pattern"].match(t)
        if not m:
            continue
        head = m.group(1).strip()
        rel = str(spec["relation"])
        if spec["kind"] == "immune":
            return (head, rel, "the immune system")
        return (head, rel, _tail_obj(m))
    return None


def parse_fact_subject_object(fact: str) -> Optional[Tuple[str, str]]:
    """
    Extract (drug-like subject span, outcome/object span) from the seed fact sentence.
    Used to orient extracted triples drug -> outcome when extraction flips arguments.
    """
    hit = _match_fact_spec(fact)
    if not hit:
        return None
    return hit[0], hit[2]


def parse_fact_spans(fact: str) -> Optional[Tuple[str, str, str]]:
    """Parse (subject, relation, object) for alignment tooling; schema matches qa_100 relations."""
    hit = _match_fact_spec(fact)
    if not hit:
        return None
    return hit
