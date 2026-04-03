"""
General-purpose entity surface validity (statistical / embedding heuristics).

No biomedical synonym tables: uses token/stopword statistics, optional spaCy POS,
embedding norm, and optional similarity to peer entities in the working set.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from state_engine.semantic_canonicalizer import strip_agentive_surface_prefix

logger = logging.getLogger(__name__)

_ENUM_PREFIX = re.compile(r"^\s*\d+[\.\)\:]\s*")
_LEADING_NUM = re.compile(r"^\s*\d+\s+")
_CARDINAL_PREFIX = re.compile(r"^\s*\d+\s+(.+)$")

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOP
except Exception:  # pragma: no cover
    _SK_STOP = frozenset()

_STOPWORDS = frozenset(w.lower() for w in _SK_STOP) if _SK_STOP else frozenset()


def strip_list_enumeration(text: str) -> str:
    """Remove leading enumerators (e.g. ``2. `` ``10) ``) — structural OCR/list noise."""
    t = (text or "").strip()
    if not t:
        return t
    t = _ENUM_PREFIX.sub("", t)
    t = _LEADING_NUM.sub("", t).strip()
    return t


def strip_leading_cardinal_phrase(text: str) -> str:
    """``10 aspirin`` → ``aspirin`` (leading count/token, not semantic rewriting)."""
    t = (text or "").strip()
    m = _CARDINAL_PREFIX.match(t)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return t


def normalize_entity_surface_for_validity(text: str) -> str:
    return strip_agentive_surface_prefix(
        strip_leading_cardinal_phrase(strip_list_enumeration(text))
    )


def _content_tokens(text: str) -> List[str]:
    t = (text or "").lower()
    return re.findall(r"[a-z0-9]+", t)


def _numeric_char_ratio(text: str) -> float:
    s = text or ""
    if not s:
        return 0.0
    return sum(1 for c in s if c.isdigit()) / max(len(s), 1)


def _stopword_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 1.0
    if not _STOPWORDS:
        return 0.0
    sw = sum(1 for t in tokens if t in _STOPWORDS)
    return sw / len(tokens)


def _optional_pos_tag(text: str):
    """Return spaCy coarse POS when the span is a single token (if spaCy is available)."""
    t = (text or "").strip()
    if not t:
        return None
    parts = t.split()
    if len(parts) != 1:
        return None
    tok = parts[0]
    if not tok:
        return None
    try:
        import spacy

        if not hasattr(_optional_pos_tag, "_nlp"):
            try:
                _optional_pos_tag._nlp = spacy.load("en_core_web_sm")  # type: ignore[attr-defined]
            except Exception:
                _optional_pos_tag._nlp = False  # type: ignore[attr-defined]
        nlp = getattr(_optional_pos_tag, "_nlp", False)
        if not nlp:
            return None
        doc = nlp(tok[:120])
        if not len(doc):
            return None
        return doc[0].pos_
    except Exception:
        return None


def should_drop_entity(
    text: str,
    *,
    peer_embeddings: Optional[List[np.ndarray]] = None,
    min_sim_to_any_peer: float = 0.22,
    min_norm: float = 1e-10,
) -> Tuple[bool, str]:
    """
    Return (drop, reason). Uses only generic linguistic / embedding signals.
    """
    raw = (text or "").strip()
    t = normalize_entity_surface_for_validity(raw)
    if not t:
        return True, "empty"
    if len(t) < 3:
        return True, "too_short"
    toks = _content_tokens(t)
    if not toks:
        return True, "no_content_tokens"
    if toks[0].isdigit():
        return True, "leading_numeric_token"
    if _numeric_char_ratio(t) >= 0.45:
        return True, "numeric_heavy"
    sr = _stopword_ratio(toks)
    if len(toks) == 1:
        if sr >= 1.0 or (toks[0] in _STOPWORDS if _STOPWORDS else False):
            return True, "single_stopword_token"
        pos = _optional_pos_tag(t)
        if pos is not None and pos not in ("NOUN", "PROPN", "ADJ", "X", "SYM"):
            return True, f"single_token_pos_{pos}"
    if sr >= 0.72 and len(toks) <= 4:
        return True, "stopword_heavy"

    try:
        from state_engine.embeddings import cosine_similarity_vectors, get_embedding_or_none

        emb = get_embedding_or_none(t)
        if emb is not None:
            nrm = float(np.linalg.norm(emb))
            if nrm < min_norm:
                return True, "low_embedding_norm"
            if peer_embeddings and len(peer_embeddings) >= 2:
                sims = [cosine_similarity_vectors(emb, pe) for pe in peer_embeddings if pe is not None]
                if sims and max(sims) < min_sim_to_any_peer:
                    return True, "isolated_from_peers"
    except Exception as exc:  # pragma: no cover
        logger.debug("should_drop_entity embedding check skipped: %s", exc)

    return False, "ok"


def peer_embedding_sample(strings: List[str], max_peers: int = 64) -> List[np.ndarray]:
    """Subset of embeddings for isolation checks (deterministic order)."""
    from state_engine.embeddings import get_embedding_or_none

    out: List[np.ndarray] = []
    for s in sorted(set(strings))[:max_peers]:
        e = get_embedding_or_none(s)
        if e is not None:
            out.append(e)
    return out
