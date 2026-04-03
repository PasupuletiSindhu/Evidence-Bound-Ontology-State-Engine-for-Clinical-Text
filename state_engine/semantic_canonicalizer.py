"""Text normalization and embedding-based semantic equality (no global synonym table)."""
# Same semantic meaning is not identified easily so we use embeddings

import re
import unicodedata

_clean_re = re.compile(r"[^a-z0-9 ]+")

def _rewrite_domain_synonyms(normalized: str) -> str:
    """
    Small, explicit surface rewrites for high-value demo/clinical paraphrase variants.

    This runs *after* basic normalization (lowercase + punctuation stripping), and is
    intentionally conservative to avoid over-merging unrelated entities.
    """
    t = (normalized or "").strip()
    if not t:
        return t

    # Gastric bleeding variants (keeps "bleeding" generic unless stomach/gastric present).
    if "bleeding" in t:
        if (
            t in {"bleeding in the stomach", "stomach bleeding", "bleeding in stomach"}
            or ("stomach" in t and "bleeding" in t)
            or ("gastric" in t and "bleeding" in t)
        ):
            return "gastric bleeding"

    return t

_AGENTIVE_LEADING = re.compile(
    r"^\s*(?:"
    r"caused\s+by|"
    r"prevented\s+by|"
    r"triggered\s+by|"
    r"induced\s+by|"
    r"mediated\s+by|"
    r"treat(?:ed|ing)?\s+with|"
    r"managed\s+with|"
    r"managed\s+by|"
    r"due\s+to|"
    r"metabolized\s+by|"
    r"metabolised\s+by|"
    r"increased\s+by|"
    r"reduced\s+by|"
    r"associated\s+with"
    r")\s+",
    re.IGNORECASE,
)


def strip_agentive_surface_prefix(text: str, *, max_rounds: int = 4) -> str:
    """
    Remove one or more leading passive/agentive English fragments from a span.

    Examples: ``caused by aspirin`` → ``aspirin``; ``treated with metformin`` → ``metformin``.
    """
    t = (text or "").strip()
    if not t:
        return t
    for _ in range(max_rounds):
        n = _AGENTIVE_LEADING.sub("", t).strip()
        if n == t:
            break
        t = n
    return t


def had_agentive_leading_scaffolding(text: str) -> bool:
    """True if :func:`strip_agentive_surface_prefix` would shorten the span."""
    t = (text or "").strip()
    if not t:
        return False
    return strip_agentive_surface_prefix(t) != t


def normalize_surface(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.lower().strip()
    text = _clean_re.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = _rewrite_domain_synonyms(text)
    return text


def semantic_canonicalize(text: str) -> str:
    """
    Normalization-only stage before ontology mapper / embedding registry.

    Learned clustering happens in :class:`state_engine.semantic_registry.EmbeddingEntityRegistry`,
    not here, to avoid hidden global state and cross-run coupling.
    """
    return normalize_surface(text)


def semantic_equal(a: str, b: str, threshold: float = 0.85) -> bool:
    """
    True if strings are embedding-similar enough (or identical after surface normalize).
    Uses shared embedding cache from ``state_engine.embeddings``.
    """
    if not a or not b:
        return False
    if normalize_surface(a) == normalize_surface(b):
        return True
    try:
        from state_engine.embeddings import text_cosine_similarity
    except Exception:
        return False
    return text_cosine_similarity(a, b) >= threshold
