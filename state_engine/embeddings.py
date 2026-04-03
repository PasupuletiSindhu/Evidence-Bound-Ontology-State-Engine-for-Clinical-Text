"""
Sentence embedding utilities for semantic entity clustering.

Deterministic: no randomness; same text always maps to the same vector (fixed model, greedy cache).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _norm_key(text: str) -> str:
    from state_engine.semantic_canonicalizer import normalize_surface

    return normalize_surface(text or "")

_MODEL_NAME_DEFAULT = "all-MiniLM-L6-v2"

_model = None
_model_load_error: Optional[BaseException] = None
_embedding_cache: Dict[str, np.ndarray] = {}


def _get_sentence_transformer():
    global _model, _model_load_error
    if _model is not None:
        return _model
    if _model_load_error is not None:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        _model_load_error = e
        return None
    try:
        _model = SentenceTransformer(_MODEL_NAME_DEFAULT)
    except Exception as e:  # pragma: no cover
        _model_load_error = e
        return None
    return _model


def is_embedding_backend_available() -> bool:
    return _get_sentence_transformer() is not None


def get_embedding(text: str) -> np.ndarray:
    """
    Return L2-friendly sentence embedding for ``text`` (normalized surface key).

    Caches by normalized surface string. Raises RuntimeError if the model cannot load.
    Callers that need a soft fallback should use ``get_embedding_or_none``.
    """
    key = _norm_key(text)
    if not key:
        return np.zeros(384, dtype=np.float64)

    if key in _embedding_cache:
        return _embedding_cache[key]

    model = _get_sentence_transformer()
    if model is None:
        raise RuntimeError(
            f"sentence-transformers model unavailable: {_model_load_error!r}"
        )

    vec = model.encode([key], convert_to_numpy=True, show_progress_bar=False)[0]
    arr = np.asarray(vec, dtype=np.float64)
    _embedding_cache[key] = arr
    return arr


def get_embedding_or_none(text: str) -> Optional[np.ndarray]:
    try:
        return get_embedding(text)
    except RuntimeError:
        return None


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Batch-encode strings with the same SentenceTransformer as entity clustering.
    Rows are L2-normalized when the backend supports it (cosine = dot product).
    Returns shape (len(texts), dim); on backend failure, zero matrix.
    """
    tx = [(x or "").strip() for x in texts]
    if not tx:
        return np.zeros((0, 384), dtype=np.float64)
    model = _get_sentence_transformer()
    if model is None:
        return np.zeros((len(tx), 384), dtype=np.float64)
    arr = model.encode(
        tx,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(arr, dtype=np.float64)


def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] scale; undefined norms -> 0.0."""
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def text_cosine_similarity(a: str, b: str) -> float:
    """
    Cosine similarity between embeddings of two strings (cached per normalized surface).
    If backend missing, returns 1.0 iff normalize_surface equal else 0.0.
    """
    sa = _norm_key(a)
    sb = _norm_key(b)
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0
    ea = get_embedding_or_none(a)
    eb = get_embedding_or_none(b)
    if ea is None or eb is None:
        return 0.0
    return cosine_similarity_vectors(ea, eb)
