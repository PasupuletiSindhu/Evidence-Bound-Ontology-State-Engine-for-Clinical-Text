from typing import Callable, Optional

from state_engine.semantic_canonicalizer import (
    semantic_canonicalize,
    semantic_equal,
    strip_agentive_surface_prefix,
)


def _surface_aligns(a: str, b: str) -> bool:
    x = (a or "").strip().lower()
    y = (b or "").strip().lower()
    if not x or not y:
        return False
    if x == y:
        return True
    if len(x) >= 3 and len(y) >= 3 and (x in y or y in x):
        return True
    return False


def canonical_entity(
    text: str,
    *,
    mapper_normalize: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    if not text:
        return text
    text = strip_agentive_surface_prefix((text or "").strip())
    if not text:
        return text
    if mapper_normalize:
        mapped = mapper_normalize(text)
        if mapped:
            text = mapped
    return semantic_canonicalize(text)


def orient_triple_nodes(
    subj: str,
    obj: str,
    fact_subj: str,
    fact_obj: str,
    *,
    mapper_normalize: Optional[Callable[[str], Optional[str]]] = None,
) -> tuple:
    """Swap (subj, obj) if extraction reversed w.r.t. seed fact arguments."""
    cs = canonical_entity(subj, mapper_normalize=mapper_normalize)
    co = canonical_entity(obj, mapper_normalize=mapper_normalize)
    fs = canonical_entity(fact_subj, mapper_normalize=mapper_normalize)
    fo = canonical_entity(fact_obj, mapper_normalize=mapper_normalize)

    if semantic_equal(cs, fs) and semantic_equal(co, fo):
        return subj, obj
    if semantic_equal(cs, fo) and semantic_equal(co, fs):
        return obj, subj

    fsu = (fact_subj or "").strip()
    fou = (fact_obj or "").strip()
    if fsu and fou:
        sr = (subj or "").strip()
        oraw = (obj or "").strip()
        if _surface_aligns(sr, fsu) and _surface_aligns(oraw, fou):
            return subj, obj
        if _surface_aligns(sr, fou) and _surface_aligns(oraw, fsu):
            return obj, subj

    return subj, obj


def enforce_seed_direction(
    subj: str,
    obj: str,
    fact_subj: str,
    fact_obj: str,
    relation: str,
    *,
    mapper_normalize: Optional[Callable[[str], Optional[str]]] = None,
    margin: float = 0.02,
) -> tuple:
    """
    Pick (subj, obj) orientation by **embedding similarity** to seed fact endpoints
    (no relation-type or domain-type rules). Falls back to lexical alignment if
    embeddings are unavailable.
    """
    _ = relation  # unused; kept for API compatibility
    fs = (fact_subj or "").strip()
    fo = (fact_obj or "").strip()
    if not fs or not fo:
        return subj, obj

    try:
        from state_engine.embeddings import text_cosine_similarity

        direct = text_cosine_similarity(subj, fs) + text_cosine_similarity(obj, fo)
        flipped = text_cosine_similarity(subj, fo) + text_cosine_similarity(obj, fs)
        if flipped > direct + margin:
            return obj, subj
        if direct > flipped + margin:
            return subj, obj
    except Exception:
        pass

    cs = canonical_entity(subj, mapper_normalize=mapper_normalize)
    co = canonical_entity(obj, mapper_normalize=mapper_normalize)
    cfs = canonical_entity(fs, mapper_normalize=mapper_normalize)
    cfo = canonical_entity(fo, mapper_normalize=mapper_normalize)

    subj_fs = semantic_equal(cs, cfs) or _surface_aligns(subj, fs)
    subj_fo = semantic_equal(cs, cfo) or _surface_aligns(subj, fo)
    obj_fs = semantic_equal(co, cfs) or _surface_aligns(obj, fs)
    obj_fo = semantic_equal(co, cfo) or _surface_aligns(obj, fo)

    score_dir = int(subj_fs) + int(obj_fo)
    score_flip = int(subj_fo) + int(obj_fs)
    if score_flip > score_dir:
        return obj, subj
    return subj, obj
