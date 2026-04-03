"""Map extractor triples to canonical relations and oriented (subject, object)."""
# Normalizes triples (subject, relation, object)
# Canonicalizes entities
# Orients triples (subject, object)
# Filters out low-value entities, identical endpoints, same-after-clustering endpoints, bad shape triples
# Collapses aligned triples by registry
# Deduplicates triples
# Returns list of triples (subject, relation, object, confidence, polarity) (canonicalized entities, oriented triples, filtered triples)

# First normalizes triples, then maps using UMLS mapper basically canonicalizes entities

import logging
import re
from typing import Dict, List, Optional, Tuple

from state_engine.canonicalize import enforce_seed_direction, orient_triple_nodes
from state_engine.entity_aliases import register_entity_alias
from state_engine.ontology import OntologyAligner
from state_engine.relations import (
    normalize_relation_label,
    parse_fact_subject_object,
    preprocess_extractor_triple,
)
from state_engine.entity_validity import (
    normalize_entity_surface_for_validity,
    peer_embedding_sample,
    should_drop_entity,
)
from state_engine.semantic_registry import EmbeddingEntityRegistry

logger = logging.getLogger(__name__)

# Weak evidence: unknown labels become related_to; literal related_to is down-weighted.
WEAK_RELATION_CONFIDENCE = 0.5

RawTriple = Tuple[str, str, str]
Triple5 = Tuple[str, str, str, float, str]

_TREAT_REL_CUE = re.compile(
    r"\btreat(?:ed|s|ing|ment)?\b",
    re.IGNORECASE,
)


def maybe_upgrade_prevents_to_treats(r_canon: str, raw_relation: str, preprocessed_r: str = "") -> str:
    """
    Relation map / clustering can map treatment cues to ``prevents``. If raw or
    preprocessed text clearly indicates treatment, use ``treats``.
    """
    if (r_canon or "").strip().lower() != "prevents":
        return r_canon
    cue = f"{raw_relation or ''} {preprocessed_r or ''}".lower()
    if _TREAT_REL_CUE.search(cue):
        return "treats"
    if "treated_with" in cue.replace(" ", "_") or "is_treated" in cue.replace(" ", "_"):
        return "treats"
    return r_canon


def collapse_aligned_triples_by_registry(
    aligned: List[Triple5],
    registry: Optional[EmbeddingEntityRegistry],
) -> List[Triple5]:
    """One triple per (canonical subj, rel, canonical obj, polarity); keep max confidence."""
    if registry is None or not aligned:
        return aligned
    buckets: Dict[Tuple[str, str, str, str], List[Triple5]] = {}
    for t in aligned:
        s, r, o, conf, pol = t[0], t[1], t[2], t[3], t[4]
        sj = registry.canonicalize_entity(s)
        oj = registry.canonicalize_entity(o)
        key = (sj, r, oj, pol)
        buckets.setdefault(key, []).append((sj, r, oj, float(conf), pol))
    out: List[Triple5] = []
    for group in buckets.values():
        out.append(max(group, key=lambda x: x[3]))
    # One object per (subject, relation, polarity): highest confidence, then shortest o.
    sr_pol_buckets: Dict[Tuple[str, str, str], List[Triple5]] = {}
    for row in out:
        s, r, o, conf, pol = row[0], row[1], row[2], row[3], row[4]
        sr_pol_buckets.setdefault((s, r, pol), []).append(row)
    collapsed: List[Triple5] = []
    n_sr = 0
    for _k, group in sr_pol_buckets.items():
        if len(group) == 1:
            collapsed.append(group[0])
            continue
        n_sr += len(group) - 1
        best = max(
            group,
            key=lambda x: (float(x[3]), -len(str(x[2]).strip())),
        )
        collapsed.append(best)
    if n_sr:
        logger.debug(
            "collapse_aligned_triples_by_registry: collapsed %d extra (s,r,pol) object variants",
            n_sr,
        )
    return collapsed

_GENERIC_NODES = {"condition", "drug", "high", "low", "risk", "effect"}
_GENERIC_OBJECT_PHRASES = {
    "patient",
    "patients",
    "person",
    "people",
    "a patient",
    "the patient",
    "a person",
    "the person",
    "drug",
    "a drug",
    "the drug",
    "medication",
    "a medication",
    "the medication",
    "medicine",
    "a medicine",
    "the medicine",
}


def _is_invalid_object(obj: str) -> bool:
    """
    Drop extractor-noise objects that are generic placeholders (hurts QA + creates drift).

    This is intentionally narrow: we do NOT enforce "len(obj.split())>1" because many
    valid biomedical objects are single tokens (e.g. 'liver', 'cholesterol').
    """
    t = normalize_entity_surface_for_validity(obj)
    t = re.sub(r"\s+", " ", (t or "").strip().lower())
    return (not t) or (t in _GENERIC_OBJECT_PHRASES)

# When the extractor returns no triples, seed one triple from the fact (recovery for empty runs).
_METABOLIZED_BY_FACT = re.compile(
    r"^(.+?)\s+is\s+metabolized\s+by\s+(.+?)\s*\.\s*$",
    re.IGNORECASE | re.DOTALL,
)


def fact_seeded_raw_triples(fact: str) -> List[RawTriple]:
    t = (fact or "").strip()
    if not t:
        return []
    m = _METABOLIZED_BY_FACT.match(t)
    if m:
        subj = m.group(1).strip()
        obj = re.sub(r"^(?:the|a|an)\s+", "", m.group(2).strip(), flags=re.IGNORECASE)
        return [(subj, "metabolized_by", obj)]
    return []


def _is_low_value_node(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in _GENERIC_NODES:
        return True
    if len(t) <= 2:
        return True
    if t.isdigit():
        return True
    return False


def _entity_peers_for_filter(
    registry: Optional[EmbeddingEntityRegistry],
    scratch_surfaces: Optional[List[str]],
) -> Optional[list]:
    reps: List[str] = []
    if registry is not None and registry._representatives:
        reps.extend(registry._representatives)
    if scratch_surfaces:
        reps.extend(scratch_surfaces)
    if len(reps) < 2:
        return None
    return peer_embedding_sample(reps, max_peers=48)


def _register_aliases_for_edge(subj: str, obj: str, raw_s: str, raw_o: str, s0: str, o0: str) -> None:
    register_entity_alias(subj, raw_s)
    register_entity_alias(subj, s0)
    register_entity_alias(subj, subj)
    register_entity_alias(obj, raw_o)
    register_entity_alias(obj, o0)
    register_entity_alias(obj, obj)


def _apply_embedding_clustering(
    subj: str,
    obj: str,
    raw_s: str,
    raw_o: str,
    s0: str,
    o0: str,
    registry: Optional[EmbeddingEntityRegistry],
) -> Tuple[str, str]:
    """Merge ontology-aligned entities within registry scope; record surfaces for QA."""
    if registry is None:
        return subj, obj
    sj = registry.canonicalize_entity(subj)
    oj = registry.canonicalize_entity(obj)
    for canon, surf in (
        (sj, raw_s),
        (sj, s0),
        (sj, subj),
        (oj, raw_o),
        (oj, o0),
        (oj, obj),
    ):
        registry.record_surface(canon, surf)
    return sj, oj


def prepare_triples_for_state(
    raw_block,
    fact_text: str,
    aligner: OntologyAligner,
    default_conf: float = 1.0,
    default_pol: str = "positive",
    entity_registry: Optional[EmbeddingEntityRegistry] = None,
    fact_gold_relation: Optional[str] = None,
    scratch_entity_surfaces: Optional[List[str]] = None,
) -> List[Triple5]:
    """
    Normalize relation labels from model output and canonicalize entities.
    """
    mfn = None
    if aligner.mapper is not None:
        mfn = aligner._mapper_fn()

    fact_so = parse_fact_subject_object(fact_text or "") or ("", "")
    fact_subj_guess, fact_obj_guess = fact_so

    out: List[Triple5] = []
    n_unknown_to_related = 0
    n_weak_related_literal = 0
    n_skip_bad_shape = 0
    n_skip_low_value = 0
    n_skip_same_node = 0
    n_skip_same_after_cluster = 0

    peers_emb = _entity_peers_for_filter(entity_registry, scratch_entity_surfaces)

    for t in raw_block or []:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            n_skip_bad_shape += 1
            continue
        raw_s, raw_r, raw_o = str(t[0]), str(t[1]), str(t[2])
        s, r, o = preprocess_extractor_triple(raw_s, raw_r, raw_o)
        s = normalize_entity_surface_for_validity(s)
        o = normalize_entity_surface_for_validity(o)
        r_canon = normalize_relation_label(
            r, fact_gold_relation=fact_gold_relation, fact_text=fact_text
        )
        r_canon = maybe_upgrade_prevents_to_treats(r_canon, raw_r, r)
        conf = float(default_conf)
        # Keep unknown/related_to triples: map unknown -> related_to and down-weight (weak).
        if r_canon == "unknown":
            r_canon = "related_to"
            conf *= WEAK_RELATION_CONFIDENCE
            n_unknown_to_related += 1
        elif r_canon == "related_to":
            conf *= WEAK_RELATION_CONFIDENCE
            n_weak_related_literal += 1
        s0, o0 = orient_triple_nodes(s, o, fact_subj_guess, fact_obj_guess, mapper_normalize=mfn)
        s0, o0 = enforce_seed_direction(
            s0, o0, fact_subj_guess, fact_obj_guess, r_canon, mapper_normalize=mfn
        )
        subj = aligner.normalize_entity(s0)
        obj = aligner.normalize_entity(o0)
        subj = normalize_entity_surface_for_validity(subj)
        obj = normalize_entity_surface_for_validity(obj)
        if _is_invalid_object(obj):
            n_skip_low_value += 1
            logger.debug(
                "prepare_triples_for_state: dropped generic object (%r) raw=(%r,%r,%r)",
                obj,
                raw_s,
                raw_r,
                raw_o,
            )
            continue
        drop_s, rs = should_drop_entity(subj, peer_embeddings=peers_emb)
        drop_o, ro = should_drop_entity(obj, peer_embeddings=peers_emb)
        if _is_low_value_node(subj) or _is_low_value_node(obj) or drop_s or drop_o:
            n_skip_low_value += 1
            logger.debug(
                "prepare_triples_for_state: dropped entity filter (subj=%r obj=%r) reasons=%s/%s raw=(%r,%r,%r)",
                subj,
                obj,
                rs,
                ro,
                raw_s,
                raw_r,
                raw_o,
            )
            continue
        if subj == obj:
            n_skip_same_node += 1
            logger.debug(
                "prepare_triples_for_state: dropped identical endpoints after normalize (%r,%r,%r)",
                subj,
                r_canon,
                obj,
            )
            continue
        subj, obj = _apply_embedding_clustering(
            subj, obj, raw_s, raw_o, s0, o0, entity_registry
        )
        if subj == obj:
            n_skip_same_after_cluster += 1
            logger.debug(
                "prepare_triples_for_state: dropped identical endpoints after clustering (%r,%r,%r)",
                subj,
                r_canon,
                obj,
            )
            continue
        _register_aliases_for_edge(subj, obj, raw_s, raw_o, s0, o0)
        out.append((subj, r_canon, obj, conf, default_pol))

    out = collapse_aligned_triples_by_registry(out, entity_registry)

    # Deduplicate after pruning/augmentation.
    seen = set()
    deduped: List[Triple5] = []
    for item in out:
        k = (item[0], item[1], item[2], item[4])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(item)
    n_deduped = len(out) - len(deduped)
    logger.debug(
        "prepare_triples_for_state: kept=%d dropped={bad_shape:%d,low_value:%d,same_node:%d,same_after_cluster:%d,dedup:%d} weak={unknown→related_to:%d,literal_related_to:%d}",
        len(deduped),
        n_skip_bad_shape,
        n_skip_low_value,
        n_skip_same_node,
        n_skip_same_after_cluster,
        n_deduped,
        n_unknown_to_related,
        n_weak_related_literal,
    )
    return deduped


def prepare_triples_fallback(
    raw_block,
    fact_text: str,
    aligner: OntologyAligner,
    default_conf: float = 1.0,
    default_pol: str = "positive",
    entity_registry: Optional[EmbeddingEntityRegistry] = None,
    fact_gold_relation: Optional[str] = None,
    scratch_entity_surfaces: Optional[List[str]] = None,
) -> List[Triple5]:
    """
    Relaxed path when strict normalization drops everything: no mapper on entities,
    same relation / direction preprocessing. Used only when incremental graph is empty.
    """
    mfn = aligner._mapper_fn() if aligner.mapper is not None else None
    fact_so = parse_fact_subject_object(fact_text or "") or ("", "")
    fact_subj_guess, fact_obj_guess = fact_so
    out: List[Triple5] = []
    n_unknown_to_related = 0
    n_weak_related_literal = 0
    n_skip_bad_shape = 0
    n_skip_low_value = 0
    n_skip_same_node = 0
    n_skip_same_after_cluster = 0
    peers_emb_fb = _entity_peers_for_filter(entity_registry, scratch_entity_surfaces)
    for t in raw_block or []:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            n_skip_bad_shape += 1
            continue
        raw_s, raw_r, raw_o = str(t[0]), str(t[1]), str(t[2])
        s, r, o = preprocess_extractor_triple(raw_s, raw_r, raw_o)
        s = normalize_entity_surface_for_validity(s)
        o = normalize_entity_surface_for_validity(o)
        r_canon = normalize_relation_label(
            r, fact_gold_relation=fact_gold_relation, fact_text=fact_text
        )
        r_canon = maybe_upgrade_prevents_to_treats(r_canon, raw_r, r)
        conf = float(default_conf)
        if r_canon == "unknown":
            r_canon = "related_to"
            conf *= WEAK_RELATION_CONFIDENCE
            n_unknown_to_related += 1
        elif r_canon == "related_to":
            conf *= WEAK_RELATION_CONFIDENCE
            n_weak_related_literal += 1
        s0, o0 = orient_triple_nodes(s, o, fact_subj_guess, fact_obj_guess, mapper_normalize=mfn)
        s0, o0 = enforce_seed_direction(
            s0, o0, fact_subj_guess, fact_obj_guess, r_canon, mapper_normalize=mfn
        )
        subj = aligner.normalize_entity_light(s0)
        obj = aligner.normalize_entity_light(o0)
        subj = normalize_entity_surface_for_validity(subj)
        obj = normalize_entity_surface_for_validity(obj)
        if _is_invalid_object(obj):
            n_skip_low_value += 1
            logger.debug(
                "prepare_triples_fallback: dropped generic object (%r) raw=(%r,%r,%r)",
                obj,
                raw_s,
                raw_r,
                raw_o,
            )
            continue
        drop_s, rs = should_drop_entity(subj, peer_embeddings=peers_emb_fb)
        drop_o, ro = should_drop_entity(obj, peer_embeddings=peers_emb_fb)
        if _is_low_value_node(subj) or _is_low_value_node(obj) or drop_s or drop_o:
            n_skip_low_value += 1
            logger.debug(
                "prepare_triples_fallback: dropped entity filter (subj=%r obj=%r) reasons=%s/%s raw=(%r,%r,%r)",
                subj,
                obj,
                rs,
                ro,
                raw_s,
                raw_r,
                raw_o,
            )
            continue
        if subj == obj:
            n_skip_same_node += 1
            logger.debug(
                "prepare_triples_fallback: dropped identical endpoints after normalize (%r,%r,%r)",
                subj,
                r_canon,
                obj,
            )
            continue
        subj, obj = _apply_embedding_clustering(
            subj, obj, raw_s, raw_o, s0, o0, entity_registry
        )
        if subj == obj:
            n_skip_same_after_cluster += 1
            logger.debug(
                "prepare_triples_fallback: dropped identical endpoints after clustering (%r,%r,%r)",
                subj,
                r_canon,
                obj,
            )
            continue
        _register_aliases_for_edge(subj, obj, raw_s, raw_o, s0, o0)
        out.append((subj, r_canon, obj, conf, default_pol))

    out = collapse_aligned_triples_by_registry(out, entity_registry)

    seen = set()
    deduped: List[Triple5] = []
    for item in out:
        k = (item[0], item[1], item[2], item[4])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(item)
    n_deduped = len(out) - len(deduped)
    logger.debug(
        "prepare_triples_fallback: kept=%d dropped={bad_shape:%d,low_value:%d,same_node:%d,same_after_cluster:%d,dedup:%d} weak={unknown→related_to:%d,literal_related_to:%d}",
        len(deduped),
        n_skip_bad_shape,
        n_skip_low_value,
        n_skip_same_node,
        n_skip_same_after_cluster,
        n_deduped,
        n_unknown_to_related,
        n_weak_related_literal,
    )
    return deduped
