"""
Batch entity merging via agglomerative clustering on embeddings (cosine).

Remaps triple strings to per-cluster representatives; deduplicates (subject, relation)
objects by frequency in the graph. No domain synonym tables.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from state_engine.embeddings import embed_texts, is_embedding_backend_available

logger = logging.getLogger(__name__)

Triple = Tuple[str, str, str]
Triple5 = Tuple[str, str, str, float, str]

T = TypeVar("T")


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = (x or "").strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def agglomerative_entity_map(
    labels: Sequence[str],
    merge_threshold: float = 0.88,
) -> Dict[str, str]:
    """
    Map each surface string to a canonical representative for its cluster.

    Cluster merge if cosine similarity >= merge_threshold (average linkage on
    cosine distance = 1 - similarity). Falls back to identity map if sklearn or
    embeddings are unavailable.
    """
    uniq = _unique_preserve_order(labels)
    if len(uniq) <= 1:
        return {u: u for u in uniq}

    if not is_embedding_backend_available():
        logger.debug("agglomerative_entity_map: no embedding backend; identity map")
        return {u: u for u in uniq}

    X = embed_texts(list(uniq))
    if X.shape[0] != len(uniq) or np.allclose(X, 0.0):
        return {u: u for u in uniq}

    d_thresh = float(max(1e-6, 1.0 - merge_threshold))
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as ex:  # pragma: no cover
        logger.debug("agglomerative_entity_map: sklearn unavailable (%s); greedy merge", ex)
        return _greedy_merge_map(uniq, X, merge_threshold)

    S = cosine_similarity(X)
    np.fill_diagonal(S, 1.0)
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=d_thresh,
        metric="precomputed",
        linkage="average",
    )
    try:
        lab = clustering.fit_predict(D)
    except Exception as ex:
        logger.debug("agglomerative_entity_map: clustering failed (%s); greedy merge", ex)
        return _greedy_merge_map(uniq, X, merge_threshold)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for u, c in zip(uniq, lab):
        clusters[int(c)].append(u)

    mapping: Dict[str, str] = {}
    merge_log: List[dict] = []
    for _cid, members in sorted(clusters.items()):
        members = sorted(members)
        rep = min(members, key=lambda x: (len(x.strip()), x.lower()))
        for m in members:
            mapping[m] = rep
        if len(members) > 1:
            merge_log.append(
                {
                    "representative": rep,
                    "cluster_size": len(members),
                    "members": members,
                }
            )

    if merge_log:
        logger.debug(
            "agglomerative_entity_map: %d clusters merged from %d labels (threshold=%.3f)",
            len(merge_log),
            len(uniq),
            merge_threshold,
        )
        for row in merge_log[:12]:
            logger.debug(
                "  cluster rep=%r size=%d members=%s",
                row["representative"],
                row["cluster_size"],
                row["members"],
            )
    return mapping


def _greedy_merge_map(
    uniq: List[str],
    X: np.ndarray,
    merge_threshold: float,
) -> Dict[str, str]:
    """Fallback: union-find style merge on pairwise cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Xn = X / norms
    S = Xn @ Xn.T
    n = len(uniq)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= merge_threshold:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    mapping: Dict[str, str] = {}
    for _root, idxs in groups.items():
        members = [uniq[i] for i in idxs]
        rep = min(members, key=lambda x: (len(x.strip()), x.lower()))
        for m in members:
            mapping[m] = rep
    return mapping


def collect_entity_strings_from_triples(triples: Sequence[Triple]) -> List[str]:
    out: List[str] = []
    for t in triples:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            out.append(str(t[0]).strip())
            out.append(str(t[2]).strip())
    return out


def remap_triple(t: Triple, m: Dict[str, str]) -> Triple:
    s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
    return (m.get(s, s), r, m.get(o, o))


def remap_triple_list(triples: Sequence[Triple], m: Dict[str, str]) -> List[Triple]:
    return [remap_triple(tuple(t), m) for t in triples]


def remap_triple5_list(rows: Sequence[Triple5], m: Dict[str, str]) -> List[Triple5]:
    out: List[Triple5] = []
    for t in rows:
        s, r, o, c, pol = t[0], t[1], t[2], t[3], t[4]
        ns, no = m.get(s, s), m.get(o, o)
        out.append((ns, r, no, float(c), pol))
    return out


def dedupe_subject_relation_objects(triples: Sequence[Triple]) -> Tuple[List[Triple], int]:
    """
    For each (subject, relation), keep one object: highest frequency in this list,
    then shortest string.
    """
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
        if s and r and o:
            buckets[(s, r)].append(o)
    out: List[Triple] = []
    removed = 0
    for (s, r), objs in sorted(buckets.items()):
        ctr = Counter(objs)
        max_ct = max(ctr.values())
        cands = [o for o, c in ctr.items() if c == max_ct]
        best = min(cands, key=lambda x: (len(x), x.lower()))
        removed += max(0, len(objs) - 1)
        out.append((s, r, best))
    return out, removed


def dedupe_identical_triples(triples: Sequence[Triple]) -> List[Triple]:
    seen = set()
    out: List[Triple] = []
    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        k = (str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip())
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def apply_batch_entity_pipeline(
    triple_lists: Sequence[Sequence[Triple]],
    merge_threshold: float = 0.88,
    *,
    do_cluster: bool = True,
    do_sr_dedupe: bool = True,
) -> Tuple[List[List[Triple]], Dict[str, str], Dict[str, int]]:
    """
    Union all entities across ``triple_lists``, cluster, then remap each list and
    optionally dedupe (s,r)->o.

    Returns (remapped_lists, entity_map, stats).
    """
    all_entities: List[str] = []
    for g in triple_lists:
        all_entities.extend(collect_entity_strings_from_triples(g))
    stats = {"raw_entities": len(_unique_preserve_order(all_entities))}
    if do_cluster and all_entities:
        emap = agglomerative_entity_map(all_entities, merge_threshold=merge_threshold)
    else:
        emap = {u: u for u in _unique_preserve_order(all_entities)}
    stats["cluster_reps"] = len(set(emap.values()))

    remapped: List[List[Triple]] = []
    total_sr_removed = 0
    for g in triple_lists:
        rg = remap_triple_list(g, emap)
        rg = dedupe_identical_triples(rg)
        if do_sr_dedupe:
            rg, nrm = dedupe_subject_relation_objects(rg)
            total_sr_removed += nrm
        remapped.append(rg)
    stats["sr_dedupe_collapsed"] = total_sr_removed
    logger.debug(
        "apply_batch_entity_pipeline: entities %d -> %d reps, sr_collapses=%d",
        stats["raw_entities"],
        stats["cluster_reps"],
        total_sr_removed,
    )
    return remapped, emap, stats


def remap_exported_edges(edges: List[dict], m: Dict[str, str]) -> List[dict]:
    """Remap subject/object on engine-export edge dicts (copy)."""
    out: List[dict] = []
    for e in edges or []:
        if not isinstance(e, dict):
            continue
        d = dict(e)
        s = str(d.get("subject", "")).strip()
        o = str(d.get("object", "")).strip()
        d["subject"] = m.get(s, s)
        d["object"] = m.get(o, o)
        out.append(d)
    return out


def filter_weak_triples(triples: Sequence[Triple]) -> List[Triple]:
    """Exclude ``related_to`` edges for stability metrics."""
    out: List[Triple] = []
    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        if str(t[1]).strip().lower().replace(" ", "_") == "related_to":
            continue
        out.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))
    return out
