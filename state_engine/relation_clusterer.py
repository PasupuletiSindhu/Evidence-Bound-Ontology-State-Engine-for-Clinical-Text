"""
Embedding-based relation clustering using **relation phrases**, not full facts.

Facts share similar entity-heavy surfaces; we extract a short, verb-centric cue per fact,
embed those cues, cluster with KMeans, optionally split impure clusters, then assign each
cluster a canonical relation by majority vote over cue→relation lookup.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_N_CLUSTERS = 7
DEFAULT_CLUSTER_MAP_RELATIVE = "results/relation_cluster_map.json"

# Short strings for SentenceTransformer — entity-neutral, verb-centric.
CUE_TO_CANONICAL_RELATION: Dict[str, str] = {
    "metabolized by": "metabolized_by",
    "treats disease": "treats",
    "treated with": "treats",
    "treats": "treats",
    "interacts with": "interacts_with",
    "increases risk": "increases",
    "prevents": "prevents",
    "causes adverse": "causes",
    "suppresses immune": "causes",
    "reduces symptom": "reduces",
    "increases effect": "increases",
    "regulates level": "causes",
    "diuretic drug": "causes",
    "general relation": "related_to",
}

_state: Optional[Dict[str, Any]] = None
_st_model_cache: Dict[str, Any] = {}


def extract_relation_phrase(fact_text: str) -> str:
    """
    Map a full fact to a compact relation cue (no drug/disease names).

    Order matches semantic specificity (e.g. ``can cause`` before bare ``treat``).
    """
    t = (fact_text or "").strip().lower()
    if not t:
        return "general relation"

    if re.search(r"\bmetabolized\s+by\b|\bmetabolised\s+by\b", t):
        return "metabolized by"
    if "is used to treat" in t or re.search(r"\bused\s+to\s+treat\b", t):
        return "treats disease"
    if re.search(r"\btreat(ed)?\s+with\b", t) or re.search(
        r"\bis\s+treated\s+with\b", t
    ):
        return "treated with"
    if re.search(r"\binteract\w*\s+with\b", t):
        return "interacts with"
    if re.search(r"\bincrease\w*\s+the\s+risk\b", t) or (
        "risk" in t and "increas" in t
    ):
        return "increases risk"
    if re.search(r"\bprevents?\b", t):
        return "prevents"
    if re.search(r"\bcauses\b", t) or re.search(r"\bcan\s+cause\b", t):
        return "causes adverse"
    if "suppress" in t and "immune" in t:
        return "suppresses immune"
    if re.search(r"\breduce\w*\b", t) or re.search(r"\blowers?\b", t) or re.search(
        r"\brelieve\w*", t
    ):
        return "reduces symptom"
    if re.search(r"\bincrease\w*\b", t) or re.search(r"\braises\b", t):
        return "increases effect"
    if re.search(r"\btreats\b", t) or re.search(r"\btreat\b", t):
        return "treats"
    if re.search(r"\bregulates\b", t):
        return "regulates level"
    if "diuretic" in t:
        return "diuretic drug"
    return "general relation"


def cue_to_canonical_relation(cue: str) -> str:
    return CUE_TO_CANONICAL_RELATION.get(cue.strip(), "related_to")


def embed_texts(texts: List[str], *, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """L2-normalized embeddings (one row per string)."""
    from sentence_transformers import SentenceTransformer

    tx = [(x or "").strip() for x in texts]
    model = _st_model_cache.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _st_model_cache[model_name] = model
    return np.asarray(
        model.encode(tx, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float64,
    )


def cluster_facts(
    embeddings: np.ndarray, k: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    lab = km.fit_predict(embeddings)
    ctr = np.asarray(km.cluster_centers_, dtype=np.float64)
    return lab, ctr


def majority_relation_for_cues(cues: List[str]) -> str:
    votes = Counter(cue_to_canonical_relation(c) for c in cues)
    if not votes:
        return "related_to"
    best_rel, _ = votes.most_common(1)[0]
    return best_rel


def split_impure_clusters(
    emb: np.ndarray,
    cues: List[str],
    labels: np.ndarray,
    *,
    random_state: int = 42,
    min_secondary: int = 2,
    min_fraction: float = 0.2,
) -> np.ndarray:
    """
    If a cluster mixes two or more canonical relations with enough support, run 2-means on
    that cluster's embeddings and split into two labels.
    """
    from sklearn.cluster import KMeans

    out = labels.astype(int).copy()
    next_id = int(out.max()) + 1

    for c in sorted(np.unique(out).tolist()):
        idx = np.where(out == c)[0]
        n = len(idx)
        if n < 4:
            continue
        rels = [cue_to_canonical_relation(cues[i]) for i in idx]
        cnt = Counter(rels)
        if len(cnt) < 2:
            continue
        (_, v1), (_, v2) = cnt.most_common(2)
        if v2 < max(min_secondary, int(min_fraction * n)):
            continue
        sub = emb[idx]
        sub_lab = KMeans(n_clusters=2, random_state=random_state, n_init=10).fit_predict(
            sub
        )
        move = idx[sub_lab == 1]
        if len(move) > 0 and len(move) < n:
            out[move] = next_id
            next_id += 1
            logger.info(
                "relation_clusterer: split cluster %s (n=%d, mixed relations %s)",
                c,
                n,
                dict(cnt),
            )
    return out


def refine_clusters_purity(
    emb: np.ndarray,
    cues: List[str],
    labels: np.ndarray,
    *,
    random_state: int = 42,
    max_rounds: int = 5,
    max_distinct_clusters: int = 14,
) -> np.ndarray:
    lab = labels.copy()
    for _ in range(max_rounds):
        if len(np.unique(lab)) >= max_distinct_clusters:
            break
        new_lab = split_impure_clusters(emb, cues, lab, random_state=random_state)
        if np.array_equal(new_lab, lab):
            break
        lab = new_lab
    return lab


def renumber_clusters_contiguous(
    labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    """Map arbitrary cluster ids to 0..C-1; return new labels, old→new, ordered old ids."""
    old_ids = sorted(np.unique(labels).astype(int).tolist())
    old_to_new = {oid: j for j, oid in enumerate(old_ids)}
    new_labels = np.array([old_to_new[int(x)] for x in labels], dtype=np.int32)
    return new_labels, old_to_new, old_ids


def centroids_for_labels(emb: np.ndarray, labels: np.ndarray, n_cluster: int) -> np.ndarray:
    rows = []
    for j in range(n_cluster):
        mask = labels == j
        if not np.any(mask):
            rows.append(np.zeros(emb.shape[1], dtype=np.float64))
        else:
            rows.append(emb[mask].mean(axis=0))
    return np.stack(rows, axis=0)


def _fit_kmeans_relation_clusters(
    facts: List[str],
    cues: List[str],
    *,
    emb: np.ndarray,
    n_clusters: int,
    model_name: str,
    random_state: int,
) -> Tuple[
    Dict[int, str],
    np.ndarray,
    List[str],
    np.ndarray,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[str],
    bool,
]:
    k = min(int(n_clusters), len(facts))
    if k < 2 and len(facts) >= 2:
        k = 2
    if k < 1:
        k = 1

    labels, _ = cluster_facts(emb, k=k, random_state=random_state)
    labels = refine_clusters_purity(emb, cues, labels, random_state=random_state)
    labels, _oldmap, _old_ids = renumber_clusters_contiguous(labels)
    n_c = int(labels.max()) + 1
    centroids = centroids_for_labels(emb, labels, n_c)

    cluster_to_relation: Dict[int, str] = {}
    cluster_summaries: List[Dict[str, Any]] = []

    for c in range(n_c):
        sub_cues = [cues[i] for i in range(len(facts)) if int(labels[i]) == c]
        sub_facts = [facts[i] for i in range(len(facts)) if int(labels[i]) == c]
        rel = majority_relation_for_cues(sub_cues)
        cluster_to_relation[c] = rel
        cluster_summaries.append(
            {
                "cluster_id": c,
                "relation": rel,
                "num_facts": len(sub_facts),
                "relation_phrases": sub_cues,
                "facts": sub_facts,
            }
        )

    fact_rows: List[Dict[str, Any]] = []
    for i, fact in enumerate(facts):
        cid = int(labels[i])
        rel = cluster_to_relation[cid]
        fact_rows.append(
            {
                "set_id": i,
                "fact": fact,
                "relation_phrase": cues[i],
                "cluster_id": cid,
                "relation": rel,
            }
        )

    return cluster_to_relation, centroids, facts, labels, fact_rows, cluster_summaries, cues, False


def fit_paraphrase_relation_clusters(
    paraphrase_sets: List[dict],
    *,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    model_name: str = DEFAULT_MODEL_NAME,
    random_state: int = 42,
) -> Tuple[
    Dict[int, str],
    np.ndarray,
    List[str],
    np.ndarray,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[str],
    bool,
]:
    facts = [str(s.get("fact", "")).strip() for s in paraphrase_sets or []]
    if not facts:
        raise ValueError("No facts in paraphrase sets.")

    cues = [extract_relation_phrase(f) for f in facts]
    emb = embed_texts(cues, model_name=model_name)
    n = len(facts)
    initial = [cue_to_canonical_relation(c) for c in cues]

    by_rel: Dict[str, List[int]] = {}
    for i, r in enumerate(initial):
        if r == "related_to":
            continue
        by_rel.setdefault(r, []).append(i)

    if not by_rel:
        logger.info(
            "relation_clusterer: no cue-anchored relations; falling back to KMeans (k<=%d)",
            n_clusters,
        )
        return _fit_kmeans_relation_clusters(
            facts,
            cues,
            emb=emb,
            n_clusters=n_clusters,
            model_name=model_name,
            random_state=random_state,
        )

    rel_order = sorted(by_rel.keys())
    cent_labeled = np.stack([emb[by_rel[r]].mean(axis=0) for r in rel_order], axis=0)

    final = list(initial)
    for i in range(n):
        if final[i] != "related_to":
            continue
        d = np.linalg.norm(cent_labeled - emb[i], axis=1)
        j = int(np.argmin(d))
        final[i] = rel_order[j]

    by_rel2: Dict[str, List[int]] = {}
    for i, r in enumerate(final):
        by_rel2.setdefault(r, []).append(i)

    unique_rels = sorted(by_rel2.keys(), key=lambda x: (x == "related_to", x))
    centroids = np.stack([emb[by_rel2[r]].mean(axis=0) for r in unique_rels], axis=0)
    rel_to_cid = {r: j for j, r in enumerate(unique_rels)}
    cluster_to_relation = {j: unique_rels[j] for j in range(len(unique_rels))}
    labels = np.array([rel_to_cid[r] for r in final], dtype=np.int32)

    cluster_summaries: List[Dict[str, Any]] = []
    for j, r in enumerate(unique_rels):
        idxs = by_rel2[r]
        cluster_summaries.append(
            {
                "cluster_id": j,
                "relation": r,
                "num_facts": len(idxs),
                "relation_phrases": [cues[i] for i in idxs],
                "facts": [facts[i] for i in idxs],
            }
        )

    fact_rows: List[Dict[str, Any]] = []
    for i, fact in enumerate(facts):
        cid = rel_to_cid[final[i]]
        fact_rows.append(
            {
                "set_id": i,
                "fact": fact,
                "relation_phrase": cues[i],
                "cluster_id": cid,
                "relation": final[i],
            }
        )

    logger.info(
        "relation_clusterer: supervised relation centroids (%d relations, %d facts)",
        len(unique_rels),
        n,
    )
    return cluster_to_relation, centroids, facts, labels, fact_rows, cluster_summaries, cues, True


def export_relation_cluster_map(
    *,
    cluster_to_relation: Dict[int, str],
    centroids: np.ndarray,
    fact_to_relation: Dict[str, str],
    fact_assignments: List[Dict[str, Any]],
    cluster_summaries: List[Dict[str, Any]],
    model_name: str,
    n_clusters: int,
    random_state: int,
    path: Path,
    embeds_relation_cues: bool = True,
    supervised_relation_centroids: bool = False,
) -> None:
    payload = {
        "model_name": model_name,
        "n_clusters": int(n_clusters),
        "random_state": int(random_state),
        "embeds_relation_cues": embeds_relation_cues,
        "supervised_relation_centroids": bool(supervised_relation_centroids),
        "cluster_to_relation": {str(k): v for k, v in sorted(cluster_to_relation.items())},
        "centroids": centroids.tolist(),
        "fact_to_relation": fact_to_relation,
        "fact_assignments": fact_assignments,
        "cluster_summaries": cluster_summaries,
        "cue_to_canonical_relation": CUE_TO_CANONICAL_RELATION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    logger.info("relation_clusterer: wrote %s (%d clusters)", path, n_clusters)


def fit_and_export_from_paraphrase_json(
    *,
    paraphrase_path: Path,
    cluster_map_out: Path,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    model_name: str = DEFAULT_MODEL_NAME,
    random_state: int = 42,
) -> Dict[str, Any]:
    data = json.loads(paraphrase_path.read_text(encoding="utf-8"))
    sets = data.get("sets") or []
    (
        cluster_to_relation,
        centroids,
        facts,
        _labels,
        fact_rows,
        cluster_summaries,
        _cues,
        supervised,
    ) = fit_paraphrase_relation_clusters(
        sets,
        n_clusters=n_clusters,
        model_name=model_name,
        random_state=random_state,
    )
    fact_to_relation = {row["fact"]: row["relation"] for row in fact_rows}
    k = len(cluster_to_relation)

    export_relation_cluster_map(
        cluster_to_relation=cluster_to_relation,
        centroids=centroids,
        fact_to_relation=fact_to_relation,
        fact_assignments=fact_rows,
        cluster_summaries=cluster_summaries,
        model_name=model_name,
        n_clusters=k,
        random_state=random_state,
        path=cluster_map_out,
        embeds_relation_cues=True,
        supervised_relation_centroids=supervised,
    )

    logger.info(
        "relation_clusterer: cluster -> relation: %s",
        {str(c): cluster_to_relation[c] for c in sorted(cluster_to_relation.keys())},
    )
    return {
        "cluster_to_relation": cluster_to_relation,
        "fact_assignments": fact_rows,
        "cluster_summaries": cluster_summaries,
    }


def load_cluster_map(path: str) -> None:
    global _state
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    centroids = np.asarray(raw["centroids"], dtype=np.float64)
    ctr = {int(k): str(v) for k, v in (raw.get("cluster_to_relation") or {}).items()}
    f2r = {str(k): str(v) for k, v in (raw.get("fact_to_relation") or {}).items()}
    _state = {
        "path": str(p.resolve()),
        "centroids": centroids,
        "cluster_to_relation": ctr,
        "fact_to_relation": f2r,
        "model_name": raw.get("model_name", DEFAULT_MODEL_NAME),
        "embeds_relation_cues": bool(raw.get("embeds_relation_cues", False)),
        "supervised_relation_centroids": bool(
            raw.get("supervised_relation_centroids", False)
        ),
    }
    logger.info("relation_clusterer: loaded cluster map from %s", p)


def _assign_full_fact_fallback(fact: str) -> str:
    """Legacy full-fact keyword stub when cluster map missing."""
    cue = extract_relation_phrase(fact)
    return cue_to_canonical_relation(cue)


def infer_fact_relation(fact_text: str) -> str:
    """
    Map seed fact → relation: **cue-first** (relation phrase → canonical); only if the cue is
    ``general relation`` do we embed and take the nearest centroid (supervised = one centroid
    per relation). Exact training-fact string still hits ``fact_to_relation`` when present.
    """
    fact = (fact_text or "").strip()
    if not fact:
        return "related_to"

    cue = extract_relation_phrase(fact)
    cue_rel = cue_to_canonical_relation(cue)
    if cue_rel != "related_to":
        return cue_rel

    if _state is None:
        try:
            root = Path(__file__).resolve().parent.parent
            default = root / DEFAULT_CLUSTER_MAP_RELATIVE
            if default.exists():
                load_cluster_map(str(default))
        except Exception as e:
            logger.debug("relation_clusterer: could not auto-load cluster map: %s", e)

    if _state is None:
        return cue_rel

    if fact in _state["fact_to_relation"]:
        return _state["fact_to_relation"][fact]

    model_name = str(_state.get("model_name", DEFAULT_MODEL_NAME))
    use_cues = bool(_state.get("embeds_relation_cues", False))
    embed_str = cue if use_cues else fact

    try:
        emb = embed_texts([embed_str], model_name=model_name)[0]
        d = np.linalg.norm(_state["centroids"] - emb, axis=1)
        cid = int(np.argmin(d))
        return _state["cluster_to_relation"][cid]
    except Exception as e:
        logger.warning("relation_clusterer: embedding assign failed (%s); cue fallback", e)
        return cue_rel


def build_label_cluster_relation_trace(
    paraphrase_sets: List[dict],
    fact_assignments: List[Dict[str, Any]],
    baseline_keys: Tuple[str, ...] = ("baseline_1", "baseline_2", "baseline_3"),
) -> List[Dict[str, Any]]:
    set_id_to_row = {row["set_id"]: row for row in fact_assignments}
    trace: List[Dict[str, Any]] = []
    for i, s in enumerate(paraphrase_sets or []):
        row = set_id_to_row.get(i) or {}
        cid = row.get("cluster_id")
        rel = row.get("relation")
        rp = row.get("relation_phrase")
        for bkey in baseline_keys:
            block = s.get(bkey) or {}
            for para_block in block.get("triples_per_paraphrase") or []:
                for t in para_block or []:
                    if not isinstance(t, (list, tuple)) or len(t) < 2:
                        continue
                    lab = str(t[1]).strip()
                    if not lab:
                        continue
                    trace.append(
                        {
                            "set_id": i,
                            "extractor_label": lab,
                            "relation_phrase": rp,
                            "cluster_id": cid,
                            "cluster_relation": rel,
                        }
                    )
    return trace
