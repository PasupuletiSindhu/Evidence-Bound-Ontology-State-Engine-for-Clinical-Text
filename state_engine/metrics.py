# GED, Jaccard, unsupported inference rate (Unsupported triples), drift@tau metrics

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Triple = Tuple[str, str, str]


def graph_edit_distance(a: Iterable[Triple], b: Iterable[Triple]) -> float:
    sa, sb = set(a), set(b)
    return float(len(sa - sb) + len(sb - sa))


def jaccard_similarity(a: Iterable[Triple], b: Iterable[Triple]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 1.0


def unsupported_inference_rate(
    predicted: Iterable[Triple], supported: Iterable[Triple]
) -> float:
    sp = set(predicted)
    ss = set(supported)
    if not sp:
        return 0.0
    unsupported = sum(1 for t in sp if t not in ss)
    return float(unsupported) / float(len(sp))


def unsupported_vs_note_evidence(
    final_graph: Iterable[Triple],
    note_triple_sets: Sequence[set],
) -> Dict[str, float]:
    """
    Per-note support: fraction of final-graph triples not present in each note's
    canonical extraction alone (then averaged). More meaningful than union-vs-final.
    """
    final_s = set(final_graph)
    if not final_s:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    if not note_triple_sets:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    rates = []
    for w in note_triple_sets:
        if not w:
            rates.append(1.0)
            continue
        u = sum(1 for t in final_s if t not in w) / max(len(final_s), 1)
        rates.append(float(u))
    arr = np.asarray(rates, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def drift_rate_at_tau(
    jaccard_pairwise: np.ndarray, tau: float = 0.2
) -> float:
    """Fraction of paraphrase pairs with drift (1 - Jaccard) >= tau."""
    if jaccard_pairwise is None or len(jaccard_pairwise) == 0:
        return 0.0
    drift = 1.0 - jaccard_pairwise
    return float(np.mean(drift >= tau))


def _strip_weak_edges(graph: Iterable[Triple]) -> List[Triple]:
    out: List[Triple] = []
    for t in graph:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        r = str(t[1]).strip().lower().replace(" ", "_")
        if r == "related_to":
            continue
        out.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))
    return out


def pairwise_stability(
    graphs: Sequence[List[Triple]],
    *,
    exclude_weak_edges: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Pairwise GED / Jaccard over triple sets.

    When ``exclude_weak_edges`` is True, edges with relation ``related_to`` are
    omitted so weak fallback evidence does not inflate drift vs structure.
    """
    ged_vals = []
    jac_vals = []
    n = len(graphs)
    for i in range(n):
        for j in range(i + 1, n):
            gi = _strip_weak_edges(graphs[i]) if exclude_weak_edges else list(graphs[i])
            gj = _strip_weak_edges(graphs[j]) if exclude_weak_edges else list(graphs[j])
            ged_vals.append(graph_edit_distance(gi, gj))
            jac_vals.append(jaccard_similarity(gi, gj))
    return {
        "ged_pairwise": np.asarray(ged_vals) if ged_vals else np.asarray([0.0]),
        "jaccard_pairwise": np.asarray(jac_vals) if jac_vals else np.asarray([1.0]),
    }


def summarize(arr: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
    n = int(len(arr))
    mean = float(np.mean(arr)) if n else 0.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / (n ** 0.5) if n > 0 else 0.0
    # Normal approximation to avoid mandatory scipy dependency.
    z = 1.96 if confidence == 0.95 else 1.96
    margin = z * sem
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
    }
