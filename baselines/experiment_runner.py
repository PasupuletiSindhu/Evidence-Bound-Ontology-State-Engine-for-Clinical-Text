# Incremental sim (N orderings), paraphrase pairwise metrics, 95% CI.
import random
from typing import List, Callable, Any, Set
import numpy as np
from scipy import stats

from drift_metrics import (
    graph_edit_distance,
    jaccard_similarity,
    unsupported_inference_rate,
    conflict_detection_precision,
    detect_conflicts_heuristic,
)


def compute_paraphrase_metrics(
    paraphrases: List[str],
    process_fn: Callable[[str], List],
) -> dict:
    # one graph per paraphrase, then pairwise GED/Jaccard
    metrics, _, _, _ = compute_paraphrase_metrics_full(paraphrases, process_fn)
    return metrics


def compute_paraphrase_metrics_full(
    paraphrases: List[str],
    process_fn: Callable[[str], List],
):
    # returns metrics + graphs + n×n GED and Jaccard matrices
    graphs = [list(process_fn(p)) for p in paraphrases]
    ged_list = []
    jaccard_list = []
    n = len(graphs)
    ged_matrix = np.zeros((n, n))
    jaccard_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            g = graph_edit_distance(graphs[i], graphs[j])
            jacc = jaccard_similarity(graphs[i], graphs[j])
            ged_list.append(g)
            jaccard_list.append(jacc)
            ged_matrix[i, j] = ged_matrix[j, i] = g
            jaccard_matrix[i, j] = jaccard_matrix[j, i] = jacc
    # n>=2 and every graph empty: pairwise scores are vacuous; store empty arrays → NaN stats.
    all_empty = n >= 2 and all(len(g) == 0 for g in graphs)
    if all_empty:
        ged_list.clear()
        jaccard_list.clear()
    if ged_list:
        metrics = {
            "ged_pairwise": np.asarray(ged_list, dtype=float),
            "jaccard_pairwise": np.asarray(jaccard_list, dtype=float),
        }
    elif n >= 2 and all_empty:
        metrics = {
            "ged_pairwise": np.array([], dtype=float),
            "jaccard_pairwise": np.array([], dtype=float),
        }
    else:
        metrics = {
            "ged_pairwise": np.array([0.0]),
            "jaccard_pairwise": np.array([1.0]),
        }
    return metrics, graphs, ged_matrix, jaccard_matrix


def run_incremental_simulation(
    texts: List[str],
    process_fn: Callable[[str], List],
    N: int = 10,
    seed: int = 42,
) -> List[List]:
    # N random orderings; each trajectory is a list of triple lists (one per text)
    rng = random.Random(seed)
    trajectories = []
    for _ in range(N):
        order = list(range(len(texts)))
        rng.shuffle(order)
        trajectory = []
        for t_idx in order:
            triples = process_fn(texts[t_idx])
            trajectory.append(list(triples))
        trajectories.append(trajectory)
    return trajectories


def compute_metrics_over_trajectories(
    trajectories: List[List[List]],
    supported_by_text: List[Set[tuple]] = None,
    is_true_conflict_fn=None,
) -> dict:
    # GED/Jaccard between consecutive states, unsupported rate, conflict heuristic
    ged_list = []
    jaccard_list = []
    unsupported_list = []
    conflict_precision_list = []
    for traj in trajectories:
        for i in range(1, len(traj)):
            ged_list.append(graph_edit_distance(traj[i - 1], traj[i]))
            jaccard_list.append(jaccard_similarity(traj[i - 1], traj[i]))
        if traj:
            all_triples = []
            for t in traj:
                all_triples.extend(t)
            supported = set()
            for s in (supported_by_text or [set()]):
                supported |= s
            unsupported_list.append(unsupported_inference_rate(all_triples, supported))
            last_triples = traj[-1] if traj else []
            detected = detect_conflicts_heuristic(last_triples)
            if is_true_conflict_fn is None:
                is_true_conflict_fn = lambda t1, t2: True
            conflict_precision_list.append(
                conflict_detection_precision(detected, is_true_conflict_fn)
            )
        else:
            unsupported_list.append(0.0)
            conflict_precision_list.append(1.0)
    return {
        "ged_consecutive": np.array(ged_list) if ged_list else np.array([0.0]),
        "jaccard_consecutive": np.array(jaccard_list) if jaccard_list else np.array([1.0]),
        "unsupported_rate": np.array(unsupported_list) if unsupported_list else np.array([0.0]),
        "conflict_precision": np.array(conflict_precision_list) if conflict_precision_list else np.array([1.0]),
    }


def report_statistics(metrics: dict, confidence: float = 0.95) -> dict:
    # mean, std, 95% CI per key
    out = {}
    for k, arr in metrics.items():
        n = len(arr)
        if n == 0:
            nan = float("nan")
            out[k] = {"mean": nan, "std": nan, "ci_lower": nan, "ci_upper": nan}
            continue
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        sem = std / (n ** 0.5) if n > 0 else 0.0
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0
        margin = t_crit * sem
        out[k] = {
            "mean": mean,
            "std": std,
            "ci_lower": mean - margin,
            "ci_upper": mean + margin,
        }
    return out


def paired_t_test(metrics_a: dict, metrics_b: dict) -> dict:
    # paired t-test per key, returns p-values
    result = {}
    for k in metrics_a:
        if k not in metrics_b or len(metrics_a[k]) != len(metrics_b[k]):
            continue
        t_stat, p_val = stats.ttest_rel(metrics_a[k], metrics_b[k])
        result[k] = {"t": float(t_stat), "p": float(p_val)}
    return result
