"""
Plotting helpers for graph evaluation: paraphrase metrics and knowledge-graph visualizations.
"""
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False


def _triples_to_graph(triples: List) -> "nx.DiGraph":
    """Build a directed graph from list of (s, r, o) or (s, r, o, _) triples."""
    if not _PLOTTING_AVAILABLE:
        return None
    G = nx.DiGraph()
    for t in triples:
        if len(t) >= 3:
            s, r, o = str(t[0]), str(t[1]), str(t[2])
            if s and o:
                G.add_edge(s, o, label=r)
    return G


def plot_paraphrase_metrics_distribution(
    ged_values: np.ndarray,
    jaccard_values: np.ndarray,
    save_path: Path,
) -> None:
    """Bar/distribution of pairwise GED and Jaccard across paraphrase pairs."""
    if not _PLOTTING_AVAILABLE or (len(ged_values) == 0 and len(jaccard_values) == 0):
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if len(ged_values) > 0:
        ax1.hist(ged_values, bins=min(20, max(2, len(ged_values))), color="steelblue", edgecolor="white")
        ax1.axvline(np.mean(ged_values), color="red", linestyle="--", label=f"Mean = {np.mean(ged_values):.3f}")
        ax1.set_xlabel("Graph Edit Distance")
        ax1.set_ylabel("Count")
        ax1.set_title("GED across paraphrase pairs (↓ lower is better)")
        ax1.legend()
    if len(jaccard_values) > 0:
        ax2.hist(jaccard_values, bins=min(20, max(2, len(jaccard_values))), color="seagreen", edgecolor="white", alpha=0.8)
        ax2.axvline(np.mean(jaccard_values), color="red", linestyle="--", label=f"Mean = {np.mean(jaccard_values):.3f}")
        ax2.set_xlabel("Evidence Jaccard")
        ax2.set_ylabel("Count")
        ax2.set_title("Jaccard across paraphrase pairs (↑ higher is better)")
        ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_paraphrase_heatmaps(
    ged_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
    labels: List[str],
    save_path: Path,
) -> None:
    """Heatmaps of pairwise GED and Jaccard between paraphrases (n×n)."""
    if not _PLOTTING_AVAILABLE or ged_matrix.size == 0:
        return
    n = ged_matrix.shape[0]
    short_labels = [f"P{i+1}" for i in range(n)] if not labels else [str(l)[:12] for l in labels]
    if len(short_labels) > 15:
        short_labels = [f"P{i+1}" for i in range(n)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(ged_matrix, cmap="Reds", aspect="auto", vmin=0)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(short_labels, rotation=45, ha="right")
    ax1.set_yticklabels(short_labels)
    ax1.set_xlabel("Paraphrase")
    ax1.set_ylabel("Paraphrase")
    ax1.set_title("Pairwise GED (↓ lower = more similar)")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(jaccard_matrix, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right")
    ax2.set_yticklabels(short_labels)
    ax2.set_xlabel("Paraphrase")
    ax2.set_ylabel("Paraphrase")
    ax2.set_title("Pairwise Jaccard (↑ higher = more similar)")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_paraphrase_knowledge_graphs(
    paraphrases: List[str],
    graphs_triples: List[List[Tuple[str, str, str]]],
    fact: str,
    save_path: Path,
    max_plots: int = 8,
) -> None:
    """Draw one knowledge-graph diagram per paraphrase (nodes = entities, edges = relations)."""
    if not _PLOTTING_AVAILABLE or not graphs_triples:
        return
    n = min(len(paraphrases), len(graphs_triples), max_plots)
    if n == 0:
        return
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for idx in range(n):
        ax = axes[idx]
        triples = graphs_triples[idx]
        G = _triples_to_graph(triples)
        if G is not None and G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=1.5, seed=42)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=800)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=20)
            edge_labels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        else:
            ax.text(0.5, 0.5, "No triples extracted", ha="center", va="center", transform=ax.transAxes)
        title = (paraphrases[idx][:50] + "…") if len(paraphrases[idx]) > 50 else paraphrases[idx]
        ax.set_title(f"Paraphrase {idx + 1}: {title}", fontsize=9)
        ax.axis("off")
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    plt.suptitle(f"Extracted knowledge graphs (fact: {fact[:60]}{'…' if len(fact) > 60 else ''})", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_first_five_paraphrases_all_baselines(
    paraphrases: List[str],
    baseline_to_graphs: dict,
    fact: str,
    save_path: Path,
    n_paraphrases: int = 5,
) -> None:
    """One image: first N paraphrases (columns) × all baselines (rows). Each cell = knowledge graph for that paraphrase."""
    if not _PLOTTING_AVAILABLE or not baseline_to_graphs:
        return
    baselines = sorted(k for k in baseline_to_graphs if k.startswith("baseline_"))
    n_baselines = len(baselines)
    n = min(n_paraphrases, len(paraphrases), max(len(baseline_to_graphs[b]) for b in baselines) if baselines else 0)
    if n == 0 or n_baselines == 0:
        return
    fig, axes = plt.subplots(n_baselines, n, figsize=(4 * n, 4 * n_baselines))
    if n_baselines == 1:
        axes = axes.reshape(1, -1)
    if n == 1:
        axes = axes.reshape(-1, 1)
    for row, bl_key in enumerate(baselines):
        graphs = baseline_to_graphs[bl_key]
        for col in range(n):
            ax = axes[row, col]
            triples = graphs[col] if col < len(graphs) else []
            G = _triples_to_graph(triples)
            if G is not None and G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=1.2, seed=42)
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=400)
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=14)
                edge_labels = nx.get_edge_attributes(G, "label")
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7)
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            else:
                ax.text(0.5, 0.5, "No triples", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            short = (paraphrases[col][:35] + "…") if len(paraphrases[col]) > 35 else paraphrases[col]
            ax.set_title(f"P{col + 1}: {short}", fontsize=8)
            ax.axis("off")
        axes[row, 0].set_ylabel(bl_key.replace("_", " ").title(), fontsize=10)
    plt.suptitle(f"First {n} paraphrases — all baselines (fact: {fact[:50]}{'…' if len(fact) > 50 else ''})", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_incremental_metrics_summary(stats: dict, save_path: Path) -> None:
    """Bar chart of mean metrics for incremental evaluation (GED, Jaccard, unsupported, conflict)."""
    if not _PLOTTING_AVAILABLE or not stats:
        return
    keys = ["ged_consecutive", "jaccard_consecutive", "unsupported_rate", "conflict_precision"]
    labels = ["GED ↓", "Jaccard ↑", "Unsupported ↓", "Heuristic conflict"]
    means = [stats.get(k, {}).get("mean", 0) for k in keys]
    ci_lo = [stats.get(k, {}).get("ci_lower", m) for k, m in zip(keys, means)]
    ci_hi = [stats.get(k, {}).get("ci_upper", m) for k, m in zip(keys, means)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, means, color=["#e74c3c", "#2ecc71", "#e67e22", "#3498db"], edgecolor="white")
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(ci_lo), np.array(ci_hi) - np.array(means)],
                fmt="none", color="black", capsize=4)
    ax.set_xticks(x)
    ax.set_ylabel("Metric value")
    ax.set_title("Incremental graph evaluation (95% CI)")
    ax.set_xticklabels(labels)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_baselines_comparison(
    baseline_stats: dict,
    save_path: Path,
    metric_keys: list = None,
    metric_labels: list = None,
) -> None:
    """Side-by-side bar chart comparing multiple baselines (e.g. baseline_1, baseline_2, baseline_3)."""
    if not _PLOTTING_AVAILABLE or not baseline_stats:
        return
    metric_keys = metric_keys or ["ged_consecutive", "jaccard_consecutive", "unsupported_rate", "conflict_precision"]
    metric_labels = metric_labels or ["GED ↓", "Jaccard ↑", "Unsupported ↓", "Heuristic conflict"]
    baselines = list(baseline_stats.keys())
    n_baselines = len(baselines)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.8 / n_baselines
    colors = ["#3498db", "#2ecc71", "#e67e22"][:n_baselines]
    fig, ax = plt.subplots(figsize=(4 + n_metrics * 1.5, 5))
    for i, bl in enumerate(baselines):
        stats = baseline_stats[bl]
        means = [stats.get(k, {}).get("mean", 0) for k in metric_keys]
        offset = (i - (n_baselines - 1) / 2) * width
        ax.bar(x + offset, means, width, label=bl.replace("_", " ").title(), color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Metric value")
    ax.set_title("Baselines comparison (incremental)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
