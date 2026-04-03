#!/usr/bin/env python3
"""
Generate dataset visualizations for the report: BC5CDR stats, paraphrase set structure, QA examples.
Saves PNGs to results/plots/ (or --output_dir). Run from baselines/: python plot_dataset_viz.py
"""
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def plot_bc5cdr_stats(train_data, dev_data, test_data, save_dir: Path) -> None:
    """Entity type distribution, relation type distribution, and split sizes for BC5CDR."""
    if not _HAS_MPL or (not train_data and not dev_data and not test_data):
        return
    all_ex = train_data + dev_data + test_data
    entity_counts = {}
    relation_counts = {}
    for ex in all_ex:
        for e in ex.get("entities", []):
            t = e.get("type", "Other")
            entity_counts[t] = entity_counts.get(t, 0) + 1
        for r in ex.get("relations", []):
            t = r.get("type", "CID")
            relation_counts[t] = relation_counts.get(t, 0) + 1

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Split sizes
    splits = ["Train", "Dev", "Test"]
    counts = [len(train_data), len(dev_data), len(test_data)]
    axes[0].bar(splits, counts, color=["#2ecc71", "#3498db", "#9b59b6"], edgecolor="white")
    axes[0].set_ylabel("Number of documents")
    axes[0].set_title("BC5CDR train/dev/test split")
    for i, c in enumerate(counts):
        axes[0].text(i, c + max(counts) * 0.02, str(c), ha="center", fontsize=11)

    # Entity types
    if entity_counts:
        labels = list(entity_counts.keys())
        vals = [entity_counts[k] for k in labels]
        axes[1].bar(labels, vals, color=["#e74c3c", "#3498db"], edgecolor="white")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Entity type distribution")
    else:
        axes[1].set_title("Entity type distribution (no data)")
        axes[1].set_xticks([])

    # Relation types
    if relation_counts:
        labels = list(relation_counts.keys())
        vals = [relation_counts[k] for k in labels]
        axes[2].bar(labels, vals, color=["#1abc9c", "#95a5a6"], edgecolor="white")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Relation type distribution")
    else:
        axes[2].set_title("Relation type distribution (no data)")
        axes[2].set_xticks([])

    plt.tight_layout()
    out = save_dir / "dataset_bc5cdr_stats.png"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_paraphrase_structure(paraphrase_path: Path, save_dir: Path) -> None:
    """Bar chart: number of paraphrases per fact (paraphrase_sets_5.json)."""
    if not _HAS_MPL or not paraphrase_path.exists():
        return
    data = json.loads(paraphrase_path.read_text())
    sets = data.get("paraphrase_sets", [])
    if not sets:
        paraphrases = data.get("paraphrases", [])
        if paraphrases:
            sets = [{"fact": data.get("fact", "Fact"), "paraphrases": paraphrases}]
    if not sets:
        return
    facts = [s.get("fact", "Fact")[:30] + ("..." if len(s.get("fact", "")) > 30 else "") for s in sets]
    counts = [len(s.get("paraphrases", [])) for s in sets]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(facts))
    bars = ax.bar(x, counts, color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(facts, rotation=25, ha="right")
    ax.set_ylabel("Number of paraphrases")
    ax.set_title("Controlled paraphrase dataset: paraphrases per fact")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.2, str(c), ha="center", fontsize=10)
    plt.tight_layout()
    out = save_dir / "dataset_paraphrase_structure.png"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_paraphrase_results_and_dataset_role(
    paraphrase_results_path: Path,
    save_dir: Path,
    *,
    n_facts_display: int = 5,
    dataset_role_rows: list = None,
) -> None:
    """
    One figure: (1) Top N paraphrase results — GED and Jaccard per fact, grouped by baseline;
    (2) Dataset Role Size table (Dataset | Role | Size).
    dataset_role_rows: list of [dataset_name, role, size] for the table. If None, use defaults.
    """
    if not _HAS_MPL:
        return
    default_role_rows = [
        ["BC5CDR Train/Dev/Test", "Train / Dev / Test (NER + relation)", "~2500 abstracts"],
        ["Paraphrase Set", "Evaluation", "30 facts × 10 paraphrases"],
        ["QA Dataset", "Evaluation", "50 questions"],
    ]
    rows = dataset_role_rows if dataset_role_rows is not None else default_role_rows

    facts_short = []
    ged_b1, ged_b2, ged_b3 = [], [], []
    jaccard_b1, jaccard_b2, jaccard_b3 = [], [], []

    if paraphrase_results_path.exists():
        data = json.loads(paraphrase_results_path.read_text())
        sets = data.get("sets", [])[:n_facts_display]
        for s in sets:
            f = s.get("fact", "Fact")
            facts_short.append(f[:22] + ("…" if len(f) > 22 else ""))
            for bl in ("baseline_1", "baseline_2", "baseline_3"):
                st = (s.get(bl) or {}).get("stats") or {}
                ged = (st.get("ged_pairwise") or {}).get("mean", 0)
                jacc = (st.get("jaccard_pairwise") or {}).get("mean", 0)
                if bl == "baseline_1":
                    ged_b1.append(ged)
                    jaccard_b1.append(jacc)
                elif bl == "baseline_2":
                    ged_b2.append(ged)
                    jaccard_b2.append(jacc)
                else:
                    ged_b3.append(ged)
                    jaccard_b3.append(jacc)
        n = len(facts_short)
    else:
        n = 0

    fig = plt.figure(figsize=(12, 8))
    if n > 0:
        x = np.arange(n)
        w = 0.25
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.bar(x - w, ged_b1, w, label="Baseline 1", color="#3498db")
        ax1.bar(x, ged_b2, w, label="Baseline 2", color="#2ecc71")
        ax1.bar(x + w, ged_b3, w, label="Baseline 3", color="#9b59b6")
        ax1.set_ylabel("GED (↓ lower = more stable)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(facts_short, rotation=15, ha="right")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title("Mean GED by fact (top 5 paraphrase sets)")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.bar(x - w, jaccard_b1, w, label="Baseline 1", color="#3498db")
        ax2.bar(x, jaccard_b2, w, label="Baseline 2", color="#2ecc71")
        ax2.bar(x + w, jaccard_b3, w, label="Baseline 3", color="#9b59b6")
        ax2.set_ylabel("Jaccard (↑ higher = more overlap)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(facts_short, rotation=15, ha="right")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Mean Jaccard by fact (top 5 paraphrase sets)")
    else:
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("Mean GED by fact")
        ax1.text(0.5, 0.5, "No paraphrase_results.json found", ha="center", va="center", transform=ax1.transAxes)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("Mean Jaccard by fact")

    ax_t = fig.add_subplot(2, 1, 2)
    ax_t.axis("off")
    table = ax_t.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
        colLabels=["Dataset", "Role", "Size"],
        colWidths=[0.32, 0.42, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.4)
    ax_t.set_title("Dataset role and size", fontsize=12)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "dataset_paraphrase_results_and_role_size.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_first_five_paraphrases_metrics(paraphrase_results_path: Path, save_dir: Path) -> None:
    """One image: GED and Jaccard for the first 5 paraphrases (mean vs others), all baselines."""
    if not _HAS_MPL or not paraphrase_results_path.exists():
        return
    data = json.loads(paraphrase_results_path.read_text())
    sets = data.get("sets", [])
    if not sets:
        return
    first = sets[0]
    n_paraphrases = 5
    baselines = [k for k in first if k.startswith("baseline_")]
    if not baselines:
        return
    ged_per_bl = {}
    jaccard_per_bl = {}
    for bl in baselines:
        bl_data = first.get(bl) or {}
        ged_mat = bl_data.get("pairwise_ged_matrix")
        jacc_mat = bl_data.get("pairwise_jaccard_matrix")
        if ged_mat is None or jacc_mat is None:
            continue
        G = np.array(ged_mat)
        J = np.array(jacc_mat)
        n = min(n_paraphrases, G.shape[0])
        # For each paraphrase i, mean vs all others (exclude diagonal)
        ged_means = []
        jacc_means = []
        for i in range(n):
            row_ged = [G[i, j] for j in range(G.shape[1]) if j != i]
            row_jacc = [J[i, j] for j in range(J.shape[1]) if j != i]
            ged_means.append(np.mean(row_ged) if row_ged else 0)
            jacc_means.append(np.mean(row_jacc) if row_jacc else 0)
        ged_per_bl[bl] = ged_means
        jaccard_per_bl[bl] = jacc_means
    if not ged_per_bl:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(n_paraphrases)
    w = 0.25
    colors = ["#3498db", "#2ecc71", "#9b59b6"]
    for i, bl in enumerate(sorted(ged_per_bl.keys())):
        offset = (i - 1) * w
        ax1.bar(x + offset, ged_per_bl[bl], w, label=bl.replace("_", " ").title(), color=colors[i % 3])
        ax2.bar(x + offset, jaccard_per_bl[bl], w, label=bl.replace("_", " ").title(), color=colors[i % 3])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"P{i+1}" for i in range(n_paraphrases)])
    ax1.set_ylabel("Mean GED (vs other paraphrases)")
    ax1.set_title("GED — first 5 paraphrases (↓ lower = more stable)")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{i+1}" for i in range(n_paraphrases)])
    ax2.set_ylabel("Mean Jaccard (vs other paraphrases)")
    ax2.set_title("Jaccard — first 5 paraphrases (↑ higher = more overlap)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=8)
    fact = first.get("fact", "Fact")[:50]
    plt.suptitle(f"First 5 paraphrases — metrics by baseline (fact: {fact}{'…' if len(first.get('fact','')) > 50 else ''})", fontsize=11, y=1.02)
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "first_five_paraphrases_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_qa_and_summary(qa_path: Path, save_dir: Path) -> None:
    """Summary panel: QA dataset size and dataset overview table as text in figure."""
    if not _HAS_MPL:
        return
    n_qa = 0
    if qa_path.exists():
        data = json.loads(qa_path.read_text())
        examples = data.get("examples", data.get("qa", []))
        n_qa = len(examples)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    rows = [
        ["Dataset", "Role", "Count"],
        ["BC5CDR (PubTator)", "Train / Dev / Test (NER + relation)", "See dataset_bc5cdr_stats.png"],
        ["Paraphrase sets (5 facts)", "Evaluation only (paraphrase stability)", "50 paraphrases (10 per fact)"],
        ["QA (aspirin_qa.json)", "Downstream QA over graph", f"{n_qa} questions"],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.35, 0.45, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)
    ax.set_title("Dataset overview", fontsize=12)
    plt.tight_layout()
    out = save_dir / "dataset_overview.png"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate dataset visualizations for the report.")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory (default: results/plots)")
    p.add_argument("--bc5cdr_dir", type=str, default=None, help="BC5CDR data dir (default: data/bc5cdr)")
    p.add_argument("--paraphrase_file", type=str, default=None, help="Paraphrase JSON (default: data/paraphrases/paraphrase_sets_5.json)")
    p.add_argument("--paraphrase_results", type=str, default=None, help="paraphrase_results.json path (for GED/Jaccard plot; default: ../results/ or results/)")
    p.add_argument("--qa_file", type=str, default=None, help="QA JSON (default: data/qa/aspirin_qa.json)")
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _here / "results" / "plots"
    bc5cdr_dir = args.bc5cdr_dir or str(_here / "data" / "bc5cdr")
    paraphrase_path = Path(args.paraphrase_file or _here / "data" / "paraphrases" / "paraphrase_sets_5.json")
    qa_path = Path(args.qa_file or _here / "data" / "qa" / "aspirin_qa.json")
    paraphrase_results_path = Path(args.paraphrase_results) if args.paraphrase_results else (_here.parent / "results" / "paraphrase_results.json")
    if not paraphrase_results_path.exists():
        paraphrase_results_path = _here / "results" / "paraphrase_results.json"

    if not _HAS_MPL:
        print("matplotlib not found; install with: pip install matplotlib")
        return

    # BC5CDR (optional)
    try:
        from loaders.bc5cdr_loader import load_bc5cdr
        train_data, dev_data, test_data = load_bc5cdr(bc5cdr_dir, download_if_missing=False)
        plot_bc5cdr_stats(train_data, dev_data, test_data, out_dir)
    except Exception as e:
        print(f"BC5CDR not loaded ({e}); skipping BC5CDR stats plot. Generate dataset_paraphrase_structure and dataset_overview only.")

    # Paraphrase structure
    plot_paraphrase_structure(paraphrase_path, out_dir)

    # Overview table (includes QA count)
    plot_qa_and_summary(qa_path, out_dir)

    # One figure: top 5 paraphrase GED/Jaccard + Dataset Role Size table
    plot_paraphrase_results_and_dataset_role(paraphrase_results_path, out_dir)

    # One figure: first 5 paraphrases metrics (GED and Jaccard by baseline)
    plot_first_five_paraphrases_metrics(paraphrase_results_path, out_dir)

    print("Dataset visualizations done.")


if __name__ == "__main__":
    main()
