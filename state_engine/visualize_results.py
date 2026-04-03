#!/usr/bin/env python3
"""
Build an HTML report (tables) and a PNG graph for one paraphrase set from state_engine_results.json.

Requires matplotlib + networkx (see baselines/requirements.txt).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from state_engine.canonicalize import canonical_entity
from state_engine.relations import parse_fact_spans
from state_engine.semantic_canonicalizer import normalize_surface, semantic_equal


def _is_viz_noise_entity(text: str) -> bool:
    """Drop obvious non-outcome / parser junk from the PNG (not used for metrics)."""
    ns = normalize_surface(text or "")
    if not ns or len(ns) < 2:
        return True
    if ns in frozenset(
        {
            "a drug",
            "the drug",
            "drug",
            "drugs",
            "medication",
            "medications",
            "patients",
            "patient",
            "people",
            "some patients",
        }
    ):
        return True
    if ns.startswith("patients ") or ns.startswith("patient "):
        return True
    return False


def _label_clusters(labels: set[str]) -> dict[str, str]:
    """
    Map each surface label -> one display representative by transitive semantic_equal
    (embedding + normalized string match).
    """
    labs = sorted(set(labels))
    if not labs:
        return {}
    parent = {x: x for x in labs}

    def find(a: str) -> str:
        while parent[a] != a:
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    for i, a in enumerate(labs):
        for b in labs[i + 1 :]:
            if semantic_equal(a, b):
                union(a, b)

    groups: dict[str, list[str]] = {}
    for x in labs:
        r = find(x)
        groups.setdefault(r, []).append(x)

    rep: dict[str, str] = {}
    for members in groups.values():
        chosen = max(members, key=lambda m: (len(m), m))
        for m in members:
            rep[m] = chosen
    return rep


def _display_representatives(
    labels: set[str],
    *,
    seed_fact: str,
) -> dict[str, str]:
    """
    Per-label display string: merge synonyms, then prefer seed-fact wording when
    any cluster member aligns with the fact subject/object.
    """
    if not labels:
        return {}
    base = _label_clusters(labels)
    parsed = parse_fact_spans(seed_fact or "")
    fs = (parsed[0] or "").strip() if parsed else ""
    fo = (parsed[2] or "").strip() if parsed else ""

    out = dict(base)
    for root in set(base.values()):
        members = [m for m in base if base[m] == root]
        if fs and any(semantic_equal(m, fs) for m in members):
            out.update({m: fs for m in members})
        elif fo and any(semantic_equal(m, fo) for m in members):
            out.update({m: fo for m in members})
    return out


def _status_bucket(status: str) -> str:
    s = (status or "").strip().lower()
    if s == "uncertain":
        return "uncertain"
    if s == "weak":
        return "weak"
    return "active"


def _status_color(status: str) -> str:
    s = _status_bucket(status)
    if s == "uncertain":
        return "#d62728"  # red
    if s == "weak":
        return "#ff7f0e"  # orange
    return "#555555"  # active/default


def _aggregate_pair_status(statuses: list[str]) -> str:
    """Pick one display status for a merged pair (most severe wins)."""
    buckets = {_status_bucket(s) for s in statuses}
    if "uncertain" in buckets:
        return "uncertain"
    if "weak" in buckets:
        return "weak"
    return "active"


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _table(headers: list, rows: list) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def build_html(data: dict) -> str:
    cfg = data.get("config") or {}
    summ = data.get("summary") or {}
    overall = data.get("overall") or {}
    slices = data.get("slices") or {}
    sets = data.get("sets") or []

    tau = cfg.get("drift_tau", overall.get("drift_tau", 0.2))

    summary_headers = [
        "GED mean",
        "Jaccard mean",
        f"Drift@{tau}",
        "Unsupported (per-note)",
        "Unsupported (union)",
        "Conflict records",
        "QA exact match",
        "QA Recall@1",
    ]
    summary_row = [
        _fmt(summ.get("GED_mean")),
        _fmt(summ.get("Jaccard_mean")),
        _fmt(summ.get(f"Drift@{tau}")),
        _fmt(summ.get("Unsupported_note_mean")),
        _fmt(summ.get("Unsupported_union_mean")),
        str(summ.get("conflict_candidates_records", "—")),
        _fmt(summ.get("QA_EM")),
        _fmt(summ.get("QA_Recall@1")),
    ]

    drift = slices.get("top_10_high_drift_sets") or []
    drift_rows = [
        [
            d.get("set_id", ""),
            (d.get("fact") or "")[:70] + ("…" if len(d.get("fact") or "") > 70 else ""),
            _fmt(d.get("jaccard_mean")),
            _fmt(d.get("ged_mean")),
            _fmt(d.get("drift_at_tau")),
        ]
        for d in drift
    ]

    per_set_rows = []
    for s in sets[:25]:
        pw = s.get("pairwise") or {}
        inc = s.get("incremental") or {}
        jm = (pw.get("jaccard") or {}).get("stats", {}).get("mean")
        gm = (pw.get("ged") or {}).get("stats", {}).get("mean")
        per_set_rows.append(
            [
                s.get("set_id", ""),
                (s.get("fact") or "")[:55] + ("…" if len(s.get("fact") or "") > 55 else ""),
                _fmt(jm),
                _fmt(gm),
                _fmt(pw.get("drift_at_tau")),
                _fmt(inc.get("unsupported_vs_per_note_evidence", {}).get("mean")),
            ]
        )

    misses = slices.get("top_10_qa_misses_missing_edges") or []
    miss_rows = [
        [
            (m.get("question") or "")[:45] + ("…" if len(m.get("question") or "") > 45 else ""),
            m.get("gold", ""),
            str(m.get("missing_canonical_edge", "")),
        ]
        for m in misses
    ]

    css = """
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; max-width: 1200px; }
    h1 { font-size: 1.35rem; }
    h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
    th, td { border: 1px solid #ddd; padding: 0.45rem 0.6rem; text-align: left; }
    th { background: #f4f4f4; }
    tr:nth-child(even) { background: #fafafa; }
    .meta { color: #555; font-size: 0.85rem; margin-bottom: 1.5rem; }
    """

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>State engine results</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Evidence-bound ontology state engine</h1>",
        "<div class='meta'>",
        f"<div><strong>Paraphrase results:</strong> {html.escape(str(cfg.get('paraphrase_results','')))}</div>",
        f"<div><strong>Baseline key:</strong> {html.escape(str(cfg.get('source_baseline','')))}</div>",
        f"<div><strong>QA file:</strong> {html.escape(str(cfg.get('qa_file','')))}</div>",
        "</div>",
        "<h2>Summary</h2>",
        _table(summary_headers, [summary_row]),
        "<h2>Top 10 high-drift sets (lowest Jaccard)</h2>",
        _table(["set_id", "fact", "Jaccard", "GED", f"Drift@{tau}"], drift_rows),
        "<h2>Per-set metrics (first 25 sets)</h2>",
        _table(
            ["set_id", "fact", "Jaccard", "GED", f"Drift@{tau}", "Unsupported/note"],
            per_set_rows,
        ),
    ]
    if miss_rows:
        parts.append("<h2>Sample QA misses (missing canonical edge)</h2>")
        parts.append(_table(["question", "gold", "missing edge"], miss_rows))
    parts.append(
        "<p><em>Graph: run <code>python3 state_engine/visualize_results.py --graph_png results/graph.png --set_id 0</code> "
        "(install <code>networkx</code> for nicer layout, optional).</em></p>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def _circular_positions(nodes: list):
    import math

    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}
    pos = {}
    for i, node in enumerate(nodes):
        ang = 2 * math.pi * i / n - math.pi / 2
        pos[node] = (math.cos(ang), math.sin(ang))
    return pos


def write_graph_png(data: dict, set_id: int, out_png: Path) -> None:
    import os

    _mpl_dir = _root / "results" / ".matplotlib"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Need matplotlib. Install: pip install matplotlib\n" + str(e)) from e

    try:
        import networkx as nx
    except ImportError:
        nx = None

    sets = data.get("sets") or []
    target = None
    for s in sets:
        if s.get("set_id") == set_id:
            target = s
            break
    if not target:
        raise SystemExit(f"No set with set_id={set_id}")

    edges_payload = (target.get("incremental") or {}).get("final_state") or {}
    raw_edges = edges_payload.get("edges") or []

    fact = target.get("fact", "")
    # Light cleanup then drop parser junk so the PNG matches how people read the fact.
    surf_subj, surf_obj = [], []
    for e in raw_edges:
        s, o = e.get("subject", ""), e.get("object", "")
        if not s or not o:
            continue
        s0, o0 = canonical_entity(str(s)), canonical_entity(str(o))
        if _is_viz_noise_entity(s0) or _is_viz_noise_entity(o0):
            continue
        surf_subj.append(s0)
        surf_obj.append(o0)
    rep = _display_representatives(set(surf_subj) | set(surf_obj), seed_fact=fact)

    nodes_set = set()
    pair_relations: dict = defaultdict(set)
    pair_statuses: dict = defaultdict(list)
    status_counts = {"active": 0, "weak": 0, "uncertain": 0}
    for e in raw_edges:
        s, r, o = e.get("subject", ""), e.get("relation", ""), e.get("object", "")
        if not s or not o:
            continue
        status = _status_bucket(str(e.get("status", "active")))
        s_c = canonical_entity(str(s))
        o_c = canonical_entity(str(o))
        if _is_viz_noise_entity(s_c) or _is_viz_noise_entity(o_c):
            continue
        s_c = rep.get(s_c, s_c)
        o_c = rep.get(o_c, o_c)
        nodes_set.add(s_c)
        nodes_set.add(o_c)
        pair = (s_c, o_c)
        pair_relations[pair].add(str(r))
        pair_statuses[pair].append(status)
        status_counts[status] += 1

    directed = sorted(pair_relations.keys())
    edge_label: dict = {}
    edge_status: dict = {}
    for pair in directed:
        rels = sorted(pair_relations[pair])
        st = _aggregate_pair_status(pair_statuses[pair])
        edge_status[pair] = st
        edge_label[pair] = f"{' | '.join(rels)} [{st}]"

    nodes = sorted(nodes_set) if nodes_set else ["(empty graph)"]

    if nx is not None:
        G = nx.DiGraph()
        for s, o in directed:
            G.add_edge(s, o)
        for n in nodes:
            G.add_node(n)
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(G, pos, node_color="#6baed6", node_size=2200, alpha=0.95)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="normal")
        edge_colors = [_status_color(edge_status.get((s, o), "active")) for s, o in directed]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=directed,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.08",
            width=1.2,
        )
        if edge_label:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_label,
                font_size=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999"),
            )
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], color=_status_color("active"), lw=2, label="active"),
            Line2D([0], [0], color=_status_color("weak"), lw=2, label="weak"),
            Line2D([0], [0], color=_status_color("uncertain"), lw=2, label="uncertain"),
        ]
        plt.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
    else:
        pos = _circular_positions(nodes)
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        for node, (x, y) in pos.items():
            ax.scatter([x], [y], s=2200, c="#6baed6", zorder=2, alpha=0.95)
            ax.text(x, y, node, ha="center", va="center", fontsize=8, wrap=True)

        def _arrow(x1, y1, x2, y2, rad=0.15, color="#444"):
            dx, dy = x2 - x1, y2 - y1
            dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ux, uy = dx / dist, dy / dist
            shrink = 0.18
            sx1, sy1 = x1 + ux * shrink, y1 + uy * shrink
            sx2, sy2 = x2 - ux * shrink, y2 - uy * shrink
            ax.annotate(
                "",
                xy=(sx2, sy2),
                xytext=(sx1, sy1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.2,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=1,
            )

        pair_n = defaultdict(int)
        for s, o in directed:
            x1, y1 = pos[s]
            x2, y2 = pos[o]
            pair_n[(s, o)] += 1
            rad = 0.08 * pair_n[(s, o)]
            st = edge_status.get((s, o), "active")
            _arrow(x1, y1, x2, y2, rad=rad, color=_status_color(st))
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mx,
                my + 0.06,
                edge_label.get((s, o), ""),
                fontsize=7,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#999"),
            )

        ax.set_aspect("equal")
        ax.axis("off")

    plt.title(
        (
            f"Set {set_id}: merged view (synonyms combined; junk objects omitted)\n"
            f"{(target.get('fact') or '')[:80]}\n"
            f"edge status counts: active={status_counts['active']}, "
            f"weak={status_counts['weak']}, uncertain={status_counts['uncertain']}"
        ),
        fontsize=11,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote graph: {out_png}")


def main():
    p = argparse.ArgumentParser(description="HTML tables + graph PNG for state engine results.")
    p.add_argument(
        "--json",
        type=str,
        default="results/state_engine_results.json",
        help="Path to state_engine_results.json",
    )
    p.add_argument(
        "--html_out",
        type=str,
        default="results/state_engine_report.html",
        help="Output HTML path",
    )
    p.add_argument(
        "--graph_png",
        type=str,
        default=None,
        help="If set, write a directed graph PNG for --set_id",
    )
    p.add_argument(
        "--set_id",
        type=int,
        default=0,
        help="Paraphrase set id for graph visualization",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    jpath = Path(args.json)
    if not jpath.is_absolute():
        jpath = root / jpath
    data = json.loads(jpath.read_text(encoding="utf-8"))

    hpath = Path(args.html_out)
    if not hpath.is_absolute():
        hpath = root / hpath
    hpath.parent.mkdir(parents=True, exist_ok=True)
    hpath.write_text(build_html(data), encoding="utf-8")
    print(f"Wrote HTML report: {hpath}")

    if args.graph_png:
        gpath = Path(args.graph_png)
        if not gpath.is_absolute():
            gpath = root / gpath
        write_graph_png(data, args.set_id, gpath)


if __name__ == "__main__":
    main()
