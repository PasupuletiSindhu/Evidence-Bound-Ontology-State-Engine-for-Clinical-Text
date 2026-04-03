#!/usr/bin/env python3
"""Aggregate baseline metrics from paraphrase + QA JSON and merge with state_engine_results.

Outputs a single markdown + CSV table for papers/slides. Run from repo root:

  python -m state_engine.compare_to_baselines

Fair QA comparison: use the SAME qa_file when running baselines/run_qa_eval.py and
state_engine/run_state_engine.py (e.g. results/qa_aligned_100.json).

Note on graph metrics: baseline GED/Jaccard come from raw extractor triples (LABEL_* relations).
State-engine pairwise metrics are computed after prepare_triples_for_state (canonical relations,
orientation). Numbers are comparable in spirit (paraphrase stability) but not bitwise identical
pipelines.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _read(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _drift_at_tau(jaccard_vals, tau: float) -> float:
    if not jaccard_vals:
        return float("nan")
    a = np.asarray(jaccard_vals, dtype=float)
    drift = 1.0 - a
    return float(np.mean(drift >= tau))


def _baseline_total_triples(pr: dict, baseline_key: str) -> int:
    """Count all (s,r,o) rows for a baseline across every set and paraphrase."""
    total = 0
    for s in pr.get("sets") or []:
        block = s.get(baseline_key) or {}
        for row in block.get("triples_per_paraphrase") or []:
            if isinstance(row, list):
                total += len(row)
    return total


def _mask_degenerate_baselines(
    pr: dict, rows_by_key: dict[str, dict], drift_tau: float
) -> list[str]:
    """
    If a baseline has zero triples in paraphrase JSON, GED/Jaccard/drift are vacuous (0 / 1 / 0).
    Clear those metrics and QA so the table shows — instead of misleading perfect stability.
    """
    dk = f"Drift@{drift_tau}"
    notes: list[str] = []
    for b in ("baseline_1", "baseline_2", "baseline_3"):
        if b not in rows_by_key:
            continue
        if _baseline_total_triples(pr, b) > 0:
            continue
        row = rows_by_key[b]
        row["GED_mean"] = float("nan")
        row["Jaccard_mean"] = float("nan")
        row[dk] = float("nan")
        row["QA_EM"] = None
        row["QA_R@1"] = None
        notes.append(row.get("label", b))
    return notes


def baseline_rows_from_paraphrase(pr: dict, drift_tau: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    sets = pr.get("sets") or []
    for b in ("baseline_1", "baseline_2", "baseline_3"):
        ged_means, jac_means, drift_means = [], [], []
        for s in sets:
            block = s.get(b) or {}
            st = block.get("stats") or {}
            g = (st.get("ged_pairwise") or {}).get("mean")
            j = (st.get("jaccard_pairwise") or {}).get("mean")
            if g is not None:
                ged_means.append(g)
            if j is not None:
                jac_means.append(j)
            jp = block.get("jaccard_pairwise")
            if jp:
                drift_means.append(_drift_at_tau(jp, drift_tau))
        label = {"baseline_1": "Baseline 1 (stateless NER+RE)", "baseline_2": "Baseline 2 (+ ontology IDs)", "baseline_3": "Baseline 3 (Qwen)"}.get(b, b)
        out[b] = {
            "label": label,
            "GED_mean": float(np.mean(ged_means)) if ged_means else float("nan"),
            "Jaccard_mean": float(np.mean(jac_means)) if jac_means else float("nan"),
            f"Drift@{drift_tau}": float(np.mean(drift_means)) if drift_means else float("nan"),
            "QA_EM": None,
            "QA_R@1": None,
            "Unsupported_note": None,
            "Conflict_records": None,
        }
    return out


def apply_qa(metrics_by_key: dict[str, dict], qa: dict | None) -> None:
    if not qa:
        return
    for b in ("baseline_1", "baseline_2", "baseline_3"):
        if b not in metrics_by_key:
            continue
        block = qa.get(b)
        if not isinstance(block, dict):
            continue
        m = block.get("metrics") or {}
        if m:
            metrics_by_key[b]["QA_EM"] = m.get("exact_match")
            metrics_by_key[b]["QA_R@1"] = m.get("recall_at_1")


def state_engine_row(se: dict, drift_tau: float) -> dict:
    sr = se.get("summary") or {}
    dk = f"Drift@{drift_tau}"
    return {
        "label": "Evidence-bound state engine (aligned + incremental)",
        "GED_mean": sr.get("GED_mean"),
        "Jaccard_mean": sr.get("Jaccard_mean"),
        dk: sr.get(dk),
        "QA_EM": sr.get("QA_EM"),
        "QA_R@1": sr.get("QA_Recall@1"),
        "Unsupported_note": sr.get("Unsupported_note_mean"),
        "Conflict_records": sr.get("conflict_candidates_records"),
    }


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and (np.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main():
    p = argparse.ArgumentParser(description="Baseline vs state engine comparison table.")
    p.add_argument("--paraphrase_results", type=str, default="results/paraphrase_results.json")
    p.add_argument("--qa_results", type=str, default="results/qa_results.json", help="From baselines/run_qa_eval.py; use same QA file as state engine for fair EM/R@1.")
    p.add_argument("--state_engine_results", type=str, default="results/state_engine_results.json")
    p.add_argument("--drift_tau", type=float, default=0.2)
    p.add_argument("--out_md", type=str, default="results/method_comparison.md")
    p.add_argument("--out_csv", type=str, default="results/method_comparison.csv")
    args = p.parse_args()

    pr_path = Path(args.paraphrase_results)
    qa_path = Path(args.qa_results)
    se_path = Path(args.state_engine_results)
    if not pr_path.is_absolute():
        pr_path = _root / pr_path
    if not qa_path.is_absolute():
        qa_path = _root / qa_path
    if not se_path.is_absolute():
        se_path = _root / se_path

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    if not out_md.is_absolute():
        out_md = _root / out_md
    if not out_csv.is_absolute():
        out_csv = _root / out_csv
    out_md = out_md.resolve()
    out_csv = out_csv.resolve()

    pr = _read(pr_path)
    qa = _read(qa_path)
    se = _read(se_path)

    if not pr:
        print(f"Missing paraphrase results: {pr_path}")
        sys.exit(1)

    rows_by_key = baseline_rows_from_paraphrase(pr, args.drift_tau)
    apply_qa(rows_by_key, qa)
    degenerate_labels = _mask_degenerate_baselines(pr, rows_by_key, args.drift_tau)

    order = ["baseline_1", "baseline_2", "baseline_3"]
    table_rows = [rows_by_key[k] for k in order if k in rows_by_key]

    if se:
        table_rows.append(state_engine_row(se, args.drift_tau))

    dk = f"Drift@{args.drift_tau}"
    header = ["Method", "GED mean", "Jaccard mean", dk, "QA EM", "QA R@1", "Unsupported (note)", "Conflict rec."]

    md_lines = [
        "# Method comparison: baselines vs evidence-bound state engine",
        "",
        "**Inputs** (paths relative to repository root where applicable)",
        "",
        f"- Paraphrase / extractor log: `{_repo_relative(pr_path, _root)}`",
        f"- QA-over-graph (baselines): `{_repo_relative(qa_path, _root)}`"
        + ("" if qa else " _(missing — run run_qa_eval.py)_"),
        f"- State engine: `{_repo_relative(se_path, _root)}`" + ("" if se else " _(missing)_"),
        "",
        "**Fair QA**: Regenerate QA results with the same `--qa_file` you pass to `run_state_engine.py`, e.g.",
        "`python baselines/run_qa_eval.py --paraphrase_results results/paraphrase_results.json --qa_file results/qa_aligned_100.json --output results/qa_results.json`",
        "",
        "**Graph metrics (GED / Jaccard / drift)**: Baseline columns use raw triples from each baseline pipeline. "
        "State-engine columns use **canonicalized, oriented** triples before pairwise stability (see `prepare_triples_for_state`). "
        "Trend comparison is meaningful; numerical equality across rows is not required.",
        "",
        "## Summary table",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    csv_rows = []
    for r in table_rows:
        csv_rows.append(
            [
                r["label"],
                r["GED_mean"],
                r["Jaccard_mean"],
                r[dk],
                r["QA_EM"],
                r["QA_R@1"],
                r["Unsupported_note"],
                r["Conflict_records"],
            ]
        )
        md_lines.append(
            "| "
            + " | ".join(
                [
                    r["label"],
                    _fmt(r["GED_mean"]),
                    _fmt(r["Jaccard_mean"]),
                    _fmt(r[dk]),
                    _fmt(r["QA_EM"]) if r["QA_EM"] is not None else "—",
                    _fmt(r["QA_R@1"]) if r["QA_R@1"] is not None else "—",
                    _fmt(r["Unsupported_note"]) if r["Unsupported_note"] is not None else "—",
                    str(r["Conflict_records"]) if r["Conflict_records"] is not None else "—",
                ]
            )
            + " |"
        )

    md_lines.extend(
        [
            "",
            f"Same numbers are in `{_repo_relative(out_csv, _root)}` for plotting or spreadsheets.",
            "",
            "## Results figure (example graph, set 0)",
            "",
            "Merged view for the first paraphrase set: synonyms combined; low-information object nodes omitted for clarity. "
            "**Active** edges (dark) are strongly supported; **uncertain** edges (red) need more evidence under the engine's policy.",
            "",
            "![Set 0: merged graph — aspirin and gastric outcomes](state_engine_graph_set0.png)",
            "",
            "**See also:** `state_engine_report.html` in this folder for the full HTML report.",
        ]
    )

    if degenerate_labels:
        md_lines.extend(
            [
                "",
                "**Note:** No triples in `paraphrase_results.json` for: "
                + ", ".join(degenerate_labels)
                + ". Graph metrics would be vacuous (empty vs empty); QA has no graph. "
                "Re-run neural baselines and merge on that paraphrase file, e.g. "
                "`python3 baselines/fill_paraphrase_neural_baselines.py --input <path-from-Inputs-above>`, "
                "or regenerate the full file with "
                "`python3 baselines/run_graph_eval.py --paraphrases_file baselines/data/paraphrases/paraphrase_sets_50.json "
                "--baselines 1,2,3 --output results/paraphrase_results.json`.",
            ]
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    def _csv_val(x):
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        return x

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in csv_rows:
            w.writerow([row[0], *[_csv_val(x) for x in row[1:]]])

    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
