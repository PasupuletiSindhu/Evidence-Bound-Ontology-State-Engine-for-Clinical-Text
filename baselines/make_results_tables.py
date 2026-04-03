#!/usr/bin/env python3
"""Create readable baseline comparison tables from result JSON logs.

Outputs:
- Markdown report with tables
- CSV tables for easy spreadsheet usage
"""
import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_paraphrase_table(paraphrase_json):
    if not paraphrase_json:
        return [], []
    sets = paraphrase_json.get("sets", [])
    rows = []
    for idx, item in enumerate(sets, start=1):
        row = [f"P{idx}"]
        for b in ("baseline_1", "baseline_2", "baseline_3"):
            stats = (item.get(b) or {}).get("stats", {})
            ged = (stats.get("ged_pairwise") or {}).get("mean")
            jac = (stats.get("jaccard_pairwise") or {}).get("mean")
            row.extend([ged, jac])
        rows.append(row)
    return ["Paraphrase", "B1 GED", "B1 Jaccard", "B2 GED", "B2 Jaccard", "B3 GED", "B3 Jaccard"], rows


def build_qa_baseline_table(qa_json):
    if not qa_json:
        return [], []
    rows = []
    for b in ("baseline_1", "baseline_2", "baseline_3"):
        m = (qa_json.get(b) or {}).get("metrics", qa_json.get(b) or {})
        if not isinstance(m, dict):
            continue
        rows.append([b, m.get("exact_match"), m.get("recall_at_1"), m.get("num_questions")])
    return ["Baseline", "Exact Match", "Recall@1", "Num Questions"], rows


def build_qa_question_table(qa_json):
    if not qa_json:
        return [], []
    rows = []
    for b in ("baseline_1", "baseline_2", "baseline_3"):
        per_q = (qa_json.get(b) or {}).get("per_question", [])
        for item in per_q:
            pred = item.get("predicted", [])
            pred_show = "; ".join(pred[:3]) if isinstance(pred, list) else str(pred)
            rows.append([
                b,
                item.get("question", ""),
                item.get("gold", ""),
                pred_show,
                item.get("exact_match", False),
                item.get("recall_at_1", False),
            ])
    return ["Baseline", "Question", "Gold", "Predicted (first 3)", "Exact Match", "Recall@1"], rows


def _to_markdown_table(header, rows):
    if not header:
        return "_No data_\n"
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        out.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description="Build readable markdown/csv tables from result logs.")
    p.add_argument("--paraphrase_results", type=str, default="results/paraphrase_results.json")
    p.add_argument("--qa_results", type=str, default="results/qa_results.json")
    p.add_argument("--out_md", type=str, default="results/summary_tables.md")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    paraphrase_path = Path(args.paraphrase_results)
    qa_path = Path(args.qa_results)
    if not paraphrase_path.is_absolute():
        paraphrase_path = root / paraphrase_path
    if not qa_path.is_absolute():
        qa_path = root / qa_path

    paraphrase_json = _read_json(paraphrase_path)
    qa_json = _read_json(qa_path)

    ph_h, ph_rows = build_paraphrase_table(paraphrase_json)
    qa_h, qa_rows = build_qa_baseline_table(qa_json)
    qd_h, qd_rows = build_qa_question_table(qa_json)

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = root / out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation Tables",
        "",
        f"- Paraphrase results source: `{paraphrase_path}`",
        f"- QA results source: `{qa_path}`",
        "",
        "## Baseline Comparison (Paraphrase)",
        _to_markdown_table(ph_h, ph_rows),
        "## Baseline Comparison (QA)",
        _to_markdown_table(qa_h, qa_rows),
        "## QA Per-Question Log",
        _to_markdown_table(qd_h, qd_rows),
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    _write_csv(out_md.with_name("paraphrase_baseline_table.csv"), ph_h, ph_rows)
    _write_csv(out_md.with_name("qa_baseline_table.csv"), qa_h, qa_rows)
    _write_csv(out_md.with_name("qa_per_question_table.csv"), qd_h, qd_rows)
    print(f"Wrote markdown table report: {out_md}")
    print("Wrote CSV tables: paraphrase_baseline_table.csv, qa_baseline_table.csv, qa_per_question_table.csv")


if __name__ == "__main__":
    main()
