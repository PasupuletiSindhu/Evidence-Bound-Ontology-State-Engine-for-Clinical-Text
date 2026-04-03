#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "13")

# Repo root (Capstone/), not Demo/
ROOT = Path(__file__).resolve().parent.parent
DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BSL = str(ROOT / "baselines")
if _BSL not in sys.path:
    sys.path.insert(0, _BSL)
from checkpoint_resolve import try_relation_load_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Demo dataset paths — swap these files to change data; pipeline code is unchanged.
# ---------------------------------------------------------------------------
DEMO_PARAPHRASE_RESULTS = DATA / "paraphrase_results_demo.json"
# Writable merge of demo seed + optional neural baselines 1/2 (never overwrites the seed file).
DEMO_PARAPHRASE_EFFECTIVE = DEMO / "paraphrase_results_effective.json"
DEMO_PARAPHRASE_SETS = DATA / "paraphrase_sets_demo.json"
DEMO_RELATION_MAP_SEED = DEMO / "relation_map_seed.json"
DEMO_QA_ALIGNED = DEMO / "qa_aligned_demo.json"
DEMO_RELATION_MAP = DEMO / "relation_map.json"
DEMO_RELATION_CLUSTER_MAP = DEMO / "relation_cluster_map.json"
DEMO_STATE_JSON = DEMO / "state_engine_results.json"
DEMO_STATE_MD = DEMO / "state_engine_results.md"
DEMO_QA_RESULTS = DEMO / "qa_results.json"
DEMO_CMP_MD = DEMO / "method_comparison.md"
DEMO_CMP_CSV = DEMO / "method_comparison.csv"

MAPPER = ROOT / "baselines" / "data" / "cui_map.txt"

_GENERATED = (
    DEMO_PARAPHRASE_EFFECTIVE,
    DEMO_QA_ALIGNED,
    DEMO_RELATION_MAP,
    DEMO / "relation_train.json",
    DEMO / "relation_map_stats.json",
    DEMO_RELATION_CLUSTER_MAP,
    DEMO_STATE_JSON,
    DEMO_STATE_MD,
    DEMO_QA_RESULTS,
    DEMO_CMP_MD,
    DEMO_CMP_CSV,
    DEMO / "state_engine_report.html",
    DEMO / "state_engine_graph_set0.png",
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def _clean_generated() -> None:
    for p in _GENERATED:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _rel_to_root(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT.resolve()))


def _pair_count(n: int) -> int:
    return (n * (n - 1)) // 2 if n >= 2 else 0


def _deterministic_baseline_block(fact: str, paraphrases: list[str], baseline2: bool = False) -> dict:
    from state_engine.relations import parse_fact_spans

    parsed = parse_fact_spans(fact or "")
    if not parsed:
        triples_pp = [[] for _ in paraphrases]
        stats = {
            "ged_pairwise": {"mean": None, "std": None, "ci_lower": None, "ci_upper": None},
            "jaccard_pairwise": {"mean": None, "std": None, "ci_lower": None, "ci_upper": None},
        }
        return {"stats": stats, "triples_per_paraphrase": triples_pp, "ged_pairwise": [], "jaccard_pairwise": []}

    subj, rel, obj = parsed
    row = [subj, rel, obj]
    triples_pp = [[row] for _ in paraphrases]
    n_pairs = _pair_count(len(paraphrases))
    ged = [0.0] * n_pairs
    jac = [1.0] * n_pairs
    stats = {
        "ged_pairwise": {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
        "jaccard_pairwise": {"mean": 1.0, "std": 0.0, "ci_lower": 1.0, "ci_upper": 1.0},
    }
    # For deterministic demo, baseline_2 uses same canonicalized text triples as baseline_1.
    _ = baseline2
    return {"stats": stats, "triples_per_paraphrase": triples_pp, "ged_pairwise": ged, "jaccard_pairwise": jac}


def _write_deterministic_effective_from_sets(src_sets: Path, out_effective: Path) -> None:
    data = json.loads(src_sets.read_text(encoding="utf-8"))
    psets = data.get("paraphrase_sets") or []
    sets = []
    for s in psets:
        fact = str(s.get("fact") or "").strip()
        paraphrases = [str(p).strip() for p in (s.get("paraphrases") or []) if str(p).strip()]
        if not fact or not paraphrases:
            continue
        b1 = _deterministic_baseline_block(fact, paraphrases, baseline2=False)
        b2 = _deterministic_baseline_block(fact, paraphrases, baseline2=True)
        b3 = _deterministic_baseline_block(fact, paraphrases, baseline2=False)
        sets.append(
            {
                "fact": fact,
                "paraphrases": paraphrases,
                "baseline_1": b1,
                "baseline_2": b2,
                "baseline_3": b3,
            }
        )
    payload = {
        "mode": "paraphrase",
        "description": "Deterministic demo paraphrase results (baseline_1/2/3) from fact parsing only.",
        "num_sets": len(sets),
        "config": {"demo": True, "baselines": ["1", "2", "3"], "deterministic_from_fact": True},
        "sets": sets,
    }
    out_effective.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full Capstone demo: same pipeline as state_engine.run_state_engine, Demo paths only.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previously generated Demo outputs before running.",
    )
    parser.add_argument(
        "--no_semantic_clustering",
        action="store_true",
        help="Pass through to state engine (disable entity registry/engine embedding merge).",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Do not run visualize_results (HTML + PNG).",
    )
    parser.add_argument(
        "--skip_neural_baselines",
        action="store_true",
        help="Skip Baselines 1 & 2 (local NER+RE). Default: run when weights exist under --rel_dir (or checkpoint-* subdirs).",
    )
    parser.add_argument(
        "--rel_dir",
        type=str,
        default=None,
        help="Relation training output directory (default: baselines/out/relation). May contain only checkpoint-* subfolders.",
    )
    parser.add_argument(
        "--ner_dir",
        type=str,
        default=None,
        help="Optional: local NER checkpoint for fill script (default: baselines/out/ner).",
    )
    parser.add_argument(
        "--ner_model",
        type=str,
        default=None,
        help="Optional: HuggingFace NER id when not using --ner_dir (passed to fill script).",
    )
    parser.add_argument(
        "--demo_rel_fallback",
        action="store_true",
        help="Enable NER-only chemical→disease fallback when relation model predicts NR (fill script). Default: disabled.",
    )
    parser.add_argument(
        "--relation_n_clusters",
        type=int,
        default=8,
        help="KMeans k cap when relation clustering falls back (forwarded to state engine).",
    )
    parser.add_argument(
        "--source_baseline",
        type=str,
        default="baseline_3",
        help="Which baseline block to read in paraphrase_results_demo.json (default: baseline_3).",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Qwen model id for Baseline 3 (forwarded to baselines/run_graph_eval.py when regenerating).",
    )
    args, state_engine_forward = parser.parse_known_args()

    py = sys.executable

    if not DEMO_PARAPHRASE_RESULTS.is_file() or not DEMO_PARAPHRASE_SETS.is_file():
        print(
            "Missing Demo inputs: need Demo/data/paraphrase_results_demo.json and "
            "Demo/data/paraphrase_sets_demo.json",
            file=sys.stderr,
        )
        sys.exit(1)
    if not DEMO_RELATION_MAP_SEED.is_file():
        print(
            "Missing Demo/relation_map_seed.json (may be empty JSON {}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.clean:
        _clean_generated()

    # 0) Working paraphrase JSON = copy of demo seed.
    # We do NOT create any deterministic (fact-parsed) baseline triples. If baseline_3
    # is missing coverage for all desired sets, we regenerate baseline_3 using the Qwen model.
    shutil.copy(DEMO_PARAPHRASE_RESULTS, DEMO_PARAPHRASE_EFFECTIVE)
    paraphrase_for_pipeline = DEMO_PARAPHRASE_EFFECTIVE

    # Ensure baseline_3 spans all demo paraphrase sets.
    desired_sets = json.loads(DEMO_PARAPHRASE_SETS.read_text(encoding="utf-8")).get("paraphrase_sets") or []
    desired_n = len(desired_sets)
    current_doc = json.loads(DEMO_PARAPHRASE_EFFECTIVE.read_text(encoding="utf-8"))
    current_n = len(current_doc.get("sets") or [])
    regen_baseline_3 = bool(
        (current_doc.get("config") or {}).get("deterministic_from_fact") is True
    )
    if not regen_baseline_3 and current_n < desired_n:
        regen_baseline_3 = True

    if regen_baseline_3:
        why = (
            "deterministic_from_fact=true in paraphrase_results_effective.json"
            if (current_doc.get("config") or {}).get("deterministic_from_fact") is True
            else f"seed only has {current_n}/{desired_n} sets"
        )
        print(f"Info: Regenerating baseline_3 via Qwen model ({why})...")
        _run(
            [
                py,
                str(ROOT / "baselines" / "run_graph_eval.py"),
                "--paraphrases_file",
                _rel_to_root(DEMO_PARAPHRASE_SETS),
                "--baselines",
                "3",
                "--output",
                _rel_to_root(DEMO_PARAPHRASE_EFFECTIVE),
                "--no_plots",
                "--qwen_model",
                args.qwen_model,
            ]
        )

    rel_root = Path(args.rel_dir) if args.rel_dir else ROOT / "baselines" / "out" / "relation"
    try:
        rel_resolved = try_relation_load_dir(rel_root.resolve())
    except Exception:
        rel_resolved = None

    if rel_resolved is not None:
        fill_cmd = [
            py,
            str(ROOT / "baselines" / "fill_paraphrase_neural_baselines.py"),
            "--input",
            str(DEMO_PARAPHRASE_EFFECTIVE.resolve()),
            "--output",
            str(DEMO_PARAPHRASE_EFFECTIVE.resolve()),
            "--rel_dir",
            str(rel_root.resolve()),
        ]
        if args.ner_dir:
            fill_cmd.extend(["--ner_dir", str(Path(args.ner_dir).resolve())])
        if args.ner_model:
            fill_cmd.extend(["--ner_model", args.ner_model])
        if MAPPER.is_file():
            fill_cmd.extend(["--mapper_file", str(MAPPER.resolve())])
        if args.demo_rel_fallback:
            fill_cmd.append("--demo_rel_fallback")
        _run(fill_cmd)
    else:
        print(
            f"Note: Baselines 1 & 2 skipped — no pytorch_model.bin/model.safetensors under "
            f"{rel_root} or in checkpoint-* subfolders. State engine / QA use demo baseline_3 only. "
            "Train or copy checkpoints, then re-run (or pass --rel_dir). "
            "Train: cd baselines && python3 train.py --dataset bc5cdr "
            "--model_name dmis-lab/biobert-base-cased-v1.1 --skip_ner_training",
            file=sys.stderr,
        )

    # 1) QA JSON aligned to facts + seed relation map (same as generate_aligned_qa for full data)
    ga_cmd = [
        py,
        "-m",
        "state_engine.generate_aligned_qa",
        "--paraphrase_sets",
        _rel_to_root(DEMO_PARAPHRASE_SETS),
        "--qa_count",
        "100",
        "--relation_map",
        _rel_to_root(DEMO_RELATION_MAP_SEED),
        "--output",
        _rel_to_root(DEMO_QA_ALIGNED),
    ]
    if MAPPER.is_file():
        ga_cmd.extend(["--mapper_file", _rel_to_root(MAPPER)])
    _run(ga_cmd)

    # 2) State engine — same module/flags as production; paths only differ
    se_cmd = [
        py,
        "-m",
        "state_engine.run_state_engine",
        "--paraphrase_results",
        _rel_to_root(paraphrase_for_pipeline),
        "--qa_file",
        _rel_to_root(DEMO_QA_ALIGNED),
        "--relation_map",
        _rel_to_root(DEMO_RELATION_MAP),
        "--relation_cluster_map",
        _rel_to_root(DEMO_RELATION_CLUSTER_MAP),
        "--output_json",
        _rel_to_root(DEMO_STATE_JSON),
        "--output_md",
        _rel_to_root(DEMO_STATE_MD),
        "--relation_n_clusters",
        str(args.relation_n_clusters),
        "--source_baseline",
        args.source_baseline,
    ]
    if MAPPER.is_file():
        se_cmd.extend(["--mapper_file", _rel_to_root(MAPPER)])
    if args.no_semantic_clustering:
        se_cmd.append("--no_semantic_clustering")
    se_cmd.extend(state_engine_forward)
    _run(se_cmd)

    # 3) Standalone QA eval on raw extractor triples (same QA file as state engine)
    _run(
        [
            py,
            str(ROOT / "baselines" / "run_qa_eval.py"),
            "--paraphrase_results",
            str(paraphrase_for_pipeline.resolve()),
            "--qa_file",
            str(DEMO_QA_ALIGNED.resolve()),
            "--output",
            str(DEMO_QA_RESULTS.resolve()),
        ]
        + (["--mapper_file", str(MAPPER.resolve())] if MAPPER.is_file() else [])
    )

    # 4) Method comparison table
    _run(
        [
            py,
            "-m",
            "state_engine.compare_to_baselines",
            "--paraphrase_results",
            _rel_to_root(paraphrase_for_pipeline),
            "--qa_results",
            _rel_to_root(DEMO_QA_RESULTS),
            "--state_engine_results",
            _rel_to_root(DEMO_STATE_JSON),
            "--out_md",
            _rel_to_root(DEMO_CMP_MD),
            "--out_csv",
            _rel_to_root(DEMO_CMP_CSV),
        ]
    )

    if not args.skip_plots:
        html_out = DEMO / "state_engine_report.html"
        png_out = DEMO / "state_engine_graph_set0.png"
        rc = subprocess.run(
            [
                py,
                "-m",
                "state_engine.visualize_results",
                "--json",
                _rel_to_root(DEMO_STATE_JSON),
                "--html_out",
                _rel_to_root(html_out),
                "--graph_png",
                _rel_to_root(png_out),
                "--set_id",
                "0",
            ],
            cwd=str(ROOT),
        )
        if rc.returncode != 0:
            print(
                "(visualize_results failed — pip install matplotlib networkx or use --skip_plots)",
                file=sys.stderr,
            )

    print("\nDemo pipeline finished. Artifacts under Demo/:")
    for path in _GENERATED:
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            rel = path
        print(f"  {rel}" + (" ✓" if path.is_file() else " (missing)"))


if __name__ == "__main__":
    main()
