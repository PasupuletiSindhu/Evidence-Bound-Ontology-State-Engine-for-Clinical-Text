# Answers QA over the extracted graph

#!/usr/bin/env python3
from __future__ import annotations

# Graph eval: incremental + paraphrase. Writes eval_results.json, paraphrase_results.json, plots.
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "13")

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from experiment_runner import (
    run_incremental_simulation,
    compute_metrics_over_trajectories,
    compute_paraphrase_metrics_full,
    report_statistics,
)
from models.bert_ner import BERTNERModel
from models.bert_relation import BERTRelationModel
from pipelines import StatelessNeuralExtraction, OntologyNormalizedStateless, SingleExtractorVariant
from ontology.umls_mapper import UMLSMapper
from checkpoint_resolve import relation_load_dir

try:
    from graph_eval_plots import (
        plot_paraphrase_metrics_distribution,
        plot_paraphrase_heatmaps,
        plot_paraphrase_knowledge_graphs,
        plot_first_five_paraphrases_all_baselines,
        plot_incremental_metrics_summary,
        plot_baselines_comparison,
    )
    _PLOTS_AVAILABLE = True
except ImportError:
    _PLOTS_AVAILABLE = False

try:
    from qa_eval import run_qa_eval, load_qa_dataset
    _QA_AVAILABLE = True
except ImportError:
    _QA_AVAILABLE = False

DEFAULT_QA_FILE = _here / "data" / "qa" / "qa_50.json" if (_here / "data" / "qa" / "qa_50.json").exists() else _here / "data" / "qa" / "aspirin_qa.json"


DEFAULT_TEXTS = [
    "Aspirin may cause gastric bleeding in some patients.",
    "Metformin is used to treat type 2 diabetes.",
    "Ibuprofen can reduce fever and pain.",
    "Warfarin increases the risk of bleeding.",
    "Acetaminophen is metabolized by the liver.",
]


DEFAULT_PRETRAINED_NER = "Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-LoRA-v1"
DEFAULT_PARAPHRASES_FILE = _here / "data" / "paraphrases" / "paraphrase_sets_30.json" if (_here / "data" / "paraphrases" / "paraphrase_sets_30.json").exists() else _here / "data" / "paraphrases" / "paraphrase_sets_5.json"


def _get_mapper(mapper_file: str, _here: Path, no_corpus_mapper: bool = False):
    # Baseline 2: BC5CDR corpus -> mapper_file -> cui_map.txt -> empty
    if not no_corpus_mapper:
        bc5cdr_dir = _here / "data" / "bc5cdr"
        try:
            from loaders.bc5cdr_loader import load_bc5cdr
            train_data, _, _ = load_bc5cdr(str(bc5cdr_dir), download_if_missing=False)
            if train_data:
                m = UMLSMapper.from_bc5cdr(train_data)
                print("Baseline 2: using standard mapping from BC5CDR corpus (MeSH IDs from annotations)")
                return m
        except Exception:
            pass
    if mapper_file and Path(mapper_file).exists():
        m = UMLSMapper.from_file(mapper_file)
        print("Baseline 2: loaded mapper from --mapper_file")
        return m
    cui_map = _here / "data" / "cui_map.txt"
    if cui_map.exists():
        print("Baseline 2: loaded mapper from data/cui_map.txt")
        return UMLSMapper.from_file(str(cui_map))
    if no_corpus_mapper:
        print("Baseline 2: no mapper (--no_corpus_mapper and no file); entities as-is.")
    else:
        print("Baseline 2: no mapper (no BC5CDR data or mapping file); entities as-is.")
    return UMLSMapper()


def _to_serializable(obj):
    # so we can json.dump results (NaN/Inf are not valid JSON — use null)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    if isinstance(obj, np.floating):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def _triples_to_json(triples):
    # (s,r,o) or (s,r,o,_) -> [s,r,o]
    out = []
    for t in triples:
        if len(t) >= 3:
            out.append([str(t[0]), str(t[1]), str(t[2])])
    return out


def _write_results_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2)
    print(f"Results saved to {path}")


def _write_summary_txt(path: Path, mode: str, stats: dict, extra_lines: list = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Graph evaluation summary ({mode})", "=" * 50, ""]
    for k, v in stats.items():
        if isinstance(v, dict) and "mean" in v:
            lines.append(f"  {k}: mean = {v['mean']:.4f}, 95% CI = [{v.get('ci_lower', v['mean']):.4f}, {v.get('ci_upper', v['mean']):.4f}]")
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved to {path}")


def _run_one_paraphrase_set(paraphrases: list, fact: str, baseline_ids: list, get_process_fns) -> dict:
    all_results = {}
    for bl_key, process_fn in get_process_fns():
        metrics, graphs, ged_matrix, jaccard_matrix = compute_paraphrase_metrics_full(paraphrases, process_fn)
        stats = report_statistics(metrics)
        all_results[bl_key] = {
            "stats": stats,
            "triples_per_paraphrase": [_triples_to_json(g) for g in graphs],
            "ged_pairwise": metrics["ged_pairwise"],
            "jaccard_pairwise": metrics["jaccard_pairwise"],
            "pairwise_ged_matrix": ged_matrix,
            "pairwise_jaccard_matrix": jaccard_matrix,
            "graphs_raw": graphs,
        }
    return all_results


def _run_paraphrase(path: Path, out_path: Path, args, baseline_ids: list, get_process_fns) -> None:
    data = json.loads(path.read_text())
    sets_to_run = []
    if data.get("paraphrase_sets"):
        sets_to_run = [
            {"fact": s.get("fact", "(none)"), "paraphrases": s.get("paraphrases") or s.get("texts") or []}
            for s in data["paraphrase_sets"]
        ]
        sets_to_run = [s for s in sets_to_run if s["paraphrases"]]
    else:
        paraphrases = data.get("paraphrases") or data.get("texts") or []
        if paraphrases:
            sets_to_run = [{"fact": data.get("fact", "(none)"), "paraphrases": paraphrases}]
    if not sets_to_run:
        print(f"No 'paraphrases', 'paraphrase_sets', or 'texts' in {path}")
        return
    num_sets = len(sets_to_run)
    print(f"Paraphrase experiment: {num_sets} set(s); running Baselines {', '.join(baseline_ids)}.")
    all_sets_results = []
    merged_triples_per_baseline = {bl_key: [] for bl_key in ("baseline_1", "baseline_2", "baseline_3") if bl_key in [f"baseline_{b}" for b in baseline_ids]}
    for idx, one in enumerate(sets_to_run):
        fact, paraphrases = one["fact"], one["paraphrases"]
        print(f"  Set {idx + 1}/{num_sets}: fact = {fact[:50]}... ; {len(paraphrases)} paraphrases.")
        for bl_key, process_fn in get_process_fns():
            if bl_key not in merged_triples_per_baseline:
                merged_triples_per_baseline[bl_key] = []
        all_results = _run_one_paraphrase_set(paraphrases, fact, baseline_ids, get_process_fns)
        for bl_key in all_results:
            merged_triples_per_baseline.setdefault(bl_key, []).extend(all_results[bl_key]["triples_per_paraphrase"])
        set_payload = {
            "fact": fact,
            "paraphrases": paraphrases,
            **{bl_key: {"stats": all_results[bl_key]["stats"], "triples_per_paraphrase": all_results[bl_key]["triples_per_paraphrase"],
               "ged_pairwise": all_results[bl_key]["ged_pairwise"], "jaccard_pairwise": all_results[bl_key]["jaccard_pairwise"],
               "pairwise_ged_matrix": all_results[bl_key]["pairwise_ged_matrix"], "pairwise_jaccard_matrix": all_results[bl_key]["pairwise_jaccard_matrix"]}
             for bl_key in all_results},
        }
        all_sets_results.append(set_payload)
        for bl_key in sorted(all_results.keys()):
            stats = all_results[bl_key]["stats"]
            n_pairs = len(all_results[bl_key]["ged_pairwise"])
            print(f"    [{bl_key}] pairs: {n_pairs}; GED mean = {stats.get('ged_pairwise', {}).get('mean', 0):.4f}; Jaccard mean = {stats.get('jaccard_pairwise', {}).get('mean', 0):.4f}")
    print("\n--- Paraphrase graph metrics (per set above) ---")
    results = {
        "mode": "paraphrase",
        "num_sets": num_sets,
        "config": {"paraphrases_file": str(path), "ner_dir": args.ner_dir, "rel_dir": args.rel_dir, "baselines": baseline_ids},
        "sets": all_sets_results,
    }
    if num_sets == 1:
        results["fact"] = sets_to_run[0]["fact"]
        results["paraphrases"] = sets_to_run[0]["paraphrases"]
        for bl_key in all_sets_results[0]:
            if bl_key not in ("fact", "paraphrases"):
                results[bl_key] = all_sets_results[0][bl_key]
    _write_results_json(out_path, results)
    summary_lines = ["", f"Sets: {num_sets}", f"Baselines: {', '.join(baseline_ids)}"]
    for idx, set_payload in enumerate(all_sets_results):
        fshort = set_payload["fact"][:60] + ("..." if len(set_payload["fact"]) > 60 else "")
        summary_lines.append(f"\n--- Set {idx + 1}: {fshort} ---")
        for bl_key in sorted(k for k in set_payload if k not in ("fact", "paraphrases")):
            for k, v in (set_payload[bl_key].get("stats") or {}).items():
                if isinstance(v, dict) and "mean" in v:
                    summary_lines.append(f"  [{bl_key}] {k}: mean = {v['mean']:.4f}, 95% CI = [{v.get('ci_lower', v['mean']):.4f}, {v.get('ci_upper', v['mean']):.4f}]")
    all_results_for_qa = {bl_key: {"triples_per_paraphrase": merged_triples_per_baseline.get(bl_key, [])} for bl_key in merged_triples_per_baseline}
    if num_sets == 1:
        all_results_for_qa = all_sets_results[0]

    # Downstream QA over merged graph (all sets combined per baseline)
    if _QA_AVAILABLE and DEFAULT_QA_FILE.exists():
        try:
            qa_examples, qa_aliases = load_qa_dataset(str(DEFAULT_QA_FILE))
            summary_lines.append("\n--- QA over graph (downstream, merged triples from all sets) ---")
            qa_results_for_json = {"mode": "qa_downstream", "config": {"qa_file": str(DEFAULT_QA_FILE)}, "num_questions": len(qa_examples)}
            qa_baseline_keys = sorted(k for k in all_results_for_qa if k.startswith("baseline_"))
            qa_mapper = (
                _get_mapper(
                    getattr(args, "mapper_file", None) or "",
                    _here,
                    getattr(args, "no_corpus_mapper", False),
                )
                if any(k == "baseline_2" for k in qa_baseline_keys)
                else None
            )
            entity_to_id_b2 = (lambda e: qa_mapper.normalize(e) or e) if qa_mapper else None
            for bl_key in qa_baseline_keys:
                triples = []
                for block in all_results_for_qa[bl_key].get("triples_per_paraphrase", []):
                    for t in block:
                        if isinstance(t, (list, tuple)) and len(t) >= 3:
                            triples.append((str(t[0]), str(t[1]), str(t[2])))
                entity_to_id = entity_to_id_b2 if bl_key == "baseline_2" else None
                _, qa_metrics = run_qa_eval(triples, qa_examples, relation_aliases=qa_aliases, entity_to_id=entity_to_id)
                summary_lines.append(f"  [{bl_key}] Exact Match: {qa_metrics['exact_match']:.2%}  Recall@1: {qa_metrics['recall_at_1']:.2%}  (n={qa_metrics['num_questions']})")
                qa_results_for_json[bl_key] = qa_metrics
            _write_results_json(out_path.parent / "qa_results.json", qa_results_for_json)
            qa_txt = out_path.parent / "qa_results.txt"
            qa_txt.write_text(
                "QA over graph (downstream)\n" + "=" * 50 + "\n\n"
                + "\n".join(f"  [{bl_key}] Exact Match: {qa_results_for_json[bl_key]['exact_match']:.2%}  Recall@1: {qa_results_for_json[bl_key]['recall_at_1']:.2%}  (n={qa_results_for_json[bl_key]['num_questions']})" for bl_key in sorted(qa_results_for_json) if bl_key not in ("mode", "config", "num_questions"))
                + "\n",
                encoding="utf-8",
            )
            summary_lines.append("")
            print("\n--- QA over graph (downstream) ---")
            print("Exact Match / Recall@1 written to results/qa_results.json and results/qa_results.txt")
            for bl_key in sorted(qa_results_for_json):
                if bl_key in ("mode", "config", "num_questions"):
                    continue
                m = qa_results_for_json[bl_key]
                print(f"  [{bl_key}] Exact Match: {m['exact_match']:.2%}  Recall@1: {m['recall_at_1']:.2%}  (n={m['num_questions']})")
        except Exception as e:
            summary_lines.append(f"\nQA eval skipped: {e}")
    _write_summary_txt(out_path.with_suffix(".txt"), "paraphrase (all baselines)", {}, extra_lines=summary_lines)
    if _PLOTS_AVAILABLE and not args.no_plots:
        plots_dir = Path(args.plots_dir or (out_path.parent / "plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)
        first_set = all_sets_results[0]
        first_paraphrases = first_set["paraphrases"]
        first_fact = first_set["fact"]
        baseline_to_graphs = {}
        for bl_key in (k for k in first_set if k not in ("fact", "paraphrases")):
            bl_data = first_set[bl_key]
            plot_paraphrase_metrics_distribution(
                np.asarray(bl_data["ged_pairwise"]), np.asarray(bl_data["jaccard_pairwise"]),
                plots_dir / f"paraphrase_metrics_distribution_{bl_key}.png",
            )
            plot_paraphrase_heatmaps(
                bl_data["pairwise_ged_matrix"], bl_data["pairwise_jaccard_matrix"], first_paraphrases,
                plots_dir / f"paraphrase_heatmaps_{bl_key}.png",
            )
            # Use graphs_raw if present (in-memory), else triples_per_paraphrase (e.g. from loaded JSON)
            graphs_for_plot = bl_data.get("graphs_raw")
            if graphs_for_plot is None:
                raw = bl_data.get("triples_per_paraphrase", [])
                graphs_for_plot = [[tuple(t) for t in g] for g in raw]
            baseline_to_graphs[bl_key] = graphs_for_plot
            plot_paraphrase_knowledge_graphs(
                first_paraphrases, graphs_for_plot, first_fact,
                plots_dir / f"paraphrase_knowledge_graphs_{bl_key}.png", max_plots=8,
            )
        plot_first_five_paraphrases_all_baselines(
            first_paraphrases, baseline_to_graphs, first_fact,
            plots_dir / "first_five_paraphrases_all_baselines.png", n_paraphrases=5,
        )
        print(f"Plots saved to {plots_dir} (first set only)")


def main():
    p = argparse.ArgumentParser(description="Run graph evaluation (GED, Jaccard, unsupported, conflict precision).")
    p.add_argument("--ner_dir", type=str, default=str(_here / "out" / "ner"), help="Path to NER checkpoint (if exists). Else use --ner_model.")
    p.add_argument("--ner_model", type=str, default=None, help=f"Pretrained NER from HuggingFace if --ner_dir missing (default: {DEFAULT_PRETRAINED_NER}).")
    p.add_argument("--rel_dir", type=str, default=str(_here / "out" / "relation"), help="Path to trained relation checkpoint.")
    p.add_argument("--texts_file", type=str, default=None, help="Optional: one text per line; if omitted, use built-in samples.")
    p.add_argument("--paraphrases_file", type=str, default=None, help="JSON with 'paraphrases' list (same fact, multiple phrasings). Runs paraphrase GED/Jaccard and generates graph plots.")
    p.add_argument("--N", type=int, default=10, help="Number of random orderings for trajectories (ignored if --paraphrases_file set).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="Save full results to this JSON file. Default: results/paraphrase_results.json (paraphrase mode) or results/eval_results.json (incremental).")
    p.add_argument("--plots_dir", type=str, default=None, help="Directory for plots (paraphrase: distributions, heatmaps, knowledge-graph figures). Default: results/plots or same dir as --output.")
    p.add_argument("--no_plots", action="store_true", help="Do not generate any plots.")
    p.add_argument("--baselines", type=str, default="1,2,3", help="Comma-separated baseline IDs to run (default: 1,2,3 = all). 1=Stateless, 2=Ontology-normalized, 3=Qwen single-extractor.")
    p.add_argument("--mapper_file", type=str, default=None, help="Baseline 2: optional mapping file (entity_text\\tCUI or MeSH, one per line). Used only if BC5CDR corpus is not used.")
    p.add_argument("--no_corpus_mapper", action="store_true", help="Baseline 2: do not use BC5CDR corpus; use only --mapper_file or data/cui_map.txt for mapping.")
    p.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-0.5B", help="Qwen model for Baseline 3.")
    args = p.parse_args()

    baseline_ids = [b.strip() for b in args.baselines.split(",") if b.strip()]
    if not baseline_ids:
        baseline_ids = ["1", "2", "3"]
    need_relation = "1" in baseline_ids or "2" in baseline_ids
    ner_path = Path(args.ner_dir)
    rel_path = Path(args.rel_dir)
    rel_load: Path | None = None
    if need_relation:
        if not rel_path.exists():
            print(f"Relation checkpoint not found: {rel_path}. Train first: python train.py --dataset bc5cdr --model_name dmis-lab/biobert-base-cased-v1.1 --skip_ner_training")
            sys.exit(1)
        try:
            rel_load = relation_load_dir(rel_path)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        if rel_load != rel_path.resolve():
            print(f"  Relation: loading from {rel_load} (resolved under {rel_path})")

    # Default output path for saving results
    if args.output is None:
        args.output = str(_here / "results" / ("paraphrase_results.json" if args.paraphrases_file else "eval_results.json"))

    # Load models and build process_fn per baseline
    ner, rel = None, None
    if "1" in baseline_ids or "2" in baseline_ids:
        print("Loading NER and relation models (for Baselines 1 & 2)...")
        if args.ner_model or not ner_path.exists():
            hub_id = args.ner_model or DEFAULT_PRETRAINED_NER
            print(f"  NER: pretrained from HuggingFace ({hub_id})")
            ner = BERTNERModel().load_pretrained(hub_id)
        else:
            ner = BERTNERModel().load(str(ner_path))
        assert rel_load is not None
        rel = BERTRelationModel(model_name="dmis-lab/biobert-base-cased-v1.1").load(str(rel_load))

    # Build baseline pipelines once and reuse across all sets/runs.
    process_fns = []
    if "1" in baseline_ids:
        pipe1 = StatelessNeuralExtraction(ner, rel)
        process_fns.append(("baseline_1", pipe1.process))
    if "2" in baseline_ids:
        mapper = _get_mapper(args.mapper_file, _here, getattr(args, "no_corpus_mapper", False))
        pipe2 = OntologyNormalizedStateless(ner, rel, mapper)
        process_fns.append(("baseline_2", pipe2.process))
    if "3" in baseline_ids:
        pipe3 = SingleExtractorVariant("qwen", qwen_model_name=args.qwen_model)
        print("Loading Baseline 3 (Qwen prompt-based)...")
        pipe3._get_qwen_extractor()  # load once
        process_fns.append(("baseline_3", pipe3.process))

    def get_process_fns():
        return process_fns

    if args.paraphrases_file:
        path = Path(args.paraphrases_file)
        if not path.exists():
            print(f"Paraphrases file not found: {path}")
            sys.exit(1)
        _run_paraphrase(path, Path(args.output), args, baseline_ids, get_process_fns)
        print("\nDone.")
        return

    if args.texts_file:
        texts = [line.strip() for line in Path(args.texts_file).read_text().splitlines() if line.strip()]
        if not texts:
            print("No lines in --texts_file.")
            sys.exit(1)
        print(f"Loaded {len(texts)} texts from {args.texts_file}")
    else:
        texts = DEFAULT_TEXTS
        print(f"Using {len(texts)} built-in sample texts.")

    print(f"Running incremental simulation (N={args.N} orderings, seed={args.seed}) for Baselines {', '.join(baseline_ids)}...")
    all_results = {}
    for bl_key, process_fn in get_process_fns():
        print(f"  Running {bl_key}...")
        trajectories = run_incremental_simulation(texts, process_fn, N=args.N, seed=args.seed)
        supported_by_text = [set((x[0], x[1], x[2]) for x in process_fn(text)) for text in texts]
        metrics = compute_metrics_over_trajectories(trajectories, supported_by_text=supported_by_text)
        stats = report_statistics(metrics)
        all_results[bl_key] = {"stats": stats, "metrics": metrics}

    print("\n--- Graph evaluation metrics (all baselines) ---")
    print("(GED ↓ = lower is better; Jaccard ↑ = higher is better; Unsupported ↓ = lower is better)\n")
    for key, label in [
        ("ged_consecutive", "GED (consecutive) ↓"),
        ("jaccard_consecutive", "Evidence Jaccard ↑"),
        ("unsupported_rate", "Unsupported inference rate ↓"),
        ("conflict_precision", "Heuristic conflict (candidate rate)"),
    ]:
        for bl_key in sorted(all_results.keys()):
            s = all_results[bl_key]["stats"].get(key, {})
            mean = s.get("mean", 0)
            ci_lo, ci_hi = s.get("ci_lower", mean), s.get("ci_upper", mean)
            if key == "unsupported_rate":
                print(f"  [{bl_key}] {label}: {mean:.2%}  (95% CI: {ci_lo:.2%} – {ci_hi:.2%})")
            else:
                print(f"  [{bl_key}] {label}: {mean:.4f}  (95% CI: {ci_lo:.4f} – {ci_hi:.4f})")
        print()

    # Save incremental results (all baselines)
    out_path = Path(args.output)
    results = {
        "mode": "incremental",
        "config": {"N": args.N, "seed": args.seed, "ner_dir": args.ner_dir, "rel_dir": args.rel_dir, "num_texts": len(texts), "baselines": baseline_ids},
    }
    for bl_key, bl_data in all_results.items():
        results[bl_key] = {
            "stats": bl_data["stats"],
            "ged_consecutive": bl_data["metrics"]["ged_consecutive"],
            "jaccard_consecutive": bl_data["metrics"]["jaccard_consecutive"],
            "unsupported_rate": bl_data["metrics"]["unsupported_rate"],
            "conflict_precision": bl_data["metrics"]["conflict_precision"],
        }
    _write_results_json(out_path, results)
    summary_lines = [f"Baselines: {', '.join(baseline_ids)}", ""]
    for bl_key in sorted(all_results.keys()):
        summary_lines.append(f"{bl_key}:")
        for k, v in all_results[bl_key]["stats"].items():
            if isinstance(v, dict) and "mean" in v:
                summary_lines.append(f"  {k}: mean = {v['mean']:.4f}, 95% CI = [{v.get('ci_lower', v['mean']):.4f}, {v.get('ci_upper', v['mean']):.4f}]")
        summary_lines.append("")
    _write_summary_txt(out_path.with_suffix(".txt"), "incremental (all baselines)", {}, extra_lines=summary_lines)

    if _PLOTS_AVAILABLE and not args.no_plots:
        plots_dir = Path(args.plots_dir or (out_path.parent / "plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)
        baseline_stats_only = {k: v["stats"] for k, v in all_results.items()}
        plot_baselines_comparison(baseline_stats_only, plots_dir / "incremental_baselines_comparison.png")
        for bl_key, bl_data in all_results.items():
            plot_incremental_metrics_summary(bl_data["stats"], plots_dir / f"incremental_metrics_{bl_key}.png")
        print(f"Plots saved to {plots_dir}")

    # No flags: also run paraphrase if default file exists (5 sets, or fallback to single aspirin fact)
    paraphrase_file = DEFAULT_PARAPHRASES_FILE if DEFAULT_PARAPHRASES_FILE.exists() else _here / "data" / "paraphrases" / "aspirin_fact.json"
    if paraphrase_file.exists():
        print("\n--- Paraphrase (default file) ---")
        _run_paraphrase(paraphrase_file, _here / "results" / "paraphrase_results.json", args, baseline_ids, get_process_fns)

    print("\nDone. See EVALUATION.md for report tables and interpretation.")


if __name__ == "__main__":
    main()
