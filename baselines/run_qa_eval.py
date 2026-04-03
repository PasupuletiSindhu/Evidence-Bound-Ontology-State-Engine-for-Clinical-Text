#!/usr/bin/env python3
# QA over the extracted graph: answer questions by looking up (subject, relation) or (relation, object).
# Use paraphrase_results.json (per-baseline) or --graph_file for a single graph.
import argparse
import json
import math
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from qa_eval import (
    run_qa_eval,
    load_qa_dataset,
    merge_triples_from_paraphrase_results,
)
from ontology.umls_mapper import UMLSMapper


def _get_mapper(mapper_file: str, no_corpus_mapper: bool = False):
    # Baseline 2: entity -> CUI/MeSH. Tries BC5CDR corpus, then mapper_file, then data/cui_map.txt.
    if not no_corpus_mapper:
        bc5cdr_dir = _here / "data" / "bc5cdr"
        try:
            from loaders.bc5cdr_loader import load_bc5cdr
            train_data, _, _ = load_bc5cdr(str(bc5cdr_dir), download_if_missing=False)
            if train_data:
                m = UMLSMapper.from_bc5cdr(train_data)
                return m
        except Exception:
            pass
    if mapper_file and Path(mapper_file).exists():
        return UMLSMapper.from_file(mapper_file)
    cui_map = _here / "data" / "cui_map.txt"
    if cui_map.exists():
        return UMLSMapper.from_file(str(cui_map))
    return UMLSMapper()


def _to_serializable(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return list(obj)
    return obj


def main():
    p = argparse.ArgumentParser(description="QA over extracted knowledge graph.")
    p.add_argument("--paraphrase_results", type=str, default=None, help="Path to paraphrase_results.json (per-baseline QA).")
    p.add_argument("--graph_file", type=str, default=None, help="Path to JSON list of [s, r, o] triples (single graph).")
    p.add_argument("--qa_file", type=str, default=None, help="QA dataset JSON (default: data/qa/qa_50.json if present, else aspirin_qa.json).")
    p.add_argument("--output", type=str, default=None, help="Write QA results to this JSON (default: results/qa_results.json).")
    p.add_argument("--baselines", type=str, default="baseline_1,baseline_2,baseline_3", help="Comma-separated baseline keys when using paraphrase_results.")
    p.add_argument("--mapper_file", type=str, default=None, help="Baseline 2: entity_text -> CUI/MeSH file (same as run_graph_eval) for QA over ID-based graph.")
    p.add_argument("--no_corpus_mapper", action="store_true", help="Baseline 2: do not use BC5CDR corpus for mapper; use only --mapper_file or data/cui_map.txt.")
    args = p.parse_args()

    if args.qa_file is None:
        qa_50 = _here / "data" / "qa" / "qa_50.json"
        args.qa_file = str(qa_50 if qa_50.exists() else _here / "data" / "qa" / "aspirin_qa.json")

    if not args.paraphrase_results and not args.graph_file:
        default_pr = _here / "results" / "paraphrase_results.json"
        if default_pr.exists():
            args.paraphrase_results = str(default_pr)
        else:
            print("Neither --paraphrase_results nor --graph_file given and results/paraphrase_results.json not found.")
            print("Run paraphrase evaluation first: python run_graph_eval.py --paraphrases_file data/paraphrases/aspirin_fact.json")
            print("Or pass --graph_file path/to/triples.json for a single graph.")
            sys.exit(1)

    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        sys.exit(1)
    examples, relation_aliases = load_qa_dataset(str(qa_path))
    print(f"Loaded {len(examples)} QA examples from {qa_path}")

    if args.paraphrase_results:
        pr_path = Path(args.paraphrase_results)
        if not pr_path.exists():
            print(f"Paraphrase results not found: {pr_path}")
            sys.exit(1)
        results_data = json.loads(pr_path.read_text(encoding="utf-8"))
        if results_data.get("sets"):
            baseline_keys = [k for k in results_data["sets"][0] if k.startswith("baseline_")]
        else:
            baseline_keys = [b.strip() for b in args.baselines.split(",") if b.strip() and results_data.get(b.strip())]
        if not baseline_keys:
            baseline_keys = [k for k in results_data if k.startswith("baseline_") and (results_data.get(k) or {}).get("triples_per_paraphrase") is not None]
        print(f"Running QA for baselines: {baseline_keys}")
        mapper = _get_mapper(args.mapper_file or "", getattr(args, "no_corpus_mapper", False)) if "baseline_2" in baseline_keys else None
        entity_to_id_b2 = (lambda e: mapper.normalize(e) or e) if mapper else None
        if entity_to_id_b2:
            print("  [baseline_2] using entity -> ID mapper for lookup")
        all_metrics = {}
        all_results = {}
        for bl_key in baseline_keys:
            triples = merge_triples_from_paraphrase_results(results_data, bl_key)
            print(f"  [{bl_key}] merged {len(triples)} triples")
            entity_to_id = entity_to_id_b2 if bl_key == "baseline_2" else None
            per_example, metrics = run_qa_eval(
                triples, examples, relation_aliases=relation_aliases, entity_to_id=entity_to_id
            )
            all_metrics[bl_key] = metrics
            all_results[bl_key] = {"metrics": metrics, "per_question": per_example}
        print("\n--- QA over extracted graph (downstream evaluation) ---")
        print("Exact Match: 1 iff predicted set equals {gold}. Recall@1: 1 iff gold in predicted set.\n")
        for bl_key in baseline_keys:
            m = all_metrics[bl_key]
            print(f"  [{bl_key}] Exact Match: {m['exact_match']:.2%}  Recall@1: {m['recall_at_1']:.2%}  (n={m['num_questions']})")
        out_data = {
            "mode": "qa_downstream",
            "config": {"qa_file": str(qa_path), "paraphrase_results": str(pr_path), "baselines": baseline_keys},
            "relation_aliases": relation_aliases,
            "num_questions": len(examples),
        }
        for bl_key in baseline_keys:
            out_data[bl_key] = {
                "metrics": all_results[bl_key]["metrics"],
                "per_question": _to_serializable(all_results[bl_key]["per_question"]),
            }
    else:
        graph_path = Path(args.graph_file)
        if not graph_path.exists():
            print(f"Graph file not found: {graph_path}")
            sys.exit(1)
        triples_raw = json.loads(graph_path.read_text(encoding="utf-8"))
        triples = [tuple(t[:3]) for t in triples_raw if isinstance(t, (list, tuple)) and len(t) >= 3]
        print(f"Loaded {len(triples)} triples from {graph_path}")
        per_example, metrics = run_qa_eval(triples, examples, relation_aliases=relation_aliases)
        print("\n--- QA over extracted graph ---")
        print(f"  Exact Match: {metrics['exact_match']:.2%}  Recall@1: {metrics['recall_at_1']:.2%}  (n={metrics['num_questions']})")
        out_data = {
            "mode": "qa_downstream",
            "config": {"qa_file": str(qa_path), "graph_file": str(graph_path)},
            "metrics": metrics,
            "per_question": _to_serializable(per_example),
        }

    out_path = Path(args.output) if args.output else _here / "results" / "qa_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(out_data), f, indent=2)
    txt_path = out_path.with_name("qa_results.txt")
    baseline_keys = [k for k in out_data if k.startswith("baseline_") and isinstance(out_data.get(k), dict)]
    if baseline_keys:
        bl_metrics = [(k, out_data[k].get("metrics", out_data[k])) for k in baseline_keys]
    else:
        bl_metrics = [("graph", out_data.get("metrics", {}))]
    lines = ["QA over graph (downstream)", "=" * 50, ""]
    for name, m in bl_metrics:
        if m and "exact_match" in m:
            lines.append(f"  [{name}] Exact Match: {m['exact_match']:.2%}  Recall@1: {m['recall_at_1']:.2%}  (n={m.get('num_questions', 0)})")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults saved to {out_path} and {txt_path}")
    print("Done.")


if __name__ == "__main__":
    main()
