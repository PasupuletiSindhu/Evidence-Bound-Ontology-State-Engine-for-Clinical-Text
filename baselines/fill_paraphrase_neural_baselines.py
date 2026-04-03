#!/usr/bin/env python3
"""
Re-run Baselines 1 & 2 (NER + relation) on each paraphrase set and merge into an existing
paraphrase_results.json. Keeps baseline_3 and all metadata unchanged.

Use when paraphrase JSON was built with Qwen only or baseline_1/2 blocks are empty.

  cd /path/to/Capstone/baselines
  python3 fill_paraphrase_neural_baselines.py --input ../results/paraphrase_results.json

Requires trained relation checkpoint at --rel_dir (see baselines/train.py) and NER
(local out/ner or HF --ner_model), same as run_graph_eval.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "13")

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from models.bert_ner import BERTNERModel
from models.bert_relation import BERTRelationModel
from pipelines import OntologyNormalizedStateless, StatelessNeuralExtraction

from checkpoint_resolve import relation_load_dir  # noqa: E402

from run_graph_eval import (  # noqa: E402
    DEFAULT_PRETRAINED_NER,
    _get_mapper,
    _run_one_paraphrase_set,
    _write_results_json,
)
from pipelines.qa_graph_alignment import finalize_baseline_triples, normalize_baseline_triple_row
from pipelines.stateless_pipeline import _sentencize


def _demo_fallback_row_from_entities(entities: list) -> tuple | None:
    """If relation model predicts NR for all pairs, optionally emit one (chem, causes, disease) row."""
    chems = [e for e in entities if "chemical" in str(e.get("label") or "").lower()]
    diss = [e for e in entities if "disease" in str(e.get("label") or "").lower()]
    if not chems or not diss:
        return None
    c, d = chems[0], diss[0]
    return normalize_baseline_triple_row(
        c.get("text", ""),
        "CID",
        d.get("text", ""),
        "positive",
        c,
        d,
    )


class _Baseline1DemoFallback:
    """Wrap StatelessNeuralExtraction: if relation head yields no triples, use one NER-based CID triple."""

    def __init__(self, inner: StatelessNeuralExtraction, ner: BERTNERModel, enabled: bool):
        self._inner = inner
        self._ner = ner
        self._enabled = enabled

    def process(self, text: str):
        out = self._inner.process(text)
        if out or not self._enabled:
            return out
        rows = []
        for sent in _sentencize(text):
            ents = self._ner.extract_entities(sent)
            row = _demo_fallback_row_from_entities(ents)
            if row is not None:
                rows.append(row)
        return finalize_baseline_triples(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Fill baseline_1 and baseline_2 in paraphrase_results.json.")
    p.add_argument("--input", type=str, required=True, help="Existing paraphrase_results.json path.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write path (default: overwrite --input).",
    )
    p.add_argument("--ner_dir", type=str, default=str(_here / "out" / "ner"))
    p.add_argument("--ner_model", type=str, default=None, help="HF NER id if --ner_dir missing.")
    p.add_argument("--rel_dir", type=str, default=str(_here / "out" / "relation"))
    p.add_argument("--mapper_file", type=str, default=None)
    p.add_argument("--no_corpus_mapper", action="store_true")
    p.add_argument(
        "--demo_rel_fallback",
        action="store_true",
        help="If relation model predicts no CID but NER finds chemical+disease, emit one (chem, causes, disease) "
        "triple for baseline_1 (demo / weak checkpoints). Does not change baseline_2.",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.output) if args.output else in_path

    rel_arg = Path(args.rel_dir)
    if not rel_arg.exists():
        print(
            f"Relation directory not found: {rel_arg}. Train first, e.g.\n"
            "  python train.py --dataset bc5cdr --model_name dmis-lab/biobert-base-cased-v1.1 --skip_ner_training",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        rel_path = relation_load_dir(rel_arg)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    if rel_path.resolve() != rel_arg.resolve():
        print(f"Using relation checkpoint: {rel_path}", file=sys.stderr)

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    sets = doc.get("sets") or []
    if not sets:
        print("No 'sets' in JSON.", file=sys.stderr)
        sys.exit(1)

    print("Loading NER and relation models (Baselines 1 & 2)...")
    ner_path = Path(args.ner_dir)
    if args.ner_model or not ner_path.exists():
        hub_id = args.ner_model or DEFAULT_PRETRAINED_NER
        print(f"  NER: pretrained from HuggingFace ({hub_id})")
        ner = BERTNERModel().load_pretrained(hub_id)
    else:
        ner = BERTNERModel().load(str(ner_path))
    rel = BERTRelationModel(model_name="dmis-lab/biobert-base-cased-v1.1").load(str(rel_path))

    pipe1_inner = StatelessNeuralExtraction(ner, rel)
    pipe1 = _Baseline1DemoFallback(pipe1_inner, ner, args.demo_rel_fallback)
    if args.demo_rel_fallback:
        print(
            "  demo_rel_fallback: if the relation head returns no CID, baseline_1 uses one NER-based causes triple.",
            file=sys.stderr,
        )
    mapper = _get_mapper(args.mapper_file or "", _here, args.no_corpus_mapper)
    pipe2 = OntologyNormalizedStateless(ner, rel, mapper)
    process_fns = [("baseline_1", pipe1.process), ("baseline_2", pipe2.process)]

    def get_process_fns():
        return process_fns

    baseline_ids = ["1", "2"]
    for idx, s in enumerate(sets):
        fact = s.get("fact", "(none)")
        paraphrases = s.get("paraphrases") or s.get("texts") or []
        if not paraphrases:
            print(f"  Set {idx}: skip (no paraphrases)")
            continue
        print(f"  Set {idx + 1}/{len(sets)}: {fact[:56]}...")
        all_results = _run_one_paraphrase_set(paraphrases, fact, baseline_ids, get_process_fns)
        for bl_key in ("baseline_1", "baseline_2"):
            blk = all_results[bl_key]
            s[bl_key] = {k: v for k, v in blk.items() if k != "graphs_raw"}

    doc["sets"] = sets
    if isinstance(doc.get("config"), dict):
        doc["config"]["ner_dir"] = str(args.ner_dir)
        doc["config"]["rel_dir"] = str(args.rel_dir)
        doc["config"]["neural_baselines_filled"] = True
        doc["config"]["demo_rel_fallback"] = bool(args.demo_rel_fallback)

    def _count_triples(key: str) -> int:
        total = 0
        for s in sets:
            block = s.get(key) or {}
            for row in block.get("triples_per_paraphrase") or []:
                if isinstance(row, list):
                    total += len(row)
        return total

    if _count_triples("baseline_1") == 0:
        print(
            "WARNING: baseline_1 produced no triples (NER found no entities, relation all NR, "
            "or QA schema mapping dropped edges). Try: pip install peft huggingface_hub; "
            "or --ner_model <HF id>; or --ner_dir with a full NER checkpoint.",
            file=sys.stderr,
        )
    if _count_triples("baseline_2") == 0 and _count_triples("baseline_1") > 0:
        print(
            "NOTE: baseline_2 empty but baseline_1 has triples — ontology mapper may lack "
            "CUI for both entity spans (expected for demo text vs BC5CDR mapper).",
            file=sys.stderr,
        )

    _write_results_json(out_path, doc)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
