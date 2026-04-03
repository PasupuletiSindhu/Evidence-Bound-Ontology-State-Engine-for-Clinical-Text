#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Default to GPU 13 unless caller overrides.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "13")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_span_ci(text: str, needle: str) -> tuple[int, int] | None:
    t = text or ""
    n = (needle or "").strip()
    if not t or not n:
        return None
    i = t.lower().find(n.lower())
    if i < 0:
        return None
    return i, i + len(n)


def _iter_fact_paraphrases(doc: dict) -> list[tuple[str, list[str]]]:
    """
    Accept either:
    - results/paraphrase_results.json style: {"sets":[{"fact","paraphrases",...}, ...]}
    - paraphrase_sets style: {"paraphrase_sets":[{"fact","paraphrases"}, ...]}
    """
    out: list[tuple[str, list[str]]] = []
    sets = doc.get("sets")
    if isinstance(sets, list) and sets:
        for s in sets:
            fact = str((s or {}).get("fact") or "").strip()
            paras = [str(x).strip() for x in ((s or {}).get("paraphrases") or []) if str(x).strip()]
            if fact and paras:
                out.append((fact, paras))
        if out:
            return out

    psets = doc.get("paraphrase_sets")
    if isinstance(psets, list) and psets:
        for s in psets:
            fact = str((s or {}).get("fact") or "").strip()
            paras = [str(x).strip() for x in ((s or {}).get("paraphrases") or []) if str(x).strip()]
            if fact and paras:
                out.append((fact, paras))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune relation model on full paraphrase pipeline data.",
    )
    p.add_argument(
        "--paraphrase_results",
        type=str,
        default="results/paraphrase_results.json",
        help="Path to pipeline paraphrase JSON (results/paraphrase_results.json).",
    )
    p.add_argument(
        "--base_rel_dir",
        type=str,
        default="baselines/out/relation",
        help="Existing relation checkpoint dir to start from.",
    )
    p.add_argument(
        "--out_rel_dir",
        type=str,
        default="baselines/out/relation_pipeline_finetuned",
        help="Output dir for fine-tuned relation checkpoint.",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev_ratio", type=float, default=0.2)
    p.add_argument("--max_sets", type=int, default=0, help="0 means use all sets.")
    p.add_argument(
        "--unrelated_disease",
        type=str,
        default="headache",
        help="Extra disease token appended to create NR negatives.",
    )
    args = p.parse_args()

    random.seed(int(args.seed))
    in_path = Path(args.paraphrase_results)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    sets = _iter_fact_paraphrases(doc)
    if not sets:
        raise SystemExit(f"No usable (fact, paraphrases) found in {in_path}")

    if int(args.max_sets) > 0:
        sets = sets[: int(args.max_sets)]

    from state_engine.relations import parse_fact_spans
    from baselines.models.bert_relation import BERTRelationModel, build_relation_examples

    injected = str(args.unrelated_disease).strip() or "headache"
    docs = []
    skipped = 0

    for fact, paraphrases in sets:
        parsed = parse_fact_spans(fact)
        if not parsed:
            skipped += 1
            continue
        subj, rel, obj = parsed
        for text0 in paraphrases:
            span_s = _find_span_ci(text0, subj)
            span_o = _find_span_ci(text0, obj)
            if not span_s or not span_o:
                continue
            text = f"{text0} Unrelated disease: {injected}."
            span_u = _find_span_ci(text, injected)
            if not span_u:
                continue
            entities = [
                {"start": span_s[0], "end": span_s[1], "text": text[span_s[0] : span_s[1]], "type": "Chemical"},
                {"start": span_o[0], "end": span_o[1], "text": text[span_o[0] : span_o[1]], "type": "Disease"},
                {"start": span_u[0], "end": span_u[1], "text": text[span_u[0] : span_u[1]], "type": "Disease"},
            ]
            relations = [{"head": 0, "tail": 1, "type": "CID"}]
            docs.append(
                {
                    "text": text,
                    "entities": entities,
                    "relations": relations,
                    "fact": fact,
                    "fact_relation": rel,
                }
            )

    if len(docs) < 20:
        raise SystemExit(
            f"Too few usable training docs ({len(docs)}). "
            "Likely subject/object substrings were not found in paraphrases."
        )

    rel_examples = build_relation_examples(
        docs,
        relation_key="type",
        add_negatives=True,
        max_negatives_ratio=1.0,
    )
    random.shuffle(rel_examples)

    dev_ratio = min(max(float(args.dev_ratio), 0.0), 0.5)
    dev_n = max(1, int(len(rel_examples) * dev_ratio))
    train_ex = rel_examples[:-dev_n] if len(rel_examples) > dev_n else rel_examples
    dev_ex = rel_examples[-dev_n:] if len(rel_examples) > dev_n else []

    print(
        "[pipeline_rel_finetune] "
        f"sets={len(sets)} skipped_facts={skipped} docs={len(docs)} "
        f"examples={len(rel_examples)} train={len(train_ex)} dev={len(dev_ex)}"
    )

    rel = BERTRelationModel(
        model_name="dmis-lab/biobert-base-cased-v1.1", use_binary_bce=True
    ).load(args.base_rel_dir)
    rel.use_binary_bce = True
    rel.label_list = ["NR", "CID"]
    rel.label2id = {"NR": 0, "CID": 1}
    rel.id2label = {0: "NR", 1: "CID"}
    rel._num_labels = 1

    out_dir = Path(args.out_rel_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel.train(
        train_ex,
        dev_ex if dev_ex else None,
        output_dir=str(out_dir),
        num_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
    )
    print(f"[pipeline_rel_finetune] wrote fine-tuned checkpoint to {out_dir}")


if __name__ == "__main__":
    main()

