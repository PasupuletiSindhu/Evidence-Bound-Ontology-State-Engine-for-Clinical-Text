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

# Ensure repo root is importable so `state_engine` works when running from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_span_ci(text: str, needle: str) -> tuple[int, int] | None:
    """Case-insensitive substring span; returns (start,end) or None."""
    t = text or ""
    n = (needle or "").strip()
    if not t or not n:
        return None
    i = t.lower().find(n.lower())
    if i < 0:
        return None
    return i, i + len(n)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune BC5CDR relation checkpoint on demo paraphrases (small targeted set).",
    )
    p.add_argument(
        "--paraphrase_sets",
        type=str,
        default="Demo/data/paraphrase_sets_demo.json",
        help="Path to demo paraphrase sets JSON.",
    )
    p.add_argument(
        "--base_rel_dir",
        type=str,
        default="baselines/out/relation",
        help="Existing trained relation checkpoint dir to start from.",
    )
    p.add_argument(
        "--out_rel_dir",
        type=str,
        default="baselines/out/relation_demo_finetuned",
        help="Output dir for fine-tuned checkpoint.",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--unrelated_disease",
        type=str,
        default="headache",
        help="Extra disease token appended to create NR negatives.",
    )
    args = p.parse_args()

    random.seed(int(args.seed))
    sets_path = Path(args.paraphrase_sets)
    doc = json.loads(sets_path.read_text(encoding="utf-8"))
    psets = doc.get("paraphrase_sets") or []
    if not psets:
        raise SystemExit(f"No paraphrase_sets found in {sets_path}")

    # We only need chemical–disease CID behavior for Baselines 1/2.
    # Build pseudo-docs with explicit entities + one positive relation per paraphrase,
    # then auto-add one negative (NR) by injecting an unrelated disease mention.
    from state_engine.relations import parse_fact_spans  # local module
    from baselines.models.bert_relation import build_relation_examples, BERTRelationModel

    docs = []
    injected = str(args.unrelated_disease).strip()
    if not injected:
        injected = "headache"

    for s in psets:
        fact = str(s.get("fact") or "").strip()
        parsed = parse_fact_spans(fact)
        if not parsed:
            continue
        subj, rel, obj = parsed
        # For this fine-tune we map all positive demo relations to CID (binary).
        # The goal is: "if chemical + disease appear with an asserted relation, predict CID".
        paraphrases = [str(x).strip() for x in (s.get("paraphrases") or []) if str(x).strip()]
        for text0 in paraphrases:
            span_s = _find_span_ci(text0, subj)
            span_o = _find_span_ci(text0, obj)
            if not span_s or not span_o:
                continue
            # Append unrelated disease mention to create at least one NR candidate pair.
            suffix = f" Unrelated disease: {injected}."
            text = text0 + suffix
            span_u = _find_span_ci(text, injected)
            if not span_u:
                continue
            entities = [
                {"start": span_s[0], "end": span_s[1], "text": text[span_s[0] : span_s[1]], "type": "Chemical"},
                {"start": span_o[0], "end": span_o[1], "text": text[span_o[0] : span_o[1]], "type": "Disease"},
                {"start": span_u[0], "end": span_u[1], "text": text[span_u[0] : span_u[1]], "type": "Disease"},
            ]
            relations = [{"head": 0, "tail": 1, "type": "CID"}]
            docs.append({"text": text, "entities": entities, "relations": relations, "fact": fact, "fact_relation": rel})

    if len(docs) < 10:
        raise SystemExit(
            f"Not enough usable paraphrases with substring spans (built {len(docs)} docs). "
            "Check that subjects/objects appear literally in paraphrases."
        )

    rel_examples = build_relation_examples(
        docs,
        relation_key="type",
        add_negatives=True,
        max_negatives_ratio=1.0,
    )
    random.shuffle(rel_examples)
    split = max(1, int(0.8 * len(rel_examples)))
    train_ex = rel_examples[:split]
    dev_ex = rel_examples[split:]
    print(f"[demo_rel_finetune] docs={len(docs)} relation_examples={len(rel_examples)} train={len(train_ex)} dev={len(dev_ex)}")

    # Load existing checkpoint and fine-tune in binary BCE mode with explicit NR/CID ids.
    rel = BERTRelationModel(model_name="dmis-lab/biobert-base-cased-v1.1", use_binary_bce=True).load(args.base_rel_dir)
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
    print(f"[demo_rel_finetune] wrote fine-tuned relation checkpoint to {out_dir}")


if __name__ == "__main__":
    main()

