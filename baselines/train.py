#!/usr/bin/env python3
"""
Train NER (MedMentions/BC5CDR) and relation model (BC5CDR). Backbone: Qwen 2.5 by default.
Uses a single GPU only; no DataParallel / DDP.

NER tuning (to improve F1 and extraction on sentences like "Aspirin may cause gastric bleeding..."):
  Option 1 — 5-label (BC5CDR): --no_collapse_to_entity → B-Chemical, I-Chemical, B-Disease, I-Disease, O (better span discrimination).
  Option 2 — Softer entity weight (3-label): --ner_entity_weight 3.0.
  Option 3 — More epochs + warmup: --ner_epochs 20 (default), --ner_warmup_ratio 0.1; best F1 checkpoint is loaded.
  Option 4 — CRF: --use_crf for BIO boundary consistency. With 5-label, class weighting is applied as auxiliary loss on emissions.
  Option 5 — Larger batch: --ner_batch_size 32 if GPU memory allows.

Example (BC5CDR, higher F1):
  python train.py --dataset bc5cdr --model_name dmis-lab/biobert-base-cased-v1.1 \\
    --no_collapse_to_entity --ner_entity_weight 3.0 --ner_epochs 20 --use_crf --ner_batch_size 32
"""
import os
# Use GPU 0 by default; override with CUDA_VISIBLE_DEVICES=0,1 python train.py ...
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "13")

import sys
from pathlib import Path
import torch

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import argparse
from loaders.medmentions_loader import load_medmentions
from loaders.bc5cdr_loader import load_bc5cdr
from loaders.sentence_split import (
    to_sentence_level_examples,
    sentence_level_entity_stats,
    first_example_with_entities,
)
from models.bert_ner import BERTNERModel, DEFAULT_NER_MODEL, debug_alignment
from models.bert_relation import BERTRelationModel, DEFAULT_REL_MODEL, build_relation_examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["medmentions", "bc5cdr"], required=True)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="./out")
    p.add_argument("--model_name", type=str, default=None, help="Backbone model (default: Qwen/Qwen2.5-0.5B). Use e.g. dmis-lab/biobert-base-cased-v1.1 for BioBERT.")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--ner_epochs", type=int, default=20, help="NER epochs. Best F1 checkpoint is loaded; try 15–20 for higher F1.")
    p.add_argument("--ner_lr", type=float, default=3e-5)
    p.add_argument("--ner_batch_size", type=int, default=16, help="Batch size for NER. Try 32 if GPU memory allows.")
    p.add_argument("--ner_warmup_ratio", type=float, default=0.1, help="Warmup over this fraction of steps (stabilizes F1).")
    p.add_argument("--rel_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_semantic_groups", action="store_true", help="MedMentions: map T-codes to ~15 UMLS groups (fewer labels).")
    p.add_argument("--no_class_weight", action="store_true", help="Disable inverse-frequency class weights for NER (default: class weighting on).")
    p.add_argument("--no_collapse_to_entity", action="store_true", help="Use full label set. BC5CDR: 5-label B-Chemical, I-Chemical, B-Disease, I-Disease, O (better span discrimination). Default is 3-label B-ENTITY, I-ENTITY, O.")
    p.add_argument("--collapse_to_entity", action="store_true", help="Explicitly use 3 labels (redundant with default).")
    p.add_argument("--no_relation_negatives", action="store_true", help="BC5CDR: do not add NR (negative) relation pairs; only use gold relations.")
    p.add_argument("--no_sentence_level", action="store_true", help="Disable sentence-level NER training (use full documents; entities may be truncated away).")
    p.add_argument("--use_crf", action="store_true", help="Add CRF layer for BIO span consistency (~+0.05–0.10 F1).")
    p.add_argument("--no_entity_only", action="store_true", dest="no_entity_only", help="Include sentences with no entities in NER training (default: train only on sentences with ≥1 entity).")
    p.add_argument("--ner_entity_weight", type=float, default=5.0, help="Weight for B/I entity tokens in 3-label mode (O=1.0). Try 3.0 for better inference stability.")
    p.add_argument("--skip_ner_training", action="store_true", help="Do not train NER; use pretrained model from Hub and save to out/ner (train relation only).")
    p.add_argument("--ner_pretrained_model", type=str, default="Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-LoRA-v1", help="HuggingFace NER model id when --skip_ner_training (BC5CDR chemical+disease). Requires: pip install peft")
    args = p.parse_args()
    entity_only = not args.no_entity_only

    # Default: 3-label NER (B-ENTITY, I-ENTITY, O) and class weighting for better F1 on both datasets.
    collapse_to_entity = not args.no_collapse_to_entity or args.collapse_to_entity
    class_weight = not args.no_class_weight

    data_dir = args.data_dir or str(_here / "data" / args.dataset)
    out_dir = Path(args.out_dir)
    out_ner = out_dir / "ner"
    out_rel = out_dir / "relation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "medmentions":
        train_data, dev_data, test_data = load_medmentions(data_dir, seed=args.seed)
        label_key = "semantic_type"
    else:
        train_data, dev_data, test_data = load_bc5cdr(data_dir, seed=args.seed)
        label_key = "type"

    # Sentence-level NER so entities are not lost to 512-token truncation (default: on).
    if not args.no_sentence_level:
        train_ner = to_sentence_level_examples(train_data)
        dev_ner = to_sentence_level_examples(dev_data) if dev_data else None
        print(f"[NER] Sentence-level: {len(train_data)} docs → {len(train_ner)} sentences (dev: {len(dev_data) or 0} → {len(dev_ner) or 0}).")
        # Span alignment sanity: did we keep entities? If most sentences have 0 entities, gold labels are wrong → near-zero F1.
        stats = sentence_level_entity_stats(train_data, train_ner)
        print(f"[NER] Entity span check: docs had {stats['doc_entities']} entities → sentences have {stats['sent_entities']} entities; "
              f"{stats['sentences_with_entities']} / {stats['total_sentences']} sentences contain ≥1 entity.")
        if stats["total_sentences"] > 0 and stats["sentences_with_entities"] == 0:
            print("  [WARNING] No sentence has any entity after splitting → gold labels are all O → F1 will be 0. Check sentence_split offset remapping.")
        elif stats["doc_entities"] > 0 and stats["sent_entities"] < stats["doc_entities"] * 0.5:
            print("  [WARNING] Many entities lost in sentence split (e.g. cross-sentence spans dropped). Consider checking sentence boundaries.")
        if entity_only:
            n_before = len(train_ner)
            train_ner = [s for s in train_ner if len(s.get("entities", [])) > 0]
            if dev_ner is not None:
                dev_ner = [s for s in dev_ner if len(s.get("entities", [])) > 0]
            print(f"[NER] Entity-only: {n_before} → {len(train_ner)} train sentences (dev: {len(dev_ner) or 0} with entities).")
    else:
        train_ner, dev_ner = train_data, dev_data
        if entity_only:
            n_before = len(train_ner)
            train_ner = [s for s in train_ner if len(s.get("entities", [])) > 0]
            dev_ner = [s for s in dev_ner if len(s.get("entities", [])) > 0] if dev_ner else None
            print(f"[NER] Entity-only: {n_before} → {len(train_ner)} train sentences.")

    # GPU: Trainer uses CUDA automatically when available.
    if torch.cuda.is_available():
        print("[Device] GPU:", torch.cuda.get_device_name(0), f"(cuda:{torch.cuda.current_device()})")
    else:
        print("[Device] CPU (training will be slow; set CUDA_VISIBLE_DEVICES if you have a GPU).")

    model_name = args.model_name or DEFAULT_NER_MODEL

    if args.skip_ner_training:
        # Use pretrained NER from HuggingFace. Do not save the LoRA/merged model (avoids tied-weights save error).
        # Write a pointer so load() and run_baseline will load from Hub when loading out/ner.
        print("[NER] Skipping NER training; using pretrained model from Hub:", args.ner_pretrained_model)
        ner = BERTNERModel().load_pretrained(args.ner_pretrained_model)
        out_ner.mkdir(parents=True, exist_ok=True)
        (out_ner / "ner_config.json").write_text(
            __import__("json").dumps({"from_hub": args.ner_pretrained_model})
        )
        print("NER pointer (from_hub) saved to", out_ner, "— model will be loaded from Hub when needed.")
    else:
        print("[Backbone]", model_name)
        ner = BERTNERModel(
            model_name=model_name,
            label_key=label_key,
            freeze_encoder=args.freeze_encoder,
            use_semantic_groups=args.use_semantic_groups,
            class_weight=class_weight,
            collapse_to_entity=collapse_to_entity,
            use_crf=args.use_crf,
        )
        ner.prepare_from_data(train_ner, dev_ner)
        # Debug: label mapping and alignment.
        if train_ner:
            first_with_entities = first_example_with_entities(train_ner) if not args.no_sentence_level else None
            debug_ex = first_with_entities if first_with_entities is not None else train_ner[0]
            debug_alignment(
                debug_ex,
                ner.tokenizer,
                ner.label2id,
                ner.id2label,
                label_key=label_key,
                semantic_group_map=ner.semantic_group_map,
                collapse_to_entity=ner.collapse_to_entity,
            )
            if first_with_entities is None and len(train_ner) > 0:
                print("[DEBUG] First example has 0 entities; gold labels may be all O. Check sentences_with_entities above.")
        ner.train(
            train_ner,
            dev_ner,
            output_dir=str(out_ner),
            num_epochs=args.ner_epochs,
            lr=args.ner_lr,
            batch_size=args.ner_batch_size,
            warmup_ratio=args.ner_warmup_ratio,
            entity_weight=args.ner_entity_weight if args.ner_entity_weight > 0 else None,
        )
        ner.load(str(out_ner))
        print("NER saved to", out_ner)

    if args.dataset == "bc5cdr":
        rel_train = build_relation_examples(
            train_data,
            add_negatives=not args.no_relation_negatives,
        )
        rel_dev = build_relation_examples(
            dev_data,
            add_negatives=not args.no_relation_negatives,
        ) if dev_data else None
        # Relation set size and balance (sanity check: tiny dev or all-one-class → metrics meaningless)
        from collections import Counter
        print("Relation train size:", len(rel_train))
        print("Relation dev size:", len(rel_dev) if rel_dev else 0)
        if rel_train:
            print("Train labels:", dict(Counter(x["label"] for x in rel_train)))
        if rel_dev:
            print("Dev labels:", dict(Counter(x["label"] for x in rel_dev)))
        if rel_dev and len(rel_dev) < 20:
            print("  [WARNING] Relation dev set has < 20 examples; eval accuracy/F1 may be statistically unstable (each mistake ≈ 6–7% swing).")
        if rel_train and len(set(x["label"] for x in rel_train)) < 2:
            print("  [WARNING] Relation train has only one label; add negative samples (e.g. build_relation_examples(..., add_negatives=True)).")
        if rel_train:
            rel_model_name = args.model_name or DEFAULT_REL_MODEL
            rel = BERTRelationModel(model_name=rel_model_name)
            rel.prepare_from_data(rel_train)
            rel.train(rel_train, rel_dev, output_dir=str(out_rel), num_epochs=args.rel_epochs)
            rel.load(str(out_rel))
            print("Relation model saved to", out_rel)
    print("Done.")


if __name__ == "__main__":
    main()
