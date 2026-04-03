#!/usr/bin/env python3
# Run one baseline (1, 2, or 3) on sample text. e.g. python run_baseline.py 1 --text "Aspirin causes bleeding."
import argparse
import sys
from pathlib import Path
from typing import Optional

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

SAMPLE_TEXT = "Aspirin may cause gastric bleeding in some patients."

# Pretrained BC5CDR NER (chemical + disease). Requires: pip install peft
DEFAULT_PRETRAINED_NER = "Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-LoRA-v1"


def _get_mapper_baseline2(mapper_file: Optional[str]):
    # mapper_file -> cui_map.txt -> BC5CDR -> empty
    from ontology.umls_mapper import UMLSMapper
    if mapper_file and Path(mapper_file).exists():
        print(f"Loaded CUI mapper from {mapper_file}")
        return UMLSMapper.from_file(mapper_file)
    if (_here / "data" / "cui_map.txt").exists():
        print("Loaded CUI mapper from data/cui_map.txt")
        return UMLSMapper.from_file(str(_here / "data" / "cui_map.txt"))
    try:
        from loaders.bc5cdr_loader import load_bc5cdr
        train_data, _, _ = load_bc5cdr(str(_here / "data" / "bc5cdr"), download_if_missing=False)
        if train_data:
            print("Built mapper from BC5CDR corpus (MeSH IDs)")
            return UMLSMapper.from_bc5cdr(train_data)
    except Exception:
        pass
    print("No mapper (data/cui_map.txt, --mapper_file, or BC5CDR data); entities as-is.")
    return UMLSMapper()


def _sentencize(text: str):
    import re
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]


def run_baseline_1(ner_dir: str, rel_dir: str, text: str, diagnose: bool = False, ner_model: Optional[str] = None) -> None:
    from models.bert_ner import BERTNERModel
    from models.bert_relation import BERTRelationModel
    from pipelines import StatelessNeuralExtraction

    print("[Baseline 1] Stateless Neural Extraction (NER + relation)")
    if ner_model:
        print(f"Loading pretrained NER from HuggingFace: {ner_model}")
        ner = BERTNERModel().load_pretrained(ner_model)
    elif Path(ner_dir).exists():
        print("Loading NER from checkpoint:", ner_dir)
        ner = BERTNERModel(model_name="dmis-lab/biobert-base-cased-v1.1").load(ner_dir)
    else:
        print(f"Loading pretrained NER from HuggingFace: {DEFAULT_PRETRAINED_NER}")
        ner = BERTNERModel().load_pretrained(DEFAULT_PRETRAINED_NER)
    print("Loading relation model...")
    rel = BERTRelationModel(model_name="dmis-lab/biobert-base-cased-v1.1").load(rel_dir)
    if not diagnose:
        pipe = StatelessNeuralExtraction(ner, rel)
        triples = pipe.process(text)
        print(f"Input: {text}")
        print(f"Triples ({len(triples)}): {triples}")
        return triples
    # show NER + relation predictions per sentence
    print(f"Input: {text}")
    sentences = _sentencize(text)
    all_triples = []
    for idx, sent in enumerate(sentences):
        if not sent:
            continue
        entities = ner.extract_entities(sent)
        entity_texts = [e.get("text", "") for e in entities]
        n = len(entities)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        relations = rel.extract_relations(sent, entities) if entities else []
        print(f"  Sentence {idx + 1}: {sent[:60]}{'...' if len(sent) > 60 else ''}")
        print(f"    NER entities ({len(entities)}): {entity_texts}")
        print(f"    Candidate pairs: {pairs}")
        print(f"    Relation predictions: {relations}")
        for r in relations:
            sub = entities[r["head"]].get("text", "")
            obj = entities[r["tail"]].get("text", "")
            if sub and obj:
                all_triples.append((sub, r["relation"], obj, "positive"))
    print(f"Triples ({len(all_triples)}): {all_triples}")
    return all_triples


def run_baseline_2(ner_dir: str, rel_dir: str, text: str, mapper_file: Optional[str], diagnose: bool = False, ner_model: Optional[str] = None) -> None:
    from models.bert_ner import BERTNERModel
    from models.bert_relation import BERTRelationModel
    from pipelines import OntologyNormalizedStateless
    from ontology.umls_mapper import UMLSMapper

    print("[Baseline 2] Ontology-Normalized Stateless (NER + relation + CUI)")
    if ner_model:
        print(f"Loading pretrained NER from HuggingFace: {ner_model}")
        ner = BERTNERModel().load_pretrained(ner_model)
    elif Path(ner_dir).exists():
        print("Loading NER from checkpoint:", ner_dir)
        ner = BERTNERModel(model_name="dmis-lab/biobert-base-cased-v1.1").load(ner_dir)
    else:
        print(f"Loading pretrained NER from HuggingFace: {DEFAULT_PRETRAINED_NER}")
        ner = BERTNERModel().load_pretrained(DEFAULT_PRETRAINED_NER)
    print("Loading relation model...")
    rel = BERTRelationModel(model_name="dmis-lab/biobert-base-cased-v1.1").load(rel_dir)
    mapper = _get_mapper_baseline2(mapper_file)
    if not diagnose:
        pipe = OntologyNormalizedStateless(ner, rel, mapper)
        triples = pipe.process(text)
        print(f"Input: {text}")
        print(f"Triples ({len(triples)}): {triples}")
        return triples
    print(f"Input: {text}")
    sentences = _sentencize(text)
    all_triples = []
    for idx, sent in enumerate(sentences):
        if not sent:
            continue
        entities = ner.extract_entities(sent)
        entity_texts = [e.get("text", "") for e in entities]
        n = len(entities)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        relations = rel.extract_relations(sent, entities) if entities else []
        print(f"  Sentence {idx + 1}: {sent[:60]}{'...' if len(sent) > 60 else ''}")
        print(f"    NER entities ({len(entities)}): {entity_texts}")
        print(f"    Candidate pairs: {pairs}")
        print(f"    Relation predictions: {relations}")
        cuis = [mapper.normalize(e.get("text", "")) for e in entities]
        for r in relations:
            sub_cui = cuis[r["head"]]
            obj_cui = cuis[r["tail"]]
            if sub_cui and obj_cui:
                all_triples.append((sub_cui, r["relation"], obj_cui, "positive"))
    print(f"Triples ({len(all_triples)}): {all_triples}")
    return all_triples


def run_baseline_3(text: str, qwen_model: str) -> None:
    from pipelines import SingleExtractorVariant

    print("[Baseline 3] Single-Extractor Variant (Qwen 2.5 prompt-based)")
    print("Loading Qwen 2.5 (prompt-based extraction)...")
    pipe = SingleExtractorVariant("qwen", qwen_model_name=qwen_model)
    triples = pipe.process(text)
    print(f"Input: {text}")
    print(f"Triples ({len(triples)}): {triples}")
    return triples


def main():
    p = argparse.ArgumentParser(description="Run baseline 1, 2, or 3 on sample text.")
    p.add_argument("baseline", type=int, nargs="?", choices=[1, 2, 3], help="Baseline to run (1, 2, or 3).")
    p.add_argument("--baseline", type=int, dest="baseline_opt", choices=[1, 2, 3], default=None, help="Same as positional.")
    p.add_argument("--text", type=str, default=SAMPLE_TEXT, help="Input sentence.")
    p.add_argument("--ner_dir", type=str, default=str(_here / "out" / "ner"), help="NER checkpoint dir; used only when it exists and --ner_model is not set.")
    p.add_argument("--ner_model", type=str, default=None, help=f"If set, always load this Hub NER model (overrides --ner_dir). Else default: {DEFAULT_PRETRAINED_NER} when --ner_dir missing.")
    p.add_argument("--rel_dir", type=str, default=str(_here / "out" / "relation"), help="Relation checkpoint (Baseline 1 & 2). Must train once.")
    p.add_argument("--mapper_file", type=str, default=None, help="CUI mapping file for Baseline 2 (entity_text\\tCUI).")
    p.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-0.5B", help="Qwen model for Baseline 3.")
    p.add_argument("--diagnose", action="store_true", help="Print NER entities, candidate pairs, and relation predictions (Baseline 1 & 2).")
    args = p.parse_args()

    baseline = args.baseline_opt or args.baseline
    if baseline is None:
        p.print_help()
        print("\nExample: python run_baseline.py 1   # then 2, then 3")
        sys.exit(0)

    if baseline == 1:
        if not Path(args.rel_dir).exists():
            print(f"Relation checkpoint not found: {args.rel_dir}")
            print("Train relation once: python train.py --dataset bc5cdr --model_name dmis-lab/biobert-base-cased-v1.1 --skip_ner_training --ner_pretrained_model Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-LoRA-v1")
            sys.exit(1)
        run_baseline_1(args.ner_dir, args.rel_dir, args.text, diagnose=args.diagnose, ner_model=args.ner_model)
    elif baseline == 2:
        if not Path(args.rel_dir).exists():
            print(f"Relation checkpoint not found: {args.rel_dir}")
            sys.exit(1)
        run_baseline_2(args.ner_dir, args.rel_dir, args.text, args.mapper_file, diagnose=args.diagnose, ner_model=args.ner_model)
    else:
        run_baseline_3(args.text, args.qwen_model)
    print("Done.")


if __name__ == "__main__":
    main()
