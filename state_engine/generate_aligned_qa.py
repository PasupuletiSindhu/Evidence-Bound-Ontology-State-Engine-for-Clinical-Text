#!/usr/bin/env python3
"""Build triple-consistent QA JSON aligned to canonical graph schema (entity + relation)."""

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from state_engine.ontology import OntologyAligner
from state_engine import relations as relation_utils
from state_engine.relations import parse_fact_spans

def _relation_aliases_from_learned_map() -> dict:
    aliases = {
        r: [r]
        for r in relation_utils.CANONICAL_RELATIONS
        if r not in ("unknown", "related_to")
    }
    for label, rel in relation_utils.REL_MAP.items():
        if rel in aliases:
            aliases[rel].append(label)
    for rel in list(aliases.keys()):
        aliases[rel] = sorted(set(aliases[rel]))
    return aliases


def _questions_for_fact(subj: str, rel: str, obj: str) -> tuple:
    """Returns (q_subject_slot, q_object_slot) natural-language templates filled."""
    subj_t = subj.strip()
    obj_t = obj.strip()
    if rel == "causes":
        return (
            f"What does {subj_t} cause?",
            f"What causes {obj_t}?",
        )
    if rel == "treats":
        return (
            f"What does {subj_t} treat?",
            f"What drug treats {obj_t}?",
        )
    if rel == "prevents":
        return (
            f"What does {subj_t} prevent?",
            f"What prevents {obj_t}?",
        )
    if rel == "reduces":
        return (
            f"What does {subj_t} reduce?",
            f"What reduces {obj_t}?",
        )
    if rel == "increases":
        return (
            f"What does {subj_t} increase?",
            f"What increases {obj_t}?",
        )
    if rel == "interacts_with":
        return (
            f"What does {subj_t} interact with?",
            f"What interacts with {obj_t}?",
        )
    if rel == "metabolized_by":
        return (
            f"Which organ metabolizes {subj_t}?",
            f"What drug is metabolized by the {obj_t}?",
        )
    return (
        f"What is linked to {subj_t} via {rel}?",
        f"What is linked to {obj_t} via {rel}?",
    )


def build_examples_from_facts(
    facts: list,
    aligner: OntologyAligner,
    qa_count: int,
    *,
    entity_mode: str = "text",
):
    examples = []
    for item in facts:
        fact = item.get("fact", "")
        parsed = parse_fact_spans(fact)
        if not parsed:
            continue
        raw_subj, rel, raw_obj = parsed
        if entity_mode == "canonical":
            subj = aligner.normalize_entity(raw_subj)
            obj = aligner.normalize_entity(raw_obj)
        else:
            subj = aligner.normalize_entity_light(raw_subj)
            obj = aligner.normalize_entity_light(raw_obj)
        q_sub, q_obj = _questions_for_fact(subj, rel, obj)
        examples.append(
            {
                "question": q_sub,
                "subject": subj,
                "relation": rel,
                "object": "?",
                "answer": obj,
            }
        )
        examples.append(
            {
                "question": q_obj,
                "subject": "?",
                "relation": rel,
                "object": obj,
                "answer": subj,
            }
        )
        if len(examples) >= qa_count:
            break
    return examples[:qa_count]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--paraphrase_sets",
        type=str,
        default="baselines/data/paraphrases/paraphrase_sets_50.json",
    )
    p.add_argument("--qa_count", type=int, default=100)
    p.add_argument(
        "--output",
        type=str,
        default="results/qa_aligned_100.json",
    )
    p.add_argument("--mapper_file", type=str, default=None)
    p.add_argument(
        "--qa_entity_mode",
        type=str,
        default="text",
        choices=["text", "canonical"],
        help="Entity representation in QA examples: text surfaces (recommended) or canonical IDs.",
    )
    p.add_argument(
        "--relation_map",
        type=str,
        default="results/relation_map.json",
        help="Learned label->relation mapping JSON.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    ppath = Path(args.paraphrase_sets)
    if not ppath.is_absolute():
        ppath = root / ppath
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    rel_map = Path(args.relation_map)
    if not rel_map.is_absolute():
        rel_map = root / rel_map

    data = json.loads(ppath.read_text(encoding="utf-8"))
    sets = data.get("paraphrase_sets") or data.get("sets") or []
    aligner = OntologyAligner.from_baselines_mapper(args.mapper_file)
    relation_utils.load_relation_map(str(rel_map))
    examples = build_examples_from_facts(
        sets,
        aligner,
        args.qa_count,
        entity_mode=str(args.qa_entity_mode),
    )
    rel_aliases = _relation_aliases_from_learned_map()
    payload = {
        "description": (
            f"{len(examples)} QA examples aligned to canonical triple schema "
            f"(entity_mode={args.qa_entity_mode})."
        ),
        "relation_aliases": rel_aliases,
        "examples": examples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(examples)} questions)")


if __name__ == "__main__":
    main()
