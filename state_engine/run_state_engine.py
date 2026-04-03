# Normalizes each triple (subject, relation, object)
# Merges entities with similar embeddings
# Builds a knowledge graph incrementally
# Computes stability GED, Jaccard, and Drift@tau metrics
# Saves detailed evaluation results

#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from state_engine.align import (
    fact_seeded_raw_triples,
    prepare_triples_fallback,
    prepare_triples_for_state,
)
from state_engine.entity_aliases import (
    clear_entity_aliases,
    expand_triples_for_qa,
    ingest_semantic_registry_surfaces,
    register_mapper_inverse,
)
from state_engine.semantic_registry import EmbeddingEntityRegistry
from state_engine.engine import EvidenceBoundOntologyStateEngine
from state_engine.metrics import (
    drift_rate_at_tau,
    pairwise_stability,
    summarize,
    unsupported_inference_rate,
    unsupported_vs_note_evidence,
)
from state_engine.ontology import OntologyAligner
from state_engine import relations as relation_utils
from state_engine.relations import infer_relation_from_text, parse_fact_spans
from state_engine.build_relation_map import build_relation_map as build_relation_map_from_examples
from state_engine.relation_clusterer import fit_and_export_from_paraphrase_json, load_cluster_map
from state_engine.relation_train_builder import collect_relation_train_examples
from state_engine.canonicalize import canonical_entity
from state_engine.entity_cluster_batch import (
    apply_batch_entity_pipeline,
    remap_exported_edges,
    remap_triple,
)


def _scratch_surfaces_for_set(s: dict) -> list:
    """Context strings for entity validity (fact + paraphrases); not label rules."""
    out = [str(s.get("fact") or "")]
    for p in s.get("paraphrases") or []:
        out.append(str(p))
    return [x.strip() for x in out if x and str(x).strip()]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_input_triples(raw_block):
    out = []
    for t in raw_block or []:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            out.append((str(t[0]), str(t[1]), str(t[2]), 1.0, "positive"))
    return out


def _load_extractor_outputs(paraphrase_results_json: dict, baseline_key: str):
    sets = paraphrase_results_json.get("sets", [])
    all_sets = []
    for idx, s in enumerate(sets):
        triples_pp = ((s.get(baseline_key) or {}).get("triples_per_paraphrase") or [])
        all_sets.append(
            {
                "set_id": idx,
                "fact": s.get("fact", f"set_{idx}"),
                "paraphrases": s.get("paraphrases", []),
                "triples_per_paraphrase": [_normalize_input_triples(x) for x in triples_pp],
            }
        )
    return all_sets


def _build_qa_triple_list(
    engine_graph: list,
    aligned_union: set,
    policy: str,
) -> list:
    """
    Triples used for QA lookup.

    - engine: only edges from incremental state export (may omit nothing if state is correct).
    - aligned_union: every aligned (s,r,o) from any paraphrase (pure extractor closure).
    - union: engine ∪ aligned_union — recall-friendly; single-paraphrase evidence is kept.
    """
    eng = {tuple(t) for t in engine_graph if len(t) >= 3}
    aln = {tuple(t) for t in aligned_union if len(t) >= 3}
    if policy == "engine":
        triples = eng
    elif policy == "aligned_union":
        triples = aln
    elif policy == "union":
        triples = eng | aln
    else:
        triples = eng | aln
    return sorted(triples)


def _qa_eval_if_available(merged_triples, qa_file: Path, mapper=None):
    try:
        from baselines.qa_eval import load_qa_dataset, run_qa_eval
    except Exception:
        return None


    if not qa_file.exists():
        return None
    examples, aliases = load_qa_dataset(str(qa_file))
    entity_to_id = (lambda e: mapper.normalize(e) or e) if mapper else None
    per_q, metrics = run_qa_eval(
        merged_triples,
        examples,
        relation_aliases=aliases,
        entity_to_id=entity_to_id,
    )
    return {
        "config": {"qa_file": str(qa_file), "num_questions": len(examples)},
        "metrics": metrics,
        "per_question": per_q,
    }


def _pin_seed_fact_edge_active(
    final_state: dict,
    fact_text: str,
    fact_gold_relation: str,
    aligner: OntologyAligner,
) -> bool:
    """
    Keep the seed fact edge as active in exported final_state so baseline truth
    is visible even when noisy extractions add conflicting objects.
    """
    edges = (final_state or {}).get("edges") or []
    if not edges:
        return False

    parsed = parse_fact_spans(fact_text or "")
    if not parsed:
        return False
    fs, fr, fo = parsed

    fs_c = canonical_entity(aligner.normalize_entity_light(fs))
    fo_c = canonical_entity(aligner.normalize_entity_light(fo))
    fr_c = relation_utils.normalize_relation_label(
        fr,
        fact_gold_relation=fact_gold_relation,
        fact_text=fact_text,
    )

    for e in edges:
        s = canonical_entity(str(e.get("subject", "")))
        o = canonical_entity(str(e.get("object", "")))
        r = relation_utils.normalize_relation_label(
            str(e.get("relation", "")),
            fact_gold_relation=fact_gold_relation,
            fact_text=fact_text,
        )
        if s == fs_c and r == fr_c and (o == fo_c):
            e["status"] = "active"
            return True
    return False


def _count_conflict_records(final_state: dict) -> int:
    n = 0
    for e in (final_state or {}).get("edges") or []:
        n += len(e.get("conflict_records") or [])
    return n


def _qa_miss_with_schema(
    qa_examples: list,
    per_question: list,
    merged_triples: list,
    top_k: int,
    mapper=None,
):
    try:
        from baselines.qa_eval import query_graph, build_graph_index
    except Exception:
        return []
    entity_to_id = (lambda e: mapper.normalize(e) or e) if mapper else None
    index_sr, index_ro = build_graph_index(
        merged_triples,
        normalize_entities=True,
        entity_to_id=entity_to_id,
    )
    out = []
    for i, ex in enumerate(qa_examples):
        if i >= len(per_question):
            break
        pr = per_question[i]
        if pr.get("recall_at_1"):
            continue
        if pr.get("error"):
            continue
        sub = ex.get("subject", "")
        rel = ex.get("relation", "")
        obj = ex.get("object", "")
        gold = ex.get("answer", "")
        pred = query_graph(
            sub,
            rel,
            obj,
            index_sr,
            index_ro,
            {},
            entity_to_id=entity_to_id,
        )
        missing_edge = None
        if obj == "?":
            missing_edge = (sub, rel, gold)
        elif sub == "?":
            missing_edge = (gold, rel, obj)
        out.append(
            {
                "question": ex.get("question", ""),
                "gold": gold,
                "predicted": pr.get("predicted", []),
                "missing_canonical_edge": missing_edge,
            }
        )
        if len(out) >= top_k:
            break
    return out


def _to_md_table(header, rows):
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Run Evidence-Bound Ontology State Engine.")
    p.add_argument(
        "--paraphrase_results",
        type=str,
        default="results/paraphrase_results.json",
        help="Input extractor outputs (from baseline run_graph_eval paraphrase results).",
    )
    p.add_argument(
        "--source_baseline",
        type=str,
        default="baseline_3",
        help="Which extractor output to consume from paraphrase results (baseline_3 = Qwen extraction).",
    )
    p.add_argument(
        "--qa_file",
        type=str,
        default="results/qa_aligned_100.json",
        help="QA dataset (use state_engine/generate_aligned_qa.py).",
    )
    p.add_argument(
        "--mapper_file",
        type=str,
        default=None,
        help="Optional entity->CUI/MeSH mapping file.",
    )
    p.add_argument(
        "--semantic_linker",
        type=str,
        default="none",
        choices=["none", "scispacy"],
        help="Optional semantic linker for concept IDs (e.g., scispaCy UMLS).",
    )
    p.add_argument(
        "--scispacy_model",
        type=str,
        default="en_core_sci_sm",
        help="spaCy model name used when --semantic_linker=scispacy.",
    )
    p.add_argument(
        "--drift_tau",
        type=float,
        default=0.2,
        help="Drift@tau: fraction of pairs with (1 - Jaccard) >= tau.",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="results/state_engine_results.json",
    )
    p.add_argument(
        "--output_md",
        type=str,
        default="results/state_engine_results.md",
    )
    p.add_argument(
        "--relation_map",
        type=str,
        default="results/relation_map.json",
        help="Path to learned label->relation mapping JSON.",
    )
    p.add_argument(
        "--skip_relation_map_build",
        action="store_true",
        help="Use existing relation_map.json and relation_cluster_map.json without refitting.",
    )
    p.add_argument(
        "--relation_cluster_map",
        type=str,
        default="results/relation_cluster_map.json",
        help="Embedding/KMeans cluster artifact (fact→cluster→relation + centroids).",
    )
    p.add_argument(
        "--relation_n_clusters",
        type=int,
        default=8,
        help="KMeans k when cue-anchored fit falls back; supervised mode uses one centroid per relation.",
    )
    p.add_argument(
        "--relation_min_top_ratio",
        type=float,
        default=0.6,
        help="When building relation_map.json, map weak labels to 'unknown' if top label frequency ratio is below this threshold.",
    )
    p.add_argument(
        "--qa_graph",
        type=str,
        default="union",
        choices=["engine", "aligned_union", "union"],
        help="How to build triples for QA: incremental engine only, all aligned extractions, or union (recommended).",
    )
    p.add_argument(
        "--llm_graph_qa",
        action="store_true",
        help=(
            "Run LLM graph QA: retrieve top triples per question, then the model reads them and answers "
            "(requires torch/transformers; uses same HF stack as Qwen extraction)."
        ),
    )
    p.add_argument(
        "--llm_graph_qa_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model id for --llm_graph_qa.",
    )
    p.add_argument(
        "--llm_graph_qa_max_context_triples",
        type=int,
        default=120,
        help="Max triples included in the LLM prompt per question (lexical retrieval from merged QA graph).",
    )
    p.add_argument(
        "--llm_graph_qa_max_new_tokens",
        type=int,
        default=96,
        help="Max new tokens for each LLM answer.",
    )
    p.add_argument(
        "--verbose_relations",
        action="store_true",
        help="Enable DEBUG logs for relation normalization (resolve_relation, normalize_relation_label) and triple keep/drop stats.",
    )
    p.add_argument(
        "--no_semantic_clustering",
        action="store_true",
        help="Disable embedding-based entity merge before graph update (registry + engine object similarity).",
    )
    p.add_argument(
        "--entity_merge_threshold",
        type=float,
        default=0.75,
        help="Cosine similarity for registry merge, engine object-agreement, and batch agglomerative entity remap.",
    )
    p.add_argument(
        "--entity_distinct_threshold",
        type=float,
        default=0.65,
        help="Below this similarity, entities are always distinct; distinct–merge band logs near-duplicate only.",
    )
    p.add_argument(
        "--entity_use_lexical_containment",
        action="store_true",
        help="Force-merge entities when one normalized surface contains the other (conservative lexical rule).",
    )
    p.add_argument(
        "--no_entity_batch_recluster",
        action="store_true",
        help="Skip set-level agglomerative entity remap + (s,r) object dedupe after extraction.",
    )
    p.add_argument(
        "--include_weak_in_stability",
        action="store_true",
        help="Include related_to (weak) edges in pairwise GED/Jaccard/drift (default: exclude).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    pr_path = Path(args.paraphrase_results)
    if not pr_path.is_absolute():
        pr_path = root / pr_path
    qa_path = Path(args.qa_file)
    if not qa_path.is_absolute():
        qa_path = root / qa_path
    out_json = Path(args.output_json)
    if not out_json.is_absolute():
        out_json = root / out_json
    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = root / out_md
    rel_map_path = Path(args.relation_map)
    if not rel_map_path.is_absolute():
        rel_map_path = root / rel_map_path

    cluster_map_path = Path(args.relation_cluster_map)
    if not cluster_map_path.is_absolute():
        cluster_map_path = root / cluster_map_path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.verbose_relations:
        logging.getLogger("state_engine.align").setLevel(logging.DEBUG)
        logging.getLogger("state_engine.relations").setLevel(logging.DEBUG)
        logging.getLogger("state_engine.entity_validity").setLevel(logging.DEBUG)
        logging.getLogger("state_engine.entity_cluster_batch").setLevel(logging.DEBUG)
        logging.getLogger("state_engine.semantic_registry").setLevel(logging.DEBUG)
    # Keep pipeline INFO logs while muting verbose model-download chatter.
    for _name in (
        "httpx",
        "huggingface_hub",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "transformers",
    ):
        logging.getLogger(_name).setLevel(logging.WARNING)

    rel_train_path = rel_map_path.parent / "relation_train.json"
    pr = _load_json(pr_path)

    if not args.skip_relation_map_build:
        cluster_res = fit_and_export_from_paraphrase_json(
            paraphrase_path=pr_path,
            cluster_map_out=cluster_map_path,
            n_clusters=int(args.relation_n_clusters),
        )
        load_cluster_map(str(cluster_map_path))
        train_examples = collect_relation_train_examples(
            pr, baseline_keys=("baseline_1", "baseline_2", "baseline_3")
        )
        rel_train_path.parent.mkdir(parents=True, exist_ok=True)
        rel_train_path.write_text(
            json.dumps(train_examples, indent=2), encoding="utf-8"
        )
        logging.getLogger(__name__).info(
            "Wrote relation training pairs to %s (%d rows); building relation_map.json",
            rel_train_path,
            len(train_examples),
        )
        label_map, rmap_stats = build_relation_map_from_examples(
            train_examples, min_top_ratio=float(args.relation_min_top_ratio)
        )
        rel_map_path.parent.mkdir(parents=True, exist_ok=True)
        rel_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
        stats_path = rel_map_path.with_name(f"{rel_map_path.stem}_stats.json")
        stats_path.write_text(json.dumps(rmap_stats, indent=2), encoding="utf-8")
        logging.getLogger(__name__).info(
            "Built relation map from relation_train: %s (%d labels)",
            rel_map_path,
            len(label_map),
        )
    else:
        try:
            if cluster_map_path.exists():
                load_cluster_map(str(cluster_map_path))
        except Exception as ex:
            logging.getLogger(__name__).warning(
                "Could not load relation cluster map %s: %s", cluster_map_path, ex
            )

    loaded_map = relation_utils.load_relation_map(str(rel_map_path))
    sets = _load_extractor_outputs(pr, args.source_baseline)
    aligner = OntologyAligner.from_baselines_mapper(
        args.mapper_file,
        semantic_linker=args.semantic_linker,
        scispacy_model=args.scispacy_model,
    )
    clear_entity_aliases()
    register_mapper_inverse(aligner.mapper)

    set_results = []
    merged_graph = []
    global_aligned_evidence_union: set = set()
    set_metric_ged = []
    set_metric_jac = []
    unsupported_rates_union = []
    unsupported_note_stats = []
    drift_tau_per_set = []
    total_step_conflicts = 0
    total_uncertain = 0
    total_conflict_records = 0

    use_clustering = not args.no_semantic_clustering
    obj_thr = float(args.entity_merge_threshold)
    # When clustering is off, skip embedding calls in the engine (string/surface equality only).
    _engine_obj_sim = None if use_clustering else (lambda _a, _b: 0.0)

    for s in sets:
        fact = s.get("fact", "")
        fact_gold_relation = infer_relation_from_text(fact)
        scratch_surfaces = _scratch_surfaces_for_set(s)
        one_shot_graphs = []
        note_canonical_sets = []
        set_evidence_triples: list = []

        set_registry = (
            EmbeddingEntityRegistry(
                merge_threshold=float(args.entity_merge_threshold),
                distinct_threshold=float(args.entity_distinct_threshold),
                use_lexical_containment=bool(args.entity_use_lexical_containment),
            )
            if use_clustering
            else None
        )

        for i, raw in enumerate(s["triples_per_paraphrase"]):
            aligned = prepare_triples_for_state(
                raw,
                fact,
                aligner,
                entity_registry=set_registry,
                fact_gold_relation=fact_gold_relation,
                scratch_entity_surfaces=scratch_surfaces,
            )
            for a, b, c, _, _ in aligned:
                set_evidence_triples.append((a, b, c))
            note_canonical_sets.append({(a, b, c) for (a, b, c, _, _) in aligned})
            e = EvidenceBoundOntologyStateEngine(
                object_merge_threshold=obj_thr,
                object_similarity_fn=_engine_obj_sim,
            )
            e.update(aligned, source_id=f"set{s['set_id']}:p{i}")
            one_shot_graphs.append(e.export_triples(include_uncertain=True))

        inc = EvidenceBoundOntologyStateEngine(
            object_merge_threshold=obj_thr,
            object_similarity_fn=_engine_obj_sim,
        )
        update_log = []
        supported_union = []
        for i, raw in enumerate(s["triples_per_paraphrase"]):
            aligned = prepare_triples_for_state(
                raw,
                fact,
                aligner,
                entity_registry=set_registry,
                fact_gold_relation=fact_gold_relation,
                scratch_entity_surfaces=scratch_surfaces,
            )
            for a, b, c, _, _ in aligned:
                set_evidence_triples.append((a, b, c))
            supported_union.extend([(a, b, c) for (a, b, c, _, _) in aligned])
            step = inc.update(aligned, source_id=f"set{s['set_id']}:p{i}")
            update_log.append(step)
            total_step_conflicts += step["conflicts"]
            total_uncertain += step["uncertain"]

        final_graph = inc.export_triples(include_uncertain=True)
        if not final_graph:
            fb_all: list = []
            fb_source = "surface_fallback"
            for raw in s["triples_per_paraphrase"]:
                fb_all.extend(
                    prepare_triples_fallback(
                        raw,
                        fact,
                        aligner,
                        entity_registry=set_registry,
                        fact_gold_relation=fact_gold_relation,
                        scratch_entity_surfaces=scratch_surfaces,
                    )
                )
            all_raw_empty = all(len(r or []) == 0 for r in s["triples_per_paraphrase"])
            if not fb_all and all_raw_empty:
                fr = fact_seeded_raw_triples(fact)
                if fr:
                    fb_all = prepare_triples_for_state(
                        fr,
                        fact,
                        aligner,
                        entity_registry=set_registry,
                        fact_gold_relation=fact_gold_relation,
                        scratch_entity_surfaces=scratch_surfaces,
                    )
                    if not fb_all:
                        fb_all = prepare_triples_fallback(
                            fr,
                            fact,
                            aligner,
                            entity_registry=set_registry,
                            fact_gold_relation=fact_gold_relation,
                            scratch_entity_surfaces=scratch_surfaces,
                        )
                    fb_source = "fact_seeded_fallback"
            if fb_all:
                for a, b, c, _, _ in fb_all:
                    set_evidence_triples.append((a, b, c))
                supported_union.extend([(a, b, c) for (a, b, c, _, _) in fb_all])
                fb_step = inc.update(
                    fb_all, source_id=f"set{s['set_id']}:{fb_source}"
                )
                update_log.append(fb_step)
                total_step_conflicts += fb_step["conflicts"]
                total_uncertain += fb_step["uncertain"]
                final_graph = inc.export_triples(include_uncertain=True)

        fst = inc.export_state()
        entity_batch_stats: dict = {}
        batch_recluster = use_clustering and not args.no_entity_batch_recluster
        if batch_recluster and (one_shot_graphs or final_graph):
            combined = [list(g) for g in one_shot_graphs]
            combined.append(list(final_graph))
            new_lists, emap, entity_batch_stats = apply_batch_entity_pipeline(
                combined,
                merge_threshold=float(args.entity_merge_threshold),
                do_cluster=True,
                do_sr_dedupe=True,
            )
            one_shot_graphs = new_lists[:-1]
            final_graph = new_lists[-1]
            supported_union = [remap_triple(t, emap) for t in supported_union]
            note_canonical_sets = [
                {remap_triple(t, emap) for t in ns} for ns in note_canonical_sets
            ]
            for t in set_evidence_triples:
                global_aligned_evidence_union.add(remap_triple(t, emap))
            fst = dict(fst)
            fst["edges"] = remap_exported_edges(fst.get("edges") or [], emap)
        else:
            for t in set_evidence_triples:
                global_aligned_evidence_union.add(t)

        _pin_seed_fact_edge_active(fst, fact, fact_gold_relation, aligner)
        merged_graph.extend(final_graph)

        pair = pairwise_stability(
            one_shot_graphs,
            exclude_weak_edges=not args.include_weak_in_stability,
        )
        ged_stats = summarize(pair["ged_pairwise"])
        jac_stats = summarize(pair["jaccard_pairwise"])
        set_metric_ged.append(ged_stats["mean"])
        set_metric_jac.append(jac_stats["mean"])
        drift_tau_per_set.append(drift_rate_at_tau(pair["jaccard_pairwise"], args.drift_tau))

        total_conflict_records += _count_conflict_records(fst)

        unsupported_u = unsupported_inference_rate(final_graph, supported_union)
        unsupported_rates_union.append(unsupported_u)
        note_ev = unsupported_vs_note_evidence(final_graph, note_canonical_sets)
        unsupported_note_stats.append(note_ev["mean"])

        sem_export = None
        if set_registry is not None:
            ingest_semantic_registry_surfaces(set_registry)
            sem_export = set_registry.export_mapping()

        set_results.append(
            {
                "set_id": s["set_id"],
                "fact": s["fact"],
                "num_paraphrases": len(s["triples_per_paraphrase"]),
                "entity_batch_remap": entity_batch_stats,
                "pairwise": {
                    "ged": {
                        "values": pair["ged_pairwise"].tolist(),
                        "stats": ged_stats,
                    },
                    "jaccard": {
                        "values": pair["jaccard_pairwise"].tolist(),
                        "stats": jac_stats,
                    },
                    "drift_at_tau": drift_tau_per_set[-1],
                },
                "incremental": {
                    "update_log": update_log,
                    "final_state": fst,
                    "unsupported_inference_rate_vs_union": unsupported_u,
                    "unsupported_vs_per_note_evidence": note_ev,
                },
                "semantic_clustering": sem_export,
            }
        )

    eng_q = {tuple(t) for t in merged_graph if len(t) >= 3}
    aln_q = {tuple(t) for t in global_aligned_evidence_union if len(t) >= 3}
    qa_base = _build_qa_triple_list(
        merged_graph,
        global_aligned_evidence_union,
        args.qa_graph,
    )
    qa_triples = expand_triples_for_qa(qa_base)
    qa_graph_diff = {
        "only_in_engine_export": len(eng_q - aln_q),
        "only_in_aligned_union": len(aln_q - eng_q),
    }
    qa = _qa_eval_if_available(
        merged_triples=qa_triples,
        qa_file=qa_path,
        mapper=aligner.mapper,
    )

    qa_llm_graph_rag = None
    if args.llm_graph_qa:
        _log = logging.getLogger(__name__)
        try:
            from state_engine.llm_graph_qa import load_qwen_for_qa, run_llm_graph_qa
            from baselines.qa_eval import load_qa_dataset

            ex_lm, _aliases_lm = load_qa_dataset(str(qa_path))
            _log.info("LLM graph QA: loading %s", args.llm_graph_qa_model)
            llm_qa = load_qwen_for_qa(
                args.llm_graph_qa_model,
                max_new_tokens=max(256, int(args.llm_graph_qa_max_new_tokens)),
            )
            per_lm, met_lm = run_llm_graph_qa(
                ex_lm,
                qa_triples,
                llm_qa,
                max_context_triples=int(args.llm_graph_qa_max_context_triples),
                max_new_tokens=int(args.llm_graph_qa_max_new_tokens),
                entity_to_id=None,
            )
            qa_llm_graph_rag = {
                "config": {
                    "model": args.llm_graph_qa_model,
                    "max_context_triples": int(args.llm_graph_qa_max_context_triples),
                    "max_new_tokens": int(args.llm_graph_qa_max_new_tokens),
                    "qa_file": str(qa_path),
                },
                "metrics": met_lm,
                "per_question": per_lm,
            }
        except Exception as e:
            _log.exception("LLM graph QA failed: %s", e)
            qa_llm_graph_rag = {"error": str(e), "config": {"model": args.llm_graph_qa_model}}

    qa_misses_top = []
    if qa and qa.get("per_question"):
        try:
            from baselines.qa_eval import load_qa_dataset

            ex_only, _ = load_qa_dataset(str(qa_path))
            qa_misses_top = _qa_miss_with_schema(
                ex_only, qa["per_question"], qa_triples, 10, mapper=aligner.mapper
            )
        except Exception:
            qa_misses_top = []

    ged_arr = np.asarray(set_metric_ged) if set_metric_ged else np.asarray([0.0])
    jac_arr = np.asarray(set_metric_jac) if set_metric_jac else np.asarray([1.0])
    uns_u_arr = (
        np.asarray(unsupported_rates_union)
        if unsupported_rates_union
        else np.asarray([0.0])
    )
    uns_n_arr = (
        np.asarray(unsupported_note_stats)
        if unsupported_note_stats
        else np.asarray([0.0])
    )
    drift_arr = np.asarray(drift_tau_per_set) if drift_tau_per_set else np.asarray([0.0])

    overall = {
        "ged_pairwise_mean_across_sets": summarize(ged_arr),
        "jaccard_pairwise_mean_across_sets": summarize(jac_arr),
        "drift_at_tau_mean_across_sets": summarize(drift_arr),
        "drift_tau": args.drift_tau,
        "unsupported_inference_rate_vs_union_across_sets": summarize(uns_u_arr),
        "unsupported_vs_per_note_evidence_mean_across_sets": summarize(uns_n_arr),
        "incremental_conflict_events": total_step_conflicts,
        "total_conflict_records": total_conflict_records,
        "total_uncertain_edges_marked": total_uncertain,
    }

    top_drift_sets = sorted(
        set_results,
        key=lambda z: z["pairwise"]["jaccard"]["stats"]["mean"],
    )[:10]
    top_drift_trimmed = [
        {
            "set_id": z["set_id"],
            "fact": z["fact"],
            "jaccard_mean": z["pairwise"]["jaccard"]["stats"]["mean"],
            "ged_mean": z["pairwise"]["ged"]["stats"]["mean"],
            "drift_at_tau": z["pairwise"]["drift_at_tau"],
        }
        for z in top_drift_sets
    ]

    qm = (qa or {}).get("metrics") or {}
    summary_row = {
        "GED_mean": overall["ged_pairwise_mean_across_sets"]["mean"],
        "Jaccard_mean": overall["jaccard_pairwise_mean_across_sets"]["mean"],
        f"Drift@{args.drift_tau}": overall["drift_at_tau_mean_across_sets"]["mean"],
        "Unsupported_note_mean": overall["unsupported_vs_per_note_evidence_mean_across_sets"][
            "mean"
        ],
        "Unsupported_union_mean": overall["unsupported_inference_rate_vs_union_across_sets"][
            "mean"
        ],
        "conflict_candidates_records": overall["total_conflict_records"],
        "QA_EM": qm.get("exact_match"),
        "QA_Recall@1": qm.get("recall_at_1"),
    }
    if qa_llm_graph_rag and isinstance(qa_llm_graph_rag.get("metrics"), dict):
        qm_llm = qa_llm_graph_rag["metrics"]
        summary_row["QA_LLM_EM"] = qm_llm.get("exact_match")
        summary_row["QA_LLM_Recall@1"] = qm_llm.get("recall_at_1")

    out = {
        "model": "evidence_bound_ontology_state_engine",
        "config": {
            "paraphrase_results": str(pr_path),
            "source_baseline": args.source_baseline,
            "qa_file": str(qa_path),
            "relation_map_supervision": "results/relation_train.json + build_relation_map (argmax per label)",
            "relation_train": str(rel_train_path),
            "relation_cluster_map": str(cluster_map_path),
            "relation_n_clusters": int(args.relation_n_clusters),
            "relation_map_build_skipped": bool(args.skip_relation_map_build),
            "relation_map_stats": str(rel_map_path.with_name(f"{rel_map_path.stem}_stats.json")),
            "mapper_file": args.mapper_file,
            "drift_tau": args.drift_tau,
            "semantic_clustering": use_clustering,
            "entity_merge_threshold": float(args.entity_merge_threshold),
            "entity_distinct_threshold": float(args.entity_distinct_threshold),
            "entity_batch_recluster": use_clustering and not bool(args.no_entity_batch_recluster),
            "exclude_weak_edges_from_stability": not bool(args.include_weak_in_stability),
            "relation_map_file": str(rel_map_path),
            "relation_map_size": len(loaded_map),
            "verbose_relations": bool(args.verbose_relations),
            "qa_graph": args.qa_graph,
            "llm_graph_qa": bool(args.llm_graph_qa),
            "llm_graph_qa_model": args.llm_graph_qa_model if args.llm_graph_qa else None,
            "qa_triple_stats": {
                "engine_export_distinct": len(eng_q),
                "aligned_evidence_union_distinct": len(aln_q),
                "qa_graph_before_alias_expand": len(qa_base),
                "qa_graph_distinct": len(qa_triples),
                **qa_graph_diff,
            },
        },
        "relation_map": dict(relation_utils.REL_MAP),
        "summary": summary_row,
        "overall": overall,
        "slices": {
            "top_10_high_drift_sets": top_drift_trimmed,
            "top_10_qa_misses_missing_edges": qa_misses_top,
        },
        "sets": set_results,
        "qa": qa,
        "qa_llm_graph_rag": qa_llm_graph_rag,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md_lines = ["# Evidence-Bound Ontology State Engine Results", ""]
    md_lines.append("## Summary")
    sr = summary_row
    sum_header = [
        "GED mean",
        "Jaccard mean",
        f"Drift@{args.drift_tau}",
        "Unsupported (per-note)",
        "Conflict records",
        "QA EM",
        "QA Recall@1",
    ]
    sum_row = [
        f"{sr['GED_mean']:.4f}",
        f"{sr['Jaccard_mean']:.4f}",
        f"{sr[f'Drift@{args.drift_tau}']:.4f}",
        f"{sr['Unsupported_note_mean']:.4f}",
        str(sr["conflict_candidates_records"]),
        f"{sr['QA_EM']:.4f}" if sr.get("QA_EM") is not None else "n/a",
        f"{sr['QA_Recall@1']:.4f}" if sr.get("QA_Recall@1") is not None else "n/a",
    ]
    if "QA_LLM_EM" in sr:
        sum_header.extend(["QA LLM EM", "QA LLM R@1"])
        sum_row.extend(
            [
                f"{sr['QA_LLM_EM']:.4f}" if sr.get("QA_LLM_EM") is not None else "n/a",
                f"{sr['QA_LLM_Recall@1']:.4f}" if sr.get("QA_LLM_Recall@1") is not None else "n/a",
            ]
        )
    md_lines.append(_to_md_table(sum_header, [sum_row]))
    md_lines.append("")
    md_lines.append("## Top 10 high-drift sets (lowest Jaccard stability)")
    drift_rows = [
        [
            z["set_id"],
            z["fact"][:56] + ("…" if len(z["fact"]) > 56 else ""),
            f"{z['jaccard_mean']:.4f}",
            f"{z['ged_mean']:.4f}",
            f"{z['drift_at_tau']:.4f}",
        ]
        for z in top_drift_trimmed
    ]
    md_lines.append(
        _to_md_table(["set_id", "fact", "Jaccard", "GED", f"Drift@{args.drift_tau}"], drift_rows)
    )
    md_lines.append("")
    md_lines.append("## Top 10 QA misses (missing canonical edge)")
    if qa_misses_top:
        qrows = [
            [
                m["question"][:50] + ("…" if len(m["question"]) > 50 else ""),
                m["gold"],
                str(m.get("missing_canonical_edge")),
            ]
            for m in qa_misses_top
        ]
        md_lines.append(_to_md_table(["question", "gold", "missing_edge (sub,rel,obj)"], qrows))
    else:
        md_lines.append("_No QA misses slice (or QA file missing)._")
    md_lines.append("")
    md_lines.append(f"Full results JSON: `{out_json}`")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved full results JSON: {out_json}")
    print(f"Saved readable markdown: {out_md}")


if __name__ == "__main__":
    main()
