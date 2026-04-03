# Symbolic QA over a graph: deterministic index lookup, not an LLM.
#
# build_graph_index + query_graph implement structured retrieval: given the schema
# slot (subject, relation, ?) or (?, relation, object), they return matching entity
# strings from indexed triples. This is fast and reproducible but does not "reason"
# in natural language. For LLM + graph retrieval (RAG-style QA), see
# state_engine.llm_graph_qa (optional --llm_graph_qa on run_state_engine).
#
# Original one-liner: QA over graph: (subject, relation) -> objects, or (relation, object) -> subjects.
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import json
from pathlib import Path
from collections import Counter


Triple = Tuple[str, str, str]
_RECALL_SIM_THRESHOLD = 0.75
_EXACT_SIM_THRESHOLD = 0.90


def _normalize_entity(s: str) -> str:
    s = (s or "").strip()
    # drop "1. ", "2. " from Qwen output
    if s and s[0].isdigit():
        for i, c in enumerate(s):
            if not c.isdigit() and c not in ".:":
                s = s[i:].lstrip(".: ")
                break
    return s.lower()


def _normalize_relation(r: str) -> str:
    return (r or "").strip().lower()


def _triples_to_set(triples: List) -> Set[Triple]:
    out = set()
    for t in triples:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
            if s and r and o:
                out.add((s, r, o))
    return out


def _triples_to_list(triples: List) -> List[Triple]:
    out: List[Triple] = []
    for t in triples:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            s, r, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
            if s and r and o:
                out.append((s, r, o))
    return out


def build_graph_ranked_index(
    triples: List,
    normalize_entities: bool = True,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[Dict[Tuple[str, str], Counter], Dict[Tuple[str, str], Counter]]:
    # (subject, relation) -> Counter(objects); (relation, object) -> Counter(subjects)
    t_list = _triples_to_list(triples)
    index_sr_to_o: Dict[Tuple[str, str], Counter] = {}
    index_ro_to_s: Dict[Tuple[str, str], Counter] = {}
    for s, r, o in t_list:
        s_for_lookup = (entity_to_id(s) or s) if entity_to_id else s
        o_for_lookup = (entity_to_id(o) or o) if entity_to_id else o
        sn = _normalize_entity(s_for_lookup) if normalize_entities else s_for_lookup
        rn = _normalize_relation(r)
        on = _normalize_entity(o_for_lookup) if normalize_entities else o_for_lookup
        key_sr = (sn, rn)
        key_ro = (rn, on)
        if key_sr not in index_sr_to_o:
            index_sr_to_o[key_sr] = Counter()
        if key_ro not in index_ro_to_s:
            index_ro_to_s[key_ro] = Counter()
        index_sr_to_o[key_sr][o] += 1
        index_ro_to_s[key_ro][s] += 1
    return index_sr_to_o, index_ro_to_s


def build_graph_index(
    triples: List,
    normalize_entities: bool = True,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Set[str]]]:
    # Backward-compatible set view derived from ranked counters.
    ranked_sr, ranked_ro = build_graph_ranked_index(
        triples, normalize_entities=normalize_entities, entity_to_id=entity_to_id
    )
    index_sr_to_o: Dict[Tuple[str, str], Set[str]] = {
        k: set(v.keys()) for k, v in ranked_sr.items()
    }
    index_ro_to_s: Dict[Tuple[str, str], Set[str]] = {
        k: set(v.keys()) for k, v in ranked_ro.items()
    }
    return index_sr_to_o, index_ro_to_s


def query_graph_ranked(
    subject: str,
    relation: str,
    object_slot: str,
    index_sr_to_o: Dict[Tuple[str, str], Counter],
    index_ro_to_s: Dict[Tuple[str, str], Counter],
    relation_aliases: Optional[Dict[str, List[str]]] = None,
    normalize_entities: bool = True,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> List[str]:
    relation_aliases = relation_aliases or {}
    counts: Counter = Counter()
    rel_variants = [relation] + relation_aliases.get(_normalize_relation(relation), [])
    if object_slot == "?":
        sub_for_lookup = (entity_to_id(subject) or subject) if entity_to_id else subject
        sn = _normalize_entity(sub_for_lookup) if normalize_entities else sub_for_lookup
        for r in rel_variants:
            rn = _normalize_relation(r)
            counts.update(index_sr_to_o.get((sn, rn), Counter()))
    elif subject == "?":
        obj_for_lookup = (entity_to_id(object_slot) or object_slot) if entity_to_id else object_slot
        on = _normalize_entity(obj_for_lookup) if normalize_entities else obj_for_lookup
        for r in rel_variants:
            rn = _normalize_relation(r)
            counts.update(index_ro_to_s.get((rn, on), Counter()))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    return [x for x, _ in ranked]


def query_graph(
    subject: str,
    relation: str,
    object_slot: str,
    index_sr_to_o: Dict[Tuple[str, str], Set[str]],
    index_ro_to_s: Dict[Tuple[str, str], Set[str]],
    relation_aliases: Optional[Dict[str, List[str]]] = None,
    normalize_entities: bool = True,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Set[str]:
    # Backward-compatible wrapper over ranked retrieval.
    # Convert set-view index to counters with unit weights.
    idx_sr_counter = {k: Counter({x: 1 for x in v}) for k, v in index_sr_to_o.items()}
    idx_ro_counter = {k: Counter({x: 1 for x in v}) for k, v in index_ro_to_s.items()}
    return set(
        query_graph_ranked(
            subject,
            relation,
            object_slot,
            idx_sr_counter,
            idx_ro_counter,
            relation_aliases=relation_aliases,
            normalize_entities=normalize_entities,
            entity_to_id=entity_to_id,
        )
    )


def _normalize_answer(a: str) -> str:
    return _normalize_entity(a or "")


def _semantic_similarity(pred: str, gold: str) -> float:
    """
    Similarity in [0,1]:
    - exact normalized string equality => 1.0
    - otherwise embedding cosine on raw text (no pre-normalization)
    """
    p = (pred or "").strip()
    g = (gold or "").strip()
    if not p or not g:
        return 0.0
    if _normalize_answer(p) == _normalize_answer(g):
        return 1.0
    try:
        from state_engine.embeddings import text_cosine_similarity

        return float(text_cosine_similarity(p, g))
    except Exception:
        return 0.0


def _best_similarity(predicted_list: List[str], gold: str) -> float:
    if not predicted_list:
        return 0.0
    return max(_semantic_similarity(p, gold) for p in predicted_list)


def score_qa(
    predicted: Set[str],
    gold: str,
    normalize_answer: bool = True,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    # Semantic concept match by best candidate similarity.
    predicted_list = list(predicted)
    if entity_to_id:
        predicted_ids = [(entity_to_id(p) or p) for p in predicted_list]
        gold_id = entity_to_id(gold) or gold
        best_sim = max(
            _best_similarity(predicted_list, gold),
            _best_similarity(predicted_ids, gold_id),
        )
    else:
        best_sim = _best_similarity(predicted_list, gold)
    semantic_hit = best_sim >= _RECALL_SIM_THRESHOLD
    exact_match = best_sim >= _EXACT_SIM_THRESHOLD
    return {
        "exact_match": exact_match,
        "semantic_hit": semantic_hit,
        # Backward-compatible alias for existing scripts/reports.
        "recall_at_1": semantic_hit,
        "predicted": list(predicted),
        "gold": gold,
        "best_similarity": best_sim,
    }


def score_qa_ranked(
    predicted_ranked: List[str],
    gold: str,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    best_sim = 0.0
    if entity_to_id:
        predicted_ids = [(entity_to_id(p) or p) for p in predicted_ranked]
        gold_id = entity_to_id(gold) or gold
        best_sim = max(
            _best_similarity(predicted_ranked, gold),
            _best_similarity(predicted_ids, gold_id),
        )
    else:
        best_sim = _best_similarity(predicted_ranked, gold)
    semantic_hit = best_sim >= _RECALL_SIM_THRESHOLD
    exact_match = best_sim >= _EXACT_SIM_THRESHOLD
    top1 = predicted_ranked[0] if predicted_ranked else None
    top1_sim = _semantic_similarity(top1, gold) if top1 else 0.0
    if entity_to_id and top1 is not None:
        top1_sim = max(
            top1_sim,
            _semantic_similarity(entity_to_id(top1) or top1, entity_to_id(gold) or gold),
        )
    top1_correct = top1_sim >= _EXACT_SIM_THRESHOLD
    rr = 0.0
    for i, cand in enumerate(predicted_ranked, start=1):
        sim = _semantic_similarity(cand, gold)
        if entity_to_id:
            sim = max(
                sim,
                _semantic_similarity(entity_to_id(cand) or cand, entity_to_id(gold) or gold),
            )
        if sim >= _EXACT_SIM_THRESHOLD:
            rr = 1.0 / float(i)
            break
    return {
        "exact_match": exact_match,
        "semantic_hit": semantic_hit,
        "recall_at_1": semantic_hit,
        "predicted_ranked": list(predicted_ranked),
        "predicted": list(predicted_ranked),
        "top1_prediction": top1,
        "top1_correct": top1_correct,
        "reciprocal_rank": rr,
        "gold": gold,
        "best_similarity": best_sim,
    }


def run_qa_eval(
    triples: List,
    qa_examples: List[Dict],
    relation_aliases: Optional[Dict[str, List[str]]] = None,
    normalize_entities: bool = True,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[List[Dict], Dict[str, float]]:
    # one of subject/object must be "?". entity_to_id for Baseline 2 (CUI lookup).
    index_sr_to_o, index_ro_to_s = build_graph_ranked_index(
        triples,
        normalize_entities=normalize_entities,
        entity_to_id=entity_to_id,
    )
    relation_aliases = relation_aliases or {}
    results = []
    exact_matches = 0
    semantic_hits = 0
    top1_hits = 0
    rr_sum = 0.0
    n = 0
    for ex in qa_examples:
        sub = ex.get("subject", "")
        rel = ex.get("relation", "")
        obj = ex.get("object", "")
        gold = ex.get("answer", "")
        if (sub == "?") == (obj == "?"):
            results.append({
                "question": ex.get("question", ""),
                "error": "Exactly one of subject or object must be '?'",
                "exact_match": False,
                "semantic_hit": False,
                "recall_at_1": False,
            })
            continue
        predicted_ranked = query_graph_ranked(
            sub, rel, obj, index_sr_to_o, index_ro_to_s,
            relation_aliases, normalize_entities, entity_to_id=entity_to_id,
        )
        score = score_qa_ranked(predicted_ranked, gold, entity_to_id=entity_to_id)
        score["question"] = ex.get("question", "")
        results.append(score)
        n += 1
        if score["exact_match"]:
            exact_matches += 1
        if score["semantic_hit"]:
            semantic_hits += 1
        if score["top1_correct"]:
            top1_hits += 1
        rr_sum += float(score["reciprocal_rank"])
    metrics = {
        "exact_match": exact_matches / n if n else 0.0,
        "semantic_hit": semantic_hits / n if n else 0.0,
        "top1_accuracy": top1_hits / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        # Backward-compatible alias for existing scripts/reports.
        "recall_at_1": semantic_hits / n if n else 0.0,
        "num_questions": n,
    }
    return results, metrics


def load_qa_dataset(path: str) -> Tuple[List[Dict], Dict[str, List[str]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = data.get("examples", data.get("qa", []))
    aliases = data.get("relation_aliases", {})
    return examples, aliases


def merge_triples_from_paraphrase_results(
    results_json: Dict,
    baseline_key: str,
) -> List[Tuple[str, str, str]]:
    # flatten triples_per_paraphrase for one baseline (from sets or top-level)
    triples = []
    if results_json.get("sets"):
        for set_payload in results_json["sets"]:
            for block in (set_payload.get(baseline_key) or {}).get("triples_per_paraphrase", []):
                for t in block:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        triples.append((str(t[0]), str(t[1]), str(t[2])))
    else:
        for block in results_json.get(baseline_key, {}).get("triples_per_paraphrase", []):
            for t in block:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    triples.append((str(t[0]), str(t[1]), str(t[2])))
    return triples
