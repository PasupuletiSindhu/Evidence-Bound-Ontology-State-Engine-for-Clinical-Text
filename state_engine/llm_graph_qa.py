"""
LLM QA grounded in a knowledge graph: retrieve relevant triples, then let the model read them and answer.

This is separate from baselines.qa_eval.run_qa_eval, which does symbolic index lookup only.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

log = logging.getLogger(__name__)

Triple = Tuple[str, str, str]

_LLM_SYSTEM = """You are a biomedical QA assistant. You must answer using ONLY the knowledge graph lines below.
Each line is one fact in the form: subject | relation | object

Rules:
- Output a single short answer: the entity that answers the question (use the same wording or ID style as in the graph when possible).
- If the graph does not support a confident answer, output exactly: UNKNOWN
- No explanation, no bullet points, one line only."""


def _tokens(s: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def retrieve_triples_for_example(
    example: Dict[str, Any],
    triples: List[Triple],
    max_triples: int,
) -> List[Triple]:
    """
    Simple lexical retrieval: rank triples by overlap with question + filled schema slots.
    """
    q = example.get("question", "") or ""
    sub = (example.get("subject") or "").strip()
    rel = (example.get("relation") or "").strip()
    obj = (example.get("object") or "").strip()
    anchors = [x for x in (q, sub, rel, obj) if x and x != "?"]
    anchor_toks: Set[str] = set()
    for a in anchors:
        anchor_toks |= _tokens(a)

    scored: List[Tuple[int, Triple]] = []
    for t in triples:
        s, r, o = str(t[0]), str(t[1]), str(t[2])
        line = f"{s} {r} {o}"
        score = len(anchor_toks & _tokens(line))
        low = line.lower()
        for a in anchors:
            if len(a) > 2 and a.lower() in low:
                score += 4
        scored.append((score, (s, r, o)))

    scored.sort(key=lambda x: -x[0])
    picked = [t for _, t in scored[:max_triples]]
    if not picked and triples:
        picked = [tuple(str(x) for x in t[:3]) for t in triples[:max_triples]]  # type: ignore
    return picked


def _format_graph_block(triples: List[Triple]) -> str:
    lines = [f"{s} | {r} | {o}" for s, r, o in triples]
    return "\n".join(lines)


def _parse_llm_answer(raw: str) -> str:
    if not raw:
        return ""
    line = raw.strip().splitlines()[0].strip()
    low = line.lower()
    if low.startswith("unknown"):
        return ""
    if ":" in line and len(line) < 200:
        # "Answer: foo" -> foo
        parts = line.split(":", 1)
        if parts[0].lower().strip() in ("answer", "the answer", "result"):
            line = parts[1].strip()
    line = line.strip().strip('"').strip("'")
    return line


def run_llm_graph_qa(
    examples: List[Dict[str, Any]],
    triples: List[Triple],
    llm: Any,
    *,
    max_context_triples: int = 100,
    max_new_tokens: int = 96,
    entity_to_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    For each example, retrieve triples, prompt the LLM with the natural-language question,
    score with the same logic as symbolic QA (single predicted string vs gold).

    llm: object with generate_from_messages(messages, max_new_tokens=...) -> str
         (e.g. baselines.extractors.qwen_prompt_extractor.QwenPromptExtractor)
    """
    from baselines.qa_eval import score_qa

    results: List[Dict[str, Any]] = []
    em = 0
    r1 = 0
    n = 0

    for ex in examples:
        sub = ex.get("subject", "")
        rel = ex.get("relation", "")
        obj = ex.get("object", "")
        gold = ex.get("answer", "")
        question = ex.get("question", "") or ""

        if (sub == "?") == (obj == "?"):
            results.append(
                {
                    "question": question,
                    "error": "Exactly one of subject or object must be '?'",
                    "exact_match": False,
                    "recall_at_1": False,
                    "predicted_llm": "",
                    "retrieved_triples_count": 0,
                }
            )
            continue

        ctx = retrieve_triples_for_example(ex, triples, max_context_triples)
        graph_block = _format_graph_block(ctx)
        user_msg = (
            f"Knowledge graph ({len(ctx)} facts):\n{graph_block}\n\n"
            f"Question: {question}\n\n"
            "Answer with the single missing entity (or UNKNOWN):"
        )
        messages = [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        try:
            raw = llm.generate_from_messages(messages, max_new_tokens=max_new_tokens)
        except Exception as e:
            log.exception("LLM QA generation failed: %s", e)
            results.append(
                {
                    "question": question,
                    "error": str(e),
                    "exact_match": False,
                    "recall_at_1": False,
                    "predicted_llm": "",
                    "retrieved_triples_count": len(ctx),
                }
            )
            n += 1
            continue

        pred_str = _parse_llm_answer(raw)
        pred_set: Set[str] = {pred_str} if pred_str else set()
        scored = score_qa(
            pred_set,
            gold,
            normalize_answer=not entity_to_id,
            entity_to_id=entity_to_id,
        )
        results.append(
            {
                "question": question,
                "exact_match": scored["exact_match"],
                "recall_at_1": scored["recall_at_1"],
                "predicted": scored.get("predicted", list(pred_set)),
                "predicted_llm": pred_str,
                "gold": gold,
                "llm_raw": raw[:500],
                "retrieved_triples_count": len(ctx),
            }
        )
        n += 1
        if scored["exact_match"]:
            em += 1
        if scored["recall_at_1"]:
            r1 += 1

    metrics = {
        "exact_match": em / n if n else 0.0,
        "recall_at_1": r1 / n if n else 0.0,
        "num_questions": n,
    }
    return results, metrics


def load_qwen_for_qa(model_name: str, **kwargs: Any) -> Any:
    """Import here so importing state_engine does not require torch."""
    from baselines.extractors.qwen_prompt_extractor import QwenPromptExtractor

    return QwenPromptExtractor(model_name=model_name, **kwargs)
