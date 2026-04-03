# GED, Jaccard, unsupported rate, and a heuristic for same (s,o) different relation.
from typing import List, Set, Tuple, Dict, Any, Callable

Triple = Tuple[str, str, str]


def _normalize_entity(s: str) -> str:
    # strip "1. " etc from LLM output
    s = (s or "").strip()
    if s and s[0].isdigit():
        for i, c in enumerate(s):
            if not c.isdigit() and c not in ".:":
                s = s[i:].lstrip(".: ")
                break
    return s


def _to_set(triples: List) -> Set[Tuple[str, str, str]]:
    out = set()
    for t in triples:
        if len(t) >= 3:
            s = _normalize_entity(str(t[0]))
            r = str(t[1]).strip()
            o = _normalize_entity(str(t[2]))
            out.add((s, r, o))
    return out


def graph_edit_distance(G1: List, G2: List) -> int:
    # symmetric difference size
    s1, s2 = _to_set(G1), _to_set(G2)
    return len(s1 - s2) + len(s2 - s1)


def jaccard_similarity(E1: List, E2: List) -> float:
    s1, s2 = _to_set(E1), _to_set(E2)
    if not s1 and not s2:
        return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def unsupported_inference_rate(
    triples: List,
    supported_set: Set[Tuple[str, str, str]],
) -> float:
    # fraction of triples not in supported_set
    t_set = _to_set(triples)
    if not t_set:
        return 0.0
    unsupported = len(t_set - supported_set)
    return unsupported / len(t_set)


def conflict_detection_precision(
    detected_conflicts: List[Tuple[Triple, Triple]],
    is_true_conflict_fn: Callable[[Triple, Triple], bool],
) -> float:
    # no gold conflict labels in our data, so we report heuristic candidate rate
    if not detected_conflicts:
        return 1.0
    correct = sum(1 for (t1, t2) in detected_conflicts if is_true_conflict_fn(t1, t2))
    return correct / len(detected_conflicts)


def detect_conflicts_heuristic(triples: List[Triple]) -> List[Tuple[Triple, Triple]]:
    # same (s, o), different relation
    out = []
    t_list = list(_to_set(triples))
    for i in range(len(t_list)):
        for j in range(i + 1, len(t_list)):
            s1, r1, o1 = t_list[i]
            s2, r2, o2 = t_list[j]
            if (s1, o1) == (s2, o2) and r1 != r2:
                out.append((t_list[i], t_list[j]))
    return out
