# Dictionary-based state engine for building a knowledge graph incrementally
# (Subject, Relation, Object) triples with Edge State
# Provenance - which sources contributed to the edge
# Conflicts - conflicts between edges (Human readable tags + structured conflict records)
# Status - active, uncertain, weak

"""update(triples, source_id) for each incoming batch:
If the exact triple already exists → merge provenance, bump confidence; if polarity clashes → mark uncertain and log conflict.
Else if same subject + relation and objects are equivalent (string match, surface normalize, or cosine ≥ object_merge_threshold) → merge into that edge (same polarity/conflict logic).
Else add a new edge, then check peers:
Same (subject, object) but different relation → classify clash type (e.g. causes vs treats = semantic polarity clash) and mark both uncertain.
Same (subject, relation) but non-equivalent objects → object disagreement."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Triple = Tuple[str, str, str]

EXACERBATING = frozenset({"causes", "increases"})
AMELIORATING = frozenset({"treats", "prevents", "reduces"})


@dataclass
class EvidenceRecord:
    source_id: str
    raw_subject: str
    raw_relation: str
    raw_object: str
    confidence: float = 1.0
    polarity: str = "positive"


@dataclass
class EdgeState:
    subject: str
    relation: str
    object: str
    polarity: str = "positive"
    confidence: float = 1.0
    provenance: List[EvidenceRecord] = field(default_factory=list)
    status: str = "active"  # active | uncertain | weak
    conflicts: List[str] = field(default_factory=list)
    conflict_records: List[Dict] = field(default_factory=list)


class EvidenceBoundOntologyStateEngine:
    """Persistent graph state with evidence-preserving deterministic updates.

    Each ``update()`` *adds* or *merges* edges; we never drop an edge because it
    appeared in only one paraphrase. Multiple paraphrases that assert the same
    (subject, relation, object) merge provenance; conflicting assertions mark
    edges ``uncertain`` but remain in the graph unless callers filter them.

    Object disagreement uses embedding cosine similarity when ``object_similarity_fn``
    is set (or default from ``state_engine.embeddings``): same fact if sim >=
    ``object_merge_threshold`` (default 0.85).
    """

    def __init__(
        self,
        *,
        object_merge_threshold: float = 0.85,
        object_similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self._edges: Dict[Triple, EdgeState] = {}
        self._object_merge_threshold = float(object_merge_threshold)
        self._object_similarity_fn = object_similarity_fn

    def _objects_equivalent_for_conflict(self, o1: str, o2: str) -> bool:
        """True if two object strings describe the same fact (no object conflict)."""
        if o1 == o2:
            return True
        try:
            from state_engine.semantic_canonicalizer import normalize_surface

            if normalize_surface(o1) == normalize_surface(o2):
                return True
        except Exception:
            pass
        sim_fn = self._object_similarity_fn
        if sim_fn is None:
            try:
                from state_engine.embeddings import text_cosine_similarity

                sim_fn = text_cosine_similarity
            except Exception:
                return False
        try:
            return sim_fn(o1, o2) >= self._object_merge_threshold
        except Exception:
            return False

    @staticmethod
    def _is_relation_conflict(prev_relation: str, new_relation: str) -> bool:
        return prev_relation != new_relation

    @staticmethod
    def _is_polarity_conflict(prev_polarity: str, new_polarity: str) -> bool:
        return prev_polarity != new_polarity

    @classmethod
    def _classify_pair(
        cls,
        r1: str,
        pol1: str,
        r2: str,
        pol2: str,
    ) -> Optional[str]:
        if r1 == r2:
            if cls._is_polarity_conflict(pol1, pol2):
                return "polarity_clash_same_relation"
            return None
        if cls._is_polarity_conflict(pol1, pol2):
            return "polarity_clash"
        if (r1 in EXACERBATING and r2 in AMELIORATING) or (
            r1 in AMELIORATING and r2 in EXACERBATING
        ):
            return "semantic_polarity_clash"
        if {r1, r2} == {"increases", "reduces"}:
            return "magnitude_polarity_clash"
        return "multi_relation_mismatch"

    def _append_conflict_record(
        self,
        edge: EdgeState,
        conflict_type: str,
        source_ids: List[str],
        peer_relation: str,
        peer_key: str,
    ) -> None:
        edge.conflict_records.append(
            {
                "conflict_type": conflict_type,
                "source_ids": sorted(set(source_ids)),
                "resolution_status": "unresolved",
                "peer_relation": peer_relation,
                "peer_edge": peer_key,
            }
        )
        edge.conflicts.append(
            f"{conflict_type}:{peer_relation}@{','.join(sorted(set(source_ids)))}"
        )

    def _find_peers(self, s: str, o: str, r: str) -> List[EdgeState]:
        return [
            e
            for e in self._edges.values()
            if e.subject == s and e.object == o and e.relation != r
        ]

    def _find_object_disagreement_peers(self, s: str, r: str, o: str) -> List[EdgeState]:
        """Same (subject, relation) but semantically distinct object."""
        return [
            e
            for e in self._edges.values()
            if e.subject == s
            and e.relation == r
            and not self._objects_equivalent_for_conflict(e.object, o)
        ]

    def update(
        self,
        triples: List[Tuple[str, str, str, float, str]],
        source_id: str,
    ) -> Dict[str, int]:
        """
        triples: list of (subject, relation, object, confidence, polarity)
        """
        added = 0
        merged = 0
        conflicts = 0
        uncertain = 0

        for s, r, o, conf, pol in triples:
            key = (s, r, o)
            evidence = EvidenceRecord(
                source_id=source_id,
                raw_subject=s,
                raw_relation=r,
                raw_object=o,
                confidence=float(conf),
                polarity=pol,
            )

            if key in self._edges:
                edge = self._edges[key]
                edge.provenance.append(evidence)
                edge.confidence = max(edge.confidence, float(conf))
                if self._is_polarity_conflict(edge.polarity, pol):
                    edge.status = "uncertain"
                    ct = "polarity_merge_clash"
                    self._append_conflict_record(
                        edge,
                        ct,
                        [p.source_id for p in edge.provenance],
                        edge.relation,
                        str(key),
                    )
                    conflicts += 1
                    uncertain += 1
                merged += 1
                continue

            coalesced_key = None
            for ek in sorted(self._edges.keys(), key=lambda t: (t[0], t[1], t[2])):
                es, er, eo = ek
                if es == s and er == r and self._objects_equivalent_for_conflict(eo, o):
                    coalesced_key = ek
                    break
            if coalesced_key is not None:
                edge = self._edges[coalesced_key]
                edge.provenance.append(evidence)
                edge.confidence = max(edge.confidence, float(conf))
                if self._is_polarity_conflict(edge.polarity, pol):
                    edge.status = "uncertain"
                    ct = "polarity_merge_clash"
                    self._append_conflict_record(
                        edge,
                        ct,
                        [p.source_id for p in edge.provenance],
                        edge.relation,
                        str(coalesced_key),
                    )
                    conflicts += 1
                    uncertain += 1
                merged += 1
                continue

            peer_edges = self._find_peers(s, o, r)
            obj_conflict_peers = self._find_object_disagreement_peers(s, r, o)
            edge = EdgeState(
                subject=s,
                relation=r,
                object=o,
                polarity=pol,
                confidence=float(conf),
                provenance=[evidence],
                status="weak" if r == "related_to" else "active",
                conflicts=[],
                conflict_records=[],
            )

            incoming_src = [source_id]
            for peer in peer_edges:
                ct = self._classify_relation_pair_wrapper(peer, edge)
                if ct is None:
                    continue
                conflicts += 1
                uncertain += 1
                edge.status = "uncertain"
                peer.status = "uncertain"
                peer_src = [p.source_id for p in peer.provenance]
                all_src = peer_src + incoming_src
                peer_key = f"({peer.subject},{peer.relation},{peer.object})"
                new_key_str = f"({s},{r},{o})"
                self._append_conflict_record(
                    edge, ct, all_src, peer.relation, peer_key
                )
                self._append_conflict_record(
                    peer, ct, all_src, r, new_key_str
                )

            for peer in obj_conflict_peers:
                ct = "object_disagreement_same_subject_relation"
                conflicts += 1
                uncertain += 1
                edge.status = "uncertain"
                peer.status = "uncertain"
                peer_src = [p.source_id for p in peer.provenance]
                all_src = peer_src + incoming_src
                peer_key = f"({peer.subject},{peer.relation},{peer.object})"
                new_key_str = f"({s},{r},{o})"
                self._append_conflict_record(edge, ct, all_src, peer.relation, peer_key)
                self._append_conflict_record(peer, ct, all_src, r, new_key_str)

            self._edges[key] = edge
            added += 1

        return {
            "added": added,
            "merged": merged,
            "conflicts": conflicts,
            "uncertain": uncertain,
            "num_edges": len(self._edges),
        }

    def _classify_relation_pair_wrapper(
        self, peer: EdgeState, new_edge: EdgeState
    ) -> Optional[str]:
        return self._classify_pair(
            peer.relation,
            peer.polarity,
            new_edge.relation,
            new_edge.polarity,
        )

    def export_triples(self, include_uncertain: bool = True) -> List[Triple]:
        out = []
        for edge in self._edges.values():
            if not include_uncertain and edge.status == "uncertain":
                continue
            out.append((edge.subject, edge.relation, edge.object))
        return out

    def export_state(self) -> Dict:
        return {
            "num_edges": len(self._edges),
            "edges": [
                {
                    "subject": e.subject,
                    "relation": e.relation,
                    "object": e.object,
                    "polarity": e.polarity,
                    "confidence": e.confidence,
                    "status": e.status,
                    "conflicts": list(e.conflicts),
                    "conflict_records": list(e.conflict_records),
                    "num_distinct_sources": len({p.source_id for p in e.provenance}),
                    "provenance": [
                        {
                            "source_id": p.source_id,
                            "raw_subject": p.raw_subject,
                            "raw_relation": p.raw_relation,
                            "raw_object": p.raw_object,
                            "confidence": p.confidence,
                            "polarity": p.polarity,
                        }
                        for p in e.provenance
                    ],
                }
                for e in self._edges.values()
            ],
        }
