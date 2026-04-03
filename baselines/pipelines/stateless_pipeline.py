# Baseline 1: NER + relation -> triples. Overwrites state each call.
import re
from typing import List, Dict, Any, Tuple

from models.bert_ner import BERTNERModel
from models.bert_relation import BERTRelationModel
from pipelines.qa_graph_alignment import finalize_baseline_triples, normalize_baseline_triple_row


def _sentencize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]


class StatelessNeuralExtraction:
    def __init__(self, ner_model: BERTNERModel, relation_model: BERTRelationModel):
        self.ner_model = ner_model
        self.relation_model = relation_model
        self._triples: List[Tuple[str, str, str, str]] = []

    def process(self, text: str) -> List[Tuple[str, str, str, str]]:
        self._triples = []
        for sent in _sentencize(text):
            if not sent:
                continue
            entities = self.ner_model.extract_entities(sent)
            if not entities:
                continue
            relations = self.relation_model.extract_relations(sent, entities)
            for r in relations:
                hi, ti = r["head"], r["tail"]
                sub = entities[hi].get("text", "")
                obj = entities[ti].get("text", "")
                row = normalize_baseline_triple_row(
                    sub,
                    r["relation"],
                    obj,
                    "positive",
                    entities[hi],
                    entities[ti],
                )
                if row is not None:
                    self._triples.append(row)
        return finalize_baseline_triples(list(self._triples))
