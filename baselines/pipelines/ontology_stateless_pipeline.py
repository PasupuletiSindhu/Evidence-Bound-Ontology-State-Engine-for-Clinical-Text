# Baseline 2: same as 1 but entities -> CUI/MeSH. Triples are (CUI, relation, CUI).
import re
from typing import List, Tuple

from models.bert_ner import BERTNERModel
from models.bert_relation import BERTRelationModel
from ontology.umls_mapper import UMLSMapper
from pipelines.qa_graph_alignment import (
    finalize_baseline_triples,
    map_relation_to_qa_schema,
    swap_entity_indices_for_qa,
)
from pipelines.stateless_pipeline import _sentencize


class OntologyNormalizedStateless:
    def __init__(
        self,
        ner_model: BERTNERModel,
        relation_model: BERTRelationModel,
        umls_mapper: UMLSMapper,
    ):
        self.ner_model = ner_model
        self.relation_model = relation_model
        self.umls_mapper = umls_mapper
        self._triples: List[Tuple[str, str, str, str]] = []

    def process(self, text: str) -> List[Tuple[str, str, str, str]]:
        self._triples = []
        for sent in _sentencize(text):
            if not sent:
                continue
            entities = self.ner_model.extract_entities(sent)
            if not entities:
                continue
            cuis = [self.umls_mapper.normalize(e.get("text", "")) for e in entities]
            relations = self.relation_model.extract_relations(sent, entities)
            for r in relations:
                canon = map_relation_to_qa_schema(r["relation"])
                if canon is None:
                    continue
                hi, ti = swap_entity_indices_for_qa(
                    r["head"], r["tail"], entities, canon
                )
                sub_cui = cuis[hi] if 0 <= hi < len(cuis) else None
                obj_cui = cuis[ti] if 0 <= ti < len(cuis) else None
                if sub_cui and obj_cui:
                    self._triples.append((sub_cui, canon, obj_cui, "positive"))
        return finalize_baseline_triples(list(self._triples))
