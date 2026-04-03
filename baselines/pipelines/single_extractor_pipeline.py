# Baseline 3: Qwen prompt-based triples. No NER/relation; optional BERT or custom LLM.
import re
from typing import List, Tuple, Literal, Optional, Callable

from models.bert_ner import BERTNERModel


def _sentencize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]


class SingleExtractorVariant:
    def __init__(
        self,
        mode: Literal["bert", "llm", "qwen"],
        ner_model: Optional[BERTNERModel] = None,
        llm_extract_fn: Optional[Callable[[str], List]] = None,
        qwen_model_name: str = "Qwen/Qwen2.5-0.5B",
    ):
        self.mode = mode
        self.ner_model = ner_model
        self.llm_extract_fn = llm_extract_fn
        self.qwen_model_name = qwen_model_name
        self._qwen_extractor = None
        self._triples: List[Tuple[str, str, str, str]] = []
        if mode == "bert" and ner_model is None:
            raise ValueError("mode='bert' requires ner_model")
        if mode == "llm" and llm_extract_fn is None:
            raise ValueError("mode='llm' requires llm_extract_fn")

    def _get_qwen_extractor(self):
        if self._qwen_extractor is None:
            from extractors.qwen_prompt_extractor import QwenPromptExtractor
            self._qwen_extractor = QwenPromptExtractor(model_name=self.qwen_model_name)
        return self._qwen_extractor

    def process(self, text: str) -> List[Tuple[str, str, str, str]]:
        self._triples = []
        if self.mode == "bert":
            for sent in _sentencize(text):
                if not sent:
                    continue
                entities = self.ner_model.extract_entities(sent)
                for e in entities:
                    span = e.get("text", "")
                    label = e.get("label", "")
                    if span:
                        self._triples.append((span, "entity", label, "positive"))
        elif self.mode == "qwen":
            qwen = self._get_qwen_extractor()
            for (s, r, o) in qwen.extract(text):
                self._triples.append((s, r, o, "positive"))
        else:
            if self.llm_extract_fn:
                raw = self.llm_extract_fn(text)
                for t in raw:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        s, r, o = t[0], t[1], t[2]
                        self._triples.append((str(s), str(r), str(o), "positive"))
                    elif isinstance(t, dict):
                        s = t.get("subject", t.get("head", ""))
                        r = t.get("relation", "")
                        o = t.get("object", t.get("tail", ""))
                        if s or o:
                            self._triples.append((str(s), str(r), str(o), "positive"))
        return list(self._triples)
