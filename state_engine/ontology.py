"""Canonicalizes entities via baseline UMLS mapper + local normalization."""
from pathlib import Path
from difflib import SequenceMatcher
from typing import Callable, Optional, Tuple

from state_engine.canonicalize import canonical_entity
from state_engine.semantic_canonicalizer import semantic_canonicalize

Triple5 = Tuple[str, str, str, float, str]


class OntologyAligner:
    """Canonicalizes entities via baseline UMLS mapper + local normalization."""

    def __init__(self, mapper=None, semantic_linker: Optional[Callable[[str], Optional[str]]] = None):
        self.mapper = mapper
        self.semantic_linker = semantic_linker

    @staticmethod
    def _norm_text(text: str) -> str:
        return " ".join((text or "").lower().split()).strip()

    @staticmethod
    def _looks_like_ontology_id(text: str) -> bool:
        t = (text or "").strip().upper()
        return bool(
            t.startswith("MESH:")
            or (len(t) >= 2 and t[0] in {"C", "D"} and t[1:].isdigit())
        )

    def _fuzzy_mapper_lookup(self, text: str) -> Optional[str]:
        """Approximate lookup against mapper vocabulary for misspellings/variants."""
        if self.mapper is None:
            return None
        try:
            vocab = getattr(self.mapper, "_map", None)
            if not isinstance(vocab, dict) or not vocab:
                return None
            q = self._norm_text(text)
            if not q:
                return None
            best_key = None
            best_score = 0.0
            q_toks = set(q.split())
            for k in vocab.keys():
                ks = str(k)
                kt = set(ks.split())
                tok_j = (len(q_toks & kt) / max(1, len(q_toks | kt))) if (q_toks or kt) else 0.0
                score = max(tok_j, SequenceMatcher(None, q, ks).ratio())
                if score > best_score:
                    best_score = score
                    best_key = ks
            if best_key is not None and best_score >= 0.82:
                return vocab.get(best_key)
        except Exception:
            return None
        return None

    def _mapper_fn(self) -> Optional[Callable[[str], Optional[str]]]:
        if self.mapper is None:
            return None

        def _fn(text: str):
            try:
                n = self.mapper.normalize(text)
                if not n:
                    return None
                raw_n = self._norm_text(text)
                map_n = self._norm_text(str(n))
                # Accept mapped output only if it changed meaningfully or is an ontology ID.
                if self._looks_like_ontology_id(str(n)) or map_n != raw_n:
                    return str(n)
                # Exact mapper miss: try fuzzy vocabulary lookup for near-matches.
                fz = self._fuzzy_mapper_lookup(text)
                if fz and (
                    self._looks_like_ontology_id(str(fz))
                    or self._norm_text(str(fz)) != raw_n
                ):
                    return str(fz)
                return None
            except Exception:
                return None

        return _fn

    def normalize_entity_light(self, text: str) -> str:
        """Surface-friendly normalization: no UMLS mapper / linker; keeps more literals."""
        return canonical_entity(text, mapper_normalize=None)

    def normalize_entity(self, text: str) -> str:
        # Preferred path: concept linking (e.g. scispaCy UMLS linker) to CUI.
        if self.semantic_linker is not None:
            try:
                cid = self.semantic_linker(text)
                if cid and self._looks_like_ontology_id(str(cid)):
                    return str(cid).strip()
            except Exception:
                pass
        mapped = canonical_entity(text, mapper_normalize=self._mapper_fn())
        return semantic_canonicalize(mapped)

    @staticmethod
    def _build_scispacy_linker(
        model_name: str = "en_core_sci_sm", min_score: float = 0.5
    ) -> Optional[Callable[[str], Optional[str]]]:
        """
        Build a scispaCy UMLS linker callback:
        entity text -> best CUI (or None).
        """
        try:
            import spacy
            from scispacy.linking import EntityLinker
        except Exception:
            return None

        try:
            nlp = spacy.load(model_name)
            if "scispacy_linker" not in nlp.pipe_names:
                nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})
        except Exception:
            return None

        def _link(text: str) -> Optional[str]:
            t = (text or "").strip()
            if not t:
                return None
            doc = nlp(t)
            if not doc.ents:
                return None
            kb_ents = doc.ents[0]._.kb_ents
            if not kb_ents:
                return None
            cui, score = kb_ents[0]
            if score is None or float(score) < float(min_score):
                return None
            return str(cui)

        return _link

    @classmethod
    def from_baselines_mapper(
        cls,
        mapper_file: Optional[str] = None,
        semantic_linker: str = "none",
        scispacy_model: str = "en_core_sci_sm",
    ):
        try:
            from baselines.ontology.umls_mapper import UMLSMapper
        except Exception:
            sem = None
            if semantic_linker == "scispacy":
                sem = cls._build_scispacy_linker(model_name=scispacy_model)
            return cls(mapper=None, semantic_linker=sem)

        sem = None
        if semantic_linker == "scispacy":
            sem = cls._build_scispacy_linker(model_name=scispacy_model)

        if mapper_file:
            try:
                m = UMLSMapper.from_file(mapper_file)
                return cls(m, semantic_linker=sem)
            except Exception:
                pass

        # Match baseline behavior: BC5CDR corpus mapper -> data/cui_map.txt -> empty mapper.
        try:
            from baselines.loaders.bc5cdr_loader import load_bc5cdr

            root = Path(__file__).resolve().parent.parent
            bc5cdr_dir = root / "baselines" / "data" / "bc5cdr"
            train_data, _, _ = load_bc5cdr(str(bc5cdr_dir), download_if_missing=False)
            if train_data:
                return cls(mapper=UMLSMapper.from_bc5cdr(train_data), semantic_linker=sem)
        except Exception:
            pass

        try:
            root = Path(__file__).resolve().parent.parent
            cui_map = root / "baselines" / "data" / "cui_map.txt"
            if cui_map.exists():
                return cls(mapper=UMLSMapper.from_file(str(cui_map)), semantic_linker=sem)
        except Exception:
            pass

        return cls(mapper=UMLSMapper(), semantic_linker=sem)

