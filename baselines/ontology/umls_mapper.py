# Entity text -> CUI/MeSH. From file, BC5CDR, or MedMentions.
from pathlib import Path
from typing import Dict, Optional, List


class UMLSMapper:
    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self._map: Dict[str, str] = {}
        if mapping:
            for k, v in mapping.items():
                self._map[self._norm(k)] = v

    def _norm(self, s: str) -> str:
        return " ".join((s or "").lower().split()).strip()

    def add(self, entity_text: str, cui: str):
        self._map[self._norm(entity_text)] = cui

    def normalize(self, entity_text: str) -> str:
        return self._map.get(self._norm(entity_text), entity_text)

    @classmethod
    def from_medmentions(cls, examples: List[Dict]) -> "UMLSMapper":
        m = cls()
        for ex in examples:
            text_full = ex.get("text", "")
            for e in ex.get("entities", []):
                span = e.get("text") or text_full[e["start"]:e["end"]]
                cui = e.get("cui")
                if span and cui:
                    m.add(span, cui)
        return m

    @classmethod
    def from_file(cls, path: str, sep: str = "\t") -> "UMLSMapper":
        # lines: entity_text sep cui
        mapping = {}
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(sep, 1)
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = parts[1].strip()
        return cls(mapping)

    @classmethod
    def from_bc5cdr(cls, examples: List[Dict]) -> "UMLSMapper":
        m = cls()
        for ex in examples:
            for e in ex.get("entities", []):
                span = (e.get("text") or "").strip()
                ident = e.get("identifier", "").strip()
                if span and ident and ident != "-1":
                    m.add(span, ident)
        return m
