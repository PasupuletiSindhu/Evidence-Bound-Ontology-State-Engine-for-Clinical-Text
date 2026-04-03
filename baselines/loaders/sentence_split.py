"""
Convert document-level NER examples to sentence-level so entities are not lost to truncation.
Long documents (e.g. 1800+ chars) truncated at 512 tokens can have all entities in the cut-off part
→ 0 non-O labels → model learns only O. Sentence-level training avoids this.
"""
import re
from typing import List, Dict, Any, Tuple


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return list of (start, end) character spans for each sentence. Uses pattern scan so offsets are exact."""
    if not text or not text.strip():
        return []
    # Split on sentence-ending punctuation followed by space or newline
    pattern = re.compile(r"(?<=[.!?])\s+|\n+")
    spans = []
    last_end = 0
    for m in pattern.finditer(text):
        sent_start, sent_end = last_end, m.start()
        segment = text[sent_start:sent_end]
        if segment.strip():
            spans.append((sent_start, sent_end))
        last_end = m.end()
    # Tail after last delimiter
    if last_end < len(text) and text[last_end:].strip():
        spans.append((last_end, len(text)))
    return spans


def to_sentence_level_examples(
    examples: List[Dict[str, Any]],
    min_sent_chars: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convert document-level examples to sentence-level.
    Each output item has "text" = one sentence and "entities" with start/end relative to that sentence.
    Only entities fully contained in a sentence are kept; entities spanning sentences are dropped for that sentence.
    Preserves other keys (e.g. "relations") on the first sentence of each document only (for downstream use);
    for NER we only need "text" and "entities".
    """
    out = []
    for ex in examples:
        text = ex.get("text", "")
        entities = ex.get("entities", [])
        if not text:
            continue
        spans = _sentence_spans(text)
        if not spans:
            # No sentence boundaries found: treat whole text as one segment (will be truncated but at least one block)
            out.append({"text": text, "entities": list(entities), **{k: v for k, v in ex.items() if k not in ("text", "entities")}})
            continue
        for start, end in spans:
            sent_text = text[start:end]
            if len(sent_text.strip()) < min_sent_chars:
                continue
            # Entities fully inside this sentence, with offsets relative to sentence
            sent_entities = []
            for e in entities:
                e_start, e_end = int(e["start"]), int(e["end"])
                if e_start >= start and e_end <= end:
                    sent_entities.append({
                        **e,
                        "start": e_start - start,
                        "end": e_end - start,
                    })
            out.append({"text": sent_text, "entities": sent_entities})
    return out


def sentence_level_entity_stats(
    doc_examples: List[Dict[str, Any]],
    sentence_examples: List[Dict[str, Any]],
) -> dict:
    """Return counts to verify we did not lose entities during sentence splitting."""
    doc_entities = sum(len(ex.get("entities", [])) for ex in doc_examples)
    sent_entities = sum(len(ex.get("entities", [])) for ex in sentence_examples)
    sentences_with_entities = sum(1 for ex in sentence_examples if len(ex.get("entities", [])) > 0)
    return {
        "doc_entities": doc_entities,
        "sent_entities": sent_entities,
        "sentences_with_entities": sentences_with_entities,
        "total_sentences": len(sentence_examples),
    }


def first_example_with_entities(examples: List[Dict[str, Any]]):
    """Return the first example that has at least one entity (for debugging alignment)."""
    for ex in examples:
        if ex.get("entities"):
            return ex
    return None
