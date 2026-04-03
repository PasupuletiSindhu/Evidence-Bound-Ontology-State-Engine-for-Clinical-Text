"""
BC5CDR loader — PubTator format.
Source: https://github.com/JHnlp/BioCreative-V-CDR-Corpus
Extracts Disease entities, Chemical entities, and CID relations.
Returns {"text", "entities", "relations": [{"head": entity_id, "tail": entity_id, "type": "CID"}]}.
Train/dev/test split included.
"""
import random
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_TRAIN, DEFAULT_DEV, DEFAULT_TEST = 0.8, 0.1, 0.1
BC5CDR_SAMPLE_URL = "https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_sample.txt"


def _parse_entity_line(line: str) -> Optional[Dict]:
    parts = line.strip().split("\t")
    if len(parts) < 6:
        return None
    try:
        return {
            "start": int(parts[1]),
            "end": int(parts[2]),
            "text": parts[3],
            "type": parts[4],
            "identifier": parts[5] if len(parts) > 5 else parts[3],
        }
    except (ValueError, IndexError):
        return None


def _parse_block(lines: List[str]) -> Optional[Dict[str, Any]]:
    text_parts = []
    entities = []
    relation_lines = []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("|")
        if len(parts) == 3:
            text_parts.append(parts[2])
        elif "\t" in line:
            tabs = line.split("\t")
            if len(tabs) >= 6:
                ent = _parse_entity_line(line)
                if ent:
                    ent["id"] = len(entities)
                    entities.append(ent)
            # Relation line: PMID\tCID\tChemicalID\tDiseaseID (exactly 4 fields)
            if len(tabs) == 4 and tabs[1] in ("CID", "NR"):
                relation_lines.append(tabs)
    if not text_parts:
        return None
    text = " ".join(text_parts)
    relations = []
    for r in relation_lines:
        # Relation line: PMID\tCID\tChemicalID\tDiseaseID (BioCreative-V CDR PubTator)
        if len(r) < 4:
            continue
        rel_type = r[1] if r[1] in ("CID", "NR") else "CID"
        arg1, arg2 = r[2], r[3]  # Chemical ID, Disease ID
        head_id = tail_id = None
        for e in entities:
            ref = e.get("identifier", e.get("text", ""))
            if ref == arg1:
                head_id = e["id"]
            if ref == arg2:
                tail_id = e["id"]
        if head_id is not None and tail_id is not None:
            relations.append({"head": head_id, "tail": tail_id, "type": rel_type})
    return {"text": text, "entities": entities, "relations": relations}


def download_bc5cdr_sample_if_missing(data_dir: str) -> Path:
    """Download CDR_sample.txt from JHnlp/BioCreative-V-CDR-Corpus if data_dir has no .txt."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    if list(data_path.glob("*.txt")):
        return data_path
    sample_path = data_path / "CDR_sample.txt"
    try:
        urllib.request.urlretrieve(BC5CDR_SAMPLE_URL, sample_path)
    except Exception as e:
        raise FileNotFoundError(
            f"No BC5CDR .txt under {data_dir}. Download from {BC5CDR_SAMPLE_URL} and place in {data_dir}. Error: {e}"
        ) from e
    return data_path


def load_bc5cdr(
    data_dir: str,
    train_ratio: float = DEFAULT_TRAIN,
    dev_ratio: float = DEFAULT_DEV,
    test_ratio: float = DEFAULT_TEST,
    seed: int = 42,
    download_if_missing: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load BC5CDR. Each item: {"text", "entities", "relations": [{"head", "tail", "type": "CID"}]}.
    If no .txt in data_dir and download_if_missing=True, downloads CDR_sample.txt from GitHub.
    """
    data_path = Path(data_dir)
    # Search recursively so CDR_Data/.../CDR_*Set.PubTator.txt and other .txt are included
    paths = [p for p in data_path.rglob("*.txt") if "README" not in p.name.upper()]
    if not paths and download_if_missing:
        download_bc5cdr_sample_if_missing(data_dir)
        paths = list(Path(data_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(
            f"No BC5CDR .txt under {data_dir}. Download from {BC5CDR_SAMPLE_URL} or see DATA.md"
        )

    all_examples = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            block = []
            for line in f:
                if not line.strip():
                    continue
                if "|" in line and len(line.split("|")) == 3:
                    if block and any("\t" in l for l in block):
                        ex = _parse_block(block)
                        if ex:
                            all_examples.append(ex)
                    block = [line]
                else:
                    block.append(line)
            if block:
                ex = _parse_block(block)
                if ex:
                    all_examples.append(ex)

    rng = random.Random(seed)
    rng.shuffle(all_examples)
    n = len(all_examples)
    t_end = int(n * train_ratio)
    d_end = t_end + int(n * dev_ratio)
    return all_examples[:t_end], all_examples[t_end:d_end], all_examples[d_end:]
