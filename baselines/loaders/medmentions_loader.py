"""
MedMentions loader — PubTator format.
Parses entity annotations: start, end, text span, UMLS CUI, semantic type.
Returns structured dataset with train/dev/test split (80/10/10, deterministic seed).
"""
import gzip
import random
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_TRAIN, DEFAULT_DEV, DEFAULT_TEST = 0.8, 0.1, 0.1
GITHUB_BASE = "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master"


def _parse_annotation_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one annotation: PMID\tstart\tend\tmention\ttype\tCUI"""
    parts = line.strip().split("\t")
    if len(parts) < 6:
        return None
    try:
        return {
            "start": int(parts[1]),
            "end": int(parts[2]),
            "text": parts[3],
            "semantic_type": parts[4][:50] if parts[4] else "T047",
            "cui": parts[5],
        }
    except (ValueError, IndexError):
        return None


def _parse_block(lines: List[str]) -> Optional[Dict[str, Any]]:
    text_parts = []
    entities = []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("|")
        if len(parts) == 3:
            text_parts.append(parts[2])
        elif "\t" in line:
            ann = _parse_annotation_line(line)
            if ann:
                entities.append(ann)
    if not text_parts:
        return None
    return {"text": " ".join(text_parts), "entities": entities}


def _ensure_medmentions_local(data_dir: str, subset: str = "full") -> Path:
    """Download MedMentions PubTator from GitHub if not present."""
    data_path = Path(data_dir)
    out_dir = data_path / subset / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "corpus_pubtator.txt"
    gz_path = out_dir / "corpus_pubtator.txt.gz"
    if txt_path.exists():
        return txt_path
    if not gz_path.exists():
        url = f"{GITHUB_BASE}/{subset}/data/corpus_pubtator.txt.gz"
        urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
        f_out.write(f_in.read())
    return txt_path


def load_medmentions(
    data_dir: str,
    train_ratio: float = DEFAULT_TRAIN,
    dev_ratio: float = DEFAULT_DEV,
    test_ratio: float = DEFAULT_TEST,
    seed: int = 42,
    download_if_missing: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load MedMentions in PubTator format.
    Returns (train, dev, test). Each item:
      {"text": str, "entities": [{"start": int, "end": int, "text": str, "cui": str, "semantic_type": str}]}
    """
    data_path = Path(data_dir)
    corpus_paths = []
    if (data_path / "full" / "data" / "corpus_pubtator.txt").exists():
        corpus_paths.append(data_path / "full" / "data" / "corpus_pubtator.txt")
    elif (data_path / "full" / "data" / "corpus_pubtator.txt.gz").exists():
        corpus_paths.append(data_path / "full" / "data" / "corpus_pubtator.txt.gz")
    if not corpus_paths and download_if_missing:
        _ensure_medmentions_local(data_dir)
        corpus_paths.append(Path(data_dir) / "full" / "data" / "corpus_pubtator.txt")
    if not corpus_paths:
        for p in data_path.rglob("*pubtator*"):
            if p.suffix in (".txt", ".gz"):
                corpus_paths.append(p)
                break
    if not corpus_paths:
        raise FileNotFoundError(f"No MedMentions PubTator file under {data_dir}")

    all_examples = []
    for path in corpus_paths:
        open_fn = gzip.open if path.suffix == ".gz" else open
        with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
            block = []
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                if "|" in line and len(line.split("|")) == 3:
                    if block and any("\t" in l for l in block):
                        ex = _parse_block(block)
                        if ex and (ex["text"].strip() or ex["entities"]):
                            all_examples.append(ex)
                    block = [line]
                else:
                    block.append(line)
            if block:
                ex = _parse_block(block)
                if ex and (ex["text"].strip() or ex["entities"]):
                    all_examples.append(ex)

    rng = random.Random(seed)
    rng.shuffle(all_examples)
    n = len(all_examples)
    t_end = int(n * train_ratio)
    d_end = t_end + int(n * dev_ratio)
    return all_examples[:t_end], all_examples[t_end:d_end], all_examples[d_end:]
