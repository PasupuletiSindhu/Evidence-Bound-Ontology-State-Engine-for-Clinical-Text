#!/usr/bin/env python3
"""Generate results/relation_train.json from paraphrase_results.json for build_relation_map.py."""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from state_engine.relation_train_builder import write_relation_train_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build relation_train.json from paraphrase results.")
    p.add_argument(
        "--input",
        type=str,
        default="results/paraphrase_results.json",
        help="Paraphrase JSON with sets[].fact and baseline triples.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/relation_train.json",
        help="Output path for training pairs.",
    )
    args = p.parse_args()
    root = _root
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = root / in_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path
    ex = write_relation_train_json(in_path, out_path)
    print(f"Wrote {len(ex)} examples to {out_path}")


if __name__ == "__main__":
    main()
