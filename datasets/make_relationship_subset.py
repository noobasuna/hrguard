#!/usr/bin/env python3
"""Extract a relationship-focused subset from FP-levelup.

This script filters the FP-levelup English and Chinese JSON files by category
and writes a single JSONL file that is ready for downstream judging and gating.
The output keeps the original source fields and adds a `response` field copied
from the generated text, which is what the judge/gate pipeline expects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def iter_records(paths: Iterable[Path]) -> Iterator[Dict]:
    for path in paths:
        for record in load_json(path):
            yield {"source_file": str(path), "record": record}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a relationship-focused Fraud-R1 subset.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files, typically FP-levelup-English.json and FP-levelup-Chinese.json",
    )
    parser.add_argument(
        "--category",
        default="network friendship",
        help="Category to keep. Default: network friendship",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--mode",
        default="attacker",
        choices=["attacker", "victim"],
        help="Mode assigned to the filtered subset for downstream judge/gate scripts",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Drop the very long raw_data field to make downstream judging faster",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    by_language: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as fout:
        for item in iter_records(inputs):
            record = item["record"]
            if str(record.get("category", "")).strip().lower() != args.category.lower():
                continue

            generated = record.get("generated text", "")
            if not isinstance(generated, str) or not generated.strip():
                continue

            language = str(record.get("language", "unknown"))
            by_language[language] = by_language.get(language, 0) + 1
            out = {
                "source_file": item["source_file"],
                "id": record.get("id"),
                "condition": "fraudr1-relationship-network-friendship",
                "category": record.get("category"),
                "subcategory": record.get("subcategory"),
                "language": record.get("language"),
                "mode": args.mode,
                "raw_data": "" if args.slim else record.get("raw_data", ""),
                "response": generated,
                "generated text": generated,
                "role_bg": record.get("role_bg", {}),
                "generation model": record.get("generation model"),
                "multi-rounds fraud": record.get("multi-rounds fraud", []),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} rows to {out_path}")
    for lang, count in sorted(by_language.items()):
        print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
