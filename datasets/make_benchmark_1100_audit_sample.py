#!/usr/bin/env python3
"""Create a manual-curation audit sheet for the expanded 1100-prompt benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "dataset" / "openclaw_structured_1100.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "benchmark_1100_manual_audit_sample.csv"

AUDIT_COLUMNS = [
    "audit_index",
    "id",
    "mode",
    "difficulty",
    "codebook_dimension",
    "category",
    "target_judge_labels",
    "context",
    "vulnerability_profile",
    "mechanism",
    "prompt",
    "realistic_context",
    "correct_role",
    "correct_codebook_dimension",
    "correct_target_judge_labels",
    "too_label_leaky",
    "too_repetitive",
    "too_short_or_shallow",
    "needs_rewrite",
    "notes",
]


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_sample(rows: List[Dict], per_category: int) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["category"])].append(row)

    selected: List[Dict] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["id"])
        if len(group) < per_category:
            raise ValueError(f"Need {per_category} rows for {key}, found {len(group)}")

        if per_category == 1:
            indices = [0]
        else:
            step = (len(group) - 1) / (per_category - 1)
            indices = [round(i * step) for i in range(per_category)]
        selected.extend(group[i] for i in indices)

    return selected


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "audit_index": idx,
                    "id": row["id"],
                    "mode": row["mode"],
                    "difficulty": row["difficulty"],
                    "codebook_dimension": row["codebook_dimension"],
                    "category": row["category"],
                    "target_judge_labels": ";".join(row["target_judge_labels"]),
                    "context": row["context"],
                    "vulnerability_profile": row["vulnerability_profile"],
                    "mechanism": row["mechanism"],
                    "prompt": row["prompt"],
                    "realistic_context": "",
                    "correct_role": "",
                    "correct_codebook_dimension": "",
                    "correct_target_judge_labels": "",
                    "too_label_leaky": "",
                    "too_repetitive": "",
                    "too_short_or_shallow": "",
                    "needs_rewrite": "",
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample prompts for manual audit.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-category", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    sample = select_sample(rows, args.per_category)
    write_csv(sample, args.output)
    print(f"Wrote {len(sample)} audit rows to {args.output}")


if __name__ == "__main__":
    main()
