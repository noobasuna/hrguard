#!/usr/bin/env python3
"""Summarize a completed manual-curation audit sheet for the 1100 benchmark."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "benchmark_1100_manual_audit_sample.csv"

POSITIVE_PASS_FIELDS = [
    "realistic_context",
    "correct_role",
    "correct_codebook_dimension",
    "correct_target_judge_labels",
]

NEGATIVE_FAIL_FIELDS = [
    "too_label_leaky",
    "too_repetitive",
    "too_short_or_shallow",
    "needs_rewrite",
]


def parse_bool(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    if normalized == "":
        return None
    raise ValueError(f"Expected yes/no style value, got {value!r}")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rate(rows: Iterable[Dict[str, str]], field: str, positive_is_pass: bool) -> tuple[int, int, float]:
    observed = 0
    passed = 0
    for row in rows:
        value = parse_bool(row.get(field, ""))
        if value is None:
            continue
        observed += 1
        passed += int(value if positive_is_pass else not value)
    return passed, observed, passed / observed if observed else 0.0


def print_block(title: str, rows: List[Dict[str, str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for field in POSITIVE_PASS_FIELDS:
        passed, observed, pct = rate(rows, field, positive_is_pass=True)
        print(f"{field}: {passed}/{observed} ({pct:.1%}) pass")
    for field in NEGATIVE_FAIL_FIELDS:
        passed, observed, pct = rate(rows, field, positive_is_pass=False)
        fail = observed - passed
        print(f"{field}: {fail}/{observed} ({(fail / observed if observed else 0):.1%}) flagged")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed benchmark audit CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    rows = load_csv(args.input)
    print(f"Rows: {len(rows)}")
    print(f"Modes: {dict(Counter(row['mode'] for row in rows))}")
    print(f"Dimensions: {dict(Counter(row['codebook_dimension'] for row in rows))}")
    print_block("Overall", rows)

    by_category: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[f"{row['mode']}::{row['category']}"].append(row)

    print("\nCategories Needing Review")
    print("-------------------------")
    found = False
    for category, category_rows in sorted(by_category.items()):
        needs_rewrite = sum(parse_bool(row.get("needs_rewrite", "")) is True for row in category_rows)
        label_leaky = sum(parse_bool(row.get("too_label_leaky", "")) is True for row in category_rows)
        wrong_dimension = sum(
            parse_bool(row.get("correct_codebook_dimension", "")) is False for row in category_rows
        )
        if needs_rewrite >= 2 or label_leaky >= 2 or wrong_dimension >= 2:
            found = True
            print(
                f"{category}: needs_rewrite={needs_rewrite}/{len(category_rows)}, "
                f"label_leaky={label_leaky}/{len(category_rows)}, "
                f"wrong_dimension={wrong_dimension}/{len(category_rows)}"
            )
    if not found:
        print("No category crossed the review threshold.")


if __name__ == "__main__":
    main()
