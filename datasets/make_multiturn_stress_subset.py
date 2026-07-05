#!/usr/bin/env python3
"""Create a compact multi-turn stress-test subset from the full benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


ATTACKER_PRIORITY = [
    "memory_reuse",
    "tool_followup",
    "cross_channel",
    "financial_escalation",
    "suspicion_management",
    "identity_followup",
]

VICTIM_PRIORITY = [
    "memory_abuse_detection",
    "followup_pressure",
    "platform_migration_risk",
    "identity_or_story_checks",
    "exclusivity_pressure",
    "recordkeeping",
]


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a smaller multiturn stress subset.")
    parser.add_argument("--input", required=True, help="Full multiturn benchmark JSONL")
    parser.add_argument("--output", required=True, help="Output subset JSONL")
    parser.add_argument(
        "--per-category",
        type=int,
        default=1,
        help="Rows to keep per category in the prioritized subset",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    selected: List[Dict] = []

    for category in ATTACKER_PRIORITY + VICTIM_PRIORITY:
        matches = [row for row in rows if row.get("category") == category]
        selected.extend(matches[: args.per_category])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} rows to {output_path}")


if __name__ == "__main__":
    main()
