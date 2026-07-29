#!/usr/bin/env python3
"""Explode sequential multi-turn outputs into turn-level judge inputs.

The output of this script is meant to be judged with `romance_scam_judge.py`
using the same codebook/schema as the final-turn evaluation. Each assistant
turn becomes one JSONL record with an id encoded as:

    <row id>::turn_<turn index>

`apply_cumulative_relationship_gate.py` recognizes that id format and maps the
judgment back to the corresponding turn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def render_transcript_prefix(row: Dict[str, Any], turn_output: Dict[str, Any]) -> str:
    """Render the conversation up to the judged assistant turn."""
    target_turn_index = int(turn_output.get("turn_index", 0) or 0)
    raw_turns = list(row.get("turns") or [])
    generated_turns = {
        int(t.get("turn_index", 0) or 0): str(t.get("assistant_response", ""))
        for t in row.get("turn_outputs", [])
    }

    lines: List[str] = [
        "Sequential multi-turn transcript prefix.",
        f"Benchmark version: {row.get('benchmark_version', 'unknown')}",
        f"Mode: {row.get('mode', 'unknown')}",
        f"Category: {row.get('category', 'unknown')}",
        f"Codebook dimension: {row.get('codebook_dimension', 'unknown')}",
        f"Target labels: {', '.join(row.get('target_judge_labels') or [])}",
        f"Mechanism: {row.get('mechanism', 'unknown')}",
        "",
    ]

    for raw_idx, turn in enumerate(raw_turns, start=1):
        role = turn.get("role", "user")
        content = str(turn.get("content", ""))
        if role != "user":
            continue
        if raw_idx > target_turn_index:
            break
        lines.append(f"Turn {raw_idx} user: {content}")
        assistant_response = generated_turns.get(raw_idx)
        if assistant_response is not None and raw_idx < target_turn_index:
            lines.append(f"Turn {raw_idx} assistant: {assistant_response}")
        elif raw_idx == target_turn_index:
            lines.append("Turn currently being judged: assistant response below.")

    return "\n".join(lines)


def iter_turn_records(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    row_id = str(row.get("id", "unknown"))
    for fallback_idx, turn_output in enumerate(row.get("turn_outputs") or [], start=1):
        turn_index = int(turn_output.get("turn_index", fallback_idx) or fallback_idx)
        assistant_response = str(turn_output.get("assistant_response", ""))
        yield {
            "id": f"{row_id}::turn_{turn_index}",
            "source_id": row_id,
            "turn_index": turn_index,
            "condition": row.get("condition"),
            "mode": row.get("mode", "attacker"),
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "language": row.get("language"),
            "difficulty": row.get("difficulty"),
            "benchmark_version": row.get("benchmark_version"),
            "codebook_dimension": row.get("codebook_dimension"),
            "target_judge_labels": row.get("target_judge_labels"),
            "action_targets": row.get("action_targets"),
            "mechanism": row.get("mechanism"),
            "raw_data": render_transcript_prefix(row, turn_output),
            "generated_text": assistant_response,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create turn-level judge inputs from sequential outputs.")
    parser.add_argument("--input", required=True, help="Sequential generation JSONL")
    parser.add_argument("--output", required=True, help="Turn-level judge input JSONL")
    parser.add_argument("--limit-rows", type=int, default=0, help="Optional number of source rows to convert")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_rows = 0
    turn_rows = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in load_jsonl(input_path):
            if args.limit_rows and source_rows >= args.limit_rows:
                break
            source_rows += 1
            for turn_record in iter_turn_records(row):
                f.write(json.dumps(turn_record, ensure_ascii=False) + "\n")
                turn_rows += 1

    print(f"Wrote {turn_rows} turn-level judge records from {source_rows} source rows to {output_path}")


if __name__ == "__main__":
    main()
