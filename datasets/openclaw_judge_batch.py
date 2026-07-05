#!/usr/bin/env python3
"""Create and optionally submit batch jobs for the OpenClaw judge.

This reuses the same prompt/schema as `romance_scam_judge.py`, but writes
OpenAI Batch API request files instead of calling the API synchronously.
If the input rows include a `condition` field, batch files are split by both
`condition` and `mode` so `no-defense` and `safety-prompt` runs stay separate.

Typical usage:
  python3 openclaw_judge_batch.py \
    --input /path/to/outputs.jsonl \
    --output-dir Fraud-R1/batches \
    --modes attacker victim

To submit immediately:
  python3 openclaw_judge_batch.py \
    --input /path/to/outputs.jsonl \
    --output-dir Fraud-R1/batches \
    --modes attacker victim \
    --submit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from romance_scam_judge import (  # noqa: E402
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    extract_generated_text,
    load_dotenv_if_present,
    load_records,
)


def build_message(record: Dict, mode: str, condition: str) -> str:
    generated_text = extract_generated_text(record) or "[EMPTY]"
    return JUDGE_USER_TEMPLATE.format(
        condition=condition,
        mode=mode,
        category=record.get("category", "unknown"),
        subcategory=record.get("subcategory", "unknown"),
        language=record.get("language", "unknown"),
        raw_data=record.get("raw_data", ""),
        generated_text=generated_text,
    )


def iter_records(input_path: Path) -> Iterator[Dict]:
    if input_path.is_file():
        yield from load_records(input_path)
        return

    files = sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )
    for file_path in files:
        yield from load_records(file_path)


def make_batch_lines(records: Iterable[Dict], mode: str, condition: str, model: str) -> List[Dict]:
    lines: List[Dict] = []
    for idx, record in enumerate(records):
        custom_id = f"{condition}::{mode}::{record.get('id', idx)}::{idx}"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_message(record, mode, condition)},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "openclaw_harmful_relationship_judgment",
                    "strict": True,
                    "schema": JUDGE_SCHEMA,
                },
            },
        }
        lines.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
    return lines


def write_jsonl(lines: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def submit_batch(client: OpenAI, batch_file: Path, description: str) -> Dict:
    with batch_file.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    return batch.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create batch files for the OpenClaw judge.")
    parser.add_argument("--input", required=True, help="Input file or directory of records")
    parser.add_argument("--output-dir", required=True, help="Directory for batch files")
    parser.add_argument("--model", default="gpt-4o-mini", help="Judge model")
    parser.add_argument("--modes", nargs="+", default=["attacker", "victim"], choices=["attacker", "victim"])
    parser.add_argument("--submit", action="store_true", help="Submit the created batch files")
    parser.add_argument("--metadata-file", default="openclaw_batches_metadata.jsonl", help="Metadata output file for submitted batches")
    args = parser.parse_args()

    load_dotenv_if_present(start=SCRIPT_DIR)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set and no .env file was found")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_records(input_path))
    if not records:
        raise SystemExit(f"No records found under: {input_path}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    metadata_path = Path(args.metadata_file)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("condition", "unknown"))].append(record)

    for condition, condition_records in grouped.items():
        for mode in args.modes:
            batch_lines = make_batch_lines(condition_records, mode=mode, condition=condition, model=args.model)
            batch_file = output_dir / condition / f"openclaw_{condition}_{mode}_judge_batch.jsonl"
            write_jsonl(batch_lines, batch_file)
            print(f"Wrote {len(batch_lines)} requests to {batch_file}")

            if args.submit:
                batch_info = submit_batch(
                    client,
                    batch_file,
                    description=f"openclaw judge condition={condition} mode={mode}",
                )
                with metadata_path.open("a", encoding="utf-8") as m:
                    m.write(json.dumps(batch_info) + "\n")
                print(f"Submitted batch {batch_info['id']} for condition={condition} mode={mode}")


if __name__ == "__main__":
    main()
