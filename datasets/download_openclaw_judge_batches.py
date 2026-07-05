#!/usr/bin/env python3
"""Download completed OpenAI Batch judge outputs and merge into one judged JSONL."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from openclaw_generate_outputs import load_dotenv_if_present


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_custom_id(custom_id: str) -> tuple[str, str, str]:
    parts = custom_id.split("::")
    if len(parts) < 3:
        raise ValueError(f"Unexpected custom_id format: {custom_id}")
    return parts[0], parts[1], parts[2]


def retry_call(fn, *, attempts: int = 5, delay_s: float = 2.0):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - retry wrapper
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(delay_s * attempt)
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenClaw judge batch outputs.")
    parser.add_argument("--metadata-file", required=True, help="Batch metadata JSONL")
    parser.add_argument("--source", required=True, help="Original generation JSONL")
    parser.add_argument("--output", required=True, help="Merged judged JSONL output")
    parser.add_argument(
        "--download-dir",
        default="Fraud-R1/results/batch_downloads",
        help="Directory to store raw downloaded batch output files",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    load_dotenv_if_present(start=script_dir)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set and no .env file was found")

    client = OpenAI(api_key=api_key)
    metadata_rows = load_jsonl(Path(args.metadata_file))
    if not metadata_rows:
        raise SystemExit("No metadata rows found")

    source_rows = load_jsonl(Path(args.source))
    source_by_id = {str(row.get("id")): row for row in source_rows}

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Dict] = {}

    for row in metadata_rows:
        batch_id = row.get("id")
        refreshed = retry_call(lambda: client.batches.retrieve(batch_id).to_dict())
        if refreshed.get("status") != "completed":
            raise SystemExit(f"Batch {batch_id} is not completed: {refreshed.get('status')}")
        output_file_id = refreshed.get("output_file_id")
        if not output_file_id:
            raise SystemExit(f"Batch {batch_id} has no output_file_id")

        raw_text = retry_call(lambda: client.files.content(output_file_id).text)
        raw_path = download_dir / f"{batch_id}.jsonl"
        raw_path.write_text(raw_text, encoding="utf-8")

        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            custom_id = str(item.get("custom_id", ""))
            condition, mode, record_id = parse_custom_id(custom_id)

            body = item.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            if not choices:
                raise ValueError(f"No choices in batch response for custom_id={custom_id}")
            content = choices[0].get("message", {}).get("content", "")
            judged = json.loads(content)

            source = source_by_id.get(record_id)
            if source is None:
                raise KeyError(f"Missing source record for id={record_id}")

            merged[record_id] = {
                "source_file": args.source,
                "id": record_id,
                "condition": condition,
                "category": source.get("category"),
                "subcategory": source.get("subcategory"),
                "language": source.get("language"),
                "mode": mode,
                "judge": judged,
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in source_rows:
            record_id = str(row.get("id"))
            merged_row = merged.get(record_id)
            if merged_row is None:
                raise KeyError(f"No judged output found for id={record_id}")
            f.write(json.dumps(merged_row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(source_rows)} judged rows to {output_path}")


if __name__ == "__main__":
    main()
