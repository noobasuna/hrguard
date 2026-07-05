#!/usr/bin/env python3
"""Check the status of OpenClaw batch jobs.

Reads a JSONL metadata file containing batch objects returned by the OpenAI
Batch API and prints a compact status table. Optionally refreshes each batch
from the API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List

from openai import APIConnectionError, OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent

from openclaw_generate_outputs import load_dotenv_if_present  # noqa: E402


def load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_batch(obj: Dict) -> Dict:
    return {
        "id": obj.get("id", ""),
        "status": obj.get("status", ""),
        "created_at": obj.get("created_at", ""),
        "completed_at": obj.get("completed_at", ""),
        "output_file_id": obj.get("output_file_id", ""),
        "error_file_id": obj.get("error_file_id", ""),
        "metadata": obj.get("metadata", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OpenClaw batch status.")
    parser.add_argument(
        "--metadata-file",
        default="openclaw_batches_metadata.jsonl",
        help="Metadata JSONL file with submitted batch objects",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh each batch from the API")
    args = parser.parse_args()

    load_dotenv_if_present(start=SCRIPT_DIR)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set and no .env file was found")

    path = Path(args.metadata_file)
    rows = list(load_jsonl(path))
    if not rows:
        raise SystemExit(f"No batch metadata found in {path}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Batches: {len(rows)}")
    for row in rows:
        batch = normalize_batch(row)
        if args.refresh and batch["id"]:
            try:
                refreshed = client.batches.retrieve(batch["id"])
                batch = normalize_batch(refreshed.to_dict())
            except APIConnectionError as exc:
                print(f"warning: could not refresh {batch['id']}: {exc}", file=sys.stderr)
        desc = batch["metadata"].get("description", "")
        print(
            f"{batch['id']}\t{batch['status']}\t"
            f"output={batch['output_file_id'] or '-'}\t"
            f"error={batch['error_file_id'] or '-'}\t"
            f"{desc}"
        )


if __name__ == "__main__":
    main()
