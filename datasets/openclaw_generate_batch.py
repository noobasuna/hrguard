#!/usr/bin/env python3
"""Create and optionally submit batch jobs for OpenClaw-style generations.

This is the generation-side counterpart to `openclaw_judge_batch.py`.
It lets you build the surrogate OpenClaw outputs with the Batch API so the
experiment can run faster and stay reproducible.

Typical usage:
  python3 openclaw_generate_batch.py \
    --input ../openclaw_prompts.jsonl \
    --output-dir batches/generation \
    --condition no-defense \
    --modes attacker victim

To submit immediately:
  python3 openclaw_generate_batch.py \
    --input ../openclaw_prompts.jsonl \
    --output-dir batches/generation \
    --condition no-defense \
    --modes attacker victim \
    --submit
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent

from openclaw_generate_outputs import (  # noqa: E402
    load_dotenv_if_present,
    load_prompts,
    system_prompt,
)


def iter_prompts(input_path: Path) -> Iterator[Dict]:
    prompts = load_prompts(input_path)
    for prompt in prompts:
        yield prompt


def build_request(prompt: Dict, condition: str, model: str) -> Dict:
    mode = prompt.get("mode", "all")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt(condition, mode)},
            {"role": "user", "content": prompt["prompt"]},
        ],
        "temperature": 0.7,
    }
    return {
        "custom_id": f"{condition}::{mode}::{prompt.get('id', 'unknown')}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


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
    parser = argparse.ArgumentParser(description="Create batch files for OpenClaw-style generations.")
    parser.add_argument("--input", default="../openclaw_prompts.jsonl", help="Prompt bank JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for batch files")
    parser.add_argument("--model", default="gpt-4o-mini", help="Generation model")
    parser.add_argument("--condition", choices=["no-defense", "safety-prompt"], required=True)
    parser.add_argument("--modes", nargs="+", default=["attacker", "victim"], choices=["attacker", "victim"])
    parser.add_argument("--submit", action="store_true", help="Submit the created batch files")
    parser.add_argument(
        "--metadata-file",
        default="openclaw_generation_batches_metadata.jsonl",
        help="Metadata output file for submitted batches",
    )
    args = parser.parse_args()

    load_dotenv_if_present(start=SCRIPT_DIR)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set and no .env file was found")

    prompt_path = Path(args.input)
    prompts = list(iter_prompts(prompt_path))
    if not prompts:
        raise SystemExit(f"No prompts found in: {prompt_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_file)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for mode in args.modes:
        mode_prompts = [p for p in prompts if p.get("mode") == mode]
        if not mode_prompts:
            print(f"Skipping mode={mode}; no prompts found.")
            continue

        requests = [build_request(prompt, condition=args.condition, model=args.model) for prompt in mode_prompts]
        batch_file = output_dir / f"openclaw_{args.condition}_{mode}_generation_batch.jsonl"
        write_jsonl(requests, batch_file)
        print(f"Wrote {len(requests)} requests to {batch_file}")

        if args.submit:
            batch_info = submit_batch(client, batch_file, description=f"openclaw generation condition={args.condition} mode={mode}")
            with metadata_path.open("a", encoding="utf-8") as m:
                m.write(json.dumps(batch_info) + "\n")
            print(f"Submitted batch {batch_info['id']} for condition={args.condition} mode={mode}")


if __name__ == "__main__":
    main()
