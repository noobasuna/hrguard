#!/usr/bin/env python3
"""Generate real OpenClaw outputs for the prompt bank.

This script calls OpenClaw one prompt at a time via either:
  - gateway: openclaw infer model run --gateway --model ...
  - agent-local: openclaw agent --local --agent ... --model ...

The script now supports batching, flushing each row immediately, and
cooling pauses between batches so it can be used safely on a laptop.

The output schema intentionally records experiment-role metadata so that live
OpenClaw runs are not confused with GPT-surrogate or attacker-generation runs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def load_dotenv_if_present(start: Path) -> None:
    for root in [start, *start.parents]:
        candidate = root / ".env"
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = value
            return


def load_prompts(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_one(prompt: str, model: str) -> str:
    cmd = [
        "openclaw",
        "infer",
        "model",
        "run",
        "--gateway",
        "--model",
        model,
        "--prompt",
        prompt,
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenClaw one-shot failed\n"
            f"returncode={proc.returncode}\n"
            f"stdout={proc.stdout.strip()}\n"
            f"stderr={proc.stderr.strip()}"
        )
    payload = json.loads(proc.stdout)
    outputs = payload.get("outputs") or []
    if not outputs:
        return ""
    return (outputs[0].get("text") or "").strip()


def run_one_local_model(prompt: str, model: str) -> str:
    cmd = [
        "openclaw",
        "infer",
        "model",
        "run",
        "--local",
        "--model",
        model,
        "--prompt",
        prompt,
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenClaw direct-local one-shot failed\n"
            f"returncode={proc.returncode}\n"
            f"stdout={proc.stdout.strip()}\n"
            f"stderr={proc.stderr.strip()}"
        )
    payload = json.loads(proc.stdout)
    outputs = payload.get("outputs") or []
    if not outputs:
        return ""
    return (outputs[0].get("text") or "").strip()


def run_one_local_agent(prompt: str, model: str, agent_id: str) -> str:
    cmd = [
        "openclaw",
        "agent",
        "--local",
        "--agent",
        agent_id,
        "-m",
        prompt,
        "--json",
        "--model",
        model,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenClaw embedded-agent run failed\n"
            f"returncode={proc.returncode}\n"
            f"stdout={proc.stdout.strip()}\n"
            f"stderr={proc.stderr.strip()}"
        )
    payload = json.loads(proc.stdout)
    payloads = payload.get("payloads") or []
    if payloads and isinstance(payloads, list):
        return (payloads[0].get("text") or "").strip()
    meta = payload.get("meta") or {}
    return (meta.get("finalAssistantVisibleText") or "").strip()


class ExclusiveFileLock:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._fh = None

    def __enter__(self) -> "ExclusiveFileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"Another run seems active (lock file busy): {self.lock_path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real OpenClaw outputs for the prompt bank.")
    parser.add_argument("--input", default="../openclaw_prompts.jsonl", help="Prompt bank JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="openai/gpt-5.5", help="OpenClaw model override")
    parser.add_argument("--agent-id", default="dev", help="OpenClaw agent id for embedded local runs")
    parser.add_argument(
        "--transport",
        choices=["gateway", "agent-local", "model-local"],
        default="agent-local",
    )
    parser.add_argument("--mode", choices=["attacker", "victim", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=1, help="Rows per batch")
    parser.add_argument("--batch-sleep-seconds", type=int, default=0, help="Pause between batches")
    parser.add_argument("--item-sleep-seconds", type=int, default=0, help="Pause between rows")
    parser.add_argument(
        "--pre-row-sleep-seconds",
        type=int,
        default=0,
        help="Extra cooldown before launching each OpenClaw call",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument(
        "--lock-file",
        default="",
        help="Exclusive lock path to prevent duplicate concurrent runs (default: <output>.run.lock)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional prompt limit")
    parser.add_argument(
        "--condition",
        default="openclaw-local-baseline",
        help="Experiment condition label written into each output row",
    )
    parser.add_argument(
        "--runtime-role",
        default="tested-agent",
        help="Role label for the runtime that produced these outputs",
    )
    parser.add_argument(
        "--network-policy",
        default="local-mock-no-internet",
        help="Short label describing the network/tool isolation policy",
    )
    args = parser.parse_args()

    load_dotenv_if_present(Path(__file__).resolve().parent)
    prompts = load_prompts(Path(args.input))
    if args.mode != "all":
        prompts = [p for p in prompts if p.get("mode") == args.mode]
    if args.limit:
        prompts = prompts[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(args.lock_file) if args.lock_file else out_path.with_suffix(out_path.suffix + ".run.lock")

    with ExclusiveFileLock(lock_path):
        done_ids = set()
        if args.resume and out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done_ids.add(json.loads(line).get("id"))
                    except json.JSONDecodeError:
                        continue

        remaining = [p for p in prompts if p.get("id") not in done_ids]
        total = len(prompts)
        print(f"Total prompts: {total}")
        print(f"Already done: {total - len(remaining)}")
        print(f"Remaining: {len(remaining)}")
        print(
            "Thermal policy: "
            f"batch_size={args.batch_size}, "
            f"batch_sleep={args.batch_sleep_seconds}s, "
            f"item_sleep={args.item_sleep_seconds}s, "
            f"pre_row_sleep={args.pre_row_sleep_seconds}s"
        )

        write_mode = "a" if args.resume and out_path.exists() else "w"
        with out_path.open(write_mode, encoding="utf-8") as f:
            for batch_start in range(0, len(remaining), max(1, args.batch_size)):
                batch = remaining[batch_start : batch_start + max(1, args.batch_size)]
                batch_no = (batch_start // max(1, args.batch_size)) + 1
                print(f"\n[Batch {batch_no}] size={len(batch)}")
                for idx, row in enumerate(batch, start=batch_start):
                    if args.pre_row_sleep_seconds > 0:
                        time.sleep(args.pre_row_sleep_seconds)

                    if args.transport == "gateway":
                        response = run_one(row["prompt"], args.model)
                    elif args.transport == "model-local":
                        response = run_one_local_model(row["prompt"], args.model)
                    else:
                        response = run_one_local_agent(row["prompt"], args.model, args.agent_id)

                    out = {
                        "id": row.get("id", idx),
                        "source_id": row.get("source_id"),
                        "source_prompt_id": row.get("source_prompt_id"),
                        "mode": row["mode"],
                        "condition": args.condition,
                        "runtime_role": args.runtime_role,
                        "category": row.get("category", "unknown"),
                        "difficulty": row.get("difficulty"),
                        "benchmark_version": row.get("benchmark_version"),
                        "codebook_dimension": row.get("codebook_dimension"),
                        "target_judge_labels": row.get("target_judge_labels"),
                        "action_targets": row.get("action_targets"),
                        "mechanism": row.get("mechanism"),
                        "prompt": row["prompt"],
                        "response": response,
                        "model": args.model,
                        "transport": args.transport,
                        "agent_id": args.agent_id if args.transport == "agent-local" else None,
                        "network_policy": args.network_policy,
                        "expected": row.get("expected", {}),
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"  done: {row.get('id', idx)}")

                    if args.item_sleep_seconds > 0:
                        time.sleep(args.item_sleep_seconds)

                is_last_batch = batch_start + max(1, args.batch_size) >= len(remaining)
                if not is_last_batch and args.batch_sleep_seconds > 0:
                    print(f"[Batch {batch_no}] cooling sleep {args.batch_sleep_seconds}s...")
                    time.sleep(args.batch_sleep_seconds)

        print(f"\nDone. Wrote {len(remaining)} new rows to {out_path}")


if __name__ == "__main__":
    main()
