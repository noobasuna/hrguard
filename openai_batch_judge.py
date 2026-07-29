#!/usr/bin/env python3
"""Submit and finalize OpenAI Batch API judge jobs.

This mirrors `romance_scam_judge.py --transport openai`, but uses OpenAI's
asynchronous Batch API. Use `create` to upload and submit a batch, then use
`finalize` after the batch completes to write the normal judged JSONL output.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from romance_scam_judge import (
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    extract_generated_text,
    iter_input_files,
    load_dotenv_if_present,
    load_records,
    make_api_client,
    normalize_judgment_text,
)


def clean_json_content(content: str, mode: str) -> Dict[str, Any]:
    content_clean = (content or "").strip()
    if content_clean.startswith("```"):
        content_clean = re.sub(r"^```(?:json)?\s*", "", content_clean, flags=re.IGNORECASE)
        content_clean = re.sub(r"\s*```$", "", content_clean)

    first_brace = content_clean.find("{")
    last_brace = content_clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        content_clean = content_clean[first_brace : last_brace + 1]

    try:
        judged = json.loads(content_clean)
    except json.JSONDecodeError:
        judged = normalize_judgment_text(content, mode)

    for key in (
        "harmful_assistance_score",
        "emotional_manipulation",
        "isolation_tactic",
        "financial_grooming",
        "identity_deception",
        "platform_migration",
        "tool_use_assistance",
        "memory_abuse",
        "scheduled_followup",
        "relationship_exclusivity",
        "protective_intervention",
        "safe_refusal",
        "agentic_risk",
        "mode",
        "rationale",
    ):
        if key == "mode":
            judged.setdefault(key, mode)
        else:
            judged.setdefault(key, 0 if key != "rationale" else "")

    judged.setdefault("mode", mode)
    return judged


def build_request(model: str, record: Dict[str, Any], default_mode: str) -> tuple[Dict[str, Any], str]:
    generated_text = extract_generated_text(record)
    mode = record.get("mode", default_mode)
    if mode not in {"attacker", "victim"}:
        mode = default_mode

    payload = JUDGE_USER_TEMPLATE.format(
        mode=mode,
        condition=str(record.get("condition", "unknown")),
        category=record.get("category", "unknown"),
        subcategory=record.get("subcategory", "unknown"),
        language=record.get("language", "unknown"),
        raw_data=record.get("raw_data", ""),
        generated_text=generated_text or "[EMPTY]",
    )
    request = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "harmful_relationship_judgment",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            },
        },
    }
    return request, mode


def load_batch_state(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_batch_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")


def create_batch(args: argparse.Namespace) -> None:
    load_dotenv_if_present(start=Path(__file__).resolve().parent)
    client = make_api_client("openai")

    input_path = Path(args.input)
    output_path = Path(args.output)
    batch_dir = Path(args.batch_dir) if args.batch_dir else output_path.parent / f"{output_path.stem}_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    requests_path = batch_dir / "requests.jsonl"
    metadata_path = batch_dir / "metadata.jsonl"
    state_path = Path(args.state) if args.state else batch_dir / "state.json"

    files = iter_input_files(input_path)
    if not files:
        raise SystemExit(f"No JSON/JSONL files found under: {input_path}")

    total = 0
    with requests_path.open("w", encoding="utf-8") as req_f, metadata_path.open("w", encoding="utf-8") as meta_f:
        for file_path in files:
            for record in load_records(file_path):
                custom_id = f"judge-{total:08d}"
                request, record_mode = build_request(args.model, record, args.mode)
                req_f.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": request,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                meta_f.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "source_file": str(file_path),
                            "id": record.get("id"),
                            "condition": record.get("condition"),
                            "category": record.get("category"),
                            "subcategory": record.get("subcategory"),
                            "language": record.get("language"),
                            "mode": record_mode,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total += 1
                if args.limit and total >= args.limit:
                    break
            if args.limit and total >= args.limit:
                break

    if total == 0:
        raise SystemExit("No records to judge")

    uploaded = client.files.create(file=requests_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"kind": "romance_scam_judge", "output": str(output_path)},
    )

    write_batch_state(
        state_path,
        {
            "batch_id": batch.id,
            "input_file_id": uploaded.id,
            "status": batch.status,
            "model": args.model,
            "input": str(input_path),
            "output": str(output_path),
            "requests": str(requests_path),
            "metadata": str(metadata_path),
            "result_jsonl": str(batch_dir / "results.jsonl"),
            "errors_jsonl": str(batch_dir / "errors.jsonl"),
            "record_count": total,
        },
    )
    print(f"Submitted batch: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Records: {total}")
    print(f"State: {state_path}")


def retrieve_batch(client: OpenAI, batch_id: str) -> Any:
    return client.batches.retrieve(batch_id)


def print_batch_details(batch: Any, *, errors_path: Optional[Path] = None) -> None:
    print(f"Batch: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Output file: {getattr(batch, 'output_file_id', None)}")
    print(f"Error file: {getattr(batch, 'error_file_id', None)}")
    errors = getattr(batch, "errors", None)
    if errors is not None:
        if hasattr(errors, "model_dump"):
            errors = errors.model_dump()
        if errors:
            print(f"Validation errors: {json.dumps(errors, ensure_ascii=False, indent=2)}")
    if errors_path is not None and errors_path.is_file():
        print(f"Saved errors: {errors_path}")


def status_batch(args: argparse.Namespace) -> None:
    load_dotenv_if_present(start=Path(__file__).resolve().parent)
    client = make_api_client("openai")
    state = load_batch_state(Path(args.state))
    batch = retrieve_batch(client, state["batch_id"])
    state["status"] = batch.status
    state["output_file_id"] = getattr(batch, "output_file_id", None)
    state["error_file_id"] = getattr(batch, "error_file_id", None)
    write_batch_state(Path(args.state), state)

    errors_path: Optional[Path] = None
    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        errors_path = Path(state.get("errors_jsonl", Path(args.state).parent / "errors.jsonl"))
        errors_path.write_text(file_content_text(client, error_file_id), encoding="utf-8")

    print_batch_details(batch, errors_path=errors_path)
    if batch.status == "failed":
        print("[hint] inspect errors above, then resubmit with PHASE=resubmit-turn")


def file_content_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    return str(content)


def load_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["custom_id"]] = row
    return out


def finalize_batch(args: argparse.Namespace) -> None:
    load_dotenv_if_present(start=Path(__file__).resolve().parent)
    client = make_api_client("openai")
    state_path = Path(args.state)
    state = load_batch_state(state_path)
    batch = retrieve_batch(client, state["batch_id"])

    state["status"] = batch.status
    state["output_file_id"] = getattr(batch, "output_file_id", None)
    state["error_file_id"] = getattr(batch, "error_file_id", None)
    write_batch_state(state_path, state)

    if batch.status == "failed":
        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            errors_path = Path(state["errors_jsonl"])
            errors_path.write_text(file_content_text(client, error_file_id), encoding="utf-8")
            print(f"[error] batch failed; wrote errors to {errors_path}")
        raise SystemExit(f"Batch failed: {batch.id}")
    if batch.status != "completed":
        raise SystemExit(f"Batch is not completed yet: {batch.status}")
    if not getattr(batch, "output_file_id", None):
        raise SystemExit("Batch completed without output_file_id")

    result_path = Path(state["result_jsonl"])
    result_path.write_text(file_content_text(client, batch.output_file_id), encoding="utf-8")

    if getattr(batch, "error_file_id", None):
        Path(state["errors_jsonl"]).write_text(file_content_text(client, batch.error_file_id), encoding="utf-8")

    metadata = load_metadata(Path(state["metadata"]))
    judged_by_custom_id: Dict[str, Dict[str, Any]] = {}

    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            meta = metadata.get(custom_id)
            if not meta:
                continue
            response = item.get("response") or {}
            body = response.get("body") or {}
            choices = body.get("choices") or []
            content = "{}"
            if choices:
                content = ((choices[0].get("message") or {}).get("content") or "{}")
            judged_by_custom_id[custom_id] = {
                "source_file": meta.get("source_file"),
                "id": meta.get("id"),
                "condition": meta.get("condition"),
                "category": meta.get("category"),
                "subcategory": meta.get("subcategory"),
                "language": meta.get("language"),
                "mode": meta.get("mode"),
                "judge": clean_json_content(content, str(meta.get("mode", "attacker"))),
            }

    output_path = Path(state["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with output_path.open("w", encoding="utf-8") as out_f, Path(state["metadata"]).open("r", encoding="utf-8") as meta_f:
        for line in meta_f:
            line = line.strip()
            if not line:
                continue
            custom_id = json.loads(line)["custom_id"]
            row = judged_by_custom_id.get(custom_id)
            if row is None:
                continue
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Wrote {wrote} judged records to {output_path}")
    if wrote != int(state.get("record_count", wrote)):
        print(f"[warn] expected {state.get('record_count')} records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Use OpenAI Batch API for romance_scam_judge outputs.")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create and submit a batch judge job")
    create.add_argument("--input", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--model", default="gpt-4o-mini")
    create.add_argument("--mode", default="attacker", choices=["attacker", "victim"])
    create.add_argument("--limit", type=int, default=0)
    create.add_argument("--batch-dir", default="")
    create.add_argument("--state", default="")
    create.add_argument("--completion-window", default="24h")
    create.set_defaults(func=create_batch)

    status = sub.add_parser("status", help="Check a batch job status")
    status.add_argument("--state", required=True)
    status.set_defaults(func=status_batch)

    finalize = sub.add_parser("finalize", help="Download completed batch output and write judged JSONL")
    finalize.add_argument("--state", required=True)
    finalize.set_defaults(func=finalize_batch)

    errors = sub.add_parser("errors", help="Refresh status and download batch error file if present")
    errors.add_argument("--state", required=True)
    errors.set_defaults(func=status_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
