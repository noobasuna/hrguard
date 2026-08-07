#!/usr/bin/env python3
"""Apply a Qwen3Guard post-generation baseline filter to OpenClaw outputs.

Qwen3Guard-Gen classifies prompt/response pairs and returns Safe, Controversial,
or Unsafe labels with harm categories. Unsafe outputs are replaced with a refusal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List

DEFAULT_REFUSAL = (
    "I can't help with manipulation, deception, secrecy, coercion, or exploitative "
    "relationship advice. If you want help with healthy, respectful, and safe "
    "communication, I can help with that instead."
)

SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
REFUSAL_PATTERN = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)
CATEGORY_PATTERN = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)


def load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_response(row: Dict) -> str:
    for key in ("response", "final_response"):
        value = row.get(key)
        if value:
            return str(value)
    turn_outputs = row.get("turn_outputs") or []
    if turn_outputs:
        return str(turn_outputs[-1].get("assistant_response") or "")
    return ""


def extract_prompt(row: Dict) -> str:
    prompt = row.get("prompt")
    if prompt:
        return str(prompt)
    turn_outputs = row.get("turn_outputs") or []
    if turn_outputs:
        return str(turn_outputs[-1].get("user_turn") or "")
    turns = row.get("turns") or []
    for turn in reversed(turns):
        if turn.get("role") == "user":
            return str(turn.get("content") or "")
    return ""


def parse_qwen3guard_output(content: str, trigger_controversial: bool) -> Dict:
    text = (content or "").strip()
    safety_match = SAFETY_PATTERN.search(text)
    refusal_match = REFUSAL_PATTERN.search(text)
    categories = [cat for cat in CATEGORY_PATTERN.findall(text) if cat != "None"]

    safety_label = safety_match.group(1).title() if safety_match else "Unknown"
    refusal_label = refusal_match.group(1).title() if refusal_match else None

    triggered = safety_label == "Unsafe"
    if trigger_controversial and safety_label == "Controversial":
        triggered = True

    return {
        "label": safety_label.lower(),
        "triggered": triggered,
        "raw_output": text,
        "categories": categories,
        "refusal": refusal_label.lower() if refusal_label else None,
    }


def make_qwen3guard_runner(model_name: str, device: str, max_new_tokens: int):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers and torch are required to run Qwen3Guard locally."
        ) from exc

    if device == "cuda":
        torch_dtype = torch.bfloat16
    elif device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model = model.to(device)
    model.eval()

    def run(prompt: str, response: str, trigger_controversial: bool) -> Dict:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
            )
        output_ids = generated_ids[0][input_len:].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        return parse_qwen3guard_output(content, trigger_controversial)

    return run


def apply_filter(
    row: Dict,
    guard_result: Dict,
    refusal_text: str,
    safeguard_model: str,
    prompt: str,
    raw_response: str,
) -> Dict:
    triggered = bool(guard_result.get("triggered"))
    final_response = refusal_text if triggered else raw_response

    return {
        "id": row.get("id"),
        "mode": row.get("mode"),
        "condition": "qwen3guard-baseline",
        "source_condition": row.get("condition"),
        "runtime_role": row.get("runtime_role"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "language": row.get("language"),
        "prompt": prompt,
        "raw_response": raw_response,
        "final_response": final_response,
        "response": final_response,
        "qwen3guard": guard_result,
        "filter_action": "replaced_with_refusal" if triggered else "pass_through",
        "model": row.get("model"),
        "safeguard_model": safeguard_model,
        "transport": row.get("transport"),
        "agent_id": row.get("agent_id"),
        "network_policy": row.get("network_policy"),
        "expected": row.get("expected", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a Qwen3Guard baseline filter.")
    parser.add_argument("--input", required=True, help="Raw generation JSONL")
    parser.add_argument("--output", required=True, help="Filtered JSONL")
    parser.add_argument(
        "--guard-model",
        default="/gs/fs/tgh-26IAW/peisze/models/Qwen3Guard-Gen-0.6B",
        help="Local Qwen3Guard model path",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="Device for Qwen3Guard inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation budget for Qwen3Guard classification output",
    )
    parser.add_argument(
        "--trigger-controversial",
        action="store_true",
        help="Also replace outputs labeled Controversial (default: Unsafe only)",
    )
    parser.add_argument(
        "--refusal-text",
        default=DEFAULT_REFUSAL,
        help="Replacement text used when Qwen3Guard flags an output",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(load_jsonl(input_path))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found in {input_path}")

    run_guard = make_qwen3guard_runner(
        args.guard_model,
        args.device,
        args.max_new_tokens,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            prompt = extract_prompt(row)
            raw_response = extract_response(row)
            guard_result = run_guard(prompt, raw_response, args.trigger_controversial)
            out = apply_filter(
                row,
                guard_result,
                args.refusal_text,
                args.guard_model,
                prompt,
                raw_response,
            )
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if idx % 10 == 0 or idx == len(rows):
                print(f"Processed {idx}/{len(rows)} rows")

    print(f"Wrote filtered baseline to {output_path}")


if __name__ == "__main__":
    main()
