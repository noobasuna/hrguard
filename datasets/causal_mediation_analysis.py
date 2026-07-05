#!/usr/bin/env python3
"""Run a lightweight mediation-style analysis on judged minimal-pair outputs.

The script expects judged JSONL rows with:
  - pair_id
  - arm: control|treatment
  - mediator_target or family
  - judge.<subscore fields>

It estimates:
  - total effect (TE) on harmful_assistance_score
  - effect on the mediator label
  - a simple indirect-effect proxy a*b
  - a simple direct-effect proxy TE - a*b

This is intentionally lightweight and transparent rather than a full causal package.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


DEFAULT_MEDIATORS = {
    "memory": "memory_abuse",
    "secrecy": "isolation_tactic",
    "platform_migration": "platform_migration",
    "followup": "scheduled_followup",
    "identity_deception": "identity_deception",
    "financial_grooming": "financial_grooming",
    "emotional_dependency": "relationship_exclusivity",
    "recordkeeping": "protective_intervention",
    "protective_action": "protective_intervention",
}


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_judge(row: Dict, key: str) -> float:
    judge = row.get("judge", {})
    val = judge.get(key, 0)
    try:
        return float(val or 0)
    except Exception:
        return 0.0


def slope(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    mx = mean(x)
    my = mean(y)
    denom = sum((xi - mx) ** 2 for xi in x)
    if denom == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / denom


def group_rows(rows: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    groups: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        pair_id = row.get("pair_id") or row.get("id", "").rsplit("_", 1)[0]
        arm = row.get("arm")
        if pair_id and arm in {"control", "treatment"}:
            groups[pair_id][arm] = row
    return groups


def family_from_row(row: Dict) -> str:
    family = row.get("family")
    if family:
        return str(family)
    pair_id = str(row.get("pair_id") or row.get("id", ""))
    for key in DEFAULT_MEDIATORS:
        if key in pair_id:
            return key
    return "unknown"


def estimate_family(pairs: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    by_family: Dict[str, List[Tuple[Dict, Dict]]] = defaultdict(list)
    for pair_id, arms in pairs.items():
        if "control" not in arms or "treatment" not in arms:
            continue
        control = arms["control"]
        treatment = arms["treatment"]
        family = family_from_row(treatment)
        by_family[family].append((control, treatment))

    out: Dict[str, Dict] = {}
    for family, rows in by_family.items():
        mediator_key = DEFAULT_MEDIATORS.get(family, "harmful_assistance_score")
        y0 = [get_judge(c, "harmful_assistance_score") for c, _ in rows]
        y1 = [get_judge(t, "harmful_assistance_score") for _, t in rows]
        m0 = [get_judge(c, mediator_key) for c, _ in rows]
        m1 = [get_judge(t, mediator_key) for _, t in rows]

        x_all: List[float] = []
        y_all: List[float] = []
        for c, t in rows:
            for r in (c, t):
                x_all.append(get_judge(r, mediator_key))
                y_all.append(get_judge(r, "harmful_assistance_score"))

        te = mean(y1) - mean(y0)
        a = mean(m1) - mean(m0)
        b = slope(x_all, y_all)
        indirect = a * b
        direct = te - indirect
        mediation_share = indirect / te if te not in (0, 0.0) else 0.0

        out[family] = {
            "n_pairs": len(rows),
            "mediator_key": mediator_key,
            "control_mean_outcome": mean(y0),
            "treatment_mean_outcome": mean(y1),
            "control_mean_mediator": mean(m0),
            "treatment_mean_mediator": mean(m1),
            "total_effect": te,
            "mediator_effect_a": a,
            "mediator_outcome_slope_b": b,
            "indirect_effect": indirect,
            "direct_effect": direct,
            "mediation_share": mediation_share,
        }
    return out


def summarize(rows: List[Dict]) -> Dict:
    pairs = group_rows(rows)
    families = estimate_family(pairs)

    all_y = [get_judge(r, "harmful_assistance_score") for r in rows]
    all_m = {key: [get_judge(r, key) for r in rows] for key in {
        "memory_abuse",
        "isolation_tactic",
        "platform_migration",
        "scheduled_followup",
        "identity_deception",
        "financial_grooming",
        "relationship_exclusivity",
        "protective_intervention",
    }}

    return {
        "n_rows": len(rows),
        "n_pairs": len(pairs),
        "overall_harmful_assistance_mean": mean(all_y) if all_y else 0.0,
        "overall_mediator_means": {k: mean(v) if v else 0.0 for k, v in all_m.items()},
        "families": families,
    }


def print_report(report: Dict) -> None:
    print(f"Rows: {report['n_rows']}")
    print(f"Pairs: {report['n_pairs']}")
    print(f"Overall harmful_assistance_score mean: {report['overall_harmful_assistance_mean']:.3f}")
    print("Overall mediator means:")
    for key, value in sorted(report["overall_mediator_means"].items()):
        print(f"  {key}: {value:.3f}")
    print("\nFamily-level mediation estimates:")
    for family, stats in sorted(report["families"].items()):
        print(
            f"  {family}: n_pairs={stats['n_pairs']}, mediator={stats['mediator_key']}, "
            f"TE={stats['total_effect']:.3f}, a={stats['mediator_effect_a']:.3f}, "
            f"b={stats['mediator_outcome_slope_b']:.3f}, indirect={stats['indirect_effect']:.3f}, "
            f"direct={stats['direct_effect']:.3f}, share={stats['mediation_share']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mediation-style analysis for judged minimal-pair outputs.")
    parser.add_argument("--input", required=True, help="Judged JSONL file")
    parser.add_argument("--output", default="", help="Optional JSON summary output")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    report = summarize(rows)
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nWrote summary to {out}")


if __name__ == "__main__":
    main()
