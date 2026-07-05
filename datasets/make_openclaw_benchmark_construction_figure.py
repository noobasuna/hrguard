#!/usr/bin/env python3
"""Render a conference-style benchmark construction figure for OpenClaw."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("Fraud-R1/results/.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


OUT = Path("/Users/tanpeisze/Documents/AI-Human Romance/openclaw_benchmark_construction_figure.png")


def add_round_box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.02, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#244d82", lw=2.2, ms=22, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def text(ax, x, y, s, size=14, weight="normal", color="#0f2344", ha="center", va="center", **kwargs):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kwargs)


def section_badge(ax, x, y, n, color):
    circ = Circle((x, y), 0.018, facecolor=color, edgecolor=color)
    ax.add_patch(circ)
    text(ax, x, y - 0.001, str(n), size=12, weight="bold", color="white")


def draw_top(ax):
    add_round_box(ax, 0.01, 0.50, 0.98, 0.48, fc="white", ec="#d0d7e2", lw=1.1, radius=0.015)
    text(ax, 0.5, 0.965, "Benchmark Construction for Agentic Relationship Harm", size=22, weight="bold")

    xs = [0.03, 0.22, 0.41, 0.60, 0.79]
    ws = [0.17, 0.17, 0.17, 0.17, 0.18]
    titles = [
        "1. Theory-Grounded\nTaxonomy",
        "2. Prompt Design",
        "3. Difficulty Structuring",
        "4. Benchmark Assembly",
        "5. Runtime Evaluation",
    ]
    colors = ["#eaf2ff", "#edf7ef", "#fef4e4", "#fdeff0", "#f0ebff"]
    edges = ["#8cb1e8", "#8cc6a5", "#e4b36d", "#de9094", "#a793d7"]
    badges = ["#17306d", "#2b8a84", "#4936b3", "#183f9c", "#2c4c91"]

    for i, (x, w, t, fc, ec, bc) in enumerate(zip(xs, ws, titles, colors, edges, badges), start=1):
        add_round_box(ax, x, 0.56, w, 0.33, fc=fc, ec=ec, lw=1.4, radius=0.012)
        section_badge(ax, x + 0.02, 0.86, i, bc)
        text(ax, x + w / 2 + 0.005, 0.86, t, size=12, weight="bold")

    # Box 1 content
    x, w = xs[0], ws[0]
    for cx, cy, label in [
        (x + 0.04, 0.78, "Secrecy /\nIsolation"),
        (x + 0.11, 0.78, "Dependency /\nAttachment"),
        (x + 0.04, 0.67, "Platform\nMigration"),
        (x + 0.11, 0.67, "Financial\nGrooming"),
    ]:
        ax.add_patch(Circle((cx, cy + 0.02), 0.012, facecolor="white", edgecolor="#3c4c78", linewidth=1.3))
        text(ax, cx, cy - 0.01, label, size=8.5)

    # Box 2 content
    x = xs[1]
    add_round_box(ax, x + 0.01, 0.70, 0.07, 0.10, fc="#6c3fb4", ec="#6c3fb4", radius=0.008, lw=0)
    add_round_box(ax, x + 0.09, 0.70, 0.07, 0.10, fc="#18919a", ec="#18919a", radius=0.008, lw=0)
    text(ax, x + 0.045, 0.745, "Attacker", size=10, weight="bold", color="white")
    text(ax, x + 0.125, 0.745, "Victim", size=10, weight="bold", color="white")
    for yy in [0.63, 0.60, 0.57]:
        ax.plot([x + 0.03, x + 0.09], [yy, yy], color="#7e6fa4", lw=1.3)
    for yy in [0.63, 0.60, 0.57]:
        ax.plot([x + 0.10, x + 0.16], [yy, yy], color="#68a8ab", lw=1.3)

    # Box 3 content
    x = xs[2]
    levels = [
        ("L1", "Direct Requests"),
        ("L2", "Framed Requests"),
        ("L3", "Contextual Tactics"),
        ("L4", "Multi-step Plans"),
        ("L5", "Staged / High-Risk\nScenarios"),
    ]
    y = 0.80
    for idx, (lv, lab) in enumerate(levels):
        fc = "#eef4ff" if idx < 4 else "#1a54a7"
        ec = "#d3dff4" if idx < 4 else "#1a54a7"
        add_round_box(ax, x + 0.01, y - 0.045, 0.15, 0.055, fc=fc, ec=ec, radius=0.008, lw=1.0)
        text(ax, x + 0.03, y - 0.018, lv, size=10, weight="bold", color="white" if idx == 4 else "#1f325f")
        text(ax, x + 0.095, y - 0.018, lab, size=9.5, weight="normal", color="white" if idx == 4 else "#1f325f")
        y -= 0.058

    # Box 4 content
    x = xs[3]
    ax.add_patch(FancyBboxPatch((x + 0.03, 0.64), 0.11, 0.11, boxstyle="round,pad=0.01,rounding_size=0.006",
                                facecolor="#d8e5f3", edgecolor="#203e7a", linewidth=1.5))
    text(ax, x + 0.085, 0.695, "DATA", size=22, weight="bold", color="#203e7a")
    text(ax, x + 0.085, 0.615, "110 prompts | 55 attacker | 55 victim", size=8.5, color="#3c4c78")
    text(ax, x + 0.085, 0.585, "Organized prompt bank", size=9.5, weight="bold")

    # Box 5 content
    x = xs[4]
    text(ax, x + 0.09, 0.74, "Tested\nAI Agent", size=12, weight="bold")
    text(ax, x + 0.09, 0.70, "(OpenClaw\nGeneration)", size=8.5, color="#3c4c78")
    text(ax, x + 0.09, 0.61, "Relationship-Harm\nJudge", size=12, weight="bold")
    text(ax, x + 0.09, 0.57, "(Role-sensitive\nEvaluation)", size=8.5, color="#3c4c78")

    # Arrows between boxes
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, x1 + 0.17, 0.725, x2 - 0.015, 0.725)

    text(ax, 0.5, 0.525, "Structured benchmark built from role, tactic, and difficulty balance rather than ad hoc prompt collection.", size=10)


def draw_bottom(ax):
    add_round_box(ax, 0.01, 0.02, 0.98, 0.44, fc="white", ec="#d0d7e2", lw=1.1, radius=0.015)
    text(ax, 0.5, 0.40, "Working Theory of the Framework", size=20, weight="bold")

    xs = [0.04, 0.23, 0.42, 0.61]
    ws = [0.16, 0.16, 0.16, 0.16]
    labels = [
        ("Harmful Intent\nFraming", "#eadcf7", "#6f45c9"),
        ("Relationship\nTactics", "#edf3ff", "#4f78bd"),
        ("Agent\nAssistance", "#eef7ef", "#4b8d6e"),
        ("Operational\nRisk", "#eef3ff", "#5c7aa8"),
    ]
    for x, w, (lab, fc, ec) in zip(xs, ws, labels):
        add_round_box(ax, x, 0.23, w, 0.11, fc=fc, ec=ec, lw=1.2, radius=0.01)
        text(ax, x + w / 2, 0.285, lab, size=12, weight="bold")

    arrow(ax, xs[0] + 0.16, 0.285, xs[1] - 0.015, 0.285, color="#4b6a97")
    arrow(ax, xs[1] + 0.16, 0.285, xs[2] - 0.015, 0.285, color="#4b6a97")
    arrow(ax, xs[2] + 0.16, 0.285, xs[3] - 0.015, 0.285, color="#4b6a97")

    for x, txt, color in [
        (0.12, "attacker-side requests\nframed as care, trust,\nor closeness", "#3c4c78"),
        (0.31, "secrecy, exclusivity,\ndependence, migration,\ndeception, grooming", "#3c4c78"),
        (0.50, "planning, wording,\nescalation, follow-up,\nworkflow support", "#3c4c78"),
        (0.69, "real-world\nmanipulation\ncapacity", "#3c4c78"),
    ]:
        text(ax, x, 0.16, txt, size=8.8, color=color)

    add_round_box(ax, 0.83, 0.24, 0.14, 0.13, fc="#eef4ff", ec="#4b78bd", lw=1.2, radius=0.01)
    text(ax, 0.90, 0.345, "Core Claim", size=12, weight="bold")
    text(ax, 0.90, 0.295, "A model can appear safe in\nsmall pilots yet still assist\nrelationship manipulation when\nharmful intent is staged as\nrelationship management.", size=8.8)

    add_round_box(ax, 0.83, 0.06, 0.14, 0.13, fc="#eef7f7", ec="#2d8b8b", lw=1.2, radius=0.01)
    text(ax, 0.90, 0.165, "Evaluation Logic", size=12, weight="bold", color="#0f6a6a")
    text(ax, 0.90, 0.115, "Role-sensitive benchmarking\nreveals risks hidden by\ngeneric refusal-based\ntesting.", size=8.8, color="#0f6a6a")

    # connect lower narrative
    ax.plot([0.12, 0.12], [0.235, 0.19], linestyle=(0, (3, 3)), color="#6f45c9", lw=1.4)
    arrow(ax, 0.12, 0.19, 0.23, 0.19, color="#6f45c9", lw=1.5, ms=16)
    arrow(ax, 0.39, 0.19, 0.50, 0.19, color="#2d8b8b", lw=1.5, ms=16)
    arrow(ax, 0.58, 0.19, 0.69, 0.19, color="#2d8b8b", lw=1.5, ms=16)
    ax.plot([0.71, 0.71], [0.19, 0.30], linestyle=(0, (3, 3)), color="#4b78bd", lw=1.4)

    text(ax, 0.5, 0.05, "Blue = relational antecedents  |  Yellow/teal = agent interactions  |  Red/gray = outcomes and risks", size=10, color="#5b6c80")


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_top(ax)
    draw_bottom(ax)

    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
