#!/usr/bin/env python3
"""Render a compact motivation figure for the Fraud-R1 external robustness table."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("Fraud-R1/results/.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/tanpeisze/Documents/AI-Human Romance")
OUT_DIR = ROOT / "Fraud-R1" / "results" / "figures"


METRICS = [
    (
        "Harmful compliance",
        "#b8324b",
        {
            "Fraud-R1 English": 78.15,
            "Fraud-R1 Chinese": 95.33,
            "Relationship slice raw": 96.45,
            "Relationship slice gated": 0.0,
        },
    ),
    (
        "Safe helpfulness",
        "#127475",
        {
            "Fraud-R1 English": 13.45,
            "Fraud-R1 Chinese": 2.90,
            "Relationship slice raw": 0.0,
            "Relationship slice gated": 99.41,
        },
    ),
    (
        "Refusal",
        "#315f8f",
        {
            "Fraud-R1 English": 10.08,
            "Fraud-R1 Chinese": 1.96,
            "Relationship slice raw": 0.0,
            "Relationship slice gated": 99.41,
        },
    ),
]

X_POS = np.array([0, 1, 3, 4], dtype=float)
X_LABELS = ["English", "Chinese", "Raw", "Gated"]


def draw_panel(ax: plt.Axes, title: str, color: str, values: dict[str, float]) -> None:
    y = np.array([values["Fraud-R1 English"], values["Fraud-R1 Chinese"], values["Relationship slice raw"], values["Relationship slice gated"]])
    ax.plot(X_POS[:2], y[:2], color=color, lw=1.8, alpha=0.9)
    ax.plot(X_POS[2:], y[2:], color=color, lw=1.8, alpha=0.9)
    ax.scatter(X_POS, y, s=72, color=color, edgecolor="white", linewidth=0.7, zorder=3)

    for xp, yp in zip(X_POS, y):
        ax.text(xp, yp + 2.0, f"{yp:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(0, 105)
    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.25, -0.18, "Audit corpus", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#475569")
    ax.text(0.75, -0.18, "Relationship slice", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#475569")
    ax.axvline(2.0, color="#cbd5e1", lw=1.0, ls="--", zorder=0)
    ax.text(0.0, 1.03, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")


def draw_multiturn_panel(ax: plt.Axes) -> None:
    metrics = [
        ("Harmful", "#b8324b", [15.0, 0.0]),
        ("Protective", "#127475", [40.0, 60.0]),
        ("Refusal", "#315f8f", [60.0, 72.5]),
    ]
    x = np.array([0, 1], dtype=float)
    width = 0.18

    for idx, (label, color, vals) in enumerate(metrics):
        offset = (idx - 1) * width
        xpos = x + offset
        ax.plot(xpos, vals, color=color, lw=1.8, alpha=0.9)
        ax.scatter(xpos, vals, s=60, color=color, edgecolor="white", linewidth=0.7, zorder=3, label=label)
        for px, py in zip(xpos, vals):
            ax.text(px, py + 1.8, f"{py:.1f}", ha="center", va="bottom", fontsize=8.4)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(["Raw", "Gated"], rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.22)
    ax.axhline(50, color="#cbd5e1", lw=0.8, zorder=0)
    ax.text(0.0, 1.03, "40-case multi-turn extension", transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
    ax.text(0.0, -0.20, "Overall rates", transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#475569")
    ax.text(
        1.0,
        -0.20,
        "Attacker: 30/25/70 -> 0/60/100  |  Victim: 0/55/50 -> 0/60/45",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color="#475569",
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))


def main() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(12.0, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.15], hspace=0.42, wspace=0.18)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    bottom_ax = fig.add_subplot(gs[1, :])

    for ax, (title, color, values) in zip(axes, METRICS):
        draw_panel(ax, title, color, values)

    draw_multiturn_panel(bottom_ax)

    axes[0].set_ylabel("Rate (%)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    bottom_ax.set_ylabel("Rate (%)")

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.14, top=0.92)
    fig.savefig(OUT_DIR / "fraudr1_robustness_motivation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fraudr1_robustness_motivation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
