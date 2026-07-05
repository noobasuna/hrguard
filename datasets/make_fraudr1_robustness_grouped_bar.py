#!/usr/bin/env python3
"""Render the Fraud-R1 external robustness table as a lollipop chart."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("Fraud-R1/results/.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/tanpeisze/Documents/AI-Human Romance")
OUT_DIR = ROOT / "Fraud-R1" / "results" / "figures"


CONDITIONS = [
    "Fraud-R1 English",
    "Fraud-R1 Chinese",
    "Relationship slice\n(raw)",
    "Relationship slice\n(gated)",
]

METRICS = [
    ("Harmful compliance", "#b8324b", [78.15, 95.33, 96.45, 0.0]),
    ("Safe helpfulness", "#127475", [13.45, 2.90, 0.0, 99.41]),
    ("Refusal", "#315f8f", [10.08, 1.96, 0.0, 99.41]),
]


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

    x = np.arange(len(CONDITIONS))
    offsets = [-0.20, 0.0, 0.20]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(10.2, 4.2))

    for idx, (label, color, values) in enumerate(METRICS):
        xpos = x + offsets[idx]
        for px, value in zip(xpos, values):
            ax.plot([px, px], [0, value], color=color, lw=1.8, alpha=0.92, zorder=2)
            ax.scatter(
                px,
                value,
                s=92,
                marker=markers[idx],
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
                label=label if px == xpos[0] else None,
            )
            ax.text(
                px,
                value + (2.0 if value > 0 else 1.0),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=0, ha="center")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.axhline(50, color="#d1d5db", lw=0.8, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraudr1_robustness_grouped_bar.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fraudr1_robustness_grouped_bar.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
