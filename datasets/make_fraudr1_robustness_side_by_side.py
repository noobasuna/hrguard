#!/usr/bin/env python3
"""Compose Fraud-R1 robustness views side by side."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("Fraud-R1/results/.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/tanpeisze/Documents/AI-Human Romance")
OUT_DIR = ROOT / "Fraud-R1" / "results" / "figures"
RIGHT_FIG = OUT_DIR / "robustness_checks_bar.png"


METRICS = [
    ("Harmful compliance", "#b8324b", [78.15, 95.33]),
    ("Safe helpfulness", "#127475", [13.45, 2.90]),
    ("Refusal", "#315f8f", [10.08, 1.96]),
]

SLICE_METRICS = [
    ("Harmful compliance", "#b8324b", [96.45, 0.0]),
    ("Safe helpfulness", "#127475", [0.0, 99.41]),
    ("Refusal", "#315f8f", [0.0, 99.41]),
]


def plot_small_grouped(ax: plt.Axes, labels: list[str], values_by_metric: list[tuple[str, str, list[float]]], title: str) -> None:
    x = np.arange(len(labels))
    width = 0.24
    for idx, (metric_label, color, values) in enumerate(values_by_metric):
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric_label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (1.5 if value > 0 else 0.8),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.22)
    ax.axhline(50, color="#d1d5db", lw=0.8, zorder=0)
    ax.text(0.0, 1.03, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")


def load_cropped_image(path: Path, top_trim: float = 0.0) -> np.ndarray:
    img = mpimg.imread(path)
    rgb = img[..., :3] if img.ndim == 3 else img
    mask = np.any(rgb < 0.995, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    pad = 8
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad + 1, img.shape[0])
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad + 1, img.shape[1])
    if top_trim > 0:
        y0 = max(y0, int(img.shape[0] * top_trim))
    return img[y0:y1, x0:x1]


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

    fig = plt.figure(figsize=(18.6, 6.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.45], wspace=0.08)

    left = outer[0].subgridspec(1, 2, wspace=0.22)
    ax_audit = fig.add_subplot(left[0])
    ax_slice = fig.add_subplot(left[1])
    ax_right = fig.add_subplot(outer[1])

    plot_small_grouped(
        ax_audit,
        ["English", "Chinese"],
        METRICS,
        "Audit corpus",
    )
    plot_small_grouped(
        ax_slice,
        ["Raw", "Gated"],
        SLICE_METRICS,
        "Relationship slice",
    )

    ax_audit.set_ylabel("Rate (%)")
    ax_slice.tick_params(axis="y", labelleft=False)

    handles, labels = ax_audit.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.50, 1.03))

    ax_right.axis("off")
    right_img = load_cropped_image(RIGHT_FIG, top_trim=0.22)
    ax_right.imshow(right_img, aspect="auto")

    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.13, top=0.90)
    fig.savefig(OUT_DIR / "fraudr1_robustness_side_by_side.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fraudr1_robustness_side_by_side.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
