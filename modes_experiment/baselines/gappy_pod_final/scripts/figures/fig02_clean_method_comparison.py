#!/usr/bin/env python3
"""Focused clean-data comparison using the matched far-core metric."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
FINAL_ROOT = HERE.parents[2]
sys.path.insert(0, str(FINAL_ROOT))

from figure_common import COLORS, clean_cartesian_axes, new_figure, save_figure


VALUES = FINAL_ROOT / "results" / "chapter4_values.csv"
OUTPUT = FINAL_ROOT / "figures" / "final" / "G02_clean_method_comparison.png"

ORDER = (
    "Pressure only + physics",
    "Pressure + physics + Karman prior",
    "Dense observations",
    "Gappy POD (supplied basis)",
)
LABELS = {
    "Pressure only + physics": "Pressure only + physics",
    "Pressure + physics + Karman prior": "Pressure + physics + Kármán prior",
    "Dense observations": "Dense observations (ceiling)",
    "Gappy POD (supplied basis)": "Gappy POD (supplied CFD basis)",
}
BAR_COLORS = {
    "Pressure only + physics": COLORS["pressure_only"],
    "Pressure + physics + Karman prior": COLORS["prior_network"],
    "Dense observations": COLORS["dense"],
    "Gappy POD (supplied basis)": COLORS["gappy"],
}


def main() -> None:
    with VALUES.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["figure_id"] == "G02"]
    values_by_method = {
        row["method"]: float(row["far_core_v_relative_L2"]) for row in rows
    }
    values = np.asarray([values_by_method[method] for method in ORDER])

    fig = new_figure(height=3.15)
    ax = fig.add_axes((0.33, 0.19, 0.63, 0.63))
    y = np.arange(len(ORDER))
    bars = ax.barh(
        y,
        values,
        height=0.56,
        color=[BAR_COLORS[method] for method in ORDER],
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axvline(1.0, color=COLORS["threshold"], linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(
        0.99,
        -0.35,
        r"zero-field threshold, $E_v=1$",
        color=COLORS["threshold"],
        ha="right",
        va="bottom",
        fontsize=7.8,
    )
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=8.3,
            fontweight="semibold",
        )

    ax.set_yticks(y, [LABELS[method] for method in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.10)
    ax.set_xlabel(r"Far-core vertical-velocity relative error, $E_v$  (lower is better)")
    ax.tick_params(axis="y", length=0, pad=7)
    clean_cartesian_axes(ax, grid_axis="x")

    fig.suptitle(
        "A supplied flow subspace closes the clean-data reconstruction gap",
        y=0.955,
        fontsize=11.0,
        fontweight="semibold",
    )
    fig.text(
        0.645,
        0.865,
        "Matched 201-snapshot far-core evaluation  |  32 pressure taps for sparse cases",
        ha="center",
        color="#4B5563",
        fontsize=8.2,
    )

    output = save_figure(fig, OUTPUT)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
