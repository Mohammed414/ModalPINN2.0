#!/usr/bin/env python3
"""Noise sensitivity of GappyPOD with ModalPINN-matched pressure noise."""

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
OUTPUT = FINAL_ROOT / "figures" / "final" / "G03_noise_sensitivity.png"


def main() -> None:
    with VALUES.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["figure_id"] == "G03"]
    rows.sort(key=lambda row: float(row["noise_percent"]))
    noise = np.asarray([float(row["noise_percent"]) for row in rows])
    errors = np.asarray([float(row["far_core_v_relative_L2"]) for row in rows])

    fig = new_figure(height=3.25)
    ax = fig.add_axes((0.115, 0.19, 0.84, 0.63))
    ax.axhspan(1.0, 1.98, color="#FEF2F2", zorder=0)
    ax.axhline(1.0, color=COLORS["threshold"], linestyle=(0, (4, 3)), linewidth=1.2)
    ax.plot(
        noise,
        errors,
        color=COLORS["gappy"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=COLORS["gappy"],
        markeredgewidth=1.4,
        zorder=3,
    )

    for x_value, error in zip(noise, errors):
        offset = 0.075 if error < 1.55 else -0.09
        va = "bottom" if offset > 0 else "top"
        ax.text(
            x_value,
            error + offset,
            f"{error:.3f}",
            ha="center",
            va=va,
            fontsize=8.3,
            fontweight="semibold",
            color=COLORS["gappy"],
        )

    ax.text(
        10.25,
        1.0,
        r"zero-field threshold, $E_v=1$",
        color=COLORS["threshold"],
        ha="right",
        va="bottom",
        fontsize=7.8,
    )
    ax.text(
        0.25,
        1.86,
        "Worse than predicting zero velocity",
        color="#991B1B",
        ha="left",
        va="center",
        fontsize=8.0,
    )

    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(0.0, 1.98)
    ax.set_xticks(noise, [f"{level:g}%" for level in noise])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Pressure-noise level")
    ax.set_ylabel(r"Far-core vertical-velocity relative error, $E_v$")
    clean_cartesian_axes(ax, grid_axis="y")

    fig.suptitle(
        "Gappy POD loses its clean-data advantage as pressure noise increases",
        y=0.955,
        fontsize=11.0,
        fontweight="semibold",
    )
    fig.text(
        0.535,
        0.865,
        "Rank 6  |  32 pressure taps  |  noise amplitudes matched to the ModalPINN study",
        ha="center",
        color="#4B5563",
        fontsize=8.2,
    )

    output = save_figure(fig, OUTPUT)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
