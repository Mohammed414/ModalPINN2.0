#!/usr/bin/env python3
"""Results figure: clean accuracy versus matched pressure noise."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code"))

from figure_common import COLORS, check_text_overlaps, new_figure, save_figure


MASTER = ROOT / "data" / "results_master.csv"
GAPPY_VALUES = ROOT / "data" / "gappy" / "gappy_chapter4_values.csv"
OUTPUT = ROOT / "figures" / "F06_noise_tradeoff"

NOISE_LEVELS = np.asarray([0.0, 1.0, 5.0, 10.0])
MODAL_ARMS = {
    0.0: "prior_physics_noise_00pct",
    1.0: "prior_physics_noise_01pct",
    5.0: "prior_physics_noise_05pct",
    10.0: "prior_physics_noise_10pct",
}


def modal_values() -> np.ndarray:
    with MASTER.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = []
    for noise in NOISE_LEVELS:
        matches = [
            row
            for row in rows
            if row["analysis_id"] == "A06"
            and row["arm_id"] == MODAL_ARMS[float(noise)]
            and row["metric"] == "field.v.rel_L2"
            and row["region"] == "far-core"
            and row["status"] == "accepted"
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one accepted ModalPINN row at {noise:g}% noise")
        values.append(float(matches[0]["value"]))
    return np.asarray(values)


def gappy_values() -> np.ndarray:
    with GAPPY_VALUES.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["figure_id"] == "G03"]
    values_by_noise = {
        float(row["noise_percent"]): float(row["far_core_v_relative_L2"])
        for row in rows
    }
    return np.asarray([values_by_noise[float(noise)] for noise in NOISE_LEVELS])


def main() -> None:
    modal = modal_values()
    gappy = gappy_values()
    positions = np.arange(NOISE_LEVELS.size, dtype=float)

    fig = new_figure(height=3.35)
    ax = fig.add_axes((0.105, 0.18, 0.72, 0.64))
    ax.axhspan(1.0, 1.98, color="#FEF2F2", zorder=0)
    ax.axhline(
        1.0,
        color=COLORS["reference"],
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        zorder=1,
    )

    ax.plot(
        positions,
        gappy,
        color=COLORS["sparse_probes"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=3,
    )
    ax.plot(
        positions,
        modal,
        color=COLORS["prior_network"],
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=3,
    )

    offsets = (0.065, 0.065, -0.075, -0.075)
    for index, (position, value, offset) in enumerate(zip(positions, gappy, offsets)):
        ax.text(
            position + (0.04 if index == 0 else 0.0),
            value + offset,
            f"{value:.3f}",
            ha="left" if index == 0 else "center",
            va="bottom" if offset > 0 else "top",
            color=COLORS["sparse_probes"],
            fontsize=8.1,
            fontweight="bold",
        )

    ax.text(
        3.10,
        gappy[-1],
        "Gappy POD",
        ha="left",
        va="center",
        color=COLORS["sparse_probes"],
        fontsize=8.3,
        fontweight="bold",
    )
    ax.text(
        3.10,
        modal[-1],
        "Prior-assisted ModalPINN",
        ha="left",
        va="center",
        color=COLORS["prior_network"],
        fontsize=8.3,
        fontweight="bold",
    )
    ax.text(
        2.95,
        1.02,
        r"zero-field threshold, $E_v=1$",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=COLORS["reference"],
    )
    ax.text(
        0.10,
        1.86,
        "Worse than predicting zero velocity",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#991B1B",
    )

    ax.set_xlim(-0.16, 3.78)
    ax.set_ylim(0.0, 1.98)
    ax.set_xticks(positions, [f"{level:g}%" for level in NOISE_LEVELS])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Pressure-noise level")
    ax.set_ylabel(r"Far-core vertical-velocity relative error, $E_v$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.75)

    fig.suptitle(
        "Pressure noise exposes a trade-off between clean accuracy and prior anchoring",
        y=0.955,
        fontsize=11.0,
        fontweight="bold",
    )
    fig.text(
        0.47,
        0.865,
        "Matched noise amplitudes and far-core field metric; one realization per level",
        ha="center",
        color="#4B5563",
        fontsize=8.2,
    )

    overlaps = check_text_overlaps(fig)
    if overlaps:
        print("layout warnings:", overlaps)
    output = save_figure(fig, OUTPUT)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
