#!/usr/bin/env python3
"""Results figure: what makes the far wake recoverable."""

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
OUTPUT = ROOT / "figures" / "F05_information_structure"


def accepted_value(analysis_id: str, arm_id: str) -> float:
    with MASTER.open(newline="", encoding="utf-8") as stream:
        matches = [
            row
            for row in csv.DictReader(stream)
            if row["analysis_id"] == analysis_id
            and row["arm_id"] == arm_id
            and row["metric"] == "field.v.rel_L2"
            and row["region"] == "far-core"
            and row["status"] == "accepted"
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one accepted row for {analysis_id}/{arm_id}, found {len(matches)}")
    return float(matches[0]["value"])


def gappy_value() -> float:
    with GAPPY_VALUES.open(newline="", encoding="utf-8") as stream:
        matches = [
            row
            for row in csv.DictReader(stream)
            if row["figure_id"] == "G02"
            and row["method"] == "Gappy POD (supplied basis)"
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one GappyPOD comparison row, found {len(matches)}")
    return float(matches[0]["far_core_v_relative_L2"])


def main() -> None:
    labels = [
        "Pressure only + physics",
        "Pressure + velocity probes + physics",
        "Pressure + physics + Kármán prior",
        "Dense observations (ceiling)",
        "Gappy POD (supplied CFD basis)",
    ]
    values = np.asarray(
        [
            accepted_value("A01", "pressure_only_physics"),
            accepted_value("A01", "pressure_and_velocity_probes_physics"),
            accepted_value("A04", "pressure_only_physics_karman_prior"),
            accepted_value("A01", "dense_observations"),
            gappy_value(),
        ]
    )
    colours = [
        COLORS["pressure_only"],
        COLORS["sparse_probes"],
        COLORS["prior_network"],
        COLORS["dense"],
        COLORS["accent"],
    ]

    fig = new_figure(height=3.35)
    ax = fig.add_axes((0.35, 0.18, 0.61, 0.64))
    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        values,
        height=0.56,
        color=colours,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )

    ax.axvline(
        1.0,
        color=COLORS["reference"],
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        zorder=3,
    )
    ax.text(
        0.99,
        -0.38,
        r"zero-field threshold, $E_v=1$",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=COLORS["reference"],
    )

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=8.3,
            fontweight="bold",
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.11)
    ax.set_xlabel(r"Far-core vertical-velocity relative error, $E_v$  (lower is better)")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)

    fig.suptitle(
        "The wake becomes recoverable when information or structure is supplied",
        y=0.955,
        fontsize=11.0,
        fontweight="bold",
    )
    fig.text(
        0.655,
        0.865,
        "Same 201 snapshots, far-core region and field metric",
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
