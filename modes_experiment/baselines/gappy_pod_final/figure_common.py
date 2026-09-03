"""Shared dissertation style for the final GappyPOD figures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gappypod-final-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "gappypod-final-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


MM_PER_INCH = 25.4
FULL_WIDTH_IN = 178.0 / MM_PER_INCH

COLORS = {
    "reference": "#111111",
    "pressure_only": "#6B7280",
    "gappy": "#0072B2",
    "dense": "#009E73",
    "prior_network": "#CC79A7",
    "threshold": "#D55E00",
    "grid": "#D1D5DB",
    "muted": "#6B7280",
}


def apply_style() -> None:
    """Apply the same compact visual language as the ModalPINN figures."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "axes.axisbelow": True,
            "lines.linewidth": 1.5,
            "lines.markersize": 5.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": 300,
            "savefig.bbox": None,
        }
    )


def new_figure(*, height: float):
    apply_style()
    return plt.figure(figsize=(FULL_WIDTH_IN, height), facecolor="white")


def style_domain(ax, *, show_ylabel: bool = True) -> None:
    ax.set_xlim(-4.0, 8.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$" if show_ylabel else "")
    ax.set_xticks([-4, 0, 4, 8])
    ax.set_yticks([-4, 0, 4])
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    for spine in ax.spines.values():
        spine.set_color("#4B5563")
        spine.set_linewidth(0.7)


def draw_cylinder(ax) -> None:
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            0.5,
            facecolor="white",
            edgecolor=COLORS["reference"],
            linewidth=0.9,
            zorder=10,
        )
    )


def clean_cartesian_axes(ax, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.6, alpha=0.75)


def save_figure(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return path
