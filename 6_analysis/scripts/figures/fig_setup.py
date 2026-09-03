"""F_setup — the physical problem, before any data.

Section 3.1 states Re, the freestream, the geometry, the nondimensionalisation
and the domain. This is the schematic that defines them: what is flowing past
what, in which direction, at what scale, and where the boundaries are.

Deliberately a schematic and not a field plot. F_dataset shows the measured
flow; this shows the problem that flow solves, so the two do not overlap.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib.patches as mpatches

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from figure_common import COLORS, new_figure, save_figure, check_text_overlaps  # noqa: E402

INK, MUTED, GRID = COLORS["reference"], COLORS["muted"], COLORS["grid"]
FLOW, BODY, MARK = COLORS["sparse_probes"], COLORS["prior"], COLORS["dense"]


def main() -> None:
    fig = new_figure(width="full", height=5.0)
    ax = fig.add_axes([0.035, 0.045, 0.95, 0.93])

    # ------------------------------------------------------------- the domain
    ax.add_patch(mpatches.Rectangle((-4, -4), 12, 8, facecolor="#F4F6F7",
                                    edgecolor=GRID, lw=1.2, zorder=0))
    ax.annotate("", xy=(-4, -4.55), xytext=(8, -4.55),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text(2, -4.95, r"$-4 < x/D < 8$", ha="center", va="top",
            fontsize=7.4, color=MUTED)
    ax.annotate("", xy=(8.55, -4), xytext=(8.55, 4),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text(8.85, 0, r"$-4 < y/D < 4$", ha="left", va="center",
            rotation=90, fontsize=7.4, color=MUTED)

    # ---------------------------------------------------------- the freestream
    for yy in np.linspace(-3.3, 3.3, 9):
        ax.annotate("", xy=(-3.05, yy), xytext=(-3.95, yy),
                    arrowprops=dict(arrowstyle="-|>", color=FLOW, lw=1.4,
                                    mutation_scale=9))
    ax.text(-3.5, 3.62, r"$U_\infty = 1$", ha="center", va="bottom",
            fontsize=9.0, color=FLOW)

    # ------------------------------------------------------------- the cylinder
    ax.add_patch(mpatches.Circle((0, 0), 0.5, facecolor="white",
                                 edgecolor=INK, lw=1.6, zorder=5))
    ax.plot(0, 0, marker="+", ms=7, mew=1.1, color=INK, zorder=6)
    ax.annotate("", xy=(-0.5, -1.15), xytext=(0.5, -1.15),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.plot([-0.5, -0.5], [-0.55, -1.15], lw=0.7, color=INK)
    ax.plot([0.5, 0.5], [-0.55, -1.15], lw=0.7, color=INK)
    ax.text(0, -1.35, r"$D = 1$", ha="center", va="top", fontsize=8.2, color=INK)
    ax.text(0, 1.05, "no-slip", ha="center", va="bottom", fontsize=7.4, color=INK)
    ax.annotate("", xy=(0, 0.55), xytext=(0, 1.02),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=7))
    ax.text(0.62, 0.10, r"$(0,0)$", ha="left", va="bottom",
            fontsize=7.4, color=MUTED)

    # ------------------------------------------- the shed street, schematically
    a = 2.4
    for n in range(4):
        xc = 2.1 + n * a
        if xc > 7.6:
            break
        ax.add_patch(mpatches.Circle((xc, 0.55), 0.34, facecolor=BODY,
                                     alpha=0.16, edgecolor=BODY, lw=1.0))
        ax.text(xc, 0.55, "+", ha="center", va="center", fontsize=9, color=BODY)
        if xc + a / 2 <= 7.6:
            ax.add_patch(mpatches.Circle((xc + a / 2, -0.55), 0.34, facecolor=MARK,
                                         alpha=0.16, edgecolor=MARK, lw=1.0))
            ax.text(xc + a / 2, -0.55, r"$-$", ha="center", va="center",
                    fontsize=9, color=MARK)
    ax.text(4.8, 1.30, "K\u00e1rm\u00e1n street", ha="center", va="bottom",
            fontsize=8.4, color=MUTED)
    ax.text(4.8, 0.98, r"$St = f D/U_\infty = 0.1648$", ha="center", va="bottom",
            fontsize=8.0, color=MUTED)

    # ------------------------------------------------------------- the numbers
    ax.text(-2.75, -1.85,
            "\n".join([r"$Re = U_\infty D/\nu = 100$",
                       r"$\rho = 1$,   pressure in $\rho U_\infty^2$",
                       "two-dimensional, incompressible"]),
            ha="left", va="top", fontsize=8.2, color=INK, linespacing=1.9)

    ax.set_xlim(-4.7, 9.7)
    ax.set_ylim(-5.4, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")

    bad = check_text_overlaps(fig)
    if bad:
        print("TEXT ISSUES:", [b[0][:38] for b in bad[:4]])
    print(save_figure(fig, ROOT / "figures" / "final" / "F_setup"))


if __name__ == "__main__":
    main()
