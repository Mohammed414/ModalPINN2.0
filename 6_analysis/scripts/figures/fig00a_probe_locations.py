"""F0b-1 - velocity-probe locations, standalone single-column figure.

Split out of the combined F0b so the two views (probes at domain scale, taps
at cylinder scale) can sit side by side as two \\includegraphics calls in
LaTeX rather than one wide combined PNG.

Carries the same region shading as F0a so the reader can see which evaluation
regions the probes actually cover. The final section is requested at x/D = 3;
there are no probe sections farther downstream.

Input:  derived/a00_geometry.npz
Output: figures/final/F00a_probe_locations.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figure_common import (  # noqa: E402
    COLORS, REGION_COLORS, check_text_overlaps, domain_axes, draw_cylinder,
    new_figure, region_patch, save_figure,
)

GEOM = np.load(ROOT / "derived" / "a00_geometry.npz", allow_pickle=True)
PX, PY = GEOM["probe_x"], GEOM["probe_y"]
XMIN, XMAX, YMIN, YMAX = -4.0, 8.0, -4.0, 4.0

sections = np.array(sorted({round(float(v)) for v in GEOM["probe_target_x"]}))
per_section = [int(np.sum(np.abs(PX - s) < 0.5)) for s in sections]
assert sum(per_section) == PX.size, (per_section, PX.size)

fig = new_figure(width="single", height=3.55)
ax = fig.add_subplot(111)
domain_axes(ax)
fig.subplots_adjust(left=0.155, right=0.97, bottom=0.135, top=0.90)

region_patch(ax, "band", extent=(XMIN, 0.0, YMIN, YMAX),
             color=REGION_COLORS["other"], alpha=0.40, zorder=1)
region_patch(ax, "band", extent=(0.0, 3.0, YMIN, YMAX),
             color=REGION_COLORS["near-wake"], alpha=0.24, zorder=1)
region_patch(ax, "band", extent=(3.0, XMAX, YMIN, YMAX),
             color=REGION_COLORS["far-wake"], alpha=0.22, zorder=1)
region_patch(ax, "disk", radius=0.75, color=REGION_COLORS["near-cylinder"],
             alpha=0.55, zorder=2)
draw_cylinder(ax, radius=0.5, zorder=3)

ax.scatter(PX, PY, s=16, marker="o", facecolor=COLORS["sparse_probes"],
           edgecolor="white", linewidth=0.4, zorder=6)
ax.set_title(f"{PX.size} velocity probes, {sections.size} sections",
             fontsize=8.6, loc="left")

ax.text(float(sections[0]), YMAX - 0.30, f"$x/D$ = {sections[0]:g}",
        fontsize=6.8, ha="center", va="top", color=COLORS["sparse_probes"],
        zorder=7, bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                            edgecolor="none", alpha=0.85))
ax.annotate("", xy=(1.0, YMAX - 0.46), xytext=(3.0, YMAX - 0.46),
            arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25",
                            color=COLORS["sparse_probes"], linewidth=0.6),
            zorder=7)
ax.text(2.0, YMAX - 0.26, "$x/D$ = 1, 2, 3", fontsize=6.8, ha="center",
        va="top", color=COLORS["sparse_probes"], zorder=7,
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                  edgecolor="none", alpha=0.85))
ax.text(XMIN + 0.18, YMIN + 0.24, f"{per_section[0]} probes per section",
        fontsize=6.6, ha="left", va="bottom", color=COLORS["muted"], zorder=7)
ax.text(5.25, -2.9, "no probe sections\ndownstream of $x/D=3$",
        fontsize=6.8, ha="center", va="center", color="#00614A",
        linespacing=1.25, zorder=7)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT COLLISIONS:", bad)
out = save_figure(fig, ROOT / "figures" / "final" / "F00a_probe_locations")
print(out)
print("probes/section:", dict(zip(sections.tolist(), per_section)))
