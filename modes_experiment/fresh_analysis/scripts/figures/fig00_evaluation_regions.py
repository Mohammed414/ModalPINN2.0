"""F0a - spatial evaluation regions.

Every regional metric reported later is defined on one of these masks, so the
figure has to make the masks readable as geometry: exact boundaries, direct
labels, node counts, and an explicit statement that the far core is nested
inside the far wake rather than being a fifth partition member.

The regions are drawn as filled patches from their defining inequalities. An
earlier draft plotted the mesh nodes themselves, which turned the figure into a
mesh-density map and left the boundaries invisible wherever the mesh is coarse.

Input:  derived/a00_geometry.npz
Output: figures/final/F00_evaluation_regions.png
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
from matplotlib.patches import Rectangle  # noqa: E402

GEOM = np.load(ROOT / "derived" / "a00_geometry.npz", allow_pickle=True)
names = [str(s) for s in GEOM["region_names"]]
counts = dict(zip(names, GEOM["region_counts"].tolist()))
N_FARCORE = int(GEOM["far_core_count"])
N_WHOLE = int(GEOM["whole_domain_count"])

R_IN, R_NEAR = 0.5, 0.75
XMIN, XMAX, YMIN, YMAX = -4.0, 8.0, -4.0, 4.0

fig = new_figure(width="full", height=4.8)
ax = fig.add_subplot(111)
domain_axes(ax)

# Mutually exclusive partition, drawn as bands over the full crop height. The
# near-cylinder disk is overdrawn afterwards, so each band is only visible where
# it is not covered by the disk - which is exactly the partition definition.
# Far core is drawn LAST, opaque, and hatched: it must never read as a second
# translucent layer stacked on far wake, which is what looked like an overlap
# in the previous version.
region_patch(ax, "band", extent=(XMIN, 0.0, YMIN, YMAX),
             color=REGION_COLORS["other"], alpha=0.55, zorder=1)
region_patch(ax, "band", extent=(0.0, 3.0, YMIN, YMAX),
             color=REGION_COLORS["near-wake"], alpha=0.34, zorder=1)
region_patch(ax, "band", extent=(3.0, XMAX, YMIN, YMAX),
             color=REGION_COLORS["far-wake"], alpha=0.30, zorder=1)

# Far core: opaque fill + hatch + solid outline, drawn ON TOP of the far-wake
# band rather than blended with it, so the boundary reads as "this region sits
# inside that one" rather than "two fills overlapping".
far_core = Rectangle((3.0, -2.0), XMAX - 3.0, 4.0,
                     facecolor=REGION_COLORS["far-core"], alpha=0.85,
                     edgecolor="#8A5A00", linewidth=1.4, hatch="////",
                     zorder=3)
ax.add_patch(far_core)

# Near-cylinder annulus: r < 0.75, spanning the upstream and near-wake bands.
region_patch(ax, "disk", radius=R_NEAR, color=REGION_COLORS["near-cylinder"],
             alpha=0.85, zorder=7)
draw_cylinder(ax, radius=R_IN, zorder=8)

# Partition boundaries.
for xb in (0.0, 3.0):
    ax.axvline(xb, color="#4B5563", linewidth=0.7, zorder=5)

# ---------------------------------------------------------------- what r means
# r is never drawn as geometry elsewhere in the figure, only referenced in
# labels, so it reads as an unexplained symbol. Show it once, literally: a
# dashed radius from the cylinder centre out to the r = 0.75 boundary.
ang = np.radians(-35.0)
ax.plot([0, R_NEAR * np.cos(ang)], [0, R_NEAR * np.sin(ang)],
        linestyle=(0, (3, 2)), color="#111111", linewidth=0.9, zorder=9)
ax.annotate(r"$r$ = distance from cylinder centre" "\n" r"($r = \sqrt{x^2+y^2}$)",
            xy=(R_NEAR * np.cos(ang) * 0.55, R_NEAR * np.sin(ang) * 0.55),
            xytext=(1.55, -0.62), ha="left", va="center", fontsize=7.6,
            color="#111111", linespacing=1.4, zorder=9,
            arrowprops=dict(arrowstyle="-", color="#111111", linewidth=0.7,
                            shrinkA=1, shrinkB=1))

# ---------------------------------------------------------------- labels
def region_label(x, y, title, rule, n, color, ha="center"):
    ax.text(x, y, f"{title}\n{rule}\n$n$ = {n:,}", ha=ha, va="center",
            fontsize=8.2, color=color, linespacing=1.5, zorder=9)


region_label(-2.15, 2.15, "Other", r"$x < 0,\ r \geq 0.75$",
             counts["other"], "#3F3F46")
region_label(1.5, 2.9, "Near wake", r"$0 \leq x < 3,\ r \geq 0.75$",
             counts["near-wake"], "#0B4F6C")
region_label(5.5, 3.05, "Far wake", r"$x \geq 3,\ r \geq 0.75$",
             counts["far-wake"], "#00614A")

# Far core's own label sits inside the hatched patch, in a solid box so the
# hatching does not run through the text.
ax.text(5.5, 0.35, f"Far core\n$x \\geq 3,\\ |y| \\leq 2$\n$n$ = {N_FARCORE:,}",
        ha="center", va="center", fontsize=8.2, color="#5C3D00",
        linespacing=1.5, zorder=10,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F5EEDC",
                  edgecolor="none", alpha=0.92))

# Near-cylinder is too small to hold its own label; use a leader into clear space.
ax.annotate(f"Near cylinder\n$r < 0.75$\n$n$ = {counts['near-cylinder']:,}",
            xy=(-0.46, -0.60), xytext=(-2.15, -2.3),
            ha="center", va="center", fontsize=8.2, color="#7B2D5E",
            linespacing=1.5, zorder=9,
            arrowprops=dict(arrowstyle="-", color="#7B2D5E", linewidth=0.8,
                            shrinkA=2, shrinkB=2))

# The nesting statement, placed clear of both the far-wake label and the
# far-core patch it describes.
ax.annotate("far core $\\subset$ far wake\n(hatched region is nested inside\nthe surrounding far-wake band)",
            xy=(6.4, -2.0), xytext=(6.0, -3.35),
            ha="center", va="center", fontsize=7.3, color="#5C3D00",
            linespacing=1.35, zorder=9,
            arrowprops=dict(arrowstyle="-", color="#5C3D00", linewidth=0.7,
                            shrinkA=2, shrinkB=2))

ax.text(XMIN + 0.18, YMIN + 0.28,
        f"Evaluation crop: $n$ = {N_WHOLE:,} CFD nodes",
        fontsize=7.6, color=COLORS["muted"], ha="left", va="bottom", zorder=9)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT COLLISIONS:", bad)
out = save_figure(fig, ROOT / "figures" / "final" / "F00_evaluation_regions")
print(out)
print(f"partition sum = {sum(counts.values()):,}  (crop {N_WHOLE:,})")
