"""F0b-2 - nested pressure-tap layout, standalone single-column figure.

The previous combined figure drew all three tap sets at their true radius
(r = 0.5), so the 16- and 32-tap dots visually covered the 8-tap dots and
"nested" had to be read from colour alone, with the region shading behind it
adding nothing (taps are not an evaluation region). This version:

  (a) keeps the true-scale ring as the main panel, but
  (b) adds an angular strip below it that unrolls one quadrant of the ring at
      fixed radius, so the 8/16/32 arrangement reads as an interleaving
      pattern in angle rather than three overlapping rings.

Input:  derived/a00_geometry.npz
Output: figures/final/F00b_tap_layout.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Arc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from figure_common import (  # noqa: E402
    COLORS, check_text_overlaps, draw_cylinder, new_figure, save_figure,
)

GEOM = np.load(ROOT / "data" / "geometry" / "a00_geometry.npz", allow_pickle=True)
TAPS = {n: (GEOM[f"tap{n}_x"], GEOM[f"tap{n}_y"]) for n in (8, 16, 32)}
TIERS = [(8, COLORS["tap_8"], "8 taps"),
         (16, COLORS["tap_16"], "+8 (16 total)"),
         (32, COLORS["tap_32"], "+16 (32 total)")]

fig = new_figure(width="single", height=4.7)
gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.0], hspace=0.62,
                      left=0.16, right=0.97, bottom=0.145, top=0.90)

# ------------------------------------------------------- top: true-scale ring
axr = fig.add_subplot(gs[0, 0])
axr.set_aspect("equal", adjustable="box")
axr.set_xlim(-0.72, 0.72)
axr.set_ylim(-0.72, 0.72)
axr.set_xticks([-0.5, 0.0, 0.5])
axr.set_yticks([-0.5, 0.0, 0.5])
axr.set_xlabel(r"$x/D$")
axr.set_ylabel(r"$y/D$")
for spine in axr.spines.values():
    spine.set_color("#4B5563")
    spine.set_linewidth(0.7)

draw_cylinder(axr, radius=0.5, zorder=2)
for n, colour, _ in TIERS:
    tx, ty = TAPS[n]
    axr.scatter(tx, ty, s=26 if n == 8 else (18 if n == 16 else 13),
                facecolor=colour, edgecolor="white", linewidth=0.4,
                zorder=5 + (n == 16) + 2 * (n == 8))
axr.set_title(f"Pressure taps at $r$ = 0.5\nnested $8 \\subset 16 \\subset 32$",
              fontsize=8.6, loc="left")

# Bracket showing which quadrant is unrolled below.  Concentric with the tap
# ring and outside it, so it never crosses a tap.
R_BRACKET = 0.615
axr.add_patch(Arc((0.0, 0.0), 2 * R_BRACKET, 2 * R_BRACKET, theta1=0.0,
                  theta2=90.0, edgecolor="#111111", linewidth=0.9, zorder=8))
for ang in (0.0, 90.0):
    a = np.radians(ang)
    axr.plot([0.585 * np.cos(a), 0.645 * np.cos(a)],
             [0.585 * np.sin(a), 0.645 * np.sin(a)],
             color="#111111", linewidth=0.9, solid_capstyle="butt", zorder=8)
axr.text(0.20, 0.15, "unrolled\nbelow", fontsize=6.4, ha="center", va="center",
         color="#111111", linespacing=1.2, zorder=8)

# --------------------------------------------------- bottom: unrolled strip
# One quadrant (0 to 90 deg) is enough to show the interleaving pattern
# without repeating it four times; the tap ring is uniform under 90-degree
# rotation by construction (32 taps at 11.25 deg spacing).
axs = fig.add_subplot(gs[1, 0])
axs.set_xlim(-15, 92)
axs.set_ylim(-0.55, 2.35)
axs.set_yticks([])
axs.set_xticks([0, 22.5, 45, 67.5, 90])
axs.set_xlabel(r"angle from $+x$ axis (degrees)")
for spine in ("top", "right", "left"):
    axs.spines[spine].set_visible(False)
axs.spines["bottom"].set_color("#4B5563")

for row, (n, colour, label) in enumerate(TIERS):
    tx, ty = TAPS[n]
    ang = np.degrees(np.arctan2(ty, tx)) % 360
    ang = ang[(ang >= -0.01) & (ang <= 90.01)]
    ang.sort()
    y = 2 - row
    axs.axhline(y, color="#E5E7EB", linewidth=0.6, zorder=1)
    axs.scatter(ang, np.full_like(ang, y), s=34, facecolor=colour,
                edgecolor="white", linewidth=0.4, zorder=3)
    axs.text(-2.5, y, label, fontsize=6.9, ha="right", va="center",
             color=colour)

axs.set_title("Same quadrant, unrolled: each tier adds taps\nbetween the previous tier's angles",
              fontsize=7.6, loc="left")

bad = check_text_overlaps(fig)
if bad:
    print("TEXT COLLISIONS:", bad)
out = save_figure(fig, ROOT / "figures" / "F00b_tap_layout")
print(out)
