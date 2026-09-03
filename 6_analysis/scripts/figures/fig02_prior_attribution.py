"""F02 - who supplies the oscillating wake: the prior or the network?

One sentence this figure must make true:
    Wall pressure recovers the wake near the cylinder but not downstream; the
    analytical prior does the opposite; and the trained hybrid tracks whichever
    source is better in each region rather than beating either.

The previous version plotted every metric for every region and every method as
126 bars, which shows the data but states no claim. This version plots ONE
metric (the v1 relative L2, the quantity the prior is built to fix) against
streamwise position, so the crossover between the two information sources is
the visible subject.

The upstream "other" region is deliberately excluded here: its reference norm
is near zero, so its ratio metrics are dominated by that denominator rather
than by reconstruction quality. It is the subject of F02b instead.

Input:  derived/a04_prior_only_metrics.json,
        derived/a04_prior_attribution_metrics.json
Output: figures/final/F02_prior_attribution.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figure_common import (  # noqa: E402
    COLORS, check_text_overlaps, new_figure, save_figure,
)

D = ROOT / "derived"
PRIOR_ONLY = json.loads((D / "a04_prior_only_metrics.json").read_text())
ATTRIB = json.loads((D / "a04_prior_attribution_metrics.json").read_text())
BASE = ATTRIB["models"]["arm1_baseline"]
HYBRID = ATTRIB["models"]["arm15_v1_radial_trust"]

# Wake regions ordered by distance downstream. far-core is nested inside
# far-wake, so plotting both as peers would double-count; far-core is the
# tighter of the two and is the one the project reports, so it is used and
# far-wake is dropped from the ordered axis.
REGIONS = ["near-cylinder", "near-wake", "far-core"]
NICE = {"near-cylinder": "Near cylinder\n$r<0.75$",
        "near-wake": "Near wake\n$0\\leq x<3$",
        "far-core": "Far core\n$x\\geq3,\\ |y|\\leq2$"}

SERIES = [
    ("Wall pressure + physics\n(no prior)", BASE, COLORS["pressure_only"], "o", "-"),
    ("Kármán prior alone\n(no network)", PRIOR_ONLY, COLORS["prior"], "s", "--"),
    ("Pressure + physics\n+ Kármán prior", HYBRID, COLORS["prior_network"], "D", "-"),
]


def v1_l2(source, region):
    return float(source["v1_mode_metrics"][region]["rel_L2"])


fig = new_figure(width="full", height=4.15)
gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.42,
                      left=0.085, right=0.975, bottom=0.24, top=0.74)

# ------------------------------------------------------------------ panel a
# The crossover. One metric, three methods, plotted against position.
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(len(REGIONS))

# "Predicting zero everywhere" scores exactly 1.0 on this metric, so it is the
# line that separates a real reconstruction from no reconstruction at all.
ax.axhline(1.0, color=COLORS["reference"], lw=0.8, ls=(0, (4, 2.5)), zorder=2)
ax.text(-0.22, 1.025, "predicting zero everywhere scores 1.0",
        fontsize=6.6, color=COLORS["reference"], va="bottom", ha="left")

for label, source, colour, marker, ls in SERIES:
    ys = [v1_l2(source, r) for r in REGIONS]
    ax.plot(xs, ys, ls=ls, color=colour, marker=marker, ms=5.5, lw=1.6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4,
            clip_on=False)

ax.set_xticks(xs)
ax.set_xticklabels([NICE[r] for r in REGIONS], fontsize=7.4)
ax.set_xlim(-0.28, len(REGIONS) - 1 + 0.28)
ax.set_ylim(0, 1.55)
ax.set_ylabel("$v_1$ relative $L^2$ error   (0 = exact)")
ax.set_title("Neither source works everywhere", fontsize=8.8,
             loc="left", pad=30)
ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# All three lines converge toward the right of the panel, so direct labels
# collide no matter where they are placed. A legend above the axes keys them
# without competing for plot area.
ax.legend([
    ax.lines[i] for i in (1, 2, 3)
], [lbl for lbl, *_ in SERIES],
    loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
    fontsize=7.0, handlelength=1.8, handletextpad=0.5, columnspacing=1.4)

# ------------------------------------------------------------------ panel b
# What the network actually contributes, region by region: the signed change
# from prior-alone to prior+network. This is the attribution question stated
# as a single quantity.
ax2 = fig.add_subplot(gs[0, 1])
deltas = [100.0 * (v1_l2(PRIOR_ONLY, r) - v1_l2(HYBRID, r)) / v1_l2(PRIOR_ONLY, r)
          for r in REGIONS]
bar_colours = [COLORS["prior_network"] if d > 5 else COLORS["muted"]
               for d in deltas]
ax2.axhline(0.0, color=COLORS["reference"], lw=0.8, zorder=3)
bars = ax2.bar(xs, deltas, 0.56, color=bar_colours, zorder=4,
               edgecolor="white", linewidth=0.5)
for xx, d in zip(xs, deltas):
    ax2.annotate(f"{d:+.0f}%", (xx, d), xytext=(0, 5 if d > 0 else -12),
                 textcoords="offset points", ha="center", fontsize=7.6,
                 color="#3F3F46", fontweight="bold")

ax2.set_xticks(xs)
ax2.set_xticklabels([NICE[r] for r in REGIONS], fontsize=7.4)
ax2.set_xlim(-0.52, len(REGIONS) - 1 + 0.52)
ax2.set_ylim(-26, 82)
ax2.set_ylabel("reduction in $v_1$ error from adding\nthe network to the prior (%)",
               fontsize=7.8)
ax2.set_title("Network correction helps near field;\nno gain in far core", fontsize=8.8,
              loc="left", pad=30)
ax2.grid(axis="y", color=COLORS["grid"], lw=0.5)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

# The -1% bar and its value label already say "no gain"; a second annotation
# on the same 2-pixel bar only collides with that label. The claim lives in
# the panel title instead.
ax2.text(2.0, -22.0, "far core: no gain", fontsize=6.9,
         color=COLORS["muted"], ha="center", va="center")

fig.text(0.085, 0.055,
         "First shedding harmonic of $v$, 201 snapshots, on the regions of "
         "Figure F0a.\nFar core is the nested subset of far wake; the "
         "upstream region is excluded here (see F02b).",
         fontsize=6.8, color=COLORS["muted"], ha="left", va="bottom",
         linespacing=1.45)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT ISSUES:", bad)
out = save_figure(fig, ROOT / "figures" / "final" / "F02_prior_attribution")
print(out)
for r in REGIONS:
    print(f"  {r:14s} noPrior={v1_l2(BASE, r):.4f} "
          f"priorOnly={v1_l2(PRIOR_ONLY, r):.4f} both={v1_l2(HYBRID, r):.4f}")
