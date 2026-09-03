"""F02b - the upstream region, and why its ratio metrics mislead.

One sentence this figure must make true:
    The pressure + physics + Kármán prior run does leak oscillation upstream of the cylinder where
    physically there is almost none, but the leakage is small in absolute terms
    and only looks catastrophic when divided by a near-zero reference.

This figure exists because the headline ratio for the upstream region is
amp_ratio = 11.5, which invites the reading "11x too much oscillation". Plotted
as absolute RMS per node against the same quantity in the wake regions, the
leakage is a third of the far-core signal, not eleven times anything. Both
framings are shown so the reader can see why they differ.

Input:  derived/a04_v1_absolute_check.json,
        derived/a04_prior_attribution_metrics.json
Output: figures/final/F02b_upstream_artefact.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from figure_common import (  # noqa: E402
    COLORS, check_text_overlaps, new_figure, save_figure,
)

D = ROOT / "data" / "analysis"
ABS = json.loads((D / "a04_v1_absolute_check.json").read_text())
ATTRIB = json.loads((D / "a04_prior_attribution_metrics.json").read_text())
HYB = "arm15_v1_radial_trust"

REGIONS = ["other", "near-cylinder", "near-wake", "far-core"]
NICE = {"other": "Upstream\n$x<0$", "near-cylinder": "Near cyl.\n$r<0.75$",
        "near-wake": "Near wake\n$0\\leq x<3$",
        "far-core": "Far core\n$x\\geq3$"}

fig = new_figure(width="full", height=3.75)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.34,
                      left=0.085, right=0.975, bottom=0.235, top=0.76)

# ------------------------------------------------------------------ panel a
# Absolute magnitudes. This is the honest framing: what is actually there.
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(len(REGIONS))
w = 0.38
true_rms = [ABS[HYB][r]["true_rms_per_node"] for r in REGIONS]
pred_rms = [ABS[HYB][r]["pred_rms_per_node"] for r in REGIONS]

ax.bar(xs - w / 2, true_rms, w, color=COLORS["reference"], zorder=3,
       label="CFD reference", edgecolor="white", linewidth=0.5)
ax.bar(xs + w / 2, pred_rms, w, color=COLORS["prior_network"], zorder=3,
       label="Prior + network", edgecolor="white", linewidth=0.5)

ax.set_xticks(xs)
ax.set_xticklabels([NICE[r] for r in REGIONS], fontsize=7.2)
ax.set_ylabel("$|v_1|$ RMS per node   (absolute)")
ax.set_title("In absolute terms the upstream leakage is small",
             fontsize=8.6, loc="left", pad=30)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
          frameon=False, fontsize=7.2, handlelength=1.5, handletextpad=0.5,
          columnspacing=1.6)
ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# Name the comparison that defuses the 11x reading.
ax.annotate(f"leakage {pred_rms[0]:.3f}\n= {100 * pred_rms[0] / pred_rms[3]:.0f}% of the\nfar-core signal",
            xy=(0 + w / 2, pred_rms[0]), xytext=(0.55, 0.145),
            fontsize=6.9, color="#7A3E00", ha="left", va="center",
            linespacing=1.35, zorder=6,
            arrowprops=dict(arrowstyle="-", color="#7A3E00", lw=0.6,
                            shrinkA=1, shrinkB=2))

# ------------------------------------------------------------------ panel b
# The same region as a ratio. Shown so the reader sees the artefact, not to
# argue from it.
ax2 = fig.add_subplot(gs[0, 1])
ratios = [float(ATTRIB["models"][HYB]["v1_mode_metrics"][r]["amp_ratio"])
          for r in REGIONS]
cols = [COLORS["accent"] if r == "other" else COLORS["prior_network"]
        for r in REGIONS]
ax2.axhline(1.0, color=COLORS["reference"], lw=0.8, ls=(0, (4, 2.5)), zorder=4)
ax2.bar(xs, ratios, 0.56, color=cols, zorder=3, edgecolor="white",
        linewidth=0.5)
for xx, v in zip(xs, ratios):
    # Bars near 1.0 would have their label struck through by the dashed
    # reference line, so those labels go inside the bar instead of above it.
    ax2.annotate(f"{v:.2f}" if v < 10 else f"{v:.1f}", (xx, v),
                 xytext=(0, 5), textcoords="offset points", ha="center",
                 fontsize=7.4, color="#3F3F46", fontweight="bold", zorder=6)

ax2.set_ylim(0, 14.5)
ax2.set_yticks([0, 1, 5, 10])
ax2.set_xticks(xs)
ax2.set_xticklabels([NICE[r] for r in REGIONS], fontsize=7.2)
ax2.set_ylabel("$|v_1|$ amplitude ratio to CFD   (1 = correct)",
               fontsize=7.8)
ax2.set_title("As a ratio it looks catastrophic", fontsize=8.6,
              loc="left", pad=30)
ax2.grid(axis="y", color=COLORS["grid"], lw=0.5)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

ax2.annotate("divided by a near-zero reference",
             xy=(0.30, 9.5), xytext=(0.75, 11.6), fontsize=6.9,
             color="#7A5200", ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color="#7A5200",
                             lw=0.6, shrinkA=1, shrinkB=2))

fig.text(0.085, 0.045,
         "First shedding harmonic of $v$, pressure + physics + Kármán prior, 201 "
         "snapshots. The upstream reference amplitude is\n"
         f"{ABS[HYB]['other']['true_rms_per_node']:.4f} RMS per node, so the "
         "ratio in the right panel is governed by its denominator. Report the "
         "absolute framing.",
         fontsize=6.8, color=COLORS["muted"], ha="left", va="bottom",
         linespacing=1.45)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT ISSUES:", bad)
out = save_figure(fig, ROOT / "figures" / "F02b_upstream_artefact")
print(out)
for r in REGIONS:
    print(f"  {r:14s} true={ABS[HYB][r]['true_rms_per_node']:.5f} "
          f"pred={ABS[HYB][r]['pred_rms_per_node']:.5f} "
          f"ratio={ATTRIB['models'][HYB]['v1_mode_metrics'][r]['amp_ratio']:.4f}")
