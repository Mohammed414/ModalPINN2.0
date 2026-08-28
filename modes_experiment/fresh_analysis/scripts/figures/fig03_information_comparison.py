"""F03 - focused information-comparison story for A01.

The figure intentionally shows only two views: whole-domain field accuracy and
regional first-harmonic wake accuracy.  Together they answer the A01 question
without presenting every value in the underlying tidy table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figure_common import (  # noqa: E402
    COLORS,
    METHOD_LABELS,
    check_text_overlaps,
    new_figure,
    save_figure,
)


METRICS = json.loads(
    (ROOT / "derived" / "a01_information_comparison_metrics.json").read_text()
)
METHODS = [
    "pressure_only_physics",
    "pressure_and_velocity_probes_physics",
    "dense_observations",
]
METHOD_COLORS = {
    "pressure_only_physics": COLORS["pressure_only"],
    "pressure_and_velocity_probes_physics": COLORS["sparse_probes"],
    "dense_observations": COLORS["dense"],
}
REGIONS = ["near-cylinder", "near-wake", "far-core"]
REGION_LABELS = {
    "near-cylinder": "Near cylinder\n$r<0.75$",
    "near-wake": "Near wake\n$0\u2264 x<3$",
    "far-core": "Far core\n$x\u22653$, $|y|\u22642$",
}


fig = new_figure(width="full", height=4.85)
gs = fig.add_gridspec(
    1, 2, width_ratios=[1.03, 1.0], wspace=0.39,
    left=0.095, right=0.975, bottom=0.245, top=0.73,
)

# Whole-domain field error: the three quantities are deliberately grouped in
# one compact panel so the measurement-information effect is immediately seen.
ax = fig.add_subplot(gs[0, 0])
quantities = ["u", "v", "p"]
quantity_labels = [r"$u$", r"$v$", r"$p$"]
x = np.arange(len(quantities), dtype=float)
width = 0.24
offsets = (np.arange(len(METHODS)) - 1.0) * width
for method, offset in zip(METHODS, offsets):
    values = [
        METRICS["models"][method]["field_metrics"]["whole-domain"][q]
        for q in quantities
    ]
    bars = ax.bar(
        x + offset, values, width=width, color=METHOD_COLORS[method],
        edgecolor="white", linewidth=0.45, zorder=3,
    )
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=2,
                 fontsize=6.5, color=COLORS["reference"])
ax.set_xticks(x)
ax.set_xticklabels(quantity_labels)
ax.set_ylabel(r"whole-domain relative $L^2$ error")
ax.set_ylim(0.0, 0.92)
ax.set_title("Velocity probes reduce field error", loc="left", pad=8.0,
             fontsize=9.0)
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.85, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", length=2.5, width=0.6)

# Regional v1 error: this is the physically important wake quantity and shows
# where the information change matters.  The dashed line is the zero-field
# baseline under the relative-L2 contract.
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(1.0, color=COLORS["reference"], linewidth=0.8,
            linestyle=(0, (4, 2.5)), zorder=2)
ax2.text(0.02, 0.91, "zero prediction = 1.0", transform=ax2.transAxes,
         fontsize=6.5, color=COLORS["reference"], va="bottom", ha="left")
for method in METHODS:
    values = [
        METRICS["models"][method]["v1_mode_metrics"][region]["rel_L2"]
        for region in REGIONS
    ]
    ax2.plot(
        np.arange(len(REGIONS)), values, color=METHOD_COLORS[method],
        marker={METHODS[0]: "o", METHODS[1]: "s", METHODS[2]: "D"}[method],
        markeredgecolor="white", markeredgewidth=0.6, markersize=5.2,
        linewidth=1.6, label=METHOD_LABELS[method], zorder=4,
    )
ax2.set_xticks(np.arange(len(REGIONS)))
ax2.set_xticklabels([REGION_LABELS[r] for r in REGIONS], fontsize=7.2)
ax2.set_ylabel(r"$v_1$ relative $L^2$ error")
ax2.set_ylim(0.0, 1.18)
ax2.set_title("Velocity probes recover the shedding\nharmonic", loc="left",
              pad=8.0, fontsize=9.0)
ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.85, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
ax2.tick_params(axis="x", length=0, pad=3)
ax2.tick_params(axis="y", length=2.5, width=0.6)

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.875),
           ncol=3, frameon=False, handlelength=1.8, columnspacing=1.5,
           fontsize=7.4)
fig.suptitle("More measurement information closes the wake-reconstruction gap",
             y=0.96, fontsize=10.6, color=COLORS["reference"])
fig.text(
    0.095, 0.055,
    "201 common snapshots. Pressure-only versus pressure + velocity probes is the controlled comparison;\n"
    "dense observations are shown as a representational ceiling because that run also uses a different optimizer budget.",
    ha="left", va="bottom", fontsize=6.9, color=COLORS["muted"], linespacing=1.45,
)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT ISSUES:", bad)
out = save_figure(fig, ROOT / "figures" / "draft" / "F03_information_comparison")
print(out)
