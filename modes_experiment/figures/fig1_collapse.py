"""Figure 1 - the headline result.

One sentence this figure must make true:
    Wall pressure alone cannot see the oscillating wake; supplying the missing
    structure recovers it, and amplitude alone is not evidence of recovery.

Panel a: far-core amplitude ratio for every arm, grouped by regime.
Panel b: amplitude against relative L2, which separates genuine recovery from
         arms that produce magnitude with the wrong phase.
"""
import os
import sys

# Run this script from anywhere: keep its own directory importable even under an
# isolated interpreter (PYTHONSAFEPATH=1 drops the script directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from fig_common import (load, style, save, panel_letter, check_overlaps,
                        C_COLLAPSED, C_FALSE, C_RECOVERED, C_REF,
                        COLLAPSE_AMP, ZERO_BASELINE)

style()
df = load()

# short, plain-language names - no codebase abbreviations on the axis
LABEL = {
    1: "32 taps (baseline)", 2: "32 taps + vorticity flux", 3: "32 taps + prior",
    4: "velocity probes", 6: "wake-biased random", 7: "wake-biased grid",
    8: "8 taps", 9: "16 taps", 10: "prior + wake-biased grid",
    11: "prior + 1% noise", 12: "prior + 5% noise", 13: "prior + 10% noise",
    14: "32 taps + vorticity flux (no inlet BC)", 15: "prior (matched BCs)",
    16: "prior, no freestream BC, no Adam",
}
df["label"] = df.arm.map(LABEL)

fig = plt.figure(figsize=(7.2, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.42)

# ---------------------------------------------------------------- panel a
ax = fig.add_subplot(gs[0, 0])
order = ["collapsed", "amplitude, no phase", "recovered"]
d = df.sort_values(["regime", "amp_far_core"],
                   key=lambda s: s.map({k: i for i, k in enumerate(order)}) if s.name == "regime" else s)
y = np.arange(len(d))
ax.hlines(y, 0, d.amp_far_core, color=d.color, lw=1.0, alpha=0.55)
ax.scatter(d.amp_far_core, y, s=34, color=d.color, zorder=3,
           edgecolor="white", linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(d.label)
ax.invert_yaxis()
ax.axvline(COLLAPSE_AMP, color=C_REF, lw=0.7, ls=":", zorder=1)
ax.text(COLLAPSE_AMP + 0.015, len(d) - 0.4, "collapse\nthreshold",
        fontsize=7, color=C_REF, va="bottom")
ax.set_xlabel("Far-core amplitude ratio  (1.0 = reference wake)")
ax.set_xlim(-0.02, 1.0)
ax.set_title("Wall pressure alone never recovers the oscillating wake")

# headline number only - everything else is read off the axis
b, p = d[d.arm == 1].iloc[0], d[d.arm == 4].iloc[0]
ax.annotate(f"{p.amp_far_core / b.amp_far_core:.0f}x the baseline",
            xy=(p.amp_far_core, list(d.arm).index(4)),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, color=C_RECOVERED, fontweight="bold",
            ha="left", va="center")

# ---------------------------------------------------------------- panel b
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(ZERO_BASELINE, color=C_REF, lw=0.8, ls="--", zorder=1)
ax2.text(0.97, ZERO_BASELINE + 0.015, "worse than predicting zero",
         fontsize=7, color=C_REF, va="bottom", ha="right")
for rg in order:
    s = df[df.regime == rg]
    ax2.scatter(s.amp_far_core, s.relL2_far_core, s=40, label=rg,
                color={"collapsed": C_COLLAPSED, "amplitude, no phase": C_FALSE,
                       "recovered": C_RECOVERED}[rg],
                edgecolor="white", linewidth=0.5, zorder=3)
# These two arms are the figure's point: label them as one group rather than
# twice, which would collide at this scale.
r6 = df[df.arm == 6].iloc[0]
r7 = df[df.arm == 7].iloc[0]
ax2.annotate("wake-biased\ncollocation",
             xy=(r6.amp_far_core, r6.relL2_far_core),
             xytext=(10, -6), textcoords="offset points",
             fontsize=7, color=C_FALSE, ha="left", va="top",
             arrowprops=dict(arrowstyle="-", color=C_FALSE, lw=0.6,
                             shrinkA=0, shrinkB=2))
ax2.set_xlabel("Far-core amplitude ratio")
ax2.set_ylabel("Relative $L_2$ error")
ax2.set_title("Amplitude alone is not recovery")
ax2.set_xlim(-0.02, 1.0)
ax2.margins(y=0.10)
ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), handletextpad=0.4)

panel_letter(ax, "a", dy=1.09)
panel_letter(ax2, "b", dx=-0.24, dy=1.09)

fig.canvas.draw()
bad = check_overlaps(fig)
if bad:
    print("text collisions:", bad)
print(save(fig, "fig1_collapse"))
