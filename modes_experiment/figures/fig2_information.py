"""Figure 2 - where the information has to come from.

One sentence this figure must make true:
    Adding pressure taps does not help, and moving collocation points into the
    wake does not either; only supplying wake structure recovers the mode.

Panel a: tap count against amplitude - the sensor axis is flat and collapsed.
Panel b: collocation sampler, prior off vs on - paired comparison.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from fig_common import (load, style, save, panel_letter, check_overlaps,
                        C_COLLAPSED, C_FALSE, C_RECOVERED, C_REF, COLLAPSE_AMP)

style()
df = load()
g = lambda a: df[df.arm == a].iloc[0]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                              gridspec_kw=dict(wspace=0.34))

# ---------------------------------------------------------------- panel a
# Tap count. Arms 8, 9, 1 are single-variable: pressure-only, uniform
# sampling, no prior, differing only in tap count.
taps = [(g(8).NTaps, g(8).amp_far_core), (g(9).NTaps, g(9).amp_far_core),
        (g(1).NTaps, g(1).amp_far_core)]
tx, ty = zip(*taps)
ax.plot(tx, ty, "-o", color=C_COLLAPSED, lw=1.2, ms=6,
        markeredgecolor="white", markeredgewidth=0.5, zorder=3)
ax.axhline(COLLAPSE_AMP, color=C_REF, lw=0.7, ls=":", zorder=1)
ax.text(8.4, COLLAPSE_AMP + 0.012, "collapse threshold",
        fontsize=7, color=C_REF, va="bottom")
# The comparison that matters: where the probe arm sits on the same axis.
ax.axhline(g(4).amp_far_core, color=C_RECOVERED, lw=1.0, ls="--", zorder=2)
ax.text(32, g(4).amp_far_core - 0.03, "32 velocity probes",
        fontsize=7, color=C_RECOVERED, va="top", ha="right")
ax.set_xticks([8, 16, 32])
ax.set_xlabel("Number of wall pressure taps")
ax.set_ylabel("Far-core amplitude ratio")
ax.set_ylim(-0.03, 1.0)
ax.set_title("Four times the taps buys nothing")

# ---------------------------------------------------------------- panel b
# Sampler, prior off vs on. Each pair differs only in the sampler.
pairs = [("uniform", 1, 15), ("wake-biased\ngrid", 7, 10)]
x = np.arange(len(pairs))
w = 0.34
for i, (name, off, on) in enumerate(pairs):
    a_off, a_on = g(off), g(on)
    c_off = C_FALSE if a_off.relL2_far_core >= 1.0 else C_COLLAPSED
    ax2.bar(i - w / 2, a_off.amp_far_core, w, color=c_off, zorder=3)
    ax2.bar(i + w / 2, a_on.amp_far_core, w, color=C_RECOVERED, zorder=3)
    # A near-zero bar renders as a sliver that reads as an empty slot; give it
    # a visible marker at the baseline so the value is legible.
    if a_off.amp_far_core < 0.05:
        ax2.plot([i - w / 2], [a_off.amp_far_core], "o", ms=4.5,
                 color=c_off, zorder=4)
    for xx, v in ((i - w / 2, a_off.amp_far_core), (i + w / 2, a_on.amp_far_core)):
        ax2.text(xx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax2.set_xticks(x)
ax2.set_xticklabels([p[0] for p in pairs])
ax2.set_xlabel("Collocation sampling")
ax2.set_ylabel("Far-core amplitude ratio")
ax2.set_ylim(0, 1.32)
ax2.set_title("The prior, not the sampler, recovers the wake")

# Identity labels are floor, not annotation budget.
ax2.bar(np.nan, np.nan, color=C_COLLAPSED, label="no prior (collapsed)")
ax2.bar(np.nan, np.nan, color=C_FALSE, label="no prior (wrong phase)")
ax2.bar(np.nan, np.nan, color=C_RECOVERED, label="prior active")
ax2.legend(loc="upper center", ncol=1, handlelength=1.1, handletextpad=0.5,
           borderpad=0.2, labelspacing=0.3, bbox_to_anchor=(0.5, 1.02))

panel_letter(ax, "a", dx=-0.16, dy=1.06)
panel_letter(ax2, "b", dx=-0.16, dy=1.06)

fig.canvas.draw()
bad = check_overlaps(fig)
if bad:
    print("text collisions:", bad)
print(save(fig, "fig2_information"))
