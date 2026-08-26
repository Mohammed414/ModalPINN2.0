"""Figure 5 - optimisation cost does not predict wake recovery.

One sentence this figure must make true:
    Every arm converged on its own terms, and the arms that spent the most
    optimiser effort are not the ones that recovered the wake.

This is the figure that rules out "it just needed longer to train" as an
explanation for the collapse.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from fig_common import (load, style, save, panel_letter, check_overlaps,
                        C_COLLAPSED, C_FALSE, C_RECOVERED, C_REF, REGIME_COLOR)

style()
df = load()

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                              gridspec_kw=dict(wspace=0.34))

# ---------------------------------------------------------------- panel a
# L-BFGS evaluations against amplitude. If longer training explained the
# collapse, this would trend upward. It does not.
for rg in ("collapsed", "amplitude, no phase", "recovered"):
    s = df[df.regime == rg]
    ax.scatter(s.lbfgs_nf / 1000, s.amp_far_core, s=42, label=rg,
               color=REGIME_COLOR[rg], edgecolor="white", linewidth=0.5,
               zorder=3)
CAP = df.LBFGSMaxit.iloc[0] / 1000
ax.axvline(CAP, color=C_REF, lw=0.8, ls=":", zorder=1)
ax.text(CAP - 1.0, 0.02, "iteration cap\n(never reached)", fontsize=7,
        color=C_REF, ha="right", va="bottom")
ax.set_xlabel("L-BFGS function evaluations (thousands)")
ax.set_ylabel("Far-core amplitude ratio")
ax.set_xlim(0, CAP + 3)
ax.set_ylim(-0.03, 1.0)
ax.set_title("Optimiser effort does not recover the wake")
ax.legend(loc="upper right", handletextpad=0.4, labelspacing=0.3,
          bbox_to_anchor=(1.0, 0.92))

# ---------------------------------------------------------------- panel b
# Where the wall-clock time went. Every arm ran the same 9 h budget; the split
# between the two optimiser phases is what differs.
d = df.sort_values("amp_far_core")
y = np.arange(len(d))
ax2.barh(y, d.lbfgs_s / 3600, 0.68, color=d.color, zorder=3)
ax2.barh(y, d.adam_s / 3600, 0.68, left=d.lbfgs_s / 3600, color="#E4E7EA",
         zorder=3, edgecolor="white", linewidth=0.4)
# The bar colour already encodes regime (threaded from panel a), so the phase
# legend keys only the pale Adam segment; the filled segment is named in text.
ax2.barh(np.nan, np.nan, color="#E4E7EA", label="Adam phase")
ax2.set_yticks(y)
ax2.set_yticklabels([f"arm {int(a)}" for a in d.arm], fontsize=6.5)
ax2.set_xlabel("Wall-clock training time (hours)")
ax2.set_title("Same budget, different split between phases\n")
# Legend below the x-axis label: every horizontal band inside the axes is
# occupied by a bar, so an in-axes placement would sit on data.
ax2.legend(loc="upper right", bbox_to_anchor=(1.0, -0.13),
           handlelength=1.1, handletextpad=0.5)
ax2.margins(y=0.02)

panel_letter(ax, "a", dx=-0.17, dy=1.06)
panel_letter(ax2, "b", dx=-0.13, dy=1.06)

fig.canvas.draw()
bad = check_overlaps(fig)
if bad:
    print("text collisions:", bad)
print(save(fig, "fig5_cost"))
