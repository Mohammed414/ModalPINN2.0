"""Figure 4 - integrated forces do not diagnose the wake.

One sentence this figure must make true:
    Lift and drag are reproduced to within a few percent whether or not the
    oscillating wake was recovered, so force error cannot detect the collapse.

Only four arms have force diagnostics computed (1, 2, 3, 16); this is stated in
the caption and the panel, and no other arm is implied.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fig_common import (load, style, save, panel_letter, check_overlaps,
                        C_COLLAPSED, C_RECOVERED, C_REF, COLLAPSE_AMP)

style()
df = load()
FE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "..", "force_error_summary.csv")
fe = pd.read_csv(FE)
fe["arm_n"] = [int(s.split()[0]) for s in fe.arm]
fe = fe.merge(df[["arm", "amp_far_core", "regime"]], left_on="arm_n",
              right_on="arm", how="left")

NAME = {1: "32 taps (baseline)", 2: "32 taps + vorticity flux",
        3: "32 taps + prior", 16: "prior, no freestream BC, no Adam"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                              gridspec_kw=dict(wspace=0.36))

col = lambda r: C_COLLAPSED if r.amp_far_core < COLLAPSE_AMP else C_RECOVERED

# ---------------------------------------------------------------- panel a
# Lift phase error against wake amplitude. The x-axis spans 43x; the y-axis
# spans 1.6 degrees. That flatness is the result.
for _, r in fe.iterrows():
    ax.scatter(r.amp_far_core, abs(r.CL_phase_err_deg), s=46, color=col(r),
               edgecolor="white", linewidth=0.5, zorder=3)
    # Arms 3 and 16 sit within 0.03 degrees of each other and would read as a
    # single mark; label each point with its arm so all four are countable.
    ax.annotate(f"arm {r.arm_n}", (r.amp_far_core, abs(r.CL_phase_err_deg)),
                xytext=(7, 3 if r.arm_n != 16 else -10),
                textcoords="offset points", fontsize=6.5, color=C_REF)
ax.set_xscale("log")
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
ax.minorticks_off()
ax.set_xlabel("Far-core amplitude ratio (log scale)")
ax.set_ylabel("Lift phase error (degrees)")
ax.set_ylim(2.5, 9)
ax.set_xlim(0.012, 1.5)
ax.set_title("Lift phase is flat across a 43x wake span")
ax.annotate("wake collapsed", xy=(fe[fe.arm_n == 1].amp_far_core.iloc[0], 4.80),
            xytext=(8, 10), textcoords="offset points", fontsize=7,
            color=C_REF, ha="left")
ax.annotate("wake recovered", xy=(fe[fe.arm_n == 3].amp_far_core.iloc[0], 5.65),
            xytext=(0, -16), textcoords="offset points", fontsize=7,
            color=C_REF, ha="center")

# ---------------------------------------------------------------- panel b
# Same four arms, the three force error measures side by side.
metrics = [("Lift amplitude", "CL_amp_err_pct"),
           ("Lift phase", "CL_phase_err_deg"),
           ("Mean drag", "CD_mean_err_pct")]
x = np.arange(len(metrics))
w = 0.2
for j, (_, r) in enumerate(fe.sort_values("amp_far_core").iterrows()):
    off = (j - 1.5) * w
    vals = [abs(r[k]) for _, k in metrics]
    ax2.bar(x + off, vals, w, color=col(r), zorder=3,
            edgecolor="white", linewidth=0.4)
    # Four bars per group in two colours: name each so a bar is traceable.
    for xi, v in zip(x + off, vals):
        ax2.text(xi, 0.25, f"{r.arm_n}", ha="center", va="bottom",
                 fontsize=6, color="white", zorder=4)
ax2.set_xticks(x)
ax2.set_xticklabels([n for n, _ in metrics])
ax2.set_ylabel("Absolute error\n(% or degrees)")
ax2.set_title("No force measure separates the two regimes")
ax2.set_ylim(0, 14)
ax2.bar(np.nan, np.nan, color=C_COLLAPSED, label="wake collapsed (2 arms)")
ax2.bar(np.nan, np.nan, color=C_RECOVERED, label="wake recovered (2 arms)")
ax2.legend(loc="upper left", handlelength=1.1, handletextpad=0.5,
           borderpad=0.2)

panel_letter(ax, "a", dx=-0.17, dy=1.06)
panel_letter(ax2, "b", dx=-0.19, dy=1.06)

fig.canvas.draw()
bad = check_overlaps(fig)
if bad:
    print("text collisions:", bad)
print(save(fig, "fig4_force_blind"))
