"""Figure 3 - robustness of the recovered reconstruction.

One sentence this figure must make true:
    Once the wake structure is supplied, the reconstruction is insensitive to
    sensor noise and to the boundary-condition and optimiser choices tested.

Panel a: noise level against amplitude, with the noise-free prior arm as the
         reference point. The ordering is non-monotonic and is drawn as such.
Panel b: configuration variants, all prior-active, as deviations from the
         matched-BC reference arm.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from fig_common import (load, style, save, panel_letter, check_overlaps,
                        C_RECOVERED, C_REF, C_PROBE)

style()
df = load()
g = lambda a: df[df.arm == a].iloc[0]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                              gridspec_kw=dict(wspace=0.34,
                                               width_ratios=[1.0, 1.45]))

# ---------------------------------------------------------------- panel a
# Noise arms 11/12/13 plus the noise-free reference (arm 15). All are
# prior-active, 32 taps, uniform sampling, matched BCs.
ref = g(15)
noise_arms = [(0.0, ref), (1.0, g(11)), (5.0, g(12)), (10.0, g(13))]

# Categorical spacing: the noise levels are four tested conditions, not a
# continuous sweep, and even spacing keeps the tick labels legible.
nx = np.arange(len(noise_arms))
ny = [r.amp_far_core for _, r in noise_arms]
ax.plot(nx, ny, "-o", color=C_RECOVERED, lw=1.2, ms=6,
        markeredgecolor="white", markeredgewidth=0.5, zorder=3)
for xx, yy in zip(nx, ny):
    ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=7,
                color=C_REF)
ax.set_xticks(nx)
ax.set_xticklabels(["none", "1%", "5%", "10%"])
ax.margins(x=0.12)
ax.set_xlabel("Sensor noise added to the pressure signal")
ax.set_ylabel("Far-core amplitude ratio")
ax.set_ylim(0.70, 0.95)
ax.set_title("Noise leaves recovery intact")

# ---------------------------------------------------------------- panel b
# Configuration variants, all prior-active. Deviation from arm 15.
variants = [("reference", 15), ("fluct. inlet\nBC on", 3),
            ("no freestream\nBC, no Adam", 16), ("wake-biased\ncolloc.", 10)]
vx = np.arange(len(variants))
vals = [g(a).amp_far_core for _, a in variants]
base = vals[0]
cols = [C_PROBE] + [C_RECOVERED] * (len(variants) - 1)
ax2.bar(vx, vals, 0.6, color=cols, zorder=3)
ax2.axhline(base, color=C_REF, lw=0.8, ls="--", zorder=2)
for i, v in enumerate(vals):
    d = "" if i == 0 else f"\n{100 * (v - base) / base:+.0f}%"
    ax2.text(i, v + 0.02, f"{v:.2f}{d}", ha="center", va="bottom", fontsize=7)
ax2.set_xticks(vx)
ax2.set_xticklabels([n for n, _ in variants])
ax2.set_ylabel("Far-core amplitude ratio")
ax2.set_ylim(0, 1.18)
ax2.set_title("Boundary and optimiser choices barely matter; sampling does")

panel_letter(ax, "a", dx=-0.19, dy=1.06)
panel_letter(ax2, "b", dx=-0.10, dy=1.06)

fig.canvas.draw()
bad = check_overlaps(fig)
if bad:
    print("text collisions:", bad)
print(save(fig, "fig3_robustness"))
