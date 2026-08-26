"""Figure 7 - the collapse, quantified along the wake.

One sentence this figure must make true:
    The pressure-only arm's oscillating mode dies within one diameter of the
    cylinder, and the arms that fail while carrying amplitude fail because they
    get the streamwise wavelength wrong, not because they are noisy.

Panel a: |v_1| against streamwise distance - the decay curve.
Panel b: centreline phase against x. Slope is 2*pi/wavelength, so a correct
         street is a straight line parallel to the DNS; a standing disturbance
         is flat.

Data comes from decay_profiles.json, written by build_decay_data.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from fig_common import (style, save, panel_letter, check_overlaps,
                        C_COLLAPSED, C_FALSE, C_RECOVERED, C_PROBE, C_REF)

HERE = os.path.dirname(os.path.abspath(__file__))
style()
D = json.load(open(os.path.join(HERE, "decay_profiles.json")))

xc = np.array(D["xc"])
dns = np.array(D["dns"])
COL = {"1": C_COLLAPSED, "7": C_FALSE, "15": C_RECOVERED, "4": C_PROBE}
ORDER = ["1", "7", "15", "4"]
NAME = {"1": "pressure only", "7": "wake-biased grid",
        "15": "pressure + prior", "4": "velocity probes"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                              gridspec_kw=dict(wspace=0.30))

# ---------------------------------------------------------------- panel a
ax.plot(xc, dns, color="0.15", lw=1.6, zorder=5, label="DNS reference")
for k in ORDER:
    a = D["arms"][k]
    ax.plot(xc, a["amp"], color=COL[k], lw=1.3, zorder=4, label=NAME[k])
ax.axvspan(-1, 0.5, color="0.92", zorder=0, lw=0)
ax.text(-0.25, 0.44, "cylinder", fontsize=6.5, color=C_REF, ha="center")
ax.set_xlabel("Streamwise distance $x/D$")
ax.set_ylabel("Oscillating mode magnitude  $|\\hat v_1|$")
ax.set_xlim(-1, 8)
ax.set_ylim(0, 0.47)
ax.set_title("Collapse within one diameter")
ax.legend(loc="center right", handlelength=1.4, labelspacing=0.32)

# The one headline number: where the collapsed arm falls under a tenth of DNS.
r = np.array(D["arms"]["1"]["amp"]) / dns
j = next(i for i in range(len(xc)) if xc[i] > 0.5 and r[i] < 0.10)
ax.annotate(f"under 10%\nof DNS by\n$x/D$ = {xc[j]:.2f}",
            xy=(xc[j], D["arms"]["1"]["amp"][j]),
            xytext=(5.6, 0.045), fontsize=7, color="0.42",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="0.42", lw=0.7,
                            shrinkA=2, shrinkB=2))

# ---------------------------------------------------------------- panel b
xl = np.array(D["dns_xline"])
ph0 = np.array(D["dns_phase"])
ax2.plot(xl, ph0 - ph0[0], color="0.15", lw=1.6, zorder=5,
         label=f"DNS  $\\lambda$ = {D['dns_lambda']:.2f}$D$")
for k in ORDER:
    a = D["arms"][k]
    p = np.array(a["phase"])
    ax2.plot(np.array(a["xline"]), p - p[0], color=COL[k], lw=1.3, zorder=4,
             label=f"{NAME[k]}  $\\lambda$ = {a['lam']:.2f}$D$")
ax2.set_xlabel("Streamwise distance $x/D$")
ax2.set_ylabel("Centreline phase of $\\hat v_1$ (rad)")
ax2.set_xlim(1, 8)
ax2.set_title("Failed arms get the wavelength wrong")
# Direct-label instead of a legend box: the two failed arms run flat along the
# top and the three street-carrying lines fan down the diagonal, so any box
# inside the axes sits on data.
ax2.set_ylim(-12.4, 3.4)
# Flat pair: label above their own lines, left half of the panel where the
# diagonal bundle has not yet arrived.
ax2.annotate(f"pressure only   $\\lambda$ = {D['arms']['1']['lam']:.0f}$D$",
             xy=(1.15, 2.55), fontsize=6.8, color=COL["1"], ha="left")
ax2.annotate(f"wake-biased grid   $\\lambda$ = {D['arms']['7']['lam']:.0f}$D$",
             xy=(1.15, 1.45), fontsize=6.8, color=COL["7"], ha="left")
# Diagonal bundle: stack the three labels in the empty lower-left wedge.
for k, yy in (("15", -9.4), ("4", -10.6)):
    a = D["arms"][k]
    ax2.annotate(f"{NAME[k]}   $\\lambda$ = {a['lam']:.2f}$D$",
                 xy=(1.15, yy), fontsize=6.8, color=COL[k], ha="left")
ax2.annotate(f"DNS   $\\lambda$ = {D['dns_lambda']:.2f}$D$",
             xy=(1.15, -11.8), fontsize=6.8, color="0.15", ha="left")

panel_letter(ax, "a", dx=-0.16, dy=1.06)
panel_letter(ax2, "b", dx=-0.12, dy=1.06)

fig.canvas.draw()
bad = check_overlaps(fig)
if bad:
    print("text collisions:", bad)
print(save(fig, "fig7_decay"))
