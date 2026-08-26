"""Figure 6 - the actual flow. What collapse and recovery look like as fields.

One sentence this figure must make true:
    The collapsed arm reproduces the mean wake but loses the vortex street
    within one or two diameters; the recovered arms carry it across the domain.

Rows are DNS reference, the collapsed pressure-only baseline (arm 1), and the
recovered prior arm (arm 15). Columns are the mean streamwise velocity, the
k=1 transverse mode magnitude, and an instantaneous transverse velocity
snapshot - the last being what a reader recognises as "the vortex street".

Requires dns_raw.npz: run `python parse_dns.py` once first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import griddata

import modalpinn_eval as me
from fig_common import style, save, check_overlaps

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(HERE, "..", "runs", "arms")
style()

# ---------------------------------------------------------------- data
FOLD = {int(d.split("_")[1] if d.startswith("arm_") else d.split("_")[0]): d
        for d in sorted(os.listdir(ARMS))}


def arm_paths(n):
    d = os.path.join(ARMS, FOLD[n])
    return (os.path.join(d, "training_run", "DNN2_100_100_4_tanh.pickle"),
            os.path.join(d, "run_record.json"),
            os.path.join(d, "street_prior_used.npz"))


Z = np.load(os.path.join(HERE, "dns_raw.npz"))
xs, ys, times = Z["xs"], Z["ys"], Z["times"]
U, V = Z["U"], Z["V"]
w0 = me.OMEGA_0

# Restrict the DNS to the reconstruction domain before interpolating.
dom = ((xs >= -4) & (xs <= 8) & (ys >= -4) & (ys <= 4))
xd, yd = xs[dom], ys[dom]


def dns_mode(F, k):
    if k == 0:
        return F[:, dom].mean(axis=0).astype(np.complex128)
    return 2.0 * (F[:, dom] * np.exp(-1j * k * w0 * times)[:, None]).mean(axis=0)


XX, YY, inside = me.grid(300, 200)


def to_grid(vals):
    """Interpolate scattered DNS nodes onto the plotting grid."""
    g = griddata((xd, yd), vals, (XX, YY), method="linear")
    return np.ma.masked_where(inside | np.isnan(g), g)


def nn_fields(n, t_snap):
    """(mean u, |v1|, v at t) for one arm, evaluated as it was trained."""
    pk, rr, spp = arm_paths(n)
    w_u, b_u, w_v, b_v, w_p, b_p = me.load_weights(pk)
    fl = me.assert_flags_supported(rr)
    sp = me.load_street_params(spp) if fl["prior"] else None
    fs_u = 1.0 if fl["freestream"] else None
    fs_v = 0.0 if fl["freestream"] else None
    xf, yf = XX.ravel(), YY.ravel()
    u0 = me.modes(xf, yf, w_u, b_u, fs_u)[:, 0].real
    qv = me.modes(xf, yf, w_v, b_v, fs_v, street_params=sp, is_v=True)
    vt = me.field_at_time(xf, yf, t_snap, w_v, b_v, fs_v,
                          street_params=sp, is_v=True)
    m = lambda a: np.ma.masked_where(inside, a.reshape(XX.shape))
    return m(u0), m(np.abs(qv[:, 1])), m(vt)


T_SNAP = float(times[0])
rows = [("DNS reference",
         to_grid(dns_mode(U, 0).real),
         to_grid(np.abs(dns_mode(V, 1))),
         to_grid(V[0, dom].astype(np.float64))),
        ("Pressure only\n(arm 1, collapsed)", *nn_fields(1, T_SNAP)),
        ("Pressure + prior\n(arm 15, recovered)", *nn_fields(15, T_SNAP))]

# ---------------------------------------------------------------- figure
COLS = [("Mean streamwise velocity", "viridis", None),
        ("Oscillating mode magnitude  $|\\hat v_1|$", "magma", None),
        ("Transverse velocity, one instant", "RdBu_r", "sym")]

fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.0),
                         gridspec_kw=dict(wspace=0.08, hspace=0.14))

# Shared scales per column, set from the DNS row so every row is comparable.
lims = []
for j, (_, cmap, kind) in enumerate(COLS):
    ref = rows[0][j + 1]
    if kind == "sym":
        a = np.nanpercentile(np.abs(ref.compressed()), 99)
        lims.append((-a, a))
    else:
        lims.append((0.0, np.nanpercentile(ref.compressed(), 99)))

for i, (name, *fields) in enumerate(rows):
    for j, f in enumerate(fields):
        ax = axes[i, j]
        vmin, vmax = lims[j]
        im = ax.pcolormesh(XX, YY, f, cmap=COLS[j][1], vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.add_patch(Circle((0, 0), 0.5, facecolor="0.85", edgecolor="0.35",
                            lw=0.5, zorder=5))
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.4); s.set_color("0.6")
        if i == 0:
            ax.set_title(COLS[j][0], fontsize=8, pad=4)
        if j == 0:
            ax.set_ylabel(name, fontsize=7.5)
        if i == 2:
            # shrink<1 leaves a gap between adjacent bars; without it the end
            # labels of neighbouring colorbars collide at the shared edge.
            cb = fig.colorbar(im, ax=axes[:, j], orientation="horizontal",
                              fraction=0.055, pad=0.04, aspect=24, shrink=0.82)
            cb.ax.tick_params(labelsize=6.5, length=2)
            cb.outline.set_linewidth(0.4)
            # Three ticks per bar: the ends and the semantic centre. More than
            # that collides at this width.
            vmn, vmx = lims[j]
            tk = [vmn, 0.0, vmx] if vmn < 0 else [vmn, vmx / 2, vmx]
            cb.set_ticks(tk)
            cb.set_ticklabels([f"{v:.1f}" for v in tk])

# One scale cue, not nine: the domain is identical in every panel.
axes[2, 0].plot([-3.5, -1.5], [-3.4, -3.4], color="0.2", lw=1.2,
                solid_capstyle="butt", zorder=6)
axes[2, 0].text(-2.5, -3.15, "2D", fontsize=6.5, ha="center", color="0.2")

fig.canvas.draw()
bad = check_overlaps(fig)
if bad:
    print("text collisions:", bad)
print(save(fig, "fig6_flow_fields"))
