"""F01 - representative snapshot field comparison for A04.

The midpoint snapshot (t = 410) is shown as three shared-scale rows (u, v, p)
and four columns (CFD, analytical prior, pressure-only + physics,
pressure-only + physics + Kármán prior).  A single snapshot is
used for visual interpretation; all numerical claims remain the all-snapshot
metrics in the A04 JSON/CSV outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src" / "R9" / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from figure_common import (  # noqa: E402
    COLORS, check_text_overlaps, draw_cylinder, domain_axes, new_figure,
    save_figure,
)
from street_prior import cf_modes_uv  # noqa: E402


SNAP = np.load(ROOT / "derived" / "a04_snapshot_fields.npz")
PRIOR = np.load(
    ROOT.parents[1] / "modes_experiment" / "runs" / "arms" /
    "15_karman_prior_fluct_off" / "street_prior_used.npz"
)
X, Y = np.asarray(SNAP["x"], float), np.asarray(SNAP["y"], float)
T = float(SNAP["time"])


def prior_field(x, y, t, prm):
    names = ("Gamma", "Uc", "xf", "r0", "omega", "phase", "amp_scale",
             "scale_p", "ramp", "delta")
    params = {key: float(prm[key]) for key in names}
    us, vs = cf_modes_uv(x, y, params, nk=3)
    amp = 2.0 * params["amp_scale"]
    r2 = x * x + y * y + 1e-12
    a2 = 0.5 ** 2
    u0 = 1.0 - a2 * (x * x - y * y) / r2 ** 2
    v0 = -a2 * 2.0 * x * y / r2 ** 2
    u, v = u0.copy(), v0.copy()
    p = -0.5 * ((u0 - params["Uc"]) ** 2 + v0 ** 2)
    for k, (uk, vk) in enumerate(zip(us, vs), start=1):
        phase = np.exp(1j * k * 1.036 * t)
        eu, ev = amp * uk, amp * vk
        u += np.real(eu * phase)
        v += np.real(ev * phase)
        p += np.real(-(1.0 - params["Uc"]) * params["scale_p"] * eu * phase)
    return u, v, p


u_prior, v_prior, p_prior = prior_field(X, Y, T, PRIOR)
fields = {
    "u": [SNAP["u_true"], u_prior, SNAP["arm1_baseline_u"],
          SNAP["arm15_v1_radial_trust_u"]],
    "v": [SNAP["v_true"], v_prior, SNAP["arm1_baseline_v"],
          SNAP["arm15_v1_radial_trust_v"]],
    "p": [SNAP["p_true"], p_prior, SNAP["arm1_baseline_p"],
          SNAP["arm15_v1_radial_trust_p"]],
}
titles = ["CFD reference", "Kármán prior\nonly",
          "Pressure-only\n+ physics",
          "Pressure-only\n+ physics\n+ Kármán prior"]
row_labels = {"u": r"$u/U_\infty$", "v": r"$v/U_\infty$",
              "p": r"$p/(\rho U_\infty^2)$"}

tri = mtri.Triangulation(X, Y)
tri.set_mask(np.any(np.sqrt(X[tri.triangles] ** 2 + Y[tri.triangles] ** 2) < 0.5,
                    axis=1))

fig = new_figure(width="full", height=7.0)
axes = fig.subplots(3, 4, squeeze=False)
fig.subplots_adjust(left=0.085, right=0.925, bottom=0.075, top=0.91,
                    wspace=0.055, hspace=0.16)

for row, (quantity, values) in enumerate(fields.items()):
    all_values = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    lo, hi = np.nanpercentile(all_values, [1.0, 99.0])
    limit = max(abs(float(lo)), abs(float(hi)))
    levels = np.linspace(-limit, limit, 25)
    for col, (ax, value) in enumerate(zip(axes[row], values)):
        mappable = ax.tricontourf(tri, np.asarray(value), levels=levels,
                                  cmap="RdBu_r", extend="both")
        draw_cylinder(ax, zorder=5)
        ax.axvline(0.0, color="#FFFFFF", linewidth=0.45, alpha=0.7)
        ax.axvline(3.0, color="#FFFFFF", linewidth=0.45, alpha=0.7)
        domain_axes(ax)
        ax.set_title(titles[col], pad=5.0, color=COLORS["reference"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if col:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(row_labels[quantity], labelpad=4)
        if row < 2 or col:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$x/D$")
        ax.text(0.015, 0.965, f"{chr(97 + row * 4 + col)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
                color=COLORS["muted"], fontweight="bold")
    cbar = fig.colorbar(mappable, ax=axes[row].tolist(), fraction=0.018,
                        pad=0.012, aspect=25)
    cbar.set_label(row_labels[quantity], rotation=90, labelpad=7)
    cbar.ax.tick_params(width=0.5, length=2.5)

fig.suptitle(r"Representative snapshot at $t = %.1f$ (all panels share a row scale)" % T,
             y=0.965, fontsize=10.5, color=COLORS["reference"])
fig.text(0.055, 0.018,
         "White lines mark the near-wake (x/D = 0) and far-wake (x/D = 3) boundaries; "
         "the cylinder is masked.",
         ha="left", va="bottom", fontsize=7.2, color=COLORS["muted"])

bad = check_text_overlaps(fig)
if bad:
    print("TEXT COLLISIONS:", bad)
out = save_figure(fig, ROOT / "figures" / "final" / "F01_prior_attribution_fields")
print(out)
print(f"snapshot index = 100, time = {T:.1f}, nodes = {X.size}")
