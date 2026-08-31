"""F_dataset — the reference flow, and the evidence for the frequency.

Section 3.1 asserts a set of numbers: 201 snapshots, a fully periodic record,
omega_0 = 1.0357. This figure shows the flow those numbers describe and the
evidence for the frequency, so the section can state the values in a table
rather than argue for them in prose.

(a) an instantaneous transverse velocity field, with the evaluation crop drawn
(b) the trace at a wake probe against the fitted sinusoid, over the whole record
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from matplotlib.tri import Triangulation

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from figure_common import COLORS, new_figure, save_figure, check_text_overlaps  # noqa: E402
from evaluate_common import OMEGA_0, strict_crop_indices  # noqa: E402

DATA = ROOT.parents[1].parent / "GappyPOD" / "data" / "flow_cache.npz"


def main() -> None:
    d = np.load(DATA)
    X, Y, V, times = d["X"], d["Y"], d["V"], d["times"]
    idx = strict_crop_indices(X, Y)
    x, y = X[idx], Y[idx]

    # panel (a) is drawn at equal aspect on a 12 x 6.2 domain, so its height is
    # dictated by the width it is given; the figure is sized to let it fill the
    # column rather than sit as a narrow strip in the middle.
    fig = new_figure(width="full", height=6.9)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.55, 1.0], hspace=0.30,
                          left=0.075, right=0.975, bottom=0.085, top=0.955)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---------------------------------------------------------------- panel (a)
    tri = Triangulation(x, y)
    cx, cy = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
    tri.set_mask(np.hypot(cx, cy) < 0.5)

    j = 40
    f = V[j][idx]
    lim = float(np.percentile(np.abs(f), 99))
    ax0.tricontourf(tri, f, levels=48, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax0.add_patch(__import__("matplotlib").patches.Circle(
        (0, 0), 0.5, facecolor="white", edgecolor=COLORS["reference"],
        linewidth=1.0, zorder=4))
    ax0.set_xlim(-4, 8)
    ax0.set_ylim(-3.1, 3.1)
    ax0.set_aspect("equal")
    ax0.set_xlabel("$x/D$")
    ax0.set_ylabel("$y/D$")
    ax0.set_title("Transverse velocity at one instant, on the evaluation crop",
                  loc="left", pad=6.0, fontsize=8.6)

    # the probe used in panel (b)
    probe = int(np.argmin((x - 3.0) ** 2 + (y - 0.5) ** 2))
    ax0.plot(x[probe], y[probe], marker="o", ms=4.5, mfc="none",
             mec=COLORS["reference"], mew=1.3, zorder=6)
    ax0.annotate("probe", (x[probe], y[probe]), textcoords="offset points",
                 xytext=(9, 7), fontsize=7.0, color=COLORS["reference"])

    # ---------------------------------------------------------------- panel (b)
    v = V[:, idx][:, probe]
    tau = times - times[0]
    design = np.stack([np.ones_like(tau), np.cos(OMEGA_0 * tau),
                       np.sin(OMEGA_0 * tau)], axis=1)
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    fit = design @ coef
    resid = float(np.linalg.norm(v - fit) / np.linalg.norm(v - v.mean()))

    ax1.plot(times, v, lw=1.5, color=COLORS["pressure_only"], label="DNS at the probe")
    ax1.plot(times, fit, lw=1.1, ls=(0, (3.4, 2.2)), color=COLORS["prior"],
             label=r"single harmonic at $\omega_0$")
    ax1.set_xlim(times[0], times[-1] + 0.12)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$v$")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=7.2, ncols=2, handlelength=2.4, columnspacing=1.6)
    ax1.set_title("One clean limit cycle: 3.30 shedding periods, no transient",
                  loc="left", pad=20.0, fontsize=8.6)

    T = 2 * np.pi / OMEGA_0
    for k in range(1, 4):
        ax1.axvline(times[0] + k * T, lw=0.8, color=COLORS["grid"], zorder=0)
    ax1.annotate(rf"$T = 2\pi/\omega_0 = {T:.3f}$", xy=(0.012, 0.90),
                 xycoords="axes fraction", fontsize=7.2, color=COLORS["muted"])

    fig.text(0.075, 0.014,
             rf"A single harmonic at $\omega_0$ leaves a relative residual of {resid:.2f}: "
             rf"the departure is the higher modes, not a drift in frequency.",
             fontsize=6.9, color=COLORS["muted"], ha="left")

    bad = check_text_overlaps(fig)
    if bad:
        print("TEXT ISSUES:", [b[0][:40] for b in bad[:4]])
    print(f"probe at x={x[probe]:.3f}, y={y[probe]:.3f}; "
          f"single-harmonic residual {resid:.4f}; periods {(times[-1]-times[0])/T:.2f}")
    print(save_figure(fig, ROOT / "figures" / "final" / "F_dataset"))


if __name__ == "__main__":
    main()
