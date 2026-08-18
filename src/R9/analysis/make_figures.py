"""Deliverable figures for the R9 production run (runs/R9_extracted).

Fig 1  campaign progress: wake error across every run E3 -> R9
Fig 2  instantaneous v field: truth / R7 / R9
Fig 3  fundamental-harmonic amplitude vs x + trust-region diagnosis
Fig 4  L-BFGS convergence traces (parsed from each run's out.txt)

Assumes apply_figure_style / panel_letter / end_of_line_labels are already in
the kernel namespace (figure-style skill). Run via exec() in the python tool.
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_runs as er                                    # noqa: E402
sys.path.insert(0, os.path.join(er.REPO, 'R9_wake_rescue', 'src'))
import common                                             # noqa: E402

RUNS = os.path.join(er.REPO, 'runs')
FIG = os.path.join(er.R9_ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

# ---- palette: R9 focal, comparators de-emphasised (figure-style 4.1/4.2)
C_TRUTH = '#1a1a1a'
C_R9 = '#1b6f4a'
C_R7 = '#b0472b'
C_PRIOR = '#7c9dc0'
C_OTHER = '#9a9a9a'


# ===========================================================================
# Figure 1 - campaign progress
# ===========================================================================
def fig1():
    # (label, near-wake E_v, far-wake E_v) parsed from each run's own
    # regional_evaluation.txt (the project's standard protocol)
    runs = [
        ('taps only',                      2.4949, 1.0136),
        ('+ freestream prior',             1.0538, 1.0145),
        ('+ vorticity-flux loss',          1.1435, 1.0431),
        ('+ inlet fluctuation BC',         1.0448, 1.0197),
        ('+ causal weighting',             0.9668, 0.9964),
        ('+ mean-flow & CV budgets',       0.9668, 1.0007),
        ('+ phase loss, warm start',       0.9565, 1.0016),
        ('+ fixed-phase loss',             9.4458, 40.4952),
        ('trust-street ansatz',            0.7569, 0.7881),
    ]
    tags = ['E3', 'E3b', 'R2', 'R3', 'R4', 'R5', 'R7', 'R8', 'R9']
    ys = np.arange(len(runs))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, ((lab, nw, fw), tag) in enumerate(zip(runs, tags)):
        y = ys[i]
        focal = (tag == 'R9')
        col = C_R9 if focal else C_OTHER
        ax.plot([nw, fw], [y, y], '-', color=col, lw=2.6 if focal else 1.2,
                alpha=1.0 if focal else 0.55, zorder=2, solid_capstyle='round')
        ax.plot(nw, y, 'o', ms=6 if focal else 4.5, color=col, zorder=3)
        ax.plot(fw, y, 's', ms=6 if focal else 4.5, mfc='white',
                mec=col, mew=1.8 if focal else 1.2, zorder=3)
        ax.text(2.6, y, f'{tag}  {lab}', va='center', ha='left',
                fontsize=7, color=col if focal else '#4a4a4a',
                fontweight='bold' if focal else 'normal')

    ax.axvline(1.0, color='#555555', lw=0.9, ls=':', zorder=1)
    ax.text(1.0, ys[0] + 0.62, 'no wake predicted', fontsize=6.5,
            color='#555555', ha='center')
    # R4 also falls below 1.0, but by 3.3% (near) / 0.4% (far) - i.e. within
    # noise of "no wake". Annotated so the figure is self-consistent with its
    # title (figure-style 1.3/1.4): the claim is MARGIN, not existence.
    i_r4 = tags.index('R4')
    ax.annotate('clears by\nonly 0.4%',
                (0.955, ys[i_r4]), xytext=(0.40, ys[i_r4]),
                fontsize=6.0, color='#4a4a4a', ha='center', va='center',
                arrowprops=dict(arrowstyle='-', color='#8a8a8a', lw=0.6,
                                shrinkA=1, shrinkB=2))
    ax.set_xscale('log')
    ax.set_xlim(0.30, 130)
    ax.set_xticks([0.6, 1, 2, 5, 10, 40])
    ax.set_xticklabels(['0.6', '1', '2', '5', '10', '40'])
    ax.set_yticks([])
    ax.set_ylim(ys.min() - 0.9, ys.max() + 1.0)
    ax.set_xlabel('relative $L_2$ error of transverse velocity $v$'
                  '   (lower = better)')
    # Verified against the plotted values: R4 and R9 are the only runs with
    # both metrics < 1.0, but R4's margin is 3.3%/0.4% vs R9's 24%/21%.
    ax.set_title('The trust-street ansatz is the first run to beat "no wake '
                 'predicted"\nby a real margin (21-24%, vs 0.4-3% for the '
                 'best loss-term fix)')
    # key
    ax.plot([], [], 'o', color='#4a4a4a', ms=5, label='near wake (0 < x < 3D)')
    ax.plot([], [], 's', mfc='white', mec='#4a4a4a', ms=5,
            label='far wake (x > 3D)')
    ax.legend(frameon=False, fontsize=7, loc='lower right',
              bbox_to_anchor=(1.0, -0.02))
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_campaign_progress.png'), dpi=300)
    return fig


# ===========================================================================
# shared: evaluate the two production runs on the truth node set
# ===========================================================================
def load_fields():
    x, y, times, U, V, P = common.load_truth_fields()
    out = {}
    for tag, pat, trust in (('R7', 'runs/R7_extracted/*/', False),
                            ('R9', 'runs/R9_extracted/*/', True)):
        d = glob.glob(os.path.join(er.REPO, pat))[0]
        run = er.load_run(d, trust=trust)
        u, v, p = er.eval_run(run, x, y)
        out[tag] = dict(u=u, v=v, p=p, dir=d, sp=run['sp'])
    return x, y, times, U, V, P, out


# ===========================================================================
# Figure 2 - instantaneous v field
# ===========================================================================
def fig2(x, y, times, V, fields):
    gx = np.linspace(-4, 8, 460)
    gy = np.linspace(-2.6, 2.6, 200)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([x, y], 1)
    mask = np.hypot(GX, GY) < 0.5

    panels = [('reference CFD', V[0], C_TRUTH),
              ('prior best PINN (R7)', er.reconstruct(fields['R7']['v'], times)[0], C_R7),
              ('trust-street ansatz (R9)', er.reconstruct(fields['R9']['v'], times)[0], C_R9)]

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True, sharey=True)
    vmax = 0.55
    for ax, (lab, f, col) in zip(axes, panels):
        Z = griddata(pts, f, (GX, GY), method='linear')
        Z = np.where(mask, np.nan, Z)
        im = ax.pcolormesh(GX, GY, Z, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           shading='auto', rasterized=True)
        ax.add_patch(plt.Circle((0, 0), 0.5, fc='0.88', ec='0.35', lw=0.8,
                                zorder=4))
        ax.set_ylabel('$y/D$')
        ax.text(0.012, 0.94, lab, transform=ax.transAxes, va='top', ha='left',
                fontsize=7.5, color=col, fontweight='bold',
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.6))
        ax.set_aspect('equal')
        ax.set_xlim(-4, 8)
    axes[-1].set_xlabel('$x/D$')
    axes[0].set_title('Instantaneous transverse velocity at one phase '
                      'of the shedding cycle')
    cb = fig.colorbar(im, ax=axes, shrink=0.72, pad=0.02, aspect=28)
    cb.set_label('$v / U_\\infty$')
    fig.savefig(os.path.join(FIG, 'fig2_v_field_snapshots.png'), dpi=300,
                bbox_inches='tight')
    return fig


# ===========================================================================
# Figure 3 - fundamental amplitude vs x + trust-region diagnosis
# ===========================================================================
def fig3(x, y, fields):
    """Amplitude convention: network modes are one-sided
    (q = Re[sum q_k e^{ikwt}]); truth modes from common.fit_modes are
    two-sided (q = c0 + sum 2Re[c_k e^{ikwt}]). Physical amplitude of the
    fundamental is therefore |q_1| for the networks and 2|c_1| for truth."""
    xm, ym, tm = common.load_truth_modes()
    assert np.allclose(xm, x)
    band = np.abs(y) < 0.8
    xb = np.arange(0.75, 7.9, 0.25)

    def profile(vals):
        return np.array([np.abs(vals[band & (np.abs(x - xc) < 0.125)]).mean()
                         for xc in xb])

    truth = profile(2.0 * tm['v'][1])
    r7 = profile(fields['R7']['v'][1])
    r9 = profile(fields['R9']['v'][1])

    sp = fields['R9']['sp']
    S_u, S_v, S_p = er.street_modes(x.astype(float), y.astype(float), sp, nk=1)
    fb = er.f_bc5(x, y)
    S_masked = np.abs(S_v[0]) * fb
    prior = profile(S_masked)
    rho, cap = 0.6, float(sp['scale_p']) * 0  # cap read below
    cap = 0.12
    lo = profile(np.maximum((1 - rho) * np.abs(S_v[0]) - cap, 0) * fb)
    hi = profile(((1 + rho) * np.abs(S_v[0]) + cap) * fb)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)

    ax = axes[0]
    ax.plot(xb, truth, '-', color=C_TRUTH, lw=2.0)
    ax.plot(xb, r9, '-', color=C_R9, lw=2.4)
    ax.plot(xb, prior, '--', color=C_PRIOR, lw=1.5)
    ax.plot(xb, r7, '-', color=C_R7, lw=1.4)
    for lab, arr, col in (('reference CFD', truth, C_TRUTH),
                          ('R9', r9, C_R9),
                          ('analytic street\n(no training)', prior, C_PRIOR),
                          ('R7', r7, C_R7)):
        ax.annotate(lab, (xb[-1], arr[-1]), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=6.5,
                    color=col)
    ax.set_xlabel('$x/D$')
    ax.set_ylabel('fundamental-harmonic amplitude of $v$')
    # ratio verified numerically before wording this title (figure-style 1.4):
    # far-wake (x>3) band-mean amplitude truth 0.519 vs R9 0.152 -> 29%
    ax.set_title('R9 sustains a wake downstream, but at only\n'
                 '29% of the true amplitude')
    ax.set_xlim(0.6, 10.6)
    ax.margins(y=0.08)

    ax = axes[1]
    ax.fill_between(xb, lo, hi, color=C_R9, alpha=0.14, lw=0,
                    label='reachable by the ansatz')
    ax.plot(xb, lo, '-', color=C_R9, lw=1.0, alpha=0.6)
    ax.plot(xb, truth, '-', color=C_TRUTH, lw=2.0, label='reference CFD')
    ax.plot(xb, r9, '-', color=C_R9, lw=2.4, label='R9 trained')
    ax.set_xlabel('$x/D$')
    ax.set_title('Why it fell short: the trust region allowed\n'
                 'amplitudes down to ~8% of truth')
    ax.set_xlim(0.6, 8.0)
    ax.legend(frameon=False, fontsize=6.5, loc='upper right')
    ax.annotate('permitted floor\n($\\rho=0.6$)', (5.5, lo[np.argmin(np.abs(xb - 5.5))]),
                xytext=(4.6, 0.22), textcoords='data', fontsize=6.5,
                color=C_R9, ha='center',
                arrowprops=dict(arrowstyle='-', color=C_R9, lw=0.7))
    for a in axes:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_amplitude_and_trust_region.png'),
                dpi=300)
    return fig


# ===========================================================================
# Figure 4 - convergence traces
# ===========================================================================
def fig4(x, y, fields):
    """Mean flow along the wake centreline.

    Deliberately NOT a cross-run loss-trace plot: R3/R7 trained with extra
    loss terms (vorticity-flux, mean-flow, phase) that R9 drops, so their
    objective values are not comparable quantities (figure-style 1.2).

    The mean flow is the part of the reconstruction with NO analytic prior -
    k=0 is a free network in the trust ansatz - so this panel isolates the
    network's own contribution.
    """
    xm, ym, tm = common.load_truth_modes()
    assert np.allclose(xm, x)
    band = np.abs(y) < 0.15
    xb = np.arange(-3.5, 7.9, 0.2)

    def prof(vals):
        return np.array([vals[band & (np.abs(x - xc) < 0.1)].mean()
                         for xc in xb])

    inside = np.abs(xb) < 0.5
    truth = prof(tm['u'][0].real)
    r9 = prof(fields['R9']['u'][0].real)
    r7 = prof(fields['R7']['u'][0].real)
    for a in (truth, r9, r7):
        a[inside] = np.nan

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.axhline(0.0, color='#bbbbbb', lw=0.8, zorder=1)
    ax.plot(xb, truth, '-', color=C_TRUTH, lw=2.0, zorder=3)
    ax.plot(xb, r9, '-', color=C_R9, lw=2.4, zorder=4)
    ax.plot(xb, r7, '-', color=C_R7, lw=1.4, zorder=2)
    for lab, arr, col in (('reference CFD', truth, C_TRUTH),
                          ('R9', r9, C_R9), ('R7', r7, C_R7)):
        ax.annotate(lab, (xb[-1], arr[-1]), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=6.5,
                    color=col)
    ax.axvspan(-0.5, 0.5, color='0.90', lw=0, zorder=0)
    ax.text(0.0, ax.get_ylim()[0] * 0.92, 'cylinder', ha='center',
            fontsize=6.5, color='#666666')
    ax.set_xlabel('$x/D$')
    ax.set_ylabel('mean streamwise velocity $\\bar{u}/U_\\infty$')
    # Title states both halves of what the panel shows (figure-style 1.4):
    # R9 reproduces the reversed-flow bubble (min ~ -0.18, truth ~ -0.17)
    # which R7 misses entirely, but at x=8 reaches only 0.35 vs truth 0.76.
    ax.set_title('R9 captures the recirculation bubble R7 misses, but '
                 'under-recovers\nthe mean deficit downstream — with no '
                 'analytic prior at $k=0$')
    ax.set_xlim(-3.6, 9.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_mean_flow_centreline.png'), dpi=300)
    return fig
