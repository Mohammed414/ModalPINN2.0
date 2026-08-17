"""Final deliverable figures for R9_wake_rescue.

Fig 1: mode-1 |v| amplitude vs x (truth / dead baseline / street / trust)
Fig 2: v-velocity snapshots (truth / baseline / trust) at t=0
Fig 3: regional E_v bars across the method line (R7 numbers are hardcoded
       constants transcribed from runs/R7_extracted/*/regional_evaluation.txt;
       our arms are read live from results/*.json)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

RES = os.path.join(common.R9, 'results')
FIG = os.path.join(common.R9, 'figures')


def mode1_profile(vm, x, y, xbins, band=0.8):
    out = []
    for xb in xbins:
        m = (np.abs(y) < band) & (np.abs(x - xb) < 0.125)
        out.append(np.abs(vm[m]).mean() if m.any() else np.nan)
    return np.array(out)


def main():
    from apply_style import apply  # noqa
    x, ytruth = None, None

    xm, ym, tmodes = common.load_truth_modes()
    xbins = np.arange(0.5, 8.0, 0.25)
    prof_truth = mode1_profile(tmodes['v'][1], xm, ym, xbins)

    profs = {'truth (CFD, eval only)': prof_truth}
    labels = {
        'arm_baseline_w40_s0': 'ModalPINN loss (dead wake)',
        'arm_trust_v2': 'street-anchored trust ansatz',
    }
    for tag, lab in labels.items():
        d = np.load(os.path.join(RES, f'{tag}_modes.npz'))
        profs[lab] = mode1_profile(d['v'][1], d['x'], d['y'], xbins)
    sd = np.load(os.path.join(RES, 'street_modes.npz'))
    profs['analytic street alone'] = mode1_profile(sd['v1'], sd['x'],
                                                   sd['y'], xbins)

    # ---------------- Fig 1
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colors = {'truth (CFD, eval only)': '#222222',
              'ModalPINN loss (dead wake)': '#c23b22',
              'analytic street alone': '#7aa6c2',
              'street-anchored trust ansatz': '#1f6f43'}
    for lab, pr in profs.items():
        lw = 2.4 if 'trust' in lab or 'truth' in lab else 1.6
        ls = '--' if lab.startswith('analytic') else '-'
        ax.plot(xbins, pr, ls, color=colors[lab], lw=lw)
        yend = pr[np.isfinite(pr)][-1]
        ax.annotate(lab, (xbins[-1], yend), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=7,
                    color=colors[lab])
    ax.set_xlabel('x / D downstream')
    ax.set_ylabel('mode-1 |v| amplitude (band |y| < 0.8)')
    ax.set_title('The oscillating wake survives downstream only with the '
                 'street-anchored ansatz')
    ax.set_xlim(0.4, 11.2)
    ax.margins(y=0.06)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_mode1_amplitude_vs_x.png'), dpi=300)

    # ---------------- Fig 2: snapshots
    from scipy.interpolate import griddata
    x, y, times, U, V, P = common.load_truth_fields()
    gx = np.linspace(common.LXMIN, common.LXMAX, 480)
    gy = np.linspace(common.LYMIN, common.LYMAX, 320)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([x, y], 1)

    def v_at_t0(vm_modes):
        v = vm_modes[0].real.copy()
        for k in range(1, 4):
            v = v + 2 * vm_modes[k].real
        return v

    snaps = {}
    snaps['truth (CFD)'] = V[0]
    d = np.load(os.path.join(RES, 'arm_baseline_w40_s0_modes.npz'))
    snaps['ModalPINN loss (dead wake)'] = v_at_t0([d['v'][k] for k in range(4)])
    d = np.load(os.path.join(RES, 'arm_trust_v2_modes.npz'))
    snaps['street-anchored trust ansatz'] = v_at_t0([d['v'][k] for k in range(4)])

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.2), sharex=True)
    vmax = 0.6
    for ax, (lab, f) in zip(axes, snaps.items()):
        Z = griddata(pts, f, (GX, GY), method='linear')
        r = np.hypot(GX, GY)
        Z = np.where(r < common.R_C, np.nan, Z)
        im = ax.pcolormesh(GX, GY, Z, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           shading='auto')
        ax.add_patch(plt.Circle((0, 0), common.R_C, fc='0.85', ec='0.4'))
        ax.set_ylabel('y / D')
        ax.text(0.01, 0.97, lab, transform=ax.transAxes, va='top',
                fontsize=8)
        ax.set_aspect('equal')
    axes[-1].set_xlabel('x / D')
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label('v / U∞  (t = t₀)')
    fig.savefig(os.path.join(FIG, 'fig2_v_snapshots.png'), dpi=300,
                bbox_inches='tight')

    # ---------------- Fig 3: regional E_v bars
    methods = []
    # R7 numbers: hardcoded constants transcribed from
    # runs/R7_extracted/*/regional_evaluation.txt (near-wake E_v, far-wake
    # E_v). NOT read live - update by hand if that file ever changes.
    methods.append(('R7 (best prior PINN)', 0.9565, 1.0016))
    with open(os.path.join(RES, 'arm_baseline_w40_s0.json')) as f:
        b = json.load(f)['table']
    methods.append(('testbed baseline\n(ModalPINN loss)',
                    b['near-wake']['E_v'], b['far-wake']['E_v']))
    with open(os.path.join(RES, 'street_standalone.json')) as f:
        s = json.load(f)
    methods.append(('analytic street alone\n(taps+classical physics)',
                    s['near-wake']['E_v'], s['far-wake']['E_v']))
    with open(os.path.join(RES, 'arm_trust_v2.json')) as f:
        t = json.load(f)['table']
    methods.append(('street-anchored\ntrust ansatz', t['near-wake']['E_v'],
                    t['far-wake']['E_v']))

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    xs = np.arange(len(methods))
    w = 0.36
    nearv = [m[1] for m in methods]
    farv = [m[2] for m in methods]
    b1 = ax.bar(xs - w / 2, nearv, w, color='#8f5b96', label='near-wake')
    b2 = ax.bar(xs + w / 2, farv, w, color='#c9a0ce', label='far-wake')
    ax.axhline(1.0, color='0.4', lw=0.8, ls=':')
    ax.text(len(methods) - 0.52, 1.02, 'E_v = 1 (wake absent)', fontsize=6.5,
            color='0.35', ha='right')
    for bars in (b1, b2):
        for bb in bars:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.02,
                    f'{bb.get_height():.2f}', ha='center', fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in methods], fontsize=7)
    ax.set_ylabel('relative L2 error of v')
    ax.set_title('32 taps + physics only: wake error drops below the '
                 '"wake absent" line for the first time')
    ax.legend(frameon=False, fontsize=7, loc='upper right',
              bbox_to_anchor=(0.99, 0.88))
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_regional_Ev.png'), dpi=300)
    print('figures saved to', FIG)


if __name__ == '__main__':
    main()
