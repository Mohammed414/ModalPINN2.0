"""
The CRITICAL DELIVERABLE figure: mode-1 amplitude versus downstream distance
x, estimate vs truth, for the three existing (pre-repair) runs -- plus the
per-x amplitude/phase split and the window-sensitivity table that any future
ranking claim must be checked against.

Reads experiments/metrics_v2_baseline.{json,npz} written by
run_metrics_v2_baseline.py. Writes figures/metrics_v2_baseline.png.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, '..', 'experiments')
FIG_DIR = os.path.join(HERE, '..', 'figures')

RUNS = ['free_run', 'enkf', 'shuffled']
NICE = {'free_run': 'free run (no data)', 'enkf': 'EnKF (true taps)',
        'shuffled': 'shuffled taps (control)'}
COL = {'free_run': '#7f7f7f', 'enkf': '#1f77b4', 'shuffled': '#d62728'}


def main():
    def latest(stem, ext):
        cands = [f for f in os.listdir(EXP_DIR)
                 if f.startswith(stem) and f.endswith(ext)]
        return os.path.join(EXP_DIR, sorted(cands)[-1])

    jpath = latest('metrics_v2_baseline', '.json')
    npath = latest('metrics_v2_baseline', '.npz')
    print('reading %s and %s' % (os.path.basename(jpath), os.path.basename(npath)))
    with open(jpath) as f:
        S = json.load(f)
    A = np.load(npath)
    gx = A['gx']
    xw = gx >= 1.0

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                         'legend.fontsize': 7.6, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.grid': True, 'grid.alpha': 0.25,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 150, 'savefig.dpi': 200})
    fig, axs = plt.subplots(3, 2, figsize=(11.6, 11.4))

    # ---- (0,0) THE deliverable curve -------------------------------------
    ax = axs[0, 0]
    ax.plot(gx[xw], A['truth_v1_prof_max'][xw], '-', color='k', lw=2.4, label='truth', zorder=4)
    for r in RUNS:
        ax.plot(gx[xw], A['%s_v1_prof_max' % r][xw], '-', color=COL[r], lw=1.7, label=NICE[r])
    tp = S['truth']
    ax.plot([tp['v1_peak_x']], [tp['v1_peak']], 'k*', ms=12, zorder=5)
    ax.annotate('truth peak %.3f at $x$ = %.2f' % (tp['v1_peak'], tp['v1_peak_x']),
                xy=(tp['v1_peak_x'], tp['v1_peak']), xytext=(3.15, 0.735), fontsize=7.8)
    ax.annotate('truth still %.0f%% of peak at $x$ = 7:\na street that PERSISTS'
                % (100 * tp['v1_persistence']),
                xy=(7.0, tp['v1_at_x7']), xytext=(4.75, 0.685), fontsize=7.8,
                ha='left', va='top',
                arrowprops=dict(arrowstyle='->', lw=1.0, color='k',
                                shrinkB=2, connectionstyle='arc3,rad=0.22'))
    ax.set_xlabel('downstream distance $x$   [diameters]')
    ax.set_ylabel(r'$\max_y\,|\hat v_1|(x)$')
    ax.set_title('Mode-1 amplitude vs downstream distance\n'
                 r'(fitted at $\omega_0$ = 1.036; the curve ModalPINN gets wrong)')
    ax.set_xlim(1, 8); ax.set_ylim(0, 0.78); ax.legend(loc='lower left')

    # ---- (0,1) same, refitted at each run's OWN frequency -----------------
    ax = axs[0, 1]
    ax.plot(gx[xw], A['truth_v1_prof_max'][xw], '-', color='k', lw=2.4, label='truth', zorder=4)
    for r in RUNS:
        w = S['runs'][r]['windows']['full']['modal_v_ownfreq']['omega']
        ax.plot(gx[xw], A['%s_v1_prof_max_ownfreq' % r][xw], '-', color=COL[r], lw=1.7,
                label=r'%s, $\omega_s$=%.3f' % (NICE[r], w))
    ax.set_xlabel('downstream distance $x$   [diameters]')
    ax.set_ylabel(r'$\max_y\,|\hat v_1|(x)$')
    ax.set_title('Same, refitted at the shedding frequency each run actually has\n'
                 'the deficit at $\\omega_0$ is mostly a FREQUENCY error, not a missing wake')
    ax.set_xlim(1, 8); ax.set_ylim(0, 0.78); ax.legend(loc='lower left')

    # ---- (1,0) per-x amplitude ratio and phase error ----------------------
    ax = axs[1, 0]
    for r in RUNS:
        ax.plot(gx[xw], A['%s_v1_amp_ratio_x' % r][xw], '-', color=COL[r], lw=1.7,
                label=NICE[r])
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.text(7.75, 1.02, 'correct', ha='right', fontsize=7.4)
    ax.set_xlabel('downstream distance $x$   [diameters]')
    ax.set_ylabel(r'amplitude ratio  $|c(x)|$    (1 = correct)')
    ax.set_title('AMPLITUDE error vs $x$  (phase-blind)\n'
                 r'$c(x)=\langle \hat v_1^{\rm true},\hat v_1^{\rm est}\rangle_y/\|\hat v_1^{\rm true}\|_y^2$')
    ax.set_xlim(1, 8); ax.set_ylim(0, 1.35); ax.legend(loc='lower left')

    axb = axs[1, 1]
    for r in RUNS:
        axb.plot(gx[xw], A['%s_v1_phase_err_x' % r][xw], '-', color=COL[r], lw=1.7,
                 label=NICE[r])
    axb.axhline(0.0, color='k', ls='--', lw=1.0)
    axb.set_xlabel('downstream distance $x$   [diameters]')
    axb.set_ylabel(r'phase error  $\arg c(x)$   [rad]')
    axb.set_title('PHASE error vs $x$, reported SEPARATELY\n'
                  'a wrong-phase wake is no longer scored as a missing wake')
    axb.set_xlim(1, 8); axb.legend(loc='lower left')

    # ---- (2,0) and (2,1) window sensitivity -------------------------------
    wnames = ['full', 'half_1', 'half_2', 'third_1', 'third_2', 'third_3']
    xpos = np.arange(len(wnames))
    for ax, comp, ttl, sub in [
        (axs[2, 0], 'modal_v', r'Window sensitivity of the amplitude deficit, fitted at $\omega_0$',
         'window-length-dependent leakage: rankings flip between cuts'),
        (axs[2, 1], 'modal_v_ownfreq', 'Same deficit, frequency error removed first',
         'window-stable (free run 0.099-0.101 across all six cuts)')]:
        for r in RUNS:
            y = [S['runs'][r]['windows'][w][comp]['k1_amp_ratio_deficit'] for w in wnames]
            ax.plot(xpos, y, 'o-', color=COL[r], lw=1.7, ms=5, label=NICE[r])
        ax.set_xticks(xpos)
        ax.set_xticklabels(['full\n20 t.u.', 'half 1\n10', 'half 2\n10',
                            'third 1\n6.6', 'third 2\n6.6', 'third 3\n6.6'])
        ax.set_ylabel(r'$|\,1-\overline{|c(x)|}\,|$   (lower = better)')
        ax.set_ylim(0, 0.215)
        ax.set_title('%s\n%s' % (ttl, sub))
        ax.legend(loc='upper right')

    fig.suptitle('metrics_v2 on the three existing (pre-repair) runs '
                 '- before/after baseline for the repaired filter',
                 fontsize=11, y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = os.path.join(FIG_DIR, 'metrics_v2_baseline.png')
    fig.savefig(out)
    print('Wrote %s' % out)

    r_ = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    tickset = set()
    for ax in fig.axes:
        tickset |= set(ax.get_xticklabels()) | set(ax.get_yticklabels())
    bad = [(a.get_text()[:26], b.get_text()[:26])
           for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
           if ba.overlaps(bb) and not (a in tickset and b in tickset)]
    print('text-overlap check: %d pair(s)%s' % (len(bad), '' if not bad else ' -> %s' % bad[:6]))

    # numbers for the record
    print('\nper-x amplitude ratio |c(x)| at selected stations (fitted at omega_0):')
    print('  %-24s %s' % ('x =', ''.join('%8.1f' % v for v in [1, 2, 3, 4, 5, 6, 7])))
    for r in RUNS:
        idx = [int(np.argmin(np.abs(gx - v))) for v in [1, 2, 3, 4, 5, 6, 7]]
        print('  %-24s %s' % (NICE[r], ''.join('%8.3f' % A['%s_v1_amp_ratio_x' % r][i] for i in idx)))
    print('\nper-x phase error arg c(x) [rad]:')
    for r in RUNS:
        idx = [int(np.argmin(np.abs(gx - v))) for v in [1, 2, 3, 4, 5, 6, 7]]
        print('  %-24s %s' % (NICE[r], ''.join('%+8.3f' % A['%s_v1_phase_err_x' % r][i] for i in idx)))


if __name__ == '__main__':
    main()
