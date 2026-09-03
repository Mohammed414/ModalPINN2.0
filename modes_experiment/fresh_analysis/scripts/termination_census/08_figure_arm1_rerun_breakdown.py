"""Figure: loss and gradient-norm trajectories for the original arm 01 and
the matched-effort re-run, showing they are bit-identical for 99 iterations
then diverge, and that the re-run's first L-BFGS call dies mid a much larger
gradient spike than the original's."""
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[4]
ARMS = REPO / 'modes_experiment' / 'runs' / 'arms'
NEWLOG = REPO / 'modes_experiment' / 'experiment' / 'notebooks' / 'matched_effort' / 'zx' / 'baseline_physics_only_K3_matched' / 'train_log.txt'
FA = REPO / 'modes_experiment' / 'fresh_analysis'
sys.path.insert(0, str(FA))
from figure_common import COLORS, new_figure, save_figure, check_text_overlaps  # noqa: E402


def fort(x):
    return float(x.replace('D', 'E').replace('d', 'e'))


def parse_its(text, call=0):
    seg = text.split('RUNNING THE L-BFGS-B CODE')[1:][call]
    return np.array([[int(a), fort(b), fort(c)] for a, b, c in
                      re.findall(r'At iterate\s*(\d+)\s+f=\s+(\S+)\s+\|proj g\|=\s+(\S+)', seg)])


A_old = parse_its((ARMS / '01_baseline_physics_only' / 'train_log.txt').read_text())
A_new = parse_its(NEWLOG.read_text())


def exit_blow(A):
    """Terminal gradient blow-up: median |proj g| over the final 50 logged
    iterates, divided by its level over the preceding window.

    This is deliberately the identical definition used by
    06_all_arms_death_census.py (and therefore by F_termination_anatomy.png and
    the findings.md numbers). An earlier version of this figure sliced the
    window by iteration NUMBER while the census sliced by array INDEX, which
    gave 5.7x/25.5x here against 6.5x/26.7x there -- the same caption wording
    attached to two different numbers. The choice does not affect the
    conclusion (Spearman rho -0.733 vs -0.721 across the 17 runs), but the two
    figures must not disagree.
    """
    g = A[:, 2]
    last = len(A) - 1
    return np.median(g[last - 50:last + 1]) / np.median(g[max(0, last - 600):last - 100])


BLOW_OLD, BLOW_NEW = exit_blow(A_old), exit_blow(A_new)

fig = new_figure(width="full", height=4.6)
gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.088, right=0.985, bottom=0.375, top=0.845)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

runs_plot = [('original run (kept)', A_old, COLORS['pressure_only']),
             ('overnight re-run (discarded)', A_new, COLORS['accent'])]
for lbl, A, c in runs_plot:
    ax1.plot(A[:, 0], A[:, 1], color=c, lw=1.4, label=lbl)
    ax1.plot([A[-1, 0]], [A[-1, 1]], 'o', color=c, ms=6, mec='white', mew=1.0)
    ax2.plot(A[:, 0], A[:, 2], color=c, lw=0.6, alpha=0.40)
    k = 51
    med = np.array([np.median(A[max(0, i - k):i + 1, 2]) for i in range(len(A))])
    ax2.plot(A[:, 0], med, color=c, lw=1.8, label=lbl)
    ax2.plot([A[-1, 0]], [A[-1, 2]], 'o', color=c, ms=6, mec='white', mew=1.0)

for ax in (ax1, ax2):
    ax.set_yscale('log')
    ax.set_xlim(-100, 5500)
    ax.set_xlabel('accepted L-BFGS iteration')

ax1.set_ylabel('training loss')
ax1.set_ylim(2e-4, 6.0)
ax1.set_title('a  Bit-identical for 99 iterations, then they diverge', loc='left', pad=8.0, fontsize=8.6)
ax1.legend(loc='upper right', frameon=False, fontsize=7.2)
ax1.annotate('quits at 2,516\n$f=1.18\\times10^{-3}$', xy=(2516, 1.1768e-3), xytext=(2780, 2.2e-2),
             fontsize=7.0, color=COLORS['accent'], ha='left',
             arrowprops=dict(arrowstyle='-|>', color=COLORS['muted'], lw=0.8, shrinkB=4))
ax1.annotate('quits at 5,081\n$f=3.00\\times10^{-4}$', xy=(5081, 3.0003e-4), xytext=(3450, 8.5e-4),
             fontsize=7.0, color=COLORS['pressure_only'], ha='left',
             arrowprops=dict(arrowstyle='-|>', color=COLORS['muted'], lw=0.8, shrinkB=4))

ax2.set_ylabel(r'projected gradient norm  $\|\nabla f\|_\infty$')
ax2.set_ylim(2e-4, 6e-1)
ax2.set_title('b  Both destabilise; the re-run far more', loc='left', pad=8.0, fontsize=8.6)
ax2.annotate(f'{BLOW_NEW:.0f}$\\times$ rise, loss frozen', xy=(2510, 3.2e-2), xytext=(2820, 2.4e-1),
             fontsize=7.0, color=COLORS['accent'], ha='left',
             arrowprops=dict(arrowstyle='-|>', color=COLORS['muted'], lw=0.8, shrinkB=4))
ax2.annotate(f'{BLOW_OLD:.1f}$\\times$', xy=(5081, 2.8e-3), xytext=(4500, 2.0e-2),
             fontsize=7.0, color=COLORS['pressure_only'], ha='center',
             arrowprops=dict(arrowstyle='-|>', color=COLORS['muted'], lw=0.8, shrinkB=4))
# Panel a already carries the colour legend, so repeating it here only puts a
# second box on top of the data. Keep the thin/thick note, which is specific to
# this panel, and give it the space the legend was using.
ax2.text(-40, 2.4e-4, 'thin: per iteration    thick: 51-iteration running median',
         fontsize=6.5, color=COLORS['muted'])

fig.suptitle('Why the arm-1 re-run stopped short: a numerical breakdown, not convergence',
             y=0.955, fontsize=10.5)
fig.text(0.088, 0.020,
    "Both runs: 32 wall pressure taps, pressure-only + physics, identical flags, identical seed, byte-identical source files.\n"
    "They agree to every printed digit through iteration 99, then float32 non-determinism in the GPU reduction over 50,000\n"
    "collocation points compounds chaotically. Both end on SciPy's REL_REDUCTION_OF_F test, which at ftol = 1e-12 lies below\n"
    "float32 resolution ($\\sim$1e-10 at these losses) and therefore fires whenever a single line search fails to move the loss at\n"
    f"all. Terminal gradient blow-up, median of the final 50 logged iterates over the preceding window: {BLOW_NEW:.1f}$\\times$ here against {BLOW_OLD:.1f}$\\times$\n"
    "for the original. The two runs that reached $5$-$6\\times10^{-5}$ (8 and 16 taps, 25,331 and 20,303 iterations) show no blow-up\n"
    "at all (0.9$\\times$, 1.2$\\times$) - they exhausted float32 resolution instead. When the instability strikes is not controllable.",
    fontsize=6.6, color=COLORS['muted'], va='bottom', linespacing=1.5)

bad = check_text_overlaps(fig)
if bad:
    print('TEXT ISSUES:', [b[0][:38] for b in bad[:5]])
print(save_figure(fig, REPO / 'modes_experiment' / 'fresh_analysis' / 'figures' / 'final' / 'F_arm1_rerun_breakdown'))
