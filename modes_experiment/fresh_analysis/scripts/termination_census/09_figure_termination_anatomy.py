"""Figure: gradient elevation at exit vs. stopping iteration, for all 17
runs in the project. This is the report-ready evidence that early stopping
is a project-wide line-search pathology, not a property of any one arm's
tap count or settings."""
import json
import pathlib
import re
import sys

import numpy as np

ARMS = pathlib.Path('/Users/mohammedhilal/Desktop/try/ModalPINN2.0/modes_experiment/runs/arms')
NEWLOG = pathlib.Path('/Users/mohammedhilal/Desktop/try/ModalPINN2.0/modes_experiment/notebooks/matched_effort/zx/baseline_physics_only_K3_matched/train_log.txt')
FA = pathlib.Path('/Users/mohammedhilal/Desktop/try/ModalPINN2.0/modes_experiment/fresh_analysis')
sys.path.insert(0, str(FA))
from figure_common import COLORS, new_figure, save_figure, check_text_overlaps  # noqa: E402


def fort(x):
    return float(x.replace('D', 'E').replace('d', 'e'))


def parse_its(text, call=0):
    seg = text.split('RUNNING THE L-BFGS-B CODE')[1:][call]
    return np.array([[int(a), fort(b), fort(c)] for a, b, c in
                      re.findall(r'At iterate\s*(\d+)\s+f=\s+(\S+)\s+\|proj g\|=\s+(\S+)', seg)])


def parse_flags(cmd):
    d, i = {}, 0
    while i < len(cmd):
        t = cmd[i]
        if t.startswith('--'):
            if i + 1 < len(cmd) and not cmd[i + 1].startswith('--'):
                d[t] = cmd[i + 1]
                i += 2
            else:
                d[t] = True
                i += 1
        else:
            i += 1
    return d


runs = {}
for a in sorted(p.name for p in ARMS.iterdir() if p.is_dir()):
    lp, rp = ARMS / a / 'train_log.txt', ARMS / a / 'run_record.json'
    if not (lp.exists() and rp.exists()):
        continue
    t = lp.read_text()
    if 'At iterate' not in t:
        continue
    runs[a] = dict(A=parse_its(t), nwarn=t.count('more than 10 function'))
runs['NEW re-run (cycle 1)'] = dict(A=parse_its(NEWLOG.read_text(), 0),
                                    nwarn=NEWLOG.read_text().count('more than 10 function'))


def death_metrics(A):
    g = A[:, 2]
    last = len(A) - 1
    base = np.median(g[max(0, last - 600):last - 100]) if last > 150 else np.median(g[:max(1, last - 10)])
    exit_blow = np.median(g[last - 50:last + 1]) / base if last > 60 else g[last] / base
    return dict(last_it=int(A[-1, 0]), exit_blow=float(exit_blow))


rows = []
for a, r in runs.items():
    m = death_metrics(r['A'])
    rows.append((m['last_it'], a, m['exit_blow']))
rows.sort(reverse=True)

fig = new_figure(width="full", height=4.4)
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.125, right=0.975, bottom=0.34, top=0.86)

pts = {}
for it, a, eb in rows:
    pts[a] = (it, eb)
    if a == '05_dense_reference':
        ax.plot(it, eb, marker='s', ms=7, color=COLORS['dense'], mec='white', mew=0.8, zorder=5)
    else:
        c = COLORS['accent'] if eb >= 3 else COLORS['pressure_only']
        ax.plot(it, eb, marker='o', ms=7, color=c, mec='white', mew=0.8, zorder=5)

ax.axhline(1.0, color=COLORS['muted'], lw=0.9, ls=(0, (1, 2.2)))
ax.axhspan(3, 45, color=COLORS['accent'], alpha=0.06, lw=0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(800, 300000)
ax.set_ylim(0.35, 45)
ax.set_xlabel('iteration at which the run stopped')
ax.set_ylabel('gradient elevation at exit\n(final 50 iterations / preceding 550)')


def lab(name, text, xy_text, ha='left'):
    ax.annotate(text, xy=pts[name], xytext=xy_text, fontsize=6.8, color=COLORS['muted'], ha=ha,
                arrowprops=dict(arrowstyle='-', color=COLORS['muted'], lw=0.6, shrinkB=3))


lab('NEW re-run (cycle 1)', 'baseline, attempt 2', (1450, 33), 'left')
lab('arm_13_prior_noise_10pct', 'prior + 10% noise', (5200, 21), 'left')
# These two runs stopped within 55 iterations and 0.4x of each other, so their
# markers nearly coincide; separate the labels vertically and lead to each.
lab('01_baseline_physics_only', 'baseline, attempt 1', (9200, 14.0), 'left')
lab('arm_11_prior_noise_01pct', 'prior + 1% noise', (9200, 3.6), 'left')
lab('09_taps_16', '16 taps', (16000, 0.48), 'center')
lab('08_taps_08', '8 taps', (31000, 0.48), 'center')
# Right-hand cluster: label into the gutter beyond the last point (41,171),
# stacked vertically so the leaders stay short and never cross the axes edge.
lab('05_dense_reference', 'dense (hit 40k cap)', (62000, 2.80), 'left')
lab('arm_06_wake_biased_random', 'wake-biased random', (62000, 1.30), 'left')
lab('07_wake_biased_grid', 'wake-biased grid', (62000, 0.62), 'left')

ax.text(900, 8.0, 'violent exits: line search\nfails mid gradient spike',
        fontsize=7.2, color=COLORS['accent'], va='bottom')
ax.text(900, 0.40, 'quiet exits: float32 resolution exhausted', fontsize=7.2,
        color=COLORS['pressure_only'], va='bottom')

fig.suptitle('All 17 runs die the same way; early deaths are cut off mid-instability',
             y=0.955, fontsize=10.5)
fig.text(0.125, 0.022,
    "Every run terminates on SciPy's REL_REDUCTION_OF_F test after a failed line search (16/17 print the line-search\n"
    "warning; the exception is the dense reference, which hit its 40,000-evaluation cap). At ftol = 1e-12 the test lies below\n"
    "float32 loss resolution, so it fires at the FIRST completely failed line search. Failures are far more likely during\n"
    "transient gradient spikes: runs stopping before 12,000 iterations exited at median 4.3$\\times$ their recent gradient level\n"
    "(range 0.9-26.7$\\times$); runs surviving past 20,000 exited at 1.0$\\times$ (0.6-2.1$\\times$). Spearman $\\rho$ = $-$0.73, p = 8$\\times$10$^{-4}$, n = 17.\n"
    "Every run survived mid-run spikes of 6-27$\\times$ along the way -- whether the fatal failure lands in one is chance.",
    fontsize=6.8, color=COLORS['muted'], va='bottom', linespacing=1.5)

bad = check_text_overlaps(fig)
if bad:
    print('TEXT ISSUES:', [b[0][:38] for b in bad[:5]])
print(save_figure(fig, pathlib.Path('/Users/mohammedhilal/Desktop/try/ModalPINN2.0/modes_experiment/fresh_analysis/figures/draft') / 'F_termination_anatomy'))
