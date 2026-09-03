"""Decisive test: is the baseline's early stopping caused by 32 taps, or is
it a project-wide optimizer pathology? Parses the L-BFGS iterate log for
every arm with one recorded (16 historical arms + the new re-run), then:

  1. checks whether tap count predicts run length (it does not -- two
     32-tap arms differing from the baseline only in collocation placement
     ran 7-8x longer);
  2. measures gradient elevation at the moment of death for all 17 runs and
     correlates it with stopping iteration (Spearman).
"""
import json
import pathlib
import re

import numpy as np
from scipy.stats import spearmanr

REPO = pathlib.Path(__file__).resolve().parents[3]
ARMS = REPO / '4_runs'
NEWLOG = REPO / '3_notebooks' / 'matched_effort' / 'zx' / 'baseline_physics_only_K3_matched' / 'train_log.txt'


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
    rec = json.loads(rp.read_text())
    F = parse_flags([str(x) for x in rec.get('command', [])])
    coll = ('wb-grid' if '--WakeBiasedGridSampling' in F else
            'wb-rand' if '--WakeBiasedSampling' in F else 'uniform')
    lset = 'prior' if '--StreetPrior' in F else 'bvf' if '--BVF' in F else 'plain'
    p_only = '--PressureOnly' in F
    runs[a] = dict(A=parse_its(t), ntaps=F.get('--NTaps', '-'), coll=coll, lset=lset, p_only=p_only,
                   nwarn=t.count('more than 10 function'))
runs['NEW re-run (cycle 1)'] = dict(A=parse_its(NEWLOG.read_text(), 0), ntaps='32', coll='uniform',
                                    lset='plain', p_only=True,
                                    nwarn=NEWLOG.read_text().count('more than 10 function'))


def death_metrics(A):
    g = A[:, 2]
    last = len(A) - 1
    base = np.median(g[max(0, last - 600):last - 100]) if last > 150 else np.median(g[:max(1, last - 10)])
    exit_blow = np.median(g[last - 50:last + 1]) / base if last > 60 else g[last] / base
    return dict(last_it=int(A[-1, 0]), f_exit=A[-1, 1], exit_blow=float(exit_blow))


rows = []
for a, r in runs.items():
    m = death_metrics(r['A'])
    rows.append((m['last_it'], a, r['ntaps'], r['coll'], r['lset'], m['f_exit'], m['exit_blow'], r['nwarn'], r['p_only']))
rows.sort(reverse=True)

print(f"{'stop it':>8s} {'arm':30s} {'taps':>4s} {'colloc':>8s} {'loss set':>8s} "
      f"{'exit f':>10s} {'exit blow':>10s} {'ls-warn':>8s}")
for it, a, nt, coll, ls, fx, eb, nw, po in rows:
    print(f"{it:8,} {a:30s} {nt:>4s} {coll:>8s} {ls:>8s} {fx:10.2e} {eb:9.1f}x {nw:>8d}")

# --- 1. Tap-count hypothesis: does 32 taps predict early stopping? ---
print("\n--- clean tap-count comparison (pressure-only, no extra loss terms) ---")
clean = [(it, a, nt) for it, a, nt, coll, ls, fx, eb, nw, po in rows
         if ls == 'plain' and coll == 'uniform' and po and a not in ('05_dense_reference',)]
for it, a, nt in sorted(clean, reverse=True):
    print(f"  {it:7,}  taps={nt:>3s}  {a}")

print("\n--- decisive counter-example: two 32-tap arms differing from the")
print("    baseline ONLY in collocation placement ---")
for a in ('06_wake_biased_random', '07_wake_biased_grid'):
    base_cmd = [str(x) for x in json.loads((ARMS / '01_baseline_physics_only' / 'run_record.json').read_text())['command']]
    F_base = parse_flags(base_cmd)
    F = parse_flags([str(x) for x in json.loads((ARMS / a / 'run_record.json').read_text())['command']])
    diff = {k: (F_base.get(k), F.get(k)) for k in set(F_base) | set(F) if F_base.get(k) != F.get(k)}
    it = [r[0] for r in rows if r[1] == a][0]
    print(f"  {a:30s} stopped at {it:,} iterations; differs from baseline only in: {diff}")

# --- 2. Gradient blow-up vs stopping iteration, across all 17 runs ---
stop = np.array([r[0] for r in rows], float)
blow = np.array([r[6] for r in rows], float)
rho, p = spearmanr(stop, blow)
print(f"\nSpearman(stop iteration, exit gradient blow-up): rho = {rho:.3f}, p = {p:.2e}  (n = {len(rows)})")
short = blow[stop < 12000]
long_ = blow[stop > 20000]
print(f"exit blow-up, runs stopping <12,000 iterations: median {np.median(short):.1f}x "
      f"(range {short.min():.1f}-{short.max():.1f}x, n={short.size})")
print(f"exit blow-up, runs stopping >20,000 iterations: median {np.median(long_):.1f}x "
      f"(range {long_.min():.1f}-{long_.max():.1f}x, n={long_.size})")
print(f"line-search warning present at termination: {sum(1 for r in rows if r[7] >= 1)}/{len(rows)}")
