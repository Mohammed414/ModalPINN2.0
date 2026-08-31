"""The original arm 01 and the re-run's first L-BFGS call use identical code,
flags and seed. Show that they are bit-identical for the first ~99
iterations and then diverge through float32 GPU non-determinism, confirming
that L-BFGS termination is a random event rather than a deterministic
property of the configuration."""
import json
import pathlib
import re

import numpy as np

ARMS = pathlib.Path('/Users/mohammedhilal/Desktop/try/ModalPINN2.0/modes_experiment/runs/arms')
NEWLOG = pathlib.Path('/Users/mohammedhilal/Desktop/try/ModalPINN2.0/modes_experiment/notebooks/matched_effort/zx/baseline_physics_only_K3_matched/train_log.txt')


def fort(x):
    return float(x.replace('D', 'E').replace('d', 'e'))


def parse_its(text, call=0):
    seg = text.split('RUNNING THE L-BFGS-B CODE')[1:][call]
    return np.array([[int(a), fort(b), fort(c)] for a, b, c in
                      re.findall(r'At iterate\s*(\d+)\s+f=\s+(\S+)\s+\|proj g\|=\s+(\S+)', seg)])


A_old = parse_its((ARMS / '01_baseline_physics_only' / 'train_log.txt').read_text())
A_new = parse_its(NEWLOG.read_text())

print("same seed, same flags, same code -- f at matched iterations:")
print(f"{'iteration':>10s} {'arm 01 (orig)':>15s} {'new run':>15s} {'rel diff':>11s}")
for it in (1, 5, 10, 50, 100, 250, 500, 1000, 1500, 2000, 2400, 2516):
    fo = A_old[A_old[:, 0] == it, 1]
    fn = A_new[A_new[:, 0] == it, 1]
    if fo.size and fn.size:
        print(f"{it:10d} {fo[0]:15.6e} {fn[0]:15.6e} {abs(fo[0] - fn[0]) / fo[0]:11.2e}")

# Gradient elevation at the moment each call died, relative to its own
# preceding level -- distinguishes "quiet, plateaued" from "violent, mid-spike"
# terminations.
print("\ngradient blow-up at each run's ending "
      "(median |proj g| in final 50 its vs the preceding 550)")
print("  NOTE: this slices the baseline window by ITERATION NUMBER and prints"
      " 5.7x / 25.5x.\n        The published figures and findings.md use the"
      " census definition in 06\n        (array-index window), which gives"
      " 6.5x / 26.7x. Quote those, not these.\n        The choice does not"
      " affect the conclusion: rho -0.733 vs -0.721, n = 17.")
for name, A in (('arm 01 (original)', A_old), ('new run, cycle 1', A_new)):
    last = int(A[-1, 0])
    base = A[(A[:, 0] >= last - 600) & (A[:, 0] < last - 50), 2]
    fin = A[A[:, 0] >= last - 50, 2]
    b, f_ = np.median(base), np.median(fin)
    print(f"  {name:20s} last it {last:6,}  baseline {b:.3e}  final-50 {f_:.3e}  blow-up {f_ / b:.1f}x  "
          f"exit f {A[-1, 1]:.3e}")
