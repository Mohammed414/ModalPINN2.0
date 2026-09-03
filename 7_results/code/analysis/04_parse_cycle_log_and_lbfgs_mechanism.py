"""Parse the new run's train_log.txt (per-cycle progress, Adam-kick escalation)
and confirm the L-BFGS termination mechanism: the last two accepted iterations
of a dying call are identical in both loss and gradient to the printed
precision, and every call ends with the same scipy line-search warning."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
R = REPO / '3_notebooks' / 'matched_effort' / 'zx' / 'baseline_physics_only_K3_matched'


def fort(x):
    return float(x.replace('D', 'E').replace('d', 'e'))


out = (R / 'training_run' / 'out.txt').read_text()
cyc = re.findall(r'CYCLE (\d+): L-BFGS gained (\d+) evals \(total (\d+)/\d+\), loss (\S+) -> (\S+)', out)
print(f"{'cycle':>5s} {'gained':>7s} {'total':>7s} {'loss after':>12s}")
for c, g, t, l0, l1 in cyc[:3] + [('...', '...', '...', '...', '...')] + cyc[-4:]:
    print(f"{c:>5s} {g:>7s} {t:>7s} {l1:>12s}")
gains = [int(g) for _, g, _, _, _ in cyc]
print(f"\ncycle 1 gained {gains[0]:,}; cycles 2-{len(gains)} averaged {sum(gains[1:]) / len(gains[1:]):.0f} evals each")

esc = re.findall(r'escalating Adam kick to (\d+) iterations', out)
print('kick escalations triggered:', len(esc), '| sizes used:', sorted(set(esc), key=int))
adam_pts = re.findall(r'Post Adam it (\d+) - Loss value :\s+(\S+)', out.split('Start Adam training')[-1])
if adam_pts:
    print(f"final Adam phase: entry {float(adam_pts[0][1]):.4e} -> final {float(adam_pts[-1][1]):.4e}")

# --- The termination mechanism itself, from the raw Fortran iterate log ---
log = (R / 'train_log.txt').read_text()
calls = log.split('RUNNING THE L-BFGS-B CODE')[1:]
print('\nscipy calls in this run:', len(calls))

c1 = calls[0]
its = [(int(a), fort(b), fort(c)) for a, b, c in
       re.findall(r'At iterate\s*(\d+)\s+f=\s+(\S+)\s+\|proj g\|=\s+(\S+)', c1)]
print('\nfinal 10 accepted iterations of cycle 1 (the first, longest L-BFGS call):')
print(f"{'it':>6s} {'f':>13s} {'|proj g|':>11s} {'df':>12s} {'df/f':>11s}")
for (i0, f0, g0), (i1, f1, g1) in zip(its[-11:-1], its[-10:]):
    print(f"{i1:6d} {f1:13.6e} {g1:11.4e} {f0 - f1:12.4e} {(f0 - f1) / f1:11.3e}")

n_warn = log.count('more than 10 function')
print(f"\n'more than 10 function and gradient evaluations' line-search warning count: {n_warn}"
      f" (one per scipy call = {len(calls)})")
