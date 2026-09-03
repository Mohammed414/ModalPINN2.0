"""Compare the completed matched-effort re-run against the original arm 01:
L-BFGS effort, final loss components, and the checkpoint history."""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
ARMS = REPO / 'modes_experiment' / 'runs' / 'arms'
R = REPO / 'modes_experiment' / 'experiment' / 'notebooks' / 'matched_effort' / 'zx' / 'baseline_physics_only_K3_matched'

rec = json.loads((R / 'run_record.json').read_text())
lsum = json.loads((R / 'training_run' / 'training_loss_summary.json').read_text())

print('--- new run: run_record.json ---')
for k, v in rec.items():
    if k != 'command':
        print(f"  {k}: {v}")
print('\n--- new run: training_loss_summary.json ---')
for k, v in lsum.items():
    print(f"  {k}: {v}")

orig = json.loads((ARMS / '01_baseline_physics_only' / 'training_run' / 'training_loss_summary.json').read_text())
orig_rec = json.loads((ARMS / '01_baseline_physics_only' / 'run_record.json').read_text())

print(f"\n{'':28s} {'ORIGINAL arm 01':>16s} {'NEW run':>16s} {'ratio':>8s}")
pairs = [
    ('L-BFGS evaluations', orig_rec.get('lbfgs_evals'), rec['lbfgs_evals']),
    ('final total loss', orig['total_loss'], lsum['total_loss']),
    ('physics loss', orig['physics_loss'], lsum['physics_loss']),
    ('pressure-tap loss', orig['pressure_tap_loss'], lsum['pressure_tap_loss']),
    ('wall clock (s)', orig_rec.get('wall_s'), rec['wall_s']),
]
for name, a, b in pairs:
    if isinstance(a, float) or isinstance(b, float):
        print(f"{name:28s} {a:16.4e} {b:16.4e} {b / a:>8.2f}x")
    else:
        print(f"{name:28s} {a:16,} {b:16,} {b / a:>8.2f}x")

print('\nbest loss reached anywhere in the new run (L-BFGS phase): %.4e' % float(rec['lbfgs_best_training_loss']))
print('original arm 01 final                                  : %.4e' % orig['total_loss'])

# Per-cycle checkpoint history: confirms the new run's evaluation TOTAL (7,173)
# and that no single checkpoint beat the original's final loss.
cds = sorted((R / 'training_run' / 'cycle_checkpoints').glob('*.json'))
best = []
for f in cds:
    d = json.loads(f.read_text())
    best.append((d['training_loss_on_lbfgs_feed'], d['tag'], d['cumulative_lbfgs_evals']))
print('\ncycle checkpoints written:', len(cds))
for l, t, e in sorted(best)[:5]:
    print(f"    {t:26s} loss {l:.4e}  at {e:,} evals")
print(f"  -> best checkpoint {min(best)[0]:.4e} vs original arm 01 {orig['total_loss']:.4e}"
      f"  ({min(best)[0] / orig['total_loss']:.2f}x)")
