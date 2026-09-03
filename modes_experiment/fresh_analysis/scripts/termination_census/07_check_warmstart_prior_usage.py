"""Verification of a specific claim made mid-session: that --RestoreModel
warm-starting was 'already used by your other runs'. Checked directly
against every arm's run_record.json / training_loss_summary.json -- the
claim was false (0 of 16 arms warm-started) and was corrected."""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[4]
ARMS = REPO / 'modes_experiment' / 'runs' / 'arms'

allarms = sorted(p.name for p in ARMS.iterdir() if p.is_dir())
print(f"{'arm':32s} {'warm_started':>13s} {'restore_model':>14s}")
n_warm = 0
for a in allarms:
    p = ARMS / a / 'run_record.json'
    if not p.exists():
        continue
    rec = json.loads(p.read_text())
    lsum_p = ARMS / a / 'training_run' / 'training_loss_summary.json'
    lsum = json.loads(lsum_p.read_text()) if lsum_p.exists() else {}
    ws = rec.get('warm_started', lsum.get('warm_started'))
    rm = rec.get('restore_model', lsum.get('restore_model'))
    n_warm += bool(ws)
    print(f"{a:32s} {str(ws):>13s} {str(rm):>14s}")

print(f"\narms with warm_started true: {n_warm} of {len(allarms)}")
print("directories carrying a _WARM suffix:", [a for a in allarms if '_WARM' in a] or 'none')
