"""Confirm the wake first-harmonic (v1) result is unchanged between the
original arm and the re-run, i.e. the training pathology affects effort,
not the physical conclusion."""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
ARMS = REPO / 'modes_experiment' / 'runs' / 'arms'
R = REPO / 'modes_experiment' / 'experiment' / 'notebooks' / 'matched_effort' / 'zx' / 'baseline_physics_only_K3_matched'

v1new = json.loads((R / 'v1.json').read_text())
v1old = json.loads((ARMS / '01_baseline_physics_only' / 'v1.json').read_text())


def show(tag, d):
    print(tag)
    m = d['regions']
    for reg in ('near-cylinder', 'near-wake', 'far-core', 'far-wake', 'whole-domain'):
        e = m.get(reg)
        if isinstance(e, dict):
            print(f"   {reg:14s} rel_L2={e.get('rel_L2', float('nan')):.4f}  amp_ratio={e.get('amp_ratio', float('nan')):.4f}")


show('NEW run v1:', v1new)
show('ORIGINAL arm 01 v1:', v1old)
