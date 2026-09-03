"""Build decay_profiles.json: streamwise |v_1| profiles and centreline phase.

Reads dns_raw.npz (from parse_dns.py) and each arm's trained weights, and writes
the per-bin amplitudes plus the centreline phase that fig7_decay.py plots.

Also prints the validation table: the reimplemented forward pass against the
amplitude ratios stored in each arm's v1.json, on the same far-core mask.

    python parse_dns.py && python build_decay_data.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import modalpinn_eval as me

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(HERE, os.pardir, os.pardir, "4_runs")
OUT = os.path.join(HERE, "decay_profiles.json")

# Arms shown in figure 7: the collapsed baseline, the amplitude-without-phase
# arm, the prior arm, and the information-rich probe reference.
SHOW = {1: "pressure only", 4: "velocity probes",
        7: "wake-biased grid", 15: "pressure + prior"}

# Additional arms validated but not plotted in figure 7: the validation table in
# the report covers these too, so they must be computed here rather than quoted.
VALIDATE_ONLY = {13: "pressure + prior, 10% noise"}

FOLD = {int(m.group(1)): d
        for d in sorted(os.listdir(ARMS))
        if os.path.isdir(os.path.join(ARMS, d))
        for m in [re.match(r"(?:arm_)?(\d+)_", d)] if m}


def paths(n):
    d = os.path.join(ARMS, FOLD[n])
    return (os.path.join(d, "training_run", "DNN2_100_100_4_tanh.pickle"),
            os.path.join(d, "run_record.json"),
            os.path.join(d, "street_prior_used.npz"),
            os.path.join(d, "v1.json"))


Z = np.load(os.path.join(HERE, "dns_raw.npz"))
xs, ys, times, V = Z["xs"], Z["ys"], Z["times"], Z["V"]
w0 = me.OMEGA_0

# Restrict to the reconstruction domain and take the k=1 temporal mode.
dom = (xs >= -4) & (xs <= 8) & (np.abs(ys) <= 4)
xd, yd = xs[dom], ys[dom]
v1_dns = 2.0 * (V[:, dom] * np.exp(-1j * w0 * times)[:, None]).mean(axis=0)

# Streamwise bins across the wake band used by the project's far-core metric.
edges = np.arange(-1.0, 8.01, 0.5)
xc = 0.5 * (edges[:-1] + edges[1:])
band = np.abs(yd) <= 2.0
idx = [np.where(band & (xd >= a) & (xd < b))[0]
       for a, b in zip(edges[:-1], edges[1:])]
assert min(len(i) for i in idx) > 100, "a streamwise bin is nearly empty"

# Centreline strip for the phase panel.
line = (xd >= 1.0) & (xd <= 8.0) & (np.abs(yd) <= 0.75)
o = np.argsort(xd[line])
xline = xd[line][o]


def wavelength(q, lo=2.0, hi=8.0):
    """Streamwise wavelength from d(phase)/dx along the centreline."""
    sel = (xd >= lo) & (xd <= hi) & (np.abs(yd) <= 0.75)
    oo = np.argsort(xd[sel])
    k = np.polyfit(xd[sel][oo], np.unwrap(np.angle(q[sel][oo])), 1)[0]
    return 2.0 * np.pi / abs(k)


def mode_v1(n):
    """v_1 on the DNS nodes for arm n, evaluated as that arm was trained."""
    pk, rr, spp, _ = paths(n)
    w = me.load_weights(pk)
    fl = me.assert_flags_supported(rr)
    sp = me.load_street_params(spp) if fl["prior"] else None
    return me.modes(xd, yd, w[2], w[3], 0.0 if fl["freestream"] else None,
                    street_params=sp, is_v=True)[:, 1]


out = dict(xc=xc.tolist(), edges=edges.tolist(),
           dns=[float(np.abs(v1_dns[i]).mean()) for i in idx],
           dns_lambda=float(wavelength(v1_dns)),
           dns_xline=xline.tolist(),
           dns_phase=np.unwrap(np.angle(v1_dns[line][o])).tolist(),
           arms={})

# Validation: reproduce each arm's stored far-core amplitude ratio.
fc = (xd >= 3.0) & (xd <= 8.0) & (np.abs(yd) <= 2.0)
den = np.abs(v1_dns[fc]).mean()
print(f"{'arm':>4} {'stored':>9} {'reimpl':>9} {'rel diff':>9}  {'lambda/D':>9}")
for n, label in list(SHOW.items()) + list(VALIDATE_ONLY.items()):
    q = mode_v1(n)
    stored = json.load(open(paths(n)[3]))["regions"]["far-core"]["amp_ratio"]
    mine = float(np.abs(q[fc]).mean() / den)
    lam = float(wavelength(q))
    assert abs(mine - stored) / stored < 0.15, (
        f"arm {n}: reimplementation {mine:.4f} disagrees with stored "
        f"{stored:.4f} by more than 15%")
    print(f"{n:>4} {stored:>9.4f} {mine:>9.4f} "
          f"{100 * abs(mine - stored) / stored:>8.2f}%  {lam:>9.2f}")
    if n not in SHOW:
        continue
    out["arms"][str(n)] = dict(
        label=label, lam=lam,
        amp=[float(np.abs(q[i]).mean()) for i in idx],
        xline=xline.tolist(),
        phase=np.unwrap(np.angle(q[line][o])).tolist())

json.dump(out, open(OUT, "w"))
print(f"\nwrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
print(f"DNS streamwise wavelength: {out['dns_lambda']:.2f} D")
