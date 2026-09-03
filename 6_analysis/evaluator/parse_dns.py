"""Parse the Boudina Re=100 cylinder DNS text file into a compact .npz.

Source format (Zenodo 5039610, src/text_flow.py):
    Re Ur / blank / Nt N_nodes / blank / t0 / <N_nodes lines: x y u v p> / t1 / ...

No interpolation and no surrogate: the node coordinates and values are taken
as given. Output holds U, V, P as float32 (201 x 82872 x 3 x 4 B = 200 MB).
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
PATH = os.path.join(REPO, "data", "fixed_cylinder_atRe100")
OUT = os.path.join(HERE, "dns_raw.npz")

with open(PATH) as f:
    Re, Ur = [float(v) for v in f.readline().split()]
    f.readline()
    Nt, Nn = [int(v) for v in f.readline().split()]
assert (Nt, Nn) == (201, 82872), (Nt, Nn)

U = np.empty((Nt, Nn), np.float32)
V = np.empty((Nt, Nn), np.float32)
P = np.empty((Nt, Nn), np.float32)
times = np.empty(Nt, np.float64)
X0 = np.empty(Nn, np.float64)
Y0 = np.empty(Nn, np.float64)

reader = pd.read_csv(PATH, sep=r"\s+", header=None, names=list(range(5)),
                     skiprows=4, skip_blank_lines=True, engine="c",
                     dtype=np.float64, chunksize=4_000_000)
snap = -1
carry = None
for chunk in reader:
    a = chunk.to_numpy()
    if carry is not None:
        a = np.vstack([carry, a])
        carry = None
    ti = np.flatnonzero(np.isnan(a[:, 4]))     # time rows have NaN in col 4
    for j, idx in enumerate(ti):
        end = ti[j + 1] if j + 1 < len(ti) else len(a)
        block = a[idx + 1:end]
        if len(block) < Nn:                    # snapshot split across chunks
            carry = a[idx:]
            break
        assert len(block) == Nn, len(block)
        snap += 1
        times[snap] = a[idx, 0]
        if snap == 0:
            X0[:] = block[:, 0]
            Y0[:] = block[:, 1]
        else:
            assert np.array_equal(block[:, 0], X0), "node order changed"
        U[snap] = block[:, 2]
        V[snap] = block[:, 3]
        P[snap] = block[:, 4]

assert snap == Nt - 1, f"got {snap + 1} snapshots, expected {Nt}"
np.savez_compressed(OUT, xs=X0, ys=Y0, times=times, U=U, V=V, P=P,
                    Re=Re, Ur=Ur)
print(f"{OUT}  {os.path.getsize(OUT) / 1e6:.0f} MB")
print(f"snapshots {Nt}  nodes {Nn}  t in [{times[0]:.2f}, {times[-1]:.2f}]")
print(f"x in [{X0.min():.2f}, {X0.max():.2f}]  y in [{Y0.min():.2f}, {Y0.max():.2f}]")
