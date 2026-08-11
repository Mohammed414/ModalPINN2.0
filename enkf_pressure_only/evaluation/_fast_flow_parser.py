"""
Self-contained fast parser for the raw CFD flow file (../../data/fixed_cylinder_atRe100).

Deliberately duplicated (not imported) from extras/parse_flow.py: this folder
is kept fully isolated from every other part of the repo, per the same
"separate folder, don't touch the rest" policy used for R6. The format is
documented in src/text_flow.py; that reference reader parses one Python
float() at a time, which is far too slow at this file's scale
(Nt=201, N_nodes=82872, ~16.6M data lines) -- this reads each per-timestep
block with a single np.loadtxt(f, max_rows=N_nodes) call instead.

This module belongs to the EVALUATION side only. It is never imported by
estimator/ code (see docs/anti_leakage.py for the enforced guard).
"""
import os
import numpy as np


def load_flow(infile, cache=None, force_reparse=False):
    """Returns Re, Ur, times (Nt,), X, Y (N_nodes,), U, V, p (Nt, N_nodes)."""
    if cache and os.path.exists(cache) and not force_reparse:
        d = np.load(cache)
        return (d['Re'], d['Ur'], d['times'], d['X'], d['Y'],
                d['U'], d['V'], d['p'])

    with open(infile, 'r') as f:
        Re, Ur = (float(x) for x in f.readline().split())
        f.readline()
        Nt, N_nodes = (int(x) for x in f.readline().split())
        f.readline()

        times = np.empty(Nt, dtype=np.float64)
        X = Y = None
        U = np.empty((Nt, N_nodes), dtype=np.float32)
        V = np.empty((Nt, N_nodes), dtype=np.float32)
        p = np.empty((Nt, N_nodes), dtype=np.float32)

        for n in range(Nt):
            times[n] = float(f.readline())
            block = np.loadtxt(f, max_rows=N_nodes, dtype=np.float32)
            if X is None:
                X = block[:, 0].copy()
                Y = block[:, 1].copy()
            U[n] = block[:, 2]
            V[n] = block[:, 3]
            p[n] = block[:, 4]

    if cache:
        np.savez_compressed(cache, Re=Re, Ur=Ur, times=times, X=X, Y=Y, U=U, V=V, p=p)

    return Re, Ur, times, X, Y, U, V, p
