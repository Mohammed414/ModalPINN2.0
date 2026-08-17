"""
Fast parser for the raw CFD flow file (../data/fixed_cylinder_atRe100).

Canonical shared copy (previously duplicated separately in extras/,
enkf_pressure_only/evaluation/, and effectively depended on by
src/pressure_only/audit_r7_*.py and R9_wake_rescue/src/common.py via that
enkf copy - consolidated here when enkf_pressure_only/ was removed, so nothing
outside that folder depends on it anymore).

text_flow.py's read_flow parses one Python float() at a time, which is far
too slow at this file's scale (Nt=201, N_nodes=82872, ~16.6M data lines) -
this reads each per-timestep block with a single
np.loadtxt(f, max_rows=N_nodes) call instead (~15-20s total vs. many minutes).
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
