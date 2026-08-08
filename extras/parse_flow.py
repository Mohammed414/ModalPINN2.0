"""
Fast parser for the raw CFD flow file (data/fixed_cylinder_atRe100).

The file format (see src/text_flow.py for the reference/slow implementation):
    Re Ur
    (blank)
    Nt N_nodes
    (blank)
    t0
    x y U V p          <- repeated N_nodes times
    t1
    x y U V p          <- repeated N_nodes times
    ...

src/text_flow.py's read_flow() parses this one Python float() at a time,
which is fine for the small sparse-sensor .npz caches ModalPINN trains on,
but is far too slow for the full field (Nt=201, N_nodes=82872 -> ~16.6M
data lines). This module reads each per-timestep block with a single
np.loadtxt(f, max_rows=N_nodes) call instead, and caches the parsed arrays
to a .npz so repeated runs (e.g. while tuning the animation) don't have to
re-parse the 1.1GB text file every time.
"""
import os
import numpy as np


def load_flow(infile, cache=None, force_reparse=False):
    """Returns Re, Ur, times (Nt,), X, Y (N_nodes,), U, V, p (Nt, N_nodes)."""
    if cache and os.path.exists(cache) and not force_reparse:
        print('Loading cached flow arrays from %s' % cache)
        d = np.load(cache)
        return (d['Re'], d['Ur'], d['times'], d['X'], d['Y'],
                d['U'], d['V'], d['p'])

    print('Parsing raw flow file %s (this only needs to happen once; '
          'result is cached)...' % infile)
    with open(infile, 'r') as f:
        Re, Ur = (float(x) for x in f.readline().split())
        f.readline()  # blank
        Nt, N_nodes = (int(x) for x in f.readline().split())
        f.readline()  # blank

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
            if (n + 1) % 20 == 0 or n == Nt - 1:
                print('  parsed timestep %d/%d (t=%.3f)' % (n + 1, Nt, times[n]))

    print('Done parsing. Re=%.0f, Ur=%.1f, Nt=%d, N_nodes=%d' % (Re, Ur, Nt, N_nodes))

    if cache:
        print('Caching parsed arrays to %s' % cache)
        np.savez_compressed(cache, Re=Re, Ur=Ur, times=times, X=X, Y=Y, U=U, V=V, p=p)

    return Re, Ur, times, X, Y, U, V, p
