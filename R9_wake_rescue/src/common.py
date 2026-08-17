"""Shared loaders + the standard evaluation protocol for R9_wake_rescue.

LEGITIMACY BOUNDARY, enforced by convention in this module:
- `load_taps()` is the ONLY input any reconstruction method may use.
- `load_truth_fields()` / `load_truth_modes()` are EVALUATION/DIAGNOSTIC ONLY.
  Any import of them from a training/reconstruction script is a bug.

Geometry/conventions match src/pressure_only/evaluate_regions.py exactly.
"""
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R9 = os.path.join(ROOT, 'R9_wake_rescue')
CACHE = os.path.join(R9, 'cache')

X_C, Y_C, R_C = 0.0, 0.0, 0.5
LXMIN, LXMAX, LYMIN, LYMAX = -4.0, 8.0, -4.0, 4.0
D = 2 * R_C
RE = 100.0
OMEGA_0 = 1.036  # value used by the whole ModalPINN line of work

# --------------------------------------------------------------------------
# taps: the only legitimate observational input
# --------------------------------------------------------------------------

def load_taps():
    """Returns dict with x, y (32,), times (201,), pressure (201,32)."""
    d = np.load(os.path.join(ROOT, 'data', 'sensor_indices', 'taps_32.npz'))
    return {k: d[k] for k in ('node_indices', 'x', 'y', 'times', 'pressure')}


# --------------------------------------------------------------------------
# truth: EVALUATION / DIAGNOSTIC ONLY
# --------------------------------------------------------------------------

def load_truth_fields():
    """Scattered truth fields cropped to the training box.

    Returns (x, y (N,), times (201,), U, V, P (201, N) float32).
    Cached in cache/truth_box.npz; built from the enkf evaluation flow cache
    (read-only). Raises FileNotFoundError if that cache is absent — re-parse
    the raw text file with enkf_pressure_only/evaluation/_fast_flow_parser.py
    first (no automatic fallback is implemented here).
    """
    box_cache = os.path.join(CACHE, 'truth_box.npz')
    if os.path.exists(box_cache):
        d = np.load(box_cache)
        return d['x'], d['y'], d['times'], d['U'], d['V'], d['P']

    flow_cache = os.path.join(ROOT, 'enkf_pressure_only', 'evaluation',
                              '_flow_cache.npz')
    if os.path.exists(flow_cache):
        d = np.load(flow_cache)
        times, X, Y, U, V, P = (d['times'], d['X'], d['Y'],
                                d['U'], d['V'], d['p'])
    else:
        raise FileNotFoundError('flow cache missing; parse raw file first')

    in_box = (X < LXMAX) & (X > LXMIN) & (Y > LYMIN) & (Y < LYMAX)
    x, y = X[in_box], Y[in_box]
    U, V, P = U[:, in_box], V[:, in_box], P[:, in_box]
    np.savez(box_cache, x=x, y=y, times=times, U=U, V=V, P=P)
    return x, y, times, U, V, P


def fit_modes(times, F, omega0, nmodes=3):
    """Least-squares fit of one-sided Fourier modes at harmonics of omega0.

    F: (Nt, N). Returns list of (N,) arrays [c0(real), c1, ..., cn(complex)]
    with convention F(t) ~ c0 + sum_k 2*Re(ck * exp(i k w0 t)).
    """
    t = np.asarray(times, dtype=np.float64)
    cols = [np.ones_like(t)]
    for k in range(1, nmodes + 1):
        cols += [np.cos(k * omega0 * t), np.sin(k * omega0 * t)]
    A = np.stack(cols, axis=1)                      # (Nt, 1+2n)
    coef, *_ = np.linalg.lstsq(A, F.astype(np.float64), rcond=None)
    out = [coef[0]]
    for k in range(1, nmodes + 1):
        a, b = coef[2 * k - 1], coef[2 * k]
        out.append(0.5 * (a - 1j * b))              # ck: 2*Re(ck e^{ikwt}) = a cos + b sin
    return out


def load_truth_modes(nmodes=3):
    """Truth modal fields on the scattered in-box nodes (EVAL ONLY). Cached."""
    mode_cache = os.path.join(CACHE, 'truth_modes.npz')
    if os.path.exists(mode_cache):
        d = np.load(mode_cache)
        modes = {q: [d[f'{q}{k}'] for k in range(nmodes + 1)] for q in 'uvp'}
        return d['x'], d['y'], modes
    x, y, times, U, V, P = load_truth_fields()
    modes = {}
    for q, F in (('u', U), ('v', V), ('p', P)):
        modes[q] = fit_modes(times, F, OMEGA_0, nmodes)
    save = {'x': x, 'y': y}
    for q in 'uvp':
        for k in range(nmodes + 1):
            save[f'{q}{k}'] = modes[q][k]
    np.savez(os.path.join(CACHE, 'truth_modes.npz'), **save)
    return x, y, modes


# --------------------------------------------------------------------------
# the standard regional-error protocol (identical to evaluate_regions.py)
# --------------------------------------------------------------------------

def region_masks(x, y):
    r = np.sqrt((x - X_C) ** 2 + (y - Y_C) ** 2)
    near_cyl = r < 1.5 * R_C
    near_wake = (~near_cyl) & (x >= X_C) & (x < X_C + 3 * D)
    far_wake = (~near_cyl) & (~near_wake) & (x >= X_C + 3 * D)
    other = ~(near_cyl | near_wake | far_wake)
    return {
        'near-cylinder': near_cyl,
        'near-wake': near_wake,
        'far-wake': far_wake,
        'other (upstream/off-axis)': other,
        'whole domain': np.ones_like(near_cyl, dtype=bool),
    }


def rel_l2(pred, true, mask):
    if mask.sum() == 0:
        return float('nan')
    diff = pred[:, mask] - true[:, mask]
    return float(np.linalg.norm(diff) / np.linalg.norm(true[:, mask]))


def regional_table(pred_U, pred_V, pred_P, x, y, U, V, P):
    """Returns dict region -> (n_nodes, E_u, E_v, E_p)."""
    out = {}
    for name, m in region_masks(x, y).items():
        out[name] = (int(m.sum()),
                     rel_l2(pred_U, U, m), rel_l2(pred_V, V, m),
                     rel_l2(pred_P, P, m))
    return out


def print_regional_table(tbl, title=''):
    if title:
        print(title)
    header = f"{'Region':<26}{'n_nodes':>9}{'E_u':>10}{'E_v':>10}{'E_p':>10}"
    print(header)
    print('-' * len(header))
    for name, (n, eu, ev, ep) in tbl.items():
        print(f"{name:<26}{n:>9}{eu:>10.4f}{ev:>10.4f}{ep:>10.4f}")
