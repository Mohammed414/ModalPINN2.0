"""
STAGE D2 (audit repair): multi-direction ensemble construction.

Stage D built its 16 members as 16 *time jitters* of one and the same
solver trajectory.  Every member is therefore (to leading order) the same
field evaluated at a slightly different phase, so the state anomaly matrix
is dominated by a single direction -- d(state)/d(phase) -- and the EnKF can
only ever correct the flow along that one direction.  The singular value
spectrum of the anomaly matrix in Stage D confirms this (see
experiments/stage_d2_ensemble_check.py).

This module adds a second, genuinely multi-dimensional perturbation family
on top of the phase jitter:

    psi   = low-pass-filtered white noise on the CORNER (node) grid
    psi  *= W(x,y)                       (smooth window, see below)
    du    =  d psi / d y     at u-points
    dv    = -d psi / d x     at v-points

Two properties, both exact rather than approximate:

1.  DISCRETE DIVERGENCE-FREE BY CONSTRUCTION.  With psi at nodes
    (x_edges[i], y_edges[j]) and the MAC staggering of ns_solver.py,

        du[j,i] = ( psi[j+1,i] - psi[j,i] ) / dy      (u-point (x_e[i], y_c[j]))
        dv[j,i] = -( psi[j,i+1] - psi[j,i] ) / dx     (v-point (x_c[i], y_e[j]))

    the discrete divergence of a cell (j,i),
        (du[j,i+1]-du[j,i])/dx + (dv[j+1,i]-dv[j,i])/dy,
    telescopes to exactly zero in floating point up to the rounding of the
    four-term sum -- it is the same four psi values added and subtracted.
    No projection, no Poisson solve, no residual.

2.  VANISHES ON THE SOLID AND ON THE BOUNDARY.  Rather than zeroing du,dv
    there (which would destroy property 1), the window W is applied to psi
    itself.  Where W == 0 over a whole neighbourhood, psi is locally
    constant and its curl is identically zero, so du = dv = 0 there while
    the field stays a curl everywhere.  W is a product of a radial ramp
    around the cylinder and a ramp away from each domain edge, each
    smoothed by the same Gaussian used for the low-pass, so the transition
    is smooth on the perturbation's own length scale.

The resulting perturbation lives entirely on the EnKF's active DOF set
(estimator/state_vector.py), which is what makes the analysis states remain
valid: every forecast member is div-free, so every linear combination of
forecast anomalies is too.

No truth data is read anywhere in this module.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def _gaussian_lowpass(field, sigma_cells):
    """Periodic-free (reflect) Gaussian low-pass. Wrapper kept separate so
    the filter choice is documented in one place."""
    return gaussian_filter(field, sigma=sigma_cells, mode='nearest')


def taper_window(solver, wall_margin=0.30, edge_margin=0.60, smooth_cells=2.0):
    """Smooth window W on the NODE grid: 0 inside the cylinder (plus
    ``wall_margin``), 0 within ``edge_margin`` of the domain boundary,
    1 elsewhere, with the step smoothed over ``smooth_cells`` cells.

    Returns W with shape (Ny+1, Nx+1) matching the node grid.
    """
    Xn, Yn = np.meshgrid(solver.x_edges, solver.y_edges)
    r = np.hypot(Xn - solver.x_c, Yn - solver.y_c)

    W = np.ones_like(Xn)
    W[r < solver.r_c + wall_margin] = 0.0
    W[(Xn - solver.Lxmin) < edge_margin] = 0.0
    W[(solver.Lxmax - Xn) < edge_margin] = 0.0
    W[(Yn - solver.Lymin) < edge_margin] = 0.0
    W[(solver.Lymax - Yn) < edge_margin] = 0.0

    W = _gaussian_lowpass(W, smooth_cells)
    # re-impose the hard zeros: smoothing bleeds a little back inside
    W[r < solver.r_c + 0.5 * wall_margin] = 0.0
    W[(Xn - solver.Lxmin) < 0.5 * edge_margin] = 0.0
    W[(solver.Lxmax - Xn) < 0.5 * edge_margin] = 0.0
    W[(Yn - solver.Lymin) < 0.5 * edge_margin] = 0.0
    W[(solver.Lymax - Yn) < 0.5 * edge_margin] = 0.0
    return W


def streamfunction_perturbation(solver, rng, length_scale=0.8, amplitude=1.0,
                                window=None, wake_bias=None):
    """One divergence-free (du, dv) perturbation drawn from a random
    low-pass-filtered streamfunction.

    length_scale : correlation length of psi in flow units (D = 1).
    amplitude    : RMS of the resulting velocity perturbation over the
                   region the perturbation actually occupies -- cells where
                   the total envelope (taper window x wake bias) exceeds
                   half its maximum.  Normalising over the whole window
                   instead would make ``amplitude`` meaningless whenever
                   ``wake_bias`` concentrates the energy: the RMS would be
                   diluted by the large quiet region and the actual peak
                   velocity perturbation would be ~10x ``amplitude``.
    wake_bias    : optional (x0, sigma_x, sigma_y); multiplies psi by a
                   Gaussian centred at x0 downstream so the perturbation
                   energy is concentrated in the wake, where the
                   uncertainty actually is. None = uniform.

    Returns (du, dv) with shapes matching solver.u and solver.v.
    """
    Ny, Nx = solver.Ny, solver.Nx
    sigma_cells = length_scale / min(solver.dx, solver.dy)

    psi = rng.standard_normal((Ny + 1, Nx + 1))
    psi = _gaussian_lowpass(psi, sigma_cells)

    envelope = taper_window(solver) if window is None else window
    if wake_bias is not None:
        x0, sx, sy = wake_bias
        Xn, Yn = np.meshgrid(solver.x_edges, solver.y_edges)
        envelope = envelope * np.exp(
            -0.5 * (((Xn - x0) / sx) ** 2 + ((Yn - solver.y_c) / sy) ** 2))

    psi = psi * envelope

    du = np.diff(psi, axis=0) / solver.dy          # (Ny, Nx+1)  at u-points
    dv = -np.diff(psi, axis=1) / solver.dx         # (Ny+1, Nx)  at v-points

    core = envelope[:-1, :-1] > 0.5 * envelope.max()
    if not np.any(core):
        raise RuntimeError('envelope left no active region')
    rms = np.sqrt(np.mean(du[:, :-1][core] ** 2 + dv[:-1, :][core] ** 2))
    if rms <= 0:
        raise RuntimeError('degenerate perturbation (zero RMS)')
    scale = amplitude / rms
    return du * scale, dv * scale


def discrete_divergence(solver, u=None, v=None):
    """Cell-centred discrete divergence, shape (Ny, Nx)."""
    u = solver.u if u is None else u
    v = solver.v if v is None else v
    return ((u[:, 1:] - u[:, :-1]) / solver.dx
            + (v[1:, :] - v[:-1, :]) / solver.dy)


def max_div_interior(solver, u=None, v=None, halo=3):
    """Stage-B convention: interior only, excluding the boundary halo (the
    outflow column carries an O(0.1) divergence by construction of the
    zero-gradient outflow BC applied after projection)."""
    div = discrete_divergence(solver, u, v)
    return float(np.abs(div[halo:-halo, halo:-halo]).max())


def anomaly_spectrum(X):
    """Singular values of the state anomaly matrix X (n_state, q) after
    removing the ensemble mean, plus the participation-ratio effective
    number of directions

        n_eff = (sum s_i^2)^2 / sum s_i^4

    which is 1 for a rank-1 (single-direction) ensemble and q-1 for an
    isotropic one.
    """
    A = X - X.mean(axis=1, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    e = s ** 2
    tot = e.sum()
    n_eff = float(tot ** 2 / np.sum(e ** 2)) if tot > 0 else 0.0
    return dict(singular_values=s, energy_fraction=e / tot if tot > 0 else e,
                n_eff=n_eff)
