"""
metrics_v2 -- evaluation metrics that cannot be improved by deleting the wake.

WHY THIS MODULE EXISTS
----------------------
Stage F's metric E_v(t) (relative L2 error of the cross-stream velocity
against the instantaneous truth) is unusable as a reconstruction-skill
score for this problem, for two independent reasons established by the
Phase-1 audit:

  (a) It is dominated by PHASE. Corr(E_v(t), |phase error|(t)) = 0.92.
      Because the forward solver sheds ~13% fast, every run sweeps through
      phase alignment and back out during a 20-time-unit window, so the
      ranking of free-run / EnKF / shuffled reverses depending on where the
      window is cut.

  (b) Worse, it is MINIMISED BY REMOVING THE OSCILLATION. For a signal at
      the wrong phase, ||a - b|| with b a phase-shifted copy of a is larger
      than ||mean(a) - b||: at a quarter-period offset, E_v = 0.906 with the
      wake deleted versus 1.299 with the wake reproduced at full amplitude
      (numbers reproduced by validate_metrics_v2.py). A reconstruction that
      DELETES the vortex street outscores one that reproduces it with the
      wrong timing.

That is exactly the ModalPINN failure mode (oscillating modes k>=1 collapsing
to ~0 a couple of diameters downstream) appearing inside the evaluation
metric itself. Any metric used to judge whether the wake was recovered must
therefore separate three things that E_v confounds:

    STRUCTURE  -- is the spatial shape of the oscillation right?
    AMPLITUDE  -- is the oscillation the right SIZE (this is the one that
                  a wake-deleting reconstruction gets catastrophically
                  wrong, and the one E_v rewards getting wrong)?
    TIMING     -- is it at the right phase / frequency?

The two metric families here do that:

  1. phase_aligned_field_error()  -- relative L2 field error after an
     optimal sub-sample TIME SHIFT of the estimate. Reports the aligned
     error AND the fitted shift tau*, so "how wrong is the structure" and
     "how wrong is the timing" are two numbers, not one. Phase alignment
     removes reason (a); it does NOT by itself remove reason (b) --- a
     deleted wake still scores ~0.91 --- but with the timing degree of
     freedom taken out, a correct-amplitude reconstruction scores ~0 and
     therefore wins outright, which it did not before.

  2. modal_metrics()  -- the PRIMARY metric. Fits the same harmonic
     decomposition the truth used (harmonic least squares at k*omega_0,
     k = 0,1,2) to the estimate's velocity history and compares complex
     mode shapes q_hat_k(x,y) against reference_truth_modal.npz. The
     headline deliverable is the mode-1 amplitude profile versus DOWNSTREAM
     DISTANCE x --- the exact curve ModalPINN gets wrong. The amplitude
     comparison |q_hat_e| vs |q_hat_t| is phase-blind BY CONSTRUCTION, so a
     wrong-phase reconstruction is not penalised for its phase here at all,
     while a deleted wake scores the worst value the metric can take
     (relative error 1.0, i.e. 100% of the truth's modal energy missing).
     Phase is reported separately, per mode and per x.

GAUGE / SIGN CONVENTION
-----------------------
Matches build_reference_truth.py exactly (verified numerically: refitting
the raw scattered truth with harmonic_modes() below reproduces Mtrue_v1 to a
relative error of 2.6e-8):

    f(t) = f0 + Re[f1 exp(i*omega_0*t)] + Re[f2 exp(i*2*omega_0*t)]

with times in the truth's ABSOLUTE clock (t = 400.0 ... 420.0). Runs store
both `exp_times` (0..20) and `tap_times_true` (400..420); the absolute clock
must be used, otherwise every phase is offset by exp(i*k*omega_0*400).

TRUTH ACCESS
------------
This module lives in evaluation/ and is the only consumer of the withheld
truth besides stage_f_evaluate.py and build_reference_truth.py. All truth
reads go through `with allow_truth_access():`. The leakage guard is never
weakened or bypassed.

USAGE
-----
    from metrics_v2 import load_truth, evaluate_run
    truth = load_truth()
    res = evaluate_run(truth, u_hist, v_hist, times_abs, config, label='enkf')

All metric functions take run arrays as ARGUMENTS (no file paths baked in),
so they apply unchanged to the repaired EnKF runs produced later.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator, RegularGridInterpolator
from scipy.optimize import minimize_scalar

from estimator._leakage_guard import allow_truth_access

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')
EXP_DIR = os.path.join(HERE, '..', 'experiments')
FIG_DIR = os.path.join(HERE, '..', 'figures')

OMEGA_0 = 1.036
R_C = 0.5

# Fluid points closer than this to the cylinder centre are excluded from every
# field/modal norm. 0.65 = r_c + 1.5*dx at the solver's dx = 0.1, i.e. the
# immersed-boundary stair-step band where the solver's representation is
# known-bad by construction (same exclusion Stage F used, kept for
# comparability). Not a hiding place: the wake metrics of interest live at
# x >= 1.
R_EXCLUDE = 0.65


# --------------------------------------------------------------------------
# harmonic decomposition (identical convention to build_reference_truth.py)
# --------------------------------------------------------------------------
def harmonic_modes(times, F, omega, K=2):
    """Harmonic least squares of F(t) at k*omega, k = 0..K.

    times : (Nt,) absolute times.
    F     : (Nt, ...) real field history; NaNs must already be removed
            (use np.nan_to_num) -- lstsq cannot handle them.

    Returns [f0 (real), f1 (complex), ..., fK (complex)] with
        F(t) ~= f0 + sum_{k>=1} Re[f_k * exp(i k omega t)].
    """
    times = np.asarray(times, float)
    cols = [np.ones_like(times)]
    for k in range(1, K + 1):
        cols += [np.cos(k * omega * times), np.sin(k * omega * times)]
    Phi = np.stack(cols, axis=1)                       # (Nt, 2K+1)
    shape = F.shape[1:]
    coeffs, *_ = np.linalg.lstsq(Phi, F.reshape(len(times), -1), rcond=None)
    modes = [coeffs[0].reshape(shape)]
    for k in range(1, K + 1):
        modes.append((coeffs[2 * k - 1] - 1j * coeffs[2 * k]).reshape(shape))
    return modes


def dominant_omega(times, F, w_lo=0.7, w_hi=1.6, n_coarse=200):
    """The estimate's OWN shedding frequency: the omega maximising the energy
    of the k=1 harmonic component of F. Reported as a diagnostic so that
    "wrong frequency" is not silently charged to "no wake": a run shedding at
    omega != omega_0 has genuinely little energy AT omega_0, which is a real
    (and separately reportable) defect, not a missing wake."""
    ws = np.linspace(w_lo, w_hi, n_coarse)
    en = np.array([np.sum(np.abs(harmonic_modes(times, F, w, K=2)[1]) ** 2) for w in ws])
    i = int(np.argmax(en))
    if 0 < i < n_coarse - 1:                            # parabolic refinement
        y0, y1, y2 = en[i - 1], en[i], en[i + 1]
        denom = (y0 - 2 * y1 + y2)
        d = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        return float(ws[i] + d * (ws[1] - ws[0]))
    return float(ws[i])


def leakage_attenuation(omega_est, omega_ref, T):
    """Amplitude attenuation suffered by a signal oscillating at omega_est
    when it is fitted onto a harmonic basis at omega_ref over a window of
    length T.

    For a monochromatic signal the k=1 least-squares coefficient is damped by
    |sinc(dw*T/2)| = |sin(z)/z|, z = (omega_est - omega_ref) * T / 2. This is
    NOT a defect of the metric --- ModalPINN asserts its modes AT omega_0, so
    energy sitting at a different frequency genuinely is not in the mode ---
    but the SIZE of the penalty depends on window length, so:

        * modal amplitudes at omega_0 may only be compared between runs
          evaluated on the SAME window, and
        * a run's modal amplitude deficit at omega_0 must always be read
          alongside modal_metrics(..., omega=dominant_omega(...)), which
          removes the frequency error and isolates the pure amplitude error.

    Verified numerically on the Stage C free run (omega_est = 1.17083,
    omega_0 = 1.036): predicted 0.7234 over T = 20, measured peak-amplitude
    ratio 0.7184; predicted 0.9274 over T = 9.9, measured 0.9165.
    """
    z = 0.5 * (omega_est - omega_ref) * T
    if abs(z) < 1e-12:
        return 1.0
    return float(abs(np.sin(z) / z))


# --------------------------------------------------------------------------
# truth bundle
# --------------------------------------------------------------------------
class Truth:
    """Withheld ground truth, resampled once onto the modal evaluation grid.

    Attributes
    ----------
    gx, gy      : (161,), (107,) evaluation grid (same grid as
                  reference_truth_modal.npz).
    times       : (201,) absolute truth times, 400.0 .. 420.0, dt = 0.1.
    U, V        : (201, 107, 161) instantaneous truth velocity on that grid,
                  NaN inside the cylinder / outside the CFD mesh.
    fluid       : (107, 161) bool, valid-and-outside-R_EXCLUDE mask.
    modes_u/v   : [q0, q1, q2] truth modes read straight from
                  reference_truth_modal.npz (NOT refit), so the comparison is
                  against the file the task specifies.
    """

    def __init__(self, gx, gy, times, U, V, fluid, modes_u, modes_v, omega_0):
        self.gx, self.gy, self.times = gx, gy, times
        self.U, self.V, self.fluid = U, V, fluid
        self.modes_u, self.modes_v = modes_u, modes_v
        self.omega_0 = omega_0
        self.period = 2 * np.pi / omega_0
        self._sp_u = CubicSpline(times, np.nan_to_num(U), axis=0)
        self._sp_v = CubicSpline(times, np.nan_to_num(V), axis=0)

    def at(self, t):
        """Truth velocity at arbitrary times (cubic in time). Only valid
        inside [times[0], times[-1]]; callers must not extrapolate."""
        return self._sp_u(t), self._sp_v(t)


def load_truth(grid_cache=os.path.join(HERE, '_truth_grid_cache.npz')):
    """Load reference_truth_modal.npz (mode shapes) and reference_truth_full.npz
    (raw scattered instantaneous CFD), resampling the latter onto the modal
    grid. The resample costs ~4 s and is cached to a gitignorable npz."""
    with allow_truth_access():
        tm = np.load(os.path.join(DATA_DIR, 'reference_truth_modal.npz'))
        gx, gy = tm['gx'].astype(float), tm['gy'].astype(float)
        omega_0 = float(tm['omega_0'])
        modes_u = [tm['Mtrue_u0'].astype(np.complex128), tm['Mtrue_u1'].astype(np.complex128),
                   tm['Mtrue_u2'].astype(np.complex128)]
        modes_v = [tm['Mtrue_v0'].astype(np.complex128), tm['Mtrue_v1'].astype(np.complex128),
                   tm['Mtrue_v2'].astype(np.complex128)]

        if os.path.exists(grid_cache):
            c = np.load(grid_cache)
            times, U, V = c['times'], c['U'], c['V']
        else:
            tf = np.load(os.path.join(DATA_DIR, 'reference_truth_full.npz'))
            rx = tf['ref_x'].astype(float); ry = tf['ref_y'].astype(float)
            times = tf['ref_times'].astype(float)
            cu, cv = tf['ref_cu'], tf['ref_cv']
            GX, GY = np.meshgrid(gx, gy, indexing='xy')
            pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
            itp = LinearNDInterpolator(np.stack([rx, ry], axis=1), cu[0].astype(float))
            U = np.empty((len(times), len(gy), len(gx)), np.float32)
            V = np.empty_like(U)
            for k in range(len(times)):
                itp.values = cu[k].astype(float).reshape(-1, 1)
                U[k] = itp(pts).reshape(GX.shape)
                itp.values = cv[k].astype(float).reshape(-1, 1)
                V[k] = itp(pts).reshape(GX.shape)
            np.savez_compressed(grid_cache, times=times, U=U, V=V)

    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    fluid = (np.hypot(GX, GY) > R_EXCLUDE) & np.isfinite(U[0]) & np.isfinite(V[0]) \
        & np.isfinite(modes_v[1]) & np.isfinite(modes_u[1])
    return Truth(gx, gy, times, U.astype(float), V.astype(float), fluid,
                 modes_u, modes_v, omega_0)


# --------------------------------------------------------------------------
# projecting a solver run onto the evaluation grid
# --------------------------------------------------------------------------
def solver_grid(config):
    dx = (config['Lxmax'] - config['Lxmin']) / config['Nx']
    dy = (config['Lymax'] - config['Lymin']) / config['Ny']
    xc = config['Lxmin'] + (np.arange(config['Nx']) + 0.5) * dx
    yc = config['Lymin'] + (np.arange(config['Ny']) + 0.5) * dy
    return xc, yc


def project_run(u_hist, v_hist, config, gx, gy):
    """MAC-staggered run history -> (Nt, len(gy), len(gx)) cell-centre fields
    on the truth's evaluation grid, by bilinear interpolation."""
    xc, yc = solver_grid(config)
    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    q = np.stack([GY.ravel(), GX.ravel()], axis=-1)
    Nt = u_hist.shape[0]
    U = np.empty((Nt, len(gy), len(gx))); V = np.empty_like(U)
    for k in range(Nt):
        uc = 0.5 * (u_hist[k][:, :-1] + u_hist[k][:, 1:])
        vc = 0.5 * (v_hist[k][:-1, :] + v_hist[k][1:, :])
        U[k] = RegularGridInterpolator((yc, xc), uc, bounds_error=False,
                                       fill_value=np.nan)(q).reshape(GX.shape)
        V[k] = RegularGridInterpolator((yc, xc), vc, bounds_error=False,
                                       fill_value=np.nan)(q).reshape(GX.shape)
    return U, V


# --------------------------------------------------------------------------
# METRIC 1 -- phase-aligned field error
# --------------------------------------------------------------------------
def phase_aligned_field_error(t_est, F_est, truth, component='v',
                              t_eval=None, max_shift=None, n_coarse=41):
    """Relative L2 field error after an optimal sub-sample TIME SHIFT.

    ALIGNMENT METHOD AND WHY. Two options were available: (i) a sub-sample
    time shift of the estimate, applied by cubic interpolation in time, and
    (ii) a phase rotation exp(i*psi) applied to the estimate's complex modes.
    They are equivalent only if the estimate is monochromatic at exactly
    omega_0. The runs here are NOT: the forward solver sheds at
    omega_s = 1.1707 (13.0% fast), so a single psi cannot align mode 1 and
    mode 2 simultaneously, and the mean flow (k=0) must not be shifted at
    all. A TIME shift is the physically correct operation --- it is what
    "the same flow, observed at a different instant" means, it acts
    correctly and consistently on every harmonic at once (k-th mode picks up
    exp(i*k*omega*tau) automatically), and it leaves the time-mean
    untouched. The complex phase rotation IS used, but in metric 2, where it
    is applied per mode and reported as that mode's phase error --- which is
    its correct role, as a REPORTED quantity, not a fitted nuisance.

    The shift is searched over one full shedding period [-T/2, +T/2] on a
    coarse grid then refined by bounded Brent minimisation, giving a
    sub-sample tau*. The estimate is evaluated at t_eval + tau by cubic
    spline in time, so t_eval must be inset from the ends of t_est by
    max_shift; the inset is applied automatically.

    Returns dict:
        E_aligned    -- time-mean relative L2 error at the optimal shift
        tau_opt      -- fitted shift (time units); positive means the
                        estimate LAGS the truth by tau_opt
        phase_lag    -- tau_opt * omega_0, in radians, wrapped to (-pi, pi]
        E_unaligned  -- the same error at tau = 0 (i.e. the old Stage F
                        metric restricted to this window), for comparison
        E_t          -- (n_eval,) per-instant aligned error
        taus, curve  -- the coarse scan, so the minimum can be inspected
    """
    F_est = np.asarray(F_est, float)
    T = truth.period
    if max_shift is None:
        max_shift = 0.5 * T
    t_est = np.asarray(t_est, float)
    if t_eval is None:
        lo, hi = t_est[0] + max_shift, t_est[-1] - max_shift
        lo = max(lo, truth.times[0]); hi = min(hi, truth.times[-1])
        t_eval = truth.times[(truth.times >= lo) & (truth.times <= hi)]
    t_eval = np.asarray(t_eval, float)
    if len(t_eval) < 8:
        raise ValueError('evaluation window too short after inset (%d samples)' % len(t_eval))

    fl = truth.fluid.ravel()
    F_true = (truth.U if component == 'u' else truth.V)
    sp_t = CubicSpline(truth.times, np.nan_to_num(F_true), axis=0)
    Yt = sp_t(t_eval).reshape(len(t_eval), -1)[:, fl]
    den = np.linalg.norm(Yt, axis=1)

    sp_e = CubicSpline(t_est, np.nan_to_num(F_est), axis=0)

    def err_series(tau):
        Ye = sp_e(t_eval + tau).reshape(len(t_eval), -1)[:, fl]
        return np.linalg.norm(Ye - Yt, axis=1) / den

    def score(tau):
        return float(np.mean(err_series(tau)))

    taus = np.linspace(-max_shift, max_shift, n_coarse)
    curve = np.array([score(tau) for tau in taus])
    i = int(np.argmin(curve))
    lo = taus[max(i - 1, 0)]; hi = taus[min(i + 1, n_coarse - 1)]
    if hi > lo:
        tau_opt = float(minimize_scalar(score, bounds=(lo, hi), method='bounded',
                                        options=dict(xatol=1e-4)).x)
    else:
        tau_opt = float(taus[i])
    E_t = err_series(tau_opt)
    return dict(E_aligned=float(np.mean(E_t)), tau_opt=tau_opt,
                phase_lag=float(np.angle(np.exp(1j * tau_opt * truth.omega_0))),
                E_unaligned=score(0.0), E_t=E_t, t_eval=t_eval,
                taus=taus, curve=curve)


# --------------------------------------------------------------------------
# METRIC 2 -- modal amplitude comparison (PRIMARY)
# --------------------------------------------------------------------------
def _reduce_profile(A, fluid, how):
    A = np.where(fluid, A, np.nan)
    with np.errstate(invalid='ignore'):
        if how == 'max':
            out = np.full(A.shape[1], np.nan)
            ok = np.any(np.isfinite(A), axis=0)
            out[ok] = np.nanmax(A[:, ok], axis=0)
            return out
        out = np.full(A.shape[1], np.nan)
        ok = np.any(np.isfinite(A), axis=0)
        out[ok] = np.sqrt(np.nanmean(A[:, ok] ** 2, axis=0))
        return out


def modal_metrics(times, F_est, truth, component='v', K=2, omega=None,
                  x_min_profile=1.0):
    """Compare complex Fourier modes of an estimate against the truth modes.

    times  : (Nt,) ABSOLUTE times of the estimate (400.0-based clock).
    F_est  : (Nt, Ny, Nx) estimate velocity on truth.gx/gy.
    omega  : fit frequency; default truth.omega_0 (the frequency at which the
             truth modes were defined and at which ModalPINN asserts its
             modes). Pass dominant_omega(...) to get the own-frequency
             diagnostic instead.

    Per mode k = 1..K returns
      prof_est/prof_true (max and rms over y, versus x)  <-- the deliverable
      amp_rel   : || |q_e| - |q_t| || / || |q_t| ||  over fluid points.
                  PHASE-BLIND by construction. A deleted wake gives exactly
                  1.0 (the worst value); an amplitude-correct but
                  wrong-phase field gives ~0.
      cplx_rel_aligned : min_psi || e^{i psi} q_e - q_t || / || q_t ||
                  -- structure error with a single global timing offset
                  forgiven.
      psi_opt   : that global phase offset (radians), i.e. the mode's timing
                  error, REPORTED not hidden.
      cplx_rel_raw : no phase freedom (for reference).
      amp_ratio_x / phase_err_x : per-x complex projection of the estimate's
                  y-profile onto the truth's,
                      c(x) = <q_t(x,.), q_e(x,.)> / <q_t(x,.), q_t(x,.)>,
                  so |c(x)| is the local amplitude ratio (1 = correct) and
                  arg(c(x)) is the local phase error. This is the
                  amplitude/phase split as a function of downstream distance
                  that the task requires.
      profile_rel_err : relative L2 of the max-over-y profile for
                  x >= x_min_profile (the wake proper).
      persistence_est/_true : profile(x=7) / max(profile), the single number
                  that most directly encodes the ModalPINN failure --- the
                  truth's is 0.807 for |v1| (a vortex street that persists);
                  a collapsing wake gives a small value.
    """
    if omega is None:
        omega = truth.omega_0
    modes_t = truth.modes_v if component == 'v' else truth.modes_u
    modes_e = harmonic_modes(times, np.nan_to_num(np.asarray(F_est, float)), omega, K=K)
    fluid = truth.fluid
    gx = truth.gx
    i7 = int(np.argmin(np.abs(gx - 7.0)))
    xsel = gx >= x_min_profile

    out = dict(omega=float(omega), component=component)
    # k = 0 (mean flow) -- real, no phase
    m0e = np.real(modes_e[0]); m0t = np.real(modes_t[0])
    out[0] = dict(mean_rel=float(np.linalg.norm((m0e - m0t)[fluid]) /
                                 np.linalg.norm(m0t[fluid])))
    for k in range(1, K + 1):
        qe, qt = modes_e[k], np.asarray(modes_t[k])
        e, t = qe[fluid], qt[fluid]
        amp_rel = float(np.linalg.norm(np.abs(e) - np.abs(t)) / np.linalg.norm(np.abs(t)))
        inner = np.vdot(t, e)
        psi = float(-np.angle(inner))
        cplx_al = float(np.linalg.norm(e * np.exp(1j * psi) - t) / np.linalg.norm(t))
        cplx_raw = float(np.linalg.norm(e - t) / np.linalg.norm(t))

        pe_max = _reduce_profile(np.abs(qe), fluid, 'max')
        pt_max = _reduce_profile(np.abs(qt), fluid, 'max')
        pe_rms = _reduce_profile(np.abs(qe), fluid, 'rms')
        pt_rms = _reduce_profile(np.abs(qt), fluid, 'rms')

        # per-x complex projection
        amp_ratio_x = np.full(len(gx), np.nan)
        phase_err_x = np.full(len(gx), np.nan)
        for j in range(len(gx)):
            col = fluid[:, j]
            if col.sum() < 4:
                continue
            a = qt[col, j]; b = qe[col, j]
            nn = np.vdot(a, a).real
            if nn <= 0:
                continue
            c = np.vdot(a, b) / nn
            amp_ratio_x[j] = np.abs(c)
            phase_err_x[j] = np.angle(c)

        # Window-robust summary of the per-x amplitude ratio. Unlike amp_rel,
        # |c(x)| is a RATIO, so the leakage attenuation |sinc(dw*T/2)| that a
        # frequency-offset run suffers enters it as a multiplicative factor
        # common to the whole profile rather than as a window-length-dependent
        # shift of an absolute error --- which is what made amp_rel's ranking
        # of the three runs flip between sub-windows.
        with np.errstate(invalid='ignore'):
            sel_ar = xsel & np.isfinite(amp_ratio_x)
            amp_ratio_mean = float(np.mean(amp_ratio_x[sel_ar])) if sel_ar.any() else np.nan
            phase_err_mean = float(np.mean(phase_err_x[sel_ar])) if sel_ar.any() else np.nan
            phase_err_std = float(np.std(phase_err_x[sel_ar])) if sel_ar.any() else np.nan

        ok = xsel & np.isfinite(pe_max) & np.isfinite(pt_max)
        prof_rel = float(np.linalg.norm(pe_max[ok] - pt_max[ok]) / np.linalg.norm(pt_max[ok]))
        pk_e = np.nanmax(pe_max[xsel]) if np.any(np.isfinite(pe_max[xsel])) else np.nan
        pk_t = np.nanmax(pt_max[xsel])
        out[k] = dict(
            amp_rel=amp_rel, cplx_rel_aligned=cplx_al, cplx_rel_raw=cplx_raw,
            psi_opt=psi, prof_rel_err=prof_rel,
            prof_x=gx, prof_est_max=pe_max, prof_true_max=pt_max,
            prof_est_rms=pe_rms, prof_true_rms=pt_rms,
            amp_ratio_x=amp_ratio_x, phase_err_x=phase_err_x,
            amp_ratio_mean=amp_ratio_mean, amp_ratio_deficit=abs(1.0 - amp_ratio_mean),
            phase_err_mean=phase_err_mean, phase_err_std=phase_err_std,
            peak_est=float(pk_e), peak_true=float(pk_t),
            peak_x_est=float(gx[xsel][int(np.nanargmax(pe_max[xsel]))]) if np.any(np.isfinite(pe_max[xsel])) else np.nan,
            peak_x_true=float(gx[xsel][int(np.nanargmax(pt_max[xsel]))]),
            persistence_est=float(pe_max[i7] / pk_e) if pk_e and np.isfinite(pk_e) and pk_e > 0 else np.nan,
            persistence_true=float(pt_max[i7] / pk_t),
            at_x7_est=float(pe_max[i7]), at_x7_true=float(pt_max[i7]),
        )
    return out


# --------------------------------------------------------------------------
# full evaluation of one run, with window sensitivity
# --------------------------------------------------------------------------
def _windows(n, times):
    """Full window plus halves and thirds, for the window-cut sensitivity
    check that invalidated the original Stage F conclusion."""
    w = [('full', 0, n)]
    h = n // 2
    w += [('half_1', 0, h), ('half_2', h, n)]
    a, b = n // 3, 2 * n // 3
    w += [('third_1', 0, a), ('third_2', a, b), ('third_3', b, n)]
    return w


def evaluate_run(truth, u_hist, v_hist, times_abs, config, label='run',
                 K=2, do_aligned=True, windows=True):
    """Everything, on the truth grid. times_abs must be the ABSOLUTE clock."""
    U, V = project_run(u_hist, v_hist, config, truth.gx, truth.gy)
    return evaluate_gridded(truth, U, V, times_abs, label=label, K=K,
                            do_aligned=do_aligned, windows=windows)


def evaluate_gridded(truth, U, V, times_abs, label='run', K=2,
                     do_aligned=True, windows=True):
    times_abs = np.asarray(times_abs, float)
    probe = truth.fluid & (np.meshgrid(truth.gx, truth.gy, indexing='xy')[0] > 1.0) \
        & (np.abs(np.meshgrid(truth.gx, truth.gy, indexing='xy')[1]) < 2.0)
    res = dict(label=label, n_t=len(times_abs), t0=float(times_abs[0]),
               t1=float(times_abs[-1]))
    res['omega_est'] = dominant_omega(times_abs, np.nan_to_num(V)[:, probe])

    wins = _windows(len(times_abs), times_abs) if windows else [('full', 0, len(times_abs))]
    res['windows'] = {}
    for name, i0, i1 in wins:
        sl = slice(i0, i1)
        w = {}
        w['t0'] = float(times_abs[i0]); w['t1'] = float(times_abs[i1 - 1])
        w['modal_v'] = modal_metrics(times_abs[sl], V[sl], truth, 'v', K=K)
        w['modal_u'] = modal_metrics(times_abs[sl], U[sl], truth, 'u', K=K)
        w['omega_est'] = dominant_omega(times_abs[sl], np.nan_to_num(V)[sl][:, probe])
        w['modal_v_ownfreq'] = modal_metrics(times_abs[sl], V[sl], truth, 'v', K=K,
                                             omega=w['omega_est'])
        w['T_window'] = float(times_abs[i1 - 1] - times_abs[i0])
        w['leakage_att_k1'] = leakage_attenuation(w['omega_est'], truth.omega_0,
                                                  w['T_window'])
        w['leakage_att_k2'] = leakage_attenuation(2 * w['omega_est'],
                                                  2 * truth.omega_0, w['T_window'])
        if do_aligned and (i1 - i0) >= 60:
            try:
                w['aligned_v'] = phase_aligned_field_error(times_abs[sl], V[sl], truth, 'v')
                w['aligned_u'] = phase_aligned_field_error(times_abs[sl], U[sl], truth, 'u')
            except ValueError:
                w['aligned_v'] = None; w['aligned_u'] = None
        else:
            w['aligned_v'] = None; w['aligned_u'] = None
        res['windows'][name] = w
    res['full'] = res['windows']['full']
    return res


# --------------------------------------------------------------------------
# the old (broken) metric, kept so the damping test can plot both
# --------------------------------------------------------------------------
def Ev_old(times_est, F_est, truth, component='v'):
    """Stage F's metric: time-mean relative L2 against the instantaneous
    truth, no phase alignment. Reproduced here only so that
    validate_metrics_v2.py can show old and new on the same axes."""
    fl = truth.fluid.ravel()
    F_true = truth.U if component == 'u' else truth.V
    sp_t = CubicSpline(truth.times, np.nan_to_num(F_true), axis=0)
    Yt = sp_t(times_est).reshape(len(times_est), -1)[:, fl]
    Ye = np.nan_to_num(np.asarray(F_est, float)).reshape(len(times_est), -1)[:, fl]
    return float(np.mean(np.linalg.norm(Ye - Yt, axis=1) / np.linalg.norm(Yt, axis=1)))


def damp_fluctuation(F, scale):
    """Hold phase fixed, scale the fluctuation about the time-mean."""
    m = F.mean(axis=0, keepdims=True)
    return m + scale * (F - m)
