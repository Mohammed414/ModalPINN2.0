"""
Frequency parameterization study (audit repair, branch enkf-repairs).

Question: is there a single scalar parameter that moves the forward solver's
shedding frequency predictably and stably over the ~15% range needed to close
the gap between the solver (omega_s ~ 1.171) and the truth (omega_0 = 1.036)?

Candidate A -- TIME DILATION.  Step the solver with dt' = dt_nom * gamma while
the OBSERVER's clock still advances dt_nom per step (i.e. the solver's internal
clock runs gamma times faster per observer second).  If temporal discretisation
error is negligible, omega_eff = gamma * omega_s.

Candidate B -- FREESTREAM RESCALING at fixed Re (U_inf = s, nu = s/Re_ref*D).
Exact dynamic similarity => predicted to be equivalent to (A); included to
confirm the two are the same handle in different bookkeeping.

Candidate C -- REYNOLDS NUMBER.  Physically real, but St(Re) is weak at Re=100.

No truth data is read anywhere in this script.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import estimator  # noqa: F401  (installs the leakage guard)
from estimator.ns_solver import CylinderFlowSolver

OMEGA_TRUTH = 1.036


# ----------------------------------------------------------------------
class ScalableCylinderFlowSolver(CylinderFlowSolver):
    """CylinderFlowSolver with a settable freestream speed U_inf.

    The base class hardcodes u=1 at the inflow face and in the far-field
    ghost rows.  Overriding those two hooks is the minimal change needed;
    nothing else in the discretisation references the freestream value.
    """

    U_inf = 1.0

    def _apply_velocity_bc(self, u, v):
        u[:, 0] = self.U_inf
        u[:, -1] = u[:, -2]
        v[0, :] = 0.0
        v[-1, :] = 0.0
        return u, v

    def _ghost_row(self, u, side):
        if side == 0:
            return 2.0 * self.U_inf - u[0, :]
        return 2.0 * self.U_inf - u[-1, :]


# ----------------------------------------------------------------------
def fit_sinusoid(t, f, w0):
    """Nonlinear single-frequency fit with a linear trend term."""
    def model(tt, A, w, ph, off, drift):
        return off + drift * (tt - tt.mean()) + A * np.cos(w * tt + ph)

    A0 = (f.max() - f.min()) / 2.0
    popt, pcov = curve_fit(model, t, f, p0=[A0, w0, 0.0, f.mean(), 0.0], maxfev=80000)
    resid = f - model(t, *popt)
    perr = np.sqrt(np.diag(pcov))
    return dict(A=float(abs(popt[0])), omega=float(abs(popt[1])),
                phase=float(popt[2]), offset=float(popt[3]), drift=float(popt[4]),
                sigma_omega=float(perr[1]),
                r2=float(1 - np.var(resid) / np.var(f)),
                resid_rms=float(np.std(resid)))


def max_div_interior(s):
    """Stage-B convention: interior only, excluding the 3-cell boundary halo
    (the outflow column carries an O(0.1) divergence by construction of the
    zero-gradient outflow BC applied after projection)."""
    div = ((s.u[:, 1:] - s.u[:, :-1]) / s.dx
           + (s.v[1:, :] - s.v[:-1, :]) / s.dy)
    return float(np.abs(div[3:-3, 3:-3]).max())


def run_case(cfg, u0, v0, gamma=1.0, Re=None, U_inf=1.0,
             T_obs=220.0, t_burn=20.0, record_dt=0.05, w0_guess=1.17):
    """Advance from (u0,v0) with dt' = dt_nom*gamma, recording Fy against the
    OBSERVER clock (which advances dt_nom per step).  Returns fit dict."""
    dt_nom = cfg['dt']
    s = ScalableCylinderFlowSolver(
        Nx=cfg['Nx'], Ny=cfg['Ny'], Lxmin=cfg['Lxmin'], Lxmax=cfg['Lxmax'],
        Lymin=cfg['Lymin'], Lymax=cfg['Lymax'], x_c=cfg['x_c'], y_c=cfg['y_c'],
        r_c=cfg['r_c'], Re=(cfg['Re'] if Re is None else Re),
        dt=dt_nom * gamma)
    s.U_inf = U_inf
    if U_inf != 1.0:
        s.u = u0 * U_inf
        s.v = v0 * U_inf
    else:
        s.u = u0.copy()
        s.v = v0.copy()
    s.t = 0.0

    n = int(round(T_obs / dt_nom))
    every = max(1, int(round(record_dt / dt_nom)))
    ts, Fy, Fx = [], [], []
    t_wall = time.time()
    blew_up = False
    for k in range(n):
        s.step()
        if k % every == 0:
            fx, fy = s.force_on_body()
            ts.append((k + 1) * dt_nom)     # OBSERVER clock
            Fx.append(fx)
            Fy.append(fy)
            if not np.isfinite(fy) or abs(fy) > 1e3:
                blew_up = True
                break
    wall = time.time() - t_wall
    ts = np.array(ts); Fy = np.array(Fy); Fx = np.array(Fx)

    out = dict(gamma=float(gamma), Re=float(s.Re), U_inf=float(U_inf),
               dt_solver=float(s.dt), T_obs_reached=float(ts[-1]) if len(ts) else 0.0,
               blew_up=bool(blew_up),
               nan_in_field=bool(not np.isfinite(s.u).all() or not np.isfinite(s.v).all()),
               max_div_interior=float('nan') if blew_up else max_div_interior(s),
               wall_s=float(wall))
    if blew_up or len(ts) < 100:
        out.update(dict(omega_eff=float('nan'), sigma_omega=float('nan'),
                        A=float('nan'), r2=float('nan'), resid_rms=float('nan'),
                        omega_h1=float('nan'), omega_h2=float('nan'),
                        omega_solver_clock=float('nan')))
        return out, ts, Fy, Fx

    m = ts > t_burn
    tf, ff = ts[m], Fy[m]
    fit = fit_sinusoid(tf, ff, w0_guess)
    nh = len(tf) // 2
    f1 = fit_sinusoid(tf[:nh], ff[:nh], fit['omega'])
    f2 = fit_sinusoid(tf[nh:], ff[nh:], fit['omega'])
    out.update(dict(omega_eff=fit['omega'], sigma_omega=fit['sigma_omega'],
                    A=fit['A'], r2=fit['r2'], resid_rms=fit['resid_rms'],
                    offset=fit['offset'], drift=fit['drift'],
                    omega_h1=f1['omega'], omega_h2=f2['omega'],
                    # frequency measured in the SOLVER's own internal clock:
                    # tau = gamma * t_observer  =>  omega_internal = omega_eff/gamma
                    omega_solver_clock=fit['omega'] / gamma,
                    mean_Fx=float(Fx[m].mean())))
    return out, ts, Fy, Fx
