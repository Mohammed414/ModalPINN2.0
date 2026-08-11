"""
STAGE C: free-run control. Continue the observer's own solver -- no
pressure assimilation at all -- from a single snapshot of its own
saturated limit cycle (spin_up_solver.py's output), for the same duration
and at the same instants as the tap dataset (Delta_t=0.1, 201 instants
spanning what the CFD calls t=[400,420]).

The initial condition is picked from the observer's OWN dynamics only --
never from the reference CFD -- so its phase relative to the hidden truth
is, by construction, arbitrary/uncontrolled by us. This is exactly the
"deliberately wrong phase, not selected using the reference field"
requirement: we didn't have to engineer disagreement, two independent
dynamical systems (this solver vs. the real CFD) simply have no reason to
share a phase.

This run is essential: it is what "no information at all" looks like, the
baseline Experiment 2 (EnKF) must clearly beat.
"""
import json
import os
import numpy as np

import estimator
from estimator.ns_solver import CylinderFlowSolver
from estimator.data_interface import TapObservations

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
np.random.seed(SEED)

CANONICAL_IC_TIME = 310.0  # first snapshot in spinup_snapshots.npz


def main():
    spin = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    c = json.loads(str(spin['solver_config']))
    times_avail = spin['times']
    ic_idx = int(np.argmin(np.abs(times_avail - CANONICAL_IC_TIME)))

    solver = CylinderFlowSolver(Nx=c['Nx'], Ny=c['Ny'],
                                 Lxmin=c['Lxmin'], Lxmax=c['Lxmax'],
                                 Lymin=c['Lymin'], Lymax=c['Lymax'],
                                 x_c=c['x_c'], y_c=c['y_c'], r_c=c['r_c'],
                                 Re=c['Re'], dt=c['dt'])
    solver.u = spin['u'][ic_idx].copy()
    solver.v = spin['v'][ic_idx].copy()
    solver.p = spin['p'][ic_idx].copy()  # dynamically-consistent pressure at this instant
    solver.t = 0.0  # reset to the experiment's own clock

    obs = TapObservations(n_taps=32)
    n_assim = len(obs.tap_times)          # 201
    dt_assim = obs.tap_times[1] - obs.tap_times[0]  # 0.1
    substeps = int(round(dt_assim / c['dt']))
    assert substeps * c['dt'] == dt_assim or abs(substeps * c['dt'] - dt_assim) < 1e-9

    exp_times = np.arange(n_assim) * dt_assim  # 0, 0.1, ..., 20.0 (relative clock)

    u_hist = np.empty((n_assim,) + solver.u.shape, dtype=np.float32)
    v_hist = np.empty((n_assim,) + solver.v.shape, dtype=np.float32)
    tap_p_pred = np.empty((n_assim, obs.n_taps))
    Fy_hist = np.empty(n_assim)

    # instant 0 = the IC itself, no step taken yet in this run -> force_on_body()
    # isn't defined until after a step; Fy_hist[0] is backfilled from Fy_hist[1]
    # below (off by one solver substep, dt_solver=0.005 << dt_assim=0.1, negligible).
    u_hist[0] = solver.u; v_hist[0] = solver.v
    tap_p_pred[0] = solver.sample_pressure(obs.tap_x, obs.tap_y)
    for k in range(1, n_assim):
        for _ in range(substeps):
            solver.step()
        u_hist[k] = solver.u
        v_hist[k] = solver.v
        tap_p_pred[k] = solver.sample_pressure(obs.tap_x, obs.tap_y)
        _, Fy_hist[k] = solver.force_on_body()  # lift-like, clean fundamental-freq content (Stage B)
    Fy_hist[0] = Fy_hist[1]

    out_path = os.path.join(HERE, 'stage_c_free_run_control.npz')
    np.savez_compressed(out_path,
                         exp_times=exp_times, tap_times_true=obs.tap_times,
                         u_hist=u_hist, v_hist=v_hist, tap_p_pred=tap_p_pred, Fy_hist=Fy_hist,
                         tap_p_measured=obs.tap_p, tap_x=obs.tap_x, tap_y=obs.tap_y,
                         ic_time_in_spinup=times_avail[ic_idx],
                         solver_config=json.dumps(c))
    print('Wrote %s' % out_path)

    # quick diagnostic: pressure "innovation" (informational only, nothing
    # is corrected in this run) and its RMS, for comparison against Stage D
    rmse = np.sqrt(np.mean((tap_p_pred - obs.tap_p) ** 2))
    print('Free-run tap-pressure RMSE vs measured (informational only): %.4f' % rmse)
    print('(no assimilation performed; this run is the "zero information" baseline)')


if __name__ == '__main__':
    main()
