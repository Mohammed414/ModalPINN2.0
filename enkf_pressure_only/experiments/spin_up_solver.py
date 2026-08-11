"""
Spin up the observer's own solver from rest to its saturated periodic
limit cycle, and save a short window of full (u,v) snapshots spanning
more than one shedding period. This is the ONLY source of initial
conditions for Stage C (free-run control) and Stage D (EnKF): every
initial state used anywhere downstream is a snapshot of the solver's own
dynamics, never anything derived from the reference CFD truth.

Re-uses the exact solver configuration validated in Stage B
(stage_b_report.json).
"""
import json
import os
import numpy as np

import estimator  # installs the leakage guard
from estimator.ns_solver import CylinderFlowSolver

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, 'stage_b_report.json')) as f:
    stage_b = json.load(f)
c = stage_b['config']
assert stage_b['overall_pass'], 'Stage B did not pass -- refusing to spin up on an unvalidated solver.'

SNAPSHOT_WINDOW_START = 310.0   # safely past saturation onset (~t=280-300, see stage_b figure)
SNAPSHOT_WINDOW_END = 322.0     # spans > 2 periods (T = 2pi/1.1707 ~= 5.37)
SNAPSHOT_DT = 0.02


def main():
    solver = CylinderFlowSolver(Nx=c['Nx'], Ny=c['Ny'],
                                 Lxmin=c['Lxmin'], Lxmax=c['Lxmax'],
                                 Lymin=c['Lymin'], Lymax=c['Lymax'],
                                 x_c=c['x_c'], y_c=c['y_c'], r_c=c['r_c'],
                                 Re=c['Re'], dt=c['dt'])

    n_to_window = int(round(SNAPSHOT_WINDOW_START / c['dt']))
    print('Running %d steps (t=0 -> %.1f) to reach the saturated limit cycle...'
          % (n_to_window, SNAPSHOT_WINDOW_START))
    for _ in range(n_to_window):
        solver.step()
    print('Reached t=%.2f' % solver.t)

    snap_every = int(round(SNAPSHOT_DT / c['dt']))
    n_window_steps = int(round((SNAPSHOT_WINDOW_END - SNAPSHOT_WINDOW_START) / c['dt']))

    times, us, vs, ps = [], [], [], []
    for n in range(n_window_steps + 1):
        if n % snap_every == 0:
            times.append(solver.t)
            us.append(solver.u.copy())
            vs.append(solver.v.copy())
            ps.append(solver.p.copy())  # dynamically-consistent pressure at this instant
        if n < n_window_steps:
            solver.step()
    print('Saved %d snapshots spanning t=[%.2f, %.2f]' % (len(times), times[0], times[-1]))

    out_path = os.path.join(HERE, 'spinup_snapshots.npz')
    np.savez_compressed(out_path,
                         times=np.array(times),
                         u=np.array(us), v=np.array(vs), p=np.array(ps),
                         solver_config=json.dumps(c))
    print('Wrote %s' % out_path)


if __name__ == '__main__':
    main()
