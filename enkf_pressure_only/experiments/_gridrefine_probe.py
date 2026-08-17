"""Grid-refinement probe (NEW file, writes a NEW npz): does the static
tap-pressure bias shrink when the immersed-boundary grid is refined?
Not part of the Stage A-F chain; diagnostic only."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import estimator
from estimator.ns_solver import CylinderFlowSolver
from estimator.data_interface import TapObservations

obs = TapObservations(n_taps=32)
out = {}
for tag, (Nx, Ny, dt) in {'coarse120x80': (120, 80, 0.005),
                          'fine240x160': (240, 160, 0.0025)}.items():
    s = CylinderFlowSolver(Nx=Nx, Ny=Ny, Lxmin=-4., Lxmax=8., Lymin=-4., Lymax=4.,
                           x_c=0., y_c=0., r_c=0.5, Re=100., dt=dt)
    t0 = time.time()
    nspin = int(round(310.0 / dt))
    for n in range(nspin):
        s.step()
    substeps = int(round(0.1 / dt))
    pw = np.empty((201, 32)); Fy = np.empty(201)
    for k in range(201):
        if k > 0:
            for _ in range(substeps):
                s.step()
        pw[k] = s.sample_pressure(obs.tap_x, obs.tap_y, method='wall_probe')
        Fy[k] = s.force_on_body()[1]
    out[tag + '_pw'] = pw
    out[tag + '_Fy'] = Fy
    print('%s done in %.0fs  mean-bias rms %.4f' % (
        tag, time.time() - t0,
        np.sqrt(np.mean((pw.mean(0) - obs.tap_p.mean(0)) ** 2))), flush=True)

np.savez_compressed(os.path.join(HERE, 'sensor_model_gridrefine.npz'),
                    tap_p_measured=obs.tap_p, tap_x=obs.tap_x, tap_y=obs.tap_y, **out)
print('wrote sensor_model_gridrefine.npz')
