"""Blockage probe (NEW file, NEW npz): the reconstruction domain y in [-4,4]
gives a 12.5% blockage ratio with slip-wall-like far-field BCs (v=0 exactly),
whereas the reference CFD spans y in [-60,60] (~0.8%). Hypothesis: confinement
over-accelerates the flow around the cylinder, over-amplifying the mean surface
Cp AND raising the shedding frequency. Test at fixed dx=dy=0.1 by widening y."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import estimator
from estimator.ns_solver import CylinderFlowSolver
from estimator.data_interface import TapObservations

obs = TapObservations(n_taps=32)
out = {}
for tag, Ly in [('Ly4', 4.0), ('Ly8', 8.0), ('Ly16', 16.0)]:
    Ny = int(round(2 * Ly / 0.1))
    s = CylinderFlowSolver(Nx=120, Ny=Ny, Lxmin=-4., Lxmax=8., Lymin=-Ly, Lymax=Ly,
                           x_c=0., y_c=0., r_c=0.5, Re=100., dt=0.005)
    t0 = time.time()
    for n in range(int(round(310.0 / 0.005))):
        s.step()
    pw = np.empty((201, 32)); Fy = np.empty(201); Fx = np.empty(201)
    for k in range(201):
        if k > 0:
            for _ in range(20): s.step()
        pw[k] = s.sample_pressure(obs.tap_x, obs.tap_y, method='wall_probe')
        Fx[k], Fy[k] = s.force_on_body()
    out[tag + '_pw'] = pw; out[tag + '_Fy'] = Fy; out[tag + '_Fx'] = Fx
    mb = pw.mean(0) - obs.tap_p.mean(0)
    print('%s (Ny=%d, blockage %.1f%%) %.0fs  bias rms %.4f  front-to-base %.4f'
          % (tag, Ny, 100.0 / (2 * Ly), time.time() - t0,
             np.sqrt(np.mean(mb ** 2)), pw.mean(0).max() - pw.mean(0).min()), flush=True)

np.savez_compressed(os.path.join(HERE, 'sensor_model_blockage.npz'),
                    tap_p_measured=obs.tap_p, tap_x=obs.tap_x, tap_y=obs.tap_y, **out)
print('wrote sensor_model_blockage.npz')
