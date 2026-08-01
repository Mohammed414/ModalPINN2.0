"""
Generate a tiny synthetic flow file in the format expected by text_flow.read_flow(),
for use as a stand-in for the real Boudina et al. dataset (Data/fixed_cylinder_atRe100,
see README) when running a quick smoke test of the training pipeline.

Not physically meaningful data - only shaped correctly (fixed mesh across
timesteps, points scattered within the domain box used by ModalPINN_VortexShedding.py)
so that Load_train_data_desync.read_cut_simulation_data() and friends succeed.
"""
import os
import numpy as np

Lxmin, Lxmax, Lymin, Lymax = -4., 8., -4., 4.
Re, Ur = 100., 5.  # arbitrary, unused by the smoke test
Nt, N_nodes = 10, 100

rng = np.random.default_rng(0)
nodes_x = rng.uniform(Lxmin, Lxmax, N_nodes)
nodes_y = rng.uniform(Lymin, Lymax, N_nodes)
times = np.linspace(0., 5., Nt)

os.makedirs('Data', exist_ok=True)
with open('Data/fixed_cylinder_atRe100', 'w') as f:
    f.write('%.1f %.1f\n\n' % (Re, Ur))
    f.write('%d %d\n\n' % (Nt, N_nodes))
    for t in times:
        f.write('%.6f\n' % t)
        u = np.sin(t) + 0.01 * rng.standard_normal(N_nodes)
        v = np.cos(t) + 0.01 * rng.standard_normal(N_nodes)
        p = 0.01 * rng.standard_normal(N_nodes)
        for k in range(N_nodes):
            f.write('%.9f %.9f %.9f %.9f %.9f\n' %
                    (nodes_x[k], nodes_y[k], u[k], v[k], p[k]))

print('Wrote Data/fixed_cylinder_atRe100 (%d timesteps x %d nodes)' % (Nt, N_nodes))
