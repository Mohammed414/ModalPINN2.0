"""
The ONE sanctioned way for estimator/EnKF code to read observation data.

Wraps enkf_pressure_only/data/tap_observations.npz and exposes exactly the
fields the task spec permits the estimator to access: tap coordinates,
tap times, tap pressure, Re, and geometry/domain constants. Nothing else is
reachable through this object -- there is no method that returns velocity,
vorticity, full-field pressure, or any Mtrue_* array, and the leakage guard
(_leakage_guard.py) additionally blocks any direct np.load of the withheld
files regardless of how they'd be reached.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TAP_FILE = os.path.join(_HERE, '..', 'data', 'tap_observations.npz')


class TapObservations:
    """Estimator-facing view of the sparse pressure-tap dataset.

    Permitted attributes only: tap_x, tap_y, tap_times, tap_p, Re, r_c,
    x_c, y_c, domain (Lxmin,Lxmax,Lymin,Lymax), omega_0.
    """

    def __init__(self, n_taps=32, path=_TAP_FILE):
        if n_taps not in (4, 8, 16, 32):
            raise ValueError('n_taps must be one of 4, 8, 16, 32 (got %r)' % n_taps)
        d = np.load(path)

        self.n_taps = n_taps
        self.tap_x = d['tap_x_%d' % n_taps].copy()
        self.tap_y = d['tap_y_%d' % n_taps].copy()
        self.tap_times = d['tap_times_%d' % n_taps].copy()
        self.tap_p = d['tap_p_%d' % n_taps].copy()  # (Nt, n_taps)

        self.Re = float(d['Re'])
        self.r_c = float(d['r_c'])
        self.x_c = float(d['x_c'])
        self.y_c = float(d['y_c'])
        self.domain = d['domain'].copy()  # [Lxmin, Lxmax, Lymin, Lymax]
        self.omega_0 = float(d['omega_0'])

        assert self.tap_p.shape == (len(self.tap_times), n_taps)

    def theta(self):
        """Tap angular position around the cylinder, radians in (-pi, pi]."""
        return np.arctan2(self.tap_y - self.y_c, self.tap_x - self.x_c)

    def __repr__(self):
        return ('TapObservations(n_taps=%d, Nt=%d, Re=%.0f, r_c=%.2f, '
                'domain=%s, omega_0=%.4f)' % (
                    self.n_taps, len(self.tap_times), self.Re, self.r_c,
                    list(self.domain), self.omega_0))


if __name__ == '__main__':
    for n in (4, 8, 16, 32):
        obs = TapObservations(n_taps=n)
        print(obs)
