"""
STAGE D: perturbed-observation Ensemble Kalman Filter, pressure-only.

Implements exactly the mathematics in docs/DESIGN.md / the task spec:

  xbar_f, ybar_f         ensemble means (forecast state, predicted tap pressure)
  X = 1/sqrt(q-1) [x_f^j - xbar_f]      scaled state anomalies      (n_state, q)
  Y = 1/sqrt(q-1) [y_f^j* - ybar_f*]    scaled pressure anomalies   (n_taps, q)
  K = X Y^T (Y Y^T + R)^-1                                          (n_state, n_taps)
  xbar_a = xbar_f + K (y_measured - ybar_f*)
  x_a^j  = x_f^j  + K (y_measured + eps^j - y_f^j*)   [perturbed obs, eps^j ~ N(0,R), centered]

Never forms an n_state x n_state matrix: Y Y^T is only (n_taps, n_taps).
The correction K(...) is by construction in the column span of X, i.e. a
linear combination of forecast state anomalies -- see docs/DESIGN.md Sec 9
for why this keeps every analysis state divergence-free automatically.

y_f^j* denotes the GAUGE-CORRECTED prediction (see gauge_correct()): the
observer's own pressure Poisson solve is only defined up to an additive
constant, generally different from the CFD file's gauge, so the common
offset is removed per member before forming any innovation.
"""
import numpy as np

from .state_vector import StateVectorizer


def gauge_correct(p_predicted, p_measured):
    """c = mean_s[p_measured,s - p_predicted,s]; return p_predicted + c.
    Removes only the common additive offset, preserves spatial structure."""
    c = np.mean(p_measured - p_predicted)
    return p_predicted + c, c


class EnKFRun:
    def __init__(self, members, taps, obs_noise_std, inflation=1.0, seed=0):
        """members: list of q CylinderFlowSolver instances (already at their
        forecast/initial states). taps: estimator.data_interface.TapObservations.
        obs_noise_std: sigma_p for R = sigma_p^2 I (scalar, numerical-
        conditioning floor -- see DESIGN.md Sec 8)."""
        self.members = members
        self.q = len(members)
        self.taps = taps
        self.sigma_p = obs_noise_std
        self.alpha = inflation
        self.rng = np.random.default_rng(seed)
        self.vec = StateVectorizer(members[0])
        self.n_taps = taps.n_taps

    def _predict_pressure(self, solver):
        return solver.sample_pressure(self.taps.tap_x, self.taps.tap_y)

    def assimilate_step(self, p_measured):
        q, n_taps = self.q, self.n_taps

        Xf = np.stack([self.vec.flatten(m) for m in self.members], axis=1)  # (n_state, q)
        Yf_raw = np.stack([self._predict_pressure(m) for m in self.members], axis=1)  # (n_taps, q)

        # gauge-correct each member's prediction independently
        Yf = np.empty_like(Yf_raw)
        gauges = np.empty(q)
        for j in range(q):
            Yf[:, j], gauges[j] = gauge_correct(Yf_raw[:, j], p_measured)

        xbar_f = Xf.mean(axis=1)
        ybar_f = Yf.mean(axis=1)

        Xa = (Xf - xbar_f[:, None]) * self.alpha  # multiplicative inflation on anomalies
        Ya = (Yf - ybar_f[:, None]) * self.alpha
        Xs = Xa / np.sqrt(q - 1)
        Ys = Ya / np.sqrt(q - 1)

        R = (self.sigma_p ** 2) * np.eye(n_taps)
        S = Ys @ Ys.T + R  # (n_taps, n_taps) -- NEVER an n_state x n_state matrix
        K = Xs @ Ys.T @ np.linalg.solve(S, np.eye(n_taps))  # (n_state, n_taps)

        innovation_mean = p_measured - ybar_f
        xbar_a = xbar_f + K @ innovation_mean

        # perturbed observations, exactly zero-mean-centered across the
        # ensemble (Burgers et al. 1998) so the analysis-ensemble spread
        # isn't systematically inflated/deflated by sampling noise in eps
        eps = self.rng.normal(0.0, self.sigma_p, size=(n_taps, q))
        eps -= eps.mean(axis=1, keepdims=True)

        Xf_analysis = Xf + K @ (p_measured[:, None] + eps - Yf)  # (n_state, q), per-member

        for j in range(q):
            self.vec.unflatten_into(self.members[j], Xf_analysis[:, j])

        diag = dict(
            xbar_f=xbar_f, xbar_a=xbar_a,
            ybar_f=ybar_f, y_measured=p_measured.copy(),
            innovation=innovation_mean,
            innovation_norm=float(np.linalg.norm(innovation_mean)),
            kalman_correction_norm=float(np.linalg.norm(K @ innovation_mean)),
            pressure_rmse=float(np.sqrt(np.mean(innovation_mean ** 2))),
            ensemble_spread_state=float(np.sqrt(np.mean(Xs ** 2)) * np.sqrt(q - 1)),
            ensemble_spread_pressure=float(np.sqrt(np.mean(Ys ** 2)) * np.sqrt(q - 1)),
            gauges=gauges,
            NIS=float(innovation_mean @ np.linalg.solve(S, innovation_mean)),
        )
        return diag

    def forecast_step(self, substeps):
        for m in self.members:
            for _ in range(substeps):
                m.step()
