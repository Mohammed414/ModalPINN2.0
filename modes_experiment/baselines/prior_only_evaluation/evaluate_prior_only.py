#!/usr/bin/env python3
"""Evaluate the taps-only Karman street prior without a neural network.

This uses the same ``Street`` model and saved parameters consumed/created by
``street_prior.py``.  It evaluates the complete analytic velocity and pressure
field on the DNS mesh, using the regional relative-L2 convention from notebook
15.  It also reports the k=1 v-mode metrics used by ``evaluate_v1_smoke.py``.

Example (from the repository root)::

    python modes_experiment/prior_only_evaluation/evaluate_prior_only.py \
      --Prior modes_experiment/runs/arms/15_karman_prior_fluct_off/street_prior_used.npz \
      --Data GappyPOD/data/flow_cache.npz

The pressure is the prior's Bernoulli-like surrogate
``p=-0.5*((u-Uc)**2+v**2)``; it is not a pressure solution of a separate
Poisson solve.  No training checkpoint or TensorFlow is used.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]       # ModalPINN2.0/
WORKSPACE_ROOT = HERE.parents[3]  # directory containing ModalPINN2.0/
SRC = REPO_ROOT / "src" / "R9" / "src"
sys.path.insert(0, str(SRC))
# street_prior imports the repository's lightweight text-flow reader.
sys.path.insert(0, str(REPO_ROOT / "src"))
from street_prior import Street, cf_modes_uv  # noqa: E402


LXMIN, LXMAX, LYMIN, LYMAX = -4.0, 8.0, -4.0, 4.0
R_C = 0.5
OMEGA_0 = 1.036                 # notebook-15 evaluator convention


def load_data(path):
    """Load either the compact cache or the original text-flow file."""
    p = pathlib.Path(path)
    if p.suffix == ".npz":
        z = np.load(p)
        # flow_cache.npz uses one-dimensional X/Y and the same field layout as
        # text_flow.read_flow after transposition.
        return (float(z["Re"]), float(z["Ur"]), np.asarray(z["times"]),
                np.asarray(z["X"]), np.asarray(z["Y"]),
                np.asarray(z["U"]), np.asarray(z["V"]), np.asarray(z["p"]))
    from text_flow import read_flow
    return read_flow(str(p))


def relative_l2(pred, true, mask):
    a = pred[:, mask]
    b = true[:, mask]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def complex_metrics(pred, true, mask):
    p, q = pred[mask], true[mask]
    nq, np_ = np.linalg.norm(q), np.linalg.norm(p)
    return {
        "n": int(mask.sum()),
        "rel_L2": float(np.linalg.norm(p - q) / (nq + 1e-30)),
        "corr": float(abs(np.vdot(p, q)) / (np_ * nq + 1e-30)),
        "amp_ratio": float(np_ / (nq + 1e-30)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Prior", required=True, help="street_prior_used.npz")
    ap.add_argument("--Data", default=str(WORKSPACE_ROOT / "GappyPOD/data/flow_cache.npz"),
                    help="flow_cache.npz or fixed_cylinder_atRe100")
    ap.add_argument("--Out", default=str(HERE / "prior_only_metrics.json"))
    ap.add_argument("--Figure", default=str(HERE / "prior_only_fields.png"))
    ap.add_argument("--Chunk", type=int, default=4000)
    ap.add_argument("--NTaps", type=int, default=32)
    ap.add_argument("--Model", choices=("modal", "numeric"), default="modal",
                    help="modal = exact closed-form prior used by NN_functions; "
                         "numeric = reference Lamb-Oseen/image street")
    args = ap.parse_args()

    z = np.load(args.Prior)
    names = ("Gamma", "Uc", "xf", "r0", "omega", "phase", "amp_scale",
             "scale_p", "ramp", "delta")
    prm = {k: float(z[k]) for k in names}
    street = Street(prm["Gamma"], prm["Uc"], prm["xf"], prm["r0"],
                    prm["phase"], prm["omega"], prm["ramp"])

    Re, Ur, times, X, Y, U, V, P = load_data(args.Data)
    times = np.asarray(times, dtype=float)
    X, Y = np.asarray(X).reshape(-1), np.asarray(Y).reshape(-1)
    # Exactly the crop used by notebook 15's evaluate_regions.py.
    box = ((X > LXMIN) & (X < LXMAX) & (Y > LYMIN) & (Y < LYMAX))
    idx = np.where(box)[0]
    x, y = X[idx], Y[idx]
    U, V, P = U[:, idx], V[:, idx], P[:, idx]
    r = np.sqrt(x * x + y * y)
    regions = {
        "near-cylinder": r < 1.5 * R_C,
        "near-wake": (r >= 1.5 * R_C) & (x >= 0) & (x < 3),
        "far-wake": (r >= 1.5 * R_C) & (x >= 3),
        "far-core": (r >= 1.5 * R_C) & (x >= 3) & (np.abs(y) <= 2.0),
        "other (upstream/off-axis)": ~((r < 1.5 * R_C) |
                                       ((r >= 1.5 * R_C) & (x >= 0) & (x < 3)) |
                                       ((r >= 1.5 * R_C) & (x >= 3))),
        "whole domain": np.ones(len(x), dtype=bool),
    }
    # The prior is identified from 32 uniformly spaced wall taps.  Keep an
    # explicit in-sample pressure score so it is clear what was fitted versus
    # what is genuinely predicted in the interior.
    ntaps = int(args.NTaps)
    wall = np.where((np.sqrt(X * X + Y * Y) - R_C) ** 2 < 1e-5)[0]
    theta_wall = np.arctan2(Y[wall], X[wall])
    target_theta = np.linspace(0.0, 1.0, ntaps, endpoint=False) * 2.0 * np.pi
    tap_global = wall[np.array([np.argmin((X[wall] - R_C*np.cos(th))**2 +
                                          (Y[wall] - R_C*np.sin(th))**2)
                                for th in target_theta])]
    crop_lookup = {int(g): i for i, g in enumerate(idx)}
    tap = np.array([crop_lookup[int(g)] for g in tap_global if int(g) in crop_lookup], dtype=int)
    print("Prior parameters:", json.dumps(prm, sort_keys=True))
    print("Prior field model:", args.Model)
    print("DNS crop: %d snapshots x %d nodes (Re=%.1f)" %
          (U.shape[0], U.shape[1], Re))

    # Accumulate relative-L2 numerators and denominators without retaining a
    # second 201 x 51k prediction array.  Also accumulate temporal Fourier
    # projections for the v1 comparison used in notebook 15.
    num = {n: {q: 0.0 for q in ("u", "v", "p")} for n in regions}
    den = {n: {q: 0.0 for q in ("u", "v", "p")} for n in regions}
    B = np.stack([np.ones_like(times),
                  np.cos(OMEGA_0 * (times - times[0])),
                  np.sin(OMEGA_0 * (times - times[0]))], axis=1)
    pinvB = np.linalg.inv(B.T @ B) @ B.T
    vproj = np.zeros((3, len(x)), dtype=float)
    vtrue_proj = pinvB @ V
    # A few representative snapshots for the diagnostic image.
    image_snap = [0, len(times) // 2, len(times) - 1]
    image_pred = {}
    tap_num = tap_den = 0.0

    def modal_values(xc, yc, t):
        """Exact NumPy port of NN_functions.street_modes_k + NN_time_*.

        The network convention is one-sided with no factor two in
        reconstruction, hence cf_modes_uv (conventional coefficients) is
        multiplied by ``2*amp_scale`` here.  The k=0 velocity is the same
        potential-flow dipole used by the numeric Street reference.  Pressure
        uses the prior's linearized-Bernoulli harmonic anchor and a Bernoulli
        mean; this is the only pressure approximation in this evaluator.
        """
        prm_cf = prm
        us, vs = cf_modes_uv(xc, yc, prm_cf, nk=3)
        amp = 2.0 * prm["amp_scale"]
        a2 = R_C ** 2
        r2 = xc * xc + yc * yc + 1e-12
        u0 = 1.0 - a2 * (xc * xc - yc * yc) / r2 ** 2
        v0 = -a2 * 2.0 * xc * yc / r2 ** 2
        # Use a one-sided reconstruction exactly as NN_time_uv/p does.
        up, vp = u0.copy(), v0.copy()
        p0 = -0.5 * ((u0 - prm["Uc"]) ** 2 + v0 ** 2)
        pp = p0.copy()
        for kk, (uk, vk) in enumerate(zip(us, vs), start=1):
            eu = amp * uk
            ev = amp * vk
            ep = -(1.0 - prm["Uc"]) * prm["scale_p"] * eu
            phase_t = np.exp(1j * kk * OMEGA_0 * (t - times[0]))
            up += np.real(eu * phase_t)
            vp += np.real(ev * phase_t)
            pp += np.real(ep * phase_t)
        return up, vp, pp

    for it, t in enumerate(times):
        up = np.empty(len(x), dtype=float)
        vp = np.empty(len(x), dtype=float)
        pp = np.empty(len(x), dtype=float)
        for j in range(0, len(x), max(1, args.Chunk)):
            sl = slice(j, min(j + max(1, args.Chunk), len(x)))
            if args.Model == "numeric":
                pts = np.column_stack((x[sl], y[sl]))
                up[sl], vp[sl] = street.velocity(pts, float(t))
                pp[sl] = street.pressure(pts, float(t))
            else:
                up[sl], vp[sl], pp[sl] = modal_values(x[sl], y[sl], float(t))
        vproj += np.outer(B[it], vp)
        if it in image_snap:
            image_pred[it] = (up.copy(), vp.copy(), pp.copy())
        for n, m in regions.items():
            for q, a, b in (("u", up, U[it]), ("v", vp, V[it]), ("p", pp, P[it])):
                d = a[m] - b[m]
                num[n][q] += float(np.dot(d, d))
                den[n][q] += float(np.dot(b[m], b[m]))
        if len(tap):
            dp = pp[tap] - P[it, tap]
            tap_num += float(np.dot(dp, dp))
            tap_den += float(np.dot(P[it, tap], P[it, tap]))
        if (it + 1) % 25 == 0 or it == len(times) - 1:
            print("  %d/%d snapshots" % (it + 1, len(times)))

    result = {
        "prior": str(pathlib.Path(args.Prior)),
        "data": str(pathlib.Path(args.Data)),
        "parameters": prm,
        "crop_nodes": int(len(x)),
        "snapshots": int(len(times)),
        "regions": {},
    }
    print("\nPRIOR-ONLY FIELD ERRORS (not trained)")
    print("%-28s %8s %10s %10s %10s" % ("region", "n", "E_u", "E_v", "E_p"))
    for n, m in regions.items():
        e = {q: float(np.sqrt(num[n][q] / (den[n][q] + 1e-30)))
             for q in ("u", "v", "p")}
        result["regions"][n] = {"n_nodes": int(m.sum()), **e}
        print("%-28s %8d %10.4f %10.4f %10.4f" %
              (n, int(m.sum()), e["u"], e["v"], e["p"]))

    # v1 mode: notebook 15 uses q1 = (cos-coefficient - i*sin-coefficient)/2.
    # vproj is the same least-squares projection, accumulated as B.T @ pred.
    vcoef = (np.linalg.inv(B.T @ B) @ vproj)
    v1_pred = 0.5 * (vcoef[1] - 1j * vcoef[2])
    v1_true = 0.5 * (vtrue_proj[1] - 1j * vtrue_proj[2])
    result["v1_mode"] = {}
    print("\nPRIOR-ONLY v1 MODE")
    print("%-16s %8s %10s %10s %10s" % ("region", "n", "rel_L2", "corr", "amp_ratio"))
    for n, m in regions.items():
        z1 = complex_metrics(v1_pred, v1_true, m)
        result["v1_mode"][n] = z1
        print("%-16s %8d %10.4f %10.4f %10.4f" %
              (n, z1["n"], z1["rel_L2"], z1["corr"], z1["amp_ratio"]))
    result["tap_pressure"] = {
        "n_taps": int(len(tap)),
        "relative_L2": float(np.sqrt(tap_num / (tap_den + 1e-30))),
    }
    print("\nIn-sample pressure taps: %d, relative L2 = %.4f" %
          (len(tap), result["tap_pressure"]["relative_L2"]))

    # Plot one representative DNS/prior/error triplet for u, v, p at the
    # middle snapshot.  This is intentionally a compact diagnostic, not a
    # training artifact.
    k = image_snap[1]
    up, vp, pp = image_pred[k]
    fig, ax = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for row, (name, truth, pred) in enumerate((("u", U[k], up), ("v", V[k], vp), ("p", P[k], pp))):
        lo, hi = np.nanpercentile(truth, [1, 99])
        for col, vals, title in ((0, truth, "DNS " + name),
                                 (1, pred, "prior " + name),
                                 (2, np.abs(pred - truth), "|error|")):
            sc = ax[row, col].scatter(x, y, c=vals, s=1, cmap="viridis",
                                      vmin=(lo if col < 2 else None),
                                      vmax=(hi if col < 2 else None))
            ax[row, col].set_aspect("equal")
            ax[row, col].set_title(title)
            fig.colorbar(sc, ax=ax[row, col], fraction=.046, pad=.02)
    fig.savefig(args.Figure, dpi=160)
    plt.close(fig)
    result["figure"] = args.Figure
    pathlib.Path(args.Out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.Out, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved", args.Out)
    print("Saved", args.Figure)


if __name__ == "__main__":
    main()
