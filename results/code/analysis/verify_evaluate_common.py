"""Deterministic checks for ``evaluate_common``."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_common import (  # noqa: E402
    OMEGA_0,
    complex_metrics,
    region_masks,
    relative_l2,
    strict_crop_indices,
    temporal_harmonic_coefficients,
)


def main() -> None:
    true = np.array([1.0, -2.0, 3.0])
    assert relative_l2(true, true) == 0.0
    assert np.isclose(relative_l2(2.0 * true, true), 1.0)

    theta = np.deg2rad(30.0)
    q = np.array([1.0 + 0.5j, -0.25 + 0.75j])
    shifted = q * np.exp(1j * theta)
    m = complex_metrics(shifted, q)
    assert np.isclose(m["amp_ratio"], 1.0)
    assert np.isclose(m["corr"], 1.0)
    assert np.isclose(m["phase_deg"], 30.0)
    assert np.isclose(m["rel_L2"], 2.0 * np.sin(theta / 2.0))

    times = np.linspace(0.0, 20.0, 201)
    q0 = np.array([0.2, -0.1])
    q1 = np.array([0.7 + 0.2j, -0.3 + 0.5j])
    field = q0[None, :] + np.real(
        2.0 * q1[None, :] * np.exp(1j * OMEGA_0 * times[:, None])
    )
    modes = temporal_harmonic_coefficients(field, times, kmax=1)
    assert np.allclose(modes[0], q0, atol=1e-10)
    assert np.allclose(modes[1], q1, atol=1e-10)

    # A non-zero time origin is the case that previously exposed the bug:
    # fitting on tau=t-t0 is stable, but the returned coefficient must still
    # be expressed against exp(i*omega*t), as the ModalPINN mode is.
    shifted_times = times + 400.0
    shifted_field = q0[None, :] + np.real(
        2.0 * q1[None, :] * np.exp(1j * OMEGA_0 * shifted_times[:, None])
    )
    shifted_modes = temporal_harmonic_coefficients(
        shifted_field, shifted_times, kmax=1
    )
    assert np.allclose(shifted_modes[1], q1, atol=1e-10)

    x = np.array([-4.1, -3.0, 0.0, 2.0, 3.0, 7.9, 8.1])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    idx = strict_crop_indices(x, y)
    assert np.array_equal(idx, np.array([1, 2, 3, 4, 5]))
    regions = region_masks(x[idx], y[idx])
    assert sum(int(v.sum()) for k, v in regions.items()
               if k in ("near-cylinder", "near-wake", "far-wake", "other")) == 5
    assert np.all(regions["far-core"] <= regions["far-wake"])

    report = {
        "status": "passed",
        "checks": [
            "relative_l2_identity_and_scale",
            "complex_phase_and_amplitude",
            "temporal_harmonic_recovery",
            "temporal_harmonic_recovery_nonzero_time_origin",
            "strict_crop_and_nested_regions",
        ],
    }
    out = HERE.parent / "derived" / "common_evaluator_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
