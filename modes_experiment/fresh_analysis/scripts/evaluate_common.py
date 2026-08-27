"""Shared, deterministic metrics for the fresh-analysis arms.

This module contains no model-loading or plotting code. Each arm supplies
already aligned arrays with shape ``(time, node)`` for ``u``, ``v``, and ``p``;
all arms then use these same functions for regional field and first-harmonic
metrics.

Conventions are frozen in ``data_contract.md``:

* strict crop ``-4 < x < 8`` and ``-4 < y < 4``;
* all available snapshots are aggregated together;
* raw pressure is the primary metric;
* the first-harmonic coefficient is ``0.5*(cos_coeff - 1j*sin_coeff)``;
* phase is ``arg(true.conj() @ prediction)`` in degrees.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-30
XMIN, XMAX = -4.0, 8.0
YMIN, YMAX = -4.0, 4.0
R_C = 0.5
R_REGION = 0.75
X_WAKE_SPLIT = 3.0
Y_CORE = 2.0
OMEGA_0 = 1.036


def strict_crop_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices in the strict evaluation crop used by every arm."""
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    return np.flatnonzero(
        (x > XMIN) & (x < XMAX) & (y > YMIN) & (y < YMAX)
    )


def region_masks(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Build the fixed region masks for already-cropped node coordinates.

    The four partition members are ``near-cylinder``, ``near-wake``,
    ``far-wake``, and ``other``. ``far-core`` is deliberately a nested subset
    of ``far-wake``. Call :func:`strict_crop_indices` before this function when
    starting from the complete CFD mesh.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    r = np.sqrt(x * x + y * y)
    near_cylinder = r < R_REGION
    near_wake = (~near_cylinder) & (x >= 0.0) & (x < X_WAKE_SPLIT)
    far_wake = (~near_cylinder) & (x >= X_WAKE_SPLIT)
    other = ~(near_cylinder | near_wake | far_wake)
    far_core = far_wake & (np.abs(y) <= Y_CORE)
    regions = {
        "near-cylinder": near_cylinder,
        "near-wake": near_wake,
        "far-wake": far_wake,
        "far-core": far_core,
        "other": other,
        "whole-domain": np.ones(x.size, dtype=bool),
    }
    if np.count_nonzero(near_cylinder | near_wake | far_wake | other) != x.size:
        raise AssertionError("partition does not cover all cropped nodes")
    if np.any(far_core & ~far_wake):
        raise AssertionError("far core is not a subset of far wake")
    return regions


def _select(values: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Select a node mask from a 1-D modal or 2-D time/node array."""
    values = np.asarray(values)
    if mask is None:
        return values.reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.ndim == 1:
        if values.shape[0] != mask.shape[0]:
            raise ValueError("mask length does not match values")
        return values[mask]
    if values.ndim == 2:
        if values.shape[1] != mask.shape[0]:
            raise ValueError("mask length does not match node dimension")
        return values[:, mask].reshape(-1)
    raise ValueError("values must be one- or two-dimensional")


def relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = EPS,
) -> float:
    """Return ``||prediction-reference||_2 / (||reference||_2 + eps)``."""
    prediction = _select(prediction, mask)
    reference = _select(reference, mask)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes do not match")
    return float(np.linalg.norm(prediction - reference) /
                 (np.linalg.norm(reference) + eps))


def complex_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = EPS,
) -> Dict[str, float]:
    """Return first-harmonic error, amplitude, correlation, and phase metrics.

    Correlation is deliberately phase-insensitive (absolute inner product).
    Phase uses ``reference.conj() @ prediction`` so a positive value means the
    prediction leads the reference in the stated complex convention.
    """
    p = _select(prediction, mask).astype(np.complex128, copy=False)
    q = _select(reference, mask).astype(np.complex128, copy=False)
    if p.shape != q.shape:
        raise ValueError("prediction and reference shapes do not match")
    norm_p = float(np.linalg.norm(p))
    norm_q = float(np.linalg.norm(q))
    denom = norm_q + eps
    if norm_p <= eps or norm_q <= eps:
        correlation = float("nan")
        phase_deg = float("nan")
    else:
        correlation = float(abs(np.vdot(q, p)) /
                           (norm_p * norm_q + eps))
        phase_deg = float(np.degrees(np.angle(np.vdot(q, p))))
    return {
        "n": int(q.size),
        "rel_L2": float(np.linalg.norm(p - q) / denom),
        "amp_ratio": float(norm_p / denom),
        "corr": correlation,
        "phase_deg": phase_deg,
    }


def temporal_harmonic_coefficients(
    field: np.ndarray,
    times: Sequence[float],
    omega: float = OMEGA_0,
    kmax: int = 3,
) -> np.ndarray:
    """Fit conventional complex Fourier coefficients at every node.

    ``field`` has shape ``(time, node)``. The output has shape
    ``(kmax + 1, node)`` and follows
    ``q(t)=q0 + q1 exp(i omega t) + conj(q1) exp(-i omega t) + ...``.
    """
    field = np.asarray(field)
    times = np.asarray(times, dtype=float).reshape(-1)
    if field.ndim != 2 or field.shape[0] != times.size:
        raise ValueError("field must have shape (time, node) matching times")
    if int(kmax) < 0:
        raise ValueError("kmax must be non-negative")
    tau = times - times[0]
    columns = [np.ones_like(tau)]
    for k in range(1, int(kmax) + 1):
        columns.extend((np.cos(k * omega * tau), np.sin(k * omega * tau)))
    design = np.stack(columns, axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, field, rcond=None)
    modes = np.empty((int(kmax) + 1, field.shape[1]), dtype=np.complex128)
    modes[0] = coefficients[0]
    for k in range(1, int(kmax) + 1):
        modes[k] = 0.5 * (coefficients[2 * k - 1] -
                          1j * coefficients[2 * k])
    return modes


def regional_field_metrics(
    predictions: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
    regions: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Return relative field errors keyed as ``region -> variable``."""
    if set(predictions) != set(references):
        raise ValueError("prediction/reference variable sets do not match")
    return {
        region: {
            variable: relative_l2(predictions[variable], references[variable], mask)
            for variable in predictions
        }
        for region, mask in regions.items()
    }


def regional_complex_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    regions: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Return complex metrics keyed by region."""
    return {
        region: complex_metrics(prediction, reference, mask)
        for region, mask in regions.items()
    }
