"""Diversity weighting pattern (US 1,747,221).

Bown's 1930 patent: split the incoming signal into sub-bands, measure the
quality of each sub-band independently, and combine them with weights
proportional to measured quality.  This is the core of CHIME's channel
grading and also appears in GRIM-S's weight-capped stacking.

This module provides the generic building block that both instruments share.
"""

from __future__ import annotations

import numpy as np


def diversity_weight(
    values: np.ndarray,
    uncertainties: np.ndarray,
    max_weight_ratio: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """Inverse-variance weighted combination with optional weight capping.

    Parameters
    ----------
    values : array of float
        Per-channel (or per-event) measurements.
    uncertainties : array of float
        Per-channel (or per-event) 1-sigma uncertainties.
    max_weight_ratio : float, optional
        Cap the maximum weight at this multiple of the mean weight.
        Prevents single-channel dominance (Bown's sub-band principle:
        no single sub-band should dominate the combination).

    Returns
    -------
    combined : float
        Weighted mean.
    combined_unc : float
        Uncertainty on the weighted mean.
    weights : array
        Normalized weights used.
    """
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)

    mask = (uncertainties > 0) & np.isfinite(values) & np.isfinite(uncertainties)
    if not np.any(mask):
        return np.nan, np.nan, np.zeros_like(values)

    v = values[mask]
    u = uncertainties[mask]

    raw_weights = 1.0 / u**2

    if max_weight_ratio is not None and max_weight_ratio > 0:
        mean_w = np.mean(raw_weights)
        cap = max_weight_ratio * mean_w
        raw_weights = np.minimum(raw_weights, cap)

    w_sum = np.sum(raw_weights)
    norm_weights = raw_weights / w_sum

    combined = float(np.sum(norm_weights * v))
    combined_unc = float(1.0 / np.sqrt(w_sum))

    # Map back to full array
    full_weights = np.zeros_like(values)
    full_weights[mask] = norm_weights

    return combined, combined_unc, full_weights
