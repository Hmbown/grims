"""Sub-band diversity combining for JWST transmission spectra.

Splits the wavelength band into sub-bands, grades each by empirical noise
quality, and selectively combines using quality-based weights. The
analytical approach is inspired by Bown's diversity reception
(US 1,747,221, 1930): weight channels by measured quality rather than
treating all channels equally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubBand:
    """Result for a single spectral sub-band."""

    index: int
    wl_center: float
    wl_range: tuple[float, float]
    n_pixels: int
    scatter_ppm: float
    photon_noise_ppm: float
    systematic_excess: float
    grade: str
    depth_ppm: float
    depth_err_ppm: float
    weight: float


@dataclass
class DiversityResult:
    """Result of diversity-combined transmission spectrum."""

    diversity_depth_ppm: float
    diversity_err_ppm: float
    naive_depth_ppm: float
    naive_err_ppm: float
    improvement_factor: float
    subbands: list[SubBand]
    n_subbands: int
    n_dropped: int
    n_degraded: int
    wl_centers: np.ndarray
    depths_ppm: np.ndarray
    depths_err_ppm: np.ndarray
    grades: list[str]
    noise_power_ppm: np.ndarray


def compute_diversity(
    flux_cube: np.ndarray,
    wavelength: np.ndarray,
    in_transit_mask: np.ndarray,
    n_subbands: int = 15,
    excess_threshold_drop: float = 10.0,
    excess_threshold_deweight: float = 5.0,
    log_spaced: bool = True,
) -> DiversityResult:
    """Compute transmission spectrum using sub-band diversity combining.

    Parameters
    ----------
    flux_cube : ndarray, shape (n_integrations, n_wavelengths)
        Per-integration extracted spectra.
    wavelength : ndarray, shape (n_wavelengths,)
        Wavelength grid in microns.
    in_transit_mask : ndarray, shape (n_integrations,), dtype bool
        True for in-transit integrations.
    n_subbands : int
        Number of spectral sub-bands.
    excess_threshold_drop : float
        Systematic excess above which sub-bands are dropped (grade D).
    excess_threshold_deweight : float
        Systematic excess above which sub-bands are deweighted (grade C).
    log_spaced : bool
        Use log-spaced sub-band edges (better for NIRSpec PRISM).

    Returns
    -------
    DiversityResult

    Notes
    -----
    The combining weights are w_k = 1 / scatter_k^2 for A/B grades,
    soft rolloff for C grade, and zero for D grade.
    The combined depth is sum(w_k * depth_k) / sum(w_k).
    """
    n_int, n_wl = flux_cube.shape
    oot_mask = ~in_transit_mask

    wl_valid = np.isfinite(wavelength) & (wavelength > 0)
    wl_min = wavelength[wl_valid].min()
    wl_max = wavelength[wl_valid].max()

    if log_spaced and wl_min > 0:
        edges = np.geomspace(wl_min, wl_max * 1.001, n_subbands + 1)
    else:
        edges = np.linspace(wl_min, wl_max * 1.001, n_subbands + 1)

    subbands = []

    for k in range(n_subbands):
        mask = wl_valid & (wavelength >= edges[k]) & (wavelength < edges[k + 1])
        n_pix = int(np.sum(mask))
        if n_pix < 2:
            continue

        wl_flux = np.nansum(flux_cube[:, mask], axis=1)
        oot_flux = wl_flux[oot_mask]
        if len(oot_flux) < 5:
            continue

        oot_median = np.nanmedian(oot_flux)
        if oot_median <= 0:
            continue

        # Robust scatter
        scatter = np.nanmedian(np.abs(oot_flux - oot_median)) * 1.4826
        scatter_ppm = scatter / oot_median * 1e6

        # Photon noise
        photon_noise = np.sqrt(oot_median)
        photon_ppm = photon_noise / oot_median * 1e6

        excess = scatter_ppm / photon_ppm if photon_ppm > 0 else float("inf")

        # Grade
        if excess < 2:
            grade = "A"
        elif excess < excess_threshold_deweight:
            grade = "B"
        elif excess < excess_threshold_drop:
            grade = "C"
        else:
            grade = "D"

        # Transit depth
        in_flux = np.nanmean(wl_flux[in_transit_mask])
        out_flux = np.nanmean(oot_flux)
        depth = 1.0 - in_flux / out_flux if out_flux > 0 else 0.0
        depth_err = (
            scatter / out_flux / np.sqrt(np.sum(in_transit_mask)) if out_flux > 0 else float("inf")
        )

        # Weight
        if grade == "D":
            weight = 0.0
        elif grade == "C":
            base = 1.0 / scatter_ppm**2 if scatter_ppm > 0 else 0.0
            rolloff = max(
                0,
                1.0
                - (excess - excess_threshold_deweight)
                / (excess_threshold_drop - excess_threshold_deweight),
            )
            weight = base * rolloff
        else:
            weight = 1.0 / scatter_ppm**2 if scatter_ppm > 0 else 0.0

        subbands.append(
            SubBand(
                index=k,
                wl_center=float(np.mean(wavelength[mask])),
                wl_range=(float(edges[k]), float(edges[k + 1])),
                n_pixels=n_pix,
                scatter_ppm=float(scatter_ppm),
                photon_noise_ppm=float(photon_ppm),
                systematic_excess=float(excess),
                grade=grade,
                depth_ppm=float(depth * 1e6),
                depth_err_ppm=float(depth_err * 1e6),
                weight=float(weight),
            )
        )

    if not subbands:
        raise ValueError("No valid sub-bands found")

    # Normalize weights
    total_weight = sum(sb.weight for sb in subbands)
    if total_weight > 0:
        for sb in subbands:
            sb.weight /= total_weight

    weights = np.array([sb.weight for sb in subbands])
    depths = np.array([sb.depth_ppm for sb in subbands])
    depth_errs = np.array([sb.depth_err_ppm for sb in subbands])

    active = weights > 0
    if not np.any(active):
        raise ValueError("All sub-bands dropped (all D-grade)")

    diversity_depth = float(np.sum(weights * depths))
    diversity_err = float(np.sqrt(np.sum(weights**2 * depth_errs**2)))

    # Naive combination (equal weight)
    n_valid = len(subbands)
    naive_depth = float(np.mean(depths))
    naive_err = float(np.sqrt(np.sum(depth_errs**2)) / n_valid)

    improvement = naive_err / diversity_err if diversity_err > 0 else 1.0

    return DiversityResult(
        diversity_depth_ppm=diversity_depth,
        diversity_err_ppm=diversity_err,
        naive_depth_ppm=naive_depth,
        naive_err_ppm=naive_err,
        improvement_factor=improvement,
        subbands=subbands,
        n_subbands=len(subbands),
        n_dropped=sum(1 for sb in subbands if sb.grade == "D"),
        n_degraded=sum(1 for sb in subbands if sb.grade == "C"),
        wl_centers=np.array([sb.wl_center for sb in subbands]),
        depths_ppm=depths,
        depths_err_ppm=depth_errs,
        grades=[sb.grade for sb in subbands],
        noise_power_ppm=np.array([sb.scatter_ppm for sb in subbands]),
    )
