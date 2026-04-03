"""Extract per-integration spectra and light curves from JWST x1dints FITS files.

Parses the EXTRACT1D extension from Stage 2 x1dints products to produce
flux cubes (n_integrations, n_wavelengths) and timestamps for transit analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass
class TransitData:
    """Extracted transit spectroscopy data from an x1dints file."""

    wavelength: np.ndarray  # (n_wl,) in microns
    flux_cube: np.ndarray  # (n_int, n_wl) in erg/s/cm2/Å
    flux_error_cube: np.ndarray  # (n_int, n_wl)
    times_mjd: np.ndarray  # (n_int,) mid-integration MJD
    n_integrations: int
    n_wavelengths: int
    header: dict = field(default_factory=dict)


def extract_transit_data(filepath: str) -> TransitData | None:
    """Extract per-integration spectra from an x1dints FITS file.

    Parameters
    ----------
    filepath : str
        Path to x1dints FITS file.

    Returns
    -------
    TransitData or None if EXTRACT1D extension is missing.
    """
    with fits.open(filepath) as hdul:
        # Timestamps from INT_TIMES extension
        times_mjd = None
        ext_names = [h.name for h in hdul]

        if "INT_TIMES" in ext_names:
            int_times = hdul["INT_TIMES"].data
            times_mjd = np.array(int_times["int_mid_MJD_UTC"], dtype=np.float64)

        if "EXTRACT1D" not in ext_names:
            return None

        data = hdul["EXTRACT1D"].data
        wl = np.array(data["WAVELENGTH"], dtype=np.float64)
        fl = np.array(data["FLUX"], dtype=np.float64)

        fe = None
        if "FLUX_ERROR" in data.dtype.names:
            fe = np.array(data["FLUX_ERROR"], dtype=np.float64)

        if fl.ndim != 2:
            return None

        n_int, n_wl = fl.shape

        # Use first integration's wavelength as reference grid
        wl_ref = wl[0] if wl.ndim > 1 else wl

        if times_mjd is not None and len(times_mjd) != n_int:
            # Combined x1dints: INT_TIMES has entries for all nod/segment
            # positions but EXTRACT1D is stacked. Subsample evenly.
            n_times = len(times_mjd)
            if n_times > n_int and n_times % n_int == 0:
                # Exact multiple — take first segment's timestamps
                times_mjd = times_mjd[:n_int]
            elif n_times > n_int:
                # Non-exact — interpolate evenly
                idx = np.linspace(0, n_times - 1, n_int, dtype=int)
                times_mjd = times_mjd[idx]
            else:
                times_mjd = None  # too few — fall back below

        if times_mjd is None:
            # Reconstruct from header if possible
            h0 = hdul[0].header
            t_start = h0.get("MJD-BEG") or h0.get("TSTART")
            t_end = h0.get("MJD-END") or h0.get("TSTOP")
            if t_start is not None and t_end is not None:
                times_mjd = np.linspace(float(t_start), float(t_end), n_int)
            else:
                times_mjd = np.arange(n_int, dtype=np.float64)

        # Extract key header fields
        header = {}
        h0 = hdul[0].header
        for key in ["TARGNAME", "INSTRUME", "FILTER", "GRATING", "DATE-OBS", "PROGRAM", "OBSERVTN"]:
            if key in h0:
                header[key] = str(h0[key])

        return TransitData(
            wavelength=wl_ref,
            flux_cube=fl,
            flux_error_cube=fe if fe is not None else np.zeros_like(fl),
            times_mjd=times_mjd,
            n_integrations=n_int,
            n_wavelengths=n_wl,
            header=header,
        )


def compute_white_light_curve(
    transit_data: TransitData,
    wl_min: float = 0.6,
    wl_max: float = 5.3,
) -> dict:
    """Compute broadband (white) light curve from transit data.

    Sums flux across all wavelengths for each integration, normalizes by
    out-of-transit baseline, and identifies the transit.

    Returns
    -------
    dict with keys: times_mjd, flux, flux_error, baseline, transit_depth,
                    transit_depth_ppm, transit_center_idx, oot_scatter_ppm
    """
    wl = transit_data.wavelength
    fl = transit_data.flux_cube
    fe = transit_data.flux_error_cube
    times = transit_data.times_mjd

    # Wavelength mask
    wl_mask = (wl >= wl_min) & (wl <= wl_max) & np.isfinite(wl)
    if np.sum(wl_mask) < 5:
        return {"error": "too few valid wavelength points"}

    white_flux = np.nansum(fl[:, wl_mask], axis=1)
    white_err = np.sqrt(np.nansum(fe[:, wl_mask] ** 2, axis=1))

    # Normalize by baseline (first and last 20%)
    n = len(times)
    n_edge = max(1, n // 5)
    baseline_idx = np.concatenate([np.arange(n_edge), np.arange(n - n_edge, n)])
    baseline = np.nanmedian(white_flux[baseline_idx])

    if baseline <= 0:
        return {"error": "zero baseline flux"}

    norm_flux = white_flux / baseline
    norm_err = white_err / baseline

    # Find transit: lowest point in middle 60%
    mid_start = n // 5
    mid_end = n - n // 5
    mid_flux = norm_flux[mid_start:mid_end]
    transit_idx = mid_start + np.nanargmin(mid_flux)
    transit_depth = 1.0 - norm_flux[transit_idx]

    oot_std = np.nanstd(norm_flux[baseline_idx])

    return {
        "times_mjd": times,
        "flux": norm_flux,
        "flux_error": norm_err,
        "baseline": float(baseline),
        "transit_depth": float(transit_depth),
        "transit_depth_ppm": float(transit_depth * 1e6),
        "transit_center_idx": int(transit_idx),
        "oot_scatter_ppm": float(oot_std * 1e6),
        "n_integrations": n,
    }
