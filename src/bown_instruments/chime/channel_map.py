"""Channel quality map — per-wavelength empirical noise diagnostics.

The core measurement of chime. For each wavelength bin, computes:
  1. Empirical scatter (MAD of out-of-transit flux)
  2. Photon noise limit (sqrt(flux)/flux)
  3. Systematic excess (ratio of empirical to photon noise)
  4. Allan deviation diagnostic (does noise average down as sqrt(n)?)
  5. Quality grade (A/B/C/D)

Analytical approach inspired by Bown's channel-quality measurement
methods (US 1,794,393, 1931): measure the channel, not just the signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ #
# Molecular absorption band definitions
# ------------------------------------------------------------------ #

MOLECULAR_BANDS: list[tuple[float, float, str]] = [
    (0.580, 0.600, "Na I"),
    (0.760, 0.780, "K I"),
    (1.10, 1.20, "H2O 1.1um"),
    (1.30, 1.50, "H2O 1.4um"),
    (1.75, 2.05, "H2O 1.9um"),
    (2.20, 2.40, "CH4 2.3um"),
    (2.50, 2.90, "H2O 2.7um"),
    (3.00, 3.15, "CH4 3.0um"),
    (3.20, 3.50, "CH4 3.3um"),
    (4.15, 4.45, "CO2 4.3um"),
    (4.50, 4.80, "CO 4.6um"),
    (5.50, 6.50, "H2O 6.0um"),
    (9.00, 10.0, "NH3 9.5um"),
]


@dataclass
class BinResult:
    """Channel quality result for a single wavelength bin."""

    wl_center: float
    wl_range: tuple[float, float]
    n_pixels: int
    median_flux: float
    scatter_ppm: float
    photon_noise_ppm: float
    systematic_excess: float
    depth_ppm: float
    allan_ratios: list[dict]
    allan_worst_ratio: float
    grade: str


@dataclass
class ChannelMap:
    """Full channel quality map result."""

    bins: list[BinResult]
    n_bins: int
    summary: dict
    trust_regions: list[dict]
    weights: np.ndarray
    wl_centers: np.ndarray

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "n_bins": self.n_bins,
            "bins": [
                {
                    "wl_center": b.wl_center,
                    "wl_range": list(b.wl_range),
                    "n_pixels": b.n_pixels,
                    "median_flux": b.median_flux,
                    "scatter_ppm": b.scatter_ppm,
                    "photon_noise_ppm": b.photon_noise_ppm,
                    "systematic_excess": b.systematic_excess,
                    "depth_ppm": b.depth_ppm,
                    "allan_worst_ratio": b.allan_worst_ratio,
                    "grade": b.grade,
                }
                for b in self.bins
            ],
            "summary": self.summary,
            "trust_regions": self.trust_regions,
            "weights": self.weights.tolist(),
        }


def _grade_bin(excess: float, allan_ratio: float) -> str:
    """Grade a wavelength bin by systematic excess and Allan ratio.

    A: excess < 2 AND allan < 1.5  (photon-limited, white noise)
    B: excess < 5                   (moderate systematics)
    C: excess < 10                   (significant systematics)
    D: excess >= 10                  (systematic-dominated, untrustworthy)
    """
    if excess < 2 and allan_ratio < 1.5:
        return "A"
    elif excess < 5:
        return "B"
    elif excess < 10:
        return "C"
    else:
        return "D"


def _compute_allan(oot_flux: np.ndarray, scatter_single: float) -> list[dict]:
    """Compute Allan deviation diagnostic.

    Tests whether noise averages down as sqrt(n) (white noise) or
    faster/slower (correlated or non-stationary).

    Parameters
    ----------
    oot_flux : array
        Out-of-transit flux values.
    scatter_single : float
        Scatter at block_size=1 (MAD-based sigma).

    Returns
    -------
    list of dicts with block_size, expected_reduction, actual_reduction, ratio.
    """
    allan = []
    for block_size in [1, 2, 5, 10, 20]:
        if block_size >= len(oot_flux) // 3:
            break
        n_blocks = len(oot_flux) // block_size
        block_means = np.array(
            [np.nanmean(oot_flux[j * block_size : (j + 1) * block_size]) for j in range(n_blocks)]
        )
        block_scatter = np.nanmedian(np.abs(block_means - np.nanmedian(block_means))) * 1.4826
        expected_reduction = 1.0 / np.sqrt(block_size)
        actual_reduction = block_scatter / scatter_single if scatter_single > 0 else 1.0
        ratio = actual_reduction / expected_reduction if expected_reduction > 0 else float("inf")
        allan.append(
            {
                "block_size": block_size,
                "expected_reduction": float(expected_reduction),
                "actual_reduction": float(actual_reduction),
                "ratio": float(ratio),
            }
        )
    return allan


def channel_quality(
    flux_cube: np.ndarray,
    wavelength: np.ndarray,
    in_transit_mask: np.ndarray,
    n_bins: int = 50,
    flux_error_cube: np.ndarray | None = None,
) -> list[BinResult]:
    """Compute per-wavelength channel quality diagnostics.

    Parameters
    ----------
    flux_cube : ndarray, shape (n_integrations, n_wavelengths)
        Per-integration extracted spectra.
    wavelength : ndarray, shape (n_wavelengths,)
        Wavelength grid in microns.
    in_transit_mask : ndarray, shape (n_integrations,), dtype bool
        True for in-transit integrations.
    n_bins : int
        Number of wavelength bins (log-spaced).
    flux_error_cube : ndarray, optional
        Per-integration flux errors (from pipeline). When provided, used as
        the noise reference instead of sqrt(flux). This gives correct
        systematic excess for calibrated data where flux is not in photon
        counts.

    Returns
    -------
    list[BinResult], one per valid wavelength bin.
    """
    oot_mask = ~in_transit_mask

    if np.sum(oot_mask) < 10:
        raise ValueError("Need at least 10 out-of-transit integrations")

    wl_valid = np.isfinite(wavelength) & (wavelength > 0)
    wl_min, wl_max = wavelength[wl_valid].min(), wavelength[wl_valid].max()
    edges = np.geomspace(wl_min, wl_max * 1.001, n_bins + 1)

    bins = []
    for i in range(n_bins):
        mask = wl_valid & (wavelength >= edges[i]) & (wavelength < edges[i + 1])
        n_pix = int(np.sum(mask))
        if n_pix < 1:
            continue

        # White light in this bin
        wl_flux = np.nansum(flux_cube[:, mask], axis=1)
        oot_flux = wl_flux[oot_mask]

        if len(oot_flux) < 10 or np.nanmedian(oot_flux) <= 0:
            continue

        median_flux = np.nanmedian(oot_flux)

        # Empirical scatter (MAD-based robust sigma)
        scatter = np.nanmedian(np.abs(oot_flux - median_flux)) * 1.4826
        scatter_ppm = scatter / median_flux * 1e6

        # Noise reference: use pipeline flux errors if available,
        # otherwise fall back to sqrt(flux) for count data
        if flux_error_cube is not None:
            # Sum errors in quadrature across pixels in bin
            wl_err = np.sqrt(np.nansum(flux_error_cube[:, mask] ** 2, axis=1))
            oot_err = wl_err[oot_mask]
            photon_noise = np.nanmedian(oot_err) if len(oot_err) > 0 else scatter
        else:
            photon_noise = np.sqrt(median_flux)

        photon_ppm = photon_noise / median_flux * 1e6 if median_flux > 0 else 0

        # Systematic excess: how much worse is scatter than the noise floor?
        excess = scatter / photon_noise if photon_noise > 0 else float("inf")

        # Transit depth in this bin
        if np.sum(in_transit_mask) > 0:
            in_flux = np.nanmedian(wl_flux[in_transit_mask])
            depth_ppm = (1.0 - in_flux / median_flux) * 1e6
        else:
            depth_ppm = 0.0

        # Allan deviation diagnostic
        allan = _compute_allan(oot_flux, scatter)
        worst_allan = max((ar["ratio"] for ar in allan), default=1.0)

        grade = _grade_bin(excess, worst_allan)

        bins.append(
            BinResult(
                wl_center=float(np.mean(wavelength[mask])),
                wl_range=(float(edges[i]), float(edges[i + 1])),
                n_pixels=n_pix,
                median_flux=float(median_flux),
                scatter_ppm=float(scatter_ppm),
                photon_noise_ppm=float(photon_ppm),
                systematic_excess=float(excess),
                depth_ppm=float(depth_ppm),
                allan_ratios=allan,
                allan_worst_ratio=float(worst_allan),
                grade=grade,
            )
        )

    return bins


def _find_trust_regions(bins: list[BinResult]) -> list[dict]:
    """Identify contiguous wavelength regions with low systematic excess.

    Trust region criteria:
      - scatter < 2 * photon noise (excess < 2)
      - Allan ratio < 1.5 (noise averages down properly)

    Returns list of dicts with wl_min, wl_max, mean_scatter_ppm, n_bins.
    """
    trust_bins = [b for b in bins if b.systematic_excess < 2 and b.allan_worst_ratio < 1.5]

    if not trust_bins:
        return []

    # Group into contiguous regions
    regions = []
    current = {
        "bins": [trust_bins[0]],
        "wl_min": trust_bins[0].wl_range[0],
        "wl_max": trust_bins[0].wl_range[1],
    }

    for b in trust_bins[1:]:
        if b.wl_range[0] <= current["wl_max"] * 1.1:
            current["bins"].append(b)
            current["wl_max"] = b.wl_range[1]
        else:
            regions.append(current)
            current = {
                "bins": [b],
                "wl_min": b.wl_range[0],
                "wl_max": b.wl_range[1],
            }
    regions.append(current)

    return [
        {
            "wl_min": round(r["wl_min"], 4),
            "wl_max": round(r["wl_max"], 4),
            "n_bins": len(r["bins"]),
            "mean_scatter_ppm": round(float(np.mean([b.scatter_ppm for b in r["bins"]])), 1),
            "mean_excess": round(float(np.mean([b.systematic_excess for b in r["bins"]])), 2),
        }
        for r in regions
        if len(r["bins"]) >= 2  # require at least 2 contiguous bins
    ]


def _compute_weights(bins: list[BinResult]) -> np.ndarray:
    """Compute diversity combining weights from channel quality.

    Weights: w_k = 1/scatter_k^2 for A/B grades, soft rolloff for C, zero for D.
    Normalized so they sum to 1.
    """
    weights = np.zeros(len(bins))
    for i, b in enumerate(bins):
        if b.grade == "D":
            weights[i] = 0.0
        elif b.grade == "C":
            base = 1.0 / b.scatter_ppm**2 if b.scatter_ppm > 0 else 0.0
            rolloff = max(0, 1.0 - (b.systematic_excess - 5) / 5)
            weights[i] = base * rolloff
        else:
            weights[i] = 1.0 / b.scatter_ppm**2 if b.scatter_ppm > 0 else 0.0

    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def _detectable_molecules(
    trust_regions: list[dict],
    noise_ppm: float,
) -> list[dict]:
    """Estimate which molecules are detectable in trust regions.

    For each trust region, computes 3-sigma detection limit and compares
    against typical feature depths for common molecules.
    """
    # Typical peak feature depths in ppm (order-of-magnitude)
    molecule_features = {
        "H2O 1.4um": (1.30, 1.50, 200),
        "H2O 1.9um": (1.75, 2.05, 300),
        "H2O 2.7um": (2.50, 2.90, 500),
        "CO2 4.3um": (4.15, 4.45, 400),
        "CO 4.6um": (4.50, 4.80, 150),
        "CH4 3.3um": (3.20, 3.50, 200),
        "NH3 9.5um": (9.00, 10.0, 100),
        "Na I": (0.580, 0.600, 50),
        "K I": (0.760, 0.780, 80),
    }

    detectable = []
    for region in trust_regions:
        wl_min, wl_max = region["wl_min"], region["wl_max"]
        scatter = region["mean_scatter_ppm"]
        limit_3sigma = 3 * scatter

        for mol, (mol_wl_min, mol_wl_max, expected_depth) in molecule_features.items():
            # Check if molecular band overlaps with trust region
            overlap_min = max(wl_min, mol_wl_min)
            overlap_max = min(wl_max, mol_wl_max)
            if overlap_max > overlap_min:
                detectable.append(
                    {
                        "molecule": mol,
                        "trust_region": f"{wl_min:.2f}-{wl_max:.2f} um",
                        "expected_depth_ppm": expected_depth,
                        "detection_limit_3sigma_ppm": round(limit_3sigma, 1),
                        "detectable": expected_depth > limit_3sigma,
                    }
                )

    return detectable


def compute_channel_map(
    flux_cube: np.ndarray,
    wavelength: np.ndarray,
    in_transit_mask: np.ndarray,
    n_bins: int = 50,
    flux_error_cube: np.ndarray | None = None,
) -> ChannelMap:
    """Compute full channel quality map with trust regions and weights.

    This is the main entry point. Computes per-wavelength quality diagnostics,
    identifies trustworthy wavelength regions, and produces combining weights.

    Parameters
    ----------
    flux_cube : ndarray, shape (n_integrations, n_wavelengths)
    wavelength : ndarray, shape (n_wavelengths,) in microns
    in_transit_mask : ndarray, shape (n_integrations,), dtype bool
    n_bins : int
    flux_error_cube : ndarray, optional
        Per-integration flux errors. If provided, used as noise floor
        reference for computing systematic excess.

    Returns
    -------
    ChannelMap
    """
    bins = channel_quality(
        flux_cube, wavelength, in_transit_mask, n_bins, flux_error_cube=flux_error_cube
    )

    if not bins:
        return ChannelMap(
            bins=[],
            n_bins=0,
            summary={"error": "no valid wavelength bins"},
            trust_regions=[],
            weights=np.array([]),
            wl_centers=np.array([]),
        )

    excesses = [b.systematic_excess for b in bins]
    scatters = [b.scatter_ppm for b in bins]
    allans = [b.allan_worst_ratio for b in bins]
    grades = [b.grade for b in bins]

    trust_regions = _find_trust_regions(bins)
    weights = _compute_weights(bins)
    wl_centers = np.array([b.wl_center for b in bins])

    detectable = _detectable_molecules(trust_regions, float(np.median(scatters)))

    summary = {
        "median_excess": float(np.median(excesses)),
        "max_excess": float(np.max(excesses)),
        "min_scatter_ppm": float(np.min(scatters)),
        "max_scatter_ppm": float(np.max(scatters)),
        "scatter_ratio": float(np.max(scatters) / np.min(scatters)) if np.min(scatters) > 0 else 0,
        "n_photon_limited": grades.count("A"),
        "n_moderate": grades.count("B"),
        "n_degraded": grades.count("C"),
        "n_systematic_dominated": grades.count("D"),
        "median_allan_ratio": float(np.median(allans)),
        "max_allan_ratio": float(np.max(allans)),
        "n_in_transit": int(np.sum(in_transit_mask)),
        "n_out_of_transit": int(np.sum(~in_transit_mask)),
        "n_trust_regions": len(trust_regions),
        "detectable_molecules": detectable,
    }

    return ChannelMap(
        bins=bins,
        n_bins=len(bins),
        summary=summary,
        trust_regions=trust_regions,
        weights=weights,
        wl_centers=wl_centers,
    )
