"""Tests for chime.channel_map — synthetic data, no network required."""

import numpy as np
import pytest

from bown_instruments.chime.channel_map import (
    channel_quality,
    compute_channel_map,
    _grade_bin,
    _compute_allan,
    _find_trust_regions,
    _compute_weights,
)


def make_synthetic_data(
    n_int: int = 200,
    n_wl: int = 500,
    wl_range: tuple = (0.6, 5.3),
    transit_depth_ppm: float = 20000,
    systematic_excess_by_band: dict | None = None,
    seed: int = 42,
):
    """Create synthetic JWST-like transit data.

    Parameters
    ----------
    systematic_excess_by_band : dict
        Maps (wl_min, wl_max) -> excess factor. Default: 1.0 everywhere.
    """
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(wl_range[0], wl_range[1], n_wl)

    # Base flux (flat, ~1e4 counts per pixel per integration)
    base_flux = 1e4 * np.ones(n_wl)

    # Transit: 20% of integrations in transit
    in_transit = np.zeros(n_int, dtype=bool)
    transit_start = n_int // 3
    transit_end = transit_start + n_int // 5
    in_transit[transit_start:transit_end] = True

    # Flux cube
    flux_cube = np.zeros((n_int, n_wl))
    for i in range(n_int):
        flux_cube[i] = base_flux.copy()

        # Add transit depth
        if in_transit[i]:
            flux_cube[i] *= 1 - transit_depth_ppm / 1e6

        # Add noise per wavelength
        for j in range(n_wl):
            noise_factor = 1.0
            if systematic_excess_by_band:
                for (wl_lo, wl_hi), excess in systematic_excess_by_band.items():
                    if wl_lo <= wavelength[j] <= wl_hi:
                        noise_factor = excess
                        break

            photon_noise = np.sqrt(base_flux[j])
            flux_cube[i, j] += rng.normal(0, photon_noise * noise_factor)

    return flux_cube, wavelength, in_transit


class TestGradeBin:
    def test_grade_a(self):
        assert _grade_bin(1.5, 1.0) == "A"

    def test_grade_a_requires_allan(self):
        # excess < 2 but allan too high -> B, not A
        assert _grade_bin(1.5, 2.0) == "B"

    def test_grade_b(self):
        assert _grade_bin(3.0, 1.0) == "B"

    def test_grade_c(self):
        assert _grade_bin(7.0, 1.0) == "C"

    def test_grade_d(self):
        assert _grade_bin(15.0, 1.0) == "D"


class TestAllan:
    def test_white_noise_averages_down(self):
        rng = np.random.default_rng(42)
        data = rng.normal(1000, 10, size=500)
        scatter = np.nanmedian(np.abs(data - np.nanmedian(data))) * 1.4826
        result = _compute_allan(data, scatter)
        # For white noise, ratios should be close to 1
        for r in result:
            assert r["ratio"] < 2.0, f"block {r['block_size']}: ratio {r['ratio']}"

    def test_correlated_noise_doesnt_average(self):
        # Perfectly correlated: all values the same + small noise
        rng = np.random.default_rng(42)
        drift = np.cumsum(rng.normal(0, 0.1, size=500))
        data = 1000 + drift
        scatter = np.nanmedian(np.abs(data - np.nanmedian(data))) * 1.4826
        result = _compute_allan(data, scatter)
        # Ratios should be elevated
        worst = max(r["ratio"] for r in result)
        assert worst > 1.5


class TestChannelQuality:
    def test_basic(self):
        flux_cube, wavelength, in_transit = make_synthetic_data()
        bins = channel_quality(flux_cube, wavelength, in_transit, n_bins=20)
        assert len(bins) > 0
        for b in bins:
            assert b.scatter_ppm > 0
            assert b.photon_noise_ppm > 0
            assert b.systematic_excess > 0
            assert b.grade in ("A", "B", "C", "D")

    def test_photon_limited_data(self):
        # Low-noise data should be mostly A-grade
        flux_cube, wavelength, in_transit = make_synthetic_data(
            systematic_excess_by_band=None,
            seed=42,
        )
        bins = channel_quality(flux_cube, wavelength, in_transit, n_bins=10)
        a_grades = sum(1 for b in bins if b.grade == "A")
        assert a_grades > 0

    def test_systematic_dominated(self):
        # High-excess data should be D-grade
        flux_cube, wavelength, in_transit = make_synthetic_data(
            systematic_excess_by_band={(0.6, 5.3): 50.0},
            seed=42,
        )
        bins = channel_quality(flux_cube, wavelength, in_transit, n_bins=10)
        d_grades = sum(1 for b in bins if b.grade == "D")
        assert d_grades > 0

    def test_too_few_oot(self):
        flux_cube = np.ones((10, 100))
        wavelength = np.linspace(1, 5, 100)
        in_transit = np.ones(10, dtype=bool)
        with pytest.raises(ValueError, match="at least 10"):
            channel_quality(flux_cube, wavelength, in_transit)


class TestChannelMap:
    def test_full_map(self):
        flux_cube, wavelength, in_transit = make_synthetic_data()
        cmap = compute_channel_map(flux_cube, wavelength, in_transit, n_bins=20)
        assert cmap.n_bins > 0
        assert "median_excess" in cmap.summary
        assert "n_photon_limited" in cmap.summary
        assert len(cmap.weights) == cmap.n_bins
        assert abs(cmap.weights.sum() - 1.0) < 1e-6

    def test_trust_regions(self):
        flux_cube, wavelength, in_transit = make_synthetic_data(seed=42)
        cmap = compute_channel_map(flux_cube, wavelength, in_transit, n_bins=20)
        # With photon-limited synthetic data, we should have trust regions
        assert isinstance(cmap.trust_regions, list)

    def test_to_dict(self):
        flux_cube, wavelength, in_transit = make_synthetic_data()
        cmap = compute_channel_map(flux_cube, wavelength, in_transit, n_bins=10)
        d = cmap.to_dict()
        assert "n_bins" in d
        assert "bins" in d
        assert "summary" in d
        assert "weights" in d

    def test_detectable_molecules(self):
        flux_cube, wavelength, in_transit = make_synthetic_data(seed=42)
        cmap = compute_channel_map(flux_cube, wavelength, in_transit, n_bins=30)
        detectable = cmap.summary.get("detectable_molecules", [])
        assert isinstance(detectable, list)


class TestWeights:
    def test_weights_sum_to_one(self):
        from bown_instruments.chime.channel_map import BinResult

        bins = [
            BinResult(1.0, (0.9, 1.1), 10, 1e4, 100, 50, 2.0, 20000, [], 1.0, "A"),
            BinResult(2.0, (1.9, 2.1), 10, 1e4, 500, 50, 10.0, 20000, [], 1.0, "D"),
            BinResult(3.0, (2.9, 3.1), 10, 1e4, 200, 50, 4.0, 20000, [], 1.0, "B"),
        ]
        weights = _compute_weights(bins)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert weights[1] == 0.0  # D-grade is zeroed
