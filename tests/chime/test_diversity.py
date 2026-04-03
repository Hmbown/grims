"""Tests for chime.diversity — synthetic data, no network required."""

import numpy as np
import pytest

from bown_instruments.chime.diversity import compute_diversity, DiversityResult


def make_diversity_data(
    n_int: int = 200,
    n_wl: int = 500,
    wl_range: tuple = (0.6, 5.3),
    transit_depth_ppm: float = 20000,
    bad_bands: list | None = None,
    seed: int = 42,
):
    """Create synthetic data with varying quality across wavelength.

    bad_bands: list of (wl_min, wl_max, excess_factor) for noisy regions.
    """
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(wl_range[0], wl_range[1], n_wl)

    base_flux = 1e4 * np.ones(n_wl)

    in_transit = np.zeros(n_int, dtype=bool)
    transit_start = n_int // 3
    transit_end = transit_start + n_int // 5
    in_transit[transit_start:transit_end] = True

    flux_cube = np.zeros((n_int, n_wl))
    for i in range(n_int):
        flux_cube[i] = base_flux.copy()
        if in_transit[i]:
            flux_cube[i] *= 1 - transit_depth_ppm / 1e6

        for j in range(n_wl):
            noise_factor = 1.0
            if bad_bands:
                for wl_lo, wl_hi, excess in bad_bands:
                    if wl_lo <= wavelength[j] <= wl_hi:
                        noise_factor = excess
                        break
            photon_noise = np.sqrt(base_flux[j])
            flux_cube[i, j] += rng.normal(0, photon_noise * noise_factor)

    return flux_cube, wavelength, in_transit


class TestDiversity:
    def test_basic(self):
        flux_cube, wavelength, in_transit = make_diversity_data()
        result = compute_diversity(flux_cube, wavelength, in_transit, n_subbands=10)
        assert isinstance(result, DiversityResult)
        assert result.n_subbands > 0
        assert result.diversity_depth_ppm != 0
        assert result.improvement_factor > 0

    def test_improvement_with_bad_bands(self):
        # With some bad bands, diversity should improve over naive
        flux_cube, wavelength, in_transit = make_diversity_data(
            bad_bands=[(0.6, 1.0, 30.0), (4.5, 5.3, 20.0)],
        )
        result = compute_diversity(flux_cube, wavelength, in_transit, n_subbands=15)
        assert result.improvement_factor > 1.0
        assert result.n_dropped >= 0

    def test_photon_limited_no_improvement(self):
        # If all bands are photon-limited, diversity ~= naive
        flux_cube, wavelength, in_transit = make_diversity_data(
            bad_bands=None,
            seed=42,
        )
        result = compute_diversity(flux_cube, wavelength, in_transit, n_subbands=10)
        # Improvement should be modest (close to 1)
        assert result.improvement_factor > 0.5

    def test_grades(self):
        flux_cube, wavelength, in_transit = make_diversity_data(
            bad_bands=[(0.6, 1.5, 25.0)],
        )
        result = compute_diversity(flux_cube, wavelength, in_transit, n_subbands=15)
        for sb in result.subbands:
            assert sb.grade in ("A", "B", "C", "D")
            if sb.grade == "D":
                assert sb.weight == 0.0
            if sb.grade == "A":
                assert sb.weight > 0.0

    def test_weights_normalized(self):
        flux_cube, wavelength, in_transit = make_diversity_data()
        result = compute_diversity(flux_cube, wavelength, in_transit, n_subbands=10)
        total = sum(sb.weight for sb in result.subbands)
        assert abs(total - 1.0) < 1e-6

    def test_all_dropped_raises(self):
        # All bands D-grade -> ValueError
        flux_cube, wavelength, in_transit = make_diversity_data(
            bad_bands=[(0.6, 5.3, 100.0)],
        )
        with pytest.raises(ValueError, match="dropped"):
            compute_diversity(
                flux_cube,
                wavelength,
                in_transit,
                n_subbands=5,
                excess_threshold_drop=1.0,  # everything is D
            )

    def test_output_arrays(self):
        flux_cube, wavelength, in_transit = make_diversity_data()
        result = compute_diversity(flux_cube, wavelength, in_transit, n_subbands=10)
        assert len(result.wl_centers) == result.n_subbands
        assert len(result.depths_ppm) == result.n_subbands
        assert len(result.depths_err_ppm) == result.n_subbands
        assert len(result.grades) == result.n_subbands
        assert len(result.noise_power_ppm) == result.n_subbands
