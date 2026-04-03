"""Tests for chime.transit_fit — synthetic data, no network required."""

import json

import numpy as np
import pytest

from bown_instruments.chime.transit_fit import (
    mandel_agol_flux,
    fit_transit_with_gp,
    fit_transmission_spectrum,
    TransitFitResult,
    _uniform_source_flux,
    _quad_ld_flux,
    _compute_z,
    _estimate_orbital_params,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
WASP39_EPHEMERIS = {
    "period_days": 4.055259,
    "t0_bjd": 2455342.9168,
    "duration_hours": 2.8056,
    "rp_rs": 0.1460,
    "expected_depth_ppm": 21316,
    "ref": "Faedi+ 2011",
}


# ---------------------------------------------------------------------------
# Helper: synthetic transit generator
# ---------------------------------------------------------------------------
def make_synthetic_transit(
    rp_rs=0.146,
    period=4.055259,
    duration_hours=2.8056,
    n_int=500,
    cadence_s=10.0,
    noise_ppm=1500,
    ramp_frac=0.01,
    u1=0.0,
    u2=0.0,
    seed=42,
):
    """Create synthetic transit data for testing.

    Returns: (times_bjd, flux, flux_err, ephemeris)

    - times centered on a transit epoch
    - flux = transit_model * (1 + ramp) + noise
    - ramp is linear from 0 to ramp_frac over the observation
    """
    rng = np.random.default_rng(seed)

    # Time array centered on transit
    t0 = 2455342.9168  # WASP-39b epoch
    total_time = n_int * cadence_s / 86400  # days
    times = t0 + np.linspace(-total_time / 2, total_time / 2, n_int)

    # Orbital params from duration
    a_rs = period * 24 / (np.pi * duration_hours)
    inc = np.pi / 2  # central transit

    # Import the function we're testing
    model = mandel_agol_flux(times, rp_rs, t0, period, a_rs, inc, u1=u1, u2=u2)

    # Add systematics (linear ramp)
    ramp = 1 + ramp_frac * np.linspace(0, 1, n_int)

    # Add noise
    noise_level = noise_ppm / 1e6
    noise = rng.normal(0, noise_level, n_int)

    flux = model * ramp + noise
    flux_err = np.full(n_int, noise_level)

    ephemeris = {
        "period_days": period,
        "t0_bjd": t0,
        "duration_hours": duration_hours,
        "rp_rs": rp_rs,
        "expected_depth_ppm": int(rp_rs**2 * 1e6),
    }

    return times, flux, flux_err, ephemeris


# ---------------------------------------------------------------------------
# Tests for _uniform_source_flux
# ---------------------------------------------------------------------------
class TestUniformSource:
    def test_no_transit_far_away(self):
        """When z > 1+p for all points, flux should be 1.0 everywhere."""
        p = 0.1
        z = np.array([2.0, 3.0, 5.0, 10.0])
        flux = _uniform_source_flux(z, p)
        np.testing.assert_allclose(flux, 1.0, atol=1e-12)

    def test_full_transit_centered(self):
        """When z=0, flux = 1 - p^2."""
        p = 0.1
        z = np.array([0.0])
        flux = _uniform_source_flux(z, p)
        expected = 1.0 - p**2
        assert flux[0] == pytest.approx(expected, abs=1e-10)

    def test_partial_overlap(self):
        """When z is between 1-p and 1+p, flux should be between 1-p^2 and 1.0."""
        p = 0.1
        # z values in the partial overlap zone (ingress/egress)
        z = np.array([0.95, 0.98, 1.02, 1.05])
        flux = _uniform_source_flux(z, p)
        for f in flux:
            assert 1.0 - p**2 <= f <= 1.0 + 1e-10

    def test_symmetry(self):
        """Flux at z should equal flux at the same |z| (symmetric around center)."""
        p = 0.1
        z_pos = np.array([0.0, 0.2, 0.5, 0.8, 0.95, 1.05])
        # z is always >= 0 (projected separation), so symmetry means
        # equal z gives equal flux — test via the transit model at +/- dt
        t0 = 0.0
        period = 4.0
        a_rs = 11.0
        inc = np.pi / 2
        dt = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        times_before = t0 - dt
        times_after = t0 + dt
        z_before = _compute_z(times_before, t0, period, a_rs, inc)
        z_after = _compute_z(times_after, t0, period, a_rs, inc)
        flux_before = _uniform_source_flux(z_before, p)
        flux_after = _uniform_source_flux(z_after, p)
        np.testing.assert_allclose(flux_before, flux_after, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests for _quad_ld_flux (quadratic limb darkening)
# ---------------------------------------------------------------------------
class TestLimbDarkening:
    def test_ld_reduces_to_uniform(self):
        """With u1=0, u2=0, LD transit should match uniform source flux."""
        p = 0.1
        z = np.linspace(0, 1.5, 200)
        flux_uniform = _uniform_source_flux(z, p)
        flux_ld = _quad_ld_flux(z, p, u1=0.0, u2=0.0)
        np.testing.assert_allclose(flux_ld, flux_uniform, atol=1e-4)

    def test_ld_deeper_at_center(self):
        """LD transit is deeper at center than uniform.

        Limb darkening makes the stellar center brighter relative to average,
        so a planet at disk center blocks more fractional flux.
        """
        p = 0.1
        u1, u2 = 0.3, 0.1  # typical LD coefficients
        z_center = np.array([0.0])
        flux_uniform = _uniform_source_flux(z_center, p)
        flux_ld = _quad_ld_flux(z_center, p, u1, u2)
        # LD flux should be lower (deeper transit) at center
        assert flux_ld[0] < flux_uniform[0]

    def test_ld_shallower_at_limb(self):
        """Near ingress/egress (z near 1-p), LD transit is shallower than uniform.

        The stellar limb is dimmer under LD, so a planet there blocks
        less fractional flux than the uniform-disk model predicts.
        """
        p = 0.1
        u1, u2 = 0.3, 0.1
        # z just inside full transit, near the limb
        z_limb = np.array([1.0 - p - 0.01])
        flux_uniform = _uniform_source_flux(z_limb, p)
        flux_ld = _quad_ld_flux(z_limb, p, u1, u2)
        # LD flux should be higher (shallower transit) at the limb
        assert flux_ld[0] > flux_uniform[0]


# ---------------------------------------------------------------------------
# Tests for mandel_agol_flux
# ---------------------------------------------------------------------------
class TestMandelAgolFlux:
    def test_out_of_transit_is_one(self):
        """Points well before/after transit should have flux = 1.0."""
        t0 = WASP39_EPHEMERIS["t0_bjd"]
        period = WASP39_EPHEMERIS["period_days"]
        a_rs = 11.55
        inc = np.radians(87.83)
        rp_rs = 0.146

        # Times far from any transit (quarter orbit away)
        times = np.array([t0 - period / 4, t0 + period / 4])
        flux = mandel_agol_flux(times, rp_rs, t0, period, a_rs, inc)
        np.testing.assert_allclose(flux, 1.0, atol=1e-10)

    def test_transit_depth_matches(self):
        """Mid-transit depth should be close to rp_rs^2 for a uniform source."""
        rp_rs = 0.146
        t0 = WASP39_EPHEMERIS["t0_bjd"]
        period = WASP39_EPHEMERIS["period_days"]
        a_rs = 11.55
        inc = np.pi / 2  # central transit

        times = np.array([t0])  # exact mid-transit
        flux = mandel_agol_flux(times, rp_rs, t0, period, a_rs, inc)
        expected_depth = rp_rs**2
        measured_depth = 1.0 - flux[0]
        assert measured_depth == pytest.approx(expected_depth, rel=0.05)

    def test_circular_orbit_symmetry(self):
        """Ingress and egress should be symmetric for ecc=0."""
        rp_rs = 0.146
        t0 = WASP39_EPHEMERIS["t0_bjd"]
        period = WASP39_EPHEMERIS["period_days"]
        a_rs = 11.55
        inc = np.pi / 2

        # Create times symmetric about t0
        dt = np.linspace(0.001, 0.08, 50)  # days
        times_before = t0 - dt
        times_after = t0 + dt

        flux_before = mandel_agol_flux(
            times_before, rp_rs, t0, period, a_rs, inc, ecc=0.0,
        )
        flux_after = mandel_agol_flux(
            times_after, rp_rs, t0, period, a_rs, inc, ecc=0.0,
        )
        np.testing.assert_allclose(flux_before, flux_after, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for _estimate_orbital_params
# ---------------------------------------------------------------------------
class TestEstimateOrbitalParams:
    def test_wasp39_params(self):
        """For WASP-39 ephemeris, a_rs ~ 11.5 and inc ~ pi/2."""
        a_rs, inc_rad = _estimate_orbital_params(WASP39_EPHEMERIS)
        # a_rs should be approximately 11.0-12.0
        assert a_rs == pytest.approx(11.0, rel=0.1)
        # inc should be close to pi/2 (central transit assumed)
        assert inc_rad == pytest.approx(np.pi / 2, abs=0.1)


# ---------------------------------------------------------------------------
# Tests for fit_transit_with_gp
# ---------------------------------------------------------------------------
class TestFitTransitWithGP:
    def test_recover_rp_rs(self):
        """Inject transit with known rp_rs, recover within tolerance.

        Uses low noise (500 ppm) and seed=42 for reproducibility.
        Allows 20% tolerance since GP + grid search is not MCMC-precise.
        """
        times, flux, flux_err, ephemeris = make_synthetic_transit(
            rp_rs=0.146,
            noise_ppm=500,
            ramp_frac=0.01,
            seed=42,
        )
        result = fit_transit_with_gp(times, flux, flux_err, ephemeris)
        assert result["rp_rs"] == pytest.approx(0.146, rel=0.20)

    def test_fit_returns_expected_keys(self):
        """Result dict should contain all expected keys."""
        times, flux, flux_err, ephemeris = make_synthetic_transit(
            noise_ppm=500,
            seed=42,
        )
        result = fit_transit_with_gp(times, flux, flux_err, ephemeris)
        expected_keys = [
            "rp_rs",
            "rp_rs_err",
            "depth_ppm",
            "depth_err_ppm",
            "gp_params",
            "model_flux",
            "systematics",
            "residuals",
            "chi2_reduced",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Sanity-check types
        assert isinstance(result["rp_rs"], float)
        assert isinstance(result["rp_rs_err"], float)
        assert isinstance(result["depth_ppm"], float)
        assert result["depth_ppm"] > 0
        assert isinstance(result["model_flux"], np.ndarray)
        assert len(result["model_flux"]) == len(times)
        assert isinstance(result["residuals"], np.ndarray)
        assert len(result["residuals"]) == len(times)
        assert result["chi2_reduced"] > 0


# ---------------------------------------------------------------------------
# Tests for TransitFitResult
# ---------------------------------------------------------------------------
class TestTransitFitResult:
    @pytest.fixture
    def sample_result(self):
        """Build a minimal TransitFitResult for serialization tests."""
        n_bins = 10
        return TransitFitResult(
            target="WASP-39",
            wl_centers=np.linspace(1.0, 5.0, n_bins),
            rp_rs=np.full(n_bins, 0.146),
            rp_rs_err=np.full(n_bins, 0.005),
            depth_ppm=np.full(n_bins, 21316.0),
            depth_err_ppm=np.full(n_bins, 1500.0),
            grades=["A"] * 8 + ["B", "C"],
            weights=np.ones(n_bins) / n_bins,
            combined_depth_ppm=21316.0,
            combined_depth_err_ppm=500.0,
            naive_depth_ppm=21000.0,
            naive_depth_err_ppm=800.0,
            improvement_factor=1.6,
            chi2_reduced=np.ones(n_bins),
            n_bins=n_bins,
            n_dropped=0,
            ephemeris=WASP39_EPHEMERIS.copy(),
        )

    def test_to_dict_serializable(self, sample_result):
        """to_dict() output should be JSON-serializable."""
        d = sample_result.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        # Round-trip check
        parsed = json.loads(serialized)
        assert parsed["target"] == "WASP-39"
        assert parsed["n_bins"] == 10

    def test_to_ecsv_rows_length(self, sample_result):
        """to_ecsv_rows() should return n_bins rows."""
        rows = sample_result.to_ecsv_rows()
        assert len(rows) == sample_result.n_bins
        # Each row should be a dict with wavelength and depth info
        for row in rows:
            assert isinstance(row, dict)
