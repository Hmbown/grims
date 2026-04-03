"""Tests for chime.ephemeris — no network required."""

import pytest

from bown_instruments.chime.ephemeris import get_ephemeris, list_targets, EPHEMERIDES


class TestEphemeris:
    def test_list_targets(self):
        targets = list_targets()
        assert len(targets) > 0
        assert "WASP-39" in targets
        assert "TRAPPIST-1" in targets

    def test_exact_match(self):
        eph = get_ephemeris("WASP-39")
        assert eph["period_days"] > 0
        assert eph["t0_bjd"] > 2400000
        assert eph["duration_hours"] > 0
        assert eph["rp_rs"] > 0
        assert eph["expected_depth_ppm"] > 0

    def test_case_insensitive(self):
        eph1 = get_ephemeris("WASP-39")
        eph2 = get_ephemeris("wasp-39")
        assert eph1 == eph2

    def test_with_planet_suffix(self):
        eph = get_ephemeris("WASP-39b")
        assert eph["period_days"] > 0

    def test_trappist_1_planets(self):
        for planet in [
            "TRAPPIST-1b",
            "TRAPPIST-1c",
            "TRAPPIST-1d",
            "TRAPPIST-1e",
            "TRAPPIST-1f",
            "TRAPPIST-1g",
        ]:
            eph = get_ephemeris(planet)
            assert eph["period_days"] > 0
            assert eph["expected_depth_ppm"] > 0

    def test_missing_target_raises(self):
        with pytest.raises(KeyError, match="No ephemeris"):
            get_ephemeris("NONEXISTENT-999")

    def test_all_ephemerides_have_required_keys(self):
        required = ["period_days", "t0_bjd", "duration_hours", "rp_rs", "expected_depth_ppm", "ref"]
        for name, eph in EPHEMERIDES.items():
            for key in required:
                assert key in eph, f"{name} missing key: {key}"
