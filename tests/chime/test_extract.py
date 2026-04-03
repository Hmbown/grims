"""Tests for chime.extract — synthetic FITS data, no network required."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from bown_instruments.chime.extract import extract_transit_data, compute_white_light_curve, TransitData


def make_synthetic_x1dints(
    filepath: str,
    n_int: int = 100,
    n_wl: int = 300,
    wl_range: tuple = (0.6, 5.3),
    transit_depth_ppm: float = 20000,
    seed: int = 42,
):
    """Write a synthetic x1dints FITS file."""
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(wl_range[0], wl_range[1], n_wl)
    base_flux = 1e4 * np.ones(n_wl)

    # Create flux cube
    flux = np.zeros((n_int, n_wl))
    in_transit_start = n_int // 3
    in_transit_end = in_transit_start + n_int // 5

    for i in range(n_int):
        flux[i] = base_flux.copy()
        if in_transit_start <= i < in_transit_end:
            flux[i] *= 1 - transit_depth_ppm / 1e6
        flux[i] += rng.normal(0, np.sqrt(base_flux))

    flux_error = np.sqrt(np.abs(flux))
    wavelength_cube = np.tile(wavelength, (n_int, 1))

    # Build FITS
    primary = fits.PrimaryHDU()
    primary.header["TARGNAME"] = "TEST-TARGET"
    primary.header["INSTRUME"] = "NIRSPEC"
    primary.header["FILTER"] = "PRISM"
    primary.header["DATE-OBS"] = "2024-01-01"
    primary.header["PROGRAM"] = "9999"
    primary.header["OBSERVTN"] = "test_obs"

    # EXTRACT1D table
    col_wl = fits.Column(name="WAVELENGTH", format=f"{n_wl}D", array=wavelength_cube)
    col_fl = fits.Column(name="FLUX", format=f"{n_wl}E", array=flux)
    col_fe = fits.Column(name="FLUX_ERROR", format=f"{n_wl}E", array=flux_error)
    extract_hdu = fits.BinTableHDU.from_columns(
        [col_wl, col_fl, col_fe],
        name="EXTRACT1D",
    )

    # INT_TIMES table
    mjd_start = 60000.0
    int_time = 10.0 / 86400.0  # 10 seconds in days
    mjd_mid = mjd_start + np.arange(n_int) * int_time + int_time / 2
    col_mjd = fits.Column(name="int_mid_MJD_UTC", format="D", array=mjd_mid)
    int_times_hdu = fits.BinTableHDU.from_columns([col_mjd], name="INT_TIMES")

    hdul = fits.HDUList([primary, extract_hdu, int_times_hdu])
    hdul.writeto(filepath, overwrite=True)
    hdul.close()


class TestExtractTransitData:
    def test_basic_extraction(self, tmp_path):
        filepath = str(tmp_path / "test_x1dints.fits")
        make_synthetic_x1dints(filepath)
        td = extract_transit_data(filepath)

        assert td is not None
        assert isinstance(td, TransitData)
        assert td.n_integrations == 100
        assert td.n_wavelengths == 300
        assert td.wavelength.shape == (300,)
        assert td.flux_cube.shape == (100, 300)
        assert td.header["TARGNAME"] == "TEST-TARGET"

    def test_timestamps(self, tmp_path):
        filepath = str(tmp_path / "test_x1dints.fits")
        make_synthetic_x1dints(filepath)
        td = extract_transit_data(filepath)

        assert len(td.times_mjd) == 100
        assert td.times_mjd[0] > 60000.0  # reasonable MJD
        assert np.all(np.diff(td.times_mjd) > 0)  # monotonic

    def test_no_extract1d_returns_none(self, tmp_path):
        filepath = str(tmp_path / "bad.fits")
        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary])
        hdul.writeto(filepath, overwrite=True)
        hdul.close()

        td = extract_transit_data(filepath)
        assert td is None


class TestWhiteLightCurve:
    def test_basic(self, tmp_path):
        filepath = str(tmp_path / "test_x1dints.fits")
        make_synthetic_x1dints(filepath, transit_depth_ppm=20000)
        td = extract_transit_data(filepath)
        wlc = compute_white_light_curve(td)

        assert "error" not in wlc
        assert wlc["transit_depth_ppm"] > 0
        assert wlc["n_integrations"] == 100
        assert wlc["baseline"] > 0

    def test_detects_transit(self, tmp_path):
        filepath = str(tmp_path / "test_x1dints.fits")
        make_synthetic_x1dints(filepath, transit_depth_ppm=20000)
        td = extract_transit_data(filepath)
        wlc = compute_white_light_curve(td)

        # Should detect ~20000 ppm transit (within factor of 2)
        assert 5000 < wlc["transit_depth_ppm"] < 50000
