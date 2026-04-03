#!/usr/bin/env python
"""End-to-end WASP-39b transit fitting with chime.

Demonstrates the full pipeline:
  1. Look up ephemeris from chime's database
  2. Find and download x1dints products from MAST
  3. Extract per-integration spectra
  4. Identify in-transit integrations via ephemeris
  5. Compute channel quality map (per-wavelength noise diagnostics)
  6. Fit Mandel & Agol transit model with GP systematics per wavelength
  7. Diversity-combine the transmission spectrum
  8. Save results

Requires network access for MAST download.

Usage:
    python examples/wasp39_full_fit.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from bown_instruments.chime.ephemeris import get_ephemeris
from bown_instruments.chime.mast import find_x1dints, download_product
from bown_instruments.chime.extract import extract_transit_data
from bown_instruments.chime.channel_map import compute_channel_map
from bown_instruments.chime.diversity import compute_diversity
from bown_instruments.chime.transit_fit import fit_transmission_spectrum
from bown_instruments.chime.cli import identify_transit


def main():
    target = "WASP-39"
    outdir = Path("chime_output")
    outdir.mkdir(exist_ok=True)

    # 1. Ephemeris
    eph = get_ephemeris(target)
    print(f"Target: {target}")
    print(f"  Period: {eph['period_days']:.6f} d")
    print(f"  Duration: {eph['duration_hours']:.3f} h")
    print(f"  Expected Rp/Rs: {eph['rp_rs']:.4f}")
    print(f"  Expected depth: {eph['expected_depth_ppm']} ppm")
    print()

    # 2. Find x1dints on MAST
    pairs = find_x1dints(target, max_obs=1)
    if not pairs:
        print("No x1dints products found on MAST.")
        sys.exit(1)

    obs, prod = pairs[0]
    print(f"Downloading: {prod.product_filename}")

    # 3. Download and extract
    filepath = download_product(prod)
    td = extract_transit_data(filepath)
    if td is None:
        print("Failed to extract transit data.")
        sys.exit(1)

    print(f"  {td.n_integrations} integrations, {td.n_wavelengths} wavelength points")
    print(f"  Wavelength range: {td.wavelength.min():.2f} — {td.wavelength.max():.2f} µm")

    # 4. Identify transit from ephemeris
    in_transit = identify_transit(td.times_mjd, eph)
    print(f"  In-transit: {np.sum(in_transit)}, out-of-transit: {np.sum(~in_transit)}")
    print()

    # 5. Channel quality map
    print("Computing channel quality map...")
    cmap = compute_channel_map(
        td.flux_cube, td.wavelength, in_transit,
        n_bins=30, flux_error_cube=td.flux_error_cube,
    )
    smry = cmap.summary
    print(f"  Grades: A={smry['n_photon_limited']} B={smry['n_moderate']} "
          f"C={smry['n_degraded']} D={smry['n_systematic_dominated']}")
    print(f"  Noise range: {smry['min_scatter_ppm']:.0f} — {smry['max_scatter_ppm']:.0f} ppm")
    print()

    # 6. Transit model fitting
    print("Fitting transit model per wavelength bin...")
    result = fit_transmission_spectrum(
        flux_cube=td.flux_cube,
        wavelength=td.wavelength,
        times_mjd=td.times_mjd,
        in_transit_mask=in_transit,
        ephemeris=eph,
        channel_map=cmap,
        n_bins=30,
        ld_coeffs=(0.06, 0.18),  # WASP-39 NIRSpec PRISM LD coefficients
    )

    # 7. Print results
    active = result.weights > 0
    print(f"\nTransmission Spectrum ({result.n_bins} bins, {result.n_dropped} dropped):")
    print(f"{'Wavelength':>12s} {'Rp/Rs':>8s} {'Depth(ppm)':>11s} {'Err(ppm)':>9s} {'Grade':>6s} {'Weight':>7s}")
    print("-" * 60)
    for i in range(result.n_bins):
        print(f"{result.wl_centers[i]:12.3f} {result.rp_rs[i]:8.4f} "
              f"{result.depth_ppm[i]:11.0f} {result.depth_err_ppm[i]:9.0f} "
              f"{result.grades[i]:>6s} {result.weights[i]:7.4f}")

    print()
    print(f"Combined (diversity-weighted): {result.combined_depth_ppm:.0f} "
          f"± {result.combined_depth_err_ppm:.0f} ppm")
    print(f"Naive (unweighted):            {result.naive_depth_ppm:.0f} "
          f"± {result.naive_depth_err_ppm:.0f} ppm")
    print(f"Improvement factor:            {result.improvement_factor:.2f}×")

    # 8. Save
    out_path = outdir / "wasp39_transit_fit.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
