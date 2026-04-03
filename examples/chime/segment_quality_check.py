"""Segment quality diagnostic script for JWST x1dints files.

Usage:
    python examples/segment_quality_check.py path/to/file.fits
    python examples/segment_quality_check.py --download WASP-39

Takes any x1dints FITS file and prints a one-page quality report:
per-channel grades, Allan ratios, trust regions, and a clear
PASS/WARN/FAIL assessment.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from bown_instruments.chime.extract import extract_transit_data
from bown_instruments.chime.channel_map import compute_channel_map
from bown_instruments.chime.ephemeris import get_ephemeris
from bown_instruments.chime.mast import find_x1dints, download_product


def identify_transit(times_mjd: np.ndarray, ephemeris: dict) -> np.ndarray:
    """Identify in-transit integrations using ephemeris."""
    bjd = times_mjd + 2400000.5
    period = ephemeris["period_days"]
    t0 = ephemeris["t0_bjd"]
    half_dur = ephemeris["duration_hours"] / 24.0 / 2.0

    phase = ((bjd - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    return np.abs(phase) < half_dur / period


def assess_quality(cmap) -> tuple[str, str]:
    """Return PASS/WARN/FAIL assessment with reason."""
    n_total = cmap.n_bins
    if n_total == 0:
        return "FAIL", "No valid wavelength bins"

    n_a = cmap.summary["n_photon_limited"]
    n_d = cmap.summary["n_systematic_dominated"]
    median_excess = cmap.summary["median_excess"]
    median_allan = cmap.summary["median_allan_ratio"]
    n_trust = cmap.summary["n_trust_regions"]

    a_fraction = n_a / n_total

    if n_d > n_total * 0.5:
        return "FAIL", f"{n_d}/{n_total} channels systematic-dominated (D-grade)"
    elif median_excess > 5:
        return "FAIL", f"Median systematic excess {median_excess:.1f}x (threshold: 5x)"
    elif median_allan > 3:
        return "FAIL", f"Median Allan ratio {median_allan:.2f} (threshold: 3.0)"
    elif a_fraction < 0.5:
        return "WARN", f"Only {n_a}/{n_total} photon-limited channels (need >50%)"
    elif median_excess > 2:
        return "WARN", f"Median systematic excess {median_excess:.1f}x (moderate)"
    elif n_trust == 0:
        return "WARN", "No contiguous trust regions found"
    else:
        return (
            "PASS",
            f"{n_a}/{n_total} A-grade, excess={median_excess:.2f}x, allan={median_allan:.2f}",
        )


def print_report(filepath: str, td, cmap, ephemeris: dict | None = None):
    """Print a one-page quality report."""
    # Overall assessment
    verdict, reason = assess_quality(cmap)

    # Header
    print("=" * 72)
    print(f"  chime — Segment Quality Report")
    print("=" * 72)

    # File info
    print(f"\n  File: {Path(filepath).name}")
    print(f"  Target: {td.header.get('TARGNAME', 'unknown')}")
    print(f"  Instrument: {td.header.get('INSTRUME', 'unknown')}")
    print(f"  Grating: {td.header.get('GRATING', 'unknown')}")
    print(f"  Filter: {td.header.get('FILTER', 'unknown')}")
    print(f"  Program: {td.header.get('PROGRAM', 'unknown')}")
    print(f"  Obs: {td.header.get('OBSERVTN', 'unknown')}")
    print(f"  Date: {td.header.get('DATE-OBS', 'unknown')}")

    # Time info
    print(f"\n  Integrations: {td.n_integrations}")
    print(f"  Wavelength points: {td.n_wavelengths}")
    print(f"  Time range: {td.times_mjd[0]:.5f} - {td.times_mjd[-1]:.5f} MJD")
    print(f"  Duration: {(td.times_mjd[-1] - td.times_mjd[0]) * 24 * 60:.1f} minutes")

    if ephemeris:
        bjd_start = td.times_mjd[0] + 2400000.5
        bjd_end = td.times_mjd[-1] + 2400000.5
        period = ephemeris["period_days"]
        t0 = ephemeris["t0_bjd"]
        phase_start = ((bjd_start - t0) / period) % 1.0
        phase_end = ((bjd_end - t0) / period) % 1.0
        print(f"  Orbital phase: {phase_start:.3f} - {phase_end:.3f}")

    # Transit info
    in_transit = identify_transit(td.times_mjd, ephemeris) if ephemeris else None
    if in_transit is not None:
        n_in = int(np.sum(in_transit))
        n_out = int(np.sum(~in_transit))
        print(f"  In-transit: {n_in} ({n_in / td.n_integrations:.0%})")
        print(f"  Out-of-transit: {n_out} ({n_out / td.n_integrations:.0%})")

    # Quality summary
    print(f"\n  {'─' * 72}")
    print(f"  CHANNEL QUALITY SUMMARY")
    print(f"  {'─' * 72}")

    smry = cmap.summary
    print(f"\n  Wavelength bins analyzed: {cmap.n_bins}")
    print(
        f"  Wavelength range: {cmap.bins[0].wl_range[0]:.2f} - {cmap.bins[-1].wl_range[1]:.2f} µm"
    )

    print(f"\n  Quality grades:")
    print(f"    A (photon-limited):     {smry['n_photon_limited']:>3}/{cmap.n_bins}")
    print(f"    B (moderate systematics): {smry['n_moderate']:>3}/{cmap.n_bins}")
    print(f"    C (significant systematics): {smry['n_degraded']:>3}/{cmap.n_bins}")
    print(f"    D (systematic-dominated): {smry['n_systematic_dominated']:>3}/{cmap.n_bins}")

    print(f"\n  Noise statistics:")
    print(f"    Median systematic excess: {smry['median_excess']:.2f}x")
    print(f"    Median Allan ratio: {smry['median_allan_ratio']:.2f}")
    print(f"    Scatter range: {smry['min_scatter_ppm']:.0f} - {smry['max_scatter_ppm']:.0f} ppm")
    print(f"    Scatter ratio: {smry['scatter_ratio']:.1f}x")

    # Trust regions
    print(f"\n  Trust regions: {smry['n_trust_regions']}")
    for i, r in enumerate(cmap.trust_regions):
        print(
            f"    {i + 1}. {r['wl_min']:.2f} - {r['wl_max']:.2f} µm "
            f"(scatter: {r['mean_scatter_ppm']:.0f} ppm, excess: {r['mean_excess']:.2f}x)"
        )

    # Per-channel table
    print(f"\n  {'─' * 72}")
    print(f"  PER-CHANNEL DETAILS")
    print(f"  {'─' * 72}")
    print(f"  {'Wavelength(µm)':>14} {'Grade':>5} {'Excess':>8} {'Allan':>7} {'Scatter(ppm)':>13}")
    print(f"  {'─' * 50}")

    for b in cmap.bins:
        print(
            f"  {b.wl_center:>14.3f} {b.grade:>5} {b.systematic_excess:>8.2f} "
            f"{b.allan_worst_ratio:>7.2f} {b.scatter_ppm:>13.0f}"
        )

    # Assessment
    print(f"\n  {'─' * 72}")
    print(f"  ASSESSMENT")
    print(f"  {'─' * 72}")

    verdict_symbol = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}
    verdict_color = {"PASS": "GREEN", "WARN": "YELLOW", "FAIL": "RED"}
    symbol = verdict_symbol.get(verdict, "?")

    print(f"\n  [{verdict}] {symbol} {verdict_color[verdict]}")
    print(f"  Reason: {reason}")

    if verdict == "FAIL":
        print(f"\n  Recommendation: Do not use this segment for precision spectroscopy.")
        print(f"  The noise is dominated by systematics that will not average down.")
    elif verdict == "WARN":
        print(f"\n  Recommendation: Use with caution. Consider downweighting or")
        print(f"  applying additional systematic corrections.")
    else:
        print(f"\n  Recommendation: This segment is suitable for precision spectroscopy.")
        print(f"  Noise is photon-limited with white noise characteristics.")

    # Detectable molecules
    detectable = smry.get("detectable_molecules", [])
    if detectable:
        detected = [d["molecule"] for d in detectable if d["detectable"]]
        if detected:
            print(f"\n  Detectable molecules (3-sigma): {', '.join(detected)}")

    print(f"\n  {'=' * 72}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment quality diagnostic for JWST x1dints files"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Path to x1dints FITS file",
    )
    parser.add_argument(
        "--download",
        type=str,
        help="Download x1dints for target name (e.g., WASP-39)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=19,
        help="Number of wavelength bins (default: 19)",
    )
    parser.add_argument(
        "--wl-min",
        type=float,
        default=2.86,
        help="Minimum wavelength in µm (default: 2.86)",
    )
    parser.add_argument(
        "--wl-max",
        type=float,
        default=3.72,
        help="Maximum wavelength in µm (default: 3.72)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of text report",
    )

    args = parser.parse_args()

    filepath = args.filepath

    # Download if requested
    if args.download and not filepath:
        print(f"Searching MAST for {args.download}...")
        pairs = find_x1dints(args.download, max_obs=10)
        if not pairs:
            print("No x1dints products found.")
            sys.exit(1)

        print(f"Found {len(pairs)} products:")
        for i, (obs, prod) in enumerate(pairs):
            print(f"  [{i + 1}] {prod.product_filename} ({prod.size / 1e6:.1f} MB)")

        # Download the first one
        print(f"\nDownloading first product...")
        filepath = download_product(pairs[0][1])
        print(f"Downloaded: {filepath}")

    if not filepath:
        parser.print_help()
        sys.exit(1)

    # Extract data
    td = extract_transit_data(filepath)
    if td is None:
        print("Error: No EXTRACT1D data found in file.")
        sys.exit(1)

    # Wavelength mask
    wl_mask = (
        (td.wavelength >= args.wl_min) & (td.wavelength <= args.wl_max) & np.isfinite(td.wavelength)
    )
    if np.sum(wl_mask) < 10:
        print(
            f"Error: Too few valid wavelengths in {args.wl_min}-{args.wl_max} µm band: {np.sum(wl_mask)}"
        )
        sys.exit(1)

    # Ephemeris
    target = td.header.get("TARGNAME", "")
    ephemeris = None
    if target:
        try:
            ephemeris = get_ephemeris(target)
        except KeyError:
            pass

    # Transit mask
    if ephemeris:
        in_transit = identify_transit(td.times_mjd, ephemeris)
        n_in = int(np.sum(in_transit))
        n_out = int(np.sum(~in_transit))
        if n_out < 10:
            in_transit = np.zeros(td.n_integrations, dtype=bool)
            mid = td.n_integrations // 2
            hw = max(1, td.n_integrations // 10)
            in_transit[mid - hw : mid + hw] = True
    else:
        in_transit = np.zeros(td.n_integrations, dtype=bool)
        mid = td.n_integrations // 2
        hw = max(1, td.n_integrations // 10)
        in_transit[mid - hw : mid + hw] = True

    # Channel quality
    wl_band = td.wavelength[wl_mask]
    fl_band = td.flux_cube[:, wl_mask]
    fe_band = td.flux_error_cube[:, wl_mask]

    cmap = compute_channel_map(
        fl_band,
        wl_band,
        in_transit,
        n_bins=args.bins,
        flux_error_cube=fe_band,
    )

    if cmap.n_bins == 0:
        print("Error: No valid wavelength bins produced.")
        sys.exit(1)

    # Output
    if args.json:
        result = {
            "file": filepath,
            "target": target,
            "n_integrations": td.n_integrations,
            "n_bins": cmap.n_bins,
            "assessment": assess_quality(cmap)[0],
            "reason": assess_quality(cmap)[1],
            "summary": cmap.summary,
            "bins": [
                {
                    "wl_center": b.wl_center,
                    "grade": b.grade,
                    "excess": b.systematic_excess,
                    "allan": b.allan_worst_ratio,
                    "scatter_ppm": b.scatter_ppm,
                }
                for b in cmap.bins
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        print_report(filepath, td, cmap, ephemeris)


if __name__ == "__main__":
    main()
