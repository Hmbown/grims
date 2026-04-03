"""Command-line interface for chime.

Usage:
    chime WASP-39                  # Run channel quality map
    chime TRAPPIST-1 --planet b    # Specify planet
    chime WASP-39 --bins 30        # Custom bin count
    chime WASP-39 --fit            # Run with transit model fitting
    chime --targets                # List available targets
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import QTable
import astropy.units as u

from bown_instruments.chime.ephemeris import get_ephemeris, list_targets, EPHEMERIDES
from bown_instruments.chime.mast import find_x1dints, download_product
from bown_instruments.chime.extract import extract_transit_data
from bown_instruments.chime.channel_map import compute_channel_map
from bown_instruments.chime.diversity import compute_diversity
from bown_instruments.chime.plot import plot_channel_map, plot_diversity, plot_segment_comparison


def identify_transit(times_mjd: np.ndarray, ephemeris: dict) -> np.ndarray:
    """Identify in-transit integrations using known ephemeris.

    Converts MJD to BJD_TDB (approximate) and phase-folds.
    """
    bjd = times_mjd + 2400000.5
    period = ephemeris["period_days"]
    t0 = ephemeris["t0_bjd"]
    half_dur = ephemeris["duration_hours"] / 24.0 / 2.0

    phase = ((bjd - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    phase_threshold = half_dur / period
    return np.abs(phase) < phase_threshold


def _write_outputs(
    all_results: list[dict],
    target: str,
    ephemeris: dict,
    outdir: Path,
):
    """Write JSON and ECSV output files."""
    slug = target.lower().replace(" ", "_").replace("-", "_")

    # JSON output
    json_path = outdir / f"chime_{slug}.json"
    json_data = {
        "target": target,
        "tool": "chime",
        "version": "0.1.0",
        "ephemeris": {k: v for k, v in ephemeris.items() if k != "ref"},
        "ephemeris_ref": ephemeris.get("ref", ""),
        "method": (
            "Empirical channel quality diagnostics using sub-band diversity combining "
            "inspired by Bown (US 1,747,221, 1930)"
        ),
        "observations": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # ECSV output (one row per wavelength bin, all observations)
    rows = []
    for obs_result in all_results:
        if "channel_map" not in obs_result:
            continue
        obs_id = obs_result.get("obs_id", "unknown")
        for b in obs_result["channel_map"].get("bins", []):
            rows.append(
                {
                    "obs_id": obs_id,
                    "wl_center_um": b["wl_center"],
                    "wl_min_um": b["wl_range"][0],
                    "wl_max_um": b["wl_range"][1],
                    "scatter_ppm": b["scatter_ppm"],
                    "photon_noise_ppm": b["photon_noise_ppm"],
                    "systematic_excess": b["systematic_excess"],
                    "allan_ratio": b["allan_worst_ratio"],
                    "grade": b["grade"],
                    "depth_ppm": b["depth_ppm"],
                }
            )

    if rows:
        ecsv_path = outdir / f"chime_{slug}.ecsv"
        table = QTable(rows=rows)
        table["wl_center_um"].unit = u.um
        table["wl_min_um"].unit = u.um
        table["wl_max_um"].unit = u.um
        table.write(str(ecsv_path), format="ascii.ecsv", overwrite=True)
        print(f"  ECSV: {ecsv_path}")


def _print_summary(all_results: list[dict], target: str):
    """Print a human-readable summary of results."""
    print()
    print("=" * 60)
    print(f"  chime summary: {target}")
    print("=" * 60)

    for i, obs in enumerate(all_results):
        smry = obs.get("summary", {})
        print(f"\n  Observation {i + 1}: {obs.get('obs_id', '?')}")
        print(
            f"    Integrations: {obs.get('n_integrations', '?')} "
            f"({obs.get('n_in_transit', '?')} in-transit)"
        )

        if smry:
            print(
                f"    Noise range: {smry.get('min_scatter_ppm', 0):.0f} — "
                f"{smry.get('max_scatter_ppm', 0):.0f} ppm "
                f"(ratio {smry.get('scatter_ratio', 0):.1f}×)"
            )
            print(
                f"    Quality grades: "
                f"A={smry.get('n_photon_limited', 0)} "
                f"B={smry.get('n_moderate', 0)} "
                f"C={smry.get('n_degraded', 0)} "
                f"D={smry.get('n_systematic_dominated', 0)}"
            )
            print(f"    Median Allan ratio: {smry.get('median_allan_ratio', 0):.2f}")

            # Trust regions
            regions = obs.get("trust_regions", [])
            if regions:
                region_strs = [f"{r['wl_min']:.2f}-{r['wl_max']:.2f} µm" for r in regions]
                print(f"    Trust regions: {', '.join(region_strs)}")
            else:
                print(f"    Trust regions: none")

            # Detectable molecules
            detectable = smry.get("detectable_molecules", [])
            detected = [d["molecule"] for d in detectable if d["detectable"]]
            if detected:
                print(f"    Detectable molecules: {', '.join(detected)}")

        # Diversity
        div = obs.get("diversity", {})
        if div:
            print(
                f"    Diversity improvement: "
                f"{div.get('improvement_factor', 0):.2f}× "
                f"(dropped {div.get('n_dropped', 0)}, "
                f"deweighted {div.get('n_degraded', 0)})"
            )

        # Transit fit
        fit = obs.get("transit_fit", {})
        if fit:
            print(
                f"    Transit fit: "
                f"{fit.get('combined_depth_ppm', 0):.0f} ± "
                f"{fit.get('combined_depth_err_ppm', 0):.0f} ppm "
                f"(improvement {fit.get('improvement_factor', 0):.2f}×)"
            )


def _run_segments_mode(target: str, ephemeris: dict, args, outdir: Path):
    """Download all per-segment files and produce segment-by-segment comparison."""
    from collections import defaultdict
    import re

    from bown_instruments.chime.mast import search_jwst, list_products, download_product
    from bown_instruments.chime.extract import extract_transit_data
    from bown_instruments.chime.channel_map import compute_channel_map

    print("=" * 60)
    print(f"  chime — Segment Quality Comparison")
    print(f"  Target: {target}")
    print("=" * 60)

    # Find all observations
    observations = search_jwst(
        target=target, instrument="NIRSPEC", calib_level_min=2, max_results=50
    )
    if not observations:
        print("  No NIRSpec observations found.")
        return

    # Collect per-segment files
    segment_files = []
    for obs in observations:
        try:
            products = list_products(obs, product_type="SCIENCE", subgroup="X1DINTS")
            for prod in products:
                fn = prod.product_filename.lower()
                if "x1dints" in fn and "nrs1" in fn and re.search(r"seg\d+", fn):
                    segment_files.append((obs, prod))
        except Exception:
            continue

    if not segment_files:
        print("  No per-segment x1dints files found.")
        return

    print(f"  Found {len(segment_files)} per-segment files")

    # Group by observation
    by_obs = defaultdict(list)
    for obs, prod in segment_files:
        by_obs[obs.obs_id].append((obs, prod))

    # Process each observation
    all_segment_maps = []

    for obs_id, segs in sorted(by_obs.items()):
        print(f"\n  Observation: {obs_id}")

        seg_results = []
        for obs, prod in segs:
            try:
                filepath = download_product(prod)
                td = extract_transit_data(filepath)
                if td is None:
                    continue

                # Verify G395H
                grating = td.header.get("GRATING", "")
                if "G395H" not in grating:
                    continue

                # Wavelength mask
                wl_min, wl_max = 2.86, 3.72
                wl_mask = (
                    (td.wavelength >= wl_min)
                    & (td.wavelength <= wl_max)
                    & np.isfinite(td.wavelength)
                )
                if np.sum(wl_mask) < 10:
                    continue

                # Transit mask
                bjd = td.times_mjd + 2400000.5
                period = ephemeris["period_days"]
                t0 = ephemeris["t0_bjd"]
                half_dur = ephemeris["duration_hours"] / 24.0 / 2.0
                phase = ((bjd - t0) / period) % 1.0
                phase[phase > 0.5] -= 1.0
                in_transit = np.abs(phase) < half_dur / period

                n_out = int(np.sum(~in_transit))
                if n_out < 10:
                    in_transit = np.zeros(td.n_integrations, dtype=bool)
                    mid = td.n_integrations // 2
                    hw = max(1, td.n_integrations // 10)
                    in_transit[mid - hw : mid + hw] = True

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
                    continue

                seg_match = re.search(r"seg(\d+)", prod.product_filename)
                seg_num = int(seg_match.group(1)) if seg_match else 0

                seg_results.append(
                    {
                        "label": f"seg{seg_num:03d}",
                        "segment": seg_num,
                        "wl_centers": [b.wl_center for b in cmap.bins],
                        "systematic_excess": [b.systematic_excess for b in cmap.bins],
                        "allan_ratios": [b.allan_worst_ratio for b in cmap.bins],
                        "scatter_ppm": [b.scatter_ppm for b in cmap.bins],
                        "grades": [b.grade for b in cmap.bins],
                        "n_A": cmap.summary["n_photon_limited"],
                        "n_bins": cmap.n_bins,
                        "median_excess": cmap.summary["median_excess"],
                        "median_allan": cmap.summary["median_allan_ratio"],
                    }
                )

            except Exception as e:
                print(f"    Error processing {prod.product_filename}: {e}")
                continue

        if not seg_results:
            continue

        # Sort by segment number
        seg_results.sort(key=lambda x: x["segment"])

        # Print table
        print(
            f"  {'Segment':>8} {'N_int':>6} {'A':>3} {'B':>3} {'C':>3} {'D':>3} {'Excess':>8} {'Allan':>7}"
        )
        for sr in seg_results:
            n_b = sum(1 for g in sr["grades"] if g == "B")
            n_c = sum(1 for g in sr["grades"] if g == "C")
            n_d = sum(1 for g in sr["grades"] if g == "D")
            print(
                f"  {sr['label']:>8} {'':>6} {sr['n_A']:>3} {n_b:>3} {n_c:>3} {n_d:>3} "
                f"{sr['median_excess']:>8.2f} {sr['median_allan']:>7.2f}"
            )

        # Plot
        if not args.no_plot and len(seg_results) >= 2:
            slug = target.lower().replace(" ", "_").replace("-", "_")
            obs_slug = obs_id.replace("jw", "").replace("-", "_")
            plot_path = str(outdir / f"segment_comparison_{slug}_{obs_slug}.png")
            plot_segment_comparison(seg_results, target, plot_path)
            print(f"  Plot: {plot_path}")

        all_segment_maps.extend(seg_results)

    if not all_segment_maps:
        print("\n  No segment results produced.")


def main(argv: list[str] | None = None):
    """Main entry point for the chime CLI."""
    parser = argparse.ArgumentParser(
        prog="chime",
        description="CHannel quality & Instrument Metrology for Exoplanets",
        epilog="Analytical approach inspired by Bown's sub-band diversity combining "
        "(US 1,747,221, 1930).",
    )
    parser.add_argument("target", nargs="?", help="Target name (e.g., WASP-39)")
    parser.add_argument("--targets", action="store_true", help="List available targets and exit")
    parser.add_argument(
        "--planet",
        type=str,
        default=None,
        help="Planet letter for multi-planet systems (e.g., b, c)",
    )
    parser.add_argument(
        "--bins", type=int, default=50, help="Number of wavelength bins (default: 50)"
    )
    parser.add_argument(
        "--subbands", type=int, default=15, help="Number of diversity sub-bands (default: 15)"
    )
    parser.add_argument(
        "--max-obs", type=int, default=3, help="Maximum observations to process (default: 3)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="chime_output",
        help="Output directory (default: chime_output)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--fit", action="store_true", help="Run transit model fitting")
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Download all per-segment files and produce segment-by-segment quality comparison",
    )

    args = parser.parse_args(argv)

    if args.targets:
        print("Available targets:")
        for t in list_targets():
            eph = EPHEMERIDES[t]
            print(
                f"  {t:20s}  P={eph['period_days']:.4f} d  "
                f"depth={eph['expected_depth_ppm']:.0f} ppm  "
                f"({eph['ref']})"
            )
        return

    if not args.target:
        parser.print_help()
        sys.exit(1)

    target = args.target

    # Look up ephemeris
    eph_key = target
    try:
        ephemeris = get_ephemeris(target)
    except KeyError:
        # Try with planet suffix
        if args.planet:
            try:
                ephemeris = get_ephemeris(f"{target}{args.planet}")
            except KeyError:
                try:
                    ephemeris = get_ephemeris(f"{target}-{args.planet}")
                except KeyError:
                    print(f"  No ephemeris for {target}. Use --targets to list available.")
                    sys.exit(1)
        else:
            print(f"  No ephemeris for {target}. Use --targets to list available.")
            sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --segments mode: per-segment quality comparison
    if args.segments:
        _run_segments_mode(target, ephemeris, args, outdir)
        return

    print("=" * 60)
    print(f"  chime — Channel Quality Map")
    print(f"  Target: {target}")
    print(f"  Period: {ephemeris['period_days']:.6f} d")
    print(f"  Duration: {ephemeris['duration_hours']:.3f} h")
    print(f"  Expected depth: {ephemeris['expected_depth_ppm']:.0f} ppm")
    print(f"  Ref: {ephemeris.get('ref', '')}")
    print("=" * 60)

    t0 = time.time()

    # Find x1dints
    pairs = find_x1dints(target, max_obs=args.max_obs)
    if not pairs:
        print("\n  No x1dints products found.")
        sys.exit(1)

    print(f"\n  Found {len(pairs)} x1dints products")

    all_results = []

    for i, (obs, prod) in enumerate(pairs):
        print(f"\n  [{i + 1}/{len(pairs)}] {prod.product_filename}")

        try:
            filepath = download_product(prod)
            td = extract_transit_data(filepath)
            if td is None:
                print("    No EXTRACT1D data")
                continue

            print(f"    {td.n_integrations} integrations, {td.n_wavelengths} wl points")
            print(f"    MJD {td.times_mjd[0]:.4f} — {td.times_mjd[-1]:.4f}")

            # Ephemeris-based transit identification
            in_transit = identify_transit(td.times_mjd, ephemeris)
            n_in = int(np.sum(in_transit))
            n_out = int(np.sum(~in_transit))
            print(f"    Transit ID: {n_in} in, {n_out} out")

            if n_in < 3:
                print("    Too few in-transit — using middle 20% as proxy")
                in_transit = np.zeros(td.n_integrations, dtype=bool)
                mid = td.n_integrations // 2
                hw = max(1, td.n_integrations // 10)
                in_transit[mid - hw : mid + hw] = True

            # Channel quality map (use pipeline flux errors as noise floor)
            cmap = compute_channel_map(
                td.flux_cube,
                td.wavelength,
                in_transit,
                n_bins=args.bins,
                flux_error_cube=td.flux_error_cube,
            )

            smry = cmap.summary
            print(
                f"    Noise: {smry['min_scatter_ppm']:.0f} — "
                f"{smry['max_scatter_ppm']:.0f} ppm "
                f"({smry['scatter_ratio']:.0f}× range)"
            )
            print(
                f"    Grades: A={smry['n_photon_limited']} "
                f"B={smry['n_moderate']} "
                f"C={smry['n_degraded']} "
                f"D={smry['n_systematic_dominated']}"
            )

            regions = cmap.trust_regions
            if regions:
                for r in regions:
                    print(
                        f"    Trust region: {r['wl_min']:.2f}—"
                        f"{r['wl_max']:.2f} µm "
                        f"(mean scatter {r['mean_scatter_ppm']:.0f} ppm)"
                    )

            # Diversity combining
            try:
                div = compute_diversity(
                    td.flux_cube,
                    td.wavelength,
                    in_transit,
                    n_subbands=args.subbands,
                )
                print(
                    f"    Diversity: {div.diversity_depth_ppm:.0f} ± "
                    f"{div.diversity_err_ppm:.0f} ppm "
                    f"(naive {div.naive_depth_ppm:.0f} ± "
                    f"{div.naive_err_ppm:.0f})"
                )
                print(f"    Improvement: {div.improvement_factor:.2f}×")
            except ValueError:
                div = None
                print("    Diversity: skipped (no valid sub-bands)")

            # Plot
            if not args.no_plot:
                slug = target.lower().replace(" ", "_").replace("-", "_")
                plot_path = str(outdir / f"chime_{slug}_{i + 1}.png")
                plot_channel_map(
                    cmap,
                    target,
                    plot_path,
                    obs_label=prod.product_filename,
                    ephemeris_ref=ephemeris.get("ref", ""),
                )
                print(f"    Plot: {plot_path}")

                if div:
                    div_path = str(outdir / f"diversity_{slug}_{i + 1}.png")
                    plot_diversity(div, target, div_path)
                    print(f"    Diversity plot: {div_path}")

            # Transit model fitting
            fit_result = None
            if args.fit:
                from bown_instruments.chime.transit_fit import fit_transmission_spectrum

                try:
                    fit_result = fit_transmission_spectrum(
                        flux_cube=td.flux_cube,
                        wavelength=td.wavelength,
                        times_mjd=td.times_mjd,
                        in_transit_mask=in_transit,
                        ephemeris=ephemeris,
                        channel_map=cmap,
                        n_bins=args.bins,
                        flux_error_cube=td.flux_error_cube,
                    )
                    active = fit_result.weights > 0
                    if np.any(active):
                        med_rp = np.median(fit_result.rp_rs[active])
                        print(f"    Transit fit: Rp/Rs = {med_rp:.4f}")
                    print(
                        f"    Combined depth: {fit_result.combined_depth_ppm:.0f}"
                        f" ± {fit_result.combined_depth_err_ppm:.0f} ppm"
                    )
                    print(f"    Improvement: {fit_result.improvement_factor:.2f}×")
                except Exception as e:
                    print(f"    Transit fit failed: {e}")

            result_entry = {
                "obs_id": obs.obs_id,
                "product": prod.product_filename,
                "n_integrations": td.n_integrations,
                "n_in_transit": n_in,
                "summary": smry,
                "trust_regions": regions,
                "channel_map": cmap.to_dict(),
            }

            if div:
                result_entry["diversity"] = {
                    "diversity_depth_ppm": div.diversity_depth_ppm,
                    "diversity_err_ppm": div.diversity_err_ppm,
                    "naive_depth_ppm": div.naive_depth_ppm,
                    "naive_err_ppm": div.naive_err_ppm,
                    "improvement_factor": div.improvement_factor,
                    "n_dropped": div.n_dropped,
                    "n_degraded": div.n_degraded,
                }

            if fit_result is not None:
                result_entry["transit_fit"] = fit_result.to_dict()

            all_results.append(result_entry)

        except Exception as e:
            print(f"    Error: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_results:
        print("\n  No results produced.")
        sys.exit(1)

    # Write outputs
    _write_outputs(all_results, target, ephemeris, outdir)
    _print_summary(all_results, target)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
