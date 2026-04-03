#!/usr/bin/env python3
"""Multi-target survey — process multiple JWST targets.

Runs the channel quality analysis on several well-studied exoplanets
and compares their channel quality characteristics. Useful for
identifying which targets have the best-quality data for atmospheric
characterization.

Usage:
    python examples/multi_target_survey.py
    python examples/multi_target_survey.py --targets WASP-39 TRAPPIST-1
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from bown_instruments.chime.ephemeris import get_ephemeris, list_targets
from bown_instruments.chime.mast import find_x1dints, download_product
from bown_instruments.chime.extract import extract_transit_data
from bown_instruments.chime.channel_map import compute_channel_map
from bown_instruments.chime.diversity import compute_diversity
from bown_instruments.chime.cli import identify_transit

DEFAULT_TARGETS = ["WASP-39", "TRAPPIST-1", "WASP-107", "HD-189733"]


def survey_target(target: str, outdir: Path) -> dict | None:
    """Run channel quality analysis on a single target."""
    print(f"\n{'=' * 60}")
    print(f"  {target}")
    print(f"{'=' * 60}")

    try:
        ephemeris = get_ephemeris(target)
    except KeyError:
        print(f"  No ephemeris found, skipping")
        return None

    pairs = find_x1dints(target, max_obs=2)
    if not pairs:
        print(f"  No x1dints found, skipping")
        return None

    print(f"  Found {len(pairs)} observations")

    results = []
    for i, (obs, prod) in enumerate(pairs):
        try:
            filepath = download_product(prod)
            td = extract_transit_data(filepath)
            if td is None:
                continue

            in_transit = identify_transit(td.times_mjd, ephemeris)

            if np.sum(in_transit) < 3:
                in_transit = np.zeros(td.n_integrations, dtype=bool)
                mid = td.n_integrations // 2
                hw = max(1, td.n_integrations // 10)
                in_transit[mid - hw : mid + hw] = True

            cmap = compute_channel_map(td.flux_cube, td.wavelength, in_transit, n_bins=30)

            try:
                div = compute_diversity(td.flux_cube, td.wavelength, in_transit, n_subbands=10)
                improvement = div.improvement_factor
            except ValueError:
                improvement = 0.0

            smry = cmap.summary
            results.append(
                {
                    "obs_id": obs.obs_id,
                    "n_integrations": td.n_integrations,
                    "n_in_transit": int(np.sum(in_transit)),
                    "median_scatter_ppm": smry.get("min_scatter_ppm", 0),
                    "max_scatter_ppm": smry.get("max_scatter_ppm", 0),
                    "scatter_ratio": smry.get("scatter_ratio", 0),
                    "n_a_grade": smry.get("n_photon_limited", 0),
                    "n_d_grade": smry.get("n_systematic_dominated", 0),
                    "n_trust_regions": smry.get("n_trust_regions", 0),
                    "diversity_improvement": improvement,
                    "trust_regions": cmap.trust_regions,
                }
            )

            print(
                f"  [{i + 1}] {obs.obs_id}: "
                f"noise {smry.get('min_scatter_ppm', 0):.0f}-"
                f"{smry.get('max_scatter_ppm', 0):.0f} ppm, "
                f"A={smry.get('n_photon_limited', 0)} "
                f"D={smry.get('n_systematic_dominated', 0)}, "
                f"diversity {improvement:.1f}x"
            )

        except Exception as e:
            print(f"  [{i + 1}] Error: {e}")
            continue

    if not results:
        return None

    return {
        "target": target,
        "n_observations": len(results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-target chime survey")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, help="Targets to survey")
    parser.add_argument("--outdir", default="chime_output/survey", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  chime multi-target survey")
    print(f"  Targets: {', '.join(args.targets)}")
    print("=" * 60)

    t0 = time.time()
    survey = []

    for target in args.targets:
        result = survey_target(target, outdir)
        if result:
            survey.append(result)

    # Rank by diversity improvement
    ranked = sorted(
        survey,
        key=lambda s: max((r["diversity_improvement"] for r in s["results"]), default=0),
        reverse=True,
    )

    print("\n" + "=" * 60)
    print("  SURVEY SUMMARY")
    print("=" * 60)
    for i, s in enumerate(ranked):
        best = max(s["results"], key=lambda r: r["diversity_improvement"])
        print(
            f"  {i + 1}. {s['target']:20s}  "
            f"best diversity: {best['diversity_improvement']:.1f}x  "
            f"A={best['n_a_grade']} D={best['n_d_grade']}"
        )

    # Save
    json_path = outdir / "survey_results.json"
    with open(json_path, "w") as f:
        json.dump(survey, f, indent=2, default=str)
    print(f"\n  Results: {json_path}")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
