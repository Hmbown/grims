#!/usr/bin/env python3
"""Run the Phase 3 GRIM-S empirical null campaign."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_PHASE3_RESULTS = PROJECT_ROOT / "results" / "grims" / "phase3_results.json"
DEFAULT_CATALOG = PROJECT_ROOT / "results" / "grims" / "gwtc_full_catalog.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "grims" / "phase3_null_distribution.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-null", type=int, default=1000, help="Number of null realizations")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument(
        "--method",
        choices=["circular_time_shift", "fourier_phase_randomization"],
        default="circular_time_shift",
        help="Null-generation method",
    )
    parser.add_argument(
        "--phase3-results",
        type=Path,
        default=DEFAULT_PHASE3_RESULTS,
        help="Path to the Phase 3 stacked result JSON",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to the full GWTC catalog JSON",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for cached GWOSC HDF5 strain files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the null-distribution artifact",
    )
    parser.add_argument(
        "--max-weight-ratio",
        type=float,
        default=5.5,
        help="Weight cap for the final event stack",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-event and per-realization progress output",
    )
    return parser.parse_args()


def main() -> int:
    from bown_instruments.grims.null_distribution import (
        recommendation_for_claim_language,
        run_phase3_null_campaign,
        save_null_campaign,
    )

    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    progress = not args.quiet

    print("=" * 70)
    print("GRIM-S Phase 3: Empirical Null Calibration")
    print("=" * 70)
    print(f"Phase 3 results: {args.phase3_results}")
    print(f"Catalog:         {args.catalog}")
    print(f"Data cache:      {args.data_dir}")
    print(f"Method:          {args.method}")
    print(f"N_null:          {args.n_null}")
    print(f"Seed:            {args.seed}")
    print()

    start = time.time()
    result = run_phase3_null_campaign(
        phase3_results_path=args.phase3_results,
        catalog_path=args.catalog,
        data_dir=args.data_dir,
        n_null=args.n_null,
        seed=args.seed,
        method=args.method,
        max_weight_ratio=args.max_weight_ratio,
        progress=progress,
    )
    elapsed = time.time() - start
    result["runtime_seconds"] = elapsed

    save_null_campaign(result, args.output)

    print()
    print("=" * 70)
    print("EMPIRICAL NULL SUMMARY")
    print("=" * 70)
    print(f"Observed kappa:           {result['observed_kappa']:+.6f}")
    print(f"Observed sigma:           {result['observed_sigma']:.6f}")
    print(f"Null mean:                {result['null_mean']:+.6f}")
    print(f"Null std:                 {result['null_std']:.6f}")
    print(f"Empirical p-value:        {result['empirical_p_value']:.6f}")
    print(f"Empirical significance:   {result['empirical_sigma']:.2f} sigma")
    print(f"Asymptotic significance:  {result['asymptotic_sigma']:.2f} sigma")
    print(f"Calibration ratio:        {result['calibration_ratio']:.3f}")
    print(f"Well calibrated?:         {'YES' if result['is_well_calibrated'] else 'NO'}")
    print(
        "Sigma ratios consistent?: "
        f"{'YES' if result['per_event_null_check']['sigma_ratios_consistent'] else 'NO'}"
    )
    print(f"Stacked ln B (observed):  {result['observed_stacked_log_bayes_factor']:.3f}")
    print(f"Stacked ln B (null mean): {result['null_log_bayes_factor_mean']:.3f}")
    print(f"Runtime:                  {elapsed:.1f} s")
    print(f"Artifact:                 {args.output}")
    print()
    print(f"Recommendation: {recommendation_for_claim_language(result)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
