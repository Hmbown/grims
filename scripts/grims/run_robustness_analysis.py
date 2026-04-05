#!/usr/bin/env python3
"""
Run comprehensive robustness analysis on Phase 3 stacked kappa.

Addresses SHA-4223: quantify event influence and robustness of the Phase 3 stack.

This script extends beyond leave-one-out jackknife to:
  - Leave-k-out analysis (k=2, 3, 5)
  - Bootstrap resampling
  - Alternative weighting schemes
  - Detector subset analysis
  - Clear pass/fail criteria
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "results" / "grims"
RESULTS_PATH = DATA_DIR / "phase3_results.json"
ROBUSTNESS_PATH = DATA_DIR / "phase3_robustness.json"


def main():
    from bown_instruments.grims.robustness import (
        run_comprehensive_robustness,
        print_robustness_summary,
    )
    from bown_instruments.grims.phase_locked_search import PhaseLockResult

    print("=" * 70)
    print("GRIM-S Phase 3: Comprehensive Robustness Analysis")
    print("=" * 70)
    print()

    if not RESULTS_PATH.exists():
        print(f"ERROR: Phase 3 results not found at {RESULTS_PATH}")
        print("Run scripts/grims/run_phase3_analysis.py first.")
        return 1

    with open(RESULTS_PATH) as f:
        phase3_data = json.load(f)

    print(f"Loaded Phase 3 results: {phase3_data['n_analyzed']} events analyzed")
    print()

    phase_lock_results = []
    results_with_metadata = []

    for r in phase3_data["individual"]:
        res = r.get("result", {})

        plr = PhaseLockResult(
            event_name=r["event"],
            kappa_hat=r["kappa_hat"],
            kappa_sigma=r["kappa_sigma"],
            snr=r["snr_nl"],
            a_220_fit=r.get("a_220_fit", 0.0),
            phi_220_fit=res.get("phi_220_fit", 0.0),
            template_norm=res.get("template_norm", 0.0),
            residual_overlap=r["kappa_hat"] * res.get("template_norm", 0.0) ** 2,
            noise_rms=r.get("noise_rms", 1.0),
        )

        phase_lock_results.append(plr)

        results_with_metadata.append(
            {
                "event": r["event"],
                "detectors_used": r.get("detectors_used", ["H1"]),
                "kappa_hat": r["kappa_hat"],
                "kappa_sigma": r["kappa_sigma"],
            }
        )

    print(f"Running comprehensive robustness analysis on {len(phase_lock_results)} events...")
    print()

    robustness = run_comprehensive_robustness(
        results=phase_lock_results,
        results_with_metadata=results_with_metadata,
        max_weight_ratio=5.5,
        n_bootstrap=500,
        seed=42,
    )

    print_robustness_summary(robustness)

    def make_serializable(obj):
        if hasattr(obj, "__dict__"):
            return {k: make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(v) for v in obj)
        return obj

    robustness_dict = {
        "n_events": robustness.n_events,
        "full_kappa": robustness.full_kappa,
        "full_sigma": robustness.full_sigma,
        "full_snr": robustness.full_snr,
        "leave_one_out": make_serializable(robustness.leave_one_out),
        "leave_two_out": make_serializable(robustness.leave_two_out),
        "leave_three_out": make_serializable(robustness.leave_three_out),
        "leave_five_out": make_serializable(robustness.leave_five_out),
        "bootstrap_mean": robustness.bootstrap_mean,
        "bootstrap_std": robustness.bootstrap_std,
        "bootstrap_bias": robustness.bootstrap_bias,
        "bootstrap_n_samples": robustness.bootstrap_n_samples,
        "weighting_schemes": make_serializable(robustness.weighting_schemes),
        "detector_subsets": make_serializable(robustness.detector_subsets),
        "sigma_quality_cuts": make_serializable(robustness.sigma_quality_cuts),
        "top_influential_events": make_serializable(robustness.top_influential_events),
        "gini_coefficient": robustness.gini_coefficient,
        "n_eff": robustness.n_eff,
        "is_robust": robustness.is_robust,
        "robustness_score": robustness.robustness_score,
        "failures": robustness.failures,
        "caveats": robustness.caveats,
    }

    with open(ROBUSTNESS_PATH, "w") as f:
        json.dump(robustness_dict, f, indent=2)

    print(f"\nRobustness analysis saved to {ROBUSTNESS_PATH}")

    print("\n" + "=" * 70)
    print("SUMMARY FOR LINEAR ISSUE SHA-4223")
    print("=" * 70)
    print()
    print(f"Robustness Status: {'ROBUST' if robustness.is_robust else 'NOT ROBUST'}")
    print(f"Robustness Score: {robustness.robustness_score:.2f}/1.00")
    print()

    if robustness.failures:
        print("Critical Failures:")
        for f in robustness.failures:
            print(f"  - {f}")
        print()

    if robustness.caveats:
        print("Important Caveats:")
        for c in robustness.caveats:
            print(f"  - {c}")
        print()

    print("Key Metrics:")
    print(f"  - Gini coefficient: {robustness.gini_coefficient:.3f}")
    print(f"  - Effective sample size: {robustness.n_eff:.1f} / {robustness.n_events}")
    print(f"  - Top influential event shifts: ", end="")
    if robustness.top_influential_events:
        top3 = robustness.top_influential_events[:3]
        print(", ".join([f"{e['event']} ({e['shift_in_sigma']:.2f}σ)" for e in top3]))
    else:
        print("None")

    print()
    print("Recommendation for Phase 3 publication:")
    if robustness.is_robust:
        print("  The stacked result appears ROBUST and could be reported")
        print("  with appropriate caveats as a hint or upper limit.")
    else:
        print("  The stacked result is NOT ROBUST enough for external claims.")
        print("  Present as provisional/local analysis only.")
        print("  Complete SHA-4221 (empirical null) before publication.")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    import numpy as np

    sys.exit(main())
