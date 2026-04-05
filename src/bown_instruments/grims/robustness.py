"""
Comprehensive robustness analysis for GRIM-S stacked kappa.

Extends beyond leave-one-out jackknife to include:
  - Leave-k-out analysis
  - Bootstrap resampling
  - Alternative weighting schemes
  - Detector subset analysis
  - Clear pass/fail criteria

Bown principle: a measurement that cannot detect its own failure
is not a measurement. A result that depends critically on a small
subset of events is not a measurement — it is an observation.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from itertools import combinations
import warnings

from .phase_locked_search import (
    PhaseLockResult,
    StackedPhaseLockResult,
    stack_phase_locked,
)


@dataclass
class RobustnessResult:
    """Comprehensive robustness analysis result."""

    # Basic info
    n_events: int
    full_kappa: float
    full_sigma: float
    full_snr: float

    # Leave-k-out results
    leave_one_out: dict  # From jackknife
    leave_two_out: Optional[dict] = None
    leave_three_out: Optional[dict] = None
    leave_five_out: Optional[dict] = None

    # Bootstrap results
    bootstrap_mean: Optional[float] = None
    bootstrap_std: Optional[float] = None
    bootstrap_bias: Optional[float] = None
    bootstrap_n_samples: int = 0

    # Weighting scheme comparison
    weighting_schemes: dict = field(default_factory=dict)

    # Detector subset analysis
    detector_subsets: dict = field(default_factory=dict)

    # Sigma quality cut analysis
    sigma_quality_cuts: dict = field(default_factory=dict)

    # Influence metrics
    top_influential_events: List[dict] = field(default_factory=list)
    gini_coefficient: float = 0.0
    n_eff: float = 0.0

    # Overall assessment
    is_robust: bool = False
    robustness_score: float = 0.0  # 0-1 scale
    failures: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


def compute_gini_coefficient(shifts: np.ndarray) -> float:
    """Compute Gini coefficient of jackknife shifts.

    Gini = 0: perfectly uniform influence
    Gini = 1: all influence in one event

    Uses the standard formula: G = sum_i sum_j |x_i - x_j| / (2 * n * mean(|x|))
    """
    abs_shifts = np.abs(shifts)
    if len(abs_shifts) == 0 or np.sum(abs_shifts) == 0:
        return 0.0

    n = len(abs_shifts)
    mean_abs = np.mean(abs_shifts)
    if mean_abs == 0:
        return 0.0

    gini = np.sum(np.abs(abs_shifts[:, np.newaxis] - abs_shifts)) / (2 * n * n * mean_abs)
    return float(gini)


def run_leave_k_out(
    results: List[PhaseLockResult],
    k: int,
    max_weight_ratio: Optional[float] = None,
    n_combinations: Optional[int] = None,
) -> dict:
    """Run leave-k-out analysis.

    Parameters
    ----------
    results : list of PhaseLockResult
    k : number of events to remove at once
    max_weight_ratio : weight cap
    n_combinations : max number of combinations to test (for large k)

    Returns
    -------
    dict with statistics on the distribution of jackknife estimates
    """
    n = len(results)
    if n < k + 2:
        return {"error": f"Need at least {k + 2} events for leave-{k}-out"}

    all_kappas = []
    all_sigmas = []
    removed_sets = []

    all_combos = list(combinations(range(n), k))
    if n_combinations is not None and len(all_combos) > n_combinations:
        np.random.seed(42)
        all_combos = [
            all_combos[i] for i in np.random.choice(len(all_combos), n_combinations, replace=False)
        ]

    for remove_indices in all_combos:
        keep_indices = [i for i in range(n) if i not in remove_indices]
        subset = [results[i] for i in keep_indices]

        try:
            stacked = stack_phase_locked(subset, max_weight_ratio=max_weight_ratio)
            all_kappas.append(stacked.kappa_hat)
            all_sigmas.append(stacked.kappa_sigma)
            removed_sets.append([results[i].event_name for i in remove_indices])
        except Exception as e:
            warnings.warn(f"Leave-{k}-out failed for {remove_indices}: {e}")
            continue

    if len(all_kappas) == 0:
        return {"error": "All leave-{k}-out combinations failed"}

    all_kappas = np.array(all_kappas)
    all_sigmas = np.array(all_sigmas)

    return {
        "k": k,
        "n_combinations_tested": len(all_kappas),
        "mean": float(np.mean(all_kappas)),
        "std": float(np.std(all_kappas)),
        "median": float(np.median(all_kappas)),
        "min": float(np.min(all_kappas)),
        "max": float(np.max(all_kappas)),
        "range": float(np.max(all_kappas) - np.min(all_kappas)),
        "q25": float(np.percentile(all_kappas, 25)),
        "q75": float(np.percentile(all_kappas, 75)),
        "removed_sets": removed_sets,
        "kappas": all_kappas.tolist(),
        "sigmas": all_sigmas.tolist(),
    }


def run_bootstrap(
    results: List[PhaseLockResult],
    max_weight_ratio: Optional[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Run bootstrap resampling analysis.

    Resample events WITH REPLACEMENT and recompute stacked kappa.
    This tests the stability of the estimate to sampling variation.
    """
    np.random.seed(seed)
    n = len(results)

    bootstrap_kappas = []
    bootstrap_sigmas = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        resampled = [results[i] for i in indices]

        try:
            stacked = stack_phase_locked(resampled, max_weight_ratio=max_weight_ratio)
            bootstrap_kappas.append(stacked.kappa_hat)
            bootstrap_sigmas.append(stacked.kappa_sigma)
        except:
            continue

    if len(bootstrap_kappas) == 0:
        return {"error": "All bootstrap samples failed"}

    bootstrap_kappas = np.array(bootstrap_kappas)

    full_stacked = stack_phase_locked(results, max_weight_ratio=max_weight_ratio)
    bias = np.mean(bootstrap_kappas) - full_stacked.kappa_hat

    return {
        "n_samples": len(bootstrap_kappas),
        "mean": float(np.mean(bootstrap_kappas)),
        "std": float(np.std(bootstrap_kappas)),
        "bias": float(bias),
        "bias_in_sigma": float(bias / full_stacked.kappa_sigma)
        if full_stacked.kappa_sigma > 0
        else 0.0,
        "median": float(np.median(bootstrap_kappas)),
        "q025": float(np.percentile(bootstrap_kappas, 2.5)),
        "q975": float(np.percentile(bootstrap_kappas, 97.5)),
        "kappas": bootstrap_kappas.tolist(),
    }


def test_weighting_schemes(
    results: List[PhaseLockResult],
    schemes: Optional[dict] = None,
) -> dict:
    """Test different weighting schemes.

    Parameters
    ----------
    results : list of PhaseLockResult
    schemes : dict mapping name to max_weight_ratio value
              If None, tests a default set

    Returns
    -------
    dict mapping scheme name to stacked result
    """
    if schemes is None:
        schemes = {
            "uncapped": None,
            "ratio_3": 3.0,
            "ratio_5": 5.0,
            "ratio_7": 7.0,
            "ratio_10": 10.0,
            "equal_weights": -1,  # Special flag for equal weights
        }

    results_dict = {}

    for name, ratio in schemes.items():
        try:
            if ratio == -1:
                stacked = stack_phase_locked(
                    results, max_weight_ratio=None, force_equal_weights=True
                )
            else:
                stacked = stack_phase_locked(results, max_weight_ratio=ratio)

            results_dict[name] = {
                "kappa": float(stacked.kappa_hat),
                "sigma": float(stacked.kappa_sigma),
                "snr": float(stacked.snr),
                "max_weight_ratio": ratio,
            }
        except Exception as e:
            results_dict[name] = {"error": str(e)}

    return results_dict


def analyze_detector_subsets(
    results: List[PhaseLockResult],
    results_with_metadata: List[dict],
) -> dict:
    """Analyze robustness to detector selection.

    Parameters
    ----------
    results : list of PhaseLockResult
    results_with_metadata : list of dicts with event metadata including detectors

    Returns
    -------
    dict with subset analysis results
    """
    event_to_detectors = {}
    for r in results_with_metadata:
        event_to_detectors[r["event"]] = r.get("detectors_used", ["H1"])

    subsets = {
        "all": results,
    }

    h1_only = [r for r in results if event_to_detectors.get(r.event_name, []) == ["H1"]]
    if h1_only:
        subsets["H1_only"] = h1_only

    multi_det = [r for r in results if len(event_to_detectors.get(r.event_name, [])) > 1]
    if multi_det:
        subsets["multi_detector"] = multi_det

    three_det = [r for r in results if len(event_to_detectors.get(r.event_name, [])) == 3]
    if three_det:
        subsets["three_detector"] = three_det

    results_dict = {}

    for subset_name, subset_results in subsets.items():
        if len(subset_results) < 3:
            results_dict[subset_name] = {
                "n_events": len(subset_results),
                "error": "Too few events (< 3)",
            }
            continue

        try:
            stacked = stack_phase_locked(subset_results, max_weight_ratio=5.5)
            results_dict[subset_name] = {
                "n_events": len(subset_results),
                "kappa": float(stacked.kappa_hat),
                "sigma": float(stacked.kappa_sigma),
                "snr": float(stacked.snr),
            }
        except Exception as e:
            results_dict[subset_name] = {
                "n_events": len(subset_results),
                "error": str(e),
            }

    return results_dict


def run_sigma_quality_cuts(
    results: List[PhaseLockResult],
    cuts: Optional[List[float]] = None,
) -> dict:
    """Test IVW stack stability across sigma quality cuts.

    If kappa is stable when progressively removing noisy events,
    the result is not driven by noise artifacts.

    Parameters
    ----------
    results : list of PhaseLockResult
    cuts : sigma thresholds to test

    Returns
    -------
    dict with per-cut stacked results and overall stability assessment
    """
    if cuts is None:
        cuts = [0.05, 0.1, 0.5, 1.0, 5.0]

    sigmas = np.array([r.kappa_sigma for r in results])
    cut_results = {}

    for sigma_cut in cuts:
        mask = sigmas < sigma_cut
        n = int(np.sum(mask))
        if n < 3:
            cut_results[f"sigma_lt_{sigma_cut}"] = {
                "n_events": n,
                "error": "Too few events (< 3)",
            }
            continue

        subset = [r for r, m in zip(results, mask) if m]
        try:
            stacked = stack_phase_locked(subset, max_weight_ratio=None)
            cut_results[f"sigma_lt_{sigma_cut}"] = {
                "n_events": n,
                "kappa": float(stacked.kappa_hat),
                "sigma": float(stacked.kappa_sigma),
                "snr": float(stacked.kappa_hat / stacked.kappa_sigma)
                if stacked.kappa_sigma > 0
                else 0.0,
            }
        except Exception as e:
            cut_results[f"sigma_lt_{sigma_cut}"] = {
                "n_events": n,
                "error": str(e),
            }

    # Assess stability: compute range of kappa across valid cuts
    valid_kappas = [
        v["kappa"] for v in cut_results.values() if "kappa" in v
    ]
    if len(valid_kappas) >= 2:
        stability = {
            "kappa_range": float(max(valid_kappas) - min(valid_kappas)),
            "kappa_mean": float(np.mean(valid_kappas)),
            "kappa_std": float(np.std(valid_kappas)),
            "is_stable": (max(valid_kappas) - min(valid_kappas)) < 0.01,
        }
    else:
        stability = {"is_stable": False, "error": "Insufficient valid cuts"}

    return {"cuts": cut_results, "stability": stability}


def compute_influence_metrics(
    jackknife_result: dict,
    full_kappa: float,
    full_sigma: float,
) -> dict:
    """Compute detailed influence metrics from jackknife result.

    Returns
    -------
    dict with top influential events and concentration metrics
    """
    per_event = jackknife_result.get("per_event", [])
    if not per_event:
        return {
            "top_influential_events": [],
            "gini_coefficient": 0.0,
            "n_eff": jackknife_result.get("n_eff", 0),
        }

    shifts = np.array([abs(e["shift"]) for e in per_event])

    gini = compute_gini_coefficient(shifts)

    sorted_events = sorted(per_event, key=lambda e: abs(e["shift"]), reverse=True)
    top_events = []
    for e in sorted_events[:10]:
        top_events.append(
            {
                "event": e["removed_event"],
                "shift": e["shift"],
                "shift_in_sigma": abs(e["shift"]) / full_sigma if full_sigma > 0 else 0,
                "kappa_without": e["kappa_jack"],
            }
        )

    return {
        "top_influential_events": top_events,
        "gini_coefficient": gini,
        "n_eff": jackknife_result.get("n_eff", 0),
        "max_fractional_influence": jackknife_result.get("max_fractional_influence", 0),
    }


def assess_robustness(
    robustness_result: RobustnessResult,
    criteria: Optional[dict] = None,
) -> tuple[bool, float, List[str], List[str]]:
    """Assess overall robustness and compute a score.

    Parameters
    ----------
    robustness_result : RobustnessResult
    criteria : dict with thresholds for various tests

    Returns
    -------
    (is_robust, score, failures, caveats)
    """
    if criteria is None:
        criteria = {
            "max_gini": 0.5,
            "min_n_eff_fraction": 0.3,
            "max_bootstrap_bias_sigma": 0.5,
            "max_leave_k_range_sigma": 2.0,
            "weighting_consistency_sigma": 1.0,
            "min_detector_subset_agreement": 0.8,
        }

    failures = []
    caveats = []
    score_components = []

    if robustness_result.gini_coefficient > criteria["max_gini"]:
        failures.append(
            f"Gini coefficient {robustness_result.gini_coefficient:.3f} > "
            f"{criteria['max_gini']}: extreme influence concentration"
        )
        score_components.append(0.0)
    else:
        score_components.append(1.0 - robustness_result.gini_coefficient / criteria["max_gini"])

    min_n_eff = criteria["min_n_eff_fraction"] * robustness_result.n_events
    if robustness_result.n_eff < min_n_eff:
        failures.append(
            f"Effective sample size {robustness_result.n_eff:.1f} < "
            f"{min_n_eff:.1f}: too few independent events"
        )
        score_components.append(robustness_result.n_eff / min_n_eff * 0.5)
    else:
        score_components.append(1.0)

    if robustness_result.bootstrap_bias is not None:
        bias_sigma = abs(robustness_result.bootstrap_bias / robustness_result.full_sigma)
        if bias_sigma > criteria["max_bootstrap_bias_sigma"]:
            caveats.append(
                f"Bootstrap bias {bias_sigma:.2f} sigma indicates potential estimation bias"
            )
            score_components.append(1.0 - bias_sigma / (criteria["max_bootstrap_bias_sigma"] * 2))
        else:
            score_components.append(1.0)

    if robustness_result.leave_three_out is not None:
        l3o = robustness_result.leave_three_out
        if "error" not in l3o:
            range_sigma = l3o["range"] / robustness_result.full_sigma
            if range_sigma > criteria["max_leave_k_range_sigma"]:
                caveats.append(
                    f"Leave-3-out range {range_sigma:.2f} sigma indicates "
                    "sensitivity to event selection"
                )
                score_components.append(
                    1.0 - range_sigma / (criteria["max_leave_k_range_sigma"] * 2)
                )
            else:
                score_components.append(1.0)

    if robustness_result.weighting_schemes:
        # Only compare IVW variants for consistency. Equal weighting is
        # statistically invalid when per-event sigma spans orders of
        # magnitude — noisy events with |kappa| >> 1 dominate the average,
        # producing a spurious "disagreement" that is not a robustness
        # failure but an expected consequence of heterogeneous precision.
        ivw_kappas = []
        for name, res in robustness_result.weighting_schemes.items():
            if "error" not in res and name != "equal_weights":
                ivw_kappas.append(res["kappa"])

        if len(ivw_kappas) >= 2:
            kappa_range = max(ivw_kappas) - min(ivw_kappas)
            range_sigma = kappa_range / robustness_result.full_sigma
            if range_sigma > criteria["weighting_consistency_sigma"]:
                caveats.append(
                    f"IVW weighting schemes disagree by {range_sigma:.2f} sigma"
                )
                score_components.append(max(0, 1.0 - range_sigma / 2))
            else:
                score_components.append(1.0)

        # Report the equal-weight divergence as context, not a failure.
        # When per-event sigma spans orders of magnitude, the equal-weight
        # mean is dominated by noise in high-sigma events and is not a valid
        # comparator for IVW.
        eq = robustness_result.weighting_schemes.get("equal_weights", {})
        if "kappa" in eq and ivw_kappas:
            eq_divergence = abs(eq["kappa"] - np.mean(ivw_kappas))
            if eq_divergence / robustness_result.full_sigma > 5:
                caveats.append(
                    f"Equal-weight estimator diverges from IVW by "
                    f"{eq_divergence / robustness_result.full_sigma:.0f} sigma "
                    f"(expected when per-event sigma varies by orders of "
                    f"magnitude; not a robustness failure)"
                )

    # Sigma quality cut stability: if kappa is stable across quality cuts,
    # the result is not driven by noise artifacts in low-SNR events
    if robustness_result.sigma_quality_cuts:
        stability = robustness_result.sigma_quality_cuts.get("stability", {})
        if stability.get("is_stable", False):
            score_components.append(1.0)
        elif "kappa_range" in stability:
            range_sigma = stability["kappa_range"] / robustness_result.full_sigma
            if range_sigma < 1.0:
                score_components.append(0.8)
            else:
                caveats.append(
                    f"Sigma quality cuts show kappa varies by "
                    f"{range_sigma:.1f} sigma across event selection"
                )
                score_components.append(max(0, 1.0 - range_sigma / 3))

    if robustness_result.top_influential_events:
        max_influence = robustness_result.top_influential_events[0].get("shift_in_sigma", 0)
        if max_influence > 1.0:
            caveats.append(
                f"Top event shifts result by {max_influence:.2f} sigma "
                f"({robustness_result.top_influential_events[0]['event']})"
            )

    score = np.mean(score_components) if score_components else 0.0
    is_robust = len(failures) == 0 and score >= 0.5

    return is_robust, score, failures, caveats


def run_comprehensive_robustness(
    results: List[PhaseLockResult],
    results_with_metadata: List[dict],
    max_weight_ratio: Optional[float] = 5.5,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> RobustnessResult:
    """Run comprehensive robustness analysis.

    Parameters
    ----------
    results : list of PhaseLockResult
    results_with_metadata : list of dicts with event metadata
    max_weight_ratio : weight cap for primary analysis
    n_bootstrap : number of bootstrap samples
    seed : random seed

    Returns
    -------
    RobustnessResult with comprehensive diagnostics
    """
    from .jackknife import run_jackknife

    n_events = len(results)

    full_stacked = stack_phase_locked(results, max_weight_ratio=max_weight_ratio)

    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE ROBUSTNESS ANALYSIS")
    print(f"{'=' * 70}")
    print(f"N events: {n_events}")
    print(f"Full stack: kappa = {full_stacked.kappa_hat:.4f} ± {full_stacked.kappa_sigma:.4f}")
    print(f"{'=' * 70}\n")

    print("Running leave-one-out jackknife...")
    jack = run_jackknife(results, max_weight_ratio=max_weight_ratio)

    leave_one_out = {
        "mean": jack.jackknife_mean,
        "std": jack.jackknife_std,
        "max_shift": jack.max_shift,
        "max_shift_event": jack.max_shift_event,
        "is_stable": jack.is_stable,
        "n_eff": jack.n_eff,
        "influential_events": jack.influential_events,
        "per_event": [
            {
                "removed_event": name,
                "kappa_jack": float(k),
                "sigma_jack": float(s),
                "shift": float(k - jack.full_kappa),
            }
            for name, k, s in zip(
                jack.removed_event_names,
                jack.jackknife_kappas,
                jack.jackknife_sigmas,
            )
        ],
    }

    print("Running leave-2-out analysis...")
    leave_two_out = run_leave_k_out(
        results, k=2, max_weight_ratio=max_weight_ratio, n_combinations=100
    )

    print("Running leave-3-out analysis...")
    leave_three_out = run_leave_k_out(
        results, k=3, max_weight_ratio=max_weight_ratio, n_combinations=200
    )

    print("Running leave-5-out analysis...")
    leave_five_out = run_leave_k_out(
        results, k=5, max_weight_ratio=max_weight_ratio, n_combinations=200
    )

    print(f"Running bootstrap ({n_bootstrap} samples)...")
    bootstrap_result = run_bootstrap(
        results, max_weight_ratio=max_weight_ratio, n_bootstrap=n_bootstrap, seed=seed
    )

    print("Testing weighting schemes...")
    weighting_schemes = test_weighting_schemes(results)

    print("Analyzing detector subsets...")
    detector_subsets = analyze_detector_subsets(results, results_with_metadata)

    print("Running sigma quality cut analysis...")
    sigma_quality_cuts = run_sigma_quality_cuts(results)

    print("Computing influence metrics...")
    influence_metrics = compute_influence_metrics(
        leave_one_out, full_stacked.kappa_hat, full_stacked.kappa_sigma
    )

    robustness_result = RobustnessResult(
        n_events=n_events,
        full_kappa=full_stacked.kappa_hat,
        full_sigma=full_stacked.kappa_sigma,
        full_snr=full_stacked.snr,
        leave_one_out=leave_one_out,
        leave_two_out=leave_two_out,
        leave_three_out=leave_three_out,
        leave_five_out=leave_five_out,
        bootstrap_mean=bootstrap_result.get("mean"),
        bootstrap_std=bootstrap_result.get("std"),
        bootstrap_bias=bootstrap_result.get("bias"),
        bootstrap_n_samples=bootstrap_result.get("n_samples", 0),
        weighting_schemes=weighting_schemes,
        detector_subsets=detector_subsets,
        sigma_quality_cuts=sigma_quality_cuts,
        top_influential_events=influence_metrics["top_influential_events"],
        gini_coefficient=influence_metrics["gini_coefficient"],
        n_eff=influence_metrics["n_eff"],
    )

    print("\nAssessing robustness...")
    is_robust, score, failures, caveats = assess_robustness(robustness_result)

    robustness_result.is_robust = is_robust
    robustness_result.robustness_score = score
    robustness_result.failures = failures
    robustness_result.caveats = caveats

    return robustness_result


def print_robustness_summary(result: RobustnessResult) -> None:
    """Print human-readable robustness summary."""
    print(f"\n{'=' * 70}")
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nFull stack: kappa = {result.full_kappa:.4f} ± {result.full_sigma:.4f}")
    print(f"N events: {result.n_events}  (N_eff = {result.n_eff:.1f})")
    print(f"Gini coefficient: {result.gini_coefficient:.3f}")
    print(f"\n{'=' * 70}")

    print("\nLEAVE-K-OUT ANALYSIS:")
    l1o = result.leave_one_out
    print(
        f"  Leave-1-out: mean={l1o['mean']:.4f}, std={l1o['std']:.4f}, "
        f"max_shift={l1o['max_shift']:.4f} ({l1o['max_shift_event']})"
    )

    if result.leave_two_out and "error" not in result.leave_two_out:
        l2o = result.leave_two_out
        print(
            f"  Leave-2-out: mean={l2o['mean']:.4f}, std={l2o['std']:.4f}, range={l2o['range']:.4f}"
        )

    if result.leave_three_out and "error" not in result.leave_three_out:
        l3o = result.leave_three_out
        print(
            f"  Leave-3-out: mean={l3o['mean']:.4f}, std={l3o['std']:.4f}, range={l3o['range']:.4f}"
        )

    print("\nBOOTSTRAP ANALYSIS:")
    if result.bootstrap_mean is not None:
        print(f"  Mean: {result.bootstrap_mean:.4f}")
        print(f"  Std:  {result.bootstrap_std:.4f}")
        print(
            f"  Bias: {result.bootstrap_bias:.4f} "
            f"({result.bootstrap_bias / result.full_sigma:.2f} sigma)"
        )

    print("\nWEIGHTING SCHEMES:")
    for name, res in result.weighting_schemes.items():
        if "error" not in res:
            print(f"  {name:<20}: kappa={res['kappa']:+.4f} ± {res['sigma']:.4f}")

    print("\nDETECTOR SUBSETS:")
    for name, res in result.detector_subsets.items():
        if "error" not in res:
            print(
                f"  {name:<20}: N={res['n_events']}, kappa={res['kappa']:+.4f} ± {res['sigma']:.4f}"
            )

    if result.sigma_quality_cuts:
        print("\nSIGMA QUALITY CUTS (IVW, no cap):")
        cuts = result.sigma_quality_cuts.get("cuts", {})
        for name, res in cuts.items():
            if "error" not in res:
                print(
                    f"  {name:<20}: N={res['n_events']}, "
                    f"kappa={res['kappa']:+.4f} ± {res['sigma']:.4f} "
                    f"({res['snr']:.2f} sigma)"
                )
        stability = result.sigma_quality_cuts.get("stability", {})
        if "kappa_range" in stability:
            print(
                f"  Stability: range={stability['kappa_range']:.4f}, "
                f"stable={stability['is_stable']}"
            )

    print("\nTOP INFLUENTIAL EVENTS:")
    for i, ev in enumerate(result.top_influential_events[:5], 1):
        print(f"  {i}. {ev['event']}: shift={ev['shift']:+.4f} ({ev['shift_in_sigma']:.2f} sigma)")

    print(f"\n{'=' * 70}")
    print("OVERALL ASSESSMENT:")
    print(f"  Robustness score: {result.robustness_score:.2f}/1.00")
    print(f"  Status: {'ROBUST' if result.is_robust else 'NOT ROBUST'}")

    if result.failures:
        print("\nFAILURES:")
        for f in result.failures:
            print(f"  ✗ {f}")

    if result.caveats:
        print("\nCAVEATS:")
        for c in result.caveats:
            print(f"  ⚠ {c}")

    print(f"{'=' * 70}\n")
