#!/usr/bin/env python3
"""
GRIM-S Phase 3: Multi-detector + adaptive segment analysis.

Improvements over Phase 2.5:
  1. Multi-detector coherent stacking (H1 + L1 + V1 where available)
  2. Per-event optimal segment length (3-5x tau_220 instead of fixed 0.15s)

Run validation on subset first, then full catalog.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_PATH = DATA_DIR / "phase3_results.json"
JACKKNIFE_PATH = DATA_DIR / "phase3_jackknife.json"
BASELINE_PATH = DATA_DIR / "full_catalog_colored_results.json"


def run_subset_validation():
    """Validate improvements on top-20 SNR events before full run."""
    from bown_instruments.grims.mass_analysis import run_mass_analysis

    print("=" * 70)
    print("PHASE 3 VALIDATION: Top-20 SNR events")
    print("=" * 70)

    # Load catalog, pick top 20 by SNR
    with open(DATA_DIR / "gwtc_full_catalog.json") as f:
        catalog = json.load(f)

    high_snr = sorted(catalog, key=lambda e: e.get("snr", 0), reverse=True)[:20]
    min_mass = min(e["total_mass"] for e in high_snr) - 1.0

    # Run Phase 2.5 baseline (H1-only, fixed segment)
    print("\n--- Baseline: H1-only, fixed segment ---")
    baseline = run_mass_analysis(
        data_dir=str(DATA_DIR),
        catalog_path=str(DATA_DIR / "gwtc_full_catalog.json"),
        min_total_mass=min_mass,
        use_colored=True,
        multi_detector=False,
        adaptive_segment=False,
        verbose=False,
    )
    # Filter to our top-20
    top_names = {e["name"] for e in high_snr}
    baseline_results = [r for r in baseline["individual"] if r["event"] in top_names]

    if baseline["stacked"]:
        print(
            f"  Stacked ({len(baseline_results)} events): "
            f"kappa = {baseline['stacked'].kappa_hat:+.4f} +/- {baseline['stacked'].kappa_sigma:.4f}"
        )

    # Run with multi-detector only
    print("\n--- Multi-detector, fixed segment ---")
    multi_det = run_mass_analysis(
        data_dir=str(DATA_DIR),
        catalog_path=str(DATA_DIR / "gwtc_full_catalog.json"),
        min_total_mass=min_mass,
        use_colored=True,
        multi_detector=True,
        adaptive_segment=False,
        verbose=False,
    )
    multi_results = [r for r in multi_det["individual"] if r["event"] in top_names]

    if multi_det["stacked"]:
        n_multi = sum(1 for r in multi_results if r.get("n_detectors", 1) > 1)
        print(
            f"  Stacked ({len(multi_results)} events, {n_multi} multi-det): "
            f"kappa = {multi_det['stacked'].kappa_hat:+.4f} +/- {multi_det['stacked'].kappa_sigma:.4f}"
        )

    # Run with adaptive segment only
    print("\n--- H1-only, adaptive segment ---")
    adaptive = run_mass_analysis(
        data_dir=str(DATA_DIR),
        catalog_path=str(DATA_DIR / "gwtc_full_catalog.json"),
        min_total_mass=min_mass,
        use_colored=True,
        multi_detector=False,
        adaptive_segment=True,
        verbose=False,
    )
    adaptive_results = [r for r in adaptive["individual"] if r["event"] in top_names]

    if adaptive["stacked"]:
        print(
            f"  Stacked ({len(adaptive_results)} events): "
            f"kappa = {adaptive['stacked'].kappa_hat:+.4f} +/- {adaptive['stacked'].kappa_sigma:.4f}"
        )

    # Run with both improvements
    print("\n--- Multi-detector + adaptive segment ---")
    both = run_mass_analysis(
        data_dir=str(DATA_DIR),
        catalog_path=str(DATA_DIR / "gwtc_full_catalog.json"),
        min_total_mass=min_mass,
        use_colored=True,
        multi_detector=True,
        adaptive_segment=True,
        verbose=False,
    )
    both_results = [r for r in both["individual"] if r["event"] in top_names]

    if both["stacked"]:
        n_multi = sum(1 for r in both_results if r.get("n_detectors", 1) > 1)
        print(
            f"  Stacked ({len(both_results)} events, {n_multi} multi-det): "
            f"kappa = {both['stacked'].kappa_hat:+.4f} +/- {both['stacked'].kappa_sigma:.4f}"
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY (top-20 events)")
    print(f"{'=' * 70}")

    configs = [
        ("Baseline (H1, fixed seg)", baseline),
        ("Multi-det, fixed seg", multi_det),
        ("H1, adaptive seg", adaptive),
        ("Multi-det + adaptive", both),
    ]

    baseline_sigma = baseline["stacked"].kappa_sigma if baseline["stacked"] else float("inf")

    for name, res in configs:
        if res["stacked"]:
            s = res["stacked"]
            improvement = (1 - s.kappa_sigma / baseline_sigma) * 100 if baseline_sigma > 0 else 0
            print(
                f"  {name:<30} kappa={s.kappa_hat:+.4f} +/- {s.kappa_sigma:.4f}  "
                f"sigma improvement: {improvement:+.1f}%"
            )

    print()

    return both


def run_full_analysis():
    """Run the full catalog with all Phase 3 improvements."""
    from bown_instruments.grims.mass_analysis import run_mass_analysis

    print("=" * 70)
    print("GRIM-S Phase 3: Full Catalog — Multi-Detector + Adaptive Segment")
    print("=" * 70)
    print()

    start = time.time()
    results = run_mass_analysis(
        data_dir=str(DATA_DIR),
        catalog_path=str(DATA_DIR / "gwtc_full_catalog.json"),
        min_total_mass=30.0,
        use_colored=True,
        multi_detector=True,
        adaptive_segment=True,
        max_weight_ratio=5.5,
        verbose=True,
    )
    elapsed = time.time() - start

    print(f"\nAnalysis completed in {elapsed:.1f}s")

    stacked = results["stacked"]
    if stacked:
        n_multi = sum(1 for r in results["individual"] if r.get("n_detectors", 1) > 1)
        total_det_measurements = sum(r.get("n_detectors", 1) for r in results["individual"])
        avg_seg = np.mean([r.get("seg_duration", 0.15) for r in results["individual"]]) * 1000

        print(f"\n{'=' * 70}")
        print(f"PHASE 3 STACKED RESULT ({stacked.n_events} events):")
        print(f"  kappa = {stacked.kappa_hat:+.4f} +/- {stacked.kappa_sigma:.4f}")
        print(f"  SNR   = {stacked.snr:.2f}")
        print(f"  significance = {abs(stacked.kappa_hat) / stacked.kappa_sigma:.1f} sigma")
        print(f"  multi-detector events: {n_multi}")
        print(f"  total detector measurements: {total_det_measurements}")
        print(f"  mean segment duration: {avg_seg:.1f} ms")
        print(f"{'=' * 70}")

        # Compare with Phase 2.5 baseline
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH) as f:
                baseline = json.load(f)
            b_kappa = baseline["stacked"]["kappa_hat"]
            b_sigma = baseline["stacked"]["kappa_sigma"]
            b_snr = baseline["stacked"]["snr"]
            b_n = baseline["stacked"]["n_events"]

            sigma_improvement = (1 - stacked.kappa_sigma / b_sigma) * 100

            print(f"\n{'=' * 70}")
            print("COMPARISON: Phase 3 vs Phase 2.5")
            print(f"{'=' * 70}")
            print(f"  Phase 2.5: kappa={b_kappa:+.4f} +/- {b_sigma:.4f}  SNR={b_snr:.2f}  N={b_n}")
            print(
                f"  Phase 3:   kappa={stacked.kappa_hat:+.4f} +/- {stacked.kappa_sigma:.4f}  "
                f"SNR={stacked.snr:.2f}  N={stacked.n_events}"
            )
            print(f"  Sigma improvement: {sigma_improvement:+.1f}%")
            print(f"{'=' * 70}")

    # Save results
    serializable = {
        "individual": [
            {k: v for k, v in r.items() if k != "result"} for r in results["individual"]
        ],
        "stacked": {
            "kappa_hat": stacked.kappa_hat if stacked else 0,
            "kappa_sigma": stacked.kappa_sigma if stacked else float("inf"),
            "snr": stacked.snr if stacked else 0,
            "n_events": stacked.n_events if stacked else 0,
        },
        "n_analyzed": results["n_analyzed"],
        "n_skipped": results["n_skipped"],
        "method": "phase3_multidet_adaptive",
        "improvements": [
            "multi-detector coherent stacking (H1+L1+V1)",
            "per-event adaptive segment length (5x tau_220)",
        ],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


def run_jackknife(results):
    """Run jackknife on Phase 3 results."""
    from bown_instruments.grims.jackknife import run_jackknife, print_jackknife_summary
    from bown_instruments.grims.phase_locked_search import PhaseLockResult

    print(f"\n{'=' * 70}")
    print("JACKKNIFE: Phase 3 Stack Stability Test")
    print(f"{'=' * 70}")

    phase_lock_results = []
    for r in results["individual"]:
        res = r["result"]
        phase_lock_results.append(
            PhaseLockResult(
                event_name=r["event"],
                kappa_hat=r["kappa_hat"],
                kappa_sigma=r["kappa_sigma"],
                snr=r["snr_nl"],
                a_220_fit=r["a_220_fit"],
                phi_220_fit=res.phi_220_fit,
                template_norm=res.template_norm,
                residual_overlap=r["kappa_hat"] * res.template_norm**2,
                noise_rms=r["noise_rms"],
            )
        )

    if len(phase_lock_results) < 3:
        print("Too few events for jackknife.")
        return None

    jack = run_jackknife(phase_lock_results, max_weight_ratio=5.5)
    print_jackknife_summary(jack)

    jack_data = {
        "full_kappa": jack.full_kappa,
        "full_sigma": jack.full_sigma,
        "full_snr": jack.full_snr,
        "jackknife_mean": jack.jackknife_mean,
        "jackknife_std": jack.jackknife_std,
        "max_shift": jack.max_shift,
        "max_shift_event": jack.max_shift_event,
        "is_stable": jack.is_stable,
        "n_eff": jack.n_eff,
        "max_fractional_influence": jack.max_fractional_influence,
        "influential_events": jack.influential_events,
        "max_weight_ratio": 5.5,
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

    with open(JACKKNIFE_PATH, "w") as f:
        json.dump(jack_data, f, indent=2, default=str)
    print(f"\nJackknife saved to {JACKKNIFE_PATH}")

    return jack


def generate_plots(results):
    """Generate Phase 3 diagnostic plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    individual = results["individual"]
    kappas = np.array([r["kappa_hat"] for r in individual])
    sigmas = np.array([r["kappa_sigma"] for r in individual])
    snrs = np.array([r["snr_event"] for r in individual])
    names = [r["event"] for r in individual]
    n_dets = np.array([r.get("n_detectors", 1) for r in individual])
    seg_ms = np.array([r.get("seg_duration", 0.15) * 1000 for r in individual])

    # Plot 1: Per-event kappa, colored by number of detectors
    order = np.argsort(snrs)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, len(kappas) * 0.25)))
    y_pos = np.arange(len(kappas))
    colors = [
        "steelblue" if n_dets[i] == 1 else "darkorange" if n_dets[i] == 2 else "forestgreen"
        for i in order
    ]
    for j, i in enumerate(order):
        ax.errorbar(
            kappas[i],
            j,
            xerr=sigmas[i],
            fmt="o",
            color=colors[j],
            capsize=3,
            alpha=0.8,
            markersize=4,
        )
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvspan(0.01, 0.05, alpha=0.1, color="red", label="NR range (~0.01-0.04)")

    # Legend for detector count
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="steelblue",
            label="1 detector",
            markersize=6,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="darkorange",
            label="2 detectors",
            markersize=6,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="forestgreen",
            label="3 detectors",
            markersize=6,
            linestyle="None",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order], fontsize=6)
    ax.set_xlabel("kappa (Phase 3: multi-det + adaptive segment)", fontsize=11)
    ax.set_title(
        f"Per-Event Kappa — Phase 3 ({len(kappas)} events)", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase3_per_event.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Segment duration vs remnant mass
    masses = np.array([r["mass"] for r in individual])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        masses, seg_ms, c=n_dets, cmap="viridis", s=30, alpha=0.7, edgecolors="k", linewidths=0.3
    )
    plt.colorbar(sc, ax=ax, label="# detectors")
    ax.set_xlabel("Remnant Mass (Msun)", fontsize=12)
    ax.set_ylabel("Segment Duration (ms)", fontsize=12)
    ax.set_title("Adaptive Segment Duration vs Mass", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase3_segment_vs_mass.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Sigma improvement from multi-detector
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        baseline_map = {r["event"]: r for r in baseline["individual"]}
        common_events = [r for r in individual if r["event"] in baseline_map]

        if common_events:
            sigma_baseline = np.array(
                [baseline_map[r["event"]]["kappa_sigma"] for r in common_events]
            )
            sigma_phase3 = np.array([r["kappa_sigma"] for r in common_events])
            ratio = sigma_phase3 / sigma_baseline
            n_det_common = np.array([r.get("n_detectors", 1) for r in common_events])

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            ax = axes[0]
            ax.hist(ratio, bins=30, color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.5)
            ax.axvline(1.0, color="red", linestyle="--", label="No change")
            ax.axvline(
                np.median(ratio),
                color="green",
                linestyle="--",
                label=f"Median: {np.median(ratio):.3f}",
            )
            ax.set_xlabel("Phase 3 sigma / Phase 2.5 sigma", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("Per-Event Error Ratio", fontsize=13, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            colors_scatter = ["darkorange" if n > 1 else "steelblue" for n in n_det_common]
            ax.scatter(
                sigma_baseline,
                sigma_phase3,
                c=colors_scatter,
                s=30,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.3,
            )
            lim = max(sigma_baseline.max(), sigma_phase3.max()) * 1.1
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="No change")
            ax.set_xlabel("Phase 2.5 sigma", fontsize=12)
            ax.set_ylabel("Phase 3 sigma", fontsize=12)
            ax.set_title("Per-Event Sigma Comparison", fontsize=13, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "phase3_vs_baseline.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"\nPlots saved to {PLOTS_DIR}/phase3_*.png")


if __name__ == "__main__":
    # Step 1: Validate on subset
    run_subset_validation()

    # Step 2: Full catalog
    results = run_full_analysis()

    # Step 3: Jackknife
    run_jackknife(results)

    # Step 4: Plots
    generate_plots(results)

    print(f"\n{'=' * 70}")
    print("PHASE 3 COMPLETE.")
    print(f"{'=' * 70}")
