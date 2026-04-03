#!/usr/bin/env python3
"""
GRIM-S Phase 2.5: Colored-noise + t_start marginalization re-analysis.

Re-runs the full 134-event catalog with:
  1. Frequency-domain colored-noise likelihood (uses actual PSD)
  2. Ringdown start time marginalization over [5M, 8M, 10M, 12M, 15M, 20M]
  3. Jackknife stability test
  4. Comparison against original white-noise results
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
RESULTS_PATH = DATA_DIR / "full_catalog_colored_results.json"
JACKKNIFE_PATH = DATA_DIR / "colored_jackknife.json"


def run_colored_analysis():
    """Run the colored-noise + marginalized analysis on the full catalog."""
    from bown_instruments.grims.mass_analysis import run_mass_analysis
    from bown_instruments.grims.phase_locked_search import PhaseLockResult

    print("=" * 70)
    print("GRIM-S Phase 2.5: Colored-Noise + t_start Marginalization")
    print("=" * 70)
    print()

    # Run with colored noise + t_start marginalization
    start = time.time()
    results = run_mass_analysis(
        data_dir=str(DATA_DIR),
        catalog_path=str(DATA_DIR / "gwtc_full_catalog.json"),
        min_total_mass=30.0,
        use_colored=True,
        verbose=True,
    )
    elapsed = time.time() - start

    print(f"\nAnalysis completed in {elapsed:.1f}s")

    # Stacked result
    stacked = results["stacked"]
    if stacked:
        print(f"\n{'=' * 70}")
        print(f"COLORED-NOISE STACKED RESULT ({stacked.n_events} events):")
        print(f"  kappa = {stacked.kappa_hat:+.4f} +/- {stacked.kappa_sigma:.4f}")
        print(f"  SNR   = {stacked.snr:.2f}")
        print(
            f"  significance = {abs(stacked.kappa_hat) / stacked.kappa_sigma:.1f} sigma"
        )
        print(f"{'=' * 70}")

        # Compare with original white-noise result
        original_path = DATA_DIR / "full_catalog_results.json"
        if original_path.exists():
            with open(original_path) as f:
                orig = json.load(f)
            orig_stacked = orig.get("stacked", {})
            orig_kappa = orig_stacked.get("kappa_hat", 0)
            orig_sigma = orig_stacked.get("kappa_sigma", float("inf"))
            orig_snr = orig_stacked.get("snr", 0)

            sigma_improvement = (
                (1 - stacked.kappa_sigma / orig_sigma) * 100 if orig_sigma > 0 else 0
            )

            print(f"\n{'=' * 70}")
            print("COMPARISON: Colored vs White Noise")
            print(f"{'=' * 70}")
            print(
                f"  White noise:  κ = {orig_kappa:+.4f} +/- {orig_sigma:.4f}  SNR = {orig_snr:.2f}"
            )
            print(
                f"  Colored noise: κ = {stacked.kappa_hat:+.4f} +/- {stacked.kappa_sigma:.4f}  SNR = {stacked.snr:.2f}"
            )
            print(f"  Error bar change: {sigma_improvement:+.1f}%")
            print(f"{'=' * 70}")

    # Save results (strip non-serializable objects)
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
        "method": "colored_noise_tstart_marginalized",
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


def run_jackknife_on_results(results):
    """Run jackknife on the colored-noise results."""
    from bown_instruments.grims.jackknife import run_jackknife, print_jackknife_summary
    from bown_instruments.grims.phase_locked_search import PhaseLockResult

    print(f"\n{'=' * 70}")
    print("JACKKNIFE: Colored-Noise Stack Stability Test")
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

    jack = run_jackknife(phase_lock_results)
    print_jackknife_summary(jack)

    # Save jackknife results
    jack_data = {
        "full_kappa": jack.full_kappa,
        "full_sigma": jack.full_sigma,
        "full_snr": jack.full_snr,
        "jackknife_mean": jack.jackknife_mean,
        "jackknife_std": jack.jackknife_std,
        "max_shift": jack.max_shift,
        "max_shift_event": jack.max_shift_event,
        "is_stable": jack.is_stable,
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

    with open(JACKKNIFE_PATH, "w") as f:
        json.dump(jack_data, f, indent=2, default=str)
    print(f"\nJackknife results saved to {JACKKNIFE_PATH}")

    return jack


def generate_comparison_plots(results_colored, results_white=None):
    """Generate comparison plots between colored and white noise results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    kappas = np.array([r["kappa_hat"] for r in results_colored["individual"]])
    sigmas = np.array([r["kappa_sigma"] for r in results_colored["individual"]])
    snrs = np.array([r["snr_event"] for r in results_colored["individual"]])
    names = [r["event"] for r in results_colored["individual"]]
    best_t = np.array(
        [r.get("best_t_start_m", 10) for r in results_colored["individual"]]
    )

    # Plot 1: Per-event kappa with colored noise
    order = np.argsort(snrs)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, len(kappas) * 0.25)))
    y_pos = np.arange(len(kappas))
    ax.errorbar(
        kappas[order],
        y_pos,
        xerr=sigmas[order],
        fmt="o",
        color="steelblue",
        capsize=3,
        alpha=0.8,
        markersize=4,
    )
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5, label="kappa=0")
    ax.axvspan(0.01, 0.05, alpha=0.1, color="red", label="NR range (~0.01-0.04)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("kappa (colored noise, t_start marginalized)", fontsize=11)
    ax.set_title(
        f"Per-Event Kappa — Colored Noise ({len(kappas)} events)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "colored_per_event.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Best t_start vs SNR
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        snrs,
        best_t,
        c=kappas,
        cmap="RdBu_r",
        s=40,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label="kappa")
    ax.set_xlabel("Event SNR", fontsize=12)
    ax.set_ylabel("Best t_start (M)", fontsize=12)
    ax.set_title(
        "Ringdown Start Time Preference vs SNR", fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "colored_tstart_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Error bar comparison if white results available
    if results_white:
        white_kappas = np.array([r["kappa_hat"] for r in results_white["individual"]])
        white_sigmas = np.array([r["kappa_sigma"] for r in results_white["individual"]])

        # Match events
        common = set(names) & set([r["event"] for r in results_white["individual"]])
        if len(common) > 5:
            white_map = {r["event"]: r for r in results_white["individual"]}
            common_kappa_c = []
            common_sigma_c = []
            common_sigma_w = []
            common_names = []

            for r in results_colored["individual"]:
                if r["event"] in white_map:
                    common_kappa_c.append(r["kappa_hat"])
                    common_sigma_c.append(r["kappa_sigma"])
                    common_sigma_w.append(white_map[r["event"]]["kappa_sigma"])
                    common_names.append(r["event"])

            common_sigma_c = np.array(common_sigma_c)
            common_sigma_w = np.array(common_sigma_w)
            ratio = common_sigma_c / common_sigma_w

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(
                ratio,
                bins=30,
                color="steelblue",
                alpha=0.7,
                edgecolor="k",
                linewidth=0.5,
            )
            ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="No change")
            ax.axvline(
                0.7, color="green", linestyle="--", alpha=0.7, label="30% improvement"
            )
            ax.set_xlabel("Colored noise error / White noise error", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(
                f"Error Bar Ratio: Colored vs White Noise ({len(common)} events)",
                fontsize=13,
                fontweight="bold",
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            mean_ratio = np.mean(ratio)
            ax.text(
                0.95,
                0.95,
                f"Mean ratio: {mean_ratio:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )
            plt.tight_layout()
            plt.savefig(
                PLOTS_DIR / "colored_vs_white_errorbars.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    print(f"\nPlots saved to {PLOTS_DIR}/colored_*.png")


if __name__ == "__main__":
    # Step 1: Run colored-noise analysis
    results_colored = run_colored_analysis()

    # Step 2: Jackknife
    jack = run_jackknife_on_results(results_colored)

    # Step 3: Load white-noise results for comparison
    results_white = None
    white_path = DATA_DIR / "full_catalog_results.json"
    if white_path.exists():
        with open(white_path) as f:
            results_white = json.load(f)

    # Step 4: Comparison plots
    generate_comparison_plots(results_colored, results_white)

    print(f"\n{'=' * 70}")
    print("PHASE 2.5 COMPLETE.")
    print(f"{'=' * 70}")
