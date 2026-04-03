#!/usr/bin/env python3
"""End-to-end demonstration: chime output → atmospheric sensitivity assessment.

Shows how an astronomer would use chime output to:
  1. Identify trustworthy wavelength regions
  2. Estimate which molecules are detectable
  3. Feed quality weights into a retrieval framework

Usage:
    python examples/end_to_end_demo.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_best_observation(json_path: str) -> dict:
    """Load the observation with the most in-transit integrations."""
    with open(json_path) as f:
        results = json.load(f)
    return max(results["observations"], key=lambda o: o["n_in_transit"])


def main():
    # Prefer checked-in example results, then fall back to locally generated output.
    candidate_paths = [
        Path(__file__).parent.parent / "results" / "wasp39_fit" / "chime_wasp_39.json",
        Path(__file__).parent.parent / "chime_output" / "wasp39_v2" / "chime_wasp_39.json",
        Path(__file__).parent.parent / "chime_output" / "wasp39" / "chime_wasp_39.json",
        Path("chime_output/wasp39/chime_wasp_39.json"),
    ]
    json_path = next((path for path in candidate_paths if path.exists()), None)
    if json_path is None:
        print("Cannot find example results.")
        print("Run `chime WASP-39 --outdir chime_output/wasp39` first.")
        sys.exit(1)

    outdir = json_path.parent / "end_to_end"
    outdir.mkdir(exist_ok=True)

    obs = load_best_observation(str(json_path))
    bins = obs["channel_map"]["bins"]
    weights = obs["channel_map"]["weights"]

    print("=" * 60)
    print("  chime End-to-End Demo: WASP-39b")
    print("=" * 60)
    print(f"  Observation: {obs['obs_id']}")
    print(f"  In-transit: {obs['n_in_transit']}")
    print()

    # ================================================================
    # Step 1: Channel quality map
    # ================================================================
    print("  STEP 1: Channel Quality Assessment")
    print("  " + "-" * 40)

    wl = np.array([b["wl_center"] for b in bins])
    scatter = np.array([b["scatter_ppm"] for b in bins])
    excess = np.array([b["systematic_excess"] for b in bins])
    allan = np.array([b["allan_worst_ratio"] for b in bins])
    grades = [b["grade"] for b in bins]

    print(
        f"  Noise range: {scatter.min():.0f} — {scatter.max():.0f} ppm "
        f"({scatter.max() / scatter.min():.0f}× variation)"
    )
    print(
        f"  Grades: A={grades.count('A')} B={grades.count('B')} "
        f"C={grades.count('C')} D={grades.count('D')}"
    )

    # Correlated noise regions (Allan > 2)
    correlated = [
        (b["wl_center"], b["allan_worst_ratio"], b["scatter_ppm"])
        for b in bins
        if b["allan_worst_ratio"] > 2
    ]
    if correlated:
        print(f"\n  CORRELATED NOISE (Allan ratio > 2):")
        for wl_c, a, s in correlated:
            print(f"    {wl_c:.3f} µm: Allan = {a:.1f}×, scatter = {s:.0f} ppm")
            print(f"      → Averaging does NOT improve this wavelength")
            print(f"      → More data won't help — systematic floor")

    # ================================================================
    # Step 2: Trust regions
    # ================================================================
    print(f"\n  STEP 2: Trust Regions")
    print("  " + "-" * 40)

    regions = obs.get("trust_regions", [])
    if regions:
        for r in regions:
            print(
                f"    {r['wl_min']:.2f} — {r['wl_max']:.2f} µm "
                f"(mean scatter {r['mean_scatter_ppm']:.0f} ppm)"
            )
    else:
        print("    No trust regions found")

    # ================================================================
    # Step 3: Atmospheric sensitivity
    # ================================================================
    print(f"\n  STEP 3: Atmospheric Sensitivity")
    print("  " + "-" * 40)

    mol_bands = [
        (1.30, 1.50, "H2O 1.4µm", 200),
        (1.75, 2.05, "H2O 1.9µm", 300),
        (2.50, 2.90, "H2O 2.7µm", 500),
        (3.20, 3.50, "CH4 3.3µm", 200),
        (4.15, 4.45, "CO2 4.3µm", 400),
        (4.50, 4.80, "CO 4.6µm", 150),
    ]

    n_in = obs.get("n_in_transit", 1)

    print(f"  {'Molecule':<14s} {'Wavelength':<12s} {'3σ Limit':<12s} {'Expected':<10s} {'Status'}")
    print(f"  {'-' * 60}")

    for lo, hi, name, expected in mol_bands:
        band_mask = (wl >= lo) & (wl <= hi)
        if np.any(band_mask):
            band_scatter = np.median(scatter[band_mask])
            # Detection limit: scatter / sqrt(n_integrations) * 3
            limit = 3 * band_scatter / np.sqrt(n_in)
            status = "DETECTABLE" if expected > limit else "below noise"
            print(
                f"  {name:<14s} {lo:.2f}-{hi:.2f} µm  {limit:>8.0f} ppm  "
                f"{expected:>6.0f} ppm  {status}"
            )
        else:
            print(f"  {name:<14s} {lo:.2f}-{hi:.2f} µm  no data")

    # ================================================================
    # Step 4: Recommended weights for retrieval
    # ================================================================
    print(f"\n  STEP 4: Recommended Retrieval Weights")
    print("  " + "-" * 40)

    w = np.array(weights)
    active = w > 0
    print(f"  Active bins: {np.sum(active)}/{len(w)}")
    print(f"  Weight range: {w[active].min():.6f} — {w[active].max():.6f}")
    print(
        f"  Top 5 weighted wavelengths: "
        f"{', '.join(f'{wl[i]:.2f}µm ({w[i]:.4f})' for i in np.argsort(w)[-5:])}"
    )

    # ================================================================
    # Step 5: Diversity combining
    # ================================================================
    div = obs.get("diversity", {})
    if div:
        print(f"\n  STEP 5: Diversity Combining")
        print("  " + "-" * 40)
        print(
            f"  Diversity depth: {div['diversity_depth_ppm']:.0f} ± "
            f"{div['diversity_err_ppm']:.0f} ppm"
        )
        print(f"  Naive depth: {div['naive_depth_ppm']:.0f} ± {div['naive_err_ppm']:.0f} ppm")
        print(f"  Improvement: {div['improvement_factor']:.2f}×")

    # ================================================================
    # Summary plot
    # ================================================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor("#0a0e17")

    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#1e293b")

    grade_colors = {"A": "#34d399", "B": "#38bdf8", "C": "#fbbf24", "D": "#f87171"}
    bar_colors = [grade_colors[g] for g in grades]
    bar_w = np.diff(wl, append=wl[-1] * 1.05) * 0.8

    # Panel 1: Scatter
    ax = axes[0]
    ax.semilogy(
        wl, scatter, "o-", color="#f87171", markersize=4, linewidth=1.2, label=f"Empirical scatter"
    )
    # Highlight correlated noise
    for wl_c, a, s in correlated:
        ax.axvspan(wl_c - 0.03, wl_c + 0.03, alpha=0.3, color="#f87171")
    if correlated:
        ax.axvspan(0, 0, alpha=0.3, color="#f87171", label="Correlated noise (Allan>2)")
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Scatter (ppm)", color="#e2e8f0")
    ax.set_title("Channel Noise Map: WASP-39b", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor="#e2e8f0", facecolor="#111827", edgecolor="#1e293b"
    )

    # Panel 2: Systematic excess
    ax = axes[1]
    ax.bar(wl, excess, width=bar_w, color=bar_colors, alpha=0.7)
    ax.axhline(1, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(2, color="#fbbf24", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Systematic Excess", color="#e2e8f0")
    ax.set_title("Systematic Excess (scatter / pipeline noise floor)", color="#e2e8f0", fontsize=11)

    # Panel 3: Allan ratio
    ax = axes[2]
    allan_colors = ["#34d399" if a < 1.5 else "#fbbf24" if a < 3 else "#f87171" for a in allan]
    ax.bar(wl, np.minimum(allan, 50), width=bar_w, color=allan_colors, alpha=0.7)
    ax.axhline(
        1.5,
        color="#34d399",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        label="White noise threshold",
    )
    ax.axhline(
        2.0,
        color="#f87171",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        label="Correlated noise threshold",
    )
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Allan Ratio", color="#e2e8f0")
    ax.set_title("Allan Deviation — Does Averaging Help?", color="#e2e8f0", fontsize=11)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor="#e2e8f0", facecolor="#111827", edgecolor="#1e293b"
    )

    # Panel 4: Weights
    ax = axes[3]
    ax.bar(wl, w, width=bar_w, color=bar_colors, alpha=0.7)
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Diversity Weight", color="#e2e8f0")
    ax.set_title("Quality-Based Combining Weights (1/scatter²)", color="#e2e8f0", fontsize=11)

    fig.suptitle(
        "chime End-to-End: WASP-39b Channel Quality Assessment",
        color="#94a3b8",
        fontsize=10,
        y=0.995,
    )
    plt.savefig(
        str(outdir / "end_to_end_wasp39.png"), dpi=150, bbox_inches="tight", facecolor="#0a0e17"
    )
    plt.close()
    print(f"\n  Plot: {outdir / 'end_to_end_wasp39.png'}")

    # Save summary
    summary = {
        "target": "WASP-39b",
        "observation": obs["obs_id"],
        "n_integrations": obs["n_integrations"],
        "n_in_transit": obs["n_in_transit"],
        "noise_range_ppm": [float(scatter.min()), float(scatter.max())],
        "scatter_ratio": float(scatter.max() / scatter.min()),
        "correlated_noise_regions": [
            {"wl_um": wl_c, "allan_ratio": a, "scatter_ppm": s} for wl_c, a, s in correlated
        ],
        "trust_regions": regions,
        "diversity_improvement": div.get("improvement_factor", 0),
    }

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
