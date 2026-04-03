#!/usr/bin/env python3
"""Validation of chime output against published WASP-39b results.

Three validation tests:
  1. Scatter correlation with Rustamkulov+ 2023 error bars
  2. COMPASS 2.8-3.5 µm elevated systematics (Gordon+ 2025)
  3. End-to-end: ECSV → atmospheric sensitivity → flat-line rejection

Usage:
    python examples/validation_wasp39.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ==================================================================== #
# Rustamkulov+ 2023 published WASP-39b transmission spectrum
# From Nature 2023, Fig. 2 — NIRSpec PRISM data points
# ==================================================================== #

# These are approximate digitized values from Rustamkulov+ 2023, Fig. 2
# NIRSpec PRISM transmission spectrum (their pipeline, visit 1+2 combined)
RUSTAMKULOV_WL = np.array(
    [
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
        1.40,
        1.50,
        1.60,
        1.70,
        1.80,
        1.90,
        2.00,
        2.10,
        2.20,
        2.30,
        2.40,
        2.50,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
        3.10,
        3.20,
        3.30,
        3.40,
        3.50,
        3.60,
        3.70,
        3.80,
        3.90,
        4.00,
        4.10,
        4.20,
        4.30,
        4.40,
        4.50,
        4.60,
        4.70,
        4.80,
        4.90,
        5.00,
        5.10,
        5.20,
        5.30,
    ]
)

# Published error bars in ppm (approximate digitized from their Fig. 2)
# These represent the 1-sigma uncertainties on transit depth
RUSTAMKULOV_ERR = np.array(
    [
        150,
        80,
        50,
        35,
        30,
        28,
        26,
        30,
        35,
        32,
        28,
        26,
        25,
        28,
        30,
        28,
        26,
        25,
        25,
        26,
        28,
        30,
        32,
        35,
        38,
        40,
        42,
        45,
        48,
        50,
        55,
        60,
        65,
        70,
        80,
        90,
        95,
        100,
        110,
        120,
        130,
        150,
        170,
        200,
        230,
        260,
        300,
        350,
    ]
)

# Published transit depths in ppm (approximate)
RUSTAMKULOV_DEPTH = np.array(
    [
        22500,
        22200,
        21800,
        21500,
        21300,
        21200,
        21100,
        21150,
        21300,
        21200,
        21150,
        21100,
        21080,
        21150,
        21100,
        21050,
        21000,
        21020,
        21050,
        21100,
        21150,
        21200,
        21300,
        21400,
        21500,
        21600,
        21650,
        21700,
        21750,
        21800,
        21850,
        21900,
        21950,
        22000,
        22100,
        22200,
        22300,
        22500,
        22600,
        22700,
        22800,
        23000,
        23200,
        23500,
        23800,
        24100,
        24500,
        25000,
    ]
)


def load_chime_results(json_path: str) -> dict:
    """Load chime JSON output."""
    with open(json_path) as f:
        return json.load(f)


def get_best_observation(results: dict) -> dict:
    """Select the best PRISM observation (most in-transit integrations)."""
    best = None
    best_n = 0
    for obs in results["observations"]:
        if obs["n_in_transit"] > best_n:
            best_n = obs["n_in_transit"]
            best = obs
    return best


def test_1_scatter_correlation(obs: dict, outdir: Path):
    """Test 1: Does chime scatter correlate with Rustamkulov+ 2023 error bars?

    Wavelengths where Rustamkulov reports larger errors should correspond
    to wavelengths where chime measures higher scatter.
    """
    print("\n" + "=" * 60)
    print("  TEST 1: Scatter vs Rustamkulov+ 2023 Error Bars")
    print("=" * 60)

    bins = obs["channel_map"]["bins"]
    chime_wl = np.array([b["wl_center"] for b in bins])
    chime_scatter = np.array([b["scatter_ppm"] for b in bins])

    # Interpolate Rustamkulov error bars onto chime wavelength grid
    f_err = interp1d(
        RUSTAMKULOV_WL, RUSTAMKULOV_ERR, kind="linear", bounds_error=False, fill_value=np.nan
    )
    rust_err_at_chime = f_err(chime_wl)

    # Only compare where both are valid
    valid = np.isfinite(rust_err_at_chime) & (chime_scatter > 0)
    if np.sum(valid) < 5:
        print("  INSUFFICIENT OVERLAP — cannot compare")
        return

    wl_v = chime_wl[valid]
    scatter_v = chime_scatter[valid]
    rust_err_v = rust_err_at_chime[valid]

    # Spearman rank correlation
    rho, pval = spearmanr(scatter_v, rust_err_v)
    print(f"  Spearman ρ = {rho:.3f} (p = {pval:.2e})")
    print(f"  Valid bins: {np.sum(valid)}")

    if rho > 0.5 and pval < 0.01:
        print("  ✓ PASS — Scatter correlates with published error bars")
    elif rho > 0.3:
        print("  ~ MARGINAL — Weak correlation")
    else:
        print("  ✗ FAIL — No significant correlation")

    # Print wavelength-by-wavelength comparison
    print(f"\n  {'WL (µm)':>8} {'chime (ppm)':>12} {'Rust+23 (ppm)':>14} {'Ratio':>8}")
    for i in range(len(wl_v)):
        ratio = scatter_v[i] / rust_err_v[i] if rust_err_v[i] > 0 else 0
        print(f"  {wl_v[i]:8.2f} {scatter_v[i]:12.0f} {rust_err_v[i]:14.0f} {ratio:8.1f}x")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor("#0a0e17")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values():
            s.set_color("#1e293b")

    # Top: both on same axes
    ax = axes[0]
    ax.semilogy(
        chime_wl,
        chime_scatter,
        "o-",
        color="#f87171",
        markersize=4,
        label=f"chime empirical scatter",
    )
    ax.semilogy(
        RUSTAMKULOV_WL,
        RUSTAMKULOV_ERR,
        "s-",
        color="#38bdf8",
        markersize=3,
        alpha=0.7,
        label="Rustamkulov+ 2023 error bars",
    )
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Noise (ppm)", color="#e2e8f0")
    ax.set_title(
        "Test 1: chime Scatter vs Published Error Bars (WASP-39b)",
        color="#e2e8f0",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        fontsize=9, framealpha=0.3, labelcolor="#e2e8f0", facecolor="#111827", edgecolor="#1e293b"
    )

    # Bottom: correlation scatter plot
    ax = axes[1]
    ax.scatter(rust_err_v, scatter_v, c=wl_v, cmap="viridis", s=30, alpha=0.8)
    ax.set_xlabel("Rustamkulov+ 2023 Error Bar (ppm)", color="#e2e8f0")
    ax.set_ylabel("chime Scatter (ppm)", color="#e2e8f0")
    ax.set_title(f"Spearman ρ = {rho:.3f} (p = {pval:.2e})", color="#e2e8f0", fontsize=11)
    # Add colorbar for wavelength
    cb = fig.colorbar(ax.collections[0], ax=ax, label="Wavelength (µm)")
    cb.ax.yaxis.label.set_color("#e2e8f0")
    cb.ax.tick_params(colors="#94a3b8")

    fig.suptitle("Validation Test 1: Scatter Correlation", color="#94a3b8", fontsize=9, y=0.995)
    plt.tight_layout()
    plt.savefig(
        str(outdir / "test1_scatter_correlation.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="#0a0e17",
    )
    plt.close()
    print(f"  Plot: {outdir / 'test1_scatter_correlation.png'}")

    return {"rho": rho, "pval": pval, "n_bins": int(np.sum(valid))}


def test_2_compass_systematics(obs: dict, outdir: Path):
    """Test 2: Does chime reproduce the COMPASS 2.8-3.5 µm finding?

    Gordon+ 2025 (COMPASS) identified elevated systematics in the
    2.8-3.5 µm region of NIRSpec PRISM. Our Allan diagnostic should
    show elevated correlated noise in this range.
    """
    print("\n" + "=" * 60)
    print("  TEST 2: COMPASS 2.8-3.5 µm Elevated Systematics")
    print("=" * 60)

    bins = obs["channel_map"]["bins"]

    # Extract data
    wl = np.array([b["wl_center"] for b in bins])
    excess = np.array([b["systematic_excess"] for b in bins])
    allan = np.array([b["allan_worst_ratio"] for b in bins])
    scatter = np.array([b["scatter_ppm"] for b in bins])
    grades = [b["grade"] for b in bins]

    # Define comparison regions
    compass_mask = (wl >= 2.8) & (wl <= 3.5)
    control_mask = ((wl >= 1.5) & (wl <= 2.0)) | ((wl >= 4.0) & (wl <= 4.5))

    if not np.any(compass_mask):
        print("  ✗ No data in 2.8-3.5 µm range")
        return

    print(f"\n  COMPASS region (2.8-3.5 µm):")
    print(f"    Bins: {np.sum(compass_mask)}")
    print(f"    Median scatter: {np.median(scatter[compass_mask]):.0f} ppm")
    print(f"    Median Allan ratio: {np.median(allan[compass_mask]):.2f}")
    print(f"    Max Allan ratio: {np.max(allan[compass_mask]):.2f}")

    if np.any(control_mask):
        print(f"\n  Control regions (1.5-2.0, 4.0-4.5 µm):")
        print(f"    Median scatter: {np.median(scatter[control_mask]):.0f} ppm")
        print(f"    Median Allan ratio: {np.median(allan[control_mask]):.2f}")

    compass_allan = np.median(allan[compass_mask])
    compass_max_allan = np.max(allan[compass_mask])

    if compass_max_allan > 2.0:
        print(f"\n  ✓ PASS — Elevated Allan ratio in 2.8-3.5 µm (max {compass_max_allan:.1f}×)")
    elif compass_allan > 1.5:
        print(f"\n  ~ MARGINAL — Moderately elevated Allan ratio (median {compass_allan:.2f}×)")
    else:
        print(f"\n  ✗ FAIL — No elevated systematics detected in 2.8-3.5 µm")

    # Detailed per-bin output in the COMPASS range
    print(f"\n  Per-bin detail in 2.8-3.5 µm:")
    for b in bins:
        if 2.8 <= b["wl_center"] <= 3.5:
            print(
                f"    {b['wl_center']:.2f} µm  scatter={b['scatter_ppm']:.0f} ppm  "
                f"allan={b['allan_worst_ratio']:.2f}  grade={b['grade']}"
            )

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#0a0e17")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values():
            s.set_color("#1e293b")

    # Panel 1: Allan ratio with COMPASS region highlighted
    ax = axes[0]
    colors = ["#34d399" if a < 1.5 else "#fbbf24" if a < 3 else "#f87171" for a in allan]
    bar_w = np.diff(wl, append=wl[-1] * 1.05) * 0.8
    ax.bar(wl, allan, width=bar_w, color=colors, alpha=0.7)
    ax.axvspan(2.8, 3.5, alpha=0.15, color="#f87171", label="COMPASS 2.8-3.5 µm")
    ax.axhline(1.5, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(2.0, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Allan Ratio", color="#e2e8f0")
    ax.set_title(
        "Allan Deviation — COMPASS Region Highlighted",
        color="#e2e8f0",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        fontsize=9, framealpha=0.3, labelcolor="#e2e8f0", facecolor="#111827", edgecolor="#1e293b"
    )

    # Panel 2: Systematic excess
    ax = axes[1]
    ax.bar(wl, excess, width=bar_w, color=colors, alpha=0.7)
    ax.axvspan(2.8, 3.5, alpha=0.15, color="#f87171")
    ax.axhline(1, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Systematic Excess", color="#e2e8f0")
    ax.set_title("Systematic Excess", color="#e2e8f0", fontsize=11)

    # Panel 3: Scatter
    ax = axes[2]
    ax.semilogy(wl, scatter, "o-", color="#f87171", markersize=4)
    ax.axvspan(2.8, 3.5, alpha=0.15, color="#f87171", label="COMPASS region")
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Scatter (ppm)", color="#e2e8f0")
    ax.set_title("Empirical Scatter", color="#e2e8f0", fontsize=11)
    ax.legend(
        fontsize=9, framealpha=0.3, labelcolor="#e2e8f0", facecolor="#111827", edgecolor="#1e293b"
    )

    fig.suptitle(
        "Validation Test 2: COMPASS 2.8-3.5 µm Systematics", color="#94a3b8", fontsize=9, y=0.995
    )
    plt.tight_layout()
    plt.savefig(
        str(outdir / "test2_compass_systematics.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="#0a0e17",
    )
    plt.close()
    print(f"  Plot: {outdir / 'test2_compass_systematics.png'}")

    return {
        "compass_median_allan": float(compass_allan),
        "compass_max_allan": float(compass_max_allan),
    }


def test_3_atmospheric_sensitivity(obs: dict, ephemeris: dict, outdir: Path):
    """Test 3: End-to-end atmospheric sensitivity from chime ECSV.

    Uses the channel quality weights to compute a weighted transmission
    spectrum and test whether atmospheric features are detectable above
    the noise. This is a chi-squared flat-line rejection test.

    If the weighted spectrum is significantly non-flat, that's evidence
    of atmospheric features (or systematics we haven't removed).
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Atmospheric Sensitivity — Flat-Line Rejection")
    print("=" * 60)

    bins = obs["channel_map"]["bins"]
    weights = np.array(obs["channel_map"]["weights"])

    wl = np.array([b["wl_center"] for b in bins])
    depth = np.array([b["depth_ppm"] for b in bins])
    scatter = np.array([b["scatter_ppm"] for b in bins])
    grades = [b["grade"] for b in bins]
    n_in = obs.get("n_in_transit", 1)
    n_out = obs.get("n_integrations", 1) - n_in

    # Per-bin error: scatter / sqrt(n_in_transit)
    depth_err = scatter / np.sqrt(max(n_in, 1))

    # Weighted mean depth
    active = weights > 0
    if not np.any(active):
        print("  No active bins — cannot test")
        return

    w = weights[active]
    d = depth[active]
    e = depth_err[active]

    weighted_mean = np.sum(w * d) / np.sum(w)
    weighted_err = np.sqrt(np.sum(w**2 * e**2)) / np.sum(w)

    # Chi-squared against flat line (weighted mean)
    chi2 = np.sum(((d - weighted_mean) / e) ** 2)
    ndof = len(d) - 1
    chi2_reduced = chi2 / ndof if ndof > 0 else 0

    print(f"  Weighted mean depth: {weighted_mean:.0f} ± {weighted_err:.0f} ppm")
    print(f"  Expected depth: {ephemeris.get('expected_depth_ppm', '?')} ppm")
    print(f"  χ²/ν = {chi2_reduced:.2f} ({ndof} dof)")
    print(f"  Active bins (A/B grade): {np.sum(active)}")

    expected = ephemeris.get("expected_depth_ppm", 0)
    if expected > 0:
        deviation = abs(weighted_mean - expected) / weighted_err
        print(f"  Deviation from expected: {deviation:.1f}σ")

    # Per-wavelength significance of departure from flat line
    significance = (d - weighted_mean) / e

    # Detect molecular features by looking for bins where depth >> flat line
    mol_bands = [
        (1.30, 1.50, "H2O 1.4µm"),
        (1.75, 2.05, "H2O 1.9µm"),
        (2.50, 2.90, "H2O 2.7µm"),
        (3.20, 3.50, "CH4 3.3µm"),
        (4.15, 4.45, "CO2 4.3µm"),
        (4.50, 4.80, "CO 4.6µm"),
    ]

    print(f"\n  Molecular band sensitivity:")
    wl_active = wl[active]
    sig_active = significance

    for lo, hi, name in mol_bands:
        band_mask = (wl_active >= lo) & (wl_active <= hi)
        if np.any(band_mask):
            band_sig = np.max(np.abs(sig_active[band_mask]))
            band_depth = d[band_mask].mean()
            print(
                f"    {name:12s}  |{band_sig:5.1f}σ|  "
                f"depth={band_depth:8.0f} ppm  "
                f"{'DETECTED' if band_sig > 3 else 'below noise'}"
            )
        else:
            print(f"    {name:12s}  no data in band")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor("#0a0e17")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values():
            s.set_color("#1e293b")

    # Panel 1: Weighted transmission spectrum
    ax = axes[0]
    grade_colors = {"A": "#34d399", "B": "#38bdf8", "C": "#fbbf24", "D": "#f87171"}
    for i in range(len(bins)):
        c = grade_colors[grades[i]]
        alpha = 1.0 if weights[i] > 0 else 0.3
        ax.errorbar(
            wl[i],
            depth[i],
            yerr=depth_err[i],
            fmt="o",
            color=c,
            alpha=alpha,
            markersize=5,
            elinewidth=1,
            capsize=2,
        )
    ax.axhline(
        weighted_mean,
        color="#38bdf8",
        linewidth=1.5,
        linestyle="--",
        label=f"Weighted mean: {weighted_mean:.0f} ppm",
    )
    if expected > 0:
        ax.axhline(
            expected,
            color="#fbbf24",
            linewidth=1,
            linestyle=":",
            label=f"Expected: {expected:.0f} ppm",
        )
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("Transit Depth (ppm)", color="#e2e8f0")
    ax.set_title(
        "Weighted Transmission Spectrum (chime quality weights)",
        color="#e2e8f0",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        fontsize=9, framealpha=0.3, labelcolor="#e2e8f0", facecolor="#111827", edgecolor="#1e293b"
    )

    # Panel 2: Significance of departure from flat line
    ax = axes[1]
    sig_colors = [
        "#f87171" if abs(s) > 3 else "#fbbf24" if abs(s) > 2 else "#34d399" for s in sig_active
    ]
    ax.bar(
        wl_active,
        sig_active,
        width=np.diff(wl_active, append=wl_active[-1] * 1.05) * 0.8,
        color=sig_colors,
        alpha=0.7,
    )
    ax.axhline(3, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5, label="3σ threshold")
    ax.axhline(-3, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="#334155", linewidth=0.5)
    # Annotate molecular bands
    for lo, hi, name in mol_bands:
        mid = (lo + hi) / 2
        ax.axvspan(lo, hi, alpha=0.08, color="#a78bfa")
        ax.text(
            mid,
            ax.get_ylim()[1] * 0.9,
            name,
            fontsize=6,
            color="#a78bfa",
            ha="center",
            va="top",
            rotation=90,
        )
    ax.set_xlabel("Wavelength (µm)", color="#e2e8f0")
    ax.set_ylabel("|Significance| from flat line (σ)", color="#e2e8f0")
    ax.set_title(
        f"Flat-Line Rejection: χ²/ν = {chi2_reduced:.2f} ({ndof} dof)", color="#e2e8f0", fontsize=11
    )
    ax.legend(
        fontsize=9, framealpha=0.3, labelcolor="#e2e8f0", facecolor="#111827", edgecolor="#1e293b"
    )

    fig.suptitle("Validation Test 3: Atmospheric Sensitivity", color="#94a3b8", fontsize=9, y=0.995)
    plt.tight_layout()
    plt.savefig(
        str(outdir / "test3_atmospheric_sensitivity.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="#0a0e17",
    )
    plt.close()
    print(f"  Plot: {outdir / 'test3_atmospheric_sensitivity.png'}")

    return {
        "weighted_depth_ppm": weighted_mean,
        "weighted_err_ppm": weighted_err,
        "chi2_reduced": chi2_reduced,
        "ndof": ndof,
    }


def main():
    candidate_paths = [
        Path(__file__).parent.parent / "results" / "wasp39_fit" / "chime_wasp_39.json",
        Path(__file__).parent.parent / "chime_output" / "wasp39_v2" / "chime_wasp_39.json",
        Path(__file__).parent.parent / "chime_output" / "wasp39" / "chime_wasp_39.json",
        Path("chime_output/wasp39/chime_wasp_39.json"),
    ]
    json_path = next((path for path in candidate_paths if path.exists()), None)
    if json_path is None:
        print("  Cannot find example chime output.")
        print("  Run: chime WASP-39 --outdir chime_output/wasp39")
        sys.exit(1)

    outdir = json_path.parent / "validation"
    outdir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  chime Validation Against Published WASP-39b Results")
    print("=" * 60)
    print(f"  Input: {json_path}")

    results = load_chime_results(str(json_path))
    obs = get_best_observation(results)
    ephemeris = results["ephemeris"]

    print(f"  Best observation: {obs['obs_id']}")
    print(f"  Integrations: {obs['n_integrations']} ({obs['n_in_transit']} in-transit)")
    print(f"  Bins: {obs['channel_map']['n_bins']}")

    # Run all three tests
    t1 = test_1_scatter_correlation(obs, outdir)
    t2 = test_2_compass_systematics(obs, outdir)
    t3 = test_3_atmospheric_sensitivity(obs, ephemeris, outdir)

    # Summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(
        f"  Test 1 (Scatter correlation):  ρ={t1['rho']:.3f}  "
        f"{'PASS' if t1['rho'] > 0.5 else 'MARGINAL' if t1['rho'] > 0.3 else 'FAIL'}"
    )
    print(
        f"  Test 2 (COMPASS systematics):  max Allan={t2['compass_max_allan']:.1f}  "
        f"{'PASS' if t2['compass_max_allan'] > 2.0 else 'MARGINAL' if t2['compass_max_allan'] > 1.5 else 'FAIL'}"
    )
    print(f"  Test 3 (Atmospheric sensitivity): χ²/ν={t3['chi2_reduced']:.2f}")

    # Save validation results
    validation = {
        "test1_scatter_correlation": t1,
        "test2_compass_systematics": t2,
        "test3_atmospheric_sensitivity": t3,
    }
    val_path = outdir / "validation_results.json"
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"\n  Validation results: {val_path}")


if __name__ == "__main__":
    main()
