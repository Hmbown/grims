"""
Deep visualization suite for GRIM-S analysis.

Comprehensive plotting for:
- Per-event kappa estimates with error bars
- Stacked posterior vs individual posteriors
- Jackknife stability plots
- Fisher correlation matrices
- NR prediction comparisons
- Catalog-wide summaries
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_per_event_kappa(
    results: List[Dict],
    save_path: str = None,
    title: str = "Per-Event Kappa Estimates",
) -> plt.Figure:
    """Plot kappa estimates for each event with error bars.

    Parameters
    ----------
    results : list of dicts with keys:
        - event_name: str
        - kappa_hat: float
        - kappa_sigma: float
        - snr: float (optional)
    """
    n = len(results)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.35)))

    names = [r.get("event_name", r.get("event", "unknown")) for r in results]
    kappas = [r.get("kappa_hat", 0) for r in results]
    sigmas = [r.get("kappa_sigma", 0) for r in results]

    # Color by SNR if available
    if "snr" in results[0]:
        snrs = [r.get("snr", 0) for r in results]
        colors = plt.cm.viridis(np.array(snrs) / max(max(snrs), 1))
    else:
        colors = ["steelblue"] * n

    y_pos = np.arange(n)
    ax.errorbar(
        kappas, y_pos, xerr=sigmas, fmt="o", color="steelblue", capsize=4, alpha=0.8
    )

    # GR prediction line
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="kappa = 1 (GR)")
    ax.axvline(x=0.0, color="gray", linestyle=":", alpha=0.5, label="kappa = 0 (no NL)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("kappa", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_stacked_posterior(
    stacked_result,
    individual_results: Optional[List] = None,
    save_path: str = None,
) -> plt.Figure:
    """Plot stacked posterior with individual event posteriors overlaid.

    Parameters
    ----------
    stacked_result : StackedPosterior or StackedPhaseLockResult
    individual_results : list of PosteriorResult or PhaseLockResult
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    kappa_grid = stacked_result.kappa_grid

    # Plot individual posteriors (faint)
    if individual_results:
        for p in individual_results:
            if hasattr(p, "posterior") and p.posterior is not None:
                # Normalize to same scale
                post_norm = (
                    p.posterior / np.max(p.posterior)
                    if np.max(p.posterior) > 0
                    else p.posterior
                )
                ax.plot(kappa_grid, post_norm, "gray", alpha=0.15, linewidth=0.8)

    # Stacked posterior (bold)
    if hasattr(stacked_result, "stacked_posterior"):
        post = stacked_result.stacked_posterior
    elif hasattr(stacked_result, "posterior"):
        post = stacked_result.posterior
    else:
        # Construct from log_likelihood
        log_l = stacked_result.stacked_log_likelihood
        post = np.exp(log_l - np.max(log_l))

    post_norm = post / np.max(post) if np.max(post) > 0 else post
    ax.plot(kappa_grid, post_norm, "blue", linewidth=2.5, label="Stacked")

    # MAP estimate
    map_idx = np.argmax(post)
    ax.axvline(kappa_grid[map_idx], color="blue", linestyle="--", alpha=0.7)

    # GR reference
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="kappa = 1 (GR)")
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5, label="kappa = 0")

    ax.set_xlabel("kappa", fontsize=12)
    ax.set_ylabel("Posterior density (normalized)", fontsize=12)
    ax.set_title("Stacked Posterior Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_kappa_vs_spin(
    events: List[Dict],
    save_path: str = None,
) -> plt.Figure:
    """Plot measured kappa vs remnant spin with NR prediction curve.

    Parameters
    ----------
    events : list of dicts with:
        - event_name: str
        - kappa_hat: float
        - kappa_sigma: float
        - remnant_spin: float
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # NR prediction curve
    try:
        from .nr_predictions import generate_kappa_curve

        spins_curve, kappa_curve, unc_curve = generate_kappa_curve()
        ax.plot(spins_curve, kappa_curve, "r-", linewidth=2, label="NR prediction")
        ax.fill_between(
            spins_curve,
            kappa_curve - 1.645 * unc_curve,
            kappa_curve + 1.645 * unc_curve,
            alpha=0.2,
            color="red",
            label="NR 90% CI",
        )
    except ImportError:
        pass

    # Data points
    spins = [e["remnant_spin"] for e in events]
    kappas = [e["kappa_hat"] for e in events]
    sigmas = [e["kappa_sigma"] for e in events]
    names = [e["event_name"] for e in events]

    ax.errorbar(
        spins, kappas, yerr=sigmas, fmt="o", color="steelblue", capsize=4, alpha=0.8
    )

    # Label points
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (spins[i], kappas[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("Remnant spin (chi)", fontsize=12)
    ax.set_ylabel("kappa", fontsize=12)
    ax.set_title("Kappa vs Remnant Spin", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_catalog_summary(
    results: List[Dict],
    stacked_result,
    save_path: str = None,
) -> plt.Figure:
    """Comprehensive catalog summary plot.

    Three panels:
    1. Per-event kappa estimates
    2. Stacked posterior
    3. SNR vs event mass
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Per-event kappa
    ax1 = fig.add_subplot(gs[0, 0])
    n = len(results)
    names = [r.get("event_name", r.get("event", "unknown")) for r in results]
    kappas = [r.get("kappa_hat", 0) for r in results]
    sigmas = [r.get("kappa_sigma", 0) for r in results]

    y_pos = np.arange(n)
    ax1.errorbar(
        kappas, y_pos, xerr=sigmas, fmt="o", color="steelblue", capsize=3, alpha=0.8
    )
    ax1.axvline(1.0, color="red", linestyle="--", alpha=0.5)
    ax1.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("kappa", fontsize=10)
    ax1.set_title("Per-Event Estimates", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Stacked posterior
    ax2 = fig.add_subplot(gs[0, 1])
    if hasattr(stacked_result, "kappa_grid"):
        kappa_grid = stacked_result.kappa_grid
        if hasattr(stacked_result, "stacked_posterior"):
            post = stacked_result.stacked_posterior
        elif hasattr(stacked_result, "posterior"):
            post = stacked_result.posterior
        else:
            log_l = stacked_result.stacked_log_likelihood
            post = np.exp(log_l - np.max(log_l))

        post_norm = post / np.max(post) if np.max(post) > 0 else post
        ax2.plot(kappa_grid, post_norm, "blue", linewidth=2)
        ax2.axvline(1.0, color="red", linestyle="--", alpha=0.5)
        ax2.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
        ax2.set_xlabel("kappa", fontsize=10)
        ax2.set_ylabel("Posterior density", fontsize=10)
        ax2.set_title("Stacked Posterior", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)

    # Panel 3: SNR vs mass
    ax3 = fig.add_subplot(gs[1, :])
    masses = [r.get("mass", r.get("remnant_mass", 0)) for r in results]
    snrs = [r.get("snr_nl", r.get("snr", 0)) for r in results]

    ax3.scatter(masses, snrs, c="steelblue", s=50, alpha=0.7)
    for i, name in enumerate(names):
        ax3.annotate(
            name,
            (masses[i], snrs[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.7,
        )
    ax3.set_xlabel("Remnant mass (Msun)", fontsize=10)
    ax3.set_ylabel("Nonlinear SNR", fontsize=10)
    ax3.set_title("SNR vs Mass", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("GRIM-S Catalog Summary", fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_measurement_vs_nr(
    comparisons: List[Dict],
    save_path: str = None,
) -> plt.Figure:
    """Plot measured kappa vs NR prediction for each event.

    Parameters
    ----------
    comparisons : list of dicts from compare_measurement_to_nr()
    """
    n = len(comparisons)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.4)))

    names = [c.get("event_name", f"Event {i}") for i, c in enumerate(comparisons)]
    measured = [c["measured_kappa"] for c in comparisons]
    measured_err = [c["measured_sigma"] for c in comparisons]
    nr_pred = [c["nr_kappa"] for c in comparisons]
    nr_err = [c["nr_sigma"] for c in comparisons]

    y_pos = np.arange(n)

    # NR predictions (gray bars)
    ax.errorbar(
        nr_pred,
        y_pos,
        xerr=nr_err,
        fmt="s",
        color="gray",
        capsize=3,
        alpha=0.5,
        label="NR prediction",
    )

    # Measurements (blue bars)
    ax.errorbar(
        measured,
        y_pos,
        xerr=measured_err,
        fmt="o",
        color="steelblue",
        capsize=3,
        alpha=0.8,
        label="Measurement",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("kappa", fontsize=11)
    ax.set_title("Measurement vs NR Prediction", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
