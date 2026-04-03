"""
Leave-one-out jackknife analysis for GRIM-S stacking.

Bown's principle: a measurement that cannot detect its own failure
is not a measurement. The jackknife tests whether the stacked result
is dominated by a single event or is robust across the catalog.

For each event in the stack, remove it and recompute the combined
kappa estimate. If one event dominates, removing it should cause
a large shift. If the result is robust, all jackknife estimates
should cluster around the full-stack value.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from .phase_locked_search import (
    PhaseLockResult,
    StackedPhaseLockResult,
    stack_phase_locked,
)


@dataclass
class JackknifeResult:
    """Result of leave-one-out jackknife analysis."""

    # Full stack result (all events)
    full_kappa: float
    full_sigma: float
    full_snr: float

    # Jackknife estimates (one per removed event)
    removed_event_names: List[str]
    jackknife_kappas: np.ndarray
    jackknife_sigmas: np.ndarray
    jackknife_snrs: np.ndarray

    # Statistics
    jackknife_mean: float
    jackknife_std: float
    max_shift: float  # largest |kappa_jack - kappa_full|
    max_shift_event: str  # which event caused the largest shift

    # Concentration metrics
    n_eff: float = 0.0  # effective number of events (Herfindahl inverse)
    max_fractional_influence: float = 0.0  # max_shift / full_sigma

    # Diagnostics
    is_stable: bool = True  # True if max_shift < 2 * jackknife_std
    influential_events: List[str] = field(default_factory=list)  # shifts > 1 sigma


def run_jackknife(
    results: List[PhaseLockResult],
    max_weight_ratio: float | None = None,
) -> JackknifeResult:
    """Run leave-one-out jackknife on a list of phase-locked results.

    Parameters
    ----------
    results : list of PhaseLockResult from individual events
    max_weight_ratio : float, optional
        Passed to stack_phase_locked. Caps each event's weight at
        this multiple of the average weight.

    Returns
    -------
    JackknifeResult with full stack and jackknife estimates
    """
    if len(results) < 3:
        raise ValueError("Need at least 3 events for jackknife")

    # Full stack
    full = stack_phase_locked(results, max_weight_ratio=max_weight_ratio)
    full_kappa = full.kappa_hat
    full_sigma = full.kappa_sigma
    full_snr = full.snr

    # Leave-one-out
    jack_kappas = []
    jack_sigmas = []
    jack_snrs = []
    removed_names = []

    for i, r in enumerate(results):
        # Remove event i
        subset = results[:i] + results[i + 1 :]
        jack = stack_phase_locked(subset, max_weight_ratio=max_weight_ratio)

        jack_kappas.append(jack.kappa_hat)
        jack_sigmas.append(jack.kappa_sigma)
        jack_snrs.append(jack.snr)
        removed_names.append(r.event_name)

    jack_kappas = np.array(jack_kappas)
    jack_sigmas = np.array(jack_sigmas)
    jack_snrs = np.array(jack_snrs)

    # Statistics
    jack_mean = np.mean(jack_kappas)
    jack_std = np.std(jack_kappas)

    shifts = np.abs(jack_kappas - full_kappa)
    max_shift_idx = np.argmax(shifts)
    max_shift = shifts[max_shift_idx]
    max_shift_event = removed_names[max_shift_idx]

    # Stability: is the max shift consistent with jackknife variance?
    is_stable = max_shift < 2.0 * jack_std if jack_std > 0 else True

    # Influential events: those causing shifts > 1 sigma
    influential = []
    if jack_std > 0:
        for i, shift in enumerate(shifts):
            if shift > jack_std:
                influential.append(removed_names[i])

    # Max fractional influence: how much of the full sigma can one event shift?
    max_frac = max_shift / full_sigma if full_sigma > 0 else 0.0

    # Effective number of events (Herfindahl index on weights)
    weights = []
    for r in results:
        if r.kappa_sigma > 0 and np.isfinite(r.kappa_sigma):
            weights.append(1.0 / r.kappa_sigma**2)
    weights = np.array(weights)
    if max_weight_ratio is not None and max_weight_ratio > 0:
        w_avg = np.sum(weights) / len(weights)
        weights = np.minimum(weights, max_weight_ratio * w_avg)
    w_frac = weights / np.sum(weights) if np.sum(weights) > 0 else weights
    n_eff = 1.0 / np.sum(w_frac**2) if np.sum(w_frac**2) > 0 else len(results)

    return JackknifeResult(
        full_kappa=full_kappa,
        full_sigma=full_sigma,
        full_snr=full_snr,
        removed_event_names=removed_names,
        jackknife_kappas=jack_kappas,
        jackknife_sigmas=jack_sigmas,
        jackknife_snrs=jack_snrs,
        jackknife_mean=jack_mean,
        jackknife_std=jack_std,
        max_shift=max_shift,
        max_shift_event=max_shift_event,
        n_eff=n_eff,
        max_fractional_influence=max_frac,
        is_stable=is_stable,
        influential_events=influential,
    )


def print_jackknife_summary(result: JackknifeResult) -> None:
    """Print a human-readable jackknife summary."""
    print("=" * 60)
    print("JACKKNIFE ANALYSIS: Leave-One-Out Stability Test")
    print("=" * 60)
    print()
    print(f"Full stack:  kappa = {result.full_kappa:.4f} +/- {result.full_sigma:.4f}")
    print(f"             SNR   = {result.full_snr:.2f}")
    print(f"             significance = {result.full_kappa / result.full_sigma:.2f} sigma")
    print()
    print(f"Jackknife mean:  {result.jackknife_mean:.4f}")
    print(f"Jackknife std:   {result.jackknife_std:.4f}")
    print(f"Max shift:       {result.max_shift:.4f} (from {result.max_shift_event})")
    print(f"Max frac. infl.: {result.max_fractional_influence:.3f} sigma_full")
    print(f"N_eff:           {result.n_eff:.1f}")
    print(f"Stable:          {'Yes' if result.is_stable else 'No'}")
    print()

    if result.influential_events:
        print(f"Influential events (shift > 1 sigma):")
        for ev in result.influential_events:
            print(f"  - {ev}")
        print()

    print(f"{'Event':<30} {'kappa_jack':>12} {'sigma_jack':>12} {'shift':>10}")
    print("-" * 60)
    for i, name in enumerate(result.removed_event_names):
        shift = result.jackknife_kappas[i] - result.full_kappa
        print(
            f"{name:<30} {result.jackknife_kappas[i]:>12.4f} "
            f"{result.jackknife_sigmas[i]:>12.4f} {shift:>+10.4f}"
        )
    print()


def plot_jackknife(result: JackknifeResult, save_path: str = None):
    """Plot jackknife results.

    Shows the full stack estimate with error bars, and each jackknife
    estimate as a point. Events that cause large shifts stand out.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result.removed_event_names)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.3)))

    # Full stack line
    ax.axhline(
        result.full_kappa,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"Full stack: {result.full_kappa:.3f}",
    )
    ax.axhspan(
        result.full_kappa - result.full_sigma,
        result.full_kappa + result.full_sigma,
        alpha=0.2,
        color="black",
        label=f"Full ± {result.full_sigma:.3f}",
    )

    # Jackknife points
    colors = [
        "red" if name in result.influential_events else "blue"
        for name in result.removed_event_names
    ]

    y_pos = np.arange(n)
    ax.errorbar(
        result.jackknife_kappas,
        y_pos,
        xerr=result.jackknife_sigmas,
        fmt="o",
        color="steelblue",
        capsize=3,
        alpha=0.7,
    )

    # Highlight influential events
    for i, name in enumerate(result.removed_event_names):
        if name in result.influential_events:
            ax.errorbar(
                result.jackknife_kappas[i],
                i,
                xerr=result.jackknife_sigmas[i],
                fmt="o",
                color="red",
                capsize=3,
                alpha=0.9,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(result.removed_event_names, fontsize=9)
    ax.set_xlabel("kappa (leave-one-out)", fontsize=11)
    ax.set_title(
        "Jackknife Stability Test: Remove One Event at a Time",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved jackknife plot to {save_path}")

    plt.close()
    return fig
