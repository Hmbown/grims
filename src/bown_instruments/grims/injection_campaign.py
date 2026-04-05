"""
Realistic-SNR injection campaign for the Phase 3 GRIM-S stack.

This module exercises the same phase-locked estimator used by the O4-inclusive
Phase 3 result:

  1. Load the Phase 3 event selection and detector choices.
  2. Rebuild the whitening + bandpass preprocessing around each event.
  3. Estimate realistic linear-mode amplitudes from the on-source segment.
  4. Inject full synthetic ringdowns into off-source detector noise.
  5. Re-run the phase-locked search and stack the recovered kappas.

The campaign is designed to answer SHA-4222:
  - Is the stacked kappa estimator biased at realistic SNR?
  - Are the quoted uncertainties calibrated?
  - How sensitive are the answers to t_start and whitening bounds?

The implementation preserves an important correlation structure: within a given
detector realization, every t_start configuration is evaluated on the same
off-source noise segment. That exposes whether the current inverse-variance
combination over t_start choices is overconfident.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from .bayesian_analysis import fit_linear_modes
from .gwtc_pipeline import M_SUN_SECONDS, is_valid_hdf5_file, load_gwosc_strain_hdf5
from .mass_analysis import (
    DEFAULT_MIN_SEGMENT_DURATION,
    DEFAULT_T_START_STRATEGY,
    compute_optimal_segment_duration,
    find_local_strain_detector,
)
from .null_distribution import _crop_around_merger, _download_catalog_strain_file, _sanitize_strain
from .phase_locked_search import (
    LatentAmplitudeEvent,
    estimate_kappa_posterior_latent_amplitude,
    phase_locked_search_colored,
)
from .qnm_modes import KerrQNMCatalog
from .ringdown_templates import RingdownTemplateBuilder
from .whiten import bandpass, estimate_asd, whiten_strain


DEFAULT_PHASE3_RESULTS_PATH = Path("results/grims/phase3_results.json")
DEFAULT_CATALOG_PATH = Path("results/grims/gwtc_full_catalog.json")
DEFAULT_OUTPUT_PATH = Path("results/grims/phase3_injection_campaign.json")
DEFAULT_SUMMARY_CSV_PATH = Path("results/grims/phase3_injection_campaign_summary.csv")
DEFAULT_WEIGHT_DIAGNOSTICS_PATH = Path("results/grims/phase3_weight_diagnostics.json")
DEFAULT_PLOT_PATH = Path("plots/grims/phase3_injection_campaign.png")

DEFAULT_KAPPA_VALUES = (0.01, 0.02, 0.03, 0.04, 0.05)
DEFAULT_T_START_VALUES = (5.0, 8.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_PAD_BEFORE_S = 0.05


@dataclass(frozen=True)
class WhiteningBand:
    """Bandpass definition relative to the event's QNM frequencies."""

    label: str
    fmin_pad: float
    fmax_pad: float


@dataclass(frozen=True)
class ScenarioDefinition:
    """One preprocessing scenario evaluated by the campaign."""

    name: str
    band_label: str
    t_start_mode: str  # "fixed" or "marginalized"
    t_start_m: float | None = None


@dataclass
class EventScenarioContribution:
    """Per-event contribution to one stacked scenario."""

    event_name: str
    kappa_hat: np.ndarray  # shape = (n_realizations, n_kappa)
    kappa_sigma: np.ndarray  # shape = (n_realizations, n_kappa)
    snr: np.ndarray  # shape = (n_realizations, n_kappa)
    detectors_used: list[str]
    segment_duration: float
    best_t_start_m: float | None = None
    a_220_fit: np.ndarray | None = None  # shape = (n_realizations,) — per-realization recovery A_220
    a_220_noise_var: float = 0.0  # Fisher noise variance on A_220 (constant per event)
    spin: float = 0.0
    noise_rms: float = 0.0


def default_whitening_bands() -> list[WhiteningBand]:
    """Reasonable whitening-band sweep around the Phase 3 default."""
    return [
        WhiteningBand("default", 0.50, 1.30),
        WhiteningBand("tight", 0.65, 1.15),
        WhiteningBand("wide", 0.40, 1.50),
    ]


def build_scenarios(
    t_start_values: list[float] | tuple[float, ...] = DEFAULT_T_START_VALUES,
    bands: list[WhiteningBand] | None = None,
) -> list[ScenarioDefinition]:
    """Construct the default t_start and whitening sweeps."""
    bands = list(default_whitening_bands() if bands is None else bands)
    scenarios: list[ScenarioDefinition] = []

    for t_start_m in t_start_values:
        scenarios.append(
            ScenarioDefinition(
                name=f"fixed_tstart_{int(t_start_m)}M",
                band_label="default",
                t_start_mode="fixed",
                t_start_m=float(t_start_m),
            )
        )

    for band in bands:
        scenarios.append(
            ScenarioDefinition(
                name=f"marginalized_{band.label}",
                band_label=band.label,
                t_start_mode="marginalized",
                t_start_m=None,
            )
        )

    return scenarios


def _format_kappa(kappa: float) -> str:
    return f"{kappa:.3f}".rstrip("0").rstrip(".")


def _load_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def _select_phase3_entries(
    phase3_individual: list[dict[str, Any]],
    max_events: int | None = None,
    selection: str = "largest_weight",
) -> list[dict[str, Any]]:
    """Select which Phase 3 events to use for the injection campaign."""
    if max_events is None or max_events >= len(phase3_individual):
        return list(phase3_individual)

    if selection == "largest_weight":
        scored = sorted(
            phase3_individual,
            key=lambda row: 0.0
            if row.get("kappa_sigma", np.inf) in (0, np.inf)
            else 1.0 / row["kappa_sigma"] ** 2,
            reverse=True,
        )
    elif selection == "highest_snr":
        scored = sorted(
            phase3_individual,
            key=lambda row: abs(row.get("snr_event", 0.0)),
            reverse=True,
        )
    elif selection == "catalog_order":
        scored = list(phase3_individual)
    else:
        raise ValueError(f"Unknown event selection strategy: {selection}")

    return scored[:max_events]


def _nearest_start_index(time: np.ndarray, start_time: float, n_samples: int) -> int | None:
    """Choose the closest sample index for a desired segment start."""
    idx = int(np.searchsorted(time, start_time))
    if idx >= len(time):
        idx = len(time) - 1
    if idx > 0 and abs(time[idx - 1] - start_time) <= abs(time[idx] - start_time):
        idx -= 1

    if idx < 0 or idx + n_samples > len(time):
        return None
    return idx


def _valid_noise_window_starts(noise_mask: np.ndarray, n_samples: int) -> np.ndarray:
    """Return all segment starts that stay completely inside off-source data."""
    if n_samples <= 0 or len(noise_mask) < n_samples:
        return np.array([], dtype=int)

    counts = np.convolve(
        noise_mask.astype(int),
        np.ones(n_samples, dtype=int),
        mode="valid",
    )
    return np.flatnonzero(counts == n_samples)


def _build_injection_waveform(
    t_dimless: np.ndarray,
    spin: float,
    fit: dict[str, dict[str, float]],
    kappa: float,
    include_higher_linear_modes: bool = True,
) -> np.ndarray:
    """Construct a full ringdown waveform in the whitened analysis domain."""
    builder = RingdownTemplateBuilder()
    template = builder.build_nonlinear_template(
        spin=spin,
        A_220=float(fit["220"]["amplitude"]),
        A_330=float(fit.get("330", {}).get("amplitude", 0.0)) if include_higher_linear_modes else 0.0,
        A_440_linear=float(fit.get("440", {}).get("amplitude", 0.0)) if include_higher_linear_modes else 0.0,
        kappa=float(kappa),
        phi_220=float(fit["220"]["phase"]),
        phi_330=float(fit.get("330", {}).get("phase", 0.0)),
        phi_440_linear=float(fit.get("440", {}).get("phase", 0.0)),
    )
    return template.waveform(t_dimless)


def _combine_measurement_arrays(
    kappa_arrays: list[np.ndarray],
    sigma_arrays: list[np.ndarray],
    snr_arrays: list[np.ndarray] | None = None,
    strategy: str = "independent",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine several measurement arrays.

    Each input array has shape (n_realizations, n_kappa).
    """
    if not kappa_arrays:
        raise ValueError("Need at least one measurement array to combine")

    kappas = np.stack(kappa_arrays, axis=0)
    sigmas = np.stack(sigma_arrays, axis=0)

    valid = np.isfinite(sigmas) & (sigmas > 0)
    weights = np.where(valid, 1.0 / sigmas**2, 0.0)

    total_weight = np.sum(weights, axis=0)
    combined_kappa = np.full_like(total_weight, np.nan, dtype=float)
    combined_sigma = np.full_like(total_weight, np.inf, dtype=float)

    nonzero = total_weight > 0
    normalized_weights = np.zeros_like(weights)
    normalized_weights[:, nonzero] = weights[:, nonzero] / total_weight[nonzero]

    if strategy == "independent":
        combined_kappa[nonzero] = (
            np.sum(weights * kappas, axis=0)[nonzero] / total_weight[nonzero]
        )
        combined_sigma[nonzero] = 1.0 / np.sqrt(total_weight[nonzero])
    elif strategy == "shared_noise":
        combined_kappa[nonzero] = np.sum(normalized_weights * kappas, axis=0)[nonzero]
        sigma_stat = np.sum(normalized_weights * sigmas, axis=0)
        centered = kappas - combined_kappa[None, :, :]
        sigma_sys = np.sqrt(np.sum(normalized_weights * centered**2, axis=0))
        combined_sigma[nonzero] = np.hypot(sigma_stat, sigma_sys)[nonzero]
    else:
        raise ValueError(f"Unknown array-combination strategy: {strategy}")

    if snr_arrays:
        snrs = np.stack(snr_arrays, axis=0)
        if strategy == "independent":
            combined_snr = np.sqrt(np.sum(np.where(valid, snrs, 0.0) ** 2, axis=0))
        else:
            signed_abs = np.where(valid, np.abs(snrs), -np.inf)
            best_idx = np.argmax(signed_abs, axis=0)
            combined_snr = np.take_along_axis(snrs, best_idx[None, :, :], axis=0)[0]
            combined_snr[~nonzero] = 0.0
    else:
        combined_snr = np.zeros_like(combined_sigma)

    return combined_kappa, combined_sigma, combined_snr


def _stack_event_contributions(
    contributions: list[EventScenarioContribution],
    max_weight_ratio: float | None = 5.5,
) -> dict[str, np.ndarray]:
    """Apply the Phase 3 event stack, including the optional weight cap."""
    kappas = np.stack([entry.kappa_hat for entry in contributions], axis=0)
    sigmas = np.stack([entry.kappa_sigma for entry in contributions], axis=0)
    snrs = np.stack([entry.snr for entry in contributions], axis=0)

    valid = np.isfinite(sigmas) & (sigmas > 0)
    weights = np.where(valid, 1.0 / sigmas**2, 0.0)

    if max_weight_ratio is not None and max_weight_ratio > 0:
        for real_idx in range(weights.shape[1]):
            for kappa_idx in range(weights.shape[2]):
                valid_event = weights[:, real_idx, kappa_idx] > 0
                if not np.any(valid_event):
                    continue
                mean_weight = np.mean(weights[valid_event, real_idx, kappa_idx])
                cap = max_weight_ratio * mean_weight
                weights[valid_event, real_idx, kappa_idx] = np.minimum(
                    weights[valid_event, real_idx, kappa_idx],
                    cap,
                )

    total_weight = np.sum(weights, axis=0)
    stacked_kappa = np.full_like(total_weight, np.nan, dtype=float)
    stacked_sigma = np.full_like(total_weight, np.inf, dtype=float)
    stacked_snr = np.sqrt(np.sum(np.where(valid, snrs, 0.0) ** 2, axis=0))

    nonzero = total_weight > 0
    stacked_kappa[nonzero] = (
        np.sum(weights * kappas, axis=0)[nonzero] / total_weight[nonzero]
    )
    stacked_sigma[nonzero] = 1.0 / np.sqrt(total_weight[nonzero])

    return {
        "kappa_hat": stacked_kappa,
        "kappa_sigma": stacked_sigma,
        "snr": stacked_snr,
        "weights": weights,
    }


def _stack_event_contributions_marginalized(
    contributions: list[EventScenarioContribution],
    kappa_min: float = -0.15,
    kappa_max: float = 0.15,
    n_kappa: int = 301,
    n_amplitude: int = 101,
) -> dict[str, np.ndarray]:
    """Stack events using the latent-amplitude marginalized estimator.

    For each noise realization and injected kappa, builds per-event
    ``LatentAmplitudeEvent`` objects from the per-realization recovery
    ``(kappa_hat, kappa_sigma, a_220_fit)`` and runs the marginalized
    posterior.  Returns arrays of the same shape as the legacy stacker
    so the downstream summary code works unchanged.
    """
    if not contributions:
        raise ValueError("Need at least one contribution")

    n_real = contributions[0].kappa_hat.shape[0]
    n_kappa_inj = contributions[0].kappa_hat.shape[1]

    # Check which contributions have per-realization a_220
    has_a220 = all(
        c.a_220_fit is not None and len(c.a_220_fit) == n_real
        for c in contributions
    )

    stacked_kappa = np.full((n_real, n_kappa_inj), np.nan, dtype=float)
    stacked_sigma = np.full((n_real, n_kappa_inj), np.inf, dtype=float)

    for real_idx in range(n_real):
        for kappa_idx in range(n_kappa_inj):
            events: list[LatentAmplitudeEvent] = []
            for c in contributions:
                kh = c.kappa_hat[real_idx, kappa_idx]
                ks = c.kappa_sigma[real_idx, kappa_idx]
                if not (np.isfinite(kh) and np.isfinite(ks) and ks > 0):
                    continue

                if has_a220:
                    a_220 = max(float(c.a_220_fit[real_idx]), 1e-12)
                else:
                    # Fallback: infer from kappa_sigma structure
                    a_220 = max(1e-12, 0.1)  # placeholder

                a_sq = a_220**2
                sigma_a = float(np.sqrt(max(c.a_220_noise_var, 0.0)))

                events.append(LatentAmplitudeEvent(
                    event_name=c.event_name,
                    c_nl_hat=float(kh * a_sq),
                    sigma_c=max(float(ks * a_sq), 1e-12),
                    a_220_hat=a_220,
                    sigma_a_220=sigma_a,
                    kappa_hat_plugin=float(kh),
                    kappa_sigma_plugin=max(float(ks), 1e-12),
                ))

            if len(events) < 2:
                continue

            try:
                post = estimate_kappa_posterior_latent_amplitude(
                    events,
                    kappa_min=kappa_min,
                    kappa_max=kappa_max,
                    n_kappa=n_kappa,
                    n_amplitude=n_amplitude,
                )
                stacked_kappa[real_idx, kappa_idx] = post.kappa_mean
                stacked_sigma[real_idx, kappa_idx] = post.kappa_std
            except Exception:
                pass

    return {
        "kappa_hat": stacked_kappa,
        "kappa_sigma": stacked_sigma,
    }


def summarize_measurements(
    kappa_hat: np.ndarray,
    kappa_sigma: np.ndarray,
    snr: np.ndarray,
    kappa_true: float,
) -> dict[str, Any]:
    """Compute bias, RMSE, pull, and interval coverage metrics."""
    valid = np.isfinite(kappa_hat) & np.isfinite(kappa_sigma) & (kappa_sigma > 0)
    if not np.any(valid):
        return {
            "n_valid": 0,
            "mean_kappa_hat": np.nan,
            "mean_sigma": np.nan,
            "mean_snr": np.nan,
            "bias": np.nan,
            "bias_over_sigma": np.nan,
            "rmse": np.nan,
            "pull_mean": np.nan,
            "pull_std": np.nan,
            "coverage_68": np.nan,
            "coverage_90": np.nan,
        }

    hat = np.asarray(kappa_hat[valid], dtype=float)
    sigma = np.asarray(kappa_sigma[valid], dtype=float)
    snr_valid = np.asarray(snr[valid], dtype=float)
    pull = (hat - kappa_true) / sigma

    return {
        "n_valid": int(np.count_nonzero(valid)),
        "mean_kappa_hat": float(np.mean(hat)),
        "median_kappa_hat": float(np.median(hat)),
        "mean_sigma": float(np.mean(sigma)),
        "mean_snr": float(np.mean(snr_valid)),
        "bias": float(np.mean(hat - kappa_true)),
        "bias_over_sigma": float(np.mean((hat - kappa_true) / sigma)),
        "rmse": float(np.sqrt(np.mean((hat - kappa_true) ** 2))),
        "pull_mean": float(np.mean(pull)),
        "pull_std": float(np.std(pull)),
        "coverage_68": float(np.mean(np.abs(pull) <= 1.0)),
        "coverage_90": float(np.mean(np.abs(pull) <= norm.ppf(0.95))),
    }


def summarize_campaign_results(
    stacked_results: dict[str, dict[str, Any]],
    kappa_values: list[float],
) -> dict[str, Any]:
    """Summarize the full stacked campaign into scenario-by-scenario metrics."""
    summary: dict[str, Any] = {"scenarios": {}}

    for scenario_name, result in stacked_results.items():
        per_kappa: dict[str, Any] = {}
        aggregate_bias = []
        aggregate_pull_std = []
        aggregate_cov68 = []
        aggregate_cov90 = []

        for idx, kappa_true in enumerate(kappa_values):
            metrics = summarize_measurements(
                result["kappa_hat"][:, idx],
                result["kappa_sigma"][:, idx],
                result["snr"][:, idx],
                kappa_true=kappa_true,
            )
            per_kappa[_format_kappa(kappa_true)] = metrics

            if np.isfinite(metrics["bias"]):
                aggregate_bias.append(abs(metrics["bias"]))
                aggregate_pull_std.append(metrics["pull_std"])
                aggregate_cov68.append(metrics["coverage_68"])
                aggregate_cov90.append(metrics["coverage_90"])

        summary["scenarios"][scenario_name] = {
            "per_kappa": per_kappa,
            "overall": {
                "mean_abs_bias": float(np.mean(aggregate_bias)) if aggregate_bias else np.nan,
                "mean_pull_std": float(np.mean(aggregate_pull_std)) if aggregate_pull_std else np.nan,
                "mean_coverage_68": float(np.mean(aggregate_cov68)) if aggregate_cov68 else np.nan,
                "mean_coverage_90": float(np.mean(aggregate_cov90)) if aggregate_cov90 else np.nan,
            },
        }

    return summary


def write_campaign_csv(summary: dict[str, Any], path: str | Path) -> None:
    """Write the scenario summaries to a flat CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "kappa_true",
                "n_valid",
                "mean_kappa_hat",
                "median_kappa_hat",
                "mean_sigma",
                "mean_snr",
                "bias",
                "bias_over_sigma",
                "rmse",
                "pull_mean",
                "pull_std",
                "coverage_68",
                "coverage_90",
            ]
        )

        for scenario_name, scenario_summary in summary["scenarios"].items():
            for kappa_key, metrics in scenario_summary["per_kappa"].items():
                writer.writerow(
                    [
                        scenario_name,
                        kappa_key,
                        metrics["n_valid"],
                        metrics["mean_kappa_hat"],
                        metrics["median_kappa_hat"],
                        metrics["mean_sigma"],
                        metrics["mean_snr"],
                        metrics["bias"],
                        metrics["bias_over_sigma"],
                        metrics["rmse"],
                        metrics["pull_mean"],
                        metrics["pull_std"],
                        metrics["coverage_68"],
                        metrics["coverage_90"],
                    ]
                )


def plot_campaign_summary(summary: dict[str, Any], path: str | Path) -> None:
    """Plot bias and coverage for the t_start and band sweeps."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scenario_map = summary["scenarios"]
    tstart_names = sorted(
        [name for name in scenario_map if name.startswith("fixed_tstart_")],
        key=lambda name: int(name.split("_")[2].rstrip("M")),
    )
    band_names = sorted(
        [name for name in scenario_map if name.startswith("marginalized_")]
    )

    def _extract_series(names: list[str], metric: str) -> tuple[np.ndarray, dict[str, list[float]]]:
        kappa_grid = np.array(
            [float(key) for key in next(iter(scenario_map.values()))["per_kappa"].keys()],
            dtype=float,
        )
        series: dict[str, list[float]] = {}
        for name in names:
            values = []
            for key in next(iter(scenario_map.values()))["per_kappa"].keys():
                values.append(float(scenario_map[name]["per_kappa"][key][metric]))
            series[name] = values
        return kappa_grid, series

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    kappa_grid, tstart_bias = _extract_series(tstart_names, "bias")
    _, tstart_cov = _extract_series(tstart_names, "coverage_90")
    _, band_bias = _extract_series(band_names, "bias")
    _, band_cov = _extract_series(band_names, "coverage_90")

    for name, values in tstart_bias.items():
        axes[0, 0].plot(kappa_grid, values, marker="o", label=name.replace("fixed_tstart_", "t=").replace("M", "M"))
    axes[0, 0].axhline(0.0, color="black", lw=1, alpha=0.6)
    axes[0, 0].set_title("Fixed t_start Bias")
    axes[0, 0].set_xlabel("Injected kappa")
    axes[0, 0].set_ylabel("Mean bias")

    for name, values in tstart_cov.items():
        axes[0, 1].plot(kappa_grid, values, marker="o", label=name.replace("fixed_tstart_", "t=").replace("M", "M"))
    axes[0, 1].axhline(0.90, color="black", lw=1, alpha=0.6, linestyle="--")
    axes[0, 1].set_title("Fixed t_start 90% Coverage")
    axes[0, 1].set_xlabel("Injected kappa")
    axes[0, 1].set_ylabel("Coverage")

    for name, values in band_bias.items():
        axes[1, 0].plot(kappa_grid, values, marker="o", label=name.replace("marginalized_", ""))
    axes[1, 0].axhline(0.0, color="black", lw=1, alpha=0.6)
    axes[1, 0].set_title("Whitening-Band Bias")
    axes[1, 0].set_xlabel("Injected kappa")
    axes[1, 0].set_ylabel("Mean bias")

    for name, values in band_cov.items():
        axes[1, 1].plot(kappa_grid, values, marker="o", label=name.replace("marginalized_", ""))
    axes[1, 1].axhline(0.90, color="black", lw=1, alpha=0.6, linestyle="--")
    axes[1, 1].set_title("Whitening-Band 90% Coverage")
    axes[1, 1].set_xlabel("Injected kappa")
    axes[1, 1].set_ylabel("Coverage")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("GRIM-S Phase 3 Injection Campaign", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_phase3_weight_diagnostics(
    phase3_results_path: str | Path = DEFAULT_PHASE3_RESULTS_PATH,
    jackknife_path: str | Path | None = Path("results/grims/phase3_jackknife.json"),
) -> dict[str, Any]:
    """Summarize the current event-weight concentration in the Phase 3 stack."""
    phase3 = _load_json(phase3_results_path)
    rows = phase3["individual"]

    weighted_rows = []
    total_weight = 0.0
    for row in rows:
        sigma = float(row.get("kappa_sigma", np.inf))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        weight = 1.0 / sigma**2
        total_weight += weight
        weighted_rows.append(
            {
                "event": row["event"],
                "weight": weight,
                "best_t_start_m": row.get("best_t_start_m"),
                "segment_duration": row.get("seg_duration"),
                "n_detectors": row.get("n_detectors", 1),
                "kappa_sigma": sigma,
                "kappa_hat": row.get("kappa_hat"),
            }
        )

    weighted_rows.sort(key=lambda row: row["weight"], reverse=True)
    for row in weighted_rows:
        row["weight_fraction"] = row["weight"] / total_weight if total_weight > 0 else 0.0

    top10 = weighted_rows[:10]
    top20 = weighted_rows[:20]
    top40 = weighted_rows[:40]
    top10_events = {row["event"] for row in top10}

    influential_events = []
    if jackknife_path is not None and Path(jackknife_path).exists():
        jackknife = _load_json(jackknife_path)
        influential_events = list(jackknife.get("influential_events", []))

    overlap = sorted(top10_events.intersection(influential_events))
    thresholds_ms = (30.0, 40.0, 50.0, 80.0)
    segment_weight_fractions = {}
    for threshold_ms in thresholds_ms:
        threshold_s = threshold_ms / 1000.0
        fraction = sum(
            row["weight_fraction"]
            for row in weighted_rows
            if row["segment_duration"] is not None and row["segment_duration"] <= threshold_s + 1e-9
        )
        segment_weight_fractions[f"le_{int(threshold_ms)}ms"] = float(fraction)

    return {
        "n_events": len(weighted_rows),
        "top10_cumulative_weight": float(sum(row["weight_fraction"] for row in top10)),
        "top5_cumulative_weight": float(sum(row["weight_fraction"] for row in weighted_rows[:5])),
        "top20_cumulative_weight": float(sum(row["weight_fraction"] for row in top20)),
        "top40_cumulative_weight": float(sum(row["weight_fraction"] for row in top40)),
        "top10_segment_durations": [row["segment_duration"] for row in top10],
        "top10_best_t_start_m": [row["best_t_start_m"] for row in top10],
        "top10_n_detectors": [row["n_detectors"] for row in top10],
        "segment_weight_fractions": segment_weight_fractions,
        "top10_events": top10,
        "jackknife_influential_overlap": overlap,
        "jackknife_overlap_count": len(overlap),
    }


def _prepare_detector_data(
    catalog_event: dict[str, Any],
    detector: str,
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Load a detector time series and estimate its ASD once."""
    strain_path = find_local_strain_detector(catalog_event, str(data_dir), detector)
    if strain_path is None or not is_valid_hdf5_file(strain_path):
        strain_path = _download_catalog_strain_file(catalog_event, detector, data_dir)

    loaded = load_gwosc_strain_hdf5(strain_path)
    strain = _sanitize_strain(loaded["strain"])
    time = loaded["time"]
    sample_rate = float(loaded["sample_rate"])

    strain, time = _crop_around_merger(
        strain,
        time,
        float(catalog_event["gps"]),
        window_seconds=32.0,
    )
    strain = _sanitize_strain(strain)

    asd_freqs, asd = estimate_asd(
        strain,
        sample_rate,
        merger_time=float(catalog_event["gps"]),
        time=time,
        exclusion_window=2.0,
    )

    return strain, time, sample_rate, asd_freqs, asd


def _compute_band_edges(
    mass: float,
    spin: float,
    sample_rate: float,
    band: WhiteningBand,
) -> tuple[float, float]:
    """Compute physical band edges for one event and band definition."""
    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)
    mode_440 = catalog.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)

    f_low = max(20.0, f_220 * band.fmin_pad)
    f_high = min(0.45 * sample_rate, max(f_nl, f_440) * band.fmax_pad)
    return float(f_low), float(f_high)


def run_phase3_injection_campaign(
    phase3_results_path: str | Path = DEFAULT_PHASE3_RESULTS_PATH,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    data_dir: str | Path = "data",
    kappa_values: list[float] | tuple[float, ...] = DEFAULT_KAPPA_VALUES,
    t_start_values: list[float] | tuple[float, ...] = DEFAULT_T_START_VALUES,
    bands: list[WhiteningBand] | None = None,
    max_events: int | None = None,
    event_selection: str = "largest_weight",
    n_realizations: int = 16,
    seed: int = 42,
    max_weight_ratio: float | None = 5.5,
    t_start_strategy: str = DEFAULT_T_START_STRATEGY,
    min_segment_duration: float = DEFAULT_MIN_SEGMENT_DURATION,
    include_higher_linear_modes: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the realistic-SNR injection campaign on the Phase 3 stack."""
    kappa_values = [float(value) for value in kappa_values]
    t_start_values = [float(value) for value in t_start_values]
    bands = list(default_whitening_bands() if bands is None else bands)
    phase3 = _load_json(phase3_results_path)
    catalog = _load_json(catalog_path)
    catalog_by_name = {event["name"]: event for event in catalog}

    selected_entries = _select_phase3_entries(
        phase3["individual"],
        max_events=max_events,
        selection=event_selection,
    )
    scenarios = build_scenarios(t_start_values=t_start_values, bands=bands)

    scenario_contributions: dict[str, list[EventScenarioContribution]] = {
        scenario.name: [] for scenario in scenarios
    }
    per_event_metadata: list[dict[str, Any]] = []

    master_rng = np.random.default_rng(seed)

    for event_idx, entry in enumerate(selected_entries, start=1):
        event_name = entry["event"]
        catalog_event = catalog_by_name.get(event_name)
        if catalog_event is None:
            raise KeyError(f"Missing catalog entry for {event_name}")

        mass = float(catalog_event["remnant_mass"])
        spin = float(catalog_event["remnant_spin"])
        gps = float(catalog_event["gps"])
        m_seconds = mass * M_SUN_SECONDS
        segment_duration = compute_optimal_segment_duration(
            mass,
            spin,
            n_damping_times=5.0,
            min_duration=min_segment_duration,
        )
        detectors = list(entry.get("detectors_used", ["H1"]))

        if progress:
            print(
                f"[{event_idx:3d}/{len(selected_entries)}] {event_name} "
                f"({','.join(detectors)}) seg={segment_duration * 1000:.0f} ms",
                flush=True,
            )

        detector_results: dict[str, dict[str, dict[float, dict[str, Any]]]] = {
            detector: {} for detector in detectors
        }
        detector_metadata: dict[str, Any] = {}

        for detector in detectors:
            strain, time, sample_rate, asd_freqs, asd = _prepare_detector_data(
                catalog_event,
                detector,
                data_dir,
            )

            n_samples = int(round((DEFAULT_PAD_BEFORE_S + segment_duration) * sample_rate)) + 1
            noise_mask = np.abs(time - gps) > 4.0
            candidate_starts = _valid_noise_window_starts(noise_mask, n_samples)
            if len(candidate_starts) == 0:
                continue

            realization_starts = master_rng.choice(
                candidate_starts,
                size=n_realizations,
                replace=True,
            )

            detector_metadata[detector] = {
                "candidate_windows": int(len(candidate_starts)),
                "sample_rate": float(sample_rate),
            }

            for band in bands:
                f_low, f_high = _compute_band_edges(mass, spin, sample_rate, band)
                whitened = whiten_strain(
                    strain,
                    sample_rate,
                    asd_freqs,
                    asd,
                    fmin=f_low * 0.8,
                )
                filtered = bandpass(whitened, sample_rate, f_low, f_high)
                noise_rms = float(np.sqrt(np.var(filtered[noise_mask])))

                band_results: dict[float, dict[str, Any]] = {}
                for t_start_m in t_start_values:
                    on_source_start = gps + t_start_m * m_seconds - DEFAULT_PAD_BEFORE_S
                    on_index = _nearest_start_index(time, on_source_start, n_samples)
                    if on_index is None:
                        continue

                    seg_time = time[on_index : on_index + n_samples]
                    t_dimless = (seg_time - (gps + t_start_m * m_seconds)) / m_seconds
                    if np.count_nonzero(t_dimless >= 0) < 4:
                        continue

                    seg_strain = filtered[on_index : on_index + n_samples]
                    fit = fit_linear_modes(seg_strain, t_dimless, spin)
                    if fit["220"]["amplitude"] <= 0:
                        continue

                    injection_waveforms = [
                        _build_injection_waveform(
                            t_dimless,
                            spin=spin,
                            fit=fit,
                            kappa=kappa,
                            include_higher_linear_modes=include_higher_linear_modes,
                        )
                        for kappa in kappa_values
                    ]

                    kappa_hat = np.full((n_realizations, len(kappa_values)), np.nan, dtype=float)
                    kappa_sigma = np.full((n_realizations, len(kappa_values)), np.inf, dtype=float)
                    snr = np.zeros((n_realizations, len(kappa_values)), dtype=float)
                    a_220_per_real = np.zeros(n_realizations, dtype=float)
                    a_220_noise_var_val = 0.0

                    for real_idx, start_idx in enumerate(realization_starts):
                        noise_segment = filtered[start_idx : start_idx + n_samples]
                        if len(noise_segment) != n_samples:
                            continue

                        for kappa_idx, waveform in enumerate(injection_waveforms):
                            result = phase_locked_search_colored(
                                noise_segment + waveform,
                                t_dimless,
                                spin,
                                noise_rms,
                                event_name=(
                                    f"{event_name}_{detector}_{band.label}"
                                    f"_t{int(t_start_m)}_k{_format_kappa(kappa_values[kappa_idx])}"
                                ),
                            )
                            kappa_hat[real_idx, kappa_idx] = result.kappa_hat
                            kappa_sigma[real_idx, kappa_idx] = result.kappa_sigma
                            snr[real_idx, kappa_idx] = result.snr
                            if kappa_idx == 0:
                                a_220_per_real[real_idx] = result.a_220_fit
                                a_220_noise_var_val = getattr(result, "a_220_noise_var", 0.0)

                    band_results[float(t_start_m)] = {
                        "kappa_hat": kappa_hat,
                        "kappa_sigma": kappa_sigma,
                        "snr": snr,
                        "a_220_fit": float(fit["220"]["amplitude"]),
                        "a_220_per_real": a_220_per_real,
                        "a_220_noise_var": a_220_noise_var_val,
                        "a_330_fit": float(fit.get("330", {}).get("amplitude", 0.0)),
                        "a_440_fit": float(fit.get("440", {}).get("amplitude", 0.0)),
                        "noise_rms": noise_rms,
                    }

                detector_results[detector][band.label] = band_results

        # Fixed t_start scenarios on the default band.
        for scenario in scenarios:
            per_detector_kappa: list[np.ndarray] = []
            per_detector_sigma: list[np.ndarray] = []
            per_detector_snr: list[np.ndarray] = []
            first_a220_per_real: np.ndarray | None = None
            first_a220_noise_var: float = 0.0

            for detector in detectors:
                detector_band_results = detector_results.get(detector, {})
                if scenario.band_label not in detector_band_results:
                    continue

                if scenario.t_start_mode == "fixed":
                    payload = detector_band_results[scenario.band_label].get(float(scenario.t_start_m))
                    if payload is None:
                        continue
                    per_detector_kappa.append(payload["kappa_hat"])
                    per_detector_sigma.append(payload["kappa_sigma"])
                    per_detector_snr.append(payload["snr"])
                    if first_a220_per_real is None and "a_220_per_real" in payload:
                        first_a220_per_real = payload["a_220_per_real"]
                        first_a220_noise_var = payload.get("a_220_noise_var", 0.0)
                else:
                    t_payloads = list(detector_band_results[scenario.band_label].values())
                    if not t_payloads:
                        continue
                    combined = _combine_measurement_arrays(
                        [payload["kappa_hat"] for payload in t_payloads],
                        [payload["kappa_sigma"] for payload in t_payloads],
                        [payload["snr"] for payload in t_payloads],
                        strategy=t_start_strategy,
                    )
                    per_detector_kappa.append(combined[0])
                    per_detector_sigma.append(combined[1])
                    per_detector_snr.append(combined[2])
                    # Use first t_start's a_220 as representative
                    if first_a220_per_real is None and "a_220_per_real" in t_payloads[0]:
                        first_a220_per_real = t_payloads[0]["a_220_per_real"]
                        first_a220_noise_var = t_payloads[0].get("a_220_noise_var", 0.0)

            if not per_detector_kappa:
                continue

            combined_event = _combine_measurement_arrays(
                per_detector_kappa,
                per_detector_sigma,
                per_detector_snr,
                strategy="independent",
            )
            scenario_contributions[scenario.name].append(
                EventScenarioContribution(
                    event_name=event_name,
                    kappa_hat=combined_event[0],
                    kappa_sigma=combined_event[1],
                    snr=combined_event[2],
                    detectors_used=detectors,
                    segment_duration=segment_duration,
                    best_t_start_m=entry.get("best_t_start_m"),
                    a_220_fit=first_a220_per_real,
                    a_220_noise_var=first_a220_noise_var,
                    spin=spin,
                    noise_rms=float(noise_rms) if isinstance(noise_rms, (int, float)) else 0.0,
                )
            )

        per_event_metadata.append(
            {
                "event": event_name,
                "detectors_used": detectors,
                "segment_duration": segment_duration,
                "phase3_best_t_start_m": entry.get("best_t_start_m"),
                "phase3_sigma": entry.get("kappa_sigma"),
                "phase3_kappa_hat": entry.get("kappa_hat"),
                "detector_metadata": detector_metadata,
            }
        )

    stacked_results: dict[str, dict[str, Any]] = {}
    marginalized_results: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        contributions = scenario_contributions[scenario.name]
        if not contributions:
            continue
        stacked = _stack_event_contributions(
            contributions,
            max_weight_ratio=max_weight_ratio,
        )
        stacked["event_names"] = [entry.event_name for entry in contributions]
        stacked_results[scenario.name] = stacked

        # Run the marginalized estimator alongside the legacy stack
        has_a220 = all(
            c.a_220_fit is not None and hasattr(c.a_220_fit, '__len__')
            for c in contributions
        )
        if has_a220 and progress:
            print(f"  Running marginalized estimator for {scenario.name}...", flush=True)
        if has_a220:
            try:
                marg = _stack_event_contributions_marginalized(
                    contributions,
                    kappa_min=-0.15,
                    kappa_max=0.15,
                    n_kappa=201,
                    n_amplitude=81,
                )
                marginalized_results[scenario.name] = marg
            except Exception as exc:
                if progress:
                    print(f"  Marginalized stacking failed for {scenario.name}: {exc}", flush=True)

    summary = summarize_campaign_results(stacked_results, kappa_values)
    weight_diagnostics = compute_phase3_weight_diagnostics(phase3_results_path=phase3_results_path)

    # Summarize marginalized results alongside legacy
    marginalized_summary: dict[str, Any] = {}
    for scenario_name, marg_payload in marginalized_results.items():
        marg_kh = marg_payload["kappa_hat"]
        marg_ks = marg_payload["kappa_sigma"]
        per_kappa: dict[str, Any] = {}
        for kidx, ktrue in enumerate(kappa_values):
            valid = np.isfinite(marg_kh[:, kidx])
            if np.any(valid):
                hats = marg_kh[valid, kidx]
                per_kappa[str(ktrue)] = {
                    "mean_kappa_hat": float(np.mean(hats)),
                    "recovery": float(np.mean(hats) / ktrue) if ktrue > 0 else None,
                    "n_valid": int(np.sum(valid)),
                }
        marginalized_summary[scenario_name] = per_kappa

    serializable_results = {
        "metadata": {
            "phase3_results_path": str(phase3_results_path),
            "catalog_path": str(catalog_path),
            "data_dir": str(data_dir),
            "n_realizations": int(n_realizations),
            "seed": int(seed),
            "kappa_values": list(kappa_values),
            "t_start_values": list(t_start_values),
            "bands": [band.__dict__ for band in bands],
            "event_selection": event_selection,
            "max_events": max_events,
            "selected_events": [entry["event"] for entry in selected_entries],
            "max_weight_ratio": max_weight_ratio,
            "t_start_strategy": t_start_strategy,
            "min_segment_duration": min_segment_duration,
        },
        "summary": summary,
        "weight_diagnostics": weight_diagnostics,
        "per_event_metadata": per_event_metadata,
        "stacked_realizations": {
            scenario: {
                "kappa_hat": payload["kappa_hat"].tolist(),
                "kappa_sigma": payload["kappa_sigma"].tolist(),
                "snr": payload["snr"].tolist(),
                "event_names": payload["event_names"],
            }
            for scenario, payload in stacked_results.items()
        },
        "marginalized_realizations": {
            scenario: {
                "kappa_hat": payload["kappa_hat"].tolist(),
                "kappa_sigma": payload["kappa_sigma"].tolist(),
            }
            for scenario, payload in marginalized_results.items()
        },
        "marginalized_summary": marginalized_summary,
    }
    return serializable_results
