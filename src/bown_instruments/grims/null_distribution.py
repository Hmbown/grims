"""
Empirical null calibration for the stacked GRIM-S kappa estimator.

The Phase 3 measurement is a weighted stack of per-event matched-filter
estimators. This module builds an empirical null by preserving each
event's fitted linear ringdown template and noise level, while breaking
the phase coherence between the residual and the nonlinear template.

Primary method
--------------
``circular_time_shift``
    Apply a circular shift to the post-subtraction residual for each
    event component. This preserves the residual amplitude distribution
    while scrambling its phase relative to the nonlinear template.

Backup method
-------------
``fourier_phase_randomization``
    Preserve the residual amplitude spectrum while replacing its Fourier
    phases with random values.

The expensive work is done once per event: load strain, whiten, bandpass,
fit the linear ringdown channel, and cache the residual/template pair.
Each null realization then recomputes only the residual-template inner
product and the stack.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from .gwtc_pipeline import M_SUN_SECONDS, is_valid_hdf5_file, load_gwosc_strain_hdf5
from .mass_analysis import find_local_strain_detector
from .phase_locked_search import PhaseLockResult, stack_phase_locked
from .qnm_modes import KerrQNMCatalog
from .whiten import bandpass, estimate_asd, whiten_strain


DEFAULT_PHASE3_RESULTS_PATH = Path("results/grims/phase3_results.json")
DEFAULT_CATALOG_PATH = Path("results/grims/gwtc_full_catalog.json")
DEFAULT_OUTPUT_PATH = Path("results/grims/phase3_null_distribution.json")


@dataclass
class NullProjectionComponent:
    """Cached residual/template pair for one detector and start time."""

    detector: str
    t_start_m: float
    residual: np.ndarray
    template_fft: np.ndarray
    noise_var: float
    template_inner: float
    template_norm: float
    kappa_hat_real: float
    kappa_sigma_real: float
    min_shift_samples: int
    n_samples: int


@dataclass
class EventNullPreparation:
    """All cached ingredients needed to null-scramble one Phase 3 event."""

    event_name: str
    detectors_used: list[str]
    detector_components: dict[str, list[NullProjectionComponent]] = field(default_factory=dict)
    real_kappa_hat: float = 0.0
    real_kappa_sigma: float = np.inf
    per_detector_real: dict[str, dict[str, float]] = field(default_factory=dict)
    noise_rms_reference: float = 1.0


def _fd_inner(
    a_fft: np.ndarray,
    b_fft: np.ndarray,
    n_samples: int,
    noise_var: float,
) -> float:
    """Mirror the colored-search inner-product normalization exactly."""
    safe_noise_var = noise_var if noise_var > 0 else 1.0
    return float(np.sum((a_fft.conj() * b_fft).real) / (n_samples * safe_noise_var))


def _combine_kappas(kappas: list[float], sigmas: list[float]) -> tuple[float, float]:
    """Inverse-variance combine kappa measurements."""
    if len(kappas) != len(sigmas):
        raise ValueError("kappas and sigmas must have the same length")

    weights = []
    valid_kappas = []
    for kappa, sigma in zip(kappas, sigmas):
        if sigma > 0 and np.isfinite(sigma):
            weights.append(1.0 / sigma**2)
            valid_kappas.append(kappa)

    if not weights:
        raise ValueError("No finite measurements available for combination")

    weights_array = np.asarray(weights, dtype=float)
    kappas_array = np.asarray(valid_kappas, dtype=float)
    total_weight = float(np.sum(weights_array))
    return (
        float(np.sum(weights_array * kappas_array) / total_weight),
        float(1.0 / np.sqrt(total_weight)),
    )


def _phase_randomize_residual(residual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve the residual amplitude spectrum while randomizing phase."""
    residual_fft = np.fft.rfft(residual)
    randomized_fft = residual_fft.copy()

    if len(randomized_fft) <= 1:
        return residual.copy()

    if len(residual) % 2 == 0:
        phase_slice = slice(1, -1)
    else:
        phase_slice = slice(1, None)

    amplitudes = np.abs(randomized_fft[phase_slice])
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(amplitudes))
    randomized_fft[phase_slice] = amplitudes * np.exp(1j * phases)

    return np.fft.irfft(randomized_fft, n=len(residual))


def _shift_samples_from_fraction(
    n_samples: int,
    min_shift_samples: int,
    fraction: float,
) -> int:
    """Map a unit-interval draw onto a valid circular shift."""
    if n_samples < 2:
        raise ValueError("Need at least two samples for a circular shift")

    min_shift = max(1, min_shift_samples)
    low = min_shift
    high = n_samples - min_shift

    if low > high:
        valid_shifts = np.arange(1, n_samples, dtype=int)
    else:
        valid_shifts = np.arange(low, high + 1, dtype=int)

    if len(valid_shifts) == 0:
        raise ValueError("No valid circular shifts available")

    clipped = float(np.clip(fraction, 0.0, np.nextafter(1.0, 0.0)))
    index = min(int(np.floor(clipped * len(valid_shifts))), len(valid_shifts) - 1)
    return int(valid_shifts[index])


def _gaussian_log_bayes_factor(
    kappa_hat: float,
    sigma: float,
    kappa_min: float = 0.0,
    kappa_max: float = 5.0,
) -> float:
    """Approximate log Bayes factor for a Gaussian matched-filter measurement."""
    if sigma <= 0 or not np.isfinite(sigma):
        return float("-inf")

    width = kappa_max - kappa_min
    if width <= 0:
        raise ValueError("kappa_max must be greater than kappa_min")

    cdf_span = norm.cdf((kappa_max - kappa_hat) / sigma) - norm.cdf((kappa_min - kappa_hat) / sigma)
    cdf_span = max(float(cdf_span), 1e-300)

    log_integral = 0.5 * np.log(2.0 * np.pi) + np.log(sigma) + np.log(cdf_span)
    return float(-np.log(width) + log_integral + 0.5 * (kappa_hat / sigma) ** 2)


def _prepare_colored_component(
    strain: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    noise_rms: float,
    detector: str,
    t_start_m: float,
) -> NullProjectionComponent | None:
    """Mirror ``phase_locked_search_colored`` and cache its ingredients.

    This duplicates the search logic intentionally so the null campaign can
    retain the residual and template rather than only the summary statistic.
    The normalization and template construction match the current analysis
    code exactly.
    """
    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    nl_mode = catalog.nonlinear_mode_quadratic(spin)

    mask = t_dimless >= 0
    t_pos = t_dimless[mask]
    d_pos = strain[mask]
    n_samples = len(d_pos)

    if n_samples < 4:
        return None

    data_fft = np.fft.rfft(d_pos)

    omega = mode_220.omega
    envelope = np.exp(omega.imag * t_pos)
    basis_cos = envelope * np.cos(omega.real * t_pos)
    basis_sin = envelope * np.sin(omega.real * t_pos)

    cos_fft = np.fft.rfft(basis_cos)
    sin_fft = np.fft.rfft(basis_sin)

    noise_var = noise_rms**2 if noise_rms > 0 else 1.0
    cc = _fd_inner(cos_fft, cos_fft, n_samples, noise_var)
    ss = _fd_inner(sin_fft, sin_fft, n_samples, noise_var)
    cs = _fd_inner(cos_fft, sin_fft, n_samples, noise_var)
    dc = _fd_inner(data_fft, cos_fft, n_samples, noise_var)
    ds = _fd_inner(data_fft, sin_fft, n_samples, noise_var)

    determinant = cc * ss - cs * cs
    if abs(determinant) < 1e-30:
        return None

    a_coeff = (dc * ss - ds * cs) / determinant
    b_coeff = (ds * cc - dc * cs) / determinant

    amplitude = np.sqrt(a_coeff**2 + b_coeff**2)
    phase = np.arctan2(-b_coeff, a_coeff)

    fitted_fft = a_coeff * cos_fft + b_coeff * sin_fft
    residual_fft = data_fft - fitted_fft
    residual = np.fft.irfft(residual_fft, n=n_samples)

    # Match phase_locked_search_colored exactly: it reuses the 220 envelope.
    omega_nl = nl_mode.omega
    a_nl = amplitude**2
    phi_nl = 2.0 * phase
    nl_template = a_nl * envelope * np.cos(omega_nl.real * t_pos + phi_nl)
    template_fft = np.fft.rfft(nl_template)

    template_inner = _fd_inner(template_fft, template_fft, n_samples, noise_var)
    if template_inner <= 0:
        return None

    residual_overlap = _fd_inner(residual_fft, template_fft, n_samples, noise_var)
    template_norm = np.sqrt(template_inner)
    kappa_hat = residual_overlap / template_inner
    kappa_sigma = 1.0 / template_norm

    dt_dimless = float(np.median(np.diff(t_pos))) if len(t_pos) > 1 else 1.0
    nl_period_dimless = 2.0 * np.pi / max(omega_nl.real, 1e-12)
    min_shift_samples = max(1, int(np.ceil(nl_period_dimless / max(dt_dimless, 1e-12))))

    return NullProjectionComponent(
        detector=detector,
        t_start_m=float(t_start_m),
        residual=residual,
        template_fft=template_fft,
        noise_var=float(noise_var),
        template_inner=float(template_inner),
        template_norm=float(template_norm),
        kappa_hat_real=float(kappa_hat),
        kappa_sigma_real=float(kappa_sigma),
        min_shift_samples=min_shift_samples,
        n_samples=n_samples,
    )


@lru_cache(maxsize=None)
def _fetch_event_bundle(json_url: str) -> dict[str, Any]:
    """Fetch and cache a GWOSC event metadata bundle."""
    with _urlopen_with_retry(json_url, timeout=60) as response:
        payload = json.load(response)
    return next(iter(payload["events"].values()))


def _urlopen_with_retry(url: str, timeout: float, attempts: int = 5):
    """Open a URL with retries for transient GWOSC proxy failures."""
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except Exception:  # pragma: no cover - exercised in live runs
            if attempt + 1 == attempts:
                raise
            time.sleep(1.0 + attempt)


def _download_catalog_strain_file(
    catalog_event: dict[str, Any],
    detector: str,
    data_dir: str | Path,
    sample_rate: int = 4096,
    duration: int = 32,
) -> str:
    """Download a GWOSC strain file for a catalog event when not cached locally."""
    event_bundle = _fetch_event_bundle(catalog_event["jsonurl"])
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    matches = [
        item
        for item in event_bundle["strain"]
        if item["detector"] == detector
        and item["sampling_rate"] == sample_rate
        and item["duration"] == duration
        and item["format"] == "hdf5"
    ]

    if not matches:
        all_hdf5 = [item for item in event_bundle["strain"] if item["format"] == "hdf5"]
        ranked = sorted(
            all_hdf5,
            key=lambda item: (
                0 if item["detector"] == detector else 1,
                abs(item["duration"] - duration),
                abs(item["sampling_rate"] - sample_rate),
            ),
        )
        matches = ranked[:1]

    if not matches:
        raise FileNotFoundError(f"No HDF5 strain file found for {catalog_event['name']} {detector}")

    remote = matches[0]
    local_file = data_path / Path(remote["url"]).name
    if is_valid_hdf5_file(local_file):
        return str(local_file)
    if local_file.exists():
        local_file.unlink()

    partial_file = local_file.with_name(f"{local_file.name}.partial")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if partial_file.exists():
                partial_file.unlink()
            with (
                _urlopen_with_retry(remote["url"], timeout=120) as response,
                partial_file.open("wb") as handle,
            ):
                handle.write(response.read())
            if not is_valid_hdf5_file(partial_file):
                raise OSError(f"Downloaded invalid HDF5 strain file from {remote['url']}")
            partial_file.replace(local_file)
            return str(local_file)
        except Exception as exc:  # pragma: no cover - exercised in live runs
            last_error = exc
            if partial_file.exists():
                partial_file.unlink()
            if local_file.exists() and not is_valid_hdf5_file(local_file):
                local_file.unlink()
            if attempt + 1 < 3:
                time.sleep(1.0 + attempt)
    if last_error is not None:
        raise OSError(
            f"Failed to download a valid HDF5 strain file for {catalog_event['name']} {detector}"
        ) from last_error

    return str(local_file)


def _load_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def _crop_around_merger(
    strain: np.ndarray,
    time: np.ndarray,
    merger_time: float,
    window_seconds: float = 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce very long LOSC files to the local 32-second analysis window."""
    if len(time) == 0 or (time[-1] - time[0]) <= window_seconds:
        return strain, time

    half_window = 0.5 * window_seconds
    mask = (time >= merger_time - half_window) & (time <= merger_time + half_window)
    if np.count_nonzero(mask) < 256:
        return strain, time
    return strain[mask], time[mask]


def _sanitize_strain(strain: np.ndarray) -> np.ndarray:
    """Fill non-finite samples so ASD estimation and whitening remain defined."""
    finite = np.isfinite(strain)
    if np.all(finite):
        return strain
    if not np.any(finite):
        raise ValueError("strain series contains no finite samples")

    x = np.arange(len(strain), dtype=float)
    sanitized = np.asarray(strain, dtype=float).copy()
    sanitized[~finite] = np.interp(x[~finite], x[finite], sanitized[finite])
    return sanitized


def prepare_phase3_null_inputs(
    phase3_results_path: str | Path = DEFAULT_PHASE3_RESULTS_PATH,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    data_dir: str | Path = "data",
    progress: bool = False,
) -> tuple[list[EventNullPreparation], dict[str, Any]]:
    """Prepare cached residual/template inputs for the full Phase 3 catalog."""
    phase3 = _load_json(phase3_results_path)
    catalog = _load_json(catalog_path)
    catalog_by_name = {event["name"]: event for event in catalog}

    preparations: list[EventNullPreparation] = []
    missing_events: list[str] = []

    for index, entry in enumerate(phase3["individual"], start=1):
        event_name = entry["event"]
        catalog_event = catalog_by_name.get(event_name)
        if catalog_event is None:
            missing_events.append(event_name)
            continue

        if progress:
            print(f"[{index:3d}/{len(phase3['individual'])}] preparing {event_name}", flush=True)

        preparation = _prepare_single_phase3_event(entry, catalog_event, data_dir)
        preparations.append(preparation)

    if missing_events:
        raise KeyError(f"Missing catalog entries for: {', '.join(missing_events)}")

    return preparations, phase3["stacked"]


def _prepare_single_phase3_event(
    phase3_entry: dict[str, Any],
    catalog_event: dict[str, Any],
    data_dir: str | Path,
) -> EventNullPreparation:
    """Build cached null inputs for one Phase 3 event."""
    mass = float(catalog_event["remnant_mass"])
    spin = float(catalog_event["remnant_spin"])
    gps = float(catalog_event["gps"])
    m_seconds = mass * M_SUN_SECONDS

    detectors = list(phase3_entry.get("detectors_used", ["H1"]))
    t_start_values = list(
        phase3_entry.get("t_start_values", [phase3_entry.get("best_t_start_m", 10.0)])
    )
    seg_duration = float(phase3_entry.get("seg_duration", 0.15))

    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)
    mode_440 = catalog.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)
    f_low = max(20.0, f_220 * 0.5)

    preparation = EventNullPreparation(
        event_name=phase3_entry["event"],
        detectors_used=detectors,
        real_kappa_hat=float(phase3_entry["kappa_hat"]),
        real_kappa_sigma=float(phase3_entry["kappa_sigma"]),
        noise_rms_reference=float(phase3_entry.get("noise_rms", 1.0)),
    )

    detector_kappas: list[float] = []
    detector_sigmas: list[float] = []

    for detector in detectors:
        strain_path = find_local_strain_detector(catalog_event, str(data_dir), detector)
        if strain_path is None or not is_valid_hdf5_file(strain_path):
            strain_path = _download_catalog_strain_file(catalog_event, detector, data_dir)

        loaded = load_gwosc_strain_hdf5(strain_path)
        strain = loaded["strain"]
        time = loaded["time"]
        strain, time = _crop_around_merger(strain, time, gps, window_seconds=32.0)
        strain = _sanitize_strain(strain)
        sample_rate = loaded["sample_rate"]

        f_high = min(0.45 * sample_rate, max(f_nl, f_440) * 1.3)
        asd_freqs, asd = estimate_asd(
            strain,
            sample_rate,
            merger_time=gps,
            time=time,
            exclusion_window=2.0,
        )
        whitened = whiten_strain(strain, sample_rate, asd_freqs, asd, fmin=f_low * 0.8)
        whitened_bp = bandpass(whitened, sample_rate, f_low, f_high)

        noise_mask = np.abs(time - gps) > 4.0
        noise_var = float(np.var(whitened_bp[noise_mask]))
        noise_rms = float(np.sqrt(noise_var))

        components: list[NullProjectionComponent] = []
        for t_start_m in t_start_values:
            ringdown_start = gps + float(t_start_m) * m_seconds
            t_start = ringdown_start - 0.05
            t_end = ringdown_start + seg_duration
            mask = (time >= t_start) & (time <= t_end)

            if int(np.sum(mask)) < 50:
                continue

            seg_strain = whitened_bp[mask]
            seg_time = time[mask]
            t_dimless = (seg_time - ringdown_start) / m_seconds

            component = _prepare_colored_component(
                seg_strain,
                t_dimless,
                spin,
                noise_rms,
                detector=detector,
                t_start_m=float(t_start_m),
            )
            if component is not None:
                components.append(component)

        if not components:
            raise RuntimeError(
                f"No valid null components prepared for {phase3_entry['event']} {detector}"
            )

        det_kappa, det_sigma = _combine_kappas(
            [component.kappa_hat_real for component in components],
            [component.kappa_sigma_real for component in components],
        )
        preparation.detector_components[detector] = components
        preparation.per_detector_real[detector] = {
            "kappa_hat": det_kappa,
            "kappa_sigma": det_sigma,
        }
        detector_kappas.append(det_kappa)
        detector_sigmas.append(det_sigma)

    event_kappa, event_sigma = _combine_kappas(detector_kappas, detector_sigmas)
    preparation.real_kappa_hat = event_kappa
    preparation.real_kappa_sigma = event_sigma

    return preparation


def generate_event_null_realization(
    preparation: EventNullPreparation,
    rng: np.random.Generator,
    method: str = "circular_time_shift",
    shift_fraction: float | None = None,
    event_seed: int | None = None,
) -> tuple[PhaseLockResult, dict[str, Any]]:
    """Generate one null realization for a single stacked Phase 3 event."""
    detector_kappas: list[float] = []
    detector_sigmas: list[float] = []
    component_shift_samples: dict[str, list[int | None]] = {}

    if method == "circular_time_shift":
        used_shift_fraction = float(rng.random() if shift_fraction is None else shift_fraction)
        used_event_seed = None
    elif method == "fourier_phase_randomization":
        used_shift_fraction = None
        used_event_seed = int(rng.integers(0, 2**32 - 1)) if event_seed is None else int(event_seed)
    else:
        raise ValueError(f"Unknown null method: {method}")

    component_index = 0
    for detector in preparation.detectors_used:
        components = preparation.detector_components.get(detector, [])
        det_kappas: list[float] = []
        det_sigmas: list[float] = []
        det_shift_samples: list[int | None] = []

        for component in components:
            if method == "circular_time_shift":
                shift_samples = _shift_samples_from_fraction(
                    component.n_samples,
                    component.min_shift_samples,
                    used_shift_fraction,
                )
                transformed = np.roll(component.residual, shift_samples)
            else:
                shift_samples = None
                component_rng = np.random.default_rng(used_event_seed + component_index)
                transformed = _phase_randomize_residual(component.residual, component_rng)

            transformed_fft = np.fft.rfft(transformed)
            residual_overlap = _fd_inner(
                transformed_fft,
                component.template_fft,
                component.n_samples,
                component.noise_var,
            )
            kappa_hat = residual_overlap / component.template_inner
            kappa_sigma = 1.0 / component.template_norm

            det_kappas.append(float(kappa_hat))
            det_sigmas.append(float(kappa_sigma))
            det_shift_samples.append(shift_samples)
            component_index += 1

        det_kappa, det_sigma = _combine_kappas(det_kappas, det_sigmas)
        detector_kappas.append(det_kappa)
        detector_sigmas.append(det_sigma)
        component_shift_samples[detector] = det_shift_samples

    event_kappa, event_sigma = _combine_kappas(detector_kappas, detector_sigmas)
    event_snr = event_kappa / event_sigma if event_sigma > 0 else 0.0

    result = PhaseLockResult(
        event_name=preparation.event_name,
        kappa_hat=float(event_kappa),
        kappa_sigma=float(event_sigma),
        snr=float(event_snr),
        a_220_fit=0.0,
        phi_220_fit=0.0,
        template_norm=float(1.0 / event_sigma) if event_sigma > 0 else 0.0,
        residual_overlap=float(event_kappa / event_sigma**2) if event_sigma > 0 else 0.0,
        noise_rms=float(preparation.noise_rms_reference),
    )

    return result, {
        "event": preparation.event_name,
        "shift_fraction": used_shift_fraction,
        "event_seed": used_event_seed,
        "component_shift_samples": component_shift_samples,
    }


def analyze_null_distribution(
    null_kappas: np.ndarray | list[float],
    null_sigmas: np.ndarray | list[float],
    observed_kappa: float,
    observed_sigma: float,
    sigma_ratios: np.ndarray | list[float],
) -> dict[str, Any]:
    """Summarize the empirical null and compare it to the asymptotic estimate."""
    null_kappas_array = np.asarray(null_kappas, dtype=float)
    null_sigmas_array = np.asarray(null_sigmas, dtype=float)
    sigma_ratios_array = np.asarray(sigma_ratios, dtype=float)

    null_mean = float(np.mean(null_kappas_array))
    null_std = float(np.std(null_kappas_array, ddof=0))
    null_median = float(np.median(null_kappas_array))

    if observed_kappa >= null_mean:
        tail_mask = null_kappas_array >= observed_kappa
    else:
        tail_mask = null_kappas_array <= observed_kappa

    empirical_p_value = float(np.mean(tail_mask))
    empirical_sigma = (
        float((observed_kappa - null_mean) / null_std) if null_std > 0 else float("inf")
    )
    asymptotic_sigma = (
        float(observed_kappa / observed_sigma) if observed_sigma > 0 else float("inf")
    )

    empirical_scale = abs(empirical_sigma)
    if empirical_scale > 0 and np.isfinite(empirical_scale):
        calibration_ratio = float(abs(asymptotic_sigma) / empirical_scale)
    else:
        calibration_ratio = float("inf")

    return {
        "null_mean": null_mean,
        "null_std": null_std,
        "null_median": null_median,
        "observed_kappa": float(observed_kappa),
        "observed_sigma": float(observed_sigma),
        "empirical_p_value": empirical_p_value,
        "empirical_sigma": empirical_sigma,
        "asymptotic_sigma": asymptotic_sigma,
        "calibration_ratio": calibration_ratio,
        "is_well_calibrated": bool(
            np.isfinite(calibration_ratio) and abs(calibration_ratio - 1.0) < 0.2
        ),
        "per_event_null_check": {
            "mean_sigma_ratio": float(np.mean(sigma_ratios_array)),
            "max_sigma_ratio": float(np.max(sigma_ratios_array)),
            "sigma_ratios_consistent": bool(np.all(np.abs(sigma_ratios_array - 1.0) <= 0.05)),
        },
        "null_sigma_mean": float(np.mean(null_sigmas_array)),
    }


def recommendation_for_claim_language(summary: dict[str, Any]) -> str:
    """Translate the calibration result into one line of claim guidance."""
    p_value = float(summary["empirical_p_value"])
    ratio = float(summary["calibration_ratio"])

    if p_value > 0.05:
        return "Use upper-limit language: the empirical null is weaker than the asymptotic 2.2-sigma estimate."
    if ratio > 1.2:
        return "Use provisional or upper-limit language: the asymptotic estimate appears optimistic under the empirical null."
    if ratio < 0.8:
        return "Treat the null generation as conservative and verify it before strengthening the external claim."
    if p_value < 0.03:
        return "Hint-level language is supported: the empirical null is consistent with a calibrated low-significance excess."
    return "Keep tentative hint language only: the null is suggestive but not yet strong enough for a firmer claim."


def run_null_campaign(
    preparations: list[EventNullPreparation],
    observed_kappa: float,
    observed_sigma: float,
    n_null: int = 1000,
    seed: int = 42,
    method: str = "circular_time_shift",
    max_weight_ratio: float | None = 5.5,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the full empirical null campaign across all prepared events."""
    if n_null <= 0:
        raise ValueError("n_null must be positive")
    if not preparations:
        raise ValueError("No prepared events supplied")

    rng = np.random.default_rng(seed)
    event_names = [prep.event_name for prep in preparations]

    null_kappas: list[float] = []
    null_sigmas: list[float] = []
    null_log_bayes_factors: list[float] = []
    realization_shift_fractions: list[list[float | None]] = []
    realization_event_seeds: list[list[int | None]] = []

    sigma_ratios: list[float] | None = None

    for realization in range(n_null):
        if progress and (
            (realization + 1) == 1 or (realization + 1) % 50 == 0 or (realization + 1) == n_null
        ):
            print(f"  null realization {realization + 1}/{n_null}", flush=True)

        per_event_results: list[PhaseLockResult] = []
        shift_fractions_for_realization: list[float | None] = []
        seeds_for_realization: list[int | None] = []
        per_event_log_bf: list[float] = []

        for preparation in preparations:
            result, metadata = generate_event_null_realization(preparation, rng, method=method)
            per_event_results.append(result)
            shift_fractions_for_realization.append(metadata["shift_fraction"])
            seeds_for_realization.append(metadata["event_seed"])
            per_event_log_bf.append(
                _gaussian_log_bayes_factor(result.kappa_hat, result.kappa_sigma)
            )

        if sigma_ratios is None:
            sigma_ratios = [
                result.kappa_sigma / prep.real_kappa_sigma
                for result, prep in zip(per_event_results, preparations)
            ]

        stacked = stack_phase_locked(per_event_results, max_weight_ratio=max_weight_ratio)
        null_kappas.append(float(stacked.kappa_hat))
        null_sigmas.append(float(stacked.kappa_sigma))
        null_log_bayes_factors.append(float(np.sum(per_event_log_bf)))
        realization_shift_fractions.append(shift_fractions_for_realization)
        realization_event_seeds.append(seeds_for_realization)

    observed_log_bayes_factor = float(
        np.sum(
            [
                _gaussian_log_bayes_factor(prep.real_kappa_hat, prep.real_kappa_sigma)
                for prep in preparations
            ]
        )
    )

    summary = analyze_null_distribution(
        null_kappas=np.asarray(null_kappas, dtype=float),
        null_sigmas=np.asarray(null_sigmas, dtype=float),
        observed_kappa=observed_kappa,
        observed_sigma=observed_sigma,
        sigma_ratios=np.asarray(sigma_ratios if sigma_ratios is not None else [], dtype=float),
    )

    result = {
        "n_null": int(n_null),
        "seed": int(seed),
        "method": method,
        "event_names": event_names,
        "null_kappas": [float(value) for value in null_kappas],
        "null_sigmas": [float(value) for value in null_sigmas],
        **summary,
        "realization_shift_fractions": realization_shift_fractions,
        "realization_event_seeds": realization_event_seeds,
        "null_stacked_log_bayes_factors": [float(value) for value in null_log_bayes_factors],
        "observed_stacked_log_bayes_factor": observed_log_bayes_factor,
        "null_log_bayes_factor_mean": float(np.mean(null_log_bayes_factors)),
        "recommendation": recommendation_for_claim_language(summary),
    }

    return result


def run_phase3_null_campaign(
    phase3_results_path: str | Path = DEFAULT_PHASE3_RESULTS_PATH,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    data_dir: str | Path = "data",
    n_null: int = 1000,
    seed: int = 42,
    method: str = "circular_time_shift",
    max_weight_ratio: float | None = 5.5,
    progress: bool = False,
) -> dict[str, Any]:
    """Prepare Phase 3 inputs and run the full null calibration campaign."""
    preparations, observed = prepare_phase3_null_inputs(
        phase3_results_path=phase3_results_path,
        catalog_path=catalog_path,
        data_dir=data_dir,
        progress=progress,
    )

    result = run_null_campaign(
        preparations=preparations,
        observed_kappa=float(observed["kappa_hat"]),
        observed_sigma=float(observed["kappa_sigma"]),
        n_null=n_null,
        seed=seed,
        method=method,
        max_weight_ratio=max_weight_ratio,
        progress=progress,
    )
    result["phase3_results_path"] = str(Path(phase3_results_path))
    result["catalog_path"] = str(Path(catalog_path))
    result["data_dir"] = str(Path(data_dir))
    result["max_weight_ratio"] = max_weight_ratio
    return result


def save_null_campaign(
    result: dict[str, Any], output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Write a null-campaign result artifact to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
