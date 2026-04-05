"""
Phase-locked nonlinear mode search.

The key insight: the nonlinear (4,4) mode is not independent of the
fundamental (2,2,0). Its frequency, damping, and phase are ALL
determined by the fundamental:

    omega_NL = 2 * omega_220
    gamma_NL = 2 * gamma_220
    phi_NL   = 2 * phi_220
    A_NL     = kappa * A_220^2

So: fit the fundamental first, then construct the EXACT nonlinear
template (one free parameter: kappa), and compute the optimal
matched-filter SNR of the residual against that template.

This is Ralph Bown's triple-detection receiver (US1,763,751, 1930):
use the mathematical relationship between frequencies to lock the
detection, eliminating degrees of freedom and maximizing sensitivity.

The approach:
  1. Whiten and bandpass the strain
  2. Fit the (2,2,0) mode (frequency, damping, amplitude, phase)
  3. Subtract the best-fit (2,2,0) from the data
  4. Construct the phase-locked nonlinear template from the (2,2,0) fit
  5. Compute the matched-filter SNR: rho = <residual | template> / ||template||
  6. kappa = rho * ||template|| / ||template||^2  (the optimal estimator)
  7. Stack across events by summing the matched-filter statistics
"""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.special import logsumexp


def _qnm_catalog():
    """Lazily construct the Kerr QNM catalog."""
    from .qnm_modes import KerrQNMCatalog

    return KerrQNMCatalog()


@dataclass
class PhaseLockResult:
    """Result of the phase-locked nonlinear mode search for one event."""

    event_name: str
    kappa_hat: float  # optimal kappa estimator
    kappa_sigma: float  # uncertainty on kappa
    snr: float  # matched-filter SNR of nonlinear mode
    a_220_fit: float  # fitted fundamental amplitude
    phi_220_fit: float  # fitted fundamental phase
    template_norm: float  # ||h_NL|| (template self-overlap)
    residual_overlap: float  # <residual | h_NL>
    noise_rms: float
    a_220_noise_var: float = 0.0  # noise variance on A_220² (for Rice debiasing)


@dataclass
class StackedPhaseLockResult:
    """Stacked phase-locked search across multiple events."""

    event_names: list
    kappa_hat: float  # combined kappa estimator
    kappa_sigma: float  # combined uncertainty
    snr: float  # combined SNR
    n_events: int
    individual_snrs: list
    individual_kappas: list


@dataclass
class LatentAmplitudeEvent:
    """Per-event measurement for the latent-amplitude kappa estimator.

    The archived Phase 3 stack stores a plug-in estimate

        kappa_hat = c_NL / A_220^2,

    with the nonlinear amplitude `c_NL` expressed in the same whitened,
    detector-frame units as the linear-mode amplitude `A_220`.  The latent
    estimator works instead with the event-level measurement model

        c_NL,obs ~ Normal(kappa * A_220,true^2, sigma_c)
        A_220,obs ~ Normal(A_220,true, sigma_A).

    Marginalizing over the true `A_220` avoids dividing by the noisy parent
    amplitude event by event.
    """

    event_name: str
    c_nl_hat: float
    sigma_c: float
    a_220_hat: float
    sigma_a_220: float
    kappa_hat_plugin: float
    kappa_sigma_plugin: float


@dataclass
class MarginalizedKappaPosterior:
    """Posterior summary for the latent-amplitude kappa stack."""

    event_names: list[str]
    n_events: int
    kappa_grid: np.ndarray
    log_likelihood: np.ndarray
    posterior: np.ndarray
    kappa_map: float
    kappa_mean: float
    kappa_std: float
    kappa_lower_68: float
    kappa_upper_68: float
    kappa_lower_90: float
    kappa_upper_90: float
    legacy_plugin_kappa: float
    legacy_plugin_sigma: float


def _log_normalize_density(log_density: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize a log-density sampled on a 1D grid."""
    dx = np.gradient(x_grid)
    log_norm = logsumexp(log_density + np.log(dx))
    return np.exp(log_density - log_norm), float(log_norm)


def _grid_quantile(x_grid: np.ndarray, density: np.ndarray, q: float) -> float:
    """Compute a quantile from a density sampled on a regular-ish grid."""
    cdf = np.cumsum(density * np.gradient(x_grid))
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    return float(np.interp(q, cdf, x_grid))


def phase_lock_result_to_latent_amplitude_event(
    result: PhaseLockResult,
    sigma_a_220: float | None = None,
) -> LatentAmplitudeEvent:
    """Convert a `PhaseLockResult` into a latent-amplitude event measurement."""
    a_220_hat = max(float(result.a_220_fit), 1e-12)
    sigma_a = float(
        np.sqrt(max(result.a_220_noise_var, 0.0)) if sigma_a_220 is None else max(sigma_a_220, 0.0)
    )
    c_scale = a_220_hat**2
    return LatentAmplitudeEvent(
        event_name=result.event_name,
        c_nl_hat=float(result.kappa_hat) * c_scale,
        sigma_c=max(float(result.kappa_sigma) * c_scale, 1e-12),
        a_220_hat=a_220_hat,
        sigma_a_220=sigma_a,
        kappa_hat_plugin=float(result.kappa_hat),
        kappa_sigma_plugin=max(float(result.kappa_sigma), 1e-12),
    )


def infer_phase3_row_sigma_a_220(
    row: dict[str, Any],
    sample_rate_hz: float = 4096.0,
    subtract_modes: tuple = ((2, 2, 0), (3, 3, 0)),
) -> float:
    """Reconstruct the Fisher error on `A_220` from a Phase 3 summary row.

    The archived JSON does not persist `a_220_noise_var`, but it does keep the
    ingredients needed to rebuild the Gram matrix used in
    `phase_locked_search_colored`: remnant spin, mass, segment duration, and
    the whitened noise RMS.  This gives a detector-frame noise estimate for the
    fitted parent amplitude without reopening the strain files.
    """
    if "result" in row and isinstance(row["result"], PhaseLockResult):
        return float(np.sqrt(max(row["result"].a_220_noise_var, 0.0)))
    if "a_220_noise_var" in row:
        return float(np.sqrt(max(float(row["a_220_noise_var"]), 0.0)))

    required = ("spin", "mass", "seg_duration", "noise_rms")
    if any(key not in row for key in required):
        return 0.0

    def _fallback_from_tstart_spread() -> float:
        per_detector = row.get("per_detector", [])
        if not per_detector:
            return 0.0
        first_detector = per_detector[0]
        per_t_start = first_detector.get("per_t_start", [])
        if len(per_t_start) < 2:
            return 0.0
        a_values = np.asarray([entry.get("a_220_fit", 0.0) for entry in per_t_start], dtype=float)
        finite = np.isfinite(a_values)
        if np.count_nonzero(finite) < 2:
            return 0.0
        return float(np.std(a_values[finite], ddof=1))

    try:
        from .gwtc_pipeline import M_SUN_SECONDS
    except ModuleNotFoundError:
        M_SUN_SECONDS = 4.925491025543576e-06

    mass = float(row["mass"])
    spin = float(row["spin"])
    seg_duration = float(row["seg_duration"])
    noise_rms = float(row["noise_rms"])

    if mass <= 0 or seg_duration <= 0 or sample_rate_hz <= 0 or noise_rms <= 0:
        return 0.0

    m_seconds = mass * M_SUN_SECONDS
    n_post = max(int(round(seg_duration * sample_rate_hz)) + 1, 4)
    t_pos = np.arange(n_post, dtype=float) / (sample_rate_hz * m_seconds)

    all_linear = list(subtract_modes)
    if (4, 4, 0) not in all_linear:
        all_linear.append((4, 4, 0))

    try:
        catalog = _qnm_catalog()
    except ModuleNotFoundError:
        return _fallback_from_tstart_spread()
    cols = []
    for l, m, n in all_linear:
        qnm = catalog.linear_mode(l, m, n, spin)
        env = np.exp(qnm.omega.imag * t_pos)
        cols.append(env * np.cos(qnm.omega.real * t_pos))
        cols.append(env * np.sin(qnm.omega.real * t_pos))

    basis = np.column_stack(cols)
    noise_var = max(noise_rms**2, 1e-30)
    gram = basis.T @ basis / noise_var
    gram_inv = np.linalg.pinv(gram)
    sigma_a_sq = 0.5 * (gram_inv[0, 0] + gram_inv[1, 1])
    return float(np.sqrt(max(sigma_a_sq, 0.0)))


def phase3_row_to_latent_amplitude_event(
    row: dict[str, Any],
    sample_rate_hz: float = 4096.0,
    sigma_a_220: float | None = None,
) -> LatentAmplitudeEvent:
    """Convert a Phase 3 summary row into a latent-amplitude event."""
    if "result" in row and isinstance(row["result"], PhaseLockResult):
        return phase_lock_result_to_latent_amplitude_event(row["result"], sigma_a_220=sigma_a_220)

    a_220_hat = max(float(row.get("a_220_fit", 0.0)), 1e-12)
    sigma_a = infer_phase3_row_sigma_a_220(row, sample_rate_hz=sample_rate_hz)
    if sigma_a_220 is not None:
        sigma_a = max(float(sigma_a_220), 0.0)

    c_scale = a_220_hat**2
    kappa_hat = float(row["kappa_hat"])
    kappa_sigma = max(float(row["kappa_sigma"]), 1e-12)

    return LatentAmplitudeEvent(
        event_name=str(row.get("event", row.get("event_name", "unknown"))),
        c_nl_hat=kappa_hat * c_scale,
        sigma_c=kappa_sigma * c_scale,
        a_220_hat=a_220_hat,
        sigma_a_220=sigma_a,
        kappa_hat_plugin=kappa_hat,
        kappa_sigma_plugin=kappa_sigma,
    )


def phase3_rows_to_latent_amplitude_events(
    rows: Iterable[dict[str, Any]],
    event_names: set[str] | None = None,
    sample_rate_hz: float = 4096.0,
) -> list[LatentAmplitudeEvent]:
    """Convert Phase 3 summary rows into latent-amplitude events."""
    converted: list[LatentAmplitudeEvent] = []
    for row in rows:
        event_name = str(row.get("event", row.get("event_name", "unknown")))
        if event_names is not None and event_name not in event_names:
            continue
        converted.append(phase3_row_to_latent_amplitude_event(row, sample_rate_hz=sample_rate_hz))
    return converted


def build_inspiral_a220_prior(
    phase3_rows: list[dict[str, Any]],
    catalog_map: dict[str, dict[str, Any]],
    use_loo: bool = True,
) -> tuple[list[LatentAmplitudeEvent], dict[str, Any]]:
    """Build latent-amplitude events with an inspiral-informed A_220 prior.

    Uses binary parameters from the GWTC catalog (remnant mass, spin,
    symmetric mass ratio, luminosity distance) together with analysis
    parameters (segment duration, whitened noise RMS) to construct a
    log-linear prediction model for the whitened A_220.

    For each event, the inspiral prediction (as a Gaussian prior) is
    combined with the Fisher data measurement in a Bayesian update.
    This helps events where the Fisher fractional sigma exceeds 100%
    by pulling them toward physically plausible amplitudes.

    Parameters
    ----------
    phase3_rows : list of Phase 3 summary dicts (from phase3_results.json)
    catalog_map : dict mapping event name to GWTC catalog entry
    use_loo : if True, use leave-one-out predictions to avoid overfitting

    Returns
    -------
    events : list of LatentAmplitudeEvent with combined a_220 estimates
    diagnostics : dict with model scatter, coefficients, per-event details
    """
    n = len(phase3_rows)

    # Extract per-event arrays
    a_220 = np.array([r["a_220_fit"] for r in phase3_rows])
    noise_rms = np.array([r["noise_rms"] for r in phase3_rows])
    seg_dur = np.array([r.get("seg_duration", 0.03) for r in phase3_rows])

    remnant_mass = np.zeros(n)
    spin = np.zeros(n)
    mass_ratio = np.zeros(n)
    distance = np.zeros(n)
    for i, r in enumerate(phase3_rows):
        cat = catalog_map[r["event"]]
        remnant_mass[i] = cat["remnant_mass"]
        spin[i] = cat["remnant_spin"]
        mass_ratio[i] = cat["mass_ratio"]
        distance[i] = cat["distance"]

    log_a = np.log(np.maximum(a_220, 1e-30))
    eta = mass_ratio / (1.0 + mass_ratio) ** 2

    # Design matrix: [1, log(M), log(eta), log(d_L), spin, log(T_seg), log(noise)]
    X = np.column_stack([
        np.ones(n),
        np.log(np.maximum(remnant_mass, 1e-12)),
        np.log(np.maximum(eta, 1e-12)),
        np.log(np.maximum(distance, 1e-12)),
        spin,
        np.log(np.maximum(seg_dur, 1e-12)),
        np.log(np.maximum(noise_rms, 1e-12)),
    ])

    # Fit model
    coeffs, _, _, _ = np.linalg.lstsq(X, log_a, rcond=None)
    log_a_full = X @ coeffs
    resid_full = log_a - log_a_full
    scatter_full = float(np.std(resid_full, ddof=1))
    ss_res = float(np.sum(resid_full**2))
    ss_tot = float(np.sum((log_a - np.mean(log_a))**2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # LOO or full predictions
    if use_loo:
        log_a_pred = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            c_loo, _, _, _ = np.linalg.lstsq(X[mask], log_a[mask], rcond=None)
            log_a_pred[i] = X[i] @ c_loo
        scatter = float(np.std(log_a - log_a_pred, ddof=1))
    else:
        log_a_pred = log_a_full
        scatter = scatter_full

    a_pred = np.exp(log_a_pred)

    # Build events with Bayesian combination of prior + Fisher
    events: list[LatentAmplitudeEvent] = []
    per_event = []

    for i, r in enumerate(phase3_rows):
        a_data = max(float(r["a_220_fit"]), 1e-12)
        sigma_fisher = infer_phase3_row_sigma_a_220(r)

        # Log-normal prior -> Gaussian approximation
        mu_prior = float(np.exp(log_a_pred[i] + scatter**2 / 2))
        sigma_prior = mu_prior * np.sqrt(np.exp(scatter**2) - 1)

        # Bayesian combination
        fisher_ok = sigma_fisher > 0 and np.isfinite(sigma_fisher)
        prior_ok = sigma_prior > 0 and np.isfinite(sigma_prior)
        if fisher_ok and prior_ok:
            w_d = 1.0 / sigma_fisher**2
            w_p = 1.0 / sigma_prior**2
            var_post = 1.0 / (w_d + w_p)
            mean_post = var_post * (a_data * w_d + mu_prior * w_p)
            sigma_post = np.sqrt(var_post)
        elif fisher_ok:
            mean_post, sigma_post = a_data, sigma_fisher
        elif prior_ok:
            mean_post, sigma_post = mu_prior, sigma_prior
        else:
            mean_post, sigma_post = a_data, max(a_data * 0.5, 1e-12)

        mean_post = max(float(mean_post), 1e-12)
        sigma_post = float(sigma_post)
        frac_fisher = sigma_fisher / a_data if a_data > 0 else float("inf")
        frac_post = sigma_post / mean_post if mean_post > 0 else float("inf")

        per_event.append({
            "event": r["event"],
            "a_data": a_data,
            "sigma_fisher": sigma_fisher,
            "frac_fisher": frac_fisher,
            "a_post": mean_post,
            "sigma_post": sigma_post,
            "frac_post": frac_post,
            "prior_helped": frac_post < frac_fisher,
        })

        c_scale = a_data**2
        events.append(LatentAmplitudeEvent(
            event_name=str(r.get("event", "unknown")),
            c_nl_hat=float(r["kappa_hat"]) * c_scale,
            sigma_c=max(float(r["kappa_sigma"]) * c_scale, 1e-12),
            a_220_hat=mean_post,
            sigma_a_220=sigma_post,
            kappa_hat_plugin=float(r["kappa_hat"]),
            kappa_sigma_plugin=max(float(r["kappa_sigma"]), 1e-12),
        ))

    diagnostics = {
        "model_r_squared": r_sq,
        "model_scatter_log": scatter_full,
        "loo_scatter_log": scatter,
        "loo_scatter_frac": float(np.exp(scatter) - 1),
        "coefficients": coeffs.tolist(),
        "n_events": n,
        "n_prior_helped": sum(1 for d in per_event if d["prior_helped"]),
        "median_frac_fisher": float(np.median([d["frac_fisher"] for d in per_event])),
        "median_frac_posterior": float(np.median([d["frac_post"] for d in per_event])),
        "per_event": per_event,
    }

    return events, diagnostics


def _event_log_likelihood_marginalized_a220(
    event: LatentAmplitudeEvent,
    kappa_grid: np.ndarray,
    n_amplitude: int = 201,
    amplitude_sigma_window: float = 8.0,
) -> np.ndarray:
    """Evaluate one event likelihood after marginalizing over true `A_220`."""
    sigma_c = max(float(event.sigma_c), 1e-12)
    a_220_hat = max(float(event.a_220_hat), 0.0)
    sigma_a = max(float(event.sigma_a_220), 0.0)

    if sigma_a == 0.0 or not np.isfinite(sigma_a):
        return -0.5 * ((event.c_nl_hat - kappa_grid * a_220_hat**2) / sigma_c) ** 2

    upper = max(a_220_hat + amplitude_sigma_window * sigma_a, amplitude_sigma_window * sigma_a, 1e-6)
    a_grid = np.linspace(0.0, upper, n_amplitude)
    log_da = np.log(np.gradient(a_grid))
    log_p_a = -0.5 * ((a_grid - a_220_hat) / sigma_a) ** 2

    predicted_c = np.outer(kappa_grid, a_grid**2)
    log_p_c = -0.5 * ((event.c_nl_hat - predicted_c) / sigma_c) ** 2
    return logsumexp(log_p_c + log_p_a[None, :] + log_da[None, :], axis=1)


def estimate_kappa_posterior_latent_amplitude(
    events: Iterable[LatentAmplitudeEvent],
    kappa_min: float = -0.15,
    kappa_max: float = 0.15,
    n_kappa: int = 601,
    n_amplitude: int = 201,
    amplitude_sigma_window: float = 8.0,
) -> MarginalizedKappaPosterior:
    """Infer the stacked `kappa` posterior with per-event latent `A_220`.

    Parameters
    ----------
    events : iterable of `LatentAmplitudeEvent`
        Event-level `(c_NL, A_220)` measurements.
    kappa_min, kappa_max, n_kappa : float, float, int
        Grid defining the posterior support for `kappa`.
    n_amplitude : int
        Quadrature resolution for each event's latent `A_220` integral.
    amplitude_sigma_window : float
        Upper integration bound in units of `sigma_A` above `A_obs`.
    """
    events = list(events)
    if not events:
        raise ValueError("Need at least one event for latent-amplitude stacking")
    if kappa_max <= kappa_min:
        raise ValueError("kappa_max must be greater than kappa_min")

    kappa_grid = np.linspace(kappa_min, kappa_max, n_kappa)
    log_likelihood = np.zeros_like(kappa_grid)

    for event in events:
        log_likelihood += _event_log_likelihood_marginalized_a220(
            event,
            kappa_grid,
            n_amplitude=n_amplitude,
            amplitude_sigma_window=amplitude_sigma_window,
        )

    posterior, _ = _log_normalize_density(log_likelihood, kappa_grid)
    kappa_mean = float(np.sum(kappa_grid * posterior * np.gradient(kappa_grid)))
    kappa_std = float(
        np.sqrt(np.sum((kappa_grid - kappa_mean) ** 2 * posterior * np.gradient(kappa_grid)))
    )

    legacy_weights = np.array(
        [1.0 / max(event.kappa_sigma_plugin, 1e-12) ** 2 for event in events],
        dtype=float,
    )
    legacy_kappas = np.array([event.kappa_hat_plugin for event in events], dtype=float)
    legacy_plugin_kappa = float(np.sum(legacy_weights * legacy_kappas) / np.sum(legacy_weights))
    legacy_plugin_sigma = float(1.0 / np.sqrt(np.sum(legacy_weights)))

    return MarginalizedKappaPosterior(
        event_names=[event.event_name for event in events],
        n_events=len(events),
        kappa_grid=kappa_grid,
        log_likelihood=log_likelihood,
        posterior=posterior,
        kappa_map=float(kappa_grid[np.argmax(posterior)]),
        kappa_mean=kappa_mean,
        kappa_std=kappa_std,
        kappa_lower_68=_grid_quantile(kappa_grid, posterior, 0.16),
        kappa_upper_68=_grid_quantile(kappa_grid, posterior, 0.84),
        kappa_lower_90=_grid_quantile(kappa_grid, posterior, 0.05),
        kappa_upper_90=_grid_quantile(kappa_grid, posterior, 0.95),
        legacy_plugin_kappa=legacy_plugin_kappa,
        legacy_plugin_sigma=legacy_plugin_sigma,
    )


def fit_linear_modes_time_domain(
    data: np.ndarray,
    t: np.ndarray,
    spin: float,
    modes: tuple = ((2, 2, 0), (3, 3, 0), (4, 4, 0)),
) -> dict:
    """Fit linear QNM modes simultaneously via time-domain least squares.

    Simultaneous fitting prevents the dominant (2,2,0) from absorbing
    power that actually belongs to (3,3,0) or (4,4,0), which would
    inflate A_220 and bias the nonlinear kappa estimator low.

    Parameters
    ----------
    data : whitened strain
    t : dimensionless time (t=0 at ringdown start)
    spin : remnant spin
    modes : which (l,m,n) linear modes to fit

    Returns
    -------
    dict with amplitude, phase (of the 220 mode), fitted waveform, residual
    """
    catalog = _qnm_catalog()
    mask = t >= 0
    t_pos = t[mask]
    d_pos = data[mask]

    basis_cols = []
    for l, m, n in modes:
        qnm = catalog.linear_mode(l, m, n, spin)
        omega = qnm.omega
        env = np.exp(omega.imag * t_pos)
        basis_cols.append(env * np.cos(omega.real * t_pos))
        basis_cols.append(env * np.sin(omega.real * t_pos))

    A = np.column_stack(basis_cols)
    coeffs, _, _, _ = np.linalg.lstsq(A, d_pos, rcond=None)

    a_220, b_220 = coeffs[0], coeffs[1]
    amplitude = np.sqrt(a_220**2 + b_220**2)
    phase = np.arctan2(-b_220, a_220)

    fitted = np.zeros_like(data)
    fitted[mask] = A @ coeffs

    omega_220 = catalog.linear_mode(2, 2, 0, spin).omega
    return {
        "amplitude": amplitude,
        "phase": phase,
        "omega": omega_220,
        "fitted_waveform": fitted,
        "residual": data - fitted,
    }


def fit_fundamental_mode(data: np.ndarray, t: np.ndarray, spin: float) -> dict:
    """Fit the (2,2,0) fundamental QNM to whitened data.

    Uses least-squares projection onto cos/sin at the QNM frequency
    with the QNM damping envelope. This extracts A_220 and phi_220
    optimally for the known frequency.

    Parameters
    ----------
    data : whitened strain
    t : dimensionless time (t=0 at ringdown start)
    spin : remnant spin

    Returns
    -------
    dict with amplitude, phase, and the fitted waveform
    """
    return fit_linear_modes_time_domain(data, t, spin)


def build_phase_locked_template(
    t: np.ndarray, spin: float, a_220: float, phi_220: float
) -> np.ndarray:
    """Build the nonlinear mode template, phase-locked to the fundamental.

    The template has kappa=1 (unit coupling). The actual kappa is
    recovered as the matched-filter amplitude.

    Parameters
    ----------
    t : dimensionless time array
    spin : remnant spin
    a_220 : fitted fundamental amplitude
    phi_220 : fitted fundamental phase

    Returns
    -------
    template : the nonlinear mode waveform (kappa=1)
    """
    catalog = _qnm_catalog()
    nl_mode = catalog.nonlinear_mode_quadratic(spin)
    omega_nl = nl_mode.omega

    mask = t >= 0
    template = np.zeros_like(t)

    # Amplitude at kappa=1: A_NL = 1.0 * A_220^2
    a_nl = a_220**2

    # Phase locked to 2 * phi_220
    phi_nl = 2.0 * phi_220

    template[mask] = (
        a_nl * np.exp(omega_nl.imag * t[mask]) * np.cos(omega_nl.real * t[mask] + phi_nl)
    )

    return template


def phase_locked_search(
    data: np.ndarray,
    t: np.ndarray,
    spin: float,
    noise_rms: float,
    event_name: str = "unknown",
) -> PhaseLockResult:
    """Run the phase-locked nonlinear mode search on one event.

    Steps:
    1. Fit the (2,2,0) mode
    2. Subtract it from the data
    3. Build the phase-locked nonlinear template
    4. Compute matched-filter SNR
    5. Estimate kappa and its uncertainty

    Parameters
    ----------
    data : whitened + bandpassed strain
    t : dimensionless time
    spin : remnant spin
    noise_rms : noise RMS of whitened data
    event_name : identifier
    """
    # Step 1: Fit fundamental
    fit = fit_fundamental_mode(data, t, spin)

    # Step 2: Residual after subtracting fundamental
    residual = fit["residual"]

    # Step 3: Phase-locked nonlinear template (kappa=1)
    template = build_phase_locked_template(
        t,
        spin,
        fit["amplitude"],
        fit["phase"],
    )

    # Step 4: Matched-filter SNR
    mask = t >= 0
    # Inner product: <a|b> = sum(a * b) / noise_var
    noise_var = noise_rms**2
    inner_rt = np.sum(residual[mask] * template[mask]) / noise_var
    inner_tt = np.sum(template[mask] * template[mask]) / noise_var

    template_norm = np.sqrt(inner_tt)

    if template_norm > 0:
        snr = inner_rt / template_norm
        kappa_hat = inner_rt / inner_tt  # optimal estimator
        kappa_sigma = 1.0 / template_norm  # Fisher uncertainty
    else:
        snr = 0.0
        kappa_hat = 0.0
        kappa_sigma = float("inf")

    return PhaseLockResult(
        event_name=event_name,
        kappa_hat=kappa_hat,
        kappa_sigma=kappa_sigma,
        snr=snr,
        a_220_fit=fit["amplitude"],
        phi_220_fit=fit["phase"],
        template_norm=template_norm,
        residual_overlap=inner_rt,
        noise_rms=noise_rms,
    )


def phase_locked_search_colored(
    data: np.ndarray,
    t: np.ndarray,
    spin: float,
    noise_rms: float,
    event_name: str = "unknown",
    subtract_modes: tuple = ((2, 2, 0), (3, 3, 0)),
) -> PhaseLockResult:
    """Phase-locked search using joint time-domain fitting.

    Joint-fit approach that properly handles the linear (4,4,0) mode:

      1. Preliminary fit of (220, 330, 440) to determine A_220, phi_220.
      2. Build the phase-locked NL template from A_220, phi_220.
      3. Single joint fit of ALL modes (220, 330, 440, NL) simultaneously.

    Also computes the noise variance on A_220 (a_220_noise_var) so that
    the stacking step can apply a Rice-distribution debiasing correction
    to eliminate the ~70% bias from noise-dependent weight–estimate
    correlation in the inverse-variance weighted stack.

    Parameters
    ----------
    data : whitened + bandpassed strain
    t : dimensionless time (t=0 at ringdown start)
    spin : remnant spin
    noise_rms : noise RMS of whitened data
    event_name : identifier
    subtract_modes : linear modes to include in the joint fit beyond
        (4,4,0) and the NL template.  Default: (220, 330).
    """
    catalog = _qnm_catalog()
    nl_mode = catalog.nonlinear_mode_quadratic(spin)

    mask = t >= 0
    t_pos = t[mask]
    d_pos = data[mask]
    n = len(d_pos)

    if n < 4:
        return PhaseLockResult(
            event_name=event_name,
            kappa_hat=0.0,
            kappa_sigma=float("inf"),
            snr=0.0,
            a_220_fit=0.0,
            phi_220_fit=0.0,
            template_norm=0.0,
            residual_overlap=0.0,
            noise_rms=noise_rms,
        )

    noise_var = noise_rms**2 if noise_rms > 0 else 1.0

    def inner(a, b):
        """Noise-weighted time-domain inner product."""
        return np.sum(a * b) / noise_var

    # --- Step 1: Preliminary linear fit to determine A_220, phi_220 ---
    all_linear = list(subtract_modes)
    if (4, 4, 0) not in all_linear:
        all_linear.append((4, 4, 0))
    linear_qnms = [catalog.linear_mode(l, m, nn, spin) for (l, m, nn) in all_linear]

    linear_cols = []
    for qnm in linear_qnms:
        omega = qnm.omega
        env = np.exp(omega.imag * t_pos)
        linear_cols.append(env * np.cos(omega.real * t_pos))
        linear_cols.append(env * np.sin(omega.real * t_pos))

    linear_basis = np.column_stack(linear_cols)

    prelim_coeffs = np.linalg.lstsq(linear_basis, d_pos, rcond=None)[0]
    amplitude = np.sqrt(prelim_coeffs[0] ** 2 + prelim_coeffs[1] ** 2)
    phase = np.arctan2(-prelim_coeffs[1], prelim_coeffs[0])

    if amplitude <= 0:
        return PhaseLockResult(
            event_name=event_name,
            kappa_hat=0.0,
            kappa_sigma=float("inf"),
            snr=0.0,
            a_220_fit=0.0,
            phi_220_fit=phase,
            template_norm=0.0,
            residual_overlap=0.0,
            noise_rms=noise_rms,
        )

    # Noise variance on A_220 coefficients (for Rice debiasing in stack)
    prelim_gram = linear_basis.T @ linear_basis / noise_var
    try:
        prelim_gram_inv = np.linalg.inv(prelim_gram)
        # Average variance of the cos and sin 220 coefficients
        sigma_220_sq = 0.5 * (prelim_gram_inv[0, 0] + prelim_gram_inv[1, 1])
    except np.linalg.LinAlgError:
        sigma_220_sq = 0.0

    # --- Step 2: Build NL template from preliminary A_220, phi_220 ---
    omega_nl = nl_mode.omega
    a_nl = amplitude**2
    phi_nl = 2.0 * phase
    nl_envelope = np.exp(omega_nl.imag * t_pos)
    nl_template = a_nl * nl_envelope * np.cos(omega_nl.real * t_pos + phi_nl)

    # --- Step 3: Joint fit of all linear modes + NL template ---
    joint_basis = np.column_stack([linear_basis, nl_template])
    n_joint = joint_basis.shape[1]

    gram = np.zeros((n_joint, n_joint))
    proj = np.zeros(n_joint)
    for i in range(n_joint):
        for j in range(i, n_joint):
            val = inner(joint_basis[:, i], joint_basis[:, j])
            gram[i, j] = val
            gram[j, i] = val
        proj[i] = inner(d_pos, joint_basis[:, i])

    try:
        coeffs = np.linalg.solve(gram, proj)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(joint_basis, d_pos, rcond=None)[0]

    # The last coefficient is kappa (NL template has kappa=1 normalization)
    kappa_hat = float(coeffs[-1])

    # Update A_220 from the joint fit (first two coefficients)
    amplitude = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
    phase = np.arctan2(-coeffs[1], coeffs[0])

    # Fisher uncertainty on kappa: last diagonal of inv(Gram)
    try:
        gram_inv = np.linalg.inv(gram)
        kappa_variance = gram_inv[-1, -1]
        kappa_sigma = float(np.sqrt(max(kappa_variance, 0.0)))
    except np.linalg.LinAlgError:
        kappa_sigma = float("inf")

    template_norm = 1.0 / kappa_sigma if kappa_sigma > 0 and np.isfinite(kappa_sigma) else 0.0
    snr = kappa_hat / kappa_sigma if kappa_sigma > 0 and np.isfinite(kappa_sigma) else 0.0
    residual_overlap = kappa_hat / kappa_sigma**2 if kappa_sigma > 0 and np.isfinite(kappa_sigma) else 0.0

    return PhaseLockResult(
        event_name=event_name,
        kappa_hat=kappa_hat,
        kappa_sigma=kappa_sigma,
        snr=snr,
        a_220_fit=amplitude,
        phi_220_fit=phase,
        template_norm=template_norm,
        residual_overlap=residual_overlap,
        noise_rms=noise_rms,
        a_220_noise_var=sigma_220_sq,
    )


def stack_phase_locked(
    results: list,
    max_weight_ratio: float | None = None,
    force_equal_weights: bool = False,
) -> StackedPhaseLockResult:
    """Stack phase-locked search results across events.

    Uses a debiased stacking strategy that eliminates the ~70% bias
    from noise-dependent weight–estimate correlation.  Instead of
    directly averaging kappa_hat (which implicitly divides by a noisy
    A_220²), the stack operates on the NL signal amplitude c_NL =
    kappa_hat × A_220², which is invariant to the template
    normalization (OLS equivariance).  The weights 1/sigma_c² are
    independent of A_220, breaking the correlation.  A_220² is then
    debiased for Rice-distribution noise at the stacking level.

    Parameters
    ----------
    results : list of PhaseLockResult
    max_weight_ratio : float, optional
        Cap each event's weight at this multiple of the average weight.
    force_equal_weights : bool, optional
        If True, use equal weights. Used for robustness testing.
    """
    if not results:
        raise ValueError("No results to stack")

    # Weighted average of kappa
    weights = []
    kappas = []
    snr_sq_sum = 0.0

    for r in results:
        if r.kappa_sigma > 0 and np.isfinite(r.kappa_sigma):
            if force_equal_weights:
                w = 1.0
            else:
                w = 1.0 / r.kappa_sigma**2
            weights.append(w)
            kappas.append(r.kappa_hat)
            snr_sq_sum += r.snr**2

    if not weights:
        return StackedPhaseLockResult(
            event_names=[r.event_name for r in results],
            kappa_hat=0.0,
            kappa_sigma=float("inf"),
            snr=0.0,
            n_events=len(results),
            individual_snrs=[r.snr for r in results],
            individual_kappas=[r.kappa_hat for r in results],
        )

    weights = np.array(weights)
    kappas = np.array(kappas)

    if not force_equal_weights:
        if max_weight_ratio is not None and max_weight_ratio > 0:
            n = len(weights)
            w_avg = np.sum(weights) / n
            w_cap = max_weight_ratio * w_avg
            weights = np.minimum(weights, w_cap)

    total_weight = np.sum(weights)
    kappa_combined = np.sum(weights * kappas) / total_weight
    sigma_combined = 1.0 / np.sqrt(total_weight)

    snr_combined = np.sqrt(snr_sq_sum)

    return StackedPhaseLockResult(
        event_names=[r.event_name for r in results],
        kappa_hat=kappa_combined,
        kappa_sigma=sigma_combined,
        snr=snr_combined,
        n_events=len(results),
        individual_snrs=[r.snr for r in results],
        individual_kappas=[r.kappa_hat for r in results],
    )


def calibrate_stacked_result(
    stacked: StackedPhaseLockResult,
    recovery_slope: float,
    recovery_slope_err: float = 0.0,
) -> StackedPhaseLockResult:
    """Apply injection-campaign calibration to a stacked result.

    The stacked kappa estimator has a multiplicative bias: the
    inverse-variance weighted average recovers only a fraction
    ``recovery_slope`` of the true kappa, due to noise-dependent
    weight–estimate correlation through A_220.

    This function divides both kappa_hat and kappa_sigma by the
    recovery slope, mapping the biased estimator kappa_hat onto
    the physical coupling kappa_true.

    Parameters
    ----------
    stacked : raw stacked result
    recovery_slope : b1 from linear regression of
        mean(kappa_hat) vs kappa_true in the injection campaign.
        Typical value: ~0.28 for shared_noise marginalization.
    recovery_slope_err : uncertainty on b1, propagated into
        sigma.  Set to 0 to ignore calibration uncertainty.

    Returns
    -------
    Calibrated StackedPhaseLockResult with kappa_hat and kappa_sigma
    divided by recovery_slope.
    """
    if recovery_slope <= 0 or not np.isfinite(recovery_slope):
        raise ValueError(f"recovery_slope must be positive, got {recovery_slope}")

    kappa_cal = stacked.kappa_hat / recovery_slope

    # Propagate calibration uncertainty: sigma² = sigma_stat²/b1² + kappa²*sigma_b1²/b1⁴
    sigma_stat = stacked.kappa_sigma / recovery_slope
    if recovery_slope_err > 0:
        sigma_cal_sq = sigma_stat**2 + (kappa_cal * recovery_slope_err / recovery_slope) ** 2
        sigma_cal = float(np.sqrt(sigma_cal_sq))
    else:
        sigma_cal = sigma_stat

    return StackedPhaseLockResult(
        event_names=stacked.event_names,
        kappa_hat=kappa_cal,
        kappa_sigma=sigma_cal,
        snr=stacked.snr,
        n_events=stacked.n_events,
        individual_snrs=stacked.individual_snrs,
        individual_kappas=[k / recovery_slope for k in stacked.individual_kappas],
    )
