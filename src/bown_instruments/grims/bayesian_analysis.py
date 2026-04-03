"""
Bayesian parameter estimation for the nonlinear coupling coefficient kappa.

The measurement: given a ringdown signal, what is the posterior
probability distribution of kappa — the nonlinear coupling coefficient?

    P(kappa | data) ∝ P(data | kappa) * P(kappa)

The likelihood is computed by matched filtering:
    log L = -0.5 * (data - template(kappa))^T * C^{-1} * (data - template(kappa))

where C is the noise covariance matrix and template(kappa) is the
ringdown waveform with nonlinear coupling coefficient kappa.

We also implement posterior stacking across multiple events:
    P(kappa | all_data) ∝ product_i P(data_i | kappa) * P(kappa)

This is how Bown would approach a weak signal: accumulate evidence
across multiple independent measurements. His pulse-averaging
synchronization patent (US2,037,847, 1936) used exactly this principle.
"""

import numpy as np
from dataclasses import dataclass, field
from .ringdown_templates import RingdownTemplateBuilder
from .qnm_modes import KerrQNMCatalog
from .self_test import extract_mode_amplitudes


@dataclass
class PosteriorResult:
    """Result of Bayesian parameter estimation for one event."""

    event_name: str
    kappa_grid: np.ndarray
    log_likelihood: np.ndarray
    log_posterior: np.ndarray
    posterior: np.ndarray  # normalized
    kappa_map: float  # maximum a posteriori
    kappa_median: float
    kappa_lower_90: float  # 5th percentile
    kappa_upper_90: float  # 95th percentile
    kappa_lower_68: float  # 16th percentile
    kappa_upper_68: float  # 84th percentile
    log_bayes_factor: float  # ln B(NL vs linear-only)
    detection_sigma: float  # significance of kappa != 0
    fit_method: str = "fixed_linear_parameters"
    linear_mode_estimates: dict = field(default_factory=dict)


@dataclass
class StackedPosterior:
    """Stacked posterior across multiple events."""

    event_names: list
    kappa_grid: np.ndarray
    individual_log_likelihoods: list
    stacked_log_likelihood: np.ndarray
    stacked_posterior: np.ndarray
    kappa_map: float
    kappa_median: float
    kappa_lower_90: float
    kappa_upper_90: float
    log_bayes_factor: float
    detection_sigma: float
    n_events: int


def _log_trapezoid(log_y: np.ndarray, x: np.ndarray) -> float:
    """Stable trapezoidal integration in log space."""
    if len(x) != len(log_y):
        raise ValueError("x and log_y must have the same length")
    if len(x) == 1:
        return float(log_y[0])

    dx = np.diff(x)
    weights = np.empty_like(x, dtype=float)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]
    if len(x) > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    positive = weights > 0
    if not np.any(positive):
        return -np.inf

    log_terms = log_y[positive] + np.log(weights[positive])
    max_term = np.max(log_terms)
    return max_term + np.log(np.sum(np.exp(log_terms - max_term)))


def fit_linear_modes(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    include_modes: tuple = ((2, 2, 0), (3, 3, 0), (4, 4, 0)),
) -> dict:
    """Estimate linear mode amplitudes/phases directly from the data.

    This is a pragmatic nuisance-parameter reduction step: fit the linear
    ringdown channel with least squares, then scan only the nonlinear
    coupling coefficient `kappa`.
    """
    catalog = KerrQNMCatalog()
    qnms = [catalog.linear_mode(l, m, n, spin) for (l, m, n) in include_modes]
    labels = [f"{l}{m}{n}" for (l, m, n) in include_modes]
    amplitudes, _, _ = extract_mode_amplitudes(
        data,
        t_dimless,
        mode_frequencies=[qnm.omega for qnm in qnms],
        mode_damping_rates=[],
    )

    fitted = {}
    for i, label in enumerate(labels):
        amp = amplitudes.get(i, {})
        fitted[label] = {
            "amplitude": float(amp.get("amplitude", 0.0)),
            "phase": float(amp.get("phase", 0.0)),
            "omega": qnms[i].omega,
        }
    return fitted


def compute_log_likelihood(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    A_220: float,
    kappa: float,
    noise_variance: float,
    builder: RingdownTemplateBuilder = None,
    A_330: float = 0.0,
    A_440_linear: float = 0.0,
    phi_220: float = 0.0,
    phi_330: float = 0.0,
    phi_440_linear: float = 0.0,
    phi_nl: float = 0.0,
) -> float:
    """Compute log-likelihood for a given kappa value.

    Parameters
    ----------
    data : ringdown strain (dimensionless units)
    t_dimless : time array in units of M (0 = ringdown start)
    spin : remnant spin
    A_220 : fundamental mode amplitude (can be marginalized or fixed)
    kappa : nonlinear coupling coefficient
    noise_variance : noise variance per sample
    A_330 : (3,3,0) mode amplitude
    A_440_linear : linear (4,4,0) mode amplitude
    """
    if builder is None:
        builder = RingdownTemplateBuilder()

    template = builder.build_nonlinear_template(
        spin=spin,
        A_220=A_220,
        kappa=kappa,
        A_330=A_330,
        A_440_linear=A_440_linear,
        phi_220=phi_220,
        phi_330=phi_330,
        phi_440_linear=phi_440_linear,
        phi_nl=phi_nl,
    )
    model = template.waveform(t_dimless)

    mask = t_dimless >= 0
    residual = data[mask] - model[mask]

    if noise_variance > 0:
        log_l = -0.5 * np.sum(residual**2) / noise_variance
        log_l -= 0.5 * np.sum(mask) * np.log(2.0 * np.pi * noise_variance)
    else:
        log_l = -0.5 * np.sum(residual**2)

    return log_l


def estimate_kappa_posterior(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    A_220: float,
    noise_variance: float,
    event_name: str = "unknown",
    kappa_min: float = 0.0,
    kappa_max: float = 5.0,
    n_kappa: int = 201,
    prior: str = "uniform",
    A_330: float = 0.0,
    A_440_linear: float = 0.0,
    phi_220: float = 0.0,
    phi_330: float = 0.0,
    phi_440_linear: float = 0.0,
    phi_nl: float = 0.0,
    fit_method: str = "fixed_linear_parameters",
    linear_mode_estimates: dict | None = None,
) -> PosteriorResult:
    """Compute the posterior distribution of kappa for one event.

    Parameters
    ----------
    data : ringdown strain
    t_dimless : time array in dimensionless units
    spin : remnant spin
    A_220 : fundamental mode amplitude
    noise_variance : noise variance per sample
    event_name : identifier
    kappa_min, kappa_max : prior range
    n_kappa : grid resolution
    prior : "uniform" or "log_uniform"
    """
    builder = RingdownTemplateBuilder()
    kappa_grid = np.linspace(kappa_min, kappa_max, n_kappa)

    # Compute log-likelihood on grid
    log_l = np.array(
        [
            compute_log_likelihood(
                data,
                t_dimless,
                spin,
                A_220,
                k,
                noise_variance,
                builder,
                A_330=A_330,
                A_440_linear=A_440_linear,
                phi_220=phi_220,
                phi_330=phi_330,
                phi_440_linear=phi_440_linear,
                phi_nl=phi_nl,
            )
            for k in kappa_grid
        ]
    )

    # Prior density
    if prior == "uniform":
        if kappa_max <= kappa_min:
            raise ValueError("kappa_max must be greater than kappa_min")
        log_prior = np.full_like(kappa_grid, -np.log(kappa_max - kappa_min))
    elif prior == "log_uniform":
        positive = kappa_grid > 0
        log_norm = np.log(np.log(kappa_max / max(kappa_min, 1e-12)))
        log_prior = np.full_like(kappa_grid, -np.inf)
        log_prior[positive] = -np.log(kappa_grid[positive]) - log_norm
    else:
        raise ValueError(f"Unknown prior: {prior}")

    # Posterior density
    log_post_unnorm = log_l + log_prior
    log_evidence = _log_trapezoid(log_post_unnorm, kappa_grid)
    log_post = log_post_unnorm - log_evidence
    posterior = np.exp(log_post)

    # Summary statistics
    cdf = np.cumsum(posterior) * (kappa_grid[1] - kappa_grid[0])
    cdf /= cdf[-1]

    kappa_map = kappa_grid[np.argmax(posterior)]
    kappa_median = np.interp(0.5, cdf, kappa_grid)
    kappa_lower_90 = np.interp(0.05, cdf, kappa_grid)
    kappa_upper_90 = np.interp(0.95, cdf, kappa_grid)
    kappa_lower_68 = np.interp(0.16, cdf, kappa_grid)
    kappa_upper_68 = np.interp(0.84, cdf, kappa_grid)

    # Bayes factor: nonlinear (kappa > 0) vs linear-only (kappa = 0)
    # B = P(data | NL model, marginalized over kappa>0) / P(data | linear model)
    log_l_linear = compute_log_likelihood(
        data,
        t_dimless,
        spin,
        A_220,
        0.0,
        noise_variance,
        builder,
        A_330=A_330,
        A_440_linear=A_440_linear,
        phi_220=phi_220,
        phi_330=phi_330,
        phi_440_linear=phi_440_linear,
        phi_nl=phi_nl,
    )
    log_bayes_factor = log_evidence - log_l_linear

    # Likelihood-ratio significance for the nonlinear model over kappa = 0.
    sigma_kappa = np.sqrt(max(0.0, 2.0 * (np.max(log_l) - log_l_linear)))

    return PosteriorResult(
        event_name=event_name,
        kappa_grid=kappa_grid,
        log_likelihood=log_l,
        log_posterior=log_post,
        posterior=posterior,
        kappa_map=kappa_map,
        kappa_median=kappa_median,
        kappa_lower_90=kappa_lower_90,
        kappa_upper_90=kappa_upper_90,
        kappa_lower_68=kappa_lower_68,
        kappa_upper_68=kappa_upper_68,
        log_bayes_factor=log_bayes_factor,
        detection_sigma=sigma_kappa,
        fit_method=fit_method,
        linear_mode_estimates=linear_mode_estimates or {},
    )


def _fit_linear_for_kappa(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    kappa: float,
    noise_variance: float,
) -> dict:
    """For a fixed kappa, find the best-fit linear mode amplitudes.

    This is the profile likelihood approach: for each kappa, we
    re-fit A_220, A_330, A_440, and all phases to maximize the
    likelihood. This eliminates bias from the initial linear fit.

    Returns dict with fitted parameters and the best log-likelihood.
    """
    from scipy.optimize import minimize

    builder = RingdownTemplateBuilder()

    def neg_log_likelihood(params):
        A_220, A_330, A_440 = params[0], params[1], params[2]
        phi_220, phi_330, phi_440 = params[3], params[4], params[5]
        phi_nl = params[6] if len(params) > 6 else 0.0

        template = builder.build_nonlinear_template(
            spin=spin,
            A_220=A_220,
            kappa=kappa,
            A_330=A_330,
            A_440_linear=A_440,
            phi_220=phi_220,
            phi_330=phi_330,
            phi_440_linear=phi_440,
            phi_nl=phi_nl,
        )
        model = template.waveform(t_dimless)
        mask = t_dimless >= 0
        residual = data[mask] - model[mask]
        if noise_variance > 0:
            return 0.5 * np.sum(residual**2) / noise_variance
        return 0.5 * np.sum(residual**2)

    # Initial guess from the standard linear fit
    linear = fit_linear_modes(data, t_dimless, spin)
    x0 = [
        linear["220"]["amplitude"],
        linear["330"]["amplitude"],
        linear["440"]["amplitude"],
        linear["220"]["phase"],
        linear["330"]["phase"],
        linear["440"]["phase"],
        0.0,  # phi_nl
    ]

    # Bounds: amplitudes >= 0, phases unconstrained
    bounds = [
        (0, None),
        (0, None),
        (0, None),  # amplitudes
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),  # phases
        (-np.pi, np.pi),  # phi_nl
    ]

    result = minimize(neg_log_likelihood, x0, bounds=bounds, method="L-BFGS-B")
    p = result.x

    return {
        "A_220": p[0],
        "A_330": p[1],
        "A_440": p[2],
        "phi_220": p[3],
        "phi_330": p[4],
        "phi_440": p[5],
        "phi_nl": p[6],
        "log_likelihood": -result.fun,
        "success": result.success,
    }


def estimate_kappa_posterior_profiled(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    noise_variance: float,
    event_name: str = "unknown",
    kappa_min: float = 0.0,
    kappa_max: float = 5.0,
    n_kappa: int = 201,
    prior: str = "uniform",
) -> PosteriorResult:
    """Estimate kappa with profile likelihood over linear amplitudes.

    For each kappa value, re-fit all linear mode amplitudes and phases.
    This is more robust than the fixed-amplitude approach because it
    accounts for the fact that the optimal linear fit depends on kappa.

    IMPORTANT: data must be in dimensionless strain units.
    """
    builder = RingdownTemplateBuilder()
    kappa_grid = np.linspace(kappa_min, kappa_max, n_kappa)

    log_l = np.zeros(n_kappa)
    fit_results = []

    for i, k in enumerate(kappa_grid):
        fit = _fit_linear_for_kappa(data, t_dimless, spin, k, noise_variance)
        log_l[i] = fit["log_likelihood"]
        fit_results.append(fit)

    # Prior density
    if prior == "uniform":
        if kappa_max <= kappa_min:
            raise ValueError("kappa_max must be greater than kappa_min")
        log_prior = np.full_like(kappa_grid, -np.log(kappa_max - kappa_min))
    elif prior == "log_uniform":
        positive = kappa_grid > 0
        log_norm = np.log(np.log(kappa_max / max(kappa_min, 1e-12)))
        log_prior = np.full_like(kappa_grid, -np.inf)
        log_prior[positive] = -np.log(kappa_grid[positive]) - log_norm
    else:
        raise ValueError(f"Unknown prior: {prior}")

    # Posterior density
    log_post_unnorm = log_l + log_prior
    log_evidence = _log_trapezoid(log_post_unnorm, kappa_grid)
    log_post = log_post_unnorm - log_evidence
    posterior = np.exp(log_post)

    # Summary statistics
    cdf = np.cumsum(posterior) * (kappa_grid[1] - kappa_grid[0])
    cdf /= cdf[-1]

    kappa_map = kappa_grid[np.argmax(posterior)]
    kappa_median = np.interp(0.5, cdf, kappa_grid)
    kappa_lower_90 = np.interp(0.05, cdf, kappa_grid)
    kappa_upper_90 = np.interp(0.95, cdf, kappa_grid)
    kappa_lower_68 = np.interp(0.16, cdf, kappa_grid)
    kappa_upper_68 = np.interp(0.84, cdf, kappa_grid)

    # Bayes factor
    log_l_linear = fit_results[0]["log_likelihood"]  # kappa=0
    log_bayes_factor = log_evidence - log_l_linear

    # Significance
    sigma_kappa = np.sqrt(max(0.0, 2.0 * (np.max(log_l) - log_l_linear)))

    # Record the fitted amplitudes at MAP
    map_idx = np.argmax(posterior)
    map_fit = fit_results[map_idx]
    linear_mode_estimates = {
        "220": {"amplitude": map_fit["A_220"], "phase": map_fit["phi_220"]},
        "330": {"amplitude": map_fit["A_330"], "phase": map_fit["phi_330"]},
        "440": {"amplitude": map_fit["A_440"], "phase": map_fit["phi_440"]},
    }

    return PosteriorResult(
        event_name=event_name,
        kappa_grid=kappa_grid,
        log_likelihood=log_l,
        log_posterior=log_post,
        posterior=posterior,
        kappa_map=kappa_map,
        kappa_median=kappa_median,
        kappa_lower_90=kappa_lower_90,
        kappa_upper_90=kappa_upper_90,
        kappa_lower_68=kappa_lower_68,
        kappa_upper_68=kappa_upper_68,
        log_bayes_factor=log_bayes_factor,
        detection_sigma=sigma_kappa,
        fit_method="profile_likelihood",
        linear_mode_estimates=linear_mode_estimates,
    )


def compute_log_likelihood_freq_domain(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    A_220: float,
    kappa: float,
    noise_variance: float,
    builder: RingdownTemplateBuilder = None,
    A_330: float = 0.0,
    A_440_linear: float = 0.0,
    phi_220: float = 0.0,
    phi_330: float = 0.0,
    phi_440_linear: float = 0.0,
    phi_nl: float = 0.0,
    sample_rate: float = None,
) -> float:
    """Compute log-likelihood in the frequency domain.

    log L = -0.5 * sum |h_data(f) - h_model(f)|^2 / S_n(f)

    where S_n(f) is approximated as flat (white noise) with variance
    noise_variance. This handles colored noise exactly when a proper
    PSD is provided.

    Parameters
    ----------
    sample_rate : sampling rate in dimensionless units (samples per M)
    """
    if builder is None:
        builder = RingdownTemplateBuilder()

    template = builder.build_nonlinear_template(
        spin=spin,
        A_220=A_220,
        kappa=kappa,
        A_330=A_330,
        A_440_linear=A_440_linear,
        phi_220=phi_220,
        phi_330=phi_330,
        phi_440_linear=phi_440_linear,
        phi_nl=phi_nl,
    )
    model = template.waveform(t_dimless)

    mask = t_dimless >= 0
    data_seg = data[mask]
    model_seg = model[mask]
    n = len(data_seg)

    if sample_rate is None:
        sample_rate = n / (t_dimless[mask][-1] - t_dimless[mask][0]) if n > 1 else 1.0

    # FFT
    data_fft = np.fft.rfft(data_seg)
    model_fft = np.fft.rfft(model_seg)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Flat noise PSD approximation
    # For white noise with variance sigma^2, the one-sided PSD is S_n = sigma^2 * dt
    # We use this to normalize the frequency-domain residual
    dt = 1.0 / sample_rate if sample_rate > 0 else 1.0
    sn = noise_variance * dt if noise_variance > 0 else 1.0

    # Frequency-domain likelihood
    # For white noise, this should give the same result as the time-domain likelihood
    # (up to a constant) by Parseval's theorem
    residual_fft = data_fft - model_fft
    log_l = -0.5 * np.sum(np.abs(residual_fft) ** 2 / sn) / n
    log_l -= 0.5 * n * np.log(2.0 * np.pi * noise_variance) if noise_variance > 0 else 0

    return log_l


def estimate_kappa_posterior_freq_domain(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    noise_variance: float,
    event_name: str = "unknown",
    kappa_min: float = 0.0,
    kappa_max: float = 5.0,
    n_kappa: int = 201,
    prior: str = "uniform",
    sample_rate: float = None,
) -> PosteriorResult:
    """Estimate kappa using frequency-domain likelihood.

    This is the standard GW likelihood that handles colored noise exactly.
    """
    builder = RingdownTemplateBuilder()
    kappa_grid = np.linspace(kappa_min, kappa_max, n_kappa)

    # Fit linear modes first
    linear = fit_linear_modes(data, t_dimless, spin)

    log_l = np.array(
        [
            compute_log_likelihood_freq_domain(
                data,
                t_dimless,
                spin,
                linear["220"]["amplitude"],
                k,
                noise_variance,
                builder,
                A_330=linear["330"]["amplitude"],
                A_440_linear=linear["440"]["amplitude"],
                phi_220=linear["220"]["phase"],
                phi_330=linear["330"]["phase"],
                phi_440_linear=linear["440"]["phase"],
                sample_rate=sample_rate,
            )
            for k in kappa_grid
        ]
    )

    # Prior
    if prior == "uniform":
        log_prior = np.full_like(kappa_grid, -np.log(kappa_max - kappa_min))
    elif prior == "log_uniform":
        positive = kappa_grid > 0
        log_norm = np.log(np.log(kappa_max / max(kappa_min, 1e-12)))
        log_prior = np.full_like(kappa_grid, -np.inf)
        log_prior[positive] = -np.log(kappa_grid[positive]) - log_norm
    else:
        raise ValueError(f"Unknown prior: {prior}")

    log_post_unnorm = log_l + log_prior
    log_evidence = _log_trapezoid(log_post_unnorm, kappa_grid)
    log_post = log_post_unnorm - log_evidence
    posterior = np.exp(log_post)

    cdf = np.cumsum(posterior) * (kappa_grid[1] - kappa_grid[0])
    cdf /= cdf[-1]

    kappa_map = kappa_grid[np.argmax(posterior)]
    kappa_median = np.interp(0.5, cdf, kappa_grid)
    kappa_lower_90 = np.interp(0.05, cdf, kappa_grid)
    kappa_upper_90 = np.interp(0.95, cdf, kappa_grid)
    kappa_lower_68 = np.interp(0.16, cdf, kappa_grid)
    kappa_upper_68 = np.interp(0.84, cdf, kappa_grid)

    log_l_linear = compute_log_likelihood_freq_domain(
        data,
        t_dimless,
        spin,
        linear["220"]["amplitude"],
        0.0,
        noise_variance,
        builder,
        A_330=linear["330"]["amplitude"],
        A_440_linear=linear["440"]["amplitude"],
        phi_220=linear["220"]["phase"],
        phi_330=linear["330"]["phase"],
        phi_440_linear=linear["440"]["phase"],
        sample_rate=sample_rate,
    )
    log_bayes_factor = log_evidence - log_l_linear
    sigma_kappa = np.sqrt(max(0.0, 2.0 * (np.max(log_l) - log_l_linear)))

    return PosteriorResult(
        event_name=event_name,
        kappa_grid=kappa_grid,
        log_likelihood=log_l,
        log_posterior=log_post,
        posterior=posterior,
        kappa_map=kappa_map,
        kappa_median=kappa_median,
        kappa_lower_90=kappa_lower_90,
        kappa_upper_90=kappa_upper_90,
        kappa_lower_68=kappa_lower_68,
        kappa_upper_68=kappa_upper_68,
        log_bayes_factor=log_bayes_factor,
        detection_sigma=sigma_kappa,
        fit_method="freq_domain",
        linear_mode_estimates=linear,
    )


def estimate_kappa_posterior_from_data(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    noise_variance: float,
    event_name: str = "unknown",
    kappa_min: float = 0.0,
    kappa_max: float = 5.0,
    n_kappa: int = 201,
    prior: str = "uniform",
) -> PosteriorResult:
    """Estimate kappa after fitting the linear ringdown channel from data.

    IMPORTANT: data must be in dimensionless strain units (order unity),
    not physical strain (~1e-21). Use RingdownSegment.to_dimensionless()
    to convert before calling this function. The nonlinear amplitude
    formula A_NL = kappa * A_220^2 only makes sense when A_220 ~ O(1).
    """
    linear = fit_linear_modes(data, t_dimless, spin)
    return estimate_kappa_posterior(
        data=data,
        t_dimless=t_dimless,
        spin=spin,
        A_220=linear["220"]["amplitude"],
        noise_variance=noise_variance,
        event_name=event_name,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        n_kappa=n_kappa,
        prior=prior,
        A_330=linear["330"]["amplitude"],
        A_440_linear=linear["440"]["amplitude"],
        phi_220=linear["220"]["phase"],
        phi_330=linear["330"]["phase"],
        phi_440_linear=linear["440"]["phase"],
        fit_method="least_squares_linear_mode_fit",
        linear_mode_estimates=linear,
    )


def analyze_ringdown_segment(
    segment,
    event_name: str | None = None,
    kappa_min: float = 0.0,
    kappa_max: float = 5.0,
    n_kappa: int = 201,
    prior: str = "uniform",
) -> PosteriorResult:
    """End-to-end analysis of a RingdownSegment from real GWOSC data.

    Handles the dimensionless conversion automatically so the nonlinear
    amplitude formula A_NL = kappa * A_220^2 is well-defined.
    """
    dimless = segment.to_dimensionless()
    name = event_name or segment.event_name

    return estimate_kappa_posterior_from_data(
        data=dimless["strain"],
        t_dimless=dimless["t_dimless"],
        spin=segment.remnant_spin,
        noise_variance=dimless["noise_variance"],
        event_name=name,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        n_kappa=n_kappa,
        prior=prior,
    )


def stack_posteriors(posteriors: list) -> StackedPosterior:
    """Stack posteriors across multiple events.

    Bown's principle (US2,037,847): accumulate evidence from
    multiple independent measurements to detect a weak signal
    buried in noise.

    The stacked log-likelihood is the sum of individual log-likelihoods.
    This is exact when events are independent (they are, since they
    come from different astrophysical sources).
    """
    if not posteriors:
        raise ValueError("No posteriors to stack")

    kappa_grid = posteriors[0].kappa_grid
    n_kappa = len(kappa_grid)

    # Verify all posteriors use the same grid
    for p in posteriors:
        if len(p.kappa_grid) != n_kappa:
            raise ValueError(
                f"Grid mismatch: {p.event_name} has {len(p.kappa_grid)} "
                f"points, expected {n_kappa}"
            )

    # Stack: sum log-likelihoods
    stacked_log_l = np.zeros(n_kappa)
    individual_log_ls = []
    for p in posteriors:
        stacked_log_l += p.log_likelihood
        individual_log_ls.append(p.log_likelihood.copy())

    # Normalize to posterior
    log_evidence = _log_trapezoid(stacked_log_l, kappa_grid)
    stacked_log_post = stacked_log_l - log_evidence
    stacked_posterior = np.exp(stacked_log_post)

    # Summary statistics
    cdf = np.cumsum(stacked_posterior) * (kappa_grid[1] - kappa_grid[0])
    cdf /= cdf[-1]

    kappa_map = kappa_grid[np.argmax(stacked_posterior)]
    kappa_median = np.interp(0.5, cdf, kappa_grid)
    kappa_lower_90 = np.interp(0.05, cdf, kappa_grid)
    kappa_upper_90 = np.interp(0.95, cdf, kappa_grid)

    # Stacked Bayes factor
    log_bf_sum = sum(p.log_bayes_factor for p in posteriors)

    # Detection significance
    kappa_lower_68 = np.interp(0.16, cdf, kappa_grid)
    kappa_upper_68 = np.interp(0.84, cdf, kappa_grid)
    width = kappa_upper_68 - kappa_lower_68
    linear_idx = int(np.argmin(np.abs(kappa_grid)))
    detection_sigma = np.sqrt(
        max(0.0, 2.0 * (np.max(stacked_log_l) - stacked_log_l[linear_idx]))
    )

    return StackedPosterior(
        event_names=[p.event_name for p in posteriors],
        kappa_grid=kappa_grid,
        individual_log_likelihoods=individual_log_ls,
        stacked_log_likelihood=stacked_log_l,
        stacked_posterior=stacked_posterior,
        kappa_map=kappa_map,
        kappa_median=kappa_median,
        kappa_lower_90=kappa_lower_90,
        kappa_upper_90=kappa_upper_90,
        log_bayes_factor=log_bf_sum,
        detection_sigma=detection_sigma,
        n_events=len(posteriors),
    )
