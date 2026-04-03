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

import numpy as np
from dataclasses import dataclass
from .qnm_modes import KerrQNMCatalog


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
    catalog = KerrQNMCatalog()
    mode = catalog.linear_mode(2, 2, 0, spin)
    omega = mode.omega

    mask = t >= 0
    t_pos = t[mask]
    d_pos = data[mask]

    # Basis functions: damped cos and sin
    envelope = np.exp(omega.imag * t_pos)
    basis_cos = envelope * np.cos(omega.real * t_pos)
    basis_sin = envelope * np.sin(omega.real * t_pos)

    # Least squares: d = a * cos_basis + b * sin_basis
    A = np.column_stack([basis_cos, basis_sin])
    coeffs, _, _, _ = np.linalg.lstsq(A, d_pos, rcond=None)
    a, b = coeffs

    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(-b, a)

    # Reconstruct the fitted fundamental
    fitted = np.zeros_like(data)
    fitted[mask] = amplitude * envelope * np.cos(omega.real * t_pos + phase)

    return {
        "amplitude": amplitude,
        "phase": phase,
        "omega": omega,
        "fitted_waveform": fitted,
        "residual": data - fitted,
    }


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
    catalog = KerrQNMCatalog()
    nl_mode = catalog.nonlinear_mode_quadratic(spin)
    omega_nl = nl_mode.omega

    mask = t >= 0
    template = np.zeros_like(t)

    # Amplitude at kappa=1: A_NL = 1.0 * A_220^2
    a_nl = a_220**2

    # Phase locked to 2 * phi_220
    phi_nl = 2.0 * phi_220

    template[mask] = (
        a_nl
        * np.exp(omega_nl.imag * t[mask])
        * np.cos(omega_nl.real * t[mask] + phi_nl)
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
) -> PhaseLockResult:
    """Phase-locked search using frequency-domain matched filtering.

    Works on whitened + bandpassed data (same input as the time-domain
    version). Computes the inner product in the frequency domain:

        <a|b> = sum_f a*(f) b(f)

    By Parseval's theorem this equals the time-domain inner product for
    white noise, but the frequency-domain formulation naturally extends
    to colored noise when a non-flat PSD is supplied.

    Parameters
    ----------
    data : whitened + bandpassed strain
    t : dimensionless time (t=0 at ringdown start)
    spin : remnant spin
    noise_rms : noise RMS of whitened data
    event_name : identifier
    """
    from .qnm_modes import KerrQNMCatalog

    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    nl_mode = catalog.nonlinear_mode_quadratic(spin)

    # Work on t >= 0 segment
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

    # FFT of data
    data_fft = np.fft.rfft(d_pos)

    # --- Fit fundamental (2,2,0) in frequency domain ---
    omega = mode_220.omega
    envelope = np.exp(omega.imag * t_pos)
    basis_cos = envelope * np.cos(omega.real * t_pos)
    basis_sin = envelope * np.sin(omega.real * t_pos)

    cos_fft = np.fft.rfft(basis_cos)
    sin_fft = np.fft.rfft(basis_sin)

    # Frequency-domain inner product with Parseval normalization:
    # sum(a * b) = (1/N) * Re[sum(a_fft* * b_fft)]
    # For whitened data, noise_var scales the inner product.
    noise_var = noise_rms**2 if noise_rms > 0 else 1.0

    def fd_inner(a_fft, b_fft):
        return np.sum((a_fft.conj() * b_fft).real) / (n * noise_var)

    cc = fd_inner(cos_fft, cos_fft)
    ss = fd_inner(sin_fft, sin_fft)
    cs = fd_inner(cos_fft, sin_fft)
    dc = fd_inner(data_fft, cos_fft)
    ds = fd_inner(data_fft, sin_fft)

    # Solve [cc cs; cs ss] [a; b] = [dc; ds]
    det = cc * ss - cs * cs
    if abs(det) < 1e-30:
        a_coeff, b_coeff = 0.0, 0.0
    else:
        a_coeff = (dc * ss - ds * cs) / det
        b_coeff = (ds * cc - dc * cs) / det

    amplitude = np.sqrt(a_coeff**2 + b_coeff**2)
    phase = np.arctan2(-b_coeff, a_coeff)

    # Build residual: data - fitted fundamental
    fitted_fft = a_coeff * cos_fft + b_coeff * sin_fft
    residual_fft = data_fft - fitted_fft

    # --- Build nonlinear template and compute SNR ---
    omega_nl = nl_mode.omega
    a_nl = amplitude**2  # kappa=1 template
    phi_nl = 2.0 * phase

    nl_template = a_nl * envelope * np.cos(omega_nl.real * t_pos + phi_nl)
    nl_fft = np.fft.rfft(nl_template)

    # Frequency-domain inner products for nonlinear mode
    nn = fd_inner(nl_fft, nl_fft)
    rn = fd_inner(residual_fft, nl_fft)

    template_norm = np.sqrt(max(nn, 0.0))

    if template_norm > 0:
        snr = rn / template_norm
        kappa_hat = rn / nn
        kappa_sigma = 1.0 / template_norm
    else:
        snr = 0.0
        kappa_hat = 0.0
        kappa_sigma = float("inf")

    return PhaseLockResult(
        event_name=event_name,
        kappa_hat=kappa_hat,
        kappa_sigma=kappa_sigma,
        snr=snr,
        a_220_fit=amplitude,
        phi_220_fit=phase,
        template_norm=template_norm,
        residual_overlap=rn,
        noise_rms=noise_rms,
    )


def stack_phase_locked(
    results: list, max_weight_ratio: float | None = None
) -> StackedPhaseLockResult:
    """Stack phase-locked search results across events.

    The optimal stacking for matched-filter searches:
    - Combined SNR^2 = sum of individual SNR^2
    - Combined kappa = weighted average (weights = 1/sigma^2)

    This is Bown's pulse-averaging principle: each event is a noisy
    measurement of kappa. The stack is the optimal combination.

    Parameters
    ----------
    results : list of PhaseLockResult
    max_weight_ratio : float, optional
        If set, cap each event's weight at this multiple of the
        average weight (w_avg = total_weight / N). For example,
        max_weight_ratio=5 means no event carries more than 5×
        the average weight. This reduces jackknife influence
        concentration without biasing kappa when the underlying
        signal is constant across events. Standard technique in
        meta-analysis (Cochrane Handbook, §10.10.4.1).
    """
    if not results:
        raise ValueError("No results to stack")

    # Weighted average of kappa
    weights = []
    kappas = []
    snr_sq_sum = 0.0

    for r in results:
        if r.kappa_sigma > 0 and np.isfinite(r.kappa_sigma):
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

    # Apply weight cap if requested
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
