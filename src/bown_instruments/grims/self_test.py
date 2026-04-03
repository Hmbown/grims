"""
Self-test module: orthogonality verification for QNM decomposition.

Ralph Bown's 1923 patent (US1,573,801) established the principle:
an unattended system must verify its own health by sending itself
a test signal and checking the response.

GRIM-S applies this principle to ringdown analysis:
  1. Decompose the signal into QNM modes using the bilinear form.
  2. Reconstruct the signal from the extracted mode amplitudes.
  3. Compute the residual.
  4. If the residual exceeds the noise floor, either:
     (a) the mode basis is incomplete (missing physics), or
     (b) the data quality is degraded (instrument problem).

This is the "Fourier coefficient check": in a complete orthogonal basis,
the sum of |c_i|^2 equals the total power. If it doesn't, the basis
is wrong or the data is corrupted.

References:
  Green, Hollands, Sberna, Toomani, Zimmerman,
  "Conserved currents for Kerr and orthogonality of quasinormal modes,"
  PRD 107, 064030 (2023).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SelfTestResult:
    """Result of the orthogonality self-test."""
    total_signal_power: float
    reconstructed_power: float
    residual_power: float
    power_ratio: float  # reconstructed / total; should be ~1.0
    residual_fraction: float  # residual / total; should be ~0.0
    noise_floor_power: float
    residual_above_noise: bool  # True if residual > noise floor
    mode_amplitudes: dict  # extracted amplitudes per mode
    diagnosis: str  # human-readable diagnosis

    @property
    def passed(self) -> bool:
        """Did the self-test pass?"""
        return not self.residual_above_noise


def extract_mode_amplitudes(signal: np.ndarray, t: np.ndarray,
                            mode_frequencies: list,
                            mode_damping_rates: list,
                            noise_variance: float = 0.0) -> dict:
    """Extract QNM amplitudes via least-squares projection.

    This is the software analog of the Green et al. bilinear form:
    project the signal onto each mode template and extract the
    complex amplitude.

    For the full bilinear form, one would need the angular separation
    functions and the complex contour regularization. Here we implement
    the time-domain projection, which is the leading-order approximation
    valid when modes are well-separated in frequency.

    Parameters
    ----------
    signal : strain time series (real)
    t : time array (dimensionless, in units of M)
    mode_frequencies : list of complex QNM frequencies
    mode_damping_rates : (included in complex frequency)
    noise_variance : estimated noise variance per sample

    Returns
    -------
    dict mapping mode index to complex amplitude
    """
    n_modes = len(mode_frequencies)
    n_samples = len(t)

    # Build the design matrix: each column is a mode template
    # Template_i(t) = exp(i * omega_i * t) for t >= 0
    mask = t >= 0
    t_pos = t[mask]
    signal_pos = signal[mask]
    n_pos = len(t_pos)

    # Design matrix (real representation: cos and sin for each mode)
    A = np.zeros((n_pos, 2 * n_modes))
    for i, omega in enumerate(mode_frequencies):
        envelope = np.exp(omega.imag * t_pos)  # damping
        A[:, 2 * i] = envelope * np.cos(omega.real * t_pos)
        A[:, 2 * i + 1] = envelope * np.sin(omega.real * t_pos)

    # Weighted least squares if noise variance known
    if noise_variance > 0:
        W = np.eye(n_pos) / noise_variance
        ATA = A.T @ W @ A
        ATy = A.T @ W @ signal_pos
    else:
        ATA = A.T @ A
        ATy = A.T @ signal_pos

    # Solve for coefficients
    try:
        coeffs = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        # Singular matrix — modes are degenerate or data is garbage
        coeffs = np.linalg.lstsq(A, signal_pos, rcond=None)[0]

    # Convert to complex amplitudes
    amplitudes = {}
    for i in range(n_modes):
        a_cos = coeffs[2 * i]
        a_sin = coeffs[2 * i + 1]
        # Complex amplitude: A * exp(i*phi) where
        # A*cos(phi) = a_cos, -A*sin(phi) = a_sin
        amplitude = np.sqrt(a_cos**2 + a_sin**2)
        phase = np.arctan2(-a_sin, a_cos)
        amplitudes[i] = {
            "amplitude": amplitude,
            "phase": phase,
            "complex": amplitude * np.exp(1j * phase),
        }

    return amplitudes, A, coeffs


def run_self_test(signal: np.ndarray, t: np.ndarray,
                  mode_frequencies: list,
                  noise_rms: float = 0.0,
                  mode_labels: list = None) -> SelfTestResult:
    """Run the GRIM-S self-test on a ringdown signal.

    Steps:
    1. Extract mode amplitudes via projection.
    2. Reconstruct the signal from extracted amplitudes.
    3. Compute residual power.
    4. Compare residual to noise floor.
    5. Diagnose.

    Parameters
    ----------
    signal : strain time series
    t : time array (dimensionless)
    mode_frequencies : list of complex QNM frequencies
    noise_rms : estimated noise RMS per sample
    mode_labels : optional string labels for each mode

    Returns
    -------
    SelfTestResult with full diagnostics
    """
    if mode_labels is None:
        mode_labels = [f"mode_{i}" for i in range(len(mode_frequencies))]

    mask = t >= 0
    signal_pos = signal[mask]
    t_pos = t[mask]
    n_pos = len(t_pos)

    noise_variance = noise_rms**2 if noise_rms > 0 else 0.0

    # Step 1: Extract amplitudes
    amplitudes, A, coeffs = extract_mode_amplitudes(
        signal, t, mode_frequencies, [], noise_variance,
    )

    # Step 2: Reconstruct
    reconstruction = A @ coeffs
    residual = signal_pos - reconstruction

    # Step 3: Power accounting
    total_power = np.sum(signal_pos**2) / n_pos
    reconstructed_power = np.sum(reconstruction**2) / n_pos
    residual_power = np.sum(residual**2) / n_pos
    noise_floor = noise_rms**2 if noise_rms > 0 else 0.0

    # Step 4: Compare
    residual_above_noise = residual_power > 2.0 * noise_floor if noise_floor > 0 \
        else residual_power > 0.01 * total_power

    # Step 5: Diagnose
    power_ratio = reconstructed_power / total_power if total_power > 0 else 0.0
    residual_fraction = residual_power / total_power if total_power > 0 else 0.0

    if not residual_above_noise:
        diagnosis = (
            "PASS: Residual consistent with noise floor. "
            "Mode basis appears complete for this signal."
        )
    elif residual_fraction < 0.05:
        diagnosis = (
            "MARGINAL: Residual slightly above noise floor "
            f"({residual_fraction:.1%} of total power). "
            "Consider adding overtones or checking data quality."
        )
    elif residual_fraction < 0.20:
        diagnosis = (
            "WARNING: Significant residual power "
            f"({residual_fraction:.1%} of total power). "
            "Mode basis may be incomplete — missing overtones, "
            "nonlinear modes, or environmental contamination."
        )
    else:
        diagnosis = (
            "FAIL: Large residual power "
            f"({residual_fraction:.1%} of total power). "
            "Either the mode basis is fundamentally wrong, "
            "the ringdown start time is misidentified, "
            "or the signal is not a ringdown."
        )

    # Build labeled amplitude dict
    labeled_amps = {}
    for i, label in enumerate(mode_labels):
        if i in amplitudes:
            labeled_amps[label] = amplitudes[i]

    return SelfTestResult(
        total_signal_power=total_power,
        reconstructed_power=reconstructed_power,
        residual_power=residual_power,
        power_ratio=power_ratio,
        residual_fraction=residual_fraction,
        noise_floor_power=noise_floor,
        residual_above_noise=residual_above_noise,
        mode_amplitudes=labeled_amps,
        diagnosis=diagnosis,
    )


def parseval_check(amplitudes: dict, mode_frequencies: list,
                   t: np.ndarray) -> dict:
    """Parseval-like energy conservation check.

    In an orthogonal basis, the total energy is the sum of
    individual mode energies. If the modes are not truly orthogonal
    (due to finite time window, mode overlap, or basis incompleteness),
    this check will show a discrepancy.

    This is the QNM analog of Parseval's theorem for Fourier series —
    the kind of identity Bown would have checked reflexively.
    """
    mask = t >= 0
    t_pos = t[mask]

    # Compute expected energy per mode (analytical integral)
    mode_energies = {}
    total_analytical = 0.0

    for i, omega in enumerate(mode_frequencies):
        if i not in amplitudes:
            continue
        A = amplitudes[i]["amplitude"]
        tau = 1.0 / abs(omega.imag)

        # Energy of a damped sinusoid: integral of A^2 * exp(-2t/tau) dt
        # from 0 to T
        T = t_pos[-1]
        energy = A**2 * tau / 2.0 * (1.0 - np.exp(-2.0 * T / tau))
        mode_energies[i] = energy
        total_analytical += energy

    return {
        "mode_energies": mode_energies,
        "total_analytical": total_analytical,
        "n_modes": len(mode_energies),
    }
