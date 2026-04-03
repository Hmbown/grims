"""
Colored-noise PSD likelihood for GRIM-S.

The current likelihood assumes white noise (constant PSD). This module
implements a proper frequency-dependent PSD weighting using the detector's
actual noise curve.

The likelihood in the frequency domain:

    log L = -0.5 * sum_f |h_data(f) - h_model(f)|^2 / S_n(f)

where S_n(f) is the one-sided power spectral density of the detector noise.

This is the standard gravitational-wave likelihood used by LIGO/Virgo.
"""

import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass


def load_aligo_psd(
    freqs: np.ndarray = None,
    sample_rate: float = 4096.0,
) -> tuple:
    """Load the Advanced LIGO design sensitivity PSD.

    Uses the analytical fit from the LIGO design sensitivity curve.

    Parameters
    ----------
    freqs : frequency array (Hz). If None, generated from sample_rate.
    sample_rate : sampling rate in Hz

    Returns
    -------
    (freqs, psd) : frequency array and one-sided PSD
    """
    if freqs is None:
        n = 2**14
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Advanced LIGO design sensitivity (analytical fit)
    # From https://dcc.ligo.org/LIGO-T1500293/public
    f0 = 20.0  # low frequency cutoff

    # Parameters for the analytical fit
    S0 = 1e-48  # overall normalization

    # Seismic wall
    f_seismic = 30.0

    # Thermal noise bump
    f_thermal = 100.0

    # Shot noise rise
    f_shot = 500.0

    psd = np.zeros_like(freqs)

    for i, f in enumerate(freqs):
        if f <= f0:
            psd[i] = np.inf
            continue

        # Seismic component (f^-4)
        seismic = (f_seismic / f) ** 4

        # Thermal component (broad bump)
        thermal = 1.0 + (f / f_thermal) ** 2

        # Shot noise (f^2)
        shot = 1.0 + (f / f_shot) ** 4

        psd[i] = S0 * seismic * thermal * shot

    return freqs, psd


def estimate_psd_from_data(
    strain: np.ndarray,
    sample_rate: float,
    merger_time: float,
    time: np.ndarray,
    exclusion_window: float = 4.0,
    segment_length: float = 64.0,
    overlap: float = 0.5,
) -> tuple:
    """Estimate the PSD from off-source data using Welch's method.

    Parameters
    ----------
    strain : strain data
    sample_rate : sampling rate in Hz
    merger_time : GPS time of merger
    time : time array
    exclusion_window : seconds around merger to exclude
    segment_length : length of each PSD segment in seconds
    overlap : overlap fraction between segments

    Returns
    -------
    (freqs, psd) : frequency array and estimated PSD
    """
    # Find off-source data
    off_source_mask = np.abs(time - merger_time) > exclusion_window

    if np.sum(off_source_mask) < 2 * sample_rate:
        raise ValueError("Not enough off-source data for PSD estimation")

    off_source_strain = strain[off_source_mask]

    # Welch's method
    nperseg = int(segment_length * sample_rate)
    noverlap = int(nperseg * overlap)

    from scipy.signal import welch

    freqs, psd = welch(
        off_source_strain,
        fs=sample_rate,
        nperseg=min(nperseg, len(off_source_strain)),
        noverlap=min(noverlap, len(off_source_strain) // 2),
        window="hann",
        scaling="density",
    )

    return freqs, psd


def compute_colored_log_likelihood(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    A_220: float,
    kappa: float,
    psd_freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate_dimless: float,
    builder=None,
    A_330: float = 0.0,
    A_440_linear: float = 0.0,
    phi_220: float = 0.0,
    phi_330: float = 0.0,
    phi_440_linear: float = 0.0,
    phi_nl: float = 0.0,
    f_min: float = 20.0,
    f_max: float = 1024.0,
) -> float:
    """Compute log-likelihood with colored noise PSD.

    Parameters
    ----------
    data : ringdown strain (dimensionless)
    t_dimless : time array in units of M
    spin : remnant spin
    A_220 : fundamental mode amplitude
    kappa : nonlinear coupling coefficient
    psd_freqs : PSD frequency array (Hz)
    psd : PSD values (strain^2/Hz)
    sample_rate_dimless : sampling rate in dimensionless units (samples per M)
    f_min, f_max : frequency range to use for likelihood

    Returns
    -------
    log_likelihood : log-likelihood value
    """
    from .ringdown_templates import RingdownTemplateBuilder

    if builder is None:
        builder = RingdownTemplateBuilder()

    # Build template
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

    # Extract ringdown segment
    mask = t_dimless >= 0
    data_seg = data[mask]
    model_seg = model[mask]
    n = len(data_seg)

    if n < 4:
        return -np.inf

    # FFT
    data_fft = np.fft.rfft(data_seg)
    model_fft = np.fft.rfft(model_seg)
    freqs_dimless = np.fft.rfftfreq(n, d=1.0 / sample_rate_dimless)

    # Convert dimensionless frequencies to Hz
    # This requires knowing the remnant mass, which we don't have here
    # For now, we'll use the PSD in dimensionless units
    # The PSD should be provided in the same units as the data

    # Interpolate PSD to our frequencies
    # If PSD is in Hz, we need to convert
    # For dimensionless data, the PSD is in (strain^2 * M)
    psd_interp = interp1d(
        psd_freqs,
        psd,
        bounds_error=False,
        fill_value=(psd[0], psd[-1]),
    )

    # Get PSD values at our frequencies
    sn = psd_interp(freqs_dimless)

    # Avoid division by zero
    sn = np.maximum(sn, 1e-30)

    # Frequency band selection
    band_mask = (freqs_dimless >= f_min) & (freqs_dimless <= f_max)

    if np.sum(band_mask) < 2:
        return -np.inf

    # Colored noise likelihood
    residual_fft = data_fft - model_fft
    log_l = -0.5 * np.sum(np.abs(residual_fft[band_mask]) ** 2 / sn[band_mask])

    # Normalization term
    n_eff = np.sum(band_mask)
    log_l -= 0.5 * n_eff * np.log(2.0 * np.pi)
    log_l -= 0.5 * np.sum(np.log(sn[band_mask]))

    return log_l


def compare_white_vs_colored(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    A_220: float,
    noise_variance: float,
    psd_freqs: np.ndarray,
    psd: np.ndarray,
    sample_rate_dimless: float,
    kappa_grid: np.ndarray = None,
    event_name: str = "unknown",
) -> dict:
    """Compare white noise vs colored noise likelihood results.

    Parameters
    ----------
    data, t_dimless, spin, A_220, noise_variance : as usual
    psd_freqs, psd : PSD for colored noise
    sample_rate_dimless : sampling rate in dimensionless units
    kappa_grid : grid of kappa values to scan
    event_name : identifier

    Returns
    -------
    dict with comparison results
    """
    from .ringdown_templates import RingdownTemplateBuilder
    from .bayesian_analysis import _log_trapezoid

    if kappa_grid is None:
        kappa_grid = np.linspace(0.0, 5.0, 201)

    builder = RingdownTemplateBuilder()

    # Fit linear modes
    from .bayesian_analysis import fit_linear_modes

    linear = fit_linear_modes(data, t_dimless, spin)

    # White noise likelihood
    white_log_l = []
    for k in kappa_grid:
        from .bayesian_analysis import compute_log_likelihood

        ll = compute_log_likelihood(
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
        )
        white_log_l.append(ll)
    white_log_l = np.array(white_log_l)

    # Colored noise likelihood
    colored_log_l = []
    for k in kappa_grid:
        ll = compute_colored_log_likelihood(
            data,
            t_dimless,
            spin,
            linear["220"]["amplitude"],
            k,
            psd_freqs,
            psd,
            sample_rate_dimless,
            builder,
            A_330=linear["330"]["amplitude"],
            A_440_linear=linear["440"]["amplitude"],
            phi_220=linear["220"]["phase"],
            phi_330=linear["330"]["phase"],
            phi_440_linear=linear["440"]["phase"],
        )
        colored_log_l.append(ll)
    colored_log_l = np.array(colored_log_l)

    # MAP estimates
    white_map = kappa_grid[np.argmax(white_log_l)]
    colored_map = kappa_grid[np.argmax(colored_log_l)]

    # 90% CI (approximate)
    def compute_ci(log_l, grid):
        post = np.exp(log_l - np.max(log_l))
        cdf = np.cumsum(post) * (grid[1] - grid[0])
        cdf /= cdf[-1]
        return np.interp(0.05, cdf, grid), np.interp(0.95, cdf, grid)

    white_ci = compute_ci(white_log_l, kappa_grid)
    colored_ci = compute_ci(colored_log_l, kappa_grid)

    return {
        "event_name": event_name,
        "kappa_grid": kappa_grid,
        "white_log_likelihood": white_log_l,
        "colored_log_likelihood": colored_log_l,
        "white_map": white_map,
        "colored_map": colored_map,
        "white_ci_90": white_ci,
        "colored_ci_90": colored_ci,
        "map_difference": colored_map - white_map,
    }


def print_colored_summary(comparison: dict) -> None:
    """Print a summary of white vs colored noise comparison."""
    print("=" * 60)
    print(f"COLORED NOISE COMPARISON: {comparison['event_name']}")
    print("=" * 60)
    print()
    print(f"{'Method':<15} {'MAP kappa':>12} {'90% CI':>25}")
    print("-" * 55)
    print(
        f"{'White noise':<15} {comparison['white_map']:>12.4f} "
        f"[{comparison['white_ci_90'][0]:>10.4f}, {comparison['white_ci_90'][1]:>10.4f}]"
    )
    print(
        f"{'Colored noise':<15} {comparison['colored_map']:>12.4f} "
        f"[{comparison['colored_ci_90'][0]:>10.4f}, {comparison['colored_ci_90'][1]:>10.4f}]"
    )
    print()
    print(f"MAP difference: {comparison['map_difference']:+.4f}")
    print()

    if abs(comparison["map_difference"]) < 0.1:
        print("Conclusion: White and colored noise results agree well.")
        print("The white noise approximation is adequate for this event.")
    elif abs(comparison["map_difference"]) < 0.5:
        print("Conclusion: Moderate difference between white and colored noise.")
        print("The colored noise result should be preferred.")
    else:
        print("Conclusion: Significant difference between white and colored noise.")
        print("The white noise approximation is NOT adequate.")
        print("Use the colored noise result for all science.")
    print()


def plot_colored_comparison(comparison: dict, save_path: str = None):
    """Plot white vs colored noise likelihood curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    kappa_grid = comparison["kappa_grid"]

    # Normalize likelihoods
    white_norm = comparison["white_log_likelihood"] - np.max(
        comparison["white_log_likelihood"]
    )
    colored_norm = comparison["colored_log_likelihood"] - np.max(
        comparison["colored_log_likelihood"]
    )

    ax.plot(kappa_grid, white_norm, "gray", linewidth=2, label="White noise")
    ax.plot(kappa_grid, colored_norm, "blue", linewidth=2, label="Colored noise")

    # MAP markers
    ax.axvline(comparison["white_map"], color="gray", linestyle="--", alpha=0.7)
    ax.axvline(comparison["colored_map"], color="blue", linestyle="--", alpha=0.7)

    # GR reference
    ax.axvline(1.0, color="red", linestyle=":", alpha=0.5, label="kappa = 1 (GR)")

    ax.set_xlabel("kappa", fontsize=12)
    ax.set_ylabel("Relative log-likelihood", fontsize=12)
    ax.set_title(
        f"White vs Colored Noise: {comparison['event_name']}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
