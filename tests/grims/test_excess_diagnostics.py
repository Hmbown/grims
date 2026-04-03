"""
Excess diagnostics for the 2.05-sigma nonlinear mode candidate in GW150914 H1.

Five stress tests to determine whether the excess at f_NL ~ 541 Hz is:
  (a) a real nonlinear QNM coupling signal, or
  (b) a noise artifact, instrument line, or statistical fluctuation.

Tests:
  A. Time-frequency map -- does the excess correlate with the ringdown epoch?
  B. Frequency precision -- is the peak at exactly 2*f_220?
  C. Phase coherence -- is the residual phase consistent with 2*phi_220?
  D. Adjacent frequency bands -- is the excess narrowband at f_NL or broadband?
  E. Null test (time shifts) -- does the excess vanish when the ringdown is absent?

Bown's principle: "You cannot improve what you cannot observe."
This script is the instrument we point at our own measurement.

Usage:
    python tests/test_excess_diagnostics.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from bown_instruments.grims.whiten import (
    estimate_asd,
    whiten_strain,
    bandpass,
    prepare_ringdown_for_analysis,
)
from bown_instruments.grims.phase_locked_search import (
    fit_fundamental_mode,
    build_phase_locked_template,
    phase_locked_search,
    PhaseLockResult,
)
from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.gwtc_pipeline import (
    M_SUN_SECONDS,
    get_candidate_event,
    load_gwosc_strain_hdf5,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EVENT_NAME = "GW150914"
DETECTOR = "H1"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def ctx():
    """Shared whitened GW150914 context for all excess diagnostics."""
    try:
        return load_full_whitened_strain()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


# ---------------------------------------------------------------------------
# Helpers: load the full 32-second strain and prepare whitened data
# ---------------------------------------------------------------------------


def load_full_whitened_strain():
    """Load GW150914 H1, whiten, and return physical-time arrays.

    Returns a dict with the full whitened+bandpassed strain in physical
    GPS time, plus event/mode metadata needed by every test.
    """
    event = get_candidate_event(EVENT_NAME)
    mass = event["remnant_mass_msun"]
    spin = event["remnant_spin"]
    m_seconds = mass * M_SUN_SECONDS
    merger_gps = float(event["gps_time"])

    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    tau_220_s = mode_220.physical_damping_time_s(mass)

    # Load the 32-second HDF5 file
    from pathlib import Path

    data_path = Path(DATA_DIR)
    candidates = sorted(data_path.glob(f"*{DETECTOR}*1126259447*.hdf5"))
    if not candidates:
        # Broader search by GPS prefix
        candidates = sorted(data_path.glob(f"*H1*112625*.hdf5"))
    if not candidates:
        raise FileNotFoundError(
            f"No HDF5 file found for {EVENT_NAME} {DETECTOR} in {DATA_DIR}"
        )

    loaded = load_gwosc_strain_hdf5(str(candidates[0]))
    full_strain = loaded["strain"]
    full_time = loaded["time"]
    sample_rate = loaded["sample_rate"]

    # Bandpass: wide enough to cover f_220 and f_NL plus margin
    f_low = max(20.0, f_220 * 0.5)
    f_high = min(0.45 * sample_rate, f_nl * 1.5)

    # Estimate ASD from off-source data
    asd_freqs, asd = estimate_asd(
        full_strain,
        sample_rate,
        merger_time=merger_gps,
        time=full_time,
        exclusion_window=2.0,
    )

    # Whiten full 32 seconds
    whitened = whiten_strain(
        full_strain, sample_rate, asd_freqs, asd, fmin=f_low * 0.8
    )

    # Bandpass
    whitened_bp = bandpass(whitened, sample_rate, f_low, f_high)

    # Ringdown start: 10 M after merger
    t_start_m = 10.0
    ringdown_start_gps = merger_gps + t_start_m * m_seconds

    return {
        "strain": whitened_bp,
        "time": full_time,
        "sample_rate": sample_rate,
        "merger_gps": merger_gps,
        "ringdown_start_gps": ringdown_start_gps,
        "f_220": f_220,
        "f_nl": f_nl,
        "tau_220_s": tau_220_s,
        "m_seconds": m_seconds,
        "mass": mass,
        "spin": spin,
        "mode_220": mode_220,
        "mode_nl": mode_nl,
        "event": event,
    }


def extract_ringdown_segment(ctx, duration_s=0.15, pad_before_s=0.05):
    """Extract the ringdown segment in dimensionless time."""
    t0 = ctx["ringdown_start_gps"] - pad_before_s
    t1 = ctx["ringdown_start_gps"] + duration_s
    mask = (ctx["time"] >= t0) & (ctx["time"] <= t1)
    seg_strain = ctx["strain"][mask]
    seg_time = ctx["time"][mask]
    t_dimless = (seg_time - ctx["ringdown_start_gps"]) / ctx["m_seconds"]

    # Noise variance from far off-source
    noise_mask = np.abs(ctx["time"] - ctx["merger_gps"]) > 4.0
    noise_var = np.var(ctx["strain"][noise_mask])

    return seg_strain, t_dimless, np.sqrt(noise_var)


# ===========================================================================
# TEST A: Time-Frequency Map of the Excess
# ===========================================================================


def test_a_time_frequency_map(ctx):
    """Spectrogram around the merger: does 541 Hz appear only at ringdown?"""
    print("=" * 70)
    print("TEST A: Time-Frequency Map of the Excess")
    print("=" * 70)

    sr = ctx["sample_rate"]
    merger_gps = ctx["merger_gps"]
    f_220 = ctx["f_220"]
    f_nl = ctx["f_nl"]

    # Window around the event: -0.5s to +0.5s relative to merger
    window_s = 0.5
    mask = (ctx["time"] >= merger_gps - window_s) & (
        ctx["time"] <= merger_gps + window_s
    )
    seg = ctx["strain"][mask]
    seg_t = ctx["time"][mask] - merger_gps  # relative to merger

    if len(seg) == 0:
        print("  ERROR: no data in window around merger.")
        return

    # Short-time FFT with ~2 ms windows
    nperseg = int(0.002 * sr)  # ~8 samples at 4096 Hz
    # Ensure nperseg is at least 8 and a power of 2 for FFT efficiency
    nperseg = max(8, nperseg)
    # Use zero-padded FFT for better frequency resolution in the plot
    nfft = max(nperseg, 256)

    # Compute spectrogram manually for full control
    hop = nperseg // 2
    n_windows = (len(seg) - nperseg) // hop + 1
    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    spectrogram = np.zeros((len(freqs), n_windows))
    t_centers = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * hop
        chunk = seg[start : start + nperseg] * window
        # Zero-pad to nfft
        padded = np.zeros(nfft)
        padded[:nperseg] = chunk
        fft_vals = np.fft.rfft(padded)
        spectrogram[:, i] = np.abs(fft_vals) ** 2
        t_centers[i] = seg_t[start + nperseg // 2]

    # Convert to dB (relative to median noise floor)
    spec_db = 10.0 * np.log10(spectrogram + 1e-30)
    noise_floor = np.median(spec_db)

    # --- Analysis: compare power at f_NL during ringdown vs. off-source ---
    # Ringdown window: 0 to +50 ms after merger
    rd_mask = (t_centers >= 0.0) & (t_centers <= 0.05)
    # Off-source: -0.4 to -0.1 s before merger
    off_mask = (t_centers >= -0.4) & (t_centers <= -0.1)

    f_nl_idx = np.argmin(np.abs(freqs - f_nl))
    f_220_idx = np.argmin(np.abs(freqs - f_220))

    # Frequency band: average over a few bins around the target
    bw = 3  # bins on each side
    f_nl_lo = max(0, f_nl_idx - bw)
    f_nl_hi = min(len(freqs), f_nl_idx + bw + 1)
    f_220_lo = max(0, f_220_idx - bw)
    f_220_hi = min(len(freqs), f_220_idx + bw + 1)

    power_nl_rd = np.mean(spectrogram[f_nl_lo:f_nl_hi, :][:, rd_mask])
    power_nl_off = np.mean(spectrogram[f_nl_lo:f_nl_hi, :][:, off_mask])
    power_220_rd = np.mean(spectrogram[f_220_lo:f_220_hi, :][:, rd_mask])
    power_220_off = np.mean(spectrogram[f_220_lo:f_220_hi, :][:, off_mask])

    ratio_nl = power_nl_rd / (power_nl_off + 1e-30)
    ratio_220 = power_220_rd / (power_220_off + 1e-30)

    print(f"\n  f_220 = {f_220:.1f} Hz")
    print(f"  f_NL  = {f_nl:.1f} Hz")
    print(f"  Spectrogram: {nperseg} samples/window ({nperseg/sr*1e3:.1f} ms), "
          f"nfft={nfft}, {n_windows} windows")
    print(f"\n  Power ratio (ringdown / off-source):")
    print(f"    at f_220 ({f_220:.0f} Hz): {ratio_220:.2f}x")
    print(f"    at f_NL  ({f_nl:.0f} Hz): {ratio_nl:.2f}x")

    if ratio_nl > 2.0:
        print(f"  --> f_NL power is {ratio_nl:.1f}x higher during ringdown.")
        print("      Consistent with event-associated signal.")
    elif ratio_nl > 1.2:
        print(f"  --> f_NL power is marginally elevated ({ratio_nl:.1f}x).")
        print("      Inconclusive -- could be signal or fluctuation.")
    else:
        print(f"  --> f_NL power is NOT elevated during ringdown ({ratio_nl:.1f}x).")
        print("      Suggests noise line or broadband artifact, not event-associated.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    # Limit frequency range to the region of interest
    f_plot_mask = (freqs >= f_220 * 0.5) & (freqs <= f_nl * 1.5)
    extent = [
        t_centers[0] * 1e3,
        t_centers[-1] * 1e3,
        freqs[f_plot_mask][0],
        freqs[f_plot_mask][-1],
    ]
    ax.imshow(
        spec_db[f_plot_mask, :],
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=noise_floor - 5,
        vmax=noise_floor + 25,
    )
    ax.axhline(f_220, color="cyan", ls="--", lw=1.5, label=f"f_220 = {f_220:.0f} Hz")
    ax.axhline(f_nl, color="lime", ls="--", lw=1.5, label=f"f_NL = {f_nl:.0f} Hz")
    ax.axvline(0, color="white", ls=":", lw=1, label="merger")

    ax.set_xlabel("Time relative to merger (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Test A: Time-Frequency Map -- {EVENT_NAME} {DETECTOR}")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "test_a_spectrogram.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Saved: plots/test_a_spectrogram.png")


# ===========================================================================
# TEST B: Frequency Precision
# ===========================================================================


def test_b_frequency_precision(ctx):
    """Is the peak frequency in the residual at exactly 2*f_220?"""
    print("\n" + "=" * 70)
    print("TEST B: Frequency Precision of the Excess")
    print("=" * 70)

    seg_strain, t_dimless, noise_rms = extract_ringdown_segment(ctx)

    # Fit the (2,2,0) mode
    fit = fit_fundamental_mode(seg_strain, t_dimless, ctx["spin"])
    residual = fit["residual"]

    # Use only the post-ringdown portion (t >= 0 in dimensionless time)
    mask = t_dimless >= 0
    residual_rd = residual[mask]

    if len(residual_rd) < 16:
        print("  ERROR: ringdown segment too short for spectral analysis.")
        return

    sr = ctx["sample_rate"]
    f_220 = ctx["f_220"]
    f_nl = ctx["f_nl"]

    # Compute power spectrum of the residual
    # Zero-pad for finer frequency resolution
    nfft = max(2048, len(residual_rd) * 4)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    window = np.hanning(len(residual_rd))
    padded = np.zeros(nfft)
    padded[: len(residual_rd)] = residual_rd * window
    spectrum = np.abs(np.fft.rfft(padded)) ** 2

    # Spectral resolution
    T_ringdown = len(residual_rd) / sr
    delta_f = 1.0 / T_ringdown
    delta_f_padded = sr / nfft  # resolution of the zero-padded FFT

    # Find the peak near f_NL
    search_band = 100.0  # Hz around f_NL
    band_mask = (freqs >= f_nl - search_band) & (freqs <= f_nl + search_band)
    if not np.any(band_mask):
        print("  ERROR: f_NL outside frequency range.")
        return

    band_freqs = freqs[band_mask]
    band_power = spectrum[band_mask]
    peak_idx = np.argmax(band_power)
    f_peak = band_freqs[peak_idx]

    # Offset from prediction
    delta = f_peak - f_nl
    delta_relative = delta / f_nl

    print(f"\n  Predicted f_NL = 2 * f_220 = {f_nl:.2f} Hz")
    print(f"  Observed peak frequency     = {f_peak:.2f} Hz")
    print(f"  Offset: {delta:+.2f} Hz ({delta_relative:+.4%})")
    print(f"\n  Spectral resolution (1/T_ringdown) = {delta_f:.1f} Hz")
    print(f"  Zero-padded bin width               = {delta_f_padded:.2f} Hz")

    if abs(delta) < delta_f:
        print(f"\n  --> Offset ({abs(delta):.1f} Hz) < spectral resolution "
              f"({delta_f:.1f} Hz)")
        print("      CONSISTENT with the nonlinear mode prediction.")
    else:
        print(f"\n  --> Offset ({abs(delta):.1f} Hz) > spectral resolution "
              f"({delta_f:.1f} Hz)")
        print("      INCONSISTENT -- the peak is not at the predicted frequency.")

    # SNR of the peak relative to the local noise floor
    # Use the median of the search band as the noise estimate
    noise_level = np.median(band_power)
    peak_snr = (band_power[peak_idx] - noise_level) / (noise_level + 1e-30)
    print(f"\n  Peak power / median noise in band: {peak_snr:.1f}x")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_mask = (freqs >= f_nl - 200) & (freqs <= f_nl + 200)
    ax.semilogy(freqs[plot_mask], spectrum[plot_mask], "k-", lw=0.8, label="residual PSD")
    ax.axvline(f_nl, color="red", ls="--", lw=1.5,
               label=f"predicted f_NL = {f_nl:.1f} Hz")
    ax.axvline(f_220, color="blue", ls=":", lw=1.5,
               label=f"f_220 = {f_220:.1f} Hz")
    ax.axvline(f_peak, color="green", ls="-", lw=1.5,
               label=f"observed peak = {f_peak:.1f} Hz")

    # Shade the spectral resolution band
    ax.axvspan(f_nl - delta_f / 2, f_nl + delta_f / 2, alpha=0.15, color="red",
               label=f"resolution = {delta_f:.0f} Hz")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (arb. units)")
    ax.set_title(f"Test B: Frequency Precision -- Residual Spectrum after (2,2,0) Subtraction")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "test_b_frequency_precision.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/test_b_frequency_precision.png")


# ===========================================================================
# TEST C: Phase Coherence
# ===========================================================================


def test_c_phase_coherence(ctx):
    """Is the residual phase at 2*f_220 consistent with 2*phi_220?"""
    print("\n" + "=" * 70)
    print("TEST C: Phase Coherence Check")
    print("=" * 70)

    seg_strain, t_dimless, noise_rms = extract_ringdown_segment(ctx)

    # Fit the (2,2,0) mode
    fit = fit_fundamental_mode(seg_strain, t_dimless, ctx["spin"])
    residual = fit["residual"]
    phi_220 = fit["phase"]

    # Predicted nonlinear phase
    phi_nl_predicted = 2.0 * phi_220
    # Wrap to [-pi, pi]
    phi_nl_predicted = (phi_nl_predicted + np.pi) % (2 * np.pi) - np.pi

    # Extract post-ringdown residual
    mask = t_dimless >= 0
    residual_rd = residual[mask]
    t_rd = t_dimless[mask]

    if len(residual_rd) < 8:
        print("  ERROR: ringdown segment too short.")
        return

    sr = ctx["sample_rate"]
    f_nl = ctx["f_nl"]

    # Compute analytic signal via Hilbert transform
    analytic = hilbert(residual_rd)

    # Heterodyne: multiply by exp(-i * 2*pi*f_NL*t) to demodulate at f_NL
    # Convert dimensionless time back to seconds for the heterodyne
    t_seconds = t_rd * ctx["m_seconds"]
    heterodyne = analytic * np.exp(-1j * 2.0 * np.pi * f_nl * t_seconds)

    # The phase of the heterodyned signal at the nonlinear frequency
    # Average over the ringdown to get a stable phase estimate
    # Weight by the expected damping envelope of the nonlinear mode
    omega_nl = ctx["mode_nl"].omega
    envelope = np.exp(omega_nl.imag * t_rd)  # damping envelope
    envelope /= np.sum(envelope) + 1e-30

    # Weighted average of the complex heterodyne
    weighted_mean = np.sum(heterodyne * envelope)
    phi_observed = np.angle(weighted_mean)
    amplitude_at_fnl = np.abs(weighted_mean)

    # Phase offset
    phase_offset = phi_observed - phi_nl_predicted
    phase_offset = (phase_offset + np.pi) % (2 * np.pi) - np.pi

    # Phase uncertainty: for a signal with SNR rho, sigma_phi ~ 1/rho
    # Estimate the SNR at f_NL from the heterodyne amplitude vs noise
    noise_power = np.var(residual_rd)
    snr_at_fnl = amplitude_at_fnl / (np.sqrt(noise_power) + 1e-30)
    sigma_phi = 1.0 / (snr_at_fnl + 1e-10)

    print(f"\n  Fitted phi_220         = {phi_220:.4f} rad ({np.degrees(phi_220):.1f} deg)")
    print(f"  Predicted phi_NL       = 2 * phi_220 = {phi_nl_predicted:.4f} rad "
          f"({np.degrees(phi_nl_predicted):.1f} deg)")
    print(f"  Observed phase at f_NL = {phi_observed:.4f} rad "
          f"({np.degrees(phi_observed):.1f} deg)")
    print(f"\n  Phase offset           = {phase_offset:.4f} rad "
          f"({np.degrees(phase_offset):.1f} deg)")
    print(f"  Phase uncertainty      ~ {sigma_phi:.2f} rad "
          f"({np.degrees(sigma_phi):.1f} deg)")
    print(f"  SNR at f_NL            = {snr_at_fnl:.2f}")

    # Interpretation
    if snr_at_fnl < 1.0:
        print(f"\n  --> SNR at f_NL is below 1. Phase measurement is noise-dominated.")
        print("      Cannot meaningfully constrain phase coherence at this SNR.")
    elif abs(phase_offset) < 2.0 * sigma_phi:
        print(f"\n  --> Phase offset ({np.degrees(phase_offset):.0f} deg) within "
              f"2-sigma ({np.degrees(2*sigma_phi):.0f} deg).")
        print("      CONSISTENT with nonlinear phase locking.")
    else:
        n_sigma = abs(phase_offset) / sigma_phi
        print(f"\n  --> Phase offset is {n_sigma:.1f} sigma from prediction.")
        print("      INCONSISTENT with phase-locked nonlinear mode.")

    # For reference: a random noise fluctuation has uniform phase
    print(f"\n  Note: random noise has uniform phase distribution over [-180, 180] deg.")
    print(f"  A real nonlinear mode would have phase offset ~ 0 +/- {np.degrees(sigma_phi):.0f} deg.")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: phase vs time
    inst_phase = np.angle(heterodyne)
    ax1.plot(t_rd, np.degrees(inst_phase), "k.", ms=2, alpha=0.5,
             label="instantaneous phase at f_NL")
    ax1.axhline(np.degrees(phi_nl_predicted), color="red", ls="--", lw=2,
                label=f"predicted 2*phi_220 = {np.degrees(phi_nl_predicted):.0f} deg")
    ax1.axhline(np.degrees(phi_observed), color="green", ls="-", lw=2,
                label=f"weighted mean = {np.degrees(phi_observed):.0f} deg")
    ax1.set_xlabel("Dimensionless time (t/M)")
    ax1.set_ylabel("Phase at f_NL (deg)")
    ax1.set_title("Phase Evolution at f_NL")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-200, 200)

    # Right: polar plot showing the coherence
    theta_grid = np.linspace(-np.pi, np.pi, 100)
    ax2 = fig.add_subplot(122, projection="polar")
    ax2.set_theta_zero_location("E")
    ax2.scatter([phi_observed], [amplitude_at_fnl], c="green", s=100,
                zorder=5, label="observed")
    ax2.scatter([phi_nl_predicted], [amplitude_at_fnl * 0.8], c="red", s=100,
                marker="x", zorder=5, label="predicted (2*phi_220)")
    # Draw uncertainty arc
    arc_theta = np.linspace(phi_observed - sigma_phi, phi_observed + sigma_phi, 50)
    ax2.plot(arc_theta, [amplitude_at_fnl] * 50, "g-", lw=3, alpha=0.5)
    ax2.set_title("Phase Coherence (polar)", pad=20)
    ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "test_c_phase_coherence.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/test_c_phase_coherence.png")


# ===========================================================================
# TEST D: Adjacent Frequency Bands
# ===========================================================================


def test_d_adjacent_bands(ctx):
    """Does the matched-filter SNR peak sharply at f_NL, or is it broad?"""
    print("\n" + "=" * 70)
    print("TEST D: Adjacent Frequency Band Test")
    print("=" * 70)

    seg_strain, t_dimless, noise_rms = extract_ringdown_segment(ctx)

    # Fit the (2,2,0) mode and get the residual
    fit = fit_fundamental_mode(seg_strain, t_dimless, ctx["spin"])
    residual = fit["residual"]
    a_220 = fit["amplitude"]
    phi_220 = fit["phase"]

    catalog = KerrQNMCatalog()
    nl_mode = catalog.nonlinear_mode_quadratic(ctx["spin"])
    omega_nl = nl_mode.omega

    f_nl = ctx["f_nl"]
    noise_var = noise_rms ** 2

    # Frequency offsets to test (Hz)
    offsets_hz = [-150, -100, -50, -25, 0, 25, 50, 100, 150]
    results = []

    mask = t_dimless >= 0

    for offset in offsets_hz:
        f_test = f_nl + offset
        # Convert to dimensionless angular frequency
        omega_test_real = 2.0 * np.pi * f_test * ctx["m_seconds"]
        # Keep the same damping rate as the nonlinear mode
        omega_test = complex(omega_test_real, omega_nl.imag)

        # Build template at the test frequency
        template = np.zeros_like(t_dimless)
        a_nl = a_220 ** 2  # kappa=1 amplitude
        phi_nl = 2.0 * phi_220
        template[mask] = (
            a_nl
            * np.exp(omega_test.imag * t_dimless[mask])
            * np.cos(omega_test.real * t_dimless[mask] + phi_nl)
        )

        # Matched-filter SNR
        inner_rt = np.sum(residual[mask] * template[mask]) / noise_var
        inner_tt = np.sum(template[mask] * template[mask]) / noise_var
        template_norm = np.sqrt(inner_tt)

        if template_norm > 0:
            snr = inner_rt / template_norm
            kappa = inner_rt / inner_tt
        else:
            snr = 0.0
            kappa = 0.0

        results.append({
            "offset_hz": offset,
            "f_test": f_test,
            "snr": snr,
            "kappa": kappa,
        })

    # Print results
    print(f"\n  {'Offset (Hz)':>12}  {'f_test (Hz)':>12}  {'SNR':>8}  {'kappa':>8}")
    print("  " + "-" * 48)
    for r in results:
        marker = " <-- f_NL" if r["offset_hz"] == 0 else ""
        print(f"  {r['offset_hz']:>+12.0f}  {r['f_test']:>12.1f}  "
              f"{r['snr']:>8.2f}  {r['kappa']:>8.2f}{marker}")

    # Analysis: is the peak at f_NL?
    snrs = np.array([r["snr"] for r in results])
    offsets = np.array([r["offset_hz"] for r in results])
    peak_idx = np.argmax(np.abs(snrs))
    peak_offset = offsets[peak_idx]
    snr_at_fnl = snrs[offsets == 0][0]
    snr_at_peak = snrs[peak_idx]

    print(f"\n  SNR at f_NL (offset=0): {snr_at_fnl:.2f}")
    print(f"  Peak |SNR| at offset {peak_offset:+.0f} Hz: {snr_at_peak:.2f}")

    # Check if f_NL is the clear peak
    off_snrs = np.abs(snrs[offsets != 0])
    if abs(snr_at_fnl) > 1.5 * np.max(off_snrs):
        print("\n  --> f_NL has the dominant excess (>1.5x any adjacent band).")
        print("      CONSISTENT with a narrowband signal at the predicted frequency.")
    elif abs(snr_at_fnl) > np.max(off_snrs):
        print("\n  --> f_NL has the highest SNR but not dramatically so.")
        print("      Marginal -- could be signal or broad fluctuation.")
    else:
        print(f"\n  --> Adjacent band at {peak_offset:+.0f} Hz has higher |SNR|.")
        print("      INCONSISTENT with a narrowband nonlinear mode.")
        print("      Suggests broadband noise or a glitch.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if o == 0 else "steelblue" for o in offsets]
    ax.bar(offsets, np.abs(snrs), width=20, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Frequency offset from f_NL (Hz)")
    ax.set_ylabel("|Matched-filter SNR|")
    ax.set_title(f"Test D: Adjacent Frequency Bands -- {EVENT_NAME} {DETECTOR}")
    ax.annotate("f_NL", xy=(0, abs(snr_at_fnl)), ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="green")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "test_d_adjacent_bands.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/test_d_adjacent_bands.png")


# ===========================================================================
# TEST E: Null Test -- Time-Shifted Data
# ===========================================================================


def test_e_time_shift_null(ctx):
    """Does the excess persist when we shift away from the ringdown?"""
    print("\n" + "=" * 70)
    print("TEST E: Null Test -- Time-Shifted Data")
    print("=" * 70)

    m_seconds = ctx["m_seconds"]
    spin = ctx["spin"]
    sr = ctx["sample_rate"]
    merger_gps = ctx["merger_gps"]

    # Shifts: the real ringdown, plus offsets of +/- 0.5s, +/- 1.0s
    shifts_s = [-1.0, -0.5, 0.0, +0.5, +1.0]
    results = []

    for shift in shifts_s:
        # The "ringdown start" is shifted
        fake_ringdown_gps = ctx["ringdown_start_gps"] + shift

        # Extract segment around this fake ringdown
        pad_before = 0.05
        duration = 0.15
        t0 = fake_ringdown_gps - pad_before
        t1 = fake_ringdown_gps + duration
        mask = (ctx["time"] >= t0) & (ctx["time"] <= t1)

        if np.sum(mask) < 32:
            print(f"  Shift {shift:+.1f}s: insufficient data, skipping.")
            results.append({
                "shift_s": shift,
                "snr": np.nan,
                "kappa": np.nan,
                "is_null": shift != 0.0,
            })
            continue

        seg_strain = ctx["strain"][mask]
        seg_time = ctx["time"][mask]
        t_dimless = (seg_time - fake_ringdown_gps) / m_seconds

        # Noise estimate from off-source (always relative to the real merger)
        noise_mask = np.abs(ctx["time"] - merger_gps) > 4.0
        noise_rms = np.sqrt(np.var(ctx["strain"][noise_mask]))

        # Run the phase-locked search
        result = phase_locked_search(
            seg_strain, t_dimless, spin, noise_rms,
            event_name=f"{EVENT_NAME}_shift{shift:+.1f}s",
        )

        results.append({
            "shift_s": shift,
            "snr": result.snr,
            "kappa": result.kappa_hat,
            "kappa_sigma": result.kappa_sigma,
            "is_null": shift != 0.0,
        })

    # Print results
    print(f"\n  {'Shift (s)':>10}  {'SNR':>8}  {'kappa':>8}  {'sigma':>8}  {'Type':>10}")
    print("  " + "-" * 52)
    for r in results:
        label = "SIGNAL" if not r["is_null"] else "NULL"
        snr_str = f"{r['snr']:.2f}" if np.isfinite(r.get("snr", np.nan)) else "N/A"
        kappa_str = f"{r['kappa']:.2f}" if np.isfinite(r.get("kappa", np.nan)) else "N/A"
        sigma_str = f"{r.get('kappa_sigma', np.nan):.2f}" if np.isfinite(r.get("kappa_sigma", np.nan)) else "N/A"
        print(f"  {r['shift_s']:>+10.1f}  {snr_str:>8}  {kappa_str:>8}  "
              f"{sigma_str:>8}  {label:>10}")

    # Analysis
    signal_result = [r for r in results if not r["is_null"] and np.isfinite(r.get("snr", np.nan))]
    null_results = [r for r in results if r["is_null"] and np.isfinite(r.get("snr", np.nan))]

    if signal_result and null_results:
        snr_signal = abs(signal_result[0]["snr"])
        snr_nulls = [abs(r["snr"]) for r in null_results]
        max_null_snr = max(snr_nulls)
        mean_null_snr = np.mean(snr_nulls)

        print(f"\n  Signal SNR (shift=0):    {snr_signal:.2f}")
        print(f"  Max null SNR:            {max_null_snr:.2f}")
        print(f"  Mean null |SNR|:         {mean_null_snr:.2f}")

        if snr_signal > 2.0 * max_null_snr:
            print("\n  --> Signal SNR is >2x any null. Excess is event-associated.")
        elif snr_signal > max_null_snr:
            print("\n  --> Signal SNR exceeds nulls, but not by a large margin.")
            print("      Marginal evidence for event association.")
        else:
            print("\n  --> Null segments show comparable or higher SNR.")
            print("      The excess is likely a NOISE ARTIFACT, not event-associated.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = [r for r in results if np.isfinite(r.get("snr", np.nan))]
    shifts = [r["shift_s"] for r in valid]
    snrs = [abs(r["snr"]) for r in valid]
    colors = ["green" if not r["is_null"] else "gray" for r in valid]

    ax.bar(shifts, snrs, width=0.3, color=colors, edgecolor="black", lw=0.5)
    ax.set_xlabel("Time shift from true ringdown (seconds)")
    ax.set_ylabel("|Matched-filter SNR|")
    ax.set_title(f"Test E: Null Test -- Time Shifts -- {EVENT_NAME} {DETECTOR}")
    ax.axhline(2.0, color="red", ls=":", lw=1, label="2-sigma")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "test_e_time_shift_null.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/test_e_time_shift_null.png")


# ===========================================================================
# Main
# ===========================================================================


def main():
    print("=" * 70)
    print("GRIM-S Excess Diagnostics: Stress-Testing the 2.05-sigma Candidate")
    print(f"Event: {EVENT_NAME}  Detector: {DETECTOR}")
    print("'Where is the instrument?' -- Ralph Bown")
    print("=" * 70)
    print()

    print("Loading and whitening GW150914 H1 strain...")
    ctx = load_full_whitened_strain()
    print(f"  f_220 = {ctx['f_220']:.1f} Hz")
    print(f"  f_NL  = {ctx['f_nl']:.1f} Hz  (predicted 2*f_220)")
    print(f"  tau_220 = {ctx['tau_220_s']*1e3:.2f} ms")
    print(f"  Sample rate = {ctx['sample_rate']:.0f} Hz")
    print()

    test_a_time_frequency_map(ctx)
    test_b_frequency_precision(ctx)
    test_c_phase_coherence(ctx)
    test_d_adjacent_bands(ctx)
    test_e_time_shift_null(ctx)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  Test A (Time-Frequency): Does the 541 Hz excess appear only during ringdown?
  Test B (Freq Precision): Is the peak at exactly 2*f_220 within resolution?
  Test C (Phase Coherence): Is the residual phase consistent with 2*phi_220?
  Test D (Adjacent Bands): Is the excess narrowband at f_NL?
  Test E (Null / Time Shift): Does the excess vanish in off-source data?

  A REAL nonlinear mode passes ALL five tests.
  A noise fluctuation typically fails Tests A, D, and E.
  An instrument line fails Tests A and E (persistent, not event-associated).

  Review the plots in plots/ for visual confirmation.
  Review the printed numbers above for quantitative verdicts.
""")
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
