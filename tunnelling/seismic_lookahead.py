"""
Seismic lookahead for tunnel boring machines.

Adapted from the GRIM-S gravitational-wave ringdown detection pipeline.
Core idea: the same signal processing that detects weak oscillation modes
in LIGO noise can detect weak seismic reflections in TBM vibration noise.

The pipeline:
  1. Characterize TBM noise from a recent off-source window
  2. Matched-filter against the known source wavelet, swept across travel times
     (frequency-domain, PSD-weighted — no separate whitening step needed)
  3. Combine across geophones with inverse-variance weighting
  4. Report detection confidence calibrated against the noise distribution

The matched filter operates directly in the frequency domain with PSD
weighting, which is equivalent to whitening both data and template but
avoids spectral leakage from short wavelets.  A separate whiten + bandpass
path is provided for visualisation and diagnostics.

Adapted from whiten.py, phase_locked_search.py, and injection_campaign.py
in bown_instruments.grims.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.special import erfc
from scipy.stats import norm as normal_dist


# ---------------------------------------------------------------------------
# Noise characterisation
# ---------------------------------------------------------------------------

def estimate_noise_psd(
    trace: np.ndarray,
    sample_rate: float,
    nperseg: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the TBM vibration noise power spectral density.

    Record geophone data while the TBM is running but no active seismic
    source is fired.  This gives a pure-noise PSD that characterises the
    machine vibration environment.

    Parameters
    ----------
    trace : 1-D array of geophone voltage / velocity samples (noise only).
    sample_rate : Sample rate in Hz.
    nperseg : Welch segment length.  0 = auto (1 second or half the trace).

    Returns
    -------
    freqs : Frequency array (Hz).
    psd : Power spectral density (units^2 / Hz).
    """
    trace = np.asarray(trace, dtype=float)
    if len(trace) < 256:
        raise ValueError(f"Need >= 256 samples for PSD, got {len(trace)}")

    if nperseg <= 0:
        nperseg = min(int(sample_rate), len(trace) // 2)

    freqs, psd = welch(trace, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    psd = np.where(np.isfinite(psd) & (psd > 0), psd, np.inf)
    return freqs, psd


# ---------------------------------------------------------------------------
# Whitening and bandpass (for visualisation / diagnostics)
# ---------------------------------------------------------------------------

def whiten_seismic(
    trace: np.ndarray,
    sample_rate: float,
    psd_freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float = 20.0,
) -> np.ndarray:
    """Whiten a geophone trace by dividing by the noise ASD in frequency domain.

    After whitening, every frequency bin contributes equal noise power.
    This is useful for visualisation and diagnostics.  The main detection
    path (matched_filter_reflection) does PSD weighting internally and
    does not require a separate whitening step.

    Parameters
    ----------
    trace : Time-domain geophone recording (contains signal + noise).
    sample_rate : Hz.
    psd_freqs : Frequency array from estimate_noise_psd.
    psd : Power spectral density from estimate_noise_psd.
    fmin : Suppress frequencies below this (Hz).  Default 20 Hz
           suppresses TBM low-frequency rumble and ground roll.

    Returns
    -------
    whitened : Whitened trace, approximately unit variance if noise-dominated.
    """
    n = len(trace)
    dt = 1.0 / sample_rate

    trace_fft = np.fft.rfft(trace)
    freqs_fft = np.fft.rfftfreq(n, d=dt)

    # Interpolate ASD (sqrt of PSD) onto FFT grid
    asd_interp = np.sqrt(np.interp(freqs_fft, psd_freqs, psd))

    # Suppress below fmin and clamp non-finite / zero values
    asd_interp[freqs_fft < fmin] = np.inf
    asd_interp[~(np.isfinite(asd_interp) & (asd_interp > 0))] = np.inf

    # Whiten: divide FFT by ASD
    whitened_fft = np.divide(
        trace_fft, asd_interp,
        out=np.zeros_like(trace_fft),
        where=np.isfinite(asd_interp),
    )

    whitened = np.fft.irfft(whitened_fft, n=n)
    whitened *= np.sqrt(2.0 * sample_rate / n)
    return whitened


def bandpass(
    trace: np.ndarray,
    sample_rate: float,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass filter.

    Restricts the trace to the frequency range where seismic reflections
    are expected, removing out-of-band noise.
    """
    nyquist = 0.5 * sample_rate
    low = fmin / nyquist
    high = min(fmax / nyquist, 0.99)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, trace)


# ---------------------------------------------------------------------------
# Matched-filter detection
# ---------------------------------------------------------------------------

def matched_filter_reflection(
    trace: np.ndarray,
    wavelet: np.ndarray,
    sample_rate: float,
    psd_freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float = 20.0,
    fmax: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-domain matched filter: SNR vs two-way travel time.

    Computes the optimal detection statistic by correlating data with the
    source wavelet, weighted by the inverse noise PSD:

        SNR(tau) = Re[ IFFT( d~(f) * h~*(f) / S_n(f) ) ] / sigma_h

    where sigma_h = sqrt( 4 * sum |h~(f)|^2 / S_n(f) * df ).

    The returned SNR is raw (uncalibrated).  To obtain properly normalised
    SNR where noise has unit variance, divide by the standard deviation
    measured on a noise-only segment.  This empirical calibration is
    standard practice in both GW and seismic matched filtering — it avoids
    subtle issues with discrete FFT normalisation conventions.

    Parameters
    ----------
    trace : Raw (un-whitened) geophone recording.
    wavelet : Source wavelet (raw, un-whitened).
    sample_rate : Hz.
    psd_freqs : Frequency array from estimate_noise_psd.
    psd : Noise PSD from estimate_noise_psd.
    fmin : Low-frequency cutoff (Hz).  Default 20 Hz.
    fmax : High-frequency cutoff (Hz).  Default 500 Hz.

    Returns
    -------
    travel_times : Two-way travel time array (seconds).
    snr_raw : Raw matched-filter output (divide by noise sigma to calibrate).
    """
    n = len(trace)
    dt = 1.0 / sample_rate

    # FFT data and zero-padded template
    data_f = np.fft.rfft(trace, n=n)
    wav_f = np.fft.rfft(wavelet, n=n)
    freqs = np.fft.rfftfreq(n, d=dt)

    # Interpolate PSD onto FFT grid; suppress out-of-band
    psd_interp = np.interp(freqs, psd_freqs, psd)
    psd_interp[(freqs < fmin) | (freqs > fmax)] = np.inf
    psd_interp[psd_interp <= 0] = np.inf

    inv_psd = np.zeros_like(psd_interp)
    ok = np.isfinite(psd_interp)
    inv_psd[ok] = 1.0 / psd_interp[ok]

    # Template norm
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    sigma_h_sq = 4.0 * np.sum(np.abs(wav_f) ** 2 * inv_psd) * df
    if sigma_h_sq <= 0:
        raise ValueError("Template has zero power in the analysis band")
    sigma_h = np.sqrt(sigma_h_sq)

    # Filtered correlation
    filtered = np.fft.irfft(data_f * np.conj(wav_f) * inv_psd, n=n)
    snr_raw = filtered / sigma_h

    travel_times = np.arange(n) / sample_rate
    return travel_times, snr_raw


def scan_travel_times(
    trace: np.ndarray,
    wavelet: np.ndarray,
    sample_rate: float,
    psd_freqs: np.ndarray,
    psd: np.ndarray,
    velocity: float,
    fmin: float = 20.0,
    fmax: float = 500.0,
    max_distance: float | None = None,
    noise_sigma: float | None = None,
    peak_threshold: float = 3.0,
) -> dict:
    """Build a reflectivity-vs-distance profile ahead of the tunnel face.

    Runs the frequency-domain matched filter and converts two-way travel
    time to one-way distance using the assumed P-wave velocity.

    Parameters
    ----------
    trace : Raw geophone recording (un-whitened).
    wavelet : Raw source wavelet (un-whitened).
    sample_rate : Hz.
    psd_freqs : Frequency array from estimate_noise_psd.
    psd : Noise PSD from estimate_noise_psd.
    velocity : Assumed P-wave velocity in m/s (e.g. 3000-5000 for rock).
    fmin : Low-frequency cutoff (Hz).  Default 20 Hz.
    fmax : High-frequency cutoff (Hz).  Default 500 Hz.
    max_distance : Maximum lookahead distance in metres (None = no limit).
    noise_sigma : Std of matched-filter output on noise-only data.  If
                  provided, SNR is normalised so noise has unit variance.
    peak_threshold : Minimum |SNR| for a peak to be reported (default 3.0).

    Returns
    -------
    dict with keys: distances, travel_times, snr, peak_distances, peak_snrs.
    """
    travel_times, snr = matched_filter_reflection(
        trace, wavelet, sample_rate, psd_freqs, psd, fmin=fmin, fmax=fmax,
    )

    if noise_sigma is not None and noise_sigma > 0:
        snr = snr / noise_sigma

    distances = travel_times * velocity / 2.0  # two-way -> one-way

    if max_distance is not None:
        mask = distances <= max_distance
        distances = distances[mask]
        travel_times = travel_times[mask]
        snr = snr[mask]

    # Find peaks above threshold
    peaks, _ = find_peaks(
        np.abs(snr), height=peak_threshold, distance=int(0.001 * sample_rate),
    )

    return {
        "distances": distances,
        "travel_times": travel_times,
        "snr": snr,
        "peak_distances": distances[peaks] if len(peaks) > 0 else np.array([]),
        "peak_snrs": snr[peaks] if len(peaks) > 0 else np.array([]),
    }


# ---------------------------------------------------------------------------
# Multi-channel stacking
# ---------------------------------------------------------------------------

@dataclass
class GeophoneResult:
    """Detection result from a single geophone channel."""
    channel_id: str
    distances: np.ndarray
    snr: np.ndarray
    noise_rms: float
    weight: float = 0.0


def multi_geophone_stack(
    results: list[GeophoneResult],
    max_weight_ratio: float = 5.0,
) -> dict:
    """Inverse-variance weighted stack across geophone channels.

    Channels with lower noise get more weight.  A maximum weight ratio
    prevents any single channel from dominating (protects against a
    miscalibrated channel or poor coupling).

    Parameters
    ----------
    results : Per-channel results (must share the same distance grid).
    max_weight_ratio : Cap on weight_max / weight_min.

    Returns
    -------
    dict with keys: distances, snr_stacked, n_channels, channel_weights.
    """
    if not results:
        raise ValueError("No geophone results to stack")

    for r in results:
        r.weight = 1.0 / (r.noise_rms ** 2) if r.noise_rms > 0 else 0.0

    weights = np.array([r.weight for r in results])
    if np.any(weights > 0):
        w_min = np.min(weights[weights > 0])
        cap = w_min * max_weight_ratio
        weights = np.minimum(weights, cap)

    total_weight = np.sum(weights)
    if total_weight == 0:
        raise ValueError("All channels have zero weight")

    norm_weights = weights / total_weight

    ref_distances = results[0].distances
    stacked_snr = np.zeros_like(ref_distances, dtype=float)
    for r, w in zip(results, norm_weights):
        n = min(len(r.snr), len(stacked_snr))
        stacked_snr[:n] += w * r.snr[:n]

    stacked_snr *= np.sqrt(total_weight)

    return {
        "distances": ref_distances,
        "snr_stacked": stacked_snr,
        "n_channels": len(results),
        "channel_weights": norm_weights,
    }


# ---------------------------------------------------------------------------
# Detection confidence (false alarm probability)
# ---------------------------------------------------------------------------

def detection_confidence(
    snr: float,
    n_trials: int = 1,
) -> dict:
    """Convert matched-filter SNR to detection confidence.

    Under H0 (no reflection), the calibrated matched-filter output is
    Gaussian(0, 1).  The false alarm probability for a single trial is:

        p_fa = erfc(|SNR| / sqrt(2)) / 2

    With multiple independent trials (travel time bins), apply a
    trials factor:

        p_fa_corrected = 1 - (1 - p_fa)^n_trials

    Parameters
    ----------
    snr : Matched-filter SNR value (calibrated, i.e. after noise_sigma division).
    n_trials : Number of independent travel-time bins searched.

    Returns
    -------
    dict with keys: snr, p_false_alarm, p_false_alarm_corrected,
                    sigma_equivalent, confidence_pct.
    """
    abs_snr = np.abs(snr)
    p_fa = 0.5 * erfc(abs_snr / np.sqrt(2.0))
    p_fa_corrected = 1.0 - (1.0 - p_fa) ** n_trials
    sigma_eq = normal_dist.isf(p_fa_corrected / 2.0) if p_fa_corrected < 1.0 else 0.0

    return {
        "snr": float(snr),
        "p_false_alarm": float(p_fa),
        "p_false_alarm_corrected": float(p_fa_corrected),
        "sigma_equivalent": float(sigma_eq),
        "confidence_pct": float(100.0 * (1.0 - p_fa_corrected)),
    }


# ---------------------------------------------------------------------------
# Injection calibration
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    """Result of one synthetic injection test."""
    true_distance: float
    true_amplitude: float
    recovered_distance: float
    recovered_snr: float
    distance_error: float
    detected: bool


def run_injection_test(
    noise_trace: np.ndarray,
    wavelet: np.ndarray,
    sample_rate: float,
    psd_freqs: np.ndarray,
    psd: np.ndarray,
    velocity: float,
    injection_distances: list[float],
    injection_amplitudes: list[float],
    fmin: float = 20.0,
    fmax: float = 500.0,
    detection_threshold: float = 4.0,
) -> list[InjectionResult]:
    """Inject synthetic reflections into real TBM noise and measure recovery.

    Takes a noise-only recording, adds reflected wavelets at known distances
    and amplitudes, runs the detection pipeline, and checks recovery.

    Parameters
    ----------
    noise_trace : Noise-only geophone recording (TBM running, no source).
    wavelet : Source wavelet (raw, not whitened).
    sample_rate : Hz.
    psd_freqs, psd : Noise PSD from estimate_noise_psd.
    velocity : P-wave velocity (m/s).
    injection_distances : Distances ahead of face to inject reflections (m).
    injection_amplitudes : Reflected amplitude of each injection.
    fmin, fmax : Analysis band limits (Hz).
    detection_threshold : SNR threshold for claiming a detection.

    Returns
    -------
    List of InjectionResult, one per injection.
    """
    # Calibrate noise level from the raw noise trace
    _, snr_null = matched_filter_reflection(
        noise_trace, wavelet, sample_rate, psd_freqs, psd, fmin=fmin, fmax=fmax,
    )
    noise_sigma = max(np.std(snr_null), 1e-30)

    results = []
    max_inj_dist = max(injection_distances)

    for dist, amp in zip(injection_distances, injection_amplitudes):
        injected = noise_trace.copy()
        travel_time = 2.0 * dist / velocity
        sample_offset = int(travel_time * sample_rate)

        if sample_offset + len(wavelet) > len(injected):
            continue

        injected[sample_offset:sample_offset + len(wavelet)] += amp * wavelet

        scan = scan_travel_times(
            injected, wavelet, sample_rate, psd_freqs, psd, velocity,
            fmin=fmin, fmax=fmax,
            max_distance=max_inj_dist * 1.5,
            noise_sigma=noise_sigma,
        )

        detected = False
        recovered_dist = 0.0
        recovered_snr = 0.0

        if len(scan["peak_distances"]) > 0:
            diffs = np.abs(scan["peak_distances"] - dist)
            closest_idx = np.argmin(diffs)
            tolerance = velocity / (2.0 * sample_rate) * 5  # 5 samples
            if diffs[closest_idx] < tolerance:
                recovered_dist = scan["peak_distances"][closest_idx]
                recovered_snr = scan["peak_snrs"][closest_idx]
                detected = np.abs(recovered_snr) >= detection_threshold

        results.append(InjectionResult(
            true_distance=dist,
            true_amplitude=amp,
            recovered_distance=recovered_dist,
            recovered_snr=recovered_snr,
            distance_error=recovered_dist - dist,
            detected=detected,
        ))

    return results
