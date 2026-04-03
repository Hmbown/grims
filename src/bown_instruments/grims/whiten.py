"""
Strain whitening and bandpass filtering for real GWOSC data.

LIGO noise is strongly colored — orders of magnitude louder at low
frequencies (seismic wall below ~20 Hz) and at high frequencies
(shot noise above ~1 kHz). A raw time-domain fit picks up noise power
across all bands, drowning out the ringdown signal.

Whitening divides the strain by the noise amplitude spectral density
(ASD) in the frequency domain, flattening the noise floor so that
every frequency bin contributes equally. After whitening, the strain
is in units of "sigma per frequency bin" — a matched filter in white
noise is optimal.

Bandpassing further restricts to the frequency range where the QNM
modes live, removing out-of-band noise that contributes nothing to
the measurement.

Bown's principle: you cannot measure a signal you cannot see.
Whitening is the instrument that makes the signal visible.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def estimate_asd(
    strain: np.ndarray,
    sample_rate: float,
    merger_time: float,
    time: np.ndarray,
    exclusion_window: float = 1.0,
    nperseg: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the noise ASD from off-source strain.

    Uses data far from the merger to characterize the detector noise.

    Parameters
    ----------
    strain : full strain time series
    sample_rate : Hz
    merger_time : GPS time of merger
    time : GPS time array corresponding to strain
    exclusion_window : seconds around merger to exclude
    nperseg : Welch segment length (0 = auto)

    Returns
    -------
    freqs : frequency array (Hz)
    asd : amplitude spectral density (strain / sqrt(Hz))
    """
    # Use data far from merger for noise estimation
    offsource = np.abs(time - merger_time) > exclusion_window
    noise_strain = strain[offsource]

    if len(noise_strain) < 256:
        raise ValueError(
            f"Only {len(noise_strain)} off-source samples — need at least 256"
        )

    if nperseg <= 0:
        nperseg = min(int(sample_rate), len(noise_strain) // 2)

    freqs, psd = welch(
        noise_strain, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2
    )
    asd = np.sqrt(psd)

    return freqs, asd


def whiten_strain(
    strain: np.ndarray,
    sample_rate: float,
    asd_freqs: np.ndarray,
    asd: np.ndarray,
    fmin: float = 20.0,
) -> np.ndarray:
    """Whiten strain by dividing by the noise ASD in frequency domain.

    Parameters
    ----------
    strain : time-domain strain
    sample_rate : Hz
    asd_freqs : frequency array from ASD estimation
    asd : amplitude spectral density
    fmin : minimum frequency (below this, ASD is set to infinity to
           suppress seismic noise)

    Returns
    -------
    whitened : whitened strain in time domain (dimensionless, ~unit variance
               per frequency bin if noise-dominated)
    """
    n = len(strain)
    dt = 1.0 / sample_rate

    # FFT
    strain_fft = np.fft.rfft(strain)
    freqs_fft = np.fft.rfftfreq(n, d=dt)

    # Interpolate ASD onto FFT frequency grid
    asd_interp = np.interp(freqs_fft, asd_freqs, asd)

    # Suppress below fmin (seismic wall)
    asd_interp[freqs_fft < fmin] = np.inf

    # Avoid division by zero
    asd_interp[asd_interp <= 0] = np.inf

    # Whiten: divide FFT by ASD, then normalize
    whitened_fft = strain_fft / asd_interp

    # Inverse FFT
    whitened = np.fft.irfft(whitened_fft, n=n)

    # Normalize so that pure noise has unit variance
    # The normalization factor is sqrt(2 / (sample_rate * n))
    whitened *= np.sqrt(2.0 * sample_rate / n)

    return whitened


def bandpass(
    strain: np.ndarray, sample_rate: float, fmin: float, fmax: float, order: int = 4
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Parameters
    ----------
    strain : time-domain strain (whitened or raw)
    sample_rate : Hz
    fmin : low cutoff (Hz)
    fmax : high cutoff (Hz)
    order : filter order

    Returns
    -------
    filtered : bandpassed strain
    """
    nyquist = 0.5 * sample_rate
    low = fmin / nyquist
    high = min(fmax / nyquist, 0.99)  # stay below Nyquist

    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, strain)


def prepare_ringdown_for_analysis(
    event_name: str,
    data_dir: str = "data/",
    detector: str | None = None,
    fmin_pad: float = 0.5,
    fmax_pad: float = 1.3,
    t_start_m: float = 10.0,
) -> dict:
    """Full whitening + bandpass + ringdown extraction pipeline.

    Loads the full 32-second GWOSC file, estimates the noise ASD from
    off-source data, whitens and bandpasses the full strain, then
    extracts the ringdown segment. This ensures the noise estimation
    has enough data (the short ringdown segment alone is too small).

    Parameters
    ----------
    event_name : GWTC event name (e.g. 'GW150914')
    data_dir : where GWOSC HDF5 files are cached
    detector : detector name (None = use first available)
    fmin_pad : multiply lowest QNM frequency by this for low cutoff
    fmax_pad : multiply highest QNM frequency (nonlinear) by this for high cutoff

    t_start_m : ringdown start time in units of M after merger (default 10.0)

    Returns
    -------
    dict with whitened strain, dimensionless time, noise variance, etc.
    """
    from .qnm_modes import KerrQNMCatalog
    from .gwtc_pipeline import (
        M_SUN_SECONDS,
        get_candidate_event,
        download_gwosc_strain,
        load_gwosc_strain_hdf5,
    )

    event = get_candidate_event(event_name)
    mass = event["remnant_mass_msun"]
    spin = event["remnant_spin"]
    m_seconds = mass * M_SUN_SECONDS
    det = detector or event["detectors"][0]

    # Compute QNM frequencies in Hz
    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)
    mode_440 = catalog.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)

    # Bandpass range: from below the fundamental to above the nonlinear mode
    f_low = max(20.0, f_220 * fmin_pad)

    # Load the FULL 32-second file (not just the ringdown segment).
    # Try local cache first; fall back to download if needed.
    from pathlib import Path

    data_path = Path(data_dir)
    local_candidates = sorted(
        data_path.glob(f"*{det}*{event_name.replace('_', '')}*4KHZ*.hdf5")
    )
    if not local_candidates:
        # Try matching by GPS time prefix
        gps_prefix = str(int(event["gps_time"]))[:4]
        local_candidates = sorted(data_path.glob(f"*{det[0]}*{gps_prefix}*.hdf5"))
    if not local_candidates:
        # Match any HDF5 that contains the GPS start time
        gps_int = int(event["gps_time"])
        for f in sorted(data_path.glob("*.hdf5")):
            # GWOSC filenames contain the GPS start: H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5
            parts = f.stem.split("-")
            if len(parts) >= 3:
                try:
                    file_gps = int(parts[-2])
                    file_dur = int(parts[-1])
                    if file_gps <= gps_int <= file_gps + file_dur:
                        local_candidates.append(f)
                except ValueError:
                    continue

    if local_candidates:
        local_path = str(local_candidates[0])
        actual_det = det
    else:
        local_path, actual_det = download_gwosc_strain(
            event_name,
            detector=det,
            data_dir=data_dir,
        )

    loaded = load_gwosc_strain_hdf5(local_path)
    full_strain = loaded["strain"]
    full_time = loaded["time"]
    sample_rate = loaded["sample_rate"]

    f_high = min(0.45 * sample_rate, max(f_nl, f_440) * fmax_pad)

    merger_time = float(event["gps_time"])
    ringdown_start = merger_time + t_start_m * m_seconds

    # Estimate noise ASD from the full file, excluding 2s around merger
    asd_freqs, asd = estimate_asd(
        full_strain,
        sample_rate,
        merger_time=merger_time,
        time=full_time,
        exclusion_window=2.0,
    )

    # Whiten the full strain
    whitened = whiten_strain(
        full_strain,
        sample_rate,
        asd_freqs,
        asd,
        fmin=f_low * 0.8,
    )

    # Bandpass
    whitened_bp = bandpass(whitened, sample_rate, f_low, f_high)

    # Extract the ringdown segment (with some padding before)
    pad_before = 0.05  # 50ms before ringdown start
    seg_duration = 0.15  # 150ms total
    t_start = ringdown_start - pad_before
    t_end = ringdown_start + seg_duration
    mask = (full_time >= t_start) & (full_time <= t_end)

    seg_strain = whitened_bp[mask]
    seg_time = full_time[mask]

    # Dimensionless time
    t_dimless = (seg_time - ringdown_start) / m_seconds

    # Noise variance from off-source whitened data (far from merger)
    noise_mask = np.abs(full_time - merger_time) > 4.0
    noise_var = np.var(whitened_bp[noise_mask])

    return {
        "strain_whitened": seg_strain,
        "t_dimless": t_dimless,
        "noise_variance": noise_var,
        "noise_rms": np.sqrt(noise_var),
        "f_band": (f_low, f_high),
        "qnm_freqs_hz": {
            "f_220": f_220,
            "f_nl": f_nl,
            "f_440": f_440,
        },
        "mass_seconds": m_seconds,
        "sample_rate": sample_rate,
        "detector": actual_det,
        "event": event,
    }


def scan_start_time(
    event_name: str,
    data_dir: str = "data/",
    detector: str | None = None,
    t_start_values: list = None,
) -> dict:
    """Scan ringdown start time and measure kappa at each value.

    This tests whether the kappa measurement is robust to the choice
    of when the ringdown begins. If kappa is stable across start times,
    the measurement is robust. If it varies, the result is not trustworthy.

    Parameters
    ----------
    event_name : GWTC event name
    data_dir : where GWOSC HDF5 files are cached
    detector : detector name (None = use first available)
    t_start_values : list of start times in M after merger
                     (default: [5, 8, 10, 12, 15, 20])

    Returns
    -------
    dict with start times, posteriors, and summary statistics
    """
    from .bayesian_analysis import (
        estimate_kappa_posterior_from_data,
        PosteriorResult,
    )
    from .gwtc_pipeline import get_candidate_event

    if t_start_values is None:
        t_start_values = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]

    event = get_candidate_event(event_name)
    spin = event["remnant_spin"]

    results = []
    for t_m in t_start_values:
        prep = prepare_ringdown_for_analysis(
            event_name,
            data_dir=data_dir,
            detector=detector,
            t_start_m=t_m,
        )
        result = estimate_kappa_posterior_from_data(
            prep["strain_whitened"],
            prep["t_dimless"],
            spin=spin,
            noise_variance=prep["noise_variance"],
            event_name=event_name,
        )
        results.append(
            {
                "t_start_m": t_m,
                "posterior": result,
                "a_220": result.linear_mode_estimates.get("220", {}).get(
                    "amplitude", 0
                ),
                "noise_rms": prep["noise_rms"],
                "n_samples": len(prep["strain_whitened"]),
            }
        )

    return {
        "event_name": event_name,
        "spin": spin,
        "t_start_values": t_start_values,
        "results": results,
    }
