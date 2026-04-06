#!/usr/bin/env python3
"""
Synthetic demonstration: seismic lookahead for tunnel boring.

Generates realistic TBM noise, injects reflections from geological
boundaries at known distances, and runs the GRIM-S-derived detection
pipeline.

Run:
    python synthetic_demo.py

Produces: tunnelling_demo.png (four-panel figure)
No external data required — everything is synthetic.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic_lookahead import (
    estimate_noise_psd,
    whiten_seismic,
    bandpass,
    matched_filter_reflection,
    scan_travel_times,
    detection_confidence,
    run_injection_test,
)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE = 2000.0       # Hz (typical for tunnel seismic)
DURATION = 0.5             # seconds of recording
VELOCITY = 4000.0          # m/s P-wave velocity (competent rock)
RNG = np.random.default_rng(42)

# Source amplitude: represents a hydraulic hammer or small charge.
# Reflected amplitude = SOURCE_AMPLITUDE * impedance_contrast.
SOURCE_AMPLITUDE = 100.0

# Three geological boundaries ahead of the face
BOUNDARIES = [
    {"distance_m": 15.0, "impedance_contrast": 0.08, "label": "Clay lens (8%)"},
    {"distance_m": 35.0, "impedance_contrast": 0.20, "label": "Fault zone (20%)"},
    {"distance_m": 52.0, "impedance_contrast": 0.05, "label": "Weathering front (5%)"},
]

# Analysis band
FMIN, FMAX = 20.0, 500.0


def make_ricker_wavelet(
    f_peak: float = 150.0,
    sample_rate: float = SAMPLE_RATE,
    duration: float = 0.015,
) -> np.ndarray:
    """Ricker (Mexican hat) wavelet — common seismic source pulse."""
    t = np.arange(-duration / 2, duration / 2, 1.0 / sample_rate)
    u = (np.pi * f_peak * t) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)


def make_tbm_noise(
    n_samples: int,
    sample_rate: float = SAMPLE_RATE,
    rng: np.random.Generator = RNG,
) -> np.ndarray:
    """Synthetic TBM vibration noise.

    Three components:
      1. Low-frequency ground coupling (< 30 Hz) — cutter rotation, thrust
      2. Narrowband harmonics at cutter RPM and multiples
      3. Broadband mechanical noise (gears, conveyors)
    """
    t = np.arange(n_samples) / sample_rate

    # Low-frequency rumble: 1/f^2 shaped below 30 Hz
    lf_noise = rng.normal(0, 1.0, n_samples)
    fft = np.fft.rfft(lf_noise)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    shaping = np.ones_like(freqs)
    mask = freqs > 0
    shaping[mask] = np.where(freqs[mask] < 30, (30.0 / freqs[mask]) ** 2, 1.0)
    fft *= shaping
    lf_noise = np.fft.irfft(fft, n=n_samples) * 5.0

    # Cutter harmonics (RPM = 6 rev/s) with frequency wander
    rpm_freq = 6.0
    harmonics = np.zeros(n_samples)
    for k in range(1, 8):
        freq = rpm_freq * k
        phase = rng.uniform(0, 2 * np.pi)
        amp = 3.0 / k
        harmonics += amp * np.sin(2 * np.pi * freq * t + phase)
        harmonics += 0.3 * amp * np.sin(
            2 * np.pi * (freq + 0.5 * np.sin(0.3 * t)) * t
        )

    # Broadband mechanical noise
    bb_noise = rng.normal(0, 0.5, n_samples)

    return lf_noise + harmonics + bb_noise


def inject_reflections(
    trace: np.ndarray,
    wavelet: np.ndarray,
    boundaries: list[dict],
    velocity: float,
    sample_rate: float,
    source_amplitude: float = 1.0,
) -> np.ndarray:
    """Add synthetic reflections at specified geological boundaries."""
    result = trace.copy()
    for b in boundaries:
        travel_time = 2.0 * b["distance_m"] / velocity
        idx = int(travel_time * sample_rate)
        n_wav = len(wavelet)
        if idx + n_wav <= len(result):
            result[idx:idx + n_wav] += source_amplitude * b["impedance_contrast"] * wavelet
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_samples = int(DURATION * SAMPLE_RATE)
    t = np.arange(n_samples) / SAMPLE_RATE

    wavelet = make_ricker_wavelet()
    noise_only = make_tbm_noise(n_samples)
    signal_trace = inject_reflections(
        noise_only, wavelet, BOUNDARIES, VELOCITY, SAMPLE_RATE, SOURCE_AMPLITUDE,
    )

    # Separate noise segment for PSD estimation (record before firing source)
    noise_for_psd = make_tbm_noise(n_samples * 4, rng=np.random.default_rng(99))

    # --- Pipeline ---

    # 1. Noise PSD
    psd_freqs, psd = estimate_noise_psd(noise_for_psd, SAMPLE_RATE)

    # 2. Whiten + bandpass (for panel b visualisation only — the matched
    #    filter does its own PSD weighting internally)
    whitened = whiten_seismic(signal_trace, SAMPLE_RATE, psd_freqs, psd, fmin=FMIN)
    whitened = bandpass(whitened, SAMPLE_RATE, fmin=FMIN, fmax=FMAX)

    # 3. Calibrate noise level on a noise-only segment
    noise_cal = make_tbm_noise(n_samples, rng=np.random.default_rng(77))
    _, snr_null = matched_filter_reflection(
        noise_cal, wavelet, SAMPLE_RATE, psd_freqs, psd, fmin=FMIN, fmax=FMAX,
    )
    noise_sigma = np.std(snr_null)

    # 4. Matched-filter scan
    scan = scan_travel_times(
        signal_trace, wavelet, SAMPLE_RATE, psd_freqs, psd, VELOCITY,
        fmin=FMIN, fmax=FMAX, max_distance=80.0, noise_sigma=noise_sigma,
    )

    # 5. Detection confidence
    n_trials = len(scan["distances"])
    peak_info = []
    for d, s in zip(scan["peak_distances"], scan["peak_snrs"]):
        conf = detection_confidence(s, n_trials=n_trials)
        peak_info.append({"distance": d, "snr": s, **conf})

    # 6. Injection campaign — sensitivity curve
    test_contrasts = np.linspace(0.01, 0.25, 12)
    n_realisations = 20
    detection_rates = []

    for contrast in test_contrasts:
        reflected_amp = SOURCE_AMPLITUDE * contrast
        detections = 0
        for trial in range(n_realisations):
            trial_noise = make_tbm_noise(
                n_samples, rng=np.random.default_rng(1000 + trial),
            )
            inj_results = run_injection_test(
                noise_trace=trial_noise,
                wavelet=wavelet,
                sample_rate=SAMPLE_RATE,
                psd_freqs=psd_freqs,
                psd=psd,
                velocity=VELOCITY,
                injection_distances=[30.0],
                injection_amplitudes=[reflected_amp],
                fmin=FMIN,
                fmax=FMAX,
                detection_threshold=3.5,
            )
            if inj_results and inj_results[0].detected:
                detections += 1
        detection_rates.append(detections / n_realisations)

    # --- Plot ---

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Seismic Lookahead — Matched-Filter Detection Through TBM Noise",
        fontsize=14, fontweight="bold", y=0.98,
    )

    boundary_colors = ["#d62728", "#2ca02c", "#9467bd"]

    # (a) Raw trace
    ax = axes[0, 0]
    ax.plot(t * 1000, signal_trace, "k", linewidth=0.3, alpha=0.8)
    ymin, ymax = ax.get_ylim()
    for i, b in enumerate(BOUNDARIES):
        tt = 2 * b["distance_m"] / VELOCITY * 1000
        ax.axvline(tt, color=boundary_colors[i], linestyle="--", alpha=0.6, linewidth=1)
        ax.text(
            tt + 2, ymax * 0.85 - i * (ymax - ymin) * 0.12,
            b["label"], fontsize=8, color=boundary_colors[i], fontweight="bold",
        )
    ax.set_xlabel("Two-way travel time (ms)")
    ax.set_ylabel("Geophone amplitude")
    ax.set_title("(a) Raw geophone trace — reflections buried in TBM noise")
    ax.set_xlim(0, DURATION * 1000)

    # (b) Noise PSD (more informative than whitened trace)
    ax = axes[0, 1]
    ax.semilogy(psd_freqs, psd, "k", linewidth=1)
    ax.axvspan(FMIN, FMAX, alpha=0.1, color="steelblue", label=f"Analysis band ({FMIN:.0f}-{FMAX:.0f} Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("(b) TBM noise spectrum — coloured, like LIGO")
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.03, 0.05,
        "Cutter harmonics\n+ low-freq rumble\ndominate below 50 Hz",
        transform=ax.transAxes, fontsize=7, color="gray", va="bottom",
    )

    # (c) Matched-filter SNR vs distance
    ax = axes[1, 0]
    ax.plot(scan["distances"], np.abs(scan["snr"]), "steelblue", linewidth=0.8)
    ax.axhline(3.5, color="orange", linestyle="--", linewidth=1,
               label="Detection threshold (3.5$\\sigma$)")

    for i, b in enumerate(BOUNDARIES):
        ax.axvline(b["distance_m"], color=boundary_colors[i], linestyle=":",
                   alpha=0.5, linewidth=1)

    # Mark detected peaks that match known boundaries
    for pi in peak_info:
        is_real = any(abs(pi["distance"] - b["distance_m"]) < 3.0 for b in BOUNDARIES)
        if is_real:
            ax.plot(pi["distance"], abs(pi["snr"]), "^", color="green", markersize=10,
                    zorder=5, label=f'{pi["distance"]:.0f}m: {pi["confidence_pct"]:.1f}% conf')
        elif abs(pi["snr"]) > 3.5:
            ax.plot(pi["distance"], abs(pi["snr"]), "o", color="orange", markersize=6,
                    alpha=0.6)

    ax.set_xlabel("Distance ahead of face (m)")
    ax.set_ylabel("|Matched-filter SNR|")
    ax.set_title("(c) Reflectivity profile — peaks at geological boundaries")
    ax.legend(fontsize=7, loc="upper right")

    # (d) Injection sensitivity curve
    ax = axes[1, 1]
    ax.plot(test_contrasts * 100, detection_rates, "o-", color="steelblue",
            linewidth=1.5, markersize=5)
    ax.axhline(0.9, color="orange", linestyle="--", linewidth=1,
               label="90% detection rate")
    ax.set_xlabel("Impedance contrast (%)")
    ax.set_ylabel("Detection rate")
    ax.set_title("(d) Injection calibration — what can we detect at 30m?")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # Annotate 90% crossing
    for i in range(len(detection_rates) - 1):
        if detection_rates[i] < 0.9 <= detection_rates[i + 1]:
            threshold = np.interp(
                0.9,
                [detection_rates[i], detection_rates[i + 1]],
                [test_contrasts[i] * 100, test_contrasts[i + 1] * 100],
            )
            ax.annotate(
                f"90% detection at ~{threshold:.0f}% contrast",
                xy=(threshold, 0.9),
                xytext=(threshold + 3, 0.65),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray",
            )
            break

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("tunnelling_demo.png", dpi=150, bbox_inches="tight")
    print("Saved: tunnelling_demo.png")

    # --- Console summary ---

    print("\n" + "=" * 70)
    print("  SEISMIC LOOKAHEAD DEMO — DETECTION SUMMARY")
    print("=" * 70)
    print(f"  TBM noise:     synthetic, {SAMPLE_RATE:.0f} Hz sample rate")
    print(f"  P-wave vel:    {VELOCITY:.0f} m/s")
    print(f"  Source:         Ricker wavelet, 150 Hz peak, amplitude {SOURCE_AMPLITUDE:.0f}")
    print(f"  Analysis band:  {FMIN:.0f}-{FMAX:.0f} Hz")
    print("-" * 70)

    header = f"  {'Distance':>10s}  {'|SNR|':>7s}  {'Confidence':>11s}  {'Match':20s}  Status"
    print(f"\n{header}")
    print("  " + "-" * 66)

    for pi in peak_info:
        match = ""
        for b in BOUNDARIES:
            if abs(pi["distance"] - b["distance_m"]) < 3.0:
                match = b["label"]
        status = "DETECTED" if pi["confidence_pct"] > 95 else "marginal"
        print(
            f'  {pi["distance"]:>8.1f} m  {abs(pi["snr"]):>7.1f}  '
            f'{pi["confidence_pct"]:>9.1f}%   {match:20s}  [{status}]'
        )

    print(f"\n  Injection calibration (30m, {n_realisations} trials per point):")
    for contrast, rate in zip(test_contrasts, detection_rates):
        bar = "#" * int(rate * 30)
        print(f"    {contrast*100:5.1f}% contrast: {rate*100:5.1f}% detected  {bar}")


if __name__ == "__main__":
    main()
