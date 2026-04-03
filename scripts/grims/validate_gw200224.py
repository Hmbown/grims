"""
GW200224_222234 Validation: Five Diagnostic Tests
==================================================

The single most important measurement in the GRIM-S stack.
kappa = +0.281 +/- 0.096, SNR = 2.93 (2.93 sigma positive).

Before trusting it, run these five tests:

  1a. Start time scan: t_start = [5, 8, 10, 12, 15, 20] M
  1b. Multi-detector: H1, L1, V1 separately
  1c. Frequency precision: residual peak at exactly 2*f_220?
  1d. Time-shift null test: +/-1s, does excess vanish?
  1e. Adjacent bands: SNR at f_NL vs f_NL +/- 50, 100, 150 Hz

If all 5 pass -> candidate detection.
If any fails -> noise; report which test killed it.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from bown_instruments.grims.whiten import estimate_asd, whiten_strain, bandpass
from bown_instruments.grims.phase_locked_search import (
    fit_fundamental_mode,
    build_phase_locked_template,
    phase_locked_search,
)
from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5

EVENT = {
    "name": "GW200224_222234",
    "gps": 1266618172.4,
    "remnant_mass": 68.7,
    "remnant_spin": 0.678,
    "total_mass": 72.3,
    "distance": 1710.0,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MASS = EVENT["remnant_mass"]
SPIN = EVENT["remnant_spin"]
M_SEC = MASS * M_SUN_SECONDS
GPS = EVENT["gps"]

catalog = KerrQNMCatalog()
mode_220 = catalog.linear_mode(2, 2, 0, SPIN)
mode_nl = catalog.nonlinear_mode_quadratic(SPIN)
mode_440 = catalog.linear_mode(4, 4, 0, SPIN)

F_220 = mode_220.physical_frequency_hz(MASS)
F_NL = mode_nl.physical_frequency_hz(MASS)
F_440 = mode_440.physical_frequency_hz(MASS)

DETECTORS = {
    "H1": f"{DATA_DIR}/H-H1_GWOSC_4KHZ_R1-1266618157-32.hdf5",
    "L1": f"{DATA_DIR}/L-L1_GWOSC_4KHZ_R1-1266618157-32.hdf5",
    "V1": f"{DATA_DIR}/V-V1_GWOSC_4KHZ_R1-1266618157-32.hdf5",
}


def load_and_prepare(detector, t_start_m=10.0, pad_before_s=0.05, seg_duration_s=0.15):
    """Load strain, whiten, bandpass, extract ringdown segment."""
    path = DETECTORS[detector]
    if not os.path.exists(path):
        return None

    loaded = load_gwosc_strain_hdf5(path)
    strain = loaded["strain"]
    time = loaded["time"]
    sr = loaded["sample_rate"]

    f_low = max(20.0, F_220 * 0.5)
    f_high = min(0.45 * sr, max(F_NL, F_440) * 1.3)

    asd_freqs, asd = estimate_asd(
        strain, sr, merger_time=GPS, time=time, exclusion_window=2.0
    )
    whitened = whiten_strain(strain, sr, asd_freqs, asd, fmin=f_low * 0.8)
    whitened_bp = bandpass(whitened, sr, f_low, f_high)

    ringdown_start = GPS + t_start_m * M_SEC
    t0 = ringdown_start - pad_before_s
    t1 = ringdown_start + seg_duration_s
    mask = (time >= t0) & (time <= t1)

    seg_strain = whitened_bp[mask]
    seg_time = time[mask]
    t_dimless = (seg_time - ringdown_start) / M_SEC

    noise_mask = np.abs(time - GPS) > 4.0
    noise_rms = np.sqrt(np.var(whitened_bp[noise_mask]))

    return {
        "seg_strain": seg_strain,
        "t_dimless": t_dimless,
        "noise_rms": noise_rms,
        "full_whitened": whitened_bp,
        "full_time": time,
        "sr": sr,
        "ringdown_start_gps": ringdown_start,
    }


# =========================================================================
# TEST 1a: Start Time Scan
# =========================================================================
def test_1a_start_time_scan():
    print("\n" + "=" * 70)
    print("TEST 1a: Ringdown Start Time Scan")
    print("=" * 70)

    t_starts = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    results = []

    for det in ["H1", "L1", "V1"]:
        print(f"\n  Detector: {det}")
        for t_m in t_starts:
            prep = load_and_prepare(det, t_start_m=t_m)
            if prep is None:
                continue
            r = phase_locked_search(
                prep["seg_strain"],
                prep["t_dimless"],
                SPIN,
                prep["noise_rms"],
                event_name=f"{EVENT['name']}_{det}_t{t_m:.0f}M",
            )
            results.append(
                {
                    "det": det,
                    "t_start_m": t_m,
                    "kappa": r.kappa_hat,
                    "sigma": r.kappa_sigma,
                    "snr": r.snr,
                    "a_220": r.a_220_fit,
                }
            )
            print(
                f"    t_start={t_m:5.1f}M  kappa={r.kappa_hat:+.3f} +/- {r.kappa_sigma:.3f}  "
                f"SNR={r.snr:+.3f}  A_220={r.a_220_fit:.3f}"
            )

    print("\n  Stability check:")
    for det in ["H1", "L1", "V1"]:
        det_res = [r for r in results if r["det"] == det]
        if not det_res:
            continue
        kappas = [r["kappa"] for r in det_res]
        kappa_range = max(kappas) - min(kappas)
        signs = [np.sign(k) for k in kappas]
        all_same_sign = len(set(signs)) == 1
        print(
            f"    {det}: kappa range = [{min(kappas):.3f}, {max(kappas):.3f}], "
            f"span = {kappa_range:.3f}, same sign: {all_same_sign}"
        )

    return results


# =========================================================================
# TEST 1b: Multi-Detector Consistency
# =========================================================================
def test_1b_multi_detector():
    print("\n" + "=" * 70)
    print("TEST 1b: Multi-Detector Consistency (H1, L1, V1)")
    print("=" * 70)

    results = {}
    for det in ["H1", "L1", "V1"]:
        prep = load_and_prepare(det)
        if prep is None:
            print(f"  {det}: no data file")
            continue
        r = phase_locked_search(
            prep["seg_strain"],
            prep["t_dimless"],
            SPIN,
            prep["noise_rms"],
            event_name=f"{EVENT['name']}_{det}",
        )
        results[det] = r
        print(
            f"  {det}: kappa={r.kappa_hat:+.3f} +/- {r.kappa_sigma:.3f}  "
            f"SNR={r.snr:+.3f}  A_220={r.a_220_fit:.3f}"
        )

    if len(results) >= 2:
        dets = list(results.keys())
        kappas = [results[d].kappa_hat for d in dets]
        sigmas = [results[d].kappa_sigma for d in dets]

        if len(kappas) >= 2:
            mean_k = np.mean(kappas)
            spread = max(kappas) - min(kappas)
            mean_sigma = np.mean(sigmas)

            consistent = all(
                abs(results[d].kappa_hat - mean_k) < 2.0 * results[d].kappa_sigma
                for d in dets
            )
            print(f"\n  Mean kappa: {mean_k:.3f}")
            print(f"  Spread: {spread:.3f}")
            print(f"  Mean sigma: {mean_sigma:.3f}")
            print(
                f"  Detectors consistent (within 2-sigma of each other): {consistent}"
            )

            if consistent:
                print("  PASS: All detectors agree.")
            else:
                print("  FAIL: Detectors disagree. Likely a noise artifact.")
    return results


# =========================================================================
# TEST 1c: Frequency Precision
# =========================================================================
def test_1c_frequency_precision():
    print("\n" + "=" * 70)
    print("TEST 1c: Frequency Precision -- Is the residual peak at exactly 2*f_220?")
    print("=" * 70)

    results = {}
    for det in ["H1", "L1", "V1"]:
        prep = load_and_prepare(det)
        if prep is None:
            continue

        fit = fit_fundamental_mode(prep["seg_strain"], prep["t_dimless"], SPIN)
        residual = fit["residual"]
        mask = prep["t_dimless"] >= 0
        residual_rd = residual[mask]

        if len(residual_rd) < 16:
            print(f"  {det}: segment too short")
            continue

        sr = prep["sr"]
        nfft = max(2048, len(residual_rd) * 4)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
        window = np.hanning(len(residual_rd))
        padded = np.zeros(nfft)
        padded[: len(residual_rd)] = residual_rd * window
        spectrum = np.abs(np.fft.rfft(padded)) ** 2

        T_rd = len(residual_rd) / sr
        delta_f = 1.0 / T_rd

        search_band = 100.0
        band_mask = (freqs >= F_NL - search_band) & (freqs <= F_NL + search_band)
        band_freqs = freqs[band_mask]
        band_power = spectrum[band_mask]

        if len(band_power) == 0:
            print(f"  {det}: no frequency bins near f_NL")
            continue

        peak_idx = np.argmax(band_power)
        f_peak = band_freqs[peak_idx]
        delta = f_peak - F_NL

        noise_level = np.median(band_power)
        peak_snr = (band_power[peak_idx] - noise_level) / (noise_level + 1e-30)

        results[det] = {
            "f_peak": f_peak,
            "delta": delta,
            "delta_f": delta_f,
            "peak_snr": peak_snr,
        }
        print(
            f"  {det}: predicted f_NL = {F_NL:.2f} Hz, "
            f"observed peak = {f_peak:.2f} Hz, "
            f"offset = {delta:+.2f} Hz ({delta / F_NL:+.4%})"
        )
        print(f"         resolution = {delta_f:.1f} Hz, peak SNR = {peak_snr:.1f}x")

        if abs(delta) < delta_f:
            print(f"         Offset < resolution. CONSISTENT.")
        else:
            print(f"         Offset > resolution. INCONSISTENT.")

    return results


# =========================================================================
# TEST 1d: Time-Shift Null Test
# =========================================================================
def test_1d_time_shift_null():
    print("\n" + "=" * 70)
    print("TEST 1d: Time-Shift Null Test -- Does excess vanish when shifted?")
    print("=" * 70)

    shifts_s = [-1.0, -0.5, 0.0, +0.5, +1.0]
    results = {}

    for det in ["H1", "L1", "V1"]:
        path = DETECTORS.get(det)
        if not os.path.exists(path):
            continue

        loaded = load_gwosc_strain_hdf5(path)
        strain_full = loaded["strain"]
        time_full = loaded["time"]
        sr = loaded["sample_rate"]

        f_low = max(20.0, F_220 * 0.5)
        f_high = min(0.45 * sr, max(F_NL, F_440) * 1.3)

        asd_freqs, asd = estimate_asd(
            strain_full, sr, merger_time=GPS, time=time_full, exclusion_window=2.0
        )
        whitened = whiten_strain(strain_full, sr, asd_freqs, asd, fmin=f_low * 0.8)
        whitened_bp = bandpass(whitened, sr, f_low, f_high)

        noise_mask = np.abs(time_full - GPS) > 4.0
        noise_rms = np.sqrt(np.var(whitened_bp[noise_mask]))

        det_results = []
        for shift in shifts_s:
            ringdown_start = GPS + 10.0 * M_SEC + shift
            t0 = ringdown_start - 0.05
            t1 = ringdown_start + 0.15
            mask = (time_full >= t0) & (time_full <= t1)

            if np.sum(mask) < 50:
                det_results.append({"shift": shift, "snr": np.nan, "kappa": np.nan})
                continue

            seg_s = whitened_bp[mask]
            seg_t = time_full[mask]
            t_dl = (seg_t - ringdown_start) / M_SEC

            r = phase_locked_search(
                seg_s,
                t_dl,
                SPIN,
                noise_rms,
                event_name=f"{EVENT['name']}_{det}_shift{shift:+.1f}s",
            )
            det_results.append(
                {
                    "shift": shift,
                    "snr": r.snr,
                    "kappa": r.kappa_hat,
                    "kappa_sigma": r.kappa_sigma,
                }
            )

        results[det] = det_results

        print(f"\n  {det}:")
        print(f"  {'Shift':>8} {'SNR':>8} {'kappa':>8} {'sigma':>8}")
        print("  " + "-" * 36)
        for r in det_results:
            snr_s = f"{r['snr']:+.2f}" if np.isfinite(r["snr"]) else "N/A"
            k_s = f"{r['kappa']:+.3f}" if np.isfinite(r["kappa"]) else "N/A"
            sig_s = (
                f"{r['kappa_sigma']:.3f}"
                if np.isfinite(r.get("kappa_sigma", np.nan))
                else "N/A"
            )
            marker = " <-- SIGNAL" if r["shift"] == 0.0 else "    null"
            print(f"  {r['shift']:>+8.1f} {snr_s:>8} {k_s:>8} {sig_s:>8}{marker}")

        signal = [r for r in det_results if r["shift"] == 0.0 and np.isfinite(r["snr"])]
        nulls = [r for r in det_results if r["shift"] != 0.0 and np.isfinite(r["snr"])]
        if signal and nulls:
            sig_snr = abs(signal[0]["snr"])
            null_snrs = [abs(r["snr"]) for r in nulls]
            print(
                f"  Signal |SNR| = {sig_snr:.2f}, Max null |SNR| = {max(null_snrs):.2f}"
            )
            if sig_snr > 2.0 * max(null_snrs):
                print(f"  PASS: Signal >2x any null.")
            else:
                print(f"  FAIL: Signal not clearly above nulls.")

    return results


# =========================================================================
# TEST 1e: Adjacent Frequency Bands
# =========================================================================
def test_1e_adjacent_bands():
    print("\n" + "=" * 70)
    print("TEST 1e: Adjacent Frequency Bands -- Narrowband or broadband?")
    print("=" * 70)

    offsets = [-150, -100, -50, -25, 0, 25, 50, 100, 150]
    all_results = {}

    for det in ["H1", "L1", "V1"]:
        prep = load_and_prepare(det)
        if prep is None:
            continue

        fit = fit_fundamental_mode(prep["seg_strain"], prep["t_dimless"], SPIN)
        residual = fit["residual"]
        a_220 = fit["amplitude"]
        phi_220 = fit["phase"]
        noise_var = prep["noise_rms"] ** 2

        omega_nl = mode_nl.omega
        mask = prep["t_dimless"] >= 0
        det_results = []

        for offset in offsets:
            f_test = F_NL + offset
            omega_test_real = 2.0 * np.pi * f_test * M_SEC
            omega_test = complex(omega_test_real, omega_nl.imag)

            template = np.zeros_like(prep["t_dimless"])
            a_nl = a_220**2
            phi_nl = 2.0 * phi_220
            template[mask] = (
                a_nl
                * np.exp(omega_test.imag * prep["t_dimless"][mask])
                * np.cos(omega_test.real * prep["t_dimless"][mask] + phi_nl)
            )

            inner_rt = np.sum(residual[mask] * template[mask]) / noise_var
            inner_tt = np.sum(template[mask] * template[mask]) / noise_var
            tnorm = np.sqrt(inner_tt)

            if tnorm > 0:
                snr = inner_rt / tnorm
                kappa = inner_rt / inner_tt
            else:
                snr = 0.0
                kappa = 0.0

            det_results.append(
                {"offset": offset, "f_test": f_test, "snr": snr, "kappa": kappa}
            )

        all_results[det] = det_results

        print(f"\n  {det}:")
        print(f"  {'Offset':>8} {'f_test':>10} {'SNR':>8} {'kappa':>8}")
        print("  " + "-" * 40)
        for r in det_results:
            marker = " <-- f_NL" if r["offset"] == 0 else ""
            print(
                f"  {r['offset']:>+8.0f} {r['f_test']:>10.1f} {r['snr']:>8.2f} {r['kappa']:>8.2f}{marker}"
            )

        snrs = np.array([abs(r["snr"]) for r in det_results])
        offsets_arr = np.array(offsets)
        snr_fnl = snrs[offsets_arr == 0][0]
        off_snrs = snrs[offsets_arr != 0]

        if snr_fnl > 1.5 * np.max(off_snrs):
            print(
                f"  PASS: f_NL peak dominant ({snr_fnl:.2f} > 1.5x max adjacent {np.max(off_snrs):.2f})"
            )
        elif snr_fnl > np.max(off_snrs):
            print(
                f"  MARGINAL: f_NL highest but not dominant ({snr_fnl:.2f} vs {np.max(off_snrs):.2f})"
            )
        else:
            print(
                f"  FAIL: f_NL not the peak ({snr_fnl:.2f} vs {np.max(off_snrs):.2f})"
            )

    return all_results


# =========================================================================
# SUMMARY
# =========================================================================
def summarize(results_1a, results_1b, results_1c, results_1d, results_1e):
    print("\n" + "=" * 70)
    print("GW200224_222234 VALIDATION SUMMARY")
    print("=" * 70)

    passes = []
    fails = []

    # 1a: Start time stability
    all_kappas = [r["kappa"] for r in results_1a]
    if all_kappas:
        same_sign = all(k > 0 for k in all_kappas if abs(k) > 0.01)
        if same_sign:
            passes.append(
                "1a: Start time scan - kappa consistently positive across t_start values"
            )
        else:
            fails.append("1a: Start time scan - kappa changes sign with t_start")

    # 1b: Multi-detector
    if results_1b and len(results_1b) >= 2:
        dets = list(results_1b.keys())
        mean_k = np.mean([results_1b[d].kappa_hat for d in dets])
        consistent = all(
            abs(results_1b[d].kappa_hat - mean_k) < 2.0 * results_1b[d].kappa_sigma
            for d in dets
        )
        if consistent:
            passes.append("1b: Multi-detector - H1/L1/V1 consistent")
        else:
            fails.append("1b: Multi-detector - detectors disagree")
    else:
        fails.append("1b: Multi-detector - insufficient detectors")

    # 1c: Frequency precision
    for det, r in results_1c.items():
        if abs(r["delta"]) < r["delta_f"]:
            passes.append(
                f"1c: Freq precision ({det}) - peak within resolution of 2*f_220"
            )
        else:
            fails.append(
                f"1c: Freq precision ({det}) - peak offset {r['delta']:.1f} Hz exceeds resolution"
            )

    # 1d: Time-shift null
    for det, det_res in results_1d.items():
        signal = [r for r in det_res if r["shift"] == 0.0 and np.isfinite(r["snr"])]
        nulls = [r for r in det_res if r["shift"] != 0.0 and np.isfinite(r["snr"])]
        if signal and nulls:
            sig_snr = abs(signal[0]["snr"])
            null_snrs = [abs(r["snr"]) for r in nulls]
            if sig_snr > 2.0 * max(null_snrs):
                passes.append(f"1d: Null test ({det}) - signal >2x all nulls")
            else:
                fails.append(f"1d: Null test ({det}) - signal not clearly above nulls")

    # 1e: Adjacent bands
    for det, det_res in results_1e.items():
        snrs = np.array([abs(r["snr"]) for r in det_res])
        offsets_arr = np.array([r["offset"] for r in det_res])
        snr_fnl = snrs[offsets_arr == 0][0]
        off_snrs = snrs[offsets_arr != 0]
        if snr_fnl > 1.5 * np.max(off_snrs):
            passes.append(f"1e: Adjacent bands ({det}) - narrowband peak at f_NL")
        else:
            fails.append(
                f"1e: Adjacent bands ({det}) - excess is broadband, not narrowband"
            )

    print(f"\n  PASSES ({len(passes)}):")
    for p in passes:
        print(f"    [PASS] {p}")

    print(f"\n  FAILURES ({len(fails)}):")
    for f in fails:
        print(f"    [FAIL] {f}")

    print(f"\n  Total: {len(passes)} passes, {len(fails)} failures")

    if len(fails) == 0:
        print("\n  VERDICT: GW200224 passes all diagnostics -> CANDIDATE DETECTION")
    elif len(passes) > len(fails):
        print("\n  VERDICT: Mixed results -> INCONCLUSIVE, needs further study")
    else:
        print("\n  VERDICT: GW200224 fails key diagnostics -> LIKELY NOISE")
        print("  The 2.93 sigma excess is not a reliable nonlinear mode signal.")


def main():
    print("=" * 70)
    print("GRIM-S: GW200224_222234 Validation Suite")
    print(f"kappa = +0.281 +/- 0.096, SNR = 2.93")
    print(f"f_220 = {F_220:.2f} Hz, f_NL = {F_NL:.2f} Hz")
    print(f"M = {MASS} Msun, a = {SPIN}")
    print("=" * 70)

    r1a = test_1a_start_time_scan()
    r1b = test_1b_multi_detector()
    r1c = test_1c_frequency_precision()
    r1d = test_1d_time_shift_null()
    r1e = test_1e_adjacent_bands()

    summarize(r1a, r1b, r1c, r1d, r1e)


if __name__ == "__main__":
    main()
