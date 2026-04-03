"""
Task 2: Debias the phase-locked estimator.

The bias: A_220_fit^2 = A_220_true^2 + sigma_A^2 (noise inflates squared amplitude)
So kappa_hat = kappa_true * A_220_true^2 / (A_220_true^2 + sigma_A^2) < kappa_true

Correction: kappa_debiased = kappa_hat * A_220_fit^2 / (A_220_fit^2 - sigma_A^2)

This requires extracting sigma_A from the least-squares covariance matrix.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from bown_instruments.grims.whiten import estimate_asd, whiten_strain, bandpass
from bown_instruments.grims.phase_locked_search import (
    fit_fundamental_mode,
    phase_locked_search,
    stack_phase_locked,
    PhaseLockResult,
    StackedPhaseLockResult,
)
from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5


def fit_fundamental_with_covariance(data, t, spin):
    """Like fit_fundamental_mode but also returns sigma_A from the covariance."""
    from bown_instruments.grims.qnm_modes import KerrQNMCatalog

    catalog = KerrQNMCatalog()
    mode = catalog.linear_mode(2, 2, 0, spin)
    omega = mode.omega

    mask = t >= 0
    t_pos = t[mask]
    d_pos = data[mask]

    envelope = np.exp(omega.imag * t_pos)
    basis_cos = envelope * np.cos(omega.real * t_pos)
    basis_sin = envelope * np.sin(omega.real * t_pos)

    A = np.column_stack([basis_cos, basis_sin])

    # Least squares with covariance estimation
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, d_pos, rcond=None)
    a, b = coeffs

    # Estimate noise variance from residuals
    n = len(d_pos)
    p = 2  # number of parameters
    if n > p:
        dof = n - p
        residual_ss = np.sum((d_pos - A @ coeffs) ** 2)
        sigma2_noise = residual_ss / dof
    else:
        sigma2_noise = 0.0

    # Covariance matrix of coefficients
    # cov = sigma2_noise * (A^T A)^{-1}
    try:
        ATA_inv = np.linalg.inv(A.T @ A)
        cov = sigma2_noise * ATA_inv
    except np.linalg.LinAlgError:
        cov = np.zeros((2, 2))

    # sigma_A from error propagation:
    # A = sqrt(a^2 + b^2), so sigma_A^2 = (a^2 * sigma_a^2 + b^2 * sigma_b^2) / (a^2 + b^2)
    sigma_a2 = cov[0, 0]
    sigma_b2 = cov[1, 1]
    amplitude = np.sqrt(a**2 + b**2)

    if amplitude > 0:
        sigma_A_sq = (a**2 * sigma_a2 + b**2 * sigma_b2) / (a**2 + b**2)
    else:
        sigma_A_sq = 0.0

    phase = np.arctan2(-b, a)

    return {
        "amplitude": amplitude,
        "phase": phase,
        "omega": omega,
        "sigma_A": np.sqrt(max(0.0, sigma_A_sq)),
        "sigma_A_sq": max(0.0, sigma_A_sq),
        "cov": cov,
        "sigma2_noise": sigma2_noise,
    }


def load_and_prepare(event, data_dir="data/", t_start_m=10.0, detector=None):
    """Load and prepare strain for a catalog event."""
    from pathlib import Path

    mass = event.get("remnant_mass", 0)
    spin = event.get("remnant_spin", 0.69)
    gps = event.get("gps", 0)

    if mass <= 0 or gps <= 0:
        return None

    m_sec = mass * M_SUN_SECONDS

    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)
    mode_440 = catalog.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)

    f_low = max(20.0, f_220 * 0.5)
    f_high_target = max(f_nl, f_440) * 1.3

    # Find data file
    data_path = Path(data_dir)
    gps_int = int(gps)
    local_file = None
    actual_det = detector

    if detector:
        # Search for specific detector
        for f in sorted(
            data_path.glob(f"*{detector[0]}-{detector}*{gps_int - 50}*.hdf5")
        ):
            pass
        # Broader search
        candidates = []
        for f in sorted(data_path.glob("*.hdf5")):
            if detector[0] in f.stem or detector in f.stem:
                parts = f.stem.split("-")
                if len(parts) >= 3:
                    try:
                        file_gps = int(parts[-2])
                        file_dur = int(parts[-1])
                        if file_gps <= gps_int <= file_gps + file_dur:
                            candidates.append(f)
                    except ValueError:
                        continue
        if candidates:
            local_file = str(candidates[0])
            actual_det = detector
    else:
        for f in sorted(data_path.glob("*.hdf5")):
            parts = f.stem.split("-")
            if len(parts) >= 3:
                try:
                    file_gps = int(parts[-2])
                    file_dur = int(parts[-1])
                    if file_gps <= gps_int <= file_gps + file_dur:
                        local_file = str(f)
                        parts2 = f.stem.split("_")
                        if len(parts2) >= 2:
                            actual_det = parts2[1]
                        break
                except ValueError:
                    continue

    if local_file is None:
        return None

    try:
        loaded = load_gwosc_strain_hdf5(local_file)
    except Exception:
        return None

    strain = loaded["strain"]
    time = loaded["time"]
    sr = loaded["sample_rate"]
    f_high = min(0.45 * sr, f_high_target)

    if f_220 < 20 or f_nl > 0.45 * sr:
        return None

    merger_time = float(gps)
    ringdown_start = merger_time + t_start_m * m_sec

    if merger_time < time[0] or merger_time > time[-1]:
        return None

    try:
        asd_freqs, asd = estimate_asd(
            strain, sr, merger_time=merger_time, time=time, exclusion_window=2.0
        )
        whitened = whiten_strain(strain, sr, asd_freqs, asd, fmin=f_low * 0.8)
        whitened_bp = bandpass(whitened, sr, f_low, f_high)
    except Exception:
        return None

    pad_before = 0.05
    seg_duration = 0.15
    t0 = ringdown_start - pad_before
    t1 = ringdown_start + seg_duration
    mask = (time >= t0) & (time <= t1)

    if np.sum(mask) < 50:
        return None

    seg_strain = whitened_bp[mask]
    seg_time = time[mask]
    t_dimless = (seg_time - ringdown_start) / m_sec

    noise_mask = np.abs(time - merger_time) > 4.0
    if np.sum(noise_mask) < 100:
        return None
    noise_var = np.var(whitened_bp[noise_mask])
    noise_rms = np.sqrt(noise_var)

    return {
        "seg_strain": seg_strain,
        "t_dimless": t_dimless,
        "noise_rms": noise_rms,
        "sr": sr,
        "detector": actual_det,
    }


def main():
    print("=" * 70)
    print("TASK 2: Debias the Phase-Locked Estimator")
    print("=" * 70)

    with open("data/gwtc_full_catalog.json") as f:
        catalog = json.load(f)

    targets = [e for e in catalog if e["total_mass"] >= 40.0]
    print(f"Catalog: {len(targets)} events with M_total >= 40")

    # First pass: standard phase-locked results (baseline)
    print("\n--- BASELINE (biased) ---")
    baseline_results = []
    for ev in targets:
        prep = load_and_prepare(ev)
        if prep is None:
            continue
        r = phase_locked_search(
            prep["seg_strain"],
            prep["t_dimless"],
            ev["remnant_spin"],
            prep["noise_rms"],
            event_name=ev["name"],
        )
        baseline_results.append(r)
        print(
            f"  {ev['name']:<30} kappa={r.kappa_hat:+.3f} +/- {r.kappa_sigma:.3f}  "
            f"SNR={r.snr:+.3f}  A_220={r.a_220_fit:.3f}"
        )

    if baseline_results:
        stacked_baseline = stack_phase_locked(baseline_results)
        print(
            f"\n  STACKED (biased): kappa={stacked_baseline.kappa_hat:+.3f} +/- "
            f"{stacked_baseline.kappa_sigma:.3f}  SNR={stacked_baseline.snr:.3f}"
        )

    # Second pass: debiased results
    print("\n--- DEBIASED (sigma_A correction) ---")
    debiased_results = []
    for ev in targets:
        prep = load_and_prepare(ev)
        if prep is None:
            continue

        spin = ev["remnant_spin"]

        # Get biased result
        r = phase_locked_search(
            prep["seg_strain"],
            prep["t_dimless"],
            spin,
            prep["noise_rms"],
            event_name=ev["name"],
        )

        # Get sigma_A from covariance
        cov_fit = fit_fundamental_with_covariance(
            prep["seg_strain"], prep["t_dimless"], spin
        )
        a_220 = cov_fit["amplitude"]
        sigma_A = cov_fit["sigma_A"]

        # Apply debiasing correction
        a_220_sq = a_220**2
        sigma_A_sq = sigma_A**2

        if a_220_sq > sigma_A_sq and a_220_sq > 0:
            correction = a_220_sq / (a_220_sq - sigma_A_sq)
            kappa_debiased = r.kappa_hat * correction
            # Uncertainty also scales
            sigma_debiased = r.kappa_sigma * correction
        else:
            correction = float("inf")
            kappa_debiased = float("nan")
            sigma_debiased = float("nan")

        # Build a debiased PhaseLockResult
        debiased_r = PhaseLockResult(
            event_name=ev["name"],
            kappa_hat=kappa_debiased if np.isfinite(kappa_debiased) else 0.0,
            kappa_sigma=sigma_debiased if np.isfinite(sigma_debiased) else float("inf"),
            snr=r.snr * (correction if np.isfinite(correction) else 1.0),
            a_220_fit=a_220,
            phi_220_fit=r.phi_220_fit,
            template_norm=r.template_norm,
            residual_overlap=r.residual_overlap,
            noise_rms=r.noise_rms,
        )

        debiased_results.append(
            {
                "event": ev["name"],
                "result": debiased_r,
                "kappa_hat": r.kappa_hat,
                "kappa_debiased": kappa_debiased,
                "sigma_debiased": sigma_debiased,
                "correction": correction,
                "a_220": a_220,
                "sigma_A": sigma_A,
                "snr": r.snr,
            }
        )

        corr_str = f"{correction:.2f}x" if np.isfinite(correction) else "INF"
        k_str = f"{kappa_debiased:+.3f}" if np.isfinite(kappa_debiased) else "N/A"
        s_str = f"{sigma_debiased:.3f}" if np.isfinite(sigma_debiased) else "N/A"
        print(
            f"  {ev['name']:<30} kappa: {r.kappa_hat:+.3f} -> {k_str}  "
            f"corr={corr_str}  A_220={a_220:.3f} sigma_A={sigma_A:.3f}"
        )

    if debiased_results:
        valid = [
            d
            for d in debiased_results
            if np.isfinite(d["kappa_debiased"]) and d["result"].kappa_sigma < 1e6
        ]
        if valid:
            phase_results = [d["result"] for d in valid]
            stacked_debiased = stack_phase_locked(phase_results)
            print(
                f"\n  STACKED (debiased): kappa={stacked_debiased.kappa_hat:+.3f} +/- "
                f"{stacked_debiased.kappa_sigma:.3f}  SNR={stacked_debiased.snr:.3f}"
            )
            print(f"  N events: {stacked_debiased.n_events}")

            # 95% CI
            k = stacked_debiased.kappa_hat
            s = stacked_debiased.kappa_sigma
            ci_95 = (k - 1.96 * s, k + 1.96 * s)
            print(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
            print(f"  kappa > 0? {'Yes' if ci_95[0] > 0 else 'No'}")
            print(
                f"  kappa = 1 in 95% CI? {'Yes' if ci_95[0] <= 1 <= ci_95[1] else 'No'}"
            )
            print(
                f"  kappa = 0 in 95% CI? {'Yes' if ci_95[0] <= 0 <= ci_95[1] else 'No'}"
            )
        else:
            print("\n  No valid debiased results to stack.")


if __name__ == "__main__":
    main()
