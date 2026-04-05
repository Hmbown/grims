#!/usr/bin/env python3
"""Diagnose the kappa estimator bias in the injection/recovery pipeline.

Traces the full signal path for a single event to identify where the
~73% underestimate of kappa originates.

Usage:
    source venv/bin/activate
    python scripts/grims/diagnose_estimator_bias.py
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bown_instruments.grims.bayesian_analysis import fit_linear_modes
from bown_instruments.grims.gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5
from bown_instruments.grims.injection_campaign import (
    _build_injection_waveform,
    _compute_band_edges,
    _prepare_detector_data,
    WhiteningBand,
    DEFAULT_PAD_BEFORE_S,
)
from bown_instruments.grims.mass_analysis import (
    compute_optimal_segment_duration,
    find_local_strain_detector,
)
from bown_instruments.grims.phase_locked_search import (
    phase_locked_search_colored,
    build_phase_locked_template,
)
from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.ringdown_templates import RingdownTemplateBuilder
from bown_instruments.grims.whiten import bandpass, estimate_asd, whiten_strain


def load_phase3_results():
    with open("results/grims/phase3_results.json") as f:
        return json.load(f)


def load_catalog():
    with open("results/grims/gwtc_full_catalog.json") as f:
        events = json.load(f)
    return {e["name"]: e for e in events}


def diagnose_single_event(event_name, kappa_true=0.03, detector="H1"):
    """Trace the full injection/recovery path for one event."""
    phase3 = load_phase3_results()
    catalog = load_catalog()

    entry = None
    for e in phase3["individual"]:
        if e["event"] == event_name:
            entry = e
            break
    if entry is None:
        print(f"Event {event_name} not found in Phase 3 results")
        return

    cat_event = catalog[event_name]
    mass = float(cat_event["remnant_mass"])
    spin = float(cat_event["remnant_spin"])
    gps = float(cat_event["gps"])
    m_seconds = mass * M_SUN_SECONDS

    print(f"=== DIAGNOSING: {event_name} ===")
    print(f"Mass={mass:.2f} Msun, Spin={spin:.4f}, M_seconds={m_seconds:.6e}")
    print()

    # QNM frequencies
    qnm_cat = KerrQNMCatalog()
    mode_220 = qnm_cat.linear_mode(2, 2, 0, spin)
    mode_nl = qnm_cat.nonlinear_mode_quadratic(spin)
    mode_440 = qnm_cat.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)

    print(f"QNM frequencies: f_220={f_220:.1f} Hz, f_NL={f_nl:.1f} Hz, f_440={f_440:.1f} Hz")
    print(f"omega_220 = {mode_220.omega}")
    print(f"omega_NL  = {mode_nl.omega}")
    print(f"omega_440 = {mode_440.omega}")
    print(f"NL/440 freq ratio: {mode_nl.omega.real / mode_440.omega.real:.4f}")
    print()

    # Load and prepare data
    band = WhiteningBand("default", 0.50, 1.30)
    strain, time, sample_rate, asd_freqs, asd = _prepare_detector_data(
        cat_event, detector, "data"
    )
    f_low, f_high = _compute_band_edges(mass, spin, sample_rate, band)
    print(f"Band: f_low={f_low:.1f} Hz, f_high={f_high:.1f} Hz")
    print(f"Sample rate: {sample_rate:.0f} Hz")

    # Whiten and bandpass
    whitened = whiten_strain(strain, sample_rate, asd_freqs, asd, fmin=f_low * 0.8)
    filtered = bandpass(whitened, sample_rate, f_low, f_high)

    noise_mask = np.abs(time - gps) > 4.0
    noise_rms = float(np.sqrt(np.var(filtered[noise_mask])))
    print(f"Noise RMS (whitened+bandpassed): {noise_rms:.6f}")

    seg_duration = compute_optimal_segment_duration(mass, spin, min_duration=0.03)
    n_samples = int(round((DEFAULT_PAD_BEFORE_S + seg_duration) * sample_rate)) + 1
    print(f"Segment duration: {seg_duration * 1000:.1f} ms, n_samples={n_samples}")
    print()

    # === ON-SOURCE: fit linear modes (this is what the injection uses) ===
    t_start_m = 10.0
    ringdown_start = gps + t_start_m * m_seconds
    on_source_start = ringdown_start - DEFAULT_PAD_BEFORE_S
    idx = int(np.searchsorted(time, on_source_start))
    seg_time = time[idx : idx + n_samples]
    t_dimless = (seg_time - ringdown_start) / m_seconds
    seg_strain = filtered[idx : idx + n_samples]

    # Injection's fit: (220, 330, 440) from bayesian_analysis
    fit_inj = fit_linear_modes(seg_strain, t_dimless, spin)
    A_220_inj = fit_inj["220"]["amplitude"]
    phi_220_inj = fit_inj["220"]["phase"]
    A_330_inj = fit_inj.get("330", {}).get("amplitude", 0)
    A_440_inj = fit_inj.get("440", {}).get("amplitude", 0)

    print("=== ON-SOURCE FIT (used for injection amplitudes) ===")
    print(f"A_220 = {A_220_inj:.8f}")
    print(f"phi_220 = {phi_220_inj:.6f} rad")
    print(f"A_330 = {A_330_inj:.8f}")
    print(f"A_440 = {A_440_inj:.8f}")
    print(f"Injected NL amplitude (kappa={kappa_true}): {kappa_true * A_220_inj**2:.8e}")
    print()

    # === BUILD INJECTION WAVEFORM ===
    waveform = _build_injection_waveform(
        t_dimless, spin=spin, fit=fit_inj, kappa=kappa_true,
        include_higher_linear_modes=True,
    )
    waveform_linear_only = _build_injection_waveform(
        t_dimless, spin=spin, fit=fit_inj, kappa=0.0,
        include_higher_linear_modes=True,
    )

    mask = t_dimless >= 0
    print(f"=== INJECTION WAVEFORM (t>=0 region) ===")
    print(f"Full waveform RMS: {np.sqrt(np.mean(waveform[mask]**2)):.8e}")
    print(f"Linear-only RMS:   {np.sqrt(np.mean(waveform_linear_only[mask]**2)):.8e}")
    print(f"NL-only RMS:       {np.sqrt(np.mean((waveform[mask] - waveform_linear_only[mask])**2)):.8e}")
    print(f"Waveform / noise_rms: {np.sqrt(np.mean(waveform[mask]**2)) / noise_rms:.4f}")
    print()

    # === TEST 1: NOISELESS RECOVERY ===
    print("=" * 60)
    print("TEST 1: NOISELESS RECOVERY (waveform only, no noise)")
    print("=" * 60)
    result_noiseless = phase_locked_search_colored(
        waveform, t_dimless, spin, noise_rms=1.0,
        event_name="noiseless_test",
    )
    print(f"A_220 recovered: {result_noiseless.a_220_fit:.8f}")
    print(f"phi_220 recovered: {result_noiseless.phi_220_fit:.6f} rad")
    print(f"A_220 ratio (recovered/injected): {result_noiseless.a_220_fit / A_220_inj:.6f}")
    print(f"phi_220 difference: {result_noiseless.phi_220_fit - phi_220_inj:.6f} rad")
    print(f"kappa_hat: {result_noiseless.kappa_hat:.8f}")
    print(f"kappa_true: {kappa_true}")
    print(f"Recovery ratio: {result_noiseless.kappa_hat / kappa_true:.6f}")
    print(f"Template norm: {result_noiseless.template_norm:.8f}")
    print(f"Residual overlap: {result_noiseless.residual_overlap:.8e}")
    print()

    # === TEST 2: MANUAL TIME-DOMAIN MATCHED FILTER ===
    print("=" * 60)
    print("TEST 2: MANUAL TIME-DOMAIN MATCHED FILTER")
    print("=" * 60)
    t_pos = t_dimless[mask]
    d_pos = waveform[mask]
    n = len(d_pos)

    linear_qnms = [
        qnm_cat.linear_mode(2, 2, 0, spin),
        qnm_cat.linear_mode(3, 3, 0, spin),
    ]
    basis_cols = []
    for qnm in linear_qnms:
        omega = qnm.omega
        env = np.exp(omega.imag * t_pos)
        basis_cols.append(env * np.cos(omega.real * t_pos))
        basis_cols.append(env * np.sin(omega.real * t_pos))

    basis = np.column_stack(basis_cols)
    coeffs_td, _, _, _ = np.linalg.lstsq(basis, d_pos, rcond=None)

    a_220_td = np.sqrt(coeffs_td[0]**2 + coeffs_td[1]**2)
    phi_220_td = np.arctan2(-coeffs_td[1], coeffs_td[0])

    fitted_td = basis @ coeffs_td
    residual_td = d_pos - fitted_td

    # Build NL template in time domain
    omega_nl = mode_nl.omega
    nl_tmpl_td = a_220_td**2 * np.exp(omega_nl.imag * t_pos) * np.cos(
        omega_nl.real * t_pos + 2 * phi_220_td
    )

    # Time-domain matched filter
    rn_td = np.sum(residual_td * nl_tmpl_td)
    nn_td = np.sum(nl_tmpl_td * nl_tmpl_td)
    kappa_td = rn_td / nn_td if nn_td > 0 else 0.0

    print(f"A_220 = {a_220_td:.8f} (ratio: {a_220_td / A_220_inj:.6f})")
    print(f"phi_220 = {phi_220_td:.6f} (diff: {phi_220_td - phi_220_inj:.6f})")
    print(f"NL template A_220^2 = {a_220_td**2:.8e}")
    print(f"Injected NL A_220^2 = {A_220_inj**2:.8e}")
    print(f"<residual|template> = {rn_td:.8e}")
    print(f"<template|template> = {nn_td:.8e}")
    print(f"kappa_hat (TD) = {kappa_td:.8f}")
    print(f"Recovery ratio: {kappa_td / kappa_true:.6f}")
    print()

    # === TEST 3: FD vs TD comparison ===
    print("=" * 60)
    print("TEST 3: FD INNER PRODUCT ANALYSIS")
    print("=" * 60)
    basis_fft = np.fft.rfft(basis, axis=0)
    data_fft = np.fft.rfft(d_pos)
    noise_var = 1.0

    def fd_inner_current(a_fft, b_fft):
        """Current code's inner product."""
        return np.sum((a_fft.conj() * b_fft).real) / (n * noise_var)

    def fd_inner_correct(a_fft, b_fft):
        """Parseval-correct rfft inner product."""
        prod = (a_fft.conj() * b_fft).real
        result = prod[0] + prod[-1] + 2 * np.sum(prod[1:-1])
        return result / (n * noise_var)

    # Check Parseval ratio
    test_sig = d_pos
    test_fft = np.fft.rfft(test_sig)
    td_norm = np.sum(test_sig**2)
    fd_norm_current = fd_inner_current(test_fft, test_fft) * n * noise_var
    fd_norm_correct = fd_inner_correct(test_fft, test_fft) * n * noise_var
    print(f"Parseval check:")
    print(f"  TD sum(x^2) = {td_norm:.8e}")
    print(f"  FD current  = {fd_norm_current:.8e} (ratio: {fd_norm_current / td_norm:.6f})")
    print(f"  FD correct  = {fd_norm_correct:.8e} (ratio: {fd_norm_correct / td_norm:.6f})")
    print()

    # FD linear fit with current inner product
    n_basis = basis.shape[1]
    gram_fd = np.zeros((n_basis, n_basis))
    proj_fd = np.zeros(n_basis)
    for i in range(n_basis):
        for j in range(i, n_basis):
            val = fd_inner_current(basis_fft[:, i], basis_fft[:, j])
            gram_fd[i, j] = val
            gram_fd[j, i] = val
        proj_fd[i] = fd_inner_current(data_fft, basis_fft[:, i])

    coeffs_fd = np.linalg.solve(gram_fd, proj_fd)
    a_220_fd = np.sqrt(coeffs_fd[0]**2 + coeffs_fd[1]**2)
    phi_220_fd = np.arctan2(-coeffs_fd[1], coeffs_fd[0])

    fitted_fft = basis_fft @ coeffs_fd
    residual_fft = data_fft - fitted_fft

    nl_tmpl_fd = a_220_fd**2 * np.exp(omega_nl.imag * t_pos) * np.cos(
        omega_nl.real * t_pos + 2 * phi_220_fd
    )
    nl_fft = np.fft.rfft(nl_tmpl_fd)

    nn_fd = fd_inner_current(nl_fft, nl_fft)
    rn_fd = fd_inner_current(residual_fft, nl_fft)
    kappa_fd = rn_fd / nn_fd if nn_fd > 0 else 0

    print(f"FD fit (current code):")
    print(f"  A_220 = {a_220_fd:.8f} (ratio: {a_220_fd / A_220_inj:.6f})")
    print(f"  phi_220 = {phi_220_fd:.6f}")
    print(f"  kappa_hat = {kappa_fd:.8f}")
    print(f"  Recovery ratio: {kappa_fd / kappa_true:.6f}")
    print(f"  sigma = {1/np.sqrt(nn_fd) if nn_fd > 0 else float('inf'):.8f}")
    print()

    # === TEST 4: RESIDUAL CONTENT ===
    print("=" * 60)
    print("TEST 4: RESIDUAL CONTENT ANALYSIS")
    print("=" * 60)
    residual_td_power = np.sum(residual_td**2)
    nl_signal_td = kappa_true * A_220_inj**2 * np.exp(omega_nl.imag * t_pos) * np.cos(
        omega_nl.real * t_pos + 2 * phi_220_inj
    )
    nl_power = np.sum(nl_signal_td**2)
    linear_440_td = A_440_inj * np.exp(mode_440.omega.imag * t_pos) * np.cos(
        mode_440.omega.real * t_pos + fit_inj["440"]["phase"]
    )
    l440_power = np.sum(linear_440_td**2)

    print(f"Residual total power: {residual_td_power:.8e}")
    print(f"Expected NL power:    {nl_power:.8e}")
    print(f"Linear 440 power:     {l440_power:.8e}")
    print(f"NL / total residual: {nl_power / residual_td_power if residual_td_power > 0 else 0:.6f}")
    print(f"440 / total residual: {l440_power / residual_td_power if residual_td_power > 0 else 0:.6f}")
    print()

    nl_440_overlap = np.sum(nl_tmpl_td * linear_440_td)
    nl_nl_overlap = np.sum(nl_tmpl_td * nl_tmpl_td)
    print(f"<NL_template | linear_440>: {nl_440_overlap:.8e}")
    print(f"<NL_template | NL_template>: {nl_nl_overlap:.8e}")
    print(f"440 leakage fraction: {nl_440_overlap / nl_nl_overlap if nl_nl_overlap > 0 else 0:.6f}")
    print()

    # === TEST 5: NOISE + INJECTION ===
    print("=" * 60)
    print("TEST 5: NOISY RECOVERY (3 realizations)")
    print("=" * 60)
    rng = np.random.default_rng(42)

    from bown_instruments.grims.injection_campaign import _valid_noise_window_starts
    candidate_starts = _valid_noise_window_starts(noise_mask, n_samples)
    if len(candidate_starts) == 0:
        print("No valid noise windows!")
        return

    for real_idx in range(3):
        start_idx = rng.choice(candidate_starts)
        noise_segment = filtered[start_idx : start_idx + n_samples]
        data_injected = noise_segment + waveform

        result = phase_locked_search_colored(
            data_injected, t_dimless, spin, noise_rms,
            event_name=f"noisy_{real_idx}",
        )
        print(f"  Realization {real_idx}: kappa_hat={result.kappa_hat:+.6f}, "
              f"sigma={result.kappa_sigma:.6f}, "
              f"A_220={result.a_220_fit:.6f} (ratio={result.a_220_fit / A_220_inj:.4f}), "
              f"SNR={result.snr:+.4f}")


if __name__ == "__main__":
    phase3 = load_phase3_results()
    entries = sorted(
        phase3["individual"],
        key=lambda e: 1.0 / e["kappa_sigma"]**2 if e.get("kappa_sigma", 0) > 0 else 0,
        reverse=True,
    )

    event_name = entries[0]["event"]
    det = entries[0].get("detectors_used", ["H1"])[0]
    print(f"Using top-weight event: {event_name}, detector: {det}")
    print()

    diagnose_single_event(event_name, kappa_true=0.03, detector=det)
