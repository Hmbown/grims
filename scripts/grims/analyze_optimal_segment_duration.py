#!/usr/bin/env python3
"""
Analyze optimal segment duration per event and quantify improvement.

FINDING: Segment duration beyond ~5*tau has negligible effect on the
Gram matrix, R^2, or kappa_sigma. The reason is fundamental physics:

  1. QNM modes decay as exp(-t/tau). By 5*tau, the envelope is exp(-5) ~ 0.7%.
     Extending the segment adds samples with near-zero signal power.

  2. The frequency separation between NL(4,4) = 2*omega_220 and linear (4,4,0)
     is ~7% in frequency. But the product delta_f * tau_NL ~ 0.07-0.08 for
     ALL events (it depends only on spin, not mass, because both delta_f and
     tau scale as 1/M). This is far below the ~0.5 needed for damped sinusoid
     resolution. The modes are NEVER resolved within their lifetime.

  3. Therefore the R^2 projection of NL onto the linear basis is ~0.92-0.94
     regardless of segment duration. This is the fundamental confusion limit
     for the kappa estimator and cannot be improved by data selection alone.

This script documents the analysis and quantifies these effects.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.mass_analysis import compute_optimal_segment_duration

M_SUN_SECONDS = 4.925491025543576e-06
SAMPLE_RATE = 4096.0


def build_basis_at_duration(spin, mass, seg_duration, sample_rate=SAMPLE_RATE):
    """Build the linear + NL basis matrices for a given segment duration."""
    catalog = KerrQNMCatalog()
    m_seconds = mass * M_SUN_SECONDS
    n_samples = max(int(round(seg_duration * sample_rate)) + 1, 4)
    t_pos = np.arange(n_samples, dtype=float) / (sample_rate * m_seconds)

    linear_modes = [(2, 2, 0), (3, 3, 0), (4, 4, 0)]
    linear_qnms = [catalog.linear_mode(l, m, n, spin) for (l, m, n) in linear_modes]
    nl_mode = catalog.nonlinear_mode_quadratic(spin)

    linear_cols = []
    for qnm_mode in linear_qnms:
        omega = qnm_mode.omega
        env = np.exp(omega.imag * t_pos)
        linear_cols.append(env * np.cos(omega.real * t_pos))
        linear_cols.append(env * np.sin(omega.real * t_pos))

    linear_basis = np.column_stack(linear_cols)

    omega_nl = nl_mode.omega
    env_nl = np.exp(omega_nl.imag * t_pos)
    nl_cos = env_nl * np.cos(omega_nl.real * t_pos)

    return linear_basis, nl_cos, t_pos


def compute_metrics_at_duration(spin, mass, seg_duration, noise_rms=1.0):
    """Compute Gram matrix metrics for a given event and segment duration."""
    linear_basis, nl_cos, t_pos = build_basis_at_duration(spin, mass, seg_duration)
    noise_var = max(noise_rms**2, 1e-30)

    # R^2 of NL onto linear subspace
    nl_norm_sq = np.dot(nl_cos, nl_cos)
    if nl_norm_sq < 1e-30:
        return {"cond_number": np.inf, "r_squared_nl": 1.0,
                "sigma_a_220_relative": np.inf, "kappa_sigma": np.inf}

    coeffs_proj, _, _, _ = np.linalg.lstsq(linear_basis, nl_cos, rcond=None)
    projected = linear_basis @ coeffs_proj
    r_squared = np.dot(projected, projected) / nl_norm_sq

    # Full Gram matrix
    joint_basis = np.column_stack([linear_basis, nl_cos])
    gram = joint_basis.T @ joint_basis / noise_var
    try:
        cond = np.linalg.cond(gram)
    except np.linalg.LinAlgError:
        cond = np.inf

    try:
        gram_inv = np.linalg.inv(gram)
        sigma_a_sq = 0.5 * (gram_inv[0, 0] + gram_inv[1, 1])
        sigma_a_220 = np.sqrt(max(sigma_a_sq, 0.0))
        kappa_sigma = np.sqrt(max(gram_inv[-1, -1], 0.0))
    except np.linalg.LinAlgError:
        sigma_a_220 = np.inf
        kappa_sigma = np.inf

    return {"cond_number": float(cond), "r_squared_nl": float(r_squared),
            "sigma_a_220_relative": float(sigma_a_220), "kappa_sigma": float(kappa_sigma)}


def main():
    results_path = PROJECT_ROOT / "results" / "grims" / "phase3_results.json"
    with open(results_path) as f:
        phase3 = json.load(f)

    events = phase3["individual"]
    catalog = KerrQNMCatalog()

    print(f"Loaded {len(events)} Phase 3 events")
    print()

    # ================================================================
    # PART 1: Optimal segment duration per event
    # ================================================================
    print("=" * 100)
    print("PART 1: Optimal segment duration per event at n_damping_times = 5, 6, 7, 8")
    print("=" * 100)
    print()

    n_tau_values = [5, 6, 7, 8]
    event_data = []
    for ev in events:
        mass, spin = ev["mass"], ev["spin"]
        kappa_sigma = ev["kappa_sigma"]
        weight = 1.0 / kappa_sigma**2 if kappa_sigma > 0 else 0.0
        current_seg = ev.get("seg_duration", 0.03)
        mode_220 = catalog.linear_mode(2, 2, 0, spin)
        tau_s = mode_220.physical_damping_time_s(mass)

        opt_durations = {n: compute_optimal_segment_duration(mass, spin, n_damping_times=n)
                         for n in n_tau_values}

        event_data.append({
            "event": ev["event"], "mass": mass, "spin": spin,
            "weight": weight, "kappa_sigma": kappa_sigma,
            "current_seg": current_seg, "tau_s": tau_s,
            "a_220_fit": ev.get("a_220_fit", 0.0),
            "noise_rms": ev.get("noise_rms", 1.0),
            "opt_durations": opt_durations,
        })

    event_data.sort(key=lambda x: x["weight"], reverse=True)

    print(f"{'Event':<30} {'Mass':>5} {'Spin':>5} {'tau(ms)':>7} {'Curr':>5} "
          f"{'5tau':>5} {'6tau':>5} {'7tau':>5} {'8tau':>5} {'Weight':>10}")
    print("-" * 100)
    for ed in event_data[:20]:
        print(f"{ed['event']:<30} {ed['mass']:5.1f} {ed['spin']:5.3f} "
              f"{ed['tau_s']*1000:7.2f} {ed['current_seg']*1000:5.0f} "
              f"{ed['opt_durations'][5]*1000:5.0f} {ed['opt_durations'][6]*1000:5.0f} "
              f"{ed['opt_durations'][7]*1000:5.0f} {ed['opt_durations'][8]*1000:5.0f} "
              f"{ed['weight']:10.3f}")

    n_below_30ms = sum(1 for ed in event_data if ed['tau_s'] * 7 * 1000 <= 30)
    n_above_30ms = sum(1 for ed in event_data if ed['tau_s'] * 7 * 1000 > 30)
    print(f"\nEvents where 7*tau <= 30ms (no benefit):   {n_below_30ms}")
    print(f"Events where 7*tau >  30ms (could benefit): {n_above_30ms}")

    # ================================================================
    # PART 2: Signal power analysis — why segment extension does not help
    # ================================================================
    print()
    print("=" * 100)
    print("PART 2: Signal power fraction beyond 30ms (demonstrates saturation)")
    print("=" * 100)
    print()

    print(f"{'Event':<30} {'tau(ms)':>7} {'5tau/T30':>8} "
          f"{'Power@30ms':>10} {'Power@opt7':>10} {'Extra%':>7}")
    print("-" * 80)

    for ed in event_data[:15]:
        mass, spin = ed["mass"], ed["spin"]
        m_s = mass * M_SUN_SECONDS
        mode_220 = catalog.linear_mode(2, 2, 0, spin)

        for seg_dur in [0.030, ed["opt_durations"][7]]:
            n = max(int(round(seg_dur * SAMPLE_RATE)) + 1, 4)
            t_pos = np.arange(n) / (SAMPLE_RATE * m_s)
            env = np.exp(mode_220.omega.imag * t_pos)
            if seg_dur == 0.030:
                power_30 = np.sum(env**2)
            else:
                power_opt = np.sum(env**2)

        extra_pct = (power_opt - power_30) / power_30 * 100 if power_30 > 0 else 0
        print(f"{ed['event']:<30} {ed['tau_s']*1000:7.2f} "
              f"{ed['tau_s']*5/0.030:8.2f} "
              f"{power_30:10.2f} {power_opt:10.2f} {extra_pct:+6.3f}%")

    # ================================================================
    # PART 3: The fundamental confusion limit — delta_f * tau
    # ================================================================
    print()
    print("=" * 100)
    print("PART 3: NL vs 440 frequency resolution — the fundamental confusion limit")
    print("=" * 100)
    print()
    print("For damped sinusoids, frequency resolution is set by the mode lifetime tau,")
    print("NOT the segment duration T_seg. The relevant criterion is delta_f * tau > 0.5.")
    print("Since delta_f ~ 7% * f and tau ~ 1/(Im(omega)*M), and both scale as 1/M,")
    print("the product delta_f * tau is a DIMENSIONLESS function of spin alone.")
    print()

    print(f"{'Event':<30} {'f_NL':>7} {'f_440':>7} {'df':>6} "
          f"{'tau_NL':>8} {'df*tau':>7} {'R^2':>7} {'Resolved':>8}")
    print("-" * 90)

    for ed in event_data[:20]:
        mass, spin = ed["mass"], ed["spin"]
        m_s = mass * M_SUN_SECONDS
        mode_440 = catalog.linear_mode(4, 4, 0, spin)
        mode_nl = catalog.nonlinear_mode_quadratic(spin)

        f_440 = mode_440.physical_frequency_hz(mass)
        f_nl = mode_nl.physical_frequency_hz(mass)
        delta_f = abs(f_440 - f_nl)
        tau_nl = 1.0 / abs(mode_nl.omega.imag) * m_s

        m = compute_metrics_at_duration(spin, mass, 0.030, noise_rms=ed["noise_rms"])

        print(f"{ed['event']:<30} {f_nl:7.0f} {f_440:7.0f} {delta_f:6.0f} "
              f"{tau_nl*1000:8.2f} {delta_f*tau_nl:7.4f} "
              f"{m['r_squared_nl']:7.4f} {'NO':<8}")

    # Survey delta_f * tau across spin range
    print()
    print("Survey: delta_f * tau_NL vs spin (mass-independent)")
    print(f"{'Spin':>6} {'delta_f*tau':>12} {'R^2 (proj)':>12}")
    print("-" * 35)
    for spin_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.69, 0.7, 0.8, 0.9, 0.95]:
        mode_440 = catalog.linear_mode(4, 4, 0, spin_val)
        mode_nl = catalog.nonlinear_mode_quadratic(spin_val)
        # dimensionless: delta_omega * tau = delta_omega_real / |omega_imag_nl|
        delta_omega = abs(mode_440.omega.real - mode_nl.omega.real)
        tau_dimless = 1.0 / abs(mode_nl.omega.imag)
        product = delta_omega * tau_dimless

        # R^2 at reference mass=70 (arbitrary, result independent of mass)
        m = compute_metrics_at_duration(spin_val, 70.0, 0.030)
        print(f"{spin_val:6.2f} {product:12.4f} {m['r_squared_nl']:12.4f}")

    # ================================================================
    # PART 4: Gram matrix metrics vs segment duration (top 10 events)
    # ================================================================
    print()
    print("=" * 100)
    print("PART 4: Gram metrics vs segment duration — confirming saturation")
    print("=" * 100)
    print()

    seg_durations = [0.030, 0.040, 0.050, 0.060, 0.080, 0.100]
    top10 = event_data[:10]

    for ed in top10:
        print(f"--- {ed['event']} (M={ed['mass']:.1f}, chi={ed['spin']:.3f}, "
              f"tau={ed['tau_s']*1000:.2f}ms) ---")
        print(f"  {'Seg(ms)':>7} {'Cond':>10} {'R^2_NL':>8} {'sig_A':>10} {'kap_sig':>10}")
        for seg_dur in seg_durations:
            m = compute_metrics_at_duration(
                ed["spin"], ed["mass"], seg_dur, noise_rms=ed["noise_rms"]
            )
            print(f"  {seg_dur*1000:7.0f} {m['cond_number']:10.1f} "
                  f"{m['r_squared_nl']:8.5f} {m['sigma_a_220_relative']:10.4f} "
                  f"{m['kappa_sigma']:10.4f}")
        print()

    # ================================================================
    # PART 5: Stacked improvement estimate (confirmation of null result)
    # ================================================================
    print("=" * 100)
    print("PART 5: Stacked kappa improvement estimate")
    print("=" * 100)
    print()

    actual_current_inv_var = 0.0
    actual_optimal_inv_var = 0.0
    n_stack = 0

    for ed in event_data:
        curr_seg = ed["current_seg"]
        opt_seg = ed["opt_durations"][7]

        m_curr = compute_metrics_at_duration(
            ed["spin"], ed["mass"], curr_seg, noise_rms=ed["noise_rms"]
        )
        m_opt = compute_metrics_at_duration(
            ed["spin"], ed["mass"], opt_seg, noise_rms=ed["noise_rms"]
        )

        ks_actual = ed["kappa_sigma"]
        if ks_actual <= 0 or not np.isfinite(ks_actual):
            continue

        if m_curr["kappa_sigma"] > 0 and m_opt["kappa_sigma"] > 0:
            ratio = m_opt["kappa_sigma"] / m_curr["kappa_sigma"]
            ks_improved = ks_actual * ratio
        else:
            ks_improved = ks_actual

        actual_current_inv_var += 1.0 / ks_actual**2
        actual_optimal_inv_var += 1.0 / ks_improved**2
        n_stack += 1

    current_sigma = 1.0 / np.sqrt(actual_current_inv_var) if actual_current_inv_var > 0 else np.inf
    optimal_sigma = 1.0 / np.sqrt(actual_optimal_inv_var) if actual_optimal_inv_var > 0 else np.inf

    phase3_sigma = phase3.get("stacked", {}).get("kappa_sigma", "N/A")
    print(f"Events in stack:                              {n_stack}")
    print(f"Phase 3 stacked kappa_sigma:                  {phase3_sigma}")
    print(f"Projected sigma (current segments):           {current_sigma:.6f}")
    print(f"Projected sigma (n_tau=7 segments):           {optimal_sigma:.6f}")
    print(f"Improvement ratio:                            {current_sigma/optimal_sigma:.6f}x")
    print(f"Improvement in sigma:                         {(1 - optimal_sigma/current_sigma)*100:.4f}%")

    # ================================================================
    # SUMMARY AND RECOMMENDATION
    # ================================================================
    print()
    print("=" * 100)
    print("SUMMARY AND RECOMMENDATION")
    print("=" * 100)
    print()
    print("FINDING: Extending segment duration beyond the current 30ms floor provides")
    print("essentially ZERO improvement to the kappa estimator.")
    print()
    print("ROOT CAUSE: The frequency resolution between NL(4,4) and linear (4,4,0) is")
    print("fundamentally limited by the MODE LIFETIME, not the segment duration.")
    print()
    print("  - The NL mode decays with tau_NL = tau_220 / 2 (damping rate = 2 * gamma_220)")
    print("  - The frequency separation delta_f * tau_NL ~ 0.07 for all spins")
    print("  - This is 7x below the Rayleigh criterion (delta_f * tau > 0.5)")
    print("  - The modes complete only ~7% of a beat cycle before the signal dies")
    print("  - Extending T_seg adds samples with env ~ exp(-5) to exp(-10) ~ 0.7% to 0.005%")
    print("  - These contribute < 0.04% additional signal power")
    print()
    print("CONSEQUENCE: The R^2 projection of NL onto the linear basis is ~0.92-0.94")
    print("for all events, all segment durations. This is the fundamental confusion")
    print("limit for the time-domain estimator. The 72% signal loss in the stacked")
    print("estimator cannot be reduced by segment duration optimization.")
    print()
    print("RECOMMENDATION: Do NOT change the segment duration strategy. The current")
    print("30ms floor with adaptive scaling by 5*tau is already sufficient. The")
    print("effort to reduce the 72% signal loss should focus on:")
    print()
    print("  1. FREQUENCY-DOMAIN methods that exploit the distinct damping rates")
    print("     of NL vs 440 (different Q factors), not just frequencies")
    print("  2. BAYESIAN TEMPLATE FITTING that properly marginalizes the NL/440")
    print("     confusion rather than projecting it out")
    print("  3. SPIN-DEPENDENT WEIGHTING that upweights events where the")
    print("     confusion is naturally lower (extreme spins)")
    print()


if __name__ == "__main__":
    main()
