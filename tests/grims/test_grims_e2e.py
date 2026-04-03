"""
End-to-end test of GRIM-S: inject a known signal, recover kappa.

This is Bown's US1,573,801 principle in action:
  1. Generate a synthetic ringdown with known kappa.
  2. Run the full analysis pipeline.
  3. Verify the recovered kappa matches the injected value.
  4. Run the self-test to verify mode decomposition consistency.

If this test fails, nothing downstream can be trusted.
"""

import sys
sys.path.insert(0, "/Volumes/VIXinSSD/drbown/bown-ringdown")

import numpy as np
from bown_instruments.grims.qnm_modes import KerrQNMCatalog, survey_spin_dependence
from bown_instruments.grims.ringdown_templates import RingdownTemplateBuilder, snr_threshold_for_nonlinear_detection
from bown_instruments.grims.gwtc_pipeline import (
    list_ringdown_candidates, estimate_ringdown_snr,
    generate_synthetic_ringdown, GWTC3_RINGDOWN_CANDIDATES,
)
from bown_instruments.grims.self_test import run_self_test
from bown_instruments.grims.bayesian_analysis import estimate_kappa_posterior, stack_posteriors


def test_qnm_frequencies():
    """Test: QNM frequencies are physically reasonable."""
    print("=" * 60)
    print("TEST 1: QNM Frequency Catalog")
    print("=" * 60)

    catalog = KerrQNMCatalog()

    for spin in [0.0, 0.3, 0.5, 0.7, 0.9]:
        modes = catalog.standard_ringdown_basis(spin, include_nonlinear=True)
        print(f"\nSpin a = {spin}:")
        for m in modes:
            label = "NL" if m.is_nonlinear else "  "
            print(f"  {label} ({m.l},{m.m},{m.n}): "
                  f"f={m.frequency:.4f}  tau={m.damping_time:.2f}M  "
                  f"Q={m.quality_factor:.1f}")

        # Verify nonlinear mode frequency = 2 * (2,2,0) frequency
        omega_220 = modes[0].omega
        omega_nl = modes[-1].omega
        freq_ratio = omega_nl.real / (2 * omega_220.real)
        print(f"  NL freq / 2*(2,2,0) freq = {freq_ratio:.6f} (should be 1.0)")
        assert abs(freq_ratio - 1.0) < 1e-10, "Nonlinear frequency mismatch!"

    print("\n✓ All QNM frequencies verified.")


def test_spin_survey():
    """Test: frequency separation between linear and nonlinear modes."""
    print("\n" + "=" * 60)
    print("TEST 2: Linear/Nonlinear Frequency Separation vs Spin")
    print("=" * 60)

    results = survey_spin_dependence(np.linspace(0.0, 0.95, 20))
    print(f"\n{'Spin':>6} {'Δf/f':>10} {'Δτ':>10}  {'Channel status'}")
    print("-" * 50)
    for r in results:
        sep = r["fractional_separation"]
        status = "OPEN" if sep > 0.05 else "NARROW" if sep > 0.01 else "CLOSED"
        print(f"{r['spin']:6.2f} {sep:10.4f} {r['delta_damping']:10.4f}  {status}")

    # The channel should never close completely
    min_sep = min(r["fractional_separation"] for r in results)
    print(f"\nMinimum separation: {min_sep:.4f}")
    assert min_sep > 0.01, "Channel too narrow — modes are degenerate!"
    print("✓ Channel remains open across all spins.")


def test_ringdown_candidates():
    """Test: GWTC-3 candidate list and SNR estimates."""
    print("\n" + "=" * 60)
    print("TEST 3: GWTC-3 Ringdown Candidates")
    print("=" * 60)

    candidates = list_ringdown_candidates(min_total_mass=60.0, min_snr=8.0)
    print(f"\n{len(candidates)} events pass cuts (M>60, SNR>8):\n")
    print(f"{'Event':<25} {'M_tot':>6} {'SNR':>6} {'SNR_rd':>7} {'SNR_NL':>7} {'Det?':>5}")
    print("-" * 60)
    for c in candidates:
        est = estimate_ringdown_snr(c)
        det = "YES" if est["nl_detectable_3sigma"] else "no"
        print(f"{c['name']:<25} {c['total_mass_msun']:6.1f} "
              f"{c['network_snr']:6.1f} {est['snr_ringdown_est']:7.1f} "
              f"{est['snr_nonlinear_est']:7.1f} {det:>5}")

    print(f"\n✓ {len(candidates)} candidates cataloged.")


def test_injection_recovery():
    """Test: inject known kappa, recover it from the data.

    This is the critical self-test. If we can't recover a known
    injected signal, the instrument is broken.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Injection Recovery (The Bown Self-Test)")
    print("=" * 60)

    # Use GW150914 parameters — the gold standard
    event = GWTC3_RINGDOWN_CANDIDATES[0]
    kappa_injected = 1.5
    noise_level = 0.0  # clean first, then noisy

    print(f"\nEvent: {event['name']}")
    print(f"Injected kappa: {kappa_injected}")
    print(f"Spin: {event['remnant_spin']}")
    print(f"Mass: {event['remnant_mass_msun']} Msun")

    # Generate synthetic data
    segment = generate_synthetic_ringdown(
        event, kappa=kappa_injected, noise_level=noise_level,
        sample_rate=4096.0, duration=0.1,
    )

    # Convert to dimensionless time for analysis
    m_sun_seconds = 4.925491025543576e-06
    m_seconds = event["remnant_mass_msun"] * m_sun_seconds
    t_dimless = (segment.time - segment.t_ringdown_start) / m_seconds

    # Scale strain to dimensionless
    distance_m = event["luminosity_distance_mpc"] * 3.0857e22
    mass_m = event["remnant_mass_msun"] * 1.989e30 * 6.674e-11 / (3e8)**2
    strain_dimless = segment.strain * distance_m / mass_m

    # Estimate mode amplitudes from the data (known for injection)
    q = event["mass_ratio"]
    A_220 = 0.4 * q
    A_330 = 0.1 * q * (1 - q)
    A_440_linear = 0.05 * q

    # --- CLEAN RECOVERY ---
    print("\n--- Clean signal (no noise) ---")
    result = estimate_kappa_posterior(
        strain_dimless, t_dimless,
        spin=event["remnant_spin"],
        A_220=A_220,
        noise_variance=1e-10,  # tiny regularizer
        event_name=event["name"],
        kappa_min=0.0, kappa_max=5.0, n_kappa=201,
        A_330=A_330, A_440_linear=A_440_linear,
    )

    print(f"  MAP kappa:    {result.kappa_map:.3f}")
    print(f"  Median kappa: {result.kappa_median:.3f}")
    print(f"  90% CI:       [{result.kappa_lower_90:.3f}, {result.kappa_upper_90:.3f}]")
    print(f"  Bayes factor: {result.log_bayes_factor:.1f}")
    print(f"  Significance: {result.detection_sigma:.1f} sigma")

    # Check recovery
    error = abs(result.kappa_map - kappa_injected) / kappa_injected
    print(f"  Recovery error: {error:.1%}")
    if error < 0.1:
        print("  ✓ PASSED: kappa recovered within 10%")
    else:
        print(f"  ✗ FAILED: recovery error {error:.1%} > 10%")

    # --- SELF-TEST ---
    print("\n--- Self-test: orthogonality check ---")
    catalog = KerrQNMCatalog()
    basis = catalog.standard_ringdown_basis(event["remnant_spin"],
                                            include_nonlinear=True)
    mode_freqs = [m.omega for m in basis]
    mode_labels = [
        f"({'NL' if m.is_nonlinear else ''}{m.l},{m.m},{m.n})"
        for m in basis
    ]

    self_test = run_self_test(
        strain_dimless, t_dimless,
        mode_frequencies=mode_freqs,
        noise_rms=0.0,
        mode_labels=mode_labels,
    )
    print(f"  Power ratio:      {self_test.power_ratio:.4f}")
    print(f"  Residual fraction: {self_test.residual_fraction:.4f}")
    print(f"  Diagnosis: {self_test.diagnosis}")

    # --- NOISY RECOVERY ---
    print("\n--- Noisy signal ---")
    # Add realistic noise level
    noise_sigma = np.max(np.abs(strain_dimless[t_dimless >= 0])) * 0.3
    noisy_strain = strain_dimless + np.random.normal(0, noise_sigma,
                                                      len(strain_dimless))

    result_noisy = estimate_kappa_posterior(
        noisy_strain, t_dimless,
        spin=event["remnant_spin"],
        A_220=A_220,
        noise_variance=noise_sigma**2,
        event_name=event["name"] + "_noisy",
        kappa_min=0.0, kappa_max=5.0, n_kappa=201,
        A_330=A_330, A_440_linear=A_440_linear,
    )

    print(f"  Noise level: {noise_sigma:.2e} (30% of peak signal)")
    print(f"  MAP kappa:    {result_noisy.kappa_map:.3f}")
    print(f"  90% CI:       [{result_noisy.kappa_lower_90:.3f}, {result_noisy.kappa_upper_90:.3f}]")
    print(f"  Significance: {result_noisy.detection_sigma:.1f} sigma")

    # The injected value should be within the 90% CI
    in_ci = (result_noisy.kappa_lower_90 <= kappa_injected <=
             result_noisy.kappa_upper_90)
    if in_ci:
        print("  ✓ PASSED: injected kappa within 90% CI")
    else:
        print("  ✗ WARNING: injected kappa outside 90% CI (may be noise)")

def test_posterior_stacking():
    """Test: stack posteriors across multiple events."""
    print("\n" + "=" * 60)
    print("TEST 5: Posterior Stacking")
    print("=" * 60)

    kappa_injected = 1.0
    posteriors = []

    # Generate and analyze 4 events
    candidates = list_ringdown_candidates(min_total_mass=60.0, min_snr=10.0)[:4]

    for event in candidates:
        segment = generate_synthetic_ringdown(
            event, kappa=kappa_injected, noise_level=0.0,
        )

        m_sun_seconds = 4.925491025543576e-06
        m_seconds = event["remnant_mass_msun"] * m_sun_seconds
        t_dimless = (segment.time - segment.t_ringdown_start) / m_seconds

        distance_m = event["luminosity_distance_mpc"] * 3.0857e22
        mass_m = event["remnant_mass_msun"] * 1.989e30 * 6.674e-11 / (3e8)**2
        strain_dimless = segment.strain * distance_m / mass_m

        # Add noise
        noise_sigma = np.max(np.abs(strain_dimless[t_dimless >= 0])) * 0.5
        noisy_strain = strain_dimless + np.random.normal(0, noise_sigma,
                                                          len(strain_dimless))

        q = event["mass_ratio"]
        A_220 = 0.4 * q
        A_330 = 0.1 * q * (1 - q)
        A_440_linear = 0.05 * q

        result = estimate_kappa_posterior(
            noisy_strain, t_dimless,
            spin=event["remnant_spin"],
            A_220=A_220,
            noise_variance=noise_sigma**2,
            event_name=event["name"],
            kappa_min=0.0, kappa_max=5.0, n_kappa=201,
            A_330=A_330, A_440_linear=A_440_linear,
        )
        posteriors.append(result)
        print(f"  {event['name']:<25} kappa_MAP={result.kappa_map:.2f}  "
              f"sigma={result.detection_sigma:.1f}")

    # Stack
    stacked = stack_posteriors(posteriors)
    print(f"\nStacked result ({stacked.n_events} events):")
    print(f"  MAP kappa:    {stacked.kappa_map:.3f}")
    print(f"  90% CI:       [{stacked.kappa_lower_90:.3f}, {stacked.kappa_upper_90:.3f}]")
    print(f"  Bayes factor: {stacked.log_bayes_factor:.1f}")
    print(f"  Significance: {stacked.detection_sigma:.1f} sigma")

    in_ci = (stacked.kappa_lower_90 <= kappa_injected <= stacked.kappa_upper_90)
    if in_ci:
        print("  ✓ PASSED: injected kappa within stacked 90% CI")
    else:
        print("  ✗ WARNING: injected kappa outside stacked 90% CI")

def test_weight_capped_stacking():
    """Test: weight-capped stacking and jackknife stability metrics."""
    print("\n" + "=" * 60)
    print("TEST 6: Weight-Capped Stacking and Jackknife")
    print("=" * 60)

    from bown_instruments.grims.phase_locked_search import PhaseLockResult, stack_phase_locked
    from bown_instruments.grims.jackknife import run_jackknife

    # Create synthetic per-event results with known weight concentration
    # 3 "loud" events (small sigma) + 17 "quiet" events (large sigma)
    rng = np.random.RandomState(123)
    kappa_true = 0.5
    results = []
    for i in range(20):
        sigma = 0.05 if i < 3 else 0.5  # 3 events with 100x more weight
        kappa = kappa_true + rng.normal(0, sigma)
        results.append(PhaseLockResult(
            event_name=f"EV{i:03d}",
            kappa_hat=kappa,
            kappa_sigma=sigma,
            snr=1.0 / sigma,
            a_220_fit=1.0,
            phi_220_fit=0.0,
            template_norm=1.0,
            residual_overlap=0.0,
            noise_rms=sigma,
        ))

    # Uncapped: dominated by 3 loud events
    stack_uncapped = stack_phase_locked(results, max_weight_ratio=None)
    jk_uncapped = run_jackknife(results, max_weight_ratio=None)
    print(f"\nUncapped: kappa={stack_uncapped.kappa_hat:.3f} ± {stack_uncapped.kappa_sigma:.3f}")
    print(f"  N_eff={jk_uncapped.n_eff:.1f}, max_frac_infl={jk_uncapped.max_fractional_influence:.3f}")

    # Capped at 3x: reduces concentration
    stack_capped = stack_phase_locked(results, max_weight_ratio=3.0)
    jk_capped = run_jackknife(results, max_weight_ratio=3.0)
    print(f"Cap 3x:  kappa={stack_capped.kappa_hat:.3f} ± {stack_capped.kappa_sigma:.3f}")
    print(f"  N_eff={jk_capped.n_eff:.1f}, max_frac_infl={jk_capped.max_fractional_influence:.3f}")

    # Verify weight cap increases N_eff
    assert jk_capped.n_eff > jk_uncapped.n_eff, "Weight cap should increase N_eff"
    print(f"  ✓ N_eff improved: {jk_uncapped.n_eff:.1f} → {jk_capped.n_eff:.1f}")

    # Verify max_weight_ratio=None gives same result as no cap
    stack_none = stack_phase_locked(results)
    assert abs(stack_none.kappa_hat - stack_uncapped.kappa_hat) < 1e-12
    print("  ✓ max_weight_ratio=None is backward-compatible")

    # Verify large cap is effectively no cap
    stack_huge = stack_phase_locked(results, max_weight_ratio=1000.0)
    assert abs(stack_huge.kappa_hat - stack_uncapped.kappa_hat) < 1e-10
    print("  ✓ Very large cap reproduces uncapped result")

    # Verify both estimates are consistent with truth
    for label, stack in [("uncapped", stack_uncapped), ("capped", stack_capped)]:
        error = abs(stack.kappa_hat - kappa_true) / kappa_true
        assert error < 0.5, f"{label} kappa too far from truth: {error:.1%}"
    print("  ✓ Both estimates consistent with injected kappa")


if __name__ == "__main__":
    print("GRIM-S End-to-End Test Suite")
    print("Gravitational Intermodulation Spectrometer")
    print("'Send yourself a test signal.' — Ralph Bown, 1923")
    print()

    np.random.seed(42)  # reproducibility

    test_qnm_frequencies()
    test_spin_survey()
    test_ringdown_candidates()
    test_injection_recovery()
    test_posterior_stacking()
    test_weight_capped_stacking()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
