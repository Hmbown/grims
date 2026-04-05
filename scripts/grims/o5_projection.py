#!/usr/bin/env python3
"""
GRIM-S O4b/O5 Sensitivity Projection.

Projects when the nonlinear QNM coupling coefficient kappa can be
constrained at the GR-predicted level kappa_GR ~ 0.032, given:

  - Current Phase 3 results (135 BBH events, O3+O4a)
  - O4b projections (~100 additional events, similar sensitivity)
  - O5 projections (~2-3x better strain sensitivity, 200-500 BBH/year)

The key bottleneck is the ~28% recovery slope in the stacking estimator,
caused by noise-dependent weights correlated with the kappa estimator
through A_220.  Better per-event SNR in O5 reduces sigma_A/A_220,
which both improves per-event precision AND reduces the weight-bias
coupling.

Author: Hunter Bown (with projections based on GRIM-S Phase 3 pipeline)
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "grims"


# ---------------------------------------------------------------------------
# 1. Load current Phase 3 results
# ---------------------------------------------------------------------------

def load_phase3():
    """Load per-event results and stacked measurement."""
    with open(RESULTS_DIR / "phase3_results.json") as f:
        data = json.load(f)
    sigmas = np.array([r["kappa_sigma"] for r in data["individual"]])
    snrs = np.array([r["snr_event"] for r in data["individual"]])
    a220s = np.array([r["a_220_fit"] for r in data["individual"]])
    noises = np.array([r["noise_rms"] for r in data["individual"]])
    return {
        "sigmas": sigmas,
        "snrs": snrs,
        "a220s": a220s,
        "noises": noises,
        "stacked_kappa": data["stacked"]["kappa_hat"],
        "stacked_sigma": data["stacked"]["kappa_sigma"],
        "n_events": len(sigmas),
    }


# ---------------------------------------------------------------------------
# 2. Characterize the current per-event sensitivity distribution
# ---------------------------------------------------------------------------

def characterize_current(phase3):
    """Print statistics on the current per-event kappa_sigma distribution."""
    sigmas = phase3["sigmas"]
    n = len(sigmas)

    print("=" * 72)
    print("SECTION 1: Current Phase 3 Per-Event Sensitivity Distribution")
    print("=" * 72)
    print(f"  Total events analyzed:  {n}")
    print(f"  Stacked result:         kappa = {phase3['stacked_kappa']:+.4f}"
          f" +/- {phase3['stacked_sigma']:.4f}")
    print()

    # Distribution statistics
    print("  Per-event kappa_sigma distribution:")
    print(f"    min:      {sigmas.min():.3f}")
    print(f"    25th pct: {np.percentile(sigmas, 25):.3f}")
    print(f"    median:   {np.median(sigmas):.3f}")
    print(f"    75th pct: {np.percentile(sigmas, 75):.3f}")
    print(f"    max:      {sigmas.max():.1f}")
    print()

    # Fraction below thresholds
    thresholds = [0.2, 0.5, 1.0, 2.0, 5.0]
    print("  Events below sigma threshold:")
    for t in thresholds:
        count = np.sum(sigmas < t)
        print(f"    sigma < {t:<4.1f}:  {count:3d} events ({count/n*100:5.1f}%)")
    print()

    # Inverse-variance weighting diagnostics
    w = 1.0 / sigmas**2
    w_total = w.sum()
    sigma_ideal = 1.0 / np.sqrt(w_total)
    n_eff = w_total**2 / np.sum(w**2)

    print(f"  Inverse-variance stacking:")
    print(f"    Ideal stacked sigma:   {sigma_ideal:.4f}")
    print(f"    Actual stacked sigma:  {phase3['stacked_sigma']:.4f}")
    print(f"    Efficiency:            {sigma_ideal/phase3['stacked_sigma']*100:.1f}%")
    print(f"    N_eff (effective):     {n_eff:.1f} of {n} events")
    print()

    # Weight concentration
    w_sorted = np.sort(w)[::-1]
    w_cum = np.cumsum(w_sorted) / w_total
    for frac in [0.5, 0.8, 0.9]:
        n_needed = int(np.searchsorted(w_cum, frac)) + 1
        print(f"    {frac*100:.0f}% of weight from top {n_needed} events")
    print()

    # Fractional A_220 uncertainty (the key diagnostic)
    frac_unc = phase3["noises"] / phase3["a220s"]
    print(f"  Fractional A_220 uncertainty (sigma_A / A_220):")
    print(f"    median:  {np.median(frac_unc)*100:.0f}%")
    print(f"    mean:    {np.mean(frac_unc)*100:.0f}%")
    # For the top 22 events (sigma < 0.5)
    mask = sigmas < 0.5
    if mask.any():
        print(f"    median (sigma<0.5 events): {np.median(frac_unc[mask])*100:.0f}%")
    print()


# ---------------------------------------------------------------------------
# 3. Stacking model with bias correction
# ---------------------------------------------------------------------------

# Recovery slope from injection campaign (Phase 3, shared_noise, 30ms)
# This is the fraction of true kappa recovered by the inverse-variance
# weighted estimator, due to noise-dependent weight-estimate correlation.
CURRENT_RECOVERY_SLOPE = 0.28

# GR prediction for kappa from NR simulations
KAPPA_GR = 0.032


def compute_stacked_constraint(
    sigmas: np.ndarray,
    recovery_slope: float,
    max_weight_ratio: float = 5.5,
) -> dict:
    """Compute the bias-corrected stacked sigma for a set of per-event sigmas.

    The raw inverse-variance estimator has a multiplicative bias:
        <kappa_hat> = recovery_slope * kappa_true

    After bias correction (dividing by recovery_slope), the effective
    stacked sigma becomes sigma_raw / recovery_slope.

    Parameters
    ----------
    sigmas : per-event kappa uncertainties
    recovery_slope : fraction of true kappa recovered (0 < b1 <= 1)
    max_weight_ratio : cap on individual event weight / mean weight

    Returns
    -------
    dict with sigma_raw, sigma_calibrated, n_eff, etc.
    """
    w = 1.0 / sigmas**2

    # Apply weight cap (same as pipeline)
    if max_weight_ratio is not None:
        w_mean = w.mean()
        w = np.minimum(w, max_weight_ratio * w_mean)

    w_total = w.sum()
    sigma_raw = 1.0 / np.sqrt(w_total)
    sigma_calibrated = sigma_raw / recovery_slope
    n_eff = w_total**2 / np.sum(w**2)

    return {
        "sigma_raw": sigma_raw,
        "sigma_calibrated": sigma_calibrated,
        "recovery_slope": recovery_slope,
        "n_eff": n_eff,
        "n_events": len(sigmas),
    }


def recovery_slope_model(frac_uncertainty_A220: float) -> float:
    """Model how recovery slope improves as sigma_A/A_220 decreases.

    The bias arises because inverse-variance weights w_i ~ 1/sigma_i^2
    depend on noise through A_220, and the kappa estimator also depends
    on A_220.  When sigma_A/A_220 is large (>> 1), the weights are
    dominated by noise fluctuations in A_220, creating strong
    weight-estimate correlation.

    As sigma_A/A_220 -> 0, the weights become deterministic (set by the
    true signal amplitude), and the bias vanishes (slope -> 1).

    We model this empirically:
        recovery_slope = 1 / (1 + alpha * (sigma_A/A_220)^2)

    Calibrated against:
        - Current: sigma_A/A_220 ~ 0.50 (for sensitive events),
                   recovery_slope ~ 0.28
        - Limit: sigma_A/A_220 -> 0, recovery_slope -> 1.0

    Solving: alpha = (1/0.28 - 1) / 0.50^2 = 10.3
    """
    alpha = 10.3  # calibrated from Phase 3 injection campaign
    return 1.0 / (1.0 + alpha * frac_uncertainty_A220**2)


# ---------------------------------------------------------------------------
# 4. O4b and O5 event population models
# ---------------------------------------------------------------------------

def generate_o4b_population(phase3_sigmas: np.ndarray, n_new: int = 100,
                            rng: np.random.Generator = None) -> np.ndarray:
    """Generate O4b events: similar sensitivity to O4a.

    O4b has similar detector sensitivity to O4a, so we draw from the
    same per-event sigma distribution.  We bootstrap from the existing
    distribution (which already captures the astrophysical mass/spin
    distribution convolved with detector sensitivity).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Bootstrap from existing distribution
    return rng.choice(phase3_sigmas, size=n_new, replace=True)


def generate_o5_population(
    phase3_sigmas: np.ndarray,
    n_events: int,
    strain_improvement: float = 2.5,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate O5 events with improved strain sensitivity.

    O5 strain sensitivity is ~2-3x better than O4.

    For ringdown:
      - A_220 scales linearly with strain amplitude (for a given source)
      - noise_rms scales inversely with strain sensitivity
      - So A_220/noise_rms improves by strain_improvement^2 for the same source

    For kappa_sigma:
      - sigma_kappa ~ noise_rms / (A_220 * template_norm)
      - For the same source, sigma_kappa improves by ~strain_improvement^2

    We scale the existing per-event sigma distribution down by this factor,
    then bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # The sigma improvement factor for each event is strain^2 because
    # both numerator (noise) and denominator (A_220) improve by strain factor.
    improvement_factor = strain_improvement**2

    # Bootstrap and scale
    base_sigmas = rng.choice(phase3_sigmas, size=n_events, replace=True)
    return base_sigmas / improvement_factor


# ---------------------------------------------------------------------------
# 5. Run projections
# ---------------------------------------------------------------------------

def run_projections(phase3):
    """Run all projection scenarios."""
    sigmas_current = phase3["sigmas"]
    rng = np.random.default_rng(2026)

    print("=" * 72)
    print("SECTION 2: O4b Projection (adding ~100 events at O4a sensitivity)")
    print("=" * 72)
    print()

    # O4b: add 100 events at current sensitivity
    o4b_new = generate_o4b_population(sigmas_current, n_new=100, rng=rng)
    sigmas_o4b = np.concatenate([sigmas_current, o4b_new])

    result_current = compute_stacked_constraint(sigmas_current, CURRENT_RECOVERY_SLOPE)
    result_o4b = compute_stacked_constraint(sigmas_o4b, CURRENT_RECOVERY_SLOPE)

    print(f"  Current (O3+O4a, {len(sigmas_current)} events):")
    print(f"    Raw stacked sigma:        {result_current['sigma_raw']:.4f}")
    print(f"    Recovery slope:            {CURRENT_RECOVERY_SLOPE:.2f}")
    print(f"    Calibrated stacked sigma:  {result_current['sigma_calibrated']:.4f}")
    print(f"    |kappa_GR|/sigma:          {KAPPA_GR/result_current['sigma_calibrated']:.2f}")
    print()

    print(f"  O4b added ({len(sigmas_o4b)} total events):")
    print(f"    Raw stacked sigma:        {result_o4b['sigma_raw']:.4f}")
    print(f"    Calibrated stacked sigma:  {result_o4b['sigma_calibrated']:.4f}")
    print(f"    |kappa_GR|/sigma:          {KAPPA_GR/result_o4b['sigma_calibrated']:.2f}")
    print(f"    N_eff:                     {result_o4b['n_eff']:.1f}")
    print(f"    Improvement over current:  {result_current['sigma_calibrated']/result_o4b['sigma_calibrated']:.2f}x")
    print()
    print("  --> O4b alone is insufficient. Scaling as sqrt(N_eff), we need")
    n_needed_o4 = (result_current['sigma_calibrated'] / (KAPPA_GR / 2.0))**2 * result_current['n_eff']
    print(f"      ~{n_needed_o4:.0f} O4-sensitivity effective events for 2-sigma detection.")
    print(f"      That requires ~{n_needed_o4 * len(sigmas_current) / result_current['n_eff']:.0f}"
          f" raw events (given current N_eff/N ratio).")
    print()

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SECTION 3: O5 Projection")
    print("=" * 72)
    print()

    # Model parameters for O5
    strain_improvements = [2.0, 2.5, 3.0]
    o5_events_per_year = [200, 350, 500]
    o5_years = [1, 2, 3]

    print("  O5 detector improvements:")
    print("    Strain sensitivity: 2-3x better than O4")
    print("    Per-event sigma_kappa improvement: (strain)^2 = 4-9x")
    print("    Event rate: 200-500 BBH/year")
    print()

    # Compute recovery slope improvement for O5
    # Current fractional A_220 uncertainty for sensitive events (~top quartile)
    mask_good = sigmas_current < np.percentile(sigmas_current, 25)
    frac_unc_current = np.median(phase3["noises"][mask_good] / phase3["a220s"][mask_good])
    print(f"  Current sigma_A/A_220 (sensitive events): {frac_unc_current*100:.0f}%")
    print()

    print("  Recovery slope vs detector improvement:")
    print(f"  {'Strain':>8s}  {'sigma_A/A_220':>14s}  {'Recovery slope':>15s}  {'Slope improvement':>18s}")
    print(f"  {'------':>8s}  {'-'*14:>14s}  {'-'*15:>15s}  {'-'*18:>18s}")

    recovery_slopes = {}
    for si in strain_improvements:
        # A_220 increases by si, noise decreases by si -> fractional unc drops by si^2
        frac_unc_o5 = frac_unc_current / si**2
        slope = recovery_slope_model(frac_unc_o5)
        recovery_slopes[si] = slope
        print(f"  {si:8.1f}x  {frac_unc_o5*100:13.1f}%  {slope:15.3f}  {slope/CURRENT_RECOVERY_SLOPE:17.1f}x")
    print()

    # -----------------------------------------------------------------------
    # Main projection table
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SECTION 4: Projected Constraint on |kappa| (2-sigma upper limit)")
    print("=" * 72)
    print()
    print("  Target: kappa_GR ~ 0.032 detectable at 2-sigma")
    print("  (i.e., sigma_calibrated < 0.016)")
    print()

    # Header
    col_w = 12
    header = f"  {'Scenario':<40s}"
    header += f"{'N_events':>{col_w}s}"
    header += f"{'N_eff':>{col_w}s}"
    header += f"{'slope':>{col_w}s}"
    header += f"{'sig_raw':>{col_w}s}"
    header += f"{'sig_cal':>{col_w}s}"
    header += f"{'kGR/sig':>{col_w}s}"
    header += f"{'2sig UL':>{col_w}s}"
    print(header)
    print("  " + "-" * (40 + 7 * col_w))

    scenarios = []

    def add_scenario(name, sigmas, slope):
        r = compute_stacked_constraint(sigmas, slope)
        kgr_over_sig = KAPPA_GR / r["sigma_calibrated"]
        ul_2sig = 2.0 * r["sigma_calibrated"]
        scenarios.append((name, r, slope, kgr_over_sig, ul_2sig))
        row = f"  {name:<40s}"
        row += f"{r['n_events']:>{col_w}d}"
        row += f"{r['n_eff']:>{col_w}.1f}"
        row += f"{slope:>{col_w}.3f}"
        row += f"{r['sigma_raw']:>{col_w}.4f}"
        row += f"{r['sigma_calibrated']:>{col_w}.4f}"
        row += f"{kgr_over_sig:>{col_w}.2f}"
        row += f"{ul_2sig:>{col_w}.4f}"
        print(row)

    # Current
    add_scenario("Current (O3+O4a, 135 events)", sigmas_current, CURRENT_RECOVERY_SLOPE)

    # O4b
    add_scenario("+ O4b (235 events)", sigmas_o4b, CURRENT_RECOVERY_SLOPE)

    print()

    # O5 scenarios
    for si in strain_improvements:
        slope = recovery_slopes[si]
        for years in o5_years:
            for rate in [200, 500]:
                n_o5 = rate * years
                o5_sigmas = generate_o5_population(
                    sigmas_current, n_events=n_o5,
                    strain_improvement=si, rng=np.random.default_rng(2026 + int(si*100) + years*10 + rate)
                )
                # Combine with O4b pool
                combined = np.concatenate([sigmas_o4b, o5_sigmas])
                label = f"O4b + O5({si:.1f}x) {years}yr {rate}/yr = {n_o5}"
                add_scenario(label, combined, slope)

        print()

    # -----------------------------------------------------------------------
    # Find the minimum O5 events needed for 2-sigma detection
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SECTION 5: Minimum O5 Events for 2-sigma Detection of kappa_GR")
    print("=" * 72)
    print()

    target_sigma = KAPPA_GR / 2.0  # 0.016
    print(f"  Target: sigma_calibrated < {target_sigma:.4f}")
    print()

    print(f"  {'Strain improvement':<25s}  {'Recovery slope':>15s}  {'N_O5 needed':>12s}  {'Observing time':>15s}")
    print(f"  {'-'*25:<25s}  {'-'*15:>15s}  {'-'*12:>12s}  {'-'*15:>15s}")

    for si in strain_improvements:
        slope = recovery_slopes[si]

        # Binary search for N_O5 needed
        n_low, n_high = 1, 50000
        while n_high - n_low > 10:
            n_mid = (n_low + n_high) // 2
            o5_sigmas = generate_o5_population(
                sigmas_current, n_events=n_mid,
                strain_improvement=si,
                rng=np.random.default_rng(2026)
            )
            combined = np.concatenate([sigmas_o4b, o5_sigmas])
            r = compute_stacked_constraint(combined, slope)
            if r["sigma_calibrated"] <= target_sigma:
                n_high = n_mid
            else:
                n_low = n_mid

        # Observing time at 350 events/year (median estimate)
        years_needed = n_high / 350.0

        print(f"  {si:.1f}x{'':<22s}  {slope:15.3f}  {n_high:12d}  {years_needed:12.1f} yr")

    print()

    # -----------------------------------------------------------------------
    # Section 6: Effect of inspiral-informed A_220 prior
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SECTION 6: Impact of Inspiral-Informed A_220 Prior")
    print("=" * 72)
    print()
    print("  If inspiral waveform models provide an A_220 prior with ~10%")
    print("  fractional uncertainty, the weight-estimate bias is largely")
    print("  eliminated because weights become nearly deterministic.")
    print()

    # With inspiral prior, sigma_A/A_220 ~ 0.10 regardless of detector noise
    frac_unc_prior = 0.10
    slope_prior = recovery_slope_model(frac_unc_prior)
    print(f"  sigma_A/A_220 with prior:  {frac_unc_prior*100:.0f}%")
    print(f"  Recovery slope with prior:  {slope_prior:.3f}")
    print(f"  (vs current slope:          {CURRENT_RECOVERY_SLOPE:.3f})")
    print()

    # The per-event sigma also improves because we're not relying on
    # noisy A_220 estimates for the kappa extraction.  Model this as
    # sigma_kappa_prior ~ sigma_kappa_current * (0.10 / frac_unc_current)
    # but only for events where frac_unc > 0.10
    frac_uncs = phase3["noises"] / phase3["a220s"]

    # With prior, per-event sigma improves proportionally to the
    # fractional uncertainty improvement
    sigmas_with_prior = sigmas_current.copy()
    for i in range(len(sigmas_with_prior)):
        if frac_uncs[i] > frac_unc_prior:
            # sigma_kappa scales roughly as sigma_A / A_220 for the
            # noise-dependent part.  But kappa_sigma also has an
            # irreducible noise floor from detector noise in the NL band.
            # Conservatively: sigma improves by min(frac_unc/prior, 3)
            improvement = min(frac_uncs[i] / frac_unc_prior, 3.0)
            sigmas_with_prior[i] /= improvement

    print(f"  {'Scenario':<45s}  {'sig_cal':>8s}  {'kGR/sig':>8s}  {'2sig UL':>8s}")
    print(f"  {'-'*45:<45s}  {'-'*8:>8s}  {'-'*8:>8s}  {'-'*8:>8s}")

    # Current events with prior
    r = compute_stacked_constraint(sigmas_with_prior, slope_prior)
    kgr_s = KAPPA_GR / r["sigma_calibrated"]
    print(f"  {'Current 135 events + inspiral prior':<45s}"
          f"  {r['sigma_calibrated']:8.4f}  {kgr_s:8.2f}  {2*r['sigma_calibrated']:8.4f}")

    # O4b events with prior
    o4b_with_prior = np.concatenate([sigmas_with_prior,
                                      generate_o4b_population(sigmas_with_prior, 100, rng=np.random.default_rng(99))])
    r = compute_stacked_constraint(o4b_with_prior, slope_prior)
    kgr_s = KAPPA_GR / r["sigma_calibrated"]
    print(f"  {'O4b 235 events + inspiral prior':<45s}"
          f"  {r['sigma_calibrated']:8.4f}  {kgr_s:8.2f}  {2*r['sigma_calibrated']:8.4f}")

    # O5 with prior
    for si in [2.0, 2.5, 3.0]:
        for n_o5 in [200, 700, 1000]:
            o5_sigs = generate_o5_population(
                sigmas_with_prior, n_events=n_o5,
                strain_improvement=si,
                rng=np.random.default_rng(42 + int(si*100) + n_o5)
            )
            combined = np.concatenate([o4b_with_prior, o5_sigs])
            # O5 + prior: fractional uncertainty even lower
            frac_unc_o5_prior = min(frac_unc_prior, frac_unc_current / si**2)
            slope_o5_prior = recovery_slope_model(frac_unc_o5_prior)
            r = compute_stacked_constraint(combined, slope_o5_prior)
            kgr_s = KAPPA_GR / r["sigma_calibrated"]
            label = f"O4b + O5({si:.0f}x) {n_o5} events + prior"
            print(f"  {label:<45s}"
                  f"  {r['sigma_calibrated']:8.4f}  {kgr_s:8.2f}  {2*r['sigma_calibrated']:8.4f}")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SECTION 7: Summary and Key Findings")
    print("=" * 72)
    print()
    print("  The measurement kappa = +0.005 +/- 0.20 (bias-corrected) is 6x too")
    print("  weak to constrain kappa_GR ~ 0.032.  The bottleneck is twofold:")
    print()
    print("  1. RECOVERY SLOPE (28%): The inverse-variance weighted estimator")
    print("     recovers only 28% of the true kappa because noise fluctuations")
    print("     in A_220 correlate the weights with the estimator.  This is a")
    print("     ~3.6x penalty on the effective sigma.")
    print()
    print("  2. PER-EVENT PRECISION: Only 22 of 135 events (16%) have")
    print("     sigma_kappa < 0.5.  The effective sample size is ~14 events.")
    print()
    print("  PATH TO DETECTION:")
    print()
    print("  A. Without inspiral prior (noise-only estimation of A_220):")
    print("     - O4b adds ~100 events but only shrinks sigma by ~sqrt(2).")
    print("       Still ~4x too weak.")
    print("     - O5 at 2.5x strain improvement gives ~6x better per-event")
    print("       sigma AND recovery slope improves from 0.28 to ~0.90+.")
    print("     - ~200-500 O5 events (1 year) likely sufficient for 2-sigma")
    print("       detection, depending on exact strain improvement.")
    print()
    print("  B. With inspiral-informed A_220 prior (sigma_A/A_220 ~ 10%):")
    print("     - Recovery slope jumps to ~0.91, nearly eliminating the bias.")
    print("     - Even with current O4a data, sigma drops to ~0.02-0.04 range.")
    print("     - O4b data with inspiral prior may be sufficient for ~2-sigma.")
    print("     - This is the HIGHEST-LEVERAGE near-term improvement.")
    print()
    print("  RECOMMENDATION: Prioritize implementing the inspiral-informed A_220")
    print("  prior.  This converts the existing 135-event dataset from a null")
    print("  result into a near-detection, and makes O4b data immediately useful.")
    print("  O5 then provides the definitive measurement with sigma << kappa_GR.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    phase3 = load_phase3()
    characterize_current(phase3)
    print()
    run_projections(phase3)
