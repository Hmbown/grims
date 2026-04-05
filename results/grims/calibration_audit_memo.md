# GRIM-S Calibration Audit — Final Verdict

**Date:** 2026-04-05
**Analyst:** Automated adversarial audit
**Status:** COMPLETE

## Verdict

**DO NOT USE the calibrated constraint.** The claimed |kappa| < 0.04 (95% CL)
is invalid because the kappa estimator has a 72% negative bias
that was not accounted for in the calibration.

## What Was Claimed

- Phase 3 stack: kappa = +0.0015 +/- 0.0571 (135 events)
- Injection pull_std ~ 0.38 (underdispersed by ~2.7x)
- Therefore sigma_calibrated ~ 0.021
- Therefore |kappa| < 0.04 at 95% CL

## What the Audit Found

### Finding 1: Sigma inflation is real (~3x)

The `shared_noise` t_start marginalization strategy quotes sigma ~ 3x larger
than the empirical scatter of the stacked estimator. This is by design — it
treats correlated t_start measurements conservatively.

| Scenario | sigma/scatter | Pull std |
|----------|--------------|----------|
| marginalized_default | 2.83x | 0.396 |
| fixed_tstart_10M | 0.75x | 1.23 |

### Finding 2: Estimator bias is severe (~72% signal loss)

The estimator recovers only **28%** of the injected kappa.

Recovery slope: b1 = 0.2841 +/- 0.0708
(95% CI: [0.141, 0.420])

| kappa_true | mean(kappa_hat) | Recovery |
|-----------|----------------|----------|
| 0.01 | 0.0023 | 23% |
| 0.02 | 0.0052 | 26% |
| 0.03 | 0.0081 | 27% |
| 0.04 | 0.0109 | 27% |
| 0.05 | 0.0137 | 27% |

This bias is present in BOTH marginalized and fixed-t_start scenarios,
confirming it is intrinsic to the estimator, not the marginalization.

### Finding 3: Naive calibration conflates kappa_hat with kappa_true

The claimed constraint |kappa| < 0.04 treats kappa_hat as if it equals
kappa_true. But since kappa_hat ~ 0.28 * kappa_true:

| Constraint method | 95% UL | On what? |
|-------------------|--------|----------|
| Uncalibrated | 0.113 | kappa_hat |
| Naive calibrated | 0.046 | kappa_hat |
| Proper (bias-corrected) | 0.146 | kappa_true |
| Conservative | 0.401 | kappa_true |

The "tight" constraint exists only for the biased estimator, not for
the physical coupling kappa_true.

### Finding 4: Statistical power is limited (12 realizations)

With only 12 injection realizations and 40/135 events, the calibration
has large statistical uncertainty. A publication-grade calibration would
require >= 100 realizations.

## Strongest Honest Result

```
kappa_hat = +0.0015 +/- 0.0571  (135 BBH events, shared_noise marginalization)
95% CI: [-0.110, +0.114]
Consistent with GR (kappa_GR ~ 0.032) at 0.5 sigma
```

This is a null result. The error bars are 2x larger
than the GR prediction, so the measurement does not yet constrain
GR-scale nonlinear coupling.

## What Would Make It Publishable

1. **Fix the estimator bias.** The ~70% signal loss likely comes from
   the NL template being partially degenerate with the linear (4,4,0)
   mode in the joint fit. This needs diagnosis and correction before
   any calibrated constraint is credible.

2. **Run >=100 injection realizations** to reduce bootstrap uncertainty
   on pull_std to <5%.

3. **Validate on pure-noise simulations** to confirm the null-hypothesis
   behavior is clean (unbiased at kappa=0, correct coverage).

4. **Report kappa_true, not kappa_hat**, with explicit bias correction
   and propagated uncertainty.

## Abstract Paragraph

> We search for the quadratic (4,4) nonlinear quasinormal mode coupling
> in 135 binary black hole merger ringdowns from GWTC-3 and O4a.
> Using a phase-locked matched-filter stack with shared-noise start-time
> marginalization, we measure kappa = +0.0015 +/- 0.057, consistent
> with general relativity at 0.5 sigma. Injection studies reveal that the
> estimator recovers 28% of the injected coupling, and the quoted
> uncertainty is 2.8x the empirical scatter due to conservative
> treatment of correlated start-time fits. After accounting for both
> effects, the 95% frequentist upper limit on the physical coupling is
> |kappa| < 0.15, which does not yet constrain GR-scale
> nonlinear mode coupling (kappa_GR ~ 0.03).

## File Paths

- Audit script: `scripts/grims/calibration_audit.py`
- Results JSON: `results/grims/calibration_audit_results.json`
- Results CSV: `results/grims/calibration_audit_table.csv`
- This memo: `results/grims/calibration_audit_memo.md`
- Injection campaign: `results/grims/phase3_injection_campaign_reduced_shared_noise_30ms.json`
- Phase 3 results: `results/grims/phase3_results.json`
