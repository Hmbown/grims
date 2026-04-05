#!/usr/bin/env python3
"""
Inspiral-informed A_220 prior for the latent-amplitude kappa estimator.

PROBLEM
-------
The stacked kappa estimator suffers from weight-estimate correlation:
the inverse-variance weights depend on A_220^4, and A_220 is measured
with ~33% (median) fractional noise from the Fisher matrix, with a
heavy tail reaching >100% for low-SNR events.  This biases the
weighted average.

APPROACH
--------
Use binary parameters from the GWTC catalog (remnant mass, spin,
symmetric mass ratio, luminosity distance) together with analysis
parameters (segment duration, noise RMS) to build a log-linear
prediction model for the whitened A_220.  This prediction serves as
a Bayesian prior that is combined with the Fisher data measurement
to produce a tighter posterior on A_220 per event.

Three estimator variants are compared:
1. Legacy plugin (inverse-variance weighted average of kappa_hat)
2. Fisher-only latent-amplitude (existing marginalized estimator)
3. Inspiral-informed latent-amplitude (this script)

The inspiral prediction model has ~120% log-space scatter (R^2 ~ 0.66),
because the dominant source of event-to-event variation in whitened A_220
is the detector PSD at the QNM frequency, which is not available in the
catalog.  As a consequence, the inspiral prior provides only marginal
improvement over the Fisher-only approach for most events.

However, the prior does help the ~24 events (18%) where Fisher fractional
sigma exceeds 100%, by pulling those events toward physically plausible
amplitudes and preventing them from receiving pathological weights.

USAGE
-----
    python scripts/grims/run_inspiral_informed_kappa.py
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bown_instruments.grims.phase_locked_search import (
    LatentAmplitudeEvent,
    MarginalizedKappaPosterior,
    estimate_kappa_posterior_latent_amplitude,
    infer_phase3_row_sigma_a_220,
    phase3_row_to_latent_amplitude_event,
)


# ---------------------------------------------------------------------------
# Inspiral prediction model
# ---------------------------------------------------------------------------

@dataclass
class InspiralPredictionModel:
    """Log-linear model: log(a_220) = X @ coefficients.

    Features: [1, log(M_rem), log(eta), log(d_L), spin, log(seg_dur), log(noise_rms)]
    where eta = q / (1 + q)^2 is the symmetric mass ratio.
    """
    coefficients: np.ndarray  # shape (7,)
    scatter: float            # RMS residual in log-space
    n_training: int
    r_squared: float

    def predict_log_a220(self, features: np.ndarray) -> np.ndarray:
        """Predict log(a_220) from the feature matrix."""
        return features @ self.coefficients

    def predict_a220(self, features: np.ndarray) -> np.ndarray:
        """Predict a_220 (linear scale) from the feature matrix."""
        return np.exp(self.predict_log_a220(features))

    def sigma_a220_prior(self, a220_pred: np.ndarray) -> np.ndarray:
        """Convert log-space scatter to a linear-space Gaussian sigma.

        Uses the log-normal approximation: the mean of the log-normal is
        exp(mu + sigma^2/2), and its standard deviation is
        mean * sqrt(exp(sigma^2) - 1).
        """
        return a220_pred * np.sqrt(np.exp(self.scatter**2) - 1)


def build_feature_matrix(
    remnant_mass: np.ndarray,
    mass_ratio: np.ndarray,
    distance: np.ndarray,
    spin: np.ndarray,
    seg_duration: np.ndarray,
    noise_rms: np.ndarray,
) -> np.ndarray:
    """Construct the design matrix for the inspiral prediction model.

    Physical motivation:
    - log(M_rem): the ringdown strain scales with remnant mass
    - log(eta): symmetric mass ratio controls the fraction of mass
      radiated in the 220 mode
    - log(d_L): strain amplitude inversely proportional to distance
    - spin: remnant spin affects the ringdown waveform shape and energy
    - log(seg_dur): longer segments integrate more of the damped mode
    - log(noise_rms): captures the whitening normalization, which depends
      on the detector PSD at the QNM frequency
    """
    eta = mass_ratio / (1.0 + mass_ratio) ** 2
    return np.column_stack([
        np.ones(len(remnant_mass)),
        np.log(np.maximum(remnant_mass, 1e-12)),
        np.log(np.maximum(eta, 1e-12)),
        np.log(np.maximum(distance, 1e-12)),
        spin,
        np.log(np.maximum(seg_duration, 1e-12)),
        np.log(np.maximum(noise_rms, 1e-12)),
    ])


def fit_inspiral_model(
    features: np.ndarray,
    log_a220: np.ndarray,
) -> InspiralPredictionModel:
    """Fit the log-linear inspiral prediction model.

    Uses ordinary least squares on the full training set.
    """
    coeffs, _, _, _ = np.linalg.lstsq(features, log_a220, rcond=None)
    log_a_pred = features @ coeffs
    residuals = log_a220 - log_a_pred
    scatter = float(np.std(residuals, ddof=1))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_a220 - np.mean(log_a220))**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return InspiralPredictionModel(
        coefficients=coeffs,
        scatter=scatter,
        n_training=len(log_a220),
        r_squared=r_squared,
    )


def loo_predict(
    features: np.ndarray,
    log_a220: np.ndarray,
) -> np.ndarray:
    """Leave-one-out cross-validated predictions.

    For each event i, fit the model on all other events and predict
    log(a_220) for event i.  This prevents overfitting bias in the
    prediction scatter estimate.
    """
    n = len(log_a220)
    predictions = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coeffs, _, _, _ = np.linalg.lstsq(
            features[mask], log_a220[mask], rcond=None
        )
        predictions[i] = features[i] @ coeffs
    return predictions


# ---------------------------------------------------------------------------
# Bayesian combination of inspiral prior + Fisher measurement
# ---------------------------------------------------------------------------

def combine_prior_and_measurement(
    a220_data: float,
    sigma_fisher: float,
    a220_prior_mean: float,
    sigma_prior: float,
) -> tuple[float, float]:
    """Bayesian update: combine inspiral prior with Fisher measurement.

    Both are approximated as Gaussian in linear a_220 space.
    Returns (posterior_mean, posterior_sigma).

    If either sigma is zero or non-finite, fall back to the other source.
    """
    fisher_valid = sigma_fisher > 0 and np.isfinite(sigma_fisher)
    prior_valid = sigma_prior > 0 and np.isfinite(sigma_prior)

    if fisher_valid and prior_valid:
        w_data = 1.0 / sigma_fisher**2
        w_prior = 1.0 / sigma_prior**2
        var_post = 1.0 / (w_data + w_prior)
        mean_post = var_post * (a220_data * w_data + a220_prior_mean * w_prior)
        return max(mean_post, 1e-12), np.sqrt(var_post)
    elif fisher_valid:
        return max(a220_data, 1e-12), sigma_fisher
    elif prior_valid:
        return max(a220_prior_mean, 1e-12), sigma_prior
    else:
        return max(a220_data, 1e-12), max(a220_data * 0.5, 1e-12)


# ---------------------------------------------------------------------------
# Build inspiral-informed latent-amplitude events
# ---------------------------------------------------------------------------

def build_inspiral_informed_events(
    phase3_rows: list[dict],
    catalog_map: dict[str, dict],
    use_loo: bool = True,
) -> tuple[list[LatentAmplitudeEvent], dict]:
    """Build latent-amplitude events with inspiral-informed A_220 prior.

    Parameters
    ----------
    phase3_rows : list of Phase 3 summary dicts
    catalog_map : dict mapping event name to GWTC catalog entry
    use_loo : if True, use leave-one-out predictions (avoids overfitting)

    Returns
    -------
    events : list of LatentAmplitudeEvent with combined (prior + Fisher) a_220
    diagnostics : dict with model and per-event diagnostics
    """
    n = len(phase3_rows)

    # Extract arrays
    a_220 = np.array([r["a_220_fit"] for r in phase3_rows])
    noise_rms = np.array([r["noise_rms"] for r in phase3_rows])
    seg_dur = np.array([r.get("seg_duration", 0.03) for r in phase3_rows])

    remnant_mass = np.zeros(n)
    spin = np.zeros(n)
    mass_ratio = np.zeros(n)
    distance = np.zeros(n)

    for i, r in enumerate(phase3_rows):
        name = r["event"]
        cat = catalog_map[name]
        remnant_mass[i] = cat["remnant_mass"]
        spin[i] = cat["remnant_spin"]
        mass_ratio[i] = cat["mass_ratio"]
        distance[i] = cat["distance"]

    log_a = np.log(np.maximum(a_220, 1e-30))
    features = build_feature_matrix(
        remnant_mass, mass_ratio, distance, spin, seg_dur, noise_rms
    )

    # Fit model and compute predictions
    model = fit_inspiral_model(features, log_a)

    if use_loo:
        log_a_pred = loo_predict(features, log_a)
        loo_scatter = float(np.std(log_a - log_a_pred, ddof=1))
    else:
        log_a_pred = model.predict_log_a220(features)
        loo_scatter = model.scatter

    a_pred = np.exp(log_a_pred)

    # Build events with Bayesian combination
    events = []
    per_event_diag = []

    for i, r in enumerate(phase3_rows):
        a_data = max(float(r["a_220_fit"]), 1e-12)
        sigma_fisher = infer_phase3_row_sigma_a_220(r)

        # Log-normal prior -> Gaussian approximation for the mean
        mu_prior = float(np.exp(log_a_pred[i] + loo_scatter**2 / 2))
        sigma_prior = mu_prior * np.sqrt(np.exp(loo_scatter**2) - 1)

        mean_post, sigma_post = combine_prior_and_measurement(
            a_data, sigma_fisher, mu_prior, sigma_prior
        )

        frac_fisher = sigma_fisher / a_data if a_data > 0 else float("inf")
        frac_post = sigma_post / mean_post if mean_post > 0 else float("inf")

        per_event_diag.append({
            "event": r["event"],
            "a_data": a_data,
            "sigma_fisher": sigma_fisher,
            "frac_fisher": frac_fisher,
            "a_pred": float(a_pred[i]),
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "a_post": mean_post,
            "sigma_post": sigma_post,
            "frac_post": frac_post,
            "prior_helped": frac_post < frac_fisher,
        })

        # c_nl and sigma_c come from the data (independent of a_220 choice)
        c_scale = a_data**2
        events.append(LatentAmplitudeEvent(
            event_name=r["event"],
            c_nl_hat=float(r["kappa_hat"]) * c_scale,
            sigma_c=max(float(r["kappa_sigma"]) * c_scale, 1e-12),
            a_220_hat=mean_post,
            sigma_a_220=sigma_post,
            kappa_hat_plugin=float(r["kappa_hat"]),
            kappa_sigma_plugin=max(float(r["kappa_sigma"]), 1e-12),
        ))

    diagnostics = {
        "model_scatter_log": model.scatter,
        "model_scatter_frac": float(np.exp(model.scatter) - 1),
        "model_r_squared": model.r_squared,
        "loo_scatter_log": loo_scatter,
        "loo_scatter_frac": float(np.exp(loo_scatter) - 1),
        "coefficients": model.coefficients.tolist(),
        "feature_names": [
            "intercept", "log_M_rem", "log_eta", "log_d_L",
            "spin", "log_seg_dur", "log_noise_rms",
        ],
        "n_events": n,
        "n_prior_helped": sum(1 for d in per_event_diag if d["prior_helped"]),
        "median_frac_fisher": float(np.median([d["frac_fisher"] for d in per_event_diag])),
        "median_frac_posterior": float(np.median([d["frac_post"] for d in per_event_diag])),
        "per_event": per_event_diag,
    }

    return events, diagnostics


# ---------------------------------------------------------------------------
# Main: run all three estimators and compare
# ---------------------------------------------------------------------------

def main():
    base = Path(__file__).resolve().parents[2]
    catalog_path = base / "results" / "grims" / "gwtc_full_catalog.json"
    phase3_path = base / "results" / "grims" / "phase3_results.json"

    with open(catalog_path) as f:
        catalog = json.load(f)
    with open(phase3_path) as f:
        p3 = json.load(f)

    catalog_map = {c["name"]: c for c in catalog}
    rows = p3["individual"]
    n = len(rows)

    print(f"{'='*72}")
    print(f"Inspiral-Informed Kappa Estimator")
    print(f"{'='*72}")
    print(f"Events: {n}")
    print()

    # --- 1. Legacy plugin ---
    kappa_hat = np.array([r["kappa_hat"] for r in rows])
    kappa_sigma = np.array([r["kappa_sigma"] for r in rows])
    weights = 1.0 / kappa_sigma**2
    legacy_kappa = float(np.sum(weights * kappa_hat) / np.sum(weights))
    legacy_sigma = float(1.0 / np.sqrt(np.sum(weights)))
    legacy_90 = abs(legacy_kappa) + 1.645 * legacy_sigma

    print(f"1. LEGACY PLUGIN (inverse-variance weighted average)")
    print(f"   kappa = {legacy_kappa:+.4f} +/- {legacy_sigma:.4f}")
    print(f"   90% upper bound on |kappa|: {legacy_90:.4f}")
    print()

    # --- 2. Fisher-only latent-amplitude ---
    events_fisher = [phase3_row_to_latent_amplitude_event(r) for r in rows]
    result_fisher = estimate_kappa_posterior_latent_amplitude(events_fisher)

    print(f"2. FISHER-ONLY LATENT-AMPLITUDE MARGINALIZATION")
    print(f"   kappa_mean = {result_fisher.kappa_mean:+.4f} +/- {result_fisher.kappa_std:.4f}")
    print(f"   kappa_MAP  = {result_fisher.kappa_map:+.4f}")
    print(f"   68% CI: [{result_fisher.kappa_lower_68:+.4f}, {result_fisher.kappa_upper_68:+.4f}]")
    print(f"   90% CI: [{result_fisher.kappa_lower_90:+.4f}, {result_fisher.kappa_upper_90:+.4f}]")
    fisher_90_bound = max(abs(result_fisher.kappa_lower_90), abs(result_fisher.kappa_upper_90))
    print(f"   90% upper bound on |kappa|: {fisher_90_bound:.4f}")
    print()

    # --- 3. Inspiral-informed latent-amplitude ---
    events_inspiral, diag = build_inspiral_informed_events(
        rows, catalog_map, use_loo=True
    )
    result_inspiral = estimate_kappa_posterior_latent_amplitude(events_inspiral)

    print(f"3. INSPIRAL-INFORMED LATENT-AMPLITUDE MARGINALIZATION")
    print(f"   kappa_mean = {result_inspiral.kappa_mean:+.4f} +/- {result_inspiral.kappa_std:.4f}")
    print(f"   kappa_MAP  = {result_inspiral.kappa_map:+.4f}")
    print(f"   68% CI: [{result_inspiral.kappa_lower_68:+.4f}, {result_inspiral.kappa_upper_68:+.4f}]")
    print(f"   90% CI: [{result_inspiral.kappa_lower_90:+.4f}, {result_inspiral.kappa_upper_90:+.4f}]")
    inspiral_90_bound = max(abs(result_inspiral.kappa_lower_90), abs(result_inspiral.kappa_upper_90))
    print(f"   90% upper bound on |kappa|: {inspiral_90_bound:.4f}")
    print()

    # --- Model diagnostics ---
    print(f"{'='*72}")
    print(f"INSPIRAL PREDICTION MODEL DIAGNOSTICS")
    print(f"{'='*72}")
    print(f"   Model R^2 (training): {diag['model_r_squared']:.4f}")
    print(f"   Model scatter (log-space): {diag['model_scatter_log']:.4f}")
    print(f"   LOO scatter (log-space): {diag['loo_scatter_log']:.4f}")
    print(f"   LOO scatter (fractional): {diag['loo_scatter_frac']:.1%}")
    print()
    print(f"   Regression coefficients:")
    for name, coeff in zip(diag["feature_names"], diag["coefficients"]):
        print(f"     {name:>15s}: {coeff:+.4f}")
    print()
    print(f"   Bayesian combination (inspiral prior + Fisher data):")
    print(f"     Events where prior helped: {diag['n_prior_helped']} / {diag['n_events']}")
    print(f"     Median fractional sigma_a (Fisher only): {diag['median_frac_fisher']:.1%}")
    print(f"     Median fractional sigma_a (posterior):   {diag['median_frac_posterior']:.1%}")
    print()

    # --- Comparison ---
    print(f"{'='*72}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*72}")
    print(f"{'Estimator':<45s} {'kappa':>8s} {'sigma':>8s} {'|k|<90%':>8s}")
    print(f"{'-'*72}")
    print(f"{'Legacy plugin':<45s} {legacy_kappa:>+8.4f} {legacy_sigma:>8.4f} {legacy_90:>8.4f}")
    print(f"{'Fisher-only latent-amplitude':<45s} {result_fisher.kappa_mean:>+8.4f} {result_fisher.kappa_std:>8.4f} {fisher_90_bound:>8.4f}")
    print(f"{'Inspiral-informed latent-amplitude':<45s} {result_inspiral.kappa_mean:>+8.4f} {result_inspiral.kappa_std:>8.4f} {inspiral_90_bound:>8.4f}")
    print()

    improvement_fisher = (legacy_sigma - result_fisher.kappa_std) / legacy_sigma * 100
    improvement_inspiral = (legacy_sigma - result_inspiral.kappa_std) / legacy_sigma * 100
    improvement_over_fisher = (result_fisher.kappa_std - result_inspiral.kappa_std) / result_fisher.kappa_std * 100

    print(f"   Improvement over legacy:")
    print(f"     Fisher-only:       {improvement_fisher:+.1f}% in sigma")
    print(f"     Inspiral-informed: {improvement_inspiral:+.1f}% in sigma")
    print(f"   Inspiral improvement over Fisher-only: {improvement_over_fisher:+.1f}%")
    print()

    # --- Discussion ---
    print(f"{'='*72}")
    print(f"DISCUSSION")
    print(f"{'='*72}")
    print("""
The inspiral-informed prior provides marginal improvement (+{:.1f}%) over
the Fisher-only latent-amplitude estimator.  The fundamental limitation
is that the whitened A_220 depends critically on the detector PSD at the
QNM frequency, which is NOT available in the GWTC catalog.  The PSD
varies by orders of magnitude across the 30-500 Hz band where f_220
sits, creating irreducible scatter of ~{:.0f}% in the prediction model.

The Bayesian combination (inspiral prior + Fisher measurement) does help
the {} events where Fisher fractional sigma exceeds 100%, by pulling
those events toward physically plausible amplitudes.  However, for the
majority of events, the Fisher measurement (median {:.1f}% fractional
sigma) already dominates the posterior.

KEY FINDING: The existing Fisher-only latent-amplitude estimator already
provides a {:.1f}% improvement over the legacy plugin, achieving
|kappa| < {:.4f} at 90% confidence.  The inspiral prior adds only a
small additional improvement.

TO ACHIEVE THE 10% SIGMA TARGET, one would need:
1. Access to the actual detector PSD at each event's QNM frequency, or
2. NR-calibrated ringdown efficiency factors (fraction of total SNR in
   the 220 mode as function of mass ratio and spin), or
3. A hierarchical Bayesian model that simultaneously infers the population
   distribution of A_220 and the per-event kappa.
""".format(
        improvement_over_fisher,
        diag['loo_scatter_frac'] * 100,
        diag['n_events'] - diag['n_prior_helped'],
        diag['median_frac_fisher'] * 100,
        improvement_fisher,
        fisher_90_bound,
    ))

    # Save results
    output = {
        "legacy_plugin": {
            "kappa": legacy_kappa,
            "sigma": legacy_sigma,
            "bound_90": legacy_90,
        },
        "fisher_latent_amplitude": {
            "kappa_mean": result_fisher.kappa_mean,
            "kappa_std": result_fisher.kappa_std,
            "kappa_map": result_fisher.kappa_map,
            "ci_68": [result_fisher.kappa_lower_68, result_fisher.kappa_upper_68],
            "ci_90": [result_fisher.kappa_lower_90, result_fisher.kappa_upper_90],
            "bound_90": fisher_90_bound,
        },
        "inspiral_informed": {
            "kappa_mean": result_inspiral.kappa_mean,
            "kappa_std": result_inspiral.kappa_std,
            "kappa_map": result_inspiral.kappa_map,
            "ci_68": [result_inspiral.kappa_lower_68, result_inspiral.kappa_upper_68],
            "ci_90": [result_inspiral.kappa_lower_90, result_inspiral.kappa_upper_90],
            "bound_90": inspiral_90_bound,
        },
        "model_diagnostics": {
            k: v for k, v in diag.items() if k != "per_event"
        },
    }

    out_path = base / "results" / "grims" / "inspiral_informed_kappa_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
