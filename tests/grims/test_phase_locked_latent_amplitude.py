import json
from math import sqrt
from pathlib import Path

import numpy as np

from bown_instruments.grims.phase_locked_search import (
    LatentAmplitudeEvent,
    estimate_kappa_posterior_latent_amplitude,
    phase3_row_to_latent_amplitude_event,
    phase3_rows_to_latent_amplitude_events,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE3_RESULTS_PATH = _PROJECT_ROOT / "results/grims/phase3_results.json"
REDUCED_INJECTION_PATH = _PROJECT_ROOT / "results/grims/phase3_injection_campaign_reduced_shared_noise_30ms.json"


def _legacy_inverse_variance_stack(events: list[LatentAmplitudeEvent]) -> float:
    kappas = np.array([event.kappa_hat_plugin for event in events], dtype=float)
    sigmas = np.array([event.kappa_sigma_plugin for event in events], dtype=float)
    weights = 1.0 / sigmas**2
    return float(np.sum(weights * kappas) / np.sum(weights))


def _through_origin_slope(kappa_true: np.ndarray, estimates: np.ndarray) -> float:
    return float(np.dot(kappa_true, estimates) / np.dot(kappa_true, kappa_true))


def _simulate_latent_events(
    base_events: list[LatentAmplitudeEvent],
    kappa_true: float,
    rng: np.random.Generator,
) -> list[LatentAmplitudeEvent]:
    simulated: list[LatentAmplitudeEvent] = []
    for event in base_events:
        a_true_sq = max(event.a_220_hat**2 - event.sigma_a_220**2, 1e-10)
        a_true = sqrt(a_true_sq)
        a_obs = max(a_true + rng.normal(0.0, event.sigma_a_220), 1e-8)
        c_obs = kappa_true * a_true_sq + rng.normal(0.0, event.sigma_c)
        kappa_hat = c_obs / (a_obs**2)
        kappa_sigma = event.sigma_c / (a_obs**2)
        simulated.append(
            LatentAmplitudeEvent(
                event_name=event.event_name,
                c_nl_hat=float(c_obs),
                sigma_c=float(event.sigma_c),
                a_220_hat=float(a_obs),
                sigma_a_220=float(event.sigma_a_220),
                kappa_hat_plugin=float(kappa_hat),
                kappa_sigma_plugin=float(kappa_sigma),
            )
        )
    return simulated


def test_latent_amplitude_matches_plugin_when_sigma_a_is_zero():
    kappa_true = 0.032
    events = [
        LatentAmplitudeEvent(
            event_name="e1",
            c_nl_hat=kappa_true * 1.2**2,
            sigma_c=0.010,
            a_220_hat=1.2,
            sigma_a_220=0.0,
            kappa_hat_plugin=kappa_true,
            kappa_sigma_plugin=0.010 / 1.2**2,
        ),
        LatentAmplitudeEvent(
            event_name="e2",
            c_nl_hat=kappa_true * 0.8**2,
            sigma_c=0.020,
            a_220_hat=0.8,
            sigma_a_220=0.0,
            kappa_hat_plugin=kappa_true,
            kappa_sigma_plugin=0.020 / 0.8**2,
        ),
    ]

    posterior = estimate_kappa_posterior_latent_amplitude(
        events,
        kappa_min=-0.02,
        kappa_max=0.08,
        n_kappa=401,
    )

    assert abs(posterior.kappa_mean - kappa_true) < 2e-3
    assert abs(posterior.legacy_plugin_kappa - kappa_true) < 1e-12


def test_phase3_row_converter_infers_finite_sigma_a():
    with PHASE3_RESULTS_PATH.open() as handle:
        row = json.load(handle)["individual"][0]

    event = phase3_row_to_latent_amplitude_event(row)

    assert np.isfinite(event.sigma_a_220)
    assert event.sigma_a_220 > 0.0
    assert np.isclose(event.c_nl_hat, row["kappa_hat"] * row["a_220_fit"] ** 2)
    assert np.isclose(event.sigma_c, row["kappa_sigma"] * row["a_220_fit"] ** 2)


def test_latent_amplitude_recovery_improves_on_reduced_injection_configuration():
    with PHASE3_RESULTS_PATH.open() as handle:
        phase3 = json.load(handle)
    with REDUCED_INJECTION_PATH.open() as handle:
        injection = json.load(handle)

    selected_events = set(injection["metadata"]["selected_events"])
    base_events = phase3_rows_to_latent_amplitude_events(
        phase3["individual"],
        event_names=selected_events,
    )
    assert len(base_events) == len(selected_events) == 40

    kappa_values = np.array(injection["metadata"]["kappa_values"], dtype=float)
    injection_legacy_means = np.mean(
        np.asarray(injection["stacked_realizations"]["marginalized_default"]["kappa_hat"], dtype=float),
        axis=0,
    )
    injection_legacy_slope = _through_origin_slope(kappa_values, injection_legacy_means)

    rng = np.random.default_rng(1234)
    legacy_means = []
    marginalized_means = []

    for kappa_true in kappa_values:
        legacy_draws = []
        marginalized_draws = []
        for _ in range(24):
            draw = _simulate_latent_events(base_events, float(kappa_true), rng)
            legacy_draws.append(_legacy_inverse_variance_stack(draw))
            posterior = estimate_kappa_posterior_latent_amplitude(
                draw,
                kappa_min=0.0,
                kappa_max=0.08,
                n_kappa=121,
                n_amplitude=81,
                amplitude_sigma_window=6.0,
            )
            marginalized_draws.append(posterior.kappa_mean)

        legacy_means.append(np.mean(legacy_draws))
        marginalized_means.append(np.mean(marginalized_draws))

    legacy_slope = _through_origin_slope(kappa_values, np.asarray(legacy_means))
    marginalized_slope = _through_origin_slope(kappa_values, np.asarray(marginalized_means))

    assert abs(legacy_slope - injection_legacy_slope) < 0.12
    assert marginalized_slope > legacy_slope + 0.15
