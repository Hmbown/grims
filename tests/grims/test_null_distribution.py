import h5py
import numpy as np

from bown_instruments.grims.null_distribution import (
    EventNullPreparation,
    NullProjectionComponent,
    _fd_inner,
    _phase_randomize_residual,
    _shift_samples_from_fraction,
    analyze_null_distribution,
    generate_event_null_realization,
    run_null_campaign,
)
from bown_instruments.grims.mass_analysis import find_local_strain_detector


def _make_component(
    detector: str,
    t_start_m: float,
    residual: np.ndarray,
    template: np.ndarray,
    noise_var: float = 1.0,
    min_shift_samples: int = 2,
) -> NullProjectionComponent:
    template_fft = np.fft.rfft(template)
    residual_fft = np.fft.rfft(residual)
    template_inner = _fd_inner(template_fft, template_fft, len(residual), noise_var)
    residual_overlap = _fd_inner(residual_fft, template_fft, len(residual), noise_var)
    template_norm = np.sqrt(template_inner)
    return NullProjectionComponent(
        detector=detector,
        t_start_m=t_start_m,
        residual=residual,
        template_fft=template_fft,
        noise_var=noise_var,
        template_inner=template_inner,
        template_norm=template_norm,
        kappa_hat_real=residual_overlap / template_inner,
        kappa_sigma_real=1.0 / template_norm,
        min_shift_samples=min_shift_samples,
        n_samples=len(residual),
    )


def _combine_sigmas(sigmas: list[float]) -> float:
    weights = np.sum([1.0 / sigma**2 for sigma in sigmas])
    return float(1.0 / np.sqrt(weights))


def _combine_kappas(kappas: list[float], sigmas: list[float]) -> float:
    weights = np.array([1.0 / sigma**2 for sigma in sigmas])
    return float(np.sum(weights * np.asarray(kappas)) / np.sum(weights))


def _make_preparation(name: str, offset: float = 0.0) -> EventNullPreparation:
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    residual_a = 0.5 * np.sin(x + offset) + 0.2 * np.cos(2.0 * x)
    residual_b = 0.3 * np.cos(x - offset) - 0.1 * np.sin(3.0 * x)
    template_a = np.cos(x)
    template_b = np.sin(x)

    component_a = _make_component("H1", 10.0, residual_a, template_a)
    component_b = _make_component("H1", 20.0, residual_b, template_b)

    detector_sigmas = [component_a.kappa_sigma_real, component_b.kappa_sigma_real]
    detector_kappas = [component_a.kappa_hat_real, component_b.kappa_hat_real]
    event_sigma = _combine_sigmas(detector_sigmas)
    event_kappa = _combine_kappas(detector_kappas, detector_sigmas)

    return EventNullPreparation(
        event_name=name,
        detectors_used=["H1"],
        detector_components={"H1": [component_a, component_b]},
        real_kappa_hat=event_kappa,
        real_kappa_sigma=event_sigma,
        per_detector_real={"H1": {"kappa_hat": event_kappa, "kappa_sigma": event_sigma}},
        noise_rms_reference=1.0,
    )


def test_shift_samples_from_fraction_respects_exclusion_window():
    assert _shift_samples_from_fraction(32, 4, 0.0) == 4
    assert _shift_samples_from_fraction(32, 4, 0.999999) == 28
    assert _shift_samples_from_fraction(8, 5, 0.5) in range(1, 8)


def test_phase_randomization_preserves_amplitude_spectrum():
    residual = np.array([0.2, -1.0, 0.5, 0.75, -0.1, 0.4, -0.3, 0.9])
    randomized = _phase_randomize_residual(residual, np.random.default_rng(7))

    assert not np.allclose(randomized, residual)
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(randomized)),
        np.abs(np.fft.rfft(residual)),
        atol=1e-12,
    )


def test_generate_event_null_realization_is_reproducible_and_preserves_sigma():
    preparation = _make_preparation("GWTEST")

    result_a, meta_a = generate_event_null_realization(
        preparation,
        np.random.default_rng(42),
        method="circular_time_shift",
    )
    result_b, meta_b = generate_event_null_realization(
        preparation,
        np.random.default_rng(42),
        method="circular_time_shift",
    )

    assert result_a.kappa_hat == result_b.kappa_hat
    assert result_a.kappa_sigma == result_b.kappa_sigma
    assert meta_a["shift_fraction"] == meta_b["shift_fraction"]
    assert result_a.kappa_sigma == preparation.real_kappa_sigma


def test_analyze_null_distribution_reports_calibration_metrics():
    summary = analyze_null_distribution(
        null_kappas=np.array([-1.0, 0.0, 1.0]),
        null_sigmas=np.array([1.0, 1.0, 1.0]),
        observed_kappa=1.0,
        observed_sigma=1.0,
        sigma_ratios=np.array([1.0, 1.01, 0.99]),
    )

    assert summary["empirical_p_value"] == 1.0 / 3.0
    assert np.isclose(summary["empirical_sigma"], 1.224744871391589)
    assert summary["per_event_null_check"]["sigma_ratios_consistent"] is True


def test_run_null_campaign_returns_expected_shapes_and_sigma_checks():
    preparations = [
        _make_preparation("GWTEST_A", offset=0.0),
        _make_preparation("GWTEST_B", offset=0.3),
    ]

    result = run_null_campaign(
        preparations=preparations,
        observed_kappa=0.25,
        observed_sigma=0.1,
        n_null=8,
        seed=11,
        method="circular_time_shift",
        max_weight_ratio=5.5,
        progress=False,
    )

    assert len(result["null_kappas"]) == 8
    assert len(result["null_sigmas"]) == 8
    assert len(result["realization_shift_fractions"]) == 8
    assert len(result["realization_shift_fractions"][0]) == 2
    assert result["per_event_null_check"]["sigma_ratios_consistent"] is True
    np.testing.assert_allclose(result["per_event_null_check"]["mean_sigma_ratio"], 1.0)


def test_find_local_strain_detector_skips_invalid_hdf5_cache(tmp_path):
    event = {"gps": 1386502144}
    invalid = tmp_path / "H-H1_GWOSC_O4a_4KHZ_R1-1386500096-4096.hdf5"
    invalid.write_bytes(b"")

    valid = tmp_path / "H-H1_GWOSC_O4a_4KHZ_R1-1386500000-8192.hdf5"
    with h5py.File(valid, "w"):
        pass

    path = find_local_strain_detector(event, str(tmp_path), "H1")
    assert path == str(valid)
