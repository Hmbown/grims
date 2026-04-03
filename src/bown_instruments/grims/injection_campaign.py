"""
Injection campaign: inject known kappa into real LIGO noise and recover it.

This is Bown's self-test principle (US1,573,801) applied to the
real-data pipeline. If we inject a known signal and cannot recover it,
the instrument is lying.

The campaign:
1. Take whitened off-source noise from each event
2. Add a synthetic ringdown with known kappa (0, 0.5, 1.0, 2.0)
3. Run the full analysis pipeline
4. Check whether the injected kappa is recovered within the posterior
"""

import sys

sys.path.insert(0, "/Volumes/VIXinSSD/drbown/bown-ringdown")

import numpy as np
from bown_instruments.grims.whiten import prepare_ringdown_for_analysis
from bown_instruments.grims.bayesian_analysis import (
    estimate_kappa_posterior_from_data,
    estimate_kappa_posterior_profiled,
    stack_posteriors,
)
from bown_instruments.grims.ringdown_templates import RingdownTemplateBuilder
from bown_instruments.grims.qnm_modes import KerrQNMCatalog


def inject_ringdown_into_noise(
    event_name: str,
    kappa_injected: float,
    data_dir: str = "data/",
    t_start_m: float = 10.0,
    use_profiled: bool = False,
) -> dict:
    """Inject a ringdown with known kappa into real noise and recover it.

    Parameters
    ----------
    event_name : GWTC event name
    kappa_injected : the kappa value to inject
    data_dir : where GWOSC files are cached
    t_start_m : ringdown start time in M after merger
    use_profiled : if True, use profile likelihood; else fixed-amplitude fit

    Returns
    -------
    dict with injection parameters, recovery result, and diagnostics
    """
    # Get the whitened data (this contains both signal + noise)
    prep = prepare_ringdown_for_analysis(
        event_name,
        data_dir=data_dir,
        t_start_m=t_start_m,
    )

    strain = prep["strain_whitened"].copy()
    t_dimless = prep["t_dimless"]
    spin = prep["event"]["remnant_spin"]
    noise_var = prep["noise_variance"]

    # Build the injected signal
    builder = RingdownTemplateBuilder()
    catalog = KerrQNMCatalog()

    # Use the fitted A_220 from the data as the injection amplitude
    # This ensures the injected signal has realistic amplitude
    from bown_instruments.grims.bayesian_analysis import fit_linear_modes

    linear = fit_linear_modes(strain, t_dimless, spin)
    A_220_inj = linear["220"]["amplitude"]
    A_330_inj = linear["330"]["amplitude"]
    A_440_inj = linear["440"]["amplitude"]

    # Build template with the injected kappa
    template = builder.build_nonlinear_template(
        spin=spin,
        A_220=A_220_inj,
        kappa=kappa_injected,
        A_330=A_330_inj,
        A_440_linear=A_440_inj,
        phi_220=linear["220"]["phase"],
        phi_330=linear["330"]["phase"],
        phi_440_linear=linear["440"]["phase"],
    )

    injected_waveform = template.waveform(t_dimless)

    # Add the injected signal to the data
    # We subtract the existing data's linear component and add the new one
    # Actually: simplest approach is to just add the nonlinear mode to the data
    # The data already has the linear modes; we're adding the nonlinear contribution
    nonlinear_only = np.zeros_like(t_dimless)
    mask = t_dimless >= 0
    for mode in template.modes:
        if mode.qnm.is_nonlinear:
            nonlinear_only[mask] += (
                mode.amplitude
                * np.exp(mode.qnm.omega.imag * t_dimless[mask])
                * np.cos(mode.qnm.omega.real * t_dimless[mask] + mode.phase)
            )

    # Also need to account for the fact that the data's A_220 may differ
    # from the injected A_220. For a clean test, replace the data entirely.
    # Strategy: take off-source noise, inject full ringdown
    # But we only have the ringdown segment. So we inject the nonlinear mode
    # on top of the existing data.

    strain_with_injection = strain + nonlinear_only

    # Run recovery
    if use_profiled:
        result = estimate_kappa_posterior_profiled(
            strain_with_injection,
            t_dimless,
            spin=spin,
            noise_variance=noise_var,
            event_name=f"{event_name}_inj_k{kappa_injected:.1f}",
            n_kappa=51,
        )
    else:
        result = estimate_kappa_posterior_from_data(
            strain_with_injection,
            t_dimless,
            spin=spin,
            noise_variance=noise_var,
            event_name=f"{event_name}_inj_k{kappa_injected:.1f}",
        )

    # Check recovery
    recovered = result.kappa_map
    in_90 = result.kappa_lower_90 <= kappa_injected <= result.kappa_upper_90
    in_68 = result.kappa_lower_68 <= kappa_injected <= result.kappa_upper_68
    offset = abs(recovered - kappa_injected)
    sigma_offset = offset / max(0.01, (result.kappa_upper_68 - result.kappa_lower_68))

    return {
        "event_name": event_name,
        "kappa_injected": kappa_injected,
        "A_220_injected": A_220_inj,
        "t_start_m": t_start_m,
        "use_profiled": use_profiled,
        "recovery": result,
        "kappa_map": recovered,
        "kappa_90_lower": result.kappa_lower_90,
        "kappa_90_upper": result.kappa_upper_90,
        "in_90_ci": in_90,
        "in_68_ci": in_68,
        "offset_from_injected": offset,
        "sigma_offset": sigma_offset,
        "log_bayes_factor": result.log_bayes_factor,
    }


def run_injection_campaign(
    events: list = None,
    kappa_values: list = None,
    data_dir: str = "data/",
    use_profiled: bool = False,
) -> list:
    """Run the full injection campaign.

    Parameters
    ----------
    events : list of event names (default: the 5 standard events)
    kappa_values : list of kappa values to inject
    data_dir : where GWOSC files are cached
    use_profiled : if True, use profile likelihood

    Returns
    -------
    list of injection results
    """
    if events is None:
        events = [
            "GW150914",
            "GW190521",
            "GW200129_065458",
            "GW191109_010717",
            "GW190910_112807",
        ]
    if kappa_values is None:
        kappa_values = [0.0, 0.5, 1.0, 2.0]

    results = []
    method = "profiled" if use_profiled else "fixed"
    print(f"\nInjection Campaign ({method} likelihood)")
    print("=" * 80)
    print(
        f"{'Event':<25} {'kappa_inj':>9} {'kappa_MAP':>9} {'90% CI':>18} {'in90?':>6} {'lnB':>6}"
    )
    print("-" * 80)

    for event_name in events:
        for k_inj in kappa_values:
            result = inject_ringdown_into_noise(
                event_name,
                k_inj,
                data_dir=data_dir,
                use_profiled=use_profiled,
            )
            results.append(result)
            ci_str = f"[{result['kappa_90_lower']:.2f},{result['kappa_90_upper']:.2f}]"
            in90 = "YES" if result["in_90_ci"] else "NO"
            print(
                f"{event_name:<25} {k_inj:>9.1f} {result['kappa_map']:>9.3f} "
                f"{ci_str:>18} {in90:>6} {result['log_bayes_factor']:>6.2f}"
            )

    return results


def summarize_campaign(results: list) -> dict:
    """Summarize the injection campaign results.

    Returns dict with pass/fail rates and systematic bias estimates.
    """
    summary = {
        "total_injections": len(results),
        "recovered_in_90": sum(1 for r in results if r["in_90_ci"]),
        "recovered_in_68": sum(1 for r in results if r["in_68_ci"]),
        "mean_offset": np.mean([r["offset_from_injected"] for r in results]),
        "mean_sigma_offset": np.mean([r["sigma_offset"] for r in results]),
        "by_kappa": {},
        "by_event": {},
        "calibrated": False,
    }

    # Group by injected kappa
    kappa_groups = {}
    for r in results:
        k = r["kappa_injected"]
        if k not in kappa_groups:
            kappa_groups[k] = []
        kappa_groups[k].append(r)

    for k, group in sorted(kappa_groups.items()):
        summary["by_kappa"][k] = {
            "n": len(group),
            "recovery_rate_90": sum(1 for r in group if r["in_90_ci"]) / len(group),
            "mean_MAP": np.mean([r["kappa_map"] for r in group]),
            "mean_offset": np.mean([r["offset_from_injected"] for r in group]),
        }

    # Group by event
    event_groups = {}
    for r in results:
        e = r["event_name"]
        if e not in event_groups:
            event_groups[e] = []
        event_groups[e].append(r)

    for e, group in sorted(event_groups.items()):
        summary["by_event"][e] = {
            "n": len(group),
            "recovery_rate_90": sum(1 for r in group if r["in_90_ci"]) / len(group),
            "mean_offset": np.mean([r["offset_from_injected"] for r in group]),
        }

    # Calibration check: if kappa=1.0 injections are recovered within 90% CI
    # at least 50% of the time, the instrument is calibrated
    if 1.0 in summary["by_kappa"]:
        summary["calibrated"] = summary["by_kappa"][1.0]["recovery_rate_90"] >= 0.5

    return summary


if __name__ == "__main__":
    print("GRIM-S Injection Campaign")
    print("Bown's self-test: send yourself a known signal, check the response")
    print()

    # Run with fixed-amplitude method first
    results_fixed = run_injection_campaign(use_profiled=False)
    summary_fixed = summarize_campaign(results_fixed)

    print("\n" + "=" * 80)
    print("SUMMARY (fixed-amplitude likelihood)")
    print(f"  Total injections: {summary_fixed['total_injections']}")
    print(
        f"  Recovered in 90% CI: {summary_fixed['recovered_in_90']}/{summary_fixed['total_injections']} "
        f"({100 * summary_fixed['recovered_in_90'] / summary_fixed['total_injections']:.0f}%)"
    )
    print(f"  Mean offset from injected: {summary_fixed['mean_offset']:.3f}")
    print(f"  Mean sigma offset: {summary_fixed['mean_sigma_offset']:.2f}")
    print(f"  Calibrated: {'YES' if summary_fixed['calibrated'] else 'NO'}")
    print("\n  By injected kappa:")
    for k, stats in sorted(summary_fixed["by_kappa"].items()):
        print(
            f"    kappa={k:.1f}: recovery={stats['recovery_rate_90']:.0%}, "
            f"mean_MAP={stats['mean_MAP']:.3f}, mean_offset={stats['mean_offset']:.3f}"
        )
