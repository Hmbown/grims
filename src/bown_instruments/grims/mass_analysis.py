"""
Mass analysis: run the phase-locked nonlinear mode search across
the full GWTC catalog.

This is the stacking engine — accumulate evidence from every available
BBH event, weighted by sensitivity. The combined SNR grows as sqrt(N).

Improvements over the original pipeline:
  - Frequency-domain colored-noise likelihood (uses actual PSD)
  - Ringdown start time marginalization over [5M, 20M]
  - Multi-detector coherent stacking (H1 + L1 + V1)
  - Per-event optimal segment length scaled by QNM damping time
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass


def load_catalog(path: str = "data/gwtc_full_catalog.json") -> list:
    """Load the full GWTC catalog."""
    with open(path) as f:
        return json.load(f)


def find_local_strain(event: dict, data_dir: str = "data/") -> str | None:
    """Find a locally cached strain file for an event (H1 preferred)."""
    return find_local_strain_detector(event, data_dir, detector_prefix="H1")


def find_local_strain_detector(
    event: dict, data_dir: str = "data/", detector_prefix: str = "H1"
) -> str | None:
    """Find a locally cached strain file for an event and specific detector."""
    from .gwtc_pipeline import is_valid_hdf5_file

    data_path = Path(data_dir)
    gps = int(event.get("gps", 0))
    if gps == 0:
        return None

    # Map detector prefix to filename prefix
    file_prefix = {"H1": "H-H1_", "L1": "L-L1_", "V1": "V-V1_"}
    prefix = file_prefix.get(detector_prefix, f"?-{detector_prefix}_")

    for f in sorted(data_path.glob("*.hdf5")):
        fname = f.name
        if not fname.startswith(prefix):
            continue
        parts = f.stem.split("-")
        if len(parts) >= 3:
            try:
                file_gps = int(parts[-2])
                file_dur = int(parts[-1])
                if file_gps <= gps <= file_gps + file_dur:
                    if is_valid_hdf5_file(f):
                        return str(f)
            except ValueError:
                continue
    return None


def find_all_detector_files(event: dict, data_dir: str = "data/") -> dict:
    """Find locally cached strain files for all available detectors.

    Returns dict mapping detector name -> file path.
    """
    result = {}
    for det in ["H1", "L1", "V1"]:
        path = find_local_strain_detector(event, data_dir, det)
        if path is not None:
            result[det] = path
    return result


def compute_optimal_segment_duration(
    mass: float, spin: float, n_damping_times: float = 5.0
) -> float:
    """Compute optimal post-ringdown segment duration for an event.

    Uses the (2,2,0) QNM damping time to set the segment length.
    The ringdown signal decays as exp(-t/tau), so after n*tau the
    signal is exp(-n) ~ negligible. Including more data past that
    just adds noise.

    Parameters
    ----------
    mass : remnant mass in solar masses
    spin : remnant spin
    n_damping_times : number of damping times to include (default 5.0)

    Returns
    -------
    seg_duration : post-ringdown segment duration in seconds
    """
    from .qnm_modes import KerrQNMCatalog

    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)

    # Physical damping time in seconds
    tau_s = mode_220.physical_damping_time_s(mass)

    # Segment = n_damping_times * tau, but clamp to reasonable range
    seg_duration = n_damping_times * tau_s

    # Floor: at least 0.03s (need enough samples for FFT)
    # Ceiling: at most 0.3s (beyond this we're just stacking noise)
    seg_duration = max(0.03, min(0.3, seg_duration))

    return seg_duration


def analyze_event_single_detector(
    event: dict,
    strain_path: str,
    detector: str,
    t_start_values: list = None,
    use_colored: bool = True,
    adaptive_segment: bool = True,
) -> dict | None:
    """Run the phase-locked search on a single event + single detector.

    Parameters
    ----------
    event : catalog entry dict
    strain_path : path to HDF5 strain file
    detector : detector name (H1, L1, V1)
    t_start_values : ringdown start times in M after merger
    use_colored : use frequency-domain colored-noise likelihood
    adaptive_segment : if True, use per-event optimal segment length

    Returns None if analysis fails.
    """
    from .whiten import estimate_asd, whiten_strain, bandpass
    from .phase_locked_search import (
        phase_locked_search,
        phase_locked_search_colored,
    )
    from .qnm_modes import KerrQNMCatalog
    from .gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5

    mass = event.get("remnant_mass", 0)
    spin = event.get("remnant_spin", 0.69)
    gps = event.get("gps", 0)

    if mass <= 0 or gps <= 0:
        return None

    m_seconds = mass * M_SUN_SECONDS

    # Compute QNM frequencies
    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)
    mode_440 = catalog.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)
    f_low = max(20.0, f_220 * 0.5)
    f_high_target = max(f_nl, f_440) * 1.3

    try:
        loaded = load_gwosc_strain_hdf5(strain_path)
    except Exception:
        return None

    strain = loaded["strain"]
    time = loaded["time"]
    sample_rate = loaded["sample_rate"]
    f_high = min(0.45 * sample_rate, f_high_target)

    if f_220 < 20 or f_nl > 0.45 * sample_rate:
        return None

    merger_time = float(gps)
    if merger_time < time[0] or merger_time > time[-1]:
        return None

    try:
        asd_freqs, asd = estimate_asd(
            strain,
            sample_rate,
            merger_time=merger_time,
            time=time,
            exclusion_window=2.0,
        )
    except Exception:
        return None

    noise_mask = np.abs(time - merger_time) > 4.0
    if np.sum(noise_mask) < 100:
        return None

    if t_start_values is None:
        t_start_values = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]

    # Compute per-event segment duration
    if adaptive_segment:
        seg_duration = compute_optimal_segment_duration(mass, spin, n_damping_times=5.0)
    else:
        seg_duration = 0.15  # original fixed duration

    pad_before = 0.05

    results_per_tstart = []

    for t_start_m in t_start_values:
        ringdown_start = merger_time + t_start_m * m_seconds

        whitened = whiten_strain(strain, sample_rate, asd_freqs, asd, fmin=f_low * 0.8)
        whitened_bp = bandpass(whitened, sample_rate, f_low, f_high)

        t_start = ringdown_start - pad_before
        t_end = ringdown_start + seg_duration
        mask = (time >= t_start) & (time <= t_end)

        if np.sum(mask) < 50:
            continue

        seg_strain = whitened_bp[mask]
        seg_time = time[mask]
        t_dimless = (seg_time - ringdown_start) / m_seconds

        noise_var = np.var(whitened_bp[noise_mask])
        noise_rms = np.sqrt(noise_var)

        search_fn = phase_locked_search_colored if use_colored else phase_locked_search
        label = f"{event['name']}_{detector}_t{t_start_m:.0f}M"

        try:
            result = search_fn(seg_strain, t_dimless, spin, noise_rms, event_name=label)
        except Exception:
            continue

        results_per_tstart.append((t_start_m, result))

    if not results_per_tstart:
        return None

    # Marginalize over t_start: inverse-variance weighted average
    weights = []
    kappas = []
    for t_start_m, result in results_per_tstart:
        if result.kappa_sigma > 0 and np.isfinite(result.kappa_sigma):
            w = 1.0 / result.kappa_sigma**2
            weights.append(w)
            kappas.append(result.kappa_hat)

    if not weights:
        return None

    weights = np.array(weights)
    kappas = np.array(kappas)
    total_weight = np.sum(weights)
    kappa_marginalized = np.sum(weights * kappas) / total_weight
    sigma_marginalized = 1.0 / np.sqrt(total_weight)

    best = max(results_per_tstart, key=lambda x: abs(x[1].snr))
    best_t, best_result = best

    return {
        "event": event["name"],
        "detector": detector,
        "mass": mass,
        "spin": spin,
        "snr_event": event.get("snr", 0),
        "f_220": f_220,
        "f_nl": f_nl,
        "kappa_hat": kappa_marginalized,
        "kappa_sigma": sigma_marginalized,
        "snr_nl": best_result.snr,
        "a_220_fit": best_result.a_220_fit,
        "noise_rms": best_result.noise_rms,
        "best_t_start_m": best_t,
        "t_start_values": t_start_values,
        "n_t_start": len(results_per_tstart),
        "seg_duration": seg_duration,
        "result": best_result,
    }


def analyze_event(
    event: dict,
    data_dir: str = "data/",
    t_start_values: list = None,
    use_colored: bool = True,
    multi_detector: bool = True,
    adaptive_segment: bool = True,
) -> dict | None:
    """Run the phase-locked search on a single event, optionally multi-detector.

    If multi_detector=True, runs on all available detectors and combines
    via inverse-variance weighting (coherent stacking across detectors).

    Parameters
    ----------
    event : catalog entry dict
    data_dir : path to data directory
    t_start_values : ringdown start times in M after merger
    use_colored : use frequency-domain colored-noise likelihood
    multi_detector : if True, use all available detector data
    adaptive_segment : if True, use per-event optimal segment length
    """
    if multi_detector:
        det_files = find_all_detector_files(event, data_dir)
    else:
        # Single-detector fallback (original behavior)
        h1_path = find_local_strain(event, data_dir)
        if h1_path is None:
            return None
        det_files = {"H1": h1_path}

    if not det_files:
        return None

    # Run analysis on each detector independently
    det_results = []
    for det, path in det_files.items():
        r = analyze_event_single_detector(
            event,
            path,
            det,
            t_start_values=t_start_values,
            use_colored=use_colored,
            adaptive_segment=adaptive_segment,
        )
        if r is not None:
            det_results.append(r)

    if not det_results:
        return None

    # Coherently combine across detectors: inverse-variance weighting
    weights = []
    kappas = []
    for r in det_results:
        if r["kappa_sigma"] > 0 and np.isfinite(r["kappa_sigma"]):
            w = 1.0 / r["kappa_sigma"] ** 2
            weights.append(w)
            kappas.append(r["kappa_hat"])

    if not weights:
        return None

    weights = np.array(weights)
    kappas = np.array(kappas)
    total_weight = np.sum(weights)
    kappa_combined = np.sum(weights * kappas) / total_weight
    sigma_combined = 1.0 / np.sqrt(total_weight)

    # Use best-SNR detector result as representative
    best_det = max(det_results, key=lambda x: abs(x["snr_nl"]))

    # Create a synthetic PhaseLockResult with combined kappa/sigma
    # so the stacker uses the multi-detector combined values
    from .phase_locked_search import PhaseLockResult

    combined_result = PhaseLockResult(
        event_name=event["name"],
        kappa_hat=kappa_combined,
        kappa_sigma=sigma_combined,
        snr=best_det["snr_nl"],
        a_220_fit=best_det["a_220_fit"],
        phi_220_fit=best_det["result"].phi_220_fit,
        template_norm=1.0 / sigma_combined if sigma_combined > 0 else 0.0,
        residual_overlap=kappa_combined / sigma_combined**2 if sigma_combined > 0 else 0.0,
        noise_rms=best_det["noise_rms"],
    )

    return {
        "event": event["name"],
        "mass": best_det["mass"],
        "spin": best_det["spin"],
        "snr_event": event.get("snr", 0),
        "f_220": best_det["f_220"],
        "f_nl": best_det["f_nl"],
        "kappa_hat": kappa_combined,
        "kappa_sigma": sigma_combined,
        "snr_nl": best_det["snr_nl"],
        "a_220_fit": best_det["a_220_fit"],
        "noise_rms": best_det["noise_rms"],
        "best_t_start_m": best_det["best_t_start_m"],
        "t_start_values": best_det["t_start_values"],
        "n_t_start": best_det["n_t_start"],
        "n_detectors": len(det_results),
        "detectors_used": [r["detector"] for r in det_results],
        "per_detector": [
            {
                "detector": r["detector"],
                "kappa_hat": r["kappa_hat"],
                "kappa_sigma": r["kappa_sigma"],
                "snr_nl": r["snr_nl"],
            }
            for r in det_results
        ],
        "seg_duration": best_det.get("seg_duration", 0.15),
        "result": combined_result,
    }


def run_mass_analysis(
    data_dir: str = "data/",
    catalog_path: str = "data/gwtc_full_catalog.json",
    min_total_mass: float = 40.0,
    t_start_values: list = None,
    use_colored: bool = True,
    multi_detector: bool = True,
    adaptive_segment: bool = True,
    max_weight_ratio: float | None = None,
    verbose: bool = True,
) -> dict:
    """Run the phase-locked search on all available events.

    Parameters
    ----------
    t_start_values : ringdown start times in M. None = [5,8,10,12,15,20].
    use_colored : if True, use frequency-domain colored-noise likelihood.
    multi_detector : if True, use all available detector data per event.
    adaptive_segment : if True, use per-event optimal segment length.
    max_weight_ratio : float, optional. Cap each event's weight at this
        multiple of the average weight. Reduces influence concentration.

    Returns a summary with individual results and the stacked measurement.
    """
    from .phase_locked_search import stack_phase_locked

    catalog = load_catalog(catalog_path)
    targets = [e for e in catalog if e["total_mass"] >= min_total_mass]

    if verbose:
        method = "colored-noise" if use_colored else "white-noise"
        t_info = (
            f"t_start={t_start_values}"
            if t_start_values
            else "t_start=[5,8,10,12,15,20]M (marginalized)"
        )
        det_info = "multi-detector" if multi_detector else "H1-only"
        seg_info = "adaptive" if adaptive_segment else "fixed 0.15s"
        print(f"Mass analysis: {len(targets)} events with M_total >= {min_total_mass}")
        print(f"Method: {method}, {t_info}")
        print(f"Detectors: {det_info}, Segment: {seg_info}")
        print()

    results = []
    skipped = 0

    for ev in targets:
        r = analyze_event(
            ev,
            data_dir,
            t_start_values=t_start_values,
            use_colored=use_colored,
            multi_detector=multi_detector,
            adaptive_segment=adaptive_segment,
        )
        if r is None:
            skipped += 1
            if verbose:
                print(f"  {ev['name']:<30} SKIP (no data or bad params)")
            continue

        results.append(r)
        if verbose:
            n_det = r.get("n_detectors", 1)
            dets = ",".join(r.get("detectors_used", ["H1"]))
            seg_ms = r.get("seg_duration", 0.15) * 1000
            print(
                f"  {r['event']:<30} k={r['kappa_hat']:+8.3f} +/- {r['kappa_sigma']:8.3f}  "
                f"SNR_NL={r['snr_nl']:+6.3f}  det={dets}({n_det})  "
                f"seg={seg_ms:.0f}ms  best_t={r.get('best_t_start_m', '?')}M"
            )

    if verbose:
        print(f"\nAnalyzed: {len(results)}, Skipped: {skipped}")

    # Stack
    if results:
        phase_lock_results = [r["result"] for r in results]
        stacked = stack_phase_locked(phase_lock_results, max_weight_ratio=max_weight_ratio)
    else:
        stacked = None

    return {
        "individual": results,
        "stacked": stacked,
        "n_analyzed": len(results),
        "n_skipped": skipped,
    }
