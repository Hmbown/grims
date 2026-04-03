#!/usr/bin/env python3
"""
Full-catalog expansion: download, analyze, and test consistency.

This script implements the plan from the last analysis session:
  1. Download H1 strain for all events in the full GWTC catalog
  2. Run the phase-locked nonlinear mode search on every event
  3. Compute the amplitude ratio R = kappa_hat per event
  4. Test whether R is consistent across events (GR predicts R = kappa(spin))
  5. Generate diagnostic plots: R vs mass, R vs spin, R vs SNR

Ralph Bown's principle: expand the measurement base before adding complexity.
sqrt(N) is cheap. Download first, model later.
"""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
CATALOG_PATH = DATA_DIR / "gwtc_full_catalog.json"
RESULTS_PATH = DATA_DIR / "full_catalog_results.json"

# ---------------------------------------------------------------------------
# Step 1: Bulk downloader
# ---------------------------------------------------------------------------

def find_local_strain_for_event(event: dict, data_dir: Path) -> str | None:
    """Find a locally cached HDF5 strain file covering this event's GPS time."""
    gps = event.get("gps", 0)
    if gps <= 0:
        return None
    gps_int = int(gps)

    for f in sorted(data_dir.glob("*.hdf5")):
        parts = f.stem.split("-")
        if len(parts) >= 3:
            try:
                file_gps = int(parts[-2])
                file_dur = int(parts[-1])
                if file_gps <= gps_int <= file_gps + file_dur:
                    return str(f)
            except ValueError:
                continue
    return None


def download_event_strain(event: dict, data_dir: Path,
                          detector: str = "H1",
                          timeout: float = 60.0) -> str | None:
    """Download 4 kHz, 32-second HDF5 strain from GWOSC for one event.

    Uses the event API JSON URL from the catalog to discover strain file URLs.
    Returns the local path on success, None on failure.
    """
    jsonurl = event.get("jsonurl", "")
    if not jsonurl:
        return None

    # Fetch event metadata from GWOSC
    try:
        req = urllib.request.Request(jsonurl, headers={"User-Agent": "GRIMS/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        print(f"    [SKIP] API error for {event['name']}: {e}")
        return None

    # Navigate the GWOSC event API structure
    events_dict = payload.get("events", {})
    if not events_dict:
        return None
    event_bundle = next(iter(events_dict.values()))
    strain_list = event_bundle.get("strain", [])
    if not strain_list:
        return None

    # Find the best match: prefer requested detector, 4kHz, 32s, HDF5
    def score(item):
        det_match = 0 if item.get("detector") == detector else 1
        sr_diff = abs(item.get("sampling_rate", 0) - 4096)
        dur_diff = abs(item.get("duration", 0) - 32)
        fmt_match = 0 if item.get("format") == "hdf5" else 10
        return (fmt_match, det_match, dur_diff, sr_diff)

    candidates = [s for s in strain_list if s.get("format") == "hdf5"]
    if not candidates:
        return None

    candidates.sort(key=score)
    best = candidates[0]
    url = best.get("url", "")
    if not url:
        return None

    local_file = data_dir / Path(url).name
    if local_file.exists():
        return str(local_file)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GRIMS/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            local_file.write_bytes(resp.read())
        return str(local_file)
    except Exception as e:
        print(f"    [SKIP] Download error for {event['name']}: {e}")
        if local_file.exists():
            local_file.unlink()
        return None


def bulk_download(catalog: list, data_dir: Path,
                  max_events: int = 200) -> dict:
    """Download strain for all catalog events that don't have local data.

    Returns summary with counts and paths.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    already_have = 0
    downloaded = 0
    failed = 0
    paths = {}

    for i, event in enumerate(catalog[:max_events]):
        name = event["name"]

        # Check if we already have data
        local = find_local_strain_for_event(event, data_dir)
        if local:
            already_have += 1
            paths[name] = local
            continue

        print(f"  [{i+1}/{len(catalog)}] Downloading {name}...", end=" ", flush=True)
        path = download_event_strain(event, data_dir)
        if path:
            downloaded += 1
            paths[name] = path
            print(f"OK ({Path(path).name})")
        else:
            failed += 1
            print("FAILED")

        # Be polite to GWOSC servers
        if downloaded > 0 and downloaded % 5 == 0:
            time.sleep(1.0)

    print(f"\n  Download summary: {already_have} cached, {downloaded} new, {failed} failed")
    return {"paths": paths, "cached": already_have, "downloaded": downloaded, "failed": failed}


# ---------------------------------------------------------------------------
# Step 2: Full-catalog analysis
# ---------------------------------------------------------------------------

def analyze_single_event(event: dict, local_path: str) -> dict | None:
    """Run the phase-locked search on one event.

    Adapted from bown_instruments.grims.mass_analysis.analyze_event but works with
    any local strain file, not just the curated set.
    """
    from bown_instruments.grims.whiten import estimate_asd, whiten_strain, bandpass
    from bown_instruments.grims.phase_locked_search import phase_locked_search
    from bown_instruments.grims.qnm_modes import KerrQNMCatalog
    from bown_instruments.grims.gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5

    mass = event.get("remnant_mass", 0)
    spin = event.get("remnant_spin", 0.69)
    gps = event.get("gps", 0)

    if mass <= 0 or gps <= 0:
        return None

    m_seconds = mass * M_SUN_SECONDS

    # QNM frequencies
    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)
    mode_440 = catalog.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)
    f_low = max(20.0, f_220 * 0.5)

    try:
        loaded = load_gwosc_strain_hdf5(local_path)
    except Exception:
        return None

    strain = loaded["strain"]
    time_arr = loaded["time"]
    sample_rate = loaded["sample_rate"]
    f_high = min(0.45 * sample_rate, max(f_nl, f_440) * 1.3)

    # Check QNM frequencies are in detector band
    if f_220 < 20 or f_nl > 0.45 * sample_rate:
        return None

    merger_time = float(gps)
    ringdown_start = merger_time + 10.0 * m_seconds

    # Check merger is within the file
    if merger_time < time_arr[0] or merger_time > time_arr[-1]:
        return None

    try:
        asd_freqs, asd = estimate_asd(
            strain, sample_rate,
            merger_time=merger_time, time=time_arr,
            exclusion_window=2.0,
        )
        whitened = whiten_strain(strain, sample_rate, asd_freqs, asd,
                                fmin=f_low * 0.8)
        whitened_bp = bandpass(whitened, sample_rate, f_low, f_high)
    except Exception:
        return None

    # Extract ringdown segment
    pad_before = 0.05
    seg_duration = 0.15
    t_start = ringdown_start - pad_before
    t_end = ringdown_start + seg_duration
    mask = (time_arr >= t_start) & (time_arr <= t_end)

    if np.sum(mask) < 50:
        return None

    seg_strain = whitened_bp[mask]
    seg_time = time_arr[mask]
    t_dimless = (seg_time - ringdown_start) / m_seconds

    # Noise variance from off-source
    noise_mask = np.abs(time_arr - merger_time) > 4.0
    if np.sum(noise_mask) < 100:
        return None
    noise_var = np.var(whitened_bp[noise_mask])
    noise_rms = np.sqrt(noise_var)

    try:
        result = phase_locked_search(
            seg_strain, t_dimless, spin, noise_rms,
            event_name=event["name"],
        )
    except Exception:
        return None

    return {
        "event": event["name"],
        "mass": mass,
        "spin": spin,
        "total_mass": event.get("total_mass", 0),
        "mass_ratio": event.get("mass_ratio", 0),
        "snr_event": event.get("snr", 0),
        "distance": event.get("distance", 0),
        "f_220": f_220,
        "f_nl": f_nl,
        "kappa_hat": result.kappa_hat,
        "kappa_sigma": result.kappa_sigma,
        "snr_nl": result.snr,
        "a_220_fit": result.a_220_fit,
        "phi_220_fit": result.phi_220_fit,
        "template_norm": result.template_norm,
        "noise_rms": noise_rms,
    }


def run_full_analysis(catalog: list, paths: dict,
                      min_total_mass: float = 30.0) -> dict:
    """Run the phase-locked search on all events with local data."""
    from bown_instruments.grims.phase_locked_search import stack_phase_locked, PhaseLockResult

    targets = [e for e in catalog
               if e["total_mass"] >= min_total_mass and e["name"] in paths]

    print(f"\nAnalyzing {len(targets)} events (M_total >= {min_total_mass} Msun)...\n")

    results = []
    skipped = 0

    for i, ev in enumerate(targets):
        local_path = paths[ev["name"]]
        r = analyze_single_event(ev, local_path)
        if r is None:
            skipped += 1
            print(f"  [{i+1:3d}] {ev['name']:<30} SKIP")
            continue

        results.append(r)
        print(f"  [{i+1:3d}] {r['event']:<30} "
              f"kappa={r['kappa_hat']:+8.3f} +/- {r['kappa_sigma']:8.3f}  "
              f"A_220={r['a_220_fit']:8.4f}  SNR_NL={r['snr_nl']:+6.2f}")

    print(f"\nAnalyzed: {len(results)}, Skipped: {skipped}")

    # Stack all results
    if len(results) >= 2:
        phase_lock_results = []
        for r in results:
            phase_lock_results.append(PhaseLockResult(
                event_name=r["event"],
                kappa_hat=r["kappa_hat"],
                kappa_sigma=r["kappa_sigma"],
                snr=r["snr_nl"],
                a_220_fit=r["a_220_fit"],
                phi_220_fit=r["phi_220_fit"],
                template_norm=r["template_norm"],
                residual_overlap=r["kappa_hat"] * r["template_norm"]**2,
                noise_rms=r["noise_rms"],
            ))
        stacked = stack_phase_locked(phase_lock_results)
        print(f"\n{'='*60}")
        print(f"STACKED RESULT ({stacked.n_events} events):")
        print(f"  kappa = {stacked.kappa_hat:+.4f} +/- {stacked.kappa_sigma:.4f}")
        print(f"  SNR   = {stacked.snr:.2f}")
        print(f"  significance = {abs(stacked.kappa_hat)/stacked.kappa_sigma:.1f} sigma")
        print(f"{'='*60}")
    else:
        stacked = None

    return {
        "individual": results,
        "stacked": {
            "kappa_hat": stacked.kappa_hat if stacked else 0,
            "kappa_sigma": stacked.kappa_sigma if stacked else float("inf"),
            "snr": stacked.snr if stacked else 0,
            "n_events": stacked.n_events if stacked else 0,
        },
        "n_analyzed": len(results),
        "n_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Step 3: Amplitude ratio consistency test
# ---------------------------------------------------------------------------

def amplitude_ratio_consistency(results: list) -> dict:
    """Test whether kappa_hat is consistent across events.

    If GR is correct, kappa_hat = A_NL / A_220^2 should equal
    kappa_NR(spin) for every event. Deviations indicate either:
      - Noise contamination (expected for weak signals)
      - Systematic errors (ringdown start time, noise model)
      - New physics (unlikely but that's the point)

    We compute:
      - Weighted mean of kappa_hat
      - Chi-squared: does the data scatter more than expected?
      - Trend tests: does kappa correlate with mass, spin, or SNR?
    """
    from bown_instruments.grims.nr_predictions import kappa_nr_from_spin

    if len(results) < 3:
        return {"error": "Need at least 3 events for consistency test"}

    kappas = np.array([r["kappa_hat"] for r in results])
    sigmas = np.array([r["kappa_sigma"] for r in results])
    masses = np.array([r["mass"] for r in results])
    spins = np.array([r["spin"] for r in results])
    snrs = np.array([r["snr_event"] for r in results])
    a_220s = np.array([r["a_220_fit"] for r in results])

    # NR predictions per event
    kappa_nr = np.array([kappa_nr_from_spin(s) for s in spins])

    # Weighted mean
    valid = (sigmas > 0) & np.isfinite(sigmas)
    if np.sum(valid) < 3:
        return {"error": "Too few valid measurements"}

    w = 1.0 / sigmas[valid]**2
    kappa_mean = np.sum(w * kappas[valid]) / np.sum(w)
    kappa_mean_sigma = 1.0 / np.sqrt(np.sum(w))

    # Chi-squared: scatter around weighted mean
    chi2 = np.sum(w * (kappas[valid] - kappa_mean)**2)
    ndof = np.sum(valid) - 1
    chi2_per_dof = chi2 / ndof if ndof > 0 else 0

    # Chi-squared around NR prediction
    chi2_nr = np.sum(((kappas[valid] - kappa_nr[valid]) / sigmas[valid])**2)
    chi2_nr_per_dof = chi2_nr / ndof if ndof > 0 else 0

    # Pearson correlations: kappa vs (mass, spin, SNR)
    def pearson_r(x, y):
        if len(x) < 3:
            return 0.0, 1.0
        xm = x - np.mean(x)
        ym = y - np.mean(y)
        r = np.sum(xm * ym) / (np.sqrt(np.sum(xm**2) * np.sum(ym**2)) + 1e-30)
        # t-test for significance
        n = len(x)
        t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-30))
        from scipy.stats import t as t_dist
        p_value = 2 * t_dist.sf(abs(t_stat), n - 2)
        return float(r), float(p_value)

    r_mass, p_mass = pearson_r(masses[valid], kappas[valid])
    r_spin, p_spin = pearson_r(spins[valid], kappas[valid])
    r_snr, p_snr = pearson_r(snrs[valid], kappas[valid])

    # Residuals from NR prediction
    residuals = kappas[valid] - kappa_nr[valid]
    r_res_mass, p_res_mass = pearson_r(masses[valid], residuals)
    r_res_snr, p_res_snr = pearson_r(snrs[valid], residuals)

    summary = {
        "n_events": int(np.sum(valid)),
        "kappa_weighted_mean": float(kappa_mean),
        "kappa_mean_sigma": float(kappa_mean_sigma),
        "significance_sigma": float(abs(kappa_mean) / kappa_mean_sigma),
        "chi2_vs_mean": float(chi2),
        "chi2_per_dof_vs_mean": float(chi2_per_dof),
        "chi2_vs_nr": float(chi2_nr),
        "chi2_per_dof_vs_nr": float(chi2_nr_per_dof),
        "ndof": int(ndof),
        "correlation_kappa_mass": {"r": r_mass, "p": p_mass},
        "correlation_kappa_spin": {"r": r_spin, "p": p_spin},
        "correlation_kappa_snr": {"r": r_snr, "p": p_snr},
        "correlation_residual_mass": {"r": r_res_mass, "p": p_res_mass},
        "correlation_residual_snr": {"r": r_res_snr, "p": p_res_snr},
    }

    print(f"\n{'='*60}")
    print("AMPLITUDE RATIO CONSISTENCY TEST")
    print(f"{'='*60}")
    print(f"  Events:              {summary['n_events']}")
    print(f"  Weighted mean kappa: {kappa_mean:+.4f} +/- {kappa_mean_sigma:.4f}")
    print(f"  Significance:        {summary['significance_sigma']:.1f} sigma")
    print(f"  chi2/dof (vs mean):  {chi2_per_dof:.2f} ({ndof} dof)")
    print(f"  chi2/dof (vs NR):    {chi2_nr_per_dof:.2f} ({ndof} dof)")
    print(f"\n  Trend correlations (kappa vs parameter):")
    print(f"    vs mass:  r={r_mass:+.3f}  p={p_mass:.3f}  {'*' if p_mass < 0.05 else ''}")
    print(f"    vs spin:  r={r_spin:+.3f}  p={p_spin:.3f}  {'*' if p_spin < 0.05 else ''}")
    print(f"    vs SNR:   r={r_snr:+.3f}  p={p_snr:.3f}  {'*' if p_snr < 0.05 else ''}")
    print(f"\n  Residual correlations (kappa - kappa_NR vs parameter):")
    print(f"    vs mass:  r={r_res_mass:+.3f}  p={p_res_mass:.3f}  {'*' if p_res_mass < 0.05 else ''}")
    print(f"    vs SNR:   r={r_res_snr:+.3f}  p={p_res_snr:.3f}  {'*' if p_res_snr < 0.05 else ''}")
    print(f"{'='*60}")

    if chi2_per_dof > 2.0:
        print("  WARNING: chi2/dof > 2 — scatter exceeds expectations.")
        print("  This could indicate systematics (ringdown start time, noise model).")
    elif chi2_per_dof < 0.5:
        print("  NOTE: chi2/dof < 0.5 — error bars may be overestimated.")

    for name, r_val, p_val in [("mass", r_mass, p_mass),
                                 ("spin", r_spin, p_spin),
                                 ("SNR", r_snr, p_snr)]:
        if p_val < 0.05:
            print(f"  WARNING: significant correlation with {name} (p={p_val:.3f}).")
            print(f"  This suggests a systematic, not a constant kappa.")

    return summary


# ---------------------------------------------------------------------------
# Step 4: Diagnostic plots
# ---------------------------------------------------------------------------

def generate_plots(results: list, consistency: dict, plots_dir: Path):
    """Generate all diagnostic plots for the full-catalog analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bown_instruments.grims.nr_predictions import kappa_nr_from_spin, generate_kappa_curve

    plots_dir.mkdir(parents=True, exist_ok=True)

    kappas = np.array([r["kappa_hat"] for r in results])
    sigmas = np.array([r["kappa_sigma"] for r in results])
    masses = np.array([r["mass"] for r in results])
    spins = np.array([r["spin"] for r in results])
    snrs = np.array([r["snr_event"] for r in results])
    names = [r["event"] for r in results]
    a_220s = np.array([r["a_220_fit"] for r in results])

    kappa_nr = np.array([kappa_nr_from_spin(s) for s in spins])

    # --- Plot 1: Per-event kappa (sorted by SNR) ---
    order = np.argsort(snrs)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.3)))
    y_pos = np.arange(len(results))
    ax.errorbar(
        kappas[order], y_pos, xerr=sigmas[order],
        fmt="o", color="steelblue", capsize=3, alpha=0.8, markersize=4,
    )
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5, label="kappa=0")
    # Show NR prediction range
    ax.axvspan(0.01, 0.05, alpha=0.1, color="red", label="NR range (~0.01-0.04)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("kappa", fontsize=11)
    ax.set_title(f"Per-Event Kappa Estimates ({len(results)} events, sorted by SNR)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "full_catalog_per_event.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Kappa vs remnant spin with NR curve ---
    fig, ax = plt.subplots(figsize=(10, 7))
    spins_curve, kappa_curve, unc_curve = generate_kappa_curve()
    ax.plot(spins_curve, kappa_curve, "r-", linewidth=2, label="NR prediction")
    ax.fill_between(spins_curve,
                    kappa_curve - 1.645 * unc_curve,
                    kappa_curve + 1.645 * unc_curve,
                    alpha=0.15, color="red", label="NR 90% CI")
    # Color by event SNR
    sc = ax.scatter(spins, kappas, c=snrs, cmap="viridis", s=40, alpha=0.7,
                    edgecolors="k", linewidths=0.3, zorder=5)
    ax.errorbar(spins, kappas, yerr=sigmas, fmt="none", ecolor="gray",
                alpha=0.3, capsize=0, zorder=4)
    plt.colorbar(sc, ax=ax, label="Event SNR")
    ax.set_xlabel("Remnant spin", fontsize=12)
    ax.set_ylabel("kappa", fontsize=12)
    ax.set_title("Kappa vs Remnant Spin (Full Catalog)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "full_catalog_kappa_vs_spin.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Kappa vs remnant mass ---
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(masses, kappas, c=snrs, cmap="viridis", s=40, alpha=0.7,
                    edgecolors="k", linewidths=0.3)
    ax.errorbar(masses, kappas, yerr=sigmas, fmt="none", ecolor="gray",
                alpha=0.3, capsize=0)
    plt.colorbar(sc, ax=ax, label="Event SNR")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhspan(0.01, 0.05, alpha=0.1, color="red", label="NR range")
    r_val = consistency.get("correlation_kappa_mass", {}).get("r", 0)
    p_val = consistency.get("correlation_kappa_mass", {}).get("p", 1)
    ax.set_xlabel("Remnant mass (Msun)", fontsize=12)
    ax.set_ylabel("kappa", fontsize=12)
    ax.set_title(f"Kappa vs Mass (r={r_val:+.3f}, p={p_val:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "full_catalog_kappa_vs_mass.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 4: Kappa vs event SNR ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(snrs, kappas, c=masses, cmap="plasma", s=40, alpha=0.7,
               edgecolors="k", linewidths=0.3)
    ax.errorbar(snrs, kappas, yerr=sigmas, fmt="none", ecolor="gray",
                alpha=0.3, capsize=0)
    plt.colorbar(ax.collections[0], ax=ax, label="Remnant mass (Msun)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhspan(0.01, 0.05, alpha=0.1, color="red", label="NR range")
    r_val = consistency.get("correlation_kappa_snr", {}).get("r", 0)
    p_val = consistency.get("correlation_kappa_snr", {}).get("p", 1)
    ax.set_xlabel("Event SNR", fontsize=12)
    ax.set_ylabel("kappa", fontsize=12)
    ax.set_title(f"Kappa vs Event SNR (r={r_val:+.3f}, p={p_val:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "full_catalog_kappa_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 5: Residuals from NR prediction ---
    residuals = kappas - kappa_nr
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, x, xlabel, label in [
        (axes[0], masses, "Remnant mass (Msun)", "mass"),
        (axes[1], spins, "Remnant spin", "spin"),
        (axes[2], snrs, "Event SNR", "SNR"),
    ]:
        ax.scatter(x, residuals, c="steelblue", s=30, alpha=0.6,
                   edgecolors="k", linewidths=0.3)
        ax.errorbar(x, residuals, yerr=sigmas, fmt="none", ecolor="gray",
                    alpha=0.2, capsize=0)
        ax.axhline(0.0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("kappa - kappa_NR", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Residuals vs Mass", fontsize=11, fontweight="bold")
    axes[1].set_title("Residuals vs Spin", fontsize=11, fontweight="bold")
    axes[2].set_title("Residuals vs SNR", fontsize=11, fontweight="bold")
    plt.suptitle("Residuals from NR Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "full_catalog_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 6: Summary dashboard ---
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # Panel A: histogram of kappa
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(kappas, bins=30, color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.axvline(consistency.get("kappa_weighted_mean", 0), color="red",
               linestyle="--", linewidth=2, label=f"mean={consistency.get('kappa_weighted_mean', 0):.3f}")
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("kappa", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Kappa Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B: significance histogram
    ax = fig.add_subplot(gs[0, 1])
    significance = np.abs(kappas) / (sigmas + 1e-30)
    ax.hist(significance, bins=30, color="orange", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="1 sigma")
    ax.axvline(2.0, color="red", linestyle="--", alpha=0.5, label="2 sigma")
    ax.set_xlabel("|kappa| / sigma", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Per-Event Significance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel C: A_220 distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(np.log10(a_220s + 1e-30), bins=30, color="green", alpha=0.7,
            edgecolor="k", linewidth=0.5)
    ax.set_xlabel("log10(A_220)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Fitted Fundamental Amplitude", fontsize=11, fontweight="bold")

    # Panel D: kappa vs spin (compact)
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(spins, kappas, c=snrs, cmap="viridis", s=20, alpha=0.6)
    spins_c, kappa_c, _ = generate_kappa_curve()
    ax.plot(spins_c, kappa_c, "r-", linewidth=1.5)
    ax.set_xlabel("Spin", fontsize=10)
    ax.set_ylabel("kappa", fontsize=10)
    ax.set_title("Kappa vs Spin", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel E: kappa vs mass (compact)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(masses, kappas, c=snrs, cmap="viridis", s=20, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Mass (Msun)", fontsize=10)
    ax.set_ylabel("kappa", fontsize=10)
    ax.set_title("Kappa vs Mass", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel F: text summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary_text = (
        f"GRIM-S Full Catalog Analysis\n"
        f"{'='*35}\n"
        f"Events analyzed: {consistency.get('n_events', 0)}\n"
        f"Weighted mean kappa: {consistency.get('kappa_weighted_mean', 0):+.4f}\n"
        f"  +/- {consistency.get('kappa_mean_sigma', 0):.4f}\n"
        f"Significance: {consistency.get('significance_sigma', 0):.1f} sigma\n"
        f"chi2/dof (vs mean): {consistency.get('chi2_per_dof_vs_mean', 0):.2f}\n"
        f"chi2/dof (vs NR):   {consistency.get('chi2_per_dof_vs_nr', 0):.2f}\n"
        f"\nCorrelations:\n"
        f"  kappa-mass: r={consistency.get('correlation_kappa_mass', {}).get('r', 0):+.3f}\n"
        f"  kappa-spin: r={consistency.get('correlation_kappa_spin', {}).get('r', 0):+.3f}\n"
        f"  kappa-SNR:  r={consistency.get('correlation_kappa_snr', {}).get('r', 0):+.3f}\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("GRIM-S: Nonlinear Mode Search — Full Catalog",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(plots_dir / "full_catalog_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to {plots_dir}/full_catalog_*.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GRIM-S: Full Catalog Expansion")
    print("Bown principle: expand the measurement base first.")
    print("=" * 60)

    # Load catalog
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    print(f"\nCatalog: {len(catalog)} events")

    # Step 1: Download all missing strain
    print(f"\n--- Step 1: Bulk download ---")
    dl = bulk_download(catalog, DATA_DIR)
    paths = dl["paths"]

    # Step 2: Full-catalog analysis
    print(f"\n--- Step 2: Full-catalog phase-locked analysis ---")
    analysis = run_full_analysis(catalog, paths, min_total_mass=30.0)

    # Save results
    results = analysis["individual"]
    with open(RESULTS_PATH, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    if len(results) < 3:
        print("\nToo few events analyzed for consistency test. Done.")
        return

    # Step 3: Consistency test
    print(f"\n--- Step 3: Amplitude ratio consistency test ---")
    consistency = amplitude_ratio_consistency(results)

    # Save consistency results
    consistency_path = DATA_DIR / "consistency_test.json"
    with open(consistency_path, "w") as f:
        json.dump(consistency, f, indent=2)

    # Step 4: Plots
    print(f"\n--- Step 4: Generate diagnostic plots ---")
    generate_plots(results, consistency, PLOTS_DIR)

    print(f"\n{'='*60}")
    print("DONE. The instrument has spoken.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
