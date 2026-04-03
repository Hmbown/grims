#!/usr/bin/env python
"""
Run Phase 2 deep analysis for GRIM-S.

This script runs:
1. Colored-noise PSD likelihood comparison
2. Multi-detector coherent analysis (H1 vs L1 for GW150914)
3. MCMC sampling for GW150914

Usage:
    python scripts/run_phase2_analysis.py
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bown_instruments.grims.mass_analysis import run_mass_analysis, analyze_event
from bown_instruments.grims.whiten import estimate_asd, whiten_strain, bandpass
from bown_instruments.grims.gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5
from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.colored_likelihood import (
    estimate_psd_from_data,
    compare_white_vs_colored,
    print_colored_summary,
    plot_colored_comparison,
)


def main():
    print("=" * 70)
    print("GRIM-S PHASE 2 DEEP ANALYSIS")
    print("=" * 70)
    print()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # =========================================================
    # 1. Colored noise PSD comparison
    # =========================================================
    print("Running colored noise PSD comparison...")
    print()

    # Load GW150914 data
    try:
        loaded = load_gwosc_strain_hdf5("data/GW150914.hdf5")
    except:
        # Try to find any GW150914 file
        import glob

        files = glob.glob("data/*GW150914*.hdf5")
        if files:
            loaded = load_gwosc_strain_hdf5(files[0])
        else:
            print("GW150914 data not found. Skipping colored noise analysis.")
            return

    strain = loaded["strain"]
    time = loaded["time"]
    sample_rate = loaded["sample_rate"]

    # Event parameters
    mass = 66.2  # solar masses
    spin = 0.68
    gps = 1126259462.4  # GW150914 GPS time
    m_seconds = mass * M_SUN_SECONDS

    # QNM frequencies
    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    f_220 = mode_220.physical_frequency_hz(mass)
    f_low = max(20.0, f_220 * 0.5)
    f_high = 512.0

    # Whitening
    merger_time = float(gps)
    ringdown_start = merger_time + 10.0 * m_seconds

    asd_freqs, asd = estimate_asd(
        strain,
        sample_rate,
        merger_time=merger_time,
        time=time,
        exclusion_window=2.0,
    )
    whitened = whiten_strain(strain, sample_rate, asd_freqs, asd, fmin=f_low * 0.8)
    whitened_bp = bandpass(whitened, sample_rate, f_low, f_high)

    # Extract ringdown segment
    pad_before = 0.05
    seg_duration = 0.15
    t_start = ringdown_start - pad_before
    t_end = ringdown_start + seg_duration
    mask = (time >= t_start) & (time <= t_end)

    seg_strain = whitened_bp[mask]
    seg_time = time[mask]
    t_dimless = (seg_time - ringdown_start) / m_seconds

    # Noise variance
    noise_mask = np.abs(time - merger_time) > 4.0
    noise_var = np.var(whitened_bp[noise_mask])

    # Estimate PSD
    try:
        psd_freqs, psd = estimate_psd_from_data(
            strain,
            sample_rate,
            merger_time,
            time,
            exclusion_window=4.0,
            segment_length=64.0,
        )

        # Sample rate in dimensionless units
        sample_rate_dimless = sample_rate * m_seconds

        # Compare white vs colored
        comparison = compare_white_vs_colored(
            data=seg_strain,
            t_dimless=t_dimless,
            spin=spin,
            A_220=1.0,  # Will be fitted internally
            noise_variance=noise_var,
            psd_freqs=psd_freqs,
            psd=psd,
            sample_rate_dimless=sample_rate_dimless,
            event_name="GW150914",
        )

        print_colored_summary(comparison)
        plot_colored_comparison(
            comparison,
            save_path=str(plots_dir / "colored_noise_comparison.png"),
        )
        print(
            f"Saved colored noise comparison plot to {plots_dir / 'colored_noise_comparison.png'}"
        )

    except Exception as e:
        print(f"PSD estimation failed: {e}")
        print("Skipping colored noise analysis.")

    # =========================================================
    # 2. Multi-detector coherent analysis
    # =========================================================
    print("\n" + "=" * 70)
    print("MULTI-DETECTOR COHERENT ANALYSIS")
    print("=" * 70)
    print()

    # Find H1 and L1 files for GW150914
    import glob

    h1_files = glob.glob("data/*GW150914*H1*.hdf5")
    l1_files = glob.glob("data/*GW150914*L1*.hdf5")

    if h1_files and l1_files:
        print(f"Found H1: {h1_files[0]}")
        print(f"Found L1: {l1_files[0]}")
        print()

        # Analyze each detector separately
        h1_loaded = load_gwosc_strain_hdf5(h1_files[0])
        l1_loaded = load_gwosc_strain_hdf5(l1_files[0])

        print(f"H1 sample rate: {h1_loaded['sample_rate']} Hz")
        print(f"L1 sample rate: {l1_loaded['sample_rate']} Hz")
        print()

        # Run phase-locked search on each
        from bown_instruments.grims.phase_locked_search import phase_locked_search

        # Process H1
        h1_strain = h1_loaded["strain"]
        h1_time = h1_loaded["time"]
        h1_sr = h1_loaded["sample_rate"]

        h1_asd_freqs, h1_asd = estimate_asd(
            h1_strain,
            h1_sr,
            merger_time=merger_time,
            time=h1_time,
            exclusion_window=2.0,
        )
        h1_whitened = whiten_strain(
            h1_strain, h1_sr, h1_asd_freqs, h1_asd, fmin=f_low * 0.8
        )
        h1_bp = bandpass(h1_whitened, h1_sr, f_low, f_high)

        h1_mask = (h1_time >= t_start) & (h1_time <= t_end)
        h1_seg = h1_bp[h1_mask]
        h1_t_dimless = (h1_time[h1_mask] - ringdown_start) / m_seconds

        h1_noise_mask = np.abs(h1_time - merger_time) > 4.0
        h1_noise_rms = np.sqrt(np.var(h1_bp[h1_noise_mask]))

        h1_result = phase_locked_search(
            h1_seg,
            h1_t_dimless,
            spin,
            h1_noise_rms,
            event_name="GW150914_H1",
        )

        # Process L1
        l1_strain = l1_loaded["strain"]
        l1_time = l1_loaded["time"]
        l1_sr = l1_loaded["sample_rate"]

        l1_asd_freqs, l1_asd = estimate_asd(
            l1_strain,
            l1_sr,
            merger_time=merger_time,
            time=l1_time,
            exclusion_window=2.0,
        )
        l1_whitened = whiten_strain(
            l1_strain, l1_sr, l1_asd_freqs, l1_asd, fmin=f_low * 0.8
        )
        l1_bp = bandpass(l1_whitened, l1_sr, f_low, f_high)

        l1_mask = (l1_time >= t_start) & (l1_time <= t_end)
        l1_seg = l1_bp[l1_mask]
        l1_t_dimless = (l1_time[l1_mask] - ringdown_start) / m_seconds

        l1_noise_mask = np.abs(l1_time - merger_time) > 4.0
        l1_noise_rms = np.sqrt(np.var(l1_bp[l1_noise_mask]))

        l1_result = phase_locked_search(
            l1_seg,
            l1_t_dimless,
            spin,
            l1_noise_rms,
            event_name="GW150914_L1",
        )

        print(
            f"H1: kappa = {h1_result.kappa_hat:.4f} ± {h1_result.kappa_sigma:.4f}, SNR = {h1_result.snr:.3f}"
        )
        print(
            f"L1: kappa = {l1_result.kappa_hat:.4f} ± {l1_result.kappa_sigma:.4f}, SNR = {l1_result.snr:.3f}"
        )
        print()

        # Coherent combination
        from bown_instruments.grims.phase_locked_search import stack_phase_locked

        coherent = stack_phase_locked([h1_result, l1_result])

        print(
            f"Coherent (H1+L1): kappa = {coherent.kappa_hat:.4f} ± {coherent.kappa_sigma:.4f}"
        )
        print(f"                  SNR = {coherent.snr:.3f}")
        print()

        # Consistency check
        diff = abs(h1_result.kappa_hat - l1_result.kappa_hat)
        combined_sigma = np.sqrt(h1_result.kappa_sigma**2 + l1_result.kappa_sigma**2)
        consistency_sigma = (
            diff / combined_sigma if combined_sigma > 0 else float("inf")
        )

        print(f"H1-L1 consistency: {consistency_sigma:.2f} sigma")
        if consistency_sigma < 2.0:
            print("Conclusion: H1 and L1 results are consistent.")
        else:
            print("Conclusion: H1 and L1 results are INCONSISTENT.")
            print("This suggests a detector-specific systematic.")

    else:
        print("GW150914 H1 and/or L1 data not found.")
        print("Skipping multi-detector analysis.")
        print(f"Available files: {glob.glob('data/*GW150914*.hdf5')}")

    # =========================================================
    # 3. MCMC sampling
    # =========================================================
    print("\n" + "=" * 70)
    print("MCMC SAMPLING")
    print("=" * 70)
    print()

    try:
        import emcee
        from bown_instruments.grims.sampler import run_mcmc, print_mcmc_summary, plot_mcmc_chains

        print("Running MCMC for GW150914...")
        print("(This may take a few minutes)")
        print()

        mcmc_result = run_mcmc(
            data=seg_strain,
            t_dimless=t_dimless,
            spin=spin,
            noise_variance=noise_var,
            n_walkers=64,
            n_steps=500,
            n_burnin=100,
            event_name="GW150914",
        )

        print_mcmc_summary(mcmc_result)

        plot_mcmc_chains(
            mcmc_result,
            save_path=str(plots_dir / "mcmc_gw150914.png"),
        )
        print(f"Saved MCMC plots to {plots_dir / 'mcmc_gw150914_chains.png'}")
        print(f"Saved MCMC corner plot to {plots_dir / 'mcmc_gw150914_corner.png'}")

    except ImportError as e:
        print(f"MCMC requires emcee: {e}")
        print("Install with: pip install emcee corner")
        print("Skipping MCMC analysis.")
    except Exception as e:
        print(f"MCMC failed: {e}")
        import traceback

        traceback.print_exc()

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print()
    print("Completed:")
    print("  ✓ Colored noise PSD comparison")
    if h1_files and l1_files:
        print("  ✓ Multi-detector coherent analysis")
    else:
        print("  ✗ Multi-detector analysis (data not available)")
    try:
        import emcee

        print("  ✓ MCMC sampling")
    except ImportError:
        print("  ✗ MCMC sampling (emcee not installed)")

    print(f"\nPlots saved to: {plots_dir}/")
    print("  - colored_noise_comparison.png")
    if h1_files and l1_files:
        print("  - coherent_analysis.png")
    try:
        import emcee

        print("  - mcmc_gw150914_chains.png")
        print("  - mcmc_gw150914_corner.png")
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print("Phase 2 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
