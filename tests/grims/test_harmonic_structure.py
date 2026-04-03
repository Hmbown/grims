"""
Harmonic structure tests for GRIM-S.

Five tests probing whether the nonlinear QNM spectrum has the structure
of a compressed harmonic series — whether the second-order modes fill
information gaps, carry independent Fisher information, reduce residual
entropy, converge to simple frequency ratios, and produce a start-time
signature consistent with a real astrophysical signal rather than noise.

These tests do not prove the 2-sigma excess is real. They probe the
*structure* of the hypothesis: if nature uses nonlinear mode coupling,
what mathematical fingerprints would that leave, and are they present?

Usage:
    python tests/test_harmonic_structure.py

Requires: qnm, numpy, scipy, matplotlib, h5py
Data:     data/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5
"""

import sys
import os
import warnings
from pathlib import Path
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — this script lives in tests/, project root is one level up.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.whiten import prepare_ringdown_for_analysis
from bown_instruments.grims.phase_locked_search import (
    phase_locked_search,
    fit_fundamental_mode,
    build_phase_locked_template,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ===========================================================================
# CONSTANTS
# ===========================================================================
M_SUN_SECONDS = 4.925491025543576e-06
GW150914_SPIN = 0.69
GW150914_MASS = 63.1  # Msun


# ===========================================================================
# TEST 1: Harmonic Completeness Spectrum
# ===========================================================================
def test_harmonic_completeness():
    """
    For spins 0.0 to 0.95, compute all QNM frequencies up to l=6
    (linear fundamental modes) plus all second-order nonlinear
    combinations omega(l,m,0) + omega(l',m',0). Ask: does the
    nonlinear (4,4) mode fill a GAP in the linear spectrum?

    Quantify gap-filling by comparing minimum adjacent-mode spacing
    with and without the nonlinear modes.
    """
    print("=" * 72)
    print("TEST 1: Harmonic Completeness Spectrum")
    print("=" * 72)
    print()
    print("Question: Does the nonlinear (4,4) mode fill a gap in the")
    print("linear QNM spectrum, or is it redundant?")
    print()

    catalog = KerrQNMCatalog()
    spins = np.linspace(0.0, 0.95, 20)

    # Collect all linear modes with l=2..6, m=l (prograde co-rotating
    # modes dominate the ringdown; m=l is the loudest for each l).
    linear_lm = [(l, l) for l in range(2, 7)]

    gap_fill_ratios = []

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for spin_idx, spin in enumerate(spins):
        # Linear frequencies (dimensionless, real part only)
        linear_freqs = []
        linear_labels = []
        for (l, m) in linear_lm:
            try:
                mode = catalog.linear_mode(l, m, 0, spin)
                linear_freqs.append(mode.frequency)
                linear_labels.append(f"({l},{m},0)")
            except Exception:
                pass  # Some modes may not converge at extreme spins

        linear_freqs_sorted = np.sort(linear_freqs)

        # All second-order nonlinear combinations
        nl_freqs = []
        nl_labels = []
        for i, (l1, m1) in enumerate(linear_lm):
            for j, (l2, m2) in enumerate(linear_lm):
                if j < i:
                    continue  # avoid duplicates
                try:
                    nl = catalog.nonlinear_mode_quadratic(
                        spin,
                        parent_l1=l1, parent_m1=m1, parent_n1=0,
                        parent_l2=l2, parent_m2=m2, parent_n2=0,
                    )
                    nl_freqs.append(nl.frequency)
                    nl_labels.append(f"NL({l1}{m1}+{l2}{m2})")
                except Exception:
                    pass

        # Combined spectrum
        all_freqs = np.sort(np.concatenate([linear_freqs_sorted, nl_freqs]))

        # Minimum spacing: linear only vs linear + nonlinear
        if len(linear_freqs_sorted) > 1:
            min_gap_linear = np.min(np.diff(linear_freqs_sorted))
        else:
            min_gap_linear = np.inf
        if len(all_freqs) > 1:
            min_gap_all = np.min(np.diff(all_freqs))
        else:
            min_gap_all = np.inf

        ratio = min_gap_all / min_gap_linear if min_gap_linear > 0 else 1.0
        gap_fill_ratios.append(ratio)

        # Plot: frequency spectrum at GW150914 spin
        if abs(spin - GW150914_SPIN) < 0.03:
            ax = axes[0]
            y_lin = np.ones_like(linear_freqs_sorted)
            y_nl = 0.5 * np.ones(len(nl_freqs))
            ax.stem(linear_freqs_sorted, y_lin, linefmt="C0-",
                    markerfmt="C0o", basefmt=" ", label="Linear modes (l=2..6)")
            ax.stem(nl_freqs, y_nl, linefmt="C1--",
                    markerfmt="C1^", basefmt=" ", label="Nonlinear 2nd-order")

            # Highlight the NL(4,4) = 2*omega(2,2,0)
            nl44_freq = 2.0 * catalog.linear_mode(2, 2, 0, spin).frequency
            ax.axvline(nl44_freq, color="red", ls=":", lw=2,
                       label=f"NL(4,4) = 2*omega_220 = {nl44_freq:.4f}")

            # Also mark linear (4,4,0)
            lin44_freq = catalog.linear_mode(4, 4, 0, spin).frequency
            ax.axvline(lin44_freq, color="green", ls=":", lw=2,
                       label=f"Linear (4,4,0) = {lin44_freq:.4f}")

            ax.set_xlabel("Dimensionless frequency (M * omega)")
            ax.set_ylabel("Mode type")
            ax.set_title(f"QNM Spectrum at spin a = {spin:.2f} (GW150914)")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_yticks([0.5, 1.0])
            ax.set_yticklabels(["Nonlinear", "Linear"])

    # Plot: gap-filling ratio vs spin
    ax2 = axes[1]
    ax2.plot(spins, gap_fill_ratios, "ko-", lw=2)
    ax2.axhline(1.0, color="gray", ls="--", label="No gap filling")
    ax2.set_xlabel("Spin parameter a")
    ax2.set_ylabel("min_gap(linear+NL) / min_gap(linear)")
    ax2.set_title("Gap-Filling Ratio: <1 means NL modes fill spectral gaps")
    ax2.legend()
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "test1_harmonic_completeness.png", dpi=150)
    plt.close(fig)
    print(f"  Figure saved: plots/test1_harmonic_completeness.png")

    # Summary
    mean_ratio = np.mean(gap_fill_ratios)
    print(f"\n  Gap-filling ratio (mean over spins): {mean_ratio:.3f}")
    if mean_ratio < 1.0:
        print("  RESULT: Nonlinear modes FILL gaps in the linear spectrum.")
        print("  The NL modes are not redundant -- they add spectral coverage")
        print("  in frequency bands that linear modes do not occupy.")
    else:
        print("  RESULT: Nonlinear modes do NOT significantly fill gaps.")
        print("  They land near existing linear modes.")

    # Detail at GW150914 spin
    cat = KerrQNMCatalog()
    lin_440 = cat.linear_mode(4, 4, 0, GW150914_SPIN)
    nl_44 = cat.nonlinear_mode_quadratic(GW150914_SPIN)
    sep = abs(lin_440.frequency - nl_44.frequency)
    print(f"\n  At spin={GW150914_SPIN}:")
    print(f"    Linear (4,4,0) freq:    {lin_440.frequency:.6f}")
    print(f"    NL(4,4) freq:           {nl_44.frequency:.6f}")
    print(f"    Separation:             {sep:.6f} ({sep/lin_440.frequency:.1%} of f_440)")
    print()

# ===========================================================================
# TEST 2: Fisher Information Content of the QNM Spectrum
# ===========================================================================
def test_fisher_information():
    """
    Compute the Fisher information matrix for (M, a) from observed QNM
    frequencies. Compare three mode sets:
      A) Linear only: (2,2,0), (3,3,0), (4,4,0)
      B) Linear + NL(4,4): add the nonlinear mode
      C) Linear + overtone: add (2,2,1)

    The Fisher information is:
      F_ij = sum_modes (d omega_k / d theta_i) * (d omega_k / d theta_j) / sigma_k^2

    where theta = (M, a) and sigma_k is the measurement uncertainty on each
    frequency. We use a fiducial sigma_k proportional to 1/SNR for each mode.

    The "compression" hypothesis predicts that NL(4,4) carries more
    independent information than the overtone (2,2,1), because the
    nonlinear mode encodes the coupling structure, not just a shifted
    decay rate.
    """
    print("=" * 72)
    print("TEST 2: Fisher Information Content of the QNM Spectrum")
    print("=" * 72)
    print()
    print("Question: Does the nonlinear mode carry more Fisher information")
    print("about (M, a) than the first overtone?")
    print()

    catalog = KerrQNMCatalog()

    # Fiducial parameters: GW150914
    M0 = GW150914_MASS
    a0 = GW150914_SPIN

    # Step sizes for numerical derivatives
    dM = 0.01 * M0  # 1% perturbation
    da = 0.005       # small spin perturbation

    def get_physical_freqs(mass, spin, mode_specs):
        """Return physical frequencies in Hz for a list of mode specifications.
        Each spec is either (l, m, n) for linear or 'NL' for nonlinear (4,4)."""
        freqs = []
        for spec in mode_specs:
            if spec == "NL":
                mode = catalog.nonlinear_mode_quadratic(spin)
            else:
                l, m, n = spec
                mode = catalog.linear_mode(l, m, n, spin)
            freqs.append(mode.physical_frequency_hz(mass))
        return np.array(freqs)

    def fisher_matrix(mass, spin, mode_specs, sigma_f):
        """Compute 2x2 Fisher information matrix for (M, a).

        sigma_f: array of frequency uncertainties (Hz) for each mode.
        """
        f0 = get_physical_freqs(mass, spin, mode_specs)
        f_dM = get_physical_freqs(mass + dM, spin, mode_specs)
        f_da = get_physical_freqs(mass, spin + da, mode_specs)

        # Numerical partial derivatives
        df_dM = (f_dM - f0) / dM
        df_da = (f_da - f0) / da

        # Fisher matrix: F_ij = sum_k (df_k/dtheta_i)(df_k/dtheta_j) / sigma_k^2
        F = np.zeros((2, 2))
        for k in range(len(mode_specs)):
            w = 1.0 / sigma_f[k]**2
            F[0, 0] += df_dM[k]**2 * w
            F[0, 1] += df_dM[k] * df_da[k] * w
            F[1, 0] += df_dM[k] * df_da[k] * w
            F[1, 1] += df_da[k]**2 * w
        return F

    # Mode sets
    set_A = [(2, 2, 0), (3, 3, 0), (4, 4, 0)]
    set_B = [(2, 2, 0), (3, 3, 0), (4, 4, 0), "NL"]
    set_C = [(2, 2, 0), (3, 3, 0), (4, 4, 0), (2, 2, 1)]

    # Fiducial measurement uncertainties: 10 Hz for each mode
    # (roughly appropriate for current LIGO ringdown SNR)
    sigma_base = 10.0  # Hz

    results = {}
    for label, modes in [("A: Linear only", set_A),
                          ("B: Linear + NL(4,4)", set_B),
                          ("C: Linear + overtone (2,2,1)", set_C)]:
        sigma_f = np.full(len(modes), sigma_base)
        F = fisher_matrix(M0, a0, modes, sigma_f)

        # Total information = determinant (volume of error ellipse shrinks)
        det_F = np.linalg.det(F)
        # Trace = total diagonal information
        trace_F = np.trace(F)
        # Conditional uncertainties
        try:
            cov = np.linalg.inv(F)
            sigma_M = np.sqrt(cov[0, 0])
            sigma_a = np.sqrt(cov[1, 1])
            correlation = cov[0, 1] / (sigma_M * sigma_a)
        except np.linalg.LinAlgError:
            sigma_M = sigma_a = correlation = np.inf

        results[label] = {
            "det_F": det_F,
            "trace_F": trace_F,
            "sigma_M": sigma_M,
            "sigma_a": sigma_a,
            "correlation": correlation,
            "F": F,
        }

        print(f"  {label}:")
        print(f"    det(F) = {det_F:.2e}  (higher = more information)")
        print(f"    sigma_M = {sigma_M:.3f} Msun,  sigma_a = {sigma_a:.5f}")
        print(f"    M-a correlation = {correlation:.3f}")
        print()

    # Compare: information gain from NL vs overtone
    det_A = results["A: Linear only"]["det_F"]
    det_B = results["B: Linear + NL(4,4)"]["det_F"]
    det_C = results["C: Linear + overtone (2,2,1)"]["det_F"]

    gain_NL = det_B / det_A if det_A > 0 else np.inf
    gain_OT = det_C / det_A if det_A > 0 else np.inf

    print(f"  Information gain from NL(4,4):      det(F_B)/det(F_A) = {gain_NL:.2f}x")
    print(f"  Information gain from overtone:      det(F_C)/det(F_A) = {gain_OT:.2f}x")
    print()

    if gain_NL > gain_OT:
        print("  RESULT: The nonlinear mode carries MORE Fisher information")
        print("  than the first overtone. Consistent with 'compression' hypothesis:")
        print("  the NL mode encodes coupling structure, not just a shifted decay.")
    else:
        print("  RESULT: The first overtone carries more Fisher information.")
        print("  The nonlinear mode is informationally subordinate to the overtone.")

    # Scan over spins
    spins_scan = np.linspace(0.1, 0.9, 17)
    gains_nl = []
    gains_ot = []
    for a in spins_scan:
        sigma_f_3 = np.full(3, sigma_base)
        sigma_f_4 = np.full(4, sigma_base)
        F_a = fisher_matrix(M0, a, set_A, sigma_f_3)
        F_b = fisher_matrix(M0, a, set_B, sigma_f_4)
        F_c = fisher_matrix(M0, a, set_C, sigma_f_4)
        d_a = np.linalg.det(F_a)
        if d_a > 0:
            gains_nl.append(np.linalg.det(F_b) / d_a)
            gains_ot.append(np.linalg.det(F_c) / d_a)
        else:
            gains_nl.append(1.0)
            gains_ot.append(1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spins_scan, gains_nl, "C1o-", lw=2, label="+ NL(4,4)")
    ax.plot(spins_scan, gains_ot, "C2s--", lw=2, label="+ overtone (2,2,1)")
    ax.axhline(1.0, color="gray", ls=":", label="Baseline (linear only)")
    ax.set_xlabel("Spin parameter a")
    ax.set_ylabel("det(F) / det(F_linear)")
    ax.set_title("Fisher Information Gain: Nonlinear Mode vs Overtone")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "test2_fisher_information.png", dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: plots/test2_fisher_information.png")
    print()

# ===========================================================================
# TEST 3: Entropy of the Residual
# ===========================================================================
def test_residual_entropy():
    """
    Take GW150914 H1 whitened data. Compute Shannon entropy of the
    ringdown residual after subtracting:
      Case A: only the (2,2,0) fundamental
      Case B: (2,2,0) + best-fit nonlinear template

    If the nonlinear mode is a real signal, subtracting it should
    REDUCE the entropy (residual becomes more Gaussian/noise-like,
    less structured). If it is noise, entropy should not change.

    Shannon entropy of a discretized signal is computed via histogram
    binning of the residual amplitudes.
    """
    print("=" * 72)
    print("TEST 3: Entropy of the Residual")
    print("=" * 72)
    print()
    print("Question: Does subtracting the NL template reduce the Shannon")
    print("entropy of the ringdown residual?")
    print()

    # Load and whiten GW150914 H1 data
    print("  Loading GW150914 H1 data...")
    prep = prepare_ringdown_for_analysis(
        "GW150914",
        data_dir=str(PROJECT_ROOT / "data"),
        detector="H1",
        t_start_m=10.0,
    )
    data = prep["strain_whitened"]
    t = prep["t_dimless"]
    spin = prep["event"]["remnant_spin"]
    noise_rms = prep["noise_rms"]

    def shannon_entropy(x, n_bins=50):
        """Compute Shannon entropy of a signal via histogram."""
        counts, _ = np.histogram(x, bins=n_bins, density=False)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def gaussian_entropy(sigma, n_bins=50):
        """Theoretical Shannon entropy of a Gaussian with the same binning."""
        # For a Gaussian with std sigma discretized into n_bins over
        # a reasonable range, the entropy approaches
        # 0.5 * log2(2*pi*e*sigma^2) + log2(n_bins / range)
        # We just use the continuous approximation.
        return 0.5 * np.log2(2.0 * np.pi * np.e * sigma**2)

    # Case A: subtract only (2,2,0)
    fit_220 = fit_fundamental_mode(data, t, spin)
    residual_A = fit_220["residual"]
    mask = t >= 0
    residual_A_pos = residual_A[mask]

    # Case B: subtract (2,2,0) + NL template
    # The NL template is phase-locked; run the matched filter to get kappa
    pl_result = phase_locked_search(data, t, spin, noise_rms, "GW150914")
    nl_template = build_phase_locked_template(
        t, spin, fit_220["amplitude"], fit_220["phase"]
    )
    residual_B = residual_A - pl_result.kappa_hat * nl_template
    residual_B_pos = residual_B[mask]

    # Pure noise reference: off-source whitened data has known properties
    noise_ref = np.random.normal(0, noise_rms, len(residual_A_pos))

    # Compute entropies
    H_A = shannon_entropy(residual_A_pos)
    H_B = shannon_entropy(residual_B_pos)
    H_noise = shannon_entropy(noise_ref)

    # Also compute Gaussianity: excess kurtosis (0 for Gaussian)
    kurt_A = float(np.mean((residual_A_pos / np.std(residual_A_pos))**4) - 3.0)
    kurt_B = float(np.mean((residual_B_pos / np.std(residual_B_pos))**4) - 3.0)
    kurt_noise = float(np.mean((noise_ref / np.std(noise_ref))**4) - 3.0)

    print(f"  Shannon entropy (50-bin histogram):")
    print(f"    Residual after (2,2,0) only:        H_A = {H_A:.3f} bits")
    print(f"    Residual after (2,2,0) + NL:        H_B = {H_B:.3f} bits")
    print(f"    Pure Gaussian noise reference:       H_n = {H_noise:.3f} bits")
    print(f"    Entropy reduction:                   dH  = {H_A - H_B:.3f} bits")
    print()
    print(f"  Excess kurtosis (0 = Gaussian):")
    print(f"    Residual after (2,2,0) only:        {kurt_A:+.3f}")
    print(f"    Residual after (2,2,0) + NL:        {kurt_B:+.3f}")
    print(f"    Gaussian noise reference:            {kurt_noise:+.3f}")
    print()

    delta_H = H_A - H_B
    if delta_H > 0.05:
        print(f"  RESULT: Subtracting the NL template REDUCES entropy by {delta_H:.3f} bits.")
        print("  The residual becomes more noise-like. Consistent with a real signal.")
    elif delta_H > 0:
        print(f"  RESULT: Marginal entropy reduction ({delta_H:.3f} bits).")
        print("  Suggestive but not conclusive.")
    else:
        print(f"  RESULT: No entropy reduction ({delta_H:.3f} bits).")
        print("  The NL template does not make the residual more Gaussian.")
        print("  This does NOT prove the signal is absent -- it may mean our")
        print("  template is imperfect or the effect is below the noise floor.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(residual_A_pos, bins=50, density=True, alpha=0.7,
                 color="C0", label=f"After (2,2,0)\nH={H_A:.2f} bits")
    axes[0].set_title("Residual: (2,2,0) subtracted")
    axes[0].legend()
    axes[0].set_xlabel("Whitened strain")

    axes[1].hist(residual_B_pos, bins=50, density=True, alpha=0.7,
                 color="C1", label=f"After (2,2,0)+NL\nH={H_B:.2f} bits")
    axes[1].set_title("Residual: (2,2,0) + NL subtracted")
    axes[1].legend()
    axes[1].set_xlabel("Whitened strain")

    axes[2].hist(noise_ref, bins=50, density=True, alpha=0.7,
                 color="C2", label=f"Gaussian noise\nH={H_noise:.2f} bits")
    axes[2].set_title("Reference: pure Gaussian noise")
    axes[2].legend()
    axes[2].set_xlabel("Whitened strain")

    for ax in axes:
        ax.set_ylabel("Probability density")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "test3_residual_entropy.png", dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: plots/test3_residual_entropy.png")
    print()

# ===========================================================================
# TEST 4: Frequency Ratio Universality
# ===========================================================================
def test_frequency_ratio_universality():
    """
    For each (l,m) linear mode, compute omega(l,m,0) / omega(2,2,0)
    as a function of spin. Do the ratios converge to simple fractions?

    The "harmonics compress into a whole" idea predicts that the QNM
    spectrum has a near-harmonic structure — the ratios should cluster
    near simple rational numbers, like musical harmonics.

    We check this by finding the closest simple fraction p/q with
    q <= 12 for each ratio and measuring the deviation.
    """
    print("=" * 72)
    print("TEST 4: Frequency Ratio Universality")
    print("=" * 72)
    print()
    print("Question: Do QNM frequency ratios converge to simple fractions")
    print("(like musical harmonics)?")
    print()

    catalog = KerrQNMCatalog()
    spins = np.linspace(0.01, 0.95, 50)  # avoid a=0 exactly

    modes_to_check = [(3, 3), (4, 4), (5, 5), (6, 6), (3, 2), (4, 3)]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Build a set of "simple" fractions with denominator <= 12
    simple_fractions = set()
    for q in range(1, 13):
        for p in range(1, 6 * q + 1):
            simple_fractions.add(Fraction(p, q))
    simple_fractions = sorted(simple_fractions)
    simple_values = np.array([float(f) for f in simple_fractions])

    all_deviations = {}

    for mode_idx, (l, m) in enumerate(modes_to_check):
        ratios = []
        for spin in spins:
            try:
                f_lm = catalog.linear_mode(l, m, 0, spin).frequency
                f_220 = catalog.linear_mode(2, 2, 0, spin).frequency
                ratios.append(f_lm / f_220)
            except Exception:
                ratios.append(np.nan)

        ratios = np.array(ratios)
        label = f"({l},{m},0)"

        axes[0].plot(spins, ratios, color=colors[mode_idx], lw=2,
                     label=label)

        # Find closest simple fraction at the GW150914 spin
        idx_gw = np.argmin(np.abs(spins - GW150914_SPIN))
        if not np.isnan(ratios[idx_gw]):
            r = ratios[idx_gw]
            closest_idx = np.argmin(np.abs(simple_values - r))
            closest_frac = simple_fractions[closest_idx]
            deviation = abs(r - float(closest_frac))
            all_deviations[label] = {
                "ratio": r,
                "closest_fraction": str(closest_frac),
                "deviation": deviation,
                "fractional_deviation": deviation / r,
            }

        # Deviation from nearest simple fraction as function of spin
        deviations = []
        for r in ratios:
            if np.isnan(r):
                deviations.append(np.nan)
            else:
                dev = np.min(np.abs(simple_values - r))
                deviations.append(dev)

        axes[1].plot(spins, deviations, color=colors[mode_idx], lw=2,
                     label=label)

    # Add the NL(4,4) ratio = 2 * omega_220 / omega_220 = 2.0 by construction
    axes[0].axhline(2.0, color="red", ls=":", lw=2,
                     label="NL(4,4) / (2,2,0) = 2 (exact)")

    axes[0].set_xlabel("Spin parameter a")
    axes[0].set_ylabel("omega(l,m,0) / omega(2,2,0)")
    axes[0].set_title("QNM Frequency Ratios vs Spin")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Mark simple fractions on the y-axis
    for frac in [Fraction(3, 2), Fraction(2, 1), Fraction(5, 2),
                 Fraction(3, 1), Fraction(7, 2), Fraction(4, 1)]:
        axes[0].axhline(float(frac), color="gray", ls="--", alpha=0.3, lw=0.5)

    axes[1].set_xlabel("Spin parameter a")
    axes[1].set_ylabel("Deviation from nearest simple fraction (q<=12)")
    axes[1].set_title("How Close to Simple Rational Ratios?")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "test4_frequency_ratios.png", dpi=150)
    plt.close(fig)
    print(f"  Figure saved: plots/test4_frequency_ratios.png")

    print(f"\n  Frequency ratios at spin = {GW150914_SPIN}:")
    print(f"  {'Mode':<12} {'Ratio':>8} {'Nearest p/q':>12} {'Deviation':>10}")
    print("  " + "-" * 46)
    for label, info in all_deviations.items():
        print(f"  {label:<12} {info['ratio']:8.4f} {info['closest_fraction']:>12}"
              f" {info['deviation']:10.4f}")

    # Assessment: are the deviations small?
    mean_dev = np.mean([v["fractional_deviation"] for v in all_deviations.values()])
    print(f"\n  Mean fractional deviation from simple fractions: {mean_dev:.4f}")

    if mean_dev < 0.02:
        print("  RESULT: QNM ratios cluster VERY close to simple fractions.")
        print("  The spectrum has near-harmonic structure.")
    elif mean_dev < 0.05:
        print("  RESULT: QNM ratios are MODERATELY close to simple fractions.")
        print("  Some harmonic tendency, but the spectrum is not truly harmonic.")
    else:
        print("  RESULT: QNM ratios are NOT close to simple fractions.")
        print("  The Kerr QNM spectrum is fundamentally anharmonic.")
        print("  The 'musical harmonics' analogy is poetic, not literal.")

    print()
    print("  CAVEAT: The NL(4,4) ratio of exactly 2.0 is true by construction")
    print("  (it is defined as 2 * omega_220). This test asks whether the")
    print("  LINEAR spectrum also tends toward simple ratios -- if so, the")
    print("  nonlinear modes are the natural completion of a near-harmonic series.")
    print()

# ===========================================================================
# TEST 5: Start Time Stability of the 2-Sigma Excess
# ===========================================================================
def test_start_time_stability():
    """
    Run the phase-locked search on GW150914 H1 at multiple start times.
    A real signal should:
      - Be absent at very early start times (before ringdown begins)
      - Peak around the ringdown onset (~10-15 M)
      - Decay for later start times (signal has damped away)

    A noise fluctuation would show random behavior across start times.
    A noise line at 541 Hz would produce constant SNR at all start times.
    """
    print("=" * 72)
    print("TEST 5: Start Time Stability of the 2-Sigma Excess")
    print("=" * 72)
    print()
    print("Question: Does the excess at the NL frequency show the expected")
    print("start-time profile of a damped ringdown signal?")
    print()

    t_starts = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]

    snrs = []
    kappas = []
    kappa_sigmas = []

    for t_m in t_starts:
        print(f"  t_start = {t_m:5.1f} M ... ", end="", flush=True)
        try:
            prep = prepare_ringdown_for_analysis(
                "GW150914",
                data_dir=str(PROJECT_ROOT / "data"),
                detector="H1",
                t_start_m=t_m,
            )
            result = phase_locked_search(
                prep["strain_whitened"],
                prep["t_dimless"],
                spin=prep["event"]["remnant_spin"],
                noise_rms=prep["noise_rms"],
                event_name="GW150914",
            )
            snrs.append(result.snr)
            kappas.append(result.kappa_hat)
            kappa_sigmas.append(result.kappa_sigma)
            print(f"SNR = {result.snr:+6.2f},  kappa = {result.kappa_hat:+6.3f}"
                  f" +/- {result.kappa_sigma:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
            snrs.append(np.nan)
            kappas.append(np.nan)
            kappa_sigmas.append(np.nan)

    snrs = np.array(snrs)
    kappas = np.array(kappas)
    kappa_sigmas = np.array(kappa_sigmas)

    # Classify the pattern
    print()
    valid = ~np.isnan(snrs)
    if np.sum(valid) < 3:
        print("  RESULT: Too few valid measurements to classify the pattern.")
        return

    snrs_valid = snrs[valid]
    t_valid = np.array(t_starts)[valid]

    # Does SNR peak at an intermediate start time?
    peak_idx = np.argmax(np.abs(snrs_valid))
    peak_t = t_valid[peak_idx]

    # Is there a decay after the peak?
    if peak_idx < len(snrs_valid) - 1:
        post_peak = np.abs(snrs_valid[peak_idx + 1:])
        decays = np.all(np.diff(post_peak) <= 0.5)  # allow small fluctuations
    else:
        decays = False

    # Is the early-time SNR low?
    early = np.abs(snrs_valid[:2])
    early_low = np.all(early < np.abs(snrs_valid[peak_idx]) * 0.5)

    # Is the SNR roughly constant (noise line)?
    snr_std = np.std(np.abs(snrs_valid))
    snr_mean = np.mean(np.abs(snrs_valid))
    is_constant = snr_std / (snr_mean + 1e-10) < 0.2

    print(f"  Peak |SNR| = {np.abs(snrs_valid[peak_idx]):.2f} at t_start = {peak_t:.0f} M")
    print(f"  SNR std/mean = {snr_std / (snr_mean + 1e-10):.2f}")
    print()

    if is_constant:
        print("  PATTERN: Constant SNR across start times.")
        print("  --> Consistent with a NOISE LINE (instrumental artifact)")
        print("      at the NL frequency, NOT a transient ringdown signal.")
    elif early_low and peak_t >= 5 and peak_t <= 20:
        print(f"  PATTERN: SNR peaks at t_start = {peak_t:.0f} M,")
        print("  low at early times, decaying at late times.")
        print("  --> Consistent with a TRANSIENT RINGDOWN signal.")
        print("      This is what a real nonlinear mode would look like:")
        print("      it turns on when the ringdown begins and damps away.")
        if not decays:
            print("  NOTE: Post-peak decay is not monotonic. This could be")
            print("  noise fluctuations or the effect of the analysis window.")
    else:
        print("  PATTERN: Irregular SNR profile.")
        print("  --> No clear ringdown signature. Could be noise.")

    print()
    print("  IMPORTANT CAVEAT: A 2-sigma excess is not a detection.")
    print("  This test checks whether the excess BEHAVES like a signal,")
    print("  not whether it IS one. A noise fluctuation can mimic any")
    print("  pattern for a single realization. Only stacking across")
    print("  events or next-generation detectors can settle this.")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(t_starts, snrs, "ko-", lw=2, markersize=8)
    ax1.axhline(0, color="gray", ls="--")
    ax1.axhline(2.0, color="red", ls=":", alpha=0.5, label="2-sigma")
    ax1.axhline(-2.0, color="red", ls=":", alpha=0.5)
    ax1.set_ylabel("Matched-filter SNR")
    ax1.set_title("GW150914 H1: Nonlinear Mode SNR vs Ringdown Start Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.errorbar(t_starts, kappas, yerr=kappa_sigmas, fmt="bs-", lw=2,
                 markersize=8, capsize=4)
    ax2.axhline(0, color="gray", ls="--")
    ax2.set_xlabel("Ringdown start time (M after merger)")
    ax2.set_ylabel("kappa (NL coupling coefficient)")
    ax2.set_title("Measured kappa vs Start Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "test5_start_time_stability.png", dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: plots/test5_start_time_stability.png")
    print()

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print()
    print("=" * 72)
    print("  GRIM-S: Harmonic Structure Tests")
    print("  Gravitational Intermodulation Spectrometer")
    print()
    print("  'Where is the mathematical elegance?' -- Ralph Bown")
    print("=" * 72)
    print()
    print("  These tests probe whether the nonlinear QNM spectrum has")
    print("  the structure of a compressed harmonic series.")
    print()
    print("  What they test:")
    print("    1. Spectral gap-filling by nonlinear modes")
    print("    2. Fisher information gain from NL vs overtone")
    print("    3. Entropy reduction when NL template is subtracted")
    print("    4. Universality of frequency ratios")
    print("    5. Start-time profile of the 2-sigma excess")
    print()
    print("  What they do NOT prove:")
    print("    - That the 2-sigma excess is real (it could be noise)")
    print("    - That GR is correct (these tests assume it)")
    print("    - That 'information compression' is physics (it is a metaphor)")
    print()

    np.random.seed(42)

    # Tests 1, 2, and 4 use only the QNM catalog -- no real data needed.
    results = {}

    results["test1"] = test_harmonic_completeness()
    results["test2"] = test_fisher_information()

    # Tests 3 and 5 require GW150914 H1 data.
    try:
        results["test3"] = test_residual_entropy()
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")
        print("  (Requires GW150914 H1 data in data/)")
        results["test3"] = None

    results["test4"] = test_frequency_ratio_universality()

    try:
        results["test5"] = test_start_time_stability()
    except Exception as e:
        print(f"  TEST 5 FAILED: {e}")
        print("  (Requires GW150914 H1 data in data/)")
        results["test5"] = None

    # Summary
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print()
    print("  Test 1 (Gap-filling): Do NL modes fill spectral gaps?")
    if results["test1"] is not None:
        mean_r = np.mean(results["test1"])
        print(f"    Mean gap-fill ratio: {mean_r:.3f} {'(YES)' if mean_r < 1.0 else '(NO)'}")
    print()
    print("  Test 2 (Fisher info): Does NL carry more info than overtone?")
    if results["test2"] is not None:
        r2 = results["test2"]
        det_A = r2["A: Linear only"]["det_F"]
        det_B = r2["B: Linear + NL(4,4)"]["det_F"]
        det_C = r2["C: Linear + overtone (2,2,1)"]["det_F"]
        print(f"    NL gain:  {det_B/det_A:.2f}x")
        print(f"    OT gain:  {det_C/det_A:.2f}x")
        print(f"    {'NL wins' if det_B > det_C else 'Overtone wins'}")
    print()
    print("  Test 3 (Entropy): Does NL subtraction reduce residual entropy?")
    if results["test3"] is not None:
        dH = results["test3"]["delta_H"]
        print(f"    Entropy reduction: {dH:.3f} bits")
        print(f"    NL matched-filter SNR: {results['test3']['snr']:.2f}")
    print()
    print("  Test 4 (Ratio universality): Are QNM ratios near simple fractions?")
    if results["test4"] is not None:
        devs = [v["fractional_deviation"] for v in results["test4"].values()]
        print(f"    Mean fractional deviation: {np.mean(devs):.4f}")
    print()
    print("  Test 5 (Start time): Does the excess behave like a ringdown?")
    if results["test5"] is not None:
        print(f"    Peak SNR: {results['test5']['peak_snr']:.2f}"
              f" at t = {results['test5']['peak_t']:.0f} M")
    print()
    print("  The honest summary: these tests establish STRUCTURE, not TRUTH.")
    print("  The nonlinear mode hypothesis makes specific, testable predictions")
    print("  about information content, spectral placement, and temporal profile.")
    print("  A 2-sigma excess that matches all five predictions is more")
    print("  interesting than one that matches none. But 2 sigma is 2 sigma.")
    print("  The universe does not owe us a detection at current sensitivity.")
    print()


if __name__ == "__main__":
    main()
