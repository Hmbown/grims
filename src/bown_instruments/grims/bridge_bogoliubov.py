"""
Bridge between GRIM-S (astrophysical ringdown) and hawking_py (acoustic analog).

Two instruments, two channels, same underlying physics:
nonlinear mode coupling across a horizon.

This module does NOT modify either codebase. It imports from both
and connects their analysis outputs for cross-validation.

The bridge answers three questions:
  1. Do both systems produce intermodulation products at the predicted
     frequencies? (frequency consistency check)
  2. Does the coupling strength scale the same way? (amplitude consistency)
  3. Can GRIM-S's self-test detect when the acoustic analog signal
     departs from the QNM template? (cross-channel self-test)

Bown's diversity receiver principle: if two independent channels
agree on the measurement, trust the result. If they disagree,
you've found something interesting.

Requirements:
  - hawking_py must be importable (add /Volumes/VIXinSSD/bogoliubov to sys.path)
  - grims must be importable (add /Volumes/VIXinSSD/drbown/bown-ringdown to sys.path)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Path management — import from both repos without modifying either
_BOGOLIUBOV_ROOT = Path("/Volumes/VIXinSSD/bogoliubov")
_GRIMS_ROOT = Path("/Volumes/VIXinSSD/drbown/bown-ringdown")

def _ensure_imports():
    """Add both repos to sys.path if not already present."""
    for p in [str(_BOGOLIUBOV_ROOT), str(_GRIMS_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
#  Data structures for cross-channel comparison
# --------------------------------------------------------------------------- #

@dataclass
class ModeCouplingSignature:
    """A frequency-domain signature of mode coupling from either channel."""
    source: str  # "grims" or "hawking_py"
    input_frequency: float  # fundamental / input frequency
    coupled_frequency: float  # nonlinear / partner frequency
    frequency_ratio: float  # coupled / input
    coupling_strength: float  # amplitude ratio or power ratio
    damping_or_width: float  # damping rate (QNM) or spectral width (acoustic)
    metadata: dict = field(default_factory=dict)


@dataclass
class CrossChannelResult:
    """Result of comparing GRIM-S and hawking_py mode coupling signatures."""
    grims_signature: ModeCouplingSignature
    hawking_signature: ModeCouplingSignature
    frequency_ratio_agreement: float  # |ratio_grims - ratio_hawking| / mean
    coupling_strength_ratio: float  # grims_coupling / hawking_coupling
    consistent: bool  # do the channels agree?
    diagnosis: str


# --------------------------------------------------------------------------- #
#  Extract mode coupling signatures from each system
# --------------------------------------------------------------------------- #

def extract_grims_signature(spin: float = 0.69,
                            kappa_nl: float = 1.0) -> ModeCouplingSignature:
    """Extract the nonlinear mode coupling signature from GRIM-S.

    Parameters
    ----------
    spin : remnant black hole spin
    kappa_nl : nonlinear coupling coefficient

    Returns the frequency ratio and coupling strength of the
    dominant nonlinear mode relative to the fundamental.
    """
    _ensure_imports()
    from bown_instruments.grims.qnm_modes import KerrQNMCatalog

    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)

    # The nonlinear mode frequency is exactly 2x the fundamental
    freq_ratio = mode_nl.frequency / mode_220.frequency

    # Coupling strength: A_NL / A_220 = kappa * A_220
    # For unit-amplitude fundamental, coupling_strength = kappa
    coupling_strength = kappa_nl

    return ModeCouplingSignature(
        source="grims",
        input_frequency=mode_220.frequency,
        coupled_frequency=mode_nl.frequency,
        frequency_ratio=freq_ratio,
        coupling_strength=coupling_strength,
        damping_or_width=abs(mode_nl.damping_rate),
        metadata={
            "spin": spin,
            "kappa_nl": kappa_nl,
            "Q_fundamental": mode_220.quality_factor,
            "Q_nonlinear": mode_nl.quality_factor,
            "freq_separation_from_linear_440":
                catalog.frequency_separation_ratio(spin)["fractional_separation"],
        },
    )


def extract_hawking_signature(kappa_surface: float = 2.0,
                              omega_drive: float = 1.0,
                              omega_input: float = 0.2,
                              amplitude: float = 2.0,
                              run_simulation: bool = True,
                              verbose: bool = False) -> ModeCouplingSignature:
    """Extract the mode coupling signature from hawking_py.

    The acoustic analog produces mode coupling when the horizon
    oscillates: input at omega_in scatters into partner at
    |omega_in - omega_drive| and blue-shifted at omega_in + omega_drive.

    Parameters
    ----------
    kappa_surface : surface gravity (controls Hawking temperature)
    omega_drive : horizon oscillation frequency
    omega_input : input wave frequency
    amplitude : horizon oscillation amplitude
    run_simulation : if True, run the actual simulation; if False,
                     return theoretical prediction only
    verbose : print progress
    """
    _ensure_imports()

    if not run_simulation:
        # Theoretical prediction without running the sim
        omega_partner = abs(omega_input - omega_drive)
        T_H = kappa_surface / (2 * np.pi)
        # Thermal prediction: power ratio ~ exp(-omega_partner / T_H)
        theoretical_ratio = np.exp(-omega_partner / T_H) if T_H > 0 else 0.0

        return ModeCouplingSignature(
            source="hawking_py (theoretical)",
            input_frequency=omega_input,
            coupled_frequency=omega_partner,
            frequency_ratio=omega_partner / omega_input if omega_input > 0 else 0,
            coupling_strength=theoretical_ratio,
            damping_or_width=0.0,  # not computed
            metadata={
                "kappa_surface": kappa_surface,
                "omega_drive": omega_drive,
                "T_H_theoretical": T_H,
                "prediction_type": "thermal_boltzmann",
            },
        )

    # Run the actual simulation using the known-good config from
    # bogoliubov/scripts/dynamical_horizon_experiment.py
    from hawking_py import (
        AcousticBlackHole, ScatteringExperiment, ExperimentConfig,
        SolverConfig, GridConfig, TimeConfig,
    )
    from hawking_py import compute_bogoliubov_coefficients

    bh = AcousticBlackHole(
        profile="tanh", v_min=0.0, v_max=-1.5,
        kappa=kappa_surface,
    )
    bh.set_horizon_oscillation(frequency=omega_drive, amplitude=amplitude)

    solver_config = SolverConfig(
        grid=GridConfig(x_min=-100.0, x_max=100.0, nx=4000),
        time=TimeConfig(t_total=200.0, cfl=0.4),
        dissipation=0.002,
    )
    exp_config = ExperimentConfig(
        solver_config=solver_config,
        monitor_position=20.0,
        analysis_start_time=50.0,
        nperseg=1024,
        source_position=40.0,
    )

    experiment = ScatteringExperiment(black_hole=bh, config=exp_config)
    spectrum = experiment.inject_mode(
        omega_in=omega_input, source_type="pulse", verbose=verbose,
    )

    # Extract coupling signature
    omega_partner = spectrum.omega_partner if spectrum.omega_partner else 0.0
    power_ratio = spectrum.power_ratio if spectrum.power_ratio else 0.0

    # Bogoliubov coefficients
    try:
        coeffs = compute_bogoliubov_coefficients(spectrum)
        beta_sq = coeffs.occupation_number
        entropy = coeffs.entropy
    except Exception:
        beta_sq = 0.0
        entropy = 0.0

    return ModeCouplingSignature(
        source="hawking_py (simulated)",
        input_frequency=omega_input,
        coupled_frequency=omega_partner,
        frequency_ratio=omega_partner / omega_input if omega_input > 0 else 0,
        coupling_strength=power_ratio,
        damping_or_width=0.0,  # could extract from spectral peak width
        metadata={
            "kappa_surface": kappa_surface,
            "omega_drive": omega_drive,
            "T_H_theoretical": kappa_surface / (2 * np.pi),
            "T_H_measured": spectrum.calculate_temperature() if spectrum.power_ratio else None,
            "beta_squared": beta_sq,
            "entropy": entropy,
            "power_primary": spectrum.power_primary,
            "power_partner": spectrum.power_partner,
        },
    )


# --------------------------------------------------------------------------- #
#  Cross-channel comparison
# --------------------------------------------------------------------------- #

def compare_channels(grims_sig: ModeCouplingSignature,
                     hawking_sig: ModeCouplingSignature) -> CrossChannelResult:
    """Compare mode coupling signatures from both channels.

    The comparison is structural, not numerical: we check whether
    the *pattern* of mode coupling is consistent across channels,
    not whether the absolute numbers match (they shouldn't — the
    systems have different scales).

    What should agree:
      - Frequency ratio pattern (both produce coupling at ~2x fundamental)
      - Coupling strength scaling (both should scale with drive amplitude)

    What will differ:
      - Absolute frequencies (gravitational vs. acoustic)
      - Absolute coupling strengths (different physics)
      - Damping rates (QNM exponential vs. acoustic spectral width)
    """
    # Frequency ratio agreement
    mean_ratio = 0.5 * (grims_sig.frequency_ratio + hawking_sig.frequency_ratio)
    if mean_ratio > 0:
        freq_agreement = abs(grims_sig.frequency_ratio - hawking_sig.frequency_ratio) / mean_ratio
    else:
        freq_agreement = float('inf')

    # Coupling strength ratio (informational, not a consistency check)
    if hawking_sig.coupling_strength > 0:
        strength_ratio = grims_sig.coupling_strength / hawking_sig.coupling_strength
    else:
        strength_ratio = float('inf')

    # Consistency: structural agreement within 20%
    consistent = freq_agreement < 0.20

    if consistent:
        diagnosis = (
            f"CONSISTENT: Both channels show mode coupling with frequency "
            f"ratio agreement within {freq_agreement:.1%}. "
            f"GRIM-S: {grims_sig.frequency_ratio:.3f}, "
            f"hawking_py: {hawking_sig.frequency_ratio:.3f}. "
            f"Bown's diversity check: two independent instruments agree."
        )
    else:
        diagnosis = (
            f"DIVERGENT: Frequency ratio disagreement of {freq_agreement:.1%}. "
            f"GRIM-S: {grims_sig.frequency_ratio:.3f}, "
            f"hawking_py: {hawking_sig.frequency_ratio:.3f}. "
            f"This may indicate different coupling mechanisms or a "
            f"parameter mapping error. Investigate before trusting either."
        )

    return CrossChannelResult(
        grims_signature=grims_sig,
        hawking_signature=hawking_sig,
        frequency_ratio_agreement=freq_agreement,
        coupling_strength_ratio=strength_ratio,
        consistent=consistent,
        diagnosis=diagnosis,
    )


# --------------------------------------------------------------------------- #
#  GRIM-S self-test applied to acoustic analog data
# --------------------------------------------------------------------------- #

def apply_grims_selftest_to_acoustic(signal: np.ndarray,
                                     dt: float,
                                     spin_equivalent: float = 0.69,
                                     mass_equivalent: float = 1.0) -> dict:
    """Apply the GRIM-S self-test to an acoustic analog signal.

    This asks: does the acoustic signal decompose cleanly into
    QNM-like modes? If yes, the mode structure is universal
    across channels. If no, the acoustic system has physics
    that the QNM template doesn't capture.

    Parameters
    ----------
    signal : time-domain signal from hawking_py experiment
    dt : time step (seconds)
    spin_equivalent : map acoustic kappa to an "equivalent" BH spin
    mass_equivalent : mass scale for time conversion (set to 1.0 for
                      dimensionless comparison)

    Returns dict with self-test results and interpretation.
    """
    _ensure_imports()
    from bown_instruments.grims.qnm_modes import KerrQNMCatalog
    from bown_instruments.grims.self_test import run_self_test

    catalog = KerrQNMCatalog()
    basis = catalog.standard_ringdown_basis(spin_equivalent,
                                            include_nonlinear=True)

    # Convert acoustic time to dimensionless units
    t_dimless = np.arange(len(signal)) * dt / mass_equivalent

    # Run self-test with QNM mode frequencies
    mode_freqs = [m.omega for m in basis]
    mode_labels = [
        f"({'NL' if m.is_nonlinear else ''}{m.l},{m.m},{m.n})"
        for m in basis
    ]

    result = run_self_test(
        signal, t_dimless,
        mode_frequencies=mode_freqs,
        mode_labels=mode_labels,
    )

    interpretation = (
        "The acoustic signal was projected onto the Kerr QNM basis. "
        f"Residual fraction: {result.residual_fraction:.1%}. "
    )

    if result.residual_fraction < 0.05:
        interpretation += (
            "The QNM basis captures most of the signal power. "
            "This suggests the mode coupling structure is similar "
            "across gravitational and acoustic horizons."
        )
    elif result.residual_fraction < 0.30:
        interpretation += (
            "Significant power outside the QNM basis. "
            "The acoustic system has additional mode structure "
            "(e.g., continuous spectrum, dispersive effects) "
            "that the Kerr QNM template does not capture."
        )
    else:
        interpretation += (
            "The QNM basis is a poor fit for the acoustic signal. "
            "This is expected: the acoustic system is not Kerr, and "
            "the mode structure diverges from gravitational QNMs. "
            "The comparison is informative precisely because it fails."
        )

    return {
        "self_test": result,
        "interpretation": interpretation,
        "spin_equivalent": spin_equivalent,
        "n_modes_tested": len(basis),
        "residual_fraction": result.residual_fraction,
    }


# --------------------------------------------------------------------------- #
#  Full cross-validation workflow
# --------------------------------------------------------------------------- #

def run_cross_validation(spin: float = 0.69,
                         kappa_nl: float = 1.0,
                         kappa_surface: float = 2.0,
                         omega_drive: float = 1.0,
                         omega_input: float = 0.3,
                         run_acoustic_sim: bool = True,
                         verbose: bool = False) -> dict:
    """Run the full cross-channel validation.

    Steps:
      1. Extract GRIM-S signature (instantaneous — just QNM math)
      2. Extract hawking_py signature (requires simulation if run_acoustic_sim)
      3. Compare channels
      4. Report

    Parameters
    ----------
    spin : Kerr BH spin for GRIM-S
    kappa_nl : nonlinear coupling coefficient for GRIM-S
    kappa_surface : surface gravity for acoustic BH
    omega_drive : acoustic horizon drive frequency
    omega_input : acoustic input wave frequency
    run_acoustic_sim : whether to actually run the hawking_py simulation
    verbose : print progress
    """
    if verbose:
        print("Cross-Channel Validation: GRIM-S x hawking_py")
        print("=" * 55)

    # Step 1: GRIM-S
    if verbose:
        print("\n[1] Extracting GRIM-S signature...")
    grims_sig = extract_grims_signature(spin=spin, kappa_nl=kappa_nl)
    if verbose:
        print(f"    Fundamental: {grims_sig.input_frequency:.4f}")
        print(f"    Nonlinear:   {grims_sig.coupled_frequency:.4f}")
        print(f"    Ratio:       {grims_sig.frequency_ratio:.4f}")

    # Step 2: hawking_py
    if verbose:
        print(f"\n[2] Extracting hawking_py signature "
              f"({'simulated' if run_acoustic_sim else 'theoretical'})...")
    hawking_sig = extract_hawking_signature(
        kappa_surface=kappa_surface,
        omega_drive=omega_drive,
        omega_input=omega_input,
        run_simulation=run_acoustic_sim,
        verbose=verbose,
    )
    if verbose:
        print(f"    Input:    {hawking_sig.input_frequency:.4f}")
        print(f"    Partner:  {hawking_sig.coupled_frequency:.4f}")
        print(f"    Ratio:    {hawking_sig.frequency_ratio:.4f}")
        print(f"    Coupling: {hawking_sig.coupling_strength:.2e}")

    # Step 3: Compare
    if verbose:
        print("\n[3] Comparing channels...")
    comparison = compare_channels(grims_sig, hawking_sig)
    if verbose:
        print(f"    {comparison.diagnosis}")

    return {
        "grims": grims_sig,
        "hawking": hawking_sig,
        "comparison": comparison,
    }
