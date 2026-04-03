"""
GRIM-S visualization: diagnostic plots for ringdown analysis.

Every plot answers a specific question that Bown would ask:
  1. "Can I see the signal?" — waveform and mode decomposition
  2. "Is the channel open?" — frequency separation vs spin
  3. "What does the measurement say?" — kappa posterior
  4. "Did the instrument check itself?" — self-test residual
  5. "Does stacking help?" — individual vs stacked posteriors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_mode_spectrum(catalog, spin, save_path=None):
    """Plot the QNM frequency spectrum: linear vs nonlinear modes.

    Answers: "Where are the modes, and can I tell them apart?"
    """
    modes = catalog.standard_ringdown_basis(spin, include_nonlinear=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for m in modes:
        color = "#d62728" if m.is_nonlinear else "#1f77b4"
        marker = "D" if m.is_nonlinear else "o"
        label_prefix = "NL " if m.is_nonlinear else ""
        label = f"{label_prefix}({m.l},{m.m},{m.n})"

        ax.plot(m.frequency, abs(m.damping_rate), marker=marker,
                color=color, markersize=12, label=label, zorder=5)

    ax.set_xlabel("Oscillation frequency (M$\\omega$)", fontsize=13)
    ax.set_ylabel("|Damping rate| (M$\\gamma$)", fontsize=13)
    ax.set_title(f"Kerr QNM Spectrum — spin a = {spin}", fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Annotate the key separation
    lin_440 = [m for m in modes if m.l == 4 and m.m == 4 and not m.is_nonlinear][0]
    nl_44 = [m for m in modes if m.is_nonlinear][0]
    ax.annotate("", xy=(lin_440.frequency, abs(lin_440.damping_rate)),
                xytext=(nl_44.frequency, abs(nl_44.damping_rate)),
                arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=2))
    mid_f = 0.5 * (lin_440.frequency + nl_44.frequency)
    mid_g = 0.5 * (abs(lin_440.damping_rate) + abs(nl_44.damping_rate))
    sep = abs(lin_440.frequency - nl_44.frequency) / mid_f * 100
    ax.text(mid_f, mid_g * 1.15, f"{sep:.1f}% separation",
            ha="center", fontsize=11, color="#2ca02c", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_channel_status(survey_results, save_path=None):
    """Plot frequency separation vs spin — the 'channel status' diagram.

    Answers: "At what spins can we distinguish the nonlinear mode?"
    """
    spins = [r["spin"] for r in survey_results]
    seps = [r["fractional_separation"] * 100 for r in survey_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(spins, seps, alpha=0.3, color="#1f77b4")
    ax.plot(spins, seps, "o-", color="#1f77b4", lw=2, markersize=6)

    # Channel status thresholds
    ax.axhline(y=5.0, color="#2ca02c", ls="--", alpha=0.7, label="OPEN threshold (5%)")
    ax.axhline(y=1.0, color="#d62728", ls="--", alpha=0.7, label="CLOSED threshold (1%)")

    ax.set_xlabel("Remnant spin $a$", fontsize=13)
    ax.set_ylabel("Frequency separation (%)", fontsize=13)
    ax.set_title("Linear/Nonlinear Mode Separation — Channel Status", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_waveform_decomposition(template, t_dimless, save_path=None):
    """Plot the ringdown waveform with individual mode contributions.

    Answers: "Can I see each mode in the signal?"
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    mask = (t_dimless >= 0) & (t_dimless <= 80)
    t_plot = t_dimless[mask]

    # Total waveform
    h_total = template.waveform(t_dimless)[mask]
    axes[0].plot(t_plot, h_total, "k-", lw=1.5, label="Total", alpha=0.8)

    # Individual modes
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
    for i, mode in enumerate(template.modes):
        h_mode = np.zeros_like(t_dimless)
        m = t_dimless >= 0
        h_mode[m] = mode.amplitude * np.exp(mode.qnm.omega.imag * t_dimless[m]) * \
            np.cos(mode.qnm.omega.real * t_dimless[m] + mode.phase)

        label_prefix = "NL " if mode.qnm.is_nonlinear else ""
        label = f"{label_prefix}({mode.qnm.l},{mode.qnm.m},{mode.qnm.n})"
        axes[0].plot(t_plot, h_mode[mask], color=colors[i % len(colors)],
                     ls="--", lw=1.2, label=label, alpha=0.7)

    axes[0].set_ylabel("Strain $h(t)$", fontsize=13)
    axes[0].set_title("Ringdown Waveform Decomposition", fontsize=14)
    axes[0].legend(fontsize=10, ncol=3, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Envelope
    for i, mode in enumerate(template.modes):
        envelope = mode.amplitude * np.exp(mode.qnm.omega.imag * t_plot)
        label_prefix = "NL " if mode.qnm.is_nonlinear else ""
        label = f"{label_prefix}({mode.qnm.l},{mode.qnm.m},{mode.qnm.n}) envelope"
        axes[1].semilogy(t_plot, np.abs(envelope), color=colors[i % len(colors)],
                         lw=2, label=label)

    axes[1].set_xlabel("Time ($t/M$)", fontsize=13)
    axes[1].set_ylabel("Mode envelope", fontsize=13)
    axes[1].legend(fontsize=9, ncol=3, loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(1e-6, None)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_kappa_posterior(result, kappa_true=None, save_path=None):
    """Plot the posterior distribution of kappa.

    Answers: "What does the measurement say about nonlinear coupling?"
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(result.kappa_grid, result.posterior, alpha=0.3, color="#1f77b4")
    ax.plot(result.kappa_grid, result.posterior, color="#1f77b4", lw=2)

    # MAP
    ax.axvline(result.kappa_map, color="#d62728", ls="-", lw=2,
               label=f"MAP = {result.kappa_map:.3f}")

    # 90% CI
    ax.axvline(result.kappa_lower_90, color="#ff7f0e", ls="--", lw=1.5, alpha=0.7)
    ax.axvline(result.kappa_upper_90, color="#ff7f0e", ls="--", lw=1.5, alpha=0.7,
               label=f"90% CI: [{result.kappa_lower_90:.2f}, {result.kappa_upper_90:.2f}]")

    # True value
    if kappa_true is not None:
        ax.axvline(kappa_true, color="#2ca02c", ls=":", lw=2.5,
                   label=f"Injected = {kappa_true:.3f}")

    ax.set_xlabel("Nonlinear coupling coefficient $\\kappa$", fontsize=13)
    ax.set_ylabel("Posterior probability density", fontsize=13)
    ax.set_title(f"{result.event_name} — "
                 f"{result.detection_sigma:.1f}$\\sigma$ detection",
                 fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_stacked_posterior(stacked, individual_posteriors=None,
                           kappa_true=None, save_path=None):
    """Plot stacked posterior with individual contributions.

    Answers: "Does accumulating evidence across events help?"
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual posteriors (faint)
    if individual_posteriors:
        for p in individual_posteriors:
            ax.plot(p.kappa_grid, p.posterior, alpha=0.3, lw=1,
                    label=p.event_name)

    # Stacked posterior (bold)
    ax.fill_between(stacked.kappa_grid, stacked.stacked_posterior,
                    alpha=0.4, color="#d62728")
    ax.plot(stacked.kappa_grid, stacked.stacked_posterior,
            color="#d62728", lw=3,
            label=f"Stacked ({stacked.n_events} events)")

    # MAP and CI
    ax.axvline(stacked.kappa_map, color="#d62728", ls="-", lw=2)
    ax.axvline(stacked.kappa_lower_90, color="#d62728", ls="--", lw=1.5, alpha=0.5)
    ax.axvline(stacked.kappa_upper_90, color="#d62728", ls="--", lw=1.5, alpha=0.5)

    if kappa_true is not None:
        ax.axvline(kappa_true, color="black", ls=":", lw=2.5,
                   label=f"Injected = {kappa_true}")

    ax.set_xlabel("Nonlinear coupling coefficient $\\kappa$", fontsize=13)
    ax.set_ylabel("Posterior probability density", fontsize=13)
    ax.set_title(f"Stacked Posterior — {stacked.detection_sigma:.1f}$\\sigma$ | "
                 f"ln B = {stacked.log_bayes_factor:.1f}", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_self_test_diagnostic(self_test_result, save_path=None):
    """Plot self-test results: power accounting pie chart + diagnosis.

    Answers: "Did the instrument check itself, and did it pass?"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Power accounting
    recon = self_test_result.reconstructed_power
    resid = self_test_result.residual_power
    total = self_test_result.total_signal_power

    if total > 0:
        sizes = [recon / total * 100, resid / total * 100]
        colors = ["#2ca02c" if self_test_result.passed else "#ff7f0e",
                  "#d62728" if not self_test_result.passed else "#95a5a6"]
        labels = [f"Reconstructed\n{sizes[0]:.1f}%",
                  f"Residual\n{sizes[1]:.1f}%"]
        ax1.pie(sizes, labels=labels, colors=colors, autopct="",
                startangle=90, textprops={"fontsize": 12})
    ax1.set_title("Power Accounting", fontsize=14)

    # Mode amplitudes bar chart
    amps = self_test_result.mode_amplitudes
    if amps:
        labels_bar = list(amps.keys())
        values = [amps[k]["amplitude"] for k in labels_bar]
        bars = ax2.bar(range(len(labels_bar)), values, color="#1f77b4", alpha=0.8)
        ax2.set_xticks(range(len(labels_bar)))
        ax2.set_xticklabels(labels_bar, rotation=45, ha="right", fontsize=10)
        ax2.set_ylabel("Extracted amplitude", fontsize=12)
    ax2.set_title("Mode Amplitudes", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add diagnosis text
    status = "PASS" if self_test_result.passed else "FAIL"
    color = "#2ca02c" if self_test_result.passed else "#d62728"
    fig.suptitle(f"Self-Test: {status}", fontsize=16, color=color,
                 fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def generate_all_diagnostics(output_dir="plots/"):
    """Generate the full diagnostic suite.

    This is the "instrument panel" — every plot Bown would want
    to see before trusting a measurement.
    """
    import os
    from .qnm_modes import KerrQNMCatalog, survey_spin_dependence
    from .ringdown_templates import RingdownTemplateBuilder
    from .gwtc_pipeline import (
        generate_synthetic_ringdown, GWTC3_RINGDOWN_CANDIDATES,
    )
    from .self_test import run_self_test
    from .bayesian_analysis import estimate_kappa_posterior

    os.makedirs(output_dir, exist_ok=True)
    catalog = KerrQNMCatalog()
    builder = RingdownTemplateBuilder()

    print("Generating GRIM-S diagnostic suite...")

    # 1. Mode spectrum
    print("  [1/5] Mode spectrum...")
    plot_mode_spectrum(catalog, 0.69,
                       save_path=os.path.join(output_dir, "mode_spectrum.png"))

    # 2. Channel status
    print("  [2/5] Channel status...")
    survey = survey_spin_dependence(np.linspace(0.0, 0.98, 50))
    plot_channel_status(survey,
                        save_path=os.path.join(output_dir, "channel_status.png"))

    # 3. Waveform decomposition
    print("  [3/5] Waveform decomposition...")
    template = builder.build_nonlinear_template(
        spin=0.69, A_220=1.0, A_330=0.3, A_440_linear=0.15, kappa=1.5,
    )
    t = np.linspace(-10, 80, 4000)
    plot_waveform_decomposition(template, t,
                                save_path=os.path.join(output_dir, "waveform.png"))

    # 4. Injection recovery + posterior
    print("  [4/5] Injection recovery...")
    np.random.seed(42)
    event = GWTC3_RINGDOWN_CANDIDATES[0]
    kappa_true = 1.5
    segment = generate_synthetic_ringdown(event, kappa=kappa_true)

    m_sun_s = 4.925491025543576e-06
    m_s = event["remnant_mass_msun"] * m_sun_s
    t_dim = (segment.time - segment.t_ringdown_start) / m_s
    dist_m = event["luminosity_distance_mpc"] * 3.0857e22
    mass_m = event["remnant_mass_msun"] * 1.989e30 * 6.674e-11 / (3e8)**2
    strain_dim = segment.strain * dist_m / mass_m

    noise_sigma = np.max(np.abs(strain_dim[t_dim >= 0])) * 0.3
    noisy = strain_dim + np.random.normal(0, noise_sigma, len(strain_dim))

    q = event["mass_ratio"]
    result = estimate_kappa_posterior(
        noisy, t_dim, spin=event["remnant_spin"],
        A_220=0.4 * q, noise_variance=noise_sigma**2,
        event_name="GW150914 (synthetic)",
        A_330=0.1 * q * (1 - q), A_440_linear=0.05 * q,
    )
    plot_kappa_posterior(result, kappa_true=kappa_true,
                         save_path=os.path.join(output_dir, "kappa_posterior.png"))

    # 5. Self-test
    print("  [5/5] Self-test...")
    basis = catalog.standard_ringdown_basis(event["remnant_spin"],
                                            include_nonlinear=True)
    st = run_self_test(
        strain_dim, t_dim,
        mode_frequencies=[m.omega for m in basis],
        mode_labels=[f"({'NL' if m.is_nonlinear else ''}{m.l},{m.m},{m.n})"
                     for m in basis],
    )
    plot_self_test_diagnostic(st,
                              save_path=os.path.join(output_dir, "self_test.png"))

    print(f"\nAll diagnostics saved to {output_dir}")
    plt.close("all")
