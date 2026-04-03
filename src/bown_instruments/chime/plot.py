"""Publication-quality plotting for chime channel quality maps.

Produces 4-panel diagnostic plots:
  1. Noise power spectrum (empirical scatter vs photon noise)
  2. Systematic excess (scatter / photon noise) with grade coloring
  3. Allan deviation ratio (correlated noise diagnostic)
  4. Rough transmission spectrum with trust regions

Uses a dark theme optimized for screen viewing and presentation.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bown_instruments.chime.channel_map import ChannelMap, BinResult
from bown_instruments.chime.diversity import DiversityResult


# Color scheme
GRADE_COLORS = {
    "A": "#34d399",
    "B": "#38bdf8",
    "C": "#fbbf24",
    "D": "#f87171",
}

BG = "#0a0e17"
PANEL_BG = "#111827"
SPINE_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
MUTED_COLOR = "#94a3b8"


def _style_axis(ax):
    """Apply dark theme to an axis."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def plot_channel_map(
    channel_map: ChannelMap,
    target: str,
    outpath: str,
    obs_label: str = "",
    ephemeris_ref: str = "",
):
    """Plot the 4-panel channel quality diagnostic.

    Parameters
    ----------
    channel_map : ChannelMap
    target : str
        Target name for title.
    outpath : str
        Output file path (PNG).
    obs_label : str
        Observation identifier for subtitle.
    ephemeris_ref : str
        Ephemeris reference string.
    """
    bins = channel_map.bins
    if not bins:
        return

    wl = np.array([b.wl_center for b in bins])
    scatter = np.array([b.scatter_ppm for b in bins])
    photon = np.array([b.photon_noise_ppm for b in bins])
    excess = np.array([b.systematic_excess for b in bins])
    depth = np.array([b.depth_ppm for b in bins])
    allan = np.array([b.allan_worst_ratio for b in bins])
    grades = [b.grade for b in bins]

    smry = channel_map.summary
    expected = smry.get("expected_depth_ppm", 0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={"hspace": 0.32})
    fig.patch.set_facecolor(BG)

    for ax in axes:
        _style_axis(ax)

    # --- Panel 1: Noise power spectrum ---
    ax = axes[0]
    ax.semilogy(
        wl,
        scatter,
        "o-",
        color="#f87171",
        markersize=4,
        linewidth=1.2,
        label=f"Empirical scatter (median {np.median(scatter):.0f} ppm)",
    )
    ax.semilogy(
        wl, photon, "s-", color="#34d399", markersize=3, linewidth=1, label="Photon noise limit"
    )
    if expected > 0:
        ax.axhline(
            expected,
            color="#fbbf24",
            linewidth=1,
            linestyle="--",
            alpha=0.6,
            label=f"Expected signal ({expected:.0f} ppm)",
        )
        signal_visible = scatter < expected * 2
        if np.any(signal_visible):
            ax.fill_between(
                wl, scatter, expected * 2, where=signal_visible, alpha=0.1, color="#fbbf24"
            )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Noise (ppm)", color=TEXT_COLOR)
    title = f"Channel Noise Map: {target}"
    if obs_label:
        title += f" — {obs_label}"
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # --- Panel 2: Systematic excess ---
    ax = axes[1]
    colors = [GRADE_COLORS[g] for g in excess_colors(excess)]
    bar_width = np.diff(wl, append=wl[-1] * 1.05) * 0.8
    ax.bar(wl, excess, width=bar_width, color=colors, alpha=0.7)
    ax.axhline(1, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5, label="Photon-limited")
    ax.axhline(
        5, color="#fbbf24", linewidth=0.8, linestyle="--", alpha=0.5, label="Systematic-dominated"
    )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Systematic Excess\n(scatter / photon noise)", color=TEXT_COLOR)
    ax.set_title(
        f"Channel Quality — "
        f"{smry.get('n_photon_limited', 0)}/{len(bins)} A-grade, "
        f"{smry.get('n_systematic_dominated', 0)}/{len(bins)} D-grade",
        color=TEXT_COLOR,
        fontsize=11,
    )
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # --- Panel 3: Allan deviation ratio ---
    ax = axes[2]
    allan_colors = ["#34d399" if a < 1.5 else "#fbbf24" if a < 3 else "#f87171" for a in allan]
    ax.bar(wl, allan, width=bar_width, color=allan_colors, alpha=0.7)
    ax.axhline(
        1.0,
        color="#34d399",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        label="White noise (averages down as √n)",
    )
    ax.axhline(
        2.0, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5, label="Correlated noise"
    )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Allan Ratio\n(actual / expected reduction)", color=TEXT_COLOR)
    ax.set_title(
        "Noise Correlation Diagnostic — Does Averaging Help?", color=TEXT_COLOR, fontsize=11
    )
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # --- Panel 4: Transmission spectrum ---
    ax = axes[3]
    n_in = smry.get("n_in_transit", 1)
    yerr = scatter / np.sqrt(max(n_in, 1))
    grade_c = [GRADE_COLORS[g] for g in grades]
    ax.errorbar(
        wl,
        depth,
        yerr=yerr,
        fmt="o",
        color="#38bdf8",
        markersize=4,
        elinewidth=1,
        capsize=2,
        alpha=0.7,
    )
    if expected > 0:
        ax.axhline(
            expected,
            color="#fbbf24",
            linewidth=1,
            linestyle="--",
            alpha=0.6,
            label=f"Expected depth ({expected:.0f} ppm)",
        )
    ax.axhline(0, color="#334155", linewidth=0.5)
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Transit Depth (ppm)", color=TEXT_COLOR)
    ax.set_title("Rough Transmission Spectrum", color=TEXT_COLOR, fontsize=11)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    if ephemeris_ref:
        fig.suptitle(f"chime — {ephemeris_ref}", color=MUTED_COLOR, fontsize=9, y=0.995)

    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_diversity(
    result: DiversityResult,
    target: str,
    outpath: str,
):
    """Plot 3-panel diversity combining diagnostic.

    Panels: transmission spectrum, systematic excess, noise power.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor(BG)

    for ax in axes:
        _style_axis(ax)

    # Panel 1: Transmission spectrum with grade coloring
    ax = axes[0]
    for sb in result.subbands:
        c = GRADE_COLORS[sb.grade]
        alpha = 0.3 if sb.grade == "D" else 0.8
        ax.errorbar(
            sb.wl_center,
            sb.depth_ppm,
            yerr=sb.depth_err_ppm,
            fmt="o",
            color=c,
            alpha=alpha,
            markersize=6,
            elinewidth=1.5,
            capsize=3,
        )

    ax.axhline(
        result.diversity_depth_ppm,
        color="#38bdf8",
        linewidth=1.5,
        linestyle="--",
        label=f"Diversity: {result.diversity_depth_ppm:.0f} ± {result.diversity_err_ppm:.0f} ppm",
    )
    ax.axhline(
        result.naive_depth_ppm,
        color=MUTED_COLOR,
        linewidth=1,
        linestyle=":",
        label=f"Naive: {result.naive_depth_ppm:.0f} ± {result.naive_err_ppm:.0f} ppm",
    )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Transit Depth (ppm)", color=TEXT_COLOR)
    ax.set_title(
        f"Diversity-Combined Transmission Spectrum: {target}", color=TEXT_COLOR, fontsize=13
    )
    ax.legend(
        fontsize=9, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # Panel 2: Systematic excess
    ax = axes[1]
    for sb in result.subbands:
        c = GRADE_COLORS[sb.grade]
        ax.bar(
            sb.wl_center,
            sb.systematic_excess,
            width=(sb.wl_range[1] - sb.wl_range[0]) * 0.8,
            color=c,
            alpha=0.7,
            edgecolor=c,
        )
    ax.axhline(2, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5, label="A/B boundary")
    ax.axhline(5, color="#fbbf24", linewidth=0.8, linestyle="--", alpha=0.5, label="B/C boundary")
    ax.axhline(10, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5, label="C/D boundary")
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Systematic Excess\n(scatter / photon noise)", color=TEXT_COLOR)
    ax.set_title("Sub-Band Quality Map", color=TEXT_COLOR, fontsize=13)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # Panel 3: Noise power
    ax = axes[2]
    ax.semilogy(
        result.wl_centers,
        result.noise_power_ppm,
        "o-",
        color="#f87171",
        alpha=0.7,
        markersize=5,
        label="Systematic noise (ppm)",
    )
    photon = np.array([sb.photon_noise_ppm for sb in result.subbands])
    ax.semilogy(
        result.wl_centers,
        photon,
        "s-",
        color="#34d399",
        alpha=0.7,
        markersize=4,
        label="Photon noise (ppm)",
    )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Noise Level (ppm)", color=TEXT_COLOR)
    ax.set_title(
        f"Noise Map — Improvement: {result.improvement_factor:.2f}× "
        f"(dropped {result.n_dropped}, deweighted {result.n_degraded})",
        color=TEXT_COLOR,
        fontsize=13,
    )
    ax.legend(
        fontsize=9, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_visit_comparison(
    visit_maps: list[dict],
    target: str,
    outpath: str,
):
    """Plot multi-visit channel quality comparison.

    Parameters
    ----------
    visit_maps : list of dicts, each with keys:
        label, wl_centers, scatter_ppm, systematic_excess, allan_ratios
    target : str
    outpath : str
    """
    n_visits = len(visit_maps)
    if n_visits < 2:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor(BG)

    for ax in axes:
        _style_axis(ax)

    visit_colors = ["#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#fb923c", "#e879f9"]

    # Panel 1: Scatter comparison
    ax = axes[0]
    for i, vm in enumerate(visit_maps):
        c = visit_colors[i % len(visit_colors)]
        ax.semilogy(
            vm["wl_centers"],
            vm["scatter_ppm"],
            "o-",
            color=c,
            markersize=3,
            linewidth=1,
            alpha=0.8,
            label=vm["label"],
        )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Scatter (ppm)", color=TEXT_COLOR)
    ax.set_title(f"Visit-to-Visit Scatter Comparison: {target}", color=TEXT_COLOR, fontsize=13)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # Panel 2: Systematic excess comparison
    ax = axes[1]
    for i, vm in enumerate(visit_maps):
        c = visit_colors[i % len(visit_colors)]
        ax.plot(
            vm["wl_centers"],
            vm["systematic_excess"],
            "o-",
            color=c,
            markersize=3,
            linewidth=1,
            alpha=0.8,
            label=vm["label"],
        )
    ax.axhline(2, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(5, color="#fbbf24", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Systematic Excess", color=TEXT_COLOR)
    ax.set_title("Systematic Excess Comparison", color=TEXT_COLOR, fontsize=11)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # Panel 3: Allan ratio comparison
    ax = axes[2]
    for i, vm in enumerate(visit_maps):
        c = visit_colors[i % len(visit_colors)]
        ax.plot(
            vm["wl_centers"],
            vm["allan_ratios"],
            "o-",
            color=c,
            markersize=3,
            linewidth=1,
            alpha=0.8,
            label=vm["label"],
        )
    ax.axhline(1.5, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(3.0, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Allan Ratio", color=TEXT_COLOR)
    ax.set_title("Allan Deviation Comparison", color=TEXT_COLOR, fontsize=11)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    fig.suptitle(
        f"chime — Multi-Visit Consistency: {target}", color=MUTED_COLOR, fontsize=10, y=0.995
    )

    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def excess_colors(excess: np.ndarray) -> list[str]:
    """Map systematic excess values to color keys."""
    return ["A" if e < 2 else "B" if e < 5 else "C" if e < 10 else "D" for e in excess]


def plot_segment_comparison(
    segment_maps: list[dict],
    target: str,
    outpath: str,
):
    """Plot per-wavelength systematic excess for all segments of one observation.

    Each segment is a separate line, color-coded. This makes quality variation
    between segments immediately visible.

    Parameters
    ----------
    segment_maps : list of dicts, each with keys:
        label (e.g. "seg001"), wl_centers, systematic_excess, allan_ratios,
        grades, scatter_ppm
    target : str
    outpath : str
    """
    if len(segment_maps) < 2:
        return

    n_segs = len(segment_maps)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor(BG)

    for ax in axes:
        _style_axis(ax)

    segment_colors = [
        "#38bdf8",
        "#f87171",
        "#34d399",
        "#fbbf24",
        "#a78bfa",
        "#fb923c",
        "#e879f9",
        "#22d3ee",
    ]

    # Panel 1: Systematic excess per wavelength
    ax = axes[0]
    for i, sm in enumerate(segment_maps):
        c = segment_colors[i % len(segment_colors)]
        ax.plot(
            sm["wl_centers"],
            sm["systematic_excess"],
            "o-",
            color=c,
            markersize=4,
            linewidth=1.5,
            alpha=0.9,
            label=f"{sm['label']} (excess={np.median(sm['systematic_excess']):.2f}x)",
        )
    ax.axhline(1, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5, label="Photon-limited")
    ax.axhline(2, color="#38bdf8", linewidth=0.8, linestyle="--", alpha=0.5, label="A/B boundary")
    ax.axhline(5, color="#fbbf24", linewidth=0.8, linestyle="--", alpha=0.5, label="B/C boundary")
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Systematic Excess\n(scatter / photon noise)", color=TEXT_COLOR)
    ax.set_title(f"Per-Segment Systematic Excess: {target}", color=TEXT_COLOR, fontsize=13)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # Panel 2: Allan ratio per wavelength
    ax = axes[1]
    for i, sm in enumerate(segment_maps):
        c = segment_colors[i % len(segment_colors)]
        ax.plot(
            sm["wl_centers"],
            sm["allan_ratios"],
            "o-",
            color=c,
            markersize=4,
            linewidth=1.5,
            alpha=0.9,
            label=sm["label"],
        )
    ax.axhline(1.0, color="#34d399", linewidth=0.8, linestyle="--", alpha=0.5, label="White noise")
    ax.axhline(
        1.5, color="#38bdf8", linewidth=0.8, linestyle="--", alpha=0.5, label="A-grade threshold"
    )
    ax.axhline(
        3.0, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.5, label="Correlated noise"
    )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Allan Ratio\n(actual / expected reduction)", color=TEXT_COLOR)
    ax.set_title("Per-Segment Allan Deviation Ratio", color=TEXT_COLOR, fontsize=13)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    # Panel 3: Scatter (noise power) per wavelength
    ax = axes[2]
    for i, sm in enumerate(segment_maps):
        c = segment_colors[i % len(segment_colors)]
        ax.semilogy(
            sm["wl_centers"],
            sm["scatter_ppm"],
            "o-",
            color=c,
            markersize=4,
            linewidth=1.5,
            alpha=0.9,
            label=sm["label"],
        )
    ax.set_xlabel("Wavelength (µm)", color=TEXT_COLOR)
    ax.set_ylabel("Empirical Scatter (ppm)", color=TEXT_COLOR)
    ax.set_title("Per-Segment Noise Level", color=TEXT_COLOR, fontsize=13)
    ax.legend(
        fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR, facecolor=PANEL_BG, edgecolor=SPINE_COLOR
    )

    fig.suptitle(
        f"chime — Segment Quality Comparison: {target}", color=MUTED_COLOR, fontsize=10, y=0.995
    )

    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
