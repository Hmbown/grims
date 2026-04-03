#!/usr/bin/env python3
"""
Generate two simple README graphics for GRIM-S:
  1. the instrument/system flow
  2. the current result and claim boundary
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def theme(mode: str) -> dict:
    if mode == "dark":
        return {
            "bg": "#0d1117",
            "panel": "#161b22",
            "panel_alt": "#1f2630",
            "line": "#30363d",
            "text": "#e6edf3",
            "muted": "#9aa4b2",
            "blue": "#58a6ff",
            "orange": "#f59e0b",
            "green": "#56d364",
            "red": "#fb7185",
            "soft_blue": "#172554",
            "soft_orange": "#3b2610",
            "soft_green": "#0f2d1c",
            "soft_red": "#3b1620",
        }
    return {
        "bg": "#ffffff",
        "panel": "#f8fafc",
        "panel_alt": "#eef2f7",
        "line": "#d1d5db",
        "text": "#111827",
        "muted": "#4b5563",
        "blue": "#2563eb",
        "orange": "#d97706",
        "green": "#16a34a",
        "red": "#dc2626",
        "soft_blue": "#dbeafe",
        "soft_orange": "#ffedd5",
        "soft_green": "#dcfce7",
        "soft_red": "#fee2e2",
    }


def box(ax, x, y, w, h, fc, ec, radius=0.025, lw=1.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
    )


def arrow(ax, x1, y1, x2, y2, color):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color=color,
        )
    )


def txt(ax, x, y, s, *, size=12, weight="normal", color="#000", ha="left", va="top", linespacing=1.3):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=linespacing)


def draw_instrument(mode: str, path: str) -> None:
    c = theme(mode)
    fig = plt.figure(figsize=(14, 6.5), facecolor=c["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    txt(ax, 0.05, 0.93, "How GRIM-S Works", size=26, weight="bold", color=c["text"])
    txt(
        ax,
        0.05,
        0.875,
        "Condition the data. Lock the daughter template. Stack weak evidence across events.",
        size=13.5,
        color=c["muted"],
    )

    # Top row: what goes in
    box(ax, 0.06, 0.63, 0.24, 0.14, c["soft_blue"], c["line"])
    txt(ax, 0.085, 0.73, "Input", size=17, weight="bold", color=c["blue"])
    txt(ax, 0.085, 0.68, "GWTC event metadata\nGWOSC strain files", size=13, color=c["text"])

    box(ax, 0.38, 0.63, 0.24, 0.14, c["panel_alt"], c["line"])
    txt(ax, 0.405, 0.73, "Conditioning", size=17, weight="bold", color=c["text"])
    txt(ax, 0.405, 0.68, "estimate ASD\nwhiten\nband-limit to the QNM band", size=13, color=c["muted"])

    box(ax, 0.70, 0.63, 0.24, 0.14, c["soft_orange"], c["line"])
    txt(ax, 0.725, 0.73, "Per-event search", size=17, weight="bold", color=c["orange"])
    txt(ax, 0.725, 0.68, "fit the parent mode\nbuild the locked daughter template\nestimate $\\kappa$", size=13, color=c["text"])

    arrow(ax, 0.305, 0.70, 0.375, 0.70, c["line"])
    arrow(ax, 0.625, 0.70, 0.695, 0.70, c["line"])

    # Bottom row: why it works
    box(ax, 0.06, 0.25, 0.40, 0.24, c["panel"], c["line"])
    txt(ax, 0.085, 0.45, "The key simplification", size=18, weight="bold", color=c["text"])
    txt(
        ax,
        0.085,
        0.39,
        "The nonlinear daughter mode is not searched as a free waveform.\n"
        "It is locked to the parent mode:\n\n"
        "$\\omega_{NL}=2\\omega_{220}$\n"
        "$\\gamma_{NL}=2\\gamma_{220}$\n"
        "$A_{NL}=\\kappa A_{220}^2$",
        size=13,
        color=c["muted"],
    )

    box(ax, 0.54, 0.25, 0.40, 0.24, c["soft_green"], c["line"])
    txt(ax, 0.565, 0.45, "Why stack many events", size=18, weight="bold", color=c["green"])
    txt(
        ax,
        0.565,
        0.39,
        "No single merger is expected to show a loud nonlinear signal.\n"
        "GRIM-S treats each event as a weak measurement and combines\n"
        "them with inverse-variance weighting to increase sensitivity.",
        size=13,
        color=c["text"],
    )

    arrow(ax, 0.50, 0.37, 0.54, 0.37, c["line"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=c["bg"], edgecolor="none")
    plt.close(fig)


def draw_result(mode: str, path: str) -> None:
    c = theme(mode)
    fig = plt.figure(figsize=(14, 6.5), facecolor=c["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    txt(ax, 0.05, 0.93, "What GRIM-S Found", size=26, weight="bold", color=c["text"])
    txt(
        ax,
        0.05,
        0.875,
        "The exploratory stack is interesting. The audited interpretation is still cautious.",
        size=13.5,
        color=c["muted"],
    )

    box(ax, 0.06, 0.56, 0.40, 0.20, c["soft_blue"], c["line"])
    txt(ax, 0.09, 0.71, "Fast phase-locked stack", size=19, weight="bold", color=c["blue"])
    txt(
        ax,
        0.09,
        0.64,
        "23 events from GWTC-1 through GWTC-3\n"
        "$\\kappa = 0.175 \\pm 0.073$\n"
        "Combined SNR = 4.2",
        size=14,
        color=c["text"],
    )

    box(ax, 0.54, 0.56, 0.40, 0.20, c["soft_orange"], c["line"])
    txt(ax, 0.57, 0.71, "Audited profile likelihood", size=19, weight="bold", color=c["orange"])
    txt(
        ax,
        0.57,
        0.64,
        "5-event audited subset\n"
        "MAP $\\kappa = 0.60$\n"
        "90% CI $[0.16, 4.63]$",
        size=14,
        color=c["text"],
    )

    box(ax, 0.06, 0.20, 0.88, 0.25, c["panel"], c["line"])
    txt(ax, 0.09, 0.40, "Honest reading", size=20, weight="bold", color=c["red"])
    txt(
        ax,
        0.09,
        0.33,
        "GRIM-S found a suggestive pattern at the expected nonlinear frequency.\n"
        "But after auditing the dominant systematics, the current public data are still too weak to support\n"
        "a clean detection claim, a sharp GR test, or a publishable astrophysical measurement.",
        size=14,
        color=c["text"],
    )

    box(ax, 0.06, 0.07, 0.88, 0.08, c["panel_alt"], c["line"])
    txt(
        ax,
        0.08,
        0.11,
        "Best framing: an open personal research instrument with interesting exploratory results, not a settled scientific claim.",
        size=12.5,
        color=c["muted"],
        va="center",
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=c["bg"], edgecolor="none")
    plt.close(fig)


def main() -> None:
    draw_instrument("dark", "plots/grims_instrument.png")
    draw_instrument("light", "plots/grims_instrument_light.png")
    draw_result("dark", "plots/grims_result.png")
    draw_result("light", "plots/grims_result_light.png")
    print("Saved README graphics")


if __name__ == "__main__":
    main()
