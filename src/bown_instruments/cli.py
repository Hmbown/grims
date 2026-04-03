"""Unified CLI for bown-instruments.

Usage:
    bown chime WASP-39              # Run CHIME channel quality analysis
    bown chime --targets            # List available CHIME targets
    bown grims                      # Run GRIM-S full catalog analysis
    bown selftest                   # Run self-tests for all instruments
"""

from __future__ import annotations

import sys


def main():
    """Entry point for the `bown` command."""
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    instrument = sys.argv[1]

    if instrument == "chime":
        # Delegate to CHIME's existing CLI, stripping 'bown' from argv
        sys.argv = ["bown-chime"] + sys.argv[2:]
        from bown_instruments.chime.cli import main as chime_main
        chime_main()

    elif instrument == "grims":
        sys.argv = ["bown-grims"] + sys.argv[2:]
        _grims_cli()

    elif instrument == "selftest":
        _run_selftests()

    elif instrument in ("--help", "-h"):
        _print_usage()

    elif instrument == "--version":
        from bown_instruments import __version__
        print(f"bown-instruments {__version__}")

    else:
        print(f"Unknown instrument: {instrument}", file=sys.stderr)
        _print_usage()
        sys.exit(1)


def _print_usage():
    print(
        "bown-instruments: Measurement tools built on Bown's engineering principles\n"
        "\n"
        "Usage:\n"
        "  bown chime [args]     JWST channel diagnostics (diversity weighting)\n"
        "  bown grims [args]     GW ringdown coupling (phase-locked stacking)\n"
        "  bown selftest         Run injection-recovery self-tests\n"
        "  bown --version        Show version\n"
        "\n"
        "Examples:\n"
        "  bown chime WASP-39              Channel quality for WASP-39\n"
        "  bown chime --targets            List available targets\n"
        "  bown grims --events 32          Analyze top 32 events\n"
        "  bown selftest                   Verify both instruments\n"
    )


def _grims_cli():
    """Minimal GRIM-S CLI wrapping mass_analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="bown grims",
        description="GRIM-S: Gravitational Intermodulation Spectrometer",
    )
    parser.add_argument(
        "--events", type=int, default=None,
        help="Maximum number of events to analyze (default: all)",
    )
    parser.add_argument(
        "--phase", type=int, default=3, choices=[1, 2, 3],
        help="Analysis phase (default: 3)",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run injection-recovery self-test before analysis",
    )
    parser.add_argument(
        "--outdir", type=str, default="results/grims",
        help="Output directory for results",
    )

    args = parser.parse_args()

    from bown_instruments.grims.mass_analysis import run_mass_analysis
    results = run_mass_analysis(max_events=args.events)
    print(f"kappa = {results.get('kappa', 'N/A')} +/- {results.get('kappa_unc', 'N/A')}")


def _run_selftests():
    """Run self-tests for all available instruments."""
    print("bown-instruments self-test protocol (US 1,573,801)")
    print("=" * 55)

    passed = True

    # GRIM-S self-test
    try:
        from bown_instruments.grims.self_test import run_self_test
        print("\n[GRIM-S] Injection-recovery self-test...")
        result = run_self_test()
        status = "PASS" if result.get("passed", False) else "FAIL"
        print(f"[GRIM-S] {status}")
        if not result.get("passed", False):
            passed = False
    except ImportError:
        print("[GRIM-S] Skipped (install with: pip install bown-instruments[grims])")
    except Exception as e:
        print(f"[GRIM-S] ERROR: {e}")
        passed = False

    # CHIME self-test (synthetic data round-trip)
    try:
        from bown_instruments.chime.channel_map import compute_channel_map
        print("\n[CHIME] Synthetic channel quality self-test...")
        import numpy as np
        rng = np.random.default_rng(42)
        n_int, n_wav = 100, 20
        flux = 1.0 + 0.01 * rng.standard_normal((n_int, n_wav))
        err = 0.01 * np.ones((n_int, n_wav))
        wavelengths = np.linspace(3.0, 5.0, n_wav)
        result = compute_channel_map(flux, err, wavelengths)
        n_graded = sum(1 for b in result.bins if b.grade in "ABCD")
        status = "PASS" if n_graded == n_wav else "FAIL"
        print(f"[CHIME] {status} ({n_graded}/{n_wav} bins graded)")
        if n_graded != n_wav:
            passed = False
    except ImportError:
        print("[CHIME] Skipped (install with: pip install bown-instruments[chime])")
    except Exception as e:
        print(f"[CHIME] ERROR: {e}")
        passed = False

    print("\n" + "=" * 55)
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
