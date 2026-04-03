#!/usr/bin/env python3
"""TRAPPIST-1 channel quality analysis.

Processes NIRSpec PRISM observations of TRAPPIST-1 (program 9256).
TRAPPIST-1 is much fainter than WASP-39, so the noise levels are
much higher and the systematic excess varies 150-280x across the band.

Expected results:
  - Trustworthy window: 1.5-3.5 um only
  - Below 1.0 um: scatter 100,000+ ppm (useless for atmosphere detection)
  - Diversity improvement: 26-47x over equal-weight combination
  - Large visit-to-visit variation in quality

Usage:
    python examples/trappist1_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bown_instruments.chime.cli import main

if __name__ == "__main__":
    main(
        [
            "TRAPPIST-1",
            "--planet",
            "b",
            "--bins",
            "50",
            "--max-obs",
            "3",
            "--outdir",
            "chime_output/trappist1",
        ]
    )
