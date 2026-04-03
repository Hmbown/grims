#!/usr/bin/env python3
"""WASP-39b channel quality map analysis.

Processes all available NIRSpec PRISM x1dints for WASP-39b and produces
the channel quality diagnostic. This is the primary validation target —
Rustamkulov+ 2023 published the first JWST transmission spectrum of this
planet from program 1366.

Expected results:
  - Noise varies ~29x across the band
  - Correlated noise at 1.0-1.3 um (Allan ratio 60-100x)
  - Trust region: ~1.5-3.5 um
  - Diversity improvement: 2.5-2.7x

Usage:
    python examples/wasp39_channel_map.py
"""

import sys
from pathlib import Path

# Add parent to path for running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bown_instruments.chime.cli import main

if __name__ == "__main__":
    main(["WASP-39", "--bins", "50", "--max-obs", "3", "--outdir", "chime_output/wasp39"])
