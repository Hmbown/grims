"""bown_instruments.core -- Shared measurement principles from Ralph Bown's engineering.

This module codifies the recurring patterns across both instruments:

1. Self-test: inject a known signal, verify recovery before trusting real data.
   Patent lineage: US 1,573,801 (1926) -- Self-testing trouble alarm.

2. Diversity weighting: grade channels independently, weight by measured quality.
   Patent lineage: US 1,747,221 (1930) -- Sub-band diversity reception.

3. Measurement-first diagnostics: instrument health before science.
"""

from bown_instruments.core.self_test import SelfTest
from bown_instruments.core.diversity import diversity_weight
from bown_instruments.core.diagnostics import check_instrument_health

__all__ = ["SelfTest", "diversity_weight", "check_instrument_health"]
