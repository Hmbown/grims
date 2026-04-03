"""chime — Channel Health & Instrument Metrology for Exoplanets.

Empirical noise diagnostics for JWST transit spectroscopy. Measures
per-wavelength channel quality, identifies correlated noise with Allan
deviation tests, and applies diversity combining inspired by Bown's
sub-band engineering methods (US 1,747,221, 1930).
"""

__version__ = "0.1.0"

from bown_instruments.chime.channel_map import channel_quality, compute_channel_map
from bown_instruments.chime.diversity import compute_diversity, DiversityResult
from bown_instruments.chime.ephemeris import get_ephemeris, list_targets, EPHEMERIDES
from bown_instruments.chime.extract import extract_transit_data, compute_white_light_curve
from bown_instruments.chime.mast import search_jwst, find_x1dints, download_product
from bown_instruments.chime.transit_fit import (
    mandel_agol_flux,
    fit_transit_with_gp,
    fit_transmission_spectrum,
    TransitFitResult,
)

__all__ = [
    "channel_quality",
    "compute_channel_map",
    "compute_diversity",
    "DiversityResult",
    "get_ephemeris",
    "list_targets",
    "EPHEMERIDES",
    "extract_transit_data",
    "compute_white_light_curve",
    "search_jwst",
    "find_x1dints",
    "download_product",
    "mandel_agol_flux",
    "fit_transit_with_gp",
    "fit_transmission_spectrum",
    "TransitFitResult",
]
