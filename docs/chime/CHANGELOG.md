# Changelog

All notable changes to the chime package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2025-04-01

Initial release.

### Added

- Per-wavelength channel quality diagnostics (`channel_map` module):
  empirical scatter, systematic excess, Allan deviation, A/B/C/D grades,
  trust regions, diversity combining weights.
- Sub-band diversity combining (`diversity` module): quality-weighted
  transmission spectrum with configurable grade thresholds.
- Mandel & Agol transit model fitting with GP systematics removal
  (`transit_fit` module).
- MAST interface for searching, listing, and downloading JWST x1dints
  products (`mast` module).
- Per-integration spectrum extraction from x1dints FITS files (`extract`
  module).
- Ephemeris table for 12 JWST exoplanet targets (`ephemeris` module).
- CLI (`chime`) for single-target diagnostics and segment comparison.
- Diagnostic plotting: 4-panel channel map, diversity comparison,
  multi-visit overlay, per-segment comparison.
- Example scripts: `segment_quality_check.py`, `multi_target_survey.py`,
  and others.
- Checked-in survey of 199 NIRSpec segments (61 observations, 10 targets,
  16 programs) with master table and cross-target analysis.
- Test suite covering channel_map, diversity, ephemeris, extraction, and
  transit fitting (synthetic data, no network required).
