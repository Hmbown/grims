# chime

**CH**annel quality and **I**nstrument **M**etrology for **E**xoplanets

`chime` is a diagnostic package for JWST transit spectroscopy. It measures
per-wavelength, per-segment noise quality in reduced time-series products and
asks:

> How clean is this data empirically, relative to the photon-noise model, and
> does it average down like white noise?

The package is aimed at observers and reduction teams who want a quality-assurance
layer before concatenating segments, combining visits, or handing spectra to a
retrieval.

## Scientific background

Most transit-reduction pipelines deliver fluxes, uncertainties, and fitted
systematics models. `chime` adds three empirical diagnostics per wavelength bin:

1. **Empirical scatter** -- robust out-of-transit scatter (MAD-based sigma).
2. **Systematic excess** -- ratio of empirical scatter to the pipeline
   photon-noise estimate.
3. **Allan ratio** -- whether noise decreases with binning as expected for
   white noise.

These are converted into per-wavelength quality grades (A/B/C/D), trust regions,
and quality-weighted combining weights.

### Analytical lineage

The approach is inspired by Ralph Bown's sub-band diversity engineering
(US 1,747,221, 1930; US 1,794,393, 1931): measure the quality of each channel
empirically, grade it, and weight the combination by measured quality rather
than treating all channels equally. The application to JWST spectroscopy is new;
the underlying idea of empirical channel grading and selective combining is not.

## Installation

```bash
pip install chime-jwst
```

Or from source:

```bash
git clone https://github.com/Hmbown/bownpower.git
cd bownpower/chime
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `astropy`, `astroquery`.

## Quick start

### CLI

```bash
# List targets with ephemerides included in the package
chime --targets

# Run a single-target diagnostic on public archive products
chime WASP-39

# Compare per-segment products for a target
chime WASP-39 --segments
```

The CLI will search MAST for `x1dints` products, download them, identify
in-transit cadences from the packaged ephemeris, compute quality diagnostics,
and write figures and machine-readable outputs.

### Python API

```python
from chime import (
    compute_channel_map,
    compute_diversity,
    download_product,
    extract_transit_data,
    find_x1dints,
    get_ephemeris,
)

pairs = find_x1dints("WASP-39")
filepath = download_product(pairs[0][1])
td = extract_transit_data(filepath)

ephemeris = get_ephemeris("WASP-39")
bjd = td.times_mjd + 2400000.5
phase = ((bjd - ephemeris["t0_bjd"]) / ephemeris["period_days"]) % 1.0
phase[phase > 0.5] -= 1.0
in_transit = abs(phase) < (
    ephemeris["duration_hours"] / 24.0 / 2.0 / ephemeris["period_days"]
)

cmap = compute_channel_map(td.flux_cube, td.wavelength, in_transit)
div = compute_diversity(td.flux_cube, td.wavelength, in_transit)
```

## Outputs

| File | Meaning |
|---|---|
| `chime_<target>.png` | per-observation diagnostic plot |
| `diversity_<target>.png` | quality-weighted combination comparison |
| `chime_<target>.json` | machine-readable results |
| `chime_<target>.ecsv` | one row per wavelength bin |

### Key columns in the ECSV table

| Column | Meaning |
|---|---|
| `scatter_ppm` | empirical out-of-transit scatter |
| `photon_noise_ppm` | pipeline photon-noise estimate |
| `systematic_excess` | scatter / photon noise |
| `allan_ratio` | departure from white-noise averaging |
| `grade` | A/B/C/D quality grade |
| `depth_ppm` | rough transit-depth estimate for context |

### Quality grades

| Grade | Criteria | Interpretation |
|---|---|---|
| `A` | excess < 2 and Allan ratio < 1.5 | near photon-limited, approximately white |
| `B` | excess < 5 | usable, moderate residual systematics |
| `C` | excess < 10 | degraded, should be downweighted |
| `D` | excess >= 10 | systematic-dominated, usually exclude |

## Survey results

This repository includes a checked-in survey of public JWST NIRSpec transit
products (199 segments, 61 observations, 10 targets, 16 programs).

Key findings from the current archive sample:

- Median segment quality is 1.97x photon noise: the typical archived segment is
  not photon-limited.
- Segment 2 is worse than segment 1 in about 80% of paired comparisons
  (Wilcoxon p = 5.44e-06).
- G395H is the noisiest grating in this sample (confounded by target mix).
- WASP-121 is a clear target-level outlier.

These are archive-sample results, not instrument-wide causal claims. Target,
grating, detector, and program are partly confounded.

Start with:
- [`results/survey/SURVEY.md`](../results/survey/SURVEY.md)
- [`results/survey/CROSS_TARGET.md`](../results/survey/CROSS_TARGET.md)
- [`results/channel_survey/FINDING.md`](results/channel_survey/FINDING.md)

### Reproducing the survey

Cross-target analysis from checked-in data (no MAST access needed):

```bash
cd ..
python run_cross_target_analysis.py
```

Full archive survey (requires network, slow):

```bash
cd ..
python run_jwst_survey.py
```

## How to use the diagnostics scientifically

1. Look at `systematic_excess` first -- how far from photon-limited.
2. Check the `allan_ratio` -- if large, more averaging will not help.
3. Compare segments before concatenating them; segment quality varies
   substantially in the current archive.
4. Use grades as weights or triage, not as a substitute for a full reduction.

## Scope and limitations

- `chime` is a diagnostic layer, not a complete transit-fitting pipeline.
- Transit identification is ephemeris-based and intended for QA.
- Quality metrics are computed on reduced archive products, not raw ramps.
- Quick-look depth estimates are not science-grade transmission spectra.

## Examples

- [`examples/segment_quality_check.py`](examples/segment_quality_check.py) --
  one-page quality report for any x1dints file
- [`examples/multi_target_survey.py`](examples/multi_target_survey.py) --
  multi-target archive survey

## License

MIT. See [`LICENSE`](LICENSE).
