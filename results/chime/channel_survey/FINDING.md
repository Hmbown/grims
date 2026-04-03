# Segment-level noise heterogeneity in JWST NIRSpec G395H transit spectroscopy

## Summary

Within a single JWST NIRSpec G395H bright object time-series (BOTS)
exposure of WASP-39b, the empirical noise quality varies by a factor of
approximately 4 between consecutive temporal segments. One segment
approaches photon-limited performance; the next is
systematic-dominated. Current pipelines do not measure or act on this
per-segment variation.

This observation has been checked against the same analysis applied to
six G395H segments across two visits (WASP-39b and WASP-107b).

## Measurement

We applied chime's per-wavelength channel quality diagnostics to six
per-segment datasets covering the 2.86--3.72 micron range with the
G395H grating:

### WASP-39b (Program 1366, Observation 3)

| Dataset | Integrations | A-grade | Sys. excess | Allan ratio | Trust coverage |
|---------|-------------|---------|-------------|-------------|----------------|
| seg001 | 155 | 17/18 | 1.19x | 1.09 | 0.86 um (100%) |
| seg002 | 155 | 0/18  | 4.89x | 3.36 | 0.00 um (0%)   |
| seg003 | 155 | 12/18 | 1.62x | 1.30 | 0.86 um (100%) |

### WASP-107b (Program 1224, Observation 3)

| Dataset | Integrations | A-grade | Sys. excess | Allan ratio | Trust coverage |
|---------|-------------|---------|-------------|-------------|----------------|
| seg001 | 411 | 18/18 | 1.40x | 1.12 | 0.86 um (100%) |
| seg002 | 411 | 13/18 | 1.36x | 1.18 | 0.86 um (100%) |
| seg003 | 408 | 15/18 | 1.28x | 1.21 | 0.86 um (100%) |

**Systematic excess** is the ratio of empirical scatter to pipeline-reported
photon noise. A value of 1.0 means the data is photon-limited. A value of
4.89 means the measured noise is nearly 5x worse than the photon statistics
predict.

**Allan ratio** measures whether noise averages down as 1/sqrt(n). A value of
1.0 indicates white noise (averaging helps). A value of 3.36 indicates
correlated noise dominates (additional averaging yields diminishing returns).

## Confound analysis

We checked five potential confounds that could explain the ~4x quality
variation between WASP-39b segments:

| Confound | Result | Explains finding? |
|----------|--------|-------------------|
| Timestamp ordering | seg001 precedes seg002 by 1.1 min | No |
| Transit phase overlap | seg001: 50% in-transit, seg002: 52% | No |
| Flux level difference | Flux ratio seg002/seg001 = 0.982 | No |
| Bad pixel / NaN count | Identical (4650 each) | No |
| INT_TIMES/EXTRACT1D mismatch | Both 155 entries, no fix triggered | No |

None of the tested confounds account for the observed variation.

## Expanded survey

We analyzed all available G395H/NRS1 per-segment x1dints files on MAST
for WASP-39 (Program 1366) and WASP-107 (Program 1224). HD-189733 and
HD-209458 have no NIRSpec G395H observations in the public archive at
the time of this analysis.

### Intra-visit variation

| Visit | Segments | Worst/best excess ratio |
|-------|----------|------------------------|
| WASP-39 P1366 Obs3 | 3 | **4.09x** |
| WASP-107 P1224 Obs3 | 3 | 1.10x |

The effect is present in 1 of 2 G395H visits examined (50%). It is not
universal but is substantial when it occurs.

### Temporal dependence

The correlation between segment number and systematic excess across all
segments is r = -0.05 -- essentially zero. This is inconsistent with a
simple monotonic detector ramp (which would produce a strong positive
correlation). The WASP-39b Obs 3 anomaly is specifically in seg002, with
seg003 recovering to near-normal quality. This good-bad-good pattern
suggests a transient systematic event rather than a detector degradation
trend.

### Wavelength dependence

The seg002 anomaly is broadband: every wavelength channel shows elevated
systematic excess (2.7x to 5.6x above seg001 levels). There is no
wavelength where seg002 is photon-limited. The effect is uniform across
the 2.86-3.72 um band, consistent with a detector-wide or optical-path
systematic rather than a wavelength-specific issue.

## Comparison to COMPASS

The JWST COMPASS study (Gordon et al. 2025, arXiv:2511.18196) analyzed
systematic noise properties of NIRSpec/G395H across seven targets. They
found that PandeXo predictions are "a relatively accurate predictor of
the precision of the spectra, with real error bars on average 5% larger
in NRS1 and 12% larger in NRS2 than predicted."

Our measurements are consistent with this for photon-limited segments
(WASP-39b seg001: 1.19x excess; WASP-107b seg001: 1.40x excess). However,
COMPASS reports visit-averaged statistics and does not resolve per-segment
variation. Our analysis shows that within a single visit, individual
segments can deviate substantially from the visit average -- a detail that
visit-level analysis would miss entirely.

The COMPASS study also notes that "systematics are particularly strong
between 2.8 and 3.5 um" -- the exact band where we observe the seg002
anomaly. This lends independent support to the reality of the effect.

## Relevance to published spectra

The WASP-39b ERS transmission spectrum, published by Alderson et al.
(2023, Nature 614, 664), was produced by 11 independent pipeline
reductions. All 11 pipelines concatenated the temporal segments and fit
a single systematics model across the full time stream. To our knowledge,
none measured per-segment noise quality or applied segment-level
weighting.

If segments within a visit have unequal noise quality (as our measurement
shows can be a factor of 4 or more), then concatenation at equal weight
dilutes photon-limited data with systematic-contaminated data. The
correlated noise in segment 2 (Allan ratio 3.36) means that averaging
more integrations from that segment does not improve the measurement
proportionally to white-noise expectations.

Whether this affects any specific published result requires re-analysis of
those datasets with per-segment quality diagnostics.

## What current pipelines do

Every major JWST exoplanet transit pipeline handles temporal segments by
concatenating them into a single time stream:

- **JWST calwebb_tso3**: Concatenates segments. No quality assessment.
- **Eureka!**: Processes the full time series through all 6 stages.
- **Tiberius**: Fits systematics models across the full observation.
- **FIREFLy**: Custom calibrations on raw data, full-stream processing.
- **ExoTiC-JEDI**: Full-stream extraction and light-curve fitting.

Inverse-variance weighting does appear in the literature, but across
independent pipeline reductions (Alderson et al. 2023 combined 11
pipelines with 1/sigma^2 weighting) or across separate visit epochs.
Within a single reduction of a single visit, integrations are typically
treated uniformly.

Allan deviation analysis has been used as a post-hoc diagnostic by
Espinoza et al. (2023), Alderson et al. (2023), and JexoPipe (Schleich
et al. 2024), but on band-integrated residuals as a pass/fail check.
It has not been applied per wavelength channel per temporal segment to
drive quality grading and weighting in the way chime does.

## What chime measures

chime computes three independent diagnostics for every wavelength bin in
every temporal segment:

1. **Empirical scatter**: MAD-based robust sigma of out-of-transit
   normalized flux. This is what the noise actually is, not what a model
   predicts.

2. **Photon noise reference**: From the pipeline's own flux error
   arrays. This is the theoretical floor.

3. **Allan deviation**: Block-average the out-of-transit flux at sizes
   1, 2, 5, 10, 20. If the scatter decreases as 1/sqrt(block_size),
   the noise is white and more data helps. If it does not, the
   noise is correlated and additional averaging is inefficient.

These three measurements combine into a quality grade:

- **A**: Photon-limited, white noise. Trust this channel completely.
- **B**: Moderate excess systematics. Usable with care.
- **C**: Significant systematics. Downweight.
- **D**: Systematic-dominated. Discard.

The grades drive diversity combining weights: A and B channels are
weighted by 1/scatter^2. C channels get a soft rolloff. D channels get
zero weight. The result is a transmission spectrum that emphasizes the
cleanest data.

## Implication

For JWST transit observations where temporal segments have unequal noise
quality, the standard concatenation approach is suboptimal. It combines
photon-limited data with systematic-contaminated data at equal weight.
The error bars it reports may underestimate the true uncertainty for
contaminated segments (which exhibit correlated noise).

The WASP-107b observation (Program 1224) demonstrates that G395H can
deliver uniformly photon-limited data across the full NRS1 band (18/18
A-grade bins, Allan ratio 1.12). The segment 2 degradation in WASP-39b
is not necessarily a fundamental instrument limitation. It may be a
time-dependent systematic that the current reduction framework does not
flag.

The proposed remedy is straightforward: measure each segment's noise
independently, grade it, and weight accordingly. This requires no change
to the calibration pipeline and no additional observations.

## Diagnostic tool

A segment quality diagnostic script is available at
`examples/segment_quality_check.py`. It takes any x1dints file and
produces a PASS/WARN/FAIL assessment with per-channel grades, Allan
ratios, and trust regions:

```
python examples/segment_quality_check.py path/to/file.fits
python examples/segment_quality_check.py --download WASP-39
```

The CLI also supports `--segments` to compare all per-segment files for
a given target:

```
chime WASP-39 --segments
```

## Provenance

Measured with chime v0.1.0. Raw x1dints files from MAST (Programs 1366,
1224). No custom calibration applied -- these diagnostics operate on
the pipeline-delivered Stage 2 products.

Data references: WASP-39b ERS (Alderson+ 2023, Nature 614, 664);
WASP-107b (Dyrek+ 2024, Nature 625, 51).

Analytical approach: Inspired by Bown's sub-band diversity engineering
(US 1,747,221, 1930) -- the idea of measuring and weighting channels by
empirical quality rather than treating all channels equally. Applied here
to JWST transit spectroscopy for the first time.

Comparison: JWST COMPASS (Gordon+ 2025, arXiv:2511.18196).
