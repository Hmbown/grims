# GRIM-S

**Gravitational Intermodulation Spectrometer**

GRIM-S measures a nonlinear property of Kerr spacetime in black hole ringdown: the quadratic quasinormal-mode coupling coefficient `kappa`. After merger, the remnant is first described by linear quasinormal modes, radiative perturbations with null propagation on the Kerr background. At second order, Einstein's equations make those modes self-couple: the squared amplitude of the dominant `(2,2,0)` mode sources a daughter response at `2ω220`. `kappa` measures the strength of that coupling.

This is a measurement of spacetime dynamics, not a generic excess-power search. Because the field equations are nonlinear, two null disturbances need not remain null: in the same Lorentzian sense that two null 4-displacements can sum to a timelike resultant, two radiative Kerr perturbations can drive a daughter with spectral content no linear mode carries. It is the classical gravitational analogue of photon-photon pair production, with no quantum vacuum involved and no new physics beyond the self-coupling already present in general relativity. GRIM-S estimates `kappa` from public LIGO/Virgo/KAGRA ringdowns by fitting the parent mode, locking the quadratic daughter to it, and stacking events.

The current Phase 3 stack over 128 binary black hole mergers yields `kappa = +0.015 ± 0.007` (2.2 sigma), with weight-capped stacking to prevent single-event dominance. That is an empirical constraint on nonlinear mode coupling in the strong-field regime.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="plots/grims_pipeline_dark.svg">
    <img alt="GRIM-S pipeline: Input, Condition, Search, Stack — with key simplification equations" src="plots/grims_pipeline_light.svg" width="840">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="plots/grims_comparison_dark.svg">
    <img alt="Two methods compared on the same kappa axis: fast stack shows narrow uncertainty, audited result shows wide uncertainty" src="plots/grims_comparison_light.svg" width="840">
  </picture>
</p>

## Results

### Phase 3 (2026-03-31): Multi-Detector + Adaptive Segment + Weight-Capped Stacking

| Metric | Phase 2.5 | Phase 3 (uncapped) | Phase 3 (capped 5.5×) |
|---|---|---|---|
| Events stacked | 122 | 128 | 128 |
| Stacked kappa | +0.016 ± 0.030 | +0.011 ± 0.006 | **+0.015 ± 0.007** |
| Stacked SNR | 13.03 | 15.68 | 15.68 |
| Significance | 0.5 sigma | 2.0 sigma | **2.2 sigma** |
| N_eff (Herfindahl) | — | 14.9 | **21.4** |
| Influential events | 11 | 19 | **16** |
| Multi-detector events | 0 | 103 (37 triple-det) | 103 (37 triple-det) |
| Total detector measurements | 122 | 266 | 266 |

Phase 3 extends the analysis to multi-detector data, replaces the fixed post-merger window with an event-by-event segment length of `5 × τ220`, and applies weight-capped stacking (`max_weight_ratio=5.5`) to reduce influence concentration. Without the cap, three events carry 44% of the total weight (N_eff = 14.9). The cap raises N_eff to 21.4 and improves significance from 2.0σ to 2.2σ.

### Phase 2.5 (2026-03-31): Colored Noise + `t_start` Marginalization

Phase 2.5 introduced a frequency-domain colored-noise likelihood using each event's measured PSD and marginalized the ringdown start over `[5M, 8M, 10M, 12M, 15M, 20M]`. The central value shifted from `+0.028` to `+0.016`, with wider errors (`±0.030`) once the fixed-`t_start` assumption was removed. Injection studies recover large couplings reliably but remain insensitive to `kappa ~ 0.03` event by event.

### Phase 2 (2026-03-30): 134-Event Expansion

| Metric | Value |
|---|---|
| Events stacked | 134 |
| Stacked kappa | +0.028 ± 0.019 |
| Stacked SNR | 11.45 |
| Significance | 1.5 sigma |
| NR prediction | kappa ~ 0.03 (Mitman/Cheung/Zhu) |
| Consistency with zero | Yes (1.5 sigma) |
| Consistency with NR | Yes (within uncertainties) |

### Phase 1 (2026-03-30): 32-Event Baseline

| Metric | Value |
|---|---|
| Events stacked | 32 |
| Stacked kappa | -0.047 ± 0.043 |
| Stacked SNR | 7.24 |
| Jackknife stability | Unstable — 2 high-SNR events dominate |

## Measurement Strategy

1. Load GWTC remnant metadata and GWOSC strain for H1, L1, and V1 where available.
2. Estimate detector ASD, whiten, and bandpass to the ringdown band.
3. Choose an adaptive post-merger segment length of `5 × τ220` for each event.
4. Fit the parent `(2,2,0)` mode in the post-merger data.
5. Build a quadratic daughter template phase-locked to the parent, with amplitude proportional to `kappa A220^2`, and evaluate it with a colored-noise matched filter.
6. Marginalize over ringdown start time, combine detectors, and inverse-variance stack across events.

## Current Limits

The dominant limitation is sensitivity. Phase 3 is a 2.2 sigma constraint, not yet a detection.

Weight-capped stacking (5.5× average) reduces influence concentration: N_eff improves from 14.9 to 21.4, and the number of influential jackknife events drops from 19 to 16. A full joint inference over remnant parameters, detector systematics, and ringdown start time remains to be done (SHA-4141). Pushing toward 3σ requires downloading the remaining ~8.5 GB of O4 strain data (SHA-4142).

Full audit history: `AUDIT_AND_FAILURE_REGISTER.md`

## Repository Structure

```text
grims/
  qnm_modes.py              Kerr QNM frequencies
  ringdown_templates.py     waveforms parameterized by kappa
  gwtc_pipeline.py          catalog, download, loading
  whiten.py                 ASD, whitening, bandpass
  phase_locked_search.py    phase-locked matched filter
  bayesian_analysis.py      posterior, profile likelihood, stacking
  mass_analysis.py          full-catalog pipeline
  self_test.py              injection self-test
  jackknife.py              leave-one-out stability test
  nr_predictions.py         NR kappa predictions vs spin
  fisher_analysis.py        parameter degeneracy analysis
  colored_likelihood.py     colored-noise PSD likelihood
  sampler.py                MCMC sampler
  visualize_deep.py         plotting suite
scripts/
  run_phase1_analysis.py    Phase 1 analysis runner
  run_phase2_analysis.py    colored-noise + MCMC runner
  run_phase3_analysis.py    multi-detector + adaptive segment runner
tests/                      injection-recovery and diagnostic tests
data/                       GWOSC strain files (not committed)
plots/                      generated figures
```

## Usage

```bash
pip install qnm numpy scipy matplotlib h5py
```

```bash
# Single event
python -c "
from grims.whiten import prepare_ringdown_for_analysis
from grims.phase_locked_search import phase_locked_search
prep = prepare_ringdown_for_analysis('GW150914', data_dir='data/')
r = phase_locked_search(prep['strain_whitened'], prep['t_dimless'],
    spin=prep['event']['remnant_spin'], noise_rms=prep['noise_rms'],
    event_name='GW150914')
print(f'kappa = {r.kappa_hat:.3f} +/- {r.kappa_sigma:.3f}, SNR = {r.snr:.3f}')
"
```

```bash
# Full catalog stack
python -c "
from grims.mass_analysis import run_mass_analysis
results = run_mass_analysis(min_total_mass=30.0)
s = results['stacked']
print(f'kappa = {s.kappa_hat:.4f} +/- {s.kappa_sigma:.4f}, SNR = {s.snr:.3f}')
"
```

## References

- Cheung et al., *PRL* **130**, 081401 (2023). [arXiv:2208.07374](https://arxiv.org/abs/2208.07374)
- Mitman et al., *PRL* **130**, 081402 (2023). [arXiv:2208.07380](https://arxiv.org/abs/2208.07380)
- Green et al., *PRD* **107**, 064030 (2023). [arXiv:2210.15935](https://arxiv.org/abs/2210.15935)
- LIGO/Virgo/KAGRA, GWTC-3: [gwosc.org](https://gwosc.org)

## License

MIT

## Acknowledgments

Gravitational-wave data are from [GWOSC](https://gwosc.org). Kerr QNM frequencies are computed with Leo C. Stein's `qnm` package.
