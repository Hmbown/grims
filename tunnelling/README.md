# Seismic Lookahead for Tunnel Boring

Technology transfer from gravitational-wave signal processing.

## The Problem

Your TBM needs to see what's ahead — faults, voids, water-bearing zones,
lithology changes — before the cutterhead hits them. Seismic-while-drilling
(SWD) methods send waves ahead of the tunnel face and listen for reflections,
but:

1. **The TBM is deafening.** Cutter vibration, thrust jacks, muck conveyors
   generate broadband noise that buries the reflections you're looking for.
2. **The noise changes constantly.** RPM varies, cutters wear, geology changes
   the coupling. Yesterday's noise model doesn't work today.
3. **False alarms are expensive.** Stopping a TBM costs tens of thousands of
   dollars per day. You need calibrated confidence before you call a halt.
4. **The interesting reflections are weak.** Gradual impedance contrasts
   (clay to silt, weathered to fresh rock) produce reflections well below
   the noise floor.

## The Technology We Built (For a Different Problem)

We built a signal processing pipeline called GRIM-S to detect extremely weak
oscillation signals in LIGO gravitational-wave detector data. The core
challenge is identical to yours:

| Your Problem | Our Problem |
|---|---|
| TBM vibration noise (broadband, non-stationary) | Detector noise (broadband, non-stationary — dominated by ground motion at low frequencies, quantum shot noise at high frequencies) |
| Seismic reflections from geology ahead | Oscillation modes from astrophysical sources |
| Reflection frequency set by rock velocity + distance | Mode frequency set by source properties (mass, spin) |
| Multiple reflections from layered geology | Multiple oscillation modes (fundamental + overtones) |
| Combine data from multiple geophones | Combine data from multiple detectors (3 observatories worldwide) |
| Stop the TBM? (high-stakes, costly decision) | Claim a detection? (high-stakes, reputation-on-the-line decision) |

## Contents

### 1. `seismic_lookahead.py` — Signal Processing Module

Direct adaptation of the GRIM-S detection pipeline, re-parameterized for
tunnel seismic:

- **`estimate_noise_psd()`** — Welch-method noise characterization from TBM
  vibration recordings.
- **`matched_filter_reflection()`** — Frequency-domain matched filter: correlates
  the raw geophone trace with the known source wavelet, weighted by the inverse
  noise PSD. This is the optimal linear detector for a known signal shape in
  coloured noise.
- **`scan_travel_times()`** — Sweeps the matched filter across travel times to
  build a reflectivity-vs-distance profile of the ground ahead.
- **`multi_geophone_stack()`** — Inverse-variance weighted combination across
  geophone channels, with weight capping to protect against miscalibrated sensors.
- **`detection_confidence()`** — False alarm probability with look-elsewhere
  correction. Answers: "given current TBM noise, what's the probability this
  peak is real?"
- **`run_injection_test()`** — Injects synthetic reflections into real noise
  and measures recovery. Calibrates your sensitivity curve.
- **`whiten_seismic()`** / **`bandpass()`** — Time-domain whitening and filtering,
  useful for visualisation and diagnostics.

### 2. `synthetic_demo.py` — Proof of Concept

Generates synthetic data mimicking a real TBM environment:

- Realistic noise model (low-frequency cutter vibration + cutter harmonics +
  broadband mechanical noise)
- Three geological boundaries at known distances and impedance contrasts
- Full pipeline run producing a four-panel figure (`tunnelling_demo.png`):
  - **(a)** Raw geophone trace — reflections invisible
  - **(b)** TBM noise power spectrum — shows the coloured noise environment
  - **(c)** Matched-filter SNR vs distance — clear peaks at boundaries
  - **(d)** Injection sensitivity curve — calibrated detection threshold

Run it: `python synthetic_demo.py`

### 3. This Document

Maps every technique to your domain so your engineers can evaluate fit
without needing to learn gravitational-wave physics.

## Key Techniques, Translated

### Adaptive Noise Characterisation

**What we do:** Estimate the noise power spectral density (PSD) from data
where no signal is present, then use that PSD to weight the matched filter
so that every frequency contributes optimally.

**How it maps to tunnelling:** Record geophone data while the TBM is running
but no active source is fired. This pure-noise PSD characterises the machine
vibration environment. The matched filter then automatically down-weights
frequency bands dominated by TBM noise (cutter harmonics, hydraulic
resonances, ground roll) and up-weights quiet bands where reflections are
visible.

**Why it matters:** TBM noise is strongly coloured — certain frequency bands
are orders of magnitude louder than others. Without PSD weighting, your
detector is dominated by whatever band is loudest, not whatever band carries
the most signal.

### Matched Filtering with a Known Wavelet

**What we do:** Correlate the data against a template whose shape is known
from physics, swept across all possible arrival times. The output is the
optimal signal-to-noise ratio (SNR) at each time.

**How it maps to tunnelling:** If you know the source wavelet (from a pilot
geophone near the source, or from a reference shot in known geology), the
reflected wavelet's shape is determined — only arrival time, amplitude, and
polarity are free. The matched filter correlates against this known shape,
swept across travel times. Each travel time maps to a distance ahead of the
face.

**Why it matters:** A simple energy detector (RMS in a time window) uses no
knowledge of the wavelet shape. The matched filter uses all of it. In our
application, template-based detection improved sensitivity by ~40% over a
generic search.

### Injection Calibration

**What we do:** Inject synthetic signals at known amplitudes into real
detector noise, then measure how well the pipeline recovers them. This
tells us: (a) is the estimator biased? (b) are the quoted uncertainties
correct? (c) at what amplitude do we lose sensitivity?

**How it maps to tunnelling:** Record TBM-only noise (no active source).
Inject synthetic reflections at known amplitudes and travel times. Run the
detection pipeline. Measure:

- **Recovery rate** at each amplitude (your sensitivity curve)
- **Travel time accuracy** (is the distance estimate biased?)
- **False alarm rate** in noise-only segments (your specificity)
- **Uncertainty calibration** (do "90% confidence" intervals actually
  contain the true value 90% of the time?)

**Why it matters:** Without injection testing, you don't know your detection
threshold. You'll either stop the TBM for noise fluctuations (expensive) or
miss real hazards (dangerous). The injection campaign gives you a calibrated
answer: *"We can detect an impedance contrast of X% at distance Y metres
with Z% confidence in current noise conditions."*

### Multi-Channel Coherent Stacking

**What we do:** Each detector sees the same signal with independent noise.
We combine them with inverse-variance weighting: quieter detectors get more
weight. Combined SNR grows as sqrt(N).

**How it maps to tunnelling:** Multiple geophones along the tunnel wall each
see the same reflection with independent (or partially correlated) noise.
Weight each channel by the inverse of its noise variance. Channels with
better coupling get more weight automatically. Same principle applies to
stacking across multiple source shots.

## What You'd Need for a Pilot

1. **TBM noise recordings** — continuous geophone data during boring (no
   active source). Multiple channels preferred. Any sample rate >= 1 kHz.
2. **Active source recordings** — same geophones, with your seismic source
   fired. Include a pilot/reference geophone near the source.
3. **Known geology section** — a stretch where you have borehole data or
   face mapping, so we can validate detections against ground truth.
4. **Operating parameters** — TBM RPM, thrust, cutter diameter, advance rate.
   These help us model the noise spectrum.

## What We'd Deliver

1. **Noise characterisation** of your specific TBM: which frequency bands
   are usable, which are dominated by machine vibration.
2. **Calibrated sensitivity curve** — detection probability vs impedance
   contrast vs distance, for your noise conditions.
3. **Real-time lookahead display** — reflectivity vs distance ahead of face,
   updated with each source shot, with colour-coded confidence levels.
4. **Injection test report** — demonstrating the pipeline is unbiased and
   confidence levels are calibrated for your data.

## Contact

Hunter Bown — hunter@bown.io
