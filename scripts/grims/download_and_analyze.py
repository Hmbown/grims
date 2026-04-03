"""
Task 3: Download and analyze more O1/O2/O3 events.

Fetch strain data from GWOSC for high-SNR events not yet in local cache,
then run the phase-locked search on them.
"""

import sys
import os
import json
import time
import urllib.request
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from bown_instruments.grims.whiten import estimate_asd, whiten_strain, bandpass
from bown_instruments.grims.phase_locked_search import phase_locked_search, stack_phase_locked
from bown_instruments.grims.qnm_modes import KerrQNMCatalog
from bown_instruments.grims.gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5

DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "gwtc_full_catalog.json"


def find_local_strain(gps_float, data_dir="data/"):
    """Check if we already have data covering this GPS time."""
    data_path = Path(data_dir)
    gps_int = int(gps_float)
    for f in sorted(data_path.glob("*.hdf5")):
        parts = f.stem.split("-")
        if len(parts) >= 3:
            try:
                file_gps = int(parts[-2])
                file_dur = int(parts[-1])
                if file_gps <= gps_int <= file_gps + file_dur:
                    return str(f)
            except ValueError:
                continue
    return None


def download_event_strain(event, data_dir="data/"):
    """Try to download 4KHZ 32s HDF5 strain from GWOSC."""
    jsonurl = event.get("jsonurl")
    if not jsonurl:
        return None, None

    try:
        req = urllib.request.Request(jsonurl)
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception as e:
        print(f"    Failed to fetch metadata: {e}")
        return None, None

    events_dict = payload.get("events", payload)
    if isinstance(events_dict, dict):
        event_data = next(iter(events_dict.values()), payload)
    else:
        event_data = payload

    strain_list = event_data.get("strain", [])
    if not strain_list:
        return None, None

    # Prefer 4KHZ 32s HDF5
    candidates = [
        s
        for s in strain_list
        if s.get("format") == "hdf5"
        and s.get("sampling_rate") == 4096
        and s.get("duration") == 32
    ]
    if not candidates:
        # Fall back to any HDF5
        candidates = [s for s in strain_list if s.get("format") == "hdf5"]
    if not candidates:
        return None, None

    # Prefer H1, then L1, then V1
    for pref in ["H1", "L1", "V1"]:
        for c in candidates:
            if c.get("detector") == pref:
                url = c["url"]
                det = c["detector"]
                break
        else:
            continue
        break
    else:
        c = candidates[0]
        url = c["url"]
        det = c.get("detector", "H1")

    local_file = Path(data_dir) / Path(url).name
    if local_file.exists():
        return str(local_file), det

    try:
        print(f"    Downloading {Path(url).name}...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        with open(local_file, "wb") as f:
            f.write(data)
        print(f"    Downloaded {len(data) / 1e6:.1f} MB")
        return str(local_file), det
    except Exception as e:
        print(f"    Download failed: {e}")
        return None, None


def analyze_downloaded_event(event, local_path, detector):
    """Run the phase-locked search on a downloaded event."""
    mass = event.get("remnant_mass", 0)
    spin = event.get("remnant_spin", 0.69)
    gps = event.get("gps", 0)

    if mass <= 0 or gps <= 0:
        return None

    m_sec = mass * M_SUN_SECONDS

    catalog = KerrQNMCatalog()
    mode_220 = catalog.linear_mode(2, 2, 0, spin)
    mode_nl = catalog.nonlinear_mode_quadratic(spin)
    mode_440 = catalog.linear_mode(4, 4, 0, spin)

    f_220 = mode_220.physical_frequency_hz(mass)
    f_nl = mode_nl.physical_frequency_hz(mass)
    f_440 = mode_440.physical_frequency_hz(mass)

    f_low = max(20.0, f_220 * 0.5)
    f_high_target = max(f_nl, f_440) * 1.3

    try:
        loaded = load_gwosc_strain_hdf5(local_path)
    except Exception as e:
        print(f"    Load failed: {e}")
        return None

    strain = loaded["strain"]
    time_arr = loaded["time"]
    sr = loaded["sample_rate"]
    f_high = min(0.45 * sr, f_high_target)

    if f_220 < 20 or f_nl > 0.45 * sr:
        return None

    # Handle O4 16KHZ files: chunk to 64s around event and downsample
    if sr > 4096:
        from scipy.signal import decimate

        chunk_start = gps - 16.0
        chunk_end = gps + 48.0
        mask = (time_arr >= chunk_start) & (time_arr <= chunk_end)
        if np.sum(mask) < 1000:
            return None
        strain = strain[mask]
        time_arr = time_arr[mask]
        # Decimate to 4096
        decim = int(sr / 4096)
        strain = decimate(strain, decim)
        sr = 4096.0
        time_arr = time_arr[np.arange(0, len(strain) * decim, decim)[: len(strain)]]

    merger_time = float(gps)
    ringdown_start = merger_time + 10.0 * m_sec

    if merger_time < time_arr[0] or merger_time > time_arr[-1]:
        return None

    try:
        asd_freqs, asd = estimate_asd(
            strain, sr, merger_time=merger_time, time=time_arr, exclusion_window=2.0
        )
        whitened = whiten_strain(strain, sr, asd_freqs, asd, fmin=f_low * 0.8)
        whitened_bp = bandpass(whitened, sr, f_low, f_high)
    except Exception as e:
        print(f"    Whitening failed: {e}")
        return None

    pad_before = 0.05
    seg_duration = 0.15
    t0 = ringdown_start - pad_before
    t1 = ringdown_start + seg_duration
    mask = (time_arr >= t0) & (time_arr <= t1)

    if np.sum(mask) < 50:
        return None

    seg_strain = whitened_bp[mask]
    seg_time = time_arr[mask]
    t_dimless = (seg_time - ringdown_start) / m_sec

    noise_mask = np.abs(time_arr - merger_time) > 4.0
    if np.sum(noise_mask) < 100:
        return None
    noise_var = np.var(whitened_bp[noise_mask])
    noise_rms = np.sqrt(noise_var)

    try:
        r = phase_locked_search(
            seg_strain,
            t_dimless,
            spin,
            noise_rms,
            event_name=event["name"],
        )
    except Exception as e:
        print(f"    Search failed: {e}")
        return None

    return r


def main():
    print("=" * 70)
    print("TASK 3: Download and Analyze More Events")
    print("=" * 70)

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    targets = [e for e in catalog if e["total_mass"] >= 40.0]
    print(f"Total events with M >= 40: {len(targets)}")

    # Sort by SNR descending
    targets.sort(key=lambda e: e.get("snr", 0), reverse=True)

    all_results = []
    newly_analyzed = 0
    already_had = 0
    download_failed = 0
    analysis_failed = 0

    for ev in targets:
        name = ev["name"]

        # Check local cache first
        local = find_local_strain(ev["gps"])
        if local:
            already_had += 1
            r = analyze_downloaded_event(ev, local, None)
            if r is not None:
                all_results.append(r)
                print(
                    f"  {name:<30} kappa={r.kappa_hat:+.3f} +/- {r.kappa_sigma:.3f}  "
                    f"SNR={r.snr:+.3f}  [cached]"
                )
            continue

        # Try downloading (limit to top events to not hammer GWOSC)
        if newly_analyzed + download_failed >= 40:
            continue

        print(
            f"  {name:<30} M={ev['total_mass']:.1f} SNR={ev.get('snr', 0):.1f} "
            f"-- downloading..."
        )
        local, det = download_event_strain(ev)
        if local is None:
            download_failed += 1
            continue

        r = analyze_downloaded_event(ev, local, det)
        if r is not None:
            all_results.append(r)
            newly_analyzed += 1
            print(
                f"    -> kappa={r.kappa_hat:+.3f} +/- {r.kappa_sigma:.3f}  "
                f"SNR={r.snr:+.3f}  [NEW]"
            )
        else:
            analysis_failed += 1

        time.sleep(0.5)  # Be nice to GWOSC

    print(f"\n{'=' * 70}")
    print(
        f"Results: {len(all_results)} events analyzed "
        f"({already_had} cached, {newly_analyzed} newly downloaded)"
    )
    print(f"Failed: {download_failed} downloads, {analysis_failed} analyses")

    if all_results:
        stacked = stack_phase_locked(all_results)
        print(f"\n  STACKED ({stacked.n_events} events):")
        print(f"  kappa = {stacked.kappa_hat:+.3f} +/- {stacked.kappa_sigma:.3f}")
        print(f"  SNR = {stacked.snr:.3f}")
        k = stacked.kappa_hat
        s = stacked.kappa_sigma
        ci_95 = (k - 1.96 * s, k + 1.96 * s)
        print(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
        print(f"  kappa > 0 at 95%? {'Yes' if ci_95[0] > 0 else 'No'}")
        print(f"  kappa = 1 in 95% CI? {'Yes' if ci_95[0] <= 1 <= ci_95[1] else 'No'}")
        print(f"  kappa = 0 in 95% CI? {'Yes' if ci_95[0] <= 0 <= ci_95[1] else 'No'}")

        # Show top 10 by |SNR|
        print(f"\n  Top 10 events by |SNR|:")
        sorted_results = sorted(all_results, key=lambda r: abs(r.snr), reverse=True)
        for r in sorted_results[:10]:
            print(
                f"    {r.event_name:<30} kappa={r.kappa_hat:+.3f} +/- {r.kappa_sigma:.3f}  "
                f"SNR={r.snr:+.3f}"
            )

        # Show sign distribution
        pos = sum(1 for r in all_results if r.kappa_hat > 0)
        neg = sum(1 for r in all_results if r.kappa_hat <= 0)
        print(f"\n  Sign distribution: {pos} positive, {neg} negative/null")
        print(f"  (For pure noise, expect ~50/50)")


if __name__ == "__main__":
    main()
