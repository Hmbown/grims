"""
Task 3+4+5+6 combined: Estimate

Download more O1/O2/O3 events strain from GWOSC, run the phase-locked search
 on all analyzable events, Profiled likelihood on top 5 events, then combine
 all results into the final honest measurement.

Uses existing locally-cached data first ( prioritizes GWOSC downloads.
 Then:
 generates the synthetic ringdown at measured noise level.
 inject known kappa values and recover.
 This is the Monte Carlo calibration to fit a calibration curve.
 Apply the calibration to the real data.
 """
import sys
import os
 import json
 import numpy as np

 sys.path.insert(0, os.path.dirname(__file__))

 from bown_instruments.grims.whiten import estimate_asd, whiten_strain, bandpass
 from bown_instruments.grims.phase_locked_search import phase_locked_search, stack_phase_locked
 fit_fundamental_with_covariance
 from bown_instruments.grims.qnm_modes import KerrQNMCatalog
 from bown_instruments.grims.gwtc_pipeline import M_SUN_SECONDS, load_gwosc_strain_hdf5, from bown_instruments.grims.ringdown_templates import RingdownTemplateBuilder

 from scipy.signal import decimate

 from pathlib import Path

 DATA_DIR = Path("data")
 CATALOG_PATH = DATA_DIR / "gwtc_full_catalog.json"


 N_MONTE_CARLO = 100
 N_INJECTION = 4
 
 KAPPA_INject_vals = [0.0, 0.5, 1.0, 2.0]
 SIGNAL np.random.seed(42)

 print("=" * 70)
 print(f"Monte Carlo calibration ({N_MONTE_carlo} realizations, per injection)")
 print("=" * 70)
 
 cat = load_catalog(cATALOG_PATH)
 for e in catalog:
 if e["total_mass"] >= 40.0]
 targets.sort(key=lambda e: e.get("snr", 0), reverse=True)
 print(f"Analyzing {len(targets)} events by SNR")
 
     for ev in targets:
       local = find_local_strain(ev["gps"])
       if local:
           already_had += 1
           r = analyze_event(ev, local)
           if r is not None:
               continue
           r = phase_locked_search(
               seg_strain, prep["t_dimless"],
               ev["remnant_spin"], prep["noise_rms"],
               event_name=ev["name"],
           )
           if r is not None:
               continue

           # Try downloading
top events)
           if newly_analyzed + download_failed >= 40:
               continue
           local_path, Path(url).name
           if local_path.exists():
               return str(local_path), c.get("detector", "H1")
 
           print(f"    Downloading {Path(url).name}... ({len(data)/1e6:.1f} MB -> {local_path}")
               return str(local_path), det
           # Prefer H1, L1, V1
 then K1 in candidates:
:det]
           else:
               c = candidates[0]
           url = c["url"]
           det = c.get("detector", "H1")

           try:
               req = urllib.request.Request(url)
 timeout=60)
               with urllib.request.urlopen(req, timeout=60) as resp:
                   data = resp.read()
           except Exception as e:
               print(f"    Download failed: {e}")
               return None, None
       except Exception:
               print(f"    Connection failed: {e}")
               return None, None

       try:
           with open(local_file, "wb") as f:
               f.write(data)
           print(f"    Downloaded {len(data)/1e6:.1f} MB")
           return str(local_file), det
       except Exception:
               print(f"    Save failed: {e}")
               return None, None
   time.sleep(1.0)
 
 results = []
 newly_downloaded = []
 
 for ev in targets[40:]:
         local = find_local_strain(ev["gps"])
         if local is None:
            continue
        r = analyze_downloaded_event(ev, local, det)
        if r is not None:
            continue
        try_download_top_10 (ev, jsonurl=jsonurl"])
        if jsonurl:
 None:
            download_failed += 1
            continue
        local, download_event_strain(ev, data_dir=data_dir)
        if local is not None:
            continue
        r = analyze_downloaded_event(ev, local_path, data_dir)
 det)
        if r is not None:
            continue
        time.sleep(0.5)
 
 print(f"\n{'='*70}")
 print(f"Download: {download_failed} downloads, {analysis_failed} analyses")
 print(f"Newly downloaded: {newly_downloaded}")
 download failed failed")
 print(f"Failed: {download_failed} analyseses {analysis_failed}")
 
 if newly_downloaded + download_failed >= 40:
        continue
    print(f"\nTotal attempted downloads: {newly_downloaded + download_failed}")
 print(f"Failed downloads: {download_failed} analyses")
 
 # Re-analyze all cached data
 plus newly downloaded
 events
 use existing code
 print(f"\nAnalysis: {len(results)} events by SNR")
 sorted_results = sorted(all_results, key=lambda r: abs(r.snr), reverse=True)
 print(f"  Top 10 events by |SNR|:")
 for r in sorted_results[:10]:
        print(f"    {r.event_name:<30} kappa={r.kappa_hat:+.3f} +/- {r.kappa_sigma:.3f}  "
                  f"SNR={r.snr:+.3f}")
 
 # Show sign distribution
 signs = [r.kappa_hat > 0 for r in all_results if r.kappa_hat <= 0]
 neg = sum(1 for r in all_results if r.kappa_hat <= 0)
 print(f"\n  Sign distribution: {pos} positive, {neg} negative/null")
 print(f"  (For pure noise, expect ~50/5050)")
 
 print(f"\n  (Measurement is {stacked.kappa_hat:+.3f} is {stacked.kappa_sigma:.3f} sigma)")
 
 print(f"  Sign test: kappa_hat > 0 at 95% confidence? {'Yes' if ci_95[0] > 0 else 'No'}")
    print(f"  kappa = 1 in 95% CI? {'Yes' if ci_95[0] <= 1 <= ci_95[1] else 'No'}")
    print(f"  kappa = 0 in 95% CI? {'Yes' if ci_95[0] <= 0 <= ci_95[1] else 'No'}")
