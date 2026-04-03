#!/usr/bin/env python3
"""
Check local data status for the GRIM-S pipeline.

Reports which strain files are present, missing, and how to fetch them.
This is the instrument Bown would insist on: a status panel that tells
you what you have and what you don't, before you start measuring.

Usage:
    python scripts/check_data.py                   # summary
    python scripts/check_data.py --detail          # per-event breakdown
    python scripts/check_data.py --ringdown-only   # only M>=30 events
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CATALOG_PATH = RESULTS_DIR / "gwtc_full_catalog.json"


def find_local_detectors(gps: int, data_dir: Path) -> set:
    """Find which detectors already have local strain files for this GPS time."""
    dets = set()
    for f in data_dir.glob("*.hdf5"):
        parts = f.stem.split("-")
        if len(parts) >= 3:
            try:
                file_gps = int(parts[-2])
                file_dur = int(parts[-1])
                if file_gps <= gps <= file_gps + file_dur:
                    det = parts[1][:2]
                    dets.add(det)
            except ValueError:
                continue
    return dets


def main():
    detail = "--detail" in sys.argv
    ringdown_only = "--ringdown-only" in sys.argv

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    if ringdown_only:
        catalog = [e for e in catalog if e["total_mass"] >= 30]

    catalog.sort(key=lambda e: e.get("snr", 0), reverse=True)

    total_events = len(catalog)
    have_h1 = 0
    have_l1 = 0
    have_v1 = 0
    have_any = 0
    have_multi = 0
    missing_all = 0

    # Track O3 vs O4
    o4_events = 0
    o4_have_any = 0
    o3_events = 0
    o3_have_any = 0

    # Storage stats
    local_files = list(DATA_DIR.glob("*.hdf5"))
    total_size_mb = sum(f.stat().st_size for f in local_files) / (1024 * 1024)

    if detail:
        print(f"{'Event':<30} {'M_tot':>5} {'SNR':>5} {'Cat':>8} {'Local Dets':>12}")
        print("-" * 70)

    for ev in catalog:
        gps = int(ev.get("gps", 0))
        if gps == 0:
            continue

        dets = find_local_detectors(gps, DATA_DIR)
        is_o4 = "GWTC-4" in ev.get("jsonurl", "")

        if is_o4:
            o4_events += 1
        else:
            o3_events += 1

        if "H1" in dets:
            have_h1 += 1
        if "L1" in dets:
            have_l1 += 1
        if "V1" in dets:
            have_v1 += 1
        if dets:
            have_any += 1
            if is_o4:
                o4_have_any += 1
            else:
                o3_have_any += 1
        if len(dets) >= 2:
            have_multi += 1
        if not dets:
            missing_all += 1

        if detail:
            cat_label = "O4" if is_o4 else "O1-O3"
            det_str = ",".join(sorted(dets)) if dets else "MISSING"
            print(f"{ev['name']:<30} {ev['total_mass']:>5.1f} {ev['snr']:>5.1f} {cat_label:>8} {det_str:>12}")

    print()
    print("=" * 50)
    print("GRIM-S Data Status Report")
    print("=" * 50)
    scope = "ringdown-relevant (M>=30)" if ringdown_only else "all"
    print(f"Catalog: {total_events} {scope} events")
    print(f"  O1-O3: {o3_events} events ({o3_have_any} with local data)")
    print(f"  O4:    {o4_events} events ({o4_have_any} with local data)")
    print()
    print(f"Local strain files: {len(local_files)}")
    print(f"Total local storage: {total_size_mb:.1f} MB")
    print()
    print(f"Events with H1: {have_h1}")
    print(f"Events with L1: {have_l1}")
    print(f"Events with V1: {have_v1}")
    print(f"Events with any detector: {have_any}")
    print(f"Events with 2+ detectors: {have_multi}")
    print(f"Events with NO local data: {missing_all}")
    print()

    # Storage estimates for missing data
    o3_missing = o3_events - o3_have_any
    o4_missing = o4_events - o4_have_any
    # O3: ~250KB per detector-file, ~2 detectors avg
    # O4: ~64MB per detector-file, ~2 detectors (H1+L1, no V1)
    o3_est_mb = o3_missing * 2 * 0.25
    o4_est_mb = o4_missing * 2 * 64
    print("Estimated download to complete:")
    print(f"  O1-O3 missing: {o3_missing} events × ~0.5 MB ≈ {o3_est_mb:.0f} MB")
    print(f"  O4 missing:    {o4_missing} events × ~128 MB ≈ {o4_est_mb/1024:.1f} GB")
    print(f"  Total:         ~{(o3_est_mb + o4_est_mb)/1024:.1f} GB")
    print()
    print("To fetch missing data:")
    print("  python scripts/download_multidet.py")
    print()
    print("To refresh catalog from GWOSC:")
    print("  python scripts/refresh_catalog.py --write")


if __name__ == "__main__":
    main()
