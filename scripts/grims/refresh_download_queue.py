#!/usr/bin/env python3
"""
Regenerate download_queue.json from the full catalog.

Queries GWOSC for strain file URLs for each ringdown-relevant event
and writes a download manifest with direct URLs. This enables
reproducible data fetching without re-querying the API each time.

Usage:
    python scripts/refresh_download_queue.py              # dry-run
    python scripts/refresh_download_queue.py --write      # overwrite queue
    python scripts/refresh_download_queue.py --min-mass 60  # custom threshold
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CATALOG_PATH = RESULTS_DIR / "gwtc_full_catalog.json"
QUEUE_PATH = RESULTS_DIR / "download_queue.json"


def fetch_strain_urls(jsonurl: str) -> list:
    """Fetch strain file metadata from GWOSC event API."""
    try:
        req = urllib.request.Request(jsonurl, headers={"User-Agent": "GRIM-S/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.load(response)
        event_data = next(iter(payload["events"].values()))
        return event_data.get("strain", [])
    except Exception:
        return []


def main():
    write_mode = "--write" in sys.argv
    min_mass = 30.0
    if "--min-mass" in sys.argv:
        idx = sys.argv.index("--min-mass")
        min_mass = float(sys.argv[idx + 1])

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    targets = [e for e in catalog if e["total_mass"] >= min_mass and e.get("jsonurl")]
    targets.sort(key=lambda e: e.get("snr", 0), reverse=True)

    print(f"Catalog: {len(catalog)} total, {len(targets)} with M>={min_mass}")

    queue = []
    failed = []

    for i, event in enumerate(targets):
        name = event["name"]
        jsonurl = event["jsonurl"]
        print(f"[{i+1}/{len(targets)}] {name:<30} ", end="", flush=True)

        strain_entries = fetch_strain_urls(jsonurl)
        if not strain_entries:
            print("API_FAIL")
            failed.append(name)
            time.sleep(0.5)
            continue

        # Select 4096Hz HDF5 files for each detector
        hdf5_4k = [
            s for s in strain_entries
            if s.get("format") == "hdf5" and s.get("sampling_rate") == 4096
        ]

        dets_found = []
        for s in hdf5_4k:
            det = s.get("detector", "?")
            queue.append({
                "event": name,
                "snr": event["snr"],
                "mass": event["total_mass"],
                "detector": det,
                "url": s["url"],
                "duration": s.get("duration", 0),
                "sr": s.get("sampling_rate", 4096),
            })
            dets_found.append(det)

        print(f"OK: {','.join(sorted(set(dets_found)))}")
        time.sleep(0.3)

    print(f"\nQueue: {len(queue)} files for {len(targets) - len(failed)} events")
    if failed:
        print(f"Failed: {len(failed)} events: {failed}")

    if write_mode:
        with open(QUEUE_PATH, "w") as f:
            json.dump(queue, f, indent=2)
            f.write("\n")
        print(f"Wrote {QUEUE_PATH}")
    else:
        print("Dry run. Use --write to update the queue file.")


if __name__ == "__main__":
    main()
