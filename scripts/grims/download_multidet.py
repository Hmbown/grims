#!/usr/bin/env python3
"""
Download L1 and V1 strain files for events that currently only have H1.

Queries each event's GWOSC JSON API to find available detectors,
then downloads 4KHz 32s HDF5 files for L1 and V1 where available.
"""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


def fetch_strain_urls(jsonurl: str) -> list:
    """Fetch strain file metadata from GWOSC event API."""
    try:
        req = urllib.request.Request(jsonurl, headers={"User-Agent": "GRIM-S/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.load(response)
        event_data = next(iter(payload["events"].values()))
        return event_data.get("strain", [])
    except Exception as e:
        return []


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to dest."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GRIM-S/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            with dest.open("wb") as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return False


def main():
    with open(DATA_DIR / "gwtc_full_catalog.json") as f:
        catalog = json.load(f)

    targets = [e for e in catalog if e["total_mass"] >= 30.0]
    targets.sort(key=lambda e: e.get("snr", 0), reverse=True)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    total_already = 0

    for i, event in enumerate(targets):
        name = event["name"]
        gps = int(event.get("gps", 0))
        jsonurl = event.get("jsonurl", "")

        if gps == 0 or not jsonurl:
            continue

        # Check what we already have
        local_dets = find_local_detectors(gps, DATA_DIR)

        # Which detectors do we still need?
        need = {"L1", "V1"} - local_dets
        if not need:
            total_already += 1
            continue

        # Fetch available strain files from GWOSC
        print(f"[{i+1}/{len(targets)}] {name:<30} have={sorted(local_dets)}  need={sorted(need)}  ", end="", flush=True)

        strain_entries = fetch_strain_urls(jsonurl)
        if not strain_entries:
            print("API_FAIL")
            total_failed += 1
            time.sleep(0.5)
            continue

        downloaded_this = []
        for det in sorted(need):
            # Find best matching file: prefer 4KHz, 32s, hdf5
            candidates = [
                s for s in strain_entries
                if s.get("detector") == det
                and s.get("format") == "hdf5"
            ]

            # Prefer 4096 Hz, 32s
            best = sorted(
                candidates,
                key=lambda s: (
                    abs(s.get("sampling_rate", 4096) - 4096),
                    abs(s.get("duration", 32) - 32),
                ),
            )

            if not best:
                continue

            entry = best[0]
            url = entry["url"]
            filename = Path(url).name
            dest = DATA_DIR / filename

            if dest.exists():
                downloaded_this.append(det)
                continue

            ok = download_file(url, dest)
            if ok:
                downloaded_this.append(det)
                total_downloaded += 1
            else:
                total_failed += 1

        if downloaded_this:
            print(f"OK: +{','.join(downloaded_this)}")
        else:
            print("no_files_available")
            total_skipped += 1

        # Be polite to GWOSC
        time.sleep(0.3)

    print(f"\nDone: {total_downloaded} downloaded, {total_already} already complete, "
          f"{total_skipped} no files available, {total_failed} failed")


if __name__ == "__main__":
    main()
