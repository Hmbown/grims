#!/usr/bin/env python3
"""
Refresh gwtc_full_catalog.json from the GWOSC event API.

Queries all configured GWTC catalogs and builds a unified catalog
with the fields needed by the GRIM-S pipeline:
  name, version_name, total_mass, remnant_mass, remnant_spin,
  mass_ratio, snr, distance, gps, jsonurl

Usage:
    python scripts/refresh_catalog.py              # dry-run: print diff
    python scripts/refresh_catalog.py --write      # overwrite catalog file

Notes:
    - Remnant mass and spin are estimated from component masses and chi_eff
      using NR-calibrated fits (Jimenez-Forteza+ 2017) when not provided
      directly by GWOSC.
    - Only events with total_mass_source > 0 and snr > 0 are included.
    - The script is idempotent: running it twice produces the same output.
"""

import json
import sys
import urllib.request
from pathlib import Path

CATALOG_SOURCES = [
    "https://gwosc.org/eventapi/json/GWTC-1-confident/",
    "https://gwosc.org/eventapi/json/GWTC-2.1-confident/",
    "https://gwosc.org/eventapi/json/GWTC-3-confident/",
    "https://gwosc.org/eventapi/json/O3_Discovery_Papers/",
    "https://gwosc.org/eventapi/json/GWTC-4.0/",
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CATALOG_PATH = RESULTS_DIR / "gwtc_full_catalog.json"


def estimate_remnant(m1: float, m2: float, chi_eff: float) -> tuple[float, float]:
    """Estimate remnant mass and spin from component masses and chi_eff.

    Uses the Jimenez-Forteza+ 2017 / Hofmann+ 2016 phenomenological fits.
    These are approximate — good enough for selecting ringdown candidates,
    not for precision parameter estimation.
    """
    M = m1 + m2
    eta = m1 * m2 / M**2  # symmetric mass ratio
    # Radiated energy fraction (Reisswig+ 2009 fit)
    E_rad_frac = 0.0559745 * eta + 0.580951 * eta**2 - 0.960 * eta**3
    # Spin correction
    E_rad_frac *= 1.0 + chi_eff * (-0.0667 + 0.436 * eta)
    remnant_mass = M * (1.0 - E_rad_frac)

    # Final spin (Barausse-Rezzolla 2009 fit)
    a_orb = 2.0 * 3**0.5 * eta - 3.516 * eta**2 + 2.548 * eta**3
    a_spin = chi_eff * (0.6865 + 0.1015 * eta)
    remnant_spin = min(abs(a_orb + a_spin), 0.998)

    return round(remnant_mass, 1), round(remnant_spin, 3)


def fetch_catalog(url: str) -> dict:
    """Fetch a GWOSC catalog and return {commonName: event_dict}."""
    req = urllib.request.Request(url, headers={"User-Agent": "GRIM-S/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data.get("events", {})


def build_entry(version_name: str, ev: dict) -> dict | None:
    """Convert a GWOSC event record to a GRIM-S catalog entry."""
    m1 = ev.get("mass_1_source") or 0
    m2 = ev.get("mass_2_source") or 0
    total_mass = ev.get("total_mass_source") or (m1 + m2)
    snr = ev.get("network_matched_filter_snr") or 0

    if total_mass <= 0 or snr <= 0:
        return None

    chi_eff = ev.get("chi_eff") or 0
    remnant_mass, remnant_spin = estimate_remnant(m1, m2, chi_eff)
    mass_ratio = min(m1, m2) / max(m1, m2) if m1 > 0 and m2 > 0 else 0

    return {
        "name": ev.get("commonName", version_name.split("-v")[0]),
        "version_name": version_name,
        "total_mass": round(total_mass, 1),
        "remnant_mass": round(remnant_mass, 1),
        "remnant_spin": round(remnant_spin, 3),
        "mass_ratio": round(mass_ratio, 4) if mass_ratio else 0,
        "snr": round(snr, 1),
        "distance": round(ev.get("luminosity_distance") or 0, 1),
        "gps": ev.get("GPS") or 0,
        "jsonurl": ev.get("jsonurl") or "",
    }


def main():
    write_mode = "--write" in sys.argv

    # Fetch all catalogs
    all_events = {}  # keyed by commonName, latest version wins
    for source_url in CATALOG_SOURCES:
        catalog_name = source_url.rstrip("/").split("/")[-1]
        print(f"Fetching {catalog_name}...", end=" ", flush=True)
        try:
            events = fetch_catalog(source_url)
            print(f"{len(events)} events")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        for vname, ev in events.items():
            entry = build_entry(vname, ev)
            if entry is None:
                continue
            name = entry["name"]
            # Keep the entry from the most recent catalog (later in CATALOG_SOURCES)
            # or the one with higher version number
            if name in all_events:
                existing = all_events[name]
                if entry["snr"] > existing["snr"]:
                    all_events[name] = entry
            else:
                all_events[name] = entry

    # Sort by SNR descending
    catalog = sorted(all_events.values(), key=lambda e: e["snr"], reverse=True)

    # Compare with existing
    try:
        with open(CATALOG_PATH) as f:
            old = json.load(f)
        old_names = set(e["name"] for e in old)
    except FileNotFoundError:
        old_names = set()
        old = []

    new_names = set(e["name"] for e in catalog)
    added = new_names - old_names
    removed = old_names - new_names

    print(f"\nCatalog: {len(catalog)} events (was {len(old)})")
    print(f"  Added: {len(added)}")
    print(f"  Removed: {len(removed)}")
    if added:
        print(f"  New events: {sorted(added)[:10]}{'...' if len(added) > 10 else ''}")

    ringdown = [e for e in catalog if e["total_mass"] >= 30]
    print(f"  Ringdown-relevant (M>=30): {len(ringdown)}")

    if write_mode:
        with open(CATALOG_PATH, "w") as f:
            json.dump(catalog, f, indent=2)
            f.write("\n")
        print(f"\nWrote {CATALOG_PATH}")
    else:
        print("\nDry run. Use --write to update the catalog file.")


if __name__ == "__main__":
    main()
