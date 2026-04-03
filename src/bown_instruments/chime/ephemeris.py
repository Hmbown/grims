"""Known transit ephemerides for JWST exoplanet targets.

All ephemerides sourced from the NASA Exoplanet Archive unless otherwise noted.
Keys: period (days), mid-transit time (BJD_TDB), transit duration (hours),
planet-to-star radius ratio, expected transit depth (ppm).

References:
    NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu
"""

from __future__ import annotations


EPHEMERIDES: dict[str, dict] = {
    "WASP-39": {
        "period_days": 4.055259,
        "t0_bjd": 2455342.9168,
        "duration_hours": 2.8056,
        "rp_rs": 0.1460,
        "expected_depth_ppm": 21316,
        "ref": "Faedi+ 2011; Rustamkulov+ 2023",
    },
    "WASP-39b": {
        "period_days": 4.055259,
        "t0_bjd": 2455342.9168,
        "duration_hours": 2.8056,
        "rp_rs": 0.1460,
        "expected_depth_ppm": 21316,
        "ref": "Faedi+ 2011; Rustamkulov+ 2023",
    },
    "TRAPPIST-1": {
        "period_days": 1.51087081,
        "t0_bjd": 2457322.51736,
        "duration_hours": 0.608,
        "rp_rs": 0.08590,
        "expected_depth_ppm": 7379,
        "ref": "Agol+ 2021 (planet b)",
    },
    "TRAPPIST-1b": {
        "period_days": 1.51087081,
        "t0_bjd": 2457322.51736,
        "duration_hours": 0.608,
        "rp_rs": 0.08590,
        "expected_depth_ppm": 7379,
        "ref": "Agol+ 2021; Lim+ 2023",
    },
    "TRAPPIST-1c": {
        "period_days": 2.42182330,
        "t0_bjd": 2457282.80728,
        "duration_hours": 0.710,
        "rp_rs": 0.08440,
        "expected_depth_ppm": 7123,
        "ref": "Agol+ 2021; Zieba+ 2023",
    },
    "TRAPPIST-1d": {
        "period_days": 4.049959,
        "t0_bjd": 2457294.7730,
        "duration_hours": 0.782,
        "rp_rs": 0.06080,
        "expected_depth_ppm": 3697,
        "ref": "Agol+ 2021",
    },
    "TRAPPIST-1e": {
        "period_days": 6.099568,
        "t0_bjd": 2457305.4490,
        "duration_hours": 0.857,
        "rp_rs": 0.06790,
        "expected_depth_ppm": 4610,
        "ref": "Agol+ 2021",
    },
    "TRAPPIST-1f": {
        "period_days": 9.206576,
        "t0_bjd": 2457311.9280,
        "duration_hours": 0.968,
        "rp_rs": 0.07360,
        "expected_depth_ppm": 5417,
        "ref": "Agol+ 2021",
    },
    "TRAPPIST-1g": {
        "period_days": 12.352877,
        "t0_bjd": 2457320.7080,
        "duration_hours": 1.080,
        "rp_rs": 0.07780,
        "expected_depth_ppm": 6053,
        "ref": "Agol+ 2021",
    },
    "WASP-107": {
        "period_days": 5.721490,
        "t0_bjd": 2456514.4106,
        "duration_hours": 2.75,
        "rp_rs": 0.1447,
        "expected_depth_ppm": 20938,
        "ref": "Anderson+ 2017; Dyrek+ 2024",
    },
    "WASP-107b": {
        "period_days": 5.721490,
        "t0_bjd": 2456514.4106,
        "duration_hours": 2.75,
        "rp_rs": 0.1447,
        "expected_depth_ppm": 20938,
        "ref": "Anderson+ 2017; Dyrek+ 2024",
    },
    "HD-189733": {
        "period_days": 2.21857567,
        "t0_bjd": 2454279.436714,
        "duration_hours": 1.827,
        "rp_rs": 0.15667,
        "expected_depth_ppm": 24545,
        "ref": "Agol+ 2010; Fu+ 2024",
    },
    "HD-189733b": {
        "period_days": 2.21857567,
        "t0_bjd": 2454279.436714,
        "duration_hours": 1.827,
        "rp_rs": 0.15667,
        "expected_depth_ppm": 24545,
        "ref": "Agol+ 2010; Fu+ 2024",
    },
    "HD-209458": {
        "period_days": 3.52474859,
        "t0_bjd": 2452826.62856,
        "duration_hours": 3.062,
        "rp_rs": 0.12070,
        "expected_depth_ppm": 14569,
        "ref": "Knutson+ 2007",
    },
    "HD-209458b": {
        "period_days": 3.52474859,
        "t0_bjd": 2452826.62856,
        "duration_hours": 3.062,
        "rp_rs": 0.12070,
        "expected_depth_ppm": 14569,
        "ref": "Knutson+ 2007",
    },
    "GJ-3470": {
        "period_days": 3.336652,
        "t0_bjd": 2456043.8930,
        "duration_hours": 1.149,
        "rp_rs": 0.07570,
        "expected_depth_ppm": 5730,
        "ref": "Dragomir+ 2019",
    },
    "GJ-3470b": {
        "period_days": 3.336652,
        "t0_bjd": 2456043.8930,
        "duration_hours": 1.149,
        "rp_rs": 0.07570,
        "expected_depth_ppm": 5730,
        "ref": "Dragomir+ 2019",
    },
    "LHS-1140": {
        "period_days": 24.73723,
        "t0_bjd": 2458386.2494,
        "duration_hours": 1.981,
        "rp_rs": 0.07090,
        "expected_depth_ppm": 5027,
        "ref": "Cadieux+ 2024 (planet c)",
    },
    "LHS-1140b": {
        "period_days": 24.73723,
        "t0_bjd": 2458386.2494,
        "duration_hours": 1.981,
        "rp_rs": 0.07090,
        "expected_depth_ppm": 5027,
        "ref": "Cadieux+ 2024",
    },
    "LHS-1140c": {
        "period_days": 3.777940,
        "t0_bjd": 2458373.2219,
        "duration_hours": 0.710,
        "rp_rs": 0.03700,
        "expected_depth_ppm": 1369,
        "ref": "Ment+ 2019",
    },
    "LP-890-9": {
        "period_days": 8.457,
        "t0_bjd": 2459614.489,
        "duration_hours": 1.42,
        "rp_rs": 0.0520,
        "expected_depth_ppm": 2704,
        "ref": "Delrez+ 2022 (planet c)",
    },
    "LP-890-9c": {
        "period_days": 8.457,
        "t0_bjd": 2459614.489,
        "duration_hours": 1.42,
        "rp_rs": 0.0520,
        "expected_depth_ppm": 2704,
        "ref": "Delrez+ 2022",
    },
    "55-CNC": {
        "period_days": 0.73654737,
        "t0_bjd": 2453290.8470,
        "duration_hours": 1.558,
        "rp_rs": 0.01920,
        "expected_depth_ppm": 369,
        "ref": "Winn+ 2011",
    },
    "55-CNCe": {
        "period_days": 0.73654737,
        "t0_bjd": 2453290.8470,
        "duration_hours": 1.558,
        "rp_rs": 0.01920,
        "expected_depth_ppm": 369,
        "ref": "Winn+ 2011",
    },
    "HAT-P-14": {
        "period_days": 4.627663,
        "t0_bjd": 2454875.2598,
        "duration_hours": 3.106,
        "rp_rs": 0.10220,
        "expected_depth_ppm": 10445,
        "ref": "Torres+ 2010",
    },
    "HAT-P-14b": {
        "period_days": 4.627663,
        "t0_bjd": 2454875.2598,
        "duration_hours": 3.106,
        "rp_rs": 0.10220,
        "expected_depth_ppm": 10445,
        "ref": "Torres+ 2010",
    },
    "WASP-96": {
        "period_days": 3.425259,
        "t0_bjd": 2457193.6861,
        "duration_hours": 2.736,
        "rp_rs": 0.11660,
        "expected_depth_ppm": 13596,
        "ref": "Nikolov+ 2018",
    },
    "WASP-96b": {
        "period_days": 3.425259,
        "t0_bjd": 2457193.6861,
        "duration_hours": 2.736,
        "rp_rs": 0.11660,
        "expected_depth_ppm": 13596,
        "ref": "Nikolov+ 2018",
    },
    "WASP-33": {
        "period_days": 1.2198696,
        "t0_bjd": 2455131.3241,
        "duration_hours": 3.598,
        "rp_rs": 0.10230,
        "expected_depth_ppm": 10465,
        "ref": "von Essen+ 2020",
    },
    "WASP-33b": {
        "period_days": 1.2198696,
        "t0_bjd": 2455131.3241,
        "duration_hours": 3.598,
        "rp_rs": 0.10230,
        "expected_depth_ppm": 10465,
        "ref": "von Essen+ 2020",
    },
    "WASP-121": {
        "period_days": 1.2749255,
        "t0_bjd": 2457043.8249,
        "duration_hours": 2.874,
        "rp_rs": 0.12240,
        "expected_depth_ppm": 14982,
        "ref": "Delrez+ 2016",
    },
    "WASP-121b": {
        "period_days": 1.2749255,
        "t0_bjd": 2457043.8249,
        "duration_hours": 2.874,
        "rp_rs": 0.12240,
        "expected_depth_ppm": 14982,
        "ref": "Delrez+ 2016",
    },
    "WASP-189": {
        "period_days": 2.7240330,
        "t0_bjd": 2458359.2321,
        "duration_hours": 3.326,
        "rp_rs": 0.06570,
        "expected_depth_ppm": 4316,
        "ref": "Lendl+ 2020",
    },
    "WASP-189b": {
        "period_days": 2.7240330,
        "t0_bjd": 2458359.2321,
        "duration_hours": 3.326,
        "rp_rs": 0.06570,
        "expected_depth_ppm": 4316,
        "ref": "Lendl+ 2020",
    },
}


def list_targets() -> list[str]:
    """Return sorted list of available target names."""
    return sorted(EPHEMERIDES.keys())


def get_ephemeris(target: str) -> dict:
    """Look up ephemeris for a target with fuzzy matching.

    Raises KeyError if no match found.
    """
    # Exact match
    if target in EPHEMERIDES:
        return EPHEMERIDES[target]

    # Case-insensitive, strip hyphens/spaces
    norm = target.upper().replace("-", "").replace(" ", "")
    for key in EPHEMERIDES:
        if norm == key.upper().replace("-", "").replace(" ", ""):
            return EPHEMERIDES[key]

    # Partial match
    for key in EPHEMERIDES:
        if norm in key.upper().replace("-", "").replace(" ", ""):
            return EPHEMERIDES[key]

    raise KeyError(f"No ephemeris for '{target}'. Known targets: {list_targets()}")
