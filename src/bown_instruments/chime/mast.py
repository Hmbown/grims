"""MAST (Mikulski Archive for Space Telescopes) interface for JWST data.

Provides search, product listing, and download for JWST x1dints files
(per-integration extracted spectra) used in transit spectroscopy.

Reference: https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html
"""

from __future__ import annotations

import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


@dataclass
class Observation:
    """A single JWST observation from MAST."""

    obsid: str
    obs_id: str
    target: str
    instrument: str
    filters: str
    calib_level: int
    dataproduct_type: str
    t_min: float
    t_max: float
    t_exptime: float
    proposal_id: str


@dataclass
class Product:
    """A downloadable data product from MAST."""

    obs_id: str
    product_filename: str
    product_type: str
    product_subgroup: str
    calib_level: int
    size: int
    uri: str


def search_jwst(
    target: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float = 0.02,
    instrument: str | None = None,
    calib_level_min: int = 2,
    max_results: int = 200,
) -> list[Observation]:
    """Search MAST for JWST observations by target name or coordinates.

    Parameters
    ----------
    target : str, optional
        Target name (e.g., "WASP-39").
    ra, dec : float, optional
        Right ascension and declination in degrees.
    radius_deg : float
        Search radius in degrees.
    instrument : str, optional
        Instrument filter (partial match, e.g., "NIRSPEC").
    calib_level_min : int
        Minimum calibration level (2 = calibrated).
    max_results : int
        Maximum number of results.

    Returns
    -------
    list[Observation]
    """
    from astroquery.mast import Observations

    kwargs: dict[str, Any] = {
        "obs_collection": "JWST",
        "dataRights": "PUBLIC",
    }

    if target:
        kwargs["target_name"] = target
    elif ra is not None and dec is not None:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        kwargs["coordinates"] = SkyCoord(ra=ra, dec=dec, unit="deg")
        kwargs["radius"] = radius_deg * u.deg

    obs_table = Observations.query_criteria(**kwargs)

    if len(obs_table) == 0:
        return []

    # Filter by calib level
    mask = [int(r.get("calib_level", 0)) >= calib_level_min for r in obs_table]
    obs_table = obs_table[mask]

    # Filter by instrument (partial match)
    if instrument:
        mask = [instrument.upper() in str(r.get("instrument_name", "")).upper() for r in obs_table]
        obs_table = obs_table[mask]

    results = []
    for row in obs_table[:max_results]:
        results.append(
            Observation(
                obsid=str(row["obsid"]),
                obs_id=str(row.get("obs_id", "")),
                target=str(row.get("target_name", "")),
                instrument=str(row.get("instrument_name", "")),
                filters=str(row.get("filters", "")),
                calib_level=int(row.get("calib_level", 0)),
                dataproduct_type=str(row.get("dataproduct_type", "")),
                t_min=float(row.get("t_min", 0)),
                t_max=float(row.get("t_max", 0)),
                t_exptime=float(row.get("t_exptime", 0)),
                proposal_id=str(row.get("proposal_id", "")),
            )
        )

    return results


def list_products(
    observation: Observation,
    product_type: str = "SCIENCE",
    subgroup: str | None = None,
) -> list[Product]:
    """List data products for an observation.

    Parameters
    ----------
    observation : Observation
        The observation to list products for.
    product_type : str
        Product type filter (SCIENCE, PREVIEW, INFO, AUXILIARY).
    subgroup : str, optional
        Subgroup filter (X1DINTS, X1D, CAL, RATE, UNCAL).

    Returns
    -------
    list[Product]
    """
    from astroquery.mast import Observations

    products = Observations.get_product_list(observation.obsid)

    results = []
    for row in products:
        ptype = str(row.get("productType", ""))
        subgrp = str(row.get("productSubGroupDescription", ""))

        if product_type and ptype.upper() != product_type.upper():
            continue
        if subgroup and subgroup.upper() not in subgrp.upper():
            continue

        results.append(
            Product(
                obs_id=str(row.get("obs_id", "")),
                product_filename=str(row.get("productFilename", "")),
                product_type=ptype,
                product_subgroup=subgrp,
                calib_level=int(row.get("calib_level", 0)),
                size=int(row.get("size", 0)),
                uri=str(row.get("dataURI", "")),
            )
        )

    return results


def download_product(
    product: Product,
    download_dir: str | None = None,
) -> str:
    """Download a single product via MAST URI.

    Parameters
    ----------
    product : Product
        The product to download.
    download_dir : str, optional
        Download directory. Uses tempdir if None.

    Returns
    -------
    str : Local file path.
    """
    if download_dir is None:
        download_dir = tempfile.mkdtemp(prefix="chime_")

    uri = product.uri
    if uri.startswith("mast:"):
        url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
    elif uri.startswith("http"):
        url = uri
    else:
        raise ValueError(f"Unknown URI format: {uri}")

    local_dir = Path(download_dir) / product.obs_id
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / product.product_filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    print(f"  downloading {product.product_filename}...", end="", flush=True)
    urllib.request.urlretrieve(url, str(local_path))
    print(" done")
    return str(local_path)


def find_x1dints(
    target: str,
    instrument: str | None = "NIRSPEC",
    max_obs: int = 10,
) -> list[tuple[Observation, Product]]:
    """Find X1DINTS products for a target.

    X1DINTS are per-integration extracted 1D spectra — the primary input
    for transit spectroscopy analysis.

    Parameters
    ----------
    target : str
        Target name.
    instrument : str, optional
        Instrument filter.
    max_obs : int
        Maximum observations to search.

    Returns
    -------
    list[tuple[Observation, Product]]
        (observation, product) pairs sorted by calib level descending.
    """
    observations = search_jwst(target=target, instrument=instrument, calib_level_min=2)

    pairs = []
    for obs in observations[:max_obs]:
        try:
            products = list_products(obs, product_type="SCIENCE", subgroup="X1DINTS")
            for prod in products:
                if "x1dints" in prod.product_filename.lower():
                    pairs.append((obs, prod))
        except Exception:
            continue

    pairs.sort(key=lambda p: (p[1].calib_level, p[1].size), reverse=True)
    return pairs
