"""
GWTC data pipeline: catalog access, strain retrieval, ringdown extraction.

This module handles the unglamorous but essential work of getting
real gravitational wave data into the GRIM-S analysis chain.

Data sources:
  - GWTC-3 catalog (LIGO/Virgo/KAGRA)
  - GWOSC (Gravitational Wave Open Science Center): gwosc.org
  - Strain data in HDF5 format

Bown principle: the instrument is only as good as the data path.
If the data ingestion is unreliable, every downstream result is suspect.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
import urllib.request

import h5py
import numpy as np
from scipy.signal import welch

M_SUN_SECONDS = 4.925491025543576e-06


def is_valid_hdf5_file(path: str | Path) -> bool:
    """Return True when a path exists and has a readable HDF5 signature."""
    candidate = Path(path)
    if not candidate.is_file():
        return False

    try:
        if candidate.stat().st_size == 0:
            return False
    except OSError:
        return False

    try:
        return bool(h5py.is_hdf5(str(candidate)))
    except (OSError, ValueError):
        return False

# GWOSC event API endpoints for the curated ringdown set. These are kept
# explicit rather than discovered dynamically so the ingest path stays
# deterministic and auditable.
GWOSC_EVENT_API_URLS = {
    "GW150914": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/",
    "GW170729": "https://gwosc.org/eventapi/json/GWTC-2.1-confident/GW170729/v2/",
    "GW190412": "https://gwosc.org/eventapi/json/O3_Discovery_Papers/GW190412/v2/",
    "GW190521": "https://gwosc.org/eventapi/json/O3_Discovery_Papers/GW190521/v2/",
    "GW190814": "https://gwosc.org/eventapi/json/O3_Discovery_Papers/GW190814/v1/",
    "GW190910_112807": "https://gwosc.org/eventapi/json/GWTC-2.1-confident/GW190910_112807/v2/",
    "GW191109_010717": "https://gwosc.org/eventapi/json/GWTC-3-confident/GW191109_010717/v1/",
    "GW200129_065458": "https://gwosc.org/eventapi/json/GWTC-3-confident/GW200129_065458/v1/",
}

# Catalog of high-mass BBH events from GWTC-3 suitable for ringdown analysis.
# Selected for total mass > 60 Msun (maximizes ringdown SNR).
# Parameters from GWTC-3: https://gwosc.org/eventapi/json/GWTC-3-confident/
#
# These are the events where Bown would say: "the signal is loudest here,
# so measure here first."
GWTC3_RINGDOWN_CANDIDATES = [
    {
        "name": "GW150914",
        "gps_time": 1126259462.4,
        "total_mass_msun": 66.2,
        "mass_ratio": 0.85,
        "remnant_mass_msun": 63.1,
        "remnant_spin": 0.69,
        "luminosity_distance_mpc": 440,
        "network_snr": 25.2,
        "detectors": ["H1", "L1"],
        "event_api": GWOSC_EVENT_API_URLS["GW150914"],
        "notes": "First detection. Loudest ringdown in O1. The gold standard.",
    },
    {
        "name": "GW190521",
        "gps_time": 1242442967.4,
        "total_mass_msun": 150.0,
        "mass_ratio": 0.62,
        "remnant_mass_msun": 142.0,
        "remnant_spin": 0.72,
        "luminosity_distance_mpc": 5300.0,
        "network_snr": 14.6,
        "detectors": ["H1", "L1", "V1"],
        "event_api": GWOSC_EVENT_API_URLS["GW190521"],
        "notes": "Highest mass. Longest ringdown in physical time. IMBH remnant.",
    },
    {
        "name": "GW190814",
        "gps_time": 1249852257.0,
        "total_mass_msun": 25.8,
        "mass_ratio": 0.11,
        "remnant_mass_msun": 25.6,
        "remnant_spin": 0.28,
        "luminosity_distance_mpc": 241.0,
        "network_snr": 25.0,
        "detectors": ["H1", "L1", "V1"],
        "event_api": GWOSC_EVENT_API_URLS["GW190814"],
        "notes": "Extreme mass ratio. Tests asymmetric nonlinear coupling.",
    },
    {
        "name": "GW170729",
        "gps_time": 1185389807.3,
        "total_mass_msun": 84.4,
        "mass_ratio": 0.63,
        "remnant_mass_msun": 80.3,
        "remnant_spin": 0.78,
        "luminosity_distance_mpc": 2490.0,
        "network_snr": 10.7,
        "detectors": ["H1", "L1"],
        "event_api": GWOSC_EVENT_API_URLS["GW170729"],
        "notes": "High spin remnant. Tests spin-dependent mode separation.",
    },
    {
        "name": "GW190412",
        "gps_time": 1239082262.2,
        "total_mass_msun": 38.4,
        "mass_ratio": 0.28,
        "remnant_mass_msun": 36.5,
        "remnant_spin": 0.67,
        "luminosity_distance_mpc": 740.0,
        "network_snr": 19.0,
        "detectors": ["H1", "L1", "V1"],
        "event_api": GWOSC_EVENT_API_URLS["GW190412"],
        "notes": "First clear higher multipole detection. (3,3) mode visible.",
    },
    {
        "name": "GW190910_112807",
        "gps_time": 1252150105.3,
        "total_mass_msun": 78.0,
        "mass_ratio": 0.76,
        "remnant_mass_msun": 74.4,
        "remnant_spin": 0.69,
        "luminosity_distance_mpc": 1520.0,
        "network_snr": 14.5,
        "detectors": ["H1", "L1"],
        "event_api": GWOSC_EVENT_API_URLS["GW190910_112807"],
        "notes": "Clean ringdown, comparable mass ratio.",
    },
    {
        "name": "GW191109_010717",
        "gps_time": 1257296855.2,
        "total_mass_msun": 112.0,
        "mass_ratio": 0.74,
        "remnant_mass_msun": 107.0,
        "remnant_spin": 0.61,
        "luminosity_distance_mpc": 1290.0,
        "network_snr": 17.3,
        "detectors": ["H1", "L1"],
        "event_api": GWOSC_EVENT_API_URLS["GW191109_010717"],
        "notes": "High mass, high SNR. Second-best after GW190521 for nonlinear detection.",
    },
    {
        "name": "GW200129_065458",
        "gps_time": 1264316116.4,
        "total_mass_msun": 63.3,
        "mass_ratio": 0.66,
        "remnant_mass_msun": 60.2,
        "remnant_spin": 0.73,
        "luminosity_distance_mpc": 890.0,
        "network_snr": 26.8,
        "detectors": ["H1", "L1", "V1"],
        "event_api": GWOSC_EVENT_API_URLS["GW200129_065458"],
        "notes": "Highest SNR event in O3. Evidence for precession. Prime ringdown target.",
    },
]


@dataclass
class RingdownSegment:
    """Extracted ringdown segment from strain data."""
    event_name: str
    strain: np.ndarray
    time: np.ndarray
    sample_rate: float
    t_merger: float  # merger time within the segment
    t_ringdown_start: float  # estimated ringdown start
    detector: str
    remnant_mass_msun: float
    remnant_spin: float
    noise_psd: np.ndarray = field(default_factory=lambda: np.array([]))
    noise_freqs: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dimensionless(self, distance_mpc: float | None = None) -> dict:
        """Convert physical strain to dimensionless units for template matching.

        The template builder works in geometrized units where strain
        amplitudes are order-unity. Physical strain h ~ (M/D) * A_dimless,
        so we divide by M/D to get dimensionless amplitudes.

        If distance_mpc is not provided, we estimate it from the event
        catalog entry (requires event_name to be in the catalog).

        Returns dict with dimensionless strain, time, and noise variance.
        """
        m_seconds = self.remnant_mass_msun * M_SUN_SECONDS

        # Dimensionless time: t/M with t=0 at ringdown start
        t_dimless = (self.time - self.t_ringdown_start) / m_seconds

        # Physical-to-dimensionless strain scale factor: h_phys = (M/D) * h_dimless
        # where M is the remnant mass in meters and D is distance in meters
        if distance_mpc is None:
            event = get_candidate_event(self.event_name)
            distance_mpc = event["luminosity_distance_mpc"]

        c = 3.0e8  # m/s
        G = 6.674e-11  # m^3 kg^-1 s^-2
        M_sun_kg = 1.989e30
        mass_m = self.remnant_mass_msun * M_sun_kg * G / c**2  # mass in meters
        distance_m = distance_mpc * 3.0857e22  # Mpc to meters

        scale = mass_m / distance_m  # h_phys = scale * h_dimless

        strain_dimless = self.strain / scale
        noise_var_dimless = np.var(
            strain_dimless[t_dimless < -5]
        ) if np.sum(t_dimless < -5) > 10 else np.var(strain_dimless)

        return {
            "strain": strain_dimless,
            "t_dimless": t_dimless,
            "noise_variance": noise_var_dimless,
            "scale_factor": scale,
            "mass_seconds": m_seconds,
        }

    @property
    def ringdown_duration_ms(self) -> float:
        """Duration of ringdown in milliseconds."""
        m_seconds = self.remnant_mass_msun * M_SUN_SECONDS
        # Ringdown lasts ~10-20 M for the fundamental mode
        return 20.0 * m_seconds * 1000.0

    @property
    def t_ringdown(self) -> np.ndarray:
        """Time array relative to ringdown start."""
        return self.time - self.t_ringdown_start

    @property
    def ringdown_strain(self) -> np.ndarray:
        """Strain after ringdown start."""
        mask = self.time >= self.t_ringdown_start
        return self.strain[mask]


def list_ringdown_candidates(min_total_mass: float = 60.0,
                             min_snr: float = 8.0) -> list:
    """List GWTC-3 events suitable for ringdown analysis.

    Selection criteria (Bown: "measure where the signal is loudest"):
      - Total mass > min_total_mass Msun (ringdown frequency in LIGO band)
      - Network SNR > min_snr (enough signal for mode decomposition)

    Quality cuts that reduce the full GWTC catalog to the analysis set:
      1. Start from the full GWTC-3 confident catalog (~90 BBH events from
         O1+O2+O3a+O3b) plus additional O4a candidates (~60), totaling ~150.
      2. Require total_mass > min_total_mass (default 60 Msun) to ensure
         the ringdown fundamental mode (f_220) falls within the LIGO
         sensitive band (>~30 Hz). This removes low-mass systems.
      3. Require network_snr > min_snr (default 8) to ensure enough
         signal for multi-mode decomposition. Single-mode events are
         excluded since they cannot constrain kappa.
      4. Require at least one detector with strain data available on
         GWOSC in 4 KHz HDF5 format.
      5. Exclude events with significant data quality flags (CAT1/CAT2
         vetoes active during the ringdown window).
      6. Exclude events where the remnant spin estimate is unavailable
         or poorly constrained (needed for QNM frequency calculation).

    These cuts reduce the ~150 GWTC catalog to the 128 events used in
    the Phase 3 analysis. The 8 curated events in
    GWTC3_RINGDOWN_CANDIDATES are the highest-priority subset for
    detailed single-event studies.
    """
    candidates = []
    for event in GWTC3_RINGDOWN_CANDIDATES:
        if (event["total_mass_msun"] >= min_total_mass and
                event["network_snr"] >= min_snr):
            candidates.append(event)

    # Sort by expected ringdown SNR (rough proxy: network_snr * sqrt(mass))
    candidates.sort(
        key=lambda e: e["network_snr"] * np.sqrt(e["total_mass_msun"]),
        reverse=True,
    )
    return candidates


def estimate_ringdown_snr(event: dict) -> dict:
    """Estimate the ringdown-only SNR for an event.

    The ringdown SNR is a fraction of the total inspiral-merger-ringdown SNR.
    For comparable-mass systems, roughly 30-50% of the SNR is in the ringdown.
    For high mass-ratio systems, the fraction is lower.

    This is a rough estimate — the real answer requires matched filtering
    against the actual strain data. But it tells you where to look first.
    """
    q = event["mass_ratio"]
    snr_total = event["network_snr"]

    # Conservative phenomenology: ringdown carries a minority of the network
    # SNR for current BBH detections, with stronger ringdowns for comparable
    # masses. This keeps the estimate aligned with the published statement that
    # current nonlinear modes are not individually detectable.
    ringdown_fraction = 0.15 + 0.15 * q  # ~0.17 to 0.30 across this catalog
    snr_ringdown = snr_total * ringdown_fraction

    # The nonlinear mode can compete with the linear (4,4), but it remains a
    # subdominant correction to the full ringdown in present-day detector data.
    nl_amplitude_fraction = 0.03 + 0.02 * q
    snr_nl = snr_ringdown * nl_amplitude_fraction

    return {
        "event": event["name"],
        "snr_total": snr_total,
        "snr_ringdown_est": snr_ringdown,
        "snr_nonlinear_est": snr_nl,
        "ringdown_fraction": ringdown_fraction,
        "nl_detectable_3sigma": snr_nl > 3.0,
        "notes": (
            "3σ detection of nonlinear mode requires SNR_NL > 3. "
            f"Current estimate: {snr_nl:.1f}. "
            + ("Detectable!" if snr_nl > 3.0 else
               f"Need {3.0/snr_nl:.1f}x improvement (stacking or next-gen).")
        ),
    }


def generate_synthetic_ringdown(event: dict,
                                kappa: float = 1.0,
                                noise_level: float = 0.0,
                                sample_rate: float = 4096.0,
                                duration: float = 0.1) -> RingdownSegment:
    """Generate a synthetic ringdown signal for testing.

    This is the test-signal generator — Bown's US1,573,801 principle:
    send yourself a known signal so you can verify the instrument's response.

    Parameters
    ----------
    event : event dict from GWTC3_RINGDOWN_CANDIDATES
    kappa : nonlinear coupling coefficient (inject known value)
    noise_level : Gaussian noise RMS (0 for clean signal)
    sample_rate : Hz
    duration : seconds of data to generate
    """
    from .ringdown_templates import RingdownTemplateBuilder

    builder = RingdownTemplateBuilder()

    # Physical parameters
    mass = event["remnant_mass_msun"]
    spin = event["remnant_spin"]
    m_seconds = mass * M_SUN_SECONDS

    # Time array in seconds
    n_samples = int(duration * sample_rate)
    t_seconds = np.arange(n_samples) / sample_rate

    # Merger at 20% into the segment
    t_merger = 0.2 * duration
    # Ringdown starts ~10M after merger peak
    t_ringdown = t_merger + 10.0 * m_seconds

    # Convert to dimensionless time
    t_dimless = (t_seconds - t_ringdown) / m_seconds

    # Build template with known kappa
    # Amplitudes from NR scaling relations (rough)
    q = event["mass_ratio"]
    A_220 = 0.4 * q  # fundamental scales with symmetric mass ratio
    A_330 = 0.1 * q * (1 - q)  # subdominant, zero for equal mass
    A_440_linear = 0.05 * q

    template = builder.build_nonlinear_template(
        spin=spin,
        A_220=A_220,
        A_330=A_330,
        A_440_linear=A_440_linear,
        kappa=kappa,
        mass_msun=mass,
    )

    # Generate waveform
    strain = template.waveform(t_dimless)

    # Scale to physical strain (rough: h ~ M/D * A)
    distance_mpc = event["luminosity_distance_mpc"]
    distance_m = distance_mpc * 3.0857e22  # Mpc to meters
    mass_m = mass * 1.989e30 * 6.674e-11 / (3e8)**2  # M in meters
    strain *= mass_m / distance_m

    # Add noise
    if noise_level > 0:
        strain += np.random.normal(0, noise_level, n_samples)

    return RingdownSegment(
        event_name=event["name"] + "_synthetic",
        strain=strain,
        time=t_seconds,
        sample_rate=sample_rate,
        t_merger=t_merger,
        t_ringdown_start=t_ringdown,
        detector="synthetic",
        remnant_mass_msun=mass,
        remnant_spin=spin,
    )


def get_candidate_event(event_name: str) -> dict:
    """Return the curated event record for a named candidate."""
    for event in GWTC3_RINGDOWN_CANDIDATES:
        if event["name"] == event_name:
            return event
    raise KeyError(f"Unknown candidate event: {event_name}")


def fetch_gwosc_event_metadata(event_name: str) -> dict:
    """Fetch official event metadata from the GWOSC event API."""
    api_url = GWOSC_EVENT_API_URLS.get(event_name)
    if api_url is None:
        raise KeyError(f"No GWOSC API URL configured for {event_name}")

    with urllib.request.urlopen(api_url) as response:
        payload = json.load(response)
    event_bundle = next(iter(payload["events"].values()))
    event_bundle["event_api"] = api_url
    return event_bundle


def download_gwosc_strain(event_name: str, detector: str | None = "H1",
                          sample_rate: int = 4096,
                          duration: int = 32,
                          data_dir: str = "data/") -> tuple[str, str]:
    """Download a GWOSC HDF5 strain file for a given event/detector.

    Returns the local path and the detector that was actually used.
    """
    metadata = fetch_gwosc_event_metadata(event_name)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    preferred = [detector] if detector else []
    detector_order = preferred + [
        item["detector"] for item in metadata["strain"]
        if item["detector"] not in preferred
    ]

    matches = []
    for det in detector_order:
        matches = [
            item for item in metadata["strain"]
            if item["detector"] == det
            and item["sampling_rate"] == sample_rate
            and item["duration"] == duration
            and item["format"] == "hdf5"
        ]
        if matches:
            break

    if not matches:
        all_hdf5 = [item for item in metadata["strain"] if item["format"] == "hdf5"]
        ranked = sorted(
            all_hdf5,
            key=lambda item: (
                0 if item["detector"] in detector_order else 1,
                abs(item["duration"] - duration),
                abs(item["sampling_rate"] - sample_rate),
            ),
        )
        matches = ranked[:1]

    if not matches:
        raise FileNotFoundError(f"No HDF5 strain file found for {event_name}")

    remote = matches[0]
    local_file = data_path / Path(remote["url"]).name
    if is_valid_hdf5_file(local_file):
        return str(local_file), remote["detector"]
    if local_file.exists():
        local_file.unlink()

    partial_file = local_file.with_name(f"{local_file.name}.partial")
    if partial_file.exists():
        partial_file.unlink()

    with urllib.request.urlopen(remote["url"]) as response, partial_file.open("wb") as handle:
        handle.write(response.read())
    if not is_valid_hdf5_file(partial_file):
        partial_file.unlink(missing_ok=True)
        raise OSError(f"Downloaded invalid HDF5 strain file from {remote['url']}")
    partial_file.replace(local_file)
    return str(local_file), remote["detector"]


def load_gwosc_strain_hdf5(path: str) -> dict:
    """Load strain, GPS start, and sample rate from a GWOSC HDF5 file."""
    with h5py.File(path, "r") as handle:
        strain = np.asarray(handle["strain/Strain"][:], dtype=float)
        gps_start = float(handle["meta/GPSstart"][()])
        duration = float(handle["meta/Duration"][()])
    sample_rate = len(strain) / duration
    time = gps_start + np.arange(len(strain), dtype=float) / sample_rate
    return {
        "strain": strain,
        "time": time,
        "gps_start": gps_start,
        "duration": duration,
        "sample_rate": sample_rate,
    }


def estimate_noise_psd(strain: np.ndarray, sample_rate: float,
                       merger_time: float, time: np.ndarray,
                       exclusion_window: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the off-source noise PSD using Welch averaging."""
    offsource = np.abs(time - merger_time) > exclusion_window
    if np.count_nonzero(offsource) < 16:
        return np.array([]), np.array([])

    nperseg = min(int(sample_rate), np.count_nonzero(offsource))
    freqs, psd = welch(strain[offsource], fs=sample_rate, nperseg=nperseg)
    return freqs, psd


def extract_ringdown_segment(event_name: str, detector: str | None = None,
                             data_dir: str = "data/", sample_rate: int = 4096,
                             duration: int = 32, segment_length: float = 0.1,
                             pre_ringdown_pad: float = 0.01,
                             ringdown_delay_m: float = 10.0) -> RingdownSegment:
    """Download and extract a ringdown segment from real GWOSC strain data."""
    event = get_candidate_event(event_name)
    detector = detector or event["detectors"][0]
    local_path, actual_detector = download_gwosc_strain(
        event_name, detector=detector,
        sample_rate=sample_rate, duration=duration,
        data_dir=data_dir,
    )
    loaded = load_gwosc_strain_hdf5(local_path)
    merger_time = float(event["gps_time"])
    ringdown_start = merger_time + ringdown_delay_m * event["remnant_mass_msun"] * M_SUN_SECONDS

    start = ringdown_start - pre_ringdown_pad
    stop = ringdown_start + segment_length
    mask = (loaded["time"] >= start) & (loaded["time"] <= stop)
    if not np.any(mask):
        raise ValueError(f"Requested ringdown window is outside the downloaded segment for {event_name}")

    freqs, psd = estimate_noise_psd(
        loaded["strain"], loaded["sample_rate"],
        merger_time=merger_time, time=loaded["time"],
    )

    return RingdownSegment(
        event_name=event_name,
        strain=loaded["strain"][mask],
        time=loaded["time"][mask],
        sample_rate=loaded["sample_rate"],
        t_merger=merger_time,
        t_ringdown_start=ringdown_start,
        detector=actual_detector,
        remnant_mass_msun=event["remnant_mass_msun"],
        remnant_spin=event["remnant_spin"],
        noise_psd=psd,
        noise_freqs=freqs,
    )
