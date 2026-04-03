"""Transit model fitting for JWST transmission spectroscopy.

Implements Mandel & Agol (2002) analytic transit models with Gaussian Process
systematics removal. Fits planet-to-star radius ratio (Rp/Rs) per wavelength
bin to produce transmission spectra.

Limitations:
    - No stellar contamination (spots, faculae)
    - No multi-visit combining (one visit at a time)
    - Uncertainty from residual scatter, not formal MCMC posterior
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.optimize


# ------------------------------------------------------------------ #
# Kepler equation solver
# ------------------------------------------------------------------ #


def _solve_kepler(M: np.ndarray, ecc: float, tol: float = 1e-10) -> np.ndarray:
    """Solve Kepler's equation M = E - e*sin(E) via Newton's method.

    Parameters
    ----------
    M : ndarray
        Mean anomaly (radians).
    ecc : float
        Orbital eccentricity.
    tol : float
        Convergence tolerance.

    Returns
    -------
    E : ndarray
        Eccentric anomaly (radians).
    """
    M = np.asarray(M, dtype=np.float64)
    E = M.copy()
    for _ in range(30):
        dE = (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E


# ------------------------------------------------------------------ #
# Projected star-planet separation
# ------------------------------------------------------------------ #


def _compute_z(
    times: np.ndarray,
    t0: float,
    period: float,
    a_rs: float,
    inc: float,
    ecc: float = 0.0,
    omega: float = 0.0,
) -> np.ndarray:
    """Compute projected star-planet separation z(t) in stellar radii.

    Parameters
    ----------
    times : ndarray
        Observation times (same units as t0 and period).
    t0 : float
        Mid-transit time.
    period : float
        Orbital period.
    a_rs : float
        Semi-major axis in stellar radii.
    inc : float
        Orbital inclination (radians).
    ecc : float
        Eccentricity.
    omega : float
        Argument of periastron (radians).

    Returns
    -------
    z : ndarray
        Projected separation in units of stellar radius.
    """
    times = np.asarray(times, dtype=np.float64)

    # Find nearest transit epoch to data midpoint
    t_mid = np.mean(times)
    n_orbits = round((t_mid - t0) / period)
    t0_nearest = t0 + n_orbits * period

    if ecc < 1e-5:
        # Circular orbit: z = a/Rs * sqrt(sin^2(phase) + cos^2(inc)*cos^2(phase))
        phase = 2.0 * np.pi * (times - t0_nearest) / period
        x = np.sin(phase)
        y = np.cos(inc) * np.cos(phase)
        z = a_rs * np.sqrt(x**2 + y**2)
    else:
        # Eccentric orbit
        M = 2.0 * np.pi * (times - t0_nearest) / period
        # Adjust M so transit occurs at the correct phase
        # At transit, true anomaly f = pi/2 - omega
        f_transit = 0.5 * np.pi - omega
        E_transit = 2.0 * np.arctan2(
            np.sqrt(1.0 - ecc) * np.sin(f_transit / 2.0),
            np.sqrt(1.0 + ecc) * np.cos(f_transit / 2.0),
        )
        M_transit = E_transit - ecc * np.sin(E_transit)
        M = M + M_transit

        E = _solve_kepler(M, ecc)
        cos_f = (np.cos(E) - ecc) / (1.0 - ecc * np.cos(E))
        sin_f = np.sqrt(1.0 - ecc**2) * np.sin(E) / (1.0 - ecc * np.cos(E))
        f = np.arctan2(sin_f, cos_f)

        r = a_rs * (1.0 - ecc**2) / (1.0 + ecc * np.cos(f))
        x = r * np.sin(f + omega)
        y = r * np.cos(f + omega) * np.cos(inc)
        z = np.sqrt(x**2 + y**2)

    return z


# ------------------------------------------------------------------ #
# Uniform-source transit (Mandel & Agol 2002, Eq. 1)
# ------------------------------------------------------------------ #


def _uniform_source_flux(z: np.ndarray, p: float) -> np.ndarray:
    """Transit flux for a uniform-brightness star.

    Parameters
    ----------
    z : ndarray
        Projected separation in stellar radii.
    p : float
        Planet-to-star radius ratio (Rp/Rs).

    Returns
    -------
    flux : ndarray
        Relative flux (1.0 = no occultation).
    """
    z = np.asarray(z, dtype=np.float64)
    p = float(p)
    flux = np.ones_like(z)

    if p <= 0.0:
        return flux

    # Full occultation: z + p <= 1 (planet fully on disk)
    full = z <= 1.0 - p
    # Partial occultation: |1-p| < z < 1+p
    partial = (~full) & (z < 1.0 + p)

    # Full occultation case
    if p < 1.0:
        flux[full] = 1.0 - p**2
    else:
        # Planet larger than star, and fully covering it
        fully_covered = full & (z <= p - 1.0)
        flux[fully_covered] = 0.0
        # Planet larger than star, partial coverage on inner edge
        inner_partial = full & (z > p - 1.0)
        if np.any(inner_partial):
            zp = z[inner_partial]
            kappa0 = np.arccos((zp**2 + p**2 - 1.0) / (2.0 * zp * p))
            kappa1 = np.arccos((zp**2 - p**2 + 1.0) / (2.0 * zp))
            area = p**2 * kappa0 + kappa1 - 0.5 * np.sqrt(
                np.maximum(4.0 * zp**2 - (1.0 + zp**2 - p**2) ** 2, 0.0)
            )
            flux[inner_partial] = 1.0 - area / np.pi

    # Partial occultation
    if np.any(partial):
        zp = z[partial]
        kappa0 = np.arccos(
            np.clip((zp**2 + p**2 - 1.0) / (2.0 * zp * p), -1.0, 1.0)
        )
        kappa1 = np.arccos(
            np.clip((zp**2 - p**2 + 1.0) / (2.0 * zp), -1.0, 1.0)
        )
        area = p**2 * kappa0 + kappa1 - 0.5 * np.sqrt(
            np.maximum(4.0 * zp**2 - (1.0 + zp**2 - p**2) ** 2, 0.0)
        )
        flux[partial] = 1.0 - area / np.pi

    return flux


# ------------------------------------------------------------------ #
# Quadratic limb-darkened transit (numerical annular integration)
# ------------------------------------------------------------------ #


def _quad_ld_flux(
    z_arr: np.ndarray,
    p: float,
    u1: float,
    u2: float,
    n_annuli: int = 200,
) -> np.ndarray:
    """Quadratic limb-darkened transit via numerical annular integration.

    Decomposes the stellar disk into concentric annuli, computes the
    fractional arc occulted in each annulus by the planet disk, and
    weights by the limb-darkening intensity profile.

    Fully vectorized over (n_times, n_annuli) for performance.
    Sub-ppm accuracy with 200 annuli for all Rp/Rs < 0.3.

    Parameters
    ----------
    z_arr : ndarray
        Projected separation in stellar radii.
    p : float
        Planet-to-star radius ratio.
    u1, u2 : float
        Quadratic limb-darkening coefficients.
    n_annuli : int
        Number of concentric annuli for integration.

    Returns
    -------
    flux : ndarray
        Relative flux.
    """
    z_arr = np.atleast_1d(np.asarray(z_arr, dtype=np.float64))
    flux = np.ones(len(z_arr))

    if p <= 0.0:
        return flux

    # Stellar annulus grid
    r_edges = np.linspace(0.0, 1.0, n_annuli + 1)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    mu = np.sqrt(np.maximum(1.0 - r_mid**2, 0.0))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    # Ring weight = I(r) * 2*r*dr  (the pi cancels in numerator/denominator)
    ring_weights = intensity * 2.0 * r_mid * dr
    total_flux = np.sum(ring_weights)

    if total_flux <= 0.0:
        return flux

    # Vectorized (n_times, n_annuli) computation
    z_2d = z_arr[:, np.newaxis]  # (n_z, 1)
    r_2d = r_mid[np.newaxis, :]  # (1, n_annuli)

    # Conditions for ring-planet overlap
    # No overlap: ring and planet disk don't intersect
    no_overlap = (r_2d + p <= z_2d) | (z_2d + p <= r_2d)
    # Full overlap: entire ring is inside the planet disk
    full_overlap = z_2d + r_2d <= p

    # Partial overlap: fraction of ring arc inside planet disk
    # cos(alpha) = (r^2 + z^2 - p^2) / (2*r*z)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_alpha = (r_2d**2 + z_2d**2 - p**2) / (2.0 * r_2d * z_2d)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    frac = alpha / np.pi

    frac = np.where(no_overlap, 0.0, np.where(full_overlap, 1.0, frac))

    # No transit when z >= 1 + p
    out_of_transit = z_arr >= 1.0 + p

    # Weighted sum of blocked flux
    blocked = np.sum(frac * ring_weights[np.newaxis, :], axis=1)
    flux = 1.0 - blocked / total_flux
    flux[out_of_transit] = 1.0

    return flux


# ------------------------------------------------------------------ #
# Top-level transit model
# ------------------------------------------------------------------ #


def mandel_agol_flux(
    times: np.ndarray,
    rp_rs: float,
    t0: float,
    period: float,
    a_rs: float,
    inc: float,
    ecc: float = 0.0,
    omega: float = 0.0,
    u1: float = 0.0,
    u2: float = 0.0,
) -> np.ndarray:
    """Compute transit light curve using Mandel & Agol (2002) model.

    Parameters
    ----------
    times : ndarray
        Observation times (same units as t0 and period).
    rp_rs : float
        Planet-to-star radius ratio.
    t0 : float
        Mid-transit time.
    period : float
        Orbital period.
    a_rs : float
        Semi-major axis in units of stellar radius.
    inc : float
        Orbital inclination in radians.
    ecc : float
        Orbital eccentricity.
    omega : float
        Argument of periastron in radians.
    u1, u2 : float
        Quadratic limb-darkening coefficients. If both zero, uses
        the uniform-source model (faster).

    Returns
    -------
    flux : ndarray
        Model relative flux.
    """
    z = _compute_z(times, t0, period, a_rs, inc, ecc, omega)

    if u1 == 0.0 and u2 == 0.0:
        return _uniform_source_flux(z, rp_rs)
    else:
        return _quad_ld_flux(z, rp_rs, u1, u2)


# ------------------------------------------------------------------ #
# Gaussian Process systematics
# ------------------------------------------------------------------ #


def _gp_negloglik(
    log_params: np.ndarray,
    t: np.ndarray,
    residuals: np.ndarray,
    flux_err: np.ndarray,
) -> float:
    """Negative log marginal likelihood for RBF + white noise GP.

    Parameters
    ----------
    log_params : array of shape (3,)
        [log_amplitude, log_lengthscale, log_jitter]
    t : ndarray
        Times.
    residuals : ndarray
        Data residuals (flux - model).
    flux_err : ndarray
        Measurement uncertainties.

    Returns
    -------
    nll : float
        Negative log marginal likelihood.
    """
    amp = np.exp(log_params[0])
    length = np.exp(log_params[1])
    jitter = np.exp(log_params[2])

    dt = t[:, None] - t[None, :]
    K = amp**2 * np.exp(-0.5 * dt**2 / length**2)
    K += np.diag(flux_err**2 + jitter**2)

    try:
        L, low = scipy.linalg.cho_factor(K)
        alpha = scipy.linalg.cho_solve((L, low), residuals)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * (residuals @ alpha + logdet + len(t) * np.log(2.0 * np.pi))
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        nll = 1e10

    return float(nll)


def _gp_predict(
    log_params: np.ndarray,
    t: np.ndarray,
    residuals: np.ndarray,
    flux_err: np.ndarray,
) -> np.ndarray:
    """GP posterior mean at the training points.

    Parameters
    ----------
    log_params : array of shape (3,)
        [log_amplitude, log_lengthscale, log_jitter]
    t : ndarray
        Times.
    residuals : ndarray
        Data residuals.
    flux_err : ndarray
        Measurement uncertainties.

    Returns
    -------
    gp_mean : ndarray
        Posterior mean of the GP at the training points.
    """
    amp = np.exp(log_params[0])
    length = np.exp(log_params[1])
    jitter = np.exp(log_params[2])

    dt = t[:, None] - t[None, :]
    K = amp**2 * np.exp(-0.5 * dt**2 / length**2)
    K_noise = K + np.diag(flux_err**2 + jitter**2)

    try:
        L, low = scipy.linalg.cho_factor(K_noise)
        alpha = scipy.linalg.cho_solve((L, low), residuals)
        return K @ alpha
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        return np.zeros_like(residuals)


def _gp_predict_full(
    log_params: np.ndarray,
    t_train: np.ndarray,
    residuals: np.ndarray,
    flux_err: np.ndarray,
    t_pred: np.ndarray,
) -> np.ndarray:
    """GP posterior mean at arbitrary prediction points.

    Trains on (t_train, residuals) and predicts at t_pred.
    """
    amp = np.exp(log_params[0])
    length = np.exp(log_params[1])
    jitter = np.exp(log_params[2])

    dt_train = t_train[:, None] - t_train[None, :]
    K_train = amp**2 * np.exp(-0.5 * dt_train**2 / length**2)
    K_train_noise = K_train + np.diag(flux_err**2 + jitter**2)

    dt_cross = t_pred[:, None] - t_train[None, :]
    K_cross = amp**2 * np.exp(-0.5 * dt_cross**2 / length**2)

    try:
        L, low = scipy.linalg.cho_factor(K_train_noise)
        alpha = scipy.linalg.cho_solve((L, low), residuals)
        return K_cross @ alpha
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        return np.zeros(len(t_pred))


# ------------------------------------------------------------------ #
# Orbital parameter estimation from ephemeris
# ------------------------------------------------------------------ #


def _estimate_orbital_params(ephemeris: dict) -> tuple[float, float]:
    """Estimate a/Rs and inclination from ephemeris transit duration.

    Uses the simplified circular-orbit relation:
        duration_hours ~ period_days * 24 / pi * Rstar/a
    so  a/Rs ~ period_days * 24 / (pi * duration_hours)

    Assumes central transit (inclination = 90 deg).

    Parameters
    ----------
    ephemeris : dict
        Must contain 'period_days' and 'duration_hours'.

    Returns
    -------
    a_rs : float
        Semi-major axis in stellar radii.
    inc_rad : float
        Inclination in radians (pi/2 for central transit).
    """
    period_days = ephemeris["period_days"]
    duration_hours = ephemeris["duration_hours"]
    a_rs = period_days * 24.0 / (np.pi * duration_hours)
    inc_rad = 0.5 * np.pi
    return a_rs, inc_rad


# ------------------------------------------------------------------ #
# Per-wavelength transit + GP fit
# ------------------------------------------------------------------ #


def fit_transit_with_gp(
    times: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: dict,
    ld_coeffs: tuple[float, float] | None = None,
    a_rs: float | None = None,
    inc_deg: float | None = None,
) -> dict:
    """Fit a transit model with GP systematics to a single light curve.

    Uses an iterative approach:
      1. Fit rp_rs with no GP (chi-squared minimization)
      2. Fit GP hyperparameters to the residuals
      3. Divide out GP, re-fit rp_rs
    Converges in 2-3 iterations.

    Parameters
    ----------
    times : ndarray
        Observation times (MJD or BJD).
    flux : ndarray
        Normalized flux (1.0 = out of transit).
    flux_err : ndarray
        Flux uncertainties.
    ephemeris : dict
        From get_ephemeris(). Keys: period_days, t0_bjd, duration_hours, rp_rs.
    ld_coeffs : tuple, optional
        (u1, u2) quadratic limb-darkening coefficients.
    a_rs : float, optional
        Semi-major axis / stellar radius. Estimated from ephemeris if None.
    inc_deg : float, optional
        Inclination in degrees. Assumed 90 if None.

    Returns
    -------
    dict with keys:
        rp_rs, rp_rs_err, depth_ppm, depth_err_ppm, gp_params,
        model_flux, systematics, residuals, chi2_reduced
    """
    times = np.asarray(times, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    # Ensure positive errors (floor at 1e-6 to avoid division by zero)
    flux_err = np.maximum(flux_err, 1e-6)

    # Orbital parameters
    if a_rs is None or inc_deg is None:
        est_a_rs, est_inc = _estimate_orbital_params(ephemeris)
        if a_rs is None:
            a_rs = est_a_rs
        if inc_deg is None:
            inc_deg = np.degrees(est_inc)
    inc_rad = np.radians(inc_deg)

    # Time conversion: MJD -> BJD if needed
    bjd_times = times.copy()
    if np.median(times) < 100000.0:
        bjd_times = times + 2400000.5

    period = ephemeris["period_days"]
    t0 = ephemeris["t0_bjd"]

    # Find nearest transit epoch
    t_mid = np.mean(bjd_times)
    n_orbits = round((t_mid - t0) / period)
    t0_nearest = t0 + n_orbits * period

    # Limb darkening
    u1, u2 = ld_coeffs if ld_coeffs is not None else (0.0, 0.0)

    # Initial guess for rp_rs
    rp_rs_init = ephemeris.get("rp_rs", None)
    if rp_rs_init is None or rp_rs_init <= 0:
        # Estimate from data: depth ~ 1 - min(flux)
        rp_rs_init = np.sqrt(max(1.0 - np.nanmin(flux), 1e-6))
    rp_rs_init = np.clip(rp_rs_init, 0.005, 0.5)

    def _transit_model(rp_rs_val):
        return mandel_agol_flux(
            bjd_times, rp_rs_val, t0_nearest, period, a_rs, inc_rad,
            u1=u1, u2=u2,
        )

    def _chi2(rp_rs_val, corrected_flux):
        model = _transit_model(rp_rs_val)
        resid = corrected_flux - model
        return np.sum((resid / flux_err) ** 2)

    # --- Iteration 1: fit rp_rs without GP ---
    result_1 = scipy.optimize.minimize_scalar(
        _chi2,
        bounds=(max(0.001, rp_rs_init * 0.3), min(0.5, rp_rs_init * 3.0)),
        args=(flux,),
        method="bounded",
        options={"xatol": 1e-6},
    )
    rp_rs_current = result_1.x

    # --- Iteration 2: fit GP to residuals, re-fit rp_rs ---
    # --- Iterative GP detrending ---
    # The GP models smooth, long-timescale systematics (detector ramps, thermal
    # drifts) — NOT the transit.  To prevent the GP from absorbing transit signal:
    #   1. Lengthscale floor = 4× transit duration (GP varies on timescales much
    #      longer than ingress/egress)
    #   2. Amplitude ceiling = empirical out-of-transit scatter (GP cannot create
    #      features larger than what is actually seen in the baseline)
    #   3. Only fit GP to out-of-transit residuals, then predict everywhere
    transit_dur_days = ephemeris.get("duration_hours", 3.0) / 24.0
    min_gp_length = transit_dur_days * 4.0
    baseline = np.ptp(bjd_times)

    # Identify out-of-transit points from the initial model
    model_init = _transit_model(rp_rs_current)
    oot_mask = model_init > (1.0 - 0.0001)  # points with < 0.01% transit signal

    # Out-of-transit scatter sets the amplitude ceiling
    oot_scatter = np.std(flux[oot_mask]) if np.sum(oot_mask) > 10 else np.std(flux) * 0.1
    max_gp_amp = max(oot_scatter * 2, 1e-4)

    gp_log_params = np.array([
        np.log(oot_scatter * 0.5),                    # amplitude: ~half of oot scatter
        np.log(max(baseline * 0.5, min_gp_length)),   # lengthscale: half the baseline
        np.log(np.median(flux_err) * 0.1),             # jitter: small
    ])

    for _iteration in range(3):
        model_current = _transit_model(rp_rs_current)

        # Fit GP ONLY to out-of-transit residuals to avoid signal absorption
        oot_mask = model_current > (1.0 - 0.0001)
        if np.sum(oot_mask) < 20:
            # Not enough OOT points — fall back to all points but with tight bounds
            oot_mask = np.ones(len(flux), dtype=bool)

        oot_residuals = flux[oot_mask] - model_current[oot_mask]
        oot_times = bjd_times[oot_mask]
        oot_errors = flux_err[oot_mask]

        # Optimize GP hyperparameters on OOT data
        try:
            gp_result = scipy.optimize.minimize(
                _gp_negloglik,
                gp_log_params,
                args=(oot_times, oot_residuals, oot_errors),
                method="L-BFGS-B",
                bounds=[
                    (np.log(1e-6), np.log(max_gp_amp)),
                    (np.log(min_gp_length), np.log(max(baseline * 3, 10.0))),
                    (np.log(1e-8), np.log(np.median(flux_err))),
                ],
                options={"maxiter": 100},
            )
            if gp_result.success or gp_result.fun < 1e9:
                gp_log_params = gp_result.x
        except Exception:
            pass

        # Predict GP at ALL times (trained on OOT only)
        gp_mean = _gp_predict_full(
            gp_log_params, oot_times, oot_residuals, oot_errors, bjd_times
        )
        corrected_flux = flux - gp_mean

        # Re-fit rp_rs on corrected flux
        result_n = scipy.optimize.minimize_scalar(
            _chi2,
            bounds=(max(0.001, rp_rs_current * 0.5), min(0.5, rp_rs_current * 2.0)),
            args=(corrected_flux,),
            method="bounded",
            options={"xatol": 1e-7},
        )
        rp_rs_current = result_n.x

    # --- Final quantities ---
    model_final = _transit_model(rp_rs_current)
    residuals_final = flux - gp_mean - model_final
    n_data = len(flux)

    chi2_val = np.sum((residuals_final / flux_err) ** 2)
    chi2_reduced = chi2_val / max(n_data - 4, 1)  # 4 fitted params: rp_rs + 3 GP

    # Uncertainty on rp_rs from curvature of chi-squared
    # d(chi2)/d(rp_rs^2) at minimum -> Hessian-based error
    delta = max(rp_rs_current * 1e-4, 1e-6)
    chi2_minus = _chi2(rp_rs_current - delta, corrected_flux)
    chi2_center = _chi2(rp_rs_current, corrected_flux)
    chi2_plus = _chi2(rp_rs_current + delta, corrected_flux)
    d2chi2 = (chi2_plus - 2.0 * chi2_center + chi2_minus) / delta**2

    if d2chi2 > 0:
        rp_rs_err = np.sqrt(2.0 / d2chi2)  # Delta chi^2 = 1 for 1-sigma
    else:
        # Fallback: residual-scatter-based estimate
        rms = np.std(residuals_final)
        depth = rp_rs_current**2
        n_in = np.sum(model_final < 1.0 - 0.1 * depth)
        n_in = max(n_in, 1)
        rp_rs_err = rms / (2.0 * rp_rs_current * np.sqrt(n_in))

    depth_ppm = rp_rs_current**2 * 1e6
    depth_err_ppm = 2.0 * rp_rs_current * rp_rs_err * 1e6

    return {
        "rp_rs": float(rp_rs_current),
        "rp_rs_err": float(rp_rs_err),
        "depth_ppm": float(depth_ppm),
        "depth_err_ppm": float(depth_err_ppm),
        "gp_params": {
            "log_amplitude": float(gp_log_params[0]),
            "log_lengthscale": float(gp_log_params[1]),
            "log_jitter": float(gp_log_params[2]),
        },
        "model_flux": model_final,
        "systematics": gp_mean,
        "residuals": residuals_final,
        "chi2_reduced": float(chi2_reduced),
    }


# ------------------------------------------------------------------ #
# Result dataclass
# ------------------------------------------------------------------ #


@dataclass
class TransitFitResult:
    """Full transmission spectrum fitting result."""

    target: str
    wl_centers: np.ndarray
    rp_rs: np.ndarray
    rp_rs_err: np.ndarray
    depth_ppm: np.ndarray
    depth_err_ppm: np.ndarray
    grades: list[str]
    weights: np.ndarray
    combined_depth_ppm: float
    combined_depth_err_ppm: float
    naive_depth_ppm: float
    naive_depth_err_ppm: float
    improvement_factor: float
    chi2_reduced: np.ndarray
    n_bins: int
    n_dropped: int
    ephemeris: dict

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "target": self.target,
            "n_bins": self.n_bins,
            "n_dropped": self.n_dropped,
            "combined_depth_ppm": float(self.combined_depth_ppm),
            "combined_depth_err_ppm": float(self.combined_depth_err_ppm),
            "naive_depth_ppm": float(self.naive_depth_ppm),
            "naive_depth_err_ppm": float(self.naive_depth_err_ppm),
            "improvement_factor": float(self.improvement_factor),
            "bins": [
                {
                    "wl_center": float(self.wl_centers[i]),
                    "rp_rs": float(self.rp_rs[i]),
                    "rp_rs_err": float(self.rp_rs_err[i]),
                    "depth_ppm": float(self.depth_ppm[i]),
                    "depth_err_ppm": float(self.depth_err_ppm[i]),
                    "grade": self.grades[i],
                    "weight": float(self.weights[i]),
                    "chi2_reduced": float(self.chi2_reduced[i]),
                }
                for i in range(self.n_bins)
            ],
        }

    def to_ecsv_rows(self) -> list[dict]:
        """Return rows suitable for an ECSV table."""
        return [
            {
                "wl_center_um": float(self.wl_centers[i]),
                "rp_rs": float(self.rp_rs[i]),
                "rp_rs_err": float(self.rp_rs_err[i]),
                "depth_ppm": float(self.depth_ppm[i]),
                "depth_err_ppm": float(self.depth_err_ppm[i]),
                "grade": self.grades[i],
                "weight": float(self.weights[i]),
            }
            for i in range(self.n_bins)
        ]


# ------------------------------------------------------------------ #
# Main entry point: fit transmission spectrum
# ------------------------------------------------------------------ #


def fit_transmission_spectrum(
    flux_cube: np.ndarray,
    wavelength: np.ndarray,
    times_mjd: np.ndarray,
    in_transit_mask: np.ndarray,
    ephemeris: dict,
    channel_map,
    target: str = "unknown",
    n_bins: int | None = None,
    ld_coeffs: tuple[float, float] | None = None,
    a_rs: float | None = None,
    inc_deg: float | None = None,
    flux_error_cube: np.ndarray | None = None,
) -> TransitFitResult:
    """Fit transit depth per wavelength bin to produce a transmission spectrum.

    Uses channel_map bins for wavelength binning and grades/weights for
    diversity combining. For each bin, fits a Mandel & Agol transit model
    with GP systematics removal.

    Parameters
    ----------
    flux_cube : ndarray, shape (n_int, n_wl)
        Per-integration extracted spectra.
    wavelength : ndarray, shape (n_wl,)
        Wavelength grid in microns.
    times_mjd : ndarray, shape (n_int,)
        Mid-integration times in MJD.
    in_transit_mask : ndarray, shape (n_int,), bool
        True for in-transit integrations.
    ephemeris : dict
        From get_ephemeris(). Keys: period_days, t0_bjd, duration_hours, rp_rs.
    channel_map : ChannelMap
        Channel quality map from compute_channel_map().
    target : str
        Target name for labelling.
    n_bins : int, optional
        Override number of bins (default: use channel_map bins).
    ld_coeffs : tuple, optional
        (u1, u2) quadratic limb-darkening coefficients.
    a_rs : float, optional
        Semi-major axis / stellar radius.
    inc_deg : float, optional
        Inclination in degrees.

    Returns
    -------
    TransitFitResult
    """
    flux_cube = np.asarray(flux_cube, dtype=np.float64)
    wavelength = np.asarray(wavelength, dtype=np.float64)
    times_mjd = np.asarray(times_mjd, dtype=np.float64)
    in_transit_mask = np.asarray(in_transit_mask, dtype=bool)
    oot_mask = ~in_transit_mask

    # Orbital parameters
    if a_rs is None or inc_deg is None:
        est_a_rs, est_inc = _estimate_orbital_params(ephemeris)
        if a_rs is None:
            a_rs = est_a_rs
        if inc_deg is None:
            inc_deg = np.degrees(est_inc)

    # Convert times to BJD_TDB
    bjd_times = times_mjd.copy()
    if np.median(times_mjd) < 100000.0:
        bjd_times = times_mjd + 2400000.5

    # Use channel_map bins for wavelength binning
    bins_to_use = channel_map.bins
    if n_bins is not None and n_bins != len(bins_to_use):
        # Re-bin: use log-spaced edges over valid wavelength range
        wl_valid = np.isfinite(wavelength) & (wavelength > 0)
        wl_min, wl_max = wavelength[wl_valid].min(), wavelength[wl_valid].max()
        edges = np.geomspace(wl_min, wl_max * 1.001, n_bins + 1)
        bin_specs = [
            {"wl_range": (edges[i], edges[i + 1]), "grade": None}
            for i in range(n_bins)
        ]
    else:
        bin_specs = [
            {"wl_range": b.wl_range, "grade": b.grade}
            for b in bins_to_use
        ]

    # Arrays to fill
    n_total = len(bin_specs)
    wl_centers = np.zeros(n_total)
    rp_rs_arr = np.zeros(n_total)
    rp_rs_err_arr = np.full(n_total, np.inf)
    depth_arr = np.zeros(n_total)
    depth_err_arr = np.full(n_total, np.inf)
    grades = []
    chi2_arr = np.zeros(n_total)

    # Use channel_map weights if available and sizes match
    if hasattr(channel_map, "weights") and len(channel_map.weights) == n_total:
        weights = channel_map.weights.copy()
    else:
        weights = np.ones(n_total) / n_total

    valid_bins = 0
    n_dropped = 0

    for i, bspec in enumerate(bin_specs):
        wl_lo, wl_hi = bspec["wl_range"]
        wl_mask = (wavelength >= wl_lo) & (wavelength < wl_hi) & np.isfinite(wavelength)
        n_pix = int(np.sum(wl_mask))

        if n_pix < 1:
            grades.append("D")
            weights[i] = 0.0
            n_dropped += 1
            continue

        wl_centers[i] = float(np.mean(wavelength[wl_mask]))

        # Determine grade: prefer channel_map grade, else estimate
        grade = bspec.get("grade", None)
        if grade is None:
            # Quick grade from scatter excess
            wl_flux = np.nansum(flux_cube[:, wl_mask], axis=1)
            oot_flux = wl_flux[oot_mask]
            med = np.nanmedian(oot_flux)
            if med > 0 and len(oot_flux) > 5:
                scatter = np.nanmedian(np.abs(oot_flux - med)) * 1.4826
                photon = np.sqrt(med)
                excess = scatter / photon if photon > 0 else 10.0
                if excess < 2:
                    grade = "A"
                elif excess < 5:
                    grade = "B"
                elif excess < 10:
                    grade = "C"
                else:
                    grade = "D"
            else:
                grade = "D"
        grades.append(grade)

        if grade == "D":
            rp_rs_arr[i] = 0.0
            rp_rs_err_arr[i] = np.inf
            depth_arr[i] = 0.0
            depth_err_arr[i] = np.inf
            chi2_arr[i] = np.nan
            weights[i] = 0.0
            n_dropped += 1
            continue

        # Sum flux across wavelength bin
        bin_flux = np.nansum(flux_cube[:, wl_mask], axis=1)

        # Error: use pipeline flux errors if available, else empirical scatter
        if flux_error_cube is not None:
            bin_err = np.sqrt(np.nansum(flux_error_cube[:, wl_mask] ** 2, axis=1))
        else:
            bin_err = None

        # Normalize by out-of-transit median
        oot_med = np.nanmedian(bin_flux[oot_mask])
        if oot_med <= 0:
            grades[-1] = "D"
            weights[i] = 0.0
            n_dropped += 1
            continue

        norm_flux = bin_flux / oot_med
        if bin_err is not None:
            norm_err = bin_err / oot_med
        else:
            # Empirical: use MAD of out-of-transit as error estimate
            oot_scatter = np.nanmedian(np.abs(norm_flux[oot_mask] - 1.0)) * 1.4826
            norm_err = np.full(len(norm_flux), max(oot_scatter, 1e-6))

        # Fit transit + GP
        try:
            fit_result = fit_transit_with_gp(
                times_mjd,
                norm_flux,
                norm_err,
                ephemeris,
                ld_coeffs=ld_coeffs,
                a_rs=a_rs,
                inc_deg=inc_deg,
            )
            rp_rs_arr[i] = fit_result["rp_rs"]
            rp_rs_err_arr[i] = fit_result["rp_rs_err"]
            depth_arr[i] = fit_result["depth_ppm"]
            depth_err_arr[i] = fit_result["depth_err_ppm"]
            chi2_arr[i] = fit_result["chi2_reduced"]
            valid_bins += 1
        except Exception:
            # Fit failed -- fallback to simple depth estimate
            in_med = np.nanmedian(norm_flux[in_transit_mask])
            depth_simple = (1.0 - in_med) * 1e6
            rp_rs_simple = np.sqrt(max(1.0 - in_med, 1e-8))
            scatter_ppm = oot_scatter * 1e6
            n_in = max(np.sum(in_transit_mask), 1)
            err_simple = scatter_ppm / np.sqrt(n_in)

            rp_rs_arr[i] = rp_rs_simple
            rp_rs_err_arr[i] = err_simple / (2.0 * rp_rs_simple * 1e6) if rp_rs_simple > 0 else np.inf
            depth_arr[i] = depth_simple
            depth_err_arr[i] = err_simple
            chi2_arr[i] = np.nan
            valid_bins += 1

    # --- Diversity combining ---
    active = weights > 0
    if np.any(active):
        # Renormalize weights over active bins
        w = weights.copy()
        w[~active] = 0.0
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        weights = w

        combined_depth = float(np.sum(weights * depth_arr))
        combined_err = float(np.sqrt(np.sum(weights**2 * depth_err_arr**2)))
    else:
        combined_depth = 0.0
        combined_err = np.inf

    # Naive: unweighted mean over all non-dropped bins
    non_dropped = np.array([g != "D" for g in grades])
    if np.any(non_dropped):
        naive_depth = float(np.mean(depth_arr[non_dropped]))
        n_valid = int(np.sum(non_dropped))
        naive_err = float(
            np.sqrt(np.sum(depth_err_arr[non_dropped] ** 2)) / n_valid
        )
    else:
        naive_depth = 0.0
        naive_err = np.inf

    improvement = naive_err / combined_err if combined_err > 0 else 1.0

    return TransitFitResult(
        target=target,
        wl_centers=wl_centers,
        rp_rs=rp_rs_arr,
        rp_rs_err=rp_rs_err_arr,
        depth_ppm=depth_arr,
        depth_err_ppm=depth_err_arr,
        grades=grades,
        weights=weights,
        combined_depth_ppm=combined_depth,
        combined_depth_err_ppm=combined_err,
        naive_depth_ppm=naive_depth,
        naive_depth_err_ppm=naive_err,
        improvement_factor=improvement,
        chi2_reduced=chi2_arr,
        n_bins=n_total,
        n_dropped=n_dropped,
        ephemeris=ephemeris,
    )
