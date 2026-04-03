"""
Ringdown waveform templates: linear + nonlinear.

The ringdown signal from a perturbed Kerr black hole is a sum of
damped sinusoids (quasinormal modes). At linear order:

    h(t) = sum_i  A_i * exp(-t/tau_i) * cos(2*pi*f_i*t + phi_i)

The nonlinear contribution (Cheung et al. 2023, Mitman et al. 2023)
adds modes whose frequencies are sums of linear mode frequencies,
and whose amplitudes scale quadratically:

    A_NL(4,4) = kappa * A(2,2,0)^2

where kappa is the nonlinear coupling coefficient — the observable
that GRIM-S is designed to measure.

In Bown's language: kappa is the intermodulation distortion coefficient
of spacetime. If GR is correct, kappa has a definite predicted value
for each mass ratio. If kappa deviates, either GR is wrong or the
environment is contaminating the signal.
"""

import numpy as np
from dataclasses import dataclass, field
from .qnm_modes import KerrQNMCatalog, QNMFrequency


@dataclass
class RingdownMode:
    """A single mode in the ringdown template."""
    qnm: QNMFrequency
    amplitude: float  # dimensionless strain amplitude
    phase: float  # initial phase (radians)


@dataclass
class RingdownTemplate:
    """Complete ringdown waveform template.

    Parameters are in dimensionless units (t in units of M).
    Convert to physical units using the remnant mass.
    """
    modes: list  # list of RingdownMode
    remnant_mass_msun: float = 0.0
    remnant_spin: float = 0.0
    t_start: float = 0.0  # ringdown start time after merger (in M)

    def waveform(self, t: np.ndarray) -> np.ndarray:
        """Compute the ringdown strain h(t).

        Parameters
        ----------
        t : time array in units of M (t=0 is ringdown start)

        Returns
        -------
        h : strain waveform (real part of h+ - i*hx)
        """
        h = np.zeros_like(t, dtype=float)
        mask = t >= 0  # causal: no signal before ringdown starts

        for mode in self.modes:
            omega = mode.qnm.omega
            # h_mode = A * exp(i*omega*t + i*phi)
            # real part gives h+, imaginary gives hx
            h[mask] += mode.amplitude * np.exp(omega.imag * t[mask]) * \
                np.cos(omega.real * t[mask] + mode.phase)

        return h

    def waveform_complex(self, t: np.ndarray) -> np.ndarray:
        """Compute complex ringdown strain h+ - i*hx."""
        h = np.zeros_like(t, dtype=complex)
        mask = t >= 0

        for mode in self.modes:
            omega = mode.qnm.omega
            # QNM convention is h ~ exp(-i * omega * t). With Im(omega) < 0
            # this gives the expected exponential decay rather than growth.
            h[mask] += mode.amplitude * np.exp(
                -1j * omega * t[mask] + 1j * mode.phase
            )

        return h

    def waveform_physical(self, t_seconds: np.ndarray) -> np.ndarray:
        """Compute strain in physical time (seconds)."""
        m_sun_seconds = 4.925491025543576e-06
        m_seconds = self.remnant_mass_msun * m_sun_seconds
        t_dimless = (t_seconds - self.t_start * m_seconds) / m_seconds
        return self.waveform(t_dimless)


class RingdownTemplateBuilder:
    """Build ringdown templates with linear and nonlinear modes.

    The key parameter is `kappa` — the nonlinear coupling coefficient.
    GR predicts a specific value of kappa for each configuration.
    Measuring kappa != kappa_GR would indicate either:
      - Modified gravity
      - Environmental contamination
      - Systematic error in the template

    Bown's self-test: the orthogonality check (see self_test module)
    catches case 3.
    """

    def __init__(self):
        self.catalog = KerrQNMCatalog()

    def build_linear_template(self, spin: float,
                              A_220: float = 1.0,
                              A_221: float = 0.0,
                              A_330: float = 0.0,
                              A_440: float = 0.0,
                              phi_220: float = 0.0,
                              phi_221: float = 0.0,
                              phi_330: float = 0.0,
                              phi_440: float = 0.0,
                              mass_msun: float = 0.0) -> RingdownTemplate:
        """Build a linear-only ringdown template."""
        modes = []

        if A_220 != 0:
            qnm_220 = self.catalog.linear_mode(2, 2, 0, spin)
            modes.append(RingdownMode(qnm_220, A_220, phi_220))

        if A_221 != 0:
            qnm_221 = self.catalog.linear_mode(2, 2, 1, spin)
            modes.append(RingdownMode(qnm_221, A_221, phi_221))

        if A_330 != 0:
            qnm_330 = self.catalog.linear_mode(3, 3, 0, spin)
            modes.append(RingdownMode(qnm_330, A_330, phi_330))

        if A_440 != 0:
            qnm_440 = self.catalog.linear_mode(4, 4, 0, spin)
            modes.append(RingdownMode(qnm_440, A_440, phi_440))

        return RingdownTemplate(
            modes=modes,
            remnant_mass_msun=mass_msun,
            remnant_spin=spin,
        )

    def build_nonlinear_template(self, spin: float,
                                 A_220: float = 1.0,
                                 A_221: float = 0.0,
                                 A_330: float = 0.0,
                                 A_440_linear: float = 0.0,
                                 kappa: float = 1.0,
                                 phi_220: float = 0.0,
                                 phi_221: float = 0.0,
                                 phi_330: float = 0.0,
                                 phi_440_linear: float = 0.0,
                                 phi_nl: float = 0.0,
                                 mass_msun: float = 0.0) -> RingdownTemplate:
        """Build a ringdown template including the nonlinear (4,4) mode.

        The nonlinear mode amplitude is:
            A_NL = kappa * A_220^2

        where kappa is the coupling coefficient to be measured.

        Parameters
        ----------
        kappa : nonlinear coupling coefficient.
            kappa=0 → no nonlinear mode (linear GR)
            kappa=kappa_GR → GR prediction
            kappa=free → the parameter GRIM-S measures
        """
        # Start with linear modes
        template = self.build_linear_template(
            spin, A_220, A_221, A_330, A_440_linear,
            phi_220, phi_221, phi_330, phi_440_linear,
            mass_msun,
        )

        # Add nonlinear mode
        if kappa != 0 and A_220 != 0:
            qnm_nl = self.catalog.nonlinear_mode_quadratic(spin)
            A_nl = kappa * A_220 ** 2
            # Phase of nonlinear mode is 2x the parent phase
            # (from the quadratic coupling)
            phi_nl_total = 2.0 * phi_220 + phi_nl
            template.modes.append(RingdownMode(qnm_nl, A_nl, phi_nl_total))

        return template

    def build_template_grid(self, spin: float, A_220: float = 1.0,
                            kappa_values: np.ndarray = None) -> list:
        """Build a grid of templates over kappa values.

        This is the matched filter bank: one template per kappa value.
        The data is compared against each template, and the best-fit
        kappa is the measurement.
        """
        if kappa_values is None:
            kappa_values = np.linspace(0.0, 5.0, 51)

        templates = []
        for kappa in kappa_values:
            t = self.build_nonlinear_template(
                spin=spin, A_220=A_220, kappa=kappa,
            )
            templates.append((kappa, t))

        return templates


def snr_threshold_for_nonlinear_detection(spin: float, A_220: float,
                                          kappa: float,
                                          sigma_n: float) -> dict:
    """Estimate the SNR needed to detect the nonlinear mode.

    The nonlinear mode amplitude is kappa * A_220^2.
    Detection requires SNR_NL = A_NL / sigma_n > threshold.

    This is a Bown-style calculation: given the channel noise,
    what signal strength do you need?

    Parameters
    ----------
    spin : remnant spin
    A_220 : fundamental mode amplitude
    kappa : nonlinear coupling coefficient
    sigma_n : noise level (strain spectral density at mode frequency)

    Returns
    -------
    dict with SNR estimates and detection thresholds
    """
    catalog = KerrQNMCatalog()
    nl_mode = catalog.nonlinear_mode_quadratic(spin)

    A_nl = kappa * A_220 ** 2
    snr_nl = A_nl / sigma_n if sigma_n > 0 else float('inf')

    # Effective cycles of the nonlinear mode before decay
    n_cycles = nl_mode.quality_factor

    # Matched filter SNR enhancement
    snr_matched = snr_nl * np.sqrt(2 * n_cycles)

    return {
        "spin": spin,
        "A_220": A_220,
        "kappa": kappa,
        "A_nonlinear": A_nl,
        "snr_single_cycle": snr_nl,
        "snr_matched_filter": snr_matched,
        "quality_factor": n_cycles,
        "nl_frequency": nl_mode.frequency,
        "nl_damping_time": nl_mode.damping_time,
        "detectable_3sigma": snr_matched > 3.0,
        "detectable_5sigma": snr_matched > 5.0,
    }
