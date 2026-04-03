"""
QNM frequency computation and mode catalog for Kerr black holes.

Uses the `qnm` package (Stein, 2019) for linear mode frequencies,
and constructs nonlinear mode frequencies from the quadratic coupling
relation: omega_NL(4,4) = 2 * omega(2,2,0).

References:
  - Green, Hollands, Sberna, Toomani, Zimmerman, PRD 107, 064030 (2023)
  - Cheung et al., PRL 130, 081401 (2023)
  - Mitman et al., PRL 130, 081402 (2023)
"""

import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

import qnm


@dataclass
class QNMFrequency:
    """A single quasinormal mode frequency."""

    s: int  # spin weight (-2 for gravitational)
    l: int  # angular quantum number
    m: int  # azimuthal quantum number
    n: int  # overtone number
    omega: complex  # dimensionless frequency (M * omega_physical)
    is_nonlinear: bool = False
    parent_modes: tuple = ()  # for nonlinear modes: which linear modes generate this

    @property
    def frequency(self) -> float:
        """Oscillation frequency (real part)."""
        return self.omega.real

    @property
    def damping_rate(self) -> float:
        """Damping rate (imaginary part, negative for decaying)."""
        return self.omega.imag

    @property
    def damping_time(self) -> float:
        """Damping time in units of M (1 / |Im(omega)|)."""
        return 1.0 / abs(self.omega.imag)

    @property
    def quality_factor(self) -> float:
        """Q-factor: number of oscillation cycles before decay."""
        return self.frequency / (2.0 * abs(self.damping_rate))

    def physical_frequency_hz(self, mass_msun: float) -> float:
        """Convert to physical frequency in Hz given BH mass in solar masses."""
        # M_sun in seconds: G * M_sun / c^3
        m_sun_seconds = 4.925491025543576e-06
        m_seconds = mass_msun * m_sun_seconds
        return self.frequency / (2.0 * np.pi * m_seconds)

    def physical_damping_time_s(self, mass_msun: float) -> float:
        """Convert damping time to seconds given BH mass in solar masses."""
        m_sun_seconds = 4.925491025543576e-06
        m_seconds = mass_msun * m_sun_seconds
        return self.damping_time * m_seconds


class KerrQNMCatalog:
    """
    Catalog of Kerr quasinormal mode frequencies, including nonlinear modes.

    The key insight (Cheung et al. 2023, Mitman et al. 2023):
    Nonlinear modes arise at frequencies that are sums of linear mode frequencies.
    The dominant nonlinear mode is (l=4, m=4) with:
        omega_NL = 2 * omega_linear(2,2,0)
    This is DISTINCT from the linear (4,4,0) mode.
    """

    def __init__(self):
        self._cache = {}

    def linear_mode(self, l: int, m: int, n: int, spin: float,
                    s: int = -2) -> QNMFrequency:
        """Compute a linear QNM frequency for a Kerr black hole.

        Parameters
        ----------
        l : angular quantum number
        m : azimuthal quantum number
        n : overtone number
        spin : dimensionless spin parameter (0 <= a < 1)
        s : spin weight (-2 for gravitational perturbations)
        """
        key = (s, l, m, n)
        if key not in self._cache:
            self._cache[key] = qnm.modes_cache(s=s, l=l, m=m, n=n)

        omega, _, _ = self._cache[key](a=spin)

        return QNMFrequency(
            s=s, l=l, m=m, n=n,
            omega=omega,
            is_nonlinear=False,
        )

    def nonlinear_mode_quadratic(self, spin: float,
                                 parent_l1: int = 2, parent_m1: int = 2,
                                 parent_n1: int = 0,
                                 parent_l2: int = 2, parent_m2: int = 2,
                                 parent_n2: int = 0) -> QNMFrequency:
        """Compute a nonlinear QNM frequency from quadratic coupling.

        The nonlinear mode frequency is:
            omega_NL = omega(l1,m1,n1) + omega(l2,m2,n2)

        For the dominant case: omega_NL = 2 * omega(2,2,0)
        This produces a mode with effective (l=4, m=4) angular structure.

        Parameters
        ----------
        spin : dimensionless spin parameter
        parent_l1, parent_m1, parent_n1 : first parent mode quantum numbers
        parent_l2, parent_m2, parent_n2 : second parent mode quantum numbers
        """
        mode1 = self.linear_mode(parent_l1, parent_m1, parent_n1, spin)
        mode2 = self.linear_mode(parent_l2, parent_m2, parent_n2, spin)

        omega_nl = mode1.omega + mode2.omega
        l_nl = parent_l1 + parent_l2  # effective angular number
        m_nl = parent_m1 + parent_m2

        return QNMFrequency(
            s=-2,
            l=l_nl, m=m_nl, n=-1,  # n=-1 signals nonlinear
            omega=omega_nl,
            is_nonlinear=True,
            parent_modes=((parent_l1, parent_m1, parent_n1),
                          (parent_l2, parent_m2, parent_n2)),
        )

    def standard_ringdown_basis(self, spin: float,
                                include_nonlinear: bool = True) -> list:
        """Build the standard ringdown mode basis.

        Linear modes:
          (2,2,0) — dominant fundamental
          (2,2,1) — first overtone
          (3,3,0) — subdominant
          (4,4,0) — linear higher harmonic

        Nonlinear modes (if include_nonlinear=True):
          NL(4,4) = 2 * (2,2,0) — quadratic coupling product

        Returns list of QNMFrequency objects.
        """
        modes = [
            self.linear_mode(2, 2, 0, spin),
            self.linear_mode(2, 2, 1, spin),
            self.linear_mode(3, 3, 0, spin),
            self.linear_mode(4, 4, 0, spin),
        ]

        if include_nonlinear:
            modes.append(self.nonlinear_mode_quadratic(spin))

        return modes

    def frequency_separation_ratio(self, spin: float) -> dict:
        """Compute the frequency separation between linear (4,4,0)
        and nonlinear (4,4) modes.

        This separation is what makes the nonlinear mode identifiable:
        if they were degenerate, you could not distinguish them.

        Returns dict with separation metrics.
        """
        linear_440 = self.linear_mode(4, 4, 0, spin)
        nl_44 = self.nonlinear_mode_quadratic(spin)

        delta_f = abs(linear_440.frequency - nl_44.frequency)
        mean_f = 0.5 * (linear_440.frequency + nl_44.frequency)

        return {
            "spin": spin,
            "omega_linear_440": linear_440.omega,
            "omega_nonlinear_44": nl_44.omega,
            "delta_freq": delta_f,
            "fractional_separation": delta_f / mean_f,
            "delta_damping": abs(linear_440.damping_rate - nl_44.damping_rate),
        }


def survey_spin_dependence(spins=None):
    """Survey how the linear/nonlinear frequency separation varies with spin.

    This is a diagnostic: if the separation shrinks to zero at some spin,
    the modes become degenerate and unresolvable. Bown would call this
    "the channel closing."
    """
    if spins is None:
        spins = np.linspace(0.0, 0.95, 20)

    catalog = KerrQNMCatalog()
    results = []
    for a in spins:
        results.append(catalog.frequency_separation_ratio(a))

    return results
