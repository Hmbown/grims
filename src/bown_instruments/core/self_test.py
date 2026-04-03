"""Self-testing trouble alarm pattern (US 1,573,801).

Bown's 1923 patent: an unattended radio receiver that periodically sends
itself a known test signal through its own circuits, checks the output
against the expected response, and raises an alarm if the system has degraded.

Both instruments use this pattern:
- GRIM-S injects synthetic ringdowns with known kappa, verifies recovery.
- CHIME injects synthetic transit signals with known depth, verifies fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SelfTestResult:
    """Result of an injection-recovery self-test."""

    injected: float
    recovered: float
    uncertainty: float
    passed: bool
    bias: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def fractional_error(self) -> float:
        if self.injected == 0:
            return float("inf") if self.recovered != 0 else 0.0
        return abs(self.recovered - self.injected) / abs(self.injected)


class SelfTest:
    """Bown self-test protocol: inject known signal, verify recovery.

    Parameters
    ----------
    inject_fn : callable
        (data, truth_value) -> modified_data.  Injects a signal of known
        strength into the data stream.
    recover_fn : callable
        (modified_data) -> (estimate, uncertainty).  Runs the measurement
        pipeline on modified data and returns the estimated parameter.
    tolerance_sigma : float
        Maximum allowed deviation in units of uncertainty (default: 3.0).
    """

    def __init__(
        self,
        inject_fn: Callable,
        recover_fn: Callable,
        tolerance_sigma: float = 3.0,
    ):
        self.inject_fn = inject_fn
        self.recover_fn = recover_fn
        self.tolerance_sigma = tolerance_sigma

    def run(
        self,
        data: np.ndarray,
        truth_value: float,
        n_trials: int = 1,
        rng: np.random.Generator | None = None,
    ) -> SelfTestResult:
        """Run the self-test.

        Parameters
        ----------
        data : array
            Baseline data (noise-only or real data to inject into).
        truth_value : float
            The known parameter value to inject.
        n_trials : int
            Number of independent injection-recovery trials (default: 1).
        rng : Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        SelfTestResult
        """
        if rng is None:
            rng = np.random.default_rng()

        estimates = []
        uncertainties = []

        for _ in range(n_trials):
            modified = self.inject_fn(data, truth_value)
            est, unc = self.recover_fn(modified)
            estimates.append(est)
            uncertainties.append(unc)

        recovered = float(np.mean(estimates))
        uncertainty = float(np.mean(uncertainties))
        if n_trials > 1:
            uncertainty = max(uncertainty, float(np.std(estimates, ddof=1)))

        bias = recovered - truth_value
        passed = abs(bias) < self.tolerance_sigma * uncertainty

        return SelfTestResult(
            injected=truth_value,
            recovered=recovered,
            uncertainty=uncertainty,
            passed=passed,
            bias=bias,
            details={
                "n_trials": n_trials,
                "tolerance_sigma": self.tolerance_sigma,
                "estimates": estimates,
            },
        )
