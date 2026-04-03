"""Measurement-first diagnostics.

Bown's principle: you cannot improve what you cannot observe.  Before
reporting a science result, verify the instrument is healthy.

This module provides a lightweight health-check protocol that both
instruments can call before running their analysis pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HealthCheck:
    """Result of a single instrument health check."""

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


def check_instrument_health(checks: list[HealthCheck]) -> bool:
    """Run a list of health checks and report results.

    Parameters
    ----------
    checks : list of HealthCheck
        Each check has already been evaluated before being passed here.

    Returns
    -------
    all_passed : bool
        True only if every check passed.
    """
    all_passed = True
    for check in checks:
        if not check.passed:
            all_passed = False
    return all_passed
