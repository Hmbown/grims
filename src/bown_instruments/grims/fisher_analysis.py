"""
Fisher matrix analysis for GRIM-S parameter degeneracies.

The question: is kappa degenerate with other parameters (remnant mass,
spin, mode amplitudes)? If yes, the wide audited intervals might just
reflect parameter correlations rather than genuine measurement uncertainty.

The Fisher information matrix:
    F_ij = <dh/dtheta_i | dh/dtheta_j>

where h is the waveform and theta_i are the parameters.
The inverse gives the parameter covariance matrix:
    C = F^{-1}

The correlation matrix shows which parameters are degenerate.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from .ringdown_templates import RingdownTemplateBuilder
from .qnm_modes import KerrQNMCatalog


@dataclass
class FisherResult:
    """Result of Fisher matrix analysis."""

    # Parameter names (in order)
    param_names: List[str]

    # Fisher matrix
    fisher_matrix: np.ndarray

    # Covariance matrix (inverse of Fisher)
    covariance_matrix: np.ndarray

    # Parameter uncertainties (1-sigma)
    uncertainties: np.ndarray

    # Correlation matrix
    correlation_matrix: np.ndarray

    # Condition number of Fisher matrix
    condition_number: float

    # Most degenerate pairs
    degenerate_pairs: List[Tuple[str, str, float]]  # (param1, param2, correlation)


def compute_fisher_matrix(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    A_220: float,
    kappa: float,
    noise_variance: float,
    params: List[str] = None,
    epsilon: float = 1e-6,
) -> FisherResult:
    """Compute Fisher information matrix for GRIM-S parameters.

    Parameters
    ----------
    data : whitened ringdown strain
    t_dimless : dimensionless time array
    spin : remnant spin
    A_220 : fundamental mode amplitude
    kappa : nonlinear coupling coefficient
    noise_variance : noise variance per sample
    params : list of parameters to include (default: all)
    epsilon : finite difference step size

    Returns
    -------
    FisherResult with Fisher matrix, covariance, correlations
    """
    if params is None:
        params = ["kappa", "A_220", "spin", "phi_220", "A_440", "phi_440"]

    builder = RingdownTemplateBuilder()
    mask = t_dimless >= 0

    # Compute derivatives numerically
    n_params = len(params)
    derivatives = np.zeros((n_params, len(data[mask])))

    # Base waveform
    base_template = builder.build_nonlinear_template(
        spin=spin,
        A_220=A_220,
        kappa=kappa,
        A_440_linear=0.0,  # simplified
        phi_220=0.0,
        phi_440_linear=0.0,
    )
    base_waveform = base_template.waveform(t_dimless)[mask]

    # Compute derivatives via finite differences
    for i, param in enumerate(params):
        # Perturb parameter
        if param == "kappa":
            h_plus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa + epsilon,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
            h_minus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa - epsilon,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
        elif param == "A_220":
            h_plus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220 + epsilon,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
            h_minus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220 - epsilon,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
        elif param == "spin":
            h_plus = builder.build_nonlinear_template(
                spin=spin + epsilon,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
            h_minus = builder.build_nonlinear_template(
                spin=spin - epsilon,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
        elif param == "phi_220":
            h_plus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=epsilon,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
            h_minus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=-epsilon,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
        elif param == "A_440":
            h_plus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=epsilon,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
            h_minus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=-epsilon,
                phi_220=0.0,
                phi_440_linear=0.0,
            ).waveform(t_dimless)[mask]
        elif param == "phi_440":
            h_plus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=epsilon,
            ).waveform(t_dimless)[mask]
            h_minus = builder.build_nonlinear_template(
                spin=spin,
                A_220=A_220,
                kappa=kappa,
                A_440_linear=0.0,
                phi_220=0.0,
                phi_440_linear=-epsilon,
            ).waveform(t_dimless)[mask]
        else:
            raise ValueError(f"Unknown parameter: {param}")

        derivatives[i] = (h_plus - h_minus) / (2 * epsilon)

    # Fisher matrix: F_ij = <dh/dtheta_i | dh/dtheta_j> / noise_var
    fisher = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(i, n_params):
            fisher[i, j] = np.sum(derivatives[i] * derivatives[j]) / noise_variance
            fisher[j, i] = fisher[i, j]

    # Add small regularization for numerical stability
    fisher += np.eye(n_params) * 1e-12

    # Covariance matrix
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        cov = np.linalg.pinv(fisher)

    # Uncertainties
    uncertainties = np.sqrt(np.abs(np.diag(cov)))

    # Correlation matrix
    corr = np.zeros_like(cov)
    for i in range(n_params):
        for j in range(n_params):
            if uncertainties[i] > 0 and uncertainties[j] > 0:
                corr[i, j] = cov[i, j] / (uncertainties[i] * uncertainties[j])

    # Condition number
    try:
        cond = np.linalg.cond(fisher)
    except:
        cond = float("inf")

    # Most degenerate pairs
    degenerate_pairs = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            degenerate_pairs.append((params[i], params[j], abs(corr[i, j])))
    degenerate_pairs.sort(key=lambda x: x[2], reverse=True)

    return FisherResult(
        param_names=params,
        fisher_matrix=fisher,
        covariance_matrix=cov,
        uncertainties=uncertainties,
        correlation_matrix=corr,
        condition_number=cond,
        degenerate_pairs=degenerate_pairs,
    )


def print_fisher_summary(result: FisherResult) -> None:
    """Print a human-readable Fisher analysis summary."""
    print("=" * 70)
    print("FISHER MATRIX ANALYSIS: Parameter Degeneracies")
    print("=" * 70)
    print()

    print("Parameter uncertainties (1-sigma):")
    print(f"{'Parameter':>15} {'Uncertainty':>15}")
    print("-" * 30)
    for name, unc in zip(result.param_names, result.uncertainties):
        print(f"{name:>15} {unc:>15.6f}")
    print()

    print(f"Condition number: {result.condition_number:.2e}")
    if result.condition_number > 1e10:
        print("WARNING: Fisher matrix is ill-conditioned!")
        print("This suggests strong parameter degeneracies.")
    print()

    print("Most degenerate parameter pairs:")
    print(f"{'Param 1':>15} {'Param 2':>15} {'|Correlation|':>15}")
    print("-" * 45)
    for p1, p2, corr in result.degenerate_pairs[:5]:
        print(f"{p1:>15} {p2:>15} {corr:>15.4f}")
    print()

    # Correlation matrix
    print("Correlation matrix:")
    header = " " * 15 + "".join(f"{p:>15}" for p in result.param_names)
    print(header)
    print("-" * (15 + 15 * len(result.param_names)))
    for i, name in enumerate(result.param_names):
        row = f"{name:>15}"
        for j in range(len(result.param_names)):
            row += f"{result.correlation_matrix[i, j]:>15.4f}"
        print(row)
    print()

    # Interpretation
    print("Interpretation:")
    kappa_idx = (
        result.param_names.index("kappa") if "kappa" in result.param_names else -1
    )
    if kappa_idx >= 0:
        for i, name in enumerate(result.param_names):
            if i != kappa_idx:
                corr = result.correlation_matrix[kappa_idx, i]
                if abs(corr) > 0.9:
                    print(
                        f"  - kappa is STRONGLY degenerate with {name} (|r|={abs(corr):.3f})"
                    )
                elif abs(corr) > 0.7:
                    print(
                        f"  - kappa is MODERATELY degenerate with {name} (|r|={abs(corr):.3f})"
                    )
                elif abs(corr) > 0.5:
                    print(
                        f"  - kappa is WEAKLY degenerate with {name} (|r|={abs(corr):.3f})"
                    )
    print()


def plot_fisher_correlations(result: FisherResult, save_path: str = None):
    """Plot Fisher correlation matrix as a heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result.param_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        result.correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal"
    )

    ax.set_xticks(range(n))
    ax.set_xticklabels(result.param_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(result.param_names)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = result.correlation_matrix[i, j]
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color
            )

    plt.colorbar(im, label="Correlation", ax=ax)
    ax.set_title("Fisher Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Fisher correlation plot to {save_path}")

    plt.close()
    return fig
