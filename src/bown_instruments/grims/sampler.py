"""
MCMC sampler for GRIM-S full parameter inference.

Replaces the 1D grid scan with a proper joint posterior over:
- kappa: nonlinear coupling coefficient
- A_220, A_330, A_440: mode amplitudes
- phi_220, phi_330, phi_440: mode phases
- t_start: ringdown start time (optional)

Uses emcee for affine-invariant ensemble sampling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MCMCResult:
    """Result of MCMC sampling."""

    # Parameter names
    param_names: List[str]

    # Chains: shape (n_walkers, n_steps, n_params)
    chains: np.ndarray

    # Log probabilities: shape (n_walkers, n_steps)
    log_prob: np.ndarray

    # Summary statistics
    medians: np.ndarray
    uncertainties: np.ndarray  # 16th-84th percentile range
    means: np.ndarray

    # Convergence diagnostics
    n_eff: np.ndarray  # effective sample size
    r_hat: np.ndarray  # Gelman-Rubin statistic (if multiple chains)

    # Metadata
    n_walkers: int
    n_steps: int
    n_burnin: int
    acceptance_fraction: float


def build_waveform_fast(
    t: np.ndarray,
    params: np.ndarray,
    spin: float,
    builder=None,
) -> np.ndarray:
    """Build waveform from parameter vector.

    Parameters
    ----------
    t : time array
    params : [kappa, A_220, A_330, A_440, phi_220, phi_330, phi_440]
    spin : remnant spin
    builder : RingdownTemplateBuilder

    Returns
    -------
    waveform : model waveform
    """
    from .ringdown_templates import RingdownTemplateBuilder

    if builder is None:
        builder = RingdownTemplateBuilder()

    kappa, A_220, A_330, A_440, phi_220, phi_330, phi_440 = params

    template = builder.build_nonlinear_template(
        spin=spin,
        A_220=A_220,
        kappa=kappa,
        A_330=A_330,
        A_440_linear=A_440,
        phi_220=phi_220,
        phi_330=phi_330,
        phi_440_linear=phi_440,
        phi_nl=2.0 * phi_220,  # phase-locked
    )

    return template.waveform(t)


def log_prior(params: np.ndarray, bounds: List[Tuple[float, float]]) -> float:
    """Compute log prior (uniform within bounds)."""
    for p, (low, high) in zip(params, bounds):
        if p < low or p > high:
            return -np.inf
    return 0.0


def log_likelihood(
    params: np.ndarray,
    data: np.ndarray,
    t: np.ndarray,
    spin: float,
    noise_variance: float,
    builder=None,
) -> float:
    """Compute log likelihood for MCMC."""
    try:
        model = build_waveform_fast(t, params, spin, builder)
    except:
        return -np.inf

    mask = t >= 0
    residual = data[mask] - model[mask]

    if noise_variance > 0:
        return -0.5 * np.sum(residual**2) / noise_variance
    return -0.5 * np.sum(residual**2)


def log_posterior(
    params: np.ndarray,
    data: np.ndarray,
    t: np.ndarray,
    spin: float,
    noise_variance: float,
    bounds: List[Tuple[float, float]],
    builder=None,
) -> float:
    """Compute log posterior = log likelihood + log prior."""
    lp = log_prior(params, bounds)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(params, data, t, spin, noise_variance, builder)
    return lp + ll


def run_mcmc(
    data: np.ndarray,
    t_dimless: np.ndarray,
    spin: float,
    noise_variance: float,
    n_walkers: int = 64,
    n_steps: int = 1000,
    n_burnin: int = 200,
    initial_guess: np.ndarray = None,
    bounds: List[Tuple[float, float]] = None,
    event_name: str = "unknown",
) -> MCMCResult:
    """Run MCMC sampler for GRIM-S parameters.

    Parameters
    ----------
    data : whitened ringdown strain
    t_dimless : dimensionless time array
    spin : remnant spin
    noise_variance : noise variance
    n_walkers : number of walkers
    n_steps : number of steps per walker
    n_burnin : number of burnin steps to discard
    initial_guess : initial parameter values [kappa, A_220, A_330, A_440, phi_220, phi_330, phi_440]
    bounds : parameter bounds [(low, high), ...]
    event_name : identifier

    Returns
    -------
    MCMCResult with chains and summary statistics
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("emcee is required for MCMC. Install with: pip install emcee")

    from .ringdown_templates import RingdownTemplateBuilder
    from .bayesian_analysis import fit_linear_modes

    builder = RingdownTemplateBuilder()

    # Parameter names
    param_names = ["kappa", "A_220", "A_330", "A_440", "phi_220", "phi_330", "phi_440"]
    n_params = len(param_names)

    # Default bounds
    if bounds is None:
        bounds = [
            (0.0, 10.0),  # kappa
            (0.0, 10.0),  # A_220
            (0.0, 10.0),  # A_330
            (0.0, 10.0),  # A_440
            (-np.pi, np.pi),  # phi_220
            (-np.pi, np.pi),  # phi_330
            (-np.pi, np.pi),  # phi_440
        ]

    # Initial guess from linear mode fit
    if initial_guess is None:
        linear = fit_linear_modes(data, t_dimless, spin)
        initial_guess = np.array(
            [
                0.1,  # kappa
                linear["220"]["amplitude"],
                linear["330"]["amplitude"],
                linear["440"]["amplitude"],
                linear["220"]["phase"],
                linear["330"]["phase"],
                linear["440"]["phase"],
            ]
        )

    # Initialize walkers
    ndim = n_params
    pos = initial_guess + 1e-4 * np.random.randn(n_walkers, ndim)

    # Ensure walkers are within bounds
    for i in range(n_walkers):
        for j in range(ndim):
            pos[i, j] = np.clip(pos[i, j], bounds[j][0] + 1e-6, bounds[j][1] - 1e-6)

    # Set up sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_likelihood,
        args=(data, t_dimless, spin, noise_variance, builder),
    )

    # Run MCMC
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Extract chains
    chains = sampler.get_chain()  # shape (n_steps, n_walkers, n_params)
    log_prob = sampler.get_log_prob()  # shape (n_steps, n_walkers)

    # Discard burnin
    flat_samples = chains[n_burnin:].reshape(-1, n_params)

    # Summary statistics
    medians = np.median(flat_samples, axis=0)
    uncertainties = np.array(
        [
            np.percentile(flat_samples[:, i], 84)
            - np.percentile(flat_samples[:, i], 16)
            for i in range(n_params)
        ]
    )
    means = np.mean(flat_samples, axis=0)

    # Effective sample size
    try:
        from emcee.autocorr import integrated_time

        tau = integrated_time(chains[n_burnin:], quiet=True)
        n_eff = chains.shape[0] / tau
    except:
        n_eff = np.full(n_params, np.nan)

    # Acceptance fraction
    acceptance = np.mean(sampler.acceptance_fraction)

    return MCMCResult(
        param_names=param_names,
        chains=chains,
        log_prob=log_prob,
        medians=medians,
        uncertainties=uncertainties,
        means=means,
        n_eff=n_eff,
        r_hat=np.ones(n_params),  # Would need multiple chains for proper R-hat
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burnin=n_burnin,
        acceptance_fraction=acceptance,
    )


def print_mcmc_summary(result: MCMCResult) -> None:
    """Print MCMC results summary."""
    print("=" * 70)
    print("MCMC SAMPLING RESULTS")
    print("=" * 70)
    print()
    print(f"Walkers: {result.n_walkers}")
    print(f"Steps: {result.n_steps}")
    print(f"Burnin: {result.n_burnin}")
    print(f"Acceptance fraction: {result.acceptance_fraction:.3f}")
    print()
    print(f"{'Parameter':>12} {'Median':>12} {'16-84%':>12} {'Mean':>12} {'N_eff':>10}")
    print("-" * 60)
    for i, name in enumerate(result.param_names):
        print(
            f"{name:>12} {result.medians[i]:>12.4f} "
            f"±{result.uncertainties[i]:>10.4f} "
            f"{result.means[i]:>12.4f} "
            f"{result.n_eff[i]:>10.1f}"
        )
    print()

    # Convergence check
    if np.all(result.n_eff > 100):
        print("Convergence: GOOD (all N_eff > 100)")
    elif np.all(result.n_eff > 50):
        print("Convergence: ACCEPTABLE (all N_eff > 50)")
    else:
        print("Convergence: POOR (some N_eff < 50)")
        print("Consider running longer or adjusting proposal distribution.")
    print()


def plot_mcmc_chains(result: MCMCResult, save_path: str = None):
    """Plot MCMC chains and corner plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(result.param_names)

    # Chain plots
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)

    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for j in range(result.n_walkers):
            ax.plot(result.log_prob[j], "k-", alpha=0.1)
        ax.set_ylabel(result.param_names[i], fontsize=10)
        ax.axvline(result.n_burnin, color="red", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Step", fontsize=11)
    plt.suptitle("MCMC Chains", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path.replace(".png", "_chains.png"), dpi=150, bbox_inches="tight"
        )
    plt.close()

    # Corner plot (if corner package available)
    try:
        import corner

        flat_samples = result.chains[result.n_burnin :].reshape(-1, n_params)

        fig = corner.corner(
            flat_samples,
            labels=result.param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 10},
        )

        if save_path:
            plt.savefig(
                save_path.replace(".png", "_corner.png"), dpi=150, bbox_inches="tight"
            )
        plt.close()
    except ImportError:
        print("corner package not available. Install with: pip install corner")
        print("Skipping corner plot.")
