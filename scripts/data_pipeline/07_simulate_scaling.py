# Usage:
#   python scripts/Fig4_simulate_neighborhoods_to_cities_scaling.py
#   python scripts/Fig4_simulate_neighborhoods_to_cities_scaling.py --plot
#   python scripts/Fig4_simulate_neighborhoods_to_cities_scaling.py --plot --uncertainty
#   python scripts/Fig4_simulate_neighborhoods_to_cities_scaling.py --s_uncertainty --plot --s_std 0.59
#   python scripts/Fig4_simulate_neighborhoods_to_cities_scaling.py --uncertainty --plot --n_iterations 25
#   python scripts/Fig4_simulate_neighborhoods_to_cities_scaling.py --delta 0.75 --s_grid 0.5 4.0 15 --outfile results/emergence_simulation/beta_vs_s.csv --plot
# 

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

# Set up base directory and paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

def simulate_beta(delta: float, s: float, n_cities: int = 3000,
                  mean_neigh_size: int = 8000, seed: Optional[int] = 42) -> float:
    """Return simulated city‑level scaling exponent β for a given Zipf s.
    The function follows the neighbourhood aggregation Monte‑Carlo schema
    used in the companion notebook.

    Parameters
    ----------
    delta : float
        Neighbourhood‑level exponent (δ).
    s : float
        Zipf exponent governing intra‑city population hierarchy.
    n_cities : int, default 3000
        Number of cities to simulate.
    mean_neigh_size : int, default 8000
        Target mean population per neighbourhood for city generation.
    seed : int | None, default 42
        RNG seed for reproducibility.

    Returns
    -------
    float
        Fitted β.
    """
    rng = np.random.default_rng(seed)
    # draw city populations rng log‑uniform between 5e4 and 2e7
    city_pops = 10 ** rng.uniform(np.log10(5e4), np.log10(2e7), size=n_cities)
    city_masses = []

    for Pc in city_pops:
        n_cells = max(5, int(Pc / mean_neigh_size))
        ranks = np.arange(1, n_cells + 1)
        weights = 1 / ranks ** s
        neigh_pops = Pc * weights / weights.sum()
        city_masses.append((neigh_pops ** delta).sum())

    coeffs = np.polyfit(np.log10(city_pops), np.log10(city_masses), 1)
    beta = coeffs[0]
    return beta

def _process_single_iteration(iteration_data):
    """
    Process a single iteration for parallel execution.
    
    Parameters
    ----------
    iteration_data : tuple
        (iteration, delta, s, n_countries, total_cities, mean_neigh_size, seed)
    
    Returns
    -------
    list
        List of beta values for this iteration
    """
    iteration, delta, s, n_countries, total_cities, mean_neigh_size, seed = iteration_data
    
    # Use different seed for each iteration to ensure variation
    iteration_seed = None if seed is None else seed + iteration * 1000
    iteration_rng = np.random.default_rng(iteration_seed)
    
    # Randomly assign cities to countries for this iteration
    # Each country gets at least 3 cities, remaining cities distributed randomly
    min_cities_per_country = 3
    remaining_cities = total_cities - (n_countries * min_cities_per_country)
    
    # Use Dirichlet distribution to create realistic country size distribution
    alpha = np.ones(n_countries) * 0.1  # Creates power-law-like distribution
    country_weights = iteration_rng.dirichlet(alpha)
    additional_cities = (country_weights * remaining_cities).astype(int)
    
    # Ensure we use all cities
    cities_per_country = min_cities_per_country + additional_cities
    cities_per_country[-1] += total_cities - cities_per_country.sum()
    
    # Generate city populations for all cities at once for this iteration
    all_city_pops = 10 ** iteration_rng.uniform(np.log10(5e4), np.log10(2e7), size=total_cities)
    
    # Calculate beta for each country in this iteration
    iteration_betas = []
    city_idx = 0
    
    for n_cities in cities_per_country:
        if n_cities < 5:  # Skip countries with too few cities for reliable fitting
            continue
            
        # Get cities for this country
        country_city_pops = all_city_pops[city_idx:city_idx + n_cities]
        city_idx += n_cities
        
        # Calculate city masses for this country
        city_masses = []
        for Pc in country_city_pops:
            n_cells = max(5, int(Pc / mean_neigh_size))
            ranks = np.arange(1, n_cells + 1)
            weights = 1 / ranks ** s
            neigh_pops = Pc * weights / weights.sum()
            city_masses.append((neigh_pops ** delta).sum())
        
        # Fit beta for this country
        try:
            coeffs = np.polyfit(np.log10(country_city_pops), np.log10(city_masses), 1)
            beta = coeffs[0]
            iteration_betas.append(beta)
        except (np.linalg.LinAlgError, ValueError):
            # Skip if fitting fails (e.g., due to insufficient variation)
            continue
    
    return iteration_betas

def simulate_beta_with_uncertainty(delta: float, s: float, n_countries: int = 50,
                                   total_cities: int = 3000, mean_neigh_size: int = 8000,
                                   n_iterations: int = 10, seed: Optional[int] = 42) -> Tuple[float, float, float, np.ndarray]:
    """
    Simulate β with uncertainty by distributing cities across multiple countries
    and repeating the process multiple times using parallel processing.
    
    This function simulates the scaling exponent β by:
    1. Distributing cities across countries using a power-law distribution
    2. Calculating β for each country with sufficient cities
    3. Repeating this process multiple times to capture uncertainty
    4. Using parallel processing to speed up computation
    
    Parameters
    ----------
    delta : float
        Neighbourhood‑level exponent (δ).
    s : float
        Zipf exponent governing intra‑city population hierarchy.
    n_countries : int, default 50
        Number of countries to simulate per iteration.
    total_cities : int, default 3000
        Total number of cities to distribute across countries per iteration.
    mean_neigh_size : int, default 8000
        Target mean population per neighbourhood for city generation.
    n_iterations : int, default 10
        Number of times to repeat the city-to-country assignment process.
    seed : int | None, default 42
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        (mean_beta, std_beta, median_beta, all_betas)
        - mean_beta: Mean β across all countries and iterations
        - std_beta: Standard deviation of β across all countries and iterations
        - median_beta: Median β across all countries and iterations
        - all_betas: Array of all β values from each country in each iteration
    """
    # Calculate number of processes to use (preserve 2 CPUs)
    n_processes = max(1, cpu_count() - 2)
    
    # Prepare data for parallel processing
    iteration_data = [
        (iteration, delta, s, n_countries, total_cities, mean_neigh_size, seed)
        for iteration in range(n_iterations)
    ]
    
    # Use multiprocessing to parallelize the iterations
    with Pool(processes=n_processes) as pool:
        results = pool.map(_process_single_iteration, iteration_data)
    
    # Flatten the results
    all_country_betas = []
    for iteration_betas in results:
        all_country_betas.extend(iteration_betas)
    
    all_country_betas = np.array(all_country_betas)
    
    return (
        np.mean(all_country_betas),
        np.std(all_country_betas),
        np.median(all_country_betas),
        all_country_betas
    )

def _process_single_iteration_with_s_uncertainty(iteration_data):
    """
    Process a single iteration for parallel execution with uncertainty in s parameter.
    
    Parameters
    ----------
    iteration_data : tuple
        (iteration, delta, s, s_std, n_countries, total_cities, mean_neigh_size, seed)
    
    Returns
    -------
    list
        List of beta values for this iteration
    """
    iteration, delta, s, s_std, n_countries, total_cities, mean_neigh_size, seed = iteration_data
    
    # Use different seed for each iteration to ensure variation
    iteration_seed = None if seed is None else seed + iteration * 1000
    iteration_rng = np.random.default_rng(iteration_seed)
    
    # Randomly assign cities to countries for this iteration
    # Each country gets at least 3 cities, remaining cities distributed randomly
    min_cities_per_country = 3
    remaining_cities = total_cities - (n_countries * min_cities_per_country)
    
    # Use Dirichlet distribution to create realistic country size distribution
    alpha = np.ones(n_countries) * 0.1  # Creates power-law-like distribution
    country_weights = iteration_rng.dirichlet(alpha)
    additional_cities = (country_weights * remaining_cities).astype(int)
    
    # Ensure we use all cities
    cities_per_country = min_cities_per_country + additional_cities
    cities_per_country[-1] += total_cities - cities_per_country.sum()
    
    # Generate city populations for all cities at once for this iteration
    all_city_pops = 10 ** iteration_rng.uniform(np.log10(5e4), np.log10(2e7), size=total_cities)
    
    # Calculate beta for each country in this iteration
    iteration_betas = []
    city_idx = 0
    
    for n_cities in cities_per_country:
        if n_cities < 5:  # Skip countries with too few cities for reliable fitting
            continue
            
        # Get cities for this country
        country_city_pops = all_city_pops[city_idx:city_idx + n_cities]
        city_idx += n_cities
        
        # Calculate city masses for this country
        city_masses = []
        for Pc in country_city_pops:
            # Draw random s value from normal distribution for each city
            s_random = iteration_rng.normal(s, s_std)
            # Ensure s_random is positive (Zipf exponent should be positive)
            s_random = max(sys.float_info.min, s_random)
            
            n_cells = max(5, int(Pc / mean_neigh_size))
            ranks = np.arange(1, n_cells + 1)
            weights = 1 / ranks ** s_random
            neigh_pops = Pc * weights / weights.sum()
            city_masses.append((neigh_pops ** delta).sum())
        
        # Fit beta for this country
        try:
            coeffs = np.polyfit(np.log10(country_city_pops), np.log10(city_masses), 1)
            beta = coeffs[0]
            iteration_betas.append(beta)
        except (np.linalg.LinAlgError, ValueError):
            # Skip if fitting fails (e.g., due to insufficient variation)
            continue
    
    return iteration_betas

def simulate_beta_with_s_uncertainty(delta: float, s: float, s_std: float = 0.1, n_countries: int = 50,
                                     total_cities: int = 3000, mean_neigh_size: int = 8000,
                                     n_iterations: int = 10, seed: Optional[int] = 42) -> Tuple[float, float, float, np.ndarray]:
    """
    Simulate β with uncertainty by distributing cities across multiple countries
    and repeating the process multiple times using parallel processing.
    This version includes uncertainty in the s parameter by drawing from N(s, s_std).
    
    This function simulates the scaling exponent β by:
    1. Distributing cities across countries using a power-law distribution
    2. For each city, drawing a random s value from N(s, s_std)
    3. Calculating β for each country with sufficient cities
    4. Repeating this process multiple times to capture uncertainty
    5. Using parallel processing to speed up computation
    
    Parameters
    ----------
    delta : float
        Neighbourhood‑level exponent (δ).
    s : float
        Mean Zipf exponent governing intra‑city population hierarchy.
    s_std : float, default 0.1
        Standard deviation for the normal distribution of s values.
    n_countries : int, default 50
        Number of countries to simulate per iteration.
    total_cities : int, default 3000
        Total number of cities to distribute across countries per iteration.
    mean_neigh_size : int, default 8000
        Target mean population per neighbourhood for city generation.
    n_iterations : int, default 10
        Number of times to repeat the city-to-country assignment process.
    seed : int | None, default 42
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        (mean_beta, std_beta, median_beta, all_betas)
        - mean_beta: Mean β across all countries and iterations
        - std_beta: Standard deviation of β across all countries and iterations
        - median_beta: Median β across all countries and iterations
        - all_betas: Array of all β values from each country in each iteration
    """
    # Calculate number of processes to use (preserve 2 CPUs)
    n_processes = max(1, cpu_count() - 2)
    
    # Prepare data for parallel processing
    iteration_data = [
        (iteration, delta, s, s_std, n_countries, total_cities, mean_neigh_size, seed)
        for iteration in range(n_iterations)
    ]
    
    # Use multiprocessing to parallelize the iterations
    with Pool(processes=n_processes) as pool:
        results = pool.map(_process_single_iteration_with_s_uncertainty, iteration_data)
    
    # Flatten the results
    all_country_betas = []
    for iteration_betas in results:
        all_country_betas.extend(iteration_betas)
    
    all_country_betas = np.array(all_country_betas)
    
    return (
        np.mean(all_country_betas),
        np.std(all_country_betas),
        np.median(all_country_betas),
        all_country_betas
    )

def create_beta_plot(df, delta, ax=None, uncertainty=False, s_uncertainty=False, observed_s=None, observed_beta=None, figsize=(8, 6)):
    """
    Create a plot of beta vs s values with optional uncertainty bands and observed data point.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the beta data. Should have columns:
        - 's': s values
        - 'beta': beta values (for uncertainty=False)
        - 'beta_mean', 'beta_lower_99', 'beta_upper_99': for uncertainty=True
        - 'beta_mean_s_unc', 'beta_std_s_unc': for s_uncertainty=True
    delta : float
        The delta value for horizontal reference line
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, creates a new figure and axes.
    uncertainty : bool, default False
        Whether to plot uncertainty bands (requires uncertainty columns in df)
    s_uncertainty : bool, default False
        Whether to plot uncertainty from s parameter uncertainty (requires s_uncertainty columns in df)
    observed_s : float, optional
        S value for observed data point
    observed_beta : float, optional
        Beta value for observed data point
    figsize : tuple, default (8, 6)
        Figure size as (width, height), only used if ax is None
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object (if ax is None) or None (if ax is provided)
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None
        plt.sca(ax)
    
    if s_uncertainty:
        # Plot with s-parameter uncertainty bands
        # Calculate 25-75% uncertainty (interquartile range) assuming normal distribution
        beta_lower_25_s = df["beta_mean_s_unc"] - 0.674 * df["beta_std_s_unc"]
        beta_upper_75_s = df["beta_mean_s_unc"] + 0.674 * df["beta_std_s_unc"]
        
        ax.fill_between(df["s"], beta_lower_25_s, beta_upper_75_s, 
                       alpha=0.2, color='blue', label='25th-75th Percentile')            
        # ax.plot(df["s"], df["beta_mean_s_unc"], 'b-', linewidth=2, label='Theoretical β')
        ax.plot(df["s"], df["beta_median_s_unc"], 'b-', linewidth=2, label='Theoretical β')
    elif uncertainty:
        # Plot with uncertainty bands
        ax.fill_between(df["s"], df["beta_lower_99"], df["beta_upper_99"], 
                       alpha=0.2, color='blue', label='99% CI')            
        ax.plot(df["s"], df["beta_mean"], 'b-', linewidth=2, label='Theoretical β')
    else:
        # Original plot
        ax.plot(df["s"], df["beta"], 'b-', linewidth=2, label='β')
    
    # Add observed values as orange dot if provided
    if observed_s is not None and observed_beta is not None:
        ax.scatter(observed_s, observed_beta, color='orange', s=100, zorder=5, 
                  label='Observed')
        # Add text annotation to the lower left of the dot
        offset = -0.1  # Try negative for left
        ax.text(observed_s+offset, observed_beta, f'  (s={observed_s}, β={observed_beta})', 
                fontsize=10, ha='right', va='bottom')
    
    ax.axhline(delta, color='orange', linestyle='--', linewidth=2)
    ax.axhline(1, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel("Across-neighborhood Disparity (s)")
    ax.set_ylabel("Across-city Scaling Coef β")
    
    # Calculate 7% of the space between 1 and delta for padding
    y_range = abs(1 - delta)
    padding = 0.07 * y_range
    ax.set_ylim(delta - padding, 1 + padding)
    ax.set_xscale("log", base=2)
    ax.legend(loc='upper right', bbox_to_anchor=(1, .9))
    
    if fig is not None:
        fig.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Monte‑Carlo exploration of β(s)")
    parser.add_argument("--delta", type=float, default=0.75,
                        help="Neighbourhood‑level scaling exponent δ (default 0.75)")
    parser.add_argument("--s_grid", type=float, nargs=3, metavar=("START", "STOP", "NUM"),
                        default=(0.33, 4.0, 50),
                        help="Range and number of s values: start stop num")
    parser.add_argument("--outfile", type=Path, default=OUTPUT_DIR / "beta_vs_s.csv",
                        help="CSV to save β(s) table")
    parser.add_argument("--plot", action="store_true", help="Produce a PNG plot")
    parser.add_argument("--uncertainty", action="store_true", 
                        help="Include uncertainty analysis with multiple countries")
    parser.add_argument("--s_uncertainty", action="store_true",
                        help="Include uncertainty analysis for the s parameter")
    parser.add_argument("--s_std", type=float, default=0.1,
                        help="Standard deviation for s uncertainty (default 0.1)")
    parser.add_argument("--n_countries", type=int, default=200,
                        help="Number of countries for uncertainty analysis (default 50)")
    parser.add_argument("--n_iterations", type=int, default=40,
                        help="Number of iterations for city-to-country assignment (default 10)")
    args = parser.parse_args()

    s_vals = np.logspace(np.log10(args.s_grid[0]), np.log10(args.s_grid[1]), int(args.s_grid[2]))

    if args.s_uncertainty:
        # Run s-uncertainty analysis
        print(f"Running s-uncertainty analysis with {args.n_iterations} iterations and s_std={args.s_std}...")
        mean_betas_s_unc = []
        std_betas_s_unc = []
        median_betas_s_unc = []
        all_country_betas_s_unc = []
        
        for i, s in enumerate(s_vals):
            print(f"Processing s = {s:.3f} ({i+1}/{len(s_vals)})")
            mean_beta, std_beta, median_beta, country_betas = simulate_beta_with_s_uncertainty(
                args.delta, s, s_std=args.s_std, n_countries=args.n_countries, 
                n_iterations=args.n_iterations, seed=42+i
            )
            mean_betas_s_unc.append(mean_beta)
            std_betas_s_unc.append(std_beta)
            median_betas_s_unc.append(median_beta)
            all_country_betas_s_unc.append(country_betas)
        
        # Create comprehensive dataframe for s-uncertainty
        df = pd.DataFrame({
            "s": s_vals,
            "beta_mean_s_unc": mean_betas_s_unc,
            "beta_std_s_unc": std_betas_s_unc,
            "beta_median_s_unc": median_betas_s_unc,
            "beta_lower_99_s_unc": [np.percentile(betas, 0.5) for betas in all_country_betas_s_unc],
            "beta_upper_99_s_unc": [np.percentile(betas, 99.5) for betas in all_country_betas_s_unc],
            "beta_lower_95_s_unc": [np.percentile(betas, 2.5) for betas in all_country_betas_s_unc],
            "beta_upper_95_s_unc": [np.percentile(betas, 97.5) for betas in all_country_betas_s_unc], 
            "beta_lower_68_s_unc": [np.percentile(betas, 16) for betas in all_country_betas_s_unc],
            "beta_upper_68_s_unc": [np.percentile(betas, 84) for betas in all_country_betas_s_unc]
        })
        
        # Save detailed results
        s_uncertainty_file = args.outfile.with_name(args.outfile.stem + "_s_uncertainty.csv")
        df.to_csv(s_uncertainty_file, index=False)
        print(f"Saved s-uncertainty analysis to {s_uncertainty_file}")
        
    elif args.uncertainty:
        # Run uncertainty analysis
        print(f"Running uncertainty analysis with {args.n_iterations} iterations...")
        mean_betas = []
        std_betas = []
        median_betas = []
        all_country_betas = []
        
        for i, s in enumerate(s_vals):
            print(f"Processing s = {s:.3f} ({i+1}/{len(s_vals)})")
            mean_beta, std_beta, median_beta, country_betas = simulate_beta_with_uncertainty(
                args.delta, s, n_countries=args.n_countries, n_iterations=args.n_iterations, seed=42+i
            )
            mean_betas.append(mean_beta)
            std_betas.append(std_beta)
            median_betas.append(median_beta)
            all_country_betas.append(country_betas)
        
        # Create comprehensive dataframe
        df = pd.DataFrame({
            "s": s_vals,
            "beta_mean": mean_betas,
            "beta_std": std_betas,
            "beta_median": median_betas,
            "beta_lower_99": [np.percentile(betas, 0.5) for betas in all_country_betas],
            "beta_upper_99": [np.percentile(betas, 99.5) for betas in all_country_betas],
            "beta_lower_95": [np.percentile(betas, 2.5) for betas in all_country_betas],
            "beta_upper_95": [np.percentile(betas, 97.5) for betas in all_country_betas], 
            "beta_lower_68": [np.percentile(betas, 16) for betas in all_country_betas],
            "beta_upper_68": [np.percentile(betas, 84) for betas in all_country_betas]
        })
        
        # Save detailed results
        uncertainty_file = args.outfile.with_name(args.outfile.stem + "_uncertainty.csv")
        df.to_csv(uncertainty_file, index=False)
        print(f"Saved uncertainty analysis to {uncertainty_file}")
        
    else:
        # Original single-beta analysis
        betas = [simulate_beta(args.delta, s) for s in s_vals]
        df = pd.DataFrame({"s": s_vals, "beta": betas})
    
    # Always save the basic results
    if not args.uncertainty and not args.s_uncertainty:
        df.to_csv(args.outfile, index=False)
        print(f"Saved β(s) table to {args.outfile}")

    if args.plot:
        # Create the plot using the new function
        fig = create_beta_plot(
            df=df, 
            delta=args.delta, 
            uncertainty=args.uncertainty,
            s_uncertainty=args.s_uncertainty,
            observed_s=1.16,
            observed_beta=0.90
        )
        
        # Save the figure
        if args.s_uncertainty:
            suffix = "_s_uncertainty"
        elif args.uncertainty:
            suffix = "_uncertainty"
        else:
            suffix = ""
        png = args.outfile.with_name(args.outfile.stem + suffix + ".png")
        fig.savefig(png, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {png}")
        # plt.show()

if __name__ == "__main__":
    main()