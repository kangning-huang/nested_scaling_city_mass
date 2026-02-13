import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from utils.paths import get_resolution_dir, get_latest_file

# Set up base directory and paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def estimate_zipf_s(values: np.ndarray, min_cells=5):
    """
    Estimate Zipf exponent from population values.
    
    Args:
        values (np.ndarray): Population values
        min_cells (int): Minimum number of cells required for estimation
        
    Returns:
        tuple: (s_estimate, n_cells) where s_estimate is the Zipf exponent
               and n_cells is the number of cells used
    """
    vals = values[values > 0]
    n = len(vals)
    if n < min_cells:
        return np.nan, n
    sorted_vals = np.sort(vals)[::-1]
    ranks = np.arange(1, n + 1)
    slope, _ = np.polyfit(np.log(ranks), np.log(sorted_vals), 1)
    return -slope, n

def estimate_s_values_from_dataframe(df, city_col, pop_col):
    """
    Estimate s values from city-neighborhood dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with city and population data
        city_col (str): Name of the city identifier column
        pop_col (str): Name of the population column
        
    Returns:
        pd.DataFrame: DataFrame with columns ['city_id', 'n_cells', 's_estimate', 'population']
    """
    results = []
    for cid, grp in df.groupby(city_col):
        s_val, n_cells = estimate_zipf_s(grp[pop_col].to_numpy(dtype=float))
        if not np.isnan(s_val):
            total_population = grp[pop_col].sum()
            results.append({
                "city_id": cid, 
                "n_cells": n_cells, 
                "s_estimate": s_val,
                "population": total_population
            })
    
    return pd.DataFrame(results)

def plot_s_distribution(s_df, ax=None, figsize=(6, 5)):
    """
    Plot the distribution of s values.
    
    Args:
        s_df (pd.DataFrame): DataFrame with s_estimate column
        ax (matplotlib.axes.Axes, optional): Axis object to plot on. If None, creates a new figure and axis.
        figsize (tuple): Figure size as (width, height), only used when ax is None
        
    Returns:
        matplotlib.axes.Axes: The axis object containing the plot
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    
    ax.hist(s_df["s_estimate"], bins=40, edgecolor="black", color="lightgrey")
    ax.set_xlabel("Across-neighborhood Disparity (s)")
    ax.set_ylabel("Number of cities")
    ax.set_xlim(0, 3)

    # ax.set_xlim(0.33, 4)
    # ax.set_xscale('log', base=2)
    
    # Add vertical line for mean value
    mean_s = s_df["s_estimate"].mean()
    ax.axvline(mean_s, color='orange', linestyle='--', linewidth=2, label=f'Mean = {mean_s:.3f}')
    ax.legend()
    
    return ax

def main():
    """
    Main function to run the Zipf analysis when script is executed directly.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Estimate Zipf distribution parameters from neighborhood data')
    parser.add_argument('--resolution', '-r', type=int, default=6,
                        help='H3 resolution level (default: 6)')
    parser.add_argument('--input', type=str, default=None,
                        help='Optional input CSV path. Defaults to latest mass file in resolution directory.')
    args = parser.parse_args()
    resolution = args.resolution

    # Set up resolution-specific directories
    DATA_DIR = get_resolution_dir(PROCESSED_DIR, resolution)
    OUTPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution)

    # Create output directory
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"H3 Resolution: {resolution}")

    # Determine input file
    if args.input:
        csv_path = Path(args.input)
    else:
        # Try to find the mass file in the resolution-specific directory
        try:
            csv_path = get_latest_file(DATA_DIR, f"Fig3_Mass_Neighborhood_H3_Resolution{resolution}_*.csv")
        except FileNotFoundError:
            # Fallback to the non-resolution-specific file in the main processed directory
            csv_path = PROCESSED_DIR / "mass_avg_Li2022_vs_Liu2024_H3_grids_world.csv"

    # Check if input file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"Required data file not found: {csv_path}")

    print(f"Using input file: {csv_path}")

    # peek to identify columns
    peek = pd.read_csv(csv_path, nrows=5)
    print("Columns in file:", peek.columns.tolist())

    # heuristic: pick city id column
    city_candidates = [c for c in peek.columns if c.lower() in {"city", "city_id", "id_hdc_g0", "city_gid"}]
    if not city_candidates:
        # try columns containing 'city' and ending with '_id'
        city_candidates = [c for c in peek.columns if "city" in c.lower() and c.lower().endswith("_id")]
    if not city_candidates:
        raise ValueError("Cannot find city identifier column automatically. Please specify.")

    city_col = city_candidates[0]
    print("Using city identifier column:", city_col)

    # heuristic: population column
    pop_candidates = [c for c in peek.columns if "pop" in c.lower()]
    pop_candidates = [c for c in pop_candidates if not any(substr in c.lower() for substr in ["mass", "volume"])]
    if not pop_candidates:
        raise ValueError("Cannot find population column automatically. Please specify.")

    pop_col = pop_candidates[0]
    print("Using population column:", pop_col)

    # load only needed columns
    df = pd.read_csv(csv_path, usecols=[city_col, pop_col])

    # Use the new function to estimate s values
    s_df = estimate_s_values_from_dataframe(df, city_col, pop_col)

    # Save the Zipf estimates dataframe as CSV
    s_df.to_csv(os.path.join(output_dir, f"zipf_estimates_resolution{resolution}.csv"), index=False)
    print(f"Zipf estimates saved to {os.path.join(output_dir, f'zipf_estimates_resolution{resolution}.csv')}")

    # summary
    summary = s_df["s_estimate"].describe(percentiles=[0.25, 0.5, 0.75]).round(3)
    print("Summary statistics:\n", summary)

    # Log-normal distribution statistics
    # Assuming s follows a log-normal distribution, calculate statistics
    s_values = s_df["s_estimate"].dropna()
    if len(s_values) > 0:
        # Calculate log-normal parameters
        log_s = np.log(s_values)
        mu = np.mean(log_s)  # Mean of log(s)
        sigma = np.std(log_s, ddof=1)  # Standard deviation of log(s)
        
        # Log-normal distribution statistics
        lognorm_mean = np.exp(mu + sigma**2 / 2)
        lognorm_median = np.exp(mu)
        lognorm_std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))
        
        print(f"\nLog-normal distribution statistics (assuming s ~ LogNormal):")
        print(f"  Estimated mean: {lognorm_mean:.3f}")
        print(f"  Estimated median: {lognorm_median:.3f}")
        print(f"  Estimated std: {lognorm_std:.3f}")
        print(f"  Underlying normal parameters: μ={mu:.3f}, σ={sigma:.3f}")
        
        # Add log-normal statistics to summary dataframe
        lognorm_stats = pd.DataFrame({
            'statistic': ['lognorm_mean', 'lognorm_median', 'lognorm_std', 'lognorm_mu', 'lognorm_sigma'],
            'value': [lognorm_mean, lognorm_median, lognorm_std, mu, sigma]
        })
        
        # Combine with existing summary
        summary_df = pd.DataFrame(summary).reset_index()
        summary_df.columns = ['statistic', 'value']
        summary_df = pd.concat([summary_df, lognorm_stats], ignore_index=True)
    else:
        summary_df = pd.DataFrame(summary).reset_index()
        summary_df.columns = ['statistic', 'value']

    # Save summary statistics as CSV
    summary_df = pd.DataFrame(summary).reset_index()
    summary_df.columns = ['statistic', 'value']
    summary_df.to_csv(os.path.join(output_dir, f"zipf_summary_statistics_resolution{resolution}.csv"), index=False)
    print(f"Summary statistics saved to {os.path.join(output_dir, f'zipf_summary_statistics_resolution{resolution}.csv')}")

    # Use the new function to create the plot
    ax = plot_s_distribution(s_df)

    # Save plot as PDF - get the figure from the axes
    plot_path = os.path.join(output_dir, f"zipf_distribution_resolution{resolution}.pdf")
    ax.figure.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved to {plot_path}")

    # plt.show()

if __name__ == '__main__':
    main()