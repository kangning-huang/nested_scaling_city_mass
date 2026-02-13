import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

# Add the scripts directory to the python path to import the functions
scripts_dir = Path(__file__).parent.absolute()
sys.path.append(str(scripts_dir))

# Import the necessary functions from each script
from 06_estimate_neighborhood_zipf import plot_s_distribution, estimate_s_values_from_dataframe
from 08_compare_beta_boxplot import create_comparison_boxplot
from 07_simulate_scaling import create_beta_plot, simulate_beta_with_s_uncertainty

def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Define output path for the combined plot
    output_plot_path = figures_dir / 'Fig4_combined_plots.pdf'

    # Set up the figure and subplot grid
    # We want a 2x2 grid, but the right column will be spanned by subplot C
    # Width ratio: 1:2 for left:right columns
    # Height ratio: 1:1 for top:bottom rows
    fig_height = 6
    fig = plt.figure(figsize=(fig_height*1.618, fig_height))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])

    # Subplot A: s distribution (top-left)
    ax_a = fig.add_subplot(gs[0, 0])
    # Load data for subplot A
    csv_path_a = project_root / "data" / "processed" / "mass_avg_Li2022_vs_Liu2024_H3_grids_world.csv"
    df_a = pd.read_csv(csv_path_a, usecols=["city_id", "population"])
    s_df = estimate_s_values_from_dataframe(df_a, "city_id", "population")
    plot_s_distribution(s_df, ax=ax_a)
    # Add bold letter 'a' to upper left outside of plot
    ax_a.text(-0.15, 1.1, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold', 
              verticalalignment='top', horizontalalignment='left')

    # Subplot B: Observed vs Simulated Beta Boxplots (bottom-left)
    ax_b = fig.add_subplot(gs[1, 0])
    # Load observed data for subplot B
    observed_data_path_b = project_root / 'data' / 'processed' / 'country_inset_slope.csv'
    country_summary_b = pd.read_csv(observed_data_path_b)
    observed_betas_b = country_summary_b['Slope'].dropna().values
    # Simulate data for subplot B
    s_value_b = 1.16
    s_std_value_b = 0.59
    _, _, _, simulated_betas_b = simulate_beta_with_s_uncertainty(
        delta=0.75, s=s_value_b, s_std=s_std_value_b, n_countries=200, n_iterations=5, seed=42
    )
    create_comparison_boxplot(observed_betas_b, simulated_betas_b, ax=ax_b)
    # Add bold letter 'b' to upper left outside of plot
    ax_b.text(-0.15, 1.1, 'B', transform=ax_b.transAxes, fontsize=14, fontweight='bold', 
              verticalalignment='top', horizontalalignment='left')

    # Subplot C: Curve plot with s uncertainty (spans right column)
    ax_c = fig.add_subplot(gs[:, 1])  # Spans both rows in the second column
    # Generate data for subplot C
    delta_c = 0.75
    s_grid_c = (0.33, 4.0, 50)
    s_vals_c = np.logspace(np.log10(s_grid_c[0]), np.log10(s_grid_c[1]), int(s_grid_c[2]))
    s_std_c = 0.59  # Default from Fig4_simulate_neighborhoods_to_cities_scaling.py
    n_countries_c = 200  # Default from Fig4_simulate_neighborhoods_to_cities_scaling.py
    n_iterations_c = 50  # Default from Fig4_simulate_neighborhoods_to_cities_scaling.py

    # Calculate beta values for each s value
    mean_betas_s_unc_c = []
    std_betas_s_unc_c = []
    median_betas_s_unc_c = []
    all_country_betas_s_unc_c = []

    for i, s_val in enumerate(tqdm(s_vals_c, desc="Simulating beta values")):
        mean_beta, std_beta, median_beta, country_betas = simulate_beta_with_s_uncertainty(
            delta_c, s_val, s_std=s_std_c, n_countries=n_countries_c,
            n_iterations=n_iterations_c, seed=42+i
        )
        mean_betas_s_unc_c.append(mean_beta)
        std_betas_s_unc_c.append(std_beta)
        median_betas_s_unc_c.append(median_beta)
        all_country_betas_s_unc_c.append(country_betas)

    # Create dataframe for plotting
    df_c = pd.DataFrame({
        "s": s_vals_c,
        "beta_mean_s_unc": mean_betas_s_unc_c,
        "beta_std_s_unc": std_betas_s_unc_c,
        "beta_median_s_unc": median_betas_s_unc_c,
        "beta_lower_99_s_unc": [np.percentile(betas, 0.5) for betas in all_country_betas_s_unc_c],
        "beta_upper_99_s_unc": [np.percentile(betas, 99.5) for betas in all_country_betas_s_unc_c],
    })

    # Observed values for subplot C
    observed_s_c = 1.16  # Mean s value from the distribution
    observed_beta_c = 0.90  # Example observed beta value (adjust as needed)

    # Create the beta plot with s uncertainty
    create_beta_plot(df_c, delta_c, ax=ax_c, s_uncertainty=True, 
                     observed_s=observed_s_c, observed_beta=observed_beta_c)
    # Add bold letter 'c' to upper left outside of plot
    ax_c.text(-0.1, 1.05, 'C', transform=ax_c.transAxes, fontsize=14, fontweight='bold', 
              verticalalignment='top', horizontalalignment='left')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_plot_path}")

    # Also save as PNG format
    output_png_path = output_plot_path.with_suffix('.png')
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot also saved as {output_png_path}")

if __name__ == '__main__':
    main()