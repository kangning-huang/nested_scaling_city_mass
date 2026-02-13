import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the scripts directory to the python path to import the simulation function
scripts_dir = Path(__file__).parent.absolute()
sys.path.append(str(scripts_dir))

from 07_simulate_scaling import simulate_beta_with_s_uncertainty

def create_comparison_boxplot(observed_betas, simulated_betas, ax=None, outfile=None):
    """
    Creates a boxplot comparing observed and simulated beta values.

    Parameters
    ----------
    observed_betas : np.ndarray
        Array of observed beta values.
    simulated_betas : np.ndarray
        Array of simulated beta values.
    ax : matplotlib.axes.Axes, optional
        Axis object to plot on. If None, creates a new figure and axis.
    outfile : Path, optional
        Path to save the output plot. If provided, the plot will be saved.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    data = [observed_betas, simulated_betas]
    labels = ['Observed', 'Simulated']

    # Boxplot properties
    boxprops = dict(facecolor='none', edgecolor='black')
    medianprops = dict(color='black')

    # Create the boxplots
    bp = ax.boxplot(data, labels=labels, patch_artist=True, 
                    boxprops=boxprops, medianprops=medianprops,
                    showfliers=False) # Do not show outlier markers

    # Jittered points
    colors = ['orange', 'blue']
    transparency = [0.5, 0.05]
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=transparency[i], color=colors[i], label=labels[i])

    ax.set_ylabel('Scaling Exponent β')
    # ax.set_title('Comparison of Observed and Simulated β')
    ax.set_ylim([0.6, 1.1])
    
    # Save the plot if outfile is provided
    if outfile is not None:
        fig.savefig(outfile, dpi=300)
        print(f"Plot saved to {outfile}")
    
    return ax

def main():
    """Main function to run the script."""
    # Define file paths
    project_root = Path(__file__).parent.parent
    observed_data_path = project_root / 'results' / 'global_scaling' / 'country_summary_table.csv'
    output_plot_path = project_root / 'results'  / 'emergence_simulation' / 'Fig4_observed_vs_simulated_beta_boxplot.png'

    # 1. Load observed data
    print(f"Loading observed data from {observed_data_path}")
    if not observed_data_path.exists():
        print(f"Error: Observed data file not found at {observed_data_path}")
        return
    
    country_summary = pd.read_csv(observed_data_path)
    observed_betas = country_summary['Slope'].dropna().values
    print(f"Found {len(observed_betas)} observed beta values.")

    # 2. Generate simulated data
    s_value = 1.16
    s_std_value = 0.59
    print(f"Simulating beta values with s = {s_value}...")
    # Using parameters from the original script for consistency
    _, _, _, simulated_betas = simulate_beta_with_s_uncertainty(
        delta=0.75, 
        s=s_value, 
        s_std=s_std_value, 
        n_countries=75, 
        n_iterations=10, 
        seed=42
    )
    print(f"Generated {len(simulated_betas)} simulated beta values.")

    # 3. Create and save the plot
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_boxplot(observed_betas, simulated_betas, outfile=output_plot_path)

if __name__ == '__main__':
    main()