#!/usr/bin/env python3
"""
03_global_neighborhood_scaling_sndi_classification.py

Global analysis of neighborhood-level scaling patterns based on Street Network
Disconnectedness Index (SNDi) classification.

This script investigates whether neighborhoods with compact street networks (SNDi ≤ 2)
show different scaling relationships compared to sprawl neighborhoods (SNDi ≥ 5.5)
across global cities.

SNDi Definition (from PNAS 2019):
- Lower SNDi values indicate more connected, compact street networks
- Higher SNDi values indicate more disconnected, sprawling street patterns
- Threshold values: Compact (≤2), Intermediate (2-5.5), Sprawl (≥5.5)

This script:
1. Loads global neighborhood mass data and SNDi values
2. Classifies neighborhoods by SNDi connectivity patterns
3. Performs separate scaling analyses for compact vs sprawl neighborhoods
4. Compares scaling slopes and intercepts between groups
5. Tests for regional/country-specific differences
6. Creates comprehensive visualizations and statistical comparisons

Research Questions:
- Do compact vs sprawl neighborhoods show consistent scaling differences globally?
- Are these patterns universal or context-dependent by region/development level?
- How much scaling variation can be explained by street network connectivity?

Author: Generated for NYU China Grant project
Date: 2025
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
# seaborn not available in requirements, using matplotlib styling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries for advanced analysis
try:
    from scipy.stats import bootstrap
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class GlobalSNDiScalingAnalyzer:
    """
    Analyzer for global neighborhood scaling patterns based on SNDi classification.
    """

    def __init__(self, base_path: str):
        """
        Initialize the global SNDi scaling analyzer.

        Args:
            base_path (str): Base path to the project directory
        """
        self.project_path = Path(base_path)
        self.data_dir = self.project_path / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.figures_dir = self.project_path / 'figures'

        # For global mass data, we still need to reference the main project
        self.main_project_dir = self.project_path.parent
        self.global_data_dir = self.main_project_dir / 'results' / 'global_scaling'

        # Create directories if they don't exist
        self.processed_data_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

        # SNDi classification thresholds based on literature
        self.sndi_thresholds = {
            'compact': 2.0,      # Well-connected street networks
            'sprawl': 5.5        # Disconnected, sprawling patterns
        }

        # Set up plotting style
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3
        })

        log.info("GlobalSNDiScalingAnalyzer initialized")
        log.info(f"Results directory: {self.processed_data_dir}")
        log.info(f"Figures directory: {self.figures_dir}")
        log.info(f"SNDi thresholds - Compact: ≤{self.sndi_thresholds['compact']}, Sprawl: ≥{self.sndi_thresholds['sprawl']}")

    def load_global_data(self) -> pd.DataFrame:
        """
        Load and merge global neighborhood mass data with SNDi values.

        Returns:
            pd.DataFrame: Merged global dataset with SNDi classifications
        """
        log.info("Loading global neighborhood data...")

        # Load the most recent global mass data
        mass_file = self.global_data_dir / 'Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv'
        if not mass_file.exists():
            raise FileNotFoundError(f"Global mass data file not found: {mass_file}")

        global_mass = pd.read_csv(mass_file)
        log.info(f"Loaded {len(global_mass)} neighborhoods with global mass data")

        # Load SNDi data - check for existing global SNDi file or create from China data
        sndi_file = self.processed_data_dir / '01_neighborhood_SNDi_2025-09-09.csv'
        if not sndi_file.exists():
            raise FileNotFoundError(f"SNDi data file not found: {sndi_file}")

        sndi_data = pd.read_csv(sndi_file)
        log.info(f"Loaded {len(sndi_data)} neighborhoods with SNDi data")

        # Check if we need to expand SNDi calculation to global dataset
        unique_countries_mass = set(global_mass['CTR_MN_ISO'].unique()) if 'CTR_MN_ISO' in global_mass.columns else set()
        unique_countries_sndi = set(sndi_data['CTR_MN_ISO'].unique()) if 'CTR_MN_ISO' in sndi_data.columns else set()

        log.info(f"Countries in mass data: {len(unique_countries_mass)}")
        log.info(f"Countries in SNDi data: {len(unique_countries_sndi)}")

        if len(unique_countries_sndi) < len(unique_countries_mass):
            log.warning("SNDi data covers fewer countries than mass data. Analysis will be limited to SNDi-covered areas.")

        # Merge datasets on h3index
        merged_data = pd.merge(
            global_mass,
            sndi_data[['h3index', 'avg_sndi', 'sndi_point_count']],
            on='h3index',
            how='inner'
        )

        log.info(f"Successfully merged data: {len(merged_data)} neighborhoods with both mass and SNDi data")

        # Filter for valid data
        initial_count = len(merged_data)
        merged_data = merged_data.dropna(subset=['avg_sndi', 'population_2015', 'total_built_mass_tons'])
        merged_data = merged_data[
            (merged_data['avg_sndi'] > 0) &
            (merged_data['population_2015'] > 0) &
            (merged_data['total_built_mass_tons'] > 0) &
            (merged_data['sndi_point_count'] > 0)  # Ensure we have actual SNDi measurements
        ]

        after_filter = len(merged_data)
        log.info(f"After filtering: {after_filter} valid neighborhoods ({initial_count - after_filter} removed)")

        return merged_data

    def classify_sndi_neighborhoods(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify neighborhoods based on SNDi values.

        Args:
            data (pd.DataFrame): Merged dataset with SNDi values

        Returns:
            pd.DataFrame: Data with SNDi classifications added
        """
        log.info("Classifying neighborhoods by SNDi connectivity patterns...")

        # Create SNDi classification
        conditions = [
            data['avg_sndi'] <= self.sndi_thresholds['compact'],
            data['avg_sndi'] >= self.sndi_thresholds['sprawl']
        ]
        choices = ['Compact', 'Sprawl']

        data['sndi_category'] = np.select(conditions, choices, default='Intermediate')

        # Log classification results
        classification_counts = data['sndi_category'].value_counts()
        log.info("SNDi Classification Results:")
        for category, count in classification_counts.items():
            percentage = count / len(data) * 100
            if category == 'Compact':
                sndi_range = f"≤{self.sndi_thresholds['compact']}"
            elif category == 'Sprawl':
                sndi_range = f"≥{self.sndi_thresholds['sprawl']}"
            else:
                sndi_range = f"{self.sndi_thresholds['compact']}-{self.sndi_thresholds['sprawl']}"

            log.info(f"  {category} (SNDi {sndi_range}): {count:,} neighborhoods ({percentage:.1f}%)")

        # Add regional information for analysis
        if 'CTR_MN_ISO' in data.columns:
            # Map countries to regions (simplified)
            region_mapping = {
                'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
                'CHN': 'East Asia', 'JPN': 'East Asia', 'KOR': 'East Asia',
                'IND': 'South Asia', 'PAK': 'South Asia', 'BGD': 'South Asia',
                'DEU': 'Europe', 'FRA': 'Europe', 'GBR': 'Europe', 'ITA': 'Europe', 'ESP': 'Europe',
                'NLD': 'Europe', 'BEL': 'Europe', 'POL': 'Europe', 'NOR': 'Europe', 'SWE': 'Europe',
                'BRA': 'South America', 'ARG': 'South America', 'COL': 'South America',
                'AUS': 'Oceania', 'NZL': 'Oceania'
            }

            data['region'] = data['CTR_MN_ISO'].map(region_mapping).fillna('Other')

            # Log regional distribution
            regional_counts = data.groupby(['region', 'sndi_category']).size().unstack(fill_value=0)
            log.info("Regional distribution of SNDi categories:")
            log.info(f"\n{regional_counts}")

        return data

    def perform_scaling_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Perform scaling analysis comparing compact vs sprawl neighborhoods.

        Args:
            data (pd.DataFrame): Classified neighborhood data

        Returns:
            Dict: Scaling analysis results
        """
        log.info("Performing scaling analysis by SNDi category...")

        # Prepare data for scaling analysis
        scaling_data = data.copy()
        scaling_data['log10_population'] = np.log10(scaling_data['population_2015'])
        scaling_data['log10_total_mass'] = np.log10(scaling_data['total_built_mass_tons'])

        # Also prepare other mass types
        if 'BuildingMass_AverageTotal' in scaling_data.columns:
            scaling_data = scaling_data[scaling_data['BuildingMass_AverageTotal'] > 0]
            scaling_data['log10_building_mass'] = np.log10(scaling_data['BuildingMass_AverageTotal'])

        if 'mobility_mass_tons' in scaling_data.columns:
            mobility_valid = scaling_data[scaling_data['mobility_mass_tons'] > 0]
            if len(mobility_valid) > 0:
                scaling_data.loc[scaling_data['mobility_mass_tons'] > 0, 'log10_mobility_mass'] = np.log10(
                    scaling_data.loc[scaling_data['mobility_mass_tons'] > 0, 'mobility_mass_tons']
                )

        results = {}

        # Mass types to analyze
        mass_types = {
            'total_built_mass': ('log10_total_mass', 'Total Built Mass'),
            'building_mass': ('log10_building_mass', 'Building Mass'),
            'mobility_mass': ('log10_mobility_mass', 'Mobility Mass')
        }

        # Analyze each mass type
        for mass_key, (log_column, mass_name) in mass_types.items():
            if log_column not in scaling_data.columns:
                log.warning(f"Skipping {mass_name} - data not available")
                continue

            results[mass_key] = {}

            # Analyze by SNDi category
            for category in ['Compact', 'Sprawl']:
                category_data = scaling_data[
                    (scaling_data['sndi_category'] == category) &
                    (scaling_data[log_column].notna())
                ]

                if len(category_data) < 10:
                    log.warning(f"Insufficient data for {category} neighborhoods in {mass_name}: {len(category_data)} points")
                    continue

                # Perform regression
                X = category_data['log10_population'].values.reshape(-1, 1)
                y = category_data[log_column].values

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)

                # Calculate additional statistics
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    category_data['log10_population'], category_data[log_column]
                )

                results[mass_key][category] = {
                    'n_points': len(category_data),
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'model': model,
                    'data': category_data
                }

                log.info(f"{mass_name} - {category}: n={len(category_data)}, slope={slope:.3f}, R²={r2:.3f}, p={p_value:.4f}")

        # Compare slopes between compact and sprawl
        log.info("Comparing scaling slopes between compact and sprawl neighborhoods...")

        for mass_key, mass_results in results.items():
            if 'Compact' in mass_results and 'Sprawl' in mass_results:
                compact_slope = mass_results['Compact']['slope']
                sprawl_slope = mass_results['Sprawl']['slope']

                # Perform t-test for slope difference (simplified)
                compact_data = mass_results['Compact']['data']
                sprawl_data = mass_results['Sprawl']['data']

                # Test for difference in residuals or direct slope comparison
                slope_diff = compact_slope - sprawl_slope

                log.info(f"{mass_types[mass_key][1]} slope comparison:")
                log.info(f"  Compact slope: {compact_slope:.4f}")
                log.info(f"  Sprawl slope: {sprawl_slope:.4f}")
                log.info(f"  Difference: {slope_diff:.4f}")

                results[mass_key]['slope_comparison'] = {
                    'compact_slope': compact_slope,
                    'sprawl_slope': sprawl_slope,
                    'difference': slope_diff,
                    'percent_difference': (slope_diff / sprawl_slope * 100) if sprawl_slope != 0 else np.nan
                }

        return results

    def regional_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Perform regional analysis of SNDi scaling patterns.

        Args:
            data (pd.DataFrame): Classified global data

        Returns:
            Dict: Regional analysis results
        """
        log.info("Performing regional analysis of SNDi scaling patterns...")

        if 'region' not in data.columns:
            log.warning("Regional information not available - skipping regional analysis")
            return {}

        regional_results = {}

        # Prepare data
        analysis_data = data.copy()
        analysis_data['log10_population'] = np.log10(analysis_data['population_2015'])
        analysis_data['log10_total_mass'] = np.log10(analysis_data['total_built_mass_tons'])

        # Analyze each region
        for region in analysis_data['region'].unique():
            if region == 'Other':
                continue

            region_data = analysis_data[analysis_data['region'] == region]

            if len(region_data) < 20:  # Minimum threshold for regional analysis
                continue

            regional_results[region] = {}

            # Count by SNDi category
            category_counts = region_data['sndi_category'].value_counts()
            regional_results[region]['category_counts'] = category_counts.to_dict()

            # Scaling analysis by category within region
            for category in ['Compact', 'Sprawl']:
                category_data = region_data[region_data['sndi_category'] == category]

                if len(category_data) < 5:
                    continue

                # Regression analysis
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    category_data['log10_population'], category_data['log10_total_mass']
                )

                regional_results[region][category] = {
                    'n_points': len(category_data),
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_err': std_err
                }

            log.info(f"Region {region}: {len(region_data)} neighborhoods, {len(category_counts)} SNDi categories")

        return regional_results

    def create_global_scatter_plot(self, data: pd.DataFrame, scaling_results: Dict):
        """
        Create global scatter plot showing scaling relationships by SNDi category.

        Args:
            data (pd.DataFrame): Classified data
            scaling_results (Dict): Scaling analysis results
        """
        log.info("Creating global scatter plot...")

        # Prepare data
        plot_data = data.copy()
        plot_data['log10_population'] = np.log10(plot_data['population_2015'])
        plot_data['log10_total_mass'] = np.log10(plot_data['total_built_mass_tons'])

        # Filter for main categories
        plot_data = plot_data[plot_data['sndi_category'].isin(['Compact', 'Sprawl'])]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Color scheme
        colors = {'Compact': '#2E86AB', 'Sprawl': '#A23B72'}

        # Scatter plot
        for category in ['Compact', 'Sprawl']:
            category_data = plot_data[plot_data['sndi_category'] == category]

            ax.scatter(
                category_data['log10_population'],
                category_data['log10_total_mass'],
                c=colors[category],
                alpha=0.6,
                s=20,
                label=f'{category} (n={len(category_data):,})',
                edgecolors='none'
            )

            # Add regression line
            if 'total_built_mass' in scaling_results and category in scaling_results['total_built_mass']:
                results = scaling_results['total_built_mass'][category]
                x_range = np.linspace(category_data['log10_population'].min(),
                                    category_data['log10_population'].max(), 100)
                y_pred = results['model'].predict(x_range.reshape(-1, 1))

                ax.plot(x_range, y_pred, color=colors[category], linewidth=2, alpha=0.8,
                       linestyle='-' if category == 'Compact' else '--')

        ax.set_xlabel('Log₁₀(Population 2015)')
        ax.set_ylabel('Log₁₀(Total Built Mass, tonnes)')
        ax.set_title('Global Neighborhood Scaling: Compact vs Sprawl Street Networks\n(Based on Street Network Disconnectedness Index)')
        ax.legend()

        # Add regression statistics
        if 'total_built_mass' in scaling_results:
            stats_text = []
            for category in ['Compact', 'Sprawl']:
                if category in scaling_results['total_built_mass']:
                    results = scaling_results['total_built_mass'][category]
                    stats_text.append(f"{category}: β={results['slope']:.3f}, R²={results['r_squared']:.3f}")

            if 'slope_comparison' in scaling_results['total_built_mass']:
                comp = scaling_results['total_built_mass']['slope_comparison']
                stats_text.append(f"Slope Δ: {comp['difference']:.3f} ({comp['percent_difference']:.1f}%)")

            ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.figures_dir / '03_global_sndi_scaling_scatter.png')
        plt.savefig(self.figures_dir / '03_global_sndi_scaling_scatter.pdf')
        plt.close()

        log.info("Saved global scatter plot")

    def create_regional_comparison(self, data: pd.DataFrame, regional_results: Dict):
        """
        Create regional comparison plots.

        Args:
            data (pd.DataFrame): Global data
            regional_results (Dict): Regional analysis results
        """
        if not regional_results:
            log.warning("No regional results available - skipping regional plots")
            return

        log.info("Creating regional comparison plots...")

        # Extract slope data for plotting
        regions = []
        compact_slopes = []
        sprawl_slopes = []

        for region, results in regional_results.items():
            if 'Compact' in results and 'Sprawl' in results:
                regions.append(region)
                compact_slopes.append(results['Compact']['slope'])
                sprawl_slopes.append(results['Sprawl']['slope'])

        if not regions:
            log.warning("No regions with both compact and sprawl data - skipping comparison")
            return

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Slope comparison
        x_pos = np.arange(len(regions))
        width = 0.35

        ax1.bar(x_pos - width/2, compact_slopes, width, label='Compact', color='#2E86AB', alpha=0.7)
        ax1.bar(x_pos + width/2, sprawl_slopes, width, label='Sprawl', color='#A23B72', alpha=0.7)

        ax1.set_xlabel('Region')
        ax1.set_ylabel('Scaling Slope (β)')
        ax1.set_title('Scaling Slopes by Region and SNDi Category')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(regions, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Slope difference
        slope_diffs = np.array(compact_slopes) - np.array(sprawl_slopes)
        colors = ['green' if diff > 0 else 'red' for diff in slope_diffs]

        ax2.bar(x_pos, slope_diffs, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Region')
        ax2.set_ylabel('Slope Difference (Compact - Sprawl)')
        ax2.set_title('Regional Differences in Scaling Slopes')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(regions, rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / '03_global_sndi_regional_comparison.png')
        plt.savefig(self.figures_dir / '03_global_sndi_regional_comparison.pdf')
        plt.close()

        log.info("Saved regional comparison plots")

    def create_sndi_distribution_plots(self, data: pd.DataFrame):
        """
        Create plots showing SNDi value distributions.

        Args:
            data (pd.DataFrame): Global data with SNDi values
        """
        log.info("Creating SNDi distribution plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Overall SNDi distribution
        axes[0, 0].hist(data['avg_sndi'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.sndi_thresholds['compact'], color='blue', linestyle='--',
                          label=f'Compact threshold (≤{self.sndi_thresholds["compact"]})')
        axes[0, 0].axvline(self.sndi_thresholds['sprawl'], color='red', linestyle='--',
                          label=f'Sprawl threshold (≥{self.sndi_thresholds["sprawl"]})')
        axes[0, 0].set_xlabel('SNDi Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Global Distribution of SNDi Values')
        axes[0, 0].legend()

        # SNDi by category
        categories = data['sndi_category'].unique()
        colors_cat = {'Compact': '#2E86AB', 'Intermediate': '#F18F01', 'Sprawl': '#A23B72'}

        for i, category in enumerate(['Compact', 'Intermediate', 'Sprawl']):
            if category in categories:
                category_data = data[data['sndi_category'] == category]['avg_sndi']
                axes[0, 1].hist(category_data, bins=30, alpha=0.6,
                               color=colors_cat.get(category, 'gray'),
                               label=f'{category} (n={len(category_data):,})')

        axes[0, 1].set_xlabel('SNDi Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('SNDi Distribution by Category')
        axes[0, 1].legend()

        # Regional SNDi patterns
        if 'region' in data.columns:
            regions = data['region'].value_counts().head(6).index  # Top 6 regions
            region_data = [data[data['region'] == region]['avg_sndi'] for region in regions]

            axes[1, 0].boxplot(region_data, labels=regions)
            axes[1, 0].set_xlabel('Region')
            axes[1, 0].set_ylabel('SNDi Value')
            axes[1, 0].set_title('SNDi Distribution by Region')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # SNDi vs Population relationship
        axes[1, 1].scatter(np.log10(data['population_2015']), data['avg_sndi'],
                          alpha=0.5, s=10, color='gray')
        axes[1, 1].set_xlabel('Log₁₀(Population 2015)')
        axes[1, 1].set_ylabel('SNDi Value')
        axes[1, 1].set_title('SNDi vs Population Size')

        # Add trend line
        log_pop = np.log10(data['population_2015'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_pop, data['avg_sndi'])
        x_trend = np.linspace(log_pop.min(), log_pop.max(), 100)
        y_trend = slope * x_trend + intercept
        axes[1, 1].plot(x_trend, y_trend, 'r-', alpha=0.8,
                       label=f'R²={r_value**2:.3f}, p={p_value:.3f}')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.figures_dir / '03_global_sndi_distributions.png')
        plt.savefig(self.figures_dir / '03_global_sndi_distributions.pdf')
        plt.close()

        log.info("Saved SNDi distribution plots")

    def export_results(self, data: pd.DataFrame, scaling_results: Dict, regional_results: Dict = None):
        """
        Export all analysis results to CSV files.

        Args:
            data (pd.DataFrame): Global dataset with classifications
            scaling_results (Dict): Scaling analysis results
            regional_results (Dict): Regional analysis results
        """
        log.info("Exporting results...")

        today = datetime.now().strftime('%Y-%m-%d')

        # Export main dataset
        main_output = self.processed_data_dir / f'03_global_sndi_scaling_merged_data_{today}.csv'
        data.to_csv(main_output, index=False)
        log.info(f"Exported main dataset to {main_output}")

        # Export scaling results summary
        scaling_summary = []
        for mass_type, results in scaling_results.items():
            for category, stats in results.items():
                if isinstance(stats, dict) and 'slope' in stats:
                    scaling_summary.append({
                        'mass_type': mass_type,
                        'sndi_category': category,
                        'n_points': stats['n_points'],
                        'slope': stats['slope'],
                        'intercept': stats['intercept'],
                        'r_squared': stats['r_squared'],
                        'p_value': stats['p_value'],
                        'std_err': stats['std_err']
                    })

        if scaling_summary:
            scaling_df = pd.DataFrame(scaling_summary)
            scaling_output = self.processed_data_dir / f'03_global_sndi_scaling_results_{today}.csv'
            scaling_df.to_csv(scaling_output, index=False)
            log.info(f"Exported scaling results to {scaling_output}")

        # Export slope comparisons
        slope_comparisons = []
        for mass_type, results in scaling_results.items():
            if 'slope_comparison' in results:
                comp = results['slope_comparison']
                slope_comparisons.append({
                    'mass_type': mass_type,
                    'compact_slope': comp['compact_slope'],
                    'sprawl_slope': comp['sprawl_slope'],
                    'slope_difference': comp['difference'],
                    'percent_difference': comp['percent_difference']
                })

        if slope_comparisons:
            comp_df = pd.DataFrame(slope_comparisons)
            comp_output = self.processed_data_dir / f'03_global_sndi_slope_comparisons_{today}.csv'
            comp_df.to_csv(comp_output, index=False)
            log.info(f"Exported slope comparisons to {comp_output}")

        # Export regional results
        if regional_results:
            regional_summary = []
            for region, results in regional_results.items():
                for category, stats in results.items():
                    if isinstance(stats, dict) and 'slope' in stats:
                        regional_summary.append({
                            'region': region,
                            'sndi_category': category,
                            'n_points': stats['n_points'],
                            'slope': stats['slope'],
                            'intercept': stats['intercept'],
                            'r_squared': stats['r_squared'],
                            'p_value': stats['p_value']
                        })

            if regional_summary:
                regional_df = pd.DataFrame(regional_summary)
                regional_output = self.processed_data_dir / f'03_global_sndi_regional_results_{today}.csv'
                regional_df.to_csv(regional_output, index=False)
                log.info(f"Exported regional results to {regional_output}")

def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description='Global neighborhood scaling analysis by SNDi classification')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited data')
    parser.add_argument('--region-filter', type=str, help='Filter to specific region for focused analysis')
    args = parser.parse_args()

    # Set project path (current directory of this script's parent)
    script_dir = Path(__file__).parent
    project_path = script_dir.parent

    log.info("Starting global SNDi-based neighborhood scaling analysis...")
    log.info(f"Project path: {project_path}")

    try:
        # Initialize analyzer
        analyzer = GlobalSNDiScalingAnalyzer(project_path)

        # Load global data
        log.info("Step 1: Loading and merging global data...")
        global_data = analyzer.load_global_data()

        # Apply region filter if specified
        if args.region_filter:
            if 'region' in global_data.columns:
                global_data = global_data[global_data['region'] == args.region_filter]
                log.info(f"Filtered to {args.region_filter}: {len(global_data)} neighborhoods")
            else:
                log.warning("Region filtering requested but region data not available")

        # Apply debug sampling
        if args.debug:
            sample_size = min(1000, len(global_data))
            global_data = global_data.sample(n=sample_size, random_state=42)
            log.info(f"DEBUG MODE: Using sample of {len(global_data)} neighborhoods")

        # Classify neighborhoods by SNDi
        log.info("Step 2: Classifying neighborhoods by SNDi connectivity...")
        classified_data = analyzer.classify_sndi_neighborhoods(global_data)

        # Perform scaling analysis
        log.info("Step 3: Performing scaling analysis...")
        scaling_results = analyzer.perform_scaling_analysis(classified_data)

        # Regional analysis
        log.info("Step 4: Performing regional analysis...")
        regional_results = analyzer.regional_analysis(classified_data)

        # Create visualizations
        log.info("Step 5: Creating visualizations...")
        analyzer.create_global_scatter_plot(classified_data, scaling_results)
        analyzer.create_regional_comparison(classified_data, regional_results)
        analyzer.create_sndi_distribution_plots(classified_data)

        # Export results
        log.info("Step 6: Exporting results...")
        analyzer.export_results(classified_data, scaling_results, regional_results)

        # Print summary
        log.info("\n" + "="*80)
        log.info("GLOBAL SNDi SCALING ANALYSIS - KEY FINDINGS")
        log.info("="*80)

        total_neighborhoods = len(classified_data)
        log.info(f"Total neighborhoods analyzed: {total_neighborhoods:,}")

        # Category distribution
        category_counts = classified_data['sndi_category'].value_counts()
        for category, count in category_counts.items():
            pct = count / total_neighborhoods * 100
            log.info(f"{category} neighborhoods: {count:,} ({pct:.1f}%)")

        # Scaling results summary
        log.info("\nSCALING SLOPE COMPARISON:")
        if 'total_built_mass' in scaling_results:
            for category in ['Compact', 'Sprawl']:
                if category in scaling_results['total_built_mass']:
                    result = scaling_results['total_built_mass'][category]
                    log.info(f"{category:>8}: β={result['slope']:.4f}, R²={result['r_squared']:.3f}, n={result['n_points']:,}")

            if 'slope_comparison' in scaling_results['total_built_mass']:
                comp = scaling_results['total_built_mass']['slope_comparison']
                log.info(f"{'Difference':>8}: Δβ={comp['difference']:.4f} ({comp['percent_difference']:.1f}%)")

                if comp['difference'] > 0:
                    log.info("INTERPRETATION: Compact neighborhoods show STEEPER scaling than sprawl")
                else:
                    log.info("INTERPRETATION: Sprawl neighborhoods show STEEPER scaling than compact")

        # Regional patterns
        if regional_results:
            log.info(f"\nREGIONAL ANALYSIS: {len(regional_results)} regions analyzed")
            for region in regional_results.keys():
                log.info(f"  {region}: Available for comparison")

        log.info("="*80)
        log.info("Global SNDi scaling analysis completed successfully!")

    except Exception as e:
        log.error(f"Analysis failed: {e}")
        raise e

if __name__ == "__main__":
    main()