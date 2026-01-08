#!/usr/bin/env python3
"""
02_enhanced_global_sndi_scaling_with_cv.py

Enhanced global analysis of neighborhood-level scaling patterns with comprehensive
SNDi classification (Compact/Medium/Sprawl) and cross-validation framework.

This enhanced script:
1. Analyzes three SNDi categories: Compact (≤2), Medium (2-5.5), Sprawl (≥5.5)
2. Implements cross-validation framework across income groups
3. Tests robustness of SNDi effects across development contexts
4. Provides comprehensive statistical validation

SNDi Definition (from PNAS 2019):
- Lower SNDi values indicate more connected, compact street networks
- Higher SNDi values indicate more disconnected, sprawling street patterns
- Three-tier classification for comprehensive analysis

Research Questions Enhanced:
- Do neighborhoods show monotonic scaling progression: Compact > Medium > Sprawl?
- Are SNDi effects robust across income groups when tested via cross-validation?
- What is the statistical significance of SNDi-based scaling differences?

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
from matplotlib.gridspec import GridSpec
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse

# Statistical libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

# Advanced statistical modeling
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Bootstrap methods
try:
    from scipy.stats import bootstrap
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class EnhancedSNDiScalingAnalyzer:
    """
    Enhanced analyzer for global neighborhood scaling patterns with three-category SNDi
    classification and cross-validation framework.
    """

    def __init__(self, base_path: str):
        """
        Initialize the enhanced SNDi scaling analyzer.

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
        self.maps_dir = self.main_project_dir / 'maps' / 'china-city-mass-index-map' / 'config'

        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced SNDi classification thresholds
        # Thresholds align with distributional breakpoints suggested by the
        # original SNDi paper (Boeing 2019) and the empirical quartiles of the
        # merged dataset: compact ≤ ~2.6, transitional 2.6-4.8, sprawl > 4.8.
        self.sndi_thresholds = {
            'compact': 2.6,       # Dense, grid-like street fabrics
            'medium_lower': 2.6,  # Lower bound for transitional category
            'medium_upper': 4.8,  # Upper bound anchored near the 75th percentile
            'sprawl': 4.8,        # Fragmented, auto-oriented networks
            'extreme_sprawl': 6.2  # Optional top-decile flag for diagnostics
        }

        # Three-category system
        self.sndi_categories = ['Compact', 'Medium', 'Sprawl']

        # Cross-validation parameters
        self.cv_params = {
            'n_folds': 5,
            'min_countries_per_income_group': 10,
            'min_neighborhoods_per_fold': 50,
            'min_cities_per_fold': 5,
            'random_seed': 42
        }

        # Bin-averaging configuration for density-balanced regression diagnostics
        self.bin_config = {
            'max_bins': 20,
            'min_points_per_bin': 30
        }

        # Set up plotting style
        plt.style.use('default')
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

        log.info("EnhancedSNDiScalingAnalyzer initialized")
        log.info(f"Processed data directory: {self.processed_data_dir}")
        log.info(f"Enhanced SNDi categories: {self.sndi_categories}")
        log.info(
            "Thresholds - Compact: ≤%.2f, Medium: %.2f-%.2f, Sprawl: >%.2f, Extreme Sprawl flag: >%.2f"
            % (
                self.sndi_thresholds['compact'],
                self.sndi_thresholds['medium_lower'],
                self.sndi_thresholds['medium_upper'],
                self.sndi_thresholds['sprawl'],
                self.sndi_thresholds['extreme_sprawl'],
            )
        )

    def load_global_data_with_income_groups(self) -> pd.DataFrame:
        """
        Load and merge global neighborhood mass data with SNDi values and income classifications.

        Returns:
            pd.DataFrame: Merged global dataset with SNDi and income classifications
        """
        log.info("Loading global neighborhood data with income group classifications...")

        # Load the most recent global mass data
        mass_file = self.global_data_dir / 'Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv'
        if not mass_file.exists():
            raise FileNotFoundError(f"Global mass data file not found: {mass_file}")

        global_mass = pd.read_csv(mass_file)
        log.info(f"Loaded {len(global_mass)} neighborhoods with global mass data")

        # Load SNDi data
        sndi_file = self.processed_data_dir / '01_neighborhood_SNDi_2025-09-09.csv'
        if not sndi_file.exists():
            raise FileNotFoundError(f"SNDi data file not found: {sndi_file}")

        sndi_data = pd.read_csv(sndi_file)
        log.info(f"Loaded {len(sndi_data)} neighborhoods with SNDi data")

        # Load income group classifications
        income_file = self.maps_dir / 'country_classification.csv'
        if not income_file.exists():
            raise FileNotFoundError(f"Income classification file not found: {income_file}")

        income_data = pd.read_csv(income_file, encoding='utf-8-sig')
        log.info(f"Loaded income classifications for {len(income_data)} countries")

        # Merge datasets
        merged_data = pd.merge(
            global_mass,
            sndi_data[['h3index', 'avg_sndi', 'sndi_point_count']],
            on='h3index',
            how='inner'
        )
        log.info(f"Merged mass and SNDi data: {len(merged_data)} neighborhoods")

        # Add income group classifications
        merged_data = pd.merge(
            merged_data,
            income_data[['Code', 'Income group']],
            left_on='CTR_MN_ISO',
            right_on='Code',
            how='left'
        )

        # Filter for valid data
        initial_count = len(merged_data)
        merged_data = merged_data.dropna(subset=['avg_sndi', 'population_2015', 'total_built_mass_tons', 'Income group'])
        merged_data = merged_data[
            (merged_data['avg_sndi'] > 0) &
            (merged_data['population_2015'] > 0) &
            (merged_data['total_built_mass_tons'] > 0) &
            (merged_data['sndi_point_count'] > 0)
        ]

        after_filter = len(merged_data)
        log.info(f"After filtering: {after_filter} valid neighborhoods ({initial_count - after_filter} removed)")

        # Apply R-style filtering criteria (from income group CV script)
        log.info("Applying R-style filtering criteria for robustness...")

        # Step 1: Filter countries with >= 3 cities
        country_city_counts = merged_data.groupby('CTR_MN_ISO')['ID_HDC_G0'].nunique()
        valid_countries = country_city_counts[country_city_counts >= 3].index
        merged_data = merged_data[merged_data['CTR_MN_ISO'].isin(valid_countries)].copy()

        # Step 2: Filter cities with total population > 50,000
        city_populations = merged_data.groupby(['CTR_MN_ISO', 'ID_HDC_G0'])['population_2015'].sum()
        valid_cities = city_populations[city_populations > 50000].index
        city_mask = pd.MultiIndex.from_tuples(
            merged_data[['CTR_MN_ISO', 'ID_HDC_G0']].apply(tuple, axis=1)
        ).isin(valid_cities)
        merged_data = merged_data[city_mask].copy()

        # Step 3: Filter cities with > 3 neighborhoods
        city_neighborhood_counts = merged_data.groupby(['CTR_MN_ISO', 'ID_HDC_G0']).size()
        valid_cities_neighborhoods = city_neighborhood_counts[city_neighborhood_counts > 3].index
        city_mask = pd.MultiIndex.from_tuples(
            merged_data[['CTR_MN_ISO', 'ID_HDC_G0']].apply(tuple, axis=1)
        ).isin(valid_cities_neighborhoods)
        merged_data = merged_data[city_mask].copy()

        final_count = len(merged_data)
        log.info(f"After R-style filtering: {final_count} neighborhoods "
                f"({after_filter - final_count} additional removed)")

        # Log income group distribution
        income_summary = merged_data.groupby('Income group').agg({
            'CTR_MN_ISO': 'nunique',
            'ID_HDC_G0': 'nunique',
            'h3index': 'count'
        }).rename(columns={'CTR_MN_ISO': 'countries', 'ID_HDC_G0': 'cities', 'h3index': 'neighborhoods'})

        log.info("Income group distribution after filtering:")
        for income_group, stats in income_summary.iterrows():
            log.info(f"  {income_group}: {stats['countries']} countries, "
                    f"{stats['cities']} cities, {stats['neighborhoods']} neighborhoods")

        return merged_data

    def classify_three_category_sndi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify neighborhoods into three SNDi categories: Compact, Medium, Sprawl.

        Args:
            data (pd.DataFrame): Merged dataset with SNDi values

        Returns:
            pd.DataFrame: Data with enhanced three-category SNDi classifications
        """
        log.info("Classifying neighborhoods into three SNDi categories...")

        # Create three-category SNDi classification
        conditions = [
            data['avg_sndi'] <= self.sndi_thresholds['compact'],
            (data['avg_sndi'] > self.sndi_thresholds['medium_lower']) &
            (data['avg_sndi'] <= self.sndi_thresholds['medium_upper']),
            data['avg_sndi'] > self.sndi_thresholds['sprawl']
        ]
        choices = ['Compact', 'Medium', 'Sprawl']

        data['sndi_category'] = np.select(conditions, choices, default='Undefined')

        # Flag extreme sprawl neighborhoods for diagnostic tracking
        extreme_sprawl_mask = (
            (data['sndi_category'] == 'Sprawl') &
            (data['avg_sndi'] > self.sndi_thresholds['extreme_sprawl'])
        )
        data['extreme_sprawl_flag'] = np.where(extreme_sprawl_mask, 1, 0)

        # Remove any undefined classifications
        initial_count = len(data)
        data = data[data['sndi_category'] != 'Undefined'].copy()
        final_count = len(data)

        if initial_count != final_count:
            log.warning(f"Removed {initial_count - final_count} neighborhoods with undefined SNDi categories")

        # Log three-category classification results
        classification_counts = data['sndi_category'].value_counts()
        log.info("Enhanced SNDi Classification Results:")
        for category, count in classification_counts.items():
            percentage = count / len(data) * 100
            if category == 'Compact':
                sndi_range = f"≤{self.sndi_thresholds['compact']:.2f}"
            elif category == 'Medium':
                sndi_range = (
                    f"{self.sndi_thresholds['medium_lower']:.2f}-"
                    f"{self.sndi_thresholds['medium_upper']:.2f}"
                )
            else:  # Sprawl
                sndi_range = f">{self.sndi_thresholds['sprawl']:.2f}"

            log.info(f"  {category} (SNDi {sndi_range}): {count:,} neighborhoods ({percentage:.1f}%)")

        extreme_count = int(extreme_sprawl_mask.sum())
        if extreme_count:
            log.info(
                "  Extreme Sprawl (SNDi > %.2f): %s neighborhoods (%.1f%% of Sprawl)"
                % (
                    self.sndi_thresholds['extreme_sprawl'],
                    f"{extreme_count:,}",
                    (extreme_count / max(classification_counts.get('Sprawl', 1), 1)) * 100,
                )
            )

        # Provide population distribution diagnostics to spot skewness by category
        if 'population_2015' in data.columns:
            percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            log.info("Population distribution (2015) by SNDi category:")
            for category in self.sndi_categories:
                group = data.loc[data['sndi_category'] == category, 'population_2015']
                if group.empty:
                    continue
                stats = {
                    'n': len(group),
                    'median': group.median(),
                    'p10': group.quantile(0.1),
                    'p25': group.quantile(0.25),
                    'p75': group.quantile(0.75),
                    'p90': group.quantile(0.9),
                }
                log.info(
                    "  %s -> n=%s, median=%.0f, p10=%.0f, p25=%.0f, p75=%.0f, p90=%.0f"
                    % (
                        category,
                        f"{stats['n']:,}",
                        stats['median'],
                        stats['p10'],
                        stats['p25'],
                        stats['p75'],
                        stats['p90'],
                    )
                )

        # Add regional information
        if 'CTR_MN_ISO' in data.columns:
            # Simplified region mapping
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

            # Log regional and income group distribution
            cross_distribution = data.groupby(['region', 'Income group', 'sndi_category']).size().unstack(fill_value=0)
            log.info("Regional × Income Group × SNDi Category distribution:")
            log.info(f"\n{cross_distribution}")

        return data

    def perform_three_way_scaling_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive scaling analysis comparing all three SNDi categories.

        Args:
            data (pd.DataFrame): Classified neighborhood data

        Returns:
            Dict: Enhanced scaling analysis results
        """
        log.info("Performing three-way scaling analysis (Compact/Medium/Sprawl)...")

        # Prepare data for scaling analysis
        scaling_data = data.copy()
        scaling_data['log10_population'] = np.log10(scaling_data['population_2015'])
        scaling_data['log10_total_mass'] = np.log10(scaling_data['total_built_mass_tons'])

        # Prepare other mass types
        if 'BuildingMass_AverageTotal' in scaling_data.columns:
            valid_building = scaling_data['BuildingMass_AverageTotal'] > 0
            scaling_data.loc[valid_building, 'log10_building_mass'] = np.log10(
                scaling_data.loc[valid_building, 'BuildingMass_AverageTotal']
            )

        if 'mobility_mass_tons' in scaling_data.columns:
            valid_mobility = scaling_data['mobility_mass_tons'] > 0
            scaling_data.loc[valid_mobility, 'log10_mobility_mass'] = np.log10(
                scaling_data.loc[valid_mobility, 'mobility_mass_tons']
            )

        results = {}

        # Mass types to analyze
        mass_types = {
            'total_built_mass': ('log10_total_mass', 'Total Built Mass'),
            'building_mass': ('log10_building_mass', 'Building Mass'),
            'mobility_mass': ('log10_mobility_mass', 'Mobility Mass')
        }

        # Analyze each mass type across all three SNDi categories
        for mass_key, (log_column, mass_name) in mass_types.items():
            if log_column not in scaling_data.columns:
                log.warning(f"Skipping {mass_name} - data not available")
                continue

            results[mass_key] = {}

            # Analyze each SNDi category
            for category in self.sndi_categories:
                category_data = scaling_data[
                    (scaling_data['sndi_category'] == category) &
                    (scaling_data[log_column].notna())
                ].copy()

                if len(category_data) < 10:
                    log.warning(f"Insufficient data for {category} neighborhoods in {mass_name}: {len(category_data)} points")
                    continue

                # Perform regression on individual observations
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
                    'population_range': (
                        float(category_data['log10_population'].min()),
                        float(category_data['log10_population'].max())
                    )
                }

                log.info(f"{mass_name} - {category}: n={len(category_data)}, slope={slope:.3f}, R²={r2:.3f}, p={p_value:.4f}")

                # Bin-averaged regression to mitigate dense-population leverage
                bin_summary_records = None
                binned_stats = None
                bin_results = self._compute_binned_regression(category_data, log_column)
                if bin_results:
                    (bin_summary_records,
                     binned_stats) = bin_results
                    results[mass_key][category]['binned_regression'] = binned_stats
                    results[mass_key][category]['bin_summary'] = bin_summary_records

                    log.info(
                        f"{mass_name} (binned) - {category}: bins={binned_stats['n_bins']}, "
                        f"slope={binned_stats['slope']:.3f}, R²={binned_stats['r_squared']:.3f}"
                    )

                # Mixed-effects regression with city-specific intercepts
                mixed_results = self._compute_mixed_effects_regression(category_data, log_column)
                if mixed_results:
                    results[mass_key][category]['mixed_effects'] = mixed_results
                    log.info(
                        f"{mass_name} (mixed) - {category}: slope={mixed_results['slope']:.3f}, "
                        f"intercept={mixed_results['intercept']:.3f}, "
                        f"random_sd={mixed_results['random_intercept_sd']:.3f}, "
                        f"n_groups={mixed_results['n_groups']}"
                    )

        # Perform pairwise slope comparisons
        log.info("Performing pairwise slope comparisons...")

        for mass_key, mass_results in results.items():
            # Filter out non-category keys and ensure we have valid slope data
            category_results = {k: v for k, v in mass_results.items()
                              if k in self.sndi_categories and isinstance(v, dict) and 'slope' in v}

            if len(category_results) >= 2:
                results[mass_key]['pairwise_comparisons'] = {}

                # All possible pairs
                categories_present = list(category_results.keys())
                for i, cat1 in enumerate(categories_present):
                    for cat2 in categories_present[i+1:]:
                        slope1 = category_results[cat1]['slope']
                        slope2 = category_results[cat2]['slope']
                        slope_diff = slope1 - slope2

                        # Store comparison
                        comparison_key = f"{cat1}_vs_{cat2}"
                        results[mass_key]['pairwise_comparisons'][comparison_key] = {
                            f'{cat1}_slope': slope1,
                            f'{cat2}_slope': slope2,
                            'slope_difference': slope_diff,
                            'percent_difference': (slope_diff / slope2 * 100) if slope2 != 0 else np.nan,
                            'comparison_direction': f"{cat1} {'>' if slope_diff > 0 else '<'} {cat2}"
                        }

                        log.info(f"{mass_types[mass_key][1]} - {comparison_key}: "
                               f"{slope1:.4f} vs {slope2:.4f} (Δ = {slope_diff:.4f})")

        # Test slope progression hypothesis
        results['slope_progression_test'] = self._test_slope_progression_hypothesis(results)

        return results

    def _compute_binned_regression(self, category_data: pd.DataFrame, log_column: str) -> Optional[Tuple[List[Dict[str, Union[str, float, int]]], Dict[str, Union[int, float]]]]:
        """Compute bin-averaged regression statistics for a given SNDi category."""

        if 'log10_population' not in category_data or log_column not in category_data:
            return None

        total_points = len(category_data)
        if total_points < max(self.bin_config['min_points_per_bin'], 30):
            return None

        ideal_bins = min(self.bin_config['max_bins'], total_points // self.bin_config['min_points_per_bin'])
        bin_count = max(3, ideal_bins)

        try:
            category_data['population_bin'] = pd.qcut(
                category_data['log10_population'],
                q=bin_count,
                duplicates='drop'
            )
        except ValueError:
            return None

        bin_summary = category_data.groupby('population_bin', observed=True).agg(
            log10_population_mean=('log10_population', 'mean'),
            log_mass_mean=(log_column, 'mean'),
            population_sum=('population_2015', 'sum'),
            n_points=('log10_population', 'size')
        ).dropna(subset=['log_mass_mean'])

        if len(bin_summary) < 3:
            return None

        bin_summary = bin_summary.reset_index()
        bin_summary['population_bin'] = bin_summary['population_bin'].astype(str)

        x_vals = bin_summary['log10_population_mean'].to_numpy(dtype=float)
        y_vals = bin_summary['log_mass_mean'].to_numpy(dtype=float)
        weights = bin_summary['n_points'].to_numpy(dtype=float)

        regression_stats = self._weighted_regression_stats(x_vals, y_vals, weights)
        if regression_stats is None:
            return None

        slope, intercept, r_squared, std_err = regression_stats

        bin_summary_records = bin_summary.to_dict(orient='records')

        binned_stats = {
            'n_bins': len(bin_summary_records),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'std_err': std_err,
            'weighted': True
        }

        return bin_summary_records, binned_stats

    def _weighted_regression_stats(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Compute weighted regression slope, intercept, R², and standard error."""

        if len(x) != len(y) or len(x) != len(weights) or len(x) < 2:
            return None

        mask = weights > 0
        if mask.sum() < 2:
            return None

        x = x[mask]
        y = y[mask]
        weights = weights[mask]

        x_mean = np.average(x, weights=weights)
        y_mean = np.average(y, weights=weights)

        cov_xy = np.sum(weights * (x - x_mean) * (y - y_mean))
        var_x = np.sum(weights * (x - x_mean) ** 2)

        if var_x == 0:
            return None

        slope = cov_xy / var_x
        intercept = y_mean - slope * x_mean

        residuals = y - (slope * x + intercept)
        ss_res = np.sum(weights * residuals ** 2)
        ss_tot = np.sum(weights * (y - y_mean) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        dof = max(len(x) - 2, 1)
        variance = ss_res / dof if dof > 0 else 0.0
        std_err = np.sqrt(variance / var_x) if var_x > 0 else np.nan

        return float(slope), float(intercept), float(r_squared), float(std_err)

    def _compute_mixed_effects_regression(self, category_data: pd.DataFrame, log_column: str) -> Optional[Dict[str, Union[float, int, bool]]]:
        """Estimate mixed-effects regression with city-specific intercepts."""

        if not STATSMODELS_AVAILABLE:
            log.warning("Statsmodels unavailable; skipping mixed-effects regression")
            return None

        required_cols = {'log10_population', log_column, 'ID_HDC_G0'}
        if not required_cols.issubset(category_data.columns):
            log.warning("Missing columns for mixed-effects regression")
            return None

        df = category_data[list(required_cols)].dropna()
        if len(df) < 100 or df['ID_HDC_G0'].nunique() < 5:
            log.warning("Insufficient data/groups for mixed-effects regression")
            return None

        exog = sm.add_constant(df['log10_population'])

        try:
            model = sm.MixedLM(df[log_column], exog, groups=df['ID_HDC_G0'])
            fit = model.fit(reml=False, method='lbfgs', maxiter=200, disp=False)
        except Exception as exc:
            log.warning(f"Mixed-effects regression failed: {exc}")
            return None

        slope = float(fit.fe_params.get('log10_population', np.nan))
        intercept = float(fit.fe_params.get('const', np.nan))
        random_var = float(fit.cov_re.iloc[0, 0]) if fit.cov_re.shape else np.nan
        random_sd = float(np.sqrt(random_var)) if random_var >= 0 else np.nan
        resid_var = float(fit.scale)

        return {
            'slope': slope,
            'intercept': intercept,
            'random_intercept_sd': random_sd,
            'residual_sd': float(np.sqrt(resid_var)) if resid_var >= 0 else np.nan,
            'n_groups': int(df['ID_HDC_G0'].nunique()),
            'n_obs': int(len(df)),
            'converged': bool(getattr(fit, 'converged', False)),
            'aic': float(getattr(fit, 'aic', np.nan)),
            'bic': float(getattr(fit, 'bic', np.nan))
        }

    def _test_slope_progression_hypothesis(self, scaling_results: Dict) -> Dict:
        """
        Test the hypothesis that slopes follow progression: Compact > Medium > Sprawl.

        Args:
            scaling_results (Dict): Results from three-way scaling analysis

        Returns:
            Dict: Hypothesis test results
        """
        log.info("Testing slope progression hypothesis: β_compact > β_medium > β_sprawl")

        progression_results = {}

        for mass_key, mass_results in scaling_results.items():
            if mass_key == 'slope_progression_test':  # Skip self-reference
                continue

            if len(mass_results) < 3:  # Need all three categories
                continue

            # Check if all three categories are present
            required_categories = set(self.sndi_categories)
            present_categories = set(mass_results.keys()) - {'pairwise_comparisons'}

            if required_categories.issubset(present_categories):
                compact_slope = mass_results['Compact']['slope']
                medium_slope = mass_results['Medium']['slope']
                sprawl_slope = mass_results['Sprawl']['slope']

                # Test progression
                compact_gt_medium = compact_slope > medium_slope
                medium_gt_sprawl = medium_slope > sprawl_slope
                compact_gt_sprawl = compact_slope > sprawl_slope

                progression_satisfied = compact_gt_medium and medium_gt_sprawl and compact_gt_sprawl

                # Calculate effect sizes
                compact_medium_diff = compact_slope - medium_slope
                medium_sprawl_diff = medium_slope - sprawl_slope
                compact_sprawl_diff = compact_slope - sprawl_slope

                progression_results[mass_key] = {
                    'compact_slope': compact_slope,
                    'medium_slope': medium_slope,
                    'sprawl_slope': sprawl_slope,
                    'compact_gt_medium': compact_gt_medium,
                    'medium_gt_sprawl': medium_gt_sprawl,
                    'compact_gt_sprawl': compact_gt_sprawl,
                    'progression_satisfied': progression_satisfied,
                    'compact_medium_diff': compact_medium_diff,
                    'medium_sprawl_diff': medium_sprawl_diff,
                    'compact_sprawl_diff': compact_sprawl_diff,
                    'total_progression': compact_sprawl_diff,
                    'progression_interpretation': (
                        f"{'CONFIRMED' if progression_satisfied else 'REJECTED'}: "
                        f"Compact({compact_slope:.3f}) > Medium({medium_slope:.3f}) > Sprawl({sprawl_slope:.3f})"
                    )
                }

                log.info(f"Slope progression test for {mass_key}: {progression_results[mass_key]['progression_interpretation']}")

        return progression_results

    def run_income_stratified_cross_validation(self, data: pd.DataFrame) -> Dict:
        """
        Run cross-validation analysis stratified by income groups to test SNDi effect robustness.

        Args:
            data (pd.DataFrame): Classified global data with income groups

        Returns:
            Dict: Cross-validation results by income group
        """
        log.info("Running income-stratified cross-validation analysis...")

        cv_results = {}

        # Get unique income groups
        income_groups = sorted(data['Income group'].unique())
        log.info(f"Income groups found: {income_groups}")

        for income_group in income_groups:
            group_result = self._run_single_income_group_cv(data, income_group)
            if group_result:
                cv_results[income_group] = group_result

        # Cross-income group analysis
        if len(cv_results) >= 2:
            cv_results['cross_income_analysis'] = self._analyze_cross_income_consistency(cv_results)

        return cv_results

    def _run_single_income_group_cv(self, data: pd.DataFrame, income_group: str) -> Optional[Dict]:
        """
        Run cross-validation for a single income group.

        Args:
            data (pd.DataFrame): Full dataset
            income_group (str): Income group to analyze

        Returns:
            Optional[Dict]: CV results for this income group, or None if insufficient data
        """
        log.info(f"Running cross-validation for {income_group}...")

        # Filter to income group
        group_data = data[data['Income group'] == income_group].copy()

        # Check minimum requirements
        unique_countries = group_data['CTR_MN_ISO'].nunique()
        if unique_countries < self.cv_params['min_countries_per_income_group']:
            log.warning(f"Insufficient countries for {income_group}: {unique_countries} < {self.cv_params['min_countries_per_income_group']}")
            return None

        # Prepare log-transformed data
        group_data['log10_population'] = np.log10(group_data['population_2015'])
        group_data['log10_total_mass'] = np.log10(group_data['total_built_mass_tons'])

        log.info(f"{income_group}: {unique_countries} countries, {group_data['ID_HDC_G0'].nunique()} cities, {len(group_data)} neighborhoods")

        # Create country-level folds
        countries = group_data['CTR_MN_ISO'].unique()
        n_folds = min(self.cv_params['n_folds'], unique_countries // 3)

        if n_folds < 3:
            log.warning(f"Insufficient countries for meaningful CV in {income_group}: {unique_countries}")
            return None

        # Shuffle countries for random folds
        np.random.seed(self.cv_params['random_seed'])
        shuffled_countries = np.random.permutation(countries)

        # Create folds
        fold_size = len(countries) // n_folds
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            if i == n_folds - 1:  # Last fold gets remaining countries
                fold_countries = shuffled_countries[start_idx:]
            else:
                fold_countries = shuffled_countries[start_idx:start_idx + fold_size]
            folds.append(fold_countries)

        log.info(f"{income_group}: Created {n_folds} CV folds with countries per fold: {[len(fold) for fold in folds]}")

        # Run CV folds
        fold_results = []
        failed_folds = []

        for fold_idx, test_countries in enumerate(folds):
            fold_result = self._run_single_cv_fold(group_data, fold_idx, test_countries, income_group)

            if fold_result['success']:
                fold_results.append(fold_result)
            else:
                failed_folds.append(fold_result)

        if not fold_results:
            log.warning(f"No successful CV folds for {income_group}")
            return None

        # Analyze CV results
        cv_summary = self._analyze_cv_results(fold_results, income_group)

        return {
            'income_group': income_group,
            'data_summary': {
                'total_countries': unique_countries,
                'total_cities': group_data['ID_HDC_G0'].nunique(),
                'total_neighborhoods': len(group_data),
                'sndi_distribution': group_data['sndi_category'].value_counts().to_dict()
            },
            'cv_setup': {
                'n_folds': n_folds,
                'countries_per_fold': [len(fold) for fold in folds]
            },
            'fold_results': fold_results,
            'failed_folds': failed_folds,
            'cv_summary': cv_summary
        }

    def _run_single_cv_fold(self, group_data: pd.DataFrame, fold_idx: int,
                           test_countries: List[str], income_group: str) -> Dict:
        """
        Run a single cross-validation fold.

        Args:
            group_data (pd.DataFrame): Income group data
            fold_idx (int): Fold index
            test_countries (List[str]): Countries for testing
            income_group (str): Income group name

        Returns:
            Dict: Fold results
        """
        try:
            # Split data
            train_countries = [c for c in group_data['CTR_MN_ISO'].unique() if c not in test_countries]

            train_data = group_data[group_data['CTR_MN_ISO'].isin(train_countries)].copy()
            test_data = group_data[group_data['CTR_MN_ISO'].isin(test_countries)].copy()

            # Check minimum requirements
            if (len(train_data) < self.cv_params['min_neighborhoods_per_fold'] or
                train_data['ID_HDC_G0'].nunique() < self.cv_params['min_cities_per_fold']):
                return {
                    'fold_idx': fold_idx,
                    'success': False,
                    'reason': 'insufficient_training_data',
                    'details': {
                        'train_neighborhoods': len(train_data),
                        'train_cities': train_data['ID_HDC_G0'].nunique() if len(train_data) > 0 else 0
                    }
                }

            # Train models for each SNDi category
            category_results = {}

            for category in self.sndi_categories:
                cat_train_data = train_data[train_data['sndi_category'] == category]
                cat_test_data = test_data[test_data['sndi_category'] == category]

                if len(cat_train_data) < 10 or len(cat_test_data) < 5:
                    continue  # Skip categories with insufficient data

                # Fit model (simple linear regression for robustness)
                X_train = cat_train_data['log10_population'].values.reshape(-1, 1)
                y_train = cat_train_data['log10_total_mass'].values

                model = LinearRegression()
                model.fit(X_train, y_train)

                # Test predictions
                X_test = cat_test_data['log10_population'].values.reshape(-1, 1)
                y_test = cat_test_data['log10_total_mass'].values
                y_pred = model.predict(X_test)

                # Calculate metrics
                test_r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
                test_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

                # Get slope and statistics
                slope = model.coef_[0]
                intercept = model.intercept_

                category_results[category] = {
                    'train_n': len(cat_train_data),
                    'test_n': len(cat_test_data),
                    'slope': slope,
                    'intercept': intercept,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                }

            # Calculate pairwise slope differences
            slope_differences = {}
            if len(category_results) >= 2:
                categories_present = list(category_results.keys())
                for i, cat1 in enumerate(categories_present):
                    for cat2 in categories_present[i+1:]:
                        slope_diff = category_results[cat1]['slope'] - category_results[cat2]['slope']
                        slope_differences[f"{cat1}_vs_{cat2}"] = slope_diff

            return {
                'fold_idx': fold_idx,
                'success': True,
                'train_countries': sorted(train_countries),
                'test_countries': sorted(test_countries),
                'category_results': category_results,
                'slope_differences': slope_differences,
                'overall_performance': {
                    'categories_analyzed': len(category_results),
                    'total_train_n': len(train_data),
                    'total_test_n': len(test_data)
                }
            }

        except Exception as e:
            return {
                'fold_idx': fold_idx,
                'success': False,
                'reason': 'execution_error',
                'error': str(e)
            }

    def _analyze_cv_results(self, fold_results: List[Dict], income_group: str) -> Dict:
        """
        Analyze cross-validation results for an income group.

        Args:
            fold_results (List[Dict]): Results from all successful folds
            income_group (str): Income group name

        Returns:
            Dict: CV analysis summary
        """
        log.info(f"Analyzing CV results for {income_group} ({len(fold_results)} successful folds)")

        # Collect slopes by category across folds
        slopes_by_category = {category: [] for category in self.sndi_categories}
        r2_by_category = {category: [] for category in self.sndi_categories}

        for fold_result in fold_results:
            for category, cat_result in fold_result['category_results'].items():
                slopes_by_category[category].append(cat_result['slope'])
                if not np.isnan(cat_result['test_r2']):
                    r2_by_category[category].append(cat_result['test_r2'])

        # Calculate summary statistics
        category_summaries = {}
        for category in self.sndi_categories:
            slopes = slopes_by_category[category]
            r2s = r2_by_category[category]

            if len(slopes) >= 2:
                category_summaries[category] = {
                    'n_folds': len(slopes),
                    'mean_slope': np.mean(slopes),
                    'std_slope': np.std(slopes, ddof=1),
                    'cv_percent': (np.std(slopes, ddof=1) / np.mean(slopes)) * 100,
                    'mean_r2': np.mean(r2s) if r2s else np.nan,
                    'std_r2': np.std(r2s, ddof=1) if len(r2s) > 1 else np.nan,
                    'slope_consistency': 'HIGH' if (np.std(slopes, ddof=1) / np.mean(slopes)) * 100 < 5 else 'MODERATE' if (np.std(slopes, ddof=1) / np.mean(slopes)) * 100 < 10 else 'LOW'
                }

                log.info(f"{income_group} - {category}: "
                        f"slope = {category_summaries[category]['mean_slope']:.3f} ± {category_summaries[category]['std_slope']:.3f} "
                        f"(CV = {category_summaries[category]['cv_percent']:.1f}%)")

        # Analyze slope differences consistency
        slope_diff_consistency = {}
        if len(category_summaries) >= 2:
            categories_present = list(category_summaries.keys())
            for i, cat1 in enumerate(categories_present):
                for cat2 in categories_present[i+1:]:
                    diff_key = f"{cat1}_vs_{cat2}"

                    # Collect differences across folds
                    diffs = []
                    for fold_result in fold_results:
                        if diff_key in fold_result['slope_differences']:
                            diffs.append(fold_result['slope_differences'][diff_key])

                    if len(diffs) >= 2:
                        mean_diff = np.mean(diffs)
                        std_diff = np.std(diffs, ddof=1)
                        consistent_direction = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

                        slope_diff_consistency[diff_key] = {
                            'mean_difference': mean_diff,
                            'std_difference': std_diff,
                            'consistent_direction': consistent_direction,
                            'all_diffs': diffs,
                            'interpretation': f"{cat1} {'consistently >' if mean_diff > 0 and consistent_direction else 'inconsistently vs'} {cat2}"
                        }

        # Overall assessment
        successful_categories = len(category_summaries)
        high_consistency_categories = sum(1 for cat_summary in category_summaries.values()
                                        if cat_summary['slope_consistency'] == 'HIGH')

        overall_assessment = {
            'successful_categories': successful_categories,
            'high_consistency_categories': high_consistency_categories,
            'cv_quality': 'EXCELLENT' if high_consistency_categories >= 2 else 'GOOD' if successful_categories >= 2 else 'LIMITED',
            'sndi_effects_robust': high_consistency_categories >= 2 and len(slope_diff_consistency) > 0,
            'interpretation': f"SNDi effects in {income_group}: {'ROBUST' if high_consistency_categories >= 2 else 'MODERATE' if successful_categories >= 2 else 'INSUFFICIENT'} evidence from cross-validation"
        }

        return {
            'successful_folds': len(fold_results),
            'category_summaries': category_summaries,
            'slope_difference_consistency': slope_diff_consistency,
            'overall_assessment': overall_assessment
        }

    def _analyze_cross_income_consistency(self, cv_results: Dict) -> Dict:
        """
        Analyze consistency of SNDi effects across income groups.

        Args:
            cv_results (Dict): CV results for all income groups

        Returns:
            Dict: Cross-income consistency analysis
        """
        log.info("Analyzing cross-income group consistency of SNDi effects...")

        # Collect slopes by category across income groups
        cross_income_slopes = {category: {} for category in self.sndi_categories}

        for income_group, income_result in cv_results.items():
            if income_group == 'cross_income_analysis':
                continue

            for category, cat_summary in income_result['cv_summary']['category_summaries'].items():
                cross_income_slopes[category][income_group] = cat_summary['mean_slope']

        # Analyze consistency across income groups
        cross_income_summary = {}
        for category in self.sndi_categories:
            slopes = list(cross_income_slopes[category].values())
            if len(slopes) >= 2:
                cross_income_summary[category] = {
                    'income_groups': len(slopes),
                    'mean_slope': np.mean(slopes),
                    'std_slope': np.std(slopes, ddof=1),
                    'cv_percent': (np.std(slopes, ddof=1) / np.mean(slopes)) * 100,
                    'consistency': 'HIGH' if (np.std(slopes, ddof=1) / np.mean(slopes)) * 100 < 10 else 'MODERATE' if (np.std(slopes, ddof=1) / np.mean(slopes)) * 100 < 20 else 'LOW',
                    'slopes_by_income': cross_income_slopes[category]
                }

        # Test universal SNDi progression across income groups
        progression_consistency = {}
        income_groups_analyzed = [ig for ig in cv_results.keys() if ig != 'cross_income_analysis']

        for income_group in income_groups_analyzed:
            income_result = cv_results[income_group]
            cat_summaries = income_result['cv_summary']['category_summaries']

            if len(cat_summaries) >= 3:  # Need all three categories
                required_cats = set(self.sndi_categories)
                present_cats = set(cat_summaries.keys())

                if required_cats.issubset(present_cats):
                    compact_slope = cat_summaries['Compact']['mean_slope']
                    medium_slope = cat_summaries['Medium']['mean_slope']
                    sprawl_slope = cat_summaries['Sprawl']['mean_slope']

                    progression_satisfied = (compact_slope > medium_slope > sprawl_slope)

                    progression_consistency[income_group] = {
                        'compact_slope': compact_slope,
                        'medium_slope': medium_slope,
                        'sprawl_slope': sprawl_slope,
                        'progression_satisfied': progression_satisfied
                    }

        # Overall cross-income assessment
        consistent_progression_count = sum(1 for result in progression_consistency.values()
                                         if result['progression_satisfied'])
        total_testable_groups = len(progression_consistency)

        overall_consistency = {
            'categories_analyzed': len(cross_income_summary),
            'income_groups_tested': len(income_groups_analyzed),
            'progression_tests': {
                'testable_income_groups': total_testable_groups,
                'consistent_progressions': consistent_progression_count,
                'consistency_rate': consistent_progression_count / total_testable_groups if total_testable_groups > 0 else 0
            },
            'universality_assessment': (
                'STRONG' if consistent_progression_count >= 3 and len(cross_income_summary) >= 2 else
                'MODERATE' if consistent_progression_count >= 2 or len(cross_income_summary) >= 2 else
                'WEAK'
            ),
            'interpretation': f"SNDi effects show {cross_income_summary.get('Compact', {}).get('consistency', 'UNKNOWN')} consistency across income groups"
        }

        log.info(f"Cross-income consistency: {overall_consistency['universality_assessment']} "
                f"({consistent_progression_count}/{total_testable_groups} income groups show consistent progression)")

        return {
            'cross_income_slopes': cross_income_summary,
            'progression_consistency': progression_consistency,
            'overall_consistency': overall_consistency
        }

    def create_enhanced_visualizations(self, data: pd.DataFrame, scaling_results: Dict, cv_results: Dict):
        """
        Create comprehensive enhanced visualizations with three-category analysis and CV results.

        Args:
            data (pd.DataFrame): Classified data
            scaling_results (Dict): Three-way scaling analysis results
            cv_results (Dict): Cross-validation results
        """
        log.info("Creating enhanced visualizations with three-category analysis and CV results...")

        # Create comprehensive 3x3 visualization
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        fig.suptitle('Enhanced SNDi Scaling Analysis: Three-Category Classification with Cross-Validation',
                     fontsize=20, fontweight='bold', y=0.98)

        # Panel A1: Three-category slope distributions
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_three_category_slopes(ax1, data, scaling_results)

        # Panel B1: Cross-validation performance by income group
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cv_performance_by_income(ax2, cv_results)

        # Panel C1: Slope progression validation
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_slope_progression(ax3, scaling_results)

        # Panel A2: Regional three-category comparison
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_regional_three_category(ax4, data, scaling_results)

        # Panel B2: CV robustness metrics
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_cv_robustness(ax5, cv_results)

        # Panel C2: SNDi distribution by income group
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_sndi_income_distribution(ax6, data)

        # Panel A3: Pairwise effect sizes
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_pairwise_effects(ax7, scaling_results)

        # Panel B3: Cross-validation fold success
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_cv_fold_success(ax8, cv_results)

        # Panel C3: Summary assessment matrix
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_summary_assessment(ax9, scaling_results, cv_results)

        # Save visualization
        today = datetime.now().strftime('%Y-%m-%d')
        output_file = self.figures_dir / f'enhanced_sndi_three_category_cv_analysis_{today}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        log.info(f"Saved enhanced visualization to {output_file}")

        overlay_file = self._create_regression_overlay_plot(scaling_results)
        per_capita_file = self._create_per_capita_line_plot(scaling_results)
        combined_per_capita_file = self._create_compact_vs_noncompact_plot(data)

        return output_file, overlay_file, per_capita_file, combined_per_capita_file

    def _create_regression_overlay_plot(self, scaling_results: Dict) -> Optional[Path]:
        """Plot regression lines for all SNDi categories on a shared figure."""

        if 'total_built_mass' not in scaling_results:
            log.warning("Cannot create regression overlay plot: total_built_mass results missing")
            return None

        mass_results = scaling_results['total_built_mass']
        categories = [cat for cat in self.sndi_categories if cat in mass_results]

        if len(categories) < 2:
            log.warning("Insufficient categories for regression overlay plot")
            return None

        colors = {'Compact': '#2E86AB', 'Medium': '#A23B72', 'Sprawl': '#F18F01'}

        # Determine shared x-range across categories
        ranges = [mass_results[cat].get('population_range') for cat in categories if mass_results[cat].get('population_range')]
        if not ranges:
            log.warning("Population ranges unavailable for regression overlay plot")
            return None

        global_min = max(1.0, min(r[0] for r in ranges))
        global_max = min(6.0, max(r[1] for r in ranges))
        if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
            log.warning("Invalid population range for regression overlay plot")
            return None

        x_vals = np.linspace(global_min, global_max, 200)

        fig, ax = plt.subplots(figsize=(10, 7))

        for category in categories:
            cat_results = mass_results[category]
            binned_stats = cat_results.get('binned_regression')
            if binned_stats:
                b_slope = binned_stats['slope']
                b_intercept = binned_stats['intercept']
                y_binned = b_slope * x_vals + b_intercept
                ax.plot(
                    x_vals,
                    y_binned,
                    color=colors.get(category, 'gray'),
                    linestyle='--',
                    linewidth=2,
                    alpha=0.8,
                    label=f"{category} (Binned β={b_slope:.3f})"
                )

                # Scatter bin centroids for context
                for bin_record in cat_results.get('bin_summary', []):
                    ax.scatter(
                        bin_record['log10_population_mean'],
                        bin_record['log_mass_mean'],
                        color=colors.get(category, 'gray'),
                        edgecolors='black',
                        linewidths=0.5,
                        alpha=0.6,
                        s=50
                    )

        ax.set_xlabel('log10(Population 2015)', fontweight='bold')
        ax.set_ylabel('log10(Total Built Mass Tons)', fontweight='bold')
        ax.set_title('Three-Category Regression Comparison\n(Raw vs Binned Fits)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        today = datetime.now().strftime('%Y-%m-%d')
        overlay_file = self.figures_dir / f'enhanced_sndi_three_category_regression_overlay_{today}.png'
        plt.savefig(overlay_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        log.info(f"Saved regression overlay plot to {overlay_file}")
        return overlay_file

    def _create_per_capita_line_plot(self, scaling_results: Dict) -> Optional[Path]:
        """Plot predicted material mass per capita lines for each SNDi category."""

        if 'total_built_mass' not in scaling_results:
            log.warning("Cannot create per-capita plot: total_built_mass results missing")
            return None

        mass_results = scaling_results['total_built_mass']
        categories = [cat for cat in self.sndi_categories if cat in mass_results]

        if len(categories) < 2:
            log.warning("Insufficient categories for per-capita plot")
            return None

        colors = {'Compact': '#2E86AB', 'Medium': '#A23B72', 'Sprawl': '#F18F01'}

        ranges = [mass_results[cat].get('population_range') for cat in categories if mass_results[cat].get('population_range')]
        if not ranges:
            log.warning("Population ranges unavailable for per-capita plot")
            return None

        global_min = max(1.0, min(r[0] for r in ranges))
        global_max = min(6.0, max(r[1] for r in ranges))
        if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
            log.warning("Invalid population range for per-capita plot")
            return None

        x_vals = np.linspace(global_min, global_max, 200)

        fig, ax = plt.subplots(figsize=(10, 6))

        for category in categories:
            cat_results = mass_results[category]
            slope = cat_results.get('binned_regression', {}).get('slope', cat_results['slope'])
            intercept = cat_results.get('binned_regression', {}).get('intercept', cat_results['intercept'])
            color = colors.get(category, 'gray')

            y_vals = (slope - 1.0) * x_vals + intercept
            ax.plot(x_vals, y_vals, color=color, linewidth=2, label=f"{category} (β={slope:.3f})")

        ax.set_xlabel('log10(Population 2015)', fontweight='bold')
        ax.set_ylabel('log10(Material Mass per Capita, tons/person)', fontweight='bold')
        ax.set_title('Material Mass per Capita vs Population Density\n(Binned Regression Lines)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        today = datetime.now().strftime('%Y-%m-%d')
        output_path = self.figures_dir / f'enhanced_sndi_per_capita_lines_{today}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        log.info(f"Saved per-capita line plot to {output_path}")
        return output_path

    def _create_compact_vs_noncompact_plot(self, data: pd.DataFrame) -> Optional[Path]:
        """Plot per-capita regression lines comparing Compact vs Medium+Sprawl."""

        required_cols = {'sndi_category', 'population_2015', 'total_built_mass_tons'}
        if not required_cols.issubset(data.columns):
            log.warning("Cannot create compact vs non-compact plot: required columns missing")
            return None

        subset = data[data['sndi_category'].isin(['Compact', 'Medium', 'Sprawl'])].copy()
        if subset.empty:
            log.warning("No data for compact vs non-compact plot")
            return None

        subset = subset[(subset['population_2015'] > 0) & (subset['total_built_mass_tons'] > 0)].copy()
        if subset.empty:
            log.warning("Filtered data empty for compact vs non-compact plot")
            return None

        subset['log10_population'] = np.log10(subset['population_2015'])
        subset['log10_total_mass'] = np.log10(subset['total_built_mass_tons'])
        subset['category_group'] = np.where(subset['sndi_category'] == 'Compact', 'Compact', 'Medium+Sprawl')

        ranges = subset.groupby('category_group')['log10_population'].agg(['min', 'max'])
        global_min = max(1.0, ranges['min'].max())
        global_max = min(6.0, ranges['max'].min())

        if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
            log.warning("Invalid population range for compact vs non-compact plot")
            return None

        x_vals = np.linspace(global_min, global_max, 200)
        colors = {'Compact': '#2E86AB', 'Medium+Sprawl': '#A23B72'}

        fig, ax = plt.subplots(figsize=(10, 6))

        for category in ['Compact', 'Medium+Sprawl']:
            category_data = subset[subset['category_group'] == category]
            if len(category_data) < 10:
                log.warning(f"Insufficient data for compact vs non-compact plot: {category}")
                continue

            bin_result = self._compute_binned_regression(category_data, 'log10_total_mass')
            if bin_result:
                _, binned_stats = bin_result
                slope = binned_stats['slope']
                intercept = binned_stats['intercept']
            else:
                slope, intercept, _, _, _ = stats.linregress(
                    category_data['log10_population'],
                    category_data['log10_total_mass']
                )

            y_vals = (slope - 1.0) * x_vals + intercept
            ax.plot(
                x_vals,
                y_vals,
                color=colors.get(category, 'gray'),
                linewidth=2,
                label=f"{category} (β={slope:.3f})"
            )

        ax.set_xlabel('log10(Population 2015)', fontweight='bold')
        ax.set_ylabel('log10(Material Mass per Capita, tons/person)', fontweight='bold')
        ax.set_title('Compact vs Medium+Sprawl\nPer-Capita Mass Scaling (Binned Regression Lines)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        today = datetime.now().strftime('%Y-%m-%d')
        output_path = self.figures_dir / f'enhanced_sndi_per_capita_compact_vs_noncompact_{today}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        log.info(f"Saved compact vs non-compact per-capita plot to {output_path}")
        return output_path

    def _plot_three_category_slopes(self, ax, data, scaling_results):
        """Panel A1: Three-category slope distributions with confidence intervals."""
        if 'total_built_mass' not in scaling_results:
            ax.text(0.5, 0.5, 'No scaling results\navailable', ha='center', va='center',
                   transform=ax.transAxes)
            return

        mass_results = scaling_results['total_built_mass']
        categories = [cat for cat in self.sndi_categories if cat in mass_results]

        if not categories:
            ax.text(0.5, 0.5, 'No category results\navailable', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Colors for three categories
        colors = {'Compact': '#2E86AB', 'Medium': '#A23B72', 'Sprawl': '#F18F01'}

        positions = range(len(categories))
        slopes = [mass_results[cat]['slope'] for cat in categories]
        std_errs = [mass_results[cat]['std_err'] for cat in categories]
        intercepts = [mass_results[cat]['intercept'] for cat in categories]

        binned_slopes = []
        binned_intercepts = []
        for cat in categories:
            binned_stats = mass_results[cat].get('binned_regression')
            if binned_stats:
                binned_slopes.append(binned_stats['slope'])
                binned_intercepts.append(binned_stats['intercept'])
            else:
                binned_slopes.append(np.nan)
                binned_intercepts.append(np.nan)

        # Bar plot with error bars
        bars = ax.bar(positions, slopes, yerr=std_errs, capsize=5,
                     color=[colors[cat] for cat in categories], alpha=0.7,
                     edgecolor='black', linewidth=1)

        # Overlay binned slope markers if available
        if not np.all(np.isnan(binned_slopes)):
            ax.plot(positions, binned_slopes, color='black', marker='D', linestyle='--',
                    linewidth=1.5, label='Binned β')

        # Add value labels
        for i, (bar, slope, stderr) in enumerate(zip(bars, slopes, std_errs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stderr + 0.01,
                   f'{slope:.3f}', ha='center', va='bottom', fontweight='bold')

        # Reference line
        ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Target β=0.75')

        ax.set_xlabel('SNDi Category', fontweight='bold')
        ax.set_ylabel('Scaling Slope (β)', fontweight='bold', color='black')
        ax.set_title('A. Three-Category Scaling Parameters\n(Total Built Mass)', fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(categories)
        ax.tick_params(axis='y', labelcolor='black')
        ax.grid(True, alpha=0.3)

        # Plot intercepts on secondary axis
        ax_intercept = ax.twinx()
        ax_intercept.plot(positions, intercepts, color='#6C7A89', marker='o', linestyle='-',
                          linewidth=1.8, label='Intercept (β₀)')

        if not np.all(np.isnan(binned_intercepts)):
            ax_intercept.plot(positions, binned_intercepts, color='#34495E', marker='s', linestyle=':',
                              linewidth=1.8, label='Binned Intercept')

        ax_intercept.set_ylabel('Intercept (log α)', fontweight='bold', color='#34495E')
        ax_intercept.tick_params(axis='y', labelcolor='#34495E')

        # Combined legend
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax_intercept.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc='upper left', frameon=True)

    def _plot_cv_performance_by_income(self, ax, cv_results):
        """Panel B1: Cross-validation performance stratified by income group."""
        income_groups = [ig for ig in cv_results.keys() if ig != 'cross_income_analysis']

        if not income_groups:
            ax.text(0.5, 0.5, 'No CV results\navailable', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Collect CV performance metrics
        performance_data = []
        for income_group in income_groups:
            income_result = cv_results[income_group]
            cv_summary = income_result['cv_summary']

            for category, cat_summary in cv_summary['category_summaries'].items():
                performance_data.append({
                    'income_group': income_group,
                    'category': category,
                    'cv_percent': cat_summary['cv_percent'],
                    'mean_r2': cat_summary['mean_r2'],
                    'n_folds': cat_summary['n_folds']
                })

        if not performance_data:
            ax.text(0.5, 0.5, 'No performance data\navailable', ha='center', va='center',
                   transform=ax.transAxes)
            return

        perf_df = pd.DataFrame(performance_data)

        # Create grouped scatter plot
        colors = {'Compact': '#2E86AB', 'Medium': '#A23B72', 'Sprawl': '#F18F01'}

        for category in perf_df['category'].unique():
            cat_data = perf_df[perf_df['category'] == category]
            ax.scatter(cat_data['cv_percent'], cat_data['mean_r2'],
                      c=colors.get(category, 'gray'), label=category, s=100, alpha=0.7)

        # Reference lines
        ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='CV threshold (5%)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Good R² (0.6)')

        ax.set_xlabel('Slope CV (%)', fontweight='bold')
        ax.set_ylabel('Mean Cross-Validation R²', fontweight='bold')
        ax.set_title('B. CV Performance by Income Group\n(Consistency vs Quality)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_slope_progression(self, ax, scaling_results):
        """Panel C1: Slope progression validation across mass types."""
        if 'slope_progression_test' not in scaling_results:
            ax.text(0.5, 0.5, 'No progression test\nresults available', ha='center', va='center',
                   transform=ax.transAxes)
            return

        progression_results = scaling_results['slope_progression_test']

        mass_types = list(progression_results.keys())
        if not mass_types:
            ax.text(0.5, 0.5, 'No progression data\navailable', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Plot progression for each mass type
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        x_positions = [0, 1, 2]  # Compact, Medium, Sprawl

        for i, mass_type in enumerate(mass_types):
            result = progression_results[mass_type]
            slopes = [result['compact_slope'], result['medium_slope'], result['sprawl_slope']]

            # Line plot showing progression
            line_style = '-' if result['progression_satisfied'] else '--'
            alpha = 0.8 if result['progression_satisfied'] else 0.5

            ax.plot(x_positions, slopes, marker='o', linewidth=2, markersize=8,
                   linestyle=line_style, alpha=alpha, label=f'{mass_type}')

        ax.set_xlabel('SNDi Category', fontweight='bold')
        ax.set_ylabel('Scaling Slope (β)', fontweight='bold')
        ax.set_title('C. Slope Progression Test\n(Compact > Medium > Sprawl)', fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['Compact', 'Medium', 'Sprawl'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_regional_three_category(self, ax, data, scaling_results):
        """Panel A2: Regional analysis with three categories."""
        # Simplified regional plot
        if 'region' not in data.columns:
            ax.text(0.5, 0.5, 'No regional data\navailable', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Count by region and category
        regional_counts = data.groupby(['region', 'sndi_category']).size().unstack(fill_value=0)

        # Plot stacked bars
        regional_counts.plot(kind='bar', stacked=True, ax=ax,
                           color=['#2E86AB', '#A23B72', '#F18F01'])

        ax.set_xlabel('Region', fontweight='bold')
        ax.set_ylabel('Number of Neighborhoods', fontweight='bold')
        ax.set_title('A2. Regional SNDi Distribution\n(Three Categories)', fontweight='bold')
        ax.legend(title='SNDi Category')
        ax.tick_params(axis='x', rotation=45)

    def _plot_cv_robustness(self, ax, cv_results):
        """Panel B2: CV robustness metrics visualization."""
        # Extract robustness metrics
        income_groups = [ig for ig in cv_results.keys() if ig != 'cross_income_analysis']

        if not income_groups:
            ax.text(0.5, 0.5, 'No CV robustness\ndata available', ha='center', va='center',
                   transform=ax.transAxes)
            return

        robustness_data = []
        for income_group in income_groups:
            income_result = cv_results[income_group]
            assessment = income_result['cv_summary']['overall_assessment']

            robustness_data.append({
                'income_group': income_group,
                'successful_categories': assessment['successful_categories'],
                'high_consistency': assessment['high_consistency_categories'],
                'cv_quality': assessment['cv_quality']
            })

        rob_df = pd.DataFrame(robustness_data)

        # Stacked bar chart
        x_pos = range(len(income_groups))
        ax.bar(x_pos, rob_df['high_consistency'], label='High Consistency', color='green', alpha=0.7)
        ax.bar(x_pos, rob_df['successful_categories'] - rob_df['high_consistency'],
               bottom=rob_df['high_consistency'], label='Moderate Consistency', color='orange', alpha=0.7)

        ax.set_xlabel('Income Group', fontweight='bold')
        ax.set_ylabel('Number of Categories', fontweight='bold')
        ax.set_title('B2. CV Robustness by Income Group\n(Category Consistency)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rob_df['income_group'], rotation=45)
        ax.legend()

    def _plot_sndi_income_distribution(self, ax, data):
        """Panel C2: SNDi distribution by income group."""
        # Box plot of SNDi values by income group
        income_groups = sorted(data['Income group'].unique())
        sndi_data = [data[data['Income group'] == ig]['avg_sndi'] for ig in income_groups]

        box_plot = ax.boxplot(sndi_data, labels=income_groups, patch_artist=True)

        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors[:len(income_groups)]):
            patch.set_facecolor(color)

        # Add threshold lines
        ax.axhline(y=2.0, color='blue', linestyle='--', alpha=0.7, label='Compact threshold')
        ax.axhline(y=5.5, color='red', linestyle='--', alpha=0.7, label='Sprawl threshold')

        ax.set_xlabel('Income Group', fontweight='bold')
        ax.set_ylabel('SNDi Value', fontweight='bold')
        ax.set_title('C2. SNDi Distribution by\nIncome Group', fontweight='bold')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    def _plot_pairwise_effects(self, ax, scaling_results):
        """Panel A3: Pairwise effect sizes."""
        if 'total_built_mass' not in scaling_results:
            ax.text(0.5, 0.5, 'No pairwise comparison\ndata available', ha='center', va='center',
                   transform=ax.transAxes)
            return

        pairwise_comps = scaling_results['total_built_mass'].get('pairwise_comparisons', {})

        if not pairwise_comps:
            ax.text(0.5, 0.5, 'No pairwise comparisons\navailable', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Extract effect sizes
        comparisons = list(pairwise_comps.keys())
        effect_sizes = [pairwise_comps[comp]['slope_difference'] for comp in comparisons]

        # Bar plot of effect sizes
        colors = ['green' if es > 0 else 'red' for es in effect_sizes]
        bars = ax.bar(range(len(comparisons)), effect_sizes, color=colors, alpha=0.7)

        # Add value labels
        for bar, es in zip(bars, effect_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                   f'{es:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Comparison', fontweight='bold')
        ax.set_ylabel('Slope Difference (β₁ - β₂)', fontweight='bold')
        ax.set_title('A3. Pairwise Effect Sizes\n(Total Built Mass)', fontweight='bold')
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons], rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_cv_fold_success(self, ax, cv_results):
        """Panel B3: Cross-validation fold success rates."""
        income_groups = [ig for ig in cv_results.keys() if ig != 'cross_income_analysis']

        if not income_groups:
            ax.text(0.5, 0.5, 'No CV fold\ndata available', ha='center', va='center',
                   transform=ax.transAxes)
            return

        # Extract fold success data
        success_data = []
        for income_group in income_groups:
            income_result = cv_results[income_group]
            successful = income_result['cv_summary']['successful_folds']
            total_attempted = len(income_result['fold_results']) + len(income_result['failed_folds'])

            success_data.append({
                'income_group': income_group,
                'successful': successful,
                'failed': total_attempted - successful,
                'success_rate': successful / total_attempted if total_attempted > 0 else 0
            })

        # Stacked bar chart
        x_pos = range(len(success_data))
        successful = [d['successful'] for d in success_data]
        failed = [d['failed'] for d in success_data]

        ax.bar(x_pos, successful, label='Successful', color='green', alpha=0.7)
        ax.bar(x_pos, failed, bottom=successful, label='Failed', color='red', alpha=0.7)

        # Add success rate labels
        for i, data in enumerate(success_data):
            total = data['successful'] + data['failed']
            ax.text(i, total + 0.1, f"{data['success_rate']:.0%}", ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Income Group', fontweight='bold')
        ax.set_ylabel('Number of Folds', fontweight='bold')
        ax.set_title('B3. CV Fold Success Rates\n(Successful vs Failed)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d['income_group'] for d in success_data], rotation=45)
        ax.legend()

    def _plot_summary_assessment(self, ax, scaling_results, cv_results):
        """Panel C3: Summary assessment matrix."""
        # Create summary table
        ax.axis('off')

        # Extract key findings
        findings = []

        # Slope progression results
        if 'slope_progression_test' in scaling_results:
            progression_results = scaling_results['slope_progression_test']
            confirmed_progressions = sum(1 for result in progression_results.values()
                                       if result['progression_satisfied'])
            total_progressions = len(progression_results)
            findings.append(f"Slope Progression: {confirmed_progressions}/{total_progressions} confirmed")

        # CV robustness
        income_groups = [ig for ig in cv_results.keys() if ig != 'cross_income_analysis']
        robust_groups = 0
        for income_group in income_groups:
            if cv_results[income_group]['cv_summary']['overall_assessment']['sndi_effects_robust']:
                robust_groups += 1
        findings.append(f"CV Robustness: {robust_groups}/{len(income_groups)} income groups")

        # Cross-income consistency
        if 'cross_income_analysis' in cv_results:
            consistency = cv_results['cross_income_analysis']['overall_consistency']['universality_assessment']
            findings.append(f"Cross-Income Consistency: {consistency}")

        # Create summary text
        summary_text = "ENHANCED SNDi ANALYSIS SUMMARY\n\n"
        summary_text += "\n".join(findings)
        total_neighborhoods = sum(cv_results.get(ig, {}).get('data_summary', {}).get('total_neighborhoods', 0) for ig in income_groups)
        summary_text += f"\n\nTotal Neighborhoods: {total_neighborhoods if income_groups else 'N/A'}"
        summary_text += f"\nIncome Groups Analyzed: {len(income_groups)}"
        summary_text += f"\nSNDi Categories: {len(self.sndi_categories)}"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        ax.set_title('C3. Summary Assessment\n(Key Findings)', fontweight='bold')

    def export_enhanced_results(self, data: pd.DataFrame, scaling_results: Dict, cv_results: Dict):
        """
        Export comprehensive enhanced results to CSV files.

        Args:
            data (pd.DataFrame): Classified global dataset
            scaling_results (Dict): Three-way scaling analysis results
            cv_results (Dict): Cross-validation results
        """
        log.info("Exporting enhanced results...")

        today = datetime.now().strftime('%Y-%m-%d')

        # Export main dataset
        main_output = self.processed_data_dir / f'enhanced_sndi_three_category_merged_data_{today}.csv'
        data.to_csv(main_output, index=False)
        log.info(f"Exported enhanced dataset to {main_output}")

        # Export three-category scaling results
        scaling_summary = []
        bin_level_summary = []
        for mass_type, results in scaling_results.items():
            if mass_type == 'slope_progression_test':
                continue
            for category, stats in results.items():
                if isinstance(stats, dict) and 'slope' in stats:
                    summary_entry = {
                        'mass_type': mass_type,
                        'sndi_category': category,
                        'n_points': stats['n_points'],
                        'slope': stats['slope'],
                        'intercept': stats['intercept'],
                        'r_squared': stats['r_squared'],
                        'p_value': stats['p_value'],
                        'std_err': stats['std_err']
                    }

                    if 'binned_regression' in stats:
                        summary_entry.update({
                            'binned_slope': stats['binned_regression']['slope'],
                            'binned_intercept': stats['binned_regression']['intercept'],
                            'binned_r_squared': stats['binned_regression']['r_squared'],
                            'binned_std_err': stats['binned_regression']['std_err'],
                            'binned_n_bins': stats['binned_regression']['n_bins']
                        })

                    scaling_summary.append(summary_entry)

                    if 'bin_summary' in stats:
                        for bin_record in stats['bin_summary']:
                            bin_level_summary.append({
                                'mass_type': mass_type,
                                'sndi_category': category,
                                **bin_record
                            })

                    if 'mixed_effects' in stats:
                        summary_entry.update({
                            'mixed_slope': stats['mixed_effects']['slope'],
                            'mixed_intercept': stats['mixed_effects']['intercept'],
                            'mixed_random_intercept_sd': stats['mixed_effects']['random_intercept_sd'],
                            'mixed_residual_sd': stats['mixed_effects']['residual_sd'],
                            'mixed_n_groups': stats['mixed_effects']['n_groups'],
                            'mixed_n_obs': stats['mixed_effects']['n_obs'],
                            'mixed_converged': stats['mixed_effects']['converged'],
                            'mixed_aic': stats['mixed_effects']['aic'],
                            'mixed_bic': stats['mixed_effects']['bic']
                        })

        if scaling_summary:
            scaling_df = pd.DataFrame(scaling_summary)
            scaling_output = self.processed_data_dir / f'enhanced_sndi_three_category_scaling_results_{today}.csv'
            scaling_df.to_csv(scaling_output, index=False)
            log.info(f"Exported three-category scaling results to {scaling_output}")

        if bin_level_summary:
            bin_df = pd.DataFrame(bin_level_summary)
            bin_output = self.processed_data_dir / f'enhanced_sndi_bin_level_summary_{today}.csv'
            bin_df.to_csv(bin_output, index=False)
            log.info(f"Exported bin-level summaries to {bin_output}")

        # Export pairwise comparisons
        pairwise_summary = []
        for mass_type, results in scaling_results.items():
            if 'pairwise_comparisons' in results:
                for comparison, comp_data in results['pairwise_comparisons'].items():
                    pairwise_summary.append({
                        'mass_type': mass_type,
                        'comparison': comparison,
                        **comp_data
                    })

        if pairwise_summary:
            pairwise_df = pd.DataFrame(pairwise_summary)
            pairwise_output = self.processed_data_dir / f'enhanced_sndi_pairwise_comparisons_{today}.csv'
            pairwise_df.to_csv(pairwise_output, index=False)
            log.info(f"Exported pairwise comparisons to {pairwise_output}")

        # Export slope progression tests
        if 'slope_progression_test' in scaling_results:
            progression_summary = []
            for mass_type, prog_result in scaling_results['slope_progression_test'].items():
                progression_summary.append({
                    'mass_type': mass_type,
                    **prog_result
                })

            if progression_summary:
                prog_df = pd.DataFrame(progression_summary)
                prog_output = self.processed_data_dir / f'enhanced_sndi_slope_progression_tests_{today}.csv'
                prog_df.to_csv(prog_output, index=False)
                log.info(f"Exported slope progression tests to {prog_output}")

        # Export CV results
        cv_summary = []
        for income_group, income_result in cv_results.items():
            if income_group == 'cross_income_analysis':
                continue

            data_summary = income_result['data_summary']
            cv_assessment = income_result['cv_summary']['overall_assessment']

            cv_summary.append({
                'income_group': income_group,
                'total_countries': data_summary['total_countries'],
                'total_cities': data_summary['total_cities'],
                'total_neighborhoods': data_summary['total_neighborhoods'],
                'successful_folds': income_result['cv_summary']['successful_folds'],
                'successful_categories': cv_assessment['successful_categories'],
                'high_consistency_categories': cv_assessment['high_consistency_categories'],
                'cv_quality': cv_assessment['cv_quality'],
                'sndi_effects_robust': cv_assessment['sndi_effects_robust']
            })

        if cv_summary:
            cv_df = pd.DataFrame(cv_summary)
            cv_output = self.processed_data_dir / f'enhanced_sndi_cv_results_by_income_{today}.csv'
            cv_df.to_csv(cv_output, index=False)
            log.info(f"Exported CV results to {cv_output}")

        # Export complete results as JSON
        complete_results = {
            'analysis_date': today,
            'data_summary': {
                'total_neighborhoods': len(data),
                'countries': data['CTR_MN_ISO'].nunique(),
                'cities': data['ID_HDC_G0'].nunique(),
                'sndi_distribution': data['sndi_category'].value_counts().to_dict(),
                'income_distribution': data['Income group'].value_counts().to_dict()
            },
            'scaling_results': scaling_results,
            'cv_results': cv_results
        }

        json_output = self.processed_data_dir / f'enhanced_sndi_complete_analysis_{today}.json'
        with open(json_output, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        log.info(f"Exported complete results to {json_output}")

def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description='Enhanced global SNDi scaling analysis with three categories and cross-validation')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited data')
    parser.add_argument('--income-filter', type=str, help='Filter to specific income group for focused analysis')
    args = parser.parse_args()

    # Set project path (current directory of this script's parent)
    script_dir = Path(__file__).parent
    project_path = script_dir.parent

    log.info("Starting enhanced global SNDi scaling analysis with cross-validation...")
    log.info(f"Project path: {project_path}")

    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedSNDiScalingAnalyzer(project_path)

        # Load global data with income groups
        log.info("Step 1: Loading global data with income group classifications...")
        global_data = analyzer.load_global_data_with_income_groups()

        # Apply income filter if specified
        if args.income_filter:
            if 'Income group' in global_data.columns:
                global_data = global_data[global_data['Income group'] == args.income_filter]
                log.info(f"Filtered to {args.income_filter}: {len(global_data)} neighborhoods")
            else:
                log.warning("Income filtering requested but income data not available")

        # Apply debug sampling
        if args.debug:
            sample_size = min(5000, len(global_data))
            global_data = global_data.sample(n=sample_size, random_state=42)
            log.info(f"DEBUG MODE: Using sample of {len(global_data)} neighborhoods")

        # Classify neighborhoods into three SNDi categories
        log.info("Step 2: Classifying neighborhoods into three SNDi categories...")
        classified_data = analyzer.classify_three_category_sndi(global_data)

        # Perform three-way scaling analysis
        log.info("Step 3: Performing three-way scaling analysis...")
        scaling_results = analyzer.perform_three_way_scaling_analysis(classified_data)

        # Run income-stratified cross-validation
        log.info("Step 4: Running income-stratified cross-validation...")
        cv_results = analyzer.run_income_stratified_cross_validation(classified_data)

        # Create enhanced visualizations
        log.info("Step 5: Creating enhanced visualizations...")
        analyzer.create_enhanced_visualizations(classified_data, scaling_results, cv_results)

        # Export enhanced results
        log.info("Step 6: Exporting enhanced results...")
        analyzer.export_enhanced_results(classified_data, scaling_results, cv_results)

        # Print comprehensive summary
        log.info("\n" + "="*90)
        log.info("ENHANCED SNDi SCALING ANALYSIS - COMPREHENSIVE FINDINGS")
        log.info("="*90)

        total_neighborhoods = len(classified_data)
        log.info(f"Total neighborhoods analyzed: {total_neighborhoods:,}")

        # Three-category distribution
        category_counts = classified_data['sndi_category'].value_counts()
        for category, count in category_counts.items():
            pct = count / total_neighborhoods * 100
            log.info(f"{category} neighborhoods: {count:,} ({pct:.1f}%)")

        # Scaling results summary
        log.info("\nTHREE-CATEGORY SCALING SLOPES:")
        if 'total_built_mass' in scaling_results:
            for category in analyzer.sndi_categories:
                if category in scaling_results['total_built_mass']:
                    result = scaling_results['total_built_mass'][category]
                    log.info(f"{category:>8}: β={result['slope']:.4f}, R²={result['r_squared']:.3f}, n={result['n_points']:,}")

        # Slope progression test
        if 'slope_progression_test' in scaling_results:
            log.info("\nSLOPE PROGRESSION TEST:")
            for mass_type, prog_result in scaling_results['slope_progression_test'].items():
                log.info(f"{mass_type}: {prog_result['progression_interpretation']}")

        # Cross-validation summary
        income_groups_analyzed = [ig for ig in cv_results.keys() if ig != 'cross_income_analysis']
        log.info(f"\nCROSS-VALIDATION ANALYSIS: {len(income_groups_analyzed)} income groups tested")

        robust_income_groups = 0
        for income_group in income_groups_analyzed:
            assessment = cv_results[income_group]['cv_summary']['overall_assessment']
            if assessment['sndi_effects_robust']:
                robust_income_groups += 1
            log.info(f"  {income_group}: {assessment['cv_quality']} quality, "
                    f"{assessment['successful_categories']} categories successful")

        log.info(f"\nROBUSTNESS ASSESSMENT: {robust_income_groups}/{len(income_groups_analyzed)} income groups show robust SNDi effects")

        # Cross-income consistency
        if 'cross_income_analysis' in cv_results:
            cross_analysis = cv_results['cross_income_analysis']
            consistency = cross_analysis['overall_consistency']['universality_assessment']
            progression_rate = cross_analysis['overall_consistency']['progression_tests']['consistency_rate']
            log.info(f"CROSS-INCOME CONSISTENCY: {consistency} (progression confirmed in {progression_rate:.0%} of testable groups)")

        log.info("="*90)
        log.info("Enhanced three-category SNDi scaling analysis with cross-validation completed successfully!")

    except Exception as e:
        log.error(f"Enhanced analysis failed: {e}")
        raise e

if __name__ == "__main__":
    main()
