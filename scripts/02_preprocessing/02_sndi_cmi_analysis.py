#!/usr/bin/env python3
"""
SNDi-CMI Multi-level Analysis Script
====================================
Joins Street Network Disconnectedness Index (SNDi) with City Mass Index (CMI) data
to explore relationships between urban form and material scaling patterns at neighborhood and city levels.

CMI captures how much the mass of a city/neighborhood deviates from expected values based on urban scaling patterns.
Larger CMI values indicate built material mass is higher than expected from population size.

Author: Claude Code Assistant
Date: 2025-09-09
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class SNDiCMIAnalyzer:
    """Multi-level analyzer for SNDi-CMI relationships"""
    
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.raw_data_dir = os.path.join(project_dir, "data", "raw")
        self.processed_data_dir = os.path.join(project_dir, "data", "processed")
        self.figures_dir = os.path.join(project_dir, "figures")
        self.sndi_data = None
        self.cmi_data = None
        self.joined_data = None
        self.city_summary = None
        
    def load_data(self):
        """Load SNDi and CMI datasets"""
        logging.info("Loading datasets...")
        
        # Load SNDi data
        sndi_path = os.path.join(self.processed_data_dir, "01_neighborhood_SNDi_2025-09-09.csv")
        self.sndi_data = pd.read_csv(sndi_path)
        logging.info(f"Loaded SNDi data: {len(self.sndi_data):,} records")
        
        # Load CMI data
        cmi_path = os.path.join(self.raw_data_dir, "china_neighborhoods_cmi.csv")
        self.cmi_data = pd.read_csv(cmi_path)
        logging.info(f"Loaded CMI data: {len(self.cmi_data):,} records")
        
        # Filter SNDi data to China only
        china_sndi = self.sndi_data[self.sndi_data['CTR_MN_ISO'] == 'CHN'].copy()
        logging.info(f"China SNDi records: {len(china_sndi):,}")
        
        return china_sndi
    
    def join_datasets(self):
        """Perform inner join on H3 hexagon identifiers"""
        logging.info("Joining datasets on h3index...")
        
        china_sndi = self.load_data()
        
        # Perform inner join
        self.joined_data = pd.merge(
            china_sndi, 
            self.cmi_data, 
            on='h3index', 
            how='inner',
            suffixes=('_sndi', '_cmi')
        )
        
        logging.info(f"Joined dataset: {len(self.joined_data):,} matched neighborhoods")
        
        # Data quality checks
        self.data_quality_summary()
        
        return self.joined_data
    
    def data_quality_summary(self):
        """Generate data quality summary"""
        logging.info("=== DATA QUALITY SUMMARY ===")
        
        # Debug: check column names after merge
        logging.info(f"Available columns: {list(self.joined_data.columns)}")
        
        # SNDi data quality
        sndi_valid = self.joined_data['avg_sndi'].notna()
        sndi_count = sndi_valid.sum()
        sndi_pct = sndi_count / len(self.joined_data) * 100
        
        logging.info(f"SNDi data availability: {sndi_count:,}/{len(self.joined_data):,} ({sndi_pct:.1f}%)")
        
        if sndi_count > 0:
            sndi_stats = self.joined_data['avg_sndi'].describe()
            logging.info(f"SNDi range: {sndi_stats['min']:.2f} - {sndi_stats['max']:.2f}")
            logging.info(f"SNDi mean: {sndi_stats['mean']:.2f} ± {sndi_stats['std']:.2f}")
        
        # CMI data quality
        cmi_valid = self.joined_data['neighborhood_cmi'].notna()
        cmi_count = cmi_valid.sum()
        cmi_pct = cmi_count / len(self.joined_data) * 100
        
        logging.info(f"CMI data availability: {cmi_count:,}/{len(self.joined_data):,} ({cmi_pct:.1f}%)")
        
        if cmi_count > 0:
            cmi_stats = self.joined_data['neighborhood_cmi'].describe()
            logging.info(f"Neighborhood CMI range: {cmi_stats['min']:.2f} - {cmi_stats['max']:.2f}")
            logging.info(f"Neighborhood CMI mean: {cmi_stats['mean']:.2f} ± {cmi_stats['std']:.2f}")
        
        # City coverage - handle suffixed column names
        city_col = None
        for col in self.joined_data.columns:
            if 'ID_HDC_G0' in col:
                city_col = col
                break
        
        if city_col:
            n_cities = self.joined_data[city_col].nunique()
            logging.info(f"Cities represented: {n_cities:,}")
        else:
            logging.warning("Could not find city ID column")
        
        # Both SNDi and CMI available
        both_valid = sndi_valid & cmi_valid
        both_count = both_valid.sum()
        both_pct = both_count / len(self.joined_data) * 100
        
        logging.info(f"Records with both SNDi and CMI: {both_count:,}/{len(self.joined_data):,} ({both_pct:.1f}%)")
        
        return both_valid
    
    def remove_outliers(self, data, columns, method='iqr', factor=1.5):
        """Remove outliers using IQR or Z-score method"""
        clean_data = data.copy()
        outliers_removed = 0
        
        for col in columns:
            if method == 'iqr':
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                before_count = len(clean_data)
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
                outliers_removed += (before_count - len(clean_data))
                
                logging.info(f"{col} outliers (IQR): removed {before_count - len(clean_data)} points, bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            elif method == 'zscore':
                from scipy.stats import zscore
                z_scores = np.abs(zscore(clean_data[col]))
                before_count = len(clean_data)
                clean_data = clean_data[z_scores < factor]
                outliers_removed += (before_count - len(clean_data))
                
                logging.info(f"{col} outliers (Z-score): removed {before_count - len(clean_data)} points, threshold: {factor}")
        
        logging.info(f"Total outliers removed: {outliers_removed}")
        return clean_data
    
    def binned_regression_analysis(self, data, x_col, y_col, n_bins=10):
        """Perform binned averages regression"""
        logging.info(f"=== BINNED REGRESSION ANALYSIS ({n_bins} bins) ===")
        
        # Create bins based on x variable
        data_sorted = data.sort_values(x_col)
        data_sorted['bin'] = pd.cut(data_sorted[x_col], bins=n_bins, labels=False)
        
        # Calculate bin averages
        bin_stats = data_sorted.groupby('bin').agg({
            x_col: ['mean', 'count', 'std'],
            y_col: ['mean', 'std'],
        }).reset_index()
        
        # Flatten column names
        bin_stats.columns = ['bin', 'x_mean', 'x_count', 'x_std', 'y_mean', 'y_std']
        
        # Filter bins with sufficient data
        min_points_per_bin = max(5, len(data) // (n_bins * 3))  # At least 5 or 1/3 of average bin size
        bin_stats = bin_stats[bin_stats['x_count'] >= min_points_per_bin]
        
        logging.info(f"Using {len(bin_stats)} bins with ≥{min_points_per_bin} points each")
        
        if len(bin_stats) < 3:
            logging.warning("Insufficient bins for regression")
            return None
        
        # Bin-level regression
        from scipy.stats import linregress
        bin_slope, bin_intercept, bin_r, bin_p, bin_stderr = linregress(
            bin_stats['x_mean'], bin_stats['y_mean']
        )
        
        logging.info(f"Binned regression: β = {bin_slope:.3f} ± {bin_stderr:.3f}, R² = {bin_r**2:.3f}, p = {bin_p:.3f}")
        
        # Weighted regression (by bin size)
        weights = bin_stats['x_count'] / bin_stats['x_count'].sum()
        weighted_slope = np.sum(weights * (bin_stats['y_mean'] - bin_stats['y_mean'].mean()) * 
                               (bin_stats['x_mean'] - bin_stats['x_mean'].mean())) / \
                        np.sum(weights * (bin_stats['x_mean'] - bin_stats['x_mean'].mean())**2)
        
        logging.info(f"Weighted binned slope: β = {weighted_slope:.3f}")
        
        return {
            'bin_stats': bin_stats,
            'bin_slope': bin_slope, 'bin_intercept': bin_intercept,
            'bin_r2': bin_r**2, 'bin_p': bin_p, 'bin_stderr': bin_stderr,
            'weighted_slope': weighted_slope
        }

    def neighborhood_level_analysis(self):
        """Analyze SNDi-CMI relationships at neighborhood level"""
        logging.info("=== NEIGHBORHOOD-LEVEL ANALYSIS ===")
        
        # Filter to complete cases
        valid_data = self.joined_data.dropna(subset=['avg_sndi', 'neighborhood_cmi'])
        n_valid = len(valid_data)
        
        if n_valid < 10:
            logging.warning(f"Insufficient data for analysis: only {n_valid} complete cases")
            return None
        
        logging.info(f"Analyzing {n_valid:,} neighborhoods with complete data")
        
        # Original correlation analysis (with outliers)
        pearson_r, pearson_p = pearsonr(valid_data['avg_sndi'], valid_data['neighborhood_cmi'])
        spearman_r, spearman_p = spearmanr(valid_data['avg_sndi'], valid_data['neighborhood_cmi'])
        
        logging.info(f"Original Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.3f}")
        logging.info(f"Original Spearman correlation: ρ = {spearman_r:.3f}, p = {spearman_p:.3f}")
        
        # Remove outliers
        logging.info("--- OUTLIER REMOVAL ---")
        clean_data = self.remove_outliers(valid_data, ['avg_sndi', 'neighborhood_cmi'], method='iqr', factor=1.5)
        n_clean = len(clean_data)
        
        logging.info(f"After outlier removal: {n_clean:,} neighborhoods ({n_valid-n_clean:,} removed, {n_clean/n_valid*100:.1f}% retained)")
        
        if n_clean < 10:
            logging.warning("Too few observations after outlier removal")
            clean_data = valid_data  # Fall back to original data
            n_clean = n_valid
        
        # Clean data correlation analysis
        clean_pearson_r, clean_pearson_p = pearsonr(clean_data['avg_sndi'], clean_data['neighborhood_cmi'])
        clean_spearman_r, clean_spearman_p = spearmanr(clean_data['avg_sndi'], clean_data['neighborhood_cmi'])
        
        logging.info(f"Clean Pearson correlation: r = {clean_pearson_r:.3f}, p = {clean_pearson_p:.3f}")
        logging.info(f"Clean Spearman correlation: ρ = {clean_spearman_r:.3f}, p = {clean_spearman_p:.3f}")
        
        # Linear regression on clean data
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            clean_data['avg_sndi'], clean_data['neighborhood_cmi']
        )
        
        logging.info(f"Clean linear regression: β = {slope:.3f} ± {std_err:.3f}, R² = {r_value**2:.3f}, p = {p_value:.3f}")
        
        # Binned averages regression
        binned_results = self.binned_regression_analysis(clean_data, 'avg_sndi', 'neighborhood_cmi', n_bins=10)
        
        # Additional binned analysis for true_neighborhood_cmi if available
        true_binned_results = None
        if 'true_neighborhood_cmi' in clean_data.columns:
            logging.info("--- TRUE NEIGHBORHOOD CMI ANALYSIS ---")
            true_binned_results = self.binned_regression_analysis(clean_data, 'avg_sndi', 'true_neighborhood_cmi', n_bins=10)
        
        # Store results
        neighborhood_results = {
            'n_observations_original': n_valid,
            'n_observations_clean': n_clean,
            'original_pearson_r': pearson_r, 'original_pearson_p': pearson_p,
            'original_spearman_r': spearman_r, 'original_spearman_p': spearman_p,
            'clean_pearson_r': clean_pearson_r, 'clean_pearson_p': clean_pearson_p,
            'clean_spearman_r': clean_spearman_r, 'clean_spearman_p': clean_spearman_p,
            'regression_slope': slope, 'regression_intercept': intercept,
            'regression_r2': r_value**2, 'regression_p': p_value,
            'regression_stderr': std_err,
            'binned_results': binned_results,
            'true_binned_results': true_binned_results
        }
        
        return neighborhood_results, clean_data, valid_data
    
    def city_level_analysis(self):
        """Analyze SNDi-CMI relationships at city level"""
        logging.info("=== CITY-LEVEL ANALYSIS ===")
        
        # Find the correct column names (handle suffixed names after merge)
        city_id_col = None
        city_name_col = None
        
        for col in self.joined_data.columns:
            if 'ID_HDC_G0' in col:
                city_id_col = col
            elif 'UC_NM_MN' in col:
                city_name_col = col
        
        if not city_id_col or not city_name_col:
            logging.error(f"Could not find city columns. Available: {list(self.joined_data.columns)}")
            return None
        
        logging.info(f"Using city columns: {city_id_col}, {city_name_col}")
        
        # Aggregate data by city
        city_agg = self.joined_data.groupby([city_id_col, city_name_col]).agg({
            'avg_sndi': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'neighborhood_cmi': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'city_cmi': 'first',  # City-level CMI should be the same for all neighborhoods in a city
            'log10_population': 'mean',
            'log10_mass': 'mean'
        }).reset_index()
        
        # Flatten column names
        city_agg.columns = [
            'city_id', 'city_name',
            'sndi_count', 'sndi_mean', 'sndi_median', 'sndi_std', 'sndi_min', 'sndi_max',
            'cmi_count', 'cmi_mean', 'cmi_median', 'cmi_std', 'cmi_min', 'cmi_max',
            'city_cmi', 'mean_log10_pop', 'mean_log10_mass'
        ]
        
        # Filter cities with sufficient data
        min_neighborhoods = 3
        city_valid = city_agg[
            (city_agg['sndi_count'] >= min_neighborhoods) & 
            (city_agg['cmi_count'] >= min_neighborhoods) &
            city_agg['sndi_mean'].notna() &
            city_agg['cmi_mean'].notna()
        ].copy()
        
        n_cities = len(city_valid)
        logging.info(f"Cities with ≥{min_neighborhoods} neighborhoods: {n_cities}")
        
        if n_cities < 5:
            logging.warning(f"Insufficient cities for analysis: only {n_cities} valid cities")
            return None
        
        # City-level correlations
        pearson_r, pearson_p = pearsonr(city_valid['sndi_mean'], city_valid['cmi_mean'])
        spearman_r, spearman_p = spearmanr(city_valid['sndi_mean'], city_valid['cmi_mean'])
        
        logging.info(f"City-level Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.3f}")
        logging.info(f"City-level Spearman correlation: ρ = {spearman_r:.3f}, p = {spearman_p:.3f}")
        
        # City CMI vs neighborhood SNDi patterns
        if 'city_cmi' in city_valid.columns:
            city_cmi_corr, city_cmi_p = pearsonr(city_valid['sndi_mean'], city_valid['city_cmi'])
            logging.info(f"City CMI vs mean neighborhood SNDi: r = {city_cmi_corr:.3f}, p = {city_cmi_p:.3f}")
        
        self.city_summary = city_valid
        
        city_results = {
            'n_cities': n_cities,
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p,
        }
        
        return city_results, city_valid
    
    def generate_visualizations(self, clean_data=None, original_data=None, binned_results=None, true_binned_results=None):
        """Generate analysis visualizations"""
        logging.info("Generating visualizations...")
        
        # Create figure directory
        import os
        fig_dir = self.figures_dir
        os.makedirs(fig_dir, exist_ok=True)
        
        # Use provided data or filter to complete cases for plotting
        if original_data is not None:
            plot_data_orig = original_data
        else:
            plot_data_orig = self.joined_data.dropna(subset=['avg_sndi', 'neighborhood_cmi'])
        
        if clean_data is not None:
            plot_data_clean = clean_data
        else:
            plot_data_clean = plot_data_orig
        
        if len(plot_data_orig) < 10:
            logging.warning("Insufficient data for visualizations")
            return
        
        # 1. Extended analysis plot with outlier removal and binned analysis
        n_plots = 4 if true_binned_results is not None else 3
        fig, axes = plt.subplots(2, n_plots, figsize=(6*n_plots, 12))
        fig.suptitle('SNDi-CMI Analysis: Multi-level CMI Comparison', fontsize=16, fontweight='bold')
        
        # Original data scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(plot_data_orig['avg_sndi'], plot_data_orig['neighborhood_cmi'], 
                   alpha=0.4, s=15, color='lightcoral', label='Original data')
        if len(plot_data_orig) > 10:
            z_orig = np.polyfit(plot_data_orig['avg_sndi'], plot_data_orig['neighborhood_cmi'], 1)
            p_orig = np.poly1d(z_orig)
            ax1.plot(plot_data_orig['avg_sndi'], p_orig(plot_data_orig['avg_sndi']), 
                    "r-", alpha=0.8, linewidth=2, label='Original regression')
        ax1.set_xlabel('Average SNDi')
        ax1.set_ylabel('Neighborhood CMI')
        ax1.set_title('Original Data (with outliers)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Clean data scatter plot
        ax2 = axes[0, 1]
        ax2.scatter(plot_data_clean['avg_sndi'], plot_data_clean['neighborhood_cmi'], 
                   alpha=0.6, s=20, color='steelblue', label='Clean data')
        if len(plot_data_clean) > 10:
            z_clean = np.polyfit(plot_data_clean['avg_sndi'], plot_data_clean['neighborhood_cmi'], 1)
            p_clean = np.poly1d(z_clean)
            ax2.plot(plot_data_clean['avg_sndi'], p_clean(plot_data_clean['avg_sndi']), 
                    "b-", alpha=0.8, linewidth=2, label='Clean regression')
        ax2.set_xlabel('Average SNDi')
        ax2.set_ylabel('Neighborhood CMI')
        ax2.set_title('Clean Data (outliers removed)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Binned regression plot
        ax3 = axes[0, 2]
        if binned_results is not None and binned_results['bin_stats'] is not None:
            bin_stats = binned_results['bin_stats']
            # Scatter plot of bin averages
            ax3.scatter(bin_stats['x_mean'], bin_stats['y_mean'], 
                       s=bin_stats['x_count']*3, alpha=0.7, color='orange', 
                       label='Bin averages')
            
            # Error bars
            ax3.errorbar(bin_stats['x_mean'], bin_stats['y_mean'],
                        xerr=bin_stats['x_std'], yerr=bin_stats['y_std'],
                        fmt='none', alpha=0.5, color='gray')
            
            # Binned regression line
            x_range = np.linspace(bin_stats['x_mean'].min(), bin_stats['x_mean'].max(), 100)
            y_binned = binned_results['bin_slope'] * x_range + binned_results['bin_intercept']
            ax3.plot(x_range, y_binned, 'g-', linewidth=2, 
                    label=f"Binned regression (R²={binned_results['bin_r2']:.3f})")
            
            ax3.set_xlabel('Average SNDi (bin means)')
            ax3.set_ylabel('Neighborhood CMI (bin means)')
            ax3.set_title('Binned Regression Analysis')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No binned analysis\navailable', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Binned Analysis')
        
        # True neighborhood CMI binned analysis (if available)
        if true_binned_results is not None:
            ax4 = axes[0, 3]
            bin_stats_true = true_binned_results['bin_stats']
            
            # Scatter plot of bin averages for true CMI
            ax4.scatter(bin_stats_true['x_mean'], bin_stats_true['y_mean'], 
                       s=bin_stats_true['x_count']*3, alpha=0.7, color='purple', 
                       label='True CMI bin averages')
            
            # Error bars
            ax4.errorbar(bin_stats_true['x_mean'], bin_stats_true['y_mean'],
                        xerr=bin_stats_true['x_std'], yerr=bin_stats_true['y_std'],
                        fmt='none', alpha=0.5, color='gray')
            
            # True CMI binned regression line
            x_range_true = np.linspace(bin_stats_true['x_mean'].min(), bin_stats_true['x_mean'].max(), 100)
            y_true = true_binned_results['bin_slope'] * x_range_true + true_binned_results['bin_intercept']
            ax4.plot(x_range_true, y_true, 'm-', linewidth=2, 
                    label=f"True CMI regression (R²={true_binned_results['bin_r2']:.3f})")
            
            ax4.set_xlabel('Average SNDi (bin means)')
            ax4.set_ylabel('True Neighborhood CMI (bin means)')
            ax4.set_title('True CMI Binned Analysis\n(Country+City+Neighborhood)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        elif n_plots == 4:
            # Handle case where true_binned_results was expected but not available
            ax4 = axes[0, 3]
            ax4.text(0.5, 0.5, 'True neighborhood CMI\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('True CMI Analysis')
        
        # SNDi distribution comparison
        ax_sndi = axes[1, 0]
        ax_sndi.hist(plot_data_orig['avg_sndi'], bins=30, alpha=0.5, color='lightcoral', 
                    edgecolor='red', label='Original', density=True)
        ax_sndi.hist(plot_data_clean['avg_sndi'], bins=30, alpha=0.7, color='steelblue', 
                    edgecolor='blue', label='Clean', density=True)
        ax_sndi.set_xlabel('Average SNDi')
        ax_sndi.set_ylabel('Density')
        ax_sndi.set_title('SNDi Distribution Comparison')
        ax_sndi.grid(True, alpha=0.3)
        ax_sndi.legend()
        
        # CMI distribution comparison
        ax_cmi = axes[1, 1]
        ax_cmi.hist(plot_data_orig['neighborhood_cmi'], bins=30, alpha=0.5, color='lightcoral', 
                   edgecolor='red', label='Original', density=True)
        ax_cmi.hist(plot_data_clean['neighborhood_cmi'], bins=30, alpha=0.7, color='steelblue', 
                   edgecolor='blue', label='Clean', density=True)
        ax_cmi.set_xlabel('Neighborhood CMI')
        ax_cmi.set_ylabel('Density')
        ax_cmi.set_title('CMI Distribution Comparison')
        ax_cmi.grid(True, alpha=0.3)
        ax_cmi.legend()
        
        # True CMI distribution comparison (if available)
        if true_binned_results is not None and 'true_neighborhood_cmi' in plot_data_clean.columns:
            ax_true_cmi = axes[1, 2]
            ax_true_cmi.hist(plot_data_orig['true_neighborhood_cmi'], bins=30, alpha=0.5, color='lightcoral', 
                           edgecolor='red', label='Original', density=True)
            ax_true_cmi.hist(plot_data_clean['true_neighborhood_cmi'], bins=30, alpha=0.7, color='steelblue', 
                           edgecolor='blue', label='Clean', density=True)
            ax_true_cmi.set_xlabel('True Neighborhood CMI')
            ax_true_cmi.set_ylabel('Density')
            ax_true_cmi.set_title('True CMI Distribution\n(Country+City+Neighborhood)')
            ax_true_cmi.grid(True, alpha=0.3)
            ax_true_cmi.legend()
            
            # City-level analysis if available
            ax_city = axes[1, 3] if n_plots == 4 else axes[1, 2]
        else:
            # City-level analysis if available
            ax_city = axes[1, 2]
        if self.city_summary is not None and len(self.city_summary) > 5:
            ax_city.scatter(self.city_summary['sndi_mean'], self.city_summary['cmi_mean'],
                           alpha=0.7, s=50, color='orange')
            ax_city.set_xlabel('Mean City SNDi')
            ax_city.set_ylabel('Mean City CMI')
            ax_city.set_title('City Level: Mean SNDi vs Mean CMI')
            ax_city.grid(True, alpha=0.3)
            
            # Add city regression line
            if len(self.city_summary) > 5:
                z_city = np.polyfit(self.city_summary['sndi_mean'], self.city_summary['cmi_mean'], 1)
                p_city = np.poly1d(z_city)
                ax_city.plot(self.city_summary['sndi_mean'], p_city(self.city_summary['sndi_mean']), 
                            "r--", alpha=0.8, linewidth=2)
        else:
            ax_city.text(0.5, 0.5, 'Insufficient city-level data\nfor visualization', 
                        ha='center', va='center', transform=ax_city.transAxes, fontsize=12)
            ax_city.set_title('City Level Analysis')
        
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/sndi_cmi_robust_analysis_2025-09-09.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation matrix (original implementation)
        corr_vars = ['avg_sndi', 'neighborhood_cmi', 'log10_population', 'log10_mass']
        corr_data = plot_data_clean[corr_vars].dropna()
        
        if len(corr_data) > 10:
            plt.figure(figsize=(8, 6))
            correlation_matrix = corr_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix: SNDi, CMI, Population, and Mass (Clean Data)')
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/sndi_cmi_correlation_matrix_clean_2025-09-09.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logging.info(f"Enhanced visualizations saved to {fig_dir}/")
    
    def save_results(self, neighborhood_results=None):
        """Save analysis results and joined dataset"""
        logging.info("Saving analysis results...")
        
        # Save joined dataset
        output_path = os.path.join(self.processed_data_dir, "02_sndi_cmi_joined_2025-09-09.csv")
        self.joined_data.to_csv(output_path, index=False)
        logging.info(f"Joined dataset saved: {output_path}")
        
        # Save city-level summary if available
        if self.city_summary is not None:
            city_output_path = os.path.join(self.processed_data_dir, "02_city_level_sndi_cmi_summary_2025-09-09.csv")
            self.city_summary.to_csv(city_output_path, index=False)
            logging.info(f"City summary saved: {city_output_path}")
        
        # Save binned averages results if available
        if neighborhood_results is not None:
            # Save neighborhood CMI binned results
            binned_results = neighborhood_results.get('binned_results')
            if binned_results is not None and binned_results['bin_stats'] is not None:
                binned_path = os.path.join(self.processed_data_dir, "02_sndi_neighborhood_cmi_binned_2025-09-09.csv")
                bin_stats = binned_results['bin_stats'].copy()
                bin_stats['regression_slope'] = binned_results['bin_slope']
                bin_stats['regression_r2'] = binned_results['bin_r2']
                bin_stats['regression_p'] = binned_results['bin_p']
                bin_stats['weighted_slope'] = binned_results['weighted_slope']
                bin_stats.to_csv(binned_path, index=False)
                logging.info(f"Neighborhood CMI binned averages saved: {binned_path}")
            
            # Save true neighborhood CMI binned results if available
            true_binned_results = neighborhood_results.get('true_binned_results')
            if true_binned_results is not None and true_binned_results['bin_stats'] is not None:
                true_binned_path = os.path.join(self.processed_data_dir, "02_sndi_true_neighborhood_cmi_binned_2025-09-09.csv")
                true_bin_stats = true_binned_results['bin_stats'].copy()
                true_bin_stats['regression_slope'] = true_binned_results['bin_slope']
                true_bin_stats['regression_r2'] = true_binned_results['bin_r2']
                true_bin_stats['regression_p'] = true_binned_results['bin_p']
                true_bin_stats['weighted_slope'] = true_binned_results['weighted_slope']
                true_bin_stats.to_csv(true_binned_path, index=False)
                logging.info(f"True neighborhood CMI binned averages saved: {true_binned_path}")
        
        return output_path
    
    def run_full_analysis(self):
        """Execute complete SNDi-CMI analysis pipeline"""
        logging.info("Starting SNDi-CMI Multi-level Analysis")
        logging.info("=" * 50)
        
        try:
            # Step 1: Join datasets
            self.join_datasets()
            
            # Step 2: Neighborhood-level analysis with outlier removal and binned regression
            neighborhood_results = self.neighborhood_level_analysis()
            if neighborhood_results:
                neighborhood_results, clean_data, original_data = neighborhood_results
                binned_results = neighborhood_results.get('binned_results')
                true_binned_results = neighborhood_results.get('true_binned_results')
            else:
                clean_data = original_data = binned_results = true_binned_results = None
            
            # Step 3: City-level analysis
            city_results = self.city_level_analysis()
            if city_results:
                city_results, city_data = city_results
            
            # Step 4: Generate enhanced visualizations
            self.generate_visualizations(
                clean_data=clean_data, 
                original_data=original_data,
                binned_results=binned_results,
                true_binned_results=true_binned_results
            )
            
            # Step 5: Save results
            output_path = self.save_results(neighborhood_results)
            
            # Final summary
            logging.info("=" * 50)
            logging.info("ROBUST ANALYSIS COMPLETE")
            logging.info(f"Joined dataset: {len(self.joined_data):,} neighborhoods")
            
            if neighborhood_results:
                logging.info(f"Original neighborhood correlation: r = {neighborhood_results['original_pearson_r']:.3f}")
                logging.info(f"Clean neighborhood correlation: r = {neighborhood_results['clean_pearson_r']:.3f}")
                if binned_results:
                    logging.info(f"Binned regression: β = {binned_results['bin_slope']:.3f}, R² = {binned_results['bin_r2']:.3f}")
                if true_binned_results:
                    logging.info(f"True CMI binned regression: β = {true_binned_results['bin_slope']:.3f}, R² = {true_binned_results['bin_r2']:.3f}")
            
            if city_results:
                logging.info(f"City-level correlation: r = {city_results['pearson_r']:.3f}")
            
            logging.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Base directory (adjust as needed)
    # Get project directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Initialize analyzer
    analyzer = SNDiCMIAnalyzer(project_dir)
    
    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()