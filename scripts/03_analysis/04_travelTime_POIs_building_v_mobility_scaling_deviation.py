"""
Exploratory Analysis: Travel Time to POIs vs Building/Mobility Mass Scaling Deviations

This script explores relationships between neighborhood accessibility (travel time to POIs)
and deviations from expected building/mobility mass scaling laws.

Key research questions explored:
1. Do neighborhoods with better accessibility have different mass scaling patterns?
2. Is there a trade-off between building mass and mobility infrastructure?
3. Do different POI categories show different relationships with mass deviations?
4. Are there spatial patterns in the relationship between accessibility and mass?
5. How do quadrants differ in their accessibility profiles?

Output: Comprehensive set of visualizations and statistical summaries in results/poi_deviation_exploration/
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# POI categories
CATEGORIES = [
    'eating', 'moving', 'outdoor_activities', 'physical_exercise',
    'supplies', 'learning', 'services', 'health_care', 'cultural_activities'
]

# ============================================================================
# SECTION 1: Data Loading and Preparation
# ============================================================================

def load_and_merge_data(tt_path, dev_path):
    """
    Load travel time and deviation data, merge on h3index

    Parameters:
    -----------
    tt_path : str or Path
        Path to neighborhood_POIs_travel_time.gpkg
    dev_path : str or Path
        Path to neighborhood_deviations_building_v_mobility.gpkg

    Returns:
    --------
    gpd.GeoDataFrame
        Merged dataset with both travel times and deviations
    """
    print("Loading data...")

    # Load travel time data
    tt_data = gpd.read_file(tt_path)
    print(f"  Travel time data: {len(tt_data):,} hexagons")

    # Load deviation data (read from gpkg and drop geometry to avoid conflicts)
    dev_data = gpd.read_file(dev_path)
    if 'geometry' in dev_data.columns:
        dev_data = pd.DataFrame(dev_data.drop(columns='geometry'))

    print(f"  Deviation data: {len(dev_data):,} hexagons")

    # Merge on h3index (from tt_data) and hex_id (from dev_data)
    merged = tt_data.merge(
        dev_data,
        left_on='h3index',
        right_on='hex_id',
        how='inner'
    )

    print(f"  Merged data: {len(merged):,} hexagons")
    print(f"  Countries: {merged['country'].unique() if 'country' in merged.columns else 'N/A'}")

    return merged

def create_analysis_variables(gdf):
    """
    Create additional analysis variables

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Merged dataset

    Returns:
    --------
    gpd.GeoDataFrame
        Dataset with additional variables
    """
    print("\nCreating analysis variables...")

    # 1. Accessibility index (inverse of average travel time)
    gdf['accessibility_walk'] = 1 / (gdf['tt_walk_avg'] + 1)  # +1 to avoid division by zero
    gdf['accessibility_motor'] = 1 / (gdf['tt_motor_avg'] + 1)

    # 2. Composite accessibility (geometric mean)
    gdf['accessibility_composite'] = np.sqrt(
        gdf['accessibility_walk'] * gdf['accessibility_motor']
    )

    # 3. Modal split indicator (walk/motor ratio)
    gdf['modal_ratio'] = gdf['tt_walk_avg'] / (gdf['tt_motor_avg'] + 1)

    # 4. Deviation ratio (building vs mobility)
    gdf['deviation_ratio'] = gdf['deviation_log_building'] / (gdf['deviation_log_mobility'] + 0.001)

    # 5. Total deviation magnitude
    gdf['deviation_magnitude'] = np.sqrt(
        gdf['deviation_log_building']**2 + gdf['deviation_log_mobility']**2
    )

    # 6. Deviation direction (in radians)
    gdf['deviation_angle'] = np.arctan2(
        gdf['deviation_log_mobility'],
        gdf['deviation_log_building']
    )

    # 7. Category-specific accessibility metrics
    for cat in CATEGORIES:
        gdf[f'access_walk_{cat}'] = 1 / (gdf[f'tt_walk_{cat}'] + 1)
        gdf[f'access_motor_{cat}'] = 1 / (gdf[f'tt_motor_{cat}'] + 1)

    # 8. Accessibility diversity (coefficient of variation across categories)
    walk_access_cols = [f'access_walk_{cat}' for cat in CATEGORIES]
    motor_access_cols = [f'access_motor_{cat}' for cat in CATEGORIES]

    gdf['accessibility_diversity_walk'] = gdf[walk_access_cols].std(axis=1) / gdf[walk_access_cols].mean(axis=1)
    gdf['accessibility_diversity_motor'] = gdf[motor_access_cols].std(axis=1) / gdf[motor_access_cols].mean(axis=1)

    print(f"  Created {len([c for c in gdf.columns if c.startswith('access')])} accessibility variables")

    return gdf

# ============================================================================
# SECTION 2: Statistical Analysis Functions
# ============================================================================

def correlation_analysis(gdf, output_dir):
    """
    Analyze correlations between travel times and mass deviations
    """
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Select key variables
    tt_vars = ['tt_walk_avg', 'tt_motor_avg', 'accessibility_composite', 'modal_ratio']
    dev_vars = ['deviation_log_building', 'deviation_log_mobility',
                'pct_deviation_building', 'pct_deviation_mobility',
                'deviation_ratio', 'deviation_magnitude']

    # Compute correlation matrix
    analysis_df = gdf[tt_vars + dev_vars].dropna()

    if len(analysis_df) == 0:
        print("  WARNING: No valid data for correlation analysis")
        return

    corr_matrix = analysis_df.corr()

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Correlation: Travel Time vs Mass Deviations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '01_correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    # Create simplified correlation heatmap with key variables only
    simple_vars = ['tt_walk_avg', 'tt_motor_avg', 'deviation_log_building',
                   'deviation_log_mobility', 'deviation_magnitude']
    simple_df = gdf[simple_vars].dropna()

    if len(simple_df) > 0:
        simple_corr = simple_df.corr()

        # Simplified labels for cleaner display
        simple_labels = {
            'tt_walk_avg': 'Walking Time',
            'tt_motor_avg': 'Motorized Time',
            'deviation_log_building': 'Building Deviation',
            'deviation_log_mobility': 'Mobility Deviation',
            'deviation_magnitude': 'Deviation Magnitude'
        }
        simple_corr.index = [simple_labels[v] for v in simple_corr.index]
        simple_corr.columns = [simple_labels[v] for v in simple_corr.columns]

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            simple_corr,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            annot_kws={'size': 12},
            cbar_kws={'label': 'Correlation'}
        )
        plt.title('Correlation: Travel Time vs Mass Deviations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / '01b_correlation_heatmap_simple.png', bbox_inches='tight')
        plt.close()

    print("\nKey Correlations:")

    # Extract key correlations
    for tt_var in tt_vars:
        for dev_var in dev_vars:
            if tt_var in corr_matrix.index and dev_var in corr_matrix.columns:
                corr = corr_matrix.loc[tt_var, dev_var]
                if abs(corr) > 0.1:
                    print(f"  {tt_var} <-> {dev_var}: {corr:.3f}")

    # Statistical tests for key relationships
    print("\nStatistical Significance Tests:")

    for tt_var in ['tt_walk_avg', 'accessibility_composite']:
        for dev_var in ['deviation_log_building', 'deviation_log_mobility']:
            if tt_var in analysis_df.columns and dev_var in analysis_df.columns:
                valid_data = analysis_df[[tt_var, dev_var]].dropna()
                if len(valid_data) > 10:
                    r, p = stats.pearsonr(valid_data[tt_var], valid_data[dev_var])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"  {tt_var} vs {dev_var}: r={r:.3f}, p={p:.4f} {sig}")

def scatter_analysis(gdf, output_dir):
    """
    Create scatter plots showing key relationships
    """
    print("\n" + "="*60)
    print("SCATTER PLOT ANALYSIS")
    print("="*60)

    # 1. Building deviation vs accessibility
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: Building deviation vs walking accessibility
    ax = axes[0, 0]
    scatter_data = gdf[['accessibility_walk', 'deviation_log_building', 'quadrant']].dropna()
    for quadrant in scatter_data['quadrant'].unique():
        qdata = scatter_data[scatter_data['quadrant'] == quadrant]
        ax.scatter(qdata['accessibility_walk'], qdata['deviation_log_building'],
                  alpha=0.3, s=10, label=quadrant)
    ax.set_xlabel('Walking Accessibility (1/min)', fontweight='bold')
    ax.set_ylabel('Building Mass Deviation (log)', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=7, loc='best')
    ax.set_title('Building Deviation vs Walking Accessibility', fontweight='bold')

    # Top right: Mobility deviation vs motorized accessibility
    ax = axes[0, 1]
    scatter_data = gdf[['accessibility_motor', 'deviation_log_mobility', 'quadrant']].dropna()
    for quadrant in scatter_data['quadrant'].unique():
        qdata = scatter_data[scatter_data['quadrant'] == quadrant]
        ax.scatter(qdata['accessibility_motor'], qdata['deviation_log_mobility'],
                  alpha=0.3, s=10, label=quadrant)
    ax.set_xlabel('Motorized Accessibility (1/min)', fontweight='bold')
    ax.set_ylabel('Mobility Mass Deviation (log)', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=7, loc='best')
    ax.set_title('Mobility Deviation vs Motorized Accessibility', fontweight='bold')

    # Bottom left: 2D deviation space colored by accessibility
    ax = axes[1, 0]
    scatter_data = gdf[['deviation_log_building', 'deviation_log_mobility', 'accessibility_composite']].dropna()
    scatter = ax.scatter(scatter_data['deviation_log_building'],
                        scatter_data['deviation_log_mobility'],
                        c=scatter_data['accessibility_composite'],
                        cmap='viridis', alpha=0.5, s=10)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Building Deviation (log)', fontweight='bold')
    ax.set_ylabel('Mobility Deviation (log)', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Composite Accessibility')
    ax.set_title('Deviation Space Colored by Accessibility', fontweight='bold')

    # Bottom right: Modal ratio vs deviation ratio
    ax = axes[1, 1]
    scatter_data = gdf[['modal_ratio', 'deviation_ratio', 'population']].dropna()
    scatter = ax.scatter(scatter_data['modal_ratio'],
                        scatter_data['deviation_ratio'],
                        c=np.log10(scatter_data['population']),
                        cmap='plasma', alpha=0.5, s=10)
    ax.set_xlabel('Modal Ratio (walk/motor time)', fontweight='bold')
    ax.set_ylabel('Deviation Ratio (building/mobility)', fontweight='bold')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label='log10(Population)')
    ax.set_title('Modal Balance vs Mass Balance', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '02_scatter_relationships.png', bbox_inches='tight')
    plt.close()

    print("  Saved scatter plot analysis")

def quadrant_analysis(gdf, output_dir):
    """
    Compare accessibility profiles across quadrants
    """
    print("\n" + "="*60)
    print("QUADRANT COMPARISON ANALYSIS")
    print("="*60)

    if 'quadrant' not in gdf.columns:
        print("  WARNING: No quadrant column found")
        return

    # 1. Box plots: Travel time by quadrant
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Walking times
    ax = axes[0, 0]
    plot_data = gdf[['quadrant', 'tt_walk_avg']].dropna()
    plot_data.boxplot(column='tt_walk_avg', by='quadrant', ax=ax)
    ax.set_xlabel('Quadrant', fontweight='bold')
    ax.set_ylabel('Average Walking Time (min)', fontweight='bold')
    ax.set_title('Walking Travel Time by Quadrant', fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right', fontsize=8)

    # Motorized times
    ax = axes[0, 1]
    plot_data = gdf[['quadrant', 'tt_motor_avg']].dropna()
    plot_data.boxplot(column='tt_motor_avg', by='quadrant', ax=ax)
    ax.set_xlabel('Quadrant', fontweight='bold')
    ax.set_ylabel('Average Motorized Time (min)', fontweight='bold')
    ax.set_title('Motorized Travel Time by Quadrant', fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right', fontsize=8)

    # Composite accessibility
    ax = axes[1, 0]
    plot_data = gdf[['quadrant', 'accessibility_composite']].dropna()
    plot_data.boxplot(column='accessibility_composite', by='quadrant', ax=ax)
    ax.set_xlabel('Quadrant', fontweight='bold')
    ax.set_ylabel('Composite Accessibility', fontweight='bold')
    ax.set_title('Composite Accessibility by Quadrant', fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right', fontsize=8)

    # Accessibility diversity
    ax = axes[1, 1]
    plot_data = gdf[['quadrant', 'accessibility_diversity_walk']].dropna()
    plot_data.boxplot(column='accessibility_diversity_walk', by='quadrant', ax=ax)
    ax.set_xlabel('Quadrant', fontweight='bold')
    ax.set_ylabel('Accessibility Diversity (CV)', fontweight='bold')
    ax.set_title('Accessibility Diversity by Quadrant', fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right', fontsize=8)

    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    plt.savefig(output_dir / '03_quadrant_boxplots.png', bbox_inches='tight')
    plt.close()

    # 2. Statistical summary by quadrant
    summary_stats = []

    for quadrant in sorted(gdf['quadrant'].unique()):
        qdata = gdf[gdf['quadrant'] == quadrant]

        stats_dict = {
            'quadrant': quadrant,
            'n': len(qdata),
            'tt_walk_mean': qdata['tt_walk_avg'].mean(),
            'tt_walk_median': qdata['tt_walk_avg'].median(),
            'tt_motor_mean': qdata['tt_motor_avg'].mean(),
            'tt_motor_median': qdata['tt_motor_avg'].median(),
            'accessibility_mean': qdata['accessibility_composite'].mean(),
            'accessibility_std': qdata['accessibility_composite'].std(),
            'modal_ratio_mean': qdata['modal_ratio'].mean(),
            'dev_building_mean': qdata['deviation_log_building'].mean(),
            'dev_mobility_mean': qdata['deviation_log_mobility'].mean()
        }
        summary_stats.append(stats_dict)

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / '03_quadrant_summary_stats.csv', index=False)

    print("\nQuadrant Summary Statistics:")
    print(summary_df.to_string(index=False))

    # 3. ANOVA tests
    print("\nANOVA Tests (differences across quadrants):")

    for var in ['tt_walk_avg', 'tt_motor_avg', 'accessibility_composite', 'modal_ratio']:
        groups = [gdf[gdf['quadrant'] == q][var].dropna() for q in gdf['quadrant'].unique()]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {var}: F={f_stat:.3f}, p={p_val:.4f} {sig}")

def category_analysis(gdf, output_dir):
    """
    Analyze relationships for each POI category separately
    """
    print("\n" + "="*60)
    print("POI CATEGORY-SPECIFIC ANALYSIS")
    print("="*60)

    # Create correlation matrix for each category
    category_correlations = []

    for cat in CATEGORIES:
        walk_col = f'tt_walk_{cat}'
        motor_col = f'tt_motor_{cat}'

        if walk_col not in gdf.columns or motor_col not in gdf.columns:
            continue

        analysis_df = gdf[[walk_col, motor_col, 'deviation_log_building', 'deviation_log_mobility']].dropna()

        if len(analysis_df) < 10:
            continue

        # Correlations
        corr_walk_building = analysis_df[walk_col].corr(analysis_df['deviation_log_building'])
        corr_walk_mobility = analysis_df[walk_col].corr(analysis_df['deviation_log_mobility'])
        corr_motor_building = analysis_df[motor_col].corr(analysis_df['deviation_log_building'])
        corr_motor_mobility = analysis_df[motor_col].corr(analysis_df['deviation_log_mobility'])

        category_correlations.append({
            'category': cat,
            'n': len(analysis_df),
            'walk_building': corr_walk_building,
            'walk_mobility': corr_walk_mobility,
            'motor_building': corr_motor_building,
            'motor_mobility': corr_motor_mobility
        })

    corr_df = pd.DataFrame(category_correlations)
    corr_df.to_csv(output_dir / '04_category_correlations.csv', index=False)

    print("\nCategory-Specific Correlations:")
    print(corr_df.to_string(index=False))

    # Heatmap of category correlations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (mode, dev_type) in enumerate([
        ('walk', 'building'),
        ('walk', 'mobility'),
        ('motor', 'building'),
        ('motor', 'mobility')
    ]):
        ax = axes[idx // 2, idx % 2]
        col_name = f'{mode}_{dev_type}'

        if col_name in corr_df.columns:
            plot_data = corr_df.sort_values(col_name)
            ax.barh(plot_data['category'], plot_data[col_name])
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Correlation', fontweight='bold')
            ax.set_ylabel('POI Category', fontweight='bold')
            ax.set_title(f'{mode.title()} Time vs {dev_type.title()} Deviation', fontweight='bold')
            ax.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(output_dir / '04_category_correlation_bars.png', bbox_inches='tight')
    plt.close()

    print("  Saved category-specific analysis")

def pca_analysis(gdf, output_dir):
    """
    Principal Component Analysis to identify key patterns
    """
    print("\n" + "="*60)
    print("PRINCIPAL COMPONENT ANALYSIS")
    print("="*60)

    # Select variables for PCA
    pca_vars = ['tt_walk_avg', 'tt_motor_avg', 'modal_ratio',
                'deviation_log_building', 'deviation_log_mobility',
                'deviation_magnitude', 'accessibility_composite']

    # Prepare data
    pca_data = gdf[pca_vars].dropna()

    if len(pca_data) < 100:
        print("  WARNING: Insufficient data for PCA")
        return

    print(f"  Analyzing {len(pca_data):,} neighborhoods")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_data)

    # Fit PCA
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    print(f"\nExplained Variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    print(f"  Cumulative (4 PCs): {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(4)],
        index=pca_vars
    )
    loadings.to_csv(output_dir / '05_pca_loadings.csv')

    print("\nPrincipal Component Loadings:")
    print(loadings.to_string())

    # Visualize loadings
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i in range(4):
        ax = axes[i // 2, i % 2]
        pc_data = loadings[f'PC{i+1}'].sort_values()
        ax.barh(pc_data.index, pc_data.values)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Loading', fontweight='bold')
        ax.set_ylabel('Variable', fontweight='bold')
        ax.set_title(f'PC{i+1} Loadings ({pca.explained_variance_ratio_[i]*100:.1f}% var)',
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '05_pca_loadings.png', bbox_inches='tight')
    plt.close()

    # Biplot for PC1 vs PC2
    fig, ax = plt.subplots(figsize=(10, 8))

    # Add quadrant colors to PCA space
    quadrant_colors = {
        'Q1: High Building, High Mobility': 'red',
        'Q2: High Building, Low Mobility': 'blue',
        'Q3: Low Building, High Mobility': 'green',
        'Q4: Low Building, Low Mobility': 'purple'
    }

    pca_df = pca_data.copy()
    pca_df['PC1'] = X_pca[:, 0]
    pca_df['PC2'] = X_pca[:, 1]
    pca_df['quadrant'] = gdf.loc[pca_data.index, 'quadrant'].values

    for quadrant, color in quadrant_colors.items():
        qdata = pca_df[pca_df['quadrant'] == quadrant]
        ax.scatter(qdata['PC1'], qdata['PC2'], c=color, alpha=0.3, s=5, label=quadrant)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontweight='bold')
    ax.set_title('PCA Biplot: PC1 vs PC2 (colored by quadrant)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '05_pca_biplot.png', bbox_inches='tight')
    plt.close()

    print("  Saved PCA analysis")

def regression_analysis(gdf, output_dir):
    """
    Regression analysis for key relationships
    """
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)

    from scipy.stats import linregress

    # Key relationships to test
    relationships = [
        ('accessibility_composite', 'deviation_log_building', 'Accessibility → Building Deviation'),
        ('accessibility_composite', 'deviation_log_mobility', 'Accessibility → Mobility Deviation'),
        ('tt_walk_avg', 'deviation_log_building', 'Walking Time → Building Deviation'),
        ('tt_motor_avg', 'deviation_log_mobility', 'Motorized Time → Mobility Deviation'),
        ('modal_ratio', 'deviation_ratio', 'Modal Ratio → Deviation Ratio'),
        ('accessibility_diversity_walk', 'deviation_magnitude', 'Access Diversity → Deviation Magnitude')
    ]

    results = []

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, (x_var, y_var, title) in enumerate(relationships):
        ax = axes[idx]

        # Get valid data
        plot_data = gdf[[x_var, y_var]].dropna()

        if len(plot_data) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(title, fontweight='bold')
            continue

        x = plot_data[x_var].values
        y = plot_data[y_var].values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Plot
        ax.scatter(x, y, alpha=0.2, s=5, color='steelblue')

        # Regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')

        # Add statistics
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        stats_text = f'R² = {r_value**2:.3f} {sig}\nn = {len(plot_data):,}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(x_var.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel(y_var.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)

        # Store results
        results.append({
            'predictor': x_var,
            'outcome': y_var,
            'n': len(plot_data),
            'slope': slope,
            'intercept': intercept,
            'r': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        })

    plt.tight_layout()
    plt.savefig(output_dir / '06_regression_analysis.png', bbox_inches='tight')
    plt.close()

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / '06_regression_results.csv', index=False)

    print("\nRegression Results:")
    print(results_df.to_string(index=False))

    print("  Saved regression analysis")

def spatial_patterns_analysis(gdf, output_dir):
    """
    Examine spatial patterns in the relationships
    """
    print("\n" + "="*60)
    print("SPATIAL PATTERNS ANALYSIS")
    print("="*60)

    if 'country' not in gdf.columns:
        print("  WARNING: No country column found")
        return

    # Country-level summary statistics
    country_stats = []

    for country in sorted(gdf['country'].unique()):
        cdata = gdf[gdf['country'] == country]

        if len(cdata) < 10:
            continue

        stats_dict = {
            'country': country,
            'n': len(cdata),
            'tt_walk_mean': cdata['tt_walk_avg'].mean(),
            'tt_motor_mean': cdata['tt_motor_avg'].mean(),
            'accessibility_mean': cdata['accessibility_composite'].mean(),
            'dev_building_mean': cdata['deviation_log_building'].mean(),
            'dev_mobility_mean': cdata['deviation_log_mobility'].mean(),
            'modal_ratio_mean': cdata['modal_ratio'].mean()
        }

        # Correlation within country
        valid = cdata[['accessibility_composite', 'deviation_log_building']].dropna()
        if len(valid) > 10:
            stats_dict['corr_access_building'] = valid['accessibility_composite'].corr(valid['deviation_log_building'])

        valid = cdata[['accessibility_composite', 'deviation_log_mobility']].dropna()
        if len(valid) > 10:
            stats_dict['corr_access_mobility'] = valid['accessibility_composite'].corr(valid['deviation_log_mobility'])

        country_stats.append(stats_dict)

    country_df = pd.DataFrame(country_stats)
    country_df.to_csv(output_dir / '07_country_summary.csv', index=False)

    print("\nCountry-Level Summary:")
    print(country_df.to_string(index=False))

    # Plot country comparisons
    if len(country_df) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Walking time
        ax = axes[0, 0]
        country_df.plot.bar(x='country', y='tt_walk_mean', ax=ax, legend=False)
        ax.set_ylabel('Mean Walking Time (min)', fontweight='bold')
        ax.set_xlabel('Country', fontweight='bold')
        ax.set_title('Average Walking Time by Country', fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

        # Accessibility
        ax = axes[0, 1]
        country_df.plot.bar(x='country', y='accessibility_mean', ax=ax, legend=False)
        ax.set_ylabel('Mean Composite Accessibility', fontweight='bold')
        ax.set_xlabel('Country', fontweight='bold')
        ax.set_title('Average Accessibility by Country', fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

        # Building deviation
        ax = axes[1, 0]
        country_df.plot.bar(x='country', y='dev_building_mean', ax=ax, legend=False)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Mean Building Deviation (log)', fontweight='bold')
        ax.set_xlabel('Country', fontweight='bold')
        ax.set_title('Average Building Deviation by Country', fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

        # Correlation strength
        ax = axes[1, 1]
        if 'corr_access_building' in country_df.columns and 'corr_access_mobility' in country_df.columns:
            x = np.arange(len(country_df))
            width = 0.35
            ax.bar(x - width/2, country_df['corr_access_building'], width, label='Building')
            ax.bar(x + width/2, country_df['corr_access_mobility'], width, label='Mobility')
            ax.set_xticks(x)
            ax.set_xticklabels(country_df['country'], rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax.set_ylabel('Correlation (Accessibility vs Deviation)', fontweight='bold')
            ax.set_xlabel('Country', fontweight='bold')
            ax.set_title('Accessibility-Deviation Correlation by Country', fontweight='bold')
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / '07_country_comparisons.png', bbox_inches='tight')
        plt.close()

    print("  Saved spatial patterns analysis")

def mobility_deviation_travel_time_analysis(gdf, output_dir):
    """
    Focused analysis: Mobility deviation vs travel times with regression lines

    Creates detailed scatter plots showing correlations between mobility mass deviation
    and both walking and motorized travel times.
    """
    print("\n" + "="*60)
    print("MOBILITY DEVIATION vs TRAVEL TIME ANALYSIS")
    print("="*60)

    from scipy.stats import linregress

    # Create 2x2 figure for detailed view
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # -------------------------------------------------------------------------
    # Panel 1: Walking Time predicted by Mobility Deviation (all data)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]

    plot_data = gdf[['tt_walk_avg', 'deviation_log_mobility']].dropna()

    if len(plot_data) >= 10:
        x = plot_data['deviation_log_mobility'].values  # Predictor: deviation
        y = plot_data['tt_walk_avg'].values  # Outcome: travel time

        # Scatter plot with transparency
        ax.scatter(x, y, alpha=0.1, s=5, color='steelblue', edgecolors='none')

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=3,
                label=f'y = {slope:.4f}x + {intercept:.3f}')

        # Add reference line at x=0 (zero deviation)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Statistics text box
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        stats_text = (f'Pearson r = {r_value:.4f} {sig}\n'
                     f'R² = {r_value**2:.4f}\n'
                     f'p-value = {p_value:.2e}\n'
                     f'n = {len(plot_data):,}')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_xlabel('Mobility Mass Deviation (log)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Walking Travel Time (minutes)', fontweight='bold', fontsize=12)
        ax.set_title('Walking Time ~ Mobility Deviation - All Data',
                    fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 2: Motorized Time predicted by Mobility Deviation (all data)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]

    plot_data = gdf[['tt_motor_avg', 'deviation_log_mobility']].dropna()

    if len(plot_data) >= 10:
        x = plot_data['deviation_log_mobility'].values  # Predictor: deviation
        y = plot_data['tt_motor_avg'].values  # Outcome: travel time

        # Scatter plot with transparency
        ax.scatter(x, y, alpha=0.1, s=5, color='darkgreen', edgecolors='none')

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=3,
                label=f'y = {slope:.4f}x + {intercept:.3f}')

        # Add reference line at x=0 (zero deviation)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Statistics text box
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        stats_text = (f'Pearson r = {r_value:.4f} {sig}\n'
                     f'R² = {r_value**2:.4f}\n'
                     f'p-value = {p_value:.2e}\n'
                     f'n = {len(plot_data):,}')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_xlabel('Mobility Mass Deviation (log)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Motorized Travel Time (minutes)', fontweight='bold', fontsize=12)
        ax.set_title('Motorized Time ~ Mobility Deviation - All Data',
                    fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 3: Walking Time ~ Mobility Deviation (by quadrant)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]

    if 'quadrant' in gdf.columns:
        quadrant_colors = {
            'Q1: High Building, High Mobility': 'red',
            'Q2: High Building, Low Mobility': 'blue',
            'Q3: Low Building, High Mobility': 'green',
            'Q4: Low Building, Low Mobility': 'purple'
        }

        for quadrant, color in quadrant_colors.items():
            qdata = gdf[gdf['quadrant'] == quadrant][['tt_walk_avg', 'deviation_log_mobility']].dropna()

            if len(qdata) > 0:
                ax.scatter(qdata['deviation_log_mobility'], qdata['tt_walk_avg'],
                          alpha=0.2, s=5, color=color, label=quadrant, edgecolors='none')

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Mobility Mass Deviation (log)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Walking Travel Time (minutes)', fontweight='bold', fontsize=12)
        ax.set_title('Walking Time ~ Mobility Deviation - By Quadrant',
                    fontweight='bold', fontsize=13)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 4: Motorized Time ~ Mobility Deviation (by quadrant)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]

    if 'quadrant' in gdf.columns:
        for quadrant, color in quadrant_colors.items():
            qdata = gdf[gdf['quadrant'] == quadrant][['tt_motor_avg', 'deviation_log_mobility']].dropna()

            if len(qdata) > 0:
                ax.scatter(qdata['deviation_log_mobility'], qdata['tt_motor_avg'],
                          alpha=0.2, s=5, color=color, label=quadrant, edgecolors='none')

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Mobility Mass Deviation (log)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Motorized Travel Time (minutes)', fontweight='bold', fontsize=12)
        ax.set_title('Motorized Time ~ Mobility Deviation - By Quadrant',
                    fontweight='bold', fontsize=13)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '08_mobility_deviation_travel_time.png', bbox_inches='tight')
    plt.close()

    # Print regression statistics
    print("\nKey Regression Statistics:")
    print("\n1. Walking Time ~ Mobility Deviation:")
    plot_data = gdf[['tt_walk_avg', 'deviation_log_mobility']].dropna()
    if len(plot_data) >= 10:
        slope, intercept, r_value, p_value, std_err = linregress(
            plot_data['deviation_log_mobility'], plot_data['tt_walk_avg']
        )
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"   Slope: {slope:.6f}")
        print(f"   Intercept: {intercept:.6f}")
        print(f"   Pearson r: {r_value:.4f}")
        print(f"   R²: {r_value**2:.4f}")
        print(f"   p-value: {p_value:.2e} {sig}")
        print(f"   n: {len(plot_data):,}")

    print("\n2. Motorized Time ~ Mobility Deviation:")
    plot_data = gdf[['tt_motor_avg', 'deviation_log_mobility']].dropna()
    if len(plot_data) >= 10:
        slope, intercept, r_value, p_value, std_err = linregress(
            plot_data['deviation_log_mobility'], plot_data['tt_motor_avg']
        )
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"   Slope: {slope:.6f}")
        print(f"   Intercept: {intercept:.6f}")
        print(f"   Pearson r: {r_value:.4f}")
        print(f"   R²: {r_value**2:.4f}")
        print(f"   p-value: {p_value:.2e} {sig}")
        print(f"   n: {len(plot_data):,}")

    # Save regression statistics to CSV
    regression_stats = []

    for y_var, y_label in [('tt_walk_avg', 'Walking Time'), ('tt_motor_avg', 'Motorized Time')]:
        plot_data = gdf[[y_var, 'deviation_log_mobility']].dropna()
        if len(plot_data) >= 10:
            slope, intercept, r_value, p_value, std_err = linregress(
                plot_data['deviation_log_mobility'], plot_data[y_var]
            )
            regression_stats.append({
                'predictor': 'Mobility Deviation (log)',
                'outcome': y_label,
                'n': len(plot_data),
                'slope': slope,
                'intercept': intercept,
                'r': r_value,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err
            })

    if regression_stats:
        stats_df = pd.DataFrame(regression_stats)
        stats_df.to_csv(output_dir / '08_mobility_deviation_regression_stats.csv', index=False)
        print("\n  Saved regression statistics to CSV")

    print("  Saved mobility deviation scatter plots")

# ============================================================================
# SECTION 3: Main Execution
# ============================================================================

def main():
    print("="*60)
    print("EXPLORATORY ANALYSIS: Travel Time vs Mass Scaling Deviations")
    print("="*60)

    # Set up paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    output_dir = results_dir / 'poi_deviation_exploration'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # File paths
    tt_path = results_dir / 'neighborhood_POIs_travel_time.gpkg'
    dev_path = results_dir / 'neighborhood_deviations_building_v_mobility.gpkg'

    # Check files exist
    if not tt_path.exists():
        print(f"\nERROR: Travel time file not found: {tt_path}")
        return

    if not dev_path.exists():
        print(f"\nERROR: Deviation file not found: {dev_path}")
        return

    # Load and merge data
    gdf = load_and_merge_data(tt_path, dev_path)

    # Create analysis variables
    gdf = create_analysis_variables(gdf)

    # Save merged dataset for reference
    print(f"\nSaving merged dataset...")
    gdf.to_file(output_dir / '00_merged_data.gpkg', driver='GPKG')
    print(f"  Saved: {output_dir / '00_merged_data.gpkg'}")

    # Run analyses
    correlation_analysis(gdf, output_dir)
    scatter_analysis(gdf, output_dir)
    quadrant_analysis(gdf, output_dir)
    category_analysis(gdf, output_dir)
    pca_analysis(gdf, output_dir)
    regression_analysis(gdf, output_dir)
    spatial_patterns_analysis(gdf, output_dir)
    mobility_deviation_travel_time_analysis(gdf, output_dir)

    # Generate summary report
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)

    report_path = output_dir / 'ANALYSIS_SUMMARY.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY ANALYSIS SUMMARY\n")
        f.write("Travel Time to POIs vs Building/Mobility Mass Scaling Deviations\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total neighborhoods analyzed: {len(gdf):,}\n")
        f.write(f"Countries: {', '.join(sorted(gdf['country'].unique()))}\n\n")

        f.write("KEY VARIABLES CREATED:\n")
        f.write("  - accessibility_walk: 1/(tt_walk_avg + 1)\n")
        f.write("  - accessibility_motor: 1/(tt_motor_avg + 1)\n")
        f.write("  - accessibility_composite: geometric mean of walk and motor\n")
        f.write("  - modal_ratio: tt_walk_avg / tt_motor_avg\n")
        f.write("  - deviation_ratio: deviation_log_building / deviation_log_mobility\n")
        f.write("  - deviation_magnitude: sqrt(dev_building² + dev_mobility²)\n")
        f.write("  - accessibility_diversity: CV of accessibility across POI categories\n\n")

        f.write("ANALYSES PERFORMED:\n")
        f.write("  1. Correlation analysis (01_correlation_heatmap.png)\n")
        f.write("  2. Scatter plot relationships (02_scatter_relationships.png)\n")
        f.write("  3. Quadrant comparisons (03_quadrant_boxplots.png)\n")
        f.write("  4. Category-specific analysis (04_category_correlation_bars.png)\n")
        f.write("  5. Principal Component Analysis (05_pca_biplot.png)\n")
        f.write("  6. Regression analysis (06_regression_analysis.png)\n")
        f.write("  7. Spatial patterns by country (07_country_comparisons.png)\n")
        f.write("  8. Mobility deviation vs travel time (08_mobility_deviation_travel_time.png)\n\n")

        f.write("OUTPUT FILES:\n")
        for file in sorted(output_dir.glob('*')):
            if file.name != 'ANALYSIS_SUMMARY.txt':
                f.write(f"  - {file.name}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Analysis completed successfully!\n")
        f.write("="*80 + "\n")

    print(f"\nSummary report saved: {report_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
