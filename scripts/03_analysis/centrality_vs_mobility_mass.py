#!/usr/bin/env python3
"""
Hexagon-level relationship between centrality and actual mobility mass.

Analyzes how population-weighted centrality metrics correlate with
actual material stocks in mobility infrastructure (roads + pavement).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
CENTRALITY_DIR = BASE_DIR / 'data' / 'osrm_pilot_results' / 'centrality_results'
MASS_DATA_PATH = Path("/Users/kangninghuang/Library/CloudStorage/GoogleDrive-kh3657@nyu.edu/My Drive/Grants_Fellowship/2024 NYU China Grant/0.CleanProject_GlobalScaling/data/processed/h3_resolution6/Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv")
OUTPUT_DIR = BASE_DIR / 'data' / 'osrm_pilot_results' / 'centrality_vs_mobility_mass'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("HEXAGON-LEVEL: CENTRALITY vs MOBILITY MASS ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load centrality data
# ============================================================================

print("STEP 1: Loading centrality data...")

centrality_df = pd.read_csv(CENTRALITY_DIR / 'centrality_all_cities.csv')
print(f"Loaded {len(centrality_df)} hexagons with centrality metrics")
print(f"Cities: {centrality_df['city_id'].nunique()}")
print(f"Columns: {list(centrality_df.columns)}")
print()

# ============================================================================
# STEP 2: Load mobility mass data
# ============================================================================

print("STEP 2: Loading mobility mass data...")

mass_df = pd.read_csv(MASS_DATA_PATH)
print(f"Loaded {len(mass_df)} hexagons with mass data")
print(f"Columns: {list(mass_df.columns)}")
print()

# Check key columns
print("Mass data columns of interest:")
mass_cols = [
    'h3index', 'BuildingMass_AverageTotal', 'RoadMass_Average',
    'OtherPavMass_Average', 'mobility_mass_tons', 'total_built_mass_tons',
    'population_2015', 'ID_HDC_G0'
]
for col in mass_cols:
    if col in mass_df.columns:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} MISSING")
print()

# ============================================================================
# STEP 3: Join datasets on H3 index
# ============================================================================

print("STEP 3: Joining centrality with mobility mass...")

# Rename h3index to h3_index for consistency
mass_df = mass_df.rename(columns={'h3index': 'h3_index'})

# Join on h3_index
merged_df = centrality_df.merge(
    mass_df[['h3_index', 'BuildingMass_AverageTotal', 'RoadMass_Average',
             'OtherPavMass_Average', 'mobility_mass_tons', 'total_built_mass_tons',
             'population_2015', 'ID_HDC_G0', 'CTR_MN_NM']],
    on='h3_index',
    how='inner'
)

print(f"Successfully joined {len(merged_df)} hexagons")
print(f"Cities matched: {merged_df['city_id'].nunique()}")
print(f"Match rate: {len(merged_df)/len(centrality_df)*100:.1f}%")
print()

# Check which cities matched
matched_cities = set(merged_df['city_id'].unique())
all_cities = set(centrality_df['city_id'].unique())
unmatched = all_cities - matched_cities

print(f"Matched cities ({len(matched_cities)}): {sorted(matched_cities)[:10]}...")
if unmatched:
    print(f"Unmatched cities ({len(unmatched)}): {sorted(unmatched)}")
print()

# ============================================================================
# STEP 4: Clean and prepare data
# ============================================================================

print("STEP 4: Cleaning and preparing data...")

# Remove rows with missing or zero mass values
initial_count = len(merged_df)

merged_df = merged_df[
    (merged_df['mobility_mass_tons'] > 0) &
    (merged_df['BuildingMass_AverageTotal'] > 0) &
    (merged_df['population_2015'] > 0)
].copy()

print(f"Removed {initial_count - len(merged_df)} hexagons with zero/missing values")
print(f"Final dataset: {len(merged_df)} hexagons from {merged_df['city_id'].nunique()} cities")
print()

# Calculate derived metrics
merged_df['mobility_mass_per_capita'] = merged_df['mobility_mass_tons'] / merged_df['population_2015']
merged_df['building_mass_per_capita'] = merged_df['BuildingMass_AverageTotal'] / merged_df['population_2015']
merged_df['ratio_building_to_mobility'] = merged_df['BuildingMass_AverageTotal'] / merged_df['mobility_mass_tons']
merged_df['pct_mobility_of_total'] = (merged_df['mobility_mass_tons'] / merged_df['total_built_mass_tons']) * 100

# Log transforms for skewed distributions
merged_df['log_mobility_mass'] = np.log10(merged_df['mobility_mass_tons'])
merged_df['log_building_mass'] = np.log10(merged_df['BuildingMass_AverageTotal'])
merged_df['log_population'] = np.log10(merged_df['population_2015'])

print("Derived metrics calculated:")
print("  - mobility_mass_per_capita")
print("  - building_mass_per_capita")
print("  - ratio_building_to_mobility")
print("  - pct_mobility_of_total")
print("  - log transforms of mass and population")
print()

# ============================================================================
# STEP 5: Calculate correlations
# ============================================================================

print("=" * 80)
print("HEXAGON-LEVEL CORRELATIONS")
print("=" * 80)
print()

# Define metric groups
centrality_vars = [
    'closeness_centrality',
    'accessibility_30min',
    'betweenness_centrality',
    'straightness'
]

mass_vars = [
    'mobility_mass_tons',
    'RoadMass_Average',
    'OtherPavMass_Average',
    'mobility_mass_per_capita',
    'pct_mobility_of_total'
]

# Calculate correlations
correlations = {}

for cent_var in centrality_vars:
    correlations[cent_var] = {}
    for mass_var in mass_vars:
        # Remove inf/nan
        valid_data = merged_df[[cent_var, mass_var]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(valid_data) > 10:
            r, p = stats.pearsonr(valid_data[cent_var], valid_data[mass_var])
            rho, p_spear = stats.spearmanr(valid_data[cent_var], valid_data[mass_var])

            correlations[cent_var][mass_var] = {
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spear,
                'n': len(valid_data)
            }

# Print correlation table
print(f"{'Centrality Metric':<30} {'Mass Metric':<35} {'Pearson r':<10} {'p-value':<12} {'n':<8}")
print("-" * 105)

for cent_var in centrality_vars:
    for mass_var in mass_vars:
        if mass_var in correlations[cent_var]:
            r = correlations[cent_var][mass_var]['pearson_r']
            p = correlations[cent_var][mass_var]['pearson_p']
            n = correlations[cent_var][mass_var]['n']
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"{cent_var:<30} {mass_var:<35} {r:>9.4f} {p:>11.6f}{sig:<1} {n:>7}")

print()

# ============================================================================
# STEP 6: Summary statistics
# ============================================================================

print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

print("Centrality Metrics:")
for var in centrality_vars:
    mean = merged_df[var].mean()
    std = merged_df[var].std()
    print(f"  {var:<30} {mean:>10.4f} ± {std:>10.4f}")
print()

print("Mobility Mass Metrics:")
print(f"  {'mobility_mass_tons':<35} {merged_df['mobility_mass_tons'].mean():>12.1f} ± {merged_df['mobility_mass_tons'].std():>12.1f}")
print(f"  {'RoadMass_Average':<35} {merged_df['RoadMass_Average'].mean():>12.1f} ± {merged_df['RoadMass_Average'].std():>12.1f}")
print(f"  {'OtherPavMass_Average':<35} {merged_df['OtherPavMass_Average'].mean():>12.1f} ± {merged_df['OtherPavMass_Average'].std():>12.1f}")
print(f"  {'mobility_mass_per_capita':<35} {merged_df['mobility_mass_per_capita'].mean():>12.1f} ± {merged_df['mobility_mass_per_capita'].std():>12.1f}")
print(f"  {'pct_mobility_of_total':<35} {merged_df['pct_mobility_of_total'].mean():>12.1f}% ± {merged_df['pct_mobility_of_total'].std():>12.1f}%")
print()

print("Building Mass (for comparison):")
print(f"  {'BuildingMass_AverageTotal':<35} {merged_df['BuildingMass_AverageTotal'].mean():>12.1f} ± {merged_df['BuildingMass_AverageTotal'].std():>12.1f}")
print(f"  {'building_mass_per_capita':<35} {merged_df['building_mass_per_capita'].mean():>12.1f} ± {merged_df['building_mass_per_capita'].std():>12.1f}")
print(f"  {'ratio_building_to_mobility':<35} {merged_df['ratio_building_to_mobility'].mean():>12.1f} ± {merged_df['ratio_building_to_mobility'].std():>12.1f}")
print()

# ============================================================================
# STEP 7: City-level aggregation
# ============================================================================

print("=" * 80)
print("CITY-LEVEL AGGREGATES")
print("=" * 80)
print()

city_summary = merged_df.groupby('city_id').agg({
    'closeness_centrality': 'mean',
    'accessibility_30min': 'mean',
    'mobility_mass_tons': 'sum',
    'mobility_mass_per_capita': 'mean',
    'BuildingMass_AverageTotal': 'sum',
    'population_2015': 'sum',
    'h3_index': 'count'
}).rename(columns={'h3_index': 'n_hexagons'})

city_summary['total_mobility_mass_MT'] = city_summary['mobility_mass_tons'] / 1e6  # Megatonnes
city_summary['total_building_mass_MT'] = city_summary['BuildingMass_AverageTotal'] / 1e6

print("Top 10 cities by total mobility mass:")
top_cities = city_summary.nlargest(10, 'total_mobility_mass_MT')
for i, (city_id, row) in enumerate(top_cities.iterrows(), 1):
    print(f"  {i}. City {city_id}: {row['total_mobility_mass_MT']:.2f} MT " +
          f"({row['n_hexagons']} hexagons, " +
          f"closeness={row['closeness_centrality']:.4f})")
print()

# ============================================================================
# STEP 8: Create visualizations
# ============================================================================

print("STEP 8: Creating visualizations...")

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Figure 1: Key relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Centrality vs Actual Mobility Mass (Hexagon-Level)', fontsize=16, y=0.995)

# Plot 1: Closeness vs Mobility Mass (log scale)
ax = axes[0, 0]
valid = merged_df[['closeness_centrality', 'log_mobility_mass']].dropna()
ax.scatter(valid['log_mobility_mass'], valid['closeness_centrality'],
           alpha=0.3, s=10, c='steelblue')
r = correlations['closeness_centrality']['mobility_mass_tons']['pearson_r']
ax.set_xlabel('log₁₀(Mobility Mass) [tons]')
ax.set_ylabel('Closeness Centrality')
ax.set_title(f'Closeness vs Mobility Mass (r={r:.3f})')
ax.grid(True, alpha=0.3)

# Plot 2: Accessibility vs Mobility Mass per Capita
ax = axes[0, 1]
valid = merged_df[['accessibility_30min', 'mobility_mass_per_capita']].dropna()
# Remove outliers for visualization
valid = valid[valid['mobility_mass_per_capita'] < valid['mobility_mass_per_capita'].quantile(0.99)]
ax.scatter(valid['mobility_mass_per_capita'], valid['accessibility_30min'],
           alpha=0.3, s=10, c='forestgreen')
r = correlations['accessibility_30min']['mobility_mass_per_capita']['pearson_r']
ax.set_xlabel('Mobility Mass per Capita [tons/person]')
ax.set_ylabel('Accessibility (30 min)')
ax.set_title(f'Accessibility vs Mobility Mass per Capita (r={r:.3f})')
ax.grid(True, alpha=0.3)

# Plot 3: Betweenness vs Road Mass
ax = axes[1, 0]
valid = merged_df[['betweenness_centrality', 'RoadMass_Average']].dropna()
valid = valid[valid['RoadMass_Average'] < valid['RoadMass_Average'].quantile(0.99)]
ax.scatter(valid['RoadMass_Average'], valid['betweenness_centrality'],
           alpha=0.3, s=10, c='coral')
r = correlations['betweenness_centrality']['RoadMass_Average']['pearson_r']
ax.set_xlabel('Road Mass [tons]')
ax.set_ylabel('Betweenness Centrality')
ax.set_title(f'Betweenness vs Road Mass (r={r:.3f})')
ax.grid(True, alpha=0.3)

# Plot 4: Straightness vs % Mobility of Total Mass
ax = axes[1, 1]
valid = merged_df[['straightness', 'pct_mobility_of_total']].dropna()
ax.scatter(valid['pct_mobility_of_total'], valid['straightness'],
           alpha=0.3, s=10, c='mediumpurple')
r = correlations['straightness']['pct_mobility_of_total']['pearson_r']
ax.set_xlabel('Mobility as % of Total Built Mass')
ax.set_ylabel('Route Straightness')
ax.set_title(f'Straightness vs % Mobility (r={r:.3f})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'centrality_vs_mobility_mass_hexagon.png', dpi=300, bbox_inches='tight')
print(f"  Saved: centrality_vs_mobility_mass_hexagon.png")

# Figure 2: Correlation heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

corr_matrix = pd.DataFrame({
    cent: {mass: correlations[cent][mass]['pearson_r']
           for mass in mass_vars if mass in correlations[cent]}
    for cent in centrality_vars
}).T

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
ax.set_title('Hexagon-Level: Centrality vs Mobility Mass Correlations', fontsize=14, pad=20)
ax.set_xlabel('Mobility Mass Metrics', fontsize=12)
ax.set_ylabel('Centrality Metrics', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap_mobility_mass.png', dpi=300, bbox_inches='tight')
print(f"  Saved: correlation_heatmap_mobility_mass.png")

plt.close('all')

# ============================================================================
# STEP 9: Save results
# ============================================================================

print()
print("STEP 9: Saving results...")

# Save merged hexagon-level data
merged_df.to_csv(OUTPUT_DIR / 'centrality_mobility_mass_hexagon_level.csv', index=False)
print(f"  Saved: centrality_mobility_mass_hexagon_level.csv ({len(merged_df)} hexagons)")

# Save city-level summary
city_summary.to_csv(OUTPUT_DIR / 'centrality_mobility_mass_city_level.csv')
print(f"  Saved: centrality_mobility_mass_city_level.csv ({len(city_summary)} cities)")

# Save correlation matrix
corr_matrix.to_csv(OUTPUT_DIR / 'correlation_matrix_mobility_mass.csv')
print(f"  Saved: correlation_matrix_mobility_mass.csv")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Results saved to: {OUTPUT_DIR}")
print()
print("Key findings:")
print(f"  - Analyzed {len(merged_df)} hexagons from {merged_df['city_id'].nunique()} cities")
print(f"  - Matched {len(merged_df)/len(centrality_df)*100:.1f}% of centrality data with mass data")
print()
print("Strongest correlations:")
for cent_var in centrality_vars:
    max_corr = max([(mass, correlations[cent_var][mass]['pearson_r'])
                    for mass in mass_vars if mass in correlations[cent_var]],
                   key=lambda x: abs(x[1]))
    print(f"  {cent_var}: {max_corr[0]} (r={max_corr[1]:.3f})")
