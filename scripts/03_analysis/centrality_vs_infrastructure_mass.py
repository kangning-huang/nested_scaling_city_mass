#!/usr/bin/env python3
"""
Investigate relationship between population-weighted centrality and
mobility infrastructure mass at H3 resolutions 6 and 7.

Analyzes correlation, spatial patterns, and scaling relationships.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
CENTRALITY_DIR = BASE_DIR / 'data' / 'osrm_pilot_results' / 'centrality_results'
OUTPUT_DIR = BASE_DIR / 'data' / 'osrm_pilot_results' / 'centrality_vs_infrastructure'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("CENTRALITY vs INFRASTRUCTURE MASS ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load centrality data (Resolution 6)
# ============================================================================

print("STEP 1: Loading centrality data (H3 resolution 6)...")

centrality_df = pd.read_csv(CENTRALITY_DIR / 'centrality_all_cities.csv')
print(f"Loaded {len(centrality_df)} grid cells from {centrality_df['city_id'].nunique()} cities")
print()

# ============================================================================
# STEP 2: Extract mobility infrastructure mass from routing matrices
# ============================================================================

print("STEP 2: Calculating infrastructure proxy from routing data...")
print()

# We'll use routing network characteristics as a proxy for infrastructure:
# 1. Network density: total road length per hexagon
# 2. Network complexity: number of unique routes passing through
# 3. Connectivity: average route directness (straightness)

# Load original routing matrices to calculate network metrics
MATRIX_DIR = BASE_DIR / 'data' / 'osrm_pilot_results'
matrix_files = sorted(MATRIX_DIR.glob('*_matrix.json'))

infrastructure_metrics = []

for matrix_file in matrix_files:
    city_id = matrix_file.stem.split('_')[0]

    # Skip if not in our centrality data
    if city_id not in centrality_df['city_id'].values:
        continue

    try:
        with open(matrix_file) as f:
            data = json.load(f)

        if 'h3_indices' not in data or 'distances' not in data:
            continue

        h3_indices = data['h3_indices']
        distances = np.array(data['distances'])  # meters
        durations = np.array(data['durations'])  # seconds

        n_grids = len(h3_indices)

        for i in range(n_grids):
            # Calculate infrastructure metrics for this hexagon

            # 1. Total network length accessible (sum of all route distances)
            total_distance_km = distances[i, :].sum() / 1000.0

            # 2. Average route length (excluding self)
            valid_dist = distances[i, distances[i, :] > 0]
            avg_route_length = valid_dist.mean() if len(valid_dist) > 0 else 0

            # 3. Network efficiency (average speed = distance/time)
            valid_times = durations[i, durations[i, :] > 0]
            valid_dists = distances[i, durations[i, :] > 0]
            avg_speed_kmh = (valid_dists.mean() / 1000.0) / (valid_times.mean() / 3600.0) if len(valid_times) > 0 else 0

            # 4. Connectivity index (number of destinations reachable)
            n_connections = (distances[i, :] > 0).sum() - 1  # exclude self

            # 5. Road density proxy: total distance / (area of hexagon)
            # H3 resolution 6 hexagon area ≈ 36.13 km²
            hexagon_area_km2 = 36.13
            road_density = total_distance_km / hexagon_area_km2

            infrastructure_metrics.append({
                'city_id': city_id,
                'h3_index': h3_indices[i],
                'total_network_distance_km': total_distance_km,
                'avg_route_length_km': avg_route_length / 1000.0,
                'avg_speed_kmh': avg_speed_kmh,
                'n_connections': n_connections,
                'road_density_km_per_km2': road_density
            })

    except Exception as e:
        print(f"  Warning: Could not process {city_id}: {e}")
        continue

infra_df = pd.DataFrame(infrastructure_metrics)
print(f"Calculated infrastructure metrics for {len(infra_df)} grid cells")
print()

# ============================================================================
# STEP 3: Join centrality with infrastructure metrics
# ============================================================================

print("STEP 3: Joining centrality with infrastructure data...")

# Merge on city_id and h3_index
merged_df = centrality_df.merge(
    infra_df,
    on=['city_id', 'h3_index'],
    how='inner'
)

print(f"Successfully joined {len(merged_df)} grid cells")
print(f"Cities in merged dataset: {merged_df['city_id'].nunique()}")
print()

# ============================================================================
# STEP 4: Calculate correlations (Resolution 6)
# ============================================================================

print("=" * 80)
print("RESOLUTION 6 ANALYSIS")
print("=" * 80)
print()

# Select metrics for correlation
centrality_vars = [
    'closeness_centrality',
    'accessibility_30min',
    'betweenness_centrality',
    'straightness'
]

infrastructure_vars = [
    'road_density_km_per_km2',
    'total_network_distance_km',
    'avg_route_length_km',
    'avg_speed_kmh',
    'n_connections'
]

# Calculate correlation matrix
print("Correlation Matrix:")
print()

correlations = {}
for cent_var in centrality_vars:
    correlations[cent_var] = {}
    for infra_var in infrastructure_vars:
        # Remove NaN/inf values
        valid_data = merged_df[[cent_var, infra_var]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(valid_data) > 10:
            corr, pval = stats.pearsonr(valid_data[cent_var], valid_data[infra_var])
            correlations[cent_var][infra_var] = {
                'correlation': corr,
                'p_value': pval,
                'n': len(valid_data)
            }

# Print correlation table
print(f"{'Centrality Metric':<30} {'Infrastructure Metric':<30} {'Corr':<8} {'p-value':<10} {'n':<8}")
print("-" * 96)

for cent_var in centrality_vars:
    for infra_var in infrastructure_vars:
        if infra_var in correlations[cent_var]:
            r = correlations[cent_var][infra_var]['correlation']
            p = correlations[cent_var][infra_var]['p_value']
            n = correlations[cent_var][infra_var]['n']
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"{cent_var:<30} {infra_var:<30} {r:>7.3f} {p:>9.4f}{sig:<1} {n:>7}")

print()

# ============================================================================
# STEP 5: Resolution 7 analysis (disaggregate)
# ============================================================================

print("=" * 80)
print("RESOLUTION 7 ANALYSIS")
print("=" * 80)
print()

print("Converting resolution 6 to resolution 7...")

# For each res 6 hexagon, get all res 7 children
res7_data = []

for idx, row in merged_df.iterrows():
    h3_res6 = row['h3_index']

    # Get all resolution 7 children
    children = h3.cell_to_children(h3_res6, 7)

    # Each child inherits the parent's values (simple disaggregation)
    # More sophisticated: could weight by actual area or population
    for child_h3 in children:
        lat, lon = h3.cell_to_latlng(child_h3)
        res7_data.append({
            **row.to_dict(),
            'h3_res7': child_h3,
            'h3_res6_parent': h3_res6,
            'lat_res7': lat,
            'lon_res7': lon
        })

res7_df = pd.DataFrame(res7_data)
print(f"Generated {len(res7_df)} resolution 7 cells from {len(merged_df)} resolution 6 cells")
print(f"Average children per parent: {len(res7_df) / len(merged_df):.1f}")
print()

# Calculate correlations at resolution 7
print("Correlation Matrix (Resolution 7):")
print()

correlations_res7 = {}
for cent_var in centrality_vars:
    correlations_res7[cent_var] = {}
    for infra_var in infrastructure_vars:
        valid_data = res7_df[[cent_var, infra_var]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(valid_data) > 10:
            corr, pval = stats.pearsonr(valid_data[cent_var], valid_data[infra_var])
            correlations_res7[cent_var][infra_var] = {
                'correlation': corr,
                'p_value': pval,
                'n': len(valid_data)
            }

print(f"{'Centrality Metric':<30} {'Infrastructure Metric':<30} {'Corr':<8} {'p-value':<10} {'n':<8}")
print("-" * 96)

for cent_var in centrality_vars:
    for infra_var in infrastructure_vars:
        if infra_var in correlations_res7[cent_var]:
            r = correlations_res7[cent_var][infra_var]['correlation']
            p = correlations_res7[cent_var][infra_var]['p_value']
            n = correlations_res7[cent_var][infra_var]['n']
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"{cent_var:<30} {infra_var:<30} {r:>7.3f} {p:>9.4f}{sig:<1} {n:>7}")

print()

# ============================================================================
# STEP 6: Create visualizations
# ============================================================================

print("STEP 6: Creating visualizations...")

# Set plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 1. Scatter plots for key relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Centrality vs Infrastructure Mass (Resolution 6)', fontsize=16, y=0.995)

# Plot 1: Closeness vs Road Density
ax = axes[0, 0]
valid = merged_df[['closeness_centrality', 'road_density_km_per_km2']].replace([np.inf, -np.inf], np.nan).dropna()
ax.scatter(valid['road_density_km_per_km2'], valid['closeness_centrality'],
           alpha=0.5, s=20, c='steelblue')
ax.set_xlabel('Road Density (km/km²)')
ax.set_ylabel('Closeness Centrality')
r = correlations['closeness_centrality']['road_density_km_per_km2']['correlation']
ax.set_title(f'Closeness vs Road Density (r={r:.3f})')
ax.grid(True, alpha=0.3)

# Plot 2: Accessibility vs Network Distance
ax = axes[0, 1]
valid = merged_df[['accessibility_30min', 'total_network_distance_km']].replace([np.inf, -np.inf], np.nan).dropna()
ax.scatter(valid['total_network_distance_km'], valid['accessibility_30min'],
           alpha=0.5, s=20, c='forestgreen')
ax.set_xlabel('Total Network Distance (km)')
ax.set_ylabel('Accessibility (30min)')
r = correlations['accessibility_30min']['total_network_distance_km']['correlation']
ax.set_title(f'Accessibility vs Network Distance (r={r:.3f})')
ax.grid(True, alpha=0.3)

# Plot 3: Betweenness vs Connections
ax = axes[1, 0]
valid = merged_df[['betweenness_centrality', 'n_connections']].replace([np.inf, -np.inf], np.nan).dropna()
ax.scatter(valid['n_connections'], valid['betweenness_centrality'],
           alpha=0.5, s=20, c='coral')
ax.set_xlabel('Number of Connections')
ax.set_ylabel('Betweenness Centrality')
r = correlations['betweenness_centrality']['n_connections']['correlation']
ax.set_title(f'Betweenness vs Connections (r={r:.3f})')
ax.grid(True, alpha=0.3)

# Plot 4: Straightness vs Average Speed
ax = axes[1, 1]
valid = merged_df[['straightness', 'avg_speed_kmh']].replace([np.inf, -np.inf], np.nan).dropna()
# Remove outliers for better visualization
valid = valid[(valid['avg_speed_kmh'] > 0) & (valid['avg_speed_kmh'] < 100)]
ax.scatter(valid['avg_speed_kmh'], valid['straightness'],
           alpha=0.5, s=20, c='mediumpurple')
ax.set_xlabel('Average Speed (km/h)')
ax.set_ylabel('Route Straightness')
r = correlations['straightness']['avg_speed_kmh']['correlation']
ax.set_title(f'Straightness vs Speed (r={r:.3f})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'centrality_vs_infrastructure_res6.png', dpi=300, bbox_inches='tight')
print(f"  Saved: centrality_vs_infrastructure_res6.png")

# 2. Correlation heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Resolution 6 heatmap
corr_matrix_res6 = pd.DataFrame({
    cent: {infra: correlations[cent][infra]['correlation']
           for infra in infrastructure_vars if infra in correlations[cent]}
    for cent in centrality_vars
}).T

sns.heatmap(corr_matrix_res6, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Correlation'})
ax1.set_title('Resolution 6 Correlations', fontsize=14)
ax1.set_xlabel('Infrastructure Metrics')
ax1.set_ylabel('Centrality Metrics')

# Resolution 7 heatmap
corr_matrix_res7 = pd.DataFrame({
    cent: {infra: correlations_res7[cent][infra]['correlation']
           for infra in infrastructure_vars if infra in correlations_res7[cent]}
    for cent in centrality_vars
}).T

sns.heatmap(corr_matrix_res7, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Correlation'})
ax2.set_title('Resolution 7 Correlations', fontsize=14)
ax2.set_xlabel('Infrastructure Metrics')
ax2.set_ylabel('Centrality Metrics')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print(f"  Saved: correlation_heatmaps.png")

plt.close('all')

# ============================================================================
# STEP 7: Save results
# ============================================================================

print()
print("STEP 7: Saving results...")

# Save merged data
merged_df.to_csv(OUTPUT_DIR / 'centrality_infrastructure_res6.csv', index=False)
print(f"  Saved: centrality_infrastructure_res6.csv")

res7_df.to_csv(OUTPUT_DIR / 'centrality_infrastructure_res7.csv', index=False)
print(f"  Saved: centrality_infrastructure_res7.csv")

# Save correlation matrices
corr_matrix_res6.to_csv(OUTPUT_DIR / 'correlation_matrix_res6.csv')
corr_matrix_res7.to_csv(OUTPUT_DIR / 'correlation_matrix_res7.csv')
print(f"  Saved: correlation matrices")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Results saved to: {OUTPUT_DIR}")
print()
print("Key findings:")
print(f"  - Analyzed {len(merged_df)} grid cells at resolution 6")
print(f"  - Generated {len(res7_df)} grid cells at resolution 7")
print(f"  - Strongest positive correlations:")
for cent_var in centrality_vars:
    max_corr = max([(infra, correlations[cent_var][infra]['correlation'])
                    for infra in infrastructure_vars if infra in correlations[cent_var]],
                   key=lambda x: x[1])
    print(f"    {cent_var}: {max_corr[0]} (r={max_corr[1]:.3f})")
