#!/usr/bin/env python3
"""
Process NY Metro grids and duration matrix for Kepler.gl visualization
"""

import json
import pandas as pd
import numpy as np

# File paths
geojson_path = "ny_metro_grids.geojson"
matrix_path = "ny_metro_duration_matrix.csv"
population_path = "/Users/kangninghuang/Library/CloudStorage/GoogleDrive-kh3657@nyu.edu/My Drive/Grants_Fellowship/2024 NYU China Grant/results/global_scaling/Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-24.csv"
output_path = "ny_metro_kepler.geojson"

# Load the duration matrix
print("Loading duration matrix...")
df = pd.read_csv(matrix_path, index_col=0)

# Convert seconds to minutes for easier interpretation
df_minutes = df / 60

# Load population data
print("Loading population data...")
pop_df = pd.read_csv(population_path, usecols=['h3index', 'population_2015', 'UC_NM_MN'])

# Filter for New York metro area
pop_df = pop_df[pop_df['UC_NM_MN'] == 'New York']
pop_df = pop_df.groupby('h3index')['population_2015'].sum()  # Aggregate in case of duplicates

# Get population for destination grids (columns in the matrix)
dest_h3_indices = df_minutes.columns.tolist()
pop_weights = pop_df.reindex(dest_h3_indices).fillna(0).values

print(f"  Found population data for {(pop_weights > 0).sum()} of {len(dest_h3_indices)} grids")
print(f"  Total population in study area: {pop_weights.sum():,.0f}")

# Calculate population-weighted average travel time for each origin
def calc_pop_weighted_avg(row, weights):
    """Calculate population-weighted average travel time"""
    total_pop = weights.sum()
    if total_pop == 0:
        return np.nan
    return (row.values * weights).sum() / total_pop

pop_weighted_avg = df_minutes.apply(lambda row: calc_pop_weighted_avg(row, pop_weights), axis=1)

# Calculate accessibility metrics for each origin grid
print("Calculating accessibility metrics...")
metrics = pd.DataFrame({
    'h3index': df_minutes.index,
    'avg_travel_time_min': df_minutes.mean(axis=1),           # Average travel time to all destinations
    'pop_weighted_avg_tt_min': pop_weighted_avg,              # Population-weighted average travel time
    'min_travel_time_min': df_minutes.min(axis=1),            # Minimum travel time (excluding self)
    'max_travel_time_min': df_minutes.max(axis=1),            # Maximum travel time
    'median_travel_time_min': df_minutes.median(axis=1),      # Median travel time
    'std_travel_time_min': df_minutes.std(axis=1),            # Standard deviation
    'pct_under_30min': (df_minutes < 30).sum(axis=1) / len(df_minutes.columns) * 100,  # % destinations reachable in 30 min
    'pct_under_60min': (df_minutes < 60).sum(axis=1) / len(df_minutes.columns) * 100,  # % destinations reachable in 60 min
})

# Load the GeoJSON
print("Loading GeoJSON...")
with open(geojson_path, 'r') as f:
    geojson = json.load(f)

# Create a lookup dictionary for metrics
metrics_dict = metrics.set_index('h3index').to_dict('index')

# Update GeoJSON features with travel time metrics
print("Merging metrics with geometry...")
for feature in geojson['features']:
    h3index = feature['properties'].get('h3index')
    if h3index and h3index in metrics_dict:
        # Add all metrics to properties
        for key, value in metrics_dict[h3index].items():
            feature['properties'][key] = round(value, 2) if not pd.isna(value) else None

# Save the updated GeoJSON
print(f"Saving to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(geojson, f)

print(f"\nDone! Output saved to: {output_path}")
print("\nMetrics added to each grid:")
print("  - avg_travel_time_min: Average travel time to all other grids (minutes)")
print("  - pop_weighted_avg_tt_min: Population-weighted average travel time (minutes)")
print("  - min_travel_time_min: Minimum travel time to nearest grid (minutes)")
print("  - max_travel_time_min: Maximum travel time to farthest grid (minutes)")
print("  - median_travel_time_min: Median travel time (minutes)")
print("  - std_travel_time_min: Standard deviation of travel times")
print("  - pct_under_30min: % of grids reachable within 30 minutes")
print("  - pct_under_60min: % of grids reachable within 60 minutes")
print("\nYou can now upload ny_metro_kepler.geojson to https://kepler.gl/demo")
