#!/usr/bin/env python3
"""
Calculate grid centrality based on route intersections.

Basic centrality: Count how many routes pass through each grid
Population-weighted centrality: Weight routes by origin and destination populations
"""

import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import shape, LineString
from collections import defaultdict
import numpy as np

print("=" * 60)
print("Grid Centrality Analysis")
print("=" * 60)

# File paths
routes_path = "test_results/945_routes_simplified.geojson"
grids_path = "ny_metro_grids.geojson"
population_path = "Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-22.csv"
output_path = "ny_metro_grids_with_centrality.geojson"

# Step 1: Load grids
print("\n1. Loading grids...")
grids_gdf = gpd.read_file(grids_path)
print(f"   Loaded {len(grids_gdf):,} grids")

# Create h3index to geometry mapping
grid_geometries = {}
for idx, row in grids_gdf.iterrows():
    h3idx = row['h3index']
    grid_geometries[h3idx] = row.geometry

# Step 2: Load population data
print("\n2. Loading population data...")
pop_df = pd.read_csv(population_path, usecols=['h3index', 'population_2015', 'UC_NM_MN'])

# Filter for New York metro area
pop_df = pop_df[pop_df['UC_NM_MN'] == 'New York']
pop_dict = pop_df.set_index('h3index')['population_2015'].to_dict()
print(f"   Loaded population for {len(pop_dict):,} grids")
print(f"   Total population: {sum(pop_dict.values()):,.0f}")

# Step 3: Load routes (in chunks due to large file size)
print("\n3. Loading routes (this may take a while)...")
print("   Opening large GeoJSON file...")

# Read the routes GeoJSON
with open(routes_path, 'r') as f:
    routes_data = json.load(f)

print(f"   Loaded {len(routes_data['features']):,} routes")

# Step 4: Calculate centrality
print("\n4. Calculating centrality metrics...")
print("   This will take several minutes for large route sets...")

# Initialize centrality counters
basic_centrality = defaultdict(int)
weighted_centrality = defaultdict(float)

# Track progress
total_routes = len(routes_data['features'])
processed = 0
update_interval = max(1000, total_routes // 20)  # Update every 5%

for feature in routes_data['features']:
    # Get route properties
    origin_h3 = feature['properties']['origin_h3']
    dest_h3 = feature['properties']['destination_h3']

    # Get route geometry
    route_geom = shape(feature['geometry'])

    # Get population weights (use 1.0 if not found)
    origin_pop = pop_dict.get(origin_h3, 0)
    dest_pop = pop_dict.get(dest_h3, 0)

    # Calculate route weight (product of populations, normalized)
    # Use geometric mean to avoid extreme values
    if origin_pop > 0 and dest_pop > 0:
        route_weight = np.sqrt(origin_pop * dest_pop)
    else:
        route_weight = 0

    # Find all grids that this route intersects
    grids_intersected = set()

    for h3idx, grid_geom in grid_geometries.items():
        if route_geom.intersects(grid_geom):
            grids_intersected.add(h3idx)

    # Update centrality for intersected grids
    for h3idx in grids_intersected:
        basic_centrality[h3idx] += 1
        weighted_centrality[h3idx] += route_weight

    # Progress update
    processed += 1
    if processed % update_interval == 0:
        print(f"   Processed {processed:,} / {total_routes:,} routes ({100*processed/total_routes:.1f}%)")

print(f"   Processed all {total_routes:,} routes")

# Step 5: Add centrality metrics to grids
print("\n5. Adding centrality metrics to grids...")

# Convert to DataFrames for easier merging
centrality_df = pd.DataFrame({
    'h3index': list(basic_centrality.keys()),
    'basic_centrality': [basic_centrality[h3] for h3 in basic_centrality.keys()],
    'weighted_centrality': [weighted_centrality[h3] for h3 in basic_centrality.keys()]
})

# Normalize weighted centrality (0-1 scale)
max_weighted = centrality_df['weighted_centrality'].max()
if max_weighted > 0:
    centrality_df['weighted_centrality_normalized'] = (
        centrality_df['weighted_centrality'] / max_weighted
    )
else:
    centrality_df['weighted_centrality_normalized'] = 0

# Merge with grids
grids_gdf = grids_gdf.merge(centrality_df, on='h3index', how='left')

# Fill NaN values with 0 (grids with no routes passing through)
grids_gdf['basic_centrality'] = grids_gdf['basic_centrality'].fillna(0).astype(int)
grids_gdf['weighted_centrality'] = grids_gdf['weighted_centrality'].fillna(0)
grids_gdf['weighted_centrality_normalized'] = grids_gdf['weighted_centrality_normalized'].fillna(0)

# Add population data
grids_gdf['population_2015'] = grids_gdf['h3index'].map(pop_dict).fillna(0)

# Step 6: Calculate statistics
print("\n6. Centrality Statistics:")
print(f"   Grids with routes: {(grids_gdf['basic_centrality'] > 0).sum():,} / {len(grids_gdf):,}")
print(f"   Max basic centrality: {grids_gdf['basic_centrality'].max():,} routes")
print(f"   Mean basic centrality: {grids_gdf['basic_centrality'].mean():.1f} routes")
print(f"   Median basic centrality: {grids_gdf['basic_centrality'].median():.1f} routes")

# Top 10 most central grids
print("\n7. Top 10 Most Central Grids (by basic centrality):")
top_grids = grids_gdf.nlargest(10, 'basic_centrality')[['h3index', 'basic_centrality', 'weighted_centrality', 'population_2015']]
for idx, row in top_grids.iterrows():
    print(f"   {row['h3index']}: {row['basic_centrality']:,} routes, "
          f"weighted={row['weighted_centrality']:,.0f}, pop={row['population_2015']:,.0f}")

# Step 7: Save results
print(f"\n8. Saving results to {output_path}...")
grids_gdf.to_file(output_path, driver='GeoJSON')

print("\nDone!")
print(f"\nOutput file: {output_path}")
print("\nMetrics added to each grid:")
print("  - basic_centrality: Number of routes passing through this grid")
print("  - weighted_centrality: Population-weighted centrality (sum of route weights)")
print("  - weighted_centrality_normalized: Normalized weighted centrality (0-1 scale)")
print("  - population_2015: Population in this grid")
print("\nYou can now visualize this in QGIS or Kepler.gl:")
print("  - Color by basic_centrality to see most central grids")
print("  - Color by weighted_centrality to see population-weighted centrality")
print("=" * 60)
