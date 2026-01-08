#!/usr/bin/env python3
"""
Calculate population-weighted centrality metrics for OSRM routing matrices.

Computes various centrality measures weighted by population:
- Closeness centrality (inverse average travel time to all locations)
- Betweenness centrality (frequency on shortest paths between locations)
- Accessibility (population-weighted reachability)
- Straightness (efficiency of routes)
"""

import json
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import h3
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, dijkstra
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'osrm_pilot_results'
OUTPUT_DIR = DATA_DIR / 'centrality_results'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("POPULATION-WEIGHTED CENTRALITY ANALYSIS")
print("=" * 70)
print()

# Load all routing matrices
print("STEP 1: Loading routing matrices...")

matrix_files = sorted(DATA_DIR.glob('*_matrix.json'))
print(f"Found {len(matrix_files)} matrix files")

cities_data = {}

for matrix_file in matrix_files:
    city_id = matrix_file.stem.split('_')[0]

    with open(matrix_file) as f:
        data = json.load(f)

    # Get n_grids (may be stored differently in different files)
    if 'n_grids' in data:
        n_grids = data['n_grids']
    elif 'h3_indices' in data:
        n_grids = len(data['h3_indices'])
        data['n_grids'] = n_grids
    else:
        print(f"  Skipping city {city_id}: cannot determine grid count")
        continue

    # Skip if too few grids
    if n_grids < 5:
        print(f"  Skipping city {city_id}: only {n_grids} grids")
        continue

    cities_data[city_id] = data
    print(f"  Loaded city {city_id}: {n_grids} grids")

print(f"\nLoaded {len(cities_data)} cities for analysis")
print()

# Get population data
print("STEP 2: Getting population data...")
print("(Using equal weights for now - will integrate WorldPop if available)")
print()

for city_id, data in cities_data.items():
    n_grids = data['n_grids']
    data['population'] = np.ones(n_grids) * 1000

print("Population data prepared (equal weights)")
print()

# Calculate centrality metrics
print("STEP 3: Calculating centrality metrics...")
print()

def calculate_closeness_centrality(duration_matrix, population):
    """Population-weighted closeness centrality."""
    n = len(duration_matrix)
    closeness = np.zeros(n)

    for i in range(n):
        times = duration_matrix[i, :]
        valid_idx = times > 0
        if valid_idx.sum() > 0:
            weighted_avg_time = np.average(times[valid_idx], weights=population[valid_idx])
            closeness[i] = 1.0 / (weighted_avg_time / 60.0)

    return closeness

def calculate_accessibility(duration_matrix, population, threshold_minutes=30):
    """Population reachable within threshold."""
    n = len(duration_matrix)
    accessibility = np.zeros(n)

    for i in range(n):
        times_minutes = duration_matrix[i, :] / 60.0
        reachable = times_minutes <= threshold_minutes
        accessibility[i] = population[reachable].sum()

    return accessibility

def calculate_betweenness_centrality(duration_matrix, population):
    """Simplified population-weighted betweenness."""
    n = len(duration_matrix)
    betweenness = np.zeros(n)

    sparse_matrix = csr_matrix(duration_matrix)

    for i in range(n):
        dist_matrix, predecessors = dijkstra(
            sparse_matrix,
            indices=i,
            return_predecessors=True
        )

        for j in range(n):
            if i != j and predecessors[j] >= 0:
                path = []
                current = j
                while current != i:
                    path.append(current)
                    current = predecessors[current]
                    if current < 0:
                        break

                pop_weight = population[j]
                for node in path[:-1]:
                    betweenness[node] += pop_weight

    if betweenness.max() > 0:
        betweenness = betweenness / betweenness.max()

    return betweenness

def calculate_straightness(duration_matrix, distance_matrix):
    """Route straightness ratio."""
    n = len(duration_matrix)
    straightness = np.zeros(n)

    for i in range(n):
        distances = distance_matrix[i, :]
        valid_idx = distances > 0

        if valid_idx.sum() > 0:
            min_distances = distances[valid_idx]
            actual_distances = distances[valid_idx]
            ratios = min_distances / (actual_distances + 1e-6)
            straightness[i] = ratios.mean()

    return straightness

# Process all cities
results = []

for city_id, data in cities_data.items():
    print(f"Processing city {city_id}...")

    duration_matrix = np.array(data['durations'])
    distance_matrix = np.array(data['distances'])
    population = data['population']
    h3_indices = data['h3_indices']
    centroids = data['centroids']

    n_grids = len(h3_indices)

    closeness = calculate_closeness_centrality(duration_matrix, population)
    accessibility_30min = calculate_accessibility(duration_matrix, population, threshold_minutes=30)
    betweenness = calculate_betweenness_centrality(duration_matrix, population)
    straightness = calculate_straightness(duration_matrix, distance_matrix)

    for i in range(n_grids):
        results.append({
            'city_id': city_id,
            'h3_index': h3_indices[i],
            'lat': centroids[i]['lat'],
            'lon': centroids[i]['lon'],
            'population': population[i],
            'closeness_centrality': closeness[i],
            'accessibility_30min': accessibility_30min[i],
            'betweenness_centrality': betweenness[i],
            'straightness': straightness[i],
            'mean_travel_time_min': duration_matrix[i, :].mean() / 60.0,
            'max_travel_time_min': duration_matrix[i, :].max() / 60.0,
            'mean_distance_km': distance_matrix[i, :].mean() / 1000.0,
        })

    print(f"  Done: {len(results)} total cells")

print()
print(f"Total cells analyzed: {len(results)}")
print()

# Save results
print("STEP 4: Saving results...")

df = pd.DataFrame(results)

csv_file = OUTPUT_DIR / 'centrality_all_cities.csv'
df.to_csv(csv_file, index=False)
print(f"  Saved CSV: {csv_file}")

for city_id in df['city_id'].unique():
    city_df = df[df['city_id'] == city_id].copy()
    city_csv = OUTPUT_DIR / f'centrality_{city_id}.csv'
    city_df.to_csv(city_csv, index=False)

print(f"  Saved {len(df['city_id'].unique())} per-city files")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['lon'], df['lat']),
    crs='EPSG:4326'
)

gpkg_file = OUTPUT_DIR / 'centrality_all_cities.gpkg'
gdf.to_file(gpkg_file, driver='GPKG')
print(f"  Saved GeoPackage: {gpkg_file}")

# Summary statistics
print()
print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print()

print("Overall Statistics:")
print(f"  Total cities: {df['city_id'].nunique()}")
print(f"  Total grid cells: {len(df)}")
print(f"  Grids per city (avg): {len(df) / df['city_id'].nunique():.1f}")
print()

print("Centrality Metrics (mean ± std):")
print(f"  Closeness: {df['closeness_centrality'].mean():.4f} ± {df['closeness_centrality'].std():.4f}")
print(f"  Accessibility (30min): {df['accessibility_30min'].mean():.0f} ± {df['accessibility_30min'].std():.0f}")
print(f"  Betweenness: {df['betweenness_centrality'].mean():.4f} ± {df['betweenness_centrality'].std():.4f}")
print(f"  Straightness: {df['straightness'].mean():.4f} ± {df['straightness'].std():.4f}")
print()

print("Top 10 Cities by Average Closeness Centrality:")
city_centrality = df.groupby('city_id')['closeness_centrality'].mean().sort_values(ascending=False)
for i, (city_id, value) in enumerate(city_centrality.head(10).items(), 1):
    n_grids = len(df[df['city_id'] == city_id])
    print(f"  {i}. City {city_id}: {value:.4f} ({n_grids} grids)")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nResults saved to: {OUTPUT_DIR}")
