#!/usr/bin/env python3
"""
Calculate grid centrality based on route intersections.

Centrality metrics:
- basic_centrality: Number of routes passing through each grid
- betweenness: Routes passing through / max routes (normalized 0-1)
- weighted_centrality: Population-weighted centrality (sum of sqrt(origin_pop * dest_pop) for routes)
- weighted_centrality_normalized: Normalized weighted centrality (0-1)

Usage:
    python3 calculate_centrality_hpc.py --city-id 11549 \
        --results-dir /scratch/kh3657/osrm/results \
        --cities-dir /scratch/kh3657/osrm/cities \
        --population-file /scratch/kh3657/osrm/data/Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-22.csv
"""

import argparse
import json
import h3
import numpy as np
from pathlib import Path
from collections import defaultdict
from shapely.geometry import shape, Polygon, mapping
import time
import csv


def load_city_boundary(geojson_path):
    """Load city boundary from GeoJSON file."""
    with open(geojson_path) as f:
        data = json.load(f)

    if data['type'] == 'FeatureCollection':
        geometry = data['features'][0]['geometry']
    elif data['type'] == 'Feature':
        geometry = data['geometry']
    else:
        geometry = data

    return shape(geometry)


def generate_h3_grids(geometry, resolution=6):
    """Generate H3 hexagonal grids covering the geometry."""
    h3_cells = list(h3.geo_to_cells(geometry, resolution))

    grids = {}
    for cell in h3_cells:
        # Get hexagon boundary
        boundary = h3.cell_to_boundary(cell)
        # Convert to Polygon (boundary is list of (lat, lon) tuples)
        coords = [(lon, lat) for lat, lon in boundary]
        coords.append(coords[0])  # Close the polygon
        grids[cell] = Polygon(coords)

    return grids


def load_population_data(population_file, city_id):
    """Load population data for a specific city."""
    pop_dict = {}

    if not Path(population_file).exists():
        print(f"Warning: Population file not found: {population_file}")
        return pop_dict

    print(f"Loading population data for city {city_id}...")

    with open(population_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get('ID_HDC_G0', '')) == str(city_id):
                h3_idx = row.get('h3index', '')
                pop = float(row.get('population_2015', 0) or 0)
                if h3_idx and pop > 0:
                    pop_dict[h3_idx] = pop

    print(f"  Loaded population for {len(pop_dict)} grids")
    if pop_dict:
        print(f"  Total population: {sum(pop_dict.values()):,.0f}")

    return pop_dict


def calculate_centrality(routes_file, grids, pop_dict=None):
    """Calculate centrality metrics for each grid."""
    print("Loading routes...")
    with open(routes_file) as f:
        routes_data = json.load(f)

    features = routes_data['features']
    total_routes = len(features)
    print(f"Loaded {total_routes} routes")

    # Initialize centrality counters
    basic_centrality = defaultdict(int)
    weighted_centrality = defaultdict(float)

    route_count = 0
    update_interval = max(1, total_routes // 10)

    start_time = time.time()
    print("Calculating route-grid intersections...")

    for feature in features:
        route_geom = shape(feature['geometry'])
        props = feature['properties']

        origin_h3 = props.get('origin_h3', '')
        dest_h3 = props.get('destination_h3', '')

        # Calculate route weight based on origin/destination population
        route_weight = 0
        if pop_dict:
            origin_pop = pop_dict.get(origin_h3, 0)
            dest_pop = pop_dict.get(dest_h3, 0)
            if origin_pop > 0 and dest_pop > 0:
                # Geometric mean of populations
                route_weight = np.sqrt(origin_pop * dest_pop)

        # Find grids that this route intersects
        for h3_idx, grid_geom in grids.items():
            if route_geom.intersects(grid_geom):
                basic_centrality[h3_idx] += 1
                weighted_centrality[h3_idx] += route_weight

        route_count += 1
        if route_count % update_interval == 0:
            elapsed = time.time() - start_time
            rate = route_count / elapsed if elapsed > 0 else 0
            print(f"  Progress: {route_count}/{total_routes} ({100*route_count/total_routes:.1f}%) - {rate:.1f} routes/sec")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    # Calculate normalized values
    max_basic = max(basic_centrality.values()) if basic_centrality else 1
    max_weighted = max(weighted_centrality.values()) if weighted_centrality else 1

    betweenness = {h3_idx: count / max_basic for h3_idx, count in basic_centrality.items()}
    weighted_normalized = {h3_idx: val / max_weighted for h3_idx, val in weighted_centrality.items()}

    return basic_centrality, betweenness, weighted_centrality, weighted_normalized, total_routes


def main():
    parser = argparse.ArgumentParser(description='Calculate grid centrality from routes')
    parser.add_argument('--city-id', required=True, help='City identifier')
    parser.add_argument('--results-dir', default='/scratch/kh3657/osrm/results', help='Results directory')
    parser.add_argument('--cities-dir', default='/scratch/kh3657/osrm/cities', help='Cities directory')
    parser.add_argument('--population-file', default='/scratch/kh3657/osrm/data/Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-22.csv',
                        help='Population data CSV file')
    parser.add_argument('--h3-resolution', type=int, default=6, help='H3 resolution')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cities_dir = Path(args.cities_dir)
    city_id = args.city_id

    print(f"=" * 60)
    print(f"Calculating centrality for city: {city_id}")
    print(f"=" * 60)

    # Load city boundary
    boundary_file = cities_dir / f"{city_id}.geojson"
    if not boundary_file.exists():
        print(f"ERROR: Boundary file not found: {boundary_file}")
        return 1

    print("\n1. Loading city boundary...")
    geometry = load_city_boundary(boundary_file)
    print(f"   Loaded boundary")

    # Generate H3 grids
    print(f"\n2. Generating H3 grids (resolution {args.h3_resolution})...")
    grids = generate_h3_grids(geometry, args.h3_resolution)
    print(f"   Generated {len(grids)} H3 cells")

    # Load population data
    print(f"\n3. Loading population data...")
    pop_dict = load_population_data(args.population_file, city_id)

    # Check for routes file
    routes_file = results_dir / f"{city_id}_routes.geojson"
    if not routes_file.exists():
        print(f"ERROR: Routes file not found: {routes_file}")
        return 1

    # Load matrix for centroid info
    matrix_file = results_dir / f"{city_id}_matrix.json"
    matrix_data = None
    if matrix_file.exists():
        with open(matrix_file) as f:
            matrix_data = json.load(f)

    # Calculate centrality
    print("\n4. Calculating centrality...")
    basic_centrality, betweenness, weighted_centrality, weighted_normalized, total_routes = \
        calculate_centrality(routes_file, grids, pop_dict)

    # Build output
    print("\n5. Building output...")
    centroids = matrix_data['centroids'] if matrix_data else []
    centroid_lookup = {c['h3_index']: c for c in centroids}

    grid_data = []
    for h3_idx, grid_geom in grids.items():
        lat, lon = h3.cell_to_latlng(h3_idx)
        population = pop_dict.get(h3_idx, 0)

        grid_data.append({
            'h3_index': h3_idx,
            'lat': lat,
            'lon': lon,
            'population': population,
            'basic_centrality': basic_centrality.get(h3_idx, 0),
            'betweenness': betweenness.get(h3_idx, 0),
            'weighted_centrality': weighted_centrality.get(h3_idx, 0),
            'weighted_centrality_normalized': weighted_normalized.get(h3_idx, 0),
            'geometry': mapping(grid_geom)
        })

    # Sort by weighted centrality (descending)
    grid_data.sort(key=lambda x: x['weighted_centrality'], reverse=True)

    # Statistics
    print("\n6. Statistics:")
    grids_with_routes = sum(1 for g in grid_data if g['basic_centrality'] > 0)
    grids_with_pop = sum(1 for g in grid_data if g['population'] > 0)
    max_basic = max(g['basic_centrality'] for g in grid_data) if grid_data else 0
    mean_basic = sum(g['basic_centrality'] for g in grid_data) / len(grid_data) if grid_data else 0
    max_weighted = max(g['weighted_centrality'] for g in grid_data) if grid_data else 0
    mean_weighted = sum(g['weighted_centrality'] for g in grid_data) / len(grid_data) if grid_data else 0
    total_pop = sum(g['population'] for g in grid_data)

    print(f"   Total grids: {len(grid_data)}")
    print(f"   Grids with routes: {grids_with_routes}")
    print(f"   Grids with population: {grids_with_pop}")
    print(f"   Total population: {total_pop:,.0f}")
    print(f"   Total routes: {total_routes}")
    print(f"   Basic centrality - max: {max_basic}, mean: {mean_basic:.1f}")
    print(f"   Weighted centrality - max: {max_weighted:,.0f}, mean: {mean_weighted:,.0f}")

    print("\n   Top 5 most central grids (by weighted centrality):")
    for g in grid_data[:5]:
        print(f"     {g['h3_index']}: basic={g['basic_centrality']}, "
              f"weighted={g['weighted_centrality']:,.0f}, pop={g['population']:,.0f}")

    # Save as GeoJSON
    features = []
    for g in grid_data:
        features.append({
            'type': 'Feature',
            'properties': {
                'h3_index': g['h3_index'],
                'lat': round(g['lat'], 6),
                'lon': round(g['lon'], 6),
                'population': round(g['population'], 0),
                'basic_centrality': g['basic_centrality'],
                'betweenness': round(g['betweenness'], 4),
                'weighted_centrality': round(g['weighted_centrality'], 2),
                'weighted_centrality_normalized': round(g['weighted_centrality_normalized'], 4)
            },
            'geometry': g['geometry']
        })

    output = {
        'type': 'FeatureCollection',
        'properties': {
            'city_id': city_id,
            'h3_resolution': args.h3_resolution,
            'n_grids': len(grid_data),
            'total_routes': total_routes,
            'total_population': round(total_pop, 0),
            'max_basic_centrality': max_basic,
            'mean_basic_centrality': round(mean_basic, 2),
            'max_weighted_centrality': round(max_weighted, 2),
            'mean_weighted_centrality': round(mean_weighted, 2)
        },
        'features': features
    }

    output_file = results_dir / f"{city_id}_centrality.geojson"
    with open(output_file, 'w') as f:
        json.dump(output, f)

    print(f"\n7. Saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

    return 0


if __name__ == '__main__':
    exit(main())
