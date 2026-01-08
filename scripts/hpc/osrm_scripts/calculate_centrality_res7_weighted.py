#!/usr/bin/env python3
"""
Calculate population-weighted grid centrality for resolution 7 routing results.

Weights each route by the population at the origin hexagon.
Uses resolution 6 population data and maps to resolution 7 via parent relationship.

Usage:
    python3 calculate_centrality_res7_weighted.py --results-dir results
    python3 calculate_centrality_res7_weighted.py --results-dir results --city-id 12400
"""

import argparse
import json
import h3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from shapely.geometry import shape, Polygon, mapping
from shapely.strtree import STRtree
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time


# Global population lookup (loaded once, shared across workers)
_population_lookup = {}


def load_population_data(csv_path):
    """Load population data from the neighborhood CSV.

    Returns dict: {h3_res6_index: population}
    """
    print(f"  Loading population data from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['h3index', 'population_2015'])

    # Create lookup dict
    pop_lookup = {}
    for _, row in df.iterrows():
        h3_idx = row['h3index']
        pop = row['population_2015']
        if pd.notna(pop) and pop > 0:
            pop_lookup[h3_idx] = pop

    print(f"  Loaded population for {len(pop_lookup)} hexagons")
    return pop_lookup


def get_population_for_res7(h3_res7, population_lookup):
    """Get population for a resolution 7 cell from its resolution 6 parent.

    Since res6 parent has ~7 res7 children, we divide by 7 to distribute.
    """
    try:
        parent_res6 = h3.cell_to_parent(h3_res7, 6)
        pop = population_lookup.get(parent_res6, 0)
        # Divide by approximate number of children (7) to distribute population
        return pop / 7.0
    except:
        return 0


def load_routes_and_grids(results_dir, city_pattern):
    """Find and load routes and matrix files for a city."""
    matrix_files = list(results_dir.glob(f"*{city_pattern}*_res7_matrix.json"))
    if not matrix_files:
        return None, None, None, None

    matrix_file = matrix_files[0]
    base_name = matrix_file.stem.replace('_res7_matrix', '')

    routes_file = results_dir / f"{base_name}_res7_routes.geojson"
    if not routes_file.exists():
        return None, None, None, None

    with open(matrix_file) as f:
        matrix_data = json.load(f)

    with open(routes_file) as f:
        routes_data = json.load(f)

    return matrix_data, routes_data, matrix_file, routes_file


def generate_h3_polygons(h3_indices):
    """Generate polygon geometries for H3 indices."""
    grids = {}
    for cell in h3_indices:
        boundary = h3.cell_to_boundary(cell)
        coords = [(lon, lat) for lat, lon in boundary]
        coords.append(coords[0])
        grids[cell] = Polygon(coords)
    return grids


def _process_route_batch_weighted(args):
    """Worker function to process a batch of routes with population weighting."""
    route_data, grid_geometries, grid_keys, population_lookup = args

    # Build spatial index
    spatial_index = STRtree(grid_geometries)

    results = defaultdict(float)  # Use float for weighted values

    for route_geom, origin_h3 in route_data:
        # Get population weight for origin
        weight = get_population_for_res7(origin_h3, population_lookup)
        if weight <= 0:
            weight = 1  # Fallback to unweighted if no population data

        # Query spatial index
        candidate_indices = spatial_index.query(route_geom)
        for idx in candidate_indices:
            if route_geom.intersects(grid_geometries[idx]):
                results[grid_keys[idx]] += weight

    return dict(results)


def calculate_centrality_weighted(routes_data, grids, population_lookup, n_workers=None):
    """Calculate population-weighted centrality with spatial indexing and parallel processing."""
    features = routes_data['features']
    total_routes = len(features)
    print(f"  Processing {total_routes} routes with population weighting...")

    if n_workers is None:
        n_workers = min(4, multiprocessing.cpu_count())

    start_time = time.time()

    # Prepare grid data
    grid_list = list(grids.items())
    grid_geometries = [geom for _, geom in grid_list]
    grid_keys = [key for key, _ in grid_list]

    # Parse route geometries and extract origin H3
    print(f"  Parsing {total_routes} routes and extracting origins...")
    route_data = []
    for f in features:
        geom = shape(f['geometry'])
        origin_h3 = f['properties'].get('origin_h3', '')
        route_data.append((geom, origin_h3))

    parse_time = time.time() - start_time
    print(f"  Parsed in {parse_time:.1f}s")

    # Check population coverage
    origins_with_pop = sum(1 for _, origin in route_data
                          if get_population_for_res7(origin, population_lookup) > 0)
    print(f"  Routes with population data: {origins_with_pop}/{total_routes} ({100*origins_with_pop/total_routes:.1f}%)")

    # Split into batches
    batch_size = max(100, len(route_data) // n_workers + 1)
    batches = []
    for i in range(0, len(route_data), batch_size):
        batch = route_data[i:i + batch_size]
        batches.append((batch, grid_geometries, grid_keys, population_lookup))

    print(f"  Processing {len(batches)} batches with {n_workers} workers...")

    # Process in parallel
    weighted_centrality = defaultdict(float)
    calc_start = time.time()

    if n_workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_route_batch_weighted, batches))

        for batch_result in results:
            for k, v in batch_result.items():
                weighted_centrality[k] += v
    else:
        result = _process_route_batch_weighted(batches[0] if batches else ([], grid_geometries, grid_keys, population_lookup))
        weighted_centrality.update(result)

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({total_routes/elapsed:.0f} routes/sec)")

    # Calculate normalized betweenness
    max_weighted = max(weighted_centrality.values()) if weighted_centrality else 1
    betweenness = {h3_idx: val / max_weighted for h3_idx, val in weighted_centrality.items()}

    return weighted_centrality, betweenness, total_routes


def process_city(results_dir, city_id, output_dir, population_lookup):
    """Process a single city."""
    print(f"\n{'='*60}")
    print(f"Processing city: {city_id}")
    print(f"{'='*60}")

    # Load data
    print("\n1. Loading routes and grid data...")
    matrix_data, routes_data, matrix_file, routes_file = load_routes_and_grids(results_dir, str(city_id))

    if not matrix_data:
        print(f"  ERROR: Could not find data for city {city_id}")
        return False

    city_name = matrix_data.get('city_name', city_id)
    h3_indices = matrix_data.get('h3_indices', [])
    h3_resolution = matrix_data.get('h3_resolution', 7)
    centroids = matrix_data.get('centroids', [])

    print(f"  City: {city_name}")
    print(f"  H3 Resolution: {h3_resolution}")
    print(f"  Grids: {len(h3_indices)}")

    # Generate grid polygons
    print("\n2. Generating H3 grid polygons...")
    grids = generate_h3_polygons(h3_indices)
    print(f"  Generated {len(grids)} polygons")

    # Calculate population-weighted centrality
    print("\n3. Calculating population-weighted centrality...")
    weighted_centrality, betweenness, total_routes = calculate_centrality_weighted(
        routes_data, grids, population_lookup
    )

    # Also calculate grid populations for reference
    print("\n4. Calculating grid populations...")
    grid_populations = {}
    for h3_idx in h3_indices:
        grid_populations[h3_idx] = get_population_for_res7(h3_idx, population_lookup)

    total_pop = sum(grid_populations.values())
    grids_with_pop = sum(1 for p in grid_populations.values() if p > 0)
    print(f"  Grids with population: {grids_with_pop}/{len(h3_indices)}")
    print(f"  Total population: {total_pop:,.0f}")

    # Build output
    print("\n5. Building output...")
    centroid_lookup = {c['h3_index']: c for c in centroids}

    grid_data = []
    for h3_idx in h3_indices:
        centroid = centroid_lookup.get(h3_idx, {})
        lat = centroid.get('lat', 0)
        lon = centroid.get('lon', 0)

        grid_data.append({
            'h3_index': h3_idx,
            'lat': lat,
            'lon': lon,
            'population': grid_populations.get(h3_idx, 0),
            'weighted_centrality': weighted_centrality.get(h3_idx, 0),
            'betweenness': betweenness.get(h3_idx, 0),
            'geometry': mapping(grids[h3_idx])
        })

    # Sort by weighted centrality
    grid_data.sort(key=lambda x: x['weighted_centrality'], reverse=True)

    # Statistics
    grids_with_routes = sum(1 for g in grid_data if g['weighted_centrality'] > 0)
    max_weighted = max(g['weighted_centrality'] for g in grid_data) if grid_data else 0
    mean_weighted = sum(g['weighted_centrality'] for g in grid_data) / len(grid_data) if grid_data else 0

    print("\n6. Statistics:")
    print(f"   Total grids: {len(grid_data)}")
    print(f"   Grids with routes: {grids_with_routes}")
    print(f"   Total routes: {total_routes}")
    print(f"   Weighted centrality - max: {max_weighted:,.1f}, mean: {mean_weighted:,.1f}")

    print("\n   Top 5 most central grids (population-weighted):")
    for g in grid_data[:5]:
        print(f"     {g['h3_index']}: pop={g['population']:.0f}, centrality={g['weighted_centrality']:,.1f}, betweenness={g['betweenness']:.4f}")

    # Save as GeoJSON
    features = []
    for g in grid_data:
        features.append({
            'type': 'Feature',
            'properties': {
                'h3_index': g['h3_index'],
                'lat': round(g['lat'], 6),
                'lon': round(g['lon'], 6),
                'population': round(g['population'], 2),
                'weighted_centrality': round(g['weighted_centrality'], 2),
                'betweenness': round(g['betweenness'], 4)
            },
            'geometry': g['geometry']
        })

    output = {
        'type': 'FeatureCollection',
        'properties': {
            'city_id': city_id,
            'city_name': city_name,
            'h3_resolution': h3_resolution,
            'n_grids': len(grid_data),
            'total_routes': total_routes,
            'total_population': round(total_pop, 2),
            'max_weighted_centrality': round(max_weighted, 2),
            'mean_weighted_centrality': round(mean_weighted, 2),
            'weighting': 'population_2015'
        },
        'features': features
    }

    output_file = output_dir / f"{city_name}_{city_id}_res7_centrality_weighted.geojson"
    with open(output_file, 'w') as f:
        json.dump(output, f)

    print(f"\n7. Saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

    return True


def main():
    parser = argparse.ArgumentParser(description='Calculate population-weighted grid centrality')
    parser.add_argument('--results-dir', default='results', help='Directory with routing results')
    parser.add_argument('--output-dir', help='Output directory (default: same as results-dir)')
    parser.add_argument('--population-csv', help='CSV file with population data')
    parser.add_argument('--city-id', help='Process only this city ID')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')

    args = parser.parse_args()

    # Find the script directory to locate data files
    script_dir = Path(__file__).parent.parent

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = script_dir / results_dir

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find population CSV
    if args.population_csv:
        pop_csv = Path(args.population_csv).expanduser()
    else:
        pop_csv = script_dir / "data" / "Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-22.csv"

    if not pop_csv.exists():
        print(f"ERROR: Population CSV not found: {pop_csv}")
        return 1

    print("=" * 60)
    print("POPULATION-WEIGHTED CENTRALITY CALCULATION")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Population CSV: {pop_csv}")
    print("=" * 60)

    # Load population data
    population_lookup = load_population_data(pop_csv)

    if args.city_id:
        success = process_city(results_dir, args.city_id, output_dir, population_lookup)
        return 0 if success else 1
    else:
        # Process all cities
        matrix_files = list(results_dir.glob("*_res7_matrix.json"))
        print(f"\nFound {len(matrix_files)} cities to process")

        processed = 0
        failed = 0

        for matrix_file in sorted(matrix_files):
            name = matrix_file.stem.replace('_res7_matrix', '')
            parts = name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                city_id = parts[1]
            else:
                city_id = name

            success = process_city(results_dir, city_id, output_dir, population_lookup)
            if success:
                processed += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"COMPLETE: {processed} processed, {failed} failed")
        print(f"{'='*60}")

        return 0


if __name__ == '__main__':
    exit(main())
