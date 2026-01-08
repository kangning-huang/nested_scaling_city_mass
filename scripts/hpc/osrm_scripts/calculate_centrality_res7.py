#!/usr/bin/env python3
"""
Calculate grid centrality for resolution 7 routing results.

This script uses the routes and grids from resolution 7 routing output.

Usage:
    python3 calculate_centrality_res7.py --results-dir ~/results_res7
    python3 calculate_centrality_res7.py --results-dir ~/results_res7 --city-id 12400
"""

import argparse
import json
import h3
import numpy as np
from pathlib import Path
from collections import defaultdict
from shapely.geometry import shape, Polygon, mapping
from shapely.strtree import STRtree
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time


def load_routes_and_grids(results_dir, city_pattern):
    """Find and load routes and matrix files for a city."""
    # Find matrix file (contains grid info)
    matrix_files = list(results_dir.glob(f"*{city_pattern}*_res7_matrix.json"))
    if not matrix_files:
        return None, None, None, None

    matrix_file = matrix_files[0]
    base_name = matrix_file.stem.replace('_res7_matrix', '')

    # Find routes file
    routes_file = results_dir / f"{base_name}_res7_routes.geojson"
    if not routes_file.exists():
        return None, None, None, None

    # Load matrix
    with open(matrix_file) as f:
        matrix_data = json.load(f)

    # Load routes
    with open(routes_file) as f:
        routes_data = json.load(f)

    return matrix_data, routes_data, matrix_file, routes_file


def generate_h3_polygons(h3_indices):
    """Generate polygon geometries for H3 indices."""
    grids = {}
    for cell in h3_indices:
        boundary = h3.cell_to_boundary(cell)
        coords = [(lon, lat) for lat, lon in boundary]
        coords.append(coords[0])  # Close the polygon
        grids[cell] = Polygon(coords)
    return grids


def _process_route_batch(args):
    """Worker function to process a batch of routes with spatial indexing."""
    route_geoms, grid_geometries, grid_keys = args

    # Build spatial index for this worker
    spatial_index = STRtree(grid_geometries)

    results = defaultdict(int)
    for route_geom in route_geoms:
        # Query spatial index - only get nearby grids (much faster than checking all)
        candidate_indices = spatial_index.query(route_geom)
        for idx in candidate_indices:
            if route_geom.intersects(grid_geometries[idx]):
                results[grid_keys[idx]] += 1
    return dict(results)


def calculate_centrality(routes_data, grids, n_workers=None):
    """Calculate centrality with spatial indexing and parallel processing.

    Uses STRtree for O(log n) spatial queries instead of O(n) brute force,
    and multiprocessing for parallel route processing.
    """
    features = routes_data['features']
    total_routes = len(features)
    print(f"  Processing {total_routes} routes...")

    if n_workers is None:
        n_workers = min(4, multiprocessing.cpu_count())

    start_time = time.time()

    # Prepare grid data
    grid_list = list(grids.items())
    grid_geometries = [geom for _, geom in grid_list]
    grid_keys = [key for key, _ in grid_list]

    # Parse all route geometries upfront
    print(f"  Parsing {total_routes} route geometries...")
    route_geoms = [shape(f['geometry']) for f in features]
    parse_time = time.time() - start_time
    print(f"  Parsed in {parse_time:.1f}s")

    # Split routes into batches for parallel processing
    batch_size = max(100, len(route_geoms) // n_workers + 1)
    batches = []
    for i in range(0, len(route_geoms), batch_size):
        batch = route_geoms[i:i + batch_size]
        batches.append((batch, grid_geometries, grid_keys))

    print(f"  Processing {len(batches)} batches with {n_workers} workers (STRtree indexed)...")

    # Process in parallel
    basic_centrality = defaultdict(int)
    calc_start = time.time()

    if n_workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_route_batch, batches))

        # Merge results from all workers
        for batch_result in results:
            for k, v in batch_result.items():
                basic_centrality[k] += v
    else:
        # Single-threaded fallback for small datasets
        result = _process_route_batch(batches[0] if batches else ([], grid_geometries, grid_keys))
        basic_centrality.update(result)

    calc_time = time.time() - calc_start
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({total_routes/elapsed:.0f} routes/sec)")

    # Calculate normalized values
    max_basic = max(basic_centrality.values()) if basic_centrality else 1
    betweenness = {h3_idx: count / max_basic for h3_idx, count in basic_centrality.items()}

    return basic_centrality, betweenness, total_routes


def process_city(results_dir, city_id, output_dir):
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

    # Calculate centrality
    print("\n3. Calculating centrality...")
    basic_centrality, betweenness, total_routes = calculate_centrality(routes_data, grids)

    # Build output
    print("\n4. Building output...")
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
            'basic_centrality': basic_centrality.get(h3_idx, 0),
            'betweenness': betweenness.get(h3_idx, 0),
            'geometry': mapping(grids[h3_idx])
        })

    # Sort by centrality
    grid_data.sort(key=lambda x: x['basic_centrality'], reverse=True)

    # Statistics
    grids_with_routes = sum(1 for g in grid_data if g['basic_centrality'] > 0)
    max_basic = max(g['basic_centrality'] for g in grid_data) if grid_data else 0
    mean_basic = sum(g['basic_centrality'] for g in grid_data) / len(grid_data) if grid_data else 0

    print("\n5. Statistics:")
    print(f"   Total grids: {len(grid_data)}")
    print(f"   Grids with routes: {grids_with_routes}")
    print(f"   Total routes: {total_routes}")
    print(f"   Basic centrality - max: {max_basic}, mean: {mean_basic:.1f}")

    print("\n   Top 5 most central grids:")
    for g in grid_data[:5]:
        print(f"     {g['h3_index']}: centrality={g['basic_centrality']}, betweenness={g['betweenness']:.4f}")

    # Save as GeoJSON
    features = []
    for g in grid_data:
        features.append({
            'type': 'Feature',
            'properties': {
                'h3_index': g['h3_index'],
                'lat': round(g['lat'], 6),
                'lon': round(g['lon'], 6),
                'basic_centrality': g['basic_centrality'],
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
            'max_centrality': max_basic,
            'mean_centrality': round(mean_basic, 2)
        },
        'features': features
    }

    output_file = output_dir / f"{city_name}_{city_id}_res7_centrality.geojson"
    with open(output_file, 'w') as f:
        json.dump(output, f)

    print(f"\n6. Saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

    return True


def main():
    parser = argparse.ArgumentParser(description='Calculate grid centrality for resolution 7 results')
    parser.add_argument('--results-dir', default='~/results_res7', help='Directory with routing results')
    parser.add_argument('--output-dir', help='Output directory (default: same as results-dir)')
    parser.add_argument('--city-id', help='Process only this city ID (optional)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CENTRALITY CALCULATION FOR RESOLUTION 7")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    if args.city_id:
        # Process single city
        success = process_city(results_dir, args.city_id, output_dir)
        return 0 if success else 1
    else:
        # Process all cities
        matrix_files = list(results_dir.glob("*_res7_matrix.json"))
        print(f"\nFound {len(matrix_files)} cities to process")

        processed = 0
        failed = 0

        for matrix_file in sorted(matrix_files):
            # Extract city_id from filename
            name = matrix_file.stem.replace('_res7_matrix', '')
            # Try to extract numeric ID
            parts = name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                city_id = parts[1]
            else:
                city_id = name

            success = process_city(results_dir, city_id, output_dir)
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
