#!/usr/bin/env python3
"""
Fetch route polylines for HPC - runs after matrix computation.
Assumes OSRM server is already running on localhost:PORT.

Usage:
    PORT=5001 python3 fetch_polylines_hpc.py --city-id 11549 --results-dir /scratch/kh3657/osrm/results
"""

import argparse
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Try to import shapely for simplification
try:
    from shapely.geometry import LineString
    from shapely import simplify as shapely_simplify
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("Warning: shapely not installed. Polylines will not be simplified.")

# Configuration
SIMPLIFY_TOLERANCE = 0.0001  # ~11 meters
COORD_PRECISION = 5  # 5 decimal places = ~1.1m precision
MAX_WORKERS = 8


def simplify_geometry(geometry):
    """Simplify a GeoJSON geometry using Douglas-Peucker algorithm."""
    if not HAS_SHAPELY:
        return geometry

    coords = geometry.get('coordinates', [])
    if len(coords) < 2:
        return geometry

    try:
        line = LineString(coords)
        simplified = shapely_simplify(line, tolerance=SIMPLIFY_TOLERANCE, preserve_topology=True)

        new_coords = [
            [round(c[0], COORD_PRECISION), round(c[1], COORD_PRECISION)]
            for c in simplified.coords
        ]

        return {'type': 'LineString', 'coordinates': new_coords}
    except:
        return {
            'type': 'LineString',
            'coordinates': [[round(c[0], COORD_PRECISION), round(c[1], COORD_PRECISION)] for c in coords]
        }


def fetch_single_route(session, origin, destination, osrm_url):
    """Fetch a single route geometry from OSRM Route API."""
    url = (
        f"{osrm_url}/route/v1/driving/"
        f"{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}"
        f"?overview=full&geometries=geojson"
    )

    try:
        response = session.get(url, timeout=30)
        data = response.json()

        if data.get('code') != 'Ok':
            return None

        route = data['routes'][0]
        simplified_geom = simplify_geometry(route['geometry'])

        return {
            'origin_h3': origin['h3_index'],
            'destination_h3': destination['h3_index'],
            'duration': route['duration'],
            'distance': route['distance'],
            'geometry': simplified_geom
        }
    except:
        return None


def fetch_all_routes(centroids, osrm_url, max_workers=MAX_WORKERS):
    """Fetch all pairwise routes using parallel requests."""
    n = len(centroids)
    total_pairs = n * (n - 1)  # Exclude self-routes

    print(f"Fetching {total_pairs} routes with {max_workers} workers...")

    # Create session with connection pooling
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max_workers + 5,
        pool_maxsize=max_workers + 5,
        max_retries=Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    )
    session.mount('http://', adapter)

    features = []
    failed = []
    completed = 0
    start_time = time.time()

    def fetch_pair(pair):
        i, j, origin, dest = pair
        result = fetch_single_route(session, origin, dest, osrm_url)
        if result:
            return {
                'type': 'Feature',
                'properties': {
                    'origin_h3': result['origin_h3'],
                    'destination_h3': result['destination_h3'],
                    'duration': result['duration'],
                    'distance': result['distance']
                },
                'geometry': result['geometry']
            }
        return None

    # Generate O-D pairs (excluding self-routes)
    pairs = []
    for i, origin in enumerate(centroids):
        for j, dest in enumerate(centroids):
            if i != j:
                pairs.append((i, j, origin, dest))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_pair, pair): pair for pair in pairs}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result:
                features.append(result)
            else:
                pair = futures[future]
                failed.append((pair[2]['h3_index'], pair[3]['h3_index']))

            if completed % max(1, total_pairs // 10) == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_pairs - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{total_pairs} ({100*completed/total_pairs:.1f}%) - {rate:.1f} routes/sec - ETA: {eta:.0f}s")

    session.close()

    elapsed = time.time() - start_time
    print(f"Completed: {len(features)} routes in {elapsed:.1f}s ({len(features)/elapsed:.1f} routes/sec)")
    if failed:
        print(f"Failed: {len(failed)} routes")

    return features, failed


def main():
    import os

    parser = argparse.ArgumentParser(description='Fetch OSRM route polylines')
    parser.add_argument('--city-id', required=True, help='City identifier')
    parser.add_argument('--results-dir', default='/scratch/kh3657/osrm/results', help='Results directory')
    parser.add_argument('--port', type=int, default=None, help='OSRM server port (default: from PORT env var or 5000)')
    parser.add_argument('--max-workers', type=int, default=8, help='Max parallel workers')

    args = parser.parse_args()

    # Get port from args, env, or default
    port = args.port or int(os.environ.get('PORT', 5000))
    osrm_url = f"http://localhost:{port}"

    results_dir = Path(args.results_dir)
    city_id = args.city_id

    print(f"=" * 60)
    print(f"Fetching polylines for city: {city_id}")
    print(f"OSRM URL: {osrm_url}")
    print(f"=" * 60)

    # Load matrix file to get centroids
    matrix_file = results_dir / f"{city_id}_matrix.json"
    if not matrix_file.exists():
        print(f"ERROR: Matrix file not found: {matrix_file}")
        return 1

    with open(matrix_file) as f:
        matrix_data = json.load(f)

    centroids = matrix_data['centroids']
    n_grids = len(centroids)
    print(f"Loaded {n_grids} centroids from matrix file")

    if n_grids < 2:
        print("ERROR: Need at least 2 centroids for routing")
        return 1

    # Check if already done
    routes_file = results_dir / f"{city_id}_routes.geojson"
    if routes_file.exists():
        print(f"Routes file already exists: {routes_file}")
        return 0

    # Fetch routes
    features, failed = fetch_all_routes(centroids, osrm_url, args.max_workers)

    # Save results
    geojson = {
        'type': 'FeatureCollection',
        'properties': {
            'city_id': city_id,
            'n_grids': n_grids,
            'total_routes': len(features),
            'failed_routes': len(failed)
        },
        'features': features
    }

    with open(routes_file, 'w') as f:
        json.dump(geojson, f)

    print(f"\nSaved routes to: {routes_file}")
    print(f"File size: {routes_file.stat().st_size / 1024 / 1024:.2f} MB")

    return 0


if __name__ == '__main__':
    exit(main())
