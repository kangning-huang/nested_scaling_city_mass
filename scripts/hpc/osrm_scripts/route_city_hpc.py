#!/usr/bin/env python3
"""
Route computation for HPC - computes travel time matrices and optionally fetches polylines.
Assumes OSRM server is already running on localhost:5000.

Usage:
    python route_city_hpc.py --city-id shanghai --boundary cities/shanghai.geojson --h3-resolution 6 --fetch-polylines
"""

import argparse
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Try to import optional dependencies
try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
    print("Warning: h3 not installed. Install with: pip install h3")

try:
    from shapely.geometry import shape, mapping
    from shapely.ops import transform
    import pyproj
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def load_city_boundary(geojson_path):
    """Load city boundary from GeoJSON file."""
    with open(geojson_path) as f:
        data = json.load(f)

    # Handle both Feature and FeatureCollection
    if data['type'] == 'FeatureCollection':
        geometry = data['features'][0]['geometry']
    elif data['type'] == 'Feature':
        geometry = data['geometry']
    else:
        geometry = data

    return shape(geometry)


def generate_h3_grids(geometry, resolution=7):
    """Generate H3 hexagonal grids covering the geometry."""
    if not HAS_H3:
        raise ImportError("h3 library required. Install with: pip install h3")

    # H3 v4 API
    h3_cells = list(h3.geo_to_cells(geometry, resolution))

    centroids = []
    for cell in h3_cells:
        lat, lon = h3.cell_to_latlng(cell)
        centroids.append({
            'h3_index': cell,
            'lat': lat,
            'lon': lon
        })

    return centroids


def compute_matrix(centroids, osrm_url="http://localhost:5000"):
    """Compute travel time/distance matrix using OSRM Table API."""
    if not centroids:
        return None

    coords = ';'.join([f"{c['lon']},{c['lat']}" for c in centroids])
    url = f"{osrm_url}/table/v1/driving/{coords}?annotations=duration,distance"

    try:
        response = requests.get(url, timeout=300)
        data = response.json()

        if data.get('code') != 'Ok':
            print(f"OSRM error: {data.get('code')} - {data.get('message', '')}")
            return None

        return data
    except Exception as e:
        print(f"Matrix computation failed: {e}")
        return None


def fetch_single_route(origin, destination, osrm_url, session):
    """Fetch a single route polyline."""
    url = f"{osrm_url}/route/v1/driving/{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}?overview=full&geometries=geojson"

    try:
        response = session.get(url, timeout=30)
        data = response.json()

        if data.get('code') == 'Ok' and data.get('routes'):
            route = data['routes'][0]
            return {
                'origin_h3': origin['h3_index'],
                'destination_h3': destination['h3_index'],
                'duration': route['duration'],
                'distance': route['distance'],
                'geometry': route['geometry']
            }
    except Exception as e:
        pass  # Silently skip failed routes

    return None


def simplify_geometry(geometry, tolerance=0.0001):
    """Simplify LineString geometry using Douglas-Peucker algorithm."""
    if not HAS_SHAPELY:
        return geometry

    try:
        geom = shape(geometry)
        simplified = geom.simplify(tolerance, preserve_topology=True)

        # Round coordinates to 5 decimal places (~1.1m precision)
        coords = [[round(x, 5), round(y, 5)] for x, y in simplified.coords]
        return {'type': 'LineString', 'coordinates': coords}
    except:
        return geometry


def fetch_all_polylines(centroids, osrm_url="http://localhost:5000", max_workers=8, simplify=True):
    """Fetch polylines for all origin-destination pairs."""
    n = len(centroids)
    total_routes = n * (n - 1)  # Exclude self-routes

    print(f"Fetching {total_routes} polylines with {max_workers} workers...")

    # Create session with connection pooling and retries
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(
        pool_connections=max_workers + 5,
        pool_maxsize=max_workers + 5,
        max_retries=retries
    )
    session.mount('http://', adapter)

    routes = []
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i, origin in enumerate(centroids):
            for j, destination in enumerate(centroids):
                if i != j:  # Skip self-routes
                    future = executor.submit(
                        fetch_single_route, origin, destination, osrm_url, session
                    )
                    futures[future] = (i, j)

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result:
                if simplify and 'geometry' in result:
                    result['geometry'] = simplify_geometry(result['geometry'])
                routes.append(result)

            # Progress update every 10%
            if completed % max(1, total_routes // 10) == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_routes - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{total_routes} ({100*completed/total_routes:.1f}%) - ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"Completed {len(routes)} routes in {elapsed:.1f}s ({len(routes)/elapsed:.1f} routes/sec)")

    return routes


def save_results(city_id, centroids, matrix_data, routes, output_dir):
    """Save matrix and routes to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save matrix JSON
    h3_indices = [c['h3_index'] for c in centroids]
    matrix_result = {
        'city_id': city_id,
        'h3_indices': h3_indices,
        'centroids': centroids,
        'durations': matrix_data['durations'],
        'distances': matrix_data['distances'],
        'n_grids': len(h3_indices)
    }

    matrix_file = output_dir / f"{city_id}_matrix.json"
    with open(matrix_file, 'w') as f:
        json.dump(matrix_result, f)
    print(f"Saved matrix: {matrix_file}")

    # Save routes GeoJSON if available
    if routes:
        features = []
        for route in routes:
            feature = {
                'type': 'Feature',
                'properties': {
                    'origin_h3': route['origin_h3'],
                    'destination_h3': route['destination_h3'],
                    'duration': route['duration'],
                    'distance': route['distance']
                },
                'geometry': route['geometry']
            }
            features.append(feature)

        routes_geojson = {
            'type': 'FeatureCollection',
            'properties': {
                'city_id': city_id,
                'total_routes': len(routes)
            },
            'features': features
        }

        routes_file = output_dir / f"{city_id}_routes.geojson"
        with open(routes_file, 'w') as f:
            json.dump(routes_geojson, f)
        print(f"Saved routes: {routes_file} ({len(routes)} routes)")


def main():
    parser = argparse.ArgumentParser(description='Compute OSRM routing matrix and polylines')
    parser.add_argument('--city-id', required=True, help='City identifier')
    parser.add_argument('--boundary', required=True, help='Path to city boundary GeoJSON')
    parser.add_argument('--h3-resolution', type=int, default=7, help='H3 grid resolution (default: 7)')
    parser.add_argument('--fetch-polylines', action='store_true', help='Fetch route polylines')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--osrm-url', default='http://localhost:5000', help='OSRM server URL')
    parser.add_argument('--max-workers', type=int, default=8, help='Max parallel workers for polylines')
    parser.add_argument('--no-simplify', action='store_true', help='Disable polyline simplification')

    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Processing city: {args.city_id}")
    print(f"H3 resolution: {args.h3_resolution}")
    print(f"Fetch polylines: {args.fetch_polylines}")
    print(f"=" * 60)

    # Load boundary
    print("\n1. Loading city boundary...")
    geometry = load_city_boundary(args.boundary)
    print(f"   Loaded boundary from {args.boundary}")

    # Generate H3 grids
    print(f"\n2. Generating H3 grids (resolution {args.h3_resolution})...")
    centroids = generate_h3_grids(geometry, args.h3_resolution)
    print(f"   Generated {len(centroids)} H3 cells")

    if not centroids:
        print("ERROR: No H3 cells generated!")
        return 1

    # Compute matrix
    print("\n3. Computing travel time matrix...")
    matrix_data = compute_matrix(centroids, args.osrm_url)

    if not matrix_data:
        print("ERROR: Matrix computation failed!")
        return 1

    print(f"   Matrix computed: {len(centroids)}x{len(centroids)} = {len(centroids)**2} pairs")

    # Fetch polylines if requested
    routes = []
    if args.fetch_polylines:
        print("\n4. Fetching route polylines...")
        routes = fetch_all_polylines(
            centroids,
            osrm_url=args.osrm_url,
            max_workers=args.max_workers,
            simplify=not args.no_simplify
        )

    # Save results
    print("\n5. Saving results...")
    save_results(args.city_id, centroids, matrix_data, routes, args.output_dir)

    print(f"\n{'=' * 60}")
    print("COMPLETE!")
    print(f"  - H3 cells: {len(centroids)}")
    print(f"  - Matrix pairs: {len(centroids)**2}")
    if routes:
        print(f"  - Routes with polylines: {len(routes)}")
    print(f"{'=' * 60}")

    return 0


if __name__ == '__main__':
    exit(main())
