#!/usr/bin/env python3
"""
Optimized Polyline Fetching for Resolution 7 Routes.

Reuses resolution 6 polylines for long-distance routes to reduce OSRM API calls by ~90%.

Strategy:
    - Routes > distance_threshold: Reuse res6 polyline via H3 parent mapping
    - Routes <= distance_threshold: Fetch from OSRM (precision matters for short routes)

Usage:
    # Process single city
    python fetch_polylines_optimized.py --city Wuhan_11549 --threshold 10000

    # Process all cities in a directory
    python fetch_polylines_optimized.py --all --threshold 10000
"""

import argparse
import json
import time
import requests
import h3
import subprocess
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def setup_logging(log_file, name):
    """Setup logging to file and console."""
    import logging

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(Path(log_file).expanduser())
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_res6_routes(routes_file):
    """Load res6 routes and build lookup index.

    Returns:
        dict: {(origin_res6, dest_res6): geometry} lookup
    """
    index = {}

    with open(routes_file) as f:
        data = json.load(f)

    for feature in data.get('features', []):
        props = feature.get('properties', {})
        origin = props.get('origin_h3')
        dest = props.get('destination_h3')
        geometry = feature.get('geometry')

        if origin and dest and geometry:
            # Only store routes with valid coordinates (not single point)
            coords = geometry.get('coordinates', [])
            if len(coords) >= 2:
                index[(origin, dest)] = geometry

    return index


def load_res7_matrix(matrix_file):
    """Load res7 matrix with distances.

    Returns:
        dict: Matrix data with h3_indices, centroids, distances, durations
    """
    with open(matrix_file) as f:
        return json.load(f)


def build_parent_mapping(h3_indices):
    """Build res7 -> res6 parent mapping.

    Args:
        h3_indices: List of res7 H3 cell indices

    Returns:
        dict: {res7_cell: res6_parent}
    """
    return {cell: h3.cell_to_parent(cell, 6) for cell in h3_indices}


def decompress_osrm(city_id, osrm_dir, logger):
    """Decompress OSRM tar.gz if needed."""
    tar_file = osrm_dir / f"{city_id}.tar.gz"
    city_osrm_dir = osrm_dir / str(city_id)
    osrm_file = city_osrm_dir / f"{city_id}.osrm"

    if osrm_file.exists():
        return city_osrm_dir

    if not tar_file.exists():
        logger.error(f"Neither OSRM dir nor tar.gz found for {city_id}")
        return None

    logger.info(f"Decompressing {city_id}...")
    city_osrm_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(city_osrm_dir)
        return city_osrm_dir
    except Exception as e:
        logger.error(f"Decompression failed: {e}")
        return None


def start_osrm_server(city_osrm_dir, city_id, logger=None):
    """Start OSRM routing server."""
    subprocess.run("docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend) 2>/dev/null",
                   shell=True, capture_output=True)
    time.sleep(2)

    cmd = f'cd {city_osrm_dir} && docker run --rm -d -p 5000:5000 -v "${{PWD}}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/{city_id}.osrm'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        for i in range(30):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:5000/health", timeout=2)
                if response.status_code == 200:
                    if logger:
                        logger.info(f"OSRM server ready for {city_id}")
                    return True
            except:
                pass
        time.sleep(5)
        return True
    return False


def stop_osrm_server():
    """Stop OSRM routing server."""
    subprocess.run("docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend) 2>/dev/null",
                   shell=True, capture_output=True)
    time.sleep(1)


def fetch_single_route(session, origin, destination):
    """Fetch a single route geometry from OSRM Route API."""
    url = (
        f"http://localhost:5000/route/v1/driving/"
        f"{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}"
        f"?overview=full&geometries=geojson"
    )

    try:
        response = session.get(url, timeout=30)
        data = response.json()

        if data.get('code') == 'Ok' and data.get('routes'):
            route = data['routes'][0]
            return {
                'geometry': route['geometry'],
                'duration': route['duration'],
                'distance': route['distance']
            }
    except Exception:
        pass
    return None


def fetch_polylines_optimized(
    city_name,
    city_id,
    res7_matrix,
    res6_routes_index,
    parent_mapping,
    distance_threshold,
    osrm_dir,
    logger,
    max_workers=8
):
    """Fetch polylines with res6 reuse optimization.

    Args:
        city_name: City name
        city_id: City ID
        res7_matrix: Res7 matrix data with centroids and distances
        res6_routes_index: {(origin_res6, dest_res6): geometry} lookup
        parent_mapping: {res7_cell: res6_parent} mapping
        distance_threshold: Routes above this (meters) reuse res6 polylines
        osrm_dir: OSRM directory path
        logger: Logger instance
        max_workers: Concurrent workers for OSRM fetching

    Returns:
        list: Route features with polylines
    """
    h3_indices = res7_matrix['h3_indices']
    centroids = res7_matrix['centroids']
    distances = res7_matrix['distances']
    durations = res7_matrix['durations']
    n_grids = len(h3_indices)

    # Build centroid lookup
    centroid_lookup = {c['h3_index']: c for c in centroids}

    # Separate routes into reuse vs fetch
    routes_to_fetch = []
    routes_reused = []
    routes_missing_res6 = []

    for i, origin_h3 in enumerate(h3_indices):
        for j, dest_h3 in enumerate(h3_indices):
            if i == j:  # Skip self-routes
                continue

            distance = distances[i][j]
            duration = durations[i][j]

            if distance > distance_threshold:
                # Try to reuse res6 polyline
                origin_parent = parent_mapping.get(origin_h3)
                dest_parent = parent_mapping.get(dest_h3)

                res6_key = (origin_parent, dest_parent)
                res6_geometry = res6_routes_index.get(res6_key)

                if res6_geometry:
                    routes_reused.append({
                        'origin_h3': origin_h3,
                        'destination_h3': dest_h3,
                        'duration': duration,
                        'distance': distance,
                        'geometry': res6_geometry,
                        'source': 'reused_res6'
                    })
                else:
                    # No res6 route available, need to fetch
                    routes_missing_res6.append((i, j, origin_h3, dest_h3, distance, duration))
            else:
                # Short route, need to fetch
                routes_to_fetch.append((i, j, origin_h3, dest_h3, distance, duration))

    # Combine routes that need fetching
    all_to_fetch = routes_to_fetch + routes_missing_res6

    total_routes = n_grids * (n_grids - 1)
    logger.info(f"Route breakdown:")
    logger.info(f"  Total routes: {total_routes}")
    logger.info(f"  Reused from res6: {len(routes_reused)} ({100*len(routes_reused)/total_routes:.1f}%)")
    logger.info(f"  To fetch (short): {len(routes_to_fetch)} ({100*len(routes_to_fetch)/total_routes:.1f}%)")
    logger.info(f"  To fetch (missing res6): {len(routes_missing_res6)} ({100*len(routes_missing_res6)/total_routes:.1f}%)")

    # Fetch routes from OSRM
    fetched_routes = []
    if all_to_fetch:
        # Start OSRM server
        city_osrm_dir = decompress_osrm(city_id, osrm_dir, logger)
        if not city_osrm_dir:
            logger.error(f"Cannot start OSRM server - OSRM files not found")
            return routes_reused  # Return what we have

        if not start_osrm_server(city_osrm_dir, city_id, logger):
            logger.error(f"Failed to start OSRM server")
            return routes_reused

        logger.info(f"Fetching {len(all_to_fetch)} routes from OSRM...")

        # Create session with connection pool
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=max_workers + 5,
            pool_maxsize=max_workers + 5,
            max_retries=Retry(total=3, backoff_factor=0.1)
        )
        session.mount('http://', adapter)

        def fetch_pair(args):
            i, j, origin_h3, dest_h3, distance, duration = args
            origin_centroid = centroid_lookup[origin_h3]
            dest_centroid = centroid_lookup[dest_h3]
            result = fetch_single_route(session, origin_centroid, dest_centroid)

            if result:
                return {
                    'origin_h3': origin_h3,
                    'destination_h3': dest_h3,
                    'duration': duration,
                    'distance': distance,
                    'geometry': result['geometry'],
                    'source': 'fetched'
                }
            return None

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_pair, args): args for args in all_to_fetch}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    fetched_routes.append(result)
                completed += 1

                if completed % 500 == 0:
                    logger.info(f"  Progress: {completed}/{len(all_to_fetch)} ({100*completed/len(all_to_fetch):.1f}%)")

        session.close()
        stop_osrm_server()

        logger.info(f"Fetched {len(fetched_routes)} routes successfully")

    # Combine all routes
    all_routes = routes_reused + fetched_routes
    logger.info(f"Total routes with polylines: {len(all_routes)}")

    return all_routes


def save_routes_geojson(city_name, city_id, routes, output_file, h3_resolution=7):
    """Save routes as GeoJSON FeatureCollection."""
    # Count sources
    reused_count = sum(1 for r in routes if r.get('source') == 'reused_res6')
    fetched_count = sum(1 for r in routes if r.get('source') == 'fetched')

    features = []
    for route in routes:
        features.append({
            'type': 'Feature',
            'properties': {
                'origin_h3': route['origin_h3'],
                'destination_h3': route['destination_h3'],
                'duration': route['duration'],
                'distance': route['distance'],
                'source': route.get('source', 'unknown')
            },
            'geometry': route['geometry']
        })

    geojson = {
        'type': 'FeatureCollection',
        'properties': {
            'city_id': city_id,
            'city_name': city_name,
            'h3_resolution': h3_resolution,
            'total_routes': len(routes),
            'reused_from_res6': reused_count,
            'fetched_from_osrm': fetched_count
        },
        'features': features
    }

    with open(output_file, 'w') as f:
        json.dump(geojson, f)

    return output_file


def process_city(
    city_name,
    city_id,
    res7_matrix_file,
    res6_routes_file,
    osrm_dir,
    results_dir,
    distance_threshold,
    logger
):
    """Process a single city."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Processing: {city_name} (ID: {city_id})")
    logger.info(f"{'=' * 60}")

    # Check output file
    output_file = results_dir / f"{city_name}_{city_id}_res7_routes.geojson"
    if output_file.exists():
        logger.info(f"Already done: {output_file}")
        return True

    # Load res7 matrix
    logger.info(f"Loading res7 matrix: {res7_matrix_file}")
    res7_matrix = load_res7_matrix(res7_matrix_file)
    n_grids = res7_matrix['n_grids']
    logger.info(f"  Grids: {n_grids}, Routes: {n_grids * (n_grids - 1)}")

    # Load res6 routes
    logger.info(f"Loading res6 routes: {res6_routes_file}")
    res6_routes_index = load_res6_routes(res6_routes_file)
    logger.info(f"  Indexed {len(res6_routes_index)} res6 routes")

    # Build parent mapping
    parent_mapping = build_parent_mapping(res7_matrix['h3_indices'])

    # Fetch polylines with optimization
    start_time = time.time()
    routes = fetch_polylines_optimized(
        city_name=city_name,
        city_id=city_id,
        res7_matrix=res7_matrix,
        res6_routes_index=res6_routes_index,
        parent_mapping=parent_mapping,
        distance_threshold=distance_threshold,
        osrm_dir=osrm_dir,
        logger=logger
    )
    elapsed = time.time() - start_time
    logger.info(f"Polyline fetching completed in {elapsed:.1f}s")

    # Save results
    save_routes_geojson(city_name, city_id, routes, output_file)
    logger.info(f"Saved to: {output_file}")

    return True


def find_matching_files(results_dir):
    """Find cities with both res7 matrix and res6 routes."""
    cities = []

    # Find res7 matrix files
    for matrix_file in results_dir.glob("*_res7_matrix.json"):
        # Parse city name and ID from filename
        # Format: {city_name}_{city_id}_res7_matrix.json
        stem = matrix_file.stem.replace('_res7_matrix', '')
        parts = stem.rsplit('_', 1)
        if len(parts) != 2:
            continue

        city_name = parts[0]
        city_id = parts[1]

        # Look for matching res6 routes
        res6_routes_file = results_dir / f"{city_name}_{city_id}_routes.geojson"
        if not res6_routes_file.exists():
            # Try without city name (old format)
            res6_routes_file = results_dir / f"{city_id}_routes.geojson"

        if res6_routes_file.exists():
            cities.append({
                'city_name': city_name,
                'city_id': city_id,
                'res7_matrix': matrix_file,
                'res6_routes': res6_routes_file
            })

    return cities


def main():
    parser = argparse.ArgumentParser(description='Fetch res7 polylines with res6 reuse optimization')
    parser.add_argument('--city', help='City to process in format: CityName_CityID (e.g., Wuhan_11549)')
    parser.add_argument('--all', action='store_true', help='Process all cities with matching res7 matrix and res6 routes')
    parser.add_argument('--threshold', type=int, default=10000, help='Distance threshold in meters (default: 10000)')
    parser.add_argument('--results-dir', default='~/results', help='Results directory')
    parser.add_argument('--osrm-dir', default='~/osrm', help='OSRM files directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    osrm_dir = Path(args.osrm_dir).expanduser()

    logger = setup_logging('~/polylines_optimized.log', 'polylines_opt')
    logger.info("=" * 60)
    logger.info("OPTIMIZED POLYLINE FETCHING")
    logger.info(f"Distance threshold: {args.threshold} meters")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"OSRM directory: {osrm_dir}")
    logger.info("=" * 60)

    if args.city:
        # Process single city
        parts = args.city.rsplit('_', 1)
        if len(parts) != 2:
            logger.error(f"Invalid city format: {args.city}. Use CityName_CityID")
            return

        city_name, city_id = parts

        # Find files
        res7_matrix = results_dir / f"{city_name}_{city_id}_res7_matrix.json"
        res6_routes = results_dir / f"{city_name}_{city_id}_routes.geojson"

        if not res7_matrix.exists():
            logger.error(f"Res7 matrix not found: {res7_matrix}")
            return

        if not res6_routes.exists():
            logger.error(f"Res6 routes not found: {res6_routes}")
            return

        process_city(
            city_name=city_name,
            city_id=city_id,
            res7_matrix_file=res7_matrix,
            res6_routes_file=res6_routes,
            osrm_dir=osrm_dir,
            results_dir=results_dir,
            distance_threshold=args.threshold,
            logger=logger
        )

    elif args.all:
        # Process all matching cities
        cities = find_matching_files(results_dir)
        logger.info(f"Found {len(cities)} cities with both res7 matrix and res6 routes")

        for city in cities:
            process_city(
                city_name=city['city_name'],
                city_id=city['city_id'],
                res7_matrix_file=city['res7_matrix'],
                res6_routes_file=city['res6_routes'],
                osrm_dir=osrm_dir,
                results_dir=results_dir,
                distance_threshold=args.threshold,
                logger=logger
            )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
