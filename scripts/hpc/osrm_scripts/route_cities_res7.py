#!/usr/bin/env python3
"""
Route cities at H3 resolution 7 using pre-computed grids.

Usage:
    python route_cities_res7.py --grids-json china_cities_h3_grids_res7.json
    python route_cities_res7.py --grids-json china_cities_h3_grids_res7.json --city-id 12400
    python route_cities_res7.py --grids-json china_cities_h3_grids_res7.json --fetch-polylines
"""

import argparse
import json
import time
import tarfile
import shutil
import subprocess
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def start_osrm_server(city_osrm_dir, city_id, max_table_size=1000, logger=None):
    """Start OSRM routing server."""
    # Stop any existing server
    subprocess.run("docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend) 2>/dev/null",
                   shell=True, capture_output=True)
    time.sleep(2)

    cmd = f'cd {city_osrm_dir} && docker run --rm -d -p 5000:5000 -v "${{PWD}}:/data" osrm/osrm-backend osrm-routed --algorithm mld --max-table-size {max_table_size} /data/{city_id}.osrm'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        # Wait for server to be ready
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


def compute_matrix(centroids, logger=None):
    """Compute travel time matrix using OSRM Table API."""
    if len(centroids) == 0:
        return None

    coords = ';'.join([f"{c['lon']},{c['lat']}" for c in centroids])
    url = f"http://localhost:5000/table/v1/driving/{coords}?annotations=duration,distance"

    try:
        response = requests.get(url, timeout=300)
        data = response.json()

        if data.get('code') != 'Ok':
            if logger:
                logger.error(f"OSRM error: {data.get('code')} - {data.get('message', '')}")
            return None
        return data
    except Exception as e:
        if logger:
            logger.error(f"Matrix computation failed: {e}")
        return None


def fetch_single_route(origin, destination):
    """Fetch a single route geometry."""
    url = f"http://localhost:5000/route/v1/driving/{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10)
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
    except:
        pass
    return None


def fetch_polylines(centroids, logger=None, max_workers=10):
    """Fetch route polylines for all pairs."""
    n = len(centroids)
    total_pairs = n * n
    if logger:
        logger.info(f"Fetching polylines for {n} centroids ({total_pairs} routes)...")

    routes = []
    pairs = []

    for origin in centroids:
        for dest in centroids:
            pairs.append((origin, dest))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_route, o, d): (o, d) for o, d in pairs}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                routes.append(result)
            completed += 1
            if completed % 1000 == 0 and logger:
                logger.info(f"  Fetched {completed}/{len(pairs)} routes...")

    return routes


def save_results(city_id, city_name, centroids, matrix_data, routes, results_dir, h3_resolution):
    """Save routing results."""
    h3_indices = [c['h3_index'] for c in centroids]

    # Save matrix JSON
    result = {
        'city_id': city_id,
        'city_name': city_name,
        'h3_resolution': h3_resolution,
        'h3_indices': h3_indices,
        'centroids': centroids,
        'durations': matrix_data['durations'],
        'distances': matrix_data['distances'],
        'n_grids': len(h3_indices)
    }

    matrix_file = results_dir / f"{city_name}_{city_id}_res7_matrix.json"
    with open(matrix_file, 'w') as f:
        json.dump(result, f)

    # Save duration CSV
    duration_df = pd.DataFrame(matrix_data['durations'], index=h3_indices, columns=h3_indices)
    duration_df.to_csv(results_dir / f"{city_name}_{city_id}_res7_duration.csv")

    # Save polylines if available
    if routes:
        features = []
        for route in routes:
            features.append({
                'type': 'Feature',
                'properties': {
                    'origin_h3': route['origin_h3'],
                    'destination_h3': route['destination_h3'],
                    'duration': route['duration'],
                    'distance': route['distance']
                },
                'geometry': route['geometry']
            })

        geojson = {
            'type': 'FeatureCollection',
            'properties': {'city_id': city_id, 'total_routes': len(routes)},
            'features': features
        }

        routes_file = results_dir / f"{city_name}_{city_id}_res7_routes.geojson"
        with open(routes_file, 'w') as f:
            json.dump(geojson, f)

    return matrix_file


def route_city(city_data, osrm_dir, results_dir, fetch_polys, logger):
    """Route a single city using pre-computed H3 grids."""
    city_id = city_data['city_id']
    city_name = city_data['city_name']
    centroids = city_data['centroids']
    h3_resolution = city_data['h3_resolution']
    n_grids = len(centroids)

    logger.info(f"{'=' * 60}")
    logger.info(f"Routing: {city_name} (ID: {city_id})")
    logger.info(f"Grids: {n_grids} at resolution {h3_resolution}")
    logger.info(f"{'=' * 60}")

    # Check if already done
    matrix_file = results_dir / f"{city_name}_{city_id}_res7_matrix.json"
    routes_file = results_dir / f"{city_name}_{city_id}_res7_routes.geojson"

    if matrix_file.exists():
        if not fetch_polys or routes_file.exists():
            logger.info(f"Already done: {city_id}")
            return True

    # Decompress OSRM if needed
    city_osrm_dir = decompress_osrm(city_id, osrm_dir, logger)
    if not city_osrm_dir:
        return False

    # Start server
    max_table_size = max(n_grids + 100, 500)
    if not start_osrm_server(city_osrm_dir, city_id, max_table_size, logger):
        logger.error(f"Failed to start server for {city_id}")
        return False

    try:
        # Compute matrix
        logger.info("Computing travel time matrix...")
        start_time = time.time()
        matrix_data = compute_matrix(centroids, logger)
        elapsed = time.time() - start_time

        if not matrix_data:
            logger.error(f"Matrix computation failed for {city_id}")
            return False

        logger.info(f"Matrix computed in {elapsed:.1f}s")

        # Fetch polylines if requested
        routes = None
        if fetch_polys:
            routes = fetch_polylines(centroids, logger)
            logger.info(f"Fetched {len(routes)} route polylines")

        # Save results
        saved_file = save_results(city_id, city_name, centroids, matrix_data, routes, results_dir, h3_resolution)
        logger.info(f"Results saved: {saved_file}")

        return True

    finally:
        stop_osrm_server()


def main():
    parser = argparse.ArgumentParser(description='Route cities at H3 resolution 7')
    parser.add_argument('--grids-json', required=True, help='Path to JSON file with H3 grid centroids')
    parser.add_argument('--osrm-dir', default='~/osrm', help='Directory with OSRM files')
    parser.add_argument('--results-dir', default='~/results_res7', help='Output directory for results')
    parser.add_argument('--fetch-polylines', action='store_true', help='Fetch route polylines')
    parser.add_argument('--city-id', help='Process only this city ID')
    args = parser.parse_args()

    # Setup paths
    osrm_dir = Path(args.osrm_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging('~/route_res7.log', 'route_res7')
    logger.info("=" * 60)
    logger.info("ROUTING AT H3 RESOLUTION 7")
    logger.info(f"Grids JSON: {args.grids_json}")
    logger.info(f"OSRM directory: {osrm_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Fetch polylines: {args.fetch_polylines}")
    logger.info("=" * 60)

    # Load grids
    with open(args.grids_json, 'r') as f:
        all_city_grids = json.load(f)

    if args.city_id:
        if args.city_id not in all_city_grids:
            logger.error(f"City not found: {args.city_id}")
            return
        all_city_grids = {args.city_id: all_city_grids[args.city_id]}

    logger.info(f"Processing {len(all_city_grids)} cities")

    # Sort by number of grids (smaller first)
    sorted_cities = sorted(all_city_grids.items(), key=lambda x: x[1]['n_grids'])

    routed = 0
    failed = 0

    for city_id, city_data in sorted_cities:
        success = route_city(city_data, osrm_dir, results_dir, args.fetch_polylines, logger)
        if success:
            routed += 1
        else:
            failed += 1

        logger.info(f"Progress: {routed} routed, {failed} failed, {len(sorted_cities) - routed - failed} remaining")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ROUTING COMPLETE")
    logger.info(f"Routed: {routed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results: {results_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
