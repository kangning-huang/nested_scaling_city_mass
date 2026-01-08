#!/usr/bin/env python3
"""
OSRM Route Polyline Fetching Script (Pass 2)

Fetches route geometries for all pairwise routes using the Route API.
Run this after process_cities.py has computed the travel time matrices.

Features:
- Parallel fetching with 8 concurrent workers
- Automatic line simplification (Douglas-Peucker, ~11m tolerance)
- Coordinate rounding to 5 decimal places (~1.1m precision)
- Reduces output file size by ~90% (e.g., 2.3GB → 223MB)

Usage:
    REGION_FILE=~/cities.geojson python3 fetch_polylines.py

Requirements:
    pip install requests shapely geopandas
"""

import json
import os
import sys
import time
import logging
import subprocess
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from shapely.geometry import LineString
from shapely import simplify as shapely_simplify

# Configuration
REGION_FILE = os.environ.get('REGION_FILE', os.path.expanduser('~/cities.geojson'))
OSM_DIR = Path(os.path.expanduser('~/osrm-data'))
CITIES_DIR = Path(os.path.expanduser('~/cities'))
RESULTS_DIR = Path(os.path.expanduser('~/results'))
MAX_WORKERS = 8  # Concurrent Route API requests (reduced to avoid overwhelming OSRM)

# Line simplification settings
SIMPLIFY_TOLERANCE = 0.0001  # ~11 meters (in degrees)
COORD_PRECISION = 5  # 5 decimal places = ~1.1m precision

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser('~/polylines.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_command(cmd, timeout=3600):
    """Run shell command with timeout."""
    logger.debug(f"Running: {cmd[:100]}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"Command error: {e}")
        return False


def find_osm_file(country_name):
    """Find the appropriate OSM file for a country."""
    country_lower = country_name.lower().replace(' ', '-').replace('_', '-')

    name_mappings = {
        'united-states': ['us', 'usa', 'united-states'],
        'united-kingdom': ['great-britain', 'uk', 'united-kingdom'],
        'democratic-republic-of-the-congo': ['congo-democratic-republic', 'drc'],
        'republic-of-the-congo': ['congo-brazzaville'],
    }

    search_names = [country_lower]
    for key, aliases in name_mappings.items():
        if country_lower in aliases or country_lower == key:
            search_names.extend(aliases)
            search_names.append(key)

    for osm_file in OSM_DIR.glob("*.osm.pbf"):
        file_lower = osm_file.stem.lower().replace('-latest', '')
        for name in search_names:
            if name in file_lower or file_lower in name:
                return osm_file

    region_files = list(OSM_DIR.glob("*-latest.osm.pbf"))
    if region_files:
        region_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        return region_files[0]

    return None


def ensure_osrm_files(city_id, country_name):
    """Ensure OSRM files exist, re-creating if necessary."""
    osrm_file = CITIES_DIR / f"{city_id}.osrm"

    if osrm_file.exists():
        logger.info(f"OSRM files exist for {city_id}")
        return True

    # Check if we have the clipped OSM file
    osm_pbf = CITIES_DIR / f"{city_id}.osm.pbf"
    if not osm_pbf.exists():
        logger.warning(f"No OSM file for {city_id}, need to re-clip")

        # Need to re-clip from source OSM
        osm_file = find_osm_file(country_name)
        if not osm_file:
            logger.error(f"No source OSM file found for {country_name}")
            return False

        # Need GeoJSON boundary - try to reconstruct from matrix file
        matrix_file = RESULTS_DIR / f"{city_id}_matrix.json"
        if not matrix_file.exists():
            logger.error(f"No matrix file for {city_id}")
            return False

        # We can't easily reconstruct the boundary, so skip this city
        logger.error(f"Cannot re-clip {city_id} - boundary not available")
        return False

    # Run OSRM preprocessing
    logger.info(f"Running OSRM preprocessing for {city_id}...")

    # Extract
    cmd = f'cd {CITIES_DIR} && docker run --rm -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/{city_id}.osm.pbf'
    if not run_command(cmd, timeout=1800):
        return False

    # Partition
    cmd = f'cd {CITIES_DIR} && docker run --rm -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-partition /data/{city_id}.osrm'
    if not run_command(cmd, timeout=1800):
        return False

    # Customize
    cmd = f'cd {CITIES_DIR} && docker run --rm -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-customize /data/{city_id}.osrm'
    if not run_command(cmd, timeout=1800):
        return False

    return True


def start_osrm_server(city_id):
    """Start OSRM routing server."""
    stop_osrm_server()
    time.sleep(2)

    cmd = f'cd {CITIES_DIR} && docker run --rm -d -p 5000:5000 -v "${{PWD}}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/{city_id}.osrm'
    if run_command(cmd):
        # Wait for server to be ready
        for i in range(30):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:5000/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
        time.sleep(5)
        return True
    return False


def stop_osrm_server():
    """Stop OSRM routing server."""
    subprocess.run("docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend) 2>/dev/null", shell=True)
    time.sleep(1)


def simplify_geometry(geometry):
    """Simplify a GeoJSON geometry using Douglas-Peucker algorithm.

    Reduces ~90% of coordinates while preserving route shape.
    - Tolerance of 0.0001 degrees ≈ 11 meters
    - Rounds coordinates to 5 decimal places (~1.1m precision)
    """
    coords = geometry.get('coordinates', [])
    if len(coords) < 2:
        return geometry

    try:
        line = LineString(coords)
        simplified = shapely_simplify(line, tolerance=SIMPLIFY_TOLERANCE, preserve_topology=True)

        # Round coordinates to reduce file size
        new_coords = [
            [round(c[0], COORD_PRECISION), round(c[1], COORD_PRECISION)]
            for c in simplified.coords
        ]

        return {
            'type': 'LineString',
            'coordinates': new_coords
        }
    except Exception:
        # If simplification fails, return original with rounded coords
        return {
            'type': 'LineString',
            'coordinates': [[round(c[0], COORD_PRECISION), round(c[1], COORD_PRECISION)] for c in coords]
        }


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

        if data.get('code') != 'Ok':
            return None

        route = data['routes'][0]

        # Simplify geometry to reduce file size by ~90%
        simplified_geom = simplify_geometry(route['geometry'])

        return {
            'geometry': simplified_geom,
            'duration': route['duration'],
            'distance': route['distance']
        }
    except Exception as e:
        return None


def fetch_all_routes(centroids, max_workers=MAX_WORKERS):
    """Fetch all pairwise routes using parallel requests."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    n = len(centroids)
    total_pairs = n * n

    # Generate all O-D pairs
    pairs = []
    for i, origin in enumerate(centroids):
        for j, destination in enumerate(centroids):
            pairs.append((i, j, origin, destination))

    logger.info(f"Fetching {total_pairs} routes with {max_workers} workers...")

    features = []
    failed = []
    completed = 0

    # Create session with proper connection pool size
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max_workers + 5,
        pool_maxsize=max_workers + 5,
        max_retries=Retry(total=3, backoff_factor=0.1)
    )
    session.mount('http://', adapter)

    def fetch_pair(pair):
        i, j, origin, dest = pair
        result = fetch_single_route(session, origin, dest)
        if result:
            return {
                'type': 'Feature',
                'properties': {
                    'origin_h3': origin['h3_index'],
                    'destination_h3': dest['h3_index'],
                    'origin_idx': i,
                    'destination_idx': j,
                    'duration': result['duration'],
                    'distance': result['distance']
                },
                'geometry': result['geometry']
            }
        else:
            return {'failed': True, 'origin': origin['h3_index'], 'dest': dest['h3_index']}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(fetch_pair, pair): pair for pair in pairs}

        for future in as_completed(future_to_pair):
            completed += 1
            result = future.result()

            if result.get('failed'):
                failed.append((result['origin'], result['dest']))
            else:
                features.append(result)

            if completed % 500 == 0 or completed == total_pairs:
                logger.info(f"Progress: {completed}/{total_pairs} ({100*completed/total_pairs:.1f}%)")

    session.close()

    logger.info(f"Completed: {len(features)} routes, {len(failed)} failed")

    return features, failed


def save_routes_geojson(city_id, features, failed):
    """Save routes as GeoJSON FeatureCollection."""
    geojson = {
        'type': 'FeatureCollection',
        'properties': {
            'city_id': city_id,
            'total_routes': len(features),
            'failed_routes': len(failed)
        },
        'features': features
    }

    output_file = RESULTS_DIR / f"{city_id}_routes.geojson"
    with open(output_file, 'w') as f:
        json.dump(geojson, f)

    # Save failed routes list if any
    if failed:
        failed_file = RESULTS_DIR / f"{city_id}_failed_routes.json"
        with open(failed_file, 'w') as f:
            json.dump(failed, f)

    logger.info(f"Saved routes to {output_file}")


def cleanup_osrm_files(city_id):
    """Remove OSRM files to save space."""
    for f in CITIES_DIR.glob(f"{city_id}.osrm*"):
        try:
            f.unlink()
        except:
            pass
    logger.debug(f"Cleaned up OSRM files for {city_id}")


def process_city_routes(city_id, country_name):
    """Process routes for a single city."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Processing routes for city {city_id}")
    logger.info(f"{'=' * 60}")

    # Check if already processed
    routes_file = RESULTS_DIR / f"{city_id}_routes.geojson"
    if routes_file.exists():
        logger.info(f"Routes already exist for {city_id}, skipping")
        return True

    # Load matrix file to get centroids
    matrix_file = RESULTS_DIR / f"{city_id}_matrix.json"
    if not matrix_file.exists():
        logger.error(f"No matrix file for {city_id}")
        return False

    try:
        with open(matrix_file) as f:
            matrix_data = json.load(f)

        centroids = matrix_data['centroids']
        n_grids = len(centroids)
        logger.info(f"Loaded {n_grids} centroids from matrix file")

        if n_grids == 0:
            logger.warning(f"No centroids for {city_id}")
            return False

        # Ensure OSRM files exist
        if not ensure_osrm_files(city_id, country_name):
            logger.error(f"Failed to ensure OSRM files for {city_id}")
            return False

        # Start OSRM server
        if not start_osrm_server(city_id):
            logger.error(f"Failed to start OSRM server for {city_id}")
            return False

        # Fetch all routes
        features, failed = fetch_all_routes(centroids)

        # Stop server
        stop_osrm_server()

        # Save results
        save_routes_geojson(city_id, features, failed)

        # Cleanup OSRM files to save space
        cleanup_osrm_files(city_id)

        return True

    except Exception as e:
        logger.error(f"Error processing {city_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        stop_osrm_server()
        return False


def main():
    """Main processing loop."""
    logger.info("=" * 60)
    logger.info("OSRM POLYLINE FETCHING STARTED")
    logger.info("=" * 60)

    # Find all completed matrix files
    matrix_files = list(RESULTS_DIR.glob("*_matrix.json"))
    logger.info(f"Found {len(matrix_files)} completed matrix files")

    if not matrix_files:
        logger.error("No matrix files found. Run process_cities.py first.")
        sys.exit(1)

    # Load city metadata if region file exists
    city_countries = {}
    if Path(REGION_FILE).exists():
        try:
            import geopandas as gpd
            cities_gdf = gpd.read_file(REGION_FILE)
            for _, row in cities_gdf.iterrows():
                city_id = str(row['ID_HDC_G0'])
                city_countries[city_id] = row.get('CTR_MN_NM', 'Unknown')
        except Exception as e:
            logger.warning(f"Could not load region file: {e}")

    processed = 0
    failed = 0
    skipped = 0

    for matrix_file in sorted(matrix_files):
        city_id = matrix_file.stem.replace('_matrix', '')
        country_name = city_countries.get(city_id, 'Unknown')

        # Check if routes already exist
        routes_file = RESULTS_DIR / f"{city_id}_routes.geojson"
        if routes_file.exists():
            skipped += 1
            continue

        success = process_city_routes(city_id, country_name)
        if success:
            processed += 1
        else:
            failed += 1

        logger.info(f"Progress: {processed} processed, {failed} failed, {skipped} skipped")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"POLYLINE FETCHING COMPLETE")
    logger.info(f"Processed: {processed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
