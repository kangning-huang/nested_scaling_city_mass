#!/usr/bin/env python3
"""
Phase 3: Route computation using OSRM Table API.

Usage:
    python route_cities.py --region ~/cities.geojson
    python route_cities.py --region ~/cities.geojson --h3-resolution 6 --fetch-polylines
"""

import argparse
import geopandas as gpd
import pandas as pd
import requests
import json
import time
import tarfile
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (
    setup_logging, run_command, ensure_dirs, is_large_city,
    DEFAULT_OSRM_DIR, DEFAULT_RESULTS_DIR
)

def decompress_osrm(city_id, osrm_dir, logger):
    """Decompress OSRM tar.gz if needed."""
    tar_file = osrm_dir / f"{city_id}.tar.gz"
    city_osrm_dir = osrm_dir / city_id
    osrm_file = city_osrm_dir / f"{city_id}.osrm"

    if osrm_file.exists():
        return city_osrm_dir

    if not tar_file.exists():
        logger.error(f"Neither OSRM dir nor tar.gz found for {city_id}")
        return None

    logger.info(f"Decompressing {city_id}...")
    ensure_dirs(city_osrm_dir)

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
    success, _ = run_command(cmd, logger=logger)

    if success:
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
    subprocess.run("docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend) 2>/dev/null",
                   shell=True, capture_output=True)
    time.sleep(1)

def generate_h3_grids(city_gdf, resolution=7):
    """Generate H3 hexagonal grids for city."""
    import h3

    city_geom = city_gdf.geometry.iloc[0]
    if city_geom is None or city_geom.is_empty:
        return pd.DataFrame()

    # H3 v4 API: use polygon_to_cells instead of polyfill_geojson
    try:
        # Convert geometry to H3 cells
        h3_cells = list(h3.geo_to_cells(city_geom, resolution))
    except Exception:
        try:
            simplified = city_geom.simplify(0.001)
            h3_cells = list(h3.geo_to_cells(simplified, resolution))
        except:
            return pd.DataFrame()

    if not h3_cells:
        return pd.DataFrame()

    # H3 v4 API: use cell_to_latlng instead of h3_to_geo
    centroids = []
    for cell in h3_cells:
        lat, lon = h3.cell_to_latlng(cell)
        centroids.append({'h3_index': cell, 'lat': lat, 'lon': lon})

    return pd.DataFrame(centroids)

def compute_matrix(centroids_df, logger=None):
    """Compute travel time matrix using OSRM Table API."""
    if len(centroids_df) == 0:
        return None

    coords = ';'.join([f"{row['lon']},{row['lat']}" for _, row in centroids_df.iterrows()])
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

def fetch_polylines(centroids_df, logger=None, max_workers=10):
    """Fetch route polylines for all pairs."""
    if logger:
        logger.info(f"Fetching polylines for {len(centroids_df)} centroids ({len(centroids_df)**2} routes)...")

    routes = []
    pairs = []

    for i, origin in centroids_df.iterrows():
        for j, dest in centroids_df.iterrows():
            pairs.append((origin.to_dict(), dest.to_dict()))

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

def save_results(city_id, city_name, centroids_df, matrix_data, routes, results_dir, h3_resolution):
    """Save routing results."""
    h3_indices = centroids_df['h3_index'].tolist()

    # Save matrix JSON
    result = {
        'city_id': city_id,
        'city_name': city_name,
        'h3_resolution': h3_resolution,
        'h3_indices': h3_indices,
        'centroids': centroids_df.to_dict('records'),
        'durations': matrix_data['durations'],
        'distances': matrix_data['distances'],
        'n_grids': len(h3_indices)
    }

    with open(results_dir / f"{city_id}_matrix.json", 'w') as f:
        json.dump(result, f)

    # Save duration CSV
    duration_df = pd.DataFrame(matrix_data['durations'], index=h3_indices, columns=h3_indices)
    duration_df.to_csv(results_dir / f"{city_id}_duration.csv")

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

        with open(results_dir / f"{city_id}_routes.geojson", 'w') as f:
            json.dump(geojson, f)

def cleanup_osrm(city_id, osrm_dir, keep_compressed, logger):
    """Clean up OSRM files after routing."""
    city_osrm_dir = osrm_dir / city_id
    tar_file = osrm_dir / f"{city_id}.tar.gz"

    if city_osrm_dir.exists():
        if keep_compressed and not tar_file.exists():
            # Re-compress before deleting
            logger.info(f"Re-compressing {city_id}...")
            try:
                with tarfile.open(tar_file, "w:gz") as tar:
                    for f in city_osrm_dir.glob(f"{city_id}.osrm*"):
                        tar.add(f, arcname=f.name)
            except Exception as e:
                logger.warning(f"Re-compression failed: {e}")

        shutil.rmtree(city_osrm_dir, ignore_errors=True)

def route_city(city_row, osrm_dir, results_dir, h3_resolution, fetch_polys, cleanup, keep_compressed, logger):
    """Route a single city."""
    city_id = str(city_row['ID_HDC_G0'])
    city_name = city_row.get('UC_NM_MN', city_id)
    if pd.isna(city_name):
        city_name = city_id

    logger.info(f"{'=' * 60}")
    logger.info(f"Routing: {city_name} (ID: {city_id})")
    logger.info(f"{'=' * 60}")

    # Check if already done
    matrix_file = results_dir / f"{city_id}_matrix.json"
    routes_file = results_dir / f"{city_id}_routes.geojson"

    if matrix_file.exists():
        if not fetch_polys or routes_file.exists():
            logger.info(f"Already done: {city_id}")
            return True

    # Decompress OSRM if needed
    city_osrm_dir = decompress_osrm(city_id, osrm_dir, logger)
    if not city_osrm_dir:
        return False

    # Generate H3 grids
    city_gdf = gpd.GeoDataFrame([city_row], crs="EPSG:4326")
    centroids_df = generate_h3_grids(city_gdf, h3_resolution)
    n_grids = len(centroids_df)
    logger.info(f"Generated {n_grids} H3 grids (resolution {h3_resolution})")

    if n_grids == 0:
        logger.warning(f"No grids for {city_id}")
        if cleanup:
            cleanup_osrm(city_id, osrm_dir, keep_compressed, logger)
        return False

    # Start server
    max_table_size = max(n_grids + 100, 500)
    if not start_osrm_server(city_osrm_dir, city_id, max_table_size, logger):
        logger.error(f"Failed to start server for {city_id}")
        if cleanup:
            cleanup_osrm(city_id, osrm_dir, keep_compressed, logger)
        return False

    try:
        # Compute matrix
        logger.info("Computing travel time matrix...")
        matrix_data = compute_matrix(centroids_df, logger)

        if not matrix_data:
            logger.error(f"Matrix computation failed for {city_id}")
            return False

        # Fetch polylines if requested
        routes = None
        if fetch_polys:
            routes = fetch_polylines(centroids_df, logger)
            logger.info(f"Fetched {len(routes)} route polylines")

        # Save results
        save_results(city_id, city_name, centroids_df, matrix_data, routes, results_dir, h3_resolution)
        logger.info(f"Results saved for {city_id}")

        return True

    finally:
        stop_osrm_server()
        if cleanup:
            cleanup_osrm(city_id, osrm_dir, keep_compressed, logger)

def main():
    parser = argparse.ArgumentParser(description='Phase 3: Route computation')
    parser.add_argument('--region', required=True, help='Path to region GeoJSON file')
    parser.add_argument('--osrm-dir', default=str(DEFAULT_OSRM_DIR), help='Directory with OSRM files')
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR), help='Output directory for results')
    parser.add_argument('--h3-resolution', type=int, default=7, help='H3 grid resolution (default: 7)')
    parser.add_argument('--fetch-polylines', action='store_true', help='Fetch route polylines')
    parser.add_argument('--cleanup', action='store_true', help='Delete OSRM files after routing')
    parser.add_argument('--keep-compressed', action='store_true', help='Keep compressed tar.gz when cleaning up')
    parser.add_argument('--city-id', help='Process only this city ID')
    args = parser.parse_args()

    # Setup
    osrm_dir = Path(args.osrm_dir)
    results_dir = Path(args.results_dir)
    ensure_dirs(results_dir)

    logger = setup_logging('~/route.log', 'route')
    logger.info("=" * 60)
    logger.info("PHASE 3: ROUTE COMPUTATION")
    logger.info(f"Region file: {args.region}")
    logger.info(f"OSRM directory: {osrm_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"H3 resolution: {args.h3_resolution}")
    logger.info(f"Fetch polylines: {args.fetch_polylines}")
    logger.info(f"Cleanup: {args.cleanup}")
    logger.info("=" * 60)

    # Load cities
    cities_gdf = gpd.read_file(args.region)

    if args.city_id:
        cities_gdf = cities_gdf[cities_gdf['ID_HDC_G0'].astype(str) == args.city_id]
        if len(cities_gdf) == 0:
            logger.error(f"City not found: {args.city_id}")
            return

    logger.info(f"Processing {len(cities_gdf)} cities")

    # Sort by area (smaller first)
    try:
        cities_gdf['area_km2'] = cities_gdf.to_crs('EPSG:3857').geometry.area / 1e6
        cities_gdf = cities_gdf.sort_values('area_km2')
    except:
        pass

    routed = 0
    failed = 0

    for idx, city_row in cities_gdf.iterrows():
        success = route_city(
            city_row, osrm_dir, results_dir,
            args.h3_resolution, args.fetch_polylines,
            args.cleanup, args.keep_compressed, logger
        )
        if success:
            routed += 1
        else:
            failed += 1

        logger.info(f"Progress: {routed} routed, {failed} failed, {len(cities_gdf) - routed - failed} remaining")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 3 COMPLETE")
    logger.info(f"Routed: {routed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results: {results_dir}")
    logger.info(f"{'=' * 60}")

if __name__ == '__main__':
    main()
