#!/usr/bin/env python3
"""
OSRM Batch Processing Script (Wrapper)
Runs all 3 phases: clip → preprocess → route

For individual phase control, use:
    python clip_cities.py --region ~/cities.geojson
    python preprocess_cities.py --clipped-dir ~/clipped --profile car --compress
    python route_cities.py --region ~/cities.geojson --h3-resolution 7

Usage:
    python process_cities.py --region ~/cities.geojson
    python process_cities.py --region ~/cities.geojson --profile bicycle --h3-resolution 6
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import setup_logging, DEFAULT_CLIPPED_DIR, DEFAULT_OSRM_DIR, DEFAULT_RESULTS_DIR

def run_phase(script_name, args, logger):
    """Run a phase script with given arguments."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"{script_name} failed with code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='OSRM Batch Processing (All Phases)')
    parser.add_argument('--region', required=True, help='Path to region GeoJSON file')
    parser.add_argument('--osm-dir', help='Directory with OSM files')
    parser.add_argument('--clipped-dir', default=str(DEFAULT_CLIPPED_DIR), help='Directory for clipped OSM files')
    parser.add_argument('--osrm-dir', default=str(DEFAULT_OSRM_DIR), help='Directory for OSRM files')
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR), help='Output directory for results')
    parser.add_argument('--profile', default='car', choices=['car', 'bicycle', 'foot'], help='Routing profile')
    parser.add_argument('--h3-resolution', type=int, default=7, help='H3 grid resolution')
    parser.add_argument('--fetch-polylines', action='store_true', help='Fetch route polylines')
    parser.add_argument('--compress', action='store_true', help='Compress OSRM files after preprocessing')
    parser.add_argument('--cleanup', action='store_true', help='Delete OSRM files after routing')
    parser.add_argument('--keep-compressed', action='store_true', help='Keep compressed tar.gz when cleaning up')
    parser.add_argument('--city-id', help='Process only this city ID')
    parser.add_argument('--skip-clip', action='store_true', help='Skip clipping phase')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing phase')
    args = parser.parse_args()

    logger = setup_logging('~/processing.log', 'process')
    logger.info("=" * 60)
    logger.info("OSRM BATCH PROCESSING - ALL PHASES")
    logger.info(f"Region: {args.region}")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"H3 Resolution: {args.h3_resolution}")
    logger.info("=" * 60)

    # Phase 1: Clip
    if not args.skip_clip:
        logger.info("\n>>> PHASE 1: CLIPPING <<<\n")
        clip_args = ['--region', args.region, '--output-dir', args.clipped_dir]
        if args.osm_dir:
            clip_args.extend(['--osm-dir', args.osm_dir])

        if not run_phase('clip_cities.py', clip_args, logger):
            logger.error("Clipping phase failed!")
            sys.exit(1)
    else:
        logger.info("Skipping clip phase (--skip-clip)")

    # Phase 2: Preprocess
    if not args.skip_preprocess:
        logger.info("\n>>> PHASE 2: PREPROCESSING <<<\n")
        preprocess_args = [
            '--clipped-dir', args.clipped_dir,
            '--osrm-dir', args.osrm_dir,
            '--profile', args.profile
        ]
        if args.compress:
            preprocess_args.append('--compress')
        if args.city_id:
            preprocess_args.extend(['--city-id', args.city_id])

        if not run_phase('preprocess_cities.py', preprocess_args, logger):
            logger.error("Preprocessing phase failed!")
            sys.exit(1)
    else:
        logger.info("Skipping preprocess phase (--skip-preprocess)")

    # Phase 3: Route
    logger.info("\n>>> PHASE 3: ROUTING <<<\n")
    route_args = [
        '--region', args.region,
        '--osrm-dir', args.osrm_dir,
        '--results-dir', args.results_dir,
        '--h3-resolution', str(args.h3_resolution)
    ]
    if args.fetch_polylines:
        route_args.append('--fetch-polylines')
    if args.cleanup:
        route_args.append('--cleanup')
    if args.keep_compressed:
        route_args.append('--keep-compressed')
    if args.city_id:
        route_args.extend(['--city-id', args.city_id])

    if not run_phase('route_cities.py', route_args, logger):
        logger.error("Routing phase failed!")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("ALL PHASES COMPLETE")
    logger.info(f"Results: {args.results_dir}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
