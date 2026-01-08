#!/usr/bin/env python3
"""
Phase 2: OSRM preprocessing (extract, partition, customize).

Usage:
    python preprocess_cities.py --clipped-dir ~/clipped
    python preprocess_cities.py --clipped-dir ~/clipped --profile bicycle --compress
"""

import argparse
import tarfile
import shutil
from pathlib import Path

from common import (
    setup_logging, run_command, ensure_dirs,
    DEFAULT_CLIPPED_DIR, DEFAULT_OSRM_DIR, DEFAULT_CITIES_DIR,
    LARGE_CITY_THRESHOLD_KM2
)

AVAILABLE_PROFILES = ['car', 'bicycle', 'foot']

def preprocess_city(city_id, clipped_file, osrm_dir, cities_dir, profile, logger):
    """Run OSRM preprocessing for a city."""
    city_osrm_dir = osrm_dir / city_id
    osrm_file = city_osrm_dir / f"{city_id}.osrm"

    # Skip if already preprocessed
    if osrm_file.exists():
        logger.info(f"Already preprocessed: {city_id}")
        return True

    # Create city OSRM directory
    ensure_dirs(city_osrm_dir)

    # Copy clipped OSM to city dir (OSRM outputs files next to input)
    city_osm = city_osrm_dir / f"{city_id}.osm.pbf"
    if not city_osm.exists():
        shutil.copy(clipped_file, city_osm)

    profile_path = f"/opt/{profile}.lua"

    # Extract
    cmd = f'cd {city_osrm_dir} && docker run --rm -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-extract -p {profile_path} /data/{city_id}.osm.pbf'
    success, _ = run_command(cmd, timeout=1800, logger=logger)
    if not success:
        logger.error(f"Extract failed for {city_id}")
        return False

    # Partition
    cmd = f'cd {city_osrm_dir} && docker run --rm -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-partition /data/{city_id}.osrm'
    success, _ = run_command(cmd, timeout=1800, logger=logger)
    if not success:
        logger.error(f"Partition failed for {city_id}")
        return False

    # Customize
    cmd = f'cd {city_osrm_dir} && docker run --rm -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-customize /data/{city_id}.osrm'
    success, _ = run_command(cmd, timeout=1800, logger=logger)
    if not success:
        logger.error(f"Customize failed for {city_id}")
        return False

    # Remove the copied OSM file to save space
    city_osm.unlink(missing_ok=True)

    return True

def compress_osrm(city_id, osrm_dir, logger):
    """Compress OSRM files to tar.gz."""
    city_osrm_dir = osrm_dir / city_id
    tar_file = osrm_dir / f"{city_id}.tar.gz"

    if tar_file.exists():
        logger.info(f"Already compressed: {city_id}")
        return True

    if not city_osrm_dir.exists():
        logger.error(f"OSRM directory not found: {city_osrm_dir}")
        return False

    logger.info(f"Compressing {city_id}...")
    try:
        with tarfile.open(tar_file, "w:gz") as tar:
            for f in city_osrm_dir.glob(f"{city_id}.osrm*"):
                tar.add(f, arcname=f.name)

        # Remove uncompressed files
        shutil.rmtree(city_osrm_dir)
        logger.info(f"Compressed: {tar_file} ({tar_file.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return False

def get_clipped_osm_size_mb(clipped_file):
    """Get size of clipped OSM in MB (proxy for city size)."""
    return clipped_file.stat().st_size / 1024 / 1024

def main():
    parser = argparse.ArgumentParser(description='Phase 2: OSRM preprocessing')
    parser.add_argument('--clipped-dir', default=str(DEFAULT_CLIPPED_DIR), help='Directory with clipped OSM files')
    parser.add_argument('--osrm-dir', default=str(DEFAULT_OSRM_DIR), help='Output directory for OSRM files')
    parser.add_argument('--cities-dir', default=str(DEFAULT_CITIES_DIR), help='Temp directory')
    parser.add_argument('--profile', default='car', choices=AVAILABLE_PROFILES, help='Routing profile')
    parser.add_argument('--compress', action='store_true', help='Compress OSRM files after preprocessing')
    parser.add_argument('--compress-large-only', action='store_true', help='Only compress large cities (>10MB OSM)')
    parser.add_argument('--city-id', help='Process only this city ID')
    args = parser.parse_args()

    # Setup
    clipped_dir = Path(args.clipped_dir)
    osrm_dir = Path(args.osrm_dir)
    cities_dir = Path(args.cities_dir)
    ensure_dirs(osrm_dir, cities_dir)

    logger = setup_logging('~/preprocess.log', 'preprocess')
    logger.info("=" * 60)
    logger.info("PHASE 2: OSRM PREPROCESSING")
    logger.info(f"Clipped directory: {clipped_dir}")
    logger.info(f"OSRM directory: {osrm_dir}")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Compress: {args.compress or args.compress_large_only}")
    logger.info("=" * 60)

    # Find all clipped OSM files
    if args.city_id:
        clipped_files = [clipped_dir / f"{args.city_id}.osm.pbf"]
        if not clipped_files[0].exists():
            logger.error(f"Clipped file not found: {clipped_files[0]}")
            return
    else:
        clipped_files = sorted(clipped_dir.glob("*.osm.pbf"))

    logger.info(f"Found {len(clipped_files)} clipped OSM files")

    # Sort by size (process smaller cities first)
    clipped_files = sorted(clipped_files, key=lambda f: f.stat().st_size)

    preprocessed = 0
    compressed = 0
    failed = 0

    for clipped_file in clipped_files:
        city_id = clipped_file.stem.replace('.osm', '')
        osm_size_mb = get_clipped_osm_size_mb(clipped_file)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {city_id} ({osm_size_mb:.1f} MB)")
        logger.info(f"{'=' * 60}")

        # Check if already done (compressed or uncompressed)
        tar_file = osrm_dir / f"{city_id}.tar.gz"
        osrm_file = osrm_dir / city_id / f"{city_id}.osrm"

        if tar_file.exists():
            logger.info(f"Already done (compressed): {city_id}")
            preprocessed += 1
            compressed += 1
            continue

        if osrm_file.exists():
            logger.info(f"Already preprocessed: {city_id}")
            preprocessed += 1
            # Compress if requested
            if args.compress or (args.compress_large_only and osm_size_mb > 10):
                if compress_osrm(city_id, osrm_dir, logger):
                    compressed += 1
            continue

        # Preprocess
        success = preprocess_city(city_id, clipped_file, osrm_dir, cities_dir, args.profile, logger)
        if success:
            preprocessed += 1

            # Compress if requested
            should_compress = args.compress or (args.compress_large_only and osm_size_mb > 10)
            if should_compress:
                if compress_osrm(city_id, osrm_dir, logger):
                    compressed += 1
        else:
            failed += 1

        logger.info(f"Progress: {preprocessed} preprocessed, {compressed} compressed, {failed} failed")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 2 COMPLETE")
    logger.info(f"Preprocessed: {preprocessed}")
    logger.info(f"Compressed: {compressed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output: {osrm_dir}")
    logger.info(f"{'=' * 60}")

if __name__ == '__main__':
    main()
