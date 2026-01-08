#!/usr/bin/env python3
"""
Phase 1: Clip OSM data for each city boundary.

Supports multi-state/province cities by:
1. Detecting which admin regions (states/provinces) overlap with each city
2. Merging OSM files from multiple regions if needed
3. Clipping from the merged file

Usage:
    python clip_cities.py --region ~/cities.geojson
    python clip_cities.py --region ~/cities.geojson --admin-bounds ~/admin_boundaries/ne_10m_admin_1_states_provinces.shp
"""

import argparse
import geopandas as gpd
import pandas as pd
from pathlib import Path

from common import (
    setup_logging, run_command, find_osm_file, ensure_dirs,
    load_admin_boundaries, find_overlapping_regions,
    get_osm_files_for_regions, merge_osm_files,
    DEFAULT_OSM_DIR, DEFAULT_CLIPPED_DIR, DEFAULT_CITIES_DIR,
    SUBNATIONAL_COUNTRIES
)


def clip_city(city_id, city_gdf, osm_file, output_dir, cities_dir, logger):
    """Clip OSM data to city boundary."""
    output_file = output_dir / f"{city_id}.osm.pbf"

    # Skip if already clipped with valid data
    if output_file.exists() and output_file.stat().st_size > 1000:
        logger.info(f"Already clipped: {city_id}")
        return True

    # Save city boundary as GeoJSON
    city_geojson = cities_dir / f"{city_id}.geojson"
    city_gdf.to_file(city_geojson, driver='GeoJSON')

    # Run osmium extract
    cmd = f"osmium extract -p {city_geojson} {osm_file} -o {output_file} --overwrite"
    success, _ = run_command(cmd, timeout=600, logger=logger)

    if success and output_file.exists() and output_file.stat().st_size > 100:
        # Clean up temporary geojson
        city_geojson.unlink(missing_ok=True)
        return True

    logger.warning(f"Clipping failed or file too small: {city_id}")
    return False


def clip_city_with_overlap(city_id, city_name, city_gdf, country_name, osm_dir,
                           output_dir, cities_dir, merge_dir, admin_gdf, logger):
    """Clip city using overlap analysis for multi-state/province cities."""
    output_file = output_dir / f"{city_id}.osm.pbf"

    # Skip if already clipped with valid data
    if output_file.exists() and output_file.stat().st_size > 1000:
        logger.info(f"Already clipped: {city_id}")
        return True

    city_geometry = city_gdf.geometry.iloc[0]

    # Find overlapping admin regions
    overlapping_regions = find_overlapping_regions(city_geometry, admin_gdf, country_name)

    if not overlapping_regions:
        logger.warning(f"No overlapping regions found for {city_name} ({city_id})")
        # Fall back to country-level file
        osm_file = find_osm_file(country_name, osm_dir)
        if osm_file:
            return clip_city(city_id, city_gdf, osm_file, output_dir, cities_dir, logger)
        return False

    logger.info(f"City {city_name} ({city_id}) overlaps: {', '.join(overlapping_regions)}")

    # Get OSM files for these regions
    osm_files = get_osm_files_for_regions(overlapping_regions, country_name, osm_dir)

    if not osm_files:
        logger.warning(f"No OSM files found for regions: {overlapping_regions}")
        # Check if we need to download missing files
        missing = []
        for region in overlapping_regions:
            from common import GEOFABRIK_NAME_MAP
            name_map = GEOFABRIK_NAME_MAP.get(country_name, {})
            geofabrik_name = name_map.get(region)
            if geofabrik_name:
                expected_file = osm_dir / f"{geofabrik_name}-latest.osm.pbf"
                if not expected_file.exists():
                    missing.append(geofabrik_name)
        if missing:
            logger.error(f"Missing OSM files - download: {', '.join(missing)}")
        return False

    logger.info(f"Using OSM files: {[f.name for f in osm_files]}")

    # Merge if multiple files
    if len(osm_files) > 1:
        # Create unique merge file name based on sorted regions
        region_key = '_'.join(sorted([r.lower().replace(' ', '-') for r in overlapping_regions]))
        merge_file = merge_dir / f"merged_{region_key}.osm.pbf"
        osm_file = merge_osm_files(osm_files, merge_file, logger)
        if not osm_file:
            logger.error(f"Failed to merge OSM files for {city_name}")
            return False
    else:
        osm_file = osm_files[0]

    # Clip from the (merged) OSM file
    return clip_city(city_id, city_gdf, osm_file, output_dir, cities_dir, logger)


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Clip OSM data for cities')
    parser.add_argument('--region', required=True, help='Path to region GeoJSON file')
    parser.add_argument('--osm-dir', default=str(DEFAULT_OSM_DIR), help='Directory with OSM files')
    parser.add_argument('--output-dir', default=str(DEFAULT_CLIPPED_DIR), help='Output directory for clipped files')
    parser.add_argument('--cities-dir', default=str(DEFAULT_CITIES_DIR), help='Temp directory for city GeoJSONs')
    parser.add_argument('--admin-bounds', default=None, help='Path to admin boundaries shapefile')
    parser.add_argument('--force', action='store_true', help='Re-clip even if output exists')
    args = parser.parse_args()

    # Setup
    osm_dir = Path(args.osm_dir)
    output_dir = Path(args.output_dir)
    cities_dir = Path(args.cities_dir)
    merge_dir = osm_dir / 'merged'
    ensure_dirs(output_dir, cities_dir, merge_dir)

    logger = setup_logging('~/clip.log', 'clip')
    logger.info("=" * 60)
    logger.info("PHASE 1: CLIPPING OSM DATA (with overlap analysis)")
    logger.info(f"Region file: {args.region}")
    logger.info(f"OSM directory: {osm_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Load admin boundaries for overlap analysis
    admin_gdf = load_admin_boundaries(args.admin_bounds)
    if admin_gdf is not None:
        logger.info(f"Loaded {len(admin_gdf)} admin regions for overlap analysis")
    else:
        logger.warning("Admin boundaries not found - will use country-level OSM files")

    # Load cities
    cities_gdf = gpd.read_file(args.region)
    logger.info(f"Loaded {len(cities_gdf)} cities")

    # If force re-clip, remove small/invalid files
    if args.force:
        for f in output_dir.glob("*.osm.pbf"):
            if f.stat().st_size < 1000:
                f.unlink()
                logger.info(f"Removed invalid file: {f.name}")

    # Group by country for efficient processing
    countries = cities_gdf.groupby('CTR_MN_NM')

    clipped = 0
    failed = 0
    skipped = 0

    for country, country_cities in countries:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Clipping {len(country_cities)} cities in {country}")
        logger.info(f"{'#' * 60}")

        # Check if this country needs subnational processing
        use_overlap = country in SUBNATIONAL_COUNTRIES and admin_gdf is not None

        if use_overlap:
            logger.info(f"Using overlap analysis for {country}")

        # Fall back to country-level OSM for countries without subnational files
        country_osm_file = None
        if not use_overlap:
            country_osm_file = find_osm_file(country, osm_dir)
            if not country_osm_file:
                logger.error(f"No OSM file found for {country} - skipping {len(country_cities)} cities")
                skipped += len(country_cities)
                continue
            logger.info(f"Using country OSM file: {country_osm_file}")

        for idx, city_row in country_cities.iterrows():
            city_id = str(city_row['ID_HDC_G0'])
            city_name = city_row.get('UC_NM_MN', city_id)
            if pd.isna(city_name):
                city_name = city_id

            city_gdf = gpd.GeoDataFrame([city_row], crs="EPSG:4326")

            if use_overlap:
                success = clip_city_with_overlap(
                    city_id, city_name, city_gdf, country, osm_dir,
                    output_dir, cities_dir, merge_dir, admin_gdf, logger
                )
            else:
                success = clip_city(city_id, city_gdf, country_osm_file,
                                   output_dir, cities_dir, logger)

            if success:
                clipped += 1
            else:
                failed += 1

            logger.info(f"Progress: {clipped} clipped, {failed} failed, {skipped} skipped")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 1 COMPLETE")
    logger.info(f"Clipped: {clipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
