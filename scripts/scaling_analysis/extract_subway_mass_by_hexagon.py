#!/usr/bin/env python3
"""
Extract Subway Material Mass by H3 Hexagon (Optimized v2)

This script calculates subway infrastructure material mass per H3 hexagon for
resolutions 5, 6, and 7 by combining:
- Chinese metro data (CPTOND-2025)
- Global OSM subway data
- Material intensity coefficients from Chinese subway construction studies

Optimizations:
- Use spatial join instead of unary_union for faster city filtering
- First filter cities that have subway infrastructure
- Only process hexagons that intersect with subway data

Output:
- results/subway_mass_by_hexagon_resolution{5,6,7}.csv
- Columns: city_name, city_id, h3index, n_stations, total_length_km, subway_mass_tonnes

Author: Claude Code
Date: 2026-02-03
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Set

# Unbuffered output for progress monitoring
sys.stdout.reconfigure(line_buffering=True)

import geopandas as gpd
import pandas as pd
from pyproj import Geod
from shapely.ops import unary_union

# Suppress warnings
warnings.filterwarnings('ignore')

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Material intensity coefficients file
MI_FILE = DATA_DIR / "subway_networks" / "material_intensity_coefficients_china.csv"

# Subway data paths
CPTOND_DIR = DATA_DIR / "subway_networks" / "CPTOND-2025" / "dataset" / "metro" / "shapefiles"
OSM_DIR = DATA_DIR / "subway_networks" / "OSM"
OSM_SUMMARY = OSM_DIR / "osm_download_summary.csv"

# City boundaries
CITY_BOUNDARIES_FILE = DATA_DIR / "all_cities_boundaries.gpkg"

# H3 grid files
H3_GRIDS = {
    5: DATA_DIR / "all_cities_h3_grids_resolution5.gpkg",
    6: DATA_DIR / "all_cities_h3_grids_resolution6.gpkg",
    7: DATA_DIR / "all_cities_h3_grids_resolution7.gpkg",
}

# Cities covered by CPTOND (skip in OSM to avoid double counting)
CPTOND_CITIES_TO_SKIP_OSM = {
    'hong_kong', 'taipei', 'kaohsiung'
}


def load_material_intensities(mi_file: Path) -> tuple[float, float]:
    """Load material intensity coefficients from CSV."""
    df = pd.read_csv(mi_file)
    tunnel_mi = df[df['unit'] == 't_per_km']['tons'].sum()
    station_mi = df[df['unit'] == 't_per_station']['tons'].sum()

    print(f"Material Intensities:")
    print(f"  Tunnel: {tunnel_mi:,.1f} tonnes/km")
    print(f"  Station: {station_mi:,.1f} tonnes/station")

    return tunnel_mi, station_mi


def load_cptond_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load CPTOND Chinese metro data."""
    print("\nLoading CPTOND (Chinese metro) data...")

    # Load and deduplicate stations
    stops_file = CPTOND_DIR / "metro_stops.shp"
    stations = gpd.read_file(stops_file)
    stations = stations.drop_duplicates(subset=['stop_id'])
    stations['source'] = 'CPTOND'
    stations['city'] = stations['city_en']
    print(f"  {len(stations):,} unique stations from {stations['city'].nunique()} Chinese cities")

    # Load routes and merge by city
    routes_file = CPTOND_DIR / "metro_routes.shp"
    routes = gpd.read_file(routes_file)

    lines_list = []
    for city, city_routes in routes.groupby('city_en'):
        merged_geom = unary_union(city_routes.geometry.values)
        lines_list.append({
            'city': city,
            'source': 'CPTOND',
            'geometry': merged_geom
        })
    lines = gpd.GeoDataFrame(lines_list, crs=routes.crs)

    return stations, lines


def load_osm_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load OSM subway data for global cities."""
    print("\nLoading OSM (global) subway data...")

    osm_summary = pd.read_csv(OSM_SUMMARY)
    all_stations = []
    all_lines = []

    for _, row in osm_summary.iterrows():
        slug = row['slug']
        city = row['city']

        if slug in CPTOND_CITIES_TO_SKIP_OSM:
            continue

        stations_file = OSM_DIR / f"{slug}_stations.geojson"
        lines_file = OSM_DIR / f"{slug}_lines.geojson"

        if stations_file.exists():
            try:
                st = gpd.read_file(stations_file)
                if len(st) > 0:
                    st['city'] = city
                    st['source'] = 'OSM'
                    all_stations.append(st[['city', 'source', 'geometry']])
            except Exception:
                pass

        if lines_file.exists():
            try:
                ln = gpd.read_file(lines_file)
                if len(ln) > 0:
                    merged_geom = unary_union(ln.geometry.values)
                    all_lines.append({
                        'city': city,
                        'source': 'OSM',
                        'geometry': merged_geom
                    })
            except Exception:
                pass

    if all_stations:
        stations = pd.concat(all_stations, ignore_index=True)
        stations = gpd.GeoDataFrame(stations, crs="EPSG:4326")
    else:
        stations = gpd.GeoDataFrame(columns=['city', 'source', 'geometry'], crs="EPSG:4326")

    if all_lines:
        lines = gpd.GeoDataFrame(all_lines, crs="EPSG:4326")
    else:
        lines = gpd.GeoDataFrame(columns=['city', 'source', 'geometry'], crs="EPSG:4326")

    print(f"  {len(stations):,} stations from {len(all_lines)} OSM cities")

    return stations, lines


def calculate_geodesic_length_km(geometry) -> float:
    """Calculate geodesic length of a geometry in kilometers."""
    if geometry is None or geometry.is_empty:
        return 0.0

    geod = Geod(ellps="WGS84")
    try:
        return geod.geometry_length(geometry) / 1000.0
    except Exception:
        return 0.0


def find_cities_with_subway(
    city_boundaries: gpd.GeoDataFrame,
    all_stations: gpd.GeoDataFrame,
    all_lines: gpd.GeoDataFrame
) -> Set[int]:
    """Find city IDs that have subway infrastructure using spatial join."""
    print("\nIdentifying cities with subway infrastructure...")

    city_ids = set()

    # Use spatial join to find cities containing stations
    print("  Checking stations...")
    if len(all_stations) > 0:
        stations_in_cities = gpd.sjoin(
            all_stations[['geometry']],
            city_boundaries[['ID_HDC_G0', 'geometry']],
            how='inner',
            predicate='within'
        )
        city_ids.update(stations_in_cities['ID_HDC_G0'].unique())
        print(f"    Found stations in {len(city_ids)} cities")

    # Use spatial join to find cities intersecting with lines
    print("  Checking lines...")
    if len(all_lines) > 0:
        lines_in_cities = gpd.sjoin(
            all_lines[['geometry']],
            city_boundaries[['ID_HDC_G0', 'geometry']],
            how='inner',
            predicate='intersects'
        )
        new_cities = set(lines_in_cities['ID_HDC_G0'].unique()) - city_ids
        city_ids.update(new_cities)
        print(f"    Found lines in {len(new_cities)} additional cities")

    print(f"  Total: {len(city_ids):,} cities with subway infrastructure")
    return city_ids


def process_single_resolution(
    resolution: int,
    hexagons: gpd.GeoDataFrame,
    all_stations: gpd.GeoDataFrame,
    all_lines: gpd.GeoDataFrame,
    tunnel_mi: float,
    station_mi: float
) -> pd.DataFrame:
    """
    Process subway data for a specific H3 resolution.
    """
    print(f"\nProcessing Resolution {resolution}...")

    # Ensure same CRS
    if hexagons.crs is None:
        hexagons = hexagons.set_crs("EPSG:4326")
    elif hexagons.crs.to_epsg() != 4326:
        hexagons = hexagons.to_crs("EPSG:4326")

    # --- Count stations per hexagon ---
    print("  Counting stations per hexagon...")

    stations_with_hex = gpd.sjoin(
        all_stations[['geometry']],
        hexagons[['h3index', 'geometry']],
        how='inner',
        predicate='within'
    )
    station_counts = stations_with_hex.groupby('h3index').size().reset_index(name='n_stations')
    print(f"    Found stations in {len(station_counts):,} hexagons")

    # --- Find hexagons that intersect with lines ---
    print("  Finding hexagons that intersect with lines...")

    if len(all_lines) > 0:
        lines_hex = gpd.sjoin(
            hexagons[['h3index', 'geometry']],
            all_lines[['geometry']],
            how='inner',
            predicate='intersects'
        )
        hex_with_lines = lines_hex['h3index'].unique()
        hexagons_for_clipping = hexagons[hexagons['h3index'].isin(hex_with_lines)].copy()
        print(f"    {len(hexagons_for_clipping):,} hexagons intersect with lines")

        # --- Calculate line length per hexagon ---
        print("  Calculating line length per hexagon...")

        if len(hexagons_for_clipping) > 0:
            # Clip lines to hexagon boundaries
            clipped_lines = gpd.overlay(
                all_lines[['geometry']],
                hexagons_for_clipping[['h3index', 'geometry']],
                how='intersection'
            )

            if len(clipped_lines) > 0:
                # Dissolve by h3index to merge overlapping segments
                clipped_lines = clipped_lines.dissolve(by='h3index').reset_index()
                clipped_lines['total_length_km'] = clipped_lines.geometry.apply(calculate_geodesic_length_km)
                line_lengths = clipped_lines[['h3index', 'total_length_km']]
                print(f"    Calculated lengths for {len(line_lengths):,} hexagons")
            else:
                line_lengths = pd.DataFrame(columns=['h3index', 'total_length_km'])
        else:
            line_lengths = pd.DataFrame(columns=['h3index', 'total_length_km'])
    else:
        line_lengths = pd.DataFrame(columns=['h3index', 'total_length_km'])

    # --- Merge station counts and line lengths ---
    print("  Merging results...")

    all_hex_indices = set(station_counts['h3index'].tolist() + line_lengths['h3index'].tolist())

    if not all_hex_indices:
        return pd.DataFrame(columns=[
            'city_name', 'city_id', 'h3index', 'n_stations',
            'total_length_km', 'subway_mass_tonnes'
        ])

    result = pd.DataFrame({'h3index': list(all_hex_indices)})

    result = result.merge(station_counts, on='h3index', how='left')
    result['n_stations'] = result['n_stations'].fillna(0).astype(int)

    result = result.merge(line_lengths, on='h3index', how='left')
    result['total_length_km'] = result['total_length_km'].fillna(0.0)

    # Calculate subway mass
    result['subway_mass_tonnes'] = (
        result['total_length_km'] * tunnel_mi +
        result['n_stations'] * station_mi
    )

    # Add city metadata
    hex_metadata = hexagons[['h3index', 'ID_HDC_G0', 'UC_NM_MN']].drop_duplicates(subset=['h3index'])
    result = result.merge(hex_metadata, on='h3index', how='left')
    result = result.rename(columns={'ID_HDC_G0': 'city_id', 'UC_NM_MN': 'city_name'})

    # Filter and order columns
    result = result[(result['n_stations'] > 0) | (result['total_length_km'] > 0)]
    result = result[[
        'city_name', 'city_id', 'h3index', 'n_stations',
        'total_length_km', 'subway_mass_tonnes'
    ]]
    result = result.sort_values(['city_id', 'h3index']).reset_index(drop=True)

    # Summary
    print(f"\n  Resolution {resolution} Summary:")
    print(f"    Hexagons with subway: {len(result):,}")
    print(f"    Total stations: {result['n_stations'].sum():,}")
    print(f"    Total line length: {result['total_length_km'].sum():,.1f} km")
    print(f"    Total mass: {result['subway_mass_tonnes'].sum() / 1e6:.2f} Mt")

    return result


def main():
    """Main execution function."""
    print("="*60)
    print("Subway Material Mass Extraction by H3 Hexagon (Optimized v2)")
    print("="*60)

    # Load material intensities
    tunnel_mi, station_mi = load_material_intensities(MI_FILE)

    # Load subway data
    cptond_stations, cptond_lines = load_cptond_data()
    osm_stations, osm_lines = load_osm_data()

    # Combine all subway data
    print("\nCombining all subway data...")
    all_stations = pd.concat([
        cptond_stations[['city', 'source', 'geometry']],
        osm_stations[['city', 'source', 'geometry']]
    ], ignore_index=True)
    all_stations = gpd.GeoDataFrame(all_stations, crs="EPSG:4326")

    all_lines = pd.concat([cptond_lines, osm_lines], ignore_index=True)
    all_lines = gpd.GeoDataFrame(all_lines, crs="EPSG:4326")
    print(f"  Combined: {len(all_stations):,} stations, {len(all_lines):,} city line networks")

    # Load city boundaries and find cities with subway
    print("\nLoading city boundaries...")
    city_boundaries = gpd.read_file(CITY_BOUNDARIES_FILE)
    print(f"  Loaded {len(city_boundaries):,} city boundaries")

    city_ids_with_subway = find_cities_with_subway(city_boundaries, all_stations, all_lines)

    # Process each resolution
    results = {}

    for resolution in [5, 6, 7]:
        print(f"\n{'='*60}")
        print(f"Loading Resolution {resolution} hexagons...")
        print(f"{'='*60}")

        h_file = H3_GRIDS[resolution]
        layer_name = f"all_cities_h3_grids_resolution{resolution}"
        hexagons = gpd.read_file(h_file, layer=layer_name)
        print(f"  Loaded {len(hexagons):,} total hexagons")

        # Filter to cities with subway
        hexagons_filtered = hexagons[hexagons['ID_HDC_G0'].isin(city_ids_with_subway)]
        print(f"  Filtered to {len(hexagons_filtered):,} hexagons in cities with subway")

        result = process_single_resolution(
            resolution, hexagons_filtered, all_stations, all_lines, tunnel_mi, station_mi
        )
        results[resolution] = result

    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)

    for resolution, result in results.items():
        output_file = RESULTS_DIR / f"subway_mass_by_hexagon_resolution{resolution}.csv"
        result.to_csv(output_file, index=False)
        print(f"  Saved: {output_file.name} ({len(result):,} rows)")

    # Validation summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    for resolution in [5, 6, 7]:
        output_file = RESULTS_DIR / f"subway_mass_by_hexagon_resolution{resolution}.csv"
        df = pd.read_csv(output_file)
        print(f"\nResolution {resolution}:")
        print(f"  Hexagons: {len(df):,}")
        print(f"  Unique cities: {df['city_id'].nunique():,}")
        print(f"  Total stations: {df['n_stations'].sum():,}")
        print(f"  Total length: {df['total_length_km'].sum():,.1f} km")
        print(f"  Total mass: {df['subway_mass_tonnes'].sum() / 1e9:.3f} Gt")

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
