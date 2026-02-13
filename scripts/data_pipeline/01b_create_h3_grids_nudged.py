#!/usr/bin/env python3
"""
01b_create_h3_grids_nudged.py - Generate nudged H3 hexagon grids for sensitivity analysis.

This script creates H3 hexagon grids that are "nudged" in different directions (N, S, E, W)
to test the sensitivity of scaling analysis results to hexagon placement.

Since H3 is a fixed global grid (hexagons cannot be moved), we achieve the nudge effect by:
1. Translating city boundaries by the nudge amount
2. Generating H3 cells for the translated boundaries
3. Translating the hexagon geometries BACK to original positions
4. Clipping to ORIGINAL city boundaries

This effectively creates hexagons as if the H3 grid was shifted.

Recommended nudge distances (approximately 25-30% of edge length):
- Resolution 5 (edge ~9.9km): 2.5 km nudge
- Resolution 6 (edge ~3.7km): 1.0 km nudge
- Resolution 7 (edge ~1.4km): 0.4 km nudge

Usage:
    # Generate all nudge directions for resolution 6
    python scripts/01b_create_h3_grids_nudged.py --resolution 6

    # Generate specific direction only
    python scripts/01b_create_h3_grids_nudged.py --resolution 6 --direction north

    # Use custom nudge distance (in kilometers)
    python scripts/01b_create_h3_grids_nudged.py --resolution 6 --nudge-km 1.5

Author: Generated for NYU China Grant project - Sensitivity Analysis
Date: 2025
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import ee
import geemap
import h3
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate
from tqdm import tqdm

from utils.paths import get_resolution_dir

GEE_PROJECT = 'ee-knhuang'
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Default nudge distances in kilometers for each resolution
DEFAULT_NUDGE_KM = {
    5: 2.5,  # ~25% of 9.85km edge
    6: 1.0,  # ~27% of 3.73km edge
    7: 0.4,  # ~28% of 1.41km edge
}

# Direction vectors (dx, dy) in terms of "to the right" and "up"
# For geographic coordinates: dx = east (+lon), dy = north (+lat)
DIRECTIONS = {
    'north': (0, 1),
    'south': (0, -1),
    'east': (1, 0),
    'west': (-1, 0),
}


def km_to_degrees(km: float, latitude: float = 0.0) -> Tuple[float, float]:
    """
    Convert kilometers to approximate degrees at a given latitude.

    Args:
        km: Distance in kilometers
        latitude: Latitude for longitude correction (degrees)

    Returns:
        (lon_degrees, lat_degrees) tuple
    """
    # 1 degree latitude ≈ 111 km everywhere
    lat_deg = km / 111.0

    # 1 degree longitude ≈ 111 * cos(latitude) km
    lon_deg = km / (111.0 * np.cos(np.radians(latitude)))

    return (lon_deg, lat_deg)


def initialize_earth_engine(project: str) -> None:
    """Authenticate and initialize the Earth Engine client."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def load_cities_from_gee() -> gpd.GeoDataFrame:
    """Load cities data from Google Earth Engine FeatureCollection."""
    print("Loading cities from GEE FeatureCollection...")

    properties_to_select = [
        'ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM',
        'GRGN_L1', 'GRGN_L2', 'E_KG_NM_LS'
    ]

    fc_UCs = ee.FeatureCollection("users/kh3657/GHS_STAT_UCDB2015").select(
        propertySelectors=properties_to_select,
        retainGeometry=True
    )

    ghs_gdf = geemap.ee_to_gdf(fc_UCs)
    print(f"Loaded {len(ghs_gdf)} cities from GHS dataset")

    return ghs_gdf


def translate_geometry(geom, dx_deg: float, dy_deg: float):
    """Translate a geometry by given degrees in x (longitude) and y (latitude)."""
    return translate(geom, xoff=dx_deg, yoff=dy_deg)


def shapely_to_h3shape(geom):
    """
    Convert a shapely Polygon or MultiPolygon to an h3 LatLngPoly or LatLngMultiPoly.
    """
    def _ring_to_latlng(ring):
        return [(y, x) for x, y in ring.coords]

    def _polygon_to_latlng_poly(poly):
        outer = _ring_to_latlng(poly.exterior)
        holes = [_ring_to_latlng(interior) for interior in poly.interiors]
        return h3.LatLngPoly(outer, *holes)

    if isinstance(geom, Polygon):
        return _polygon_to_latlng_poly(geom)
    elif isinstance(geom, MultiPolygon):
        polys = [_polygon_to_latlng_poly(p) for p in geom.geoms]
        return h3.LatLngMultiPoly(*polys)
    else:
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")


def generate_nudged_h3_grids_for_city(
    city_row,
    idx: int,
    resolution: int,
    nudge_km: float,
    direction: str
) -> Optional[gpd.GeoDataFrame]:
    """
    Generate H3 hexagons for a city with a nudge offset.

    The approach:
    1. Translate city boundary by nudge amount
    2. Generate H3 cells for translated boundary (with buffer for edge coverage)
    3. Translate hexagon geometries BACK by opposite amount
    4. Clip to ORIGINAL city boundary

    Args:
        city_row: City data row
        idx: City index
        resolution: H3 resolution level
        nudge_km: Nudge distance in kilometers
        direction: Nudge direction ('north', 'south', 'east', 'west')

    Returns:
        GeoDataFrame of nudged H3 hexagons, or None if error
    """
    try:
        city_name = city_row.get('UC_NM_MN', f'City_{idx}')
        original_geom = city_row.geometry

        if original_geom is None or original_geom.is_empty:
            return None

        # Get city centroid latitude for accurate degree conversion
        centroid_lat = original_geom.centroid.y

        # Convert nudge to degrees
        lon_deg_per_km, lat_deg_per_km = km_to_degrees(1.0, centroid_lat)

        # Get direction vector
        dx, dy = DIRECTIONS[direction]

        # Calculate translation in degrees
        translate_lon = dx * nudge_km * lon_deg_per_km
        translate_lat = dy * nudge_km * lat_deg_per_km

        # Step 1: Translate city boundary (in opposite direction to effectively nudge grid)
        # We shift the city in opposite direction, generate hexagons, then shift them back
        # This is equivalent to shifting the H3 grid in the desired direction
        shifted_geom = translate_geometry(original_geom, -translate_lon, -translate_lat)

        # Step 2: Generate H3 cells for shifted geometry
        h3_shape = shapely_to_h3shape(shifted_geom)

        try:
            cell_ids = h3.h3shape_to_cells_experimental(h3_shape, resolution, contain='overlap')
        except AttributeError:
            cell_ids = h3.h3shape_to_cells(h3_shape, resolution)
            expanded = set(cell_ids)
            for c in cell_ids:
                expanded.update(h3.grid_ring(c, 1))
            filtered = set()
            for c in expanded:
                boundary = h3.cell_to_boundary(c)
                hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
                if hex_poly.intersects(shifted_geom):
                    filtered.add(c)
            cell_ids = filtered

        if not cell_ids:
            return None

        # Step 3: Create hexagon geometries and translate them BACK
        hex_geometries = []
        hex_ids = []
        for cell_id in cell_ids:
            boundary = h3.cell_to_boundary(cell_id)
            hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
            # Translate back to align with original city position
            hex_poly_shifted = translate_geometry(hex_poly, translate_lon, translate_lat)
            hex_geometries.append(hex_poly_shifted)
            hex_ids.append(cell_id)

        hex_gdf = gpd.GeoDataFrame(
            {'h3index': hex_ids},
            geometry=hex_geometries,
            crs='EPSG:4326'
        )

        # Step 4: Clip to ORIGINAL city boundary (not the shifted one)
        city_gdf_single = gpd.GeoDataFrame([city_row], crs='EPSG:4326')
        city_h3_grids = gpd.clip(hex_gdf, city_gdf_single)

        # Remove empty geometries
        city_h3_grids = city_h3_grids[~city_h3_grids.geometry.is_empty].copy()

        if city_h3_grids.empty:
            return None

        # Add city attributes
        for col in ['ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2', 'E_KG_NM_LS']:
            if col in city_row:
                city_h3_grids[col] = city_row[col]

        city_h3_grids['neighborhood_id'] = city_h3_grids['h3index']
        city_h3_grids['nudge_direction'] = direction
        city_h3_grids['nudge_km'] = nudge_km

        return city_h3_grids

    except Exception as e:
        city_name = city_row.get('UC_NM_MN', f'City_{idx}')
        print(f"Error processing city {city_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate nudged H3 hexagonal grids for sensitivity analysis'
    )
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=6,
        help='H3 resolution level (default: 6)'
    )
    parser.add_argument(
        '--direction', '-d',
        type=str,
        choices=['north', 'south', 'east', 'west', 'all'],
        default='all',
        help='Nudge direction (default: all)'
    )
    parser.add_argument(
        '--nudge-km',
        type=float,
        default=None,
        help='Custom nudge distance in km (default: resolution-specific)'
    )
    parser.add_argument(
        '--max-cities',
        type=int,
        default=None,
        help='Maximum cities to process (for testing)'
    )
    args = parser.parse_args()

    resolution = args.resolution
    nudge_km = args.nudge_km or DEFAULT_NUDGE_KM.get(resolution, 1.0)

    # Determine which directions to process
    if args.direction == 'all':
        directions_to_process = list(DIRECTIONS.keys())
    else:
        directions_to_process = [args.direction]

    # Set output directory
    OUTPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution) / "nudged"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("H3 GRID GENERATION WITH NUDGE - Sensitivity Analysis")
    print("=" * 70)
    print(f"Resolution: {resolution}")
    print(f"Nudge distance: {nudge_km} km")
    print(f"Directions: {directions_to_process}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Initialize GEE and load cities
    initialize_earth_engine(GEE_PROJECT)
    ghs_gdf = load_cities_from_gee()

    # Ensure CRS
    if ghs_gdf.crs is None:
        ghs_gdf.set_crs('EPSG:4326', inplace=True)
    elif ghs_gdf.crs.to_string() != 'EPSG:4326':
        ghs_gdf = ghs_gdf.to_crs('EPSG:4326')

    # Limit cities if requested
    if args.max_cities:
        ghs_gdf = ghs_gdf.head(args.max_cities)
        print(f"Limited to {len(ghs_gdf)} cities for testing")

    # Process each direction
    for direction in directions_to_process:
        print(f"\n{'='*70}")
        print(f"Processing direction: {direction.upper()}")
        print(f"{'='*70}")

        all_h3_grids = []

        for idx, (_, city_row) in enumerate(tqdm(
            ghs_gdf.iterrows(),
            total=len(ghs_gdf),
            desc=f"Processing cities ({direction})"
        )):
            city_h3_grids = generate_nudged_h3_grids_for_city(
                city_row, idx, resolution, nudge_km, direction
            )

            if city_h3_grids is not None and not city_h3_grids.empty:
                all_h3_grids.append(city_h3_grids)

        # Combine results
        if all_h3_grids:
            print(f"\nCombining H3 grids from {len(all_h3_grids)} cities...")
            combined = pd.concat(all_h3_grids, ignore_index=True)

            if not isinstance(combined, gpd.GeoDataFrame):
                combined = gpd.GeoDataFrame(combined)

            # Remove duplicates
            initial_count = len(combined)
            combined = combined.drop_duplicates(
                subset=['h3index', 'ID_HDC_G0'], keep='first'
            ).reset_index(drop=True)

            if initial_count != len(combined):
                print(f"Removed {initial_count - len(combined)} duplicates")

            # Export - try GPKG first, fall back to Parquet if PROJ issues
            output_file = OUTPUT_DIR / f"all_cities_h3_grids_resolution{resolution}_nudge_{direction}_{nudge_km}km.gpkg"
            try:
                # Try using fiona engine to avoid pyogrio PROJ issues
                combined.to_file(output_file, driver='GPKG', engine='fiona')
            except Exception as e1:
                print(f"  GPKG with fiona failed: {e1}")
                try:
                    # Fallback: try pyogrio
                    combined.to_file(output_file, driver='GPKG', engine='pyogrio')
                except Exception as e2:
                    print(f"  GPKG with pyogrio failed: {e2}")
                    # Final fallback: save as Parquet (no CRS issues)
                    output_file = OUTPUT_DIR / f"all_cities_h3_grids_resolution{resolution}_nudge_{direction}_{nudge_km}km.parquet"
                    combined.to_parquet(output_file)
                    print(f"  Saved as Parquet instead")

            print(f"Exported {len(combined)} hexagons to: {output_file}")
            print(f"  Cities processed: {len(all_h3_grids)}")
            print(f"  Avg hexagons/city: {len(combined)/len(all_h3_grids):.1f}")
        else:
            print(f"No H3 grids generated for direction {direction}")

    print("\n" + "=" * 70)
    print("NUDGED H3 GRID GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files are in: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Modify 03a_submit_batch_exports.py to accept nudged grid files")
    print("  2. Submit batch export tasks for each nudge direction")
    print("  3. Compare scaling results across nudge conditions")


if __name__ == "__main__":
    main()
