#!/usr/bin/env python3
"""
Script to create H3 hexagons over all cities in GHS_STAT_UCDB2015 dataset.

This script:
1. Loads all cities from the GEE FeatureCollection "users/kh3657/GHS_STAT_UCDB2015"
2. Generates H3 hexagons (configurable resolution, default 6) for each city
3. Keeps specified columns and exports to GPKG format

Author: Generated for NYU China Grant project
Date: 2024
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import ee
import geemap
import h3
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

from utils.paths import get_resolution_dir

GEE_PROJECT = 'ee-knhuang'
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def initialize_earth_engine(project: str) -> None:
    """Authenticate and initialize the Earth Engine client on demand."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

def load_cities_from_gee():
    """
    Load cities data from Google Earth Engine FeatureCollection.
    
    Returns:
        gpd.GeoDataFrame: Cities data with selected properties
    """
    print("Loading cities from GEE FeatureCollection...")
    
    # Define properties to select
    properties_to_select = ['ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2', 'E_KG_NM_LS']
    
    # Load FeatureCollection with selected properties
    fc_UCs = ee.FeatureCollection("users/kh3657/GHS_STAT_UCDB2015").select(
        propertySelectors=properties_to_select, 
        retainGeometry=True 
    )
    
    # Convert to GeoDataFrame
    ghs_gdf = geemap.ee_to_gdf(fc_UCs)
    print(f"Loaded {len(ghs_gdf)} cities from GHS dataset")
    
    return ghs_gdf

def shapely_to_h3shape(geom):
    """
    Convert a shapely Polygon or MultiPolygon to an h3 LatLngPoly or LatLngMultiPoly.

    h3 uses (lat, lng) order while shapely uses (lng, lat), so coordinates
    are swapped during conversion.

    Args:
        geom: shapely Polygon or MultiPolygon

    Returns:
        h3.LatLngPoly or h3.LatLngMultiPoly
    """
    def _ring_to_latlng(ring):
        """Convert a shapely ring to list of (lat, lng) tuples."""
        return [(y, x) for x, y in ring.coords]

    def _polygon_to_latlng_poly(poly):
        """Convert a shapely Polygon to h3.LatLngPoly."""
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


def generate_h3_grids_for_city(city_row, idx, resolution=6):
    """
    Generate H3 hexagons for a single city, covering the full city polygon.

    Uses h3.h3shape_to_cells_experimental with contain='overlap' to include ALL
    hexagons that intersect the city polygon (not just centroids-inside). Boundary
    hexagons are then clipped to the city polygon so all resolutions cover exactly
    the same spatial area.

    Args:
        city_row (pd.Series): City data row
        idx (int): City index for naming
        resolution (int): H3 resolution level (default: 6)

    Returns:
        gpd.GeoDataFrame or None: H3 grids with city attributes, clipped to city boundary
    """
    try:
        city_name = city_row.get('UC_NM_MN', f'City_{idx}')
        city_geom = city_row.geometry

        if city_geom is None or city_geom.is_empty:
            print(f"Warning: Empty geometry for city {city_name}")
            return None

        # Convert shapely geometry to h3 shape
        h3_shape = shapely_to_h3shape(city_geom)

        # Get all H3 cells that overlap with the city polygon
        try:
            cell_ids = h3.h3shape_to_cells_experimental(h3_shape, resolution, contain='overlap')
        except AttributeError:
            # Fallback for older h3 versions without experimental overlap
            cell_ids = h3.h3shape_to_cells(h3_shape, resolution)
            # Expand by one ring to capture boundary hexagons
            expanded = set(cell_ids)
            for c in cell_ids:
                expanded.update(h3.grid_ring(c, 1))
            # Filter to only cells that actually intersect the city polygon
            filtered = set()
            for c in expanded:
                boundary = h3.cell_to_boundary(c)
                hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
                if hex_poly.intersects(city_geom):
                    filtered.add(c)
            cell_ids = filtered

        if not cell_ids:
            print(f"Warning: No hexagons generated for city {city_name}")
            return None

        # Build GeoDataFrame from H3 cell IDs
        hex_geometries = []
        hex_ids = []
        for cell_id in cell_ids:
            boundary = h3.cell_to_boundary(cell_id)
            # h3 returns (lat, lng), shapely needs (lng, lat)
            hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
            hex_geometries.append(hex_poly)
            hex_ids.append(cell_id)

        hex_gdf = gpd.GeoDataFrame(
            {'h3index': hex_ids},
            geometry=hex_geometries,
            crs='EPSG:4326'
        )

        # Clip hexagons to city boundary so all resolutions cover the same area
        city_gdf_single = gpd.GeoDataFrame([city_row], crs='EPSG:4326')
        city_h3_grids = gpd.clip(hex_gdf, city_gdf_single)

        # Remove any empty geometries produced by clipping
        city_h3_grids = city_h3_grids[~city_h3_grids.geometry.is_empty].copy()

        if city_h3_grids.empty:
            print(f"Warning: No hexagons remain after clipping for city {city_name}")
            return None

        # Add city attributes to each hexagon
        for col in ['ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2', 'E_KG_NM_LS']:
            if col in city_row:
                city_h3_grids[col] = city_row[col]

        # Use h3index as neighborhood identifier; combined with ID_HDC_G0
        # this forms a unique key for city-neighborhood pairs
        city_h3_grids['neighborhood_id'] = city_h3_grids['h3index']

        return city_h3_grids

    except Exception as e:
        city_name = city_row.get('UC_NM_MN', f'City_{idx}')
        print(f"Error processing city {city_name}: {e}")
        return None

def main():
    """
    Main function to generate H3 grids for all cities and export results.
    """
    parser = argparse.ArgumentParser(
        description='Generate H3 hexagonal grids for all cities in GHS_STAT_UCDB2015'
    )
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=6,
        help='H3 resolution level (default: 6). Higher = smaller hexagons.'
    )
    args = parser.parse_args()
    resolution = args.resolution

    # Set output directory based on resolution
    OUTPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting H3 grid generation for all cities at resolution {resolution}...")
    print(f"Output directory: {OUTPUT_DIR}")

    initialize_earth_engine(GEE_PROJECT)
    
    # Load cities data from GEE
    ghs_gdf = load_cities_from_gee()
    
    # Ensure CRS is EPSG:4326 for H3 compatibility
    if ghs_gdf.crs is None:
        print("Warning: Input GeoDataFrame has no CRS defined. Assuming EPSG:4326")
        ghs_gdf.set_crs('EPSG:4326', inplace=True)
    elif ghs_gdf.crs.to_string() != 'EPSG:4326':
        print(f"Original CRS: {ghs_gdf.crs}. Reprojecting to EPSG:4326 for H3 compatibility.")
        ghs_gdf = ghs_gdf.to_crs('EPSG:4326')
    
    # Process each city to generate hexagons
    all_h3_grids = []
    
    print(f"\nProcessing {len(ghs_gdf)} cities to generate H3 hexagons...")
    
    for idx, (_, city_row) in enumerate(tqdm(ghs_gdf.iterrows(), total=len(ghs_gdf), desc="Processing cities")):
        city_h3_grids = generate_h3_grids_for_city(city_row, idx, resolution=resolution)
        
        if city_h3_grids is not None and not city_h3_grids.empty:
            all_h3_grids.append(city_h3_grids)
    
    # Combine all results
    if all_h3_grids:
        print(f"\nCombining H3 grids from {len(all_h3_grids)} cities...")
        combined_h3_grids = pd.concat(all_h3_grids, ignore_index=True)

        # Convert to GeoDataFrame if needed
        if not isinstance(combined_h3_grids, gpd.GeoDataFrame):
            combined_h3_grids = gpd.GeoDataFrame(combined_h3_grids)
        
        # Remove duplicate (h3index, city) entries - keep first occurrence
        # Use composite key so shared hexagons between adjacent cities each
        # retain their own clipped geometry, ensuring full spatial coverage.
        initial_count = len(combined_h3_grids)
        combined_h3_grids = combined_h3_grids.drop_duplicates(
            subset=['h3index', 'ID_HDC_G0'], keep='first'
        ).reset_index(drop=True)
        final_count = len(combined_h3_grids)

        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} duplicate (h3index, city) entries")
            print(f"Final dataset: {final_count} unique neighborhood-city pairs")
        
        # Ensure we have the required columns
        required_columns = ['neighborhood_id', 'h3index', 'ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2', 'E_KG_NM_LS']
        available_columns = [col for col in required_columns if col in combined_h3_grids.columns]

        if 'geometry' in combined_h3_grids.columns:
            available_columns.append('geometry')

        # Select only the required columns
        final_h3_grids = combined_h3_grids[available_columns].copy()

        print(f"Final dataset contains {len(final_h3_grids)} H3 hexagons")
        print(f"Columns: {list(final_h3_grids.columns)}")

        # Export to GPKG
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"all_cities_h3_grids_resolution{resolution}.gpkg"

        print(f"\nExporting results to: {output_file}")

        final_h3_grids.to_file(output_file, driver='GPKG')
        print(f"Successfully exported {len(final_h3_grids)} H3 hexagons to {output_file}")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"- Total H3 hexagons: {len(final_h3_grids)}")
        print(f"- Cities processed: {len(all_h3_grids)}")
        print(f"- Average hexagons per city: {len(final_h3_grids) / len(all_h3_grids):.1f}")
        
        if 'UC_NM_MN' in final_h3_grids.columns:
            city_counts = final_h3_grids['UC_NM_MN'].value_counts()
            print(f"- City with most hexagons: {city_counts.index[0]} ({city_counts.iloc[0]} hexagons)")
            print(f"- City with least hexagons: {city_counts.index[-1]} ({city_counts.iloc[-1]} hexagons)")
    
    else:
        print("No H3 grids were generated. Please check the input data and processing logic.")

if __name__ == "__main__":
    main()
