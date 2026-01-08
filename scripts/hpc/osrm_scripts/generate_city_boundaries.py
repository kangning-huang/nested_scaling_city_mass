#!/usr/bin/env python3
"""
Export city boundaries as individual GeoJSON files for OSRM processing.

For each city, creates a bounding polygon that encompasses all H3 grid cells.
This boundary is used by osmium to clip the OSM data.

Usage:
    python generate_city_boundaries.py --input all_cities_h3_grids.gpkg --output-dir cities/

    # For specific cities only:
    python generate_city_boundaries.py --input all_cities_h3_grids.gpkg --output-dir cities/ --city-list pilot_cities.txt
"""

import geopandas as gpd
import pandas as pd
import json
import os
import argparse
from shapely.ops import unary_union
from shapely.geometry import mapping


def generate_city_boundaries(input_path: str, output_dir: str, city_list: str = None, buffer_km: float = 2.0):
    """Generate GeoJSON boundary files for cities."""

    print(f"Reading {input_path}...")
    gdf = gpd.read_file(input_path)

    # Filter to specific cities if list provided
    if city_list:
        print(f"Filtering to cities in {city_list}...")
        with open(city_list, 'r') as f:
            city_ids = [int(line.strip()) for line in f if line.strip()]
        gdf = gdf[gdf['ID_HDC_G0'].isin(city_ids)]
        print(f"Selected {len(city_ids)} cities")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get unique cities
    cities = gdf.groupby('ID_HDC_G0').first().reset_index()
    print(f"Processing {len(cities)} cities...")

    # Convert buffer from km to degrees (approximate)
    buffer_deg = buffer_km / 111.0  # ~111 km per degree

    processed = 0
    for _, city_row in cities.iterrows():
        city_id = city_row['ID_HDC_G0']
        city_name = city_row['UC_NM_MN']
        country = city_row['CTR_MN_NM']

        # Get all grids for this city
        city_grids = gdf[gdf['ID_HDC_G0'] == city_id]

        # Create union of all grid geometries
        boundary = unary_union(city_grids.geometry)

        # Buffer slightly to ensure we capture edge roads
        boundary_buffered = boundary.buffer(buffer_deg)

        # Simplify to reduce file size (tolerance in degrees, ~100m)
        boundary_simplified = boundary_buffered.simplify(0.001)

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "city_id": int(city_id),
                "city_name": city_name,
                "country": country,
                "n_grids": len(city_grids)
            },
            "geometry": mapping(boundary_simplified)
        }

        geojson = {
            "type": "FeatureCollection",
            "features": [feature]
        }

        # Write to file
        output_path = os.path.join(output_dir, f"{city_id}.geojson")
        with open(output_path, 'w') as f:
            json.dump(geojson, f)

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{len(cities)} cities...")

    print(f"\nGenerated {processed} boundary files in {output_dir}")

    # Create a manifest file
    manifest = []
    for _, city_row in cities.iterrows():
        manifest.append({
            "city_id": int(city_row['ID_HDC_G0']),
            "city_name": city_row['UC_NM_MN'],
            "country": city_row['CTR_MN_NM'],
            "filename": f"{city_row['ID_HDC_G0']}.geojson"
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Generate city boundary GeoJSONs")
    parser.add_argument("--input", "-i", required=True, help="Input GeoPackage with H3 grids")
    parser.add_argument("--output-dir", "-o", default="cities", help="Output directory for GeoJSONs")
    parser.add_argument("--city-list", "-l", help="Optional: text file with city IDs to process")
    parser.add_argument("--buffer-km", "-b", type=float, default=2.0,
                        help="Buffer distance in km around city boundary (default: 2.0)")
    args = parser.parse_args()

    generate_city_boundaries(args.input, args.output_dir, args.city_list, args.buffer_km)


if __name__ == "__main__":
    main()
