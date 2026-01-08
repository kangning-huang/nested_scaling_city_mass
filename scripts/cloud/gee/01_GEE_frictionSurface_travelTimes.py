"""
Pairwise travel time between H3 hexagons using the MAP Global Friction Surface
2019 (v5.1) in Google Earth Engine.

The script:
1) Loads the master hex grid from `data/raw/all_cities_h3_grids.gpkg`.
2) Filters to a target city (by name or ID).
3) Uses centroids of the hexagons as origins/destinations.
4) For each origin hex, runs `cumulativeCost` on the friction surface and samples
   travel time at every destination hex centroid.
5) Downloads an origin-destination table of travel minutes.

Usage example:
    python 01_GEE_frictionSurface_travelTimes.py --city-name \"New York\" \\
        --output results/pairwise_travel_times_ny.csv

Requirements:
- earthengine-api
- geemap
- geopandas
- shapely
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

import ee
import geemap

# Friction surface (minutes per meter) from MAP v5.1
FRICTION_ASSET = "projects/malariaatlasproject/assets/accessibility/friction_surface/2019_v5_1"


def init_earth_engine() -> None:
    """Initialize Earth Engine, guiding the user if credentials are missing."""
    try:
        ee.Initialize()
    except Exception:
        print(
            "⚠️  Earth Engine not initialized. Run `earthengine authenticate` first "
            "and ensure application default credentials are available."
        )
        raise


def gdf_to_feature_collection(
    gdf: gpd.GeoDataFrame, properties: Optional[Iterable[str]] = None
) -> ee.FeatureCollection:
    """Convert a GeoDataFrame to an Earth Engine FeatureCollection."""
    props = list(properties) if properties else []
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    features = []
    for _, row in gdf.iterrows():
        geom_geojson = mapping(row.geometry)
        feat_props = {p: row[p] for p in props if p in row and pd.notna(row[p])}
        features.append(ee.Feature(ee.Geometry(geom_geojson), feat_props))

    return ee.FeatureCollection(features)


def filter_hexes_for_city(
    hexes: gpd.GeoDataFrame,
    city_name: str,
    city_id: Optional[int],
) -> gpd.GeoDataFrame:
    """
    Filter the master hex grid to the target city. Tries ID, then name columns.
    """
    id_cols = ["ID_HDC_G0", "id_hdc_g0", "city_id", "CITY_ID"]
    for col in id_cols:
        if city_id is not None and col in hexes.columns:
            subset = hexes[hexes[col] == city_id]
            if not subset.empty:
                return subset

    name_cols = ["city", "CITY", "UC_NM_MN", "uc_nm_mn", "name", "NAME"]
    for col in name_cols:
        if col in hexes.columns:
            subset = hexes[hexes[col].str.contains(city_name, case=False, na=False)]
            if not subset.empty:
                return subset

    raise ValueError("Could not filter hexes; supply a city ID or adjust the name.")


def compute_pairwise_travel_minutes(
    hex_fc: ee.FeatureCollection,
    friction_image: ee.Image,
    scale_m: int,
    max_minutes: Optional[float],
    tile_scale: int,
) -> ee.FeatureCollection:
    """
    Compute pairwise travel minutes between all hex centroids using
    `cumulativeCost`. The friction surface encodes minutes per meter.
    """
    hex_list = hex_fc.toList(hex_fc.size())

    def iterate_fn(index, acc):
        feature = ee.Feature(hex_list.get(index))
        origin_id = feature.get("h3index")
        origin_point = feature.geometry()
        source = ee.Image().paint(origin_point, 1).selfMask()

        if max_minutes is not None:
            cumulative = friction_image.cumulativeCost(
                source=source, maxDistance=max_minutes
            )
        else:
            cumulative = friction_image.cumulativeCost(source=source)
        cumulative = cumulative.rename("travel_minutes")

        samples = cumulative.sampleRegions(
            collection=hex_fc,
            scale=scale_m,
            tileScale=tile_scale,
            geometries=False,
        )

        def set_ids(f):
            return f.set(
                {
                    "origin_h3": origin_id,
                    "dest_h3": f.get("h3index"),
                    "travel_minutes": f.get("travel_minutes"),
                }
            )

        samples_with_ids = samples.map(set_ids)
        return ee.FeatureCollection(acc).merge(samples_with_ids)

    sequence = ee.List.sequence(0, hex_fc.size().subtract(1))
    merged = ee.FeatureCollection(sequence.iterate(iterate_fn, ee.FeatureCollection([])))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise travel time between hexagons using MAP friction surface."
    )
    repo_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--hex-path",
        default=repo_root / "data/raw/all_cities_h3_grids.gpkg",
        help="Path to master hex grid GeoPackage.",
    )
    parser.add_argument(
        "--city-name",
        default="New York",
        help="City name used for filtering hexes (case-insensitive).",
    )
    parser.add_argument(
        "--city-id",
        type=int,
        default=None,
        help="Optional city ID (e.g., ID_HDC_G0) for filtering hexes.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1000,
        help="Sampling scale in meters for cumulative cost results.",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Optional search radius in minutes; leave empty for full surface.",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=4,
        help="tileScale parameter for EE reducers to trade memory vs speed.",
    )
    parser.add_argument(
        "--output",
        default=repo_root / "results/pairwise_travel_times.csv",
        help="Output CSV path for the origin-destination travel minutes.",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=None,
        help="Optional limit on number of origin hexes (useful for smoke tests).",
    )
    args = parser.parse_args()

    hex_path = Path(args.hex_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing Earth Engine...")
    init_earth_engine()

    print(f"Loading hex grid from {hex_path} ...")
    hex_gdf = gpd.read_file(hex_path)
    filtered_hexes = filter_hexes_for_city(
        hexes=hex_gdf, city_name=args.city_name, city_id=args.city_id
    )

    if filtered_hexes.empty:
        raise ValueError("No hexes selected after filtering.")

    if "h3index" not in filtered_hexes.columns:
        filtered_hexes = filtered_hexes.rename(columns={"h3": "h3index"})

    if args.max_origins:
        filtered_hexes = filtered_hexes.head(args.max_origins)
        print(f"Limiting to first {len(filtered_hexes)} origins for testing.")

    print(f"✓ Selected {len(filtered_hexes):,} hexes for {args.city_name}")

    # Use centroids to keep payloads light when sending to EE.
    hex_points = filtered_hexes.to_crs("EPSG:4326").copy()
    hex_points["geometry"] = hex_points.geometry.centroid
    hex_points = hex_points[["h3index", "geometry"]]

    print("Converting hexes to Earth Engine FeatureCollection...")
    hex_fc = gdf_to_feature_collection(hex_points, properties=["h3index"])

    print("Preparing friction surface...")
    friction = ee.Image(FRICTION_ASSET).select("friction").selfMask()

    print("Computing pairwise travel minutes on Earth Engine (server-side)...")
    pairwise_fc = compute_pairwise_travel_minutes(
        hex_fc=hex_fc,
        friction_image=friction,
        scale_m=args.scale,
        max_minutes=args.max_minutes,
        tile_scale=args.tile_scale,
    )

    print("Downloading results to pandas DataFrame...")
    df = geemap.ee_to_df(pairwise_fc)  # type: ignore
    df = df[["origin_h3", "dest_h3", "travel_minutes"]]

    print(f"Writing origin-destination table to {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
