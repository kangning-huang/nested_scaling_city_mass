"""
Compute proximity time (PT) to POIs per neighborhood hexagon using the
MAP Global Friction Surface 2019 (v5.1) in Google Earth Engine (GEE).

Two PT variants are calculated:
1) Motorized: uses the `friction` band (all modes).
2) (Optional) A second band can be specified, but the v5.1 dataset only
   exposes `friction`; the walking-only band is not available in this
   release. If no second band is provided, only the motorized PT is
   produced.

This script is structured for Paris as a test case and writes outputs to
`0.CleanProject_Building_v_Mobility/tests`, but the parameters allow reuse
for other cities if POIs, boundary, and hex layers are provided.

Requirements:
- earthengine-api
- geemap
- geopandas
- shapely

Example:
    python 02_accessibility_POIs_neighborhoods.py \\
        --city-name Paris \\
        --poi-path ../data/raw/pois/Paris_pois_9cats.gpkg \\
        --boundary-path ../data/raw/pois/Paris_boundary.gpkg \\
        --hex-path ../../0.CleanProject_GlobalScaling/data/processed/all_cities_h3_grids.gpkg \\
        --output-dir ../tests
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

# Google Earth Engine
import ee
import geemap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default friction dataset (MAP v5.1)
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
    props: List[str] = list(properties) if properties else []
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
    boundary: Optional[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """
    Filter the master hex grid to the target city. Tries ID, then name columns,
    then spatial intersection with the boundary.
    """
    candidates: List[str] = [
        "ID_HDC_G0",
        "id_hdc_g0",
        "city_id",
        "CITY_ID",
    ]

    if city_id is not None:
        for col in candidates:
            if col in hexes.columns:
                subset = hexes[hexes[col] == city_id]
                if not subset.empty:
                    return subset

    # Name-based filtering
    name_cols = ["city", "CITY", "UC_NM_MN", "uc_nm_mn", "name", "NAME"]
    for col in name_cols:
        if col in hexes.columns:
            subset = hexes[hexes[col].str.contains(city_name, case=False, na=False)]
            if not subset.empty:
                return subset

    # Geometry-based filtering
    if boundary is not None and not boundary.empty:
        boundary_4326 = boundary.to_crs("EPSG:4326")
        boundary_union = boundary_4326.geometry.unary_union
        subset = hexes.to_crs("EPSG:4326")[
            hexes.to_crs("EPSG:4326").intersects(boundary_union)
        ]
        if not subset.empty:
            return subset

    raise ValueError(
        "Could not filter hexes for city; please provide a valid city_id or "
        "ensure name/geometry filtering can succeed."
    )


def compute_cumulative_cost_minutes(
    poi_fc: ee.FeatureCollection,
    hex_fc: ee.FeatureCollection,
    friction_band: str,
    max_minutes: float,
    scale_m: int,
    asset_id: str,
) -> ee.FeatureCollection:
    """
    Compute travel time (minutes) from every hex to the nearest POI using the
    specified friction band. The friction surface encodes minutes per meter.
    """
    friction = ee.Image(asset_id).select(friction_band).selfMask()

    # Paint POIs as sources
    poi_sources = ee.Image().paint(poi_fc, 1).selfMask()

    # Limit search radius (units: minutes, matching the friction bands)
    cumulative = friction.cumulativeCost(
        source=poi_sources, maxDistance=max_minutes
    )

    # Reduce to hex polygons; min travel time within each hex
    reduced = cumulative.reduceRegions(
        collection=hex_fc,
        reducer=ee.Reducer.min(),
        scale=scale_m,
        tileScale=2,
    )
    return reduced


def export_results_to_gpkg(
    fc: ee.FeatureCollection,
    out_path: Path,
    id_cols: List[str],
    category: str,
    friction_band: str,
) -> None:
    """Download the FeatureCollection to GeoPackage using a client-side export."""
    print(f"Downloading results for category '{category}' ({friction_band})...")
    gdf = geemap.ee_to_gdf(fc)  # type: ignore
    gdf["category"] = category
    gdf = gdf[id_cols + [f"pt_minutes_{friction_band}", "category", "geometry"]]
    gdf.to_file(out_path, layer=f"{category}_{friction_band}", driver="GPKG")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-hex proximity time to POIs using GEE friction surface."
    )
    parser.add_argument("--city-name", default="Paris", help="City name (default: Paris)")
    parser.add_argument("--city-id", type=int, default=None, help="Optional city ID to filter hexes")
    parser.add_argument(
        "--poi-path",
        default="../data/raw/pois/Paris_pois_9cats.gpkg",
        help="Path to city POIs GeoPackage",
    )
    parser.add_argument(
        "--boundary-path",
        default="../data/raw/pois/Paris_boundary.gpkg",
        help="Path to city boundary GeoPackage",
    )
    parser.add_argument(
        "--hex-path",
        default="../../0.CleanProject_GlobalScaling/data/processed/all_cities_h3_grids.gpkg",
        help="Path to master hex grid GeoPackage",
    )
    parser.add_argument(
        "--output-dir",
        default="../tests",
        help="Directory to write outputs (GPKG/CSV)",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=60.0,
        help="Maximum travel time search radius (minutes)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=500,
        help="Scale (meters) for reduceRegions sampling",
    )
    parser.add_argument(
        "--friction-asset",
        default=FRICTION_ASSET,
        help="Earth Engine asset ID for friction surface (default: MAP v5.1)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Earth Engine...")
    init_earth_engine()

    print("Loading POIs...")
    poi_gdf = gpd.read_file(args.poi_path)
    if "category" not in poi_gdf.columns:
        raise ValueError("POI file must contain a 'category' column.")

    print("Loading boundary...")
    boundary_gdf = gpd.read_file(args.boundary_path)

    print("Loading hex grid...")
    hex_gdf = gpd.read_file(args.hex_path)
    hex_filtered = filter_hexes_for_city(
        hexes=hex_gdf,
        city_name=args.city_name,
        city_id=args.city_id,
        boundary=boundary_gdf,
    )

    print(f"✓ Selected {len(hex_filtered):,} hexes for {args.city_name}")

    # Use centroids to reduce payload size for EE client download
    print("Using hex centroids to minimize payload...")
    hex_filtered = hex_filtered.copy()
    hex_filtered["geometry"] = hex_filtered.geometry.centroid

    # Identify ID columns for traceability
    id_cols = []
    for candidate in ("hex_id", "HEX_ID", "h3", "h3_index", "ID_HDC_G0", "city"):
        if candidate in hex_filtered.columns:
            id_cols.append(candidate)
    if not id_cols:
        hex_filtered["hex_seq"] = range(len(hex_filtered))
        id_cols = ["hex_seq"]

    # Convert to EE collections
    hex_fc = gdf_to_feature_collection(hex_filtered, properties=id_cols)

    # Process each category independently
    categories = sorted(poi_gdf["category"].unique())
    all_records: List[pd.DataFrame] = []

    for category in categories:
        print(f"\nCategory: {category}")
        pois_cat = poi_gdf[poi_gdf["category"] == category]
        if pois_cat.empty:
            print("  Skipping (no POIs).")
            continue

        # Simplify POI geometries to centroids to keep payload small
        pois_cat_simple = pois_cat.to_crs("EPSG:4326").copy()
        pois_cat_simple["geometry"] = pois_cat_simple.geometry.centroid
        poi_fc = gdf_to_feature_collection(pois_cat_simple, properties=["category"])

        band_gdfs: List[pd.DataFrame] = []

        for band in ["friction"]:
            print(f"  Computing PT with band: {band}")
            result_fc = compute_cumulative_cost_minutes(
                poi_fc=poi_fc,
                hex_fc=hex_fc,
                friction_band=band,
                max_minutes=args.max_minutes,
                scale_m=args.scale,
                asset_id=args.friction_asset,
            )
            # Pull to pandas for local storage
            gdf_band = geemap.ee_to_gdf(result_fc)  # type: ignore
            gdf_band = gdf_band.rename(columns={"min": f"pt_minutes_{band}"})
            band_gdfs.append(gdf_band)

        merged = band_gdfs[0]
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
        merged["category"] = category
        all_records.append(merged)

    if not all_records:
        print("No categories processed; exiting.")
        return

    final_gdf = gpd.GeoDataFrame(
        pd.concat(all_records, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    final_path = output_dir / "Paris_PT_friction.gpkg"
    csv_path = output_dir / "Paris_PT_friction.csv"

    print(f"\nWriting results to {final_path} and {csv_path}")
    final_gdf.to_file(final_path, driver="GPKG")

    # Drop geometry for a compact CSV
    final_df = final_gdf.drop(columns="geometry")
    final_df.to_csv(csv_path, index=False)

    print("\nDone. Review the outputs in the tests folder.")


if __name__ == "__main__":
    main()
