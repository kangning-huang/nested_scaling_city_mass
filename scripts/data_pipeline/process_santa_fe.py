#!/usr/bin/env python3
"""
Process Santa Fe, NM (USA) through the entire data pipeline.

This script:
1. Creates H3 hexagonal grids for Santa Fe
2. Extracts volume/pavement data from Google Earth Engine
3. Merges with existing road data template
4. Calculates material stocks
5. Prepares web data

Author: Generated for NYU China Grant project
Date: 2026
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import ee
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

try:
    import h3
except ImportError:
    print("h3 library not found. Installing...")
    os.system("pip install h3")
    import h3

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import get_resolution_dir

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WEB_DATA_DIR = BASE_DIR / "web" / "public" / "webdata"
GEE_PROJECT = 'ee-knhuang'

# Fishman 2024 RASMI Material Intensity values for USA (Tier 3)
# Values in kg/m³ by building height class
MI_USA_FISHMAN2024 = {
    'LW': 145.0,   # Lightweight structures < 3m
    'RS': 270.0,   # Residential Single-family 3-12m
    'RM': 290.0,   # Residential Multi-family 12-50m
    'NR': 265.0,   # Non-Residential 3-50m
    'HR': 305.0,   # High-rise 50-100m
}


def classify_building_by_height(height_m: float) -> str:
    """
    Classify a building by its height into MI categories.

    Height thresholds based on Fishman 2024 / Haberl 2025:
    - LW (Lightweight): < 3m (small outbuildings, sheds)
    - RS (Residential Single-family): 3-12m (1-4 stories)
    - RM (Residential Multi-family): 12-50m (4-15 stories)
    - HR (High-rise): >= 50m (15+ stories)

    Note: NR (Non-Residential) is determined by land use, not height.
    For residential-dominated cities like Santa Fe, we use RS/RM.
    """
    if height_m < 3:
        return 'LW'
    elif height_m < 12:
        return 'RS'
    elif height_m < 50:
        return 'RM'
    else:
        return 'HR'


# Santa Fe metadata
SANTA_FE_CONFIG = {
    'city_id': 13132,  # New unique ID for Santa Fe, NM
    'city_name': 'Santa Fe',
    'country_iso': 'USA',
    'country_name': 'United States of America',
    'region_l1': 'Northern America',
    'region_l2': 'Northern America',
    'gpkg_path': DATA_DIR / 'GUB_Global_2018_SantaFe.gpkg'
}


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialized with project: {GEE_PROJECT}")
    except Exception:
        print("Authenticating with GEE...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def shapely_to_h3shape(geom):
    """Convert shapely Polygon/MultiPolygon to h3 LatLngPoly/LatLngMultiPoly."""
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


def create_h3_grids(gdf: gpd.GeoDataFrame, resolution: int = 7) -> gpd.GeoDataFrame:
    """
    Create H3 hexagonal grids for Santa Fe.

    Args:
        gdf: GeoDataFrame with Santa Fe boundary
        resolution: H3 resolution (default 7 for neighborhood level)

    Returns:
        GeoDataFrame with H3 hexagons
    """
    print(f"\n=== Creating H3 Grids (Resolution {resolution}) ===")

    city_geom = gdf.geometry.iloc[0]

    # Convert to H3 shape
    h3_shape = shapely_to_h3shape(city_geom)

    # Get all H3 cells that overlap with the city polygon
    try:
        cell_ids = h3.h3shape_to_cells_experimental(h3_shape, resolution, contain='overlap')
    except AttributeError:
        # Fallback for older h3 versions
        cell_ids = h3.h3shape_to_cells(h3_shape, resolution)
        expanded = set(cell_ids)
        for c in cell_ids:
            expanded.update(h3.grid_ring(c, 1))
        filtered = set()
        for c in expanded:
            boundary = h3.cell_to_boundary(c)
            hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
            if hex_poly.intersects(city_geom):
                filtered.add(c)
        cell_ids = filtered

    print(f"Generated {len(cell_ids)} H3 hexagons")

    # Build GeoDataFrame
    hex_geometries = []
    hex_ids = []
    for cell_id in cell_ids:
        boundary = h3.cell_to_boundary(cell_id)
        hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
        hex_geometries.append(hex_poly)
        hex_ids.append(cell_id)

    hex_gdf = gpd.GeoDataFrame(
        {'h3index': hex_ids},
        geometry=hex_geometries,
        crs='EPSG:4326'
    )

    # Clip hexagons to city boundary
    city_gdf = gpd.GeoDataFrame([{'geometry': city_geom}], crs='EPSG:4326')
    hex_gdf = gpd.clip(hex_gdf, city_gdf)
    hex_gdf = hex_gdf[~hex_gdf.geometry.is_empty].copy()

    # Add Santa Fe metadata
    hex_gdf['ID_HDC_G0'] = SANTA_FE_CONFIG['city_id']
    hex_gdf['UC_NM_MN'] = SANTA_FE_CONFIG['city_name']
    hex_gdf['CTR_MN_ISO'] = SANTA_FE_CONFIG['country_iso']
    hex_gdf['CTR_MN_NM'] = SANTA_FE_CONFIG['country_name']
    hex_gdf['GRGN_L1'] = SANTA_FE_CONFIG['region_l1']
    hex_gdf['GRGN_L2'] = SANTA_FE_CONFIG['region_l2']
    hex_gdf['neighborhood_id'] = hex_gdf['h3index']

    print(f"Final grid: {len(hex_gdf)} hexagons after clipping")

    return hex_gdf


def get_impervious_2015():
    """Get impervious surface mask for 2015."""
    year_of_first_ISA = [1972, 1978, 1985, 1986, 1987, 1988, 1989, 1990,
                         1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
                         2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    pixel_values = list(range(1, len(year_of_first_ISA) + 1))
    lookup_table = dict(zip(year_of_first_ISA, pixel_values))
    from_glc_image = ee.ImageCollection("projects/sat-io/open-datasets/GISA_1972_2019").mosaic()
    return from_glc_image.lte(lookup_table.get(2015))


def extract_gba_building_mass(hex_gdf: gpd.GeoDataFrame, city_boundary: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract building mass from Global Building Atlas using building-level MI.

    This function:
    1. Loads GBA building polygons for Santa Fe region
    2. Intersects buildings with H3 hexagons
    3. Classifies each building by height (LW/RS/RM/HR)
    4. Applies Fishman 2024 USA MI values to each building
    5. Aggregates mass to hexagon level

    Args:
        hex_gdf: GeoDataFrame with H3 hexagons (full, unclipped)
        city_boundary: GeoDataFrame with city boundary

    Returns:
        DataFrame with population and building mass per hexagon
    """
    print("\n=== Extracting Building Mass from Global Building Atlas ===")
    print("Using Fishman 2024 RASMI MI values (USA Tier 3)")
    print(f"  LW (<3m): {MI_USA_FISHMAN2024['LW']} kg/m³")
    print(f"  RS (3-12m): {MI_USA_FISHMAN2024['RS']} kg/m³")
    print(f"  RM (12-50m): {MI_USA_FISHMAN2024['RM']} kg/m³")
    print(f"  HR (≥50m): {MI_USA_FISHMAN2024['HR']} kg/m³")

    # Get city bounding box for GBA tile selection
    bounds = city_boundary.total_bounds  # [minx, miny, maxx, maxy]
    min_lon, min_lat, max_lon, max_lat = bounds
    print(f"\nCity bounds: lon [{min_lon:.2f}, {max_lon:.2f}], lat [{min_lat:.2f}, {max_lat:.2f}]")

    # GBA tile naming: w{west}_n{north}_w{east}_n{south} with 5-degree tiles
    # Santa Fe ~(-106, 35.7) falls in tile w110_n40_w105_n35
    tile_name = "w110_n40_w105_n35"
    gba_asset = f"projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS/{tile_name}"
    print(f"Loading GBA tile: {tile_name}")

    # Create city boundary geometry for GEE
    city_geom = city_boundary.geometry.iloc[0]
    city_geojson = json.loads(gpd.GeoDataFrame([{'geometry': city_geom}], crs='EPSG:4326').to_json())
    city_fc = ee.FeatureCollection(city_geojson)
    city_ee_geom = city_fc.geometry()

    # Load GBA buildings and filter to city boundary
    gba = ee.FeatureCollection(gba_asset)
    gba_city = gba.filterBounds(city_ee_geom)

    # Add building properties: height, footprint, volume, classified mass
    def add_building_properties(feature):
        height = ee.Number(feature.get('height'))  # building height in meters
        footprint = feature.geometry().area()  # m²
        volume = height.multiply(footprint)  # m³

        # Classify by height and assign MI
        mi = ee.Algorithms.If(
            height.lt(3), MI_USA_FISHMAN2024['LW'],
            ee.Algorithms.If(
                height.lt(12), MI_USA_FISHMAN2024['RS'],
                ee.Algorithms.If(
                    height.lt(50), MI_USA_FISHMAN2024['RM'],
                    MI_USA_FISHMAN2024['HR']
                )
            )
        )

        # Mass in tonnes = volume (m³) × MI (kg/m³) / 1000
        mass = volume.multiply(ee.Number(mi)).divide(1000)

        return feature.set({
            'height': height,
            'footprint_m2': footprint,
            'volume_m3': volume,
            'mi_kg_m3': mi,
            'mass_tonnes': mass
        })

    gba_with_mass = gba_city.map(add_building_properties)

    # Get building count for reporting
    building_count = gba_with_mass.size().getInfo()
    print(f"Found {building_count:,} buildings in city boundary")

    # Create hexagon FeatureCollection for spatial join
    hex_geojson = json.loads(hex_gdf[['h3index', 'geometry']].to_json())
    hex_fc = ee.FeatureCollection(hex_geojson)

    # For each hexagon, aggregate building properties
    def aggregate_hex(hex_feature):
        hex_geom = hex_feature.geometry()
        h3index = hex_feature.get('h3index')

        # Filter buildings within this hexagon
        buildings_in_hex = gba_with_mass.filterBounds(hex_geom)

        # Aggregate statistics
        building_stats = buildings_in_hex.aggregate_array('mass_tonnes')
        total_mass = building_stats.reduce(ee.Reducer.sum())
        building_count_hex = buildings_in_hex.size()

        # Volume and footprint for metadata
        total_volume = buildings_in_hex.aggregate_array('volume_m3').reduce(ee.Reducer.sum())
        total_footprint = buildings_in_hex.aggregate_array('footprint_m2').reduce(ee.Reducer.sum())

        return hex_feature.set({
            'gba_mass_tonnes': total_mass,
            'gba_building_count': building_count_hex,
            'gba_volume_m3': total_volume,
            'gba_footprint_m2': total_footprint
        })

    hex_with_buildings = hex_fc.map(aggregate_hex)

    # Also get population data
    population = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2015-01-01', '2015-12-31').mosaic()
    impervious = get_impervious_2015()

    # Add population to hexagons
    hex_with_pop = population.reduceRegions(
        collection=hex_with_buildings,
        reducer=ee.Reducer.sum().setOutputs(['population_2015']),
        scale=100
    )

    # Add impervious surface for road estimation
    hex_with_all = impervious.multiply(100).reduceRegions(
        collection=hex_with_pop,
        reducer=ee.Reducer.sum().setOutputs(['impervious_2015']),
        scale=30
    )

    # Download results
    print("Downloading aggregated results from GEE...")
    result = hex_with_all.getInfo()

    # Convert to DataFrame
    rows = []
    for feature in result['features']:
        props = feature['properties']
        rows.append({
            'h3index': props.get('h3index'),
            'population_2015': props.get('population_2015', 0),
            'gba_mass_tonnes': props.get('gba_mass_tonnes', 0),
            'gba_building_count': props.get('gba_building_count', 0),
            'gba_volume_m3': props.get('gba_volume_m3', 0),
            'gba_footprint_m2': props.get('gba_footprint_m2', 0),
            'impervious_2015': props.get('impervious_2015', 0)
        })

    df = pd.DataFrame(rows)

    # Convert None/NaN to 0
    df = df.fillna(0)

    # Calculate derived metrics
    df['mean_height'] = np.where(
        df['gba_footprint_m2'] > 0,
        df['gba_volume_m3'] / df['gba_footprint_m2'],
        0
    )

    print(f"\nGBA Extraction Summary:")
    print(f"  Hexagons processed: {len(df)}")
    print(f"  Total buildings: {df['gba_building_count'].sum():,.0f}")
    print(f"  Total volume: {df['gba_volume_m3'].sum():,.0f} m³")
    print(f"  Total building mass: {df['gba_mass_tonnes'].sum():,.0f} tonnes")
    print(f"  Total population: {df['population_2015'].sum():,.0f}")

    # Report height classification distribution
    print(f"\nHeight-based MI Classification Applied:")
    print(f"  Mean building height: {df['mean_height'].mean():.1f} m")

    return df


def classify_pixels(building_height_img):
    """Classify building pixels into height categories."""
    building_type_img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018").select('built_characteristics')

    residential_mask = (
        building_type_img.eq(11).Or(building_type_img.eq(12))
        .Or(building_type_img.eq(13)).Or(building_type_img.eq(14)).Or(building_type_img.eq(15))
    )
    nonresidential_mask = (
        building_type_img.eq(21).Or(building_type_img.eq(22))
        .Or(building_type_img.eq(23)).Or(building_type_img.eq(24)).Or(building_type_img.eq(25))
    )

    lw_mask = building_height_img.lt(3)
    rs_mask = residential_mask.And(building_height_img.gte(3)).And(building_height_img.lt(12))
    rm_mask = residential_mask.And(building_height_img.gte(12)).And(building_height_img.lt(50))
    nr_mask = nonresidential_mask.And(building_height_img.gte(3)).And(building_height_img.lt(50))
    hr_mask = (residential_mask.Or(nonresidential_mask)).And(building_height_img.gte(50)).And(building_height_img.lte(100))

    return {'LW': lw_mask, 'RS': rs_mask, 'RM': rm_mask, 'NR': nr_mask, 'HR': hr_mask}


def extract_gee_data(hex_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract volume, pavement, and population data from GEE for each hexagon.

    Args:
        hex_gdf: GeoDataFrame with H3 hexagons

    Returns:
        DataFrame with extracted values
    """
    print("\n=== Extracting GEE Data ===")

    impervious_2015 = get_impervious_2015()

    # Create combined image with all bands
    # Esch2022 WSF3D (90m)
    height_esch = ee.Image("projects/ardent-spot-390007/assets/Esch2022_WSF3D/WSF3D_V02_BuildingHeight").multiply(0.1)
    footprint_esch = ee.Image("projects/ardent-spot-390007/assets/Esch2022_WSF3D/WSF3D_V02_BuildingFraction").divide(100).multiply(90*90)
    volume_esch = height_esch.multiply(footprint_esch)
    classes_esch = classify_pixels(height_esch)

    # Li2022 (1000m)
    height_li = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/height_mean")
    volume_li = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/volume_mean").multiply(100000)
    classes_li = classify_pixels(height_li)

    # Liu2024 (500m)
    volume_liu = ee.Image("projects/ardent-spot-390007/assets/Liu2024_GlobalUrbanStructure_3D/GUS3D_Volume")
    height_liu = ee.Image("projects/ardent-spot-390007/assets/Liu2024_GlobalUrbanStructure_3D/GUS3D_Height")
    classes_liu = classify_pixels(height_liu)

    # Population
    population = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2015-01-01', '2015-12-31').mosaic()

    # Build combined image for extraction
    bands = []

    # Esch volumes by class
    for cls_name, mask in classes_esch.items():
        band = volume_esch.updateMask(mask).updateMask(impervious_2015).unmask(0).rename(f'vol_Esch2022_{cls_name}')
        bands.append(band)

    # Li volumes by class
    for cls_name, mask in classes_li.items():
        band = volume_li.updateMask(mask).updateMask(impervious_2015).unmask(0).rename(f'vol_Li2022_{cls_name}')
        bands.append(band)

    # Liu volumes by class
    for cls_name, mask in classes_liu.items():
        band = volume_liu.updateMask(mask).updateMask(impervious_2015).unmask(0).rename(f'vol_Liu2024_{cls_name}')
        bands.append(band)

    # Population and impervious
    bands.append(population.unmask(0).rename('population_2015'))
    bands.append(impervious_2015.multiply(100).unmask(0).rename('impervious_2015'))

    combined_image = ee.Image.cat(bands)

    # Convert hexagons to GEE FeatureCollection
    geojson = json.loads(hex_gdf[['h3index', 'geometry']].to_json())
    fc_hex = ee.FeatureCollection(geojson)

    print(f"Processing {len(hex_gdf)} hexagons...")

    # Extract using reduceRegions at different scales
    # Use 90m scale for best resolution
    stats = combined_image.reduceRegions(
        collection=fc_hex,
        reducer=ee.Reducer.sum(),
        scale=90,
        tileScale=4
    )

    # Download results
    print("Downloading results from GEE...")
    result = stats.getInfo()

    # Convert to DataFrame
    rows = []
    for feature in result['features']:
        props = feature['properties']
        rows.append(props)

    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} records from GEE")

    return df


def calculate_material_stocks(df: pd.DataFrame, use_gba: bool = False) -> pd.DataFrame:
    """
    Calculate material stocks from volume and area data.

    Args:
        df: DataFrame with extracted data
        use_gba: If True, use GBA building mass (already calculated with building-level MI)
                 If False, use WSF3D/Li2022/Liu2024 with class-level MI

    Uses Material Intensity values from Fishman 2024 RASMI for USA.
    """
    print("\n=== Calculating Material Stocks ===")

    if use_gba:
        # GBA path: building mass already calculated with building-level MI
        print("Using Global Building Atlas with building-level MI (Fishman 2024)")
        df['BuildingMass_AverageTotal'] = df['gba_mass_tonnes'].fillna(0)
    else:
        # Legacy path: WSF3D/Li2022/Liu2024 with class-level MI
        print("Using WSF3D/Li2022/Liu2024 with class-level MI")

        # Material Intensity lookup (kg/m³) - Fishman 2024 USA values
        mi_building = {
            'LW': MI_USA_FISHMAN2024['LW'],
            'RS': MI_USA_FISHMAN2024['RS'],
            'RM': MI_USA_FISHMAN2024['RM'],
            'NR': MI_USA_FISHMAN2024['NR'],
            'HR': MI_USA_FISHMAN2024['HR']
        }

        # Calculate building mass for each source and class
        for source in ['Esch2022', 'Li2022', 'Liu2024']:
            total_mass = 0
            for cls in ['LW', 'RS', 'RM', 'NR', 'HR']:
                vol_col = f'vol_{source}_{cls}'
                if vol_col in df.columns:
                    mi = mi_building[cls]
                    mass = df[vol_col].fillna(0) * mi / 1000  # Convert to tonnes
                    df[f'mass_{source}_{cls}'] = mass
                    total_mass += mass
            df[f'BuildingMass_Total_{source}'] = total_mass

        # Average building mass across sources
        mass_cols = [col for col in df.columns if col.startswith('BuildingMass_Total_')]
        df['BuildingMass_AverageTotal'] = df[mass_cols].mean(axis=1)

    # Road mass - use defaults for Santa Fe (North America)
    # Since we don't have road data from GRIP, estimate based on impervious surface
    # Assume ~30% of impervious is roads
    road_fraction = 0.3
    road_mi = 100  # kg/m² average for roads (asphalt + base)
    df['RoadMass_Average'] = (df['impervious_2015'].fillna(0) * road_fraction * road_mi / 1000)

    # Other pavement - remaining impervious surface
    pavement_fraction = 0.2  # 20% parking/yards
    pavement_mi = 57  # kg/m² from Frantz2023
    df['OtherPavMass_Average'] = (df['impervious_2015'].fillna(0) * pavement_fraction * pavement_mi / 1000)

    # Total built mass
    df['mobility_mass_tons'] = df['RoadMass_Average'] + df['OtherPavMass_Average']
    df['total_built_mass_tons'] = df['BuildingMass_AverageTotal'] + df['mobility_mass_tons']

    print(f"Building mass: {df['BuildingMass_AverageTotal'].sum():,.0f} tonnes")
    print(f"Road mass: {df['RoadMass_Average'].sum():,.0f} tonnes")
    print(f"Total built mass: {df['total_built_mass_tons'].sum():,.0f} tonnes")
    print(f"Total population: {df['population_2015'].sum():,.0f}")
    print(f"Per capita: {df['total_built_mass_tons'].sum() / max(df['population_2015'].sum(), 1):,.1f} tonnes/person")

    return df


def compute_city_regression(df: pd.DataFrame):
    """
    Compute neighborhood-level regression for Santa Fe and save to web data.

    Uses the same OLS methodology as compute_regressions.py.
    """
    print("\n=== Computing City Neighborhood Regression ===")

    city_id = SANTA_FE_CONFIG['city_id']

    # Filter positive values
    valid = df[(df['population_2015'] > 0) & (df['total_built_mass_tons'] > 0)].copy()

    if len(valid) < 3:
        print(f"Warning: Only {len(valid)} valid hexagons, skipping regression")
        return

    # Log transform
    log_pop = np.log10(valid['population_2015'].values)
    log_mass = np.log10(valid['total_built_mass_tons'].values)

    # OLS regression
    n = len(log_pop)
    x_mean = log_pop.mean()
    y_mean = log_mass.mean()
    Sxx = ((log_pop - x_mean) ** 2).sum()
    Sxy = ((log_pop - x_mean) * (log_mass - y_mean)).sum()
    slope = Sxy / Sxx if Sxx > 0 else np.nan
    intercept = y_mean - slope * x_mean

    y_pred = intercept + slope * log_pop
    resid = log_mass - y_pred
    s2 = (resid @ resid) / (n - 2)  # residual variance
    se_slope = np.sqrt(s2 / Sxx) if Sxx > 0 else np.nan

    # 95% CI using normal approx
    z = 1.959963984540054
    slope_lo = slope - z * se_slope
    slope_hi = slope + z * se_slope

    ss_tot = ((log_mass - y_mean) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"  N hexagons: {n}")
    print(f"  Slope (δ): {slope:.4f} [{slope_lo:.4f}, {slope_hi:.4f}]")
    print(f"  R²: {r2:.4f}")

    # Save regression
    reg_dir = WEB_DATA_DIR / "regression" / "city_neighborhood"
    reg_dir.mkdir(parents=True, exist_ok=True)

    reg_data = {
        "scope": "city_neighborhood",
        "city_id": city_id,
        "slope": round(slope, 4),
        "slope_lo": round(slope_lo, 4),
        "slope_hi": round(slope_hi, 4),
        "x0": round(x_mean, 4),
        "y0": round(y_mean, 4),
        "n": n,
        "r2": round(r2, 4)
    }

    reg_file = reg_dir / f"{city_id}.json"
    with open(reg_file, 'w') as f:
        json.dump(reg_data, f)
    print(f"Saved regression to: {reg_file}")

    # Interpretation
    if slope_hi < 1:
        print(f"  Interpretation: Sub-linear scaling (slope CI below 1)")
    elif slope_lo > 1:
        print(f"  Interpretation: Super-linear scaling (slope CI above 1)")
    else:
        print(f"  Interpretation: Scaling CI includes 1 (linear)")


def prepare_web_data(df: pd.DataFrame, hex_gdf: gpd.GeoDataFrame):
    """
    Prepare web data for Santa Fe visualization.

    Creates:
    - hex/city=13132.json
    - Updates cities_agg files
    - Updates scatter_samples
    - Updates city_meta index
    """
    print("\n=== Preparing Web Data ===")

    city_id = SANTA_FE_CONFIG['city_id']
    country_iso = SANTA_FE_CONFIG['country_iso']

    # 1. Create hex feed
    hex_dir = WEB_DATA_DIR / "hex"
    hex_dir.mkdir(parents=True, exist_ok=True)

    hex_data = df[['h3index', 'population_2015', 'total_built_mass_tons']].copy()
    hex_data['city_id'] = city_id
    hex_data['country_iso'] = country_iso

    hex_file = hex_dir / f"city={city_id}.json"
    hex_data.to_json(hex_file, orient='records')
    print(f"Created: {hex_file}")

    # 2. Update city aggregates
    cities_agg_dir = WEB_DATA_DIR / "cities_agg"

    # Calculate city totals
    city_agg = {
        'country_iso': country_iso,
        'city_id': city_id,
        'city': SANTA_FE_CONFIG['city_name'],
        'pop_total': float(df['population_2015'].sum()),
        'mass_total': float(df['total_built_mass_tons'].sum()),
        'log_pop': float(np.log10(max(df['population_2015'].sum(), 1))),
        'log_mass': float(np.log10(max(df['total_built_mass_tons'].sum(), 1)))
    }

    # Update global.json
    global_file = cities_agg_dir / "global.json"
    if global_file.exists():
        with open(global_file) as f:
            global_data = json.load(f)
    else:
        global_data = []

    # Remove any existing entry for this city
    global_data = [c for c in global_data if c.get('city_id') != city_id]
    global_data.append(city_agg)

    with open(global_file, 'w') as f:
        json.dump(global_data, f)
    print(f"Updated: {global_file}")

    # Update country file
    country_file = cities_agg_dir / f"country={country_iso}.json"
    if country_file.exists():
        with open(country_file) as f:
            country_data = json.load(f)
    else:
        country_data = []

    country_data = [c for c in country_data if c.get('city_id') != city_id]
    country_data.append(city_agg)

    with open(country_file, 'w') as f:
        json.dump(country_data, f)
    print(f"Updated: {country_file}")

    # 3. Update scatter samples
    scatter_dir = WEB_DATA_DIR / "scatter_samples"

    # Prepare neighborhood samples
    neighborhood_samples = df[['population_2015', 'total_built_mass_tons']].copy()
    neighborhood_samples['log_pop'] = np.log10(neighborhood_samples['population_2015'].clip(lower=1))
    neighborhood_samples['log_mass'] = np.log10(neighborhood_samples['total_built_mass_tons'].clip(lower=1))
    neighborhood_samples['city_id'] = city_id
    neighborhood_samples['country_iso'] = country_iso

    # Add to global samples
    global_samples_file = scatter_dir / "global.json"
    if global_samples_file.exists():
        with open(global_samples_file) as f:
            global_samples = json.load(f)
    else:
        global_samples = []

    # Add new samples (limit to avoid bloating)
    new_samples = neighborhood_samples[['log_pop', 'log_mass', 'city_id', 'country_iso']].to_dict('records')
    global_samples.extend(new_samples[:100])  # Add up to 100 samples

    with open(global_samples_file, 'w') as f:
        json.dump(global_samples, f)
    print(f"Updated: {global_samples_file}")

    # Update country samples
    country_samples_file = scatter_dir / f"country={country_iso}.json"
    if country_samples_file.exists():
        with open(country_samples_file) as f:
            country_samples = json.load(f)
    else:
        country_samples = []

    country_samples.extend(new_samples)

    with open(country_samples_file, 'w') as f:
        json.dump(country_samples, f)
    print(f"Updated: {country_samples_file}")

    # 4. Update city_meta index
    index_dir = WEB_DATA_DIR / "index"
    city_meta_file = index_dir / "city_meta.json"

    with open(city_meta_file) as f:
        city_meta = json.load(f)

    # Calculate centroid
    centroid = hex_gdf.geometry.unary_union.centroid

    city_meta[str(city_id)] = {
        'city': SANTA_FE_CONFIG['city_name'],
        'country_iso': country_iso,
        'lat': centroid.y,
        'lon': centroid.x
    }

    with open(city_meta_file, 'w') as f:
        json.dump(city_meta, f)
    print(f"Updated: {city_meta_file}")

    # 5. Update country_to_cities index
    country_cities_file = index_dir / "country_to_cities.json"
    if country_cities_file.exists():
        with open(country_cities_file) as f:
            country_cities = json.load(f)
    else:
        country_cities = {}

    if country_iso not in country_cities:
        country_cities[country_iso] = []

    if city_id not in country_cities[country_iso]:
        country_cities[country_iso].append(city_id)

    with open(country_cities_file, 'w') as f:
        json.dump(country_cities, f)
    print(f"Updated: {country_cities_file}")

    print("\nWeb data preparation complete!")


def create_h3_grids_full(gdf: gpd.GeoDataFrame, resolution: int = 7) -> gpd.GeoDataFrame:
    """
    Create full (unclipped) H3 hexagonal grids for Santa Fe.

    Unlike create_h3_grids(), this returns full hexagon geometries without
    clipping to city boundary. This avoids creating tiny hexagon slivers
    that have population but no building data.

    Args:
        gdf: GeoDataFrame with Santa Fe boundary
        resolution: H3 resolution (default 7 for neighborhood level)

    Returns:
        GeoDataFrame with full H3 hexagons
    """
    print(f"\n=== Creating Full H3 Grids (Resolution {resolution}) ===")

    city_geom = gdf.geometry.iloc[0]

    # Convert to H3 shape
    h3_shape = shapely_to_h3shape(city_geom)

    # Get all H3 cells that overlap with the city polygon
    try:
        cell_ids = h3.h3shape_to_cells_experimental(h3_shape, resolution, contain='overlap')
    except AttributeError:
        # Fallback for older h3 versions
        cell_ids = h3.h3shape_to_cells(h3_shape, resolution)
        expanded = set(cell_ids)
        for c in cell_ids:
            expanded.update(h3.grid_ring(c, 1))
        filtered = set()
        for c in expanded:
            boundary = h3.cell_to_boundary(c)
            hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
            if hex_poly.intersects(city_geom):
                filtered.add(c)
        cell_ids = filtered

    print(f"Generated {len(cell_ids)} H3 hexagons (full, unclipped)")

    # Build GeoDataFrame with FULL hexagon geometries (no clipping)
    hex_geometries = []
    hex_ids = []
    for cell_id in cell_ids:
        boundary = h3.cell_to_boundary(cell_id)
        hex_poly = Polygon([(lng, lat) for lat, lng in boundary])
        hex_geometries.append(hex_poly)
        hex_ids.append(cell_id)

    hex_gdf = gpd.GeoDataFrame(
        {'h3index': hex_ids},
        geometry=hex_geometries,
        crs='EPSG:4326'
    )

    # Add Santa Fe metadata
    hex_gdf['ID_HDC_G0'] = SANTA_FE_CONFIG['city_id']
    hex_gdf['UC_NM_MN'] = SANTA_FE_CONFIG['city_name']
    hex_gdf['CTR_MN_ISO'] = SANTA_FE_CONFIG['country_iso']
    hex_gdf['CTR_MN_NM'] = SANTA_FE_CONFIG['country_name']
    hex_gdf['GRGN_L1'] = SANTA_FE_CONFIG['region_l1']
    hex_gdf['GRGN_L2'] = SANTA_FE_CONFIG['region_l2']
    hex_gdf['neighborhood_id'] = hex_gdf['h3index']

    print(f"Final grid: {len(hex_gdf)} full hexagons")

    return hex_gdf


def main():
    parser = argparse.ArgumentParser(description='Process Santa Fe through data pipeline')
    parser.add_argument('--resolution', '-r', type=int, default=7,
                        help='H3 resolution level (default: 7)')
    parser.add_argument('--skip-gee', action='store_true',
                        help='Skip GEE extraction (use cached data)')
    parser.add_argument('--use-gba', action='store_true',
                        help='Use Global Building Atlas for building mass (recommended)')
    parser.add_argument('--full-hex', action='store_true',
                        help='Use full hexagons without city boundary clipping')
    args = parser.parse_args()

    resolution = args.resolution

    print("="*70)
    print("SANTA FE DATA PIPELINE")
    print("="*70)
    print(f"City: {SANTA_FE_CONFIG['city_name']}, {SANTA_FE_CONFIG['country_iso']}")
    print(f"City ID: {SANTA_FE_CONFIG['city_id']}")
    print(f"H3 Resolution: {resolution}")
    print(f"Use GBA: {args.use_gba}")
    print(f"Full hexagons: {args.full_hex}")
    print()

    # Create output directory
    output_dir = get_resolution_dir(PROCESSED_DIR, resolution)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Santa Fe boundary
    print("\n=== Loading Santa Fe Boundary ===")
    gpkg_path = SANTA_FE_CONFIG['gpkg_path']
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Santa Fe GeoPackage not found: {gpkg_path}")

    santa_fe_gdf = gpd.read_file(gpkg_path)
    print(f"Loaded boundary with area: {santa_fe_gdf.iloc[0]['urbanArea']:.2f} km²")

    # Step 2: Create H3 grids
    if args.full_hex:
        hex_gdf = create_h3_grids_full(santa_fe_gdf, resolution)
    else:
        hex_gdf = create_h3_grids(santa_fe_gdf, resolution)

    # Save H3 grids
    suffix = "_full" if args.full_hex else ""
    h3_output = output_dir / f"santa_fe_h3_grids{suffix}_resolution{resolution}.gpkg"
    hex_gdf.to_file(h3_output, driver='GPKG')
    print(f"Saved H3 grids to: {h3_output}")

    # Step 3: Extract data from GEE
    if not args.skip_gee:
        initialize_gee()

        if args.use_gba:
            # Use Global Building Atlas with building-level MI
            gee_data = extract_gba_building_mass(hex_gdf, santa_fe_gdf)
            gee_output = output_dir / f"santa_fe_gba_extract{suffix}_resolution{resolution}.csv"
        else:
            # Use WSF3D/Li2022/Liu2024 with class-level MI
            gee_data = extract_gee_data(hex_gdf)
            gee_output = output_dir / f"santa_fe_gee_extract{suffix}_resolution{resolution}.csv"

        gee_data.to_csv(gee_output, index=False)
        print(f"Saved extraction data to: {gee_output}")
    else:
        print("\nSkipping GEE extraction (--skip-gee)")
        if args.use_gba:
            gee_output = output_dir / f"santa_fe_gba_extract{suffix}_resolution{resolution}.csv"
        else:
            gee_output = output_dir / f"santa_fe_gee_extract{suffix}_resolution{resolution}.csv"
        if gee_output.exists():
            gee_data = pd.read_csv(gee_output)
        else:
            raise FileNotFoundError(f"Cached data not found: {gee_output}")

    # Step 4: Calculate material stocks
    mass_data = calculate_material_stocks(gee_data, use_gba=args.use_gba)

    # Filter out hexagons with zero mass (data quality)
    before_filter = len(mass_data)
    mass_data = mass_data[mass_data['total_built_mass_tons'] > 0].copy()
    after_filter = len(mass_data)
    if before_filter > after_filter:
        print(f"\nFiltered out {before_filter - after_filter} hexagons with zero mass")
        print(f"Remaining: {after_filter} hexagons")

    # Add metadata
    mass_data['ID_HDC_G0'] = SANTA_FE_CONFIG['city_id']
    mass_data['UC_NM_MN'] = SANTA_FE_CONFIG['city_name']
    mass_data['CTR_MN_ISO'] = SANTA_FE_CONFIG['country_iso']

    # Save mass data
    today = datetime.now().strftime("%Y-%m-%d")
    mass_output = output_dir / f"santa_fe_mass{suffix}_resolution{resolution}_{today}.csv"
    mass_data.to_csv(mass_output, index=False)
    print(f"Saved mass data to: {mass_output}")

    # Step 5: Prepare web data
    # Filter hex_gdf to match mass_data h3indices
    hex_gdf_filtered = hex_gdf[hex_gdf['h3index'].isin(mass_data['h3index'])].copy()
    prepare_web_data(mass_data, hex_gdf_filtered)

    # Step 6: Compute regression for this city
    compute_city_regression(mass_data)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nSanta Fe ({SANTA_FE_CONFIG['city_id']}) has been added to the web visualization.")
    print(f"Total hexagons: {len(mass_data)}")
    print(f"Total population: {mass_data['population_2015'].sum():,.0f}")
    print(f"Total built mass: {mass_data['total_built_mass_tons'].sum():,.0f} tonnes")
    print(f"Per capita: {mass_data['total_built_mass_tons'].sum() / max(mass_data['population_2015'].sum(), 1):,.1f} tonnes/person")


if __name__ == "__main__":
    main()
