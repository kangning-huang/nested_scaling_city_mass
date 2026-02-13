# Usage
# python scripts/Fig3_DataExtract_Volume_Pavement_Neighborhood.py
# python scripts/Fig3_DataExtract_Volume_Pavement_Neighborhood.py --debug
# python scripts/Fig3_DataExtract_Volume_Pavement_Neighborhood.py --debug-country
# python scripts/Fig3_DataExtract_Volume_Pavement_Neighborhood.py --debug-country --country IND
# python scripts/Fig3_DataExtract_Volume_Pavement_Neighborhood.py --debug-ids

import os
from pathlib import Path
from shapely import is_empty
import ee
import geemap
import pandas as pd
from tqdm import tqdm
from datetime import date # Add this import
from functools import reduce
import sys
import geopandas as gpd
import json
import h3
from tobler.util import h3fy # Use h3fy from tobler
import multiprocessing # Added for parallel computing
import argparse # Add this import
import logging
from datetime import datetime

from utils.paths import get_resolution_dir

# Set up base directory and paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

logging.getLogger('geemap').setLevel(logging.WARNING)  # Only show warnings or errors, not info/debug

gee_proj = 'ee-knhuang'

def gdf_to_ee_featurecollection(gdf):
    # Ensure h3index is a property, not just the index
    if 'h3index' not in gdf.columns and gdf.index.name == 'h3index':
        gdf_copy = gdf.reset_index()
    else:
        gdf_copy = gdf.copy()
    
    # Select only h3index and geometry for the FeatureCollection properties
    # GEE will use the geometry column for geometries. h3index will be a property.
    if 'h3index' in gdf_copy.columns and 'geometry' in gdf_copy.columns:
         gdf_for_fc = gdf_copy[['h3index', 'geometry']]
    else: # Fallback if h3index is not a column (e.g. if h3fy returns it as index)
        print("Warning: 'h3index' column not found directly. Ensure your h3fy function provides it as a column.")
        # Attempt to create it from index if named 'h3index'
        if gdf_copy.index.name == 'h3index':
            gdf_copy['h3index'] = gdf_copy.index
            gdf_for_fc = gdf_copy[['h3index', 'geometry']]
        else: # If still not found, this will likely cause issues downstream
            gdf_for_fc = gdf_copy[['geometry']] # Will lack h3index property

    # Convert the GeoDataFrame to a GeoJSON string
    geojson_str = gdf_for_fc.to_json()
    # Parse the GeoJSON string into a Python dictionary
    geojson_dict = json.loads(geojson_str)
    # Create an Earth Engine FeatureCollection from the GeoJSON
    fc = ee.FeatureCollection(geojson_dict)
    # Return the FeatureCollection
    return ee.FeatureCollection(fc)

def ee_featurecollection_to_dataframe(fc, show_progress=True):
    dfs = []
    # Ensure 'h3index' is requested if it exists as a property
    # The properties are derived from the input FeatureCollection
    # We assume 'h3index' is a property of the features in fc
    # and 'sum' is the result of zonal_stats
    columns_to_request = ['h3index', 'sum']
    
    # Check if fc actually has 'h3index'. This is a client-side check on a sample.
    # For robustness, it's better to ensure fc_hex always has 'h3index' property.
    # sample_props = fc.first().propertyNames().getInfo()
    # if 'h3index' not in sample_props:
    #     print("Warning: 'h3index' not found as a property in FeatureCollection for ee_to_df. Results might be incomplete.")
    #     columns_to_request = ['sum'] # Or handle error

    count = fc.size().getInfo()
    chunk_size = 2000 # Reduced chunk size for potentially larger features or more properties
    num_chunks = (count // chunk_size) + 1
    iterator = tqdm(range(num_chunks), desc="Downloading GEE FC to DataFrame", leave=False) if show_progress else range(num_chunks)
    
    for i in iterator:
        start = i * chunk_size
        # end = start + chunk_size # toList takes count, not end index
        list_fc = fc.toList(chunk_size, start) # Get a list of features for the current chunk
        
        # Check if list_fc is empty or smaller than expected
        if list_fc.size().getInfo() == 0:
            continue

        subset_fc = ee.FeatureCollection(list_fc)
        try:
            subset_df = geemap.ee_to_df(subset_fc, columns=columns_to_request)
            dfs.append(subset_df)
        except Exception as e:
            print(f"Error converting EE FeatureCollection chunk to DataFrame: {e}")
            print(f"Problematic chunk properties (first feature): {subset_fc.first().getInfo().get('properties') if subset_fc.size().getInfo() > 0 else 'Empty chunk'}")
            # Decide how to handle: skip chunk, retry, or raise error
            # For now, we'll append an empty df if there's an error to avoid breaking the loop,
            # but this means data might be missing.
            # dfs.append(pd.DataFrame(columns=columns_to_request)) # Or handle more gracefully

    if not dfs: # If all chunks failed or fc was empty
        return pd.DataFrame(columns=['h3index'] + [col for col in columns_to_request if col != 'h3index']) # Ensure h3index column exists

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

def classify_pixels(building_height_img):
    building_type_img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018").select('built_characteristics')
    residential_mask = (
        building_type_img.eq(11)
        .Or(building_type_img.eq(12))
        .Or(building_type_img.eq(13))
        .Or(building_type_img.eq(14))
        .Or(building_type_img.eq(15))
    )
    nonresidential_mask = (
        building_type_img.eq(21)
        .Or(building_type_img.eq(22))
        .Or(building_type_img.eq(23))
        .Or(building_type_img.eq(24))
        .Or(building_type_img.eq(25))
    )
    lw_mask = building_height_img.lt(3)
    rs_mask = residential_mask.And(building_height_img.gte(3)).And(building_height_img.lt(12))
    rm_mask = residential_mask.And(building_height_img.gte(12)).And(building_height_img.lt(50))
    nr_mask = nonresidential_mask.And(building_height_img.gte(3)).And(building_height_img.lt(50))
    hr_mask = (residential_mask.Or(nonresidential_mask)).And(building_height_img.gte(50)).And(building_height_img.lte(100))
    classified_img = (hr_mask.multiply(4)
                      .where(nr_mask, 3)
                      .where(rm_mask, 2)
                      .where(rs_mask, 1)
                      .where(lw_mask, 0))
    return classified_img

def process_Esch2022_hex(fc_hex, impervious_2015):
    height_Esch2022 = ee.Image("projects/ardent-spot-390007/assets/Esch2022_WSF3D/WSF3D_V02_BuildingHeight").multiply(0.1) # in m
    footprint_Esch2022 = ee.Image("projects/ardent-spot-390007/assets/Esch2022_WSF3D/WSF3D_V02_BuildingFraction").divide(100).multiply(90*90) # in m^2
    volume_Esch2022 = height_Esch2022.multiply(footprint_Esch2022) # in m^3
    # zonal stats in each class
    class_Esch2022 = classify_pixels(height_Esch2022)
    class_names = ["LW", "RS", "RM", "HR", "NR"]
    zonal_stats_dfs = []
    for class_value, class_name in enumerate(class_names):
        # print(f"Processing Esch2022 class: {class_name}") # Too verbose for many cities
        # Zonal stats of volumes
        mask = class_Esch2022.eq(class_value)
        masked_volume = volume_Esch2022.updateMask(mask).updateMask(impervious_2015)
        stats = geemap.zonal_stats(
            in_zone_vector=fc_hex,
            in_value_raster=masked_volume,
            scale=90,
            statistics_type='SUM',
            return_fc=True
        )
        df_volume = ee_featurecollection_to_dataframe(stats, show_progress=False) # Disable progress for inner loop
        df_volume.rename(columns={'sum': f'vol_Esch2022_{class_name}'}, inplace=True)
        if 'h3index' not in df_volume.columns and not df_volume.empty: # Should not happen if ee_featurecollection_to_dataframe is correct
             print(f"Warning: h3index missing in DataFrame for Esch2022 class {class_name}")
        zonal_stats_dfs.append(df_volume)
    
    if not zonal_stats_dfs: return pd.DataFrame() # Should not happen

    # Iteratively merge, ensuring h3index is the key
    # Start with the first df, or an empty df with h3index if the first one is empty
    if not zonal_stats_dfs[0].empty:
        merged_df = zonal_stats_dfs[0]
    else: # If the first df is empty, try to find a non-empty one to start, or use h3_indices_df
        # This part needs careful handling if some classes have no data for any hexagon
        # A robust way is to start with a DataFrame containing all h3indices
        # For now, let's assume at least one df will have h3index
        merged_df = pd.DataFrame(columns=['h3index']) # Fallback
        for df_item in zonal_stats_dfs:
            if 'h3index' in df_item.columns:
                merged_df = df_item
                break
    
    for i in range(1, len(zonal_stats_dfs)):
        if not zonal_stats_dfs[i].empty and 'h3index' in zonal_stats_dfs[i].columns:
            merged_df = pd.merge(merged_df, zonal_stats_dfs[i], on='h3index', how='outer')
        elif zonal_stats_dfs[i].empty:
            pass # Skip empty dataframes
        else: # h3index missing
            print(f"Warning: h3index missing in a DataFrame during Esch2022 merge. Skipping.")

    # Zonal stats of footprints
    stats = geemap.zonal_stats(
        in_zone_vector=fc_hex,
        in_value_raster=footprint_Esch2022.updateMask(impervious_2015),
        scale=90,
        statistics_type='SUM',
        return_fc=True
    )
    df_footprint = ee_featurecollection_to_dataframe(stats, show_progress=False) # Disable progress for inner loop
    df_footprint.rename(columns={'sum': f'footprint_Esch2022'}, inplace=True)
    df_footprint = df_footprint[['h3index', f'footprint_Esch2022']]
    merged_df = merged_df.merge(df_footprint, on='h3index', how='inner')

    return merged_df

def process_Li2022_hex(fc_hex, impervious_2015):
    height_Li2022 = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/height_mean") # in m
    volume_Li2022 = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/volume_mean").multiply(100000) # in m^3
    footprint_Li2022 = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/footprint_mean").multiply(1000*1000) # in m^2
    class_Li2022 = classify_pixels(height_Li2022)
    class_names = ["LW", "RS", "RM", "HR", "NR"]
    zonal_stats_dfs = []
    for class_value, class_name in enumerate(class_names):
        mask = class_Li2022.eq(class_value)
        masked_volume = volume_Li2022.updateMask(mask).updateMask(impervious_2015)
        stats = geemap.zonal_stats(in_zone_vector=fc_hex, in_value_raster=masked_volume, scale=1000, statistics_type='SUM', return_fc=True)
        df_volume = ee_featurecollection_to_dataframe(stats, show_progress=False)
        df_volume.rename(columns={'sum': f'vol_Li2022_{class_name}'}, inplace=True)
        if 'h3index' not in df_volume.columns and not df_volume.empty: 
            print(f"Warning: h3index missing in DataFrame for Li2022 class {class_name}")
        zonal_stats_dfs.append(df_volume)
    
    merged_df = pd.DataFrame(columns=['h3index'])
    for df_item in zonal_stats_dfs:
        if 'h3index' in df_item.columns:
            if merged_df.empty and not df_item.empty : merged_df = df_item
            elif not df_item.empty : merged_df = pd.merge(merged_df, df_item, on='h3index', how='outer')
        elif not df_item.empty: print(f"Warning: h3index missing in a DataFrame during Li2022 merge. Skipping.")

    # Zonal stats of footprints
    stats = geemap.zonal_stats(in_zone_vector=fc_hex, 
                               in_value_raster=footprint_Li2022.updateMask(impervious_2015), 
                               scale=1000, statistics_type='SUM', 
                               return_fc=True)

    df_footprint = ee_featurecollection_to_dataframe(stats, show_progress=False)
    df_footprint.rename(columns={'sum': f'footprint_Li2022'}, inplace=True)
    df_footprint = df_footprint[['h3index', f'footprint_Li2022']]
    merged_df = merged_df.merge(df_footprint, on='h3index', how='inner')

    return merged_df

# ---------------------------------------------------------------------------
# Li-2022 • 90 m area-weighted volumes & footprints for H3-6 hexes
# ---------------------------------------------------------------------------
def process_Li2022_hex_weighted(fc_hex, impervious_2015):
    """
    Returns
    -------
    pandas.DataFrame with columns:
        h3index,
        vol_Li2022_LW, vol_Li2022_RS, vol_Li2022_RM,
        vol_Li2022_HR, vol_Li2022_NR,
        footprint_Li2022                (all in SI units, m³ or m² per hex)
    """
    import ee, pandas as pd, geemap                      # already imported at top of file

    fine_scale = 90  # ← fixed here so callers don’t have to remember it

    # -- 1.  Li-2022 assets --------------------------------------------------
    height_Li2022   = ee.Image(
        "projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/height_mean"
    )                                                     # m
    volume_total_px = ee.Image(
        "projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/volume_mean"
    )                                                     # m³ *per native 1-km pixel*
    footprint_ratio = ee.Image(
        "projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/footprint_mean"
    )                                                     # 0-1 fraction of pixel area

    # -- 2.  Convert per-pixel totals → density so we can area-weight later --
    volume_density  = volume_total_px.divide(ee.Image.pixelArea())  # m³ m⁻²

    # -- 3.  Classify height into LW / RS / … / NR ---------------------------
    class_img   = classify_pixels(height_Li2022)                   # re-uses helper in file
    class_names = ["LW", "RS", "RM", "HR", "NR"]

    dfs_volume = []
    for v, cname in enumerate(class_names):
        # Mask to one class + ISA, then re-weight by pixelArea() at 90 m
        vol_fine = (
            volume_density
            .updateMask(class_img.eq(v))
            .updateMask(impervious_2015)
            .multiply(ee.Image.pixelArea())               # now m³ per 90-m pixel
        )

        stats = geemap.zonal_stats(                       # wrapper around reduceRegions
            in_zone_vector   = fc_hex,
            in_value_raster  = vol_fine,
            scale            = fine_scale,                # 90 m
            statistics_type  = 'SUM',
            return_fc        = True
        )

        df = ee_featurecollection_to_dataframe(stats, show_progress=False)
        df.rename(columns={'sum': f'vol_Li2022_{cname}'}, inplace=True)
        dfs_volume.append(df)

    # -- 4.  Outer-merge the five class tables, keep zeros -------------------
    vol_df = dfs_volume[0]
    for df in dfs_volume[1:]:
        vol_df = vol_df.merge(df, on='h3index', how='outer')
    vol_df.fillna(0, inplace=True)

    # -- 5.  Area-weighted footprints (fraction × pixelArea at 90 m) ---------
    footprint_fine = (
        footprint_ratio
        .updateMask(impervious_2015)
        .multiply(ee.Image.pixelArea())                    # m² per 90-m pixel
    )
    stats_fp = geemap.zonal_stats(
        in_zone_vector   = fc_hex,
        in_value_raster  = footprint_fine,
        scale            = fine_scale,
        statistics_type  = 'SUM',
        return_fc        = True
    )
    df_fp = ee_featurecollection_to_dataframe(stats_fp, show_progress=False)
    df_fp.rename(columns={'sum': 'footprint_Li2022'}, inplace=True)

    # -- 6.  Combine volume & footprint --------------------------------------
    merged = vol_df.merge(df_fp[['h3index', 'footprint_Li2022']],
                          on='h3index', how='inner')

    return merged

def process_Liu2024_hex(fc_hex, impervious_2015):
    volume_Liu2024 = ee.Image("projects/ardent-spot-390007/assets/Liu2024_GlobalUrbanStructure_3D/GUS3D_Volume") # in m^3
    height_Liu2024 = ee.Image("projects/ardent-spot-390007/assets/Liu2024_GlobalUrbanStructure_3D/GUS3D_Height") # in m
    footprint_Liu2024 = volume_Liu2024.divide(height_Liu2024) # in m^2
    class_Liu2024 = classify_pixels(height_Liu2024)
    class_names = ["LW", "RS", "RM", "HR", "NR"]
    zonal_stats_dfs = []
    for class_value, class_name in enumerate(class_names):
        mask = class_Liu2024.eq(class_value)
        masked_volume = volume_Liu2024.updateMask(mask).updateMask(impervious_2015)
        stats = geemap.zonal_stats(in_zone_vector=fc_hex, in_value_raster=masked_volume, scale=500, statistics_type='SUM', return_fc=True)
        df_volume = ee_featurecollection_to_dataframe(stats, show_progress=False)
        df_volume.rename(columns={'sum': f'vol_Liu2024_{class_name}'}, inplace=True)
        if 'h3index' not in df_volume.columns and not df_volume.empty:
            print(f"Warning: h3index missing in DataFrame for Liu2024 class {class_name}")
        zonal_stats_dfs.append(df_volume)

    merged_df = pd.DataFrame(columns=['h3index'])
    for df_item in zonal_stats_dfs:
        if 'h3index' in df_item.columns:
            if merged_df.empty and not df_item.empty : merged_df = df_item
            elif not df_item.empty : merged_df = pd.merge(merged_df, df_item, on='h3index', how='outer')
        elif not df_item.empty: print(f"Warning: h3index missing in a DataFrame during Liu2024 merge. Skipping.")

    # Zonal stats of footprints
    stats = geemap.zonal_stats(in_zone_vector=fc_hex, in_value_raster=footprint_Liu2024.updateMask(impervious_2015), 
                               scale=500, statistics_type='SUM', return_fc=True)
    df_footprint = ee_featurecollection_to_dataframe(stats, show_progress=False)
    df_footprint.rename(columns={'sum': f'footprint_Liu2024'}, inplace=True)
    df_footprint = df_footprint[['h3index', f'footprint_Liu2024']]
    merged_df = merged_df.merge(df_footprint, on='h3index', how='inner')

    return merged_df

def process_WorldPop_hex(fc_hex, impervious_2015):
    population_2015 = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2015-01-01', '2015-12-31').mosaic()
    stats = geemap.zonal_stats(in_zone_vector=fc_hex, in_value_raster=population_2015.updateMask(impervious_2015), 
                               scale=100, statistics_type='SUM', return_fc=True)
    df = ee_featurecollection_to_dataframe(stats, show_progress=False)
    df.rename(columns={'sum': 'population_2015'}, inplace=True)
    return df

def process_Huang2022_hex(fc_hex, impervious_2015): # impervious_2015 is the raster itself
    stats = geemap.zonal_stats(in_zone_vector=fc_hex, in_value_raster=impervious_2015, scale=10, statistics_type='SUM', return_fc=True)
    df = ee_featurecollection_to_dataframe(stats, show_progress=False)
    df.rename(columns={'sum': 'impervious_2015'}, inplace=True) # Renamed to be more specific
    return df

def run_zonal_stats_and_merge(fc_hex, impervious_2015):
    df_esch = process_Esch2022_hex(fc_hex, impervious_2015)
    df_li = process_Li2022_hex(fc_hex, impervious_2015)
    # df_li = process_Li2022_hex_weighted(fc_hex, impervious_2015)
    df_liu = process_Liu2024_hex(fc_hex, impervious_2015)
    df_pop = process_WorldPop_hex(fc_hex, impervious_2015)
    df_imp = process_Huang2022_hex(fc_hex, impervious_2015)

    dataframes_to_merge = []
    if not df_esch.empty and 'h3index' in df_esch.columns: dataframes_to_merge.append(df_esch)
    if not df_li.empty and 'h3index' in df_li.columns: dataframes_to_merge.append(df_li)
    if not df_liu.empty and 'h3index' in df_liu.columns: dataframes_to_merge.append(df_liu)
    if not df_pop.empty and 'h3index' in df_pop.columns: dataframes_to_merge.append(df_pop)
    if not df_imp.empty and 'h3index' in df_imp.columns: dataframes_to_merge.append(df_imp)
    
    if not dataframes_to_merge:
        return pd.DataFrame() # Return empty if no data

    # Extract all unique h3index values from fc_hex to ensure a complete base for merging
    # This is a client-side operation and might be slow for very large fc_hex.
    # It's crucial for ensuring all hexagons are represented, even if some datasets have no values for them.
    try:
        base_h3_df = geemap.ee_to_df(fc_hex.select(['h3index']), columns=['h3index'])[['h3index']].drop_duplicates().reset_index(drop=True)
        if base_h3_df.empty and fc_hex.size().getInfo() > 0: # If ee_to_df failed to get h3index but fc is not empty
             print("Warning: Could not extract h3index from fc_hex for merging. Merged results might be incomplete.")
             # Fallback: use the first available df with h3index as base, or empty if none
             base_h3_df = pd.DataFrame(columns=['h3index'])
             for df in dataframes_to_merge:
                 if 'h3index' in df.columns and not df.empty:
                     base_h3_df = df[['h3index']].drop_duplicates().reset_index(drop=True)
                     break
    except Exception as e:
        print(f"Error extracting h3index from fc_hex for base_h3_df: {e}")
        base_h3_df = pd.DataFrame(columns=['h3index'])


    merged_df = base_h3_df
    for df_to_add in dataframes_to_merge:
         if not df_to_add.empty and 'h3index' in df_to_add.columns:
            merged_df = pd.merge(merged_df, df_to_add, on='h3index', how='outer')
         elif not df_to_add.empty:
            print(f"Warning: DataFrame missing 'h3index' during final merge: {df_to_add.columns}")
    
    return merged_df               

# Worker function for parallel processing
def process_city_worker(args):
    city_row_series, impervious_2015_img_asset_id, gee_project_id, city_gdf_crs_epsg, resolution = args
    MIN_H3_GRIDS = 1 # Minimum number of H3 grids required to process a city

    try:
        # Initialize EE in the worker process
        ee.Initialize(project=gee_project_id)

        city_row = city_row_series
        city_name = city_row['UC_NM_MN']
        city_id = city_row['ID_HDC_G0']
        country_name = city_row['CTR_MN_NM']
        country_iso = city_row['CTR_MN_ISO']

        city_gdf_single = gpd.GeoDataFrame([city_row], crs=f"EPSG:{city_gdf_crs_epsg}")
        impervious_2015_img = impervious_2015_img_asset_id

        # Initial attempt with clip=False and buffer
        city_h3_grids = h3fy(city_gdf_single, resolution=resolution, clip=False, buffer=True)
        city_h3_grids = city_h3_grids.clip(city_gdf_single)

        # Export city and H3 grids
        # city_gdf_single.to_file(f"results/shapefiles/{city_id}_{city_name}_city.gpkg")
        # city_h3_grids.to_file(f"results/shapefiles/{city_id}_{city_name}_h3_grids.gpkg")

        # Fallback if clip=True yields no results or too few results
        if city_h3_grids is None or city_h3_grids.empty:
            print(f"Info for city {city_name}: h3fy with clip=True returned empty. Trying with clip=False.")
            city_h3_grids = h3fy(city_gdf_single, resolution=resolution, clip=False, buffer=True) # Fallback to clip=False
            city_h3_grids = city_h3_grids.clip(city_gdf_single)

            # Check again after clip=False
            if city_h3_grids is None or city_h3_grids.empty:
                print(f"Skipping city {city_name} in worker: h3fy returned {'None' if city_h3_grids is None else 'empty'} even with clip=False.")
                return None
        
        if len(city_h3_grids) < MIN_H3_GRIDS:
            # This condition could be hit if clip=True returned some grids but less than MIN_H3_GRIDS,
            # and we didn't fall into the clip=False block, or if clip=False also returned too few.
            print(f"Skipping city {city_name} in worker: Not enough H3 grids (found {len(city_h3_grids)}, required >= {MIN_H3_GRIDS}) after attempting fallbacks. City might be too small or have geometry issues.")
            return None
        
        # Ensure h3index is a column
        # Handle 'hex_id' index name from tobler.util.h3fy
        if city_h3_grids.index.name == 'hex_id':
            city_h3_grids.index.name = 'h3index' # Rename index to 'h3index'
        
        if 'h3index' not in city_h3_grids.columns:
            if city_h3_grids.index.name == 'h3index':
                city_h3_grids = city_h3_grids.reset_index()
            else:
                # This means index is not 'h3index' (even after potential 'hex_id' rename) AND 'h3index' is not a column
                print(f"Error in worker: 'h3index' column could not be derived for city {city_name}. Index name: '{city_h3_grids.index.name}', Columns: {city_h3_grids.columns.tolist()}. Skipping.")
                return None
        
        # At this point, 'h3index' should be a column.
        if 'h3index' not in city_h3_grids.columns: # Final check, should not be hit if above logic is correct
            print(f"Critical error in worker: 'h3index' column still missing for city {city_name} after processing. Columns: {city_h3_grids.columns.tolist()}. Skipping.")
            return None

        fc_hex = gdf_to_ee_featurecollection(city_h3_grids[['h3index', 'geometry']])
        city_merged_df = run_zonal_stats_and_merge(fc_hex, impervious_2015_img)

        if city_merged_df is not None and not city_merged_df.empty:
            city_merged_df['UC_NM_MN'] = city_name
            city_merged_df['ID_HDC_G0'] = city_id
            city_merged_df['CTR_MN_NM'] = country_name
            city_merged_df['CTR_MN_ISO'] = country_iso
            return city_merged_df
        else:
            print(f"No data returned from zonal stats for city {city_name} in worker.")
            return None
    except Exception as e:
        print(f"Error processing city {city_row.get('UC_NM_MN', 'Unknown')} in worker: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for better debugging in worker
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract volume and pavement data for neighborhoods')
    parser.add_argument('--resolution', '-r', type=int, default=6,
                       help='H3 resolution level (default: 6)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to process only a few cities for testing')
    parser.add_argument('--debug-country', action='store_true',
                       help='Enable debug mode to process cities within a specific country')
    parser.add_argument('--debug-ids', action='store_true',
                       help='Enable debug mode to process only cities with specific IDs')
    parser.add_argument('--country', type=str, default='BGD',
                       help='Country ISO code to filter cities (default: BGD for Bangladesh)')
    args = parser.parse_args()

    # Set DEBUG_MODE based on command line argument
    DEBUG_MODE = args.debug
    DEBUG_COUNTRY = args.debug_country
    DEBUG_IDS = args.debug_ids
    COUNTRY_CODE = args.country
    resolution = args.resolution

    # Set up resolution-specific directories
    OUTPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"H3 Resolution: {resolution}")

    # Define specific city IDs for debug-ids mode
    DEBUG_CITY_IDS = [3512, 3506, 2035, 11424, 2061, 3350, 2390, 2442]
    
    # --- Parallel Processing Setup ---
    total_cpus = os.cpu_count()
    cpus_to_reserve = 2
    num_cpus_to_use = max(1, total_cpus - cpus_to_reserve)
    num_cities_to_test = 3 * num_cpus_to_use

    print(f"Total CPUs: {total_cpus}, Reserving: {cpus_to_reserve}, Using: {num_cpus_to_use} for parallel processing.")
    # --- End Parallel Processing Setup ---

    ee.Authenticate() # Authenticate once in the main process
    ee.Initialize(project=gee_proj)

    year_of_first_ISA = [1972, 1978, 1985, 1986, 1987, 1988, 1989, 1990, 
                         1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 
                         2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    pixel_values = list(range(1, len(year_of_first_ISA) + 1))
    lookup_table = dict(zip(year_of_first_ISA, pixel_values))
    from_glc_image = ee.ImageCollection("projects/sat-io/open-datasets/GISA_1972_2019").mosaic() # Corrected path
    impervious_2015 = from_glc_image.lte(lookup_table.get(2015)) # This is an ee.Image

    # Corrected fc_UCs definition: set retainGeometry=True
    properties_to_select = ['ID_HDC_G0','UC_NM_MN','CTR_MN_ISO','CTR_MN_NM','GRGN_L1','GRGN_L2']
    fc_UCs = ee.FeatureCollection("users/kh3657/GHS_STAT_UCDB2015").select(
        propertySelectors=properties_to_select, 
        retainGeometry=True 
    )
    # Convert fc_UCs to geodataframe
    gdf_cities_all = geemap.ee_to_gdf(fc_UCs)
    # gdf_cities_all = gpd.read_file('data/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg')

    # Sort gdf_cities_all by area
    gdf_cities_all['_calculated_geo_area'] = gdf_cities_all.geometry.area
    gdf_cities_all = gdf_cities_all.sort_values(by='_calculated_geo_area', ascending=False)
    
    # Apply country filtering if debug-country mode is enabled
    # Filter cities based on debug mode
    if DEBUG_COUNTRY:
        print(f"--- DEBUG COUNTRY MODE ENABLED: Filtering cities in country {COUNTRY_CODE} ---")
        
        # Filter cities by country ISO code
        country_cities = gdf_cities_all[gdf_cities_all['CTR_MN_ISO'] == COUNTRY_CODE]
        
        if country_cities.empty:
            print(f"Error: No cities found for country code '{COUNTRY_CODE}'. Available countries: {sorted(gdf_cities_all['CTR_MN_ISO'].unique())}")
            sys.exit(1)
        
        gdf_cities = country_cities.reset_index(drop=True)
        print(f"Found {len(gdf_cities)} cities in {COUNTRY_CODE}: {gdf_cities['UC_NM_MN'].tolist()}")
        
    elif DEBUG_IDS:
        print(f"--- DEBUG IDS MODE ENABLED: Processing cities with specific IDs: {DEBUG_CITY_IDS} ---")
        
        # Convert ID_HDC_G0 to numeric for comparison
        gdf_cities_all['ID_HDC_G0_numeric'] = pd.to_numeric(gdf_cities_all['ID_HDC_G0'], errors='coerce')
        
        # Filter cities by the specified IDs
        id_cities = gdf_cities_all[gdf_cities_all['ID_HDC_G0_numeric'].isin(DEBUG_CITY_IDS)]
        
        if id_cities.empty:
            print(f"Error: No cities found with the specified IDs {DEBUG_CITY_IDS}. Available IDs: {gdf_cities_all['ID_HDC_G0'].head(10).tolist()}...")
            sys.exit(1)
        
        gdf_cities = id_cities.reset_index(drop=True)
        print(f"Found {len(gdf_cities)} cities with specified IDs: {gdf_cities[['ID_HDC_G0', 'UC_NM_MN']].values.tolist()}")
        
    elif DEBUG_MODE:
        print(f"--- DEBUG MODE ENABLED: Targeting the {num_cities_to_test} largest cities (by GeoPandas calculated geometry area) for parallel execution testing ---")
        
        # Calculate area directly from geometries
        try:
            gdf_cities_all['_calculated_geo_area'] = gdf_cities_all.geometry.area
        except Exception as e:
            print(f"Error calculating geometry area with GeoPandas: {e}")
            print("Ensure your GeoDataFrame has valid geometries and an appropriate CRS.")
            sys.exit(1)

        # Sort by the calculated area in descending order and take the top N cities
        gdf_cities_sorted = gdf_cities_all.sort_values(by='_calculated_geo_area', ascending=False)
        gdf_cities = gdf_cities_sorted.head(num_cities_to_test).reset_index(drop=True)
        
        if gdf_cities.empty:
            print(f"Error: No cities selected for debug mode (tried to get top {num_cities_to_test} by calculated geometry area). Check dataset. Exiting.")
            sys.exit(1)
        
        print(f"Selected {len(gdf_cities)} largest cities for debug processing (by GeoPandas calculated area): {gdf_cities['UC_NM_MN'].tolist()}")
        if len(gdf_cities) < num_cpus_to_use:
            print(f"Warning: Selected only {len(gdf_cities)} cities, though {num_cities_to_test} were targeted. This might be due to the dataset having fewer than {num_cpus_to_use} cities.")

    else: # Not DEBUG_MODE
        gdf_cities = gdf_cities_all
    
    # Define output filename with current date
    today = datetime.now().strftime('%Y-%m-%d')
    if DEBUG_COUNTRY:
        output_filename = OUTPUT_DIR / f"Fig3_Volume_Pavement_Neighborhood_H3_Resolution{resolution}_debug_{COUNTRY_CODE}_{today}.csv"
    elif DEBUG_IDS:
        output_filename = OUTPUT_DIR / f"Fig3_Volume_Pavement_Neighborhood_H3_Resolution{resolution}_debug_ids_{today}.csv"
    elif DEBUG_MODE:
        output_filename = OUTPUT_DIR / f"Fig3_Volume_Pavement_Neighborhood_H3_Resolution{resolution}_debug_{today}.csv"
    else:
        output_filename = OUTPUT_DIR / f"Fig3_Volume_Pavement_Neighborhood_H3_Resolution{resolution}_{today}.csv"
    
    # Load existing data if available
    existing_df = pd.DataFrame()
    processed_cities = []
    
    if os.path.exists(output_filename):
        try:
            existing_df = pd.read_csv(output_filename)
            if 'ID_HDC_G0' in existing_df.columns:
                processed_cities = existing_df['ID_HDC_G0'].astype(str).unique().tolist()
                print(f"Found existing data with {len(processed_cities)} already processed cities.")
            else:
                print("Warning: Existing file found but no 'ID_HDC_G0' column detected.")
        except Exception as e:
            print(f"Warning: Could not load existing data from {output_filename}: {e}")
            existing_df = pd.DataFrame()
            processed_cities = []
    else:
        print(f"No existing data found. Starting fresh processing. Output will be saved to: {output_filename}")
    
    # Filter out already processed cities
    if (len(processed_cities) > 0) & (not DEBUG_MODE) & (not DEBUG_COUNTRY):
        # Ensure 'ID_HDC_G0' is a string to match with processed_cities
        gdf_cities['ID_HDC_G0'] = gdf_cities['ID_HDC_G0'].astype(str)
        # Filter out cities that have already been processed:
        city_name_col = 'ID_HDC_G0'  # Adjust this column name as needed
        if city_name_col in gdf_cities.columns:
            gdf_cities = gdf_cities[~gdf_cities[city_name_col].isin(processed_cities)].reset_index(drop=True)
            print(f"Remaining cities to process: {len(gdf_cities)}")
        else:
            print(f"Warning: city name column '{city_name_col}' not found, processing all cities")
    
    if gdf_cities.empty:
        print("All cities have already been processed!")
        return

    # Calculate batch size (5 * number of CPUs)
    batch_size = 1 * num_cpus_to_use
    total_cities = len(gdf_cities)
    
    print(f"\nProcessing {total_cities} cities in batches of {batch_size} using {num_cpus_to_use} cores...")
    
    # Process cities in batches
    # Wrap the outer loop with tqdm
    for batch_start in tqdm(range(0, total_cities, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_cities)
        batch_cities = gdf_cities.iloc[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}: cities {batch_start+1}-{batch_end} of {total_cities}")
        
        # Define the CRS EPSG code for the city GeoDataFrame
        city_gdf_crs_epsg = 4326  # Assuming WGS84, adjust if needed

        # Prepare tasks for the current batch (include resolution parameter)
        tasks = [(city_row, impervious_2015, gee_proj, city_gdf_crs_epsg, resolution) for _, city_row in batch_cities.iterrows()]
        
        batch_results = []
        with multiprocessing.Pool(processes=num_cpus_to_use) as pool:
            # Process results as they complete
            for result_df in pool.imap_unordered(process_city_worker, tasks):
                if result_df is not None and not result_df.empty:
                    batch_results.append(result_df)
        
        # Concatenate batch results
        if batch_results:
            batch_df = pd.concat(batch_results, ignore_index=True)
            
            # Combine with existing data
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            else:
                combined_df = batch_df
            
            # Save to file
            os.makedirs('results', exist_ok=True)
            os.makedirs('results/global_scaling', exist_ok=True)
            combined_df.to_csv(output_filename, index=False)
            
            # Update existing_df for next iteration
            existing_df = combined_df
            
            print(f"Batch {batch_start//batch_size + 1} completed. Data saved to {output_filename}")
            print(f"Total cities processed so far: {len(combined_df['city_name'].unique()) if 'city_name' in combined_df.columns else len(combined_df)}")
        else:
            print(f"No valid results from batch {batch_start//batch_size + 1}")
    
    print(f"\nAll batches completed. Final data saved to {output_filename}")

    # Remove the redundant code that was checking results_list
    # The batch processing already handles everything

if __name__ == "__main__":
    # Make sure the script can be run directly for multiprocessing to work correctly on some OS (e.g. Windows)
    main()