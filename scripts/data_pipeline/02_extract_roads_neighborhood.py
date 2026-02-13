#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig3_DataExtract_Roads_Neighborhood_v3.py
Extract road length data from GRIP raster files for H3 grid neighborhoods
Author: Generated script (translated from R)
Date: 2024
"""
# Run in debug mode with 10 largest cities (default) 
# python scripts/Fig3_DataExtract_Roads_Neighborhood_v3.py --debug

# Run in debug mode with 5 largest cities
# python scripts/Fig3_DataExtract_Roads_Neighborhood_v3.py --debug --debug-cities 5

# Run in debug mode with cities from Bangladesh (default country)
# python scripts/Fig3_DataExtract_Roads_Neighborhood_v3.py --debug-country

# Run in debug mode with cities from India, Peru
# python scripts/Fig3_DataExtract_Roads_Neighborhood_v3.py --debug-country --country IND
# python scripts/Fig3_DataExtract_Roads_Neighborhood_v3.py --debug-country --country PER

# Run normally (process all cities)
# python scripts/Fig3_DataExtract_Roads_Neighborhood_v3.py

import os
import sys
import gc
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
import rasterio
from exactextract import exact_extract
from pathlib import Path
import re
from typing import Dict
import argparse
from datetime import datetime

from utils.paths import get_resolution_dir

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Set up base directory and paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


class LaneDataProcessor:
    """Handles lane data processing from Excel files."""
    
    def process_lane_data(self, 
                         excel_path: Path, 
                         sheet_name: str,
                         country_col: str = 'Country Alpha-3 Code',
                         region_col: str = 'GRIP region',
                         road_type_col: str = 'GRIP road type',
                         lanes_col: str = 'Avg. number of lanes, weighted') -> pd.DataFrame:
        """Process lane data from Excel file."""
        log.info(f"Processing lane data from: {excel_path}")
        
        try:
            lanes_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        except FileNotFoundError:
            log.error(f"Excel file not found: {excel_path}")
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Error reading Excel file: {e}")
            return pd.DataFrame()
            
        # Validate required columns
        required_cols = [country_col, region_col, road_type_col, lanes_col]
        missing_cols = [col for col in required_cols if col not in lanes_df.columns]
        
        if missing_cols:
            log.error(f"Missing required columns: {missing_cols}")
            log.info(f"Available columns: {lanes_df.columns.tolist()}")
            return pd.DataFrame()
            
        # Clean and process data
        lanes_df[lanes_col] = pd.to_numeric(lanes_df[lanes_col], errors='coerce')
        
        # Fill missing values with regional averages
        lanes_df['lanes_filled'] = lanes_df.groupby([region_col, road_type_col])[lanes_col].transform(
            lambda x: x.fillna(x.mean())
        )
        
        # Pivot to wide format
        try:
            pivot_df = lanes_df.pivot_table(
                index=country_col,
                columns=road_type_col,
                values='lanes_filled',
                aggfunc='mean'  # Handle potential duplicates
            ).reset_index()
            
            # Clean column names
            pivot_df.columns = [col if col == country_col else f"lanes_{col.replace(' ', '_').replace('-', '_').lower()}" 
                              for col in pivot_df.columns]
            
            # Rename country column for clarity
            pivot_df.rename(columns={country_col: 'ISO_A3_lanes'}, inplace=True)
            
            log.info(f"Processed lane data: {pivot_df.shape[0]} countries, {pivot_df.shape[1]-1} road types")
            return pivot_df
            
        except Exception as e:
            log.error(f"Error during pivot operation: {e}")
            return pd.DataFrame()


class ClimateProcessor:
    """Handles climate classification processing."""
    
    def __init__(self):
        self.koppen_mapping = {
            'Tropical rain forest': {'koppen_code': 'Af', 'climate_class': 'Wet, non-freeze (WN)'},
            'Tropical monsoon': {'koppen_code': 'Am', 'climate_class': 'Wet, non-freeze (WN)'},
            'Tropical savannah with dry summer': {'koppen_code': 'As', 'climate_class': 'Wet, non-freeze (WN)'},
            'Tropical savannah with dry winter': {'koppen_code': 'Aw', 'climate_class': 'Wet, non-freeze (WN)'},
            'Desert (arid), and Hot arid': {'koppen_code': 'BWh', 'climate_class': 'Dry, non-freeze (DN)'},
            'Desert (arid), and Cold arid': {'koppen_code': 'BWk', 'climate_class': 'Dry, freeze (DF)'},
            'Steppe (semi-arid), and Hot arid': {'koppen_code': 'BSh', 'climate_class': 'Dry, non-freeze (DN)'},
            'Steppe (semi-arid), and Cold arid': {'koppen_code': 'BSk', 'climate_class': 'Dry, non-freeze (DN)'},
            'Mild temperate with dry summer, and Hot summer': {'koppen_code': 'Csa', 'climate_class': 'Dry, non-freeze (DN)'},
            'Mild temperate with dry summer, and Warm summer': {'koppen_code': 'Csb', 'climate_class': 'Dry, non-freeze (DN)'},
            'Mild temperate with dry winter, and Hot summer': {'koppen_code': 'Cwa', 'climate_class': 'Dry, non-freeze (DN)'},
            'Mild temperate with dry winter, and Warm summer': {'koppen_code': 'Cwb', 'climate_class': 'Dry, non-freeze (DN)'},
            'Mild temperate, fully humid, and Hot summer': {'koppen_code': 'Cfa', 'climate_class': 'Wet, non-freeze (WN)'},
            'Mild temperate, fully humid, and Warm summer': {'koppen_code': 'Cfb', 'climate_class': 'Wet, non-freeze (WN)'},
            'Mild temperate, fully humid, and Cool summer': {'koppen_code': 'Cfc', 'climate_class': 'Wet, freeze (WF)'},
            'Snow with dry summer, and Hot summer': {'koppen_code': 'Dsa', 'climate_class': 'Dry, freeze (DF)'},
            'Snow with dry summer, and Warm summer': {'koppen_code': 'Dsb', 'climate_class': 'Dry, freeze (DF)'},
            'Snow with dry summer, and Cool summer': {'koppen_code': 'Dsc', 'climate_class': 'Dry, freeze (DF)'},
            'Snow with dry winter, and Hot summer': {'koppen_code': 'Dwa', 'climate_class': 'Wet, freeze (WF)'},
            'Snow with dry winter, and Warm summer': {'koppen_code': 'Dwb', 'climate_class': 'Wet, freeze (WF)'},
            'Snow with dry winter, and Cool summer': {'koppen_code': 'Dwc', 'climate_class': 'Dry, freeze (DF)'},
            'Snow, fully humid, and Hot summer': {'koppen_code': 'Dfa', 'climate_class': 'Wet, freeze (WF)'},
            'Snow, fully humid, and Warm summer': {'koppen_code': 'Dfb', 'climate_class': 'Wet, freeze (WF)'},
            'Snow, fully humid, and Cool summer': {'koppen_code': 'Dfc', 'climate_class': 'Wet, freeze (WF)'}
        }
        
        # Create a normalized mapping for robust matching
        self.normalized_mapping = {}
        for key, value in self.koppen_mapping.items():
            normalized_key = self._normalize_text(key)
            self.normalized_mapping[normalized_key] = value
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and converting to lowercase."""
        if pd.isna(text) or text is None:
            return ''
        # Replace multiple spaces with single space and strip
        normalized = re.sub(r'\s+', ' ', str(text).strip())
        return normalized.lower()
    
    def _get_climate_mapping(self, climate_text: str) -> Dict[str, str]:
        """Get climate mapping using normalized text matching."""
        if pd.isna(climate_text) or climate_text is None:
            return {'koppen_code': 'Unknown', 'climate_class': 'Unknown'}
        
        normalized_input = self._normalize_text(climate_text)
        return self.normalized_mapping.get(normalized_input, {'koppen_code': 'Unknown', 'climate_class': 'Unknown'})

    def add_climate_classifications(self, df: pd.DataFrame, climate_col: str = 'E_KG_NM_LS') -> pd.DataFrame:
        """Add Köppen climate classifications."""
        log.info("Adding climate classifications")
        
        result_df = df.copy()
        
        if climate_col not in result_df.columns:
            log.warning(f"Climate column '{climate_col}' not found")
            return result_df
            
        # Map climate descriptions to codes and classes using robust matching
        result_df['koppen_code'] = result_df[climate_col].apply(
            lambda x: self._get_climate_mapping(x)['koppen_code']
        )
        
        result_df['climate_class'] = result_df[climate_col].apply(
            lambda x: self._get_climate_mapping(x)['climate_class']
        )
        
        log.info(f"Added climate classifications for {len(result_df)} records")
        return result_df


class RoadMetricsCalculator:
    """Calculates road width and surface area metrics."""
    
    def calculate_road_metrics(self, df, road_type_mapping):
        """Calculate road width and area estimates based on lane counts."""
        result_df = df.copy()
        
        # Debug: Check what columns are available
        log.info(f"Available columns in dataframe: {result_df.columns.tolist()}")
        lanes_cols = [col for col in result_df.columns if col.startswith('lanes_')]
        length_cols = [col for col in result_df.columns if col.startswith('sum_road_m_')]
        log.info(f"Found lanes columns: {lanes_cols}")
        log.info(f"Found length columns: {length_cols}")
        
        # Calculate metrics for each road type
        for tp_key, road_type in road_type_mapping.items():
            length_col = f'sum_road_m_{road_type}'
            lanes_col = f'lanes_{road_type}'
            width_low_col = f'width_low_{road_type}'
            width_high_col = f'width_high_{road_type}'
            area_low_col = f'road_area_low_{road_type}'
            area_high_col = f'road_area_high_{road_type}'
            
            if length_col in result_df.columns and lanes_col in result_df.columns:
                # Calculate width estimates (low and high)
                if road_type in ['highway', 'primary']:
                    result_df[width_low_col] = result_df[lanes_col] * 3.5
                    result_df[width_high_col] = (result_df[lanes_col] + 1) * 4.0
                else:
                    result_df[width_low_col] = result_df[lanes_col] * 3.0
                    result_df[width_high_col] = (result_df[lanes_col] + 1) * 3.5
                
                # Calculate surface area (m^2) for both low and high estimates
                result_df[area_low_col] = result_df[width_low_col] * result_df[length_col]
                result_df[area_high_col] = result_df[width_high_col] * result_df[length_col]
                
                log.debug(f"Calculated low/high metrics for {road_type}")
            else:
                log.warning(f"Missing columns for {road_type}: {lanes_col} or {length_col} not found")
                
        return result_df


def filter_cities_by_country(h3_grids: gpd.GeoDataFrame, country_iso: str = 'BGD') -> gpd.GeoDataFrame:
    """
    Filter H3 grids to include only those from cities in a specific country.
    
    Args:
        h3_grids: GeoDataFrame with H3 grid polygons
        country_iso: ISO 3-letter country code (default: 'BGD' for Bangladesh)
        
    Returns:
        Filtered GeoDataFrame with H3 grids from the specified country only
    """
    log.info(f"Filtering for cities in country: {country_iso}")
    
    # Check if country column exists
    country_col = 'CTR_MN_ISO'
    if country_col not in h3_grids.columns:
        # Try alternative country identifier columns
        possible_country_cols = ['ISO_A3', 'COUNTRY_ISO', 'CTR_ISO']
        for col in possible_country_cols:
            if col in h3_grids.columns:
                country_col = col
                break
        else:
            log.warning("No country identifier column found. Using all data.")
            return h3_grids
    
    # Filter by country
    country_mask = h3_grids[country_col].str.upper() == country_iso.upper()
    filtered_grids = h3_grids[country_mask].copy()
    
    if len(filtered_grids) == 0:
        log.warning(f"No cities found for country {country_iso}. Available countries: {h3_grids[country_col].unique()[:10]}")
        return h3_grids
    
    # Get unique cities in the country
    city_col = 'UC_NM_MN'
    if city_col not in filtered_grids.columns:
        possible_city_cols = ['ID_HDC_G0', 'CTR_MN_NM']
        for col in possible_city_cols:
            if col in filtered_grids.columns:
                city_col = col
                break
    
    unique_cities = filtered_grids[city_col].nunique() if city_col in filtered_grids.columns else 'unknown'
    
    log.info(f"Found {len(filtered_grids)} H3 grids from {unique_cities} cities in {country_iso}")
    
    if city_col in filtered_grids.columns:
        cities_list = filtered_grids[city_col].unique()[:10]  # Show first 10 cities
        log.info(f"Cities include: {', '.join(cities_list)}{'...' if len(cities_list) == 10 else ''}")
    
    return filtered_grids


def filter_largest_cities(h3_grids: gpd.GeoDataFrame, n_cities: int = 10) -> gpd.GeoDataFrame:
    """
    Filter H3 grids to include only those from the N largest cities by total area.
    
    Args:
        h3_grids: GeoDataFrame with H3 grid polygons
        n_cities: Number of largest cities to keep (default: 10)
        
    Returns:
        Filtered GeoDataFrame with H3 grids from largest cities only
    """
    log.info(f"Filtering for {n_cities} largest cities by area...")
    
    if h3_grids.crs is None:
        log.warning("Input H3 grids have no CRS defined; skipping area-based filtering.")
        return h3_grids

    try:
        area_crs = 'EPSG:6933'  # World Cylindrical Equal Area
        h3_grids_with_area = h3_grids.to_crs(area_crs)
    except Exception as exc:
        log.warning(f"Failed to reproject geometries for area calculation ({exc}). Using original CRS; results may be inaccurate.")
        h3_grids_with_area = h3_grids

    # Calculate area for each H3 grid (in square meters)
    h3_grids_with_area = h3_grids_with_area.copy()
    h3_grids_with_area['grid_area_m2'] = h3_grids_with_area.geometry.area
    
    # Group by city and calculate total area per city
    # Assuming 'UC_NM_MN' is the city name column, adjust if different
    city_col = 'UC_NM_MN'
    if city_col not in h3_grids_with_area.columns:
        # Try alternative city identifier columns
        possible_city_cols = ['ID_HDC_G0', 'CTR_MN_NM']
        for col in possible_city_cols:
            if col in h3_grids_with_area.columns:
                city_col = col
                break
        else:
            log.warning("No city identifier column found. Using all data.")
            return h3_grids
    
    city_areas = h3_grids_with_area.groupby(city_col)['grid_area_m2'].sum().reset_index()
    city_areas = city_areas.sort_values('grid_area_m2', ascending=False)
    
    # Get the N largest cities
    largest_cities = city_areas.head(n_cities)[city_col].tolist()
    
    log.info(f"Selected {len(largest_cities)} largest cities:")
    for i, city in enumerate(largest_cities, 1):
        area_km2 = city_areas[city_areas[city_col] == city]['grid_area_m2'].iloc[0] / 1e6
        log.info(f"  {i}. {city}: {area_km2:.2f} km²")
    
    # Filter H3 grids to include only those from largest cities
    filtered_grids = h3_grids[h3_grids[city_col].isin(largest_cities)].copy()
    
    log.info(f"Filtered from {len(h3_grids)} to {len(filtered_grids)} H3 grids")
    return filtered_grids


def extract_road_data(debug_mode: bool = False, n_debug_cities: int = 10, debug_country: bool = False, country_iso: str = 'BGD', resolution: int = 6):
    """
    Extract road data from GRIP raster files for H3 neighborhoods

    This function extracts sum of pixel values from GRIP road length rasters
    for each H3 grid polygon using exactextract package

    Args:
        debug_mode: If True, process only the largest cities for faster debugging
        n_debug_cities: Number of largest cities to process in debug mode
        debug_country: If True, process only cities from a specific country
        country_iso: ISO 3-letter country code for country filtering (default: 'BGD')
        resolution: H3 resolution level (default: 6)

    Returns:
        pd.DataFrame: A data frame with road length data by road type for each H3 grid
    """

    # Get today's date in YYYY-MM-DD format
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Set up resolution-specific directories
    DATA_DIR = get_resolution_dir(PROCESSED_DIR, resolution)
    OUTPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution)

    # Define file paths
    input_gpkg = DATA_DIR / f"all_cities_h3_grids_resolution{resolution}.gpkg"

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if debug_country:
        output_gpkg = OUTPUT_DIR / f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_debug_{country_iso}_{today_date}.gpkg"
        output_csv = OUTPUT_DIR / f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_debug_{country_iso}_{today_date}.csv"
        log.info(f"DEBUG MODE: Processing cities from country {country_iso}")
    elif debug_mode:
        output_gpkg = OUTPUT_DIR / f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_debug_{n_debug_cities}cities_{today_date}.gpkg"
        output_csv = OUTPUT_DIR / f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_debug_{n_debug_cities}cities_{today_date}.csv"
        log.info(f"DEBUG MODE: Processing only {n_debug_cities} largest cities")
    else:
        output_gpkg = OUTPUT_DIR / f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_{today_date}.gpkg"
        output_csv = OUTPUT_DIR / f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_{today_date}.csv"
    
    excel_lane_data_path = "data/Rousseau et al 2022/es2c05255_si_001.xlsx"
    
    # Check if input file exists
    if not os.path.exists(input_gpkg):
        raise FileNotFoundError(f"Input file not found: {input_gpkg}")
    
    # Read H3 grid polygons
    print("Reading H3 grid polygons...")
    h3_grids = gpd.read_file(input_gpkg)
    
    # Apply debug filtering if enabled
    if debug_country:
        h3_grids = filter_cities_by_country(h3_grids, country_iso)
    elif debug_mode:
        h3_grids = filter_largest_cities(h3_grids, n_debug_cities)
    
    # Define GRIP raster file paths and corresponding road types
    grip_files = {
        'highway': "data/GRIP/GRIP4_density_tp1/grip4_tp1_length_m.tif",
        'primary': "data/GRIP/GRIP4_density_tp2/grip4_tp2_length_m.tif",
        'secondary': "data/GRIP/GRIP4_density_tp3/grip4_tp3_length_m.tif",
        'tertiary': "data/GRIP/GRIP4_density_tp4/grip4_tp4_length_m.tif",
        'local': "data/GRIP/GRIP4_density_tp5/grip4_tp5_length_m.tif"
    }
    
    # Road type mapping for metrics calculation
    road_type_mapping = {
        'tp1': 'highway',
        'tp2': 'primary', 
        'tp3': 'secondary',
        'tp4': 'tertiary',
        'tp5': 'local'
    }
    
    # Check if all GRIP files exist
    missing_files = [path for path in grip_files.values() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(f"Missing GRIP files: {', '.join(missing_files)}")
    
    # Initialize result dataframe with existing columns
    # Drop geometry and select specific columns
    result_df = h3_grids.drop(columns=['geometry']).copy()
    # select(h3index, ID_HDC_G0, UC_NM_MN, CTR_MN_ISO, CTR_MN_NM, GRGN_L1, GRGN_L2, E_KG_NM_LS)
    columns_to_keep = ['neighborhood_id', 'h3index', 'ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2', 'E_KG_NM_LS']
    # Only keep columns that exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in result_df.columns]
    result_df = result_df[existing_columns]
    
    # Rename E_KG_NM_LS to E_KG_NM_LST for consistency with expected output
    # if 'E_KG_NM_LS' in result_df.columns:
    #     result_df = result_df.rename(columns={'E_KG_NM_LS': 'E_KG_NM_LST'})
    
    # Extract road data for each road type
    for road_type, raster_file in grip_files.items():
        print(f"Processing {road_type} roads...")
        
        # Extract sum of pixel values for each polygon using exactextract
        extracted_values = exact_extract(
            raster_file,
            h3_grids,
            'sum',
            include_cols=['neighborhood_id'],
            progress=True,
            output='pandas'
        )
        
        # remove rows with repeated 'h3index' in extracted_values
        # extracted_values = extracted_values.drop_duplicates(subset=['neighborhood_id'])
        
        # Add to result dataframe
        column_name = f"sum_road_m_{road_type}"
        merge_df = extracted_values[['neighborhood_id', 'sum']].rename(columns={'sum': column_name})
        result_df = result_df.merge(merge_df, on='neighborhood_id', how='left')
        
        # Clean up
        gc.collect()
    
    # Initialize processors
    lane_processor = LaneDataProcessor()
    climate_processor = ClimateProcessor()
    metrics_calculator = RoadMetricsCalculator()
    
    # Process lane data
    print("Processing lane data...")
    lane_df = lane_processor.process_lane_data(
        Path(excel_lane_data_path), 
        'Number_lanes'
    )
    
    # Add lane data if available
    if not lane_df.empty and 'CTR_MN_ISO' in result_df.columns:
        print("Merging with lane data...")
        
        # Ensure ISO codes are comparable (e.g., uppercase, no whitespace)
        result_df['CTR_MN_ISO_upper'] = result_df['CTR_MN_ISO'].astype(str).str.upper().str.strip()
        lane_df['ISO_A3_lanes_upper'] = lane_df['ISO_A3_lanes'].astype(str).str.upper().str.strip()
        
        result_df = result_df.merge(
            lane_df, 
            left_on='CTR_MN_ISO_upper', 
            right_on='ISO_A3_lanes_upper', 
            how='left'
        )
        
        # Drop helper columns
        result_df.drop(columns=['CTR_MN_ISO_upper', 'ISO_A3_lanes_upper'], inplace=True, errors='ignore')
        if 'ISO_A3_lanes' in result_df.columns:
            result_df.drop(columns=['ISO_A3_lanes'], inplace=True, errors='ignore')
        
        # Calculate road metrics (width, area)
        print("Calculating road metrics (width, area)...")
        result_df = metrics_calculator.calculate_road_metrics(
            result_df, road_type_mapping
        )
    else:
        print("Skipping lane data merge and road metrics calculation")
    
    # Add climate data
    if 'E_KG_NM_LS' in result_df.columns:
        print("Adding climate classifications...")
        result_df = climate_processor.add_climate_classifications(
            result_df, climate_col='E_KG_NM_LS'
        )
    else:
        print("Skipping climate classification")
    
    # Ensure output directories exist
    Path(output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Create spatial dataframe for GPKG output
    result_sf = h3_grids[['h3index', 'geometry']].merge(result_df, on='h3index', how='left')
    
    # Write outputs
    # print("Writing GPKG output...")
    # result_sf.to_file(output_gpkg, driver='GPKG')
    
    print("Writing CSV output...")
    result_df.to_csv(output_csv, index=False)
    
    print("Extraction complete!")
    print(f"GPKG output: {output_gpkg}")
    print(f"CSV output: {output_csv}")
    print(f"Number of H3 grids processed: {len(result_df)}")
    
    return result_df


def main():
    """
    Main execution function

    Runs the road data extraction process
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract and analyze road data at the H3 neighborhood level.')
    parser.add_argument('--resolution', '-r', type=int, default=6,
                       help='H3 resolution level (default: 6)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to process only the largest cities')
    parser.add_argument('--debug-cities', type=int, default=10,
                       help='Number of largest cities to process in debug mode (default: 10)')
    parser.add_argument('--debug-country', action='store_true',
                       help='Enable debug mode to process cities from a specific country')
    parser.add_argument('--country', type=str, default='BGD',
                       help='ISO 3-letter country code for country filtering (default: BGD for Bangladesh)')

    # Parse arguments
    args = parser.parse_args()

    print(f"Starting GRIP road data extraction for H3 neighborhoods (resolution {args.resolution})...")

    # Set working directory to project root
    if os.path.basename(os.getcwd()) == "scripts":
        os.chdir("..")

    # Run extraction
    try:
        result = extract_road_data(
            debug_mode=args.debug,
            n_debug_cities=args.debug_cities,
            debug_country=args.debug_country,
            country_iso=args.country,
            resolution=args.resolution
        )
        print("\nExtraction completed successfully!")
        
        # Print summary statistics
        print("\nSummary of extracted road lengths (meters):")
        road_columns = [col for col in result.columns if col.startswith('sum_road_m_')]
        for col in road_columns:
            mean_val = result[col].mean()
            max_val = result[col].max()
            print(f"{col}: Mean = {mean_val:.2f}, Max = {max_val:.2f}")
            
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
