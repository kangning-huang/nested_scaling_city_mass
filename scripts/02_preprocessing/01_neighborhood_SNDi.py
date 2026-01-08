#!/usr/bin/env python3
"""
01_neighborhood_SNDi.py

Calculate average SNDi (Street Network Disconnectedness Index) values for each 
H3 hexagon by spatially joining with SNDi grid data.

This script:
1. Loads H3 hexagon grids from all_cities_h3_grids.gpkg
2. Loads SNDi grid polygons from sndi_grid_as_polygons_UrbanCores_intersect.gpkg
3. Converts SNDi polygons to centroids for efficient spatial operations
4. Performs spatial joins to calculate weighted average SNDi per hexagon
5. Exports results as CSV

The SNDi values come from the 'pca1' column, representing the first principal 
component of street network disconnectedness metrics.

Author: Generated for NYU China Grant project
Date: 2025
"""

import os
import sys
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import argparse
import warnings
from shapely.geometry import Point
warnings.filterwarnings('ignore')

# Try to import raster processing libraries
try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.features import rasterize
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Try to import geemap for zonal statistics (following existing pattern)
try:
    import geemap
    GEEMAP_AVAILABLE = True
except ImportError:
    GEEMAP_AVAILABLE = False

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Try to import optional dependencies, fall back gracefully if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        log.info(f"Processing: {desc}" if desc else "Processing...")
        return iterable

# Try to import scikit-learn for distance-based matching
try:
    from sklearn.neighbors import BallTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not available - distance-based matching will be disabled")

# Check for rasterio dependency (will handle gracefully if not available)
if not RASTERIO_AVAILABLE:
    log.warning("rasterio is not available - raster processing will be limited")
    log.warning("For full functionality, install rasterio: pip install rasterio")

# Define base paths (reorganized for clean project structure)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, "data")
raw_data_dir = os.path.join(data_dir, "raw")
processed_data_dir = os.path.join(data_dir, "processed")

# Input data paths
h3_input_gpkg = os.path.join(raw_data_dir, "all_cities_h3_grids.gpkg")
sndi_input_raster = os.path.join(raw_data_dir, "sndi_grid_in_UrbanCores.tif")
output_dir = processed_data_dir

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    log.info(f"Created output directory: {output_dir}")

def load_h3_grids(input_file: str, country_filter: Optional[str] = None, debug_sample: Optional[int] = None) -> gpd.GeoDataFrame:
    """
    Load H3 hexagon grids from GPKG file.
    
    Args:
        input_file: Path to the all_cities_h3_grids.gpkg file
        country_filter: Optional ISO 3-letter country code to filter data (e.g., 'CHN')
        debug_sample: Optional number of hexagons to sample for testing
        
    Returns:
        GeoDataFrame containing H3 grid polygons with city metadata
    """
    log.info(f"Loading H3 grids from: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        h3_grids = gpd.read_file(input_file)
        log.info(f"Loaded {len(h3_grids)} H3 grid polygons")
        
        # Log available columns
        log.info(f"Available columns: {h3_grids.columns.tolist()}")
        
        # Apply country filter if specified
        if country_filter and 'CTR_MN_ISO' in h3_grids.columns:
            h3_grids = h3_grids[h3_grids['CTR_MN_ISO'] == country_filter]
            log.info(f"Filtered to {len(h3_grids)} H3 grids for country: {country_filter}")
            
        # Apply debug sampling if specified
        if debug_sample:
            # In debug mode, focus on Shanghai specifically
            shanghai_grids = h3_grids[h3_grids['UC_NM_MN'].str.contains('Shanghai', case=False, na=False)]
            
            if len(shanghai_grids) > 0:
                # Use Shanghai data, sample if needed
                if len(shanghai_grids) > debug_sample:
                    h3_grids = shanghai_grids.sample(n=debug_sample, random_state=42)
                else:
                    h3_grids = shanghai_grids
                log.info(f"DEBUG: Using Shanghai data - {len(h3_grids)} hexagons")
            else:
                # Fallback: look for other major Chinese cities
                chinese_cities = h3_grids[h3_grids['CTR_MN_ISO'] == 'CHN']
                if len(chinese_cities) > 0:
                    if len(chinese_cities) > debug_sample:
                        h3_grids = chinese_cities.sample(n=debug_sample, random_state=42)
                    else:
                        h3_grids = chinese_cities
                    log.info(f"DEBUG: Shanghai not found, using Chinese cities - {len(h3_grids)} hexagons")
                else:
                    # Final fallback: random sample
                    h3_grids = h3_grids.sample(n=min(debug_sample, len(h3_grids)), random_state=42)
                    log.info(f"DEBUG: No Shanghai or Chinese cities found, using random sample - {len(h3_grids)} hexagons")
            
        # Ensure CRS is WGS84
        if h3_grids.crs != 'EPSG:4326':
            h3_grids = h3_grids.to_crs('EPSG:4326')
            log.info("Reprojected H3 grids to WGS84 (EPSG:4326)")
            
        return h3_grids
        
    except Exception as e:
        log.error(f"Error loading H3 grids: {e}")
        raise

def load_sndi_raster(input_file: str) -> tuple:
    """
    Load SNDi raster data and return raster dataset info.
    
    Args:
        input_file: Path to the SNDi raster file
        
    Returns:
        Tuple containing (raster_path, dataset_info_dict)
    """
    log.info(f"Loading SNDi raster from: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"SNDi raster file not found: {input_file}")
    
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for raster processing but not available")
    
    try:
        with rasterio.open(input_file) as src:
            dataset_info = {
                'shape': src.shape,
                'crs': src.crs,
                'bounds': src.bounds,
                'dtype': src.dtypes[0],
                'count': src.count,
                'nodata': src.nodata,
                'transform': src.transform
            }
            
            log.info(f"Raster properties:")
            log.info(f"  Shape: {dataset_info['shape']}")
            log.info(f"  CRS: {dataset_info['crs']}")
            log.info(f"  Bounds: {dataset_info['bounds']}")
            log.info(f"  Data type: {dataset_info['dtype']}")
            log.info(f"  NoData value: {dataset_info['nodata']}")
            
            # Read a sample to check data range
            sample_window = rasterio.windows.Window(0, 0, min(1000, src.width), min(1000, src.height))
            sample = src.read(1, window=sample_window)
            
            # Get valid (non-nodata) values
            if dataset_info['nodata'] is not None:
                valid_sample = sample[sample != dataset_info['nodata']]
            else:
                valid_sample = sample[~np.isnan(sample)]
            
            if len(valid_sample) > 0:
                log.info(f"  Sample data statistics:")
                log.info(f"    Valid pixels: {len(valid_sample)}/{sample.size}")
                log.info(f"    Min: {valid_sample.min():.3f}")
                log.info(f"    Max: {valid_sample.max():.3f}")  
                log.info(f"    Mean: {valid_sample.mean():.3f}")
                log.info(f"    Std: {valid_sample.std():.3f}")
            else:
                log.warning("No valid data found in sample area")
        
        return input_file, dataset_info
        
    except Exception as e:
        log.error(f"Error loading SNDi raster: {e}")
        raise



def perform_zonal_statistics_raster(h3_grids: gpd.GeoDataFrame, 
                                   sndi_raster_path: str,
                                   chunk_size: int = 1000) -> pd.DataFrame:
    """
    Calculate zonal statistics to get average SNDi values for each H3 hexagon
    using raster data and chunked processing.
    
    Args:
        h3_grids: GeoDataFrame with H3 hexagon polygons
        sndi_raster_path: Path to SNDi raster file
        chunk_size: Number of hexagons to process per chunk
        
    Returns:
        DataFrame with averaged SNDi values per hexagon
    """
    total_hexagons = len(h3_grids)
    log.info(f"Performing zonal statistics for {total_hexagons} hexagons using raster data")
    log.info(f"Processing in chunks of {chunk_size}")
    
    all_results = []
    
    try:
        # Process hexagons in chunks to avoid memory issues
        for i in tqdm(range(0, total_hexagons, chunk_size), desc="Processing chunks"):
            end_idx = min(i + chunk_size, total_hexagons)
            chunk_grids = h3_grids.iloc[i:end_idx].copy()
            
            log.info(f"Processing chunk {i//chunk_size + 1}/{(total_hexagons-1)//chunk_size + 1}: hexagons {i+1}-{end_idx}")
            
            try:
                # Method 1: Try using rasterio with zonal stats
                chunk_results = calculate_zonal_stats_rasterio(chunk_grids, sndi_raster_path)
                
                if chunk_results is not None and len(chunk_results) > 0:
                    all_results.append(chunk_results)
                    valid_count = chunk_results['avg_sndi'].notna().sum()
                    log.info(f"Chunk {i//chunk_size + 1} completed: {valid_count}/{len(chunk_results)} hexagons with SNDi data")
                else:
                    log.warning(f"Chunk {i//chunk_size + 1} produced no results")
                    # Create empty results
                    empty_results = chunk_grids[['h3index']].copy()
                    empty_results['avg_sndi'] = np.nan
                    empty_results['sndi_point_count'] = 0
                    all_results.append(empty_results)
                    
            except Exception as e:
                log.error(f"Error processing chunk {i//chunk_size + 1}: {e}")
                # Create empty results for failed chunk
                empty_results = chunk_grids[['h3index']].copy()
                empty_results['avg_sndi'] = np.nan
                empty_results['sndi_point_count'] = 0
                all_results.append(empty_results)
        
        # Combine all results
        if all_results:
            result_df = pd.concat(all_results, ignore_index=True)
            log.info(f"Combined all chunks: {len(result_df)} total records")
            
            # Add metadata from original H3 grids
            metadata_cols = [col for col in h3_grids.columns if col not in ['geometry']]
            h3_metadata = h3_grids[metadata_cols]
            
            result_df = result_df.merge(h3_metadata, on='h3index', how='left')
            
            # Log summary statistics
            valid_ages = result_df['avg_sndi'].dropna()
            log.info(f"Valid SNDi calculations: {len(valid_ages)}/{len(result_df)} records")
            
            return result_df
        else:
            raise ValueError("No results from any chunks")
        
    except Exception as e:
        log.error(f"Error in zonal statistics: {e}")
        raise

def calculate_zonal_stats_rasterio(hexagons: gpd.GeoDataFrame, 
                                  raster_path: str) -> pd.DataFrame:
    """
    Calculate zonal statistics using rasterio for a batch of hexagons.
    
    Args:
        hexagons: GeoDataFrame with hexagon polygons (should be in WGS84)
        raster_path: Path to the SNDi raster
        
    Returns:
        DataFrame with h3index and avg_sndi columns
    """
    try:
        with rasterio.open(raster_path) as src:
            results = []
            
            # Ensure hexagons are in the same CRS as the raster
            if hexagons.crs != src.crs:
                log.info(f"Reprojecting hexagons from {hexagons.crs} to {src.crs} for raster processing")
                hexagons_reproj = hexagons.to_crs(src.crs)
            else:
                hexagons_reproj = hexagons
            
            # Reset index to ensure alignment between original and reprojected
            hexagons_orig = hexagons.reset_index(drop=True)
            hexagons_reproj = hexagons_reproj.reset_index(drop=True)
            
            for idx, row in hexagons_reproj.iterrows():
                h3index = hexagons_orig.iloc[idx]['h3index']  # Get h3index from original
                geom = row['geometry']
                
                try:
                    # Extract raster values for this polygon
                    masked_data, transform = mask(src, [geom], crop=True, nodata=src.nodata)
                    masked_array = masked_data[0]  # Get first (and only) band
                    
                    # Remove nodata values (0.0 in this case)
                    if src.nodata is not None:
                        valid_data = masked_array[masked_array != src.nodata]
                    else:
                        valid_data = masked_array[~np.isnan(masked_array)]
                    
                    # Also remove any additional invalid values (negative values seem unlikely for SNDi)
                    valid_data = valid_data[valid_data > 0]
                    
                    if len(valid_data) > 0:
                        avg_sndi = float(np.mean(valid_data))
                        point_count = len(valid_data)
                        log.debug(f"Hexagon {h3index}: {point_count} valid pixels, mean SNDi = {avg_sndi:.3f}")
                    else:
                        avg_sndi = np.nan
                        point_count = 0
                        log.debug(f"Hexagon {h3index}: no valid data")
                    
                    results.append({
                        'h3index': h3index,
                        'avg_sndi': avg_sndi,
                        'sndi_point_count': point_count
                    })
                    
                except Exception as e:
                    log.debug(f"Error processing hexagon {h3index}: {e}")
                    results.append({
                        'h3index': h3index,
                        'avg_sndi': np.nan,
                        'sndi_point_count': 0
                    })
            
            return pd.DataFrame(results)
            
    except Exception as e:
        log.error(f"Error in rasterio zonal stats: {e}")
        return None

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate average SNDi values for H3 hexagons')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with small sample')
    parser.add_argument('--debug-sample', type=int, default=50, help='Number of hexagons for debug sample (default: 50)')
    parser.add_argument('--country', type=str, help='Filter to specific country (e.g., CHN, USA)')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of cities to process per batch (default: 50)')
    args = parser.parse_args()
    
    log.info("Starting SNDi calculation for H3 hexagons")
    
    # Get today's date for output filename
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Set output filename based on mode
    if args.debug:
        output_csv = os.path.join(output_dir, f"01_neighborhood_SNDi_debug_{args.debug_sample}samples_{today_date}.csv")
        log.info(f"DEBUG MODE: Processing sample of {args.debug_sample} hexagons")
    elif args.country:
        output_csv = os.path.join(output_dir, f"01_neighborhood_SNDi_{args.country}_{today_date}.csv")
    else:
        output_csv = os.path.join(output_dir, f"01_neighborhood_SNDi_{today_date}.csv")
    
    try:
        # Step 1: Load H3 grids
        log.info("Step 1: Loading H3 hexagon grids")
        debug_sample = args.debug_sample if args.debug else None
        h3_grids = load_h3_grids(h3_input_gpkg, country_filter=args.country, debug_sample=debug_sample)
        
        # Step 2: Load SNDi raster data
        log.info("Step 2: Loading SNDi raster data")
        sndi_raster_path, raster_info = load_sndi_raster(sndi_input_raster)
        
        # Step 3: Perform zonal statistics
        log.info("Step 3: Calculating average SNDi per hexagon using zonal statistics")
        results_df = perform_zonal_statistics_raster(h3_grids, sndi_raster_path, chunk_size=args.batch_size)
        
        # Step 5: Save results
        log.info(f"Step 5: Saving results to {output_csv}")
        results_df.to_csv(output_csv, index=False)
        log.info(f"Results saved successfully. Total records: {len(results_df)}")
        
        # Log summary statistics
        if 'avg_sndi' in results_df.columns:
            # Overall statistics
            valid_count = results_df['avg_sndi'].notna().sum()
            total_count = len(results_df)
            
            log.info(f"Summary statistics:")
            log.info(f"  Total hexagons: {total_count}")
            log.info(f"  Hexagons with SNDi data: {valid_count} ({valid_count/total_count*100:.1f}%)")
            
            if valid_count > 0:
                sndi_stats = results_df['avg_sndi'].describe()
                log.info(f"  SNDi value statistics:\n{sndi_stats}")
                
                # Point count statistics
                if 'sndi_point_count' in results_df.columns:
                    point_stats = results_df['sndi_point_count'].describe()
                    log.info(f"  SNDi points per hexagon:\n{point_stats}")
        
        log.info("SNDi calculation completed successfully!")
        
    except Exception as e:
        log.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()