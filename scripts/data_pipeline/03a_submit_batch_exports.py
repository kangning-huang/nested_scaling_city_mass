#!/usr/bin/env python3
"""
03a_submit_batch_exports.py - Submit GEE batch export tasks for volume/pavement extraction.

This is an alternative to 03_extract_volume_pavement.py that uses GEE's batch export
system instead of synchronous API calls. This is 5-10x faster for large jobs.

Architecture:
    1. This script (03a): Submits export tasks to GEE
    2. 03b_monitor_batch_tasks.py: Monitors task completion
    3. 03c_download_batch_results.py: Downloads and combines results

Usage:
    # Submit all cities (will take ~1-2 hours to complete on GEE servers)
    python scripts/03a_submit_batch_exports.py --resolution 7

    # Submit specific country only
    python scripts/03a_submit_batch_exports.py --resolution 7 --country BGD

    # Submit in batches of N cities per export task
    python scripts/03a_submit_batch_exports.py --resolution 7 --cities-per-task 50
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import ee
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import get_resolution_dir

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
GEE_PROJECT = 'ee-knhuang'
DRIVE_FOLDER = 'GEE_Batch_Exports'  # Folder in Google Drive for exports


def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit GEE batch export tasks for volume/pavement extraction'
    )
    parser.add_argument('--resolution', '-r', type=int, default=6,
                        help='H3 resolution level (default: 6)')
    parser.add_argument('--country', type=str, default=None,
                        help='Country ISO code to filter (e.g., BGD, USA). Default: all countries')
    parser.add_argument('--cities-per-task', type=int, default=100,
                        help='Number of cities per export task (default: 100)')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Maximum number of tasks to submit (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without submitting tasks')
    return parser.parse_args()


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialized with project: {GEE_PROJECT}")
    except Exception:
        print("Authenticating with GEE...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


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

    return {
        'LW': lw_mask,
        'RS': rs_mask,
        'RM': rm_mask,
        'NR': nr_mask,
        'HR': hr_mask
    }


def create_esch2022_bands(impervious_2015):
    """
    Create Esch2022 WSF3D bands at native 90m resolution.
    Returns (image, band_names, scale).
    """
    height = ee.Image("projects/ardent-spot-390007/assets/Esch2022_WSF3D/WSF3D_V02_BuildingHeight").multiply(0.1)
    footprint = ee.Image("projects/ardent-spot-390007/assets/Esch2022_WSF3D/WSF3D_V02_BuildingFraction").divide(100).multiply(90*90)
    volume = height.multiply(footprint)
    classes = classify_pixels(height)

    bands = []
    band_names = []
    for cls_name, mask in classes.items():
        band = volume.updateMask(mask).updateMask(impervious_2015).unmask(0).rename(f'vol_Esch2022_{cls_name}')
        bands.append(band)
        band_names.append(f'vol_Esch2022_{cls_name}')

    bands.append(footprint.updateMask(impervious_2015).unmask(0).rename('footprint_Esch2022'))
    band_names.append('footprint_Esch2022')

    return ee.Image.cat(bands), band_names, 90


def create_li2022_bands(impervious_2015):
    """
    Create Li2022 Global3DBuiltup bands at native 1000m resolution.
    Returns (image, band_names, scale).
    """
    height = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/height_mean")
    volume = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/volume_mean").multiply(100000)
    footprint = ee.Image("projects/ardent-spot-390007/assets/Li2022_Global3DBuiltup/footprint_mean").multiply(1000*1000)
    classes = classify_pixels(height)

    bands = []
    band_names = []
    for cls_name, mask in classes.items():
        band = volume.updateMask(mask).updateMask(impervious_2015).unmask(0).rename(f'vol_Li2022_{cls_name}')
        bands.append(band)
        band_names.append(f'vol_Li2022_{cls_name}')

    bands.append(footprint.updateMask(impervious_2015).unmask(0).rename('footprint_Li2022'))
    band_names.append('footprint_Li2022')

    return ee.Image.cat(bands), band_names, 1000


def create_liu2024_bands(impervious_2015):
    """
    Create Liu2024 GUS3D bands at native 500m resolution.
    Returns (image, band_names, scale).
    """
    volume = ee.Image("projects/ardent-spot-390007/assets/Liu2024_GlobalUrbanStructure_3D/GUS3D_Volume")
    height = ee.Image("projects/ardent-spot-390007/assets/Liu2024_GlobalUrbanStructure_3D/GUS3D_Height")
    footprint = volume.divide(height)
    classes = classify_pixels(height)

    bands = []
    band_names = []
    for cls_name, mask in classes.items():
        band = volume.updateMask(mask).updateMask(impervious_2015).unmask(0).rename(f'vol_Liu2024_{cls_name}')
        bands.append(band)
        band_names.append(f'vol_Liu2024_{cls_name}')

    bands.append(footprint.updateMask(impervious_2015).unmask(0).rename('footprint_Liu2024'))
    band_names.append('footprint_Liu2024')

    return ee.Image.cat(bands), band_names, 500


def create_other_bands(impervious_2015):
    """
    Create population (100m) and impervious surface bands.
    Returns (image, band_names, scale).
    """
    population = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2015-01-01', '2015-12-31').mosaic()
    pop_band = population.updateMask(impervious_2015).unmask(0).rename('population_2015')
    # Impervious: Each GISA pixel is 10m x 10m = 100 m^2
    imp_band = impervious_2015.multiply(100).unmask(0).rename('impervious_2015')

    return ee.Image.cat([pop_band, imp_band]), ['population_2015', 'impervious_2015'], 100


def get_all_dataset_configs(impervious_2015):
    """
    Get all dataset configurations with their native scales.
    Returns list of (dataset_name, image, band_names, scale).
    """
    configs = []

    esch_img, esch_names, esch_scale = create_esch2022_bands(impervious_2015)
    configs.append(('esch2022', esch_img, esch_names, esch_scale))

    li_img, li_names, li_scale = create_li2022_bands(impervious_2015)
    configs.append(('li2022', li_img, li_names, li_scale))

    liu_img, liu_names, liu_scale = create_liu2024_bands(impervious_2015)
    configs.append(('liu2024', liu_img, liu_names, liu_scale))

    other_img, other_names, other_scale = create_other_bands(impervious_2015)
    configs.append(('other', other_img, other_names, other_scale))

    return configs


def load_h3_grids(resolution: int) -> gpd.GeoDataFrame:
    """Load pre-generated H3 grids for all cities."""
    grids_file = get_resolution_dir(PROCESSED_DIR, resolution) / f"all_cities_h3_grids_resolution{resolution}.gpkg"

    if not grids_file.exists():
        raise FileNotFoundError(
            f"H3 grids file not found: {grids_file}\n"
            f"Run 01_create_h3_grids.py --resolution {resolution} first."
        )

    print(f"Loading H3 grids from: {grids_file}")
    gdf = gpd.read_file(grids_file)
    print(f"Loaded {len(gdf)} H3 cells")
    return gdf


def load_cities() -> gpd.GeoDataFrame:
    """Load city boundaries from GEE asset."""
    print("Loading city boundaries from GEE...")
    fc_UCs = ee.FeatureCollection("users/kh3657/GHS_STAT_UCDB2015")

    # Get as GeoDataFrame
    import geemap
    gdf = geemap.ee_to_gdf(fc_UCs)
    print(f"Loaded {len(gdf)} cities")
    return gdf


def gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """Convert GeoDataFrame to Earth Engine FeatureCollection."""
    # Ensure we have required columns
    if 'h3index' not in gdf.columns:
        if gdf.index.name == 'h3index' or gdf.index.name == 'hex_id':
            gdf = gdf.reset_index()
            if 'hex_id' in gdf.columns:
                gdf = gdf.rename(columns={'hex_id': 'h3index'})

    # Convert to GeoJSON
    geojson = json.loads(gdf[['h3index', 'geometry']].to_json())
    return ee.FeatureCollection(geojson)


def submit_export_task(
    fc_hex: ee.FeatureCollection,
    image: ee.Image,
    task_name: str,
    description: str,
    scale: int,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Submit a single export task to GEE.

    Uses reduceRegions to compute sum of all bands for each H3 cell,
    then exports results to Google Drive.

    Args:
        fc_hex: FeatureCollection of H3 hexagons
        image: Multi-band image to reduce
        task_name: Name for the export task
        description: Human-readable description
        scale: Native resolution scale in meters for reduceRegions
        dry_run: If True, don't actually submit the task
    """
    # Compute zonal statistics (sum) for all bands at native scale
    stats = image.reduceRegions(
        collection=fc_hex,
        reducer=ee.Reducer.sum(),
        scale=scale,  # Use dataset-native scale
        tileScale=4   # Increase for memory issues
    )

    if dry_run:
        print(f"  [DRY RUN] Would submit task: {task_name} (scale={scale}m)")
        return {'task_id': None, 'name': task_name, 'status': 'dry_run', 'scale': scale}

    # Submit export task
    task = ee.batch.Export.table.toDrive(
        collection=stats,
        description=task_name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=task_name,
        fileFormat='CSV'
    )

    task.start()

    return {
        'task_id': task.id,
        'name': task_name,
        'description': description,
        'scale': scale,
        'status': 'SUBMITTED',
        'submitted_at': datetime.now().isoformat()
    }


def main():
    args = parse_args()
    resolution = args.resolution

    print("="*70)
    print("GEE BATCH EXPORT - Volume/Pavement Extraction")
    print("="*70)
    print(f"Resolution: {resolution}")
    print(f"Cities per task: {args.cities_per_task}")
    print(f"Country filter: {args.country or 'ALL'}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize GEE
    initialize_gee()

    # Set up output directory
    output_dir = get_resolution_dir(PROCESSED_DIR, resolution)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load H3 grids
    h3_gdf = load_h3_grids(resolution)

    # Filter by country if specified
    if args.country:
        if 'CTR_MN_ISO' in h3_gdf.columns:
            h3_gdf = h3_gdf[h3_gdf['CTR_MN_ISO'] == args.country]
            print(f"Filtered to {len(h3_gdf)} H3 cells in {args.country}")
        else:
            print("Warning: CTR_MN_ISO column not found, cannot filter by country")

    # Get unique city IDs
    city_ids = h3_gdf['ID_HDC_G0'].unique()
    print(f"Total cities to process: {len(city_ids)}")

    # Create dataset configurations with native scales
    print("\nCreating dataset configurations with native scales...")
    impervious_2015 = get_impervious_2015()
    dataset_configs = get_all_dataset_configs(impervious_2015)

    all_band_names = []
    print("Datasets and scales:")
    for ds_name, ds_image, ds_bands, ds_scale in dataset_configs:
        print(f"  {ds_name}: {len(ds_bands)} bands, scale={ds_scale}m")
        all_band_names.extend(ds_bands)

    # Split cities into batches
    n_batches = (len(city_ids) + args.cities_per_task - 1) // args.cities_per_task
    n_datasets = len(dataset_configs)
    total_tasks = n_batches * n_datasets
    print(f"\nWill submit {n_batches} batches × {n_datasets} datasets = {total_tasks} export tasks")

    if args.max_tasks:
        max_batches = args.max_tasks // n_datasets
        n_batches = min(n_batches, max_batches)
        print(f"Limited to {n_batches} batches (--max-tasks)")

    # Submit tasks
    task_records = []
    today = datetime.now().strftime('%Y%m%d')

    for i in tqdm(range(n_batches), desc="Submitting batches"):
        start_idx = i * args.cities_per_task
        end_idx = min((i + 1) * args.cities_per_task, len(city_ids))
        batch_city_ids = city_ids[start_idx:end_idx]

        # Filter H3 cells for this batch of cities
        batch_h3 = h3_gdf[h3_gdf['ID_HDC_G0'].isin(batch_city_ids)]

        if len(batch_h3) == 0:
            print(f"  Warning: No H3 cells for batch {i+1}, skipping")
            continue

        # Convert to EE FeatureCollection
        fc_hex = gdf_to_ee_fc(batch_h3)

        # Submit one task per dataset at its native scale
        for ds_name, ds_image, ds_bands, ds_scale in dataset_configs:
            country_suffix = f"_{args.country}" if args.country else ""
            task_name = f"vol_pav_r{resolution}{country_suffix}_{ds_name}_batch{i+1:04d}_{today}"
            description = f"{ds_name} - Cities {start_idx+1}-{end_idx} of {len(city_ids)}"

            try:
                record = submit_export_task(
                    fc_hex=fc_hex,
                    image=ds_image,
                    task_name=task_name,
                    description=description,
                    scale=ds_scale,
                    dry_run=args.dry_run
                )
                record['batch_num'] = i + 1
                record['dataset'] = ds_name
                record['band_names'] = ds_bands
                record['city_ids'] = batch_city_ids.tolist()
                record['n_h3_cells'] = len(batch_h3)
                task_records.append(record)
            except Exception as e:
                print(f"  Error submitting {ds_name} batch {i+1}: {e}")
                task_records.append({
                    'batch_num': i + 1,
                    'dataset': ds_name,
                    'name': task_name,
                    'status': 'ERROR',
                    'error': str(e)
                })

    # Save task manifest
    manifest_file = output_dir / f"batch_export_manifest_r{resolution}_{today}.json"
    manifest = {
        'resolution': resolution,
        'country': args.country,
        'cities_per_task': args.cities_per_task,
        'total_cities': len(city_ids),
        'total_h3_cells': len(h3_gdf),
        'n_batches': n_batches,
        'n_datasets': n_datasets,
        'n_tasks': len(task_records),
        'datasets': [{'name': ds[0], 'bands': ds[2], 'scale': ds[3]} for ds in dataset_configs],
        'all_band_names': all_band_names,
        'drive_folder': DRIVE_FOLDER,
        'submitted_at': datetime.now().isoformat(),
        'tasks': task_records
    }

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("SUBMISSION COMPLETE")
    print(f"{'='*70}")
    n_success = len([t for t in task_records if t.get('status') != 'ERROR'])
    print(f"Tasks submitted: {n_success} ({n_batches} batches × {n_datasets} datasets)")
    print(f"Manifest saved to: {manifest_file}")
    print(f"\nNext steps:")
    print(f"  1. Monitor tasks: python scripts/03b_monitor_batch_tasks.py --resolution {resolution}")
    print(f"  2. Download results: python scripts/03c_download_batch_results.py --resolution {resolution}")
    print(f"\nExports will appear in Google Drive folder: {DRIVE_FOLDER}/")


if __name__ == "__main__":
    main()
