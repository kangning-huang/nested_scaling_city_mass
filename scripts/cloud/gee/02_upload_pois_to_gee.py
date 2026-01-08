"""
Script to upload POI data for 9 cities to Google Earth Engine as FeatureCollections

For large datasets, this uses the recommended GCS → GEE import workflow.

Usage:
    python 02_upload_pois_to_gee.py
    python 02_upload_pois_to_gee.py --use-direct  # For small cities only
"""

import ee
import geopandas as gpd
import json
from pathlib import Path
import sys
import argparse
import tempfile
import subprocess
from google.cloud import storage

# GEE project ID
GEE_PROJECT = 'ee-knhuang'

# GCS bucket (change this to your bucket)
GCS_BUCKET = 'ee-knhuang-uploads'  # Change if needed

# Cities to process - will auto-detect all available POI files
CITIES = []  # Auto-populated in main()

def get_available_cities(data_dir='../data/raw/pois'):
    """Get list of cities with POI files (excluding empty files)"""
    poi_dir = Path(data_dir)
    cities = []
    for poi_file in poi_dir.glob('*_pois_9cats.gpkg'):
        # Skip empty files (less than 1KB)
        if poi_file.stat().st_size < 1024:
            continue
        city_name = poi_file.stem.replace('_pois_9cats', '')
        cities.append(city_name)
    return sorted(cities)

def validate_geometry(geom):
    """Validate and fix geometry if needed"""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
        if not geom.is_valid:
            return None
    return geom

def check_asset_exists(asset_id):
    """Check if asset already exists in GEE"""
    earthengine_cmd = '/Users/kangninghuang/.venvs/nyu_china_grant_env/bin/earthengine'
    result = subprocess.run(
        [earthengine_cmd, 'asset', 'info', asset_id],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def upload_via_gcs(city_name, data_dir='../data/raw/pois'):
    """
    Upload POI data via Google Cloud Storage (recommended for large datasets)

    Parameters:
    -----------
    city_name : str
        Name of the city
    data_dir : str
        Directory containing POI geopackage files

    Returns:
    --------
    task : ee.batch.Task
        GEE import task object
    """
    poi_path = Path(data_dir) / f'{city_name}_pois_9cats.gpkg'
    asset_id = f'projects/{GEE_PROJECT}/assets/global_cities_POIs/{city_name}_pois_9cats'

    if not poi_path.exists():
        print(f"ERROR: POI file not found for {city_name}: {poi_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Processing: {city_name}")
    print(f"{'='*60}")

    # Check if already exists
    if check_asset_exists(asset_id):
        print(f"✓ Asset already exists, skipping: {asset_id}")
        return 'exists'

    # Read POI data
    print(f"Reading {poi_path}...")
    gdf = gpd.read_file(poi_path)
    print(f"  Total features: {len(gdf):,}")
    print(f"  Categories: {gdf['category'].unique().tolist()}")

    # Validate geometries
    print("Validating geometries...")
    gdf['geometry'] = gdf['geometry'].apply(validate_geometry)
    gdf = gdf[gdf['geometry'].notna()].reset_index(drop=True)
    print(f"  Valid geometries: {len(gdf):,}")

    # Convert to WGS84 if needed
    if gdf.crs and gdf.crs != 'EPSG:4326':
        print(f"  Converting from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs('EPSG:4326')

    # Convert all geometries to centroids (points) to avoid mixed geometry issues
    print("Converting to point geometries (centroids)...")
    gdf['geometry'] = gdf['geometry'].centroid

    # Create temporary shapefile and ZIP it
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        shp_base = tmpdir_path / f'{city_name}_pois_9cats'
        shp_path = tmpdir_path / f'{city_name}_pois_9cats.shp'
        zip_path = tmpdir_path / f'{city_name}_pois_9cats.zip'

        print(f"Creating temporary shapefile...")
        gdf.to_file(shp_path, driver='ESRI Shapefile')

        # Create ZIP file with all shapefile components
        import zipfile
        print(f"Creating ZIP archive...")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                file_path = tmpdir_path / f'{city_name}_pois_9cats{ext}'
                if file_path.exists():
                    zipf.write(file_path, f'{city_name}_pois_9cats{ext}')

        # Upload to GCS using Python library
        gcs_blob_name = f'poi_uploads/{city_name}_pois_9cats.zip'
        gcs_uri = f'gs://{GCS_BUCKET}/{gcs_blob_name}'
        print(f"Uploading to GCS: {gcs_uri}")

        try:
            storage_client = storage.Client(project=GEE_PROJECT)
            bucket = storage_client.bucket(GCS_BUCKET)

            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(str(zip_path))
            print(f"  ✓ Uploaded {zip_path.name}")

            print(f"  Upload complete")
        except Exception as e:
            print(f"  ERROR uploading to GCS: {e}")
            return None

    # Start GEE ingestion task using earthengine CLI
    asset_id = f'projects/{GEE_PROJECT}/assets/global_cities_POIs/{city_name}_pois_9cats'

    print(f"Starting GEE ingestion task...")
    print(f"  Source: {gcs_uri}")
    print(f"  Destination: {asset_id}")

    try:
        # Use earthengine CLI to trigger ingestion
        earthengine_cmd = '/Users/kangninghuang/.venvs/nyu_china_grant_env/bin/earthengine'
        result = subprocess.run(
            [earthengine_cmd, 'upload', 'table',
             '--asset_id', asset_id,
             gcs_uri],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"  ✓ Upload task started successfully")
            if result.stdout.strip():
                print(f"  Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ✗ Upload failed (exit code: {result.returncode})")
            if result.stdout.strip():
                print(f"  Stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"  Stderr: {result.stderr.strip()}")
            return None

    except Exception as e:
        print(f"  ERROR starting ingestion: {e}")
        print(f"\n  Manual command:")
        print(f"    earthengine upload table --asset_id={asset_id} {gcs_uri}")
        return None

def upload_direct(city_name, data_dir='../data/raw/pois'):
    """
    Upload POI data directly (only for small datasets <10MB)

    Parameters:
    -----------
    city_name : str
        Name of the city
    data_dir : str
        Directory containing POI geopackage files

    Returns:
    --------
    task : ee.batch.Task
        GEE export task object
    """
    import pandas as pd
    from datetime import datetime

    poi_path = Path(data_dir) / f'{city_name}_pois_9cats.gpkg'

    if not poi_path.exists():
        print(f"ERROR: POI file not found for {city_name}: {poi_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Processing: {city_name}")
    print(f"{'='*60}")

    # Read POI data
    print(f"Reading {poi_path}...")
    gdf = gpd.read_file(poi_path)
    print(f"  Total features: {len(gdf):,}")

    # Validate geometries
    print("Validating geometries...")
    valid_count = 0
    features = []

    for idx, row in gdf.iterrows():
        geom = validate_geometry(row.geometry)
        if geom is None:
            continue

        valid_count += 1
        props = row.drop('geometry').to_dict()

        # Convert non-JSON serializable types
        for key, value in props.items():
            if pd.isna(value):
                props[key] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                props[key] = str(value)

        features.append(ee.Feature(ee.Geometry(geom.__geo_interface__), props))

    print(f"  Valid geometries: {valid_count:,}")

    if not features:
        print(f"ERROR: No valid features found for {city_name}")
        return None

    fc = ee.FeatureCollection(features)
    asset_id = f'projects/{GEE_PROJECT}/assets/global_cities_POIs/{city_name}_pois_9cats'

    print(f"Uploading to GEE asset: {asset_id}")

    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=f'Upload_{city_name}_POIs',
        assetId=asset_id
    )
    task.start()

    print(f"  Task ID: {task.id}")
    print(f"  Task Status: {task.status()['state']}")

    return task

def check_gcs_bucket():
    """Check if GCS bucket exists and is accessible, create if needed"""
    try:
        storage_client = storage.Client(project=GEE_PROJECT)
        bucket = storage_client.bucket(GCS_BUCKET)

        if not bucket.exists():
            print(f"\n⚠️  GCS bucket does not exist: gs://{GCS_BUCKET}/")
            print(f"  Creating bucket...")
            bucket = storage_client.create_bucket(GCS_BUCKET, location='US')
            print(f"  ✓ Bucket created: gs://{GCS_BUCKET}/")
        else:
            print(f"\n✓ GCS bucket accessible: gs://{GCS_BUCKET}/")

        return True
    except Exception as e:
        print(f"\nERROR: Cannot access/create GCS bucket: {e}")
        print("  Options:")
        print(f"  1. Authenticate: gcloud auth application-default login")
        print(f"  2. Or use --use-direct flag for small cities (<5000 features)")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload POI data to Google Earth Engine')
    parser.add_argument('--use-direct', action='store_true',
                       help='Use direct upload (only for small datasets)')
    parser.add_argument('--city', type=str,
                       help='Upload single city only')
    args = parser.parse_args()

    print("="*60)
    print("POI Upload to Google Earth Engine")
    print("="*60)

    # Initialize GEE
    print("\nInitializing Google Earth Engine...")
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"  Connected to project: {GEE_PROJECT}")
    except Exception as e:
        print(f"ERROR: Failed to initialize GEE: {e}")
        sys.exit(1)

    # Check upload method
    if args.use_direct:
        print("\n⚠ Using DIRECT upload (limited to small datasets)")
        upload_func = upload_direct
    else:
        print("\nUsing GCS → GEE upload (recommended for large datasets)")
        if not check_gcs_bucket():
            sys.exit(1)
        upload_func = upload_via_gcs

    # Determine cities to process
    if args.city:
        cities_to_process = [args.city]
    else:
        # Auto-detect all available cities
        cities_to_process = get_available_cities()
        print(f"\nFound {len(cities_to_process)} cities with POI data")
        print(f"  (Files > 1KB in ../data/raw/pois/)")

    # Upload each city
    tasks = {}
    skipped = []
    failed = []

    for city in cities_to_process:
        try:
            task = upload_func(city)
            if task == 'exists':
                skipped.append(city)
            elif task:
                tasks[city] = task
            else:
                failed.append(city)
        except Exception as e:
            print(f"\nERROR processing {city}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(city)

    # Summary
    print(f"\n{'='*60}")
    print("Upload Summary")
    print(f"{'='*60}")
    print(f"Total cities processed: {len(cities_to_process)}")
    print(f"Skipped (already exist): {len(skipped)}")
    print(f"Successfully started: {len(tasks)} uploads")
    print(f"Failed: {len(failed)}")

    if tasks:
        print(f"\nTask IDs:")
        for city, task in tasks.items():
            print(f"  {city}: {task.id}")

        print(f"\n{'='*60}")
        print("Monitor upload progress:")
        print("  1. GEE Code Editor → Tasks tab")
        print("  2. Command line: earthengine task list")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
