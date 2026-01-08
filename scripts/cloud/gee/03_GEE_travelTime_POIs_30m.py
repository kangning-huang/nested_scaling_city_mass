"""
Script to calculate travel times to POIs and aggregate by H3 hexagons

This script:
1. Loads POI FeatureCollections from GEE
2. Computes travel time surfaces using friction/cumulative cost analysis
3. Aggregates mean travel times for each H3 hexagon
4. Outputs results to all_cities_h3_grids.gpkg

Usage:
    python 03_travelTime_POIs_30m.py
    python 03_travelTime_POIs_30m.py --city Paris --category eating  # Test mode
"""

import ee
import geemap
import geopandas as gpd
import pandas as pd
import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from google.cloud import storage
from io import StringIO
import time

# GEE project ID
GEE_PROJECT = 'ee-knhuang'

# GCS bucket for travel time exports (reuse existing bucket from POI uploads)
GCS_BUCKET = 'ee-knhuang-uploads'

# City-specific CRS mapping (UTM zones)
CITY_PROJECTIONS = {
    'Paris': 'EPSG:32631',
    'Tokyo': 'EPSG:32654',
    'Melbourne': 'EPSG:32755',
    'Mexico_City': 'EPSG:32614',
    'Bogota': 'EPSG:32618',
    'Atlanta': 'EPSG:32616',
    'Rome': 'EPSG:32633',
    'Reykjavik': 'EPSG:32627',
    'Addis_Ababa': 'EPSG:32637'
}

# GRIP4 regional dataset mapping
GRIP4_REGIONS = {
    'Paris': 'projects/sat-io/open-datasets/GRIP4/Europe',
    'Rome': 'projects/sat-io/open-datasets/GRIP4/Europe',
    'Reykjavik': 'projects/sat-io/open-datasets/GRIP4/Europe',
    'Tokyo': 'projects/sat-io/open-datasets/GRIP4/South-East-Asia',
    'Melbourne': 'projects/sat-io/open-datasets/GRIP4/Oceania',
    'Mexico_City': 'projects/sat-io/open-datasets/GRIP4/Central-South-America',
    'Bogota': 'projects/sat-io/open-datasets/GRIP4/Central-South-America',
    'Atlanta': 'projects/sat-io/open-datasets/GRIP4/North-America',
    'Addis_Ababa': 'projects/sat-io/open-datasets/GRIP4/Africa'
}

# POI categories
CATEGORIES = [
    'eating', 'moving', 'outdoor_activities', 'physical_exercise',
    'supplies', 'learning', 'services', 'health_care', 'cultural_activities'
]

# ---------------------------------------------------------------------------
# Helper functions from Fig3_DataExtract_Volume_Pavement_Neighborhood.py
# ---------------------------------------------------------------------------

def gdf_to_ee_featurecollection(gdf):
    """
    Convert GeoDataFrame to Earth Engine FeatureCollection

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with h3index and geometry columns

    Returns:
    --------
    ee.FeatureCollection
    """
    # Ensure h3index is a property, not just the index
    if 'h3index' not in gdf.columns and gdf.index.name == 'h3index':
        gdf_copy = gdf.reset_index()
    else:
        gdf_copy = gdf.copy()

    # Select only h3index and geometry for the FeatureCollection properties
    if 'h3index' in gdf_copy.columns and 'geometry' in gdf_copy.columns:
        gdf_for_fc = gdf_copy[['h3index', 'geometry']]
    else:
        print("Warning: 'h3index' column not found directly.")
        if gdf_copy.index.name == 'h3index':
            gdf_copy['h3index'] = gdf_copy.index
            gdf_for_fc = gdf_copy[['h3index', 'geometry']]
        else:
            gdf_for_fc = gdf_copy[['geometry']]

    # Convert the GeoDataFrame to a GeoJSON string
    geojson_str = gdf_for_fc.to_json()
    # Parse the GeoJSON string into a Python dictionary
    geojson_dict = json.loads(geojson_str)
    # Create an Earth Engine FeatureCollection from the GeoJSON
    fc = ee.FeatureCollection(geojson_dict)
    return fc

def ee_featurecollection_to_dataframe(fc, show_progress=True):
    """
    Convert Earth Engine FeatureCollection to pandas DataFrame with chunking

    Parameters:
    -----------
    fc : ee.FeatureCollection
        Input FeatureCollection
    show_progress : bool
        Whether to show progress bar

    Returns:
    --------
    pd.DataFrame
    """
    dfs = []
    columns_to_request = ['h3index', 'mean']  # Changed from 'sum' to 'mean' for travel time

    count = fc.size().getInfo()
    chunk_size = 2000
    num_chunks = (count // chunk_size) + 1
    iterator = tqdm(range(num_chunks), desc="Downloading GEE FC to DataFrame", leave=False) if show_progress else range(num_chunks)

    for i in iterator:
        start = i * chunk_size
        list_fc = fc.toList(chunk_size, start)

        if list_fc.size().getInfo() == 0:
            continue

        subset_fc = ee.FeatureCollection(list_fc)
        try:
            subset_df = geemap.ee_to_df(subset_fc, columns=columns_to_request)
            dfs.append(subset_df)
        except Exception as e:
            print(f"Error converting EE FeatureCollection chunk to DataFrame: {e}")

    if not dfs:
        return pd.DataFrame(columns=['h3index', 'mean'])

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

# ---------------------------------------------------------------------------
# Travel time computation functions
# ---------------------------------------------------------------------------

def compute_friction_surfaces(region, crs, city_name):
    """
    Compute friction surfaces for walking and motorized travel

    Replicates logic from GEE_travelTimeToPOIs_30m.js lines 15-96

    Parameters:
    -----------
    region : ee.Geometry
        Analysis region
    crs : str
        Coordinate reference system (e.g., 'EPSG:32631')
    city_name : str
        City name for GRIP4 dataset selection

    Returns:
    --------
    tuple of (frictionWalk, frictionMotor) : ee.Image
        Friction surfaces in minutes per pixel
    """
    scale = 30

    # 1. LAND COVER - Base speed from ESA WorldCover
    lc = ee.Image('ESA/WorldCover/v100/2020').select('Map')

    lcClasses = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    speedValues = [3.06, 3.60, 4.86, 2.50, 5.00, 3.00, 2.00, 0.01, 4.86, 4.20, 2.00]

    baseSpeed = lc.remap(lcClasses, speedValues, 3.0)
    landMask = lc.neq(80)  # Mask out water
    baseSpeed = baseSpeed.updateMask(landMask)

    # 2. TOPOGRAPHIC ADJUSTMENTS
    dsm = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2') \
        .select('DSM') \
        .filterBounds(region) \
        .mosaic() \
        .setDefaultProjection(crs, None, scale) \
        .clip(region)

    # Slope
    slopeDeg = ee.Terrain.slope(dsm)

    # Elevation factor
    fElev = dsm.expression('1.016 * exp(-0.0001072 * elev)', {'elev': dsm})

    # Slope factor (Tobler's hiking function)
    slopeRad = slopeDeg.multiply(3.14159265359 / 180)
    s = slopeRad.tan()
    hikingSpeed = s.expression('6 * exp(-3.5 * abs(s + 0.05))', {'s': s})
    fSlope = hikingSpeed.divide(5)

    # Fallback for invalid slope/elevation factors
    fSlope = fSlope.unmask(1)
    fElev = fElev.unmask(1)

    # Combined land speed
    landSpeed = baseSpeed.multiply(fElev).multiply(fSlope)
    landSpeed = landSpeed.where(landSpeed.lte(0), 0.5)
    landSpeed = landSpeed.unmask(3)

    # 3. GRIP4 ROADS
    grip4_path = GRIP4_REGIONS[city_name]
    grip4_roads = ee.FeatureCollection(grip4_path).filterBounds(region)

    roadBuffer = grip4_roads.map(lambda f: f.buffer(15))

    # Road speed for motorized transport (km/h)
    roadSpeedMotor = ee.Image(0) \
        .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 1)), 105) \
        .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 2)), 80) \
        .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 3)), 60) \
        .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 4)), 50) \
        .paint(roadBuffer.filter(ee.Filter.eq('GP_RTP', 5)), 40)

    # Road speed for walking (km/h)
    roadSpeedWalk = ee.Image(0).paint(roadBuffer, 5)

    # 4. COMBINED SPEEDS
    walkSpeed = landSpeed.max(roadSpeedWalk).max(0.5)
    motorSpeed = landSpeed.max(roadSpeedMotor).max(0.5)

    # 5. FRICTION (minutes per pixel at 30m)
    # Formula: (60 minutes/hour × 0.030 km/pixel) / speed(km/h) = minutes/pixel
    frictionWalk = ee.Image(60 / 1000).divide(walkSpeed).rename('friction_walk')
    frictionMotor = ee.Image(60 / 1000).divide(motorSpeed).rename('friction_motor')

    return frictionWalk, frictionMotor

def rasterize_pois(pois, category):
    """
    Rasterize POIs by category

    Replicates logic from GEE_travelTimeToPOIs_30m.js lines 99-108

    Parameters:
    -----------
    pois : ee.FeatureCollection
        POI FeatureCollection with 'category' property
    category : str
        Category to filter and rasterize

    Returns:
    --------
    ee.Image
        Binary raster (1 = POI location, masked elsewhere)
    """
    fc = pois.filter(ee.Filter.eq('category', category))

    src = ee.Image(0).byte() \
        .paint(fc, 1) \
        .selfMask()

    return src

def compute_travel_time(pois, category, region, crs, city_name, max_minutes=1000):
    """
    Compute travel time to POIs for a given category

    Replicates logic from GEE_travelTimeToPOIs_30m.js lines 114-129

    Parameters:
    -----------
    pois : ee.FeatureCollection
        POI FeatureCollection
    category : str
        POI category
    region : ee.Geometry
        Analysis region
    crs : str
        Coordinate reference system
    city_name : str
        City name for GRIP4 dataset selection
    max_minutes : int
        Maximum travel time distance (minutes)

    Returns:
    --------
    tuple of (tt_walk, tt_motor) : ee.Image
        Travel time images in minutes
    """
    # Compute friction surfaces
    frictionWalk, frictionMotor = compute_friction_surfaces(region, crs, city_name)

    # Rasterize POIs
    src = rasterize_pois(pois, category)

    # Cumulative cost (travel time)
    ttWalk = frictionWalk.cumulativeCost(
        source=src,
        maxDistance=max_minutes
    ).rename(f'tt_walk_{category}')

    ttMotor = frictionMotor.cumulativeCost(
        source=src,
        maxDistance=max_minutes
    ).rename(f'tt_motor_{category}')

    return ttWalk, ttMotor

def extract_hexagon_stats(city_hexagons, tt_walk, tt_motor, category):
    """
    Extract mean travel times for each H3 hexagon

    Uses geemap.zonal_stats pattern from Fig3_DataExtract_Volume_Pavement_Neighborhood.py

    Parameters:
    -----------
    city_hexagons : gpd.GeoDataFrame
        Hexagons for the city with h3index and geometry
    tt_walk : ee.Image
        Walking travel time image
    tt_motor : ee.Image
        Motorized travel time image
    category : str
        POI category name

    Returns:
    --------
    pd.DataFrame
        DataFrame with h3index, tt_walk_{category}, tt_motor_{category}
    """
    # Convert hexagons to ee.FeatureCollection
    fc_hex = gdf_to_ee_featurecollection(city_hexagons[['h3index', 'geometry']])

    # Zonal stats for walking travel time
    stats_walk = geemap.zonal_stats(
        in_zone_vector=fc_hex,
        in_value_raster=tt_walk,
        scale=30,
        statistics_type='MEAN',
        return_fc=True
    )
    df_walk = ee_featurecollection_to_dataframe(stats_walk, show_progress=False)
    df_walk.rename(columns={'mean': f'tt_walk_{category}'}, inplace=True)

    # Zonal stats for motorized travel time
    stats_motor = geemap.zonal_stats(
        in_zone_vector=fc_hex,
        in_value_raster=tt_motor,
        scale=30,
        statistics_type='MEAN',
        return_fc=True
    )
    df_motor = ee_featurecollection_to_dataframe(stats_motor, show_progress=False)
    df_motor.rename(columns={'mean': f'tt_motor_{category}'}, inplace=True)

    # Merge walk and motor results
    merged = df_walk.merge(
        df_motor[['h3index', f'tt_motor_{category}']],
        on='h3index',
        how='inner'
    )

    return merged

def combine_city_results(city_results):
    """
    Combine all category results for a single city into one DataFrame

    Parameters:
    -----------
    city_results : list of pd.DataFrame
        List of DataFrames with travel time results for each category (from one city)

    Returns:
    --------
    pd.DataFrame or None
        Combined DataFrame with h3index + all travel time columns for this city,
        or None if no valid results
    """
    if not city_results:
        return None

    # Start with the first result
    valid_results = [df for df in city_results if not df.empty and 'h3index' in df.columns]

    if not valid_results:
        return None

    # Start with first DataFrame - keep only relevant columns
    # Filter to keep h3index and any tt_* columns
    def filter_columns(df):
        cols_to_keep = ['h3index'] + [c for c in df.columns if c.startswith('tt_')]
        return df[cols_to_keep]

    merged = filter_columns(valid_results[0]).copy()

    # Merge remaining DataFrames from this city
    for df_to_add in valid_results[1:]:
        # Filter to relevant columns before merging to avoid conflicts
        df_filtered = filter_columns(df_to_add)
        # Merge on h3index - all DataFrames should have the same h3index values (same city)
        merged = pd.merge(merged, df_filtered, on='h3index', how='outer')

    return merged

def combine_results(h3_grids, city_combined_results):
    """
    Combine all category results with original H3 grid data

    Follows merging pattern from Fig3_DataExtract_Volume_Pavement_Neighborhood.py lines 407-413

    Parameters:
    -----------
    h3_grids : gpd.GeoDataFrame
        Original H3 grid with all city data
    city_combined_results : list of pd.DataFrame
        List of DataFrames, one per city, each containing all categories for that city

    Returns:
    --------
    gpd.GeoDataFrame
        Combined GeoDataFrame with original columns + travel time columns
    """
    # Start with original h3_grids
    merged_df = h3_grids.copy()

    # Merge each city's combined results
    for city_df in city_combined_results:
        if city_df is None or city_df.empty:
            continue

        if 'h3index' not in city_df.columns:
            print(f"Warning: DataFrame missing 'h3index' during merge")
            continue

        # Get new columns that will be added (exclude 'h3index')
        new_cols = [c for c in city_df.columns if c != 'h3index']

        # Check if any columns already exist (they shouldn't, since each city is unique)
        cols_to_update = [c for c in new_cols if c in merged_df.columns]

        if cols_to_update:
            # Update existing columns with new data (fill NaN values)
            print(f"  Updating columns with new city data: {', '.join(cols_to_update)}")
            # First merge, creating _x and _y suffixes
            temp_df = pd.merge(merged_df, city_df, on='h3index', how='left', suffixes=('_old', '_new'))

            # For each column, fill NaN in _new with _old, then drop _old
            for col in cols_to_update:
                if f'{col}_new' in temp_df.columns:
                    # Combine old and new: use new if available, otherwise keep old
                    temp_df[col] = temp_df[f'{col}_new'].fillna(temp_df[f'{col}_old'])
                    temp_df = temp_df.drop(columns=[f'{col}_old', f'{col}_new'])

            # Drop any remaining _old/_new columns and rename
            merged_df = temp_df
        else:
            # No conflicts - simple left merge
            merged_df = pd.merge(merged_df, city_df, on='h3index', how='left')

    return merged_df

# ---------------------------------------------------------------------------
# Server-side export functions for parallel cloud execution
# ---------------------------------------------------------------------------

def export_city_travel_times(city_name, h3_grids, pois, categories, gcs_bucket):
    """
    Export travel time analysis results to Google Cloud Storage using server-side tasks

    This function submits all category analyses for a city as parallel GEE export tasks,
    eliminating the client-side download bottleneck.

    Parameters:
    -----------
    city_name : str
        City name
    h3_grids : gpd.GeoDataFrame
        H3 hexagon grid for all cities
    pois : ee.FeatureCollection
        POI FeatureCollection for this city
    categories : list
        List of POI categories to process
    gcs_bucket : str
        GCS bucket name (e.g., 'ee-knhuang-travel-time-exports')

    Returns:
    --------
    dict
        Dictionary mapping category names to ee.batch.Task objects
    """
    print(f"\n{'='*60}")
    print(f"Setting up server-side exports for: {city_name}")
    print(f"{'='*60}")

    # Get city-specific hexagons
    city_hexagons = h3_grids[h3_grids['UC_NM_MN'] == city_name].copy()

    if city_hexagons.empty:
        print(f"WARNING: No hexagons found for {city_name}")
        return {}

    print(f"  Hexagons: {len(city_hexagons):,}")
    print(f"  Categories: {len(categories)}")

    # Convert hexagons to ee.FeatureCollection
    fc_hex = gdf_to_ee_featurecollection(city_hexagons[['h3index', 'geometry']])

    # Set up region and CRS
    region = pois.geometry().bounds()
    crs = CITY_PROJECTIONS[city_name]

    # Compute friction surfaces ONCE for this city (optimization)
    print(f"  Computing friction surfaces...")
    friction_walk, friction_motor = compute_friction_surfaces(region, crs, city_name)

    # Submit export tasks for each category
    tasks = {}

    for category in categories:
        print(f"\n  [{category}] Setting up export task...")

        try:
            # 1. Rasterize POIs for this category
            pois_raster = rasterize_pois(pois, category)

            # 2. Compute travel times
            max_minutes = 1000

            tt_walk = friction_walk.cumulativeCost(
                source=pois_raster,
                maxDistance=max_minutes
            ).rename(f'tt_walk_{category}')

            tt_motor = friction_motor.cumulativeCost(
                source=pois_raster,
                maxDistance=max_minutes
            ).rename(f'tt_motor_{category}')

            # 3. Compute zonal stats (server-side)
            stats_walk = geemap.zonal_stats(
                in_zone_vector=fc_hex,
                in_value_raster=tt_walk,
                scale=30,
                statistics_type='MEAN',
                return_fc=True
            )

            stats_motor = geemap.zonal_stats(
                in_zone_vector=fc_hex,
                in_value_raster=tt_motor,
                scale=30,
                statistics_type='MEAN',
                return_fc=True
            )

            # 4. Merge walk and motor stats server-side
            # Add category-specific column names
            stats_walk_renamed = stats_walk.map(lambda f: f.set(f'tt_walk_{category}', f.get('mean')))
            stats_motor_renamed = stats_motor.map(lambda f: f.set(f'tt_motor_{category}', f.get('mean')))

            # Join on h3index
            combined_stats = stats_walk_renamed.map(lambda f_walk:
                f_walk.set(
                    f'tt_motor_{category}',
                    stats_motor_renamed.filter(
                        ee.Filter.eq('h3index', f_walk.get('h3index'))
                    ).first().get(f'tt_motor_{category}')
                )
            )

            # Select only h3index and the two travel time columns
            final_stats = combined_stats.select(
                ['h3index', f'tt_walk_{category}', f'tt_motor_{category}'],
                retainGeometry=False
            )

            # 5. Export to GCS
            file_prefix = f'travel_time/{city_name}/{category}'
            description = f'{city_name}_{category}_tt'

            task = ee.batch.Export.table.toCloudStorage(
                collection=final_stats,
                description=description,
                bucket=gcs_bucket,
                fileNamePrefix=file_prefix,
                fileFormat='CSV'
            )

            task.start()
            tasks[category] = task

            print(f"    ✓ Task started: {task.id}")

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            continue

    print(f"\n  Total export tasks submitted: {len(tasks)}")
    return tasks

def monitor_tasks(task_dict, poll_interval=60):
    """
    Monitor GEE export tasks until all complete

    Parameters:
    -----------
    task_dict : dict
        Dictionary mapping identifiers to ee.batch.Task objects
    poll_interval : int
        Seconds between status checks (default: 60)

    Returns:
    --------
    tuple of (completed_tasks, failed_tasks)
        Lists of task identifiers for completed and failed tasks
    """
    print(f"\n{'='*60}")
    print(f"Monitoring {len(task_dict)} export tasks...")
    print(f"{'='*60}")

    pending = list(task_dict.keys())
    completed = []
    failed = []

    while pending:
        print(f"\n  Status check ({len(pending)} pending, {len(completed)} completed, {len(failed)} failed)")

        for identifier in list(pending):
            task = task_dict[identifier]
            status = task.status()
            state = status['state']

            if state == 'COMPLETED':
                print(f"    ✓ {identifier}: COMPLETED")
                pending.remove(identifier)
                completed.append(identifier)
            elif state == 'FAILED':
                error_msg = status.get('error_message', 'Unknown error')
                print(f"    ✗ {identifier}: FAILED - {error_msg}")
                pending.remove(identifier)
                failed.append(identifier)
            elif state in ['RUNNING', 'READY']:
                print(f"    ⋯ {identifier}: {state}")

        if pending:
            print(f"\n  Waiting {poll_interval} seconds before next check...")
            time.sleep(poll_interval)

    print(f"\n{'='*60}")
    print(f"All tasks finished!")
    print(f"  Completed: {len(completed)}")
    print(f"  Failed: {len(failed)}")
    print(f"{'='*60}\n")

    return completed, failed

def download_and_combine_gcs_results(bucket_name, cities, categories, h3_grids, output_path):
    """
    Download travel time results from GCS and combine into final GeoPackage

    Parameters:
    -----------
    bucket_name : str
        GCS bucket name
    cities : list
        List of city names
    categories : list
        List of POI categories
    h3_grids : gpd.GeoDataFrame
        Original H3 grid with geometries
    output_path : str or Path
        Path to save final GeoPackage

    Returns:
    --------
    gpd.GeoDataFrame
        Combined GeoDataFrame with all travel time data
    """
    print(f"\n{'='*60}")
    print(f"Downloading and combining results from GCS")
    print(f"{'='*60}")

    storage_client = storage.Client(project=GEE_PROJECT)
    bucket = storage_client.bucket(bucket_name)

    city_combined_results = []

    for city_name in cities:
        print(f"\n  Processing {city_name}...")
        city_results = []

        for category in categories:
            blob_path = f'travel_time/{city_name}/{category}.csv'
            blob = bucket.blob(blob_path)

            try:
                # Download CSV as string
                csv_data = blob.download_as_text()

                # Parse with pandas
                df = pd.read_csv(StringIO(csv_data))

                print(f"    ✓ {category}: {len(df):,} rows")
                city_results.append(df)

            except Exception as e:
                print(f"    ✗ {category}: ERROR - {e}")
                continue

        # Combine all categories for this city
        if city_results:
            city_combined = combine_city_results(city_results)
            if city_combined is not None:
                city_combined_results.append(city_combined)
                print(f"  → Combined {len(city_results)} categories: {len(city_combined):,} hexagons")

    # Combine all cities with original H3 grid
    print(f"\n  Merging with H3 grid data...")
    final_gdf = combine_results(h3_grids, city_combined_results)

    # Add average travel time columns
    print(f"  Computing average travel times...")
    final_gdf = add_average_travel_time_columns(final_gdf)

    # Save to GeoPackage
    print(f"  Saving to: {output_path}")
    final_gdf.to_file(output_path, driver='GPKG')

    print(f"\n  ✓ Saved {len(final_gdf):,} hexagons with {len([c for c in final_gdf.columns if c.startswith('tt_')])} travel time columns")

    return final_gdf

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_city(city_name, h3_grids, categories=CATEGORIES, test_mode=False):
    """
    Process all POI categories for a single city

    Parameters:
    -----------
    city_name : str
        City name
    h3_grids : gpd.GeoDataFrame
        H3 hexagon grid for all cities
    categories : list
        List of POI categories to process
    test_mode : bool
        If True, only process first category for testing

    Returns:
    --------
    list of pd.DataFrame
        Results for each category
    """
    print(f"\n{'='*60}")
    print(f"Processing city: {city_name}")
    print(f"{'='*60}")

    # Get city-specific hexagons
    city_hexagons = h3_grids[h3_grids['UC_NM_MN'] == city_name].copy()

    if city_hexagons.empty:
        print(f"WARNING: No hexagons found for {city_name}")
        return []

    print(f"  Hexagons: {len(city_hexagons):,}")

    # Load POIs for this city
    try:
        pois = ee.FeatureCollection(
            f'projects/{GEE_PROJECT}/assets/global_cities_POIs/{city_name}_pois_9cats'
        )
        # Check if POIs exist
        poi_count = pois.size().getInfo()
        print(f"  POIs: {poi_count:,}")
    except Exception as e:
        print(f"ERROR: Could not load POIs for {city_name}: {e}")
        return []

    # Set up region and CRS
    region = pois.geometry().bounds()
    crs = CITY_PROJECTIONS[city_name]
    print(f"  CRS: {crs}")
    print(f"  GRIP4 Region: {GRIP4_REGIONS[city_name]}")

    # Process each category
    results = []
    categories_to_process = categories[:1] if test_mode else categories

    for category in tqdm(categories_to_process, desc=f"  Categories"):
        try:
            print(f"\n  Processing category: {category}")

            # Compute travel time
            tt_walk, tt_motor = compute_travel_time(pois, category, region, crs, city_name)

            # Extract hexagon stats
            hexagon_stats = extract_hexagon_stats(city_hexagons, tt_walk, tt_motor, category)

            if not hexagon_stats.empty:
                results.append(hexagon_stats)

                # Print intermediate statistics
                walk_col = f'tt_walk_{category}'
                motor_col = f'tt_motor_{category}'
                print(f"    ✓ {category}: {len(hexagon_stats)} hexagons processed")
                print(f"      Walking times (minutes):")
                print(f"        Min: {hexagon_stats[walk_col].min():.2f}")
                print(f"        Max: {hexagon_stats[walk_col].max():.2f}")
                print(f"        Mean: {hexagon_stats[walk_col].mean():.2f}")
                print(f"        Median: {hexagon_stats[walk_col].median():.2f}")
                print(f"      Motorized times (minutes):")
                print(f"        Min: {hexagon_stats[motor_col].min():.2f}")
                print(f"        Max: {hexagon_stats[motor_col].max():.2f}")
                print(f"        Mean: {hexagon_stats[motor_col].mean():.2f}")
                print(f"        Median: {hexagon_stats[motor_col].median():.2f}")
            else:
                print(f"    ✗ {category}: No results")

        except Exception as e:
            print(f"    ERROR processing {category}: {e}")
            import traceback
            traceback.print_exc()

    return results

def get_available_cities():
    """Get list of cities with POI assets in GEE"""
    import subprocess

    earthengine_cmd = '/Users/kangninghuang/.venvs/nyu_china_grant_env/bin/earthengine'
    asset_folder = f'projects/{GEE_PROJECT}/assets/global_cities_POIs'

    result = subprocess.run(
        [earthengine_cmd, 'ls', asset_folder],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return []

    # Parse asset list to extract city names
    cities = []
    for line in result.stdout.strip().split('\n'):
        if '_pois_9cats' in line:
            # Extract city name from asset path
            city_name = line.split('/')[-1].replace('_pois_9cats', '')
            cities.append(city_name)

    return sorted(cities)

def add_average_travel_time_columns(gdf):
    """
    Add two summary columns: average travel time across all categories for walking and motorized

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with travel time columns

    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with added average columns
    """
    walk_cols = [f'tt_walk_{cat}' for cat in CATEGORIES if f'tt_walk_{cat}' in gdf.columns]
    motor_cols = [f'tt_motor_{cat}' for cat in CATEGORIES if f'tt_motor_{cat}' in gdf.columns]

    if walk_cols:
        gdf['tt_walk_avg'] = gdf[walk_cols].mean(axis=1)
        print(f"  Added tt_walk_avg column (average of {len(walk_cols)} categories)")
        print(f"    Mean: {gdf['tt_walk_avg'].mean():.2f} min")
        print(f"    Median: {gdf['tt_walk_avg'].median():.2f} min")

    if motor_cols:
        gdf['tt_motor_avg'] = gdf[motor_cols].mean(axis=1)
        print(f"  Added tt_motor_avg column (average of {len(motor_cols)} categories)")
        print(f"    Mean: {gdf['tt_motor_avg'].mean():.2f} min")
        print(f"    Median: {gdf['tt_motor_avg'].median():.2f} min")

    return gdf

def check_city_processed(output_path, city_name):
    """Check if travel times for a city have already been calculated"""
    if not output_path.exists():
        return False

    try:
        # Read existing data
        gdf = gpd.read_file(output_path)

        # Check if city has any travel time columns with data
        city_hexagons = gdf[gdf['UC_NM_MN'] == city_name]

        if city_hexagons.empty:
            return False

        # Check if all category columns exist and have data
        tt_cols = [f'tt_walk_{cat}' for cat in CATEGORIES] + [f'tt_motor_{cat}' for cat in CATEGORIES]
        has_all_data = all(
            col in gdf.columns and city_hexagons[col].notna().sum() > 0
            for col in tt_cols
        )

        return has_all_data
    except Exception as e:
        print(f"  Warning: Could not check existing data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Calculate travel times to POIs for H3 hexagons')
    parser.add_argument('--city', type=str, help='Process single city (for testing)')
    parser.add_argument('--category', type=str, help='Process single category (for testing)')
    parser.add_argument('--data-dir', type=str, default='../data/raw',
                       help='Data directory containing all_cities_h3_grids.gpkg')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Output directory for results')
    args = parser.parse_args()

    print("="*60)
    print("Travel Time Analysis - POIs to H3 Hexagons")
    print("="*60)

    # Initialize GEE
    print("\nInitializing Google Earth Engine...")
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"  Connected to project: {GEE_PROJECT}")
    except Exception as e:
        print(f"ERROR: Failed to initialize GEE: {e}")
        sys.exit(1)

    # Load H3 grid
    h3_grid_path = Path(args.data_dir) / 'all_cities_h3_grids.gpkg'
    print(f"\nLoading H3 grid: {h3_grid_path}")

    if not h3_grid_path.exists():
        print(f"ERROR: H3 grid file not found: {h3_grid_path}")
        sys.exit(1)

    h3_grids = gpd.read_file(h3_grid_path)
    print(f"  Total hexagons: {len(h3_grids):,}")
    print(f"  Cities in H3 grid: {h3_grids['UC_NM_MN'].unique().tolist()}")

    # Get available cities from GEE
    print("\nChecking available POI assets in GEE...")
    gee_cities = get_available_cities()
    print(f"  Found {len(gee_cities)} cities with POI data in GEE")

    # Set up output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'neighborhood_POIs_travel_time.gpkg'

    # Load existing results if available
    if output_path.exists():
        print(f"\nLoading existing results from: {output_path}")
        existing_gdf = gpd.read_file(output_path)
        print(f"  Existing columns: {len([c for c in existing_gdf.columns if c.startswith('tt_')])} travel time columns")
    else:
        print(f"\nNo existing results found. Will create new file.")
        existing_gdf = h3_grids.copy()

    # Determine cities to process
    if args.city:
        cities_to_process = [args.city]
        test_mode = False  # Process all categories even when testing a single city
    else:
        # Process only cities that have CRS and GRIP4 mappings
        supported_cities = set(CITY_PROJECTIONS.keys())
        h3_cities = set(h3_grids['UC_NM_MN'].unique())
        cities_to_process = [c for c in gee_cities if c in h3_cities and c in supported_cities]
        test_mode = False

        print(f"\nCities to process: {len(cities_to_process)}")

        # Check which cities are already processed
        already_processed = []
        for city in cities_to_process[:]:
            if check_city_processed(output_path, city):
                already_processed.append(city)
                cities_to_process.remove(city)

        if already_processed:
            print(f"  Already processed ({len(already_processed)}): {', '.join(already_processed)}")
        print(f"  Need to process ({len(cities_to_process)}): {', '.join(cities_to_process)}")

    if not cities_to_process:
        print("\nAll cities already processed!")
        sys.exit(0)

    categories_to_process = [args.category] if args.category else CATEGORIES

    # Server-side export workflow - submit all tasks in parallel
    print(f"\n{'='*60}")
    print("Submitting server-side export tasks to GEE...")
    print(f"{'='*60}")

    all_tasks = {}  # city_category -> task

    for city_name in cities_to_process:
        # Load POIs for this city
        try:
            pois = ee.FeatureCollection(
                f'projects/{GEE_PROJECT}/assets/global_cities_POIs/{city_name}_pois_9cats'
            )
            # Verify POIs exist
            poi_count = pois.size().getInfo()
            print(f"\n{city_name}: {poi_count:,} POIs")
        except Exception as e:
            print(f"\nERROR loading POIs for {city_name}: {e}")
            continue

        # Submit export tasks for all categories in this city
        city_tasks = export_city_travel_times(
            city_name=city_name,
            h3_grids=h3_grids,
            pois=pois,
            categories=categories_to_process,
            gcs_bucket=GCS_BUCKET
        )

        # Add city prefix to task identifiers
        for category, task in city_tasks.items():
            all_tasks[f"{city_name}_{category}"] = task

    if not all_tasks:
        print("\nERROR: No tasks were successfully submitted!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Submitted {len(all_tasks)} export tasks across {len(cities_to_process)} cities")
    print(f"{'='*60}")

    # Monitor task completion
    completed, failed = monitor_tasks(all_tasks, poll_interval=60)

    if failed:
        print(f"\nWARNING: {len(failed)} tasks failed:")
        for task_id in failed:
            print(f"  - {task_id}")

    if not completed:
        print("\nERROR: No tasks completed successfully!")
        sys.exit(1)

    # Download and combine results from GCS
    final_gdf = download_and_combine_gcs_results(
        bucket_name=GCS_BUCKET,
        cities=cities_to_process,
        categories=categories_to_process,
        h3_grids=h3_grids,
        output_path=output_path
    )

    # Validation
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")

    travel_time_cols = [col for col in final_gdf.columns if col.startswith('tt_')]
    print(f"\nTotal travel time columns: {len(travel_time_cols)}")

    for col in sorted(travel_time_cols):
        if col in final_gdf.columns and final_gdf[col].notna().sum() > 0:
            stats = final_gdf[col].describe()
            print(f"\n{col}:")
            print(f"  Count: {stats['count']:.0f}")
            print(f"  Mean: {stats['mean']:.2f} min")
            print(f"  Median: {stats['50%']:.2f} min")
            print(f"  Min: {stats['min']:.2f} min")
            print(f"  Max: {stats['max']:.2f} min")
            print(f"  NaN values: {final_gdf[col].isna().sum():,}")

if __name__ == "__main__":
    main()
