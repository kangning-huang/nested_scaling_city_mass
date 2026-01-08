"""
Batch download POIs for top 500 largest cities by population using 01_download_OSM_city_pois.py

This script:
1. Reads the UCDB database to get the top 500 cities by population
2. Extracts city name and country information
3. Downloads POIs for each city sequentially
4. Skips cities that have already been downloaded
"""

import subprocess
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd

# Path to UCDB database
UCDB_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg'

def get_top_cities(n=50):
    """
    Get top N cities by population from UCDB database

    Parameters:
    -----------
    n : int
        Number of top cities to retrieve (default: 50)

    Returns:
    --------
    list of tuple
        List of (city_name, country_name) tuples
    """
    print(f"\n{'='*80}")
    print(f"Loading UCDB database: {UCDB_PATH}")
    print(f"{'='*80}\n")

    # Read UCDB data
    ucdb = gpd.read_file(UCDB_PATH)

    # Convert population to numeric (stored as string in GPKG)
    ucdb['P15_numeric'] = pd.to_numeric(ucdb['P15'], errors='coerce')

    # Sort by population and get top N
    top_n = ucdb.nlargest(n, 'P15_numeric')

    # Extract city and country names
    cities = []
    for idx, row in top_n.iterrows():
        city_name = row['UC_NM_MN']
        country_name = row['CTR_MN_NM']
        population = int(row['P15_numeric']) if pd.notna(row['P15_numeric']) else 0

        # Clean up city names (remove brackets and alternative names)
        # e.g., "Quezon City [Manila]" -> "Quezon City"
        if '[' in city_name:
            city_name = city_name.split('[')[0].strip()

        cities.append((city_name, country_name, population))

    return cities

# Get top 500 cities by population
print("Fetching top 500 cities by population from UCDB database...")
CITIES_WITH_POP = get_top_cities(500)
CITIES = [(city, country) for city, country, pop in CITIES_WITH_POP]

def check_poi_exists(city_name, data_dir='../data/raw/pois'):
    """
    Check if POI file already exists for a city

    Parameters:
    -----------
    city_name : str
        Name of the city
    data_dir : str
        Directory containing POI files

    Returns:
    --------
    bool
        True if POI file exists and is non-empty (>1KB), False otherwise
    """
    poi_file = Path(data_dir) / f'{city_name}_pois_9cats.gpkg'
    return poi_file.exists() and poi_file.stat().st_size > 1024

def run_extraction(city_name, country_name):
    """
    Run POI extraction for a single city

    Parameters:
    -----------
    city_name : str
        Name of the city
    country_name : str
        Name of the country

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / "01_download_OSM_city_pois.py"

    print(f"\n{'='*80}")
    print(f"Starting extraction for: {city_name}, {country_name}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(script_path),
        city_name,
        "--country", country_name
    ]

    try:
        # Run the extraction script
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )

        print(f"\n✓ Successfully completed: {city_name}, {country_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to extract POIs for: {city_name}, {country_name}")
        print(f"   Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error for: {city_name}, {country_name}")
        print(f"   Error: {e}")
        return False


def main():
    """Main entry point for batch processing"""
    print(f"\n{'='*80}")
    print(f"BATCH POI EXTRACTION - TOP 500 CITIES BY POPULATION")
    print(f"{'='*80}")
    print(f"\nTotal cities: {len(CITIES)}")

    # Check which cities already have POI data
    existing = []
    to_download = []
    for city_name, country_name in CITIES:
        if check_poi_exists(city_name):
            existing.append((city_name, country_name))
        else:
            to_download.append((city_name, country_name))

    print(f"Already downloaded: {len(existing)}")
    print(f"To download: {len(to_download)}")

    if to_download:
        print(f"\nCities to download:")
        for i, (city, country) in enumerate(to_download[:20], 1):
            print(f"  {i:2d}. {city:<30s} {country:<25s}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download) - 20} more")
    print()

    # Track results
    results = []
    skipped = []

    # Process each city
    for i, (city_name, country_name) in enumerate(CITIES, 1):
        # Check if already exists
        if check_poi_exists(city_name):
            print(f"\n[{i}/{len(CITIES)}] ⊘ Skipping (already exists): {city_name}, {country_name}")
            skipped.append((city_name, country_name, True))
            continue

        print(f"\n[{i}/{len(CITIES)}] Processing: {city_name}, {country_name}")
        success = run_extraction(city_name, country_name)
        results.append((city_name, country_name, success))

    # Print summary
    print(f"\n\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}\n")

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    print(f"Total cities: {len(CITIES)}")
    print(f"Skipped (already downloaded): {len(skipped)}")
    print(f"Newly downloaded: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n✓ Successfully downloaded ({len(successful)}):")
        for city, country, _ in successful[:10]:
            print(f"  - {city}, {country}")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")

    if failed:
        print(f"\n✗ Failed to download ({len(failed)}):")
        for city, country, _ in failed:
            print(f"  - {city}, {country}")

    print(f"\n{'='*80}\n")

    # Exit with error if any failed
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
