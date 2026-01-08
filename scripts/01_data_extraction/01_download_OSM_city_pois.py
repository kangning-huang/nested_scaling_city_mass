"""
Extract Points of Interest (POI) from OpenStreetMap for any city in the GHS Urban Centers Database

Usage:
    python extract_city_pois.py "New York" --country "United States"
    python extract_city_pois.py "London" --country "United Kingdom"
    python extract_city_pois.py "Tokyo" --output-dir ../results/pois
"""

import argparse
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import osmnx as ox

# Disable caching to save disk space in Google Drive
ox.settings.use_cache = False

# POI Categories Definition
POI_TAGS = {
    "outdoor_activities": {
        "leisure": ["park", "playground", "garden", "dog_park", "nature_reserve"],
        "tourism": ["viewpoint"]
    },
    "learning": {
        "amenity": ["school", "kindergarten", "college", "university", "library"]
    },
    "supplies": {
        "shop": ["supermarket", "convenience", "bakery", "butcher", "greengrocer",
                 "mall", "department_store"]
    },
    "eating": {
        "amenity": ["restaurant", "cafe", "fast_food", "bar", "pub"]
    },
    "moving": {
        "amenity": ["bus_station", "ferry_terminal", "taxi"],
        "highway": ["bus_stop", "tram_stop", "subway_entrance"]
    },
    "cultural_activities": {
        "amenity": ["cinema", "theatre", "arts_centre"],
        "tourism": ["museum", "gallery"]
    },
    "physical_exercise": {
        "leisure": ["sports_centre", "pitch", "fitness_centre", "swimming_pool"]
    },
    "services": {
        "amenity": ["bank", "atm", "post_office", "pharmacy", "police", "townhall"]
    },
    "health_care": {
        "amenity": ["hospital", "clinic", "doctors", "dentist"]
    }
}


def find_city_in_database(city_name, country_name=None, database_path=None):
    """
    Find a city in the GHS Urban Centers Database

    Parameters:
    -----------
    city_name : str
        Name of the city to search for
    country_name : str, optional
        Name of the country to narrow search
    database_path : str
        Path to the GHS database GeoPackage

    Returns:
    --------
    geopandas.GeoDataFrame or None
        GeoDataFrame with matching city, or None if not found
    """
    print(f"\n{'='*80}")
    print(f"Searching for city: {city_name}")
    if country_name:
        print(f"Country filter: {country_name}")
    print(f"{'='*80}\n")

    # Load the database
    print("Loading GHS Urban Centers Database...")
    try:
        all_cities = gpd.read_file(database_path)
        print(f"  Loaded {len(all_cities):,} cities from database")
    except Exception as e:
        print(f"Error loading database: {e}")
        return None

    # Search for the city
    city_matches = all_cities[
        all_cities['UC_NM_MN'].str.contains(city_name, case=False, na=False)
    ].copy()

    # Filter by country if provided
    if country_name:
        city_matches = city_matches[
            city_matches['CTR_MN_NM'].str.contains(country_name, case=False, na=False)
        ]

    if city_matches.empty:
        print(f"\n❌ No cities found matching '{city_name}'")
        if country_name:
            print(f"   in country '{country_name}'")
        return None

    if len(city_matches) > 1:
        print(f"\n⚠️  Found {len(city_matches)} matching cities:")
        for idx, row in city_matches.iterrows():
            print(f"   - {row['UC_NM_MN']}, {row['CTR_MN_NM']} (ID: {row['ID_HDC_G0']})")
        print("\n   Using the first match. Specify --country to narrow down the search.")

    # Use the first match
    city = city_matches.iloc[0:1].copy()
    print(f"\n✓ Found city: {city['UC_NM_MN'].iloc[0]}, {city['CTR_MN_NM'].iloc[0]}")
    print(f"  City ID: {city['ID_HDC_G0'].iloc[0]}")
    if 'P15' in city.columns:
        pop = city['P15'].iloc[0]
        if pd.notna(pop):
            print(f"  Population (2015): {float(pop):,.0f}")

    return city


def query_poi_category(polygon, tag_dict, category_name):
    """
    Query POIs for a specific category from OpenStreetMap

    Parameters:
    -----------
    polygon : shapely.geometry
        The boundary polygon to query within
    tag_dict : dict
        Dictionary of OSM tags and values for this category
    category_name : str
        Name of the POI category

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing all POIs for this category
    """
    frames = []

    for key, values in tag_dict.items():
        for v in values:
            tags = {key: v}

            try:
                print(f"  Querying {key}={v}...", end=" ")
                gdf = ox.features_from_polygon(polygon, tags=tags)

                if not gdf.empty:
                    gdf["category"] = category_name
                    gdf["osm_key"] = key
                    gdf["osm_value"] = v
                    frames.append(gdf)
                    print(f"✓ {len(gdf)} features")
                else:
                    print("(none)")

            except Exception as e:
                print(f"✗ ({str(e)[:50]})")

    if frames:
        result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
        return result
    else:
        return gpd.GeoDataFrame(
            columns=["geometry", "category", "osm_key", "osm_value"],
            crs="EPSG:4326"
        )


def extract_pois(city_name, country_name=None, output_dir=None, database_path=None):
    """
    Extract POIs for a city from OpenStreetMap

    Parameters:
    -----------
    city_name : str
        Name of the city
    country_name : str, optional
        Country name to narrow search
    output_dir : str
        Directory to save output files
    database_path : str
        Path to GHS database

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Set up output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find the city in the database
    city_gdf = find_city_in_database(city_name, country_name, database_path)

    if city_gdf is None:
        return False

    # Get city information
    city_clean_name = city_gdf['UC_NM_MN'].iloc[0].replace(" ", "_").replace("/", "-")
    city_id = city_gdf['ID_HDC_G0'].iloc[0]
    polygon = city_gdf.geometry.iloc[0]

    # Validate and repair geometry if needed
    if not polygon.is_valid:
        print(f"  ⚠️  Invalid geometry detected, repairing with buffer(0)...")
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            print(f"  ✗ Could not repair geometry")
            return False
        print(f"  ✓ Geometry repaired successfully")

    print(f"\n{'='*80}")
    print(f"Extracting POIs for: {city_gdf['UC_NM_MN'].iloc[0]}")
    print(f"{'='*80}\n")

    # Save city boundary
    boundary_file = output_path / f"{city_clean_name}_boundary.gpkg"
    city_gdf.to_file(boundary_file, driver="GPKG")
    print(f"✓ Saved city boundary to: {boundary_file}\n")

    # Extract POIs for each category
    print(f"Extracting POIs for 9 categories...\n")
    results = []

    for i, (category, tag_dict) in enumerate(POI_TAGS.items(), 1):
        print(f"[{i}/9] {category.replace('_', ' ').title()}:")
        gdf_cat = query_poi_category(polygon, tag_dict, category)

        if not gdf_cat.empty:
            results.append(gdf_cat)
            print(f"  → Total: {len(gdf_cat):,} POIs\n")
        else:
            print(f"  → No POIs found\n")

    # Combine and save results
    if not results:
        print("\n❌ No POIs were extracted!")
        return False

    print(f"{'='*80}")
    print("Combining and saving results...")
    print(f"{'='*80}\n")

    all_pois = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")

    # Keep only essential columns
    essential_cols = ["geometry", "category", "osm_key", "osm_value"]
    for col in ["name", "addr:street", "addr:housenumber", "addr:city"]:
        if col in all_pois.columns:
            essential_cols.append(col)
    if "osmid" in all_pois.columns:
        essential_cols.append("osmid")
    elif "id" in all_pois.columns:
        essential_cols.append("id")

    cols_to_keep = [col for col in essential_cols if col in all_pois.columns]
    all_pois_clean = all_pois[cols_to_keep].copy()

    # Save to GeoPackage
    output_file = output_path / f"{city_clean_name}_pois_9cats.gpkg"
    all_pois_clean.to_file(output_file, layer="pois", driver="GPKG")
    print(f"✓ Saved {len(all_pois_clean):,} POIs to: {output_file}")

    # Generate and save summary
    category_counts = all_pois.groupby("category").size().sort_values(ascending=False)
    summary_file = output_path / f"{city_clean_name}_pois_summary.csv"
    category_counts.to_csv(summary_file, header=["count"])

    print(f"\n{'='*80}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*80}\n")
    print(f"City: {city_gdf['UC_NM_MN'].iloc[0]}, {city_gdf['CTR_MN_NM'].iloc[0]}")
    print(f"Total POIs: {len(all_pois_clean):,}\n")

    print("POIs by category:")
    for cat, count in category_counts.items():
        pct = 100 * count / len(all_pois_clean)
        print(f"  {cat:25s}: {count:6,} ({pct:5.1f}%)")

    print(f"\nGeometry types:")
    geom_counts = all_pois.geometry.geom_type.value_counts()
    for geom_type, count in geom_counts.items():
        pct = 100 * count / len(all_pois_clean)
        print(f"  {geom_type:15s}: {count:6,} ({pct:5.1f}%)")

    print(f"\n✓ Summary saved to: {summary_file}")
    print(f"\n{'='*80}")
    print("✓ Extraction completed successfully!")
    print(f"{'='*80}\n")

    return True


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Extract POIs from OpenStreetMap for any city in the GHS Urban Centers Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "New York" --country "United States"
  %(prog)s "London" --country "United Kingdom"
  %(prog)s "Tokyo" --output-dir ../data/raw/pois
  %(prog)s "Paris" --database custom_database.gpkg

Default paths (relative to script location):
  Database: ../data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg
  Output:   ../data/raw/pois
        """
    )

    parser.add_argument(
        "city",
        type=str,
        help="Name of the city to extract POIs for"
    )

    parser.add_argument(
        "-c", "--country",
        type=str,
        default=None,
        help="Country name to narrow down city search (optional)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (default: ../data/raw/pois)"
    )

    parser.add_argument(
        "-d", "--database",
        type=str,
        default=None,
        help="Path to GHS database GeoPackage (default: ../data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg)"
    )

    args = parser.parse_args()

    # Set default paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.database is None:
        args.database = project_root / "data" / "raw" / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"

    if args.output_dir is None:
        args.output_dir = project_root / "data" / "raw" / "pois"

    # Check if database exists
    if not Path(args.database).exists():
        print(f"❌ Error: Database file not found: {args.database}")
        print("\nPlease specify the correct path using --database")
        sys.exit(1)

    # Run extraction
    success = extract_pois(
        city_name=args.city,
        country_name=args.country,
        output_dir=args.output_dir,
        database_path=args.database
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
