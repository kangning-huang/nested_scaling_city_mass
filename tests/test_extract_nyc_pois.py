"""
Test script to extract 9 categories of POI from OpenStreetMap for New York City
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
from pathlib import Path

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


def query_category(polygon, tag_dict, category_name):
    """
    Query POIs for a specific category from OpenStreetMap

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
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
            # Overpass tag filter
            tags = {key: v}

            try:
                print(f"  Querying {key}={v}...")
                gdf = ox.features_from_polygon(polygon, tags=tags)

                if not gdf.empty:
                    gdf["category"] = category_name
                    gdf["osm_key"] = key
                    gdf["osm_value"] = v
                    frames.append(gdf)
                    print(f"    Found {len(gdf)} features")
                else:
                    print(f"    No features found")

            except Exception as e:
                print(f"    Query failed: {key}={v} - {e}")

    if frames:
        result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
        return result
    else:
        return gpd.GeoDataFrame(columns=["geometry", "category", "osm_key", "osm_value"],
                               crs="EPSG:4326")


def extract_nyc_pois(output_dir="./", boundary_file=None, city_id=945):
    """
    Extract POIs for New York City from OpenStreetMap

    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    boundary_file : str, optional
        Path to GeoPackage containing city boundaries (H3 grids)
    city_id : int
        ID_HDC_G0 value for the city to extract (default: 945 for NYC)
    """
    print("="*80)
    print("Starting NYC POI Extraction from OpenStreetMap")
    print("="*80)

    # 1. Load NYC boundary from file or fetch from OSM
    print("\n[1/3] Loading New York City boundary...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        if boundary_file and Path(boundary_file).exists():
            # Load from provided GeoPackage file
            print(f"  Loading from file: {boundary_file}")
            all_cities = gpd.read_file(boundary_file)

            # Filter for New York City (ID_HDC_G0 = 945)
            city_grids = all_cities[all_cities['ID_HDC_G0'] == city_id].copy()

            if city_grids.empty:
                print(f"  Error: City ID {city_id} not found in the file")
                return

            # Dissolve H3 grids to get city boundary
            city = gpd.GeoDataFrame(
                {'ID_HDC_G0': [city_id],
                 'UC_NM_MN': [city_grids['UC_NM_MN'].iloc[0]],
                 'CTR_MN_NM': [city_grids['CTR_MN_NM'].iloc[0]]},
                geometry=[city_grids.union_all()],
                crs=city_grids.crs
            )
            polygon = city.geometry.iloc[0]

            print(f"  Successfully loaded NYC boundary")
            print(f"  City: {city['UC_NM_MN'].iloc[0]}, {city['CTR_MN_NM'].iloc[0]}")
            print(f"  Number of H3 grid cells: {len(city_grids)}")
            print(f"  Boundary type: {type(polygon)}")

        else:
            # Fallback: Get NYC boundary using osmnx
            print("  Fetching from OpenStreetMap (no boundary file provided)...")
            city = ox.geocode_to_gdf("New York City, New York, USA")
            polygon = city.geometry.union_all()
            print(f"  Successfully fetched NYC boundary")
            print(f"  Boundary type: {type(polygon)}")

        # Save boundary for reference
        city.to_file(output_path / "nyc_boundary.gpkg", driver="GPKG")
        print(f"  Saved boundary to: {output_path / 'nyc_boundary.gpkg'}")

    except Exception as e:
        print(f"  Error loading NYC boundary: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Extract POIs for each category
    print("\n[2/3] Extracting POIs for 9 categories...")
    results = []

    for i, (category, tag_dict) in enumerate(POI_TAGS.items(), 1):
        print(f"\n[Category {i}/9] Extracting: {category}")
        gdf_cat = query_category(polygon, tag_dict, category)

        if not gdf_cat.empty:
            results.append(gdf_cat)
            print(f"  Total features for {category}: {len(gdf_cat)}")
        else:
            print(f"  No features found for {category}")

    # 3. Combine and save results
    print("\n[3/3] Combining and saving results...")

    if results:
        all_pois = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")

        # Clean column names - remove problematic characters
        # Keep only essential columns to avoid field name issues
        essential_cols = ["geometry", "category", "osm_key", "osm_value"]

        # Add name and address columns if they exist
        for col in ["name", "addr:street", "addr:housenumber", "addr:city"]:
            if col in all_pois.columns:
                essential_cols.append(col)

        # Keep OSM ID if available
        if "osmid" in all_pois.columns:
            essential_cols.append("osmid")
        elif "id" in all_pois.columns:
            essential_cols.append("id")

        # Select only essential columns that exist
        cols_to_keep = [col for col in essential_cols if col in all_pois.columns]
        all_pois_clean = all_pois[cols_to_keep].copy()

        # Save combined results
        output_file = output_path / "nyc_pois_9cats.gpkg"
        all_pois_clean.to_file(output_file, layer="pois", driver="GPKG")
        print(f"\n  Saved {len(all_pois_clean)} POIs to: {output_file}")

        # Print summary statistics
        print("\n" + "="*80)
        print("EXTRACTION SUMMARY")
        print("="*80)
        print(f"\nTotal POIs extracted: {len(all_pois)}")
        print("\nPOIs by category:")
        category_counts = all_pois.groupby("category").size().sort_values(ascending=False)
        for cat, count in category_counts.items():
            print(f"  {cat:25s}: {count:6d} POIs")

        print("\nGeometry types:")
        geom_counts = all_pois.geometry.geom_type.value_counts()
        for geom_type, count in geom_counts.items():
            print(f"  {geom_type:15s}: {count:6d}")

        # Save summary to CSV
        summary_file = output_path / "nyc_pois_summary.csv"
        category_counts.to_csv(summary_file, header=["count"])
        print(f"\nSummary saved to: {summary_file}")

        print("\n" + "="*80)
        print("Extraction completed successfully!")
        print("="*80)

    else:
        print("\n  WARNING: No POIs were extracted!")


if __name__ == "__main__":
    # Run extraction in the tests directory
    output_directory = Path(__file__).parent

    # Path to the boundary file (relative to this script)
    boundary_file_path = output_directory.parent / "data" / "raw" / "all_cities_h3_grids.gpkg"

    # Extract POIs using the boundary file
    extract_nyc_pois(
        output_dir=output_directory,
        boundary_file=str(boundary_file_path),
        city_id=945  # New York City ID
    )
