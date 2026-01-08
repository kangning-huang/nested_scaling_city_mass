#!/usr/bin/env python3
"""
Tests for Step 1: Data Extraction

Tests the POI extraction pipeline with a small sample city.
Can run actual extraction (slow) or just validate existing data (fast).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd

# Test configuration
TEST_CITY = "Ahmedabad"
EXPECTED_POI_CATEGORIES = [
    'outdoor_activities', 'learning', 'supplies', 'eating',
    'moving', 'cultural_activities', 'physical_exercise',
    'services', 'health_care'
]


def test_ghs_database_schema():
    """Verify GHS database has required columns."""
    ghs_path = PROJECT_ROOT / "data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"

    gdf = gpd.read_file(ghs_path, rows=5)

    required_columns = ['ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'P15', 'geometry']

    missing = [col for col in required_columns if col not in gdf.columns]
    assert not missing, f"Missing columns: {missing}"

    # Check geometry is valid
    assert gdf.geometry.is_valid.all(), "Invalid geometries found"

    print(f"✓ GHS database schema valid ({len(gdf.columns)} columns)")
    return True


def test_city_boundary_extraction():
    """Test that city boundaries can be extracted correctly."""
    ghs_path = PROJECT_ROOT / "data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"

    gdf = gpd.read_file(ghs_path)

    # Find test city
    city_match = gdf[gdf['UC_NM_MN'].str.contains(TEST_CITY, case=False, na=False)]
    assert len(city_match) > 0, f"City '{TEST_CITY}' not found"

    city = city_match.iloc[0]

    # Validate city geometry
    assert city.geometry is not None, "City has no geometry"
    assert city.geometry.is_valid, "City geometry is invalid"
    assert city.geometry.area > 0, "City geometry has zero area"

    print(f"✓ City boundary valid: {city['UC_NM_MN']}")
    print(f"  - ID: {city['ID_HDC_G0']}")
    print(f"  - Population: {city['P15']:,.0f}")
    print(f"  - Area: {city.geometry.area:.4f} sq degrees")

    return True


def test_existing_poi_files():
    """Test that existing POI extracts are valid."""
    pois_dir = PROJECT_ROOT / "data/raw/pois"

    if not pois_dir.exists():
        print("⚠ POIs directory doesn't exist - skipping")
        return True

    # Find all POI files
    poi_files = list(pois_dir.glob("*/*_pois_9cats.gpkg"))

    if not poi_files:
        print("⚠ No POI files found - skipping")
        return True

    # Test a sample of files
    sample_size = min(3, len(poi_files))
    sample_files = sorted(poi_files, key=lambda p: p.stat().st_size)[:sample_size]

    for poi_file in sample_files:
        gdf = gpd.read_file(poi_file)

        # Check required columns
        assert 'geometry' in gdf.columns, f"{poi_file.name}: missing geometry"
        assert 'category' in gdf.columns, f"{poi_file.name}: missing category"

        # Check categories are valid
        categories = set(gdf['category'].unique())
        invalid = categories - set(EXPECTED_POI_CATEGORIES)
        assert not invalid, f"{poi_file.name}: unexpected categories {invalid}"

        city_name = poi_file.parent.name
        print(f"✓ {city_name}: {len(gdf)} POIs, {len(categories)} categories")

    return True


def test_poi_summary_files():
    """Test that POI summary CSVs are valid."""
    pois_dir = PROJECT_ROOT / "data/raw/pois"

    if not pois_dir.exists():
        return True

    summary_files = list(pois_dir.glob("*/*_pois_summary.csv"))

    if not summary_files:
        print("⚠ No summary files found")
        return True

    for summary_file in summary_files[:3]:
        df = pd.read_csv(summary_file)

        # Should have category and count columns
        assert len(df.columns) >= 2, f"{summary_file.name}: too few columns"

        # All counts should be non-negative
        count_col = df.columns[1]  # Usually second column
        if df[count_col].dtype in ['int64', 'float64']:
            assert (df[count_col] >= 0).all(), f"{summary_file.name}: negative counts"

        city_name = summary_file.parent.name
        print(f"✓ {city_name} summary valid")

    return True


def test_poi_extraction_dry_run():
    """
    Dry run of POI extraction - checks script can load without running.

    For actual extraction test, use test_poi_extraction_live().
    """
    import ast

    script_path = PROJECT_ROOT / "scripts/01_data_extraction/01_download_OSM_city_pois.py"

    if not script_path.exists():
        print(f"⚠ Script not found: {script_path}")
        return True

    # Check syntax
    with open(script_path) as f:
        source = f.read()

    try:
        ast.parse(source)
        print(f"✓ POI extraction script syntax valid")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

    # Check key imports are available
    try:
        import osmnx
        print(f"✓ osmnx available (version {osmnx.__version__})")
    except ImportError:
        print("⚠ osmnx not installed - extraction won't work")

    return True


def test_poi_extraction_live(city_name: str = "Ahmedabad", max_pois: int = 100):
    """
    Actually run POI extraction for a small city.

    This is a slow test (~2-5 minutes) - only run when needed.

    Args:
        city_name: City to extract POIs for
        max_pois: Stop early after this many POIs (for speed)
    """
    try:
        import osmnx as ox
    except ImportError:
        print("⚠ osmnx not installed - skipping live test")
        return True

    # Load city boundary
    ghs_path = PROJECT_ROOT / "data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"
    gdf = gpd.read_file(ghs_path)
    city = gdf[gdf['UC_NM_MN'].str.contains(city_name, case=False, na=False)].iloc[0]

    print(f"Testing POI extraction for {city['UC_NM_MN']}...")

    # Test just one category (fastest)
    test_tags = {'amenity': ['restaurant', 'cafe']}

    try:
        pois = ox.features_from_polygon(city.geometry, tags=test_tags)
        print(f"✓ Extracted {len(pois)} eating POIs from {city_name}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live extraction test")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Data Extraction Tests")
    print("=" * 60)

    tests = [
        test_ghs_database_schema,
        test_city_boundary_extraction,
        test_existing_poi_files,
        test_poi_summary_files,
        test_poi_extraction_dry_run,
    ]

    if args.live:
        tests.append(test_poi_extraction_live)

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
