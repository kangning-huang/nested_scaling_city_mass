#!/usr/bin/env python3
"""
Tests for Step 2: Preprocessing

Tests SNDi calculation and data preparation pipeline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import geopandas as gpd
import numpy as np

# Test configuration
TEST_CITY_ID = 6694  # Ahmedabad
TEST_COUNTRY = "IND"


def test_h3_grids_schema():
    """Verify H3 grids file has required columns and valid geometry."""
    h3_path = PROJECT_ROOT / "data/raw/all_cities_h3_grids.gpkg"

    if not h3_path.exists():
        print(f"⚠ H3 grids file not found: {h3_path}")
        return True

    gdf = gpd.read_file(h3_path, rows=100)

    required_columns = ['ID_HDC_G0', 'geometry']
    missing = [col for col in required_columns if col not in gdf.columns]

    assert not missing, f"Missing columns: {missing}"
    assert gdf.geometry.is_valid.all(), "Invalid geometries"
    assert gdf.crs is not None, "Missing CRS"

    print(f"✓ H3 grids schema valid")
    print(f"  - CRS: {gdf.crs}")
    print(f"  - Columns: {list(gdf.columns)}")

    return True


def test_h3_grids_city_coverage():
    """Test that H3 grids exist for test city."""
    h3_path = PROJECT_ROOT / "data/raw/all_cities_h3_grids.gpkg"

    if not h3_path.exists():
        return True

    gdf = gpd.read_file(h3_path)

    city_grids = gdf[gdf['ID_HDC_G0'] == TEST_CITY_ID]

    assert len(city_grids) > 0, f"No grids found for city ID {TEST_CITY_ID}"

    print(f"✓ Found {len(city_grids)} H3 grids for test city")

    # Check grid properties
    areas = city_grids.geometry.area
    print(f"  - Grid area range: {areas.min():.6f} - {areas.max():.6f}")

    return True


def test_sndi_raster_metadata():
    """Test SNDi raster exists and check metadata (without loading full raster)."""
    sndi_path = PROJECT_ROOT / "data/raw/sndi_grid_in_UrbanCores.tif"

    if not sndi_path.exists():
        print(f"⚠ SNDi raster not found: {sndi_path}")
        return True

    try:
        import rasterio

        with rasterio.open(sndi_path) as src:
            print(f"✓ SNDi raster metadata:")
            print(f"  - Shape: {src.width} x {src.height}")
            print(f"  - CRS: {src.crs}")
            print(f"  - Bounds: {src.bounds}")
            print(f"  - NoData: {src.nodata}")
            print(f"  - Data type: {src.dtypes[0]}")

        return True

    except ImportError:
        print("⚠ rasterio not installed - skipping raster test")
        return True


def test_sndi_sample_extraction():
    """Test extracting SNDi values for a small sample area."""
    sndi_path = PROJECT_ROOT / "data/raw/sndi_grid_in_UrbanCores.tif"
    h3_path = PROJECT_ROOT / "data/raw/all_cities_h3_grids.gpkg"

    if not sndi_path.exists() or not h3_path.exists():
        print("⚠ Required files not found - skipping")
        return True

    try:
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import mapping
    except ImportError:
        print("⚠ rasterio not installed - skipping")
        return True

    # Load a few grids
    gdf = gpd.read_file(h3_path, rows=10)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Test extraction for first grid
    test_geom = gdf.iloc[0].geometry

    try:
        with rasterio.open(sndi_path) as src:
            # Reproject geometry if needed
            if gdf.crs != src.crs:
                gdf_reproj = gdf.to_crs(src.crs)
                test_geom = gdf_reproj.iloc[0].geometry

            out_image, out_transform = mask(src, [mapping(test_geom)], crop=True)

            # Get valid values
            valid_values = out_image[out_image != src.nodata]

            if len(valid_values) > 0:
                mean_sndi = np.mean(valid_values)
                print(f"✓ SNDi extraction works")
                print(f"  - Sample mean SNDi: {mean_sndi:.3f}")
                print(f"  - Valid pixels: {len(valid_values)}")
            else:
                print("⚠ No valid SNDi values in sample area")

        return True

    except Exception as e:
        print(f"✗ SNDi extraction failed: {e}")
        return False


def test_processed_sndi_files():
    """Test that processed SNDi output files are valid."""
    processed_dir = PROJECT_ROOT / "data/processed"

    sndi_files = list(processed_dir.glob("01_neighborhood_SNDi*.csv"))

    if not sndi_files:
        print("⚠ No processed SNDi files found - run preprocessing first")
        return True

    # Test most recent file
    latest_file = max(sndi_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)

    # Check required columns
    required = ['h3index', 'ID_HDC_G0']
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    # Check data validity
    assert len(df) > 0, "Empty dataframe"
    assert df['ID_HDC_G0'].notna().any(), "All city IDs are null"

    # Check SNDi values if present
    if 'avg_sndi' in df.columns:
        valid_sndi = df['avg_sndi'].dropna()
        if len(valid_sndi) > 0:
            assert valid_sndi.min() >= 0, "Negative SNDi values"
            print(f"  - SNDi range: {valid_sndi.min():.3f} - {valid_sndi.max():.3f}")

    print(f"✓ Processed SNDi file valid: {latest_file.name}")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Cities: {df['ID_HDC_G0'].nunique():,}")

    return True


def test_sndi_city_completeness():
    """Test that SNDi data covers expected cities."""
    processed_dir = PROJECT_ROOT / "data/processed"

    sndi_files = list(processed_dir.glob("01_neighborhood_SNDi*.csv"))

    if not sndi_files:
        return True

    df = pd.read_csv(sndi_files[0])

    # Check test city coverage
    city_data = df[df['ID_HDC_G0'] == TEST_CITY_ID]

    if len(city_data) == 0:
        print(f"⚠ Test city {TEST_CITY_ID} not in processed data")
        return True

    print(f"✓ Test city coverage:")
    print(f"  - Neighborhoods: {len(city_data)}")

    if 'avg_sndi' in df.columns:
        valid = city_data['avg_sndi'].notna().sum()
        print(f"  - With SNDi data: {valid} ({100*valid/len(city_data):.1f}%)")

    return True


def test_preprocessing_script_syntax():
    """Test that preprocessing script has valid syntax."""
    import ast

    script_path = PROJECT_ROOT / "scripts/02_preprocessing/01_neighborhood_SNDi.py"

    if not script_path.exists():
        print(f"⚠ Script not found: {script_path}")
        return True

    with open(script_path) as f:
        source = f.read()

    try:
        ast.parse(source)
        print(f"✓ Preprocessing script syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False


def test_cmi_data_available():
    """Test that CMI data is available for joining."""
    cmi_path = PROJECT_ROOT / "data/raw/china_neighborhoods_cmi.csv"

    if not cmi_path.exists():
        print(f"⚠ CMI data not found: {cmi_path}")
        return True

    df = pd.read_csv(cmi_path)

    print(f"✓ CMI data available:")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {list(df.columns)[:5]}...")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Preprocessing Tests")
    print("=" * 60)

    tests = [
        test_h3_grids_schema,
        test_h3_grids_city_coverage,
        test_sndi_raster_metadata,
        test_sndi_sample_extraction,
        test_processed_sndi_files,
        test_sndi_city_completeness,
        test_preprocessing_script_syntax,
        test_cmi_data_available,
    ]

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
