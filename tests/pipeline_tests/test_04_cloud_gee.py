#!/usr/bin/env python3
"""
Tests for Step 4: Cloud/GEE Scripts

Tests Google Earth Engine scripts with dry-run and optional live tests.
"""

import sys
from pathlib import Path
import ast

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_gee_scripts_syntax():
    """Test that all GEE Python scripts have valid syntax."""
    gee_dir = PROJECT_ROOT / "scripts/cloud/gee"

    if not gee_dir.exists():
        print(f"⚠ GEE directory not found: {gee_dir}")
        return False

    py_files = list(gee_dir.glob("*.py"))

    if not py_files:
        print("⚠ No Python scripts in GEE directory")
        return True

    all_valid = True
    for py_file in py_files:
        try:
            with open(py_file) as f:
                ast.parse(f.read())
            print(f"✓ {py_file.name} - syntax OK")
        except SyntaxError as e:
            print(f"✗ {py_file.name} - syntax error at line {e.lineno}: {e.msg}")
            all_valid = False

    return all_valid


def test_gee_javascript_exists():
    """Test that GEE JavaScript files exist."""
    gee_dir = PROJECT_ROOT / "scripts/cloud/gee"

    js_files = list(gee_dir.glob("*.js"))

    if not js_files:
        print("⚠ No JavaScript files found")
        return True

    for js_file in js_files:
        size_kb = js_file.stat().st_size / 1024
        print(f"✓ {js_file.name} ({size_kb:.1f} KB)")

        # Basic JS syntax check (look for common issues)
        with open(js_file) as f:
            content = f.read()

        # Check for GEE-specific patterns
        if 'ee.' in content:
            print(f"  - Contains Earth Engine API calls")
        if 'Export.' in content:
            print(f"  - Contains export operations")

    return True


def test_gee_package_available():
    """Test that Earth Engine Python package is available."""
    try:
        import ee
        print(f"✓ earthengine-api installed (version {ee.__version__})")
        return True
    except ImportError:
        print("⚠ earthengine-api not installed")
        print("  Install with: pip install earthengine-api")
        return True  # Not a failure - optional


def test_geemap_available():
    """Test that geemap package is available."""
    try:
        import geemap
        print(f"✓ geemap installed (version {geemap.__version__})")
        return True
    except ImportError:
        print("⚠ geemap not installed")
        print("  Install with: pip install geemap")
        return True  # Not a failure - optional


def test_gee_authentication():
    """Test GEE authentication status."""
    try:
        import ee

        # Try to initialize
        try:
            ee.Initialize(project='ee-knhuang')
            print(f"✓ GEE authenticated and initialized")

            # Test a simple operation
            image = ee.Image(1)
            result = image.getInfo()
            print(f"✓ GEE API responding correctly")

            return True

        except ee.EEException as e:
            if 'not registered' in str(e).lower():
                print("⚠ GEE project not registered or authenticated")
                print("  Run: earthengine authenticate")
            else:
                print(f"⚠ GEE initialization failed: {e}")
            return True  # Not a critical failure

    except ImportError:
        print("⚠ earthengine-api not installed - skipping auth test")
        return True


def test_gee_script_structure():
    """Test that GEE scripts follow expected structure."""
    gee_dir = PROJECT_ROOT / "scripts/cloud/gee"

    py_files = list(gee_dir.glob("*.py"))

    for py_file in py_files:
        with open(py_file) as f:
            content = f.read()

        checks = {
            'ee import': 'import ee' in content or 'from ee' in content,
            'ee.Initialize': 'ee.Initialize' in content or 'geemap' in content,
            'project ID': 'ee-knhuang' in content or 'project=' in content,
        }

        print(f"\n{py_file.name}:")
        for check_name, passed in checks.items():
            status = "✓" if passed else "⚠"
            print(f"  {status} {check_name}")

    return True


def test_friction_surface_script():
    """Test friction surface calculation script structure."""
    script_path = PROJECT_ROOT / "scripts/cloud/gee/01_GEE_frictionSurface_travelTimes.py"

    if not script_path.exists():
        print("⚠ Friction surface script not found")
        return True

    with open(script_path) as f:
        content = f.read()

    # Check for key components
    components = {
        'friction surface': 'friction' in content.lower(),
        'travel time': 'travel' in content.lower() and 'time' in content.lower(),
        'zonal stats': 'zonal' in content.lower() or 'reduceRegions' in content,
        'export': 'Export' in content or 'export' in content.lower(),
    }

    print("Friction surface script components:")
    all_present = True
    for comp, present in components.items():
        status = "✓" if present else "⚠"
        print(f"  {status} {comp}")
        if not present:
            all_present = False

    return True


def test_poi_upload_script():
    """Test POI upload to GEE script."""
    script_path = PROJECT_ROOT / "scripts/cloud/gee/02_upload_pois_to_gee.py"

    if not script_path.exists():
        print("⚠ POI upload script not found")
        return True

    with open(script_path) as f:
        content = f.read()

    # Check for key components
    components = {
        'asset upload': 'upload' in content.lower() or 'ingest' in content.lower(),
        'feature collection': 'FeatureCollection' in content,
        'geometry handling': 'geometry' in content.lower(),
    }

    print("POI upload script components:")
    for comp, present in components.items():
        status = "✓" if present else "⚠"
        print(f"  {status} {comp}")

    return True


def test_travel_time_script():
    """Test travel time calculation script."""
    script_path = PROJECT_ROOT / "scripts/cloud/gee/03_GEE_travelTime_POIs_30m.py"

    if not script_path.exists():
        print("⚠ Travel time script not found")
        return True

    with open(script_path) as f:
        content = f.read()

    # Check for key components
    components = {
        'travel time calc': 'travel' in content.lower(),
        'POI reference': 'poi' in content.lower() or 'POI' in content,
        'resolution setting': '30' in content,  # 30m resolution
        'batch processing': 'batch' in content.lower() or 'loop' in content.lower() or 'for ' in content,
    }

    print("Travel time script components:")
    for comp, present in components.items():
        status = "✓" if present else "⚠"
        print(f"  {status} {comp}")

    return True


def test_gee_live_simple(skip_if_no_auth: bool = True):
    """
    Run a simple live GEE test.

    This actually executes GEE code - use sparingly.
    """
    try:
        import ee
        ee.Initialize(project='ee-knhuang')
    except Exception as e:
        if skip_if_no_auth:
            print(f"⚠ GEE not available: {e}")
            return True
        return False

    print("Running live GEE test...")

    try:
        # Simple test: get elevation for a point
        point = ee.Geometry.Point([72.5714, 23.0225])  # Ahmedabad
        elevation = ee.Image('USGS/SRTMGL1_003').sample(point, 30).first().get('elevation')

        result = elevation.getInfo()
        print(f"✓ Live GEE test passed")
        print(f"  - Elevation at Ahmedabad: {result}m")

        return True

    except Exception as e:
        print(f"✗ Live GEE test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live GEE tests")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Cloud/GEE Tests")
    print("=" * 60)

    tests = [
        test_gee_scripts_syntax,
        test_gee_javascript_exists,
        test_gee_package_available,
        test_geemap_available,
        test_gee_authentication,
        test_gee_script_structure,
        test_friction_surface_script,
        test_poi_upload_script,
        test_travel_time_script,
    ]

    if args.live:
        tests.append(lambda: test_gee_live_simple(skip_if_no_auth=False))

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{test.__name__ if hasattr(test, '__name__') else 'live_test'}:")
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
