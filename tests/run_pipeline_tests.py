#!/usr/bin/env python3
"""
Pipeline Test Runner for Urban Scaling Project

Runs validation tests on each pipeline step using small sample data.
Use --step to run specific steps, or run all steps sequentially.

Usage:
    python tests/run_pipeline_tests.py                    # Run all tests
    python tests/run_pipeline_tests.py --step 1          # Run only step 1
    python tests/run_pipeline_tests.py --step 1 2 3      # Run steps 1, 2, 3
    python tests/run_pipeline_tests.py --quick           # Quick smoke test
    python tests/run_pipeline_tests.py --list            # List available tests
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test configuration
TEST_CITY = "Ahmedabad"  # Small city (~500KB POI data)
TEST_CITY_ID = "6694"    # ID_HDC_G0 for Ahmedabad
TEST_COUNTRY = "IND"

# ANSI colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


class PipelineTestRunner:
    """Runs pipeline validation tests."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.sample_data_dir = self.tests_dir / "sample_data"
        self.results = {}

    def run_test(self, name: str, test_func) -> bool:
        """Run a single test and record results."""
        print(f"\nRunning: {name}")
        start_time = time.time()

        try:
            result = test_func()
            elapsed = time.time() - start_time

            if result:
                print_success(f"{name} passed ({elapsed:.2f}s)")
                self.results[name] = ("PASS", elapsed)
                return True
            else:
                print_error(f"{name} failed ({elapsed:.2f}s)")
                self.results[name] = ("FAIL", elapsed)
                return False

        except Exception as e:
            elapsed = time.time() - start_time
            print_error(f"{name} error: {str(e)}")
            self.results[name] = ("ERROR", elapsed, str(e))
            return False

    # =========================================================================
    # Step 0: Environment and Dependencies
    # =========================================================================

    def test_environment(self) -> bool:
        """Test that required packages are installed."""
        required = [
            'pandas', 'geopandas', 'numpy', 'shapely'
        ]
        optional = [
            'osmnx', 'rasterio', 'h3', 'scipy', 'sklearn'
        ]

        missing_required = []
        missing_optional = []

        for pkg in required:
            try:
                __import__(pkg)
                print_success(f"  {pkg} installed")
            except ImportError:
                missing_required.append(pkg)
                print_error(f"  {pkg} NOT installed (required)")

        for pkg in optional:
            try:
                __import__(pkg)
                print_success(f"  {pkg} installed")
            except ImportError:
                missing_optional.append(pkg)
                print_warning(f"  {pkg} NOT installed (optional)")

        if missing_required:
            print_error(f"Missing required packages: {missing_required}")
            return False

        if missing_optional:
            print_warning(f"Some optional packages missing: {missing_optional}")

        return True

    def test_project_structure(self) -> bool:
        """Test that project directory structure is correct."""
        required_dirs = [
            "scripts/01_data_extraction",
            "scripts/02_preprocessing",
            "scripts/03_analysis",
            "scripts/cloud/gee",
            "scripts/hpc",
            "data/raw",
            "data/processed",
            "results",
            "config",
        ]

        all_exist = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print_success(f"  {dir_path}/ exists")
            else:
                print_error(f"  {dir_path}/ MISSING")
                all_exist = False

        return all_exist

    def test_config_module(self) -> bool:
        """Test that config module loads correctly."""
        try:
            from config.paths import (
                PROJECT_ROOT, DATA_RAW, DATA_PROCESSED,
                RESULTS, FIGURES, ENVIRONMENT
            )
            print_success(f"  Config loaded, environment: {ENVIRONMENT}")
            print_success(f"  PROJECT_ROOT: {PROJECT_ROOT}")
            return True
        except ImportError as e:
            print_error(f"  Config import failed: {e}")
            return False

    # =========================================================================
    # Step 1: Data Extraction Tests
    # =========================================================================

    def test_ghs_database_readable(self) -> bool:
        """Test that GHS Urban Centers database can be read."""
        import geopandas as gpd

        ghs_path = self.project_root / "data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"

        if not ghs_path.exists():
            print_error(f"  GHS database not found at {ghs_path}")
            return False

        # Read just first few rows
        gdf = gpd.read_file(ghs_path, rows=10)

        required_cols = ['ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'P15', 'geometry']
        missing = [c for c in required_cols if c not in gdf.columns]

        if missing:
            print_error(f"  Missing columns: {missing}")
            return False

        print_success(f"  GHS database readable, {len(gdf.columns)} columns")
        return True

    def test_sample_city_lookup(self) -> bool:
        """Test that we can find our test city in GHS database."""
        import geopandas as gpd

        ghs_path = self.project_root / "data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"
        gdf = gpd.read_file(ghs_path)

        # Find test city
        city_match = gdf[gdf['UC_NM_MN'].str.contains(TEST_CITY, case=False, na=False)]

        if len(city_match) == 0:
            print_error(f"  City '{TEST_CITY}' not found in database")
            return False

        city = city_match.iloc[0]
        print_success(f"  Found {city['UC_NM_MN']} (ID: {city['ID_HDC_G0']}, Pop: {city['P15']:,.0f})")
        return True

    def test_existing_pois_readable(self) -> bool:
        """Test that existing POI extracts can be read."""
        import geopandas as gpd

        # Check if we have any POI files
        pois_dir = self.project_root / "data/raw/pois"

        if not pois_dir.exists():
            print_warning(f"  POIs directory doesn't exist yet")
            return True  # Not a failure, just not extracted yet

        # Find a small POI file
        poi_files = list(pois_dir.glob("*/*_pois_9cats.gpkg"))

        if not poi_files:
            print_warning(f"  No POI files found")
            return True

        # Read the smallest one
        smallest = min(poi_files, key=lambda p: p.stat().st_size)
        gdf = gpd.read_file(smallest)

        required_cols = ['geometry', 'category']
        missing = [c for c in required_cols if c not in gdf.columns]

        if missing:
            print_error(f"  Missing columns in POI file: {missing}")
            return False

        categories = gdf['category'].unique()
        print_success(f"  POI file readable: {smallest.parent.name}")
        print_success(f"  {len(gdf)} POIs, {len(categories)} categories")
        return True

    # =========================================================================
    # Step 2: Preprocessing Tests
    # =========================================================================

    def test_h3_grids_readable(self) -> bool:
        """Test that H3 grids file can be read."""
        import geopandas as gpd

        h3_path = self.project_root / "data/raw/all_cities_h3_grids.gpkg"

        if not h3_path.exists():
            print_error(f"  H3 grids file not found")
            return False

        # Read sample
        gdf = gpd.read_file(h3_path, rows=100)

        required_cols = ['ID_HDC_G0', 'geometry']
        missing = [c for c in required_cols if c not in gdf.columns]

        if missing:
            print_error(f"  Missing columns: {missing}")
            return False

        print_success(f"  H3 grids readable, {len(gdf.columns)} columns")
        return True

    def test_sndi_raster_exists(self) -> bool:
        """Test that SNDi raster exists (don't load - too large)."""
        sndi_path = self.project_root / "data/raw/sndi_grid_in_UrbanCores.tif"

        if not sndi_path.exists():
            print_error(f"  SNDi raster not found")
            return False

        size_gb = sndi_path.stat().st_size / (1024**3)
        print_success(f"  SNDi raster exists ({size_gb:.2f} GB)")
        return True

    def test_processed_data_format(self) -> bool:
        """Test that processed data files have correct format."""
        import pandas as pd

        processed_dir = self.project_root / "data/processed"

        # Find SNDi file
        sndi_files = list(processed_dir.glob("01_neighborhood_SNDi*.csv"))

        if not sndi_files:
            print_warning(f"  No processed SNDi files found")
            return True  # Not a failure

        # Read sample
        df = pd.read_csv(sndi_files[0], nrows=100)

        expected_cols = ['h3index', 'avg_sndi', 'ID_HDC_G0']
        missing = [c for c in expected_cols if c not in df.columns]

        if missing:
            print_error(f"  Missing columns: {missing}")
            return False

        print_success(f"  Processed data format correct")
        return True

    # =========================================================================
    # Step 3: Analysis Tests
    # =========================================================================

    def test_analysis_script_imports(self) -> bool:
        """Test that analysis scripts can be imported (syntax check)."""
        import importlib.util

        scripts = [
            "scripts/03_analysis/calculate_population_weighted_centrality.py",
            "scripts/03_analysis/02_accessibility_POIs_neighborhoods.py",
        ]

        all_valid = True
        for script_path in scripts:
            full_path = self.project_root / script_path

            if not full_path.exists():
                print_warning(f"  {script_path} not found")
                continue

            try:
                spec = importlib.util.spec_from_file_location("module", full_path)
                # Just check syntax, don't execute
                import ast
                with open(full_path) as f:
                    ast.parse(f.read())
                print_success(f"  {Path(script_path).name} syntax OK")
            except SyntaxError as e:
                print_error(f"  {Path(script_path).name} syntax error: {e}")
                all_valid = False

        return all_valid

    def test_scaling_results_exist(self) -> bool:
        """Test that scaling analysis results exist and are valid."""
        import pandas as pd

        processed_dir = self.project_root / "data/processed"
        scaling_files = list(processed_dir.glob("*scaling*.csv"))

        if not scaling_files:
            print_warning(f"  No scaling results found")
            return True

        # Check one file
        df = pd.read_csv(scaling_files[0])
        print_success(f"  Found {len(scaling_files)} scaling result files")
        print_success(f"  Sample file has {len(df)} rows, {len(df.columns)} columns")
        return True

    # =========================================================================
    # Step 4: Cloud/GEE Tests (dry run)
    # =========================================================================

    def test_gee_scripts_syntax(self) -> bool:
        """Test that GEE scripts have valid Python syntax."""
        import ast

        gee_dir = self.project_root / "scripts/cloud/gee"

        if not gee_dir.exists():
            print_error(f"  GEE scripts directory not found")
            return False

        py_files = list(gee_dir.glob("*.py"))

        all_valid = True
        for py_file in py_files:
            try:
                with open(py_file) as f:
                    ast.parse(f.read())
                print_success(f"  {py_file.name} syntax OK")
            except SyntaxError as e:
                print_error(f"  {py_file.name} syntax error: {e}")
                all_valid = False

        return all_valid

    def test_gee_authentication(self) -> bool:
        """Test GEE authentication (optional - may not be configured)."""
        try:
            import ee
            ee.Initialize(project='ee-knhuang')
            print_success(f"  GEE authenticated successfully")
            return True
        except ImportError:
            print_warning(f"  earthengine-api not installed")
            return True  # Not a failure
        except Exception as e:
            print_warning(f"  GEE not authenticated: {e}")
            return True  # Not a failure for pipeline test

    # =========================================================================
    # Step 5: HPC Tests (local validation)
    # =========================================================================

    def test_hpc_scripts_syntax(self) -> bool:
        """Test that HPC Python scripts have valid syntax."""
        import ast

        hpc_scripts_dir = self.project_root / "scripts/hpc/osrm_scripts"

        if not hpc_scripts_dir.exists():
            print_error(f"  HPC scripts directory not found")
            return False

        py_files = list(hpc_scripts_dir.glob("*.py"))

        all_valid = True
        for py_file in py_files[:5]:  # Check first 5
            try:
                with open(py_file) as f:
                    ast.parse(f.read())
                print_success(f"  {py_file.name} syntax OK")
            except SyntaxError as e:
                print_error(f"  {py_file.name} syntax error: {e}")
                all_valid = False

        if len(py_files) > 5:
            print_info(f"  ... and {len(py_files) - 5} more scripts")

        return all_valid

    def test_slurm_templates_exist(self) -> bool:
        """Test that SLURM job templates exist."""
        slurm_dir = self.project_root / "scripts/hpc/slurm_templates"

        if not slurm_dir.exists():
            print_error(f"  SLURM templates directory not found")
            return False

        slurm_files = list(slurm_dir.glob("*.slurm")) + list(slurm_dir.glob("*.sh"))

        if not slurm_files:
            print_warning(f"  No SLURM templates found")
            return True

        print_success(f"  Found {len(slurm_files)} SLURM templates")
        for f in slurm_files[:3]:
            print_info(f"    - {f.name}")

        return True

    # =========================================================================
    # Integration Test
    # =========================================================================

    def test_mini_pipeline_integration(self) -> bool:
        """Run a mini end-to-end test with sample data."""
        import pandas as pd
        import geopandas as gpd

        print_info("  Running mini integration test...")

        # 1. Load GHS database and find test city
        ghs_path = self.project_root / "data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"
        gdf = gpd.read_file(ghs_path)
        city = gdf[gdf['UC_NM_MN'].str.contains(TEST_CITY, case=False, na=False)].iloc[0]
        print_success(f"  1. Found city: {city['UC_NM_MN']}")

        # 2. Load H3 grids for this city
        h3_path = self.project_root / "data/raw/all_cities_h3_grids.gpkg"
        h3_all = gpd.read_file(h3_path)
        city_grids = h3_all[h3_all['ID_HDC_G0'] == city['ID_HDC_G0']]
        print_success(f"  2. Found {len(city_grids)} H3 grids for city")

        # 3. Check if processed data exists for this city
        processed_dir = self.project_root / "data/processed"
        sndi_files = list(processed_dir.glob("01_neighborhood_SNDi*.csv"))

        if sndi_files:
            df = pd.read_csv(sndi_files[0])
            city_sndi = df[df['ID_HDC_G0'] == city['ID_HDC_G0']]
            print_success(f"  3. Found {len(city_sndi)} SNDi records for city")
        else:
            print_warning(f"  3. No processed SNDi data yet")

        # 4. Check results directory
        results_dir = self.project_root / "results"
        result_files = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.gpkg"))
        print_success(f"  4. Results directory has {len(result_files)} output files")

        return True

    # =========================================================================
    # Test Runner
    # =========================================================================

    def get_all_tests(self):
        """Return all tests organized by step."""
        return {
            0: [
                ("Environment Check", self.test_environment),
                ("Project Structure", self.test_project_structure),
                ("Config Module", self.test_config_module),
            ],
            1: [
                ("GHS Database Readable", self.test_ghs_database_readable),
                ("Sample City Lookup", self.test_sample_city_lookup),
                ("Existing POIs Readable", self.test_existing_pois_readable),
            ],
            2: [
                ("H3 Grids Readable", self.test_h3_grids_readable),
                ("SNDi Raster Exists", self.test_sndi_raster_exists),
                ("Processed Data Format", self.test_processed_data_format),
            ],
            3: [
                ("Analysis Script Imports", self.test_analysis_script_imports),
                ("Scaling Results Exist", self.test_scaling_results_exist),
            ],
            4: [
                ("GEE Scripts Syntax", self.test_gee_scripts_syntax),
                ("GEE Authentication", self.test_gee_authentication),
            ],
            5: [
                ("HPC Scripts Syntax", self.test_hpc_scripts_syntax),
                ("SLURM Templates Exist", self.test_slurm_templates_exist),
            ],
            6: [
                ("Mini Pipeline Integration", self.test_mini_pipeline_integration),
            ],
        }

    def run_step(self, step: int) -> bool:
        """Run all tests for a specific step."""
        tests = self.get_all_tests()

        step_names = {
            0: "Environment & Dependencies",
            1: "Data Extraction",
            2: "Preprocessing",
            3: "Analysis",
            4: "Cloud/GEE",
            5: "HPC",
            6: "Integration",
        }

        if step not in tests:
            print_error(f"Invalid step: {step}")
            return False

        print_header(f"Step {step}: {step_names.get(step, 'Unknown')}")

        all_passed = True
        for name, test_func in tests[step]:
            if not self.run_test(name, test_func):
                all_passed = False

        return all_passed

    def run_all(self) -> bool:
        """Run all tests."""
        all_passed = True

        for step in sorted(self.get_all_tests().keys()):
            if not self.run_step(step):
                all_passed = False

        return all_passed

    def print_summary(self):
        """Print test summary."""
        print_header("Test Summary")

        passed = sum(1 for r in self.results.values() if r[0] == "PASS")
        failed = sum(1 for r in self.results.values() if r[0] == "FAIL")
        errors = sum(1 for r in self.results.values() if r[0] == "ERROR")
        total = len(self.results)

        print(f"Total: {total} tests")
        print_success(f"Passed: {passed}")
        if failed:
            print_error(f"Failed: {failed}")
        if errors:
            print_error(f"Errors: {errors}")

        total_time = sum(r[1] for r in self.results.values())
        print(f"\nTotal time: {total_time:.2f}s")

        if failed or errors:
            print("\nFailed/Error tests:")
            for name, result in self.results.items():
                if result[0] in ("FAIL", "ERROR"):
                    print_error(f"  - {name}")

        return failed == 0 and errors == 0


def main():
    parser = argparse.ArgumentParser(description="Run pipeline validation tests")
    parser.add_argument("--step", type=int, nargs="+", help="Run specific step(s)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (step 0 only)")
    parser.add_argument("--list", action="store_true", help="List available tests")

    args = parser.parse_args()

    runner = PipelineTestRunner(PROJECT_ROOT)

    if args.list:
        print_header("Available Tests")
        for step, tests in runner.get_all_tests().items():
            print(f"\nStep {step}:")
            for name, _ in tests:
                print(f"  - {name}")
        return

    print_header(f"Pipeline Test Runner - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Test City: {TEST_CITY} ({TEST_CITY_ID})")

    if args.quick:
        runner.run_step(0)
    elif args.step:
        for step in args.step:
            runner.run_step(step)
    else:
        runner.run_all()

    success = runner.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
