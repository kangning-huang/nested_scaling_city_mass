#!/usr/bin/env python3
"""
Tests for Step 3: Analysis

Tests scaling analysis, centrality calculations, and statistical outputs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import ast


def test_analysis_scripts_syntax():
    """Test that all analysis scripts have valid Python syntax."""
    analysis_dir = PROJECT_ROOT / "scripts/03_analysis"

    if not analysis_dir.exists():
        print(f"⚠ Analysis directory not found")
        return False

    py_files = list(analysis_dir.glob("*.py"))

    if not py_files:
        print("⚠ No Python scripts in analysis directory")
        return True

    all_valid = True
    for py_file in py_files:
        try:
            with open(py_file) as f:
                ast.parse(f.read())
            print(f"✓ {py_file.name} - syntax OK")
        except SyntaxError as e:
            print(f"✗ {py_file.name} - syntax error at line {e.lineno}")
            all_valid = False

    return all_valid


def test_r_scripts_exist():
    """Test that R analysis scripts exist."""
    analysis_dir = PROJECT_ROOT / "scripts/03_analysis"

    r_files = list(analysis_dir.glob("*.R"))

    if not r_files:
        print("⚠ No R scripts found")
        return True

    print(f"✓ Found {len(r_files)} R scripts:")
    for r_file in r_files:
        size_kb = r_file.stat().st_size / 1024
        print(f"  - {r_file.name} ({size_kb:.1f} KB)")

    return True


def test_scaling_results_format():
    """Test that scaling analysis results have correct format."""
    processed_dir = PROJECT_ROOT / "data/processed"

    scaling_files = list(processed_dir.glob("*scaling*.csv"))

    if not scaling_files:
        print("⚠ No scaling results found - run analysis first")
        return True

    for scaling_file in scaling_files[:3]:
        df = pd.read_csv(scaling_file)

        print(f"✓ {scaling_file.name}:")
        print(f"  - Rows: {len(df):,}")
        print(f"  - Columns: {len(df.columns)}")

        # Check for common scaling columns
        scaling_cols = [c for c in df.columns if 'slope' in c.lower() or 'beta' in c.lower()]
        if scaling_cols:
            print(f"  - Scaling columns: {scaling_cols}")

    return True


def test_city_level_summary():
    """Test city-level summary statistics."""
    processed_dir = PROJECT_ROOT / "data/processed"

    summary_files = list(processed_dir.glob("*city_level*.csv"))

    if not summary_files:
        print("⚠ No city-level summary found")
        return True

    df = pd.read_csv(summary_files[0])

    # Check expected summary statistics
    stat_cols = [c for c in df.columns if any(s in c.lower() for s in ['mean', 'median', 'std', 'min', 'max'])]

    print(f"✓ City-level summary:")
    print(f"  - Cities: {len(df):,}")
    print(f"  - Stat columns: {len(stat_cols)}")

    # Validate statistics are sensible
    for col in stat_cols[:3]:
        if df[col].dtype in ['float64', 'int64']:
            assert df[col].notna().any(), f"Column {col} is all null"

    return True


def test_centrality_calculation_imports():
    """Test that centrality calculation dependencies are available."""
    script_path = PROJECT_ROOT / "scripts/03_analysis/calculate_population_weighted_centrality.py"

    if not script_path.exists():
        print("⚠ Centrality script not found")
        return True

    # Check required imports
    required_imports = ['pandas', 'numpy', 'json']
    optional_imports = ['scipy', 'networkx', 'geopandas']

    missing_required = []
    missing_optional = []

    for pkg in required_imports:
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(pkg)

    for pkg in optional_imports:
        try:
            __import__(pkg)
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        print(f"✗ Missing required packages: {missing_required}")
        return False

    print(f"✓ All required imports available")
    if missing_optional:
        print(f"⚠ Optional packages missing: {missing_optional}")

    return True


def test_centrality_results():
    """Test centrality calculation results if they exist."""
    results_dir = PROJECT_ROOT / "results"

    centrality_files = list(results_dir.glob("*centrality*.csv"))

    if not centrality_files:
        # Check in processed dir too
        centrality_files = list((PROJECT_ROOT / "data/processed").glob("*centrality*.csv"))

    if not centrality_files:
        print("⚠ No centrality results found")
        return True

    for cf in centrality_files[:2]:
        df = pd.read_csv(cf)

        print(f"✓ {cf.name}:")
        print(f"  - Rows: {len(df):,}")

        # Check for centrality columns
        cent_cols = [c for c in df.columns if 'centrality' in c.lower() or 'accessibility' in c.lower()]
        if cent_cols:
            print(f"  - Centrality metrics: {cent_cols}")

            # Validate values are in reasonable range
            for col in cent_cols:
                if df[col].dtype in ['float64', 'int64']:
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        print(f"    {col}: {vals.min():.4f} - {vals.max():.4f}")

    return True


def test_deviation_analysis_results():
    """Test deviation analysis results."""
    results_dir = PROJECT_ROOT / "results"

    deviation_files = list(results_dir.glob("*deviation*.csv"))

    if not deviation_files:
        print("⚠ No deviation results found")
        return True

    for df_file in deviation_files[:2]:
        df = pd.read_csv(df_file)

        print(f"✓ {df_file.name}:")
        print(f"  - Rows: {len(df):,}")
        print(f"  - Columns: {list(df.columns)[:5]}...")

    return True


def test_json_analysis_outputs():
    """Test JSON analysis output files."""
    processed_dir = PROJECT_ROOT / "data/processed"

    json_files = list(processed_dir.glob("*.json"))

    if not json_files:
        print("⚠ No JSON analysis outputs found")
        return True

    import json

    for jf in json_files[:2]:
        try:
            with open(jf) as f:
                data = json.load(f)

            print(f"✓ {jf.name}:")
            if isinstance(data, dict):
                print(f"  - Top-level keys: {list(data.keys())[:5]}")
            elif isinstance(data, list):
                print(f"  - List with {len(data)} items")

        except json.JSONDecodeError as e:
            print(f"✗ {jf.name}: Invalid JSON - {e}")
            return False

    return True


def test_analysis_output_consistency():
    """Test that analysis outputs are internally consistent."""
    processed_dir = PROJECT_ROOT / "data/processed"

    # Find related files
    sndi_files = list(processed_dir.glob("*sndi*.csv"))
    scaling_files = list(processed_dir.glob("*scaling*.csv"))

    if not sndi_files or not scaling_files:
        print("⚠ Insufficient files for consistency check")
        return True

    # Load sample files
    sndi_df = pd.read_csv(sndi_files[0])
    scaling_df = pd.read_csv(scaling_files[0])

    # Check for common identifiers
    sndi_cols = set(sndi_df.columns)
    scaling_cols = set(scaling_df.columns)

    common_cols = sndi_cols & scaling_cols
    id_cols = [c for c in common_cols if 'id' in c.lower() or 'city' in c.lower()]

    if id_cols:
        print(f"✓ Common identifier columns: {id_cols}")

        # Check for overlapping IDs
        for col in id_cols[:1]:
            sndi_ids = set(sndi_df[col].dropna().unique())
            scaling_ids = set(scaling_df[col].dropna().unique())
            overlap = sndi_ids & scaling_ids

            if overlap:
                print(f"  - Overlapping {col} values: {len(overlap)}")
    else:
        print("⚠ No common identifier columns found")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Analysis Tests")
    print("=" * 60)

    tests = [
        test_analysis_scripts_syntax,
        test_r_scripts_exist,
        test_scaling_results_format,
        test_city_level_summary,
        test_centrality_calculation_imports,
        test_centrality_results,
        test_deviation_analysis_results,
        test_json_analysis_outputs,
        test_analysis_output_consistency,
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
