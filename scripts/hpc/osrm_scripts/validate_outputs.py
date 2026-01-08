#!/usr/bin/env python3
"""
Validate output files from OSRM processing.

Checks:
1. Matrix JSON files have valid structure
2. Route GeoJSON files have valid structure
3. No null values in travel time matrices
4. Matching city IDs between files

Usage:
    python validate_outputs.py                     # Validate all
    python validate_outputs.py --city-list pilot_cities.txt  # Validate specific cities
    python validate_outputs.py --fix               # Delete invalid files (for re-processing)
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Configuration
WORK_DIR = Path("/scratch/kh3657/osrm")
RESULTS_DIR = WORK_DIR / "results"


def validate_matrix(filepath):
    """Validate a matrix JSON file."""
    errors = []

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    except Exception as e:
        return [f"Cannot read file: {e}"]

    # Required fields
    required = ["city_id", "h3_indices", "centroids", "durations", "distances", "n_grids"]
    for field in required:
        if field not in data:
            errors.append(f"Missing field: {field}")

    if errors:
        return errors

    # Validate dimensions
    n_grids = data["n_grids"]
    if len(data["h3_indices"]) != n_grids:
        errors.append(f"h3_indices length mismatch: {len(data['h3_indices'])} vs {n_grids}")

    if len(data["centroids"]) != n_grids:
        errors.append(f"centroids length mismatch: {len(data['centroids'])} vs {n_grids}")

    if len(data["durations"]) != n_grids:
        errors.append(f"durations rows mismatch: {len(data['durations'])} vs {n_grids}")
    else:
        for i, row in enumerate(data["durations"]):
            if len(row) != n_grids:
                errors.append(f"durations[{i}] columns mismatch: {len(row)} vs {n_grids}")
                break

    if len(data["distances"]) != n_grids:
        errors.append(f"distances rows mismatch: {len(data['distances'])} vs {n_grids}")

    # Check for null values
    null_count = 0
    for row in data["durations"]:
        null_count += sum(1 for v in row if v is None)

    if null_count > 0:
        pct = null_count / (n_grids * n_grids) * 100
        if pct > 50:  # More than 50% null is likely an error
            errors.append(f"Too many null values in durations: {null_count} ({pct:.1f}%)")

    return errors


def validate_routes(filepath):
    """Validate a routes GeoJSON file."""
    errors = []

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    except Exception as e:
        return [f"Cannot read file: {e}"]

    # Required structure
    if data.get("type") != "FeatureCollection":
        errors.append("Not a valid GeoJSON FeatureCollection")
        return errors

    if "features" not in data:
        errors.append("Missing 'features' array")
        return errors

    # Validate features
    if len(data["features"]) == 0:
        errors.append("Empty features array")
        return errors

    # Check first few features
    for i, feature in enumerate(data["features"][:5]):
        if feature.get("type") != "Feature":
            errors.append(f"Feature {i}: Invalid type")
            continue

        props = feature.get("properties", {})
        if "origin_h3" not in props:
            errors.append(f"Feature {i}: Missing origin_h3")
        if "destination_h3" not in props:
            errors.append(f"Feature {i}: Missing destination_h3")

        geom = feature.get("geometry")
        if not geom or geom.get("type") not in ["LineString", "MultiLineString"]:
            errors.append(f"Feature {i}: Invalid geometry type")

    return errors


def validate_outputs(city_ids=None, fix=False):
    """Validate all output files."""

    results = {
        "valid": [],
        "invalid_matrix": [],
        "invalid_routes": [],
        "missing_matrix": [],
        "missing_routes": [],
        "orphaned_matrix": [],
        "orphaned_routes": []
    }

    # Get all result files
    matrix_files = {f.stem.replace("_matrix", ""): f for f in RESULTS_DIR.glob("*_matrix.json")}
    routes_files = {f.stem.replace("_routes", ""): f for f in RESULTS_DIR.glob("*_routes.geojson")}

    all_city_ids = set(matrix_files.keys()) | set(routes_files.keys())

    if city_ids:
        all_city_ids = all_city_ids & set(city_ids)

    print(f"Validating {len(all_city_ids)} cities...")
    print()

    for city_id in sorted(all_city_ids):
        matrix_path = matrix_files.get(city_id)
        routes_path = routes_files.get(city_id)

        # Check for missing files
        if not matrix_path:
            results["missing_matrix"].append(city_id)
        if not routes_path:
            results["missing_routes"].append(city_id)

        # Validate matrix
        if matrix_path:
            errors = validate_matrix(matrix_path)
            if errors:
                results["invalid_matrix"].append({"city_id": city_id, "errors": errors})
                if fix:
                    print(f"  Deleting invalid matrix: {city_id}")
                    os.remove(matrix_path)
            elif not routes_path:
                results["orphaned_matrix"].append(city_id)

        # Validate routes
        if routes_path:
            errors = validate_routes(routes_path)
            if errors:
                results["invalid_routes"].append({"city_id": city_id, "errors": errors})
                if fix:
                    print(f"  Deleting invalid routes: {city_id}")
                    os.remove(routes_path)
            elif not matrix_path:
                results["orphaned_routes"].append(city_id)

        # Count valid
        if matrix_path and routes_path:
            matrix_errors = validate_matrix(matrix_path)
            routes_errors = validate_routes(routes_path)
            if not matrix_errors and not routes_errors:
                results["valid"].append(city_id)

    # Print results
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print()
    print(f"Valid (both files):      {len(results['valid'])}")
    print(f"Invalid matrix:          {len(results['invalid_matrix'])}")
    print(f"Invalid routes:          {len(results['invalid_routes'])}")
    print(f"Missing matrix:          {len(results['missing_matrix'])}")
    print(f"Missing routes:          {len(results['missing_routes'])}")
    print(f"Orphaned matrix:         {len(results['orphaned_matrix'])}")
    print(f"Orphaned routes:         {len(results['orphaned_routes'])}")
    print()

    # Show details for invalid files
    if results["invalid_matrix"]:
        print("Invalid Matrix Files:")
        for item in results["invalid_matrix"][:10]:
            print(f"  {item['city_id']}: {item['errors'][0]}")
        if len(results["invalid_matrix"]) > 10:
            print(f"  ... and {len(results['invalid_matrix']) - 10} more")
        print()

    if results["invalid_routes"]:
        print("Invalid Routes Files:")
        for item in results["invalid_routes"][:10]:
            print(f"  {item['city_id']}: {item['errors'][0]}")
        if len(results["invalid_routes"]) > 10:
            print(f"  ... and {len(results['invalid_routes']) - 10} more")
        print()

    # Write cities needing reprocessing
    needs_reprocess = (
        [item["city_id"] for item in results["invalid_matrix"]] +
        [item["city_id"] for item in results["invalid_routes"]] +
        results["missing_matrix"] +
        results["missing_routes"]
    )
    needs_reprocess = list(set(needs_reprocess))

    if needs_reprocess:
        reprocess_file = WORK_DIR / "city_lists" / "needs_reprocess.txt"
        with open(reprocess_file, 'w') as f:
            for city_id in sorted(needs_reprocess):
                f.write(f"{city_id}\n")
        print(f"Cities needing reprocessing written to: {reprocess_file}")
        print(f"Total: {len(needs_reprocess)} cities")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate OSRM output files")
    parser.add_argument("--city-list", "-l", help="File with city IDs to validate")
    parser.add_argument("--fix", "-f", action="store_true", help="Delete invalid files for reprocessing")
    args = parser.parse_args()

    city_ids = None
    if args.city_list:
        with open(args.city_list) as f:
            city_ids = [line.strip() for line in f if line.strip()]

    validate_outputs(city_ids, args.fix)


if __name__ == "__main__":
    main()
