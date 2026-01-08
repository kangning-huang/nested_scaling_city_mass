#!/usr/bin/env python3
"""
Resubmit failed cities for processing.

Identifies cities that:
1. Have no output files (matrix or routes)
2. Have invalid output files
3. Are listed in needs_reprocess.txt

Usage:
    python retry_failed.py                    # Analyze only
    python retry_failed.py --submit           # Submit retry jobs
    python retry_failed.py --submit --limit 50  # Submit max 50 cities
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

# Configuration
WORK_DIR = Path("/scratch/kh3657/osrm")
CITY_LISTS_DIR = WORK_DIR / "city_lists"
RESULTS_DIR = WORK_DIR / "results"
SLURM_DIR = WORK_DIR / "slurm"


def get_failed_cities():
    """Get list of cities that need reprocessing."""
    failed = defaultdict(list)

    # Load country summary for OSM file mapping
    summary_path = CITY_LISTS_DIR / "country_summary.json"
    country_data = {}
    if summary_path.exists():
        with open(summary_path) as f:
            for item in json.load(f):
                country_name = item["filename"].replace("_cities.txt", "")
                country_data[country_name] = {
                    "osm_path": item.get("osm_path", "UNKNOWN"),
                    "city_ids": {c["id"] for c in item.get("cities", [])}
                }

    # Check each country's cities
    for city_list in CITY_LISTS_DIR.glob("*_cities.txt"):
        country = city_list.stem.replace("_cities", "")
        if country == "pilot" or country == "needs_reprocess":
            continue

        with open(city_list) as f:
            city_ids = [line.strip() for line in f if line.strip()]

        for city_id in city_ids:
            matrix_exists = (RESULTS_DIR / f"{city_id}_matrix.json").exists()
            routes_exists = (RESULTS_DIR / f"{city_id}_routes.geojson").exists()

            if not matrix_exists or not routes_exists:
                failed[country].append(city_id)

    return failed, country_data


def submit_retry_jobs(failed, country_data, limit=None, dry_run=False):
    """Submit SLURM jobs for failed cities."""

    total_submitted = 0

    for country, city_ids in sorted(failed.items(), key=lambda x: -len(x[1])):
        if limit and total_submitted >= limit:
            break

        if not city_ids:
            continue

        # Get OSM file for this country
        osm_path = country_data.get(country, {}).get("osm_path", "UNKNOWN")
        if osm_path == "UNKNOWN":
            print(f"  SKIP {country}: No OSM file mapping")
            continue

        osm_file = osm_path.replace("/", "_") + "-latest.osm.pbf"

        # Limit cities if needed
        if limit:
            remaining = limit - total_submitted
            city_ids = city_ids[:remaining]

        # Write temporary city list
        retry_list = CITY_LISTS_DIR / f"retry_{country}.txt"
        with open(retry_list, 'w') as f:
            for city_id in city_ids:
                f.write(f"{city_id}\n")

        n_cities = len(city_ids)
        print(f"  {country}: {n_cities} cities to retry")

        if not dry_run:
            # Submit job array
            cmd = [
                "sbatch",
                f"--job-name=retry_{country}",
                f"--array=1-{n_cities}%20",
                "--export", f"CITY_LIST={retry_list},COUNTRY_OSM={osm_file}",
                str(SLURM_DIR / "retry_city.slurm")
            ]
            subprocess.run(cmd)

        total_submitted += n_cities

    return total_submitted


def create_retry_slurm():
    """Create the retry SLURM template if it doesn't exist."""
    retry_slurm = SLURM_DIR / "retry_city.slurm"

    if not retry_slurm.exists():
        content = """#!/bin/bash
#SBATCH --job-name=retry
#SBATCH --output=/scratch/kh3657/osrm/logs/retry_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/retry_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

WORK_DIR=/scratch/kh3657/osrm

# Get city ID from the retry list
CITY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${CITY_LIST})

if [ -z "$CITY_ID" ]; then
    echo "ERROR: No city ID found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Retrying city $CITY_ID with OSM file $COUNTRY_OSM"

# Delete any partial outputs
rm -f $WORK_DIR/results/${CITY_ID}_matrix.json
rm -f $WORK_DIR/results/${CITY_ID}_routes.geojson

# Run the processing pipeline
cd $WORK_DIR
bash slurm/process_single_city.sh $CITY_ID $COUNTRY_OSM

echo "Completed retry for city $CITY_ID"
"""
        with open(retry_slurm, 'w') as f:
            f.write(content)
        print(f"Created {retry_slurm}")


def main():
    parser = argparse.ArgumentParser(description="Retry failed cities")
    parser.add_argument("--submit", "-s", action="store_true", help="Submit retry jobs")
    parser.add_argument("--limit", "-l", type=int, help="Maximum cities to retry")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be submitted")
    args = parser.parse_args()

    print("=" * 60)
    print("OSRM RETRY ANALYSIS")
    print("=" * 60)
    print()

    # Get failed cities
    failed, country_data = get_failed_cities()

    total_failed = sum(len(cities) for cities in failed.values())

    if total_failed == 0:
        print("No failed cities found!")
        return

    print(f"Total failed/incomplete cities: {total_failed}")
    print()
    print("Failed by country:")
    for country, city_ids in sorted(failed.items(), key=lambda x: -len(x[1])):
        if city_ids:
            print(f"  {country}: {len(city_ids)}")
    print()

    if args.submit or args.dry_run:
        print("Submitting retry jobs...")
        create_retry_slurm()
        submitted = submit_retry_jobs(failed, country_data, args.limit, args.dry_run)
        print()
        print(f"{'Would submit' if args.dry_run else 'Submitted'}: {submitted} cities")

        if not args.dry_run:
            print()
            print("Monitor with:")
            print("  squeue -u $USER")
            print("  python3 scripts/check_progress.py")
    else:
        print("Run with --submit to resubmit failed cities")
        print("Run with --dry-run to see what would be submitted")


if __name__ == "__main__":
    main()
