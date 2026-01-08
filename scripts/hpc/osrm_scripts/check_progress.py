#!/usr/bin/env python3
"""
Progress monitoring dashboard for HPC OSRM processing.

Displays completion status by country with progress bars and statistics.

Usage:
    python check_progress.py                    # Full dashboard
    python check_progress.py --summary          # Summary only
    python check_progress.py --country india    # Specific country
    python check_progress.py --failed           # Show failed cities
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
WORK_DIR = Path("/scratch/kh3657/osrm")
CITY_LISTS_DIR = WORK_DIR / "city_lists"
RESULTS_DIR = WORK_DIR / "results"
LOGS_DIR = WORK_DIR / "logs"


def load_country_summary():
    """Load country summary with city counts."""
    summary_path = CITY_LISTS_DIR / "country_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return []


def get_completion_status():
    """Get completion status for all cities."""
    status = defaultdict(lambda: {
        "total": 0,
        "matrix_done": 0,
        "routes_done": 0,
        "both_done": 0,
        "failed": [],
        "in_progress": []
    })

    # Check each country's city list
    for city_list in CITY_LISTS_DIR.glob("*_cities.txt"):
        country = city_list.stem.replace("_cities", "")
        if country == "pilot":
            continue

        with open(city_list) as f:
            city_ids = [line.strip() for line in f if line.strip()]

        status[country]["total"] = len(city_ids)

        for city_id in city_ids:
            matrix_exists = (RESULTS_DIR / f"{city_id}_matrix.json").exists()
            routes_exists = (RESULTS_DIR / f"{city_id}_routes.geojson").exists()

            if matrix_exists:
                status[country]["matrix_done"] += 1
            if routes_exists:
                status[country]["routes_done"] += 1
            if matrix_exists and routes_exists:
                status[country]["both_done"] += 1

            # Check for failures in logs
            log_file = LOGS_DIR / f"city_{city_id}.log"
            if log_file.exists():
                with open(log_file) as f:
                    content = f.read()
                    if "ERROR" in content and not matrix_exists:
                        status[country]["failed"].append(city_id)
                    elif "SUCCESS" not in content and not matrix_exists:
                        status[country]["in_progress"].append(city_id)

    return status


def print_progress_bar(done, total, width=30):
    """Print a progress bar."""
    if total == 0:
        return "[" + " " * width + "] 0%"

    pct = done / total
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {pct*100:.1f}%"


def print_dashboard(status, summary_only=False, show_country=None, show_failed=False):
    """Print the progress dashboard."""

    print("=" * 80)
    print("OSRM CITY PROCESSING - PROGRESS DASHBOARD")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Calculate totals
    total_cities = sum(s["total"] for s in status.values())
    total_done = sum(s["both_done"] for s in status.values())
    total_matrix = sum(s["matrix_done"] for s in status.values())
    total_routes = sum(s["routes_done"] for s in status.values())
    total_failed = sum(len(s["failed"]) for s in status.values())
    total_in_progress = sum(len(s["in_progress"]) for s in status.values())

    # Overall summary
    print("OVERALL PROGRESS")
    print("-" * 40)
    print(f"Total cities:     {total_cities:,}")
    print(f"Completed:        {total_done:,} {print_progress_bar(total_done, total_cities)}")
    print(f"  - Matrix only:  {total_matrix - total_done:,}")
    print(f"  - Routes only:  {total_routes - total_done:,}")
    print(f"In progress:      {total_in_progress:,}")
    print(f"Failed:           {total_failed:,}")
    print(f"Remaining:        {total_cities - total_done - total_in_progress:,}")
    print()

    if summary_only:
        return

    # Sort countries by completion percentage (completed first, then by total)
    sorted_countries = sorted(
        status.items(),
        key=lambda x: (
            -x[1]["both_done"] / max(x[1]["total"], 1),  # Completion %
            -x[1]["total"]  # Then by size
        )
    )

    # Filter if specific country requested
    if show_country:
        sorted_countries = [(k, v) for k, v in sorted_countries if k == show_country]
        if not sorted_countries:
            print(f"Country not found: {show_country}")
            return

    # Country breakdown
    print("PROGRESS BY COUNTRY")
    print("-" * 80)
    print(f"{'Country':<25} {'Total':>7} {'Done':>7} {'Progress':<35} {'Status'}")
    print("-" * 80)

    for country, data in sorted_countries:
        if data["total"] == 0:
            continue

        pct = data["both_done"] / data["total"] * 100
        progress = print_progress_bar(data["both_done"], data["total"], width=25)

        if pct == 100:
            status_str = "COMPLETE"
        elif len(data["failed"]) > 0:
            status_str = f"{len(data['failed'])} FAILED"
        elif len(data["in_progress"]) > 0:
            status_str = f"{len(data['in_progress'])} running"
        else:
            status_str = "pending"

        print(f"{country:<25} {data['total']:>7} {data['both_done']:>7} {progress} {status_str}")

    print()

    # Show failed cities if requested
    if show_failed:
        print("FAILED CITIES")
        print("-" * 40)
        any_failed = False
        for country, data in sorted_countries:
            if data["failed"]:
                any_failed = True
                print(f"\n{country}:")
                for city_id in data["failed"][:10]:  # Limit to 10
                    print(f"  - {city_id}")
                if len(data["failed"]) > 10:
                    print(f"  ... and {len(data['failed']) - 10} more")

        if not any_failed:
            print("No failed cities found.")
        print()

    # Check disk usage
    if RESULTS_DIR.exists():
        total_size = sum(f.stat().st_size for f in RESULTS_DIR.glob("*") if f.is_file())
        print("STORAGE")
        print("-" * 40)
        print(f"Results directory: {total_size / (1024**3):.2f} GB")
        print()


def main():
    parser = argparse.ArgumentParser(description="OSRM processing progress dashboard")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary only")
    parser.add_argument("--country", "-c", help="Show specific country")
    parser.add_argument("--failed", "-f", action="store_true", help="Show failed cities")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring (every 60s)")
    args = parser.parse_args()

    if args.watch:
        import time
        while True:
            os.system('clear')
            status = get_completion_status()
            print_dashboard(status, args.summary, args.country, args.failed)
            print("\nRefreshing in 60 seconds... (Ctrl+C to exit)")
            time.sleep(60)
    else:
        status = get_completion_status()
        print_dashboard(status, args.summary, args.country, args.failed)


if __name__ == "__main__":
    main()
