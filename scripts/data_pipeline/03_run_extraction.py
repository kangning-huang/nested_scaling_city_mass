#!/usr/bin/env python3
"""
03_run_extraction.py - Unified runner for volume/pavement extraction.

This script provides a single entry point to run the volume/pavement extraction
using either the original synchronous approach or the faster batch export approach.

Approaches:
    1. SYNCHRONOUS (original): Direct GEE API calls, slower but simpler
       - Best for: Small jobs, debugging, single-country runs
       - Uses: 03_extract_volume_pavement.py

    2. BATCH EXPORT (new): GEE batch export tasks, 5-10x faster
       - Best for: Large jobs (1000+ cities), full global runs
       - Uses: 03a_submit_batch_exports.py + 03b_monitor_batch_tasks.py + 03c_download_batch_results.py

Usage:
    # Run with batch export (recommended for large jobs)
    python scripts/03_run_extraction.py --resolution 7 --method batch

    # Run with original synchronous method
    python scripts/03_run_extraction.py --resolution 7 --method sync

    # Run batch export for specific country
    python scripts/03_run_extraction.py --resolution 7 --method batch --country BGD

    # Full pipeline (batch export + wait + download)
    python scripts/03_run_extraction.py --resolution 7 --method batch --wait
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified runner for volume/pavement extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
    sync    Original synchronous approach using direct GEE API calls.
            Slower but simpler. Best for small jobs or debugging.

    batch   New batch export approach using GEE export tasks.
            5-10x faster. Best for large jobs (1000+ cities).

Examples:
    # Quick test on one country
    python scripts/03_run_extraction.py -r 7 --method sync --country BGD

    # Full global extraction (recommended)
    python scripts/03_run_extraction.py -r 7 --method batch --wait

    # Submit batch jobs and check later
    python scripts/03_run_extraction.py -r 7 --method batch
    # ... wait for completion ...
    python scripts/03b_monitor_batch_tasks.py -r 7
    python scripts/03c_download_batch_results.py -r 7
"""
    )
    parser.add_argument('--resolution', '-r', type=int, default=6,
                        help='H3 resolution level (default: 6)')
    parser.add_argument('--method', choices=['sync', 'batch'], default='batch',
                        help='Extraction method: sync (original) or batch (faster)')
    parser.add_argument('--country', type=str, default=None,
                        help='Country ISO code to filter (e.g., BGD, USA)')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for batch tasks to complete and download results')
    parser.add_argument('--cities-per-task', type=int, default=100,
                        help='Cities per batch export task (batch method only)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode for sync method')
    return parser.parse_args()


def run_sync_extraction(args):
    """Run the original synchronous extraction."""
    print("="*70)
    print("Running SYNCHRONOUS extraction (original method)")
    print("="*70)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "03_extract_volume_pavement.py"),
        "--resolution", str(args.resolution)
    ]

    if args.country:
        cmd.extend(["--debug-country", "--country", args.country])
    elif args.debug:
        cmd.append("--debug")

    print(f"Command: {' '.join(cmd)}")
    print()

    return subprocess.run(cmd)


def run_batch_extraction(args):
    """Run the batch export extraction."""
    print("="*70)
    print("Running BATCH EXPORT extraction (faster method)")
    print("="*70)

    # Step 1: Submit batch export tasks
    print("\n[Step 1/3] Submitting batch export tasks...")
    submit_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "03a_submit_batch_exports.py"),
        "--resolution", str(args.resolution),
        "--cities-per-task", str(args.cities_per_task)
    ]

    if args.country:
        submit_cmd.extend(["--country", args.country])

    print(f"Command: {' '.join(submit_cmd)}")
    result = subprocess.run(submit_cmd)

    if result.returncode != 0:
        print("Error submitting batch tasks")
        return result

    if not args.wait:
        print("\n" + "="*70)
        print("BATCH TASKS SUBMITTED")
        print("="*70)
        print("""
Tasks are now running on GEE servers. To complete the extraction:

1. Monitor task progress:
   python scripts/03b_monitor_batch_tasks.py --resolution {resolution}

2. When all tasks complete, download results:
   python scripts/03c_download_batch_results.py --resolution {resolution}

Or run with --wait flag to do this automatically:
   python scripts/03_run_extraction.py --resolution {resolution} --method batch --wait
""".format(resolution=args.resolution))
        return result

    # Step 2: Monitor and wait for completion
    print("\n[Step 2/3] Monitoring task progress (this may take a while)...")
    monitor_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "03b_monitor_batch_tasks.py"),
        "--resolution", str(args.resolution),
        "--watch"
    ]

    print(f"Command: {' '.join(monitor_cmd)}")
    result = subprocess.run(monitor_cmd)

    if result.returncode != 0:
        print("Error or tasks failed during monitoring")
        return result

    # Step 3: Download and combine results
    print("\n[Step 3/3] Downloading and combining results...")
    download_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "03c_download_batch_results.py"),
        "--resolution", str(args.resolution)
    ]

    print(f"Command: {' '.join(download_cmd)}")
    return subprocess.run(download_cmd)


def main():
    args = parse_args()

    print()
    print("Volume/Pavement Extraction")
    print(f"Resolution: {args.resolution}")
    print(f"Method: {args.method.upper()}")
    print(f"Country: {args.country or 'ALL'}")
    print()

    if args.method == 'sync':
        result = run_sync_extraction(args)
    else:
        result = run_batch_extraction(args)

    sys.exit(result.returncode if result else 0)


if __name__ == "__main__":
    main()
