#!/usr/bin/env python3
"""
03b_monitor_batch_tasks.py - Monitor GEE batch export task status.

This script monitors the progress of batch export tasks submitted by 03a_submit_batch_exports.py.
It reads the task manifest and checks the status of each task on GEE.

Usage:
    # Check status of all tasks
    python scripts/03b_monitor_batch_tasks.py --resolution 7

    # Watch mode - continuously monitor until all tasks complete
    python scripts/03b_monitor_batch_tasks.py --resolution 7 --watch

    # Check specific manifest file
    python scripts/03b_monitor_batch_tasks.py --manifest path/to/manifest.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import ee

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import get_resolution_dir, get_latest_file

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
GEE_PROJECT = 'ee-knhuang'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Monitor GEE batch export task status'
    )
    parser.add_argument('--resolution', '-r', type=int, default=6,
                        help='H3 resolution level (default: 6)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to specific manifest file')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Continuously monitor until all tasks complete')
    parser.add_argument('--interval', type=int, default=60,
                        help='Seconds between status checks in watch mode (default: 60)')
    return parser.parse_args()


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def load_manifest(manifest_path: Path) -> Dict:
    """Load the task manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        return json.load(f)


def find_latest_manifest(resolution: int) -> Path:
    """Find the most recent manifest file for the given resolution."""
    output_dir = get_resolution_dir(PROCESSED_DIR, resolution)
    pattern = f"batch_export_manifest_r{resolution}_*.json"

    try:
        return get_latest_file(output_dir, pattern)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No manifest files found for resolution {resolution}.\n"
            f"Run 03a_submit_batch_exports.py first."
        )


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a single GEE task."""
    try:
        # Get task status from GEE
        tasks = ee.data.getTaskList()

        for task in tasks:
            if task.get('id') == task_id:
                return {
                    'state': task.get('state', 'UNKNOWN'),
                    'description': task.get('description', ''),
                    'creation_time': task.get('creation_timestamp_ms', 0),
                    'start_time': task.get('start_timestamp_ms', 0),
                    'update_time': task.get('update_timestamp_ms', 0),
                    'error_message': task.get('error_message', None)
                }

        return {'state': 'NOT_FOUND', 'error_message': 'Task not found in GEE task list'}
    except Exception as e:
        return {'state': 'ERROR', 'error_message': str(e)}


def get_all_tasks_status() -> List[Dict]:
    """Get status of all recent GEE tasks."""
    try:
        return ee.data.getTaskList()
    except Exception as e:
        print(f"Error getting task list: {e}")
        return []


def format_duration(ms: int) -> str:
    """Format milliseconds as human-readable duration."""
    if ms <= 0:
        return "N/A"
    seconds = ms // 1000
    minutes = seconds // 60
    hours = minutes // 60
    if hours > 0:
        return f"{hours}h {minutes % 60}m"
    elif minutes > 0:
        return f"{minutes}m {seconds % 60}s"
    else:
        return f"{seconds}s"


def print_status_summary(manifest: Dict, task_statuses: Dict[str, Dict]) -> Dict[str, int]:
    """Print a summary of task statuses."""
    print("\n" + "="*80)
    print("GEE BATCH EXPORT STATUS")
    print("="*80)

    print(f"\nManifest: {manifest.get('submitted_at', 'Unknown')}")
    print(f"Resolution: {manifest.get('resolution')}")
    print(f"Total cities: {manifest.get('total_cities')}")
    print(f"Total H3 cells: {manifest.get('total_h3_cells')}")
    print(f"Tasks submitted: {manifest.get('n_tasks')}")

    # Count statuses
    status_counts = {
        'COMPLETED': 0,
        'RUNNING': 0,
        'PENDING': 0,
        'READY': 0,
        'FAILED': 0,
        'CANCELLED': 0,
        'UNKNOWN': 0
    }

    for task in manifest.get('tasks', []):
        task_id = task.get('task_id')
        if task_id and task_id in task_statuses:
            state = task_statuses[task_id].get('state', 'UNKNOWN')
        else:
            state = task.get('status', 'UNKNOWN')

        if state in status_counts:
            status_counts[state] += 1
        else:
            status_counts['UNKNOWN'] += 1

    print(f"\n{'Status':<15} {'Count':>8}")
    print("-" * 25)
    for status, count in status_counts.items():
        if count > 0:
            print(f"{status:<15} {count:>8}")

    total = sum(status_counts.values())
    completed = status_counts['COMPLETED']
    print("-" * 25)
    print(f"{'TOTAL':<15} {total:>8}")
    print(f"\nProgress: {completed}/{total} ({100*completed/total:.1f}%)")

    return status_counts


def print_detailed_status(manifest: Dict, task_statuses: Dict[str, Dict]):
    """Print detailed status of each task."""
    print("\n" + "-"*80)
    print("DETAILED TASK STATUS")
    print("-"*80)

    print(f"\n{'Batch':<8} {'Status':<12} {'H3 Cells':<10} {'Duration':<12} {'Description'}")
    print("-" * 80)

    for task in manifest.get('tasks', []):
        task_id = task.get('task_id')
        batch_num = task.get('batch_num', '?')
        n_cells = task.get('n_h3_cells', 0)
        name = task.get('name', 'Unknown')

        if task_id and task_id in task_statuses:
            status_info = task_statuses[task_id]
            state = status_info.get('state', 'UNKNOWN')

            # Calculate duration
            start_time = status_info.get('start_time', 0)
            update_time = status_info.get('update_time', 0)
            if start_time and update_time:
                duration = format_duration(update_time - start_time)
            else:
                duration = "N/A"
        else:
            state = task.get('status', 'UNKNOWN')
            duration = "N/A"

        # Truncate name if too long
        if len(name) > 35:
            name = name[:32] + "..."

        print(f"{batch_num:<8} {state:<12} {n_cells:<10} {duration:<12} {name}")


def print_errors(manifest: Dict, task_statuses: Dict[str, Dict]):
    """Print any error messages from failed tasks."""
    errors = []

    for task in manifest.get('tasks', []):
        task_id = task.get('task_id')
        if task_id and task_id in task_statuses:
            status_info = task_statuses[task_id]
            if status_info.get('state') == 'FAILED':
                errors.append({
                    'batch': task.get('batch_num'),
                    'name': task.get('name'),
                    'error': status_info.get('error_message', 'Unknown error')
                })

    if errors:
        print("\n" + "!"*80)
        print("ERRORS")
        print("!"*80)
        for err in errors:
            print(f"\nBatch {err['batch']}: {err['name']}")
            print(f"  Error: {err['error']}")


def update_manifest_with_status(manifest: Dict, task_statuses: Dict[str, Dict], manifest_path: Path):
    """Update manifest file with current task statuses."""
    for task in manifest.get('tasks', []):
        task_id = task.get('task_id')
        if task_id and task_id in task_statuses:
            status_info = task_statuses[task_id]
            task['current_status'] = status_info.get('state')
            task['last_checked'] = datetime.now().isoformat()
            if status_info.get('error_message'):
                task['error_message'] = status_info['error_message']

    manifest['last_status_check'] = datetime.now().isoformat()

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


def main():
    args = parse_args()

    # Initialize GEE
    initialize_gee()

    # Find manifest file
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = find_latest_manifest(args.resolution)

    print(f"Using manifest: {manifest_path}")

    # Load manifest
    manifest = load_manifest(manifest_path)

    # Get all GEE tasks (more efficient than querying one by one)
    print("Fetching task status from GEE...")
    all_tasks = get_all_tasks_status()

    # Build lookup by task ID
    task_statuses = {}
    for task in all_tasks:
        task_id = task.get('id')
        if task_id:
            task_statuses[task_id] = {
                'state': task.get('state', 'UNKNOWN'),
                'description': task.get('description', ''),
                'creation_time': task.get('creation_timestamp_ms', 0),
                'start_time': task.get('start_timestamp_ms', 0),
                'update_time': task.get('update_timestamp_ms', 0),
                'error_message': task.get('error_message', None)
            }

    # Print status
    status_counts = print_status_summary(manifest, task_statuses)
    print_detailed_status(manifest, task_statuses)
    print_errors(manifest, task_statuses)

    # Update manifest with current status
    update_manifest_with_status(manifest, task_statuses, manifest_path)
    print(f"\nManifest updated: {manifest_path}")

    # Watch mode
    if args.watch:
        completed = status_counts['COMPLETED']
        total = sum(status_counts.values())

        while completed < total and status_counts['FAILED'] == 0:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Waiting {args.interval}s before next check...")
            time.sleep(args.interval)

            # Refresh status
            all_tasks = get_all_tasks_status()
            task_statuses = {t['id']: {
                'state': t.get('state', 'UNKNOWN'),
                'start_time': t.get('start_timestamp_ms', 0),
                'update_time': t.get('update_timestamp_ms', 0),
                'error_message': t.get('error_message', None)
            } for t in all_tasks if t.get('id')}

            status_counts = print_status_summary(manifest, task_statuses)
            completed = status_counts['COMPLETED']

            # Update manifest
            update_manifest_with_status(manifest, task_statuses, manifest_path)

        if status_counts['FAILED'] > 0:
            print("\n" + "!"*80)
            print("SOME TASKS FAILED - Check errors above")
            print("!"*80)
            sys.exit(1)
        else:
            print("\n" + "="*80)
            print("ALL TASKS COMPLETED!")
            print("="*80)
            print(f"\nNext step: python scripts/03c_download_batch_results.py --resolution {args.resolution}")


if __name__ == "__main__":
    main()
