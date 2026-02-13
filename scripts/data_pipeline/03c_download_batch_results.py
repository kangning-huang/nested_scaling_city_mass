#!/usr/bin/env python3
"""
03c_download_batch_results.py - Download and combine batch export results from Google Drive.

This script downloads the CSV files exported by GEE batch tasks and combines them
into a single output file compatible with the rest of the pipeline.

Prerequisites:
    - Google Drive API credentials configured
    - Or: manually download files from Google Drive to a local folder

Usage:
    # Download from Google Drive and combine
    python scripts/03c_download_batch_results.py --resolution 7

    # Combine already-downloaded files from a local folder
    python scripts/03c_download_batch_results.py --resolution 7 --local-folder /path/to/downloads

    # Specify manifest file
    python scripts/03c_download_batch_results.py --manifest path/to/manifest.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import get_resolution_dir, get_latest_file

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DRIVE_FOLDER = 'GEE_Batch_Exports'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download and combine batch export results'
    )
    parser.add_argument('--resolution', '-r', type=int, default=6,
                        help='H3 resolution level (default: 6)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to specific manifest file')
    parser.add_argument('--local-folder', type=str, default=None,
                        help='Local folder containing downloaded CSV files (skip Drive download)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, only combine existing files')
    return parser.parse_args()


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


def load_manifest(manifest_path: Path) -> dict:
    """Load the task manifest file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def find_drive_folder() -> Optional[Path]:
    """
    Try to find the Google Drive folder on the local filesystem.
    This works when Google Drive is synced locally (Google Drive for Desktop).
    """
    # Common Google Drive mount points
    possible_paths = [
        Path.home() / "Google Drive" / "My Drive" / DRIVE_FOLDER,
        Path.home() / "Library/CloudStorage/GoogleDrive-kh3657@nyu.edu/My Drive" / DRIVE_FOLDER,
        Path("/Volumes/GoogleDrive/My Drive") / DRIVE_FOLDER,
        Path.home() / "GoogleDrive" / DRIVE_FOLDER,
    ]

    # Also search for any GoogleDrive pattern in CloudStorage
    cloud_storage = Path.home() / "Library/CloudStorage"
    if cloud_storage.exists():
        for folder in cloud_storage.iterdir():
            if folder.name.startswith("GoogleDrive"):
                potential = folder / "My Drive" / DRIVE_FOLDER
                if potential.exists():
                    return potential

    for path in possible_paths:
        if path.exists():
            return path

    return None


def download_from_drive_api(task_names: List[str], output_folder: Path) -> List[Path]:
    """
    Download files from Google Drive using the Drive API.
    Requires google-api-python-client and credentials.
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        import io
        import pickle
    except ImportError:
        print("Google Drive API not available. Please either:")
        print("  1. Install: pip install google-api-python-client google-auth-oauthlib")
        print("  2. Use --local-folder to point to manually downloaded files")
        print("  3. Use Google Drive for Desktop to sync files locally")
        return []

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None

    # Token file for caching credentials
    token_file = Path.home() / '.gee_drive_token.pickle'
    creds_file = Path.home() / '.gee_credentials.json'

    if token_file.exists():
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_file.exists():
                print(f"Credentials file not found: {creds_file}")
                print("Please set up Google Drive API credentials.")
                return []
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    # Find files in the export folder
    downloaded = []
    for task_name in tqdm(task_names, desc="Downloading from Drive"):
        filename = f"{task_name}.csv"

        # Search for file
        results = service.files().list(
            q=f"name='{filename}'",
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        files = results.get('files', [])
        if not files:
            print(f"  Warning: File not found: {filename}")
            continue

        file_id = files[0]['id']
        output_path = output_folder / filename

        # Download
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        with open(output_path, 'wb') as f:
            f.write(fh.getvalue())

        downloaded.append(output_path)

    return downloaded


def find_local_files(folder: Path, task_names: List[str]) -> List[Path]:
    """Find CSV files in a local folder matching task names."""
    files = []
    for task_name in task_names:
        csv_path = folder / f"{task_name}.csv"
        if csv_path.exists():
            files.append(csv_path)
        else:
            # Try without .csv extension (GEE sometimes adds it automatically)
            alt_path = folder / task_name
            if alt_path.exists():
                files.append(alt_path)

    return files


def group_files_by_batch(csv_files: List[Path], manifest: dict) -> dict:
    """
    Group CSV files by batch number for merging.

    The new export structure creates 4 files per batch (one per dataset).
    These need to be merged on h3index before concatenating batches.
    """
    # Build mapping from task name to (batch_num, dataset)
    task_info = {}
    for task in manifest.get('tasks', []):
        name = task.get('name')
        batch_num = task.get('batch_num')
        dataset = task.get('dataset')
        if name and batch_num and dataset:
            task_info[name] = (batch_num, dataset)

    # Group files by batch
    batches = {}
    for csv_file in csv_files:
        # Extract task name from filename
        task_name = csv_file.stem  # Remove .csv extension

        if task_name in task_info:
            batch_num, dataset = task_info[task_name]
            if batch_num not in batches:
                batches[batch_num] = {}
            batches[batch_num][dataset] = csv_file
        else:
            # Old format - single file per batch, or unrecognized
            # Try to extract batch number from filename
            match = re.search(r'batch(\d+)', task_name)
            if match:
                batch_num = int(match.group(1))
                if batch_num not in batches:
                    batches[batch_num] = {}
                batches[batch_num]['all'] = csv_file

    return batches


def merge_batch_datasets(batch_files: dict) -> pd.DataFrame:
    """
    Merge multiple dataset CSVs for a single batch on h3index.

    Args:
        batch_files: Dict mapping dataset name to CSV file path

    Returns:
        Merged DataFrame with all columns from all datasets
    """
    dfs = {}
    for dataset, csv_file in batch_files.items():
        try:
            df = pd.read_csv(csv_file)
            # Clean up columns
            columns_to_drop = ['system:index', '.geo', 'geo']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Ensure h3index column exists
            if 'h3index' not in df.columns:
                possible_names = ['hex_id', 'h3_index', 'index']
                for name in possible_names:
                    if name in df.columns:
                        df = df.rename(columns={name: 'h3index'})
                        break

            dfs[dataset] = df
        except Exception as e:
            print(f"  Warning: Error loading {csv_file}: {e}")

    if not dfs:
        return None

    # If only one file (old format), return it directly
    if len(dfs) == 1:
        return list(dfs.values())[0]

    # Merge all datasets on h3index
    merged = None
    for dataset, df in dfs.items():
        if merged is None:
            merged = df
        else:
            # Get non-h3index columns from this df
            new_cols = [c for c in df.columns if c != 'h3index' and c not in merged.columns]
            if new_cols:
                merged = merged.merge(df[['h3index'] + new_cols], on='h3index', how='outer')

    return merged


def combine_csv_files(csv_files: List[Path], output_path: Path, manifest: dict):
    """Combine multiple CSV files into a single output file."""
    print(f"\nCombining {len(csv_files)} CSV files...")

    # Check if we have the new multi-dataset format
    datasets_in_manifest = manifest.get('datasets', [])
    n_datasets = len(datasets_in_manifest)

    if n_datasets > 1:
        print(f"Detected multi-dataset format ({n_datasets} datasets per batch)")
        print("Grouping files by batch and merging datasets...")

        # Group files by batch
        batches = group_files_by_batch(csv_files, manifest)
        print(f"Found {len(batches)} batches")

        # Merge datasets within each batch, then combine
        dfs = []
        for batch_num in tqdm(sorted(batches.keys()), desc="Merging batches"):
            batch_files = batches[batch_num]
            merged = merge_batch_datasets(batch_files)
            if merged is not None:
                dfs.append(merged)

        if not dfs:
            print("Error: No batches could be merged")
            return None

        combined = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataframe: {len(combined)} rows from {len(dfs)} batches")
    else:
        # Old single-dataset format
        dfs = []
        for csv_file in tqdm(csv_files, desc="Loading files"):
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Error loading {csv_file}: {e}")

        if not dfs:
            print("Error: No CSV files could be loaded")
            return None

        # Combine all dataframes
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataframe: {len(combined)} rows")

    # Clean up columns
    # GEE exports include a 'system:index' column and '.geo' column we don't need
    columns_to_drop = ['system:index', '.geo', 'geo']
    for col in columns_to_drop:
        if col in combined.columns:
            combined = combined.drop(columns=[col])

    # Rename 'h3index' if it's named differently
    if 'h3index' not in combined.columns:
        possible_names = ['hex_id', 'h3_index', 'index']
        for name in possible_names:
            if name in combined.columns:
                combined = combined.rename(columns={name: 'h3index'})
                break

    # Add metadata columns if available from manifest
    # These would need to be joined from the H3 grids file
    print(f"Columns: {list(combined.columns)}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nOutput saved to: {output_path}")
    print(f"Total rows: {len(combined)}")
    print(f"Unique H3 cells: {combined['h3index'].nunique() if 'h3index' in combined.columns else 'N/A'}")

    return combined


def enrich_with_metadata(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Add city/country metadata from H3 grids file."""
    grids_file = get_resolution_dir(PROCESSED_DIR, resolution) / f"all_cities_h3_grids_resolution{resolution}.gpkg"

    if not grids_file.exists():
        print("Warning: H3 grids file not found, skipping metadata enrichment")
        return df

    print("Enriching with city/country metadata...")
    import geopandas as gpd
    grids = gpd.read_file(grids_file)

    # Get metadata columns
    meta_cols = ['h3index', 'ID_HDC_G0', 'UC_NM_MN', 'CTR_MN_ISO', 'CTR_MN_NM', 'GRGN_L1', 'GRGN_L2']
    meta_cols = [c for c in meta_cols if c in grids.columns]

    if 'h3index' not in grids.columns:
        if grids.index.name in ['h3index', 'hex_id']:
            grids = grids.reset_index()
            if 'hex_id' in grids.columns:
                grids = grids.rename(columns={'hex_id': 'h3index'})

    meta_df = grids[meta_cols].drop_duplicates(subset=['h3index'])

    # Merge
    df = df.merge(meta_df, on='h3index', how='left')

    return df


def main():
    args = parse_args()
    resolution = args.resolution

    print("="*70)
    print("GEE BATCH EXPORT - Download and Combine Results")
    print("="*70)

    # Find manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = find_latest_manifest(resolution)

    print(f"Using manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)

    # Get task names from manifest
    task_names = [t.get('name') for t in manifest.get('tasks', []) if t.get('name')]
    print(f"Tasks in manifest: {len(task_names)}")

    # Set up output directory
    output_dir = get_resolution_dir(PROCESSED_DIR, resolution)
    downloads_dir = output_dir / "batch_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Find or download CSV files
    csv_files = []

    if args.local_folder:
        # Use specified local folder
        local_folder = Path(args.local_folder)
        print(f"\nLooking for files in: {local_folder}")
        csv_files = find_local_files(local_folder, task_names)

    elif args.skip_download:
        # Look in default downloads directory
        print(f"\nLooking for files in: {downloads_dir}")
        csv_files = find_local_files(downloads_dir, task_names)

    else:
        # Try to find Google Drive folder locally first
        drive_folder = find_drive_folder()

        if drive_folder:
            print(f"\nFound Google Drive folder: {drive_folder}")
            csv_files = find_local_files(drive_folder, task_names)
        else:
            # Try Drive API download
            print("\nGoogle Drive folder not found locally. Trying API download...")
            csv_files = download_from_drive_api(task_names, downloads_dir)

    print(f"\nFound {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        print("\nNo CSV files found. Options:")
        print("  1. Wait for GEE tasks to complete (use 03b_monitor_batch_tasks.py)")
        print("  2. Manually download files from Google Drive to a folder")
        print(f"     Then run: python {sys.argv[0]} --local-folder /path/to/downloads")
        return

    if len(csv_files) < len(task_names):
        print(f"Warning: Only {len(csv_files)}/{len(task_names)} files found")
        print("Some tasks may still be running or failed.")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        today = datetime.now().strftime('%Y-%m-%d')
        output_path = output_dir / f"Fig3_Volume_Pavement_Neighborhood_H3_Resolution{resolution}_{today}.csv"

    # Combine files
    df = combine_csv_files(csv_files, output_path, manifest)

    if df is not None:
        # Enrich with metadata
        df = enrich_with_metadata(df, resolution)
        df.to_csv(output_path, index=False)

        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Output: {output_path}")
        print(f"Rows: {len(df)}")
        if 'ID_HDC_G0' in df.columns:
            print(f"Cities: {df['ID_HDC_G0'].nunique()}")
        if 'CTR_MN_ISO' in df.columns:
            print(f"Countries: {df['CTR_MN_ISO'].nunique()}")

        print(f"\nNext step: python scripts/04_merge_building_road_data.py --resolution {resolution}")


if __name__ == "__main__":
    main()
