"""Shared path utilities for resolution-aware file handling."""
from pathlib import Path
from datetime import datetime


def get_resolution_dir(base_dir: Path, resolution: int) -> Path:
    """Return path to resolution-specific subdirectory.

    Args:
        base_dir: Base directory (e.g., data/processed)
        resolution: H3 resolution level (e.g., 6 or 7)

    Returns:
        Path to h3_resolution{N}/ subdirectory
    """
    return base_dir / f"h3_resolution{resolution}"


def get_latest_file(directory: Path, pattern: str) -> Path:
    """Find latest file matching pattern (sorted by date in filename).

    Args:
        directory: Directory to search in
        pattern: Glob pattern to match (e.g., "Fig3_Merged_*.csv")

    Returns:
        Path to the latest matching file

    Raises:
        FileNotFoundError: If no files match the pattern
    """
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def get_output_filename(prefix: str, resolution: int, extension: str = "csv") -> str:
    """Generate dated output filename with resolution.

    Args:
        prefix: Filename prefix (e.g., "Fig3_Mass_Neighborhood")
        resolution: H3 resolution level
        extension: File extension (default: "csv")

    Returns:
        Filename string like "Fig3_Mass_Neighborhood_H3_Resolution7_2025-01-15.csv"
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{prefix}_H3_Resolution{resolution}_{today}.{extension}"


def add_resolution_argument(parser, default: int = 6):
    """Add standard --resolution argument to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
        default: Default resolution value (default: 6)
    """
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=default,
        help=f'H3 resolution level (default: {default})'
    )
