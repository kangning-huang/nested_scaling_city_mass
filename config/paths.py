"""
Centralized path configuration for Urban Scaling Project.

This module provides environment-aware path definitions that work across:
- Local development (macOS/Linux)
- NYUSH HPC cluster
- NYU Greene HPC cluster
- Google Cloud VMs

Usage:
    from config.paths import DATA_RAW, RESULTS, FIGURES

    # Paths automatically adjust based on COMPUTE_ENV environment variable
    df = pd.read_csv(DATA_PROCESSED / "sndi_neighborhoods.csv")
"""

import os
from pathlib import Path

# Detect compute environment
# Set COMPUTE_ENV in your shell or SLURM script:
#   export COMPUTE_ENV=hpc_nyush
#   export COMPUTE_ENV=hpc_greene
#   export COMPUTE_ENV=gcloud
ENVIRONMENT = os.getenv("COMPUTE_ENV", "local")

# Environment-specific base paths
if ENVIRONMENT == "hpc_nyush":
    # NYU Shanghai HPC
    PROJECT_ROOT = Path("/scratch/kh3657/urban_scaling")
    OSRM_DATA = Path("/scratch/kh3657/osrm")

elif ENVIRONMENT == "hpc_greene":
    # NYU Greene HPC
    PROJECT_ROOT = Path("/scratch/netid/urban_scaling")  # Update netid
    OSRM_DATA = Path("/scratch/netid/osrm")

elif ENVIRONMENT == "gcloud":
    # Google Cloud VM
    PROJECT_ROOT = Path.home() / "urban_scaling"
    OSRM_DATA = Path.home() / "osrm"

else:
    # Local development (default)
    # Assumes this file is at: project_root/config/paths.py
    PROJECT_ROOT = Path(__file__).parent.parent
    OSRM_DATA = None  # OSRM not available locally

# =============================================================================
# Core Data Directories
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

# Raw data subdirectories
RAW_BOUNDARIES = DATA_RAW / "boundaries"
RAW_POIS = DATA_RAW / "pois"
RAW_RASTERS = DATA_RAW / "rasters"
RAW_GRIP = DATA_RAW / "GRIP"

# =============================================================================
# Output Directories
# =============================================================================

RESULTS = PROJECT_ROOT / "results"
RESULTS_TABLES = RESULTS / "tables"
RESULTS_GEOSPATIAL = RESULTS / "geospatial"

FIGURES = PROJECT_ROOT / "figures"
FIGURES_MAIN = FIGURES / "main"
FIGURES_SUPPLEMENTARY = FIGURES / "supplementary"
FIGURES_EXPLORATORY = FIGURES / "exploratory"

# =============================================================================
# Scripts and Documentation
# =============================================================================

SCRIPTS = PROJECT_ROOT / "scripts"
DOCS = PROJECT_ROOT / "docs"
MANUSCRIPTS = PROJECT_ROOT / "manuscripts"

# =============================================================================
# Key Data Files (commonly used)
# =============================================================================

# Urban centers database
GHS_UCDB = DATA_RAW / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg"

# H3 hexagon grids
H3_GRIDS = DATA_RAW / "all_cities_h3_grids.gpkg"

# SNDi raster
SNDI_RASTER = DATA_RAW / "sndi_grid_in_UrbanCores.tif"

# China neighborhood CMI
CHINA_CMI = DATA_RAW / "china_neighborhoods_cmi.csv"

# =============================================================================
# Google Earth Engine Configuration
# =============================================================================

GEE_PROJECT = "ee-knhuang"

# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dirs():
    """Create all output directories if they don't exist."""
    dirs = [
        DATA_INTERIM, DATA_PROCESSED,
        RESULTS, RESULTS_TABLES, RESULTS_GEOSPATIAL,
        FIGURES, FIGURES_MAIN, FIGURES_SUPPLEMENTARY, FIGURES_EXPLORATORY
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def get_city_pois_path(city_id: str) -> Path:
    """Get path to POI geopackage for a city."""
    return RAW_POIS / city_id / f"{city_id}_pois_9cats.gpkg"

def get_city_boundary_path(city_id: str) -> Path:
    """Get path to city boundary geopackage."""
    return RAW_POIS / city_id / f"{city_id}_boundary.gpkg"

def get_processed_file(name: str, date: str = None, ext: str = "csv") -> Path:
    """
    Get path for a processed data file.

    Args:
        name: Base filename (e.g., "sndi_neighborhoods")
        date: Optional date string (YYYY-MM-DD)
        ext: File extension (default: csv)

    Returns:
        Path object for the file
    """
    if date:
        return DATA_PROCESSED / f"{name}_{date}.{ext}"
    return DATA_PROCESSED / f"{name}.{ext}"

# =============================================================================
# Environment Info
# =============================================================================

def print_environment():
    """Print current environment configuration."""
    print(f"Compute Environment: {ENVIRONMENT}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS}")
    print(f"OSRM Data: {OSRM_DATA or 'Not available'}")

if __name__ == "__main__":
    print_environment()
