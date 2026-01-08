#!/usr/bin/env python3
"""
Common utilities shared across OSRM processing scripts.
"""

import os
import logging
import subprocess
from pathlib import Path

import geopandas as gpd

# Default directories
DEFAULT_OSM_DIR = Path(os.path.expanduser('~/osrm-data'))
DEFAULT_CLIPPED_DIR = Path(os.path.expanduser('~/clipped'))
DEFAULT_OSRM_DIR = Path(os.path.expanduser('~/osrm'))
DEFAULT_CITIES_DIR = Path(os.path.expanduser('~/cities'))
DEFAULT_RESULTS_DIR = Path(os.path.expanduser('~/results'))
DEFAULT_ADMIN_BOUNDS = Path(os.path.expanduser('~/admin_boundaries/ne_10m_admin_1_states_provinces.shp'))

# City size threshold for keeping compressed OSRM (km²)
LARGE_CITY_THRESHOLD_KM2 = 100

# Countries that require subnational OSM files
# Map various country name formats to canonical names used in Natural Earth data
SUBNATIONAL_COUNTRIES = {'United States of America', 'United States', 'China', 'India'}

# Map city file country names to Natural Earth admin boundary country names
COUNTRY_NAME_MAP = {
    'United States': 'United States of America',
    'USA': 'United States of America',
}

# Geofabrik naming conventions for subnational regions
GEOFABRIK_NAME_MAP = {
    # US States - map Natural Earth names to Geofabrik names
    'United States of America': {
        'Alabama': 'alabama',
        'Alaska': 'alaska',
        'Arizona': 'arizona',
        'Arkansas': 'arkansas',
        'California': 'california',
        'Colorado': 'colorado',
        'Connecticut': 'connecticut',
        'Delaware': 'delaware',
        'District of Columbia': 'district-of-columbia',
        'Florida': 'florida',
        'Georgia': 'georgia',
        'Hawaii': 'hawaii',
        'Idaho': 'idaho',
        'Illinois': 'illinois',
        'Indiana': 'indiana',
        'Iowa': 'iowa',
        'Kansas': 'kansas',
        'Kentucky': 'kentucky',
        'Louisiana': 'louisiana',
        'Maine': 'maine',
        'Maryland': 'maryland',
        'Massachusetts': 'massachusetts',
        'Michigan': 'michigan',
        'Minnesota': 'minnesota',
        'Mississippi': 'mississippi',
        'Missouri': 'missouri',
        'Montana': 'montana',
        'Nebraska': 'nebraska',
        'Nevada': 'nevada',
        'New Hampshire': 'new-hampshire',
        'New Jersey': 'new-jersey',
        'New Mexico': 'new-mexico',
        'New York': 'new-york',
        'North Carolina': 'north-carolina',
        'North Dakota': 'north-dakota',
        'Ohio': 'ohio',
        'Oklahoma': 'oklahoma',
        'Oregon': 'oregon',
        'Pennsylvania': 'pennsylvania',
        'Rhode Island': 'rhode-island',
        'South Carolina': 'south-carolina',
        'South Dakota': 'south-dakota',
        'Tennessee': 'tennessee',
        'Texas': 'texas',
        'Utah': 'utah',
        'Vermont': 'vermont',
        'Virginia': 'virginia',
        'Washington': 'washington',
        'West Virginia': 'west-virginia',
        'Wisconsin': 'wisconsin',
        'Wyoming': 'wyoming',
        'Puerto Rico': 'puerto-rico',
    },
    # China Provinces - Geofabrik doesn't have individual provinces, use country file
    # But we keep this for future use if needed
    'China': {},
    # India States - Geofabrik doesn't have individual states, use country file
    'India': {},
}


def setup_logging(log_file, name='osrm'):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.expanduser(log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


def run_command(cmd, timeout=3600, logger=None):
    """Run shell command with timeout."""
    if logger:
        logger.info(f"Running: {cmd[:100]}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            if logger:
                logger.error(f"Command failed: {result.stderr}")
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"Command timed out after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        if logger:
            logger.error(f"Command error: {e}")
        return False, str(e)


def load_admin_boundaries(admin_file=None):
    """Load admin boundaries for overlap analysis."""
    if admin_file is None:
        admin_file = DEFAULT_ADMIN_BOUNDS
    admin_file = Path(admin_file)

    if not admin_file.exists():
        return None

    admin_gdf = gpd.read_file(admin_file)
    # Filter to countries that need subnational processing
    return admin_gdf[admin_gdf['admin'].isin(SUBNATIONAL_COUNTRIES)]


def find_overlapping_regions(city_geometry, admin_gdf, country_name):
    """Find admin regions that overlap with a city geometry."""
    if admin_gdf is None:
        return []

    # Map country name to Natural Earth format
    canonical_country = COUNTRY_NAME_MAP.get(country_name, country_name)

    # Filter to the country
    country_admin = admin_gdf[admin_gdf['admin'] == canonical_country]
    if len(country_admin) == 0:
        return []

    # Find overlapping regions
    overlapping = country_admin[country_admin.intersects(city_geometry)]
    return overlapping['name'].tolist()


def get_osm_files_for_regions(regions, country_name, osm_dir):
    """Map region names to OSM file paths."""
    osm_dir = Path(osm_dir)
    osm_files = []

    # Map country name to canonical format for GEOFABRIK_NAME_MAP lookup
    canonical_country = COUNTRY_NAME_MAP.get(country_name, country_name)
    name_map = GEOFABRIK_NAME_MAP.get(canonical_country, {})

    for region in regions:
        geofabrik_name = name_map.get(region)
        if geofabrik_name:
            # Look for the OSM file
            osm_file = osm_dir / f"{geofabrik_name}-latest.osm.pbf"
            if osm_file.exists():
                osm_files.append(osm_file)
            else:
                # Try without -latest suffix
                for f in osm_dir.glob(f"{geofabrik_name}*.osm.pbf"):
                    osm_files.append(f)
                    break

    return list(set(osm_files))  # Remove duplicates


def merge_osm_files(osm_files, output_file, logger=None):
    """Merge multiple OSM files into one."""
    if len(osm_files) == 1:
        return osm_files[0]

    output_file = Path(output_file)

    # Check if merge already exists and is valid
    if output_file.exists() and output_file.stat().st_size > 1000:
        if logger:
            logger.info(f"Using existing merged file: {output_file}")
        return output_file

    files_str = ' '.join(str(f) for f in osm_files)
    cmd = f"osmium merge {files_str} -o {output_file} --overwrite"

    success, _ = run_command(cmd, timeout=1800, logger=logger)
    if success and output_file.exists():
        return output_file
    return None


def find_osm_file(country_name, osm_dir):
    """Find the appropriate OSM file for a country."""
    osm_dir = Path(osm_dir)
    country_lower = country_name.lower().replace(' ', '-').replace('_', '-')

    # Common name mappings
    name_mappings = {
        'united-states': ['us', 'usa', 'united-states'],
        'united-kingdom': ['great-britain', 'uk', 'united-kingdom'],
        'democratic-republic-of-the-congo': ['congo-democratic-republic', 'drc'],
        'republic-of-the-congo': ['congo-brazzaville'],
    }

    search_names = [country_lower]
    for key, aliases in name_mappings.items():
        if country_lower in aliases or country_lower == key:
            search_names.extend(aliases)
            search_names.append(key)

    # Search for matching OSM file
    for osm_file in osm_dir.glob("*.osm.pbf"):
        file_lower = osm_file.stem.lower().replace('-latest', '')
        for name in search_names:
            if name in file_lower or file_lower in name:
                return osm_file

    # Fall back to largest file (likely continent-level)
    region_files = list(osm_dir.glob("*-latest.osm.pbf"))
    if region_files:
        region_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        return region_files[0]

    return None


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_city_area_km2(city_gdf):
    """Calculate city area in km²."""
    try:
        return city_gdf.to_crs('EPSG:3857').geometry.area.iloc[0] / 1e6
    except:
        return 0


def is_large_city(city_gdf, threshold=LARGE_CITY_THRESHOLD_KM2):
    """Check if city is large enough to keep compressed OSRM."""
    return get_city_area_km2(city_gdf) > threshold
