#!/usr/bin/env python3
"""
Common utilities shared across OSRM processing scripts.
Version 2: Improved country-to-OSM file mapping.
"""

import os
import logging
import subprocess
from pathlib import Path

# Default directories
DEFAULT_OSM_DIR = Path(os.path.expanduser("~/osrm-data"))
DEFAULT_CLIPPED_DIR = Path(os.path.expanduser("~/clipped"))
DEFAULT_OSRM_DIR = Path(os.path.expanduser("~/osrm"))
DEFAULT_CITIES_DIR = Path(os.path.expanduser("~/cities"))
DEFAULT_RESULTS_DIR = Path(os.path.expanduser("~/results"))

LARGE_CITY_THRESHOLD_KM2 = 100

# Comprehensive country name to Geofabrik OSM file mapping
COUNTRY_TO_OSM = {
    # Africa
    "algeria": "algeria",
    "angola": "angola",
    "benin": "benin",
    "botswana": "botswana",
    "burkina faso": "burkina-faso",
    "burundi": "burundi",
    "cameroon": "cameroon",
    "cape verde": "cape-verde",
    "central african republic": "central-african-republic",
    "chad": "chad",
    "comoros": "comoros",
    "congo": "congo-brazzaville",
    "republic of congo": "congo-brazzaville",
    "republic of the congo": "congo-brazzaville",
    "democratic republic of the congo": "congo-democratic-republic",
    "drc": "congo-democratic-republic",
    "cote d ivoire": "ivory-coast",
    "côte d'ivoire": "ivory-coast",
    "ivory coast": "ivory-coast",
    "djibouti": "djibouti",
    "egypt": "egypt",
    "equatorial guinea": "equatorial-guinea",
    "eritrea": "eritrea",
    "eswatini": "eswatini",
    "swaziland": "eswatini",
    "ethiopia": "ethiopia",
    "gabon": "gabon",
    "gambia": "senegal-and-gambia",
    "ghana": "ghana",
    "guinea": "guinea",
    "guinea-bissau": "guinea-bissau",
    "kenya": "kenya",
    "lesotho": "lesotho",
    "liberia": "liberia",
    "libya": "libya",
    "madagascar": "madagascar",
    "malawi": "malawi",
    "mali": "mali",
    "mauritania": "mauritania",
    "mauritius": "mauritius",
    "morocco": "morocco",
    "mozambique": "mozambique",
    "namibia": "namibia",
    "niger": "niger",
    "nigeria": "nigeria",
    "rwanda": "rwanda",
    "sao tome and principe": "sao-tome-and-principe",
    "são tomé and príncipe": "sao-tome-and-principe",
    "senegal": "senegal-and-gambia",
    "sierra leone": "sierra-leone",
    "somalia": "somalia",
    "south africa": "south-africa",
    "south sudan": "south-sudan",
    "sudan": "sudan",
    "tanzania": "tanzania",
    "togo": "togo",
    "tunisia": "tunisia",
    "uganda": "uganda",
    "western sahara": "morocco",
    "zambia": "zambia",
    "zimbabwe": "zimbabwe",
    # Latin America
    "argentina": "argentina",
    "belize": "belize",
    "bolivia": "bolivia",
    "brazil": "brazil",
    "chile": "chile",
    "colombia": "colombia",
    "costa rica": "costa-rica",
    "cuba": "cuba",
    "dominican republic": "haiti-and-domrep",
    "ecuador": "ecuador",
    "el salvador": "el-salvador",
    "french guiana": "south-america",
    "guatemala": "guatemala",
    "guyana": "guyana",
    "haiti": "haiti-and-domrep",
    "honduras": "honduras",
    "jamaica": "jamaica",
    "mexico": "mexico",
    "nicaragua": "nicaragua",
    "panama": "panama",
    "paraguay": "paraguay",
    "peru": "peru",
    "puerto rico": "puerto-rico",
    "suriname": "suriname",
    "trinidad and tobago": "central-america",
    "uruguay": "uruguay",
    "venezuela": "venezuela",
    # Caribbean
    "bahamas": "central-america",
    "barbados": "central-america",
    "curaçao": "central-america",
    "curacao": "central-america",
    # Oceania
    "australia": "australia",
    "fiji": "fiji",
    "french polynesia": "australia-oceania",
    "new caledonia": "new-caledonia",
    "new zealand": "new-zealand",
    "papua new guinea": "papua-new-guinea",
    "solomon islands": "australia-oceania",
    # Europe (for reference)
    "france": "france",
}

def setup_logging(log_file, name="osrm"):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
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

def find_osm_file(country_name, osm_dir):
    """Find the appropriate OSM file for a country. Never falls back to continent files."""
    osm_dir = Path(osm_dir)
    country_lower = country_name.lower().strip()

    # Try the explicit mapping first
    osm_name = COUNTRY_TO_OSM.get(country_lower)
    if osm_name:
        osm_file = osm_dir / f"{osm_name}-latest.osm.pbf"
        if osm_file.exists():
            return osm_file

    # Try normalized name matching
    country_normalized = country_lower.replace(" ", "-").replace("_", "-")
    for osm_file in osm_dir.glob("*.osm.pbf"):
        file_name = osm_file.stem.lower().replace("-latest", "")
        if country_normalized == file_name:
            return osm_file
        if country_normalized in file_name or file_name in country_normalized:
            # Avoid matching continent files
            if file_name not in ["africa", "asia", "europe", "north-america", "south-america", "australia-oceania", "central-america"]:
                return osm_file

    # No match found - return None instead of falling back to continent file
    return None

def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def get_city_area_km2(city_gdf):
    """Calculate city area in km²."""
    try:
        return city_gdf.to_crs("EPSG:3857").geometry.area.iloc[0] / 1e6
    except:
        return 0

def is_large_city(city_gdf, threshold=LARGE_CITY_THRESHOLD_KM2):
    """Check if city is large enough to keep compressed OSRM."""
    return get_city_area_km2(city_gdf) > threshold
