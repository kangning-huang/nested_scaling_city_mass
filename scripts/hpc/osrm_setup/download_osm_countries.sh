#!/bin/bash
#
# Download OSM data for all countries with cities in the dataset
# Run on HPC: bash download_osm_countries.sh
#
# This script downloads country-level OSM files from Geofabrik.
# Country-level files are MUCH faster to process than continent files.
#
# Estimated total download size: ~30 GB
# Estimated time: 1-2 hours

set -e

OSM_DIR=/scratch/kh3657/osrm/osm-data
mkdir -p $OSM_DIR
cd $OSM_DIR

echo "Downloading OSM data to $OSM_DIR"
echo "Start time: $(date)"
echo ""

# Function to download with retry
download_osm() {
    local url=$1
    local filename=$2

    if [ -f "$filename" ]; then
        echo "  Already exists: $filename"
        return 0
    fi

    echo "  Downloading: $filename"
    wget -c -q --show-progress -O "$filename" "$url" || {
        echo "  Retrying: $filename"
        sleep 5
        wget -c -q --show-progress -O "$filename" "$url"
    }
}

# =============================================================================
# Priority 1: Large Countries (>500 cities)
# =============================================================================
echo "=== Priority 1: Large Countries ==="

download_osm "https://download.geofabrik.de/asia/india-latest.osm.pbf" "asia_india-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/china-latest.osm.pbf" "asia_china-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/ethiopia-latest.osm.pbf" "africa_ethiopia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/nigeria-latest.osm.pbf" "africa_nigeria-latest.osm.pbf"

# =============================================================================
# Priority 2: Medium Countries (100-500 cities)
# =============================================================================
echo ""
echo "=== Priority 2: Medium Countries ==="

# Asia
download_osm "https://download.geofabrik.de/asia/indonesia-latest.osm.pbf" "asia_indonesia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf" "asia_bangladesh-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/pakistan-latest.osm.pbf" "asia_pakistan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/japan-latest.osm.pbf" "asia_japan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/philippines-latest.osm.pbf" "asia_philippines-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf" "asia_vietnam-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/iran-latest.osm.pbf" "asia_iran-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/myanmar-latest.osm.pbf" "asia_myanmar-latest.osm.pbf"

# Africa
download_osm "https://download.geofabrik.de/africa/egypt-latest.osm.pbf" "africa_egypt-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/south-africa-latest.osm.pbf" "africa_south-africa-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/kenya-latest.osm.pbf" "africa_kenya-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/tanzania-latest.osm.pbf" "africa_tanzania-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/uganda-latest.osm.pbf" "africa_uganda-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/democratic-republic-of-the-congo-latest.osm.pbf" "africa_democratic-republic-of-the-congo-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/sudan-latest.osm.pbf" "africa_sudan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/algeria-latest.osm.pbf" "africa_algeria-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/morocco-latest.osm.pbf" "africa_morocco-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/ghana-latest.osm.pbf" "africa_ghana-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/cameroon-latest.osm.pbf" "africa_cameroon-latest.osm.pbf"

# Americas
download_osm "https://download.geofabrik.de/south-america/brazil-latest.osm.pbf" "south-america_brazil-latest.osm.pbf"
download_osm "https://download.geofabrik.de/south-america/colombia-latest.osm.pbf" "south-america_colombia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/mexico-latest.osm.pbf" "north-america_mexico-latest.osm.pbf"

# Europe
download_osm "https://download.geofabrik.de/russia-latest.osm.pbf" "russia-latest.osm.pbf"

# =============================================================================
# Priority 3: Other Countries
# =============================================================================
echo ""
echo "=== Priority 3: Other Countries ==="

# Asia
download_osm "https://download.geofabrik.de/asia/thailand-latest.osm.pbf" "asia_thailand-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/south-korea-latest.osm.pbf" "asia_south-korea-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/nepal-latest.osm.pbf" "asia_nepal-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf" "asia_malaysia-singapore-brunei-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/cambodia-latest.osm.pbf" "asia_cambodia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf" "asia_sri-lanka-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/iraq-latest.osm.pbf" "asia_iraq-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/afghanistan-latest.osm.pbf" "asia_afghanistan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/uzbekistan-latest.osm.pbf" "asia_uzbekistan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/kazakhstan-latest.osm.pbf" "asia_kazakhstan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/taiwan-latest.osm.pbf" "asia_taiwan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/gcc-states-latest.osm.pbf" "asia_gcc-states-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf" "asia_israel-and-palestine-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/jordan-latest.osm.pbf" "asia_jordan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/syria-latest.osm.pbf" "asia_syria-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/yemen-latest.osm.pbf" "asia_yemen-latest.osm.pbf"
download_osm "https://download.geofabrik.de/asia/lebanon-latest.osm.pbf" "asia_lebanon-latest.osm.pbf"

# Africa - remaining countries
download_osm "https://download.geofabrik.de/africa/angola-latest.osm.pbf" "africa_angola-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/mozambique-latest.osm.pbf" "africa_mozambique-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/madagascar-latest.osm.pbf" "africa_madagascar-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/ivory-coast-latest.osm.pbf" "africa_ivory-coast-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/niger-latest.osm.pbf" "africa_niger-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/burkina-faso-latest.osm.pbf" "africa_burkina-faso-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/mali-latest.osm.pbf" "africa_mali-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/malawi-latest.osm.pbf" "africa_malawi-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/zambia-latest.osm.pbf" "africa_zambia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/senegal-and-gambia-latest.osm.pbf" "africa_senegal-and-gambia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/zimbabwe-latest.osm.pbf" "africa_zimbabwe-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/chad-latest.osm.pbf" "africa_chad-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/somalia-latest.osm.pbf" "africa_somalia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/guinea-latest.osm.pbf" "africa_guinea-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/rwanda-latest.osm.pbf" "africa_rwanda-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/benin-latest.osm.pbf" "africa_benin-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/burundi-latest.osm.pbf" "africa_burundi-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/tunisia-latest.osm.pbf" "africa_tunisia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/south-sudan-latest.osm.pbf" "africa_south-sudan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/togo-latest.osm.pbf" "africa_togo-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/sierra-leone-latest.osm.pbf" "africa_sierra-leone-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/libya-latest.osm.pbf" "africa_libya-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/eritrea-latest.osm.pbf" "africa_eritrea-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/congo-brazzaville-latest.osm.pbf" "africa_congo-brazzaville-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/liberia-latest.osm.pbf" "africa_liberia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/africa/central-african-republic-latest.osm.pbf" "africa_central-african-republic-latest.osm.pbf"

# Europe
download_osm "https://download.geofabrik.de/europe/turkey-latest.osm.pbf" "europe_turkey-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf" "europe_ukraine-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/poland-latest.osm.pbf" "europe_poland-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/germany-latest.osm.pbf" "europe_germany-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/france-latest.osm.pbf" "europe_france-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/great-britain-latest.osm.pbf" "europe_great-britain-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/italy-latest.osm.pbf" "europe_italy-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/spain-latest.osm.pbf" "europe_spain-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/romania-latest.osm.pbf" "europe_romania-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/greece-latest.osm.pbf" "europe_greece-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/netherlands-latest.osm.pbf" "europe_netherlands-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/belgium-latest.osm.pbf" "europe_belgium-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf" "europe_czech-republic-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/portugal-latest.osm.pbf" "europe_portugal-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/sweden-latest.osm.pbf" "europe_sweden-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/hungary-latest.osm.pbf" "europe_hungary-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/belarus-latest.osm.pbf" "europe_belarus-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/austria-latest.osm.pbf" "europe_austria-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/serbia-latest.osm.pbf" "europe_serbia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/bulgaria-latest.osm.pbf" "europe_bulgaria-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/georgia-latest.osm.pbf" "europe_georgia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/europe/azerbaijan-latest.osm.pbf" "europe_azerbaijan-latest.osm.pbf"

# Americas
download_osm "https://download.geofabrik.de/north-america/canada-latest.osm.pbf" "north-america_canada-latest.osm.pbf"
download_osm "https://download.geofabrik.de/south-america/argentina-latest.osm.pbf" "south-america_argentina-latest.osm.pbf"
download_osm "https://download.geofabrik.de/south-america/peru-latest.osm.pbf" "south-america_peru-latest.osm.pbf"
download_osm "https://download.geofabrik.de/south-america/venezuela-latest.osm.pbf" "south-america_venezuela-latest.osm.pbf"
download_osm "https://download.geofabrik.de/south-america/chile-latest.osm.pbf" "south-america_chile-latest.osm.pbf"
download_osm "https://download.geofabrik.de/south-america/ecuador-latest.osm.pbf" "south-america_ecuador-latest.osm.pbf"
download_osm "https://download.geofabrik.de/south-america/bolivia-latest.osm.pbf" "south-america_bolivia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/central-america/guatemala-latest.osm.pbf" "central-america_guatemala-latest.osm.pbf"
download_osm "https://download.geofabrik.de/central-america/cuba-latest.osm.pbf" "central-america_cuba-latest.osm.pbf"
download_osm "https://download.geofabrik.de/central-america/haiti-and-domrep-latest.osm.pbf" "central-america_haiti-and-domrep-latest.osm.pbf"
download_osm "https://download.geofabrik.de/central-america/honduras-latest.osm.pbf" "central-america_honduras-latest.osm.pbf"

# Oceania
download_osm "https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf" "australia-oceania_australia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/australia-oceania/new-zealand-latest.osm.pbf" "australia-oceania_new-zealand-latest.osm.pbf"
download_osm "https://download.geofabrik.de/australia-oceania/papua-new-guinea-latest.osm.pbf" "australia-oceania_papua-new-guinea-latest.osm.pbf"

# =============================================================================
# US States (for cities in USA)
# =============================================================================
echo ""
echo "=== US States ==="

# Only download states that have cities in the dataset
# Top US states by city count
download_osm "https://download.geofabrik.de/north-america/us/california-latest.osm.pbf" "north-america_us_california-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf" "north-america_us_texas-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/florida-latest.osm.pbf" "north-america_us_florida-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf" "north-america_us_new-york-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/new-jersey-latest.osm.pbf" "north-america_us_new-jersey-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/pennsylvania-latest.osm.pbf" "north-america_us_pennsylvania-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf" "north-america_us_illinois-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/ohio-latest.osm.pbf" "north-america_us_ohio-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/georgia-latest.osm.pbf" "north-america_us_georgia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/michigan-latest.osm.pbf" "north-america_us_michigan-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/north-carolina-latest.osm.pbf" "north-america_us_north-carolina-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/arizona-latest.osm.pbf" "north-america_us_arizona-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/massachusetts-latest.osm.pbf" "north-america_us_massachusetts-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/washington-latest.osm.pbf" "north-america_us_washington-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/colorado-latest.osm.pbf" "north-america_us_colorado-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf" "north-america_us_maryland-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/virginia-latest.osm.pbf" "north-america_us_virginia-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/indiana-latest.osm.pbf" "north-america_us_indiana-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/tennessee-latest.osm.pbf" "north-america_us_tennessee-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/missouri-latest.osm.pbf" "north-america_us_missouri-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/wisconsin-latest.osm.pbf" "north-america_us_wisconsin-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/minnesota-latest.osm.pbf" "north-america_us_minnesota-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/oregon-latest.osm.pbf" "north-america_us_oregon-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/louisiana-latest.osm.pbf" "north-america_us_louisiana-latest.osm.pbf"
download_osm "https://download.geofabrik.de/north-america/us/connecticut-latest.osm.pbf" "north-america_us_connecticut-latest.osm.pbf"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "===== Download Complete ====="
echo "End time: $(date)"
echo ""
echo "Downloaded files:"
ls -lh $OSM_DIR/*.osm.pbf | head -20
echo "..."
echo ""
echo "Total size:"
du -sh $OSM_DIR
