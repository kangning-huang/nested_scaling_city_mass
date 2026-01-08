#!/bin/bash
# start_processing.sh - Start OSRM processing on a VM
# This script should be run ON the VM

set -e

echo "=== Starting OSRM Processing ==="
echo "Timestamp: $(date)"

# Check if downloads are complete
if pgrep -f wget > /dev/null; then
    echo "ERROR: Downloads still in progress. Wait for downloads to complete first."
    exit 1
fi

# Check if OSM files exist
osm_count=$(ls ~/osrm-data/*.osm.pbf 2>/dev/null | wc -l)
if [ "$osm_count" -eq 0 ]; then
    echo "ERROR: No OSM files found in ~/osrm-data/"
    exit 1
fi
echo "Found $osm_count OSM file(s)"

# Check if cities file exists
if [ ! -f ~/cities.geojson ]; then
    echo "ERROR: ~/cities.geojson not found"
    exit 1
fi

# Activate Python environment
source ~/osrm_env/bin/activate

# Start processing in background
echo "Starting processing script..."
cd ~
nohup python3 ~/process_cities.py > ~/processing.log 2>&1 &

echo "Processing started! PID: $!"
echo "Monitor with: tail -f ~/processing.log"
