#!/bin/bash
# Download OSM data for China
cd ~/osrm-data
echo "Downloading China OSM data..."
wget -c https://download.geofabrik.de/asia/china-latest.osm.pbf
echo "Download complete!"
ls -lh ~/osrm-data/
