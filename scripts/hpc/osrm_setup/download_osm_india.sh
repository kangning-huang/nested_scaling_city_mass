#!/bin/bash
# Download OSM data for India
cd ~/osrm-data
echo "Downloading India OSM data..."
wget -c https://download.geofabrik.de/asia/india-latest.osm.pbf
echo "Download complete!"
ls -lh ~/osrm-data/
