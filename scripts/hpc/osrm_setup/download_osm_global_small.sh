#!/bin/bash
# Download OSM data for Oceania, Africa, and Latin America
cd ~/osrm-data
echo "Downloading OSM data for Oceania, Africa, and Latin America..."

# Continent-level downloads
wget -c https://download.geofabrik.de/africa-latest.osm.pbf
wget -c https://download.geofabrik.de/south-america-latest.osm.pbf
wget -c https://download.geofabrik.de/central-america-latest.osm.pbf
wget -c https://download.geofabrik.de/australia-oceania-latest.osm.pbf

echo "Download complete!"
ls -lh ~/osrm-data/
