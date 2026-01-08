#!/bin/bash
# Download OSM data for Europe
cd ~/osrm-data
echo "Downloading OSM data for European countries..."

# Major countries with many cities
wget -c https://download.geofabrik.de/europe/russia-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/great-britain-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/germany-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/france-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/italy-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/spain-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/poland-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/ukraine-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/turkey-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/romania-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/netherlands-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/belgium-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/greece-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/portugal-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/sweden-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/hungary-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/austria-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/switzerland-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/serbia-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/bulgaria-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/denmark-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/finland-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/norway-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/ireland-and-northern-ireland-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/belarus-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/slovakia-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/croatia-latest.osm.pbf

echo "Download complete!"
ls -lh ~/osrm-data/
