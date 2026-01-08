#!/bin/bash
# Download OSM data for Northern America (US states + Canada)
cd ~/osrm-data
echo "Downloading OSM data for US states and Canada..."

# Major US states with cities
wget -c https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/california-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/michigan-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/florida-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/pennsylvania-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/ohio-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/georgia-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/new-jersey-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/massachusetts-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/arizona-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/colorado-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/washington-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/virginia-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/north-carolina-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/indiana-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/minnesota-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/missouri-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/wisconsin-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/tennessee-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/oregon-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/louisiana-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/connecticut-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/utah-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/nevada-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/kentucky-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/oklahoma-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/alabama-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/south-carolina-latest.osm.pbf
wget -c https://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf

# Canada
wget -c https://download.geofabrik.de/north-america/canada-latest.osm.pbf

echo "Download complete!"
ls -lh ~/osrm-data/
