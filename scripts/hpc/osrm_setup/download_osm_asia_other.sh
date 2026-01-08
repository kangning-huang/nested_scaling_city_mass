#!/bin/bash
# Download OSM data for Asia (excluding India and China)
cd ~/osrm-data
echo "Downloading OSM data for Asian countries..."

# Major countries with many cities
wget -c https://download.geofabrik.de/asia/japan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/indonesia-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/pakistan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/vietnam-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/iran-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/philippines-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/myanmar-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/iraq-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/thailand-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/south-korea-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/north-korea-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/afghanistan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/nepal-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/uzbekistan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/saudi-arabia-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/yemen-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/syria-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/cambodia-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/taiwan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/jordan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/mongolia-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/kazakhstan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/laos-latest.osm.pbf

echo "Download complete!"
ls -lh ~/osrm-data/
