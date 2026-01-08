#!/bin/bash
echo "Checking batch download progress..."
echo ""
ls -lh "../data/raw/pois" | grep -E "Rome|Paris|Atlanta|Tokyo|Addis|Bogota|Mexico|Melbourne" | grep "gpkg$" | wc -l | xargs echo "Cities completed:"
echo ""
echo "POI files:"
ls -lh "../data/raw/pois" | grep -E "Rome|Paris|Atlanta|Tokyo|Addis|Bogota|Mexico|Melbourne" | grep "_pois_9cats.gpkg$"
