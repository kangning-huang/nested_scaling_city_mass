#!/bin/bash
#SBATCH --job-name=osrm_pilot
#SBATCH --output=/scratch/kh3657/osrm/logs/pilot_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/pilot_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-100%20
#
# Pilot batch processing for 100 diverse cities
# Tests the full pipeline before scaling to all 13,135 cities
#
# Submit with:
#   sbatch process_pilot.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f /scratch/kh3657/osrm/logs/pilot_*.out

set -e

# Paths
WORK_DIR=/scratch/kh3657/osrm
CITY_LIST=$WORK_DIR/city_lists/pilot_cities.txt
RESULTS_DIR=$WORK_DIR/results

# Get city ID for this array task
CITY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CITY_LIST)

if [ -z "$CITY_ID" ]; then
    echo "ERROR: No city ID found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing city $CITY_ID (task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT)"

# Skip if already completed
if [ -f "$RESULTS_DIR/${CITY_ID}_matrix.json" ] && [ -f "$RESULTS_DIR/${CITY_ID}_routes.geojson" ]; then
    echo "City $CITY_ID already completed, skipping..."
    exit 0
fi

# Look up the country OSM file for this city
COUNTRY_OSM=$(python3 -c "
import json
with open('$WORK_DIR/city_lists/country_summary.json') as f:
    data = json.load(f)
for country in data:
    for city in country.get('cities', []):
        if city['id'] == $CITY_ID:
            osm_path = country['osm_path']
            if osm_path != 'UNKNOWN':
                print(osm_path.replace('/', '_') + '-latest.osm.pbf')
            break
")

if [ -z "$COUNTRY_OSM" ]; then
    echo "ERROR: Could not find OSM file for city $CITY_ID"
    exit 1
fi

echo "Using OSM file: $COUNTRY_OSM"

# Run the processing pipeline
cd $WORK_DIR
bash slurm/process_single_city.sh $CITY_ID $COUNTRY_OSM

echo "Completed city $CITY_ID"
