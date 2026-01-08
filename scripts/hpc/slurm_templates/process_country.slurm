#!/bin/bash
#SBATCH --job-name=osrm_country
#SBATCH --output=/scratch/kh3657/osrm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#
# Generic country processing template
# Array size is set dynamically when submitting
#
# Submit with:
#   # For India (3248 cities):
#   sbatch --job-name=osrm_india --array=1-3248%20 process_country.slurm india asia_india-latest.osm.pbf
#
#   # For China (1850 cities):
#   sbatch --job-name=osrm_china --array=1-1850%20 process_country.slurm china asia_china-latest.osm.pbf
#
#   # For smaller countries:
#   sbatch --job-name=osrm_ethiopia --array=1-557%20 process_country.slurm ethiopia africa_ethiopia-latest.osm.pbf

set -e

# Arguments passed from sbatch command line
COUNTRY=$1
COUNTRY_OSM=$2

if [ -z "$COUNTRY" ] || [ -z "$COUNTRY_OSM" ]; then
    echo "Usage: sbatch --array=1-N%20 process_country.slurm <country> <osm_file>"
    echo "Example: sbatch --array=1-3248%20 process_country.slurm india asia_india-latest.osm.pbf"
    exit 1
fi

# Paths
WORK_DIR=/scratch/kh3657/osrm
CITY_LIST=$WORK_DIR/city_lists/${COUNTRY}_cities.txt
RESULTS_DIR=$WORK_DIR/results

# Get city ID for this array task
CITY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CITY_LIST)

if [ -z "$CITY_ID" ]; then
    echo "ERROR: No city ID found for task $SLURM_ARRAY_TASK_ID in $CITY_LIST"
    exit 1
fi

echo "===== Job Info ====="
echo "Country: $COUNTRY"
echo "City ID: $CITY_ID"
echo "Task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "OSM File: $COUNTRY_OSM"
echo "===================="

# Skip if already completed
if [ -f "$RESULTS_DIR/${CITY_ID}_matrix.json" ] && [ -f "$RESULTS_DIR/${CITY_ID}_routes.geojson" ]; then
    echo "City $CITY_ID already completed, skipping..."
    exit 0
fi

# Run the processing pipeline
cd $WORK_DIR
bash slurm/process_single_city.sh $CITY_ID $COUNTRY_OSM

echo "Completed city $CITY_ID"
