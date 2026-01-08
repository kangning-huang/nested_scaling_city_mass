#!/bin/bash
#SBATCH --job-name=osrm_centrality
#SBATCH --output=/scratch/kh3657/osrm/logs/centrality_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/centrality_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=acc
#SBATCH --array=1-5

# Calculate centrality for test cities with >=5 grids
# Cities: 2288, 11549, 8675, 12569, 10297

WORK_DIR=/scratch/kh3657/osrm

# Map array index to city ID
case $SLURM_ARRAY_TASK_ID in
    1) CITY_ID=2288 ;;    # Dusseldorf, 7 grids
    2) CITY_ID=11549 ;;   # Wuhan, 25 grids
    3) CITY_ID=8675 ;;    # Chennai, 24 grids
    4) CITY_ID=12569 ;;   # Quanzhou, 33 grids
    5) CITY_ID=10297 ;;   # Chengdu, 37 grids
esac

echo "Calculating centrality for city $CITY_ID (task $SLURM_ARRAY_TASK_ID)"

# Check if already done
if [ -f "$WORK_DIR/results/${CITY_ID}_centrality.geojson" ]; then
    echo "SKIP: Centrality already calculated for $CITY_ID"
    exit 0
fi

# Activate Python environment
source ~/osrm_venv/bin/activate

# Calculate centrality with population weighting
python3 $WORK_DIR/scripts/calculate_centrality_hpc.py \
    --city-id $CITY_ID \
    --results-dir $WORK_DIR/results \
    --cities-dir $WORK_DIR/cities \
    --population-file $WORK_DIR/data/Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-22.csv \
    --h3-resolution 6

deactivate

echo "=== Complete ==="
ls -lh $WORK_DIR/results/${CITY_ID}_centrality.geojson 2>/dev/null || echo "No centrality file created"
