#!/bin/bash
#SBATCH --job-name=osrm_polylines
#SBATCH --output=/scratch/kh3657/osrm/logs/polylines_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/polylines_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=acc
#SBATCH --array=1-5

# Fetch polylines for test cities with >=5 grids
# Cities: 2288, 11549, 8675, 12569, 10297

module load Singularity/4.3.1-gcc-8.5.0

WORK_DIR=/scratch/kh3657/osrm
OSRM_SIF=$WORK_DIR/containers/osrm-backend_latest.sif

# Unique port per task
PORT=$((5000 + SLURM_ARRAY_TASK_ID))

# Map array index to city ID
case $SLURM_ARRAY_TASK_ID in
    1) CITY_ID=2288 ;;    # Dusseldorf, 7 grids
    2) CITY_ID=11549 ;;   # Wuhan, 25 grids
    3) CITY_ID=8675 ;;    # Chennai, 24 grids
    4) CITY_ID=12569 ;;   # Quanzhou, 33 grids
    5) CITY_ID=10297 ;;   # Chengdu, 37 grids
esac

echo "Processing city $CITY_ID (task $SLURM_ARRAY_TASK_ID) on port $PORT"

# Check if already done
if [ -f "$WORK_DIR/results/${CITY_ID}_routes.geojson" ]; then
    echo "SKIP: Routes already exist for $CITY_ID"
    exit 0
fi

# Check if OSRM files exist
OSRM_DIR=$WORK_DIR/osrm-files/$CITY_ID
if [ ! -f "$OSRM_DIR/${CITY_ID}.osrm" ]; then
    echo "ERROR: OSRM files not found for $CITY_ID"
    exit 1
fi

cd $OSRM_DIR

# Start OSRM server on unique port
echo "Starting OSRM server on port $PORT..."
singularity exec -B ${PWD}:/data $OSRM_SIF \
    osrm-routed --port $PORT --algorithm mld /data/${CITY_ID}.osrm &
OSRM_PID=$!
sleep 20

# Wait for server
for i in 1 2 3 4 5; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "OSRM server running (attempt $i)"
        break
    fi
    echo "Waiting for server... (attempt $i)"
    sleep 5
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: OSRM server failed to start"
    kill $OSRM_PID 2>/dev/null
    exit 1
fi

# Activate Python environment
source ~/osrm_venv/bin/activate

# Fetch polylines
echo "Fetching polylines..."
PORT=$PORT python3 $WORK_DIR/scripts/fetch_polylines_hpc.py \
    --city-id $CITY_ID \
    --results-dir $WORK_DIR/results \
    --port $PORT

deactivate

# Stop server
echo "Stopping OSRM server..."
kill $OSRM_PID 2>/dev/null
wait $OSRM_PID 2>/dev/null

echo "=== Complete ==="
ls -lh $WORK_DIR/results/${CITY_ID}_routes.geojson 2>/dev/null || echo "No routes file created"
