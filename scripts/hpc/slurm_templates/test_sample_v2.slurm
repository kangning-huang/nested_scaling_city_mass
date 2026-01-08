#!/bin/bash
#SBATCH --job-name=osrm_test_v2
#SBATCH --output=/scratch/kh3657/osrm/logs/test_v2_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/test_v2_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=acc
#SBATCH --array=1-11

# Minimum grid threshold
MIN_GRIDS=5
n# Unique port per array task to avoid conflicts when running on same node
PORT=$((5000 + SLURM_ARRAY_TASK_ID))

module load Singularity/4.3.1-gcc-8.5.0

WORK_DIR=/scratch/kh3657/osrm
OSRM_SIF=$WORK_DIR/containers/osrm-backend_latest.sif
OSMIUM_SIF=$WORK_DIR/containers/osmium-tool_latest.sif

# Verify containers exist
if [ ! -f "$OSRM_SIF" ] || [ ! -f "$OSMIUM_SIF" ]; then
    echo "ERROR: Container files not found"
    echo "OSRM: $OSRM_SIF"
    echo "Osmium: $OSMIUM_SIF"
    exit 1
fi

# Get city ID from array
CITY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $WORK_DIR/city_lists/test_sample_v2.txt)
echo "Processing city: $CITY_ID (task $SLURM_ARRAY_TASK_ID)"

# Map city to OSM file based on country
get_osm_file() {
    local city=$1
    if grep -q "^${city}$" $WORK_DIR/city_lists/china_cities.txt 2>/dev/null; then
        echo "china-latest.osm.pbf"
    elif grep -q "^${city}$" $WORK_DIR/city_lists/india_cities.txt 2>/dev/null; then
        echo "india-latest.osm.pbf"
    elif grep -q "^${city}$" $WORK_DIR/city_lists/germany_cities.txt 2>/dev/null; then
        echo "germany-latest.osm.pbf"
    elif grep -q "^${city}$" $WORK_DIR/city_lists/brazil_cities.txt 2>/dev/null; then
        echo "brazil-latest.osm.pbf"
    elif grep -q "^${city}$" $WORK_DIR/city_lists/argentina_cities.txt 2>/dev/null; then
        echo "argentina-latest.osm.pbf"
    else
        echo "UNKNOWN"
    fi
}

OSM_FILE=$(get_osm_file $CITY_ID)
echo "Using OSM file: $OSM_FILE"

if [ "$OSM_FILE" = "UNKNOWN" ]; then
    echo "ERROR: Cannot find OSM file for city $CITY_ID"
    exit 1
fi

# Verify OSM file exists
if [ ! -f "$WORK_DIR/osm-data/$OSM_FILE" ]; then
    echo "ERROR: OSM file not found: $WORK_DIR/osm-data/$OSM_FILE"
    exit 1
fi

# Check if already done
if [ -f "$WORK_DIR/results/${CITY_ID}_matrix.json" ]; then
    echo "SKIP: City $CITY_ID already processed"
    exit 0
fi

# Phase 1: Clip OSM
echo "=== Phase 1: Clipping OSM ==="
mkdir -p $WORK_DIR/clipped
if [ ! -f "$WORK_DIR/clipped/${CITY_ID}.osm.pbf" ]; then
    singularity exec -B $WORK_DIR:/data $OSMIUM_SIF \
        osmium extract -p /data/cities/${CITY_ID}.geojson \
        /data/osm-data/$OSM_FILE \
        -o /data/clipped/${CITY_ID}.osm.pbf --overwrite
    
    if [ ! -f "$WORK_DIR/clipped/${CITY_ID}.osm.pbf" ]; then
        echo "ERROR: Clipping failed"
        exit 1
    fi
    echo "Clipped: $(ls -lh $WORK_DIR/clipped/${CITY_ID}.osm.pbf)"
else
    echo "Using existing clipped file: $(ls -lh $WORK_DIR/clipped/${CITY_ID}.osm.pbf)"
fi

# Phase 2: OSRM Preprocessing
echo "=== Phase 2: OSRM Preprocessing ==="
mkdir -p $WORK_DIR/osrm-files/$CITY_ID
cd $WORK_DIR/osrm-files/$CITY_ID
cp $WORK_DIR/clipped/${CITY_ID}.osm.pbf .

echo "Running osrm-extract..."
singularity exec -B ${PWD}:/data $OSRM_SIF osrm-extract -p /opt/car.lua /data/${CITY_ID}.osm.pbf
echo "Running osrm-partition..."
singularity exec -B ${PWD}:/data $OSRM_SIF osrm-partition /data/${CITY_ID}.osrm
echo "Running osrm-customize..."
singularity exec -B ${PWD}:/data $OSRM_SIF osrm-customize /data/${CITY_ID}.osrm

# Check if preprocessing succeeded
if [ ! -f "${CITY_ID}.osrm" ]; then
    echo "ERROR: OSRM preprocessing failed - no .osrm file"
    exit 1
fi

rm -f ${CITY_ID}.osm.pbf
echo "OSRM files: $(du -sh .)"

# Phase 3: Routing
echo "=== Phase 3: Computing Matrix ==="
mkdir -p $WORK_DIR/results

# Start OSRM server
echo "Starting OSRM server..."
singularity exec -B ${PWD}:/data $OSRM_SIF \
    osrm-routed --port $PORT --algorithm mld --max-table-size 500 /data/${CITY_ID}.osrm &
OSRM_PID=$!
sleep 25

# Check server
for i in 1 2 3 4 5; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "OSRM server running (attempt $i)"
        break
    fi
    echo "Waiting for server... (attempt $i)"
    sleep 5
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: OSRM server failed to start after 50 seconds"
    kill $OSRM_PID 2>/dev/null
    exit 1
fi

# Compute matrix using Python
python3 << PYEOF
import json
import h3
import requests
from shapely.geometry import shape

city_id = "$CITY_ID"
MIN_GRIDS = $MIN_GRIDS

# Load city boundary and generate H3 grids
with open("$WORK_DIR/cities/${CITY_ID}.geojson") as f:
    data = json.load(f)

if data.get("type") == "FeatureCollection":
    geom = data["features"][0]["geometry"]
elif data.get("type") == "Feature":
    geom = data["geometry"]
else:
    geom = data

poly = shape(geom)
cells = list(h3.geo_to_cells(poly, 6))
n_grids = len(cells)

print(f"City {city_id}: {n_grids} grids")

if n_grids < MIN_GRIDS:
    print(f"SKIP: City has fewer than {MIN_GRIDS} grids")
    exit(0)

# Get centroids
centroids = []
for cell in cells:
    lat, lon = h3.cell_to_latlng(cell)
    centroids.append({"h3_index": cell, "lat": lat, "lon": lon})

# Build coords string
coords = ";".join([f"{c['lon']},{c['lat']}" for c in centroids])

# Query OSRM Table API
url = f"http://localhost:$PORT/table/v1/driving/{coords}?annotations=duration,distance"
print(f"Querying OSRM Table API...")
resp = requests.get(url, timeout=300)
data = resp.json()

if data.get("code") != "Ok":
    print(f"ERROR: {data.get('code')} - {data.get('message', '')}")
    exit(1)

# Save results
result = {
    "city_id": city_id,
    "n_grids": n_grids,
    "h3_resolution": 6,
    "h3_indices": [c["h3_index"] for c in centroids],
    "centroids": centroids,
    "durations": data["durations"],
    "distances": data["distances"]
}

with open(f"$WORK_DIR/results/{city_id}_matrix.json", "w") as f:
    json.dump(result, f)

print(f"SUCCESS: Matrix saved ({n_grids}x{n_grids} = {n_grids*n_grids} cells)")
PYEOF

# Stop server
kill $OSRM_PID 2>/dev/null

echo "=== Complete ==="
ls -lh $WORK_DIR/results/${CITY_ID}_matrix.json 2>/dev/null || echo "No matrix file created"
