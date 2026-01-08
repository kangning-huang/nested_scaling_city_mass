#!/bin/bash
#
# Process a single city through the full OSRM pipeline
# Usage: ./process_single_city.sh <city_id> <country_osm_file>
#
# Phases:
#   1. Clip OSM data to city boundary
#   2. OSRM preprocessing (extract, partition, customize)
#   3. Compute travel time matrix and fetch route polylines
#
# Prerequisites:
#   - City boundary GeoJSON at /scratch/kh3657/osrm/cities/<city_id>.geojson
#   - Country OSM file at /scratch/kh3657/osrm/osm-data/<country_osm_file>
#   - Singularity containers in /scratch/kh3657/osrm/containers/

set -e  # Exit on error

# Arguments
CITY_ID=$1
COUNTRY_OSM=$2

if [ -z "$CITY_ID" ] || [ -z "$COUNTRY_OSM" ]; then
    echo "Usage: $0 <city_id> <country_osm_file>"
    echo "Example: $0 12400 china-latest.osm.pbf"
    exit 1
fi

# Paths
WORK_DIR=/scratch/kh3657/osrm
CITY_BOUNDARY=$WORK_DIR/cities/${CITY_ID}.geojson
OSM_FILE=$WORK_DIR/osm-data/${COUNTRY_OSM}
CLIPPED_DIR=$WORK_DIR/clipped
OSRM_DIR=$WORK_DIR/osrm-files/${CITY_ID}
RESULTS_DIR=$WORK_DIR/results
LOG_FILE=$WORK_DIR/logs/city_${CITY_ID}.log

# Containers
OSMIUM_SIF=$WORK_DIR/containers/osmium-tool_latest.sif
OSRM_SIF=$WORK_DIR/containers/osrm-backend_latest.sif

# Output files
CLIPPED_OSM=$CLIPPED_DIR/${CITY_ID}.osm.pbf
MATRIX_JSON=$RESULTS_DIR/${CITY_ID}_matrix.json
ROUTES_GEOJSON=$RESULTS_DIR/${CITY_ID}_routes.geojson

# Create directories
mkdir -p $CLIPPED_DIR $OSRM_DIR $RESULTS_DIR $WORK_DIR/logs

# Log start
echo "===== Processing city $CITY_ID =====" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "Country OSM: $COUNTRY_OSM" | tee -a $LOG_FILE

# Check prerequisites
if [ ! -f "$CITY_BOUNDARY" ]; then
    echo "ERROR: City boundary not found: $CITY_BOUNDARY" | tee -a $LOG_FILE
    exit 1
fi

if [ ! -f "$OSM_FILE" ]; then
    echo "ERROR: OSM file not found: $OSM_FILE" | tee -a $LOG_FILE
    exit 1
fi

# Load Singularity module
module load Singularity/4.3.1-gcc-8.5.0

# =============================================================================
# Phase 1: Clip OSM Data
# =============================================================================
echo "" | tee -a $LOG_FILE
echo "--- Phase 1: Clipping OSM data ---" | tee -a $LOG_FILE
PHASE1_START=$(date +%s)

if [ -f "$CLIPPED_OSM" ]; then
    echo "Clipped OSM already exists, skipping..." | tee -a $LOG_FILE
else
    singularity exec -B $WORK_DIR:/data $OSMIUM_SIF \
        osmium extract -p /data/cities/${CITY_ID}.geojson \
        /data/osm-data/${COUNTRY_OSM} \
        -o /data/clipped/${CITY_ID}.osm.pbf --overwrite

    PHASE1_END=$(date +%s)
    echo "Phase 1 completed in $((PHASE1_END - PHASE1_START)) seconds" | tee -a $LOG_FILE
    echo "Clipped OSM size: $(ls -lh $CLIPPED_OSM | awk '{print $5}')" | tee -a $LOG_FILE
fi

# =============================================================================
# Phase 2: OSRM Preprocessing
# =============================================================================
echo "" | tee -a $LOG_FILE
echo "--- Phase 2: OSRM Preprocessing ---" | tee -a $LOG_FILE
PHASE2_START=$(date +%s)

# Check if already preprocessed
if [ -f "$OSRM_DIR/${CITY_ID}.osrm.cell_metrics" ]; then
    echo "OSRM files already exist, skipping preprocessing..." | tee -a $LOG_FILE
else
    mkdir -p $OSRM_DIR
    cp $CLIPPED_OSM $OSRM_DIR/${CITY_ID}.osm.pbf

    cd $OSRM_DIR

    # Extract
    echo "  Running osrm-extract..." | tee -a $LOG_FILE
    singularity exec -B ${PWD}:/data $OSRM_SIF \
        osrm-extract -p /opt/car.lua /data/${CITY_ID}.osm.pbf

    # Partition
    echo "  Running osrm-partition..." | tee -a $LOG_FILE
    singularity exec -B ${PWD}:/data $OSRM_SIF \
        osrm-partition /data/${CITY_ID}.osrm

    # Customize
    echo "  Running osrm-customize..." | tee -a $LOG_FILE
    singularity exec -B ${PWD}:/data $OSRM_SIF \
        osrm-customize /data/${CITY_ID}.osrm

    # Cleanup OSM file from OSRM directory
    rm -f $OSRM_DIR/${CITY_ID}.osm.pbf

    PHASE2_END=$(date +%s)
    echo "Phase 2 completed in $((PHASE2_END - PHASE2_START)) seconds" | tee -a $LOG_FILE
    echo "OSRM files size: $(du -sh $OSRM_DIR | cut -f1)" | tee -a $LOG_FILE
fi

# =============================================================================
# Phase 3: Route Matrix and Polylines
# =============================================================================
echo "" | tee -a $LOG_FILE
echo "--- Phase 3: Computing Routes ---" | tee -a $LOG_FILE
PHASE3_START=$(date +%s)

# Check if already done
if [ -f "$MATRIX_JSON" ] && [ -f "$ROUTES_GEOJSON" ]; then
    echo "Results already exist, skipping..." | tee -a $LOG_FILE
else
    cd $OSRM_DIR

    # Start OSRM server
    echo "  Starting OSRM server..." | tee -a $LOG_FILE
    singularity exec -B ${PWD}:/data $OSRM_SIF \
        osrm-routed --algorithm mld --max-table-size 500 /data/${CITY_ID}.osrm &
    OSRM_PID=$!

    # Wait for server to start
    sleep 15

    # Check if server is running
    if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "  Waiting for server..." | tee -a $LOG_FILE
        sleep 10
    fi

    # Run the routing script
    echo "  Computing matrix and fetching routes..." | tee -a $LOG_FILE

    # Activate Python environment and run routing
    source ~/osrm_venv/bin/activate

    python3 $WORK_DIR/scripts/route_city_hpc.py \
        --city-id $CITY_ID \
        --grids-path $WORK_DIR/all_cities_h3_grids.gpkg \
        --output-dir $RESULTS_DIR \
        --fetch-polylines

    deactivate

    # Stop OSRM server
    echo "  Stopping OSRM server..." | tee -a $LOG_FILE
    kill $OSRM_PID 2>/dev/null || true
    wait $OSRM_PID 2>/dev/null || true

    PHASE3_END=$(date +%s)
    echo "Phase 3 completed in $((PHASE3_END - PHASE3_START)) seconds" | tee -a $LOG_FILE
fi

# =============================================================================
# Cleanup
# =============================================================================
echo "" | tee -a $LOG_FILE
echo "--- Cleanup ---" | tee -a $LOG_FILE

# Delete clipped OSM (can regenerate from country file)
if [ -f "$CLIPPED_OSM" ]; then
    rm -f $CLIPPED_OSM
    echo "Deleted clipped OSM file" | tee -a $LOG_FILE
fi

# Delete OSRM files (keep results only)
if [ -d "$OSRM_DIR" ]; then
    rm -rf $OSRM_DIR
    echo "Deleted OSRM directory" | tee -a $LOG_FILE
fi

# =============================================================================
# Summary
# =============================================================================
echo "" | tee -a $LOG_FILE
echo "===== City $CITY_ID Complete =====" | tee -a $LOG_FILE
echo "End time: $(date)" | tee -a $LOG_FILE

if [ -f "$MATRIX_JSON" ]; then
    echo "Matrix: $(ls -lh $MATRIX_JSON | awk '{print $5}')" | tee -a $LOG_FILE
fi

if [ -f "$ROUTES_GEOJSON" ]; then
    echo "Routes: $(ls -lh $ROUTES_GEOJSON | awk '{print $5}')" | tee -a $LOG_FILE
fi

echo "SUCCESS" | tee -a $LOG_FILE
