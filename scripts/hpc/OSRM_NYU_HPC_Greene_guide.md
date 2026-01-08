# Running OSRM Routing on NYU HPC Greene

This guide explains how to run the OSRM routing codebase on NYU's Greene High Performance Computing cluster (New York).

> **Note**: This guide is adapted from the NYUSH HPC guide. See [Key Differences from NYUSH HPC](#key-differences-from-nyush-hpc) section for a detailed comparison.

## Overview

The NYU HPC Greene cluster uses:
- **SLURM** for job scheduling
- **Apptainer** (formerly Singularity) for containers - globally installed, no module needed
- **Lmod** for software module management
- **GPFS** shared filesystem with `/scratch`, `/vast`, and `/archive` storage

**Key Difference from NYUSH HPC:** Greene uses Apptainer (globally installed) instead of requiring `module load Singularity`. The storage paths and quotas are also different.

### Quick Reference: NYU Greene vs NYUSH HPC

| Aspect | NYU Greene (New York) | NYUSH HPC (Shanghai) |
|--------|----------------------|----------------------|
| **Login Node** | `greene.hpc.nyu.edu` | `hpc.shanghai.nyu.edu` |
| **Home Directory** | `/home/$USER` (50 GB) | `/gpfsnyu/home/$USER` (~50 GB) |
| **Scratch** | `/scratch/$USER` (5 TB, 60-day purge) | `/scratch/$USER` (~1-5 TB, 90-day purge) |
| **Additional Storage** | `/vast/$USER` (2 TB, 60-day purge) | Not available |
| **Archive** | `/archive/X/$USER` (2 TB, permanent) | Unknown |
| **Container System** | Apptainer (global, no module needed) | Singularity (requires module load) |
| **VPN Required** | Yes (Cisco AnyConnect) for off-campus | No (Duo 2FA instead) |
| **Web Portal** | https://ood.hpc.nyu.edu | https://ood.shanghai.nyu.edu/hpc/ |
| **Pre-built Containers** | `/scratch/work/public/singularity/` | None |

---

## Key Differences from NYUSH HPC

### 1. Container System: Apptainer vs Singularity Module

**NYUSH HPC:**
```bash
# Must load module first
module load Singularity/4.3.1-gcc-8.5.0
singularity exec ...
```

**NYU Greene:**
```bash
# Apptainer is globally installed - no module needed!
apptainer exec ...
# Or use 'singularity' alias (still works)
singularity exec ...
```

### 2. Storage Paths

**NYUSH HPC:**
```bash
# Home directory has different path
/gpfsnyu/home/kh3657

# Only scratch available
/scratch/kh3657
```

**NYU Greene:**
```bash
# Standard home path
/home/kh3657

# Multiple storage options
/scratch/kh3657    # 5 TB, 60-day purge
/vast/kh3657       # 2 TB, 60-day purge (optimized for small files)
/archive/k/kh3657  # 2 TB, permanent (login nodes only)
```

### 3. Network Access

**NYUSH HPC:**
- Direct SSH with NYU credentials + Duo 2FA
- No VPN required

**NYU Greene:**
- Requires NYU VPN (Cisco AnyConnect) for off-campus access
- Or use gateway: `ssh netid@gw.hpc.nyu.edu` then `ssh netid@greene.hpc.nyu.edu`

### 4. Pre-built Singularity Images

**NYU Greene Only:**
```bash
# Check available pre-built images
ls /scratch/work/public/singularity/

# May include CUDA, Python, ML frameworks, etc.
```

### 5. Purge Policy

| Cluster | Scratch Purge | Action Needed |
|---------|---------------|---------------|
| NYUSH | 90 days | Less urgent |
| Greene | **60 days** | More frequent backups needed |

---

## 1. Accessing NYU HPC Greene

### VPN Setup (Required for Off-Campus)

1. Install Cisco AnyConnect VPN client
2. Connect to: `vpn.nyu.edu`
3. Authenticate with NYU credentials

### SSH Access

**Option 1: Direct (on NYU network or VPN)**
```bash
ssh kh3657@greene.hpc.nyu.edu
```

**Option 2: Via Gateway (off-campus without VPN)**
```bash
# First connect to gateway
ssh kh3657@gw.hpc.nyu.edu

# Then to Greene
ssh kh3657@greene.hpc.nyu.edu
```

### SSH Config (Recommended)

Add to `~/.ssh/config`:
```
Host greene
    HostName greene.hpc.nyu.edu
    User kh3657
    ForwardAgent yes
    ServerAliveInterval 60

Host hpc-gateway
    HostName gw.hpc.nyu.edu
    User kh3657

Host greene-via-gw
    HostName greene.hpc.nyu.edu
    User kh3657
    ProxyJump hpc-gateway
```

Then connect with:
```bash
ssh greene          # Direct (requires VPN)
ssh greene-via-gw   # Via gateway (no VPN needed)
```

### Web Portal (Open OnDemand)
Access via browser: https://ood.hpc.nyu.edu

### First-Time Setup
```bash
# Check your quotas
myquota

# Create your scratch workspace
mkdir -p /scratch/kh3657/osrm
cd /scratch/kh3657/osrm
```

---

## 2. Transferring Files

### From Local Machine to Greene
```bash
# Direct SCP (requires VPN)
scp -r local_folder/ kh3657@greene.hpc.nyu.edu:/scratch/kh3657/osrm/

# Via Data Transfer Node (faster for large files)
scp -r local_folder/ kh3657@dtn.hpc.nyu.edu:/scratch/kh3657/osrm/
```

### Using rsync (Recommended for Large Transfers)
```bash
rsync -avzP --progress results/ kh3657@greene.hpc.nyu.edu:/scratch/kh3657/osrm/results/
```

### From Google Cloud VMs
```bash
# On your local machine - download from GCP
gcloud compute scp --recurse instance-20251223-055023:~/results/ . --zone=us-central1-c

# Then upload to Greene
scp -r results/ kh3657@greene.hpc.nyu.edu:/scratch/kh3657/osrm/
```

---

## 3. Downloading OSM Data on Greene

### Important: Use Country-Level Files

Same rule as NYUSH HPC - never use continent-level OSM files.

| OSM File Size | Time per City | 100 Cities |
|---------------|---------------|------------|
| Country (100-500 MB) | ~5 seconds | ~8 minutes |
| Continent (7 GB) | ~3-5 minutes | **~5-8 hours** |

### Download Script for Greene

```bash
#!/bin/bash
#SBATCH --job-name=osm_download
#SBATCH --output=/scratch/kh3657/osrm/logs/download_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

cd /scratch/kh3657/osrm/osm-data

# China (1,850 cities)
wget -c https://download.geofabrik.de/asia/china-latest.osm.pbf

# India (3,248 cities)
# wget -c https://download.geofabrik.de/asia/india-latest.osm.pbf

# US States (download individually)
# wget -c https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf

echo "Download complete!"
ls -lh /scratch/kh3657/osrm/osm-data/
```

### Quick Download Commands (Interactive)
```bash
ssh kh3657@greene.hpc.nyu.edu

mkdir -p /scratch/kh3657/osrm/osm-data
cd /scratch/kh3657/osrm/osm-data

# Download China OSM (~1.4 GB)
wget -c https://download.geofabrik.de/asia/china-latest.osm.pbf
```

---

## 4. Setting Up Containers on Greene

### Key Difference: No Module Load Needed!

On Greene, Apptainer is globally installed:

```bash
# Verify Apptainer is available (no module needed!)
apptainer --version

# Or use singularity alias
singularity --version
```

### Pull OSRM Containers

```bash
cd /scratch/kh3657/osrm

# Pull OSRM backend from Docker Hub
apptainer pull docker://osrm/osrm-backend:latest
# Creates: osrm-backend_latest.sif (37 MB)

# Pull osmium-tool for clipping
apptainer pull docker://stefda/osmium-tool:latest
# Creates: osmium-tool_latest.sif (435 MB)

# Verify containers
ls -lh *.sif
```

### Check Pre-built Containers (Greene Only)

Greene may have pre-built images available:
```bash
ls /scratch/work/public/singularity/
```

### Save Containers Permanently

Since `/scratch` has a 60-day purge policy, save containers to home:
```bash
mkdir -p $HOME/osrm-containers
cp /scratch/kh3657/osrm/*.sif $HOME/osrm-containers/

# Restore later with:
cp $HOME/osrm-containers/*.sif /scratch/kh3657/osrm/
```

---

## 5. Full Processing Pipeline on Greene

The processing pipeline has 3 independent phases (same as NYUSH):

```
Phase 1: Clip OSM      → /scratch/kh3657/osrm/clipped/{city_id}.osm.pbf
Phase 2: Preprocess    → /scratch/kh3657/osrm/osrm/{city_id}.osrm*
Phase 3: Route         → /scratch/kh3657/osrm/results/{city_id}_matrix.json
```

### GeoJSON Format for osmium (Same as NYUSH)

osmium requires a GeoJSON Feature, NOT a FeatureCollection:

```json
{
  "type": "Feature",
  "properties": {"name": "Shanghai"},
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[120.85, 30.69], [122.00, 30.69], [122.00, 31.53], [120.85, 31.53], [120.85, 30.69]]]
  }
}
```

### Phase 1: Clip OSM Data

```bash
cd /scratch/kh3657/osrm

# Note: No module load needed on Greene!
apptainer exec -B ${PWD}:/data osmium-tool_latest.sif \
    osmium extract -p /data/cities/shanghai.geojson \
    /data/osm-data/china-latest.osm.pbf \
    -o /data/clipped/shanghai.osm.pbf --overwrite
```

**SLURM Script (clip_city.slurm):**

```bash
#!/bin/bash
#SBATCH --job-name=osrm_clip
#SBATCH --output=/scratch/kh3657/osrm/logs/clip_%j.out
#SBATCH --error=/scratch/kh3657/osrm/logs/clip_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# NO module load needed on Greene!

WORK_DIR=/scratch/kh3657/osrm
OSM_DIR=$WORK_DIR/osm-data
CLIPPED_DIR=$WORK_DIR/clipped
CITIES_DIR=$WORK_DIR/cities
CITY_ID=$1

mkdir -p $CLIPPED_DIR

OSM_FILE=$OSM_DIR/china-latest.osm.pbf

apptainer exec -B $WORK_DIR:/data $WORK_DIR/osmium-tool_latest.sif \
    osmium extract -p /data/cities/${CITY_ID}.geojson $OSM_FILE \
    -o /data/clipped/${CITY_ID}.osm.pbf --overwrite

echo "Clipping complete for $CITY_ID"
ls -lh $CLIPPED_DIR/${CITY_ID}.osm.pbf
```

### Phase 2: OSRM Preprocessing

```bash
cd /scratch/kh3657/osrm/osrm-files/shanghai

# Copy clipped OSM
cp /scratch/kh3657/osrm/clipped/shanghai.osm.pbf .

# Step 1: Extract
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-extract -p /opt/car.lua /data/shanghai.osm.pbf

# Step 2: Partition
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-partition /data/shanghai.osrm

# Step 3: Customize
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-customize /data/shanghai.osrm

# Cleanup
rm -f shanghai.osm.pbf
```

**SLURM Script (preprocess_city.slurm):**

```bash
#!/bin/bash
#SBATCH --job-name=osrm_preprocess
#SBATCH --output=/scratch/kh3657/osrm/logs/preprocess_%j.out
#SBATCH --error=/scratch/kh3657/osrm/logs/preprocess_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# NO module load needed on Greene!

WORK_DIR=/scratch/kh3657/osrm
OSRM_SIF=$WORK_DIR/osrm-backend_latest.sif
CLIPPED_DIR=$WORK_DIR/clipped
OSRM_DIR=$WORK_DIR/osrm-files
CITY_ID=$1
PROFILE=${2:-car}

mkdir -p $OSRM_DIR/$CITY_ID
cd $OSRM_DIR/$CITY_ID

cp $CLIPPED_DIR/${CITY_ID}.osm.pbf .

echo "Extracting with $PROFILE profile..."
apptainer exec -B ${PWD}:/data $OSRM_SIF \
    osrm-extract -p /opt/${PROFILE}.lua /data/${CITY_ID}.osm.pbf

echo "Partitioning..."
apptainer exec -B ${PWD}:/data $OSRM_SIF \
    osrm-partition /data/${CITY_ID}.osrm

echo "Customizing..."
apptainer exec -B ${PWD}:/data $OSRM_SIF \
    osrm-customize /data/${CITY_ID}.osrm

rm -f ${CITY_ID}.osm.pbf

echo "Preprocessing complete for $CITY_ID"
ls -lh $OSRM_DIR/$CITY_ID/
```

### Phase 3: Route Computation

```bash
#!/bin/bash
#SBATCH --job-name=osrm_route
#SBATCH --output=/scratch/kh3657/osrm/logs/route_%j.out
#SBATCH --error=/scratch/kh3657/osrm/logs/route_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load Python module (Greene uses Lmod)
module load python/intel/3.8.6

WORK_DIR=/scratch/kh3657/osrm
OSRM_SIF=$WORK_DIR/osrm-backend_latest.sif
OSRM_DIR=$WORK_DIR/osrm-files
RESULTS_DIR=$WORK_DIR/results
CITY_ID=$1

mkdir -p $RESULTS_DIR
cd $OSRM_DIR/$CITY_ID

# Start OSRM server
echo "Starting OSRM server for $CITY_ID..."
apptainer exec -B ${PWD}:/data $OSRM_SIF \
    osrm-routed --algorithm mld --max-table-size 2000 /data/${CITY_ID}.osrm &
OSRM_PID=$!
sleep 15

# Verify server is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "ERROR: OSRM server failed to start"
    kill $OSRM_PID 2>/dev/null
    exit 1
fi

# Run routing
source ~/osrm_venv/bin/activate
python $WORK_DIR/scripts/route_cities_hpc.py \
    --city-id $CITY_ID \
    --region $WORK_DIR/cities/cities.geojson \
    --results-dir $RESULTS_DIR

kill $OSRM_PID 2>/dev/null

echo "Routing complete for $CITY_ID"
```

### Job Array for Parallel Processing

```bash
#!/bin/bash
#SBATCH --job-name=osrm_array
#SBATCH --output=/scratch/kh3657/osrm/logs/array_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/array_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-100%10  # Process 100 cities, max 10 concurrent

# Get city ID from array index
CITY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/kh3657/osrm/city_ids.txt)
echo "Processing city $CITY_ID (task $SLURM_ARRAY_TASK_ID)"

# Run full pipeline...
```

---

## 6. Quick Start Test (Shanghai Example)

### Step 1: Setup Environment
```bash
# SSH to Greene (requires VPN or gateway)
ssh kh3657@greene.hpc.nyu.edu

# Create workspace
mkdir -p /scratch/kh3657/osrm/{cities,results,logs,osm-data,clipped,osrm-files}
cd /scratch/kh3657/osrm

# Pull containers (NO module load needed!)
apptainer pull docker://osrm/osrm-backend:latest
apptainer pull docker://stefda/osmium-tool:latest

# Save to home for future use
mkdir -p $HOME/osrm-containers
cp *.sif $HOME/osrm-containers/
```

### Step 2: Download OSM and Create Boundary
```bash
# Download China OSM (~1.4 GB)
cd /scratch/kh3657/osrm/osm-data
wget -c https://download.geofabrik.de/asia/china-latest.osm.pbf

# Create Shanghai boundary
cat > /scratch/kh3657/osrm/cities/shanghai.geojson << 'EOF'
{
  "type": "Feature",
  "properties": {"name": "Shanghai", "city_id": "shanghai"},
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[120.85, 30.69], [122.00, 30.69], [122.00, 31.53], [120.85, 31.53], [120.85, 30.69]]]
  }
}
EOF
```

### Step 3: Full Pipeline Test
```bash
cd /scratch/kh3657/osrm

# Phase 1: Clip Shanghai (~5 seconds)
apptainer exec -B ${PWD}:/data osmium-tool_latest.sif \
    osmium extract -p /data/cities/shanghai.geojson \
    /data/osm-data/china-latest.osm.pbf \
    -o /data/clipped/shanghai.osm.pbf --overwrite

# Phase 2: OSRM Preprocessing (~15 seconds)
mkdir -p osrm-files/shanghai && cd osrm-files/shanghai
cp /scratch/kh3657/osrm/clipped/shanghai.osm.pbf .

apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-extract -p /opt/car.lua /data/shanghai.osm.pbf
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-partition /data/shanghai.osrm
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-customize /data/shanghai.osrm

rm -f shanghai.osm.pbf
ls -lh

# Phase 3: Test Routing Server
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-routed --algorithm mld --max-table-size 500 /data/shanghai.osrm &
sleep 10

# Test route: Pudong Airport to The Bund
curl -s "http://localhost:5000/route/v1/driving/121.799,31.143;121.490,31.234"

# Stop server
pkill -f osrm-routed
```

---

## 7. Directory Structure on Greene

```
/scratch/kh3657/osrm/                    # Working directory
├── osrm-backend_latest.sif              # OSRM container (37 MB)
├── osmium-tool_latest.sif               # Osmium container (435 MB)
├── osm-data/                            # Raw OSM files
│   └── china-latest.osm.pbf
├── cities/                              # City boundary GeoJSONs
│   └── shanghai.geojson
├── clipped/                             # Clipped city OSM files
│   └── shanghai.osm.pbf
├── osrm-files/                          # OSRM processed files
│   └── shanghai/
├── results/                             # Output matrices and routes
├── scripts/                             # Python scripts
└── logs/                                # SLURM job logs

/home/kh3657/                            # Permanent storage (50 GB)
├── osrm-containers/                     # Containers backup
│   ├── osrm-backend_latest.sif
│   └── osmium-tool_latest.sif
└── osrm-cities/                         # City boundaries backup
```

---

## 8. Useful SLURM Commands

```bash
# Submit a job
sbatch preprocess_city.slurm shanghai

# Check job status
squeue -u kh3657

# Cancel a job
scancel <job_id>

# View job output in real-time
tail -f /scratch/kh3657/osrm/logs/preprocess_12345.out

# Check cluster status
sinfo

# Check your quotas
myquota

# View past job history
sacct -u kh3657 --starttime=2025-01-01
```

---

## 9. Troubleshooting

### Greene-Specific Issues

| Issue | Solution |
|-------|----------|
| `apptainer: command not found` | Should not happen - Apptainer is globally installed. Check `which apptainer` |
| VPN connection issues | Use gateway method: `ssh gw.hpc.nyu.edu` then `ssh greene.hpc.nyu.edu` |
| `/scratch` files disappeared | 60-day purge policy - save important files to `/home` or `/archive` |
| Home directory full | Only 50 GB - use `/scratch` for large files |

### Common Issues (Same as NYUSH)

| Issue | Solution |
|-------|----------|
| `Expected 'type' value to be 'Feature'` | GeoJSON must be Feature, not FeatureCollection |
| `Out of memory` | Increase `--mem` in SLURM script |
| `OSRM server not responding` | Increase sleep time to 15-20s |
| `Too many table coordinates` | Increase `--max-table-size` parameter |
| `No edges remaining` in OSRM | City has no roads in OSM - expected for small areas |
| City skipped / no results | City has <5 H3 grids - too small for routing (see Section 12) |
| **All zeros in matrix** (parallel jobs) | Port conflict - use unique port per array task: `PORT=$((5000 + SLURM_ARRAY_TASK_ID))` |
| `No grids for city` / H3 fails | H3 v4 API change - see below |
| Compression "Permission denied" | Run `chmod -R u+w` on the directory |

### H3 Library v4 API Changes

```python
# H3 v4 API (use this)
h3_cells = h3.geo_to_cells(geometry, resolution)
lat, lon = h3.cell_to_latlng(cell)
```

---

## 10. Storage Best Practices on Greene

### Storage Tiers

| Location | Path | Quota | Purge | Use For |
|----------|------|-------|-------|---------|
| Home | `/home/kh3657` | 50 GB | Never | Containers, scripts, config |
| Scratch | `/scratch/kh3657` | 5 TB | **60 days** | OSM data, processing |
| Vast | `/vast/kh3657` | 2 TB | 60 days | Small files (many inodes) |
| Archive | `/archive/k/kh3657` | 2 TB | Never | Long-term storage |

### Recommended Strategy

```bash
# Save permanently (in /home)
$HOME/osrm-containers/     # 472 MB - Singularity containers
$HOME/osrm-cities/         # ~50 MB - City boundaries
$HOME/osrm-results/        # ~4.4 GB - Final results (or use archive)

# Temporary processing (in /scratch)
/scratch/kh3657/osrm/      # OSM data, OSRM files, logs
```

### Quick Restore Script

```bash
#!/bin/bash
# restore_osrm_setup.sh

SCRATCH=/scratch/kh3657/osrm

mkdir -p $SCRATCH/{osm-data,cities,clipped,osrm-files,results,logs}

# Copy containers from home
cp $HOME/osrm-containers/*.sif $SCRATCH/

# Copy city boundaries
cp $HOME/osrm-cities/*.geojson $SCRATCH/cities/

# Download fresh OSM data
cd $SCRATCH/osm-data
wget -c https://download.geofabrik.de/asia/china-latest.osm.pbf

echo "Setup restored!"
```

---

## 11. Command Comparison: Greene vs NYUSH

### Container Commands

**NYUSH HPC:**
```bash
module load Singularity/4.3.1-gcc-8.5.0
singularity exec -B ${PWD}:/data container.sif command
```

**NYU Greene:**
```bash
# No module load needed!
apptainer exec -B ${PWD}:/data container.sif command
# Or use singularity alias:
singularity exec -B ${PWD}:/data container.sif command
```

### Storage Paths

**NYUSH HPC:**
```bash
$HOME → /gpfsnyu/home/kh3657
```

**NYU Greene:**
```bash
$HOME → /home/kh3657
```

### SLURM Scripts

**NYUSH HPC:**
```bash
#SBATCH --partition=acc  # Must specify acc, not parallel
module load Singularity/4.3.1-gcc-8.5.0
singularity exec ...
```

**NYU Greene:**
```bash
# Default partition works, no module load needed
apptainer exec ...
```

---

## 12. Full City Matrix Computation with H3 Grids

This section demonstrates computing a complete travel time matrix using H3 grid centroids.

### Minimum Grid Requirement

**Cities with fewer than 5 H3 grids are skipped** during routing. This threshold ensures:
- Meaningful travel time matrices (at least 20 O-D pairs)
- Sufficient spatial coverage for centrality analysis
- Avoidance of trivial results from very small cities

| Grids | Routes (n×(n-1)) | Status |
|-------|------------------|--------|
| 1 | 0 | ❌ Skip - cannot route |
| 2 | 2 | ❌ Skip - too few pairs |
| 3 | 6 | ❌ Skip - too few pairs |
| 4 | 12 | ❌ Skip - too few pairs |
| **5** | **20** | ✅ Minimum threshold |
| 10 | 90 | ✅ OK |
| 50 | 2,450 | ✅ OK |
| 164 | 26,732 | ✅ OK (Shanghai) |

At H3 resolution 6 (~36 km² per hexagon), cities smaller than ~150 km² typically have fewer than 5 grids.

### Step 1: Extract Centroids from GeoPackage

```python
#!/usr/bin/env python3
"""Extract H3 grid centroids for a city from all_cities_h3_grids.gpkg"""

import geopandas as gpd
import json

# Read GeoPackage
gdf = gpd.read_file("all_cities_h3_grids.gpkg")

# Minimum grid threshold
MIN_GRIDS = 5

# Extract city by ID (e.g., Shanghai = 12400)
city_id = 12400
city_grids = gdf[gdf['ID_HDC_G0'] == city_id].copy()

# Check minimum grid requirement
if len(city_grids) < MIN_GRIDS:
    print(f"SKIP: City {city_id} has only {len(city_grids)} grids (minimum: {MIN_GRIDS})")
    exit(0)

# Get centroids
city_grids['lon'] = city_grids.geometry.centroid.x
city_grids['lat'] = city_grids.geometry.centroid.y

# Create coords string for OSRM Table API (semicolon-separated)
coords_list = [f"{row['lon']:.6f},{row['lat']:.6f}" for _, row in city_grids.iterrows()]
coords_str = ";".join(coords_list)

# Save for HPC
with open(f"{city_id}_coords.txt", "w") as f:
    f.write(coords_str)

# Save centroids JSON for polyline fetching
centroids_data = {
    "city_id": str(city_id),
    "city_name": city_grids['UC_NM_MN'].iloc[0],
    "n_grids": len(city_grids),
    "centroids": [
        {"h3_index": row['h3index'], "lat": row['lat'], "lon": row['lon']}
        for _, row in city_grids.iterrows()
    ]
}
with open(f"{city_id}_centroids.json", "w") as f:
    json.dump(centroids_data, f, indent=2)

print(f"City: {centroids_data['city_name']}")
print(f"Grids: {len(city_grids)}")
print(f"Saved: {city_id}_coords.txt, {city_id}_centroids.json")
```

### Step 2: Upload to HPC

```bash
# From local machine
scp 12400_coords.txt 12400_centroids.json kh3657@greene.hpc.nyu.edu:/scratch/kh3657/osrm/cities/
```

### Step 3: Compute Matrix on HPC

```bash
# SSH to Greene
ssh kh3657@greene.hpc.nyu.edu

cd /scratch/kh3657/osrm/osrm-files/shanghai

# Start OSRM server (set max-table-size >= number of grids)
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-routed --algorithm mld --max-table-size 200 /data/shanghai.osrm &
sleep 20

# Read coords and compute matrix
COORDS=$(cat /scratch/kh3657/osrm/cities/12400_coords.txt)
curl -s "http://localhost:5000/table/v1/driving/${COORDS}?annotations=duration,distance" \
    -o /scratch/kh3657/osrm/results/12400_matrix.json

# Verify results
python3 -c "
import json
with open('/scratch/kh3657/osrm/results/12400_matrix.json') as f:
    d = json.load(f)
n = len(d['durations'])
nulls = sum(1 for row in d['durations'] for v in row if v is None)
print(f'Matrix: {n}x{n} ({n*n:,} pairs)')
print(f'Null values: {nulls} ({100*nulls/(n*n):.1f}%)')
"

# Stop server
pkill -f osrm-routed
```

### Output Format (matrix JSON)

```json
{
  "code": "Ok",
  "durations": [[0, 1234.5, ...], [1200.3, 0, ...], ...],
  "distances": [[0, 25000, ...], [24500, 0, ...], ...],
  "sources": [{"location": [121.64, 30.89], "name": "..."}],
  "destinations": [{"location": [121.64, 30.89], "name": "..."}]
}
```

---

## 13. Route Polyline Fetching

After computing the travel time matrix, fetch actual route geometries for visualization.

### Why Separate from Matrix Computation?

| API | Purpose | Speed | Output |
|-----|---------|-------|--------|
| **Table API** | Duration/distance matrix | ~1.6 sec for 164×164 | Numbers only |
| **Route API** | Full route geometries | ~154 sec for 26,732 routes | LineString GeoJSON |

### Polyline Fetching Script

Save as `/scratch/kh3657/osrm/scripts/fetch_polylines_hpc.py`:

```python
#!/usr/bin/env python3
"""
Fetch route polylines for all O-D pairs.
Uses Douglas-Peucker simplification for ~90% size reduction.
"""

import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time

MAX_WORKERS = 8
SIMPLIFY_TOLERANCE = 0.0001  # ~11 meters

def simplify_coords(coords, tolerance=SIMPLIFY_TOLERANCE):
    """Douglas-Peucker line simplification."""
    if len(coords) <= 2:
        return coords

    start, end = coords[0], coords[-1]
    max_dist = 0
    max_idx = 0

    for i in range(1, len(coords) - 1):
        x, y = coords[i]
        x1, y1 = start
        x2, y2 = end
        num = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
        den = ((y2-y1)**2 + (x2-x1)**2)**0.5
        dist = num / den if den > 0 else 0
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > tolerance:
        left = simplify_coords(coords[:max_idx+1], tolerance)
        right = simplify_coords(coords[max_idx:], tolerance)
        return left[:-1] + right
    return [start, end]

def fetch_route(origin, destination, session):
    """Fetch single route with geometry."""
    url = f"http://localhost:5000/route/v1/driving/{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}?overview=full&geometries=geojson"
    try:
        resp = session.get(url, timeout=30)
        data = resp.json()
        if data.get('code') == 'Ok' and data.get('routes'):
            route = data['routes'][0]
            coords = simplify_coords(route['geometry']['coordinates'])
            return {
                'origin_h3': origin['h3_index'],
                'destination_h3': destination['h3_index'],
                'duration': route['duration'],
                'distance': route['distance'],
                'geometry': {'type': 'LineString', 'coordinates': [[round(x,5), round(y,5)] for x,y in coords]}
            }
    except:
        pass
    return None

def main(centroids_file, output_file):
    with open(centroids_file) as f:
        data = json.load(f)

    centroids = data['centroids']
    n = len(centroids)
    total = n * (n - 1)

    print(f"City: {data.get('city_name', data['city_id'])}")
    print(f"Grids: {n}, Routes: {total}")

    pairs = [(centroids[i], centroids[j]) for i in range(n) for j in range(n) if i != j]
    features = []
    session = requests.Session()
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_route, o, d, session): (o, d) for o, d in pairs}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                features.append({'type': 'Feature', 'properties': {
                    'origin_h3': result['origin_h3'],
                    'destination_h3': result['destination_h3'],
                    'duration': result['duration'],
                    'distance': result['distance']
                }, 'geometry': result['geometry']})
            if (i+1) % 5000 == 0:
                print(f"  Progress: {i+1}/{total} ({100*(i+1)/total:.0f}%)")

    elapsed = time.time() - start
    geojson = {
        'type': 'FeatureCollection',
        'properties': {'city_id': data['city_id'], 'total_routes': len(features), 'failed_routes': total-len(features), 'processing_time_sec': round(elapsed,1)},
        'features': features
    }
    with open(output_file, 'w') as f:
        json.dump(geojson, f)

    print(f"Complete: {len(features)} routes in {elapsed:.1f}s ({len(features)/elapsed:.0f}/sec)")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
```

### Running Polyline Fetching

```bash
# On Greene - ensure OSRM server is running first
cd /scratch/kh3657/osrm/osrm-files/shanghai

# Start server
apptainer exec -B ${PWD}:/data /scratch/kh3657/osrm/osrm-backend_latest.sif \
    osrm-routed --algorithm mld --max-table-size 200 /data/shanghai.osrm &
sleep 20

# Run polyline fetching
python3 /scratch/kh3657/osrm/scripts/fetch_polylines_hpc.py \
    /scratch/kh3657/osrm/cities/12400_centroids.json \
    /scratch/kh3657/osrm/results/12400_routes.geojson

# Stop server
pkill -f osrm-routed
```

### SLURM Array Job for Polyline Fetching

```bash
#!/bin/bash
#SBATCH --job-name=osrm_polylines
#SBATCH --output=/scratch/kh3657/osrm/logs/polylines_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/polylines_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-100%10

# NO module load needed on Greene!

WORK_DIR=/scratch/kh3657/osrm
OSRM_SIF=$WORK_DIR/osrm-backend_latest.sif

# IMPORTANT: Use unique port per task to avoid conflicts
PORT=$((5000 + SLURM_ARRAY_TASK_ID))

CITY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $WORK_DIR/city_ids.txt)
echo "Fetching polylines for city $CITY_ID on port $PORT"

# Skip if done
if [ -f "$WORK_DIR/results/${CITY_ID}_routes.geojson" ]; then
    echo "SKIP: Routes already exist"
    exit 0
fi

# Check OSRM files
OSRM_DIR=$WORK_DIR/osrm-files/$CITY_ID
if [ ! -f "$OSRM_DIR/${CITY_ID}.osrm" ]; then
    echo "ERROR: OSRM files not found"
    exit 1
fi

cd $OSRM_DIR

# Start OSRM server on unique port
apptainer exec -B ${PWD}:/data $OSRM_SIF \
    osrm-routed --port $PORT --algorithm mld /data/${CITY_ID}.osrm &
OSRM_PID=$!
sleep 20

# Verify server
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: OSRM server failed"
    kill $OSRM_PID 2>/dev/null
    exit 1
fi

# Fetch polylines
source ~/osrm_venv/bin/activate
PORT=$PORT python3 $WORK_DIR/scripts/fetch_polylines_hpc.py \
    --city-id $CITY_ID \
    --results-dir $WORK_DIR/results \
    --port $PORT
deactivate

kill $OSRM_PID 2>/dev/null
```

### Performance Comparison

| City Size | Grids | Routes | Matrix Time | Polyline Time |
|-----------|-------|--------|-------------|---------------|
| Small | 50 | 2,450 | <0.5 sec | ~15 sec |
| Medium (Shanghai) | 164 | 26,732 | 1.6 sec | 154 sec |
| Large | 300 | 89,700 | ~3 sec | ~9 min |

---

## 14. Grid Centrality Calculation

Calculate how central each grid is based on route intersections.

### Centrality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `basic_centrality` | Count of routes through grid | Highway hub |
| `betweenness` | basic / max_basic | Normalized (0-1) |
| `weighted_centrality` | Σ √(origin_pop × dest_pop) | Population flow hub |
| `weighted_centrality_normalized` | weighted / max_weighted | Normalized (0-1) |

**Why geometric mean?** Using √(origin_pop × dest_pop) gives balanced weight to both endpoints, avoiding bias toward high-population origins or destinations.

### Population Data Required

Population data must be in CSV format with columns:
- `h3index`: H3 cell identifier
- `ID_HDC_G0`: City ID
- `population_2015`: Population count

### SLURM Array Job for Centrality

```bash
#!/bin/bash
#SBATCH --job-name=osrm_centrality
#SBATCH --output=/scratch/kh3657/osrm/logs/centrality_%A_%a.out
#SBATCH --error=/scratch/kh3657/osrm/logs/centrality_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-100%20

WORK_DIR=/scratch/kh3657/osrm
CITY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $WORK_DIR/city_ids.txt)

echo "Calculating centrality for city $CITY_ID"

# Skip if done
if [ -f "$WORK_DIR/results/${CITY_ID}_centrality.geojson" ]; then
    echo "SKIP: Centrality already exists"
    exit 0
fi

source ~/osrm_venv/bin/activate

python3 $WORK_DIR/scripts/calculate_centrality_hpc.py \
    --city-id $CITY_ID \
    --results-dir $WORK_DIR/results \
    --cities-dir $WORK_DIR/cities \
    --population-file $WORK_DIR/data/Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-22.csv \
    --h3-resolution 6

deactivate
```

### Output Format (centrality GeoJSON)

```json
{
  "type": "FeatureCollection",
  "properties": {
    "city_id": "11549",
    "n_grids": 25,
    "total_routes": 600,
    "total_population": 5279530,
    "max_basic_centrality": 218,
    "max_weighted_centrality": 37227641.23
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "h3_index": "86d0c3a67ffffff",
        "lat": 30.567,
        "lon": 114.234,
        "population": 125000,
        "basic_centrality": 218,
        "betweenness": 1.0,
        "weighted_centrality": 37227641.23,
        "weighted_centrality_normalized": 1.0
      },
      "geometry": { "type": "Polygon", "coordinates": [...] }
    }
  ]
}
```

### Visualization in Kepler.gl

1. Upload `{city_id}_centrality.geojson`
2. Color hexagons by:
   - `betweenness` → Highway/road hubs
   - `weighted_centrality_normalized` → Population flow hubs
3. Compare patterns to identify areas with high infrastructure but low population flow, or vice versa

---

## 15. Contact & Support

- **NYU HPC Support:** hpc@nyu.edu
- **HPC Documentation:** https://sites.google.com/nyu.edu/nyu-hpc
- **Open OnDemand:** https://ood.hpc.nyu.edu

---

*Last updated: 2025-12-27*
*Adapted from NYUSH HPC guide for NYU Greene cluster*
*NetID: kh3657*
*Updated: Added H3 grid matrix computation, polyline fetching, and centrality calculation sections*
*Updated: Added minimum 5-grid threshold for routing (skip cities too small for meaningful analysis)*

---

## Appendix A: Full Comparison Table

| Feature | NYU Greene (New York) | NYUSH HPC (Shanghai) |
|---------|----------------------|----------------------|
| **Cluster Name** | Greene | Unnamed |
| **Login Node** | greene.hpc.nyu.edu | hpc.shanghai.nyu.edu |
| **Gateway** | gw.hpc.nyu.edu | None needed |
| **VPN Required** | Yes (Cisco AnyConnect) | No (Duo 2FA) |
| **Web Portal** | ood.hpc.nyu.edu | ood.shanghai.nyu.edu |
| **Container System** | Apptainer (global) | Singularity (module) |
| **Container Command** | `apptainer` or `singularity` | `singularity` |
| **Module Load** | Not needed | `module load Singularity/4.3.1-gcc-8.5.0` |
| **Home Path** | `/home/$USER` | `/gpfsnyu/home/$USER` |
| **Home Quota** | 50 GB | ~50 GB |
| **Scratch Path** | `/scratch/$USER` | `/scratch/$USER` |
| **Scratch Quota** | 5 TB | ~1-5 TB |
| **Scratch Purge** | 60 days | ~90 days |
| **Vast Storage** | `/vast/$USER` (2 TB) | Not available |
| **Archive** | `/archive/X/$USER` (2 TB) | Unknown |
| **Pre-built Images** | `/scratch/work/public/singularity/` | None |
| **Partition Issues** | None known | `parallel` has broken Singularity |
| **Python Module** | `module load python/intel/3.8.6` | `module load python/3.10` |

---

## Appendix B: Scale Estimates for Global Processing

Based on Shanghai test (verified) and `all_cities.gpkg` analysis:

### Storage Estimates by Country (Top 10)

| Country | Cities | OSRM Size | Results Size |
|---------|--------|-----------|--------------|
| China | 1,850 | 15.5 GB | 0.6 GB |
| India | 3,248 | 14.2 GB | 1.0 GB |
| United States | 324 | 8.0 GB | 0.1 GB |
| Indonesia | 393 | 3.6 GB | 0.1 GB |
| Brazil | 349 | 3.0 GB | 0.1 GB |
| Bangladesh | 301 | 2.8 GB | 0.1 GB |
| Nigeria | 483 | 2.3 GB | 0.1 GB |
| Russia | 209 | 2.3 GB | 0.1 GB |
| Japan | 109 | 2.0 GB | 0.03 GB |
| Mexico | 168 | 1.7 GB | 0.05 GB |

### Total Storage Summary

| Component | All 13,135 Cities |
|-----------|-------------------|
| Clipped OSM | ~9 GB |
| OSRM processed files | ~92 GB |
| Results (matrices) | ~4.4 GB |
| Containers | 0.5 GB |
| **Total working space** | **~106 GB** |
| **Permanent storage needed** | **~5 GB** |

### Processing Time Estimates

Based on Shanghai (~15 seconds for preprocessing):
- Average city (smaller than Shanghai): ~5-10 seconds
- With SLURM parallelism (10 concurrent jobs): ~3-4 hours for all cities
- Sequential processing: ~20-30 hours

**Note**: HPC performance depends on cluster load and job scheduling. Use job arrays for efficient parallel processing.

### Storage Estimates per Phase

| Component | Per City (avg) | 13,135 Cities | Keep Permanently? |
|-----------|----------------|---------------|-------------------|
| Clipped OSM | ~0.7 MB | ~9 GB | No - regenerate |
| OSRM files | ~7 MB | ~92 GB | **Yes, if re-routing needed** |
| Results (matrices) | ~0.3 MB | ~4.4 GB | **Yes - research output** |
| Containers | - | 0.5 GB | **Yes - in /home** |

**Shanghai reference (large city):**
- Clipped OSM: 26 MB
- OSRM files: 268 MB
- Processing time: ~15 seconds
