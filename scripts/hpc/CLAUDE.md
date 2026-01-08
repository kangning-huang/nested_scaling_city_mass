# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OSRM (Open Source Routing Machine) processing pipeline for computing travel time matrices and route geometries for 13,135 global cities across 182 countries. The project uses Google Cloud VMs to parallelize processing and supports multi-phase workflows with H3 hexagonal grids for spatial discretization.

## Environment Setup

### Google Cloud SDK
```bash
# Install on macOS
brew install --cask google-cloud-sdk

# Authenticate and configure
gcloud auth login
gcloud config set project ee-knhuang
gcloud config set compute/zone us-central1-c
```

### Python Environment

**Local Machine:**
```bash
# Create virtual environment outside Google Drive to avoid sync issues
python3 -m venv ~/.venvs/nyu_china_grant_env
source ~/.venvs/nyu_china_grant_env/bin/activate

# Install dependencies
pip install geopandas pandas h3 requests shapely
```

**On VM:**
```bash
# Create virtual environment on the VM
python3 -m venv ~/osrm_env
source ~/osrm_env/bin/activate

# Install dependencies
pip install geopandas pandas h3 requests shapely
```

**Note:** Use `~/.venvs/nyu_china_grant_env` for local development and `~/osrm_env` on VMs.

## Key Infrastructure

### VM Configuration
The project uses 6 VMs for parallel processing, each handling different geographic regions:
- `osrm-india` (asia-south1-c) - 3,248 cities
- `osrm-china` (asia-east1-b) - 1,850 cities
- `osrm-asia-other` (asia-southeast1-b) - 2,639 cities
- `osrm-global-small` (europe-west1-b) - 3,967 cities (Africa + Latin America + Oceania)
- `osrm-northam` (us-central1-c) - 372 cities
- `osrm-europe` (europe-west1-b) - 1,059 cities

### VM Management Commands
```bash
# Start/stop a VM
gcloud compute instances start instance-20251223-055023 --zone=us-central1-c
gcloud compute instances stop instance-20251223-055023 --zone=us-central1-c

# SSH with IAP tunneling (always use --tunnel-through-iap for reliability)
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap

# Upload/download files
gcloud compute scp local-file.geojson instance-20251223-055023:~/cities/ --zone=us-central1-c
gcloud compute scp instance-20251223-055023:~/results/*.json ./local-dir/ --zone=us-central1-c

# Run remote commands
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="ls -la ~/cities/"
```

## Processing Pipeline Architecture

### Three-Phase Modular Pipeline

The processing is split into 3 independent phases for maximum flexibility:

```
Phase 1: clip_cities.py     → ~/clipped/{city_id}.osm.pbf
Phase 2: preprocess_cities.py → ~/osrm/{city_id}.tar.gz (compressed) or ~/cities/{city_id}.osrm.* (directory)
Phase 3: route_cities.py    → ~/results/{city_id}_matrix.json, {city_id}_routes.geojson
```

**Why separate phases?**
- Re-run routing with different H3 resolutions without re-preprocessing
- Switch profiles (car/bicycle/foot) by re-running Phase 2 only
- Compress OSRM files (~300MB vs 1.5GB) for storage efficiency
- Resume from failed phase without starting over

### Phase 1: Clipping OSM Data
Extract city-specific OSM data from country/region files using `osmium`.

```bash
python clip_cities.py --region ~/cities.geojson --osm-dir ~/osrm-data --output-dir ~/clipped
```

### Phase 2: OSRM Preprocessing
Run OSRM's three-step preprocessing pipeline (extract, partition, customize) using Docker.

```bash
# With compression (recommended for large cities)
python preprocess_cities.py --clipped-dir ~/clipped --profile car --compress

# Without compression (faster processing in Phase 3)
python preprocess_cities.py --clipped-dir ~/clipped --profile bicycle
```

**OSRM Profiles:**
- `car` - Driving routes (default, uses `/opt/car.lua`)
- `bicycle` - Cycling routes (uses `/opt/bicycle.lua`)
- `foot` - Walking routes (uses `/opt/foot.lua`)

### Phase 3: Routing and Matrix Computation
Generate H3 grids, compute travel time matrices, and optionally fetch route polylines.

```bash
# Basic matrix computation (H3 resolution 7, no polylines)
python route_cities.py --region ~/cities.geojson

# With polylines and custom H3 resolution
python route_cities.py --region ~/cities.geojson --h3-resolution 6 --fetch-polylines

# Cleanup OSRM files after routing (keep compressed)
python route_cities.py --region ~/cities.geojson --cleanup --keep-compressed
```

### All-in-One Processing
Wrapper script that runs all three phases sequentially:

```bash
# Full processing with all options
python process_cities.py --region ~/cities.geojson \
    --profile car --compress \
    --h3-resolution 7 --fetch-polylines \
    --cleanup --keep-compressed

# Skip phases for re-runs
python process_cities.py --region ~/cities.geojson --skip-clip --skip-preprocess
```

## OSRM API Usage

### Table API (Fast Matrix Computation)
Computes full distance/duration matrices in a single request - **58x faster** than individual route queries.

```bash
# Start OSRM server with increased table size limit
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -d -p 5000:5000 -v \"\${PWD}:/data\" osrm/osrm-backend \
    osrm-routed --algorithm mld --max-table-size 500 /data/ny_metro.osrm
"

# Compute matrix (coordinates semicolon-separated: lon1,lat1;lon2,lat2;...)
curl 'http://localhost:5000/table/v1/driving/${COORDS}?annotations=duration,distance'

# Stop server
docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend)
```

**Performance:** 219×219 matrix (47,961 pairs) in 3.3 seconds vs 3.2 minutes with individual queries.

### Route API (Polyline Fetching)
Fetches actual route geometries for visualization. Run as a second pass after matrices are computed.

```bash
# Single route request
curl 'http://localhost:5000/route/v1/driving/-74.048443,40.818747;-74.497560,40.511854?overview=full&geometries=geojson'

# Batch polyline fetching (run on VM after matrices are done)
source ~/osrm_env/bin/activate
nohup python3 ~/fetch_polylines.py >> ~/polylines.log 2>&1 &
```

## OSM Data Management

### Critical Rule: Always Download Country-Specific Files
**Never use continent-level files for clipping** - `osmium extract` scans the entire file for each city:

| OSM File Size | Time per City | 100 Cities |
|---------------|---------------|------------|
| Country (100-500 MB) | ~5 seconds | ~8 minutes |
| Continent (7 GB) | ~3-5 minutes | ~5-8 hours |

**Example:** Using `africa-latest.osm.pbf` (7GB) runs **36x slower** than individual country files.

### Geofabrik Download Patterns
```bash
# Asia - download by country
wget https://download.geofabrik.de/asia/china-latest.osm.pbf
wget https://download.geofabrik.de/asia/india-latest.osm.pbf
wget https://download.geofabrik.de/asia/japan-latest.osm.pbf

# Africa - ALWAYS individual countries
wget https://download.geofabrik.de/africa/ethiopia-latest.osm.pbf
wget https://download.geofabrik.de/africa/nigeria-latest.osm.pbf

# US - by state only
wget https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf

# Merge states for multi-state regions
osmium merge new-york-latest.osm.pbf new-jersey-latest.osm.pbf -o region.osm.pbf --overwrite
```

**URL format:** `https://download.geofabrik.de/{continent}/{country}-latest.osm.pbf`

## H3 Grid System

### H3 Version 4 API (Current)
The project uses H3 v4 - **important API changes** from v3:

```python
# H3 v4 API (use this)
import h3
h3_cells = h3.geo_to_cells(geometry, resolution)
lat, lon = h3.cell_to_latlng(cell)

# Old v3 API (deprecated)
h3_cells = h3.polyfill_geojson(geometry.__geo_interface__, resolution)
lat, lon = h3.h3_to_geo(cell)
```

Check version: `python3 -c "import h3; print(h3.__version__)"`

### Resolution Guidelines
- Resolution 6: ~36.13 km² per hexagon (large cities)
- Resolution 7: ~5.16 km² per hexagon (default, medium cities)
- Resolution 8: ~0.74 km² per hexagon (small cities, high detail)

## Directory Structure on VM

```
~/
├── osrm-data/              # Regional OSM source data
│   └── {country}-latest.osm.pbf
├── cities/                 # City GeoJSON boundaries
│   └── {city_id}.geojson
├── clipped/                # Clipped OSM data (Phase 1)
│   └── {city_id}.osm.pbf
├── osrm/                   # Compressed OSRM files (Phase 2)
│   └── {city_id}.tar.gz
├── cities/                 # OSRM processed files (Phase 2, uncompressed)
│   └── {city_id}.osrm.*
└── results/                # Final outputs (Phase 3)
    ├── {city_id}_matrix.json
    └── {city_id}_routes.geojson
```

## Common Workflows

### Process Single City (New York Metro)
```bash
# 1. Upload city boundary
gcloud compute scp ny_metro.geojson instance-20251223-055023:~/cities/ --zone=us-central1-c

# 2. Download OSM data
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/osrm-data && \
  wget -c https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf && \
  wget -c https://download.geofabrik.de/north-america/us/new-jersey-latest.osm.pbf && \
  wget -c https://download.geofabrik.de/north-america/us/connecticut-latest.osm.pbf && \
  osmium merge *.osm.pbf -o region.osm.pbf --overwrite
"

# 3. Clip to city boundary
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  osmium extract -p cities/ny_metro.geojson osrm-data/region.osm.pbf -o cities/ny_metro.osm.pbf
"

# 4. OSRM preprocessing (3 steps)
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/ny_metro.osm.pbf && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-partition /data/ny_metro.osrm && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-customize /data/ny_metro.osrm
"
```

### Monitor VM Processing
```bash
# Check log
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="tail -50 ~/processing.log"

# Count completed cities
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="ls ~/results/*.json | wc -l"

# Check disk usage
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="df -h"
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `dest open "folder/": Failure` | Create destination folder first with `mkdir -p` |
| `module 'h3' has no attribute 'polyfill_geojson'` | Update to H3 v4 API: use `h3.geo_to_cells()` instead |
| `Too many table coordinates` | Increase `--max-table-size` when starting OSRM server |
| OSRM preprocessing OOM | Use VM with more RAM or process smaller cities first |
| "No edges remaining" | City has no roads in OSM - expected for remote areas |
| Compression "Permission denied" | Run `sudo chown -R $(whoami) ~/osrm/` (Docker creates root files) |
| SSH connection timeout | Always use `--tunnel-through-iap` flag |
| Wrong file downloaded (HTML instead of .osm.pbf) | Verify Geofabrik URL exists - US uses state-level, not regional |

### Performance Notes
- Table API: 219×219 matrix = 3.3 seconds
- Route API: 47,961 polylines = ~4 minutes (8 workers, c2d-highmem-4 VM)
- Clipping: 5 seconds per city with country file, 3-5 minutes with continent file

## Output Formats

### Matrix JSON
```json
{
  "city_id": "945",
  "h3_indices": ["862a13d67ffffff", ...],
  "centroids": [{"h3_index": "...", "lat": 40.38, "lon": -74.31}, ...],
  "durations": [[0, 1037, ...], [1025, 0, ...], ...],  // seconds
  "distances": [[0, 12500, ...], [11900, 0, ...], ...], // meters
  "n_grids": 219
}
```

### Routes GeoJSON
```json
{
  "type": "FeatureCollection",
  "properties": {"city_id": "945", "total_routes": 47961},
  "features": [{
    "type": "Feature",
    "properties": {
      "origin_h3": "862a13d67ffffff",
      "destination_h3": "862a12a6fffffff",
      "duration": 1989.4,
      "distance": 33795.1
    },
    "geometry": {"type": "LineString", "coordinates": [[-74.313, 40.388], ...]}
  }]
}
```

## Local Processing Scripts

### Visualization Preparation
- `process_for_kepler.py` - Calculate accessibility metrics for Kepler.gl visualization
- `process_arcs_for_kepler.py` - Convert matrix to arc format for flow visualization

Both scripts process travel time matrices and prepare data for upload to https://kepler.gl/demo

**Running locally:**
```bash
# Activate the local environment
source ~/.venvs/nyu_china_grant_env/bin/activate

# Run visualization scripts
python process_for_kepler.py
python process_arcs_for_kepler.py
```

## Storage Estimates

| Component | Per City (avg) | 13,135 Cities |
|-----------|----------------|---------------|
| Clipped OSM | ~15 MB | ~200 GB |
| OSRM uncompressed | ~1.5 GB | ~20 TB |
| OSRM compressed | ~300 MB | ~4 TB |
| Matrix JSON | ~3 MB | ~40 GB |
| Routes GeoJSON (simplified) | ~5 MB | ~65 GB |

**Recommended strategy:** Keep clipped OSM and results, delete uncompressed OSRM after routing (or keep compressed for large cities only).

## Cost Estimation

Running all 6 VMs for ~2 days:
- VM compute: $70-80
- SSD storage: $10-15
- Network egress: $5-10
- **Total: ~$85-105**
