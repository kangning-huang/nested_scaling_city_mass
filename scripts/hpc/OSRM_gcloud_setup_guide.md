# Google Cloud VM Setup Guide for OSRM Processing

## Overview
This guide documents the setup and workflow for processing OSM data on Google Cloud VM for OSRM routing analysis.

---

## 1. Install Google Cloud SDK

```bash
# Using Homebrew (recommended for macOS)
brew install --cask google-cloud-sdk
```

After installation, verify:
```bash
gcloud --version
```

---

## 2. Authentication & Configuration

### Login
```bash
gcloud auth login
```
This opens a browser for Google account authentication.

### Set Project and Zone
```bash
gcloud config set project ee-knhuang
gcloud config set compute/zone us-central1-c
```

### Verify Configuration
```bash
gcloud config list
```

**Current Configuration:**
| Setting | Value |
|---------|-------|
| Account | fytaso@gmail.com |
| Project | ee-knhuang |
| Zone | us-central1-c |

---

## 3. VM Management

### Start VM
```bash
gcloud compute instances start instance-20251223-055023 --zone=us-central1-c
```

### Stop VM
```bash
gcloud compute instances stop instance-20251223-055023 --zone=us-central1-c
```

### Check Status
```bash
gcloud compute instances list --filter="name=instance-20251223-055023"
```

---

## 4. VM Connection & File Transfer

### SSH into VM
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap
```

**Note:** Use `--tunnel-through-iap` for reliable connections. First connection auto-generates SSH keys at `~/.ssh/google_compute_engine`.

### Upload Files to VM
```bash
# Create destination folder first (required!)
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="mkdir -p ~/cities"

# Then upload
gcloud compute scp /path/to/local/file.geojson instance-20251223-055023:~/cities/ --zone=us-central1-c
```

**What didn't work:**
- Uploading to a non-existent folder fails silently with `dest open "cities/": Failure`
- Always create the destination folder first!

### Run Commands on VM
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="ls -la ~/cities/"
```

---

## 5. Download OSM Data from Geofabrik

### Working Approach: Download by State
Geofabrik provides US data by state, not by region like "Northeast".

```bash
# Download individual states
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  mkdir -p ~/osrm-data && cd ~/osrm-data && \
  wget -c https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf && \
  wget -c https://download.geofabrik.de/north-america/us/new-jersey-latest.osm.pbf && \
  wget -c https://download.geofabrik.de/north-america/us/connecticut-latest.osm.pbf
"
```

### Merge State Files
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/osrm-data && \
  osmium merge new-york-latest.osm.pbf new-jersey-latest.osm.pbf connecticut-latest.osm.pbf -o region.osm.pbf && \
  rm -f new-york-latest.osm.pbf new-jersey-latest.osm.pbf connecticut-latest.osm.pbf
"
```

**What didn't work:**
- URL `https://download.geofabrik.de/north-america/us/northeast-latest.osm.pbf` does NOT exist
- This redirects to HTML homepage, creating a tiny invalid file
- US data is only available by state on Geofabrik

### Geofabrik US State URLs
Format: `https://download.geofabrik.de/north-america/us/{state-name}-latest.osm.pbf`

Examples:
- `new-york-latest.osm.pbf` (~462 MB)
- `new-jersey-latest.osm.pbf` (~150 MB)
- `connecticut-latest.osm.pbf` (~56 MB)

---

## 6. Extract OSM Data with Osmium

### Clip OSM data using GeoJSON boundary
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  osmium extract -p cities/ny_metro.geojson osrm-data/region.osm.pbf -o cities/ny_metro.osm.pbf
"
```

**What didn't work:**
- `osmium merge ... -o region.osm.pbf` fails if file exists
- Use `--overwrite` flag or delete existing file first

---

## 7. OSRM Preprocessing with Docker

OSRM preprocessing uses the MLD (Multi-Level Dijkstra) algorithm with three steps:

### Step 1: Extract (using car.lua profile for driving)
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/ny_metro.osm.pbf
"
```

### Step 2: Partition
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-partition /data/ny_metro.osrm
"
```

### Step 3: Customize
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-customize /data/ny_metro.osrm
"
```

### Available Profiles
| Profile | File | Use Case |
|---------|------|----------|
| Driving | `/opt/car.lua` | Car routing (default) |
| Cycling | `/opt/bicycle.lua` | Bike routing |
| Walking | `/opt/foot.lua` | Pedestrian routing |

---

## 8. Running OSRM Routing Server

### Start Server (Basic)
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -d -p 5000:5000 -v \"\${PWD}:/data\" osrm/osrm-backend osrm-routed --algorithm mld /data/ny_metro.osrm
"
```

**Note:** Use `-d` flag to run in background (detached mode).

### Start Server with Higher Table Size Limit
Default table size limit is 100 coordinates. For larger matrices, increase the limit:

```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -d -p 5000:5000 -v \"\${PWD}:/data\" osrm/osrm-backend osrm-routed --algorithm mld --max-table-size 500 /data/ny_metro.osrm
"
```

### Stop Server
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  docker stop \$(docker ps -q --filter ancestor=osrm/osrm-backend)
"
```

### Check Server Status
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="docker ps"
```

### Test Single Route
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  curl -s 'http://localhost:5000/route/v1/driving/-74.048443,40.818747;-74.497560,40.511854?overview=false'
"
```

---

## 9. Computing Travel Time Matrix (Table API)

The Table API computes a full distance/duration matrix in a single request - much faster than individual route queries.

### API Format
```
GET http://localhost:5000/table/v1/driving/{coordinates}?annotations=duration,distance
```

Where `{coordinates}` is semicolon-separated: `lon1,lat1;lon2,lat2;lon3,lat3;...`

### Example: Full Matrix for 219 Grid Centroids
```bash
COORDS="lon1,lat1;lon2,lat2;..."  # semicolon-separated coordinates

gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  curl -s 'http://localhost:5000/table/v1/driving/${COORDS}?annotations=duration,distance' -o /tmp/matrix.json
"
```

### Performance Comparison
| Method | 219×219 Matrix (47,961 pairs) | Speedup |
|--------|-------------------------------|---------|
| Individual route queries | ~3.2 minutes | 1× |
| **Table API (single request)** | **3.3 seconds** | **~58×** |

### Output Format
```json
{
  "code": "Ok",
  "durations": [[0, 1037, ...], [1025, 0, ...], ...],  // seconds
  "distances": [[0, 12500, ...], [11900, 0, ...], ...], // meters
  "sources": [...],
  "destinations": [...]
}
```

### What Didn't Work
- **"Too many table coordinates" error**: Default limit is 100 coordinates
- **Solution**: Restart server with `--max-table-size 500` (or higher)

### Download Matrix from VM
```bash
gcloud compute scp instance-20251223-055023:/tmp/matrix.json ./matrix.json --zone=us-central1-c
```

---

## 10. Directory Structure on VM

```
~/
├── cities/                          # City-specific data
│   ├── ny_metro.geojson            # City boundary (37 KB)
│   ├── ny_metro.osm.pbf            # Clipped OSM data (219 MB)
│   └── ny_metro.osrm.*             # OSRM processed files (~1.1 GB total)
│       ├── ny_metro.osrm           # Main graph (135 MB)
│       ├── ny_metro.osrm.cell_metrics  # MLD metrics (221 MB)
│       ├── ny_metro.osrm.cells     # Cell data (1.6 MB)
│       ├── ny_metro.osrm.cnbg      # Compressed node-based graph (16 MB)
│       ├── ny_metro.osrm.cnbg_to_ebg   # Node to edge mapping (16 MB)
│       ├── ny_metro.osrm.datasource_names  # Data sources (68 KB)
│       ├── ny_metro.osrm.ebg       # Edge-based graph (81 MB)
│       ├── ny_metro.osrm.ebg_nodes # Edge-based nodes (22 MB)
│       ├── ny_metro.osrm.edges     # Edges (24 MB)
│       ├── ny_metro.osrm.enw       # Edge node weights (22 MB)
│       ├── ny_metro.osrm.fileIndex # File index (56 MB)
│       ├── ny_metro.osrm.geometry  # Road geometries (68 MB)
│       ├── ny_metro.osrm.icd       # Intersection data (11 MB)
│       ├── ny_metro.osrm.mldgr     # MLD graph (85 MB)
│       ├── ny_metro.osrm.names     # Street names (1 MB)
│       ├── ny_metro.osrm.nbg_nodes # Node-based graph nodes (31 MB)
│       ├── ny_metro.osrm.partition # Partition data (15 MB)
│       ├── ny_metro.osrm.properties    # Properties (6 KB)
│       ├── ny_metro.osrm.ramIndex  # RAM index (232 KB)
│       ├── ny_metro.osrm.restrictions  # Turn restrictions (4 KB)
│       ├── ny_metro.osrm.timestamp # Timestamp (3.5 KB)
│       ├── ny_metro.osrm.tld       # Turn lane data (7.5 KB)
│       ├── ny_metro.osrm.tls       # Traffic lights (12 KB)
│       ├── ny_metro.osrm.turn_duration_penalties  # Duration penalties (6.7 MB)
│       ├── ny_metro.osrm.turn_penalties_index     # Penalty index (41 MB)
│       └── ny_metro.osrm.turn_weight_penalties    # Weight penalties (6.7 MB)
│
└── osrm-data/                       # Regional OSM source data
    └── region.osm.pbf              # Merged state data (816 MB)
```

### File Size Summary
| Component | Size | Description |
|-----------|------|-------------|
| Source GeoJSON | 37 KB | City boundary polygon |
| Clipped OSM | 219 MB | Roads within boundary |
| OSRM files | ~1.1 GB | Preprocessed routing graph |
| Regional OSM | 816 MB | NY + NJ + CT merged |

---

## 9. Complete Workflow Script

```bash
#!/bin/bash
# Complete workflow for processing a new city

INSTANCE="instance-20251223-055023"
ZONE="us-central1-c"
CITY_NAME="ny_metro"
LOCAL_GEOJSON="/path/to/ny_metro.geojson"

# 1. Start VM
gcloud compute instances start $INSTANCE --zone=$ZONE

# 2. Create folders on VM
gcloud compute ssh $INSTANCE --zone=$ZONE --tunnel-through-iap --command="mkdir -p ~/cities ~/osrm-data"

# 3. Upload GeoJSON boundary
gcloud compute scp "$LOCAL_GEOJSON" $INSTANCE:~/cities/ --zone=$ZONE

# 4. Download and merge required state OSM files (adjust states as needed)
gcloud compute ssh $INSTANCE --zone=$ZONE --tunnel-through-iap --command="
  cd ~/osrm-data && \
  wget -c https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf && \
  wget -c https://download.geofabrik.de/north-america/us/new-jersey-latest.osm.pbf && \
  wget -c https://download.geofabrik.de/north-america/us/connecticut-latest.osm.pbf && \
  osmium merge *.osm.pbf -o region.osm.pbf --overwrite && \
  rm -f new-york-latest.osm.pbf new-jersey-latest.osm.pbf connecticut-latest.osm.pbf
"

# 5. Extract city boundary
gcloud compute ssh $INSTANCE --zone=$ZONE --tunnel-through-iap --command="
  osmium extract -p cities/${CITY_NAME}.geojson osrm-data/region.osm.pbf -o cities/${CITY_NAME}.osm.pbf
"

# 6. OSRM preprocessing
gcloud compute ssh $INSTANCE --zone=$ZONE --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/${CITY_NAME}.osm.pbf && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-partition /data/${CITY_NAME}.osrm && \
  docker run -t -v \"\${PWD}:/data\" osrm/osrm-backend osrm-customize /data/${CITY_NAME}.osrm
"

echo "Done! OSRM files ready at ~/cities/${CITY_NAME}.osrm.*"
```

---

## 10. Quick Reference Commands

### Check VM disk usage
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="df -h"
```

### List files on VM
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="ls -lh ~/cities/"
```

### Download files from VM
```bash
gcloud compute scp instance-20251223-055023:~/cities/output.osm.pbf /local/path/ --zone=us-central1-c
```

---

## 11. Troubleshooting

| Issue | Solution |
|-------|----------|
| `dest open "folder/": Failure` | Create folder first with `mkdir -p` |
| `No such file or directory` | Check path; use `ls` to verify |
| `File exists` error in osmium | Add `--overwrite` flag or delete file |
| Wrong file downloaded (HTML) | Verify URL exists; Geofabrik uses state-level for US |
| SSH connection timeout | Use `--tunnel-through-iap` flag (see below) |
| `Connection reset by peer` | Use IAP tunneling instead of direct SSH |

### SSH Connection Issues - Use IAP Tunneling
When direct SSH fails with timeout or connection reset errors, use IAP (Identity-Aware Proxy) tunneling:

```bash
# Instead of:
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --command="..."

# Use:
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="..."
```

This routes through Google's infrastructure and is more reliable when the VM is under heavy load.

---

## 12. VM Instance Details

- **Instance Name:** instance-20251223-055023
- **Zone:** us-central1-c
- **Project:** ee-knhuang
- **SSH Key Location:** `~/.ssh/google_compute_engine`
- **Docker Image:** `osrm/osrm-backend:latest`

---

## 13. Route Polyline Fetching (Pass 2)

After computing travel time matrices with the Table API, use the Route API to fetch actual route geometries.

### Start OSRM Server for Route API
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  cd ~/cities && \
  docker run -d -p 5000:5000 -v \"\${PWD}:/data\" osrm/osrm-backend osrm-routed --algorithm mld /data/ny_metro.osrm
"
```

### Route API Format
```
GET http://localhost:5000/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson
```

### Example Route Request
```bash
gcloud compute ssh instance-20251223-055023 --zone=us-central1-c --tunnel-through-iap --command="
  curl -s 'http://localhost:5000/route/v1/driving/-74.048443,40.818747;-74.497560,40.511854?overview=full&geometries=geojson'
"
```

### Route API Response
```json
{
  "code": "Ok",
  "routes": [{
    "geometry": {"type": "LineString", "coordinates": [[lon, lat], ...]},
    "duration": 1234.5,
    "distance": 15000.0
  }]
}
```

### Batch Polyline Fetching Script
Use `fetch_polylines.py` to fetch all pairwise routes for a city:

```bash
# On VM - run after matrix is computed
source ~/osrm_env/bin/activate
nohup python3 ~/fetch_polylines.py > ~/polylines.log 2>&1 &
```

The script:
1. Reads existing `{city_id}_matrix.json` files
2. Starts OSRM server for each city
3. Fetches all pairwise routes via Route API (8 parallel workers)
4. Saves `{city_id}_routes.geojson` with full geometries

### Performance Comparison
| Method | 219×219 Matrix (47,961 pairs) | Output |
|--------|-------------------------------|--------|
| Table API | 3.3 seconds | Duration/distance only |
| Route API (8 workers) | ~4 minutes | Full route geometries |

### Output Format (routes GeoJSON)
```json
{
  "type": "FeatureCollection",
  "properties": {"city_id": "945", "total_routes": 47961, "failed_routes": 0},
  "features": [
    {
      "type": "Feature",
      "properties": {
        "origin_h3": "862a13d67ffffff",
        "destination_h3": "862a12a6fffffff",
        "duration": 1989.4,
        "distance": 33795.1
      },
      "geometry": {"type": "LineString", "coordinates": [[-74.313, 40.388], ...]}
    }
  ]
}
```

---

*Last updated: 2025-12-24*
