# Parallel Multi-VM OSRM Processing for 13,135 Global Cities

## Overview

This document describes the setup and workflow for processing OSRM travel time matrices for 13,135 cities across 182 countries using a parallel multi-VM approach on Google Cloud Platform.

---

## 1. Scale Analysis

### Dataset Summary
| Metric | Value |
|--------|-------|
| Total cities | 13,135 |
| Total area | 975,193 km² |
| Countries | 182 |
| Continents | 6 |

### City Distribution by VM (Optimized Split)
| VM | Cities | Countries | Area (km²) | Est. Hours | Est. Storage (GB) |
|----|--------|-----------|------------|------------|-------------------|
| osrm-india | 3,248 | 1 | 89,671 | 7.5 | 32.9 |
| osrm-china | 1,850 | 1 | 154,901 | 12.9 | 56.8 |
| osrm-asia-other | 2,639 | 47 | 206,171 | 17.2 | 75.6 |
| osrm-global-small | 3,967 | 91 | 157,899 | 13.2 | 57.9 |
| osrm-northam | 372 | 2 | 167,299 | 13.9 | 61.3 |
| osrm-europe | 1,059 | 41 | 199,252 | 16.6 | 73.1 |
| **TOTAL** | **13,135** | **182** | **975,193** | **~17** (parallel) | **357.6** |

### Top Countries by City Count
| Country | Cities | Region |
|---------|--------|--------|
| India | 3,248 | Asia |
| China | 1,850 | Asia |
| Ethiopia | 557 | Africa |
| Nigeria | 483 | Africa |
| Indonesia | 393 | Asia |
| Brazil | 349 | Latin America |
| United States | 324 | Northern America |
| Pakistan | 301 | Asia |
| Bangladesh | 301 | Asia |
| Russia | 209 | Europe/Asia |

### Largest Cities (Processing Bottlenecks)
| City | Country | Area (km²) | Est. RAM Needed | Est. Time |
|------|---------|------------|-----------------|-----------|
| New York | USA | 9,388 | 13 GB | 47 mins |
| Los Angeles | USA | 8,187 | 11 GB | 41 mins |
| Tokyo | Japan | 8,060 | 11 GB | 40 mins |
| Guangzhou | China | 7,804 | 10 GB | 39 mins |
| Chicago | USA | 6,891 | 9 GB | 34 mins |
| Moscow | Russia | 5,938 | 8 GB | 30 mins |

---

## 2. Infrastructure Design

### VM Configuration (Optimized Split)

| VM Name | Zone | Machine Type | vCPUs | RAM | Disk | Cities | Est. Cost/Day |
|---------|------|--------------|-------|-----|------|--------|---------------|
| osrm-india | asia-south1-c | e2-highmem-4 | 4 | 32 GB | 50 GB | 3,248 | $5.50 |
| osrm-china | asia-east1-b | c2d-highmem-4 | 4 | 32 GB | 80 GB | 1,850 | $7.00 |
| osrm-asia-other | asia-southeast1-b | c2d-highmem-4 | 4 | 32 GB | 100 GB | 2,639 | $8.00 |
| osrm-global-small | europe-west1-b | e2-highmem-4 | 4 | 32 GB | 80 GB | 3,967 | $5.50 |
| osrm-northam | us-central1-c | c2d-highmem-4 | 4 | 32 GB | 100 GB | 372 | $7.00 |
| osrm-europe | europe-west1-b | c2d-highmem-4 | 4 | 32 GB | 100 GB | 1,059 | $8.00 |

**Total estimated cost: ~$41/day × 2 days = ~$82**

### Why This Split?
- **India & China separated**: Each has 1,850-3,248 cities - splitting prevents bottleneck
- **Asia-Other**: Handles 47 other Asian countries (Japan, Indonesia, Pakistan, etc.)
- **Global-Small**: Combines Oceania (86), Africa (2,805), Latin America (1,076) - 91 countries but smaller cities
- **Northern America**: Has New York and LA (largest cities globally) - needs high RAM
- **Europe**: Moscow is large, 41 countries - moderate resources

### VM Status (Created)
```
NAME               ZONE               MACHINE_TYPE   STATUS   DISK_SIZE_GB
osrm-india         asia-south1-c      e2-highmem-4   RUNNING  50
osrm-china         asia-east1-b       c2d-highmem-4  RUNNING  80
osrm-asia-other    asia-southeast1-b  c2d-highmem-4  RUNNING  100
osrm-global-small  europe-west1-b     e2-highmem-4   RUNNING  80
osrm-northam       us-central1-c      c2d-highmem-4  RUNNING  100
osrm-europe        europe-west1-b     c2d-highmem-4  RUNNING  100
```

---

## 3. VM Setup Commands

### Create All VMs (Already Created)
```bash
# Set project
gcloud config set project ee-knhuang

# India (3,248 cities)
gcloud compute instances create osrm-india \
  --zone=asia-south1-c \
  --machine-type=e2-highmem-4 \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=osrm

# China (1,850 cities - has Guangzhou, largest Asian city)
gcloud compute instances create osrm-china \
  --zone=asia-east1-b \
  --machine-type=c2d-highmem-4 \
  --boot-disk-size=80GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=osrm

# Asia Other (2,639 cities - Japan, Indonesia, Pakistan, etc.)
gcloud compute instances create osrm-asia-other \
  --zone=asia-southeast1-b \
  --machine-type=c2d-highmem-4 \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=osrm

# Global Small (3,967 cities - Oceania + Africa + Latin America)
gcloud compute instances create osrm-global-small \
  --zone=europe-west1-b \
  --machine-type=e2-highmem-4 \
  --boot-disk-size=80GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=osrm

# Northern America (372 cities - has NY and LA, largest cities)
gcloud compute instances create osrm-northam \
  --zone=us-central1-c \
  --machine-type=c2d-highmem-4 \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=osrm

# Europe (1,059 cities - has Moscow)
gcloud compute instances create osrm-europe \
  --zone=europe-west1-b \
  --machine-type=c2d-highmem-4 \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=osrm
```

### Initialize Each VM with Docker and Tools
Run this on each VM after creation:
```bash
#!/bin/bash
# install_dependencies.sh

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install osmium
sudo apt-get install -y osmium-tool

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-venv
python3 -m venv ~/osrm_env
source ~/osrm_env/bin/activate
pip install geopandas pandas h3 requests

# Create directories
mkdir -p ~/osrm-data ~/cities ~/results

# Pull OSRM Docker image
sudo docker pull osrm/osrm-backend

echo "Setup complete! Log out and back in for Docker permissions."
```

---

## 4. OSM Data Download Strategy

### ⚠️ IMPORTANT: Always Download Country-Specific Files

**Never use continent-level OSM files for clipping.** The `osmium extract` command must scan the entire OSM file for each city, so:

| OSM File Size | Time per City | 100 Cities |
|---------------|---------------|------------|
| Country (100-500 MB) | ~5 seconds | ~8 minutes |
| Continent (7 GB) | ~3-5 minutes | ~5-8 hours |

**Example**: Using `africa-latest.osm.pbf` (7GB) instead of individual country files caused clipping to run **36x slower**.

**Rule of thumb**: Download individual country files. Only use continent files as a last resort for countries not available separately on Geofabrik.

### Geofabrik Download URLs by Region

#### Asia
```bash
# Major countries (download individually for large ones)
wget https://download.geofabrik.de/asia/china-latest.osm.pbf          # 1.0 GB
wget https://download.geofabrik.de/asia/india-latest.osm.pbf          # 1.3 GB
wget https://download.geofabrik.de/asia/japan-latest.osm.pbf          # 1.8 GB
wget https://download.geofabrik.de/asia/indonesia-latest.osm.pbf      # 400 MB
wget https://download.geofabrik.de/asia/pakistan-latest.osm.pbf       # 200 MB
wget https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf     # 200 MB
wget https://download.geofabrik.de/asia/vietnam-latest.osm.pbf        # 150 MB
wget https://download.geofabrik.de/asia/iran-latest.osm.pbf           # 300 MB
# ... other countries as needed
```

#### Africa
```bash
# ALWAYS download individual countries - continent file is 7GB and extremely slow!
# Top countries by city count:
wget https://download.geofabrik.de/africa/ethiopia-latest.osm.pbf     # 104 MB - 557 cities
wget https://download.geofabrik.de/africa/nigeria-latest.osm.pbf      # 634 MB - 483 cities
wget https://download.geofabrik.de/africa/egypt-latest.osm.pbf        # 166 MB
wget https://download.geofabrik.de/africa/south-africa-latest.osm.pbf # 367 MB
wget https://download.geofabrik.de/africa/kenya-latest.osm.pbf        # 316 MB
wget https://download.geofabrik.de/africa/tanzania-latest.osm.pbf     # 669 MB
wget https://download.geofabrik.de/africa/algeria-latest.osm.pbf      # 280 MB
wget https://download.geofabrik.de/africa/morocco-latest.osm.pbf      # 230 MB
wget https://download.geofabrik.de/africa/ghana-latest.osm.pbf        # 103 MB
wget https://download.geofabrik.de/africa/cameroon-latest.osm.pbf     # 206 MB
wget https://download.geofabrik.de/africa/uganda-latest.osm.pbf       # 338 MB
wget https://download.geofabrik.de/africa/sudan-latest.osm.pbf        # 133 MB
wget https://download.geofabrik.de/africa/angola-latest.osm.pbf
wget https://download.geofabrik.de/africa/mozambique-latest.osm.pbf
wget https://download.geofabrik.de/africa/madagascar-latest.osm.pbf
wget https://download.geofabrik.de/africa/congo-democratic-republic-latest.osm.pbf
# ... download ALL countries you need - see https://download.geofabrik.de/africa.html
# Only use africa-latest.osm.pbf as fallback for missing countries
```

#### Europe
```bash
# Download individual countries - DO NOT use europe-latest.osm.pbf (28 GB)!
wget https://download.geofabrik.de/europe/russia-latest.osm.pbf       # 3.8 GB
wget https://download.geofabrik.de/europe/france-latest.osm.pbf       # 4.6 GB
wget https://download.geofabrik.de/europe/germany-latest.osm.pbf      # 4.4 GB
wget https://download.geofabrik.de/europe/great-britain-latest.osm.pbf # 2.0 GB
wget https://download.geofabrik.de/europe/italy-latest.osm.pbf        # 2.0 GB
wget https://download.geofabrik.de/europe/poland-latest.osm.pbf       # 1.9 GB
wget https://download.geofabrik.de/europe/spain-latest.osm.pbf        # 1.3 GB
wget https://download.geofabrik.de/europe/netherlands-latest.osm.pbf  # 1.3 GB
wget https://download.geofabrik.de/europe/ukraine-latest.osm.pbf      # 806 MB
wget https://download.geofabrik.de/europe/turkey-latest.osm.pbf       # 588 MB
# ... download all countries with cities in your dataset
```

#### Latin America
```bash
# South America continent file is acceptable (3.6 GB) but country files are faster
# For best performance, download individual countries:
wget https://download.geofabrik.de/south-america/brazil-latest.osm.pbf  # 1.4 GB - 349 cities
wget https://download.geofabrik.de/south-america/colombia-latest.osm.pbf
wget https://download.geofabrik.de/south-america/argentina-latest.osm.pbf
wget https://download.geofabrik.de/south-america/peru-latest.osm.pbf
wget https://download.geofabrik.de/south-america/venezuela-latest.osm.pbf
wget https://download.geofabrik.de/south-america/chile-latest.osm.pbf
wget https://download.geofabrik.de/south-america/ecuador-latest.osm.pbf
# Or use continent files as fallback:
wget https://download.geofabrik.de/south-america-latest.osm.pbf       # 3.6 GB
wget https://download.geofabrik.de/central-america-latest.osm.pbf     # 724 MB
```

#### Northern America
```bash
# US by state (recommended for US)
wget https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf
wget https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf
wget https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf
# ... all relevant states

# Canada
wget https://download.geofabrik.de/north-america/canada-latest.osm.pbf
```

#### Oceania
```bash
wget https://download.geofabrik.de/australia-oceania-latest.osm.pbf   # 1.0 GB
```

---

## 5. Modular 3-Phase Processing Pipeline

The processing pipeline is split into 3 independent phases for maximum flexibility:

```
Phase 1: clip_cities.py     → ~/clipped/{city_id}.osm.pbf
Phase 2: preprocess_cities.py → ~/osrm/{city_id}.tar.gz (or directory)
Phase 3: route_cities.py    → ~/results/{city_id}_matrix.json, _routes.geojson
```

### Scripts Overview

| Script | Phase | Purpose | Key Options |
|--------|-------|---------|-------------|
| `clip_cities.py` | 1 | Clip OSM to city boundaries | `--region`, `--osm-dir` |
| `preprocess_cities.py` | 2 | OSRM extract/partition/customize | `--profile`, `--compress` |
| `route_cities.py` | 3 | Compute matrices & polylines | `--h3-resolution`, `--fetch-polylines` |
| `process_cities.py` | All | Wrapper for all 3 phases | All options above |
| `common.py` | - | Shared utilities | - |

### Why Separate Phases?

1. **Re-run routing without re-preprocessing**: Change H3 resolution or fetch polylines
2. **Different profiles**: Bicycle/foot requires re-preprocessing (Phase 2) but not re-clipping
3. **Storage efficiency**: Compress OSRM files for large cities (~300MB vs 1.5GB)
4. **Failure recovery**: Resume from failed phase without starting over

### Phase 1: Clipping

```bash
# Clip all cities in a region
python clip_cities.py --region ~/cities.geojson

# With custom directories
python clip_cities.py --region ~/cities.geojson --osm-dir ~/osrm-data --output-dir ~/clipped
```

### Phase 2: Preprocessing

```bash
# Preprocess with car profile (default)
python preprocess_cities.py --clipped-dir ~/clipped

# With bicycle profile and compression
python preprocess_cities.py --clipped-dir ~/clipped --profile bicycle --compress

# Process single city
python preprocess_cities.py --clipped-dir ~/clipped --city-id 12345
```

### Phase 3: Routing

```bash
# Basic routing (H3 resolution 7, no polylines)
python route_cities.py --region ~/cities.geojson

# With polylines and different H3 resolution
python route_cities.py --region ~/cities.geojson --h3-resolution 6 --fetch-polylines

# Cleanup OSRM files after routing (keep compressed)
python route_cities.py --region ~/cities.geojson --cleanup --keep-compressed
```

### All-in-One Processing

```bash
# Run all phases
python process_cities.py --region ~/cities.geojson

# With all options
python process_cities.py --region ~/cities.geojson \
    --profile car --compress \
    --h3-resolution 7 --fetch-polylines \
    --cleanup --keep-compressed

# Skip phases (for re-runs)
python process_cities.py --region ~/cities.geojson --skip-clip --skip-preprocess
```

### Storage Estimates

| Component | Per City | 13,135 Cities |
|-----------|----------|---------------|
| Clipped OSM | ~15 MB | ~200 GB |
| OSRM compressed | ~300 MB | ~4 TB (if kept) |
| Results only | ~5 MB | ~65 GB |

**Recommended strategy**: Keep clipped OSM (~200GB), delete OSRM after routing for small cities, keep compressed OSRM only for large cities (>100 km²).

---

## 5b. Legacy Batch Processing Script (Reference)

The original monolithic script for reference:

```python
#!/usr/bin/env python3
"""
OSRM Batch Processing Script
Processes all cities in a region, computing travel time matrices.
"""

import geopandas as gpd
import pandas as pd
import subprocess
import requests
import json
import os
import time
import logging
from pathlib import Path

# Configuration
REGION_FILE = os.environ.get('REGION_FILE', 'cities.geojson')
OSM_DIR = Path(os.path.expanduser('~/osrm-data'))
CITIES_DIR = Path(os.path.expanduser('~/cities'))
RESULTS_DIR = Path(os.path.expanduser('~/results'))
H3_RESOLUTION = 7  # Adjust based on city size

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, timeout=3600):
    """Run shell command with timeout."""
    logger.info(f"Running: {cmd[:100]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        return False
    return True

def find_osm_file(city_gdf):
    """Find the appropriate OSM file for a city based on country."""
    country = city_gdf['CTR_MN_NM'].iloc[0].lower().replace(' ', '-')

    # Check for country-specific file first
    country_file = OSM_DIR / f"{country}-latest.osm.pbf"
    if country_file.exists():
        return country_file

    # Fall back to region file
    region_files = list(OSM_DIR.glob("*-latest.osm.pbf"))
    if region_files:
        return region_files[0]

    raise FileNotFoundError(f"No OSM file found for {country}")

def clip_city(city_id, city_geojson_path, osm_file):
    """Clip OSM data to city boundary."""
    output_file = CITIES_DIR / f"{city_id}.osm.pbf"
    if output_file.exists():
        logger.info(f"Clipped file exists: {output_file}")
        return output_file

    cmd = f"osmium extract -p {city_geojson_path} {osm_file} -o {output_file}"
    if run_command(cmd):
        return output_file
    return None

def osrm_preprocess(city_id, osm_file):
    """Run OSRM preprocessing (extract, partition, customize)."""
    base_name = osm_file.stem.replace('.osm', '')
    osrm_file = CITIES_DIR / f"{city_id}.osrm"

    if osrm_file.exists():
        logger.info(f"OSRM file exists: {osrm_file}")
        return osrm_file

    # Extract
    cmd = f'cd {CITIES_DIR} && docker run -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/{city_id}.osm.pbf'
    if not run_command(cmd, timeout=1800):
        return None

    # Partition
    cmd = f'cd {CITIES_DIR} && docker run -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-partition /data/{city_id}.osrm'
    if not run_command(cmd, timeout=1800):
        return None

    # Customize
    cmd = f'cd {CITIES_DIR} && docker run -t -v "${{PWD}}:/data" osrm/osrm-backend osrm-customize /data/{city_id}.osrm'
    if not run_command(cmd, timeout=1800):
        return None

    return osrm_file

def start_osrm_server(city_id, max_table_size=1000):
    """Start OSRM routing server."""
    # Stop any existing server
    subprocess.run("docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend) 2>/dev/null", shell=True)
    time.sleep(2)

    cmd = f'cd {CITIES_DIR} && docker run -d -p 5000:5000 -v "${{PWD}}:/data" osrm/osrm-backend osrm-routed --algorithm mld --max-table-size {max_table_size} /data/{city_id}.osrm'
    if run_command(cmd):
        time.sleep(5)  # Wait for server to start
        return True
    return False

def stop_osrm_server():
    """Stop OSRM routing server."""
    subprocess.run("docker stop $(docker ps -q --filter ancestor=osrm/osrm-backend) 2>/dev/null", shell=True)

def generate_h3_grids(city_gdf, resolution=7):
    """Generate H3 hexagonal grids for city."""
    import h3

    # Get city boundary
    city_geom = city_gdf.geometry.iloc[0]

    # Get H3 cells covering the city
    h3_cells = list(h3.polyfill_geojson(city_geom.__geo_interface__, resolution))

    # Get centroids
    centroids = []
    for cell in h3_cells:
        lat, lon = h3.h3_to_geo(cell)
        centroids.append({'h3_index': cell, 'lat': lat, 'lon': lon})

    return pd.DataFrame(centroids)

def compute_matrix(centroids_df):
    """Compute travel time matrix using OSRM Table API."""
    coords = ';'.join([f"{row['lon']},{row['lat']}" for _, row in centroids_df.iterrows()])

    url = f"http://localhost:5000/table/v1/driving/{coords}?annotations=duration,distance"

    try:
        response = requests.get(url, timeout=300)
        data = response.json()

        if data.get('code') != 'Ok':
            logger.error(f"OSRM error: {data.get('message', 'Unknown error')}")
            return None

        return data
    except Exception as e:
        logger.error(f"Matrix computation failed: {e}")
        return None

def save_results(city_id, centroids_df, matrix_data):
    """Save matrix results."""
    h3_indices = centroids_df['h3_index'].tolist()

    # Save JSON with full data
    result = {
        'city_id': city_id,
        'h3_indices': h3_indices,
        'centroids': centroids_df.to_dict('records'),
        'durations': matrix_data['durations'],
        'distances': matrix_data['distances'],
        'n_grids': len(h3_indices)
    }

    with open(RESULTS_DIR / f"{city_id}_matrix.json", 'w') as f:
        json.dump(result, f)

    # Save duration CSV
    duration_df = pd.DataFrame(matrix_data['durations'], index=h3_indices, columns=h3_indices)
    duration_df.to_csv(RESULTS_DIR / f"{city_id}_duration.csv")

    logger.info(f"Results saved for {city_id}: {len(h3_indices)} grids")

def cleanup_city_files(city_id):
    """Remove OSRM files to save space."""
    for f in CITIES_DIR.glob(f"{city_id}.osrm*"):
        f.unlink()
    osm_file = CITIES_DIR / f"{city_id}.osm.pbf"
    if osm_file.exists():
        osm_file.unlink()
    logger.info(f"Cleaned up files for {city_id}")

def process_city(city_row, osm_file):
    """Process a single city end-to-end."""
    city_id = str(city_row['ID_HDC_G0'])
    city_name = city_row.get('UC_NM_MN', city_id)

    logger.info(f"=" * 60)
    logger.info(f"Processing: {city_name} (ID: {city_id})")
    logger.info(f"=" * 60)

    # Check if already processed
    if (RESULTS_DIR / f"{city_id}_matrix.json").exists():
        logger.info(f"Already processed: {city_id}")
        return True

    try:
        # Create city GeoJSON
        city_gdf = gpd.GeoDataFrame([city_row], crs="EPSG:4326")
        city_geojson = CITIES_DIR / f"{city_id}.geojson"
        city_gdf.to_file(city_geojson, driver='GeoJSON')

        # Step 1: Clip OSM
        logger.info("Step 1/5: Clipping OSM data...")
        clipped_osm = clip_city(city_id, city_geojson, osm_file)
        if not clipped_osm:
            logger.error(f"Failed to clip OSM for {city_id}")
            return False

        # Step 2: OSRM preprocessing
        logger.info("Step 2/5: OSRM preprocessing...")
        osrm_file = osrm_preprocess(city_id, clipped_osm)
        if not osrm_file:
            logger.error(f"Failed OSRM preprocessing for {city_id}")
            return False

        # Step 3: Generate H3 grids
        logger.info("Step 3/5: Generating H3 grids...")
        centroids_df = generate_h3_grids(city_gdf)
        n_grids = len(centroids_df)
        logger.info(f"Generated {n_grids} H3 grids")

        if n_grids == 0:
            logger.warning(f"No grids generated for {city_id}")
            return False

        # Step 4: Start server and compute matrix
        logger.info("Step 4/5: Computing travel time matrix...")
        max_table_size = max(n_grids + 100, 500)
        if not start_osrm_server(city_id, max_table_size):
            logger.error(f"Failed to start server for {city_id}")
            return False

        matrix_data = compute_matrix(centroids_df)
        stop_osrm_server()

        if not matrix_data:
            logger.error(f"Failed to compute matrix for {city_id}")
            return False

        # Step 5: Save results
        logger.info("Step 5/5: Saving results...")
        save_results(city_id, centroids_df, matrix_data)

        # Cleanup to save space
        cleanup_city_files(city_id)

        logger.info(f"Successfully processed {city_name}")
        return True

    except Exception as e:
        logger.error(f"Error processing {city_id}: {e}")
        stop_osrm_server()
        return False

def main():
    """Main processing loop."""
    # Load cities
    cities_gdf = gpd.read_file(REGION_FILE)
    logger.info(f"Loaded {len(cities_gdf)} cities from {REGION_FILE}")

    # Sort by area (process smaller cities first)
    cities_gdf['area_km2'] = cities_gdf.to_crs('EPSG:3857').geometry.area / 1e6
    cities_gdf = cities_gdf.sort_values('area_km2')

    # Group by country for efficient OSM usage
    countries = cities_gdf.groupby('CTR_MN_NM')

    processed = 0
    failed = 0

    for country, country_cities in countries:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Processing {len(country_cities)} cities in {country}")
        logger.info(f"{'#' * 60}\n")

        # Find OSM file for this country
        try:
            osm_file = find_osm_file(country_cities)
            logger.info(f"Using OSM file: {osm_file}")
        except FileNotFoundError as e:
            logger.error(f"Skipping {country}: {e}")
            failed += len(country_cities)
            continue

        for idx, city_row in country_cities.iterrows():
            success = process_city(city_row, osm_file)
            if success:
                processed += 1
            else:
                failed += 1

            logger.info(f"Progress: {processed} processed, {failed} failed, {len(cities_gdf) - processed - failed} remaining")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"Processed: {processed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"{'=' * 60}")

if __name__ == '__main__':
    main()
```

---

## 6. Launch Commands for Each VM

### India VM (3,248 cities)
```bash
# SSH into VM
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap

# Upload cities file
gcloud compute scp regions/india.geojson osrm-india:~/cities.geojson --zone=asia-south1-c

# On VM - download India OSM
cd ~/osrm-data
wget -c https://download.geofabrik.de/asia/india-latest.osm.pbf

# Start processing
cd ~ && source osrm_env/bin/activate
REGION_FILE=~/cities.geojson python3 process_cities.py > processing.log 2>&1 &
```

### China VM (1,850 cities)
```bash
gcloud compute ssh osrm-china --zone=asia-east1-b --tunnel-through-iap
gcloud compute scp regions/china.geojson osrm-china:~/cities.geojson --zone=asia-east1-b

# On VM:
cd ~/osrm-data && wget -c https://download.geofabrik.de/asia/china-latest.osm.pbf
cd ~ && source osrm_env/bin/activate
REGION_FILE=~/cities.geojson python3 process_cities.py > processing.log 2>&1 &
```

### Asia Other VM (2,639 cities - 47 countries)
```bash
gcloud compute ssh osrm-asia-other --zone=asia-southeast1-b --tunnel-through-iap
gcloud compute scp regions/asia_other.geojson osrm-asia-other:~/cities.geojson --zone=asia-southeast1-b

# On VM - download major Asian country OSM files
cd ~/osrm-data
wget -c https://download.geofabrik.de/asia/japan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/indonesia-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/pakistan-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/vietnam-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/iran-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/philippines-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/myanmar-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/iraq-latest.osm.pbf
wget -c https://download.geofabrik.de/asia/thailand-latest.osm.pbf
# Add more as needed

cd ~ && source osrm_env/bin/activate
REGION_FILE=~/cities.geojson python3 process_cities.py > processing.log 2>&1 &
```

### Global Small VM (3,967 cities - Oceania + Africa + Latin America)
```bash
gcloud compute ssh osrm-global-small --zone=europe-west1-b --tunnel-through-iap
gcloud compute scp regions/global_small.geojson osrm-global-small:~/cities.geojson --zone=europe-west1-b

# On VM - download continent-level OSM files
cd ~/osrm-data
wget -c https://download.geofabrik.de/africa-latest.osm.pbf
wget -c https://download.geofabrik.de/south-america-latest.osm.pbf
wget -c https://download.geofabrik.de/central-america-latest.osm.pbf
wget -c https://download.geofabrik.de/australia-oceania-latest.osm.pbf

cd ~ && source osrm_env/bin/activate
REGION_FILE=~/cities.geojson python3 process_cities.py > processing.log 2>&1 &
```

### Northern America VM (372 cities)
```bash
gcloud compute ssh osrm-northam --zone=us-central1-c --tunnel-through-iap
gcloud compute scp regions/northern_america.geojson osrm-northam:~/cities.geojson --zone=us-central1-c

# On VM - download US states with major cities
cd ~/osrm-data
for state in new-york california texas illinois michigan florida pennsylvania ohio georgia \
             new-jersey massachusetts arizona colorado washington maryland virginia; do
  wget -c https://download.geofabrik.de/north-america/us/${state}-latest.osm.pbf
done
wget -c https://download.geofabrik.de/north-america/canada-latest.osm.pbf

cd ~ && source osrm_env/bin/activate
REGION_FILE=~/cities.geojson python3 process_cities.py > processing.log 2>&1 &
```

### Europe VM (1,059 cities - 41 countries)
```bash
gcloud compute ssh osrm-europe --zone=europe-west1-b --tunnel-through-iap
gcloud compute scp regions/europe.geojson osrm-europe:~/cities.geojson --zone=europe-west1-b

# On VM - download major European country OSM files
cd ~/osrm-data
wget -c https://download.geofabrik.de/europe/russia-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/great-britain-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/germany-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/france-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/italy-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/spain-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/poland-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/ukraine-latest.osm.pbf
wget -c https://download.geofabrik.de/europe/turkey-latest.osm.pbf
# Add more as needed

cd ~ && source osrm_env/bin/activate
REGION_FILE=~/cities.geojson python3 process_cities.py > processing.log 2>&1 &
```

---

## 7. Monitoring Progress

### Check Processing Status
```bash
# Check log on any VM
gcloud compute ssh osrm-asia --zone=asia-east1-b --tunnel-through-iap --command="tail -50 ~/processing.log"

# Count completed cities
gcloud compute ssh osrm-asia --zone=asia-east1-b --tunnel-through-iap --command="ls ~/results/*.json | wc -l"

# Check disk usage
gcloud compute ssh osrm-asia --zone=asia-east1-b --tunnel-through-iap --command="df -h"
```

### Monitor All VMs
```bash
#!/bin/bash
# monitor_all.sh

VMS=("osrm-india:asia-south1-c" "osrm-china:asia-east1-b" "osrm-asia-other:asia-southeast1-b"
     "osrm-global-small:europe-west1-b" "osrm-northam:us-central1-c" "osrm-europe:europe-west1-b")

for vm_zone in "${VMS[@]}"; do
  IFS=':' read -r vm zone <<< "$vm_zone"
  echo "=== $vm ==="
  gcloud compute ssh $vm --zone=$zone --tunnel-through-iap --command="
    echo \"Completed: \$(ls ~/results/*.json 2>/dev/null | wc -l)\"
    echo \"Last log: \$(tail -1 ~/processing.log 2>/dev/null)\"
    echo \"Disk: \$(df -h / | tail -1 | awk '{print \$5}')\"
  " 2>/dev/null || echo "VM not reachable"
  echo ""
done
```

---

## 8. Collecting Results

After processing completes, download results from all VMs:

```bash
#!/bin/bash
# collect_results.sh

mkdir -p all_results

# Download from each VM
gcloud compute scp "osrm-india:~/results/*" all_results/ --zone=asia-south1-c
gcloud compute scp "osrm-china:~/results/*" all_results/ --zone=asia-east1-b
gcloud compute scp "osrm-asia-other:~/results/*" all_results/ --zone=asia-southeast1-b
gcloud compute scp "osrm-global-small:~/results/*" all_results/ --zone=europe-west1-b
gcloud compute scp "osrm-northam:~/results/*" all_results/ --zone=us-central1-c
gcloud compute scp "osrm-europe:~/results/*" all_results/ --zone=europe-west1-b

echo "Downloaded $(ls all_results/*.json | wc -l) city matrices"
```

---

## 9. Cleanup

### Stop All VMs (to save cost)
```bash
gcloud compute instances stop osrm-india --zone=asia-south1-c
gcloud compute instances stop osrm-china --zone=asia-east1-b
gcloud compute instances stop osrm-asia-other --zone=asia-southeast1-b
gcloud compute instances stop osrm-global-small --zone=europe-west1-b
gcloud compute instances stop osrm-northam --zone=us-central1-c
gcloud compute instances stop osrm-europe --zone=europe-west1-b
```

### Delete All VMs (after confirming results)
```bash
gcloud compute instances delete osrm-india --zone=asia-south1-c --quiet
gcloud compute instances delete osrm-china --zone=asia-east1-b --quiet
gcloud compute instances delete osrm-asia-other --zone=asia-southeast1-b --quiet
gcloud compute instances delete osrm-global-small --zone=europe-west1-b --quiet
gcloud compute instances delete osrm-northam --zone=us-central1-c --quiet
gcloud compute instances delete osrm-europe --zone=europe-west1-b --quiet
```

---

## 10. Cost Estimation

| Component | Estimated Cost |
|-----------|---------------|
| VM compute (6 VMs × 2 days) | $70-80 |
| Storage (SSD disks) | $10-15 |
| Network egress (downloading results) | $5-10 |
| **Total** | **~$85-105** |

---

## 11. Timeline

| Phase | Duration |
|-------|----------|
| VM creation & setup | 1-2 hours |
| OSM data download (parallel) | 2-4 hours |
| City processing (parallel) | 24-36 hours |
| Result collection | 1-2 hours |
| **Total** | **~1.5-2 days** |

---

## 12. Route Polyline Fetching (Pass 2)

After travel time matrices are computed, run a second pass to fetch route geometries.

### Why Separate Passes?
- **Table API** (Pass 1): Fast matrix computation (~1.6 minutes for all cities)
- **Route API** (Pass 2): Slower but returns actual route polylines (~1.5 hours)

### Running Polyline Fetching

```bash
# On each VM (after matrices are done)
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="
    source ~/osrm_env/bin/activate
    nohup python3 ~/fetch_polylines.py >> ~/polylines.log 2>&1 &
"
```

### Script: `fetch_polylines.py`

The script:
1. Reads existing `{city_id}_matrix.json` files
2. Extracts H3 centroids
3. Fetches all pairwise routes via Route API (parallel requests)
4. **Simplifies geometries** using Douglas-Peucker a

orithm (~90% size reduction)
5. Saves `{city_id}_routes.geojson` per city

### Output Format

```json
{
  "type": "FeatureCollection",
  "properties": {"city_id": "945", "total_routes": 47961},
  "features": [
    {
      "type": "Feature",
      "properties": {
        "origin_h3": "872a1...",
        "destination_h3": "872a1...",
        "duration": 1234.5,
        "distance": 15000.0
      },
      "geometry": {"type": "LineString", "coordinates": [...]}
    }
  ]
}
```

### Monitoring Polyline Progress

```bash
# Count completed polyline files
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="ls ~/results/*_routes.geojson 2>/dev/null | wc -l"

# Check polyline log
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="tail -30 ~/polylines.log"
```

### Time Estimate
| Cities | Routes | Single VM | 6 VMs Parallel |
|--------|--------|-----------|----------------|
| 13,135 | 1,396,910 | ~2 hours | ~20 minutes |

### Performance (Tested on NY Metro - 219 grids, 47,961 routes)
| VM Type | Workers | Result | Notes |
|---------|---------|--------|-------|
| e2-medium (2 vCPU, 4GB) | 8 | Failed at ~17% | Server overwhelmed |
| c2d-highmem-4 (4 vCPU, 32GB) | 8 | **100% success** | ~4 minutes total |

**Recommendation**: Use c2d-highmem-4 or equivalent for polyline fetching.

### Script Configuration
Key settings in `fetch_polylines.py`:
```python
MAX_WORKERS = 8  # Concurrent Route API requests

# Line simplification (reduces file size by ~90%)
SIMPLIFY_TOLERANCE = 0.0001  # ~11 meters (Douglas-Peucker)
COORD_PRECISION = 5  # 5 decimal places = ~1.1m precision

# HTTPAdapter configured with:
#   pool_connections = max_workers + 5
#   pool_maxsize = max_workers + 5
#   max_retries = Retry(total=3, backoff_factor=0.1)
```

### Output Size (with simplification)
| Metric | Before | After |
|--------|--------|-------|
| File size | 2.3 GB | **223 MB** |
| Coordinates | 97M | 9.7M |
| Reduction | - | **90%** |

---

## 13. Troubleshooting

| Issue | Solution |
|-------|----------|
| OSRM preprocessing fails (OOM) | Increase VM RAM or split large cities |
| "Too many table coordinates" | Increase `--max-table-size` parameter |
| OSM file not found | Check country name mapping in script |
| Disk full | Run cleanup more aggressively or increase disk |
| VM unresponsive | Check GCP quotas, restart VM |
| "No grids for city" / H3 fails | H3 v4 API change - see below |
| Compression "Permission denied" | Docker creates files as root - run `sudo chown -R $(whoami) ~/osrm/` |
| "No edges remaining" in OSRM | City has no roads in OSM data - expected for small/remote areas |

### H3 Library v4 API Changes

If you see `module 'h3' has no attribute 'polyfill_geojson'`, the H3 library API changed in v4:

```python
# Old API (h3 v3)
h3_cells = h3.polyfill_geojson(geometry.__geo_interface__, resolution)
lat, lon = h3.h3_to_geo(cell)

# New API (h3 v4) - USE THIS
h3_cells = h3.geo_to_cells(geometry, resolution)
lat, lon = h3.cell_to_latlng(cell)
```

Check your version: `python3 -c "import h3; print(h3.__version__)"`

---

## 14. Quick Reference Scripts

All scripts are in the `scripts/` directory:

### Processing Scripts (Modular Pipeline)

| Script | Phase | Purpose | Run From |
|--------|-------|---------|----------|
| `common.py` | - | Shared utilities (logging, commands) | - |
| `clip_cities.py` | 1 | Clip OSM data to city boundaries | VM |
| `preprocess_cities.py` | 2 | OSRM extract/partition/customize | VM |
| `route_cities.py` | 3 | Compute matrices & fetch polylines | VM |
| `process_cities.py` | All | Wrapper for all 3 phases | VM |
| `fetch_polylines.py` | Post | Fetch route geometries (Pass 2) | VM |

### Management Scripts

| Script | Purpose | Run From |
|--------|---------|----------|
| `monitor_all.sh` | Check progress on all VMs | Local |
| `monitor_all.sh --loop` | Continuous monitoring (5 min refresh) | Local |
| `check_issues.sh` | Check for problems requiring attention | Local |
| `launch_all_processing.sh` | Check downloads and start processing | Local |
| `collect_results.sh` | Download all results from VMs | Local |
| `start_processing.sh` | Start processing (run ON the VM) | VM |

### Example Workflow

```bash
# From local machine:
cd scripts/

# 1. Check download progress
./monitor_all.sh

# 2. When downloads complete, start processing on each VM
# Option A: All-in-one (recommended for first run)
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="
    source ~/osrm_env/bin/activate
    nohup python3 ~/process_cities.py --region ~/cities.geojson --compress >> ~/processing.log 2>&1 &
"

# Option B: Phase by phase (for more control)
# Phase 1: Clip (run once)
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="
    source ~/osrm_env/bin/activate
    python3 ~/clip_cities.py --region ~/cities.geojson
"

# Phase 2: Preprocess (run once per profile)
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="
    source ~/osrm_env/bin/activate
    python3 ~/preprocess_cities.py --profile car --compress
"

# Phase 3: Route (can re-run with different options)
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="
    source ~/osrm_env/bin/activate
    nohup python3 ~/route_cities.py --region ~/cities.geojson --h3-resolution 7 --fetch-polylines >> ~/route.log 2>&1 &
"

# 3. Monitor progress
./monitor_all.sh --loop

# 4. Re-run routing with different H3 resolution (no re-preprocessing needed!)
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="
    source ~/osrm_env/bin/activate
    python3 ~/route_cities.py --region ~/cities.geojson --h3-resolution 6
"

# 5. When complete, collect all results
./collect_results.sh
```

### Manual VM Commands

```bash
# SSH into a specific VM
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap

# Check processing log
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="tail -50 ~/processing.log"

# Check completed count
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="ls ~/results/*.json | wc -l"

# Restart processing if it stopped
gcloud compute ssh osrm-india --zone=asia-south1-c --tunnel-through-iap --command="
    source ~/osrm_env/bin/activate
    nohup python3 ~/process_cities.py >> ~/processing.log 2>&1 &
"
```

---

*Last updated: 2025-12-24*
