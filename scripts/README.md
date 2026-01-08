# Scripts Pipeline

This directory contains all analysis scripts organized by pipeline stage and compute environment.

## Directory Structure

```
scripts/
├── 01_data_extraction/     # Data download and extraction (LOCAL)
├── 02_preprocessing/       # Data cleaning and preparation (LOCAL)
├── 03_analysis/            # Core analysis and modeling (LOCAL)
├── 04_visualization/       # Figure generation (LOCAL)
├── hpc/                    # NYUSH HPC scripts (OSRM routing, etc.)
├── cloud/                  # Cloud platform scripts (GEE)
├── shared/                 # Reusable utilities
└── archive/                # Deprecated scripts
```

## Environment Matrix

| Script | Environment | Dependencies | Notes |
|--------|-------------|--------------|-------|
| **01_data_extraction/** | | | |
| `01_download_OSM_city_pois.py` | Local | osmnx, geopandas | ~5 min per city |
| `01b_batch_download_pois.py` | Local | osmnx, geopandas | Batch processing |
| **02_preprocessing/** | | | |
| `01_neighborhood_SNDi.py` | Local | pandas, rasterio, exactextract | SNDi calculation |
| `02_sndi_cmi_analysis.py` | Local | pandas, geopandas | Join SNDi with CMI |
| **03_analysis/** | | | |
| `01_neighborhood_scaling_building_v_mobility.R` | Local | R: tidyverse, sf | Scaling regression |
| `01_neighborhood_building_v_mobility_ratio.R` | Local | R: tidyverse, sf | Ratio calculations |
| `02_neighborhood_scaling_building_v_mobility_deviation.R` | Local | R: tidyverse | Deviation analysis |
| `02_enhanced_global_sndi_scaling_with_cv.py` | Local | pandas, scipy, sklearn | Cross-validation |
| `02_global_neighborhood_scaling_sndi_classification.py` | Local | pandas, sklearn | Classification |
| `02_accessibility_POIs_neighborhoods.py` | Local | geopandas | POI accessibility |
| `04_travelTime_POIs_building_v_mobility_scaling_deviation.py` | Local | pandas, geopandas | Travel time analysis |
| `calculate_population_weighted_centrality.py` | Local | pandas, geopandas | Centrality metrics |
| `centrality_vs_infrastructure_mass.py` | Local | pandas | Infrastructure analysis |
| `centrality_vs_mobility_mass.py` | Local | pandas | Mobility analysis |
| **cloud/gee/** | | | |
| `01_GEE_frictionSurface_travelTimes.py` | GEE Cloud | geemap, ee | Friction surface |
| `02_upload_pois_to_gee.py` | GEE Cloud | geemap, ee | Asset upload |
| `03_GEE_travelTime_POIs_30m.py` | GEE Cloud | geemap, ee | Travel time rasters |
| `GEE_travelTimeToPOIs_30m.js` | GEE Code Editor | ee | JavaScript version |
| **hpc/osrm_scripts/** | | | |
| `route_cities.py` | **NYUSH HPC** | OSRM, geopandas | Routing calculations |
| `calculate_centrality_hpc.py` | **NYUSH HPC** | pandas, networkx | Graph centrality |
| `fetch_polylines_hpc.py` | **NYUSH HPC** | OSRM client | Route geometries |

## Pipeline Execution Order

### Phase 1: Data Extraction
```bash
# Local machine
python scripts/01_data_extraction/01_download_OSM_city_pois.py
python scripts/01_data_extraction/01b_batch_download_pois.py
```

### Phase 2: Preprocessing
```bash
# Local machine
python scripts/02_preprocessing/01_neighborhood_SNDi.py
python scripts/02_preprocessing/02_sndi_cmi_analysis.py
```

### Phase 3: Cloud Processing (GEE)
```bash
# Requires GEE authentication
python scripts/cloud/gee/01_GEE_frictionSurface_travelTimes.py
python scripts/cloud/gee/02_upload_pois_to_gee.py
python scripts/cloud/gee/03_GEE_travelTime_POIs_30m.py
```

### Phase 4: HPC Processing (OSRM)
```bash
# On NYUSH HPC - see hpc/README.md for setup
sbatch scripts/hpc/slurm_templates/process_country.slurm
```

### Phase 5: Analysis
```bash
# Local machine - R scripts
Rscript scripts/03_analysis/01_neighborhood_scaling_building_v_mobility.R
Rscript scripts/03_analysis/01_neighborhood_building_v_mobility_ratio.R
Rscript scripts/03_analysis/02_neighborhood_scaling_building_v_mobility_deviation.R

# Local machine - Python scripts
python scripts/03_analysis/02_enhanced_global_sndi_scaling_with_cv.py
python scripts/03_analysis/calculate_population_weighted_centrality.py
```

### Phase 6: Visualization
```bash
# Local machine
python scripts/04_visualization/*.py
```

## Environment Setup

### Local Environment
```bash
source ~/.venvs/nyu_china_grant_env/bin/activate
pip install -r requirements.txt
```

### GEE Authentication
```bash
earthengine authenticate
```

### HPC Setup
See `hpc/README.md` for NYUSH HPC configuration and SLURM job submission.

## Data Flow

```
Raw Data (data/raw/)
    │
    ▼
[01_data_extraction] ──► POIs, boundaries
    │
    ▼
[02_preprocessing] ──► SNDi, CMI joined data
    │
    ├──► [cloud/gee] ──► Travel time rasters, friction surfaces
    │
    └──► [hpc/osrm] ──► Routing matrices, centrality
    │
    ▼
[03_analysis] ──► Scaling results, deviations
    │
    ▼
[04_visualization] ──► Figures (figures/)
    │
    ▼
Results (results/)
```
