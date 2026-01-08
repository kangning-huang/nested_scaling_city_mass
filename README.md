# Nested Scaling of City Mass

Analysis of urban scaling laws and material stocks across global cities, with focus on neighborhood-level patterns and infrastructure-mobility relationships.

## Project Overview

This research project examines:
- Urban scaling laws for building and mobility infrastructure
- Material stock intensities at city and neighborhood scales
- Relationships between network centrality and material allocation
- Cross-country comparisons of urban development patterns

## Repository Structure

```
├── config/                 # Configuration and path definitions
│   └── paths.py           # Environment-aware path configuration
│
├── scripts/
│   ├── 01_data_extraction/    # POI and boundary extraction (LOCAL)
│   ├── 02_preprocessing/      # SNDi and data preparation (LOCAL)
│   ├── 03_analysis/           # Scaling and centrality analysis (LOCAL)
│   ├── 04_visualization/      # Figure generation (LOCAL)
│   ├── cloud/gee/             # Google Earth Engine scripts (CLOUD)
│   ├── hpc/                   # NYUSH HPC scripts (OSRM routing)
│   ├── shared/                # Common utilities
│   └── archive/               # Deprecated scripts
│
├── tests/
│   ├── run_pipeline_tests.py  # Main test runner
│   └── pipeline_tests/        # Step-by-step validation tests
│
└── docs/                      # Documentation and session logs
```

## Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/.venvs/nyu_china_grant_env
source ~/.venvs/nyu_china_grant_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Google Earth Engine
earthengine authenticate
```

## Data Sources

- **GHSL Urban Centres Database**: City boundary definitions
- **WorldPop**: Population estimates at 100m resolution
- **OpenStreetMap**: Points of Interest (POIs)
- **Zhou2022**: Global building heights/volumes
- **Sun2023**: China material stocks
- **GRIP**: Global Roads Inventory Project

**Note**: Data files are stored separately due to size (not in this repository).

## Pipeline Execution

See `scripts/README.md` for detailed pipeline documentation.

### Quick Test
```bash
python tests/run_pipeline_tests.py --quick
```

### Full Pipeline
```bash
# Step 1: Data extraction
python scripts/01_data_extraction/01_download_OSM_city_pois.py

# Step 2: Preprocessing
python scripts/02_preprocessing/01_neighborhood_SNDi.py

# Step 3: Analysis
python scripts/03_analysis/calculate_population_weighted_centrality.py
Rscript scripts/03_analysis/01_neighborhood_scaling_building_v_mobility.R

# Cloud processing (GEE)
python scripts/cloud/gee/03_GEE_travelTime_POIs_30m.py

# HPC processing (OSRM) - see scripts/hpc/README.md
```

## Key Dependencies

- Python: pandas, geopandas, numpy, scipy, h3, osmnx, rasterio
- R: tidyverse, sf, ggplot2
- Google Earth Engine: earthengine-api, geemap
- HPC: OSRM, Singularity

## License

This project is part of research conducted at NYU Shanghai.

## Contact

Kangning Huang - kh3657@nyu.edu
