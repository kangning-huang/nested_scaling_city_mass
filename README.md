# Nested Scaling of Urban Material Stocks

This repository contains the code and analysis pipeline for our manuscript on nested scaling relationships between population and built environment mass across cities and neighborhoods globally.

- **Status:** Under revision at *Nature Cities*
- **Preprint:** [arXiv:2507.03960](https://arxiv.org/abs/2507.03960)
- **Interactive explorer:** [https://kangning-huang.github.io/nested-scaling-city-mass/](https://kangning-huang.github.io/nested-scaling-city-mass/)
- **Repository:** [https://github.com/kangning-huang/nested-scaling-city-mass](https://github.com/kangning-huang/nested-scaling-city-mass)

## Key Findings

Urban material stocks (buildings, roads, pavement) scale **sublinearly** with population at two nested spatial scales:

| Scale | Exponent | 95% CI | N | Interpretation |
|-------|----------|--------|---|----------------|
| **City-level** | β = 0.900 | [0.890, 0.909] | 3,588 cities | A 1% increase in population is associated with only 0.9% more material stock |
| **Neighborhood-level** | δ = 0.713 | [0.712, 0.715] | 141,109 neighborhoods | Economies of scale are even stronger within cities |

The two exponents are statistically distinct (z = 39.1, p < 0.001), with confidence intervals that never overlap across any sensitivity specification. The result is robust to the choice of building volume dataset, material intensity assumptions, inclusion of underground infrastructure, spatial resolution, and grid positioning.

---

## Repository Structure

```
nested_scaling_city_mass/
├── scripts/
│   ├── data_pipeline/           # Full data processing pipeline (GEE → mass → figures)
│   │   ├── 01_create_h3_grids.py
│   │   ├── 02_extract_roads_neighborhood.py
│   │   ├── 03a_submit_batch_exports.py
│   │   ├── 04_merge_building_road_data.py
│   │   ├── 05_prep_global_mass_neighborhood.py
│   │   ├── Fig1_*.Rmd / Fig2_*.Rmd / Fig3_*.Rmd
│   │   ├── 06–10_*.py           # Zipf, simulation, Fig 4
│   │   ├── sensitivity/         # Original mixed-effects sensitivity analyses
│   │   └── utils/               # Shared path utilities
│   └── scaling_analysis/        # Revised scaling analysis (de-centering approach)
│       ├── Fig2_*.R             # City-level scaling + sensitivity
│       ├── Fig3_*.R             # Neighborhood-level scaling + sensitivity
│       ├── extract_subway_mass_by_hexagon.py
│       ├── test_zipf_vs_lognormal.py
│       ├── test_rank_correlation.py
│       └── web_prep/            # Scripts to prepare data for the interactive website
├── web/                         # Interactive web explorer (React + MapLibre + deck.gl)
├── config/                      # Path configuration for multi-environment support
└── tests/                       # Pipeline tests
```

---

## Data Sources

All input datasets are publicly available. Data files are not tracked in this repository due to size constraints.

### Primary Datasets

| Dataset | Source | Resolution | Access | Usage |
|---------|--------|------------|--------|-------|
| **GHSL Urban Centres Database (UCDB)** | [EU JRC](https://ghsl.jrc.ec.europa.eu/ghs_stat_ucdb2015mt_r2019a.php) | City polygons | Free download; also hosted on GEE as `users/kh3657/GHS_STAT_UCDB2015` | Defines 3,588 city boundaries worldwide |
| **WorldPop** | [WorldPop](https://www.worldpop.org/) | 100 m raster | GEE: `WorldPop/GP/100m/pop/2015` | Population estimates (2015) per H3 hexagon |
| **Esch et al. 2022 (WSF3D)** | [DLR](https://geoservice.dlr.de/web/maps/eoc:wsf3d) | 90 m | GEE: `DLR/WSF3D/v1` | Building heights and volumes from TanDEM-X radar |
| **Li et al. 2022** | [Zenodo](https://doi.org/10.5281/zenodo.5825801) | 1 km | GEE: hosted as user asset | Building height/volume from random forest ensemble |
| **Liu et al. 2024 (GUS3D)** | [Figshare](https://doi.org/10.6084/m9.figshare.24901575) | 500 m | GEE: hosted as user asset | Building volume from XGBoost with 11 regional models |
| **GRIP Global Roads** | [GRIP](https://www.globio.info/download-grip-dataset) | Raster (various) | Local raster files | Road length by functional class (highway, primary, secondary, tertiary, local) |
| **GAIA/GISA Impervious Surface** | [GEE](https://developers.google.com/earth-engine/datasets) | 30 m | GEE: `Tsinghua/GAIA/v1` (2015 band) | Impervious surface extent for pavement area estimation |

### Material Intensity (MI) Reference Data

| Dataset | Source | Granularity | Usage |
|---------|--------|-------------|-------|
| **Heeren & Fishman 2019** | [Scientific Data](https://doi.org/10.1038/s41597-019-0021-x) | Global average + 4 regions | Baseline MI values for converting building volume → mass |
| **Haberl et al. 2024** | Manuscript SI | 5 world regions | Sensitivity analysis: region-specific MI |
| **Fishman et al. 2024 (RASMI)** | Manuscript SI | 32 world regions | Sensitivity analysis: fine-grained regional MI |
| **Rousseau et al. 2022** | [ES&T](https://doi.org/10.1021/acs.est.2c05255) | Per-material breakdown | Cross-validation of MI values |
| **Wiedenhofer et al. 2023** | [Scientific Data](https://doi.org/10.1038/s41597-023-02565-0) | Global raster (roads + rails) | Mobility infrastructure mass validation |

### Supplementary Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| **Elhacham et al. 2020** | [Nature](https://doi.org/10.1038/s41586-020-3010-5) SI | Historical trend of anthropogenic mass vs biomass (Fig 1) |
| **CPTOND-2025** | Chinese Public Transport Open Dataset | China subway network shapefiles for underground infrastructure sensitivity |
| **OpenStreetMap** | [Overpass API](https://overpass-turbo.eu/) | Global subway line extraction for non-China cities |
| **Mao et al. 2021** | Literature | Subway material intensity coefficients (~19,500 t/km tunnel, ~170,000 t/station) |

### How Raw Data Were Obtained

1. **City boundaries:** Downloaded from EU JRC GHSL portal as GeoPackage; uploaded to GEE as FeatureCollection.
2. **Building volumes:** Three independent datasets accessed as GEE Image assets. WSF3D is a public GEE dataset; Li2022 and Liu2024 were downloaded from their respective repositories and ingested as private GEE assets.
3. **Population:** Accessed directly from GEE's public `WorldPop` ImageCollection.
4. **Road data:** GRIP rasters downloaded from globio.info and stored locally for `exactextract` zonal statistics.
5. **Impervious surface:** Accessed directly from GEE's public GAIA dataset.
6. **Material intensity:** Extracted from published supplementary information tables (Excel/CSV). Baseline MI lookup hardcoded in `05_prep_global_mass_neighborhood.py`.
7. **Subway networks:** China networks from CPTOND-2025 (AMap API crawl); global networks from OSM Overpass queries via `osm_subway_download.py`.

---

## Complete Data Processing Pipeline

The pipeline transforms raw geospatial data into scaling analysis results in 6 stages.

### Stage 1: Create Spatial Framework

**Script:** `scripts/data_pipeline/01_create_h3_grids.py`

Generates H3 hexagonal grids at configurable resolution over all 3,588 GHSL urban centres. Each city is tessellated with Uber's H3 hierarchical hexagonal grid system.

```
Input:  GEE FeatureCollection 'users/kh3657/GHS_STAT_UCDB2015'
Output: data/processed/h3_resolution{N}/all_cities_h3_grids.gpkg
```

- **Resolution 5:** ~253 km² hexagons (coarse; 8,267 neighborhoods)
- **Resolution 6:** ~36 km² hexagons (primary analysis; 34,060 neighborhoods)
- **Resolution 7:** ~5 km² hexagons (fine; 248,797 neighborhoods)

For spatial sensitivity testing, `01b_create_h3_grids_nudged.py` creates shifted grids by translating city boundaries 1 km in each cardinal direction before tessellation.

### Stage 2a: Extract Road Data

**Script:** `scripts/data_pipeline/02_extract_roads_neighborhood.py`

Computes road length (km) within each H3 hexagon using `exactextract` zonal statistics on GRIP raster files. Roads are categorized by functional class (highway, primary, secondary, tertiary, local).

```
Input:  all_cities_h3_grids.gpkg + GRIP rasters (local)
Output: Fig3_Roads_Neighborhood_H3_Resolution{N}_{date}.csv
```

`02b_extract_roads_clean.py` is an improved version adding lane-based road widths, surface areas, and climate classifications.

### Stage 2b: Extract Building Volumes and Pavement

**Scripts:**
- `scripts/data_pipeline/03a_submit_batch_exports.py` — submit GEE batch export tasks
- `scripts/data_pipeline/03b_monitor_batch_tasks.py` — monitor task completion
- `scripts/data_pipeline/03c_download_batch_results.py` — download and merge results

For each H3 hexagon, extracts from GEE:
- **Building volume** (m³) from three independent datasets (Esch2022, Li2022, Liu2024)
- **Impervious surface area** (m²) from GAIA/GISA
- **Population** from WorldPop 100m

All three building volume sources are extracted simultaneously and stored as separate columns, allowing post-hoc source selection and sensitivity analysis.

```
Input:  all_cities_h3_grids.gpkg + GEE raster datasets
Output: Fig3_Volume_Pavement_Neighborhood_H3_Resolution{N}_{date}.csv
```

The synchronous single-city alternative is `03_extract_volume_pavement.py` (slower but simpler). `03_run_extraction.py` dispatches between the two approaches.

### Stage 3: Merge Building and Road Data

**Script:** `scripts/data_pipeline/04_merge_building_road_data.py`

Joins volume/pavement extraction output with road extraction output on `h3index`.

```
Input:  Volume/pavement CSV + Roads CSV
Output: Fig3_Merged_Neighborhood_H3_Resolution{N}_{date}.csv
```

### Stage 4: Calculate Material Stocks

**Neighborhood-level:** `scripts/data_pipeline/05_prep_global_mass_neighborhood.py`

Converts physical quantities to material mass (tonnes) using region-specific material intensity (MI) values:

| Component | Formula | MI Source |
|-----------|---------|-----------|
| **Building mass** | volume (m³) × MI (kg/m³) | Heeren & Fishman 2019, stratified by building class (RS/RM/RI) and world region |
| **Road mass** | road_length (km) × lane_width (m) × depth (m) × density (kg/m³) | Literature values by road class |
| **Pavement mass** | impervious_area (m²) × depth (m) × density (kg/m³) | Standard pavement MI |

Building class assignment uses the ratio of residential to non-residential floor space from GHSL data. MI lookup provides values for 5 world regions × 3 building classes.

```
Input:  Fig3_Merged_Neighborhood_H3_Resolution{N}_{date}.csv
Output: Fig3_Mass_Neighborhood_H3_Resolution{N}_{date}.csv
```

**City-level:** `scripts/data_pipeline/Fig1_DataPrep_GlobalMass_MergedMI_.Rmd` (R Markdown)

Equivalent process for city-level aggregates, also incorporating biomass estimates.

```
Input:  merged_building_road_otherpavement.csv + biomass_by_cities.csv
Output: MasterMass_ByClass20250616.csv (3,588 cities)
```

### Stage 5: Scaling Analysis

**Scripts:** `scripts/scaling_analysis/Fig2_UniversalScaling_Decentered.R` and `Fig3_NeighborhoodScaling_Decentered.R`

Following Bettencourt & Lobo (2016), we use within-group de-centering to remove country/city baseline differences before pooled OLS:

```r
# City-level: de-center by country mean
group_by(CTR_MN_NM) %>%
  mutate(log_pop_c = log10(pop) - mean(log10(pop)),
         log_mass_c = log10(mass) - mean(log10(mass)))

# Neighborhood-level: de-center by city mean
group_by(ID_HDC_G0) %>%
  mutate(log_pop_c = log10(pop) - mean(log10(pop)),
         log_mass_c = log10(mass) - mean(log10(mass)))

# OLS on pooled de-centered data
lm(log_mass_c ~ log_pop_c, data = pooled)
```

This is algebraically equivalent to the within-group fixed-effects estimator (Frisch-Waugh-Lovell theorem) and produces virtually identical results to the mixed-effects approach.

### Stage 6: Figures and Simulation

| Script | Output |
|--------|--------|
| `Fig1_GlobalMassboxplot_Assemble_MIUpdate.Rmd` | **Figure 1:** Global mass overview with boxplots and historical trend |
| `Fig2_UniversalScaling_Decentered.R` | **Figure 2:** City-level scaling (pop vs mass scatter with OLS) |
| `Fig3_NeighborhoodScaling_Decentered.R` | **Figure 3:** Neighborhood-level scaling |
| `06_estimate_neighborhood_zipf.py` | Zipf exponent (s) for each city's population distribution |
| `07_simulate_scaling.py` | Monte Carlo simulation: neighborhood δ + Zipf s → predicted city β |
| `08_compare_beta_boxplot.py` | Observed vs simulated β comparison |
| `10_generate_fig4.py` | **Figure 4:** Assembly of Zipf, simulation, and comparison panels |

---

## Sensitivity Analyses

All sensitivity scripts are in `scripts/scaling_analysis/`. Each test shows that the sublinear finding (β < 1, δ < 1) is robust.

### Building Volume Data Source (Reviewer 1, Comment #1)

Tests whether the choice among three independent building volume datasets affects the scaling exponent.

| Script | What it tests |
|--------|---------------|
| `Fig2_*_Source_Sensitivity.R` | City β with each source individually + random selection (100 iterations) |
| `Fig3_*_Source_Sensitivity.R` | Neighborhood δ with each source individually |
| `Fig2_*_Weighted_Source.R` | Reliability-weighted averaging across sources |

**Result:** City β range = 0.024 (0.892–0.916). All sublinear regardless of source.

### Material Intensity Assumptions (Reviewer 1 #2, Reviewer 3 #3)

Tests three MI frameworks: global average, Haberl 5-region, Fishman 32-region RASMI.

| Script | Level |
|--------|-------|
| `Fig2_*_MI_sensitivity.R` | City-level |
| `Fig3_*_MI_sensitivity.R` / `*_Multiscale_MI_sensitivity.R` | Neighborhood-level |

**Result:** β range = 0.004 (negligible). MI is multiplicative and absorbed by de-centering.

### Underground Infrastructure (Reviewer 1 #3, Reviewer 3 #5)

Tests whether omitting subway infrastructure biases the exponent.

| Script | Purpose |
|--------|---------|
| `extract_subway_mass_by_hexagon.py` | Extract subway mass from CPTOND-2025 (China) + OSM (global) |
| `osm_subway_download.py` | Download subway networks from OpenStreetMap |
| `Fig2_*_WithSubwayMass.R` | City-level with subway mass added |
| `Fig3_*_WithSubwayMass.R` | Neighborhood-level with subway mass added |

**Result:** Δβ = +0.003, Δδ = +0.001. Underground infrastructure accounts for < 1% of total mass.

### Spatial Resolution and Grid Placement (Reviewer 1 #6, Reviewer 3 #4)

Tests sensitivity to H3 hexagon resolution and MAUP (Modifiable Areal Unit Problem).

| Script | What it tests |
|--------|---------------|
| `Fig3_*_Multiscale_R6Filter.R` | δ across H3 Resolutions 5, 6, 7 |
| `Fig3_*_NudgeSensitivity.R` | δ with grid shifted 1 km N/S/E/W |

**Result:** δ decreases monotonically with finer resolution (0.826 → 0.751 → 0.713 for R5 → R6 → R7). Nudging produces max deviation of 0.003.

### Statistical Distribution Tests (Reviewer 1 #7)

| Script | What it tests |
|--------|---------------|
| `test_zipf_vs_lognormal.py` | Power-law vs lognormal vs truncated power-law (Clauset-Shalizi-Newman framework) |
| `test_rank_correlation.py` | Rank correspondence and permutation tests |

**Result:** Power law strongly preferred over lognormal (Vuong R = 28.9 for population, p < 0.001).

### Neighborhood Variability (Reviewer 1 #8)

| Script | Output |
|--------|--------|
| `Fig3_ExtendedData_CityLines.R` | Extended Data figure overlaying all 3,312 city OLS lines |
| `Fig3_R7_city_candidates*.R` | Per-city slope analysis and candidate identification |

**Result:** 99.97% of cities (3,311/3,312) have positive scaling slopes.

---

## Filtering Criteria

All neighborhood analyses apply these filters:

1. **Population ≥ 1** per hexagon (excludes fractional/impossible values)
2. **City total population > 50,000**
3. **Cities with ≥ 10 qualifying neighborhoods** (ensures reliable per-city regression)
4. **Countries with ≥ 5 qualifying cities** (ensures reliable per-country statistics)

---

## Interactive Website

The [live explorer](https://kangning-huang.github.io/nested-scaling-city-mass/) provides:

- **Map view:** Navigate Global → Country → City. City view renders H3 R7 hexagons colored by population or built mass on a log scale.
- **City panel:** City-level scatter of log(pop) vs log(mass) with OLS regression and 95% CI.
- **Neighborhood panel:** Neighborhood-level scatter with density overlay option.

### Website Data Preparation

Scripts in `scripts/scaling_analysis/web_prep/` transform the analysis outputs into static JSON artifacts:

| Script | Output |
|--------|--------|
| `prep_city_aggregates.py` | City summaries per country |
| `prep_neighborhood_subsamples.py` | Stratified subsamples for scatter plots |
| `compute_regressions.py` | OLS slopes with 95% CIs and decentered anchors |
| `split_city_hex_feeds.py` | Per-city H3 hex feeds for map rendering |
| `download_countries_geojson.py` | Country boundary GeoJSON |

### Website Stack

- **Frontend:** Vite + React + MapLibre GL + deck.gl (H3HexagonLayer)
- **Basemap:** MapTiler light basemap
- **Deployment:** GitHub Pages via `.github/workflows/deploy.yml`

### Local Development

```bash
cd web && npm ci
export VITE_MAPTILER_KEY=your_key  # free at maptiler.com
npm run dev
```

### Data Dictionary

| File | Fields |
|------|--------|
| `webdata/cities_agg/*.json` | `country_iso`, `city_id`, `city`, `pop_total`, `mass_total`, `lat`, `lon`, `log_pop`, `log_mass` |
| `webdata/scatter_samples/*.json` | `country_iso`, `city_id`, `log_pop`, `log_mass` |
| `webdata/regression/*.json` | `slope`, `slope_lo`, `slope_hi`, `x0`, `y0`, `n`, `r2` |
| `webdata/hex/city=ID.json` | `h3index`, `population_2015`, `total_built_mass_tons`, `city_id`, `country_iso` |

Regression lines use: `y = y0 + slope * (x - x0)`. All logs are base-10; population in persons; mass in metric tonnes.

---

## Environment Setup

### Python

```bash
python3 -m venv ~/.venvs/urban_scaling_env
source ~/.venvs/urban_scaling_env/bin/activate
pip install pandas geopandas numpy scipy h3 tobler geemap earthengine-api rasterio exactextract statsmodels powerlaw matplotlib seaborn
```

### R

```r
install.packages(c("tidyverse", "lme4", "sf", "ggplot2", "scales", "patchwork", "broom"))
```

### Google Earth Engine

```bash
earthengine authenticate
# GEE Project ID: ee-knhuang
```

---

## Reproducing Results

### Quick Start (figures only, requires processed data)

```bash
# City-level scaling (Figure 2)
cd scripts/scaling_analysis
Rscript Fig2_UniversalScaling_Decentered.R

# Neighborhood-level scaling (Figure 3)
Rscript Fig3_NeighborhoodScaling_Decentered.R

# Simulation analysis (Figure 4)
cd ../data_pipeline
python 10_generate_fig4.py
```

### Full Pipeline (requires GEE authentication and raw data)

```bash
cd scripts/data_pipeline

# Stage 1: Create H3 grids
python 01_create_h3_grids.py --resolution 6

# Stage 2a: Extract roads (requires local GRIP rasters)
python 02_extract_roads_neighborhood.py

# Stage 2b: Extract building volumes + pavement (GEE batch)
python 03a_submit_batch_exports.py
python 03b_monitor_batch_tasks.py --watch     # wait for completion
python 03c_download_batch_results.py

# Stage 3: Merge
python 04_merge_building_road_data.py

# Stage 4: Calculate mass
python 05_prep_global_mass_neighborhood.py

# Stage 5: Scaling analysis
cd ../scaling_analysis
Rscript Fig2_UniversalScaling_Decentered.R
Rscript Fig3_NeighborhoodScaling_Decentered.R

# Stage 6: Simulation
cd ../data_pipeline
python 06_estimate_neighborhood_zipf.py
python 07_simulate_scaling.py
python 10_generate_fig4.py
```

### Sensitivity Analyses

```bash
cd scripts/scaling_analysis

# Data source sensitivity
Rscript Fig2_UniversalScaling_Decentered_Source_Sensitivity.R
Rscript Fig3_NeighborhoodScaling_Decentered_Source_Sensitivity.R

# Material intensity sensitivity
Rscript Fig2_UniversalScaling_Decentered_MI_sensitivity.R

# Underground infrastructure
python extract_subway_mass_by_hexagon.py
Rscript Fig2_UniversalScaling_Decentered_WithSubwayMass.R

# Multiscale (H3 R5/R6/R7)
Rscript Fig3_NeighborhoodScaling_Decentered_Multiscale_R6Filter.R

# Grid nudge sensitivity
Rscript Fig3_NeighborhoodScaling_Decentered_NudgeSensitivity.R

# Zipf's law test
python test_zipf_vs_lognormal.py
python test_rank_correlation.py
```

---

## Citation

If you use this code or data, please cite:

> Huang, K. & Lu, M. (2025). Nested scaling laws of urban material stocks. *arXiv:2507.03960*. [https://arxiv.org/abs/2507.03960](https://arxiv.org/abs/2507.03960)

---

## License

This project is shared for research reproducibility. Please contact the authors for commercial use.
