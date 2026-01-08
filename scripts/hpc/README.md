# HPC Scripts - NYUSH HPC & NYU Greene

This directory contains scripts and resources for running compute-intensive tasks on HPC clusters, primarily OSRM-based routing and centrality calculations.

## Directory Structure

```
hpc/
├── README.md                       # This file
├── OSRM_NYUSH_HPC_guide.md        # Detailed NYUSH HPC setup guide
├── OSRM_NYU_HPC_Greene_guide.md   # NYU Greene HPC guide
├── OSRM_gcloud_setup_guide.md     # Google Cloud alternative
├── OSRM_parallel_processing.md    # Parallel processing strategies
├── CLAUDE.md                       # AI assistant context for HPC work
│
├── osrm_setup/                     # Setup and installation scripts
│   ├── install_dependencies.sh
│   ├── download_osm_*.sh          # OSM data download scripts
│   ├── start_processing.sh
│   └── launch_all_processing.sh
│
├── osrm_scripts/                   # Main Python scripts for HPC
│   ├── route_cities.py            # City routing calculations
│   ├── route_cities_res7.py       # Resolution 7 routing
│   ├── calculate_centrality_hpc.py
│   ├── calculate_centrality_res7.py
│   ├── fetch_polylines_hpc.py
│   ├── common.py                  # Shared utilities
│   └── ...
│
├── slurm_templates/                # SLURM job submission scripts
│   ├── process_country.slurm
│   ├── process_pilot.slurm
│   ├── calculate_centrality_test.slurm
│   └── ...
│
├── cities/                         # City boundary files
├── city_lists/                     # City list CSVs by country/region
├── regions/                        # Regional OSM extracts config
├── osrm_data/                      # OSRM preprocessed data
├── osrm_results/                   # Output results
└── test_results/                   # Test outputs
```

## Quick Start - NYUSH HPC

### 1. Connect to HPC
```bash
# SSH access
ssh kh3657@hpc.shanghai.nyu.edu

# Or use web portal
# https://ood.shanghai.nyu.edu/hpc/
```

### 2. Set Up Environment
```bash
# Create workspace on scratch (fast storage)
mkdir -p /scratch/kh3657/osrm
cd /scratch/kh3657/osrm

# Load required modules
module load Singularity/4.3.1-gcc-8.5.0
module load anaconda3/2023.09

# Create conda environment
conda create -n urban_scaling python=3.10 geopandas pandas numpy requests tqdm -y
conda activate urban_scaling
```

### 3. Transfer Data
```bash
# From local machine to HPC
rsync -avzP data/raw/pois/ kh3657@hpc.shanghai.nyu.edu:/scratch/kh3657/osrm/pois/
rsync -avzP scripts/hpc/ kh3657@hpc.shanghai.nyu.edu:/scratch/kh3657/osrm/scripts/
```

### 4. Submit Jobs
```bash
# Single city test
sbatch slurm_templates/process_pilot.slurm

# Full country processing
sbatch slurm_templates/process_country.slurm --export=COUNTRY=china
```

### 5. Monitor Jobs
```bash
# Check job status
squeue -u kh3657

# View job output
tail -f slurm-<jobid>.out

# Cancel job
scancel <jobid>
```

### 6. Retrieve Results
```bash
# From local machine
rsync -avzP kh3657@hpc.shanghai.nyu.edu:/scratch/kh3657/osrm/results/ results/hpc_outputs/
```

## Key Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `route_cities.py` | Calculate routing matrices between H3 cells | `python route_cities.py --city Shanghai` |
| `calculate_centrality_hpc.py` | Compute graph centrality metrics | `python calculate_centrality_hpc.py --city Shanghai` |
| `fetch_polylines_hpc.py` | Extract route geometries | `python fetch_polylines_hpc.py --city Shanghai` |
| `calculate_centrality_res7.py` | Resolution 7 centrality | For finer spatial analysis |

## OSRM Server Setup

OSRM requires preprocessed OSM data. On HPC, we use Singularity containers:

```bash
# Pull OSRM Docker image as Singularity
singularity pull docker://osrm/osrm-backend:latest

# Preprocess OSM data (one-time per region)
singularity exec osrm-backend_latest.sif osrm-extract -p /opt/car.lua china-latest.osm.pbf
singularity exec osrm-backend_latest.sif osrm-partition china-latest.osrm
singularity exec osrm-backend_latest.sif osrm-customize china-latest.osrm

# Start OSRM server (in job script)
singularity exec osrm-backend_latest.sif osrm-routed --algorithm=MLD china-latest.osrm
```

## Resource Requirements

| Task | CPUs | Memory | Time | Notes |
|------|------|--------|------|-------|
| OSM preprocessing | 4 | 16GB | 30min | Per country |
| City routing (small) | 2 | 8GB | 10min | <100k nodes |
| City routing (large) | 4 | 32GB | 2hr | >500k nodes |
| Centrality calculation | 8 | 64GB | 4hr | Memory-intensive |

## Detailed Guides

For comprehensive instructions, see:

- **NYUSH HPC**: [OSRM_NYUSH_HPC_guide.md](OSRM_NYUSH_HPC_guide.md)
- **NYU Greene**: [OSRM_NYU_HPC_Greene_guide.md](OSRM_NYU_HPC_Greene_guide.md)
- **Google Cloud**: [OSRM_gcloud_setup_guide.md](OSRM_gcloud_setup_guide.md)
- **Parallel Processing**: [OSRM_parallel_processing.md](OSRM_parallel_processing.md)

## Troubleshooting

### Common Issues

**Job fails with OOM (Out of Memory)**
```bash
# Increase memory in SLURM script
#SBATCH --mem=64G
```

**Singularity permission denied**
```bash
# Ensure image is in scratch, not home
cp osrm-backend_latest.sif /scratch/kh3657/
```

**OSRM server connection refused**
```bash
# Check if server is running
curl http://localhost:5000/health

# Start server manually
singularity exec osrm-backend_latest.sif osrm-routed --algorithm=MLD /path/to/data.osrm &
```

### Getting Help

- NYUSH HPC docs: https://hpc.shanghai.nyu.edu/
- SLURM documentation: https://slurm.schedmd.com/
- OSRM API docs: http://project-osrm.org/docs/
