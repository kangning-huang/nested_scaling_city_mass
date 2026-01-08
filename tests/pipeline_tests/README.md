# Pipeline Validation Tests

Tests to validate each step of the urban scaling analysis pipeline after reorganization.

## Quick Start

```bash
# Activate environment
source ~/.venvs/nyu_china_grant_env/bin/activate

# Run all tests from project root
python tests/run_pipeline_tests.py

# Run specific steps
python tests/run_pipeline_tests.py --step 0 1 2

# Quick smoke test
python tests/run_pipeline_tests.py --quick

# List available tests
python tests/run_pipeline_tests.py --list
```

## Test Steps

| Step | Name | Description |
|------|------|-------------|
| 0 | Environment | Dependencies, project structure, config module |
| 1 | Data Extraction | GHS database, city lookup, POI files |
| 2 | Preprocessing | H3 grids, SNDi raster, processed outputs |
| 3 | Analysis | Script syntax, scaling results, centrality |
| 4 | Cloud/GEE | GEE scripts syntax, authentication |
| 5 | HPC | OSRM scripts, SLURM templates |
| 6 | Integration | End-to-end mini pipeline test |

## Running Individual Modules

```bash
python tests/pipeline_tests/test_01_data_extraction.py
python tests/pipeline_tests/test_02_preprocessing.py
python tests/pipeline_tests/test_03_analysis.py
python tests/pipeline_tests/test_04_cloud_gee.py
python tests/pipeline_tests/test_05_hpc.py

# With live tests (slower)
python tests/pipeline_tests/test_01_data_extraction.py --live
python tests/pipeline_tests/test_04_cloud_gee.py --live
```

## Test City

Default test city: **Ahmedabad, India** (ID: 6694)
- Small dataset (~500KB POIs)
- Fast processing
- Good data coverage

## Output Format

```
✓ Test passed
✗ Test failed
⚠ Warning (non-critical)
```

Exit code `0` = all passed, `1` = failures
