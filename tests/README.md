# OpenStreetMap POI Extraction Test

This directory contains test scripts for extracting Points of Interest (POI) from OpenStreetMap using the `osmnx` library.

## Test Overview

Successfully tested POI extraction for **New York City** using the city boundary from `all_cities_h3_grids.gpkg` across 9 functional categories:

1. **Outdoor Activities** - Parks, playgrounds, gardens, viewpoints
2. **Learning** - Schools, universities, libraries
3. **Supplies** - Supermarkets, convenience stores, bakeries
4. **Eating** - Restaurants, cafes, bars
5. **Moving** - Transit stops, stations
6. **Cultural Activities** - Museums, theaters, galleries
7. **Physical Exercise** - Sports centers, fitness facilities
8. **Services** - Banks, pharmacies, post offices
9. **Health Care** - Hospitals, clinics, doctors

## Results Summary

**Total POIs Extracted: 164,551**

Using the H3 grid-based boundary from `all_cities_h3_grids.gpkg` (ID_HDC_G0 = 945):

| Category | Count | Percentage |
|----------|-------|------------|
| Physical Exercise | 77,031 | 46.8% |
| Eating | 24,589 | 14.9% |
| Outdoor Activities | 20,966 | 12.7% |
| Moving | 16,897 | 10.3% |
| Supplies | 7,716 | 4.7% |
| Learning | 6,306 | 3.8% |
| Services | 5,728 | 3.5% |
| Health Care | 3,875 | 2.4% |
| Cultural Activities | 1,443 | 0.9% |

**Note**: This extraction yielded ~52% more POIs (164,551 vs 108,066) compared to using OSMnx's geocoded boundary, as the H3 grid-based boundary covers a more complete metropolitan area including surrounding communities.

## Files

- **`test_extract_nyc_pois.py`** - Main extraction script (uses boundary from GeoPackage)
- **`verify_poi_data.py`** - Data verification script
- **`nyc_pois_9cats.gpkg`** - Output GeoPackage (49 MB, 164K POIs)
- **`nyc_boundary.gpkg`** - NYC city boundary (dissolved from 219 H3 cells)
- **`nyc_pois_summary.csv`** - Summary statistics

## How to Run

### Prerequisites

```bash
# Activate virtual environment
source ~/.venvs/nyu_china_grant_env/bin/activate

# Install required packages (if not already installed)
pip install osmnx geopandas pandas
```

### Extract POIs

```bash
cd "0.CleanProject_Building_v_Mobility/tests"
python test_extract_nyc_pois.py
```

### Verify Results

```bash
python verify_poi_data.py
```

## Script Features

### `test_extract_nyc_pois.py`

- Automatically fetches city boundary from OSM using geocoding
- Queries each POI category systematically
- Handles errors gracefully (continues if some queries fail)
- Cleans column names to avoid field name issues
- Saves essential columns: category, OSM key/value, name, address, geometry
- Generates summary statistics

### POI Tag Configuration

The script uses the `POI_TAGS` dictionary to define OSM tags for each category:

```python
POI_TAGS = {
    "outdoor_activities": {
        "leisure": ["park", "playground", "garden", "dog_park", "nature_reserve"],
        "tourism": ["viewpoint"]
    },
    # ... (see test_extract_nyc_pois.py for full configuration)
}
```

## Data Quality

### Geometry Types
- **Polygons**: 105,449 (64.1%) - Buildings, parks, areas
- **Points**: 58,805 (35.7%) - Single location POIs
- **MultiPolygons**: 294 (0.2%) - Complex areas
- **LineStrings**: 3 (0.0%) - Linear features

### Data Completeness
- **Name**: 43.7% of POIs have names
- **Street Address**: 20.2% have street names
- **House Number**: 19.8% have house numbers
- **City**: 13.3% have city names

## Adapting for Other Cities

The script now uses the `all_cities_h3_grids.gpkg` file by default. To extract POIs for other cities:

### Option 1: Use a city from the GeoPackage

Find your city's `ID_HDC_G0` value and modify the main execution block:

```python
# In test_extract_nyc_pois.py, change:
extract_nyc_pois(
    output_dir=output_directory,
    boundary_file=str(boundary_file_path),
    city_id=945  # Change to your city's ID_HDC_G0
)
```

### Option 2: Use OSM geocoding (fallback)

If the boundary file is not available, the script will automatically fall back to OSM geocoding:

```python
# Simply don't provide a boundary_file, or set it to None
extract_nyc_pois(
    output_dir=output_directory,
    boundary_file=None  # Will use OSM geocoding
)
```

### Option 3: Custom polygon from file

```python
# Modify the script to load your custom boundary
city = gpd.read_file("your_city_boundary.gpkg").to_crs("EPSG:4326")
polygon = city.geometry.union_all()
```

## Technical Notes

### Key Considerations

1. **Rate Limiting**: OSM Overpass API has rate limits. Large cities may take time.
2. **Memory**: Large extractions can be memory-intensive
3. **Field Names**: Some OSM tags contain special characters (`:`, `/`) that can cause issues with certain file formats
4. **Missing Queries**: Some specific tags may not exist in all cities (e.g., `highway=subway_entrance`)

### Column Selection Strategy

The script keeps only essential columns to avoid issues:
- Core: `geometry`, `category`, `osm_key`, `osm_value`
- Optional: `name`, address fields, OSM ID

This approach prevents errors from problematic field names like `payment:NFC_mobile_payments`.

## Future Improvements

Possible enhancements:
- [ ] Add spatial filtering by administrative boundaries
- [ ] Include temporal tags (opening hours)
- [ ] Add quality scores based on data completeness
- [ ] Export to multiple formats (GeoJSON, Shapefile, CSV)
- [ ] Add distance calculations to nearest POI
- [ ] Create summary maps/visualizations

## References

- **OSMnx Documentation**: https://osmnx.readthedocs.io/
- **OpenStreetMap Tags**: https://wiki.openstreetmap.org/wiki/Map_features
- **Overpass API**: https://wiki.openstreetmap.org/wiki/Overpass_API
