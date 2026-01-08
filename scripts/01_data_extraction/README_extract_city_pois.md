# POI Extraction Script

`extract_city_pois.py` - Extract Points of Interest from OpenStreetMap for any city in the GHS Urban Centers Database

## Overview

This script automatically extracts 9 categories of Points of Interest (POI) from OpenStreetMap for any city in the Global Human Settlement (GHS) Urban Centers Database. It uses city boundaries from the GHS database and queries the Overpass API through the `osmnx` library.

## Features

- ✅ Supports all 13,135 cities in the GHS Urban Centers Database
- ✅ Extracts 9 functional POI categories
- ✅ Automatic city lookup by name (with optional country filter)
- ✅ Saves results to GeoPackage format with essential attributes
- ✅ Generates summary statistics (CSV)
- ✅ Clean, formatted console output with progress indicators
- ✅ Error handling for missing features and API issues

## POI Categories

1. **Outdoor Activities** - Parks, playgrounds, gardens, viewpoints
2. **Learning** - Schools, universities, libraries
3. **Supplies** - Supermarkets, convenience stores, bakeries
4. **Eating** - Restaurants, cafes, bars
5. **Moving** - Transit stops, bus stations, ferry terminals
6. **Cultural Activities** - Museums, theaters, galleries
7. **Physical Exercise** - Sports centers, fitness facilities
8. **Services** - Banks, pharmacies, post offices
9. **Health Care** - Hospitals, clinics, doctors

## Installation

Ensure the required packages are installed:

```bash
source ~/.venvs/nyu_china_grant_env/bin/activate
pip install osmnx geopandas pandas
```

## Usage

### Basic Usage

Extract POIs for a city:

```bash
python extract_city_pois.py "New York" --country "United States"
```

### Examples

```bash
# Extract POIs for Reykjavik
python extract_city_pois.py "Reykjavik" --country "Iceland"

# Extract POIs for London (no country needed if unambiguous)
python extract_city_pois.py "London" --country "United Kingdom"

# Specify custom output directory
python extract_city_pois.py "Paris" --country "France" --output-dir ~/my_pois

# Use a different database file
python extract_city_pois.py "Tokyo" --database /path/to/custom_database.gpkg
```

### Command-Line Arguments

```
positional arguments:
  city                  Name of the city to extract POIs for

options:
  -h, --help            Show help message and exit
  -c, --country COUNTRY
                        Country name to narrow down city search (optional but recommended)
  -o, --output-dir OUTPUT_DIR
                        Directory to save output files (default: ../data/raw/pois)
  -d, --database DATABASE
                        Path to GHS database GeoPackage
                        (default: ../data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_1.gpkg)
```

## Output Files

The script creates three files in the output directory:

1. **`{City_Name}_pois_9cats.gpkg`** - GeoPackage containing all extracted POIs
   - Attributes: `geometry`, `category`, `osm_key`, `osm_value`, `name`, address fields, `osmid`

2. **`{City_Name}_boundary.gpkg`** - GeoPackage with city boundary from GHS database
   - Includes city metadata (ID, country, population, etc.)

3. **`{City_Name}_pois_summary.csv`** - Summary statistics by category
   - Format: `category,count`

### Example Output

```
data/raw/pois/
├── Reykjavik_boundary.gpkg      (100 KB)
├── Reykjavik_pois_9cats.gpkg    (992 KB - 3,293 POIs)
└── Reykjavik_pois_summary.csv   (159 B)
```

## How It Works

1. **City Lookup**: Searches GHS database for matching city name (case-insensitive)
2. **Boundary Extraction**: Extracts city boundary geometry
3. **POI Querying**: For each category, queries OpenStreetMap Overpass API
4. **Data Cleaning**: Keeps only essential columns, removes problematic field names
5. **Output**: Saves to GeoPackage and generates summary

## Performance

- **Small cities** (< 200K population): ~2-5 minutes
- **Medium cities** (200K-1M): ~5-10 minutes
- **Large cities** (> 1M): ~10-20 minutes

Time varies based on:
- City size and POI density
- Internet connection speed
- Overpass API load

## Example Results

### Reykjavik, Iceland (Population: 184,357)

**Total POIs: 3,293**

| Category | Count | Percentage |
|----------|-------|------------|
| Moving | 837 | 25.4% |
| Outdoor Activities | 708 | 21.5% |
| Eating | 570 | 17.3% |
| Physical Exercise | 539 | 16.4% |
| Learning | 245 | 7.4% |
| Supplies | 170 | 5.2% |
| Services | 108 | 3.3% |
| Cultural Activities | 68 | 2.1% |
| Health Care | 48 | 1.5% |

**Geometry Distribution:**
- Points: 1,832 (55.6%)
- Polygons: 1,452 (44.1%)
- MultiPolygons: 9 (0.3%)

## Troubleshooting

### City Not Found

If the city isn't found:
- Check spelling
- Add `--country` parameter to narrow search
- Try alternative city names (e.g., "Roma" vs "Rome")
- Check if city exists in GHS database:
  ```python
  import geopandas as gpd
  gdf = gpd.read_file("path/to/GHS_database.gpkg")
  print(gdf[gdf['UC_NM_MN'].str.contains("YourCity", case=False)])
  ```

### Multiple Matching Cities

The script will show all matches and use the first one:
```
⚠️  Found 3 matching cities:
   - Springfield, United States (ID: 1234)
   - Springfield, United States (ID: 5678)
   - Springfield, United Kingdom (ID: 9012)

   Using the first match. Specify --country to narrow down the search.
```

### API Rate Limiting

If you encounter Overpass API rate limits:
- Wait a few minutes before retrying
- The script uses OSMnx's built-in caching to avoid re-downloading data
- For large extractions, run during off-peak hours

### Missing POI Categories

Some cities may have zero POIs for certain categories (e.g., no subway entrances in cities without metros). This is expected and not an error.

## Technical Notes

### Data Source

- **Boundaries**: GHS Urban Centres Database (GHS-UCDB) R2019A
  - 13,135 urban centers globally
  - Population data from 2015
  - Consistent definitions across countries

- **POIs**: OpenStreetMap via Overpass API
  - Real-time, community-contributed data
  - Quality varies by region
  - Generally excellent coverage in developed countries

### Coordinate System

All outputs use **EPSG:4326** (WGS84 lat/lon)

### Field Cleaning

The script removes problematic OSM fields that cause issues with GeoPackage format:
- Fields with special characters (`:`, `/`, etc.)
- Keeps only essential fields: category, OSM tags, name, address, ID
- This prevents "FieldError" issues when saving

## Data Quality Considerations

POI data quality varies by:
1. **Geographic region**: Better coverage in Europe/North America
2. **Urban vs rural**: Cities have more complete data
3. **OSM community activity**: Active communities = better data
4. **Update frequency**: OSM is continuously updated

Always verify results for your specific use case.

## Citation

If using this data in research, cite:

**GHS Urban Centers Database:**
```
Florczyk, A.J., et al. (2019). GHS Urban Centre Database 2015, multitemporal and
multidimensional attributes, R2019A. European Commission, Joint Research Centre (JRC).
doi:10.2905/53473144-b88c-44bc-b4a3-4583ed1f547e
```

**OpenStreetMap:**
```
OpenStreetMap contributors. (2024). Planet dump retrieved from https://planet.osm.org
https://www.openstreetmap.org
```

## Related Scripts

- `test_extract_nyc_pois.py` - Original test script for NYC (in `tests/` directory)
- `verify_poi_data.py` - Data verification script

## Author

Created for the NYU China Grant project analyzing urban scaling laws and material stocks.

## Version

- **Version**: 1.0
- **Last Updated**: 2024-12-04
- **Python**: 3.10+
- **Dependencies**: osmnx, geopandas, pandas
