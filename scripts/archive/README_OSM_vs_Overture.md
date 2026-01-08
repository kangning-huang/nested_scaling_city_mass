# POI Extraction: OpenStreetMap vs Overture Maps

This document compares two approaches for extracting Points of Interest (POI) data for cities.

## Overview

### OpenStreetMap (OSM) Approach
- **Script**: `01_download_OSM_city_pois.py`
- **Library**: `osmnx` (wraps Overpass API)
- **Data Source**: OpenStreetMap community-contributed data
- **Access Method**: Real-time API queries via Overpass API

### Overture Maps Approach
- **Script**: `01_download_Overture_city_pois.py`
- **Library**: `overturemaps` Python package
- **Data Source**: Overture Maps (Meta, Microsoft, Amazon, TomTom consortium)
- **Access Method**: Pre-processed cloud-based dataset (AWS S3/Parquet)

## Key Differences

### Data Source & Quality

**OpenStreetMap:**
- ✅ Community-contributed, constantly updated
- ✅ Very detailed in well-mapped regions (Europe, North America)
- ⚠️ Variable quality across regions
- ⚠️ May have gaps in less-mapped areas
- ✅ Highly detailed attribute tags
- ✅ Free and open (ODbL license)

**Overture Maps:**
- ✅ Aggregated from multiple commercial sources
- ✅ More consistent global coverage
- ✅ Professional data quality control
- ⚠️ Less frequently updated than OSM
- ⚠️ Simplified attribute schema
- ✅ Open data (CDLA-Permissive license)

### Performance

**OpenStreetMap (OSMnx):**
- ⏱️ **Query Speed**: Moderate to slow (depends on Overpass API load)
- 📊 **Rate Limits**: Subject to Overpass API rate limiting
- 💾 **Caching**: OSMnx caches queries locally
- 🔄 **Scalability**: Good for individual cities, challenging for batch processing
- 🌐 **Network**: Requires stable internet connection

**Overture Maps:**
- ⏱️ **Query Speed**: Generally faster (optimized Parquet files)
- 📊 **Rate Limits**: No rate limits (cloud storage access)
- 💾 **Data Transfer**: Downloads larger chunks, but more efficient
- 🔄 **Scalability**: Better for batch processing multiple cities
- 🌐 **Network**: Requires good bandwidth for initial data download

### Data Schema

**OSM Tags (Examples):**
```python
{
    "amenity": ["restaurant", "school", "hospital"],
    "leisure": ["park", "playground"],
    "shop": ["supermarket", "convenience"],
    "highway": ["bus_stop"]
}
```
- Flexible, detailed tagging system
- Hundreds of possible tags
- May require complex filtering

**Overture Categories (Examples):**
```python
{
    "categories": [
        "food_and_drink",
        "education",
        "healthcare",
        "recreation",
        "retail"
    ]
}
```
- Standardized category schema
- Simpler structure
- Easier to process

## Installation Requirements

### OSM Approach
```bash
pip install osmnx geopandas pandas
```

### Overture Approach
```bash
pip install overturemaps geopandas pandas
```

## Usage Comparison

### OSM Script
```bash
python 01_download_OSM_city_pois.py "Atlanta" --country "United States"
```

**Process:**
1. Finds city in GHS database
2. For each POI category:
   - Queries Overpass API for specific tags
   - Multiple API requests per category
3. Combines and filters results
4. Saves to GeoPackage

### Overture Script
```bash
python 01_download_Overture_city_pois.py "Atlanta" --country "United States"
```

**Process:**
1. Finds city in GHS database
2. Queries Overture "places" theme for city bounding box
3. Single bulk query
4. Filters and categorizes locally
5. Saves to GeoPackage

## Performance Expectations

Based on testing with various cities:

| City Size | OSM Time | Overture Time | Winner |
|-----------|----------|---------------|--------|
| Small (<200K) | 2-5 min | 30-90 sec | Overture |
| Medium (200K-1M) | 5-10 min | 1-3 min | Overture |
| Large (>1M) | 10-20 min | 2-5 min | Overture |
| Very Large (>10M) | 15-30 min | 3-8 min | Overture |

**Factors affecting performance:**
- City size and POI density
- Internet connection speed
- Overpass API load (OSM only)
- Geographic region data quality

## POI Count Comparison

**Expected differences:**

1. **OSM typically has more POIs in:**
   - Well-mapped urban areas (Europe, North America)
   - Categories with community interest (parks, restaurants)
   - Recent developments (fast community updates)

2. **Overture typically has more POIs in:**
   - Less-mapped regions (Global South, rural areas)
   - Commercial establishments (from commercial data sources)
   - Standardized categories

3. **Overlap:**
   - Typical overlap: 60-80% of POIs
   - Differences due to:
     - Data source timing
     - Classification differences
     - Quality filters

## Use Case Recommendations

### Choose **OpenStreetMap** if you need:
- Most up-to-date data
- Highly detailed attributes
- Specific niche categories
- Well-mapped regions (Europe, North America)
- Complete tag metadata

### Choose **Overture Maps** if you need:
- Faster processing
- Global consistency
- Batch processing many cities
- Less-mapped regions
- Commercial-grade data quality

### Use **Both** if you need:
- Maximum POI coverage
- Data validation/cross-checking
- Research comparing data sources

## Comparison Script

Run both methods and compare:

```bash
python compare_osm_vs_overture.py "Atlanta" --country "United States"
```

**Output:**
- Side-by-side performance metrics
- POI count comparison
- Category breakdown comparison
- Saved comparison CSV

## Data Licenses

### OpenStreetMap
- **License**: Open Database License (ODbL)
- **Attribution**: © OpenStreetMap contributors
- **Share-Alike**: Derivative works must use same license
- **URL**: https://www.openstreetmap.org/copyright

### Overture Maps
- **License**: CDLA-Permissive-2.0
- **Attribution**: Overture Maps Foundation
- **No Share-Alike**: More permissive than OSM
- **URL**: https://overturemaps.org/

## Troubleshooting

### OSM Issues

**Rate Limiting:**
```
Error: 429 Too Many Requests
```
- Wait 5-10 minutes
- OSMnx has built-in caching
- Reduce query complexity

**Timeout Errors:**
```
Error: Query timeout
```
- City too large - try smaller bbox
- Overpass API under load
- Retry during off-peak hours

### Overture Issues

**Large Downloads:**
```
Downloading large Parquet files...
```
- Normal for large cities
- First query downloads data
- Subsequent queries are cached

**Missing Categories:**
```
No POIs matched our categories
```
- Check category mapping
- Overture schema may differ from OSM
- Some categories may use different names

## Performance Optimization

### OSM Optimization
1. **Use caching**: OSMnx caches by default
2. **Reduce tag complexity**: Fewer tags = faster
3. **Smaller regions**: Break large cities into chunks
4. **Off-peak hours**: Night/weekend queries faster

### Overture Optimization
1. **Precise bounding boxes**: Smaller bbox = less data transfer
2. **Filter locally**: Download once, filter multiple times
3. **Batch processing**: Download multiple cities sequentially
4. **Use SSD storage**: Faster Parquet I/O

## Future Enhancements

Potential improvements to both scripts:

1. **Parallel processing**: Query multiple cities simultaneously
2. **Incremental updates**: Only download changed data
3. **Hybrid approach**: Combine OSM + Overture for maximum coverage
4. **Quality metrics**: Automated data quality assessment
5. **Temporal analysis**: Track POI changes over time

## References

### Documentation
- **OSMnx**: https://osmnx.readthedocs.io/
- **Overture Maps**: https://docs.overturemaps.org/
- **GHS Urban Centers**: https://ghsl.jrc.ec.europa.eu/

### Papers
- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126-139.
- Overture Maps Foundation (2023). Overture Maps: An open map dataset. https://overturemaps.org/

## Version History

- **v1.0** (2024-12-05): Initial implementation of both OSM and Overture scripts
- Script author: NYU China Grant Project
- Python version: 3.10+
- Dependencies: geopandas, pandas, osmnx, overturemaps
