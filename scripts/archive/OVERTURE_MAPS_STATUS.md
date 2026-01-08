# Overture Maps POI Extraction - Status Report

**Date**: December 5, 2024
**Status**: ❌ **NOT WORKING** - Overture Maps data access currently unavailable

## Summary

After extensive testing, Overture Maps POI data cannot be reliably accessed through any available method. The project should continue using OpenStreetMap (OSM) via OSMnx, which is working reliably.

## Issues Encountered

### 1. Overture Maps Python Library (`overturemaps`) - BROKEN
- **Version tested**: 0.17.0
- **Problem**: Library hardcodes future release dates that don't exist:
  - `2025-08-20.0`
  - `2025-08-20.1`
  - `2025-09-24.0`
- **Error**: All queries return `403 Forbidden` when trying to access these non-existent releases
- **Affects**: Both Python API and CLI tool (`overturemaps download` command)

### 2. Direct S3 Access via DuckDB - NOT ACCESSIBLE
- **Attempted releases**:
  - 2024-12-18.0 (latest documented release)
  - 2024-11-13.0
  - 2024-10-23.0
  - 2024-09-18.0 (from working examples)
  - 2024-07-22.0
- **Error**: `IO Error: No files found that match the pattern "s3://overturemaps-us-west-2/release/[VERSION]/theme=places/type=place/*"`
- **Possible causes**:
  - S3 bucket structure changed
  - Places data partitioning scheme changed
  - Access requires authentication
  - Data not publicly available via S3 (despite documentation)

### 3. Azure Blob Storage Access via DuckDB - NOT ACCESSIBLE
- **Attempted path**: `https://overturemapswestus2.blob.core.windows.net/release/2024-12-18.0/theme=places/type=place/*.parquet`
- **Error**: `HTTP Error 404: The specified blob does not exist`
- **Conclusion**: Azure Blob Storage also not publicly accessible for places data

### 4. Path Patterns Tested

All of the following path patterns failed:

**AWS S3:**
```
s3://overturemaps-us-west-2/release/2024-12-18.0/theme=places/type=place/*
s3://overturemaps-us-west-2/release/2024-11-13.0/theme=places/type=place/*
s3://overturemaps-us-west-2/release/2024-10-23.0/theme=places/*.parquet
s3://overturemaps-us-west-2/release/2024-09-18.0/theme=places/type=place/*
```

**Azure Blob Storage:**
```
https://overturemapswestus2.blob.core.windows.net/release/2024-12-18.0/theme=places/type=place/*.parquet
```

## Documentation vs Reality

### What the Documentation Says:
- [Overture Maps Documentation](https://docs.overturemaps.org/getting-data/) claims data is available via:
  - Python library (`overturemaps`)
  - Direct S3 access via DuckDB
  - CLI tool (`overturemaps download`)
- Latest release listed: [2024-12-18.0](https://docs.overturemaps.org/release/2024-12-18.0/)
- Examples show working queries for places data

### What Actually Works:
- ❌ Python library - broken
- ❌ CLI tool - broken (uses same library)
- ❌ Direct S3 via DuckDB - data not accessible
- ✅ **OpenStreetMap via OSMnx - WORKING**

## OpenStreetMap Status: ✅ WORKING

The OSM-based approach (`01_download_OSM_city_pois.py`) is fully functional and has successfully extracted POIs for:

| City | Country | POIs | Status |
|------|---------|------|--------|
| Rome | Italy | 27,549 | ✅ |
| Paris | France | 121,203 | ✅ |
| Atlanta | United States | 10,934 | ✅ |
| Tokyo | Japan | 236,618 | ✅ |
| Addis Ababa | Ethiopia | 4,321 | ✅ |
| Bogota | Colombia | 34,725 | ✅ |
| Mexico City | Mexico | 39,339 | ✅ |
| Melbourne | Australia | 59,449 | ✅ |

**Total**: 534,138 POIs across 8 cities

## Recommendation

**Continue using OpenStreetMap (OSMnx) for POI extraction.**

### Reasons:
1. ✅ **Proven reliability** - Successfully processed 8 cities
2. ✅ **Active community** - Real-time updates from contributors worldwide
3. ✅ **Well-documented API** - Stable OSMnx library with consistent interface
4. ✅ **Rich attributes** - Detailed tags and metadata for each POI
5. ✅ **No access issues** - Free, open API via Overpass

### Trade-offs:
- Rate limits on Overpass API (manageable with delays)
- Coverage varies by region (community-dependent)
- Tag schema variations across regions

## Alternative Solutions (if Overture Maps is needed in future)

1. **Wait for library fix** - Monitor [overturemaps GitHub](https://github.com/OvertureMaps/overturemaps-py) for updates
2. **Use OvertureMapsDownloader tool** - [Third-party Docker tool](https://github.com/Youssef-Harby/OvertureMapsDownloader) that may have working implementation
3. **Manual GeoParquet download** - Download entire themes from S3, process locally (450GB+ for full dataset)

## References

- [Overture Maps Documentation](https://docs.overturemaps.org/getting-data/)
- [2024-12-18.0 Release Notes](https://docs.overturemaps.org/blog/2024-12-18.0/)
- [DuckDB Examples](https://docs.overturemaps.org/getting-data/duckdb/)
- [Pandas Examples](https://docs.overturemaps.org/examples/pandas/)
- [Simon Willison's DuckDB Guide](https://til.simonwillison.net/overture-maps/overture-maps-parquet)

## Files Created (Non-functional)

1. `01_download_Overture_city_pois.py` - Uses broken overturemaps library
2. `01_download_Overture_city_pois_s3.py` - Direct S3 access attempt (fails)
3. `compare_osm_vs_overture.py` - Comparison script (incomplete due to Overture failures)

These files are retained for reference but should not be used in production.

## Conclusion

While Overture Maps promises better global coverage and data consistency, the current implementation barriers make it impractical for this project. OpenStreetMap via OSMnx provides reliable, high-quality POI data and should remain the primary data source.
