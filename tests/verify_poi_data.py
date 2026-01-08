"""
Quick verification script to inspect the extracted NYC POI data
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

# Load the data
data_path = Path(__file__).parent / "nyc_pois_9cats.gpkg"
gdf = gpd.read_file(data_path)

print("="*80)
print("NYC POI DATA VERIFICATION")
print("="*80)

print(f"\nTotal POIs: {len(gdf):,}")
print(f"CRS: {gdf.crs}")

print("\n" + "-"*80)
print("POI COUNTS BY CATEGORY")
print("-"*80)
category_summary = gdf.groupby('category').agg({
    'category': 'count',
    'osm_key': lambda x: len(x.unique())
}).rename(columns={'category': 'count', 'osm_key': 'unique_keys'})

for idx, row in category_summary.iterrows():
    print(f"{idx:30s}: {row['count']:6,} POIs  ({row['unique_keys']} unique keys)")

print("\n" + "-"*80)
print("GEOMETRY TYPES")
print("-"*80)
for geom_type, count in gdf.geometry.geom_type.value_counts().items():
    pct = 100 * count / len(gdf)
    print(f"{geom_type:20s}: {count:6,} ({pct:5.1f}%)")

print("\n" + "-"*80)
print("SAMPLE POIs WITH NAMES (First 10)")
print("-"*80)
sample = gdf[gdf['name'].notna()].head(10)[['category', 'osm_key', 'osm_value', 'name']]
print(sample.to_string(index=False))

print("\n" + "-"*80)
print("DATA COMPLETENESS")
print("-"*80)
for col in ['name', 'addr:street', 'addr:housenumber', 'addr:city']:
    non_null = gdf[col].notna().sum()
    pct = 100 * non_null / len(gdf)
    print(f"{col:20s}: {non_null:6,} / {len(gdf):6,} ({pct:5.1f}%)")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
