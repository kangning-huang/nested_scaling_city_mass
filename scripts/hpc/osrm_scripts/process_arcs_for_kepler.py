#!/usr/bin/env python3
"""
Process NY Metro duration matrix into arc format for Kepler.gl visualization
"""

import json
import pandas as pd
import numpy as np
import h3

# File paths
geojson_path = "ny_metro_grids.geojson"
matrix_path = "ny_metro_duration_matrix.csv"
output_path = "ny_metro_arcs.csv"

# Load the duration matrix
print("Loading duration matrix...")
df = pd.read_csv(matrix_path, index_col=0)

# Convert seconds to minutes
df_minutes = df / 60

# Get H3 centroids for coordinates
print("Calculating H3 grid centroids...")
h3_indices = df_minutes.index.tolist()

centroids = {}
for h3_idx in h3_indices:
    lat, lng = h3.cell_to_latlng(h3_idx)
    centroids[h3_idx] = {'lat': lat, 'lng': lng}

# Convert matrix to long format (origin-destination pairs)
print("Converting to arc format...")
arcs = []

for origin in df_minutes.index:
    for dest in df_minutes.columns:
        if origin != dest:  # Skip self-loops
            travel_time = df_minutes.loc[origin, dest]
            arcs.append({
                'origin_h3': origin,
                'origin_lat': centroids[origin]['lat'],
                'origin_lng': centroids[origin]['lng'],
                'dest_h3': dest,
                'dest_lat': centroids[dest]['lat'],
                'dest_lng': centroids[dest]['lng'],
                'travel_time_min': round(travel_time, 2)
            })

arcs_df = pd.DataFrame(arcs)

# Add travel time categories for easier filtering/coloring
arcs_df['travel_time_category'] = pd.cut(
    arcs_df['travel_time_min'],
    bins=[0, 15, 30, 45, 60, 90, 120, float('inf')],
    labels=['0-15min', '15-30min', '30-45min', '45-60min', '60-90min', '90-120min', '120+min']
)

# Save to CSV
print(f"Saving {len(arcs_df):,} arcs to {output_path}...")
arcs_df.to_csv(output_path, index=False)

print(f"\nDone! Output saved to: {output_path}")
print(f"\nArc data summary:")
print(f"  Total arcs: {len(arcs_df):,}")
print(f"  Unique origins: {arcs_df['origin_h3'].nunique()}")
print(f"  Unique destinations: {arcs_df['dest_h3'].nunique()}")
print(f"\nTravel time distribution:")
print(arcs_df['travel_time_category'].value_counts().sort_index())
print(f"\nColumns in output:")
print("  - origin_h3, origin_lat, origin_lng: Origin grid ID and coordinates")
print("  - dest_h3, dest_lat, dest_lng: Destination grid ID and coordinates")
print("  - travel_time_min: Travel time in minutes")
print("  - travel_time_category: Binned travel time for filtering")
print("\nTo visualize in Kepler.gl:")
print("  1. Upload ny_metro_arcs.csv")
print("  2. Add an 'Arc' layer")
print("  3. Set origin: origin_lat, origin_lng")
print("  4. Set destination: dest_lat, dest_lng")
print("  5. Color by: travel_time_min or travel_time_category")
print("\nTip: Filter to reduce visual clutter (e.g., only show arcs < 30 min)")
