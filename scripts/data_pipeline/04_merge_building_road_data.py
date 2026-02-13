import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.paths import get_resolution_dir


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge volume and road outputs for Fig. 3")
    parser.add_argument("--resolution", "-r", type=int, default=6,
                        help="H3 resolution level (default: 6)")
    parser.add_argument("--volume-file", type=str, default=None,
                        help="Optional path to the volume CSV. Defaults to latest matching export in results/global_scaling.")
    parser.add_argument("--road-file", type=str, default=None,
                        help="Optional path to the road CSV. Defaults to latest matching export in results/global_scaling.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output path. Defaults to results/global_scaling/Fig3_Merged_...<today>.csv")
    return parser.parse_args()


def resolve_path(path_str: Optional[str], pattern: str, input_dir: Path) -> Path:
    if path_str:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    matches = sorted(input_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files found matching pattern '{pattern}' in {input_dir}")
    return matches[-1]

def main():
    args = parse_args()
    resolution = args.resolution

    # Set up resolution-specific directories
    INPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution)
    OUTPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution)

    print(f"H3 Resolution: {resolution}")
    print(f"Input directory: {INPUT_DIR}")

    volume_file = resolve_path(args.volume_file, f"Fig3_Volume_Pavement_Neighborhood_H3_Resolution{resolution}_*.csv", INPUT_DIR)
    road_file = resolve_path(args.road_file, f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_*.csv", INPUT_DIR)

    if args.output:
        output_file = Path(args.output).expanduser()
        if not output_file.is_absolute():
            output_file = BASE_DIR / output_file
    else:
        today_date = datetime.now().strftime("%Y-%m-%d")
        output_file = OUTPUT_DIR / f"Fig3_Merged_Neighborhood_H3_Resolution{resolution}_{today_date}.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading volume data from {volume_file}")
    df_volume = pd.read_csv(volume_file)
    print(f"Reading road data from {road_file}")
    df_road = pd.read_csv(road_file)

    # Find common columns for merge
    common_cols = [col for col in df_volume.columns if col in df_road.columns]
    if not common_cols:
        raise ValueError("No common columns found for merging.")
    print(f"Merging on columns: {common_cols}")

    df_merged = pd.merge(df_volume, df_road, on=common_cols, how='outer')

    # Building footprint columns
    building_cols = ['footprint_Esch2022', 'footprint_Li2022', 'footprint_Liu2024']
    # Road types and width assumptions (in meters)
    road_types = ['highway', 'primary', 'secondary', 'tertiary', 'local']
    # Width assumptions from literature (high and low estimates)
    road_widths_high = {'highway': 14, 'primary': 10, 'secondary': 8, 'tertiary': 6, 'local': 5}
    road_widths_low = {'highway': 10, 'primary': 7, 'secondary': 6, 'tertiary': 4, 'local': 3}

    # Check columns exist
    for col in building_cols:
        if col not in df_merged.columns:
            raise ValueError(f"Missing building footprint column: {col}")
    if 'impervious_2015' not in df_merged.columns:
        raise ValueError("Missing 'impervious' column.")

    # Check if road data has length columns (sum_road_m_*) or area columns (road_area_*)
    has_length_cols = all(f'sum_road_m_{t}' in df_merged.columns for t in road_types)
    has_area_cols = all(f'road_area_high_{t}' in df_merged.columns for t in road_types)

    if has_area_cols:
        # Use pre-computed area columns
        road_area_high_cols = [f'road_area_high_{t}' for t in road_types]
        road_area_low_cols = [f'road_area_low_{t}' for t in road_types]
        df_merged['road_area_high_total'] = df_merged[road_area_high_cols].sum(axis=1)
        df_merged['road_area_low_total'] = df_merged[road_area_low_cols].sum(axis=1)
    elif has_length_cols:
        # Compute road areas from length columns
        print("Computing road areas from length data using width assumptions:")
        print(f"  High widths (m): {road_widths_high}")
        print(f"  Low widths (m): {road_widths_low}")

        for t in road_types:
            length_col = f'sum_road_m_{t}'
            # Fill NaN with 0 for length columns
            df_merged[length_col] = df_merged[length_col].fillna(0)
            df_merged[f'road_area_high_{t}'] = df_merged[length_col] * road_widths_high[t]
            df_merged[f'road_area_low_{t}'] = df_merged[length_col] * road_widths_low[t]

        road_area_high_cols = [f'road_area_high_{t}' for t in road_types]
        road_area_low_cols = [f'road_area_low_{t}' for t in road_types]
        df_merged['road_area_high_total'] = df_merged[road_area_high_cols].sum(axis=1)
        df_merged['road_area_low_total'] = df_merged[road_area_low_cols].sum(axis=1)
    else:
        raise ValueError("Missing road data columns. Expected either sum_road_m_* or road_area_* columns.")

    # Calculate other pavement area for all 6 combinations
    for b in building_cols:
        for road_area_type in ['high', 'low']:
            road_area_col = f'road_area_{road_area_type}_total'
            out_col = f'other_pavement_{b}_{road_area_type}'
            # Each pixel in impervious is 10m x 10m so 100m2 = 100*impervious
            df_merged[out_col] = 100*df_merged['impervious_2015'] - df_merged[b] - df_merged[road_area_col]
            # Ensure no negative values
            df_merged[out_col] = df_merged[out_col].clip(lower=0)

    # PRESERVE EXISTING CLIMATE DATA - only fill missing values
    # Check if climate columns already exist from the road data
    if 'koppen_code' not in df_merged.columns:
        df_merged['koppen_code'] = None
    if 'climate_class' not in df_merged.columns:
        df_merged['climate_class'] = None

    # Fill missing values in Koppen Code for specific cities only
    koppen_lookup = {
        'Honolulu':            'As',   # Tropical savanna, dry summer
        'El Alto [La Paz]':    'ET',   # Tundra
        'Bridgetown':          'Am',   # Tropical monsoon
        'Funchal':             'Csb',  # Warm‐summer Mediterranean
        'Valletta':            'Csa',  # Hot‐summer Mediterranean
        'Port Louis':          'Aw',   # Tropical savanna, dry winter
        'Naha':                'Cfa',  # Humid subtropical
        'Jolo':                'Af',   # Tropical rainforest
    }

    # Only fill missing koppen_code values, don't overwrite existing ones
    missing_koppen_mask = (df_merged['koppen_code'].isna() | (df_merged['koppen_code'] == '')) & df_merged['UC_NM_MN'].isin(koppen_lookup.keys())
    df_merged.loc[missing_koppen_mask, 'koppen_code'] = df_merged.loc[missing_koppen_mask, 'UC_NM_MN'].map(koppen_lookup)

    # Mapping from Köppen codes to climate classes
    koppen_to_class = {
        # Wet, non-freeze (WN)
        'Af': 'Wet, non-freeze (WN)', 'Am': 'Wet, non-freeze (WN)', 'Aw': 'Wet, non-freeze (WN)', 'As': 'Wet, non-freeze (WN)', 'Cfa': 'Wet, non-freeze (WN)', 'Cfb': 'Wet, non-freeze (WN)',        
        # Dry, non-freeze (DN)
        'Bsh': 'Dry, non-freeze (DN)', 'BSk': 'Dry, non-freeze (DN)', 'BWh': 'Dry, non-freeze (DN)', 'Csa': 'Dry, non-freeze (DN)', 'Csb': 'Dry, non-freeze (DN)',
        'Csc': 'Dry, non-freeze (DN)', 'Cwa': 'Dry, non-freeze (DN)', 'Cwb': 'Dry, non-freeze (DN)',
        # Wet, freeze (WF)
        'Cfc': 'Wet, freeze (WF)', 'Dfa': 'Wet, freeze (WF)', 'Dfb': 'Wet, freeze (WF)', 'Dfc': 'Wet, freeze (WF)', 'Dfd': 'Wet, freeze (WF)',
        'Dwa': 'Wet, freeze (WF)', 'Dwb': 'Wet, freeze (WF)',
        # Dry, freeze (DF)
        'BWk': 'Dry, freeze (DF)', 'Cwc': 'Dry, freeze (DF)', 'Dsa': 'Dry, freeze (DF)', 'Dsb': 'Dry, freeze (DF)', 'Dsc': 'Dry, freeze (DF)',
        'Dsd': 'Dry, freeze (DF)', 'Dwc': 'Dry, freeze (DF)', 'Dwd': 'Dry, freeze (DF)', 'EF': 'Dry, freeze (DF)', 'ET': 'Dry, freeze (DF)'
    }

    # Only fill missing climate_class values, don't overwrite existing ones
    missing_climate_mask = (df_merged['climate_class'].isna() | (df_merged['climate_class'] == '')) & df_merged['koppen_code'].notna()
    df_merged.loc[missing_climate_mask, 'climate_class'] = df_merged.loc[missing_climate_mask, 'koppen_code'].map(koppen_to_class)
    
    # print out cities in missing climate class
    print(df_merged[df_merged['UC_NM_MN'].isin(['Honolulu', 'El Alto [La Paz]', 'Bridgetown', 'Funchal', 'Valletta', 'Port Louis', 'Naha', 'Jolo'])][['UC_NM_MN', 'koppen_code', 'climate_class']])
    
    # Save output
    df_merged.to_csv(output_file, index=False)
    print(f"Merged and calculated file saved to: {output_file}")

if __name__ == "__main__":
    main()
