import argparse
import numpy as np
from datetime import datetime
import warnings
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

from utils.paths import get_resolution_dir, get_latest_file

# Set up base directory and paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Prepare global mass data at neighborhood level')
parser.add_argument('--resolution', '-r', type=int, default=6,
                    help='H3 resolution level (default: 6)')
args = parser.parse_args()
resolution = args.resolution

# Set up resolution-specific directories
DATA_DIR = get_resolution_dir(PROCESSED_DIR, resolution)
OUTPUT_DIR = get_resolution_dir(PROCESSED_DIR, resolution)

print(f"Fig3_GlobalMass_ByClass - Neighborhood Level (Resolution {resolution})")

# 1. Input Data
# Everything organized around the H3 index: h3index
print("\n=== Loading Data ===")

# Load merged neighborhood data (find latest file matching pattern)
merged_pattern = f"Fig3_Merged_Neighborhood_H3_Resolution{resolution}_*.csv"
try:
    merged_file = get_latest_file(DATA_DIR, merged_pattern)
except FileNotFoundError:
    raise FileNotFoundError(f"No merged data file found matching pattern: {merged_pattern} in {DATA_DIR}")
print(f"Using merged file: {merged_file}")
DF_Merged = pd.read_csv(merged_file)
if 'Unnamed: 0' in DF_Merged.columns:
    DF_Merged = DF_Merged.drop('Unnamed: 0', axis=1)

# Load biomass dataset (if available at neighborhood level)
# Note: You may need to adapt this based on your biomass data structure
try:
    DF_Biomass = pd.read_csv(DATA_DIR / "biomass_by_h3.csv")
    if 'X' in DF_Biomass.columns:
        DF_Biomass = DF_Biomass.drop('X', axis=1)
    DF_Biomass = DF_Biomass.rename(columns={'total_biomass_tons': 'total_biomassC_tons'})
    DF_Biomass['total_dryBiomass_tons'] = DF_Biomass['total_biomassC_tons'] / 0.45
    print("Biomass data loaded successfully")
except FileNotFoundError:
    print("Warning: Biomass data not found. Creating empty dataframe.")
    DF_Biomass = pd.DataFrame(columns=['h3index', 'total_biomassC_tons', 'total_dryBiomass_tons'])

print(f"Merged data shape: {DF_Merged.shape}")
print(f"Biomass data shape: {DF_Biomass.shape}")

# Building classification based on Haberl2025
building_classes = {
    'LW': 'Lightweight (<3m)',
    'RS': 'Residential single-family (3-12m)', 
    'RM': 'Residential multi-family (12-50m)',
    'NR': 'Non-residential (3-50m)',
    'HR': 'High-rise (50-100m)'
}

print("\n=== Building Classes ===")
for code, desc in building_classes.items():
    print(f"{code}: {desc}")

# Check climate class distribution
if 'climate_class' in DF_Merged.columns:
    print("\n=== Climate Class Distribution ===")
    print(DF_Merged['climate_class'].value_counts())
    
    # Check for missing climate data in populated areas
    if 'population_2015' in DF_Merged.columns:
        missing_climate = DF_Merged[(DF_Merged['population_2015'] > 0) & 
                                   (DF_Merged['climate_class'].isna() | (DF_Merged['climate_class'] == ''))]
        print(f"H3 cells with population > 1000 but missing climate data: {len(missing_climate)}")

# 2. Split into sub workflows: buildings, roads & pavement
print("\n=== Data Processing ===")

# Define key ID columns for H3 neighborhoods
id_cols = ['h3index']

# Add additional identifier columns if they exist
optional_id_cols = ['UC_NM_MN', 'ID_HDC_G0', 'CTR_MN_ISO', 'CTR_MN_NM', 'population_2015']
for col in optional_id_cols:
    if col in DF_Merged.columns:
        id_cols.append(col)

print(f"Using ID columns: {id_cols}")

# 2.1 Extract & tidy building volume columns
vol_columns = [col for col in DF_Merged.columns if col.startswith('vol_') and 
               any(building_class in col for building_class in ['LW', 'RS', 'RM', 'HR', 'NR'])]

print(f"Found {len(vol_columns)} volume columns")

# Reshape building volume data
DF_BuildingVolume_list = []
for col in vol_columns:
    # Extract data source and class from column name
    # Format: vol_DataSource_Class
    parts = col.split('_')
    if len(parts) >= 3:
        data_source = parts[1]
        building_class = parts[2]
        
        temp_df = DF_Merged[id_cols + [col]].copy()
        temp_df['DataSource'] = data_source
        temp_df['Class'] = building_class
        temp_df['Volume_m3'] = temp_df[col]
        temp_df = temp_df.drop(col, axis=1)
        DF_BuildingVolume_list.append(temp_df)

if DF_BuildingVolume_list:
    DF_BuildingVolume = pd.concat(DF_BuildingVolume_list, ignore_index=True)
    print(f"Building volume data shape: {DF_BuildingVolume.shape}")
else:
    print("Warning: No building volume data found")
    DF_BuildingVolume = pd.DataFrame()

# 2.2 Road surface area
road_id_cols = id_cols.copy()
if 'climate_class' in DF_Merged.columns:
    road_id_cols.append('climate_class')

road_area_columns = [col for col in DF_Merged.columns if col.startswith('road_area_')]
print(f"Found {len(road_area_columns)} road area columns")

# Reshape road data
DF_RoadSurface_list = []
for col in road_area_columns:
    # Format: road_area_estimate_roadtype
    parts = col.split('_')
    if len(parts) >= 4:
        estimate = parts[2]  # low or high
        road_type = parts[3]  # highway, primary, etc.
        
        if road_type != 'total':  # Skip total columns
            temp_df = DF_Merged[road_id_cols + [col]].copy()
            temp_df['Estimate'] = estimate
            temp_df['RoadType'] = road_type
            temp_df['RoadArea_m2'] = temp_df[col]
            temp_df['Source'] = 'Rousseau2022'
            temp_df = temp_df.drop(col, axis=1)
            DF_RoadSurface_list.append(temp_df)

if DF_RoadSurface_list:
    DF_RoadSurface = pd.concat(DF_RoadSurface_list, ignore_index=True)
    
    # Process climate codes
    if 'climate_class' in DF_RoadSurface.columns:
        DF_RoadSurface['climate_class'] = DF_RoadSurface['climate_class'].astype(str).str.strip()
        DF_RoadSurface['climate_class'] = DF_RoadSurface['climate_class'].replace('', np.nan)
        
        # Extract climate codes
        def extract_climate_code(climate_str):
            if pd.isna(climate_str) or climate_str == 'nan':
                return np.nan
            if '(DN)' in climate_str:
                return 'DN'
            elif '(DF)' in climate_str:
                return 'DF'
            elif '(WN)' in climate_str:
                return 'WN'
            elif '(WF)' in climate_str:
                return 'WF'
            else:
                return np.nan
        
        DF_RoadSurface['ClimateCode'] = DF_RoadSurface['climate_class'].apply(extract_climate_code)
        DF_RoadSurface = DF_RoadSurface.drop('climate_class', axis=1)
    
    # Map road types to GRIP codes
    road_type_mapping = {
        'highway': 1,
        'primary': 2,
        'secondary': 3,
        'tertiary': 4,
        'local': 5
    }
    DF_RoadSurface['GRIPRoadType'] = DF_RoadSurface['RoadType'].map(road_type_mapping)
    
    # Add GRIP region codes (if geographic info available)
    if 'CTR_MN_ISO' in DF_RoadSurface.columns:
        def get_grip_region(iso_code):
            # Simplified mapping - you may need to expand this
            if pd.isna(iso_code):
                return np.nan, np.nan
            
            iso_code = str(iso_code).upper()
            if iso_code in ['USA', 'CAN']:
                return 'North America', 1
            elif iso_code in ['CHN', 'JPN', 'KOR', 'IND', 'IDN', 'THA', 'VNM', 'MYS', 'SGP', 'PHL']:
                return 'South and East Asia', 6
            elif iso_code in ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'POL', 'NLD', 'BEL', 'GRC', 'PRT']:
                return 'Europe', 4
            else:
                return 'Rest of the world', 7
        
        DF_RoadSurface[['GRIPRegion', 'GRIPRegionCode']] = DF_RoadSurface['CTR_MN_ISO'].apply(
            lambda x: pd.Series(get_grip_region(x))
        )
    
    print(f"Road surface data shape: {DF_RoadSurface.shape}")
else:
    print("Warning: No road area data found")
    DF_RoadSurface = pd.DataFrame()

# 2.3 Other pavement (non-road)
other_pav_columns = [col for col in DF_Merged.columns if col.startswith('other_pavement_footprint_')]
print(f"Found {len(other_pav_columns)} other pavement columns")

DF_OtherPavement_list = []
for col in other_pav_columns:
    # Format: other_pavement_footprint_source_estimate
    parts = col.split('_')
    if len(parts) >= 5:
        source = parts[3]
        estimate = parts[4]  # low or high
        
        temp_df = DF_Merged[id_cols + [col]].copy()
        temp_df['Source'] = source
        temp_df['Estimate'] = estimate
        temp_df['OtherPavArea_m2'] = temp_df[col]
        temp_df = temp_df.drop(col, axis=1)
        DF_OtherPavement_list.append(temp_df)

if DF_OtherPavement_list:
    DF_OtherPavement = pd.concat(DF_OtherPavement_list, ignore_index=True)
    print(f"Other pavement data shape: {DF_OtherPavement.shape}")
else:
    print("Warning: No other pavement data found")
    DF_OtherPavement = pd.DataFrame()

# 3. Building Material Intensity
print("\n=== Building Material Intensity ===")

if not DF_BuildingVolume.empty:
    # OECD country definitions
    oecd1990 = ['AUS', 'AUT', 'BEL', 'CAN', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'ISL', 
                'IRL', 'ITA', 'JPN', 'LUX', 'NLD', 'NZL', 'NOR', 'PRT', 'ESP', 'SWE', 
                'CHE', 'TUR', 'GBR', 'USA']
    
    oecd2010 = oecd1990 + ['CHL', 'EST', 'ISR', 'SVN']
    other_oecd2010 = [x for x in oecd2010 if x not in ['USA', 'JPN']]
    
    # Material Intensity lookup table
    MI_lookup = pd.DataFrame([
        {'Class': 'RS', 'RegionMI': 'OECD (excl. NA and Japan)', 'MI': 349.64},
        {'Class': 'RS', 'RegionMI': 'North America', 'MI': 284.87},
        {'Class': 'RS', 'RegionMI': 'Japan', 'MI': 124.93},
        {'Class': 'RS', 'RegionMI': 'China', 'MI': 526.00},
        {'Class': 'RS', 'RegionMI': 'Rest of the world', 'MI': 299.45},
        {'Class': 'RM', 'RegionMI': 'OECD (excl. NA and Japan)', 'MI': 398.60},
        {'Class': 'RM', 'RegionMI': 'North America', 'MI': 314.72},
        {'Class': 'RM', 'RegionMI': 'Japan', 'MI': 601.60},
        {'Class': 'RM', 'RegionMI': 'China', 'MI': 662.06},
        {'Class': 'RM', 'RegionMI': 'Rest of the world', 'MI': 394.77},
        {'Class': 'NR', 'RegionMI': 'OECD (excl. NA and Japan)', 'MI': 375.55},
        {'Class': 'NR', 'RegionMI': 'North America', 'MI': 280.82},
        {'Class': 'NR', 'RegionMI': 'Japan', 'MI': 272.90},
        {'Class': 'NR', 'RegionMI': 'China', 'MI': 654.06},
        {'Class': 'NR', 'RegionMI': 'Rest of the world', 'MI': 367.81},
        {'Class': 'LW', 'RegionMI': 'North America', 'MI': 151.30},
        {'Class': 'LW', 'RegionMI': 'Rest of the world', 'MI': 154.20},
        {'Class': 'HR', 'RegionMI': 'North America', 'MI': 312.35},
        {'Class': 'HR', 'RegionMI': 'Rest of the world', 'MI': 329.98}
    ])
    
    # Assign region for material intensity
    def assign_region_mi(row):
        building_class = row['Class']
        country_iso = row.get('CTR_MN_ISO', '')
        
        if pd.isna(country_iso):
            country_iso = ''
        
        if building_class in ['LW', 'HR']:
            if country_iso in ['USA', 'CAN']:
                return 'North America'
            else:
                return 'Rest of the world'
        elif building_class in ['RS', 'RM', 'NR']:
            if country_iso in ['USA', 'CAN']:
                return 'North America'
            elif country_iso == 'JPN':
                return 'Japan'
            elif country_iso == 'CHN':
                return 'China'
            elif country_iso in other_oecd2010:
                return 'OECD (excl. NA and Japan)'
            else:
                return 'Rest of the world'
        else:
            return 'Rest of the world'
    
    DF_BuildingVolume['RegionMI'] = DF_BuildingVolume.apply(assign_region_mi, axis=1)
    
    # Join with MI lookup
    DF_MI_Buildings = DF_BuildingVolume.merge(MI_lookup, on=['Class', 'RegionMI'], how='left')
    
    # Calculate building mass (volume in m3, MI in kg/m3, convert to tonnes)
    DF_MI_Buildings['BuildingMass'] = DF_MI_Buildings['Volume_m3'] * DF_MI_Buildings['MI'] / 1000
    
    print(f"Building mass data shape: {DF_MI_Buildings.shape}")
    print(f"Missing MI values: {DF_MI_Buildings['MI'].isna().sum()}")
else:
    DF_MI_Buildings = pd.DataFrame()

# 4. Road Material Intensity
print("\n=== Road Material Intensity ===")

if not DF_RoadSurface.empty:
    try:
        mi_road_raw = pd.read_csv("scripts/Rousseau2022_SITable_MI.csv") # Ensure this file is accessible
        mi_road_raw.columns = mi_road_raw.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        mi_road = mi_road_raw[['country_alpha_3_code', 'grip_region', 'grip_road_type', 
                               'climate_class', 'asphalt_int_median', 'granular_int_median',
                               'cement_int_median', 'concrete_int_median']].copy()
        
        mi_road.columns = ['iso3', 'grip_region_code', 'grip_rt', 'climate', 
                           'asphalt', 'granular', 'cement', 'concrete']

        # Prepare four specialized MI lookup tables
        mi_road['iso3_lower'] = mi_road['iso3'].str.lower()
        mi_road['climate_lower'] = mi_road['climate'].str.lower()

        # 2a. Country-Specific data
        mi_spec_all = mi_road[mi_road['iso3_lower'] != "generic"].copy()
        mi_spec_climate = mi_spec_all[mi_spec_all['climate_lower'] != "multiple"].copy()
        mi_spec_multiple = mi_spec_all[mi_spec_all['climate_lower'] == "multiple"].drop(columns=['climate', 'climate_lower']).copy()

        # 2b. Generic data
        mi_gen_all = mi_road[mi_road['iso3_lower'] == "generic"].drop(columns=['iso3', 'iso3_lower']).copy()
        mi_gen_climate = mi_gen_all[mi_gen_all['climate_lower'] != "multiple"].copy()
        mi_gen_multiple = mi_gen_all[mi_gen_all['climate_lower'] == "multiple"].drop(columns=['climate', 'climate_lower']).copy()

        # Ensure join keys are of the same type if necessary (e.g., GRIPRegionCode, GRIPRoadType)
        # DF_RoadSurface['GRIPRegionCode'] = DF_RoadSurface['GRIPRegionCode'].astype(float).astype('Int64') # Example, adjust as per your data types
        # DF_RoadSurface['GRIPRoadType'] = DF_RoadSurface['GRIPRoadType'].astype(float).astype('Int64')
        # mi_spec_climate['grip_region_code'] = mi_spec_climate['grip_region_code'].astype(float).astype('Int64')
        # mi_spec_climate['grip_rt'] = mi_spec_climate['grip_rt'].astype(float).astype('Int64')
        # ... and so on for other lookup tables and their join keys

        road_data_mi = DF_RoadSurface.copy()

        # 3.1. Join Country-Specific, Climate-Specific MI
        road_data_mi = pd.merge(
            road_data_mi, 
            mi_spec_climate[['iso3', 'grip_region_code', 'grip_rt', 'climate', 'asphalt', 'granular', 'cement', 'concrete']],
            left_on=['CTR_MN_ISO', 'GRIPRegionCode', 'GRIPRoadType', 'ClimateCode'],
            right_on=['iso3', 'grip_region_code', 'grip_rt', 'climate'],
            how='left',
            suffixes=('', '_s_c') # _s_c for specific_climate
        )
        # Rename merged columns to avoid conflicts in subsequent merges and for clarity in coalesce
        rename_map_s_c = {col: col + '_s_c' for col in ['asphalt', 'granular', 'cement', 'concrete'] if col in road_data_mi.columns}
        road_data_mi = road_data_mi.rename(columns=rename_map_s_c)
        # Drop helper columns from the right table if they were merged
        road_data_mi = road_data_mi.drop(columns=['iso3', 'grip_region_code', 'grip_rt', 'climate'], errors='ignore')

        # 3.2. Join Country-Specific, "Multiple" Climate MI
        road_data_mi = pd.merge(
            road_data_mi, 
            mi_spec_multiple[['iso3', 'grip_region_code', 'grip_rt', 'asphalt', 'granular', 'cement', 'concrete']],
            left_on=['CTR_MN_ISO', 'GRIPRegionCode', 'GRIPRoadType'],
            right_on=['iso3', 'grip_region_code', 'grip_rt'],
            how='left',
            suffixes=('', '_s_m') # _s_m for specific_multiple
        )
        rename_map_s_m = {col: col + '_s_m' for col in ['asphalt', 'granular', 'cement', 'concrete'] if col + '_s_m' not in road_data_mi.columns and col in road_data_mi.columns}
        road_data_mi = road_data_mi.rename(columns=rename_map_s_m)
        road_data_mi = road_data_mi.drop(columns=['iso3', 'grip_region_code', 'grip_rt'], errors='ignore')

        # 3.3. Join Generic, Climate-Specific MI
        road_data_mi = pd.merge(
            road_data_mi, 
            mi_gen_climate[['grip_region_code', 'grip_rt', 'climate', 'asphalt', 'granular', 'cement', 'concrete']],
            left_on=['GRIPRegionCode', 'GRIPRoadType', 'ClimateCode'],
            right_on=['grip_region_code', 'grip_rt', 'climate'],
            how='left',
            suffixes=('', '_g_c') # _g_c for generic_climate
        )
        rename_map_g_c = {col: col + '_g_c' for col in ['asphalt', 'granular', 'cement', 'concrete'] if col + '_g_c' not in road_data_mi.columns and col in road_data_mi.columns}
        road_data_mi = road_data_mi.rename(columns=rename_map_g_c)
        road_data_mi = road_data_mi.drop(columns=['grip_region_code', 'grip_rt', 'climate'], errors='ignore')

        # 3.4. Join Generic, "Multiple" Climate MI
        road_data_mi = pd.merge(
            road_data_mi, 
            mi_gen_multiple[['grip_region_code', 'grip_rt', 'asphalt', 'granular', 'cement', 'concrete']],
            left_on=['GRIPRegionCode', 'GRIPRoadType'],
            right_on=['grip_region_code', 'grip_rt'],
            how='left',
            suffixes=('', '_g_m') # _g_m for generic_multiple
        )
        rename_map_g_m = {col: col + '_g_m' for col in ['asphalt', 'granular', 'cement', 'concrete'] if col + '_g_m' not in road_data_mi.columns and col in road_data_mi.columns}
        road_data_mi = road_data_mi.rename(columns=rename_map_g_m)
        road_data_mi = road_data_mi.drop(columns=['grip_region_code', 'grip_rt'], errors='ignore')

        # 4. Coalesce to get the final MI values
        # Order of coalesce: spec_climate, spec_multiple, gen_climate, gen_multiple
        road_data_mi['asphalt'] = road_data_mi['asphalt_s_c']
        if 'asphalt_s_m' in road_data_mi.columns: road_data_mi['asphalt'] = road_data_mi['asphalt'].fillna(road_data_mi['asphalt_s_m'])
        if 'asphalt_g_c' in road_data_mi.columns: road_data_mi['asphalt'] = road_data_mi['asphalt'].fillna(road_data_mi['asphalt_g_c'])
        if 'asphalt_g_m' in road_data_mi.columns: road_data_mi['asphalt'] = road_data_mi['asphalt'].fillna(road_data_mi['asphalt_g_m'])

        road_data_mi['granular'] = road_data_mi['granular_s_c']
        if 'granular_s_m' in road_data_mi.columns: road_data_mi['granular'] = road_data_mi['granular'].fillna(road_data_mi['granular_s_m'])
        if 'granular_g_c' in road_data_mi.columns: road_data_mi['granular'] = road_data_mi['granular'].fillna(road_data_mi['granular_g_c'])
        if 'granular_g_m' in road_data_mi.columns: road_data_mi['granular'] = road_data_mi['granular'].fillna(road_data_mi['granular_g_m'])

        road_data_mi['cement'] = road_data_mi['cement_s_c']
        if 'cement_s_m' in road_data_mi.columns: road_data_mi['cement'] = road_data_mi['cement'].fillna(road_data_mi['cement_s_m'])
        if 'cement_g_c' in road_data_mi.columns: road_data_mi['cement'] = road_data_mi['cement'].fillna(road_data_mi['cement_g_c'])
        if 'cement_g_m' in road_data_mi.columns: road_data_mi['cement'] = road_data_mi['cement'].fillna(road_data_mi['cement_g_m'])

        road_data_mi['concrete'] = road_data_mi['concrete_s_c']
        if 'concrete_s_m' in road_data_mi.columns: road_data_mi['concrete'] = road_data_mi['concrete'].fillna(road_data_mi['concrete_s_m'])
        if 'concrete_g_c' in road_data_mi.columns: road_data_mi['concrete'] = road_data_mi['concrete'].fillna(road_data_mi['concrete_g_c'])
        if 'concrete_g_m' in road_data_mi.columns: road_data_mi['concrete'] = road_data_mi['concrete'].fillna(road_data_mi['concrete_g_m'])
        
        # Drop intermediate columns
        cols_to_drop = [col for col in road_data_mi.columns if any(suffix in col for suffix in ['_s_c', '_s_m', '_g_c', '_g_m']) and col not in ['asphalt', 'granular', 'cement', 'concrete']]
        # Also drop the original unsuffixed columns if they were created by the first merge and are now redundant
        # For example, if the first merge created 'asphalt_x', 'granular_x', etc. and then 'asphalt_s_c', etc.
        # The current logic renames the first merge's columns to *_s_c, so this should be fine.
        road_data_mi = road_data_mi.drop(columns=cols_to_drop, errors='ignore')

        print("Hierarchical road material intensity applied.")

    except FileNotFoundError:
        print("Road MI lookup table 'Rousseau2022_SITable_MI.csv' not found. Using default values.")
        road_data_mi = DF_RoadSurface.copy()
        road_data_mi['asphalt'] = 50.0  # Default values
        road_data_mi['granular'] = 30.0
        road_data_mi['cement'] = 20.0
        road_data_mi['concrete'] = 40.0
    except Exception as e:
        print(f"An error occurred during road MI processing: {e}. Using default values.")
        road_data_mi = DF_RoadSurface.copy()
        road_data_mi['asphalt'] = 50.0
        road_data_mi['granular'] = 30.0
        road_data_mi['cement'] = 20.0
        road_data_mi['concrete'] = 40.0
    
    # Calculate road masses
    road_data_mass = road_data_mi.copy()
    road_data_mass['Mass_asphalt'] = road_data_mass['RoadArea_m2'] * road_data_mass['asphalt'] / 1000
    road_data_mass['Mass_granular'] = road_data_mass['RoadArea_m2'] * road_data_mass['granular'] / 1000
    road_data_mass['Mass_cement'] = road_data_mass['RoadArea_m2'] * road_data_mass['cement'] / 1000
    road_data_mass['Mass_concrete'] = road_data_mass['RoadArea_m2'] * road_data_mass['concrete'] / 1000
    
    print(f"Road mass data shape: {road_data_mass.shape}")
else:
    road_data_mass = pd.DataFrame()

# 5. Other Pavement Mass
print("\n=== Other Pavement Mass ===")

if not DF_OtherPavement.empty:
    # 0.57 t/m2 from Frantz2023 SI Table 15 for parking lots and yards
    DF_OtherPavement['OtherPavMass'] = DF_OtherPavement['OtherPavArea_m2'] * 0.57
    print(f"Other pavement mass data shape: {DF_OtherPavement.shape}")
else:
    DF_OtherPavement = pd.DataFrame()

# 6. Aggregation by H3 cell
print("\n=== Aggregating by H3 Cell ===")

# Filter for cells with some population (adjust threshold as needed)
population_threshold = 0  # Lower threshold for neighborhood level
if 'population_2015' in DF_Merged.columns:
    valid_h3_cells = DF_Merged[DF_Merged['population_2015'] > population_threshold]['h3index'].unique()
    print(f"Processing {len(valid_h3_cells)} H3 cells with population > {population_threshold}")
else:
    valid_h3_cells = DF_Merged['h3index'].unique()
    print(f"Processing all {len(valid_h3_cells)} H3 cells")

# 6.1 Building mass aggregation
if not DF_MI_Buildings.empty:
    building_totals = DF_MI_Buildings[DF_MI_Buildings['h3index'].isin(valid_h3_cells)].groupby(
        ['h3index', 'DataSource']
    )['BuildingMass'].sum().reset_index()
    
    # Convert zeros to NaN
    building_totals.loc[building_totals['BuildingMass'] == 0, 'BuildingMass'] = np.nan
    
    # Pivot by data source
    building_summary = building_totals.pivot(index='h3index', columns='DataSource', values='BuildingMass')
    building_summary.columns = [f'BuildingMass_Total_{col}' for col in building_summary.columns]
    building_summary = building_summary.reset_index()
    
    # Calculate average across data sources
    mass_columns = [col for col in building_summary.columns if col.startswith('BuildingMass_Total_')]
    if mass_columns:
        building_summary['BuildingMass_AverageTotal'] = building_summary[mass_columns].mean(axis=1, skipna=True)
    
    # Merge with ID columns from original data
    id_data = DF_Merged[id_cols].drop_duplicates(subset=['h3index'])
    building_summary = building_summary.merge(id_data, on='h3index', how='left')
    
    print(f"Building summary shape: {building_summary.shape}")
else:
    # Create empty building summary with all ID columns
    id_data = DF_Merged[id_cols].drop_duplicates(subset=['h3index'])
    building_summary = id_data[id_data['h3index'].isin(valid_h3_cells)].copy()
    building_summary['BuildingMass_AverageTotal'] = np.nan

# 6.2 Road mass aggregation
if not road_data_mass.empty:
    road_data_mass['TotalRoadMaterialMass'] = (road_data_mass['Mass_asphalt'] + 
                                              road_data_mass['Mass_granular'] + 
                                              road_data_mass['Mass_cement'] + 
                                              road_data_mass['Mass_concrete'])
    
    road_totals = road_data_mass[road_data_mass['h3index'].isin(valid_h3_cells)].groupby(
        ['h3index', 'Estimate']
    )['TotalRoadMaterialMass'].sum().reset_index()
    
    # Pivot by estimate
    road_summary = road_totals.pivot(index='h3index', columns='Estimate', values='TotalRoadMaterialMass')
    road_summary.columns = [f'RoadMass_{col}' for col in road_summary.columns]
    road_summary = road_summary.reset_index()
    
    # Calculate average
    if 'RoadMass_low' in road_summary.columns and 'RoadMass_high' in road_summary.columns:
        road_summary['RoadMass_Average'] = (road_summary['RoadMass_low'] + road_summary['RoadMass_high']) / 2
    
    print(f"Road summary shape: {road_summary.shape}")
else:
    road_summary = pd.DataFrame({'h3index': valid_h3_cells})
    road_summary['RoadMass_Average'] = np.nan

# 6.3 Other pavement mass aggregation
if not DF_OtherPavement.empty:
    other_pav_totals = DF_OtherPavement[DF_OtherPavement['h3index'].isin(valid_h3_cells)].groupby(
        ['h3index', 'Estimate']
    )['OtherPavMass'].mean().reset_index()
    
    # Convert zeros to NaN
    other_pav_totals.loc[other_pav_totals['OtherPavMass'] == 0, 'OtherPavMass'] = np.nan
    
    # Pivot by estimate
    other_pav_summary = other_pav_totals.pivot(index='h3index', columns='Estimate', values='OtherPavMass')
    other_pav_summary.columns = [f'OtherPavMass_{col}' for col in other_pav_summary.columns]
    other_pav_summary = other_pav_summary.reset_index()
    
    # Calculate average
    if 'OtherPavMass_low' in other_pav_summary.columns and 'OtherPavMass_high' in other_pav_summary.columns:
        other_pav_summary['OtherPavMass_Average'] = (other_pav_summary['OtherPavMass_low'] + 
                                                    other_pav_summary['OtherPavMass_high']) / 2
    
    print(f"Other pavement summary shape: {other_pav_summary.shape}")
else:
    other_pav_summary = pd.DataFrame({'h3index': valid_h3_cells})
    other_pav_summary['OtherPavMass_Average'] = np.nan

# 7. Final combination and export
print("\n=== Final Combination ===")

# Start with building summary
MasterMass_ByClass = building_summary.copy()

# Merge with road summary
MasterMass_ByClass = MasterMass_ByClass.merge(road_summary, on='h3index', how='outer')

# Merge with other pavement summary
MasterMass_ByClass = MasterMass_ByClass.merge(other_pav_summary, on='h3index', how='outer')

# Remove the duplicate population merge since population_2015 is already included in building_summary
# from the id_data merge at line ~485
# population_data = DF_Merged[['h3index', 'population_2015']].copy()
# MasterMass_ByClass = MasterMass_ByClass.merge(population_data, on='h3index', how='left')
print("Population data already included from building summary merge")

# Merge with biomass data if available
if not DF_Biomass.empty:
    MasterMass_ByClass = MasterMass_ByClass.merge(DF_Biomass, on='h3index', how='left')
    print("Biomass data merged")
else:
    MasterMass_ByClass['total_biomassC_tons'] = np.nan
    MasterMass_ByClass['total_dryBiomass_tons'] = np.nan

# Calculate derived totals
MasterMass_ByClass['mobility_mass_tons'] = (MasterMass_ByClass.get('OtherPavMass_Average', 0).fillna(0) + 
                                           MasterMass_ByClass.get('RoadMass_Average', 0).fillna(0))

MasterMass_ByClass['total_built_mass_tons'] = (MasterMass_ByClass.get('OtherPavMass_Average', 0).fillna(0) + 
                                              MasterMass_ByClass.get('RoadMass_Average', 0).fillna(0) + 
                                              MasterMass_ByClass.get('BuildingMass_AverageTotal', 0).fillna(0))

# Replace zeros with NaN for derived totals where appropriate
MasterMass_ByClass.loc[MasterMass_ByClass['mobility_mass_tons'] == 0, 'mobility_mass_tons'] = np.nan
MasterMass_ByClass.loc[MasterMass_ByClass['total_built_mass_tons'] == 0, 'total_built_mass_tons'] = np.nan

print(f"Final dataset shape: {MasterMass_ByClass.shape}")
print(f"Columns: {list(MasterMass_ByClass.columns)}")

# Export results
today = datetime.now().strftime("%Y-%m-%d")
output_filename = OUTPUT_DIR / f"Fig3_Mass_Neighborhood_H3_Resolution{resolution}_{today}.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MasterMass_ByClass.to_csv(output_filename, index=False)

print(f"\n=== Export Complete ===")
print(f"Output saved as: {output_filename}")
print(f"Total H3 cells processed: {len(MasterMass_ByClass)}")

# Display summary statistics
print("\n=== Summary Statistics ===")
numeric_columns = MasterMass_ByClass.select_dtypes(include=[np.number]).columns
for col in ['BuildingMass_AverageTotal', 'RoadMass_Average', 'OtherPavMass_Average', 
           'mobility_mass_tons', 'total_built_mass_tons']:
    if col in MasterMass_ByClass.columns:
        valid_count = MasterMass_ByClass[col].notna().sum()
        if valid_count > 0:
            mean_val = MasterMass_ByClass[col].mean()
            median_val = MasterMass_ByClass[col].median()
            print(f"{col}: {valid_count} valid values, mean={mean_val:.2f}, median={median_val:.2f}")
        else:
            print(f"{col}: No valid values")

# Sample output for inspection
print("\n=== Sample Output ===")
print(MasterMass_ByClass.head())

print("\n=== Processing Complete ===")