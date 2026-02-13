#!/usr/bin/env python3
"""
02b_extract_roads_clean.py
Clean GRIP road extraction for H3 hexagons.

Fixes the duplicate-hexagon issue in the original 02_extract_roads_neighborhood.py:
the grids file has duplicate h3 indices (hexagons belonging to multiple cities),
which caused Cartesian products during sequential left merges.

This script:
  1. Deduplicates hexagon geometries before extraction
  2. Extracts road lengths from GRIP rasters (one row per hexagon)
  3. Adds lane-based road widths and surface areas
  4. Adds climate classifications
  5. Joins metadata back from grids

Usage:
  python scripts/02b_extract_roads_clean.py -r 5
  python scripts/02b_extract_roads_clean.py -r 7
"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from exactextract import exact_extract

# Ensure project root is on sys.path so 'utils' is importable
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.paths import get_resolution_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = BASE_DIR / "data" / "processed"

# GRIP raster paths (relative to project root)
GRIP_BASE = Path(os.environ.get(
    "GRIP_DIR",
    "/Users/kangninghuang/Library/CloudStorage/GoogleDrive-kh3657@nyu.edu/"
    "My Drive/Grants_Fellowship/2024 NYU China Grant/data/GRIP"
))

GRIP_FILES = {
    "highway":   GRIP_BASE / "GRIP4_density_tp1" / "grip4_tp1_length_m.tif",
    "primary":   GRIP_BASE / "GRIP4_density_tp2" / "grip4_tp2_length_m.tif",
    "secondary": GRIP_BASE / "GRIP4_density_tp3" / "grip4_tp3_length_m.tif",
    "tertiary":  GRIP_BASE / "GRIP4_density_tp4" / "grip4_tp4_length_m.tif",
    "local":     GRIP_BASE / "GRIP4_density_tp5" / "grip4_tp5_length_m.tif",
}

LANE_EXCEL = (
    Path("/Users/kangninghuang/Library/CloudStorage/GoogleDrive-kh3657@nyu.edu/"
         "My Drive/Grants_Fellowship/2024 NYU China Grant/data/"
         "Rousseau et al 2022/es2c05255_si_001.xlsx")
)

# Road type mapping for lane data
ROAD_TYPE_MAP = {
    "tp1": "highway",
    "tp2": "primary",
    "tp3": "secondary",
    "tp4": "tertiary",
    "tp5": "local",
}

# Fixed width fallbacks (metres) used when lane data is unavailable
WIDTHS_HIGH = {"highway": 14, "primary": 10, "secondary": 8, "tertiary": 6, "local": 5}
WIDTHS_LOW  = {"highway": 10, "primary": 7,  "secondary": 6, "tertiary": 4, "local": 3}

# ── Köppen climate mapping ──────────────────────────────────────────
KOPPEN_MAP = {
    "Tropical rain forest":                           ("Af",  "Wet, non-freeze (WN)"),
    "Tropical monsoon":                               ("Am",  "Wet, non-freeze (WN)"),
    "Tropical savannah with dry summer":              ("As",  "Wet, non-freeze (WN)"),
    "Tropical savannah with dry winter":              ("Aw",  "Wet, non-freeze (WN)"),
    "Desert (arid), and Hot arid":                    ("BWh", "Dry, non-freeze (DN)"),
    "Desert (arid), and Cold arid":                   ("BWk", "Dry, freeze (DF)"),
    "Steppe (semi-arid), and Hot arid":               ("BSh", "Dry, non-freeze (DN)"),
    "Steppe (semi-arid), and Cold arid":              ("BSk", "Dry, non-freeze (DN)"),
    "Mild temperate with dry summer, and Hot summer":  ("Csa", "Dry, non-freeze (DN)"),
    "Mild temperate with dry summer, and Warm summer": ("Csb", "Dry, non-freeze (DN)"),
    "Mild temperate with dry winter, and Hot summer":  ("Cwa", "Dry, non-freeze (DN)"),
    "Mild temperate with dry winter, and Warm summer": ("Cwb", "Dry, non-freeze (DN)"),
    "Mild temperate, fully humid, and Hot summer":     ("Cfa", "Wet, non-freeze (WN)"),
    "Mild temperate, fully humid, and Warm summer":    ("Cfb", "Wet, non-freeze (WN)"),
    "Mild temperate, fully humid, and Cool summer":    ("Cfc", "Wet, freeze (WF)"),
    "Snow with dry summer, and Hot summer":            ("Dsa", "Dry, freeze (DF)"),
    "Snow with dry summer, and Warm summer":           ("Dsb", "Dry, freeze (DF)"),
    "Snow with dry summer, and Cool summer":           ("Dsc", "Dry, freeze (DF)"),
    "Snow with dry winter, and Hot summer":            ("Dwa", "Wet, freeze (WF)"),
    "Snow with dry winter, and Warm summer":           ("Dwb", "Wet, freeze (WF)"),
    "Snow with dry winter, and Cool summer":           ("Dwc", "Dry, freeze (DF)"),
    "Snow, fully humid, and Hot summer":               ("Dfa", "Wet, freeze (WF)"),
    "Snow, fully humid, and Warm summer":              ("Dfb", "Wet, freeze (WF)"),
    "Snow, fully humid, and Cool summer":              ("Dfc", "Wet, freeze (WF)"),
}


# ── Helpers ──────────────────────────────────────────────────────────

def load_grids(resolution: int) -> gpd.GeoDataFrame:
    """Load H3 grids, keeping unique hexagon geometries + metadata."""
    data_dir = get_resolution_dir(PROCESSED_DIR, resolution)
    gpkg = data_dir / f"all_cities_h3_grids_resolution{resolution}.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Grid file not found: {gpkg}")
    log.info("Loading grids from %s", gpkg)
    gdf = gpd.read_file(gpkg)
    log.info("Loaded %d rows, %d unique h3 indices", len(gdf), gdf["h3index"].nunique())
    return gdf


def deduplicate_grids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return one geometry per h3index (dissolve duplicates).

    Hexagons that appear under multiple cities get their geometries dissolved
    (union), so exactextract sees each hexagon exactly once.
    """
    n_before = len(gdf)
    # Dissolve by h3index, taking the first metadata row
    dedup = gdf.dissolve(by="h3index", aggfunc="first").reset_index()
    log.info("Deduplicated grids: %d → %d", n_before, len(dedup))
    return dedup


def extract_grip_lengths(grids: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extract GRIP road lengths (sum per hexagon) for all 5 road types."""
    # Use a temporary integer id for exactextract
    grids = grids.copy()
    grids["_eid"] = range(len(grids))

    result = grids[["_eid", "h3index"]].copy()

    for road_type, tif_path in GRIP_FILES.items():
        if not tif_path.exists():
            raise FileNotFoundError(f"GRIP raster not found: {tif_path}")
        log.info("Extracting %s from %s", road_type, tif_path.name)
        ev = exact_extract(
            str(tif_path),
            grids,
            "sum",
            include_cols=["_eid"],
            progress=True,
            output="pandas",
        )
        col = f"sum_road_m_{road_type}"
        ev = ev.rename(columns={"sum": col})

        # exactextract should return one row per polygon, but guard against duplicates
        if ev["_eid"].duplicated().any():
            log.warning("%s: %d duplicate _eid values – aggregating", road_type, ev["_eid"].duplicated().sum())
            ev = ev.groupby("_eid", as_index=False)[col].sum()

        result = result.merge(ev[["_eid", col]], on="_eid", how="left")
        gc.collect()

    result.drop(columns=["_eid"], inplace=True)
    # Fill any NaN from merge with 0
    road_cols = [c for c in result.columns if c.startswith("sum_road_m_")]
    result[road_cols] = result[road_cols].fillna(0.0)
    return result


def add_lane_data(df: pd.DataFrame, grids_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge lane count data from Rousseau et al. 2022 and compute widths/areas."""
    if not LANE_EXCEL.exists():
        log.warning("Lane data Excel not found: %s – using fixed width assumptions", LANE_EXCEL)
        return compute_areas_fixed(df)

    log.info("Loading lane data from %s", LANE_EXCEL)
    lanes_raw = pd.read_excel(LANE_EXCEL, sheet_name="Number_lanes")

    # Required columns
    country_col = "Country Alpha-3 Code"
    road_type_col = "GRIP road type"
    lanes_col = "Avg. number of lanes, weighted"
    region_col = "GRIP region"

    for c in [country_col, road_type_col, lanes_col]:
        if c not in lanes_raw.columns:
            log.warning("Missing column '%s' in lane data – using fixed widths", c)
            return compute_areas_fixed(df)

    lanes_raw[lanes_col] = pd.to_numeric(lanes_raw[lanes_col], errors="coerce")

    # Fill missing with regional average
    if region_col in lanes_raw.columns:
        lanes_raw["lanes_filled"] = lanes_raw.groupby([region_col, road_type_col])[lanes_col].transform(
            lambda x: x.fillna(x.mean())
        )
    else:
        lanes_raw["lanes_filled"] = lanes_raw[lanes_col]

    pivot = lanes_raw.pivot_table(
        index=country_col, columns=road_type_col, values="lanes_filled", aggfunc="mean"
    ).reset_index()
    pivot.columns = [
        c if c == country_col else f"lanes_{c.replace(' ', '_').replace('-', '_').lower()}"
        for c in pivot.columns
    ]
    pivot.rename(columns={country_col: "ISO_A3_lanes"}, inplace=True)

    # We need CTR_MN_ISO from the grids metadata
    meta = grids_meta[["h3index", "CTR_MN_ISO"]].drop_duplicates(subset="h3index")
    df = df.merge(meta, on="h3index", how="left")

    # Merge lane data
    df["_iso_upper"] = df["CTR_MN_ISO"].astype(str).str.upper().str.strip()
    pivot["_iso_upper"] = pivot["ISO_A3_lanes"].astype(str).str.upper().str.strip()
    df = df.merge(pivot, on="_iso_upper", how="left")
    df.drop(columns=["_iso_upper", "ISO_A3_lanes", "CTR_MN_ISO"], inplace=True, errors="ignore")

    # Calculate width and area per road type
    for tp_key, rtype in ROAD_TYPE_MAP.items():
        length_col = f"sum_road_m_{rtype}"
        lanes_col_name = f"lanes_{rtype}"
        if length_col not in df.columns:
            continue
        if lanes_col_name in df.columns:
            if rtype in ("highway", "primary"):
                df[f"width_low_{rtype}"] = df[lanes_col_name] * 3.5
                df[f"width_high_{rtype}"] = (df[lanes_col_name] + 1) * 4.0
            else:
                df[f"width_low_{rtype}"] = df[lanes_col_name] * 3.0
                df[f"width_high_{rtype}"] = (df[lanes_col_name] + 1) * 3.5

            df[f"road_area_low_{rtype}"] = df[f"width_low_{rtype}"] * df[length_col]
            df[f"road_area_high_{rtype}"] = df[f"width_high_{rtype}"] * df[length_col]
        else:
            # Fallback to fixed widths for this road type
            df[f"road_area_low_{rtype}"] = df[length_col] * WIDTHS_LOW[rtype]
            df[f"road_area_high_{rtype}"] = df[length_col] * WIDTHS_HIGH[rtype]

    return df


def compute_areas_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute road surface areas using fixed width assumptions."""
    log.info("Computing road areas using fixed width assumptions")
    for rtype in ROAD_TYPE_MAP.values():
        length_col = f"sum_road_m_{rtype}"
        if length_col in df.columns:
            df[f"road_area_low_{rtype}"] = df[length_col] * WIDTHS_LOW[rtype]
            df[f"road_area_high_{rtype}"] = df[length_col] * WIDTHS_HIGH[rtype]
    return df


def add_climate(df: pd.DataFrame, grids_meta: pd.DataFrame) -> pd.DataFrame:
    """Add Köppen climate code and class from grids metadata."""
    meta = grids_meta[["h3index", "E_KG_NM_LS"]].drop_duplicates(subset="h3index")
    df = df.merge(meta, on="h3index", how="left")

    # Build normalised lookup
    norm_map = {}
    for text, (code, cls) in KOPPEN_MAP.items():
        key = " ".join(text.lower().split())
        norm_map[key] = (code, cls)

    def _map_climate(text):
        if pd.isna(text):
            return pd.NA, pd.NA
        key = " ".join(str(text).lower().split())
        if key in norm_map:
            return norm_map[key]
        return "Unknown", "Unknown"

    mapped = df["E_KG_NM_LS"].apply(_map_climate)
    df["koppen_code"] = [m[0] for m in mapped]
    df["climate_class"] = [m[1] for m in mapped]
    return df


def attach_metadata(roads_df: pd.DataFrame, grids: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach city/country metadata from the original (non-deduplicated) grids.

    Each hexagon gets metadata from its *first* city occurrence (matching
    the convention used in the validated R6 dataset).
    """
    meta_cols = ["h3index", "neighborhood_id", "ID_HDC_G0", "UC_NM_MN",
                 "CTR_MN_ISO", "CTR_MN_NM", "GRGN_L1", "GRGN_L2"]
    existing = [c for c in meta_cols if c in grids.columns]
    meta = grids[existing].drop_duplicates(subset="h3index", keep="first")
    roads_df = roads_df.merge(meta, on="h3index", how="left")
    return roads_df


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean GRIP road extraction for H3 hexagons")
    parser.add_argument("--resolution", "-r", type=int, default=6,
                        help="H3 resolution level (default: 6)")
    args = parser.parse_args()
    resolution = args.resolution

    log.info("=== GRIP road extraction for Resolution %d ===", resolution)

    # 1. Load and deduplicate grids
    grids_raw = load_grids(resolution)
    grids_dedup = deduplicate_grids(grids_raw)

    # 2. Extract road lengths from GRIP
    roads = extract_grip_lengths(grids_dedup)
    log.info("Extracted road lengths for %d hexagons", len(roads))

    # 3. Add lane-based widths and areas
    roads = add_lane_data(roads, grids_raw)

    # 4. Add climate classifications
    roads = add_climate(roads, grids_raw)

    # 5. Attach city/country metadata
    roads = attach_metadata(roads, grids_raw)

    # 6. Reorder columns: metadata first, then road data
    meta_cols = ["neighborhood_id", "h3index", "ID_HDC_G0", "UC_NM_MN",
                 "CTR_MN_ISO", "CTR_MN_NM", "GRGN_L1", "GRGN_L2", "E_KG_NM_LS"]
    existing_meta = [c for c in meta_cols if c in roads.columns]
    other_cols = [c for c in roads.columns if c not in existing_meta]
    roads = roads[existing_meta + other_cols]

    # 7. Save
    out_dir = get_resolution_dir(PROCESSED_DIR, resolution)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out_csv = out_dir / f"Fig3_Roads_Neighborhood_H3_Resolution{resolution}_{today}.csv"
    roads.to_csv(out_csv, index=False)

    log.info("Saved %d rows to %s", len(roads), out_csv)

    # Print summary
    road_len_cols = [c for c in roads.columns if c.startswith("sum_road_m_")]
    road_area_cols = [c for c in roads.columns if c.startswith("road_area_")]
    log.info("--- Summary ---")
    log.info("Hexagons: %d", len(roads))
    log.info("Road length columns: %s", road_len_cols)
    log.info("Road area columns: %s", road_area_cols)
    for col in road_len_cols:
        log.info("  %s: mean=%.1f, max=%.1f, sum=%.0f",
                 col, roads[col].mean(), roads[col].max(), roads[col].sum())
    for col in ["road_area_high_total", "road_area_low_total"]:
        if col in roads.columns:
            log.info("  %s: sum=%.0f m²", col, roads[col].sum())


if __name__ == "__main__":
    main()
