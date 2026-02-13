#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def log_safe(x):
    x = np.asarray(x, dtype=float)
    # Add a tiny epsilon to avoid -inf for zeros
    return np.log10(np.clip(x, 1e-9, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighborhoods", required=True, help="Neighborhoods CSV (R7)")
    ap.add_argument("--out_dir", required=True, help="Output base dir (e.g., web/public/webdata)")
    ap.add_argument("--gzip", action="store_true", help="Write .json.gz files instead of .json")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    cities_agg_dir = os.path.join(out_dir, "cities_agg")
    index_dir = os.path.join(out_dir, "index")
    os.makedirs(cities_agg_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    print("Reading neighborhoods…", args.neighborhoods)
    usecols = [
        "ID_HDC_G0",
        "UC_NM_MN",
        "CTR_MN_ISO",
        "population_2015",
        "total_built_mass_tons",
        "h3index",
    ]
    dtypes = {
        "ID_HDC_G0": "int64",
        "UC_NM_MN": "string",
        "CTR_MN_ISO": "string",
        "population_2015": "float64",
        "total_built_mass_tons": "float64",
        "h3index": "string",
    }
    df = pd.read_csv(args.neighborhoods, usecols=usecols, dtype=dtypes)

    # Build city aggregates
    grp = df.groupby(["CTR_MN_ISO", "ID_HDC_G0", "UC_NM_MN"], as_index=False).agg(
        pop_total=("population_2015", "sum"),
        mass_total=("total_built_mass_tons", "sum"),
        n_hex=("h3index", "count"),
        sample_h3=("h3index", "first"),
    )
    grp["log_pop"] = log_safe(grp["pop_total"])  # log10
    grp["log_mass"] = log_safe(grp["mass_total"])  # log10
    grp.rename(columns={"CTR_MN_ISO": "country_iso", "ID_HDC_G0": "city_id", "UC_NM_MN": "city"}, inplace=True)

    # Write global city aggregates
    global_path = os.path.join(cities_agg_dir, f"global.json{'.gz' if args.gzip else ''}")
    if args.gzip:
        grp.to_json(global_path, orient="records", lines=False, compression="gzip")
    else:
        grp.to_json(global_path, orient="records")
    print("Wrote", global_path)

    # Per-country partitions
    for iso, sub in tqdm(grp.groupby("country_iso"), desc="countries"):
        out = os.path.join(cities_agg_dir, f"country={iso}.json{'.gz' if args.gzip else ''}")
        if args.gzip:
            sub.to_json(out, orient="records", compression="gzip")
        else:
            sub.to_json(out, orient="records")

    # Lookup indices
    country_to_cities = defaultdict(list)
    city_meta = {}
    for row in grp.itertuples(index=False):
        country_to_cities[row.country_iso].append(int(row.city_id))
        city_meta[str(int(row.city_id))] = {
            "city": str(row.city),
            "country_iso": str(row.country_iso),
            "sample_h3": str(row.sample_h3),
            "n_hex": int(row.n_hex),
        }
    with open(os.path.join(index_dir, "country_to_cities.json"), "w", encoding="utf-8") as f:
        json.dump(country_to_cities, f)
    with open(os.path.join(index_dir, "city_meta.json"), "w", encoding="utf-8") as f:
        json.dump(city_meta, f)
    print("Wrote indices in", index_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

