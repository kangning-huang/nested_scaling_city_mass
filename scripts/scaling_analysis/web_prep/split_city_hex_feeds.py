#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighborhoods", required=True, help="Neighborhoods CSV (R7)")
    ap.add_argument("--out_dir", required=True, help="Output dir for per-city JSON files")
    ap.add_argument("--cities", nargs="*", type=int, help="Optional list of city_id to restrict output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "ID_HDC_G0",
        "CTR_MN_ISO",
        "h3index",
        "population_2015",
        "total_built_mass_tons",
    ]
    dtypes = {
        "ID_HDC_G0": "int64",
        "CTR_MN_ISO": "string",
        "h3index": "string",
        "population_2015": "float64",
        "total_built_mass_tons": "float64",
    }
    df = pd.read_csv(args.neighborhoods, usecols=usecols, dtype=dtypes)
    df.rename(columns={"ID_HDC_G0": "city_id", "CTR_MN_ISO": "country_iso"}, inplace=True)

    if args.cities:
        df = df[df["city_id"].isin(args.cities)]

    for city_id, sub in tqdm(df.groupby("city_id"), desc="cities"):
        out = out_dir / f"city={int(city_id)}.json"
        # Keep fields required by client
        sub[["h3index", "population_2015", "total_built_mass_tons", "city_id", "country_iso"]].to_json(
            out, orient="records"
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

