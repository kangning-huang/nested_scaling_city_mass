#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


def log_safe(x):
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, 1e-9, None))


def stratified_sample(df: pd.DataFrame, group_col: str, n_total: int) -> pd.DataFrame:
    # Sample proportional to group size, min 1 per group (if available)
    counts = df[group_col].value_counts()
    if counts.empty or n_total <= 0:
        return df.iloc[[]]
    frac = n_total / len(df)
    parts = []
    for key, cnt in counts.items():
        take = max(1, int(round(cnt * frac)))
        sub = df[df[group_col] == key]
        if take >= len(sub):
            parts.append(sub)
        else:
            parts.append(sub.sample(n=take, random_state=42))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=42)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighborhoods", required=True, help="Neighborhoods CSV (R7)")
    ap.add_argument("--out_dir", required=True, help="Output base dir (e.g., web/public/webdata)")
    ap.add_argument("--global_n", type=int, default=100000, help="Global sample size")
    ap.add_argument("--per_country_n", type=int, default=20000, help="Per-country sample size")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    sample_dir = os.path.join(out_dir, "scatter_samples")
    os.makedirs(sample_dir, exist_ok=True)

    usecols = ["CTR_MN_ISO", "population_2015", "total_built_mass_tons", "ID_HDC_G0"]
    dtypes = {"CTR_MN_ISO": "string", "population_2015": "float64", "total_built_mass_tons": "float64", "ID_HDC_G0": "int64"}
    df = pd.read_csv(args.neighborhoods, usecols=usecols, dtype=dtypes)
    df = df.rename(columns={"CTR_MN_ISO": "country_iso", "ID_HDC_G0": "city_id"})
    df["log_pop"] = log_safe(df["population_2015"])
    df["log_mass"] = log_safe(df["total_built_mass_tons"])

    # Keep only fields required for scatter rendering
    cols = ["country_iso", "city_id", "log_pop", "log_mass"]
    df = df[cols]

    # Global sample
    gsample = stratified_sample(df, group_col="country_iso", n_total=args.global_n)
    gpath = os.path.join(sample_dir, "global_neighborhood.json")
    gsample.to_json(gpath, orient="records")
    print("Wrote", gpath)

    # Per-country samples
    for iso, sub in tqdm(df.groupby("country_iso"), desc="country samples"):
        sp = stratified_sample(sub, group_col="city_id", n_total=args.per_country_n)
        out = os.path.join(sample_dir, f"country={iso}.json")
        sp.to_json(out, orient="records")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

