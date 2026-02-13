#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def ols_slope_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, float, float]:
    """Compute OLS slope, intercept, slope 95% CI, R^2.

    Returns: slope, intercept, slope_lo, slope_hi, r2
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = ((x - x_mean) ** 2).sum()
    Sxy = ((x - x_mean) * (y - y_mean)).sum()
    slope = Sxy / Sxx if Sxx > 0 else np.nan
    intercept = y_mean - slope * x_mean

    y_pred = intercept + slope * x
    resid = y - y_pred
    s2 = (resid @ resid) / (n - 2)  # residual variance
    se_slope = np.sqrt(s2 / Sxx) if Sxx > 0 else np.nan

    # 95% CI using normal approx (t close to 1.96 for large n)
    z = 1.959963984540054
    slope_lo = slope - z * se_slope
    slope_hi = slope + z * se_slope

    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(slope_lo), float(slope_hi), float(r2)


def write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city_agg", required=True, help="Global city aggregates JSON (from prep_city_aggregates)")
    ap.add_argument("--neighborhoods", required=True, help="Neighborhoods CSV (R7)")
    ap.add_argument("--out_dir", required=True, help="Output base dir (e.g., web/public/webdata)")
    args = ap.parse_args()

    out_dir = args.out_dir
    reg_dir = os.path.join(out_dir, "regression")

    # City-level regressions
    cities = pd.read_json(args.city_agg)
    # Global
    s, b, lo, hi, r2 = ols_slope_ci(cities["log_pop"].values, cities["log_mass"].values)
    x0 = float(cities["log_pop"].mean())
    y0 = float(cities["log_mass"].mean())
    write_json(os.path.join(reg_dir, "global_city.json"), {
        "scope": "global_city",
        "slope": s, "slope_lo": lo, "slope_hi": hi,
        "x0": x0, "y0": y0, "n": int(len(cities)), "r2": r2
    })

    # Per-country city-level
    for iso, sub in tqdm(cities.groupby("country_iso"), desc="city-level by country"):
        s, b, lo, hi, r2 = ols_slope_ci(sub["log_pop"].values, sub["log_mass"].values)
        x0 = float(sub["log_pop"].mean())
        y0 = float(sub["log_mass"].mean())
        path = os.path.join(reg_dir, "country", f"{iso}.json")
        write_json(path, {"scope": "country_city", "country_iso": iso, "slope": s, "slope_lo": lo, "slope_hi": hi, "x0": x0, "y0": y0, "n": int(len(sub)), "r2": r2})

    # Neighborhood-level regressions (global, per-country, per-city)
    usecols = ["CTR_MN_ISO", "ID_HDC_G0", "population_2015", "total_built_mass_tons"]
    dtypes = {"CTR_MN_ISO": "string", "ID_HDC_G0": "int64", "population_2015": "float64", "total_built_mass_tons": "float64"}
    neigh = pd.read_csv(args.neighborhoods, usecols=usecols, dtype=dtypes)
    neigh = neigh.rename(columns={"CTR_MN_ISO": "country_iso", "ID_HDC_G0": "city_id"})
    neigh = neigh[(neigh["population_2015"] > 0) & (neigh["total_built_mass_tons"] > 0)].copy()
    neigh["log_pop"] = np.log10(neigh["population_2015"].values)
    neigh["log_mass"] = np.log10(neigh["total_built_mass_tons"].values)
    neigh = neigh[np.isfinite(neigh["log_pop"]) & np.isfinite(neigh["log_mass"])]

    # Global neighborhood
    s, b, lo, hi, r2 = ols_slope_ci(neigh["log_pop"].values, neigh["log_mass"].values)
    x0 = float(neigh["log_pop"].mean())
    y0 = float(neigh["log_mass"].mean())
    write_json(os.path.join(reg_dir, "global_neighborhood.json"), {
        "scope": "global_neighborhood",
        "slope": s, "slope_lo": lo, "slope_hi": hi,
        "x0": x0, "y0": y0, "n": int(len(neigh)), "r2": r2
    })

    # Country neighborhood
    for iso, sub in tqdm(neigh.groupby("country_iso"), desc="neighborhood by country"):
        s, b, lo, hi, r2 = ols_slope_ci(sub["log_pop"].values, sub["log_mass"].values)
        x0 = float(sub["log_pop"].mean())
        y0 = float(sub["log_mass"].mean())
        path = os.path.join(reg_dir, "country_neighborhood", f"{iso}.json")
        write_json(path, {"scope": "country_neighborhood", "country_iso": iso, "slope": s, "slope_lo": lo, "slope_hi": hi, "x0": x0, "y0": y0, "n": int(len(sub)), "r2": r2})

    # City neighborhood
    for city_id, sub in tqdm(neigh.groupby("city_id"), desc="neighborhood by city"):
        s, b, lo, hi, r2 = ols_slope_ci(sub["log_pop"].values, sub["log_mass"].values)
        x0 = float(sub["log_pop"].mean())
        y0 = float(sub["log_mass"].mean())
        path = os.path.join(reg_dir, "city_neighborhood", f"{int(city_id)}.json")
        write_json(path, {"scope": "city_neighborhood", "city_id": int(city_id), "slope": s, "slope_lo": lo, "slope_hi": hi, "x0": x0, "y0": y0, "n": int(len(sub)), "r2": r2})

    print("Wrote regression summaries to", reg_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

