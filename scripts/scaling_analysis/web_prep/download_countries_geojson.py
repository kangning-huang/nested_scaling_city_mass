#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output GeoJSON path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Downloading countries GeoJSON…\n  {DATA_URL}")
    with urllib.request.urlopen(DATA_URL) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Keep only necessary properties
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        iso3 = props.get("ISO_A3") or props.get("ISO3") or props.get("ADMIN_ISO3") or props.get("ADM0_A3")
        name = props.get("ADMIN") or props.get("ADMIN_NAME") or props.get("NAME") or props.get("NAME_EN")
        feat["properties"] = {"iso3": iso3, "name": name}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    sys.exit(main())

