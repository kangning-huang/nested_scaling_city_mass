"""
Batch-download subway lines and stations from OpenStreetMap (Overpass)
for the Mao et al. (2021) 219-city list.

- Input: data/subway_networks/osm_target_cities.csv (region, paper_index, city)
- Output: per-city GeoJSON files under data/subway_networks/OSM/
  * {city_slug}_lines.geojson
  * {city_slug}_stations.geojson
- Summary: data/subway_networks/OSM/osm_download_summary.csv

Designed to be robust to occasional Overpass errors via endpoint rotation,
retry, and simple backoff. Requires outbound internet access.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

# ----- Configuration -----
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "codex-subway-downloader/0.1 (contact: kangninghuang@nyu.edu)"
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "subway_networks"
INPUT_CSV = DATA_ROOT / "osm_target_cities.csv"
OUTPUT_DIR = DATA_ROOT / "OSM"
SUMMARY_CSV = OUTPUT_DIR / "osm_download_summary.csv"
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 2.0  # seconds


@dataclass
class CityJob:
    region: str
    paper_index: int
    name: str

    @property
    def slug(self) -> str:
        s = self.name.lower().replace(" ", "_").replace("/", "-")
        s = "".join(ch for ch in s if ch.isalnum() or ch in {"_", "-"})
        return s


# ----- Helpers -----

def read_city_jobs() -> List[CityJob]:
    df = pd.read_csv(INPUT_CSV)
    jobs = [CityJob(region=row.region, paper_index=int(row.paper_index), name=row.city) for row in df.itertuples()]
    return jobs


def nominatim_bbox(city: str) -> Optional[List[float]]:
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        res = r.json()
        if not res:
            return None
        bb = res[0]["boundingbox"]  # [south, north, west, east]
        south, north, west, east = map(float, bb)
        return [south, west, north, east]
    except Exception:
        return None


def overpass_query(q: str, *, timeout: int = DEFAULT_TIMEOUT) -> Optional[dict]:
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"}
    for attempt in range(MAX_RETRIES):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            resp = requests.post(endpoint, data=q.encode("utf-8"), headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        # backoff
        time.sleep(2 ** attempt)
    return None


def build_station_query(bbox: List[float]) -> str:
    s, w, n, e = bbox
    return f"""
[out:json][timeout:{DEFAULT_TIMEOUT}][bbox:{s},{w},{n},{e}];
(
  node["station"="subway"];
  node["railway"="station"]["subway"="yes"];
  node["public_transport"="station"]["subway"="yes"];
  node["railway"="halt"]["subway"="yes"];
);
out body;
"""


def build_line_query(bbox: List[float]) -> str:
    s, w, n, e = bbox
    return f"""
[out:json][timeout:{DEFAULT_TIMEOUT}][bbox:{s},{w},{n},{e}];
(
  way["railway"="subway"];
  way["railway"="light_rail"]["service"!~"yard|siding|spur"];
  relation["route"="subway"];
  way(r);
);
out geom;
"""


def overpass_to_geojson_elements(data: dict, *, wanted_types: Iterable[str]) -> List[dict]:
    features: List[dict] = []
    for elem in data.get("elements", []):
        if elem.get("type") not in wanted_types:
            continue
        if elem["type"] == "node":
            coords = [elem["lon"], elem["lat"]]
            geom = {"type": "Point", "coordinates": coords}
        elif elem["type"] == "way":
            if "geometry" not in elem:
                continue
            coords = [[pt["lon"], pt["lat"]] for pt in elem["geometry"]]
            geom = {"type": "LineString", "coordinates": coords}
        elif elem["type"] == "relation":
            # Attempt to build MultiLineString from member geometries if present.
            lines = []
            for mem in elem.get("members", []):
                geom_mem = mem.get("geometry")
                if not geom_mem:
                    continue
                coords = [[pt["lon"], pt["lat"]] for pt in geom_mem]
                lines.append(coords)
            if not lines:
                continue
            geom = {"type": "MultiLineString", "coordinates": lines}
        else:
            continue

        props = {k: v for k, v in elem.items() if k not in {"type", "id", "lat", "lon", "nodes", "geometry", "members"}}
        props["osm_id"] = elem.get("id")
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    return features


def save_geojson(path: Path, features: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload))


def process_city(job: CityJob) -> dict:
    bbox = nominatim_bbox(job.name)
    if not bbox:
        return {"city": job.name, "status": "geocode_failed"}

    station_data = overpass_query(build_station_query(bbox))
    line_data = overpass_query(build_line_query(bbox))

    if station_data is None or line_data is None:
        return {"city": job.name, "status": "overpass_failed"}

    station_features = overpass_to_geojson_elements(station_data, wanted_types=["node"])
    line_features = overpass_to_geojson_elements(line_data, wanted_types=["way", "relation"])

    save_geojson(OUTPUT_DIR / f"{job.slug}_stations.geojson", station_features)
    save_geojson(OUTPUT_DIR / f"{job.slug}_lines.geojson", line_features)

    summary = {
        "city": job.name,
        "slug": job.slug,
        "region": job.region,
        "n_stations": len(station_features),
        "n_lines": len(line_features),
        "status": "ok",
    }
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = read_city_jobs()
    summaries = []
    for job in jobs:
        print(f"\n=== {job.name} ===")
        summary = process_city(job)
        summaries.append(summary)
        print(summary)
        time.sleep(SLEEP_BETWEEN_CALLS)

    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
