from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import geopandas as gpd
import requests
from shapely.geometry import LineString


ROUTES_ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline"
DEFAULT_LAYER = "all_cities_h3_grids"


@dataclass
class RouteResult:
    distance_m: Optional[float]
    duration_min: Optional[float]
    geometry: LineString


def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    """Decode a polyline string into a list of (lat, lon) pairs.

    Implemented inline to avoid extra dependencies.
    """

    coords: List[Tuple[float, float]] = []
    index = lat = lng = 0

    while index < len(encoded):
        shift = result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat

        shift = result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))

    return coords


def parse_duration_to_minutes(duration: Optional[str]) -> Optional[float]:
    if not duration:
        return None
    if duration.endswith("s"):
        try:
            return float(duration[:-1]) / 60.0
        except ValueError:
            return None
    try:
        return float(duration) / 60.0
    except (TypeError, ValueError):
        return None


def choose_population_column(gdf: gpd.GeoDataFrame, user_choice: Optional[str]) -> Optional[str]:
    candidates: Iterable[Optional[str]] = (
        user_choice,
        "population",
        "population_2015",
        "pop_est",
        "P15_numeric",
    )
    for col in candidates:
        if col and col in gdf.columns:
            return col
    return None


def load_city_hexes(
    gpkg_path: Path, city: str, country_iso: Optional[str], layer: str = DEFAULT_LAYER
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path, layer=layer)

    if "UC_NM_MN" not in gdf.columns:
        raise ValueError("Expected column 'UC_NM_MN' to filter cities; not found in grid file.")

    city_mask = gdf["UC_NM_MN"].str.lower() == city.lower()
    if country_iso and "CTR_MN_ISO" in gdf.columns:
        city_mask &= gdf["CTR_MN_ISO"].str.lower() == country_iso.lower()

    subset = gdf.loc[city_mask].copy()
    if subset.empty:
        raise ValueError(f"No hexagons found for city '{city}' with ISO '{country_iso}'.")

    subset["centroid"] = subset.geometry.centroid
    return subset


def compute_route(
    session: requests.Session,
    api_key: str,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    travel_mode: str,
) -> Optional[RouteResult]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    body = {
        "origin": {
            "location": {
                "latLng": {"latitude": origin[1], "longitude": origin[0]}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": destination[1], "longitude": destination[0]}
            }
        },
        "travelMode": travel_mode,
        "routingPreference": "TRAFFIC_AWARE",
    }

    response = session.post(ROUTES_ENDPOINT, headers=headers, json=body, timeout=30)
    if not response.ok:
        sys.stderr.write(
            f"Routes API error {response.status_code}: {response.text}\n"
        )
        return None

    payload = response.json()
    routes = payload.get("routes", [])
    if not routes:
        return None

    route = routes[0]
    distance_m = route.get("distanceMeters")
    duration_min = parse_duration_to_minutes(route.get("duration"))

    encoded_polyline = None
    polyline_info = route.get("polyline") or {}
    encoded_polyline = polyline_info.get("encodedPolyline")

    geometry = None
    if encoded_polyline:
        try:
            coords = decode_polyline(encoded_polyline)
            geometry = LineString([(lng, lat) for lat, lng in coords])
        except Exception:
            geometry = None

    if geometry is None:
        geometry = LineString([origin, destination])

    return RouteResult(distance_m=distance_m, duration_min=duration_min, geometry=geometry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query Google Maps Routes API for pairwise travel between hexagons in a city "
            "and write results to a GPKG line layer."
        )
    )
    parser.add_argument("--city", required=True, help="City name as stored in UC_NM_MN column.")
    parser.add_argument(
        "--country-iso",
        help="Optional ISO3 country code to disambiguate cities sharing the same name.",
    )
    parser.add_argument(
        "--api-key",
        help="Google Maps API key. Defaults to GOOGLE_MAPS_API_KEY env var if omitted.",
    )
    parser.add_argument(
        "--travel-mode",
        default="DRIVE",
        choices=["DRIVE", "TWO_WHEELER", "WALK", "BICYCLE"],
        help="Travel mode for routing.",
    )
    parser.add_argument(
        "--population-column",
        help="Optional population column name in the grid file to use for end_grid_population.",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        help="Override path to all_cities_h3_grids.gpkg (defaults to repo data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults to repo results/routes_GoogleMaps).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Optional cap on number of origin-destination pairs for dry runs/testing.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between API calls to avoid rate limits.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    gpkg_path = args.grid_path or repo_root / "data" / "raw" / "all_cities_h3_grids.gpkg"
    output_dir = args.output_dir or repo_root / "results" / "routes_GoogleMaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise SystemExit("Provide a Google Maps API key via --api-key or GOOGLE_MAPS_API_KEY env var.")

    hexes = load_city_hexes(gpkg_path=gpkg_path, city=args.city, country_iso=args.country_iso)

    pop_col = choose_population_column(hexes, args.population_column)
    if pop_col is None:
        sys.stderr.write(
            "No population column found; end_grid_population will be null.\n"
        )

    records = []
    session = requests.Session()

    total_pairs = len(hexes) * (len(hexes) - 1)
    pair_counter = 0
    max_pairs = args.max_pairs if args.max_pairs and args.max_pairs > 0 else total_pairs

    for origin_idx, origin_row in hexes.iterrows():
        origin_point = (origin_row.centroid.x, origin_row.centroid.y)
        for dest_idx, dest_row in hexes.iterrows():
            if origin_idx == dest_idx:
                continue
            if pair_counter >= max_pairs:
                break

            destination_point = (dest_row.centroid.x, dest_row.centroid.y)
            result = compute_route(
                session=session,
                api_key=api_key,
                origin=origin_point,
                destination=destination_point,
                travel_mode=args.travel_mode,
            )

            pair_counter += 1

            if result is None:
                continue

            pop_value = None
            if pop_col:
                value = dest_row.get(pop_col)
                if isinstance(value, str):
                    try:
                        pop_value = float(value)
                    except ValueError:
                        pop_value = None
                else:
                    pop_value = value

            records.append(
                {
                    "start_grid": origin_row.get("h3index") or origin_row.get("h3_index"),
                    "end_grid": dest_row.get("h3index") or dest_row.get("h3_index"),
                    "end_grid_population": pop_value,
                    "distance_m": result.distance_m,
                    "duration_min": result.duration_min,
                    "geometry": result.geometry,
                }
            )

            if args.sleep > 0:
                time.sleep(args.sleep)
        if pair_counter >= max_pairs:
            break

    if not records:
        raise SystemExit("No routes generated; check inputs and API response.")

    routes_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    safe_city = args.city.replace(" ", "_")
    out_path = output_dir / f"{safe_city}_GoogleMaps_routes.gpkg"
    routes_gdf.to_file(out_path, driver="GPKG")

    print(f"Wrote {len(routes_gdf)} routes to {out_path}")


if __name__ == "__main__":
    main()
