Web data preparation pipeline

This folder contains Python scripts to generate static artifacts for the interactive website.

Key outputs (written under `web/public/webdata/` by default):
- countries.geojson: country polygons and ISO3 codes for clickable map regions.
- cities_agg/: aggregated city totals for city-level scatter (global and per-country partitions).
- scatter_samples/: subsampled neighborhood points for global and per-country neighborhood scatters.
- hex/: per-city H3 hex feeds with population and built mass for map coloring.
- regression/: regression summaries with 95% CIs for city and neighborhood levels across scopes.
- index/: lookup tables (country_to_cities, city_meta with a representative H3 index per city).

Environment
- Python 3.9+
- Recommended packages: pandas, numpy, pyproj (optional), requests, tqdm, pydantic (optional), (optional) h3 if you want to compute centroids offline.

Install (conda or pip)
    pip install pandas numpy requests tqdm

Usage
1) Download countries GeoJSON (Natural Earth derived) once:
    python download_countries_geojson.py --out ../../web/public/webdata/countries.geojson

2) Build lookup indices and city aggregates:
    python prep_city_aggregates.py \
      --neighborhoods ../../data/Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv \
      --out_dir ../../web/public/webdata

3) Build neighborhood scatter subsamples:
    python prep_neighborhood_subsamples.py \
      --neighborhoods ../../data/Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv \
      --out_dir ../../web/public/webdata \
      --global_n 100000 --per_country_n 20000

4) Split per-city hex feeds (H3 indices + metrics):
    python split_city_hex_feeds.py \
      --neighborhoods ../../data/Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv \
      --out_dir ../../web/public/webdata/hex

5) Compute regressions (slope, 95% CI) for city and neighborhood levels:
    python compute_regressions.py \
      --city_agg ../../web/public/webdata/cities_agg/global.json \
      --neighborhoods ../../data/Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv \
      --out_dir ../../web/public/webdata

Notes
- All outputs are JSON (uncompressed) to work smoothly on GitHub Pages. If you prefer gzip, add the `--gzip` flag where available and use client-side decompression.
- City centroids for rendering are computed client-side from a representative H3 index (stored in `index/city_meta.json`). This avoids requiring Geo packages locally.
- Countries GeoJSON is sourced from a public dataset and reduced to attributes we need.

