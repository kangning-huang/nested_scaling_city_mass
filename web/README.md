Web app (Vite + React + MapLibre + deck.gl)

Getting started
1) Install Node 18+ and npm.
2) Install deps: `npm ci` (inside `web/`).
3) Provide a MapTiler key via `VITE_MAPTILER_KEY` env var for the basemap.
4) Run dev: `npm run dev`.
5) Avoid unzipping `public/webdata/*.zip` locally (Google Drive friendly); the CI workflow expands them during deploy.

Remote data base (optional)
- To avoid storing heavy per-city hex files locally, set `VITE_DATA_BASE` to a remote base URL hosting `webdata/`:
  - Example: `VITE_DATA_BASE=https://kangning-huang.github.io/nested-scaling-city-mass/webdata`
  - The app fetches all data from `${VITE_DATA_BASE}/...` instead of local `/webdata/...`.
  - You can choose to host only `webdata/hex/` remotely while keeping light JSON locally; or host the entire `webdata/` remotely.

Data
- Static artifacts are served from `public/webdata/`. Generate with scripts in `scripts/web_prep/`.
- Minimal required files to render basic UI:
  - `webdata/countries.geojson`
  - `webdata/cities_agg/global.json`
  - `webdata/scatter_samples/global_neighborhood.json`
  - `webdata/regression/global_city.json`
  - `webdata/regression/global_neighborhood.json`
  - Per-country and per-city files enhance the experience.

Deploy to GitHub Pages
- Push to `main` or `master`. The workflow builds `web/` and publishes to `gh-pages`.
- Set `MAPTILER_KEY` repository secret.
- Pages URL: `https://<user>.github.io/<repo>/`.

Packaging heavy data (optional)
- To keep local sync fast (e.g., with Google Drive), you can zip folders with many JSON files and remove the originals:
  - Hex: `python scripts/web_prep/pack_hex_to_zip.py --hex_dir web/public/webdata/hex --zip_path web/public/webdata/hex.zip --remove`
  - City aggregates: `(cd web/public/webdata/cities_agg && zip -q -r ../cities_agg.zip . -i '*.json'); rm -rf web/public/webdata/cities_agg`
  - Neighborhood samples: `(cd web/public/webdata/scatter_samples && zip -q -r ../scatter_samples.zip . -i '*.json'); rm -rf web/public/webdata/scatter_samples`
- The GitHub Actions workflow will automatically unzip any of `hex.zip`, `cities_agg.zip`, and `scatter_samples.zip` during deployment; do not unzip locally.
  - Alternatively, host these remotely and set `VITE_DATA_BASE` to the remote base URL.
