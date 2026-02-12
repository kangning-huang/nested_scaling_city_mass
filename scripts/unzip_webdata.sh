#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_ROOT="${SCRIPT_DIR}/../web"
WEBDATA_DIR="${WEB_ROOT}/public/webdata"

if [ ! -d "${WEBDATA_DIR}" ]; then
  echo "Expected web data directory at ${WEBDATA_DIR} (from repo root)."
  exit 1
fi

shopt -s nullglob
zips=("${WEBDATA_DIR}"/*.zip)

if [ ${#zips[@]} -eq 0 ]; then
  echo "No *.zip archives found under ${WEBDATA_DIR}."
  exit 0
fi

for zip_path in "${zips[@]}"; do
  zip_name="$(basename "${zip_path}")"
  base_name="${zip_name%.zip}"
  dest_dir="${WEBDATA_DIR}/${base_name}"

  echo "Unzipping ${zip_name} -> ${dest_dir#${WEB_ROOT}/}"
  rm -rf "${dest_dir}"
  mkdir -p "${dest_dir}"
  unzip -qo "${zip_path}" -d "${dest_dir}"
  rm -rf "${dest_dir}/__MACOSX"
done

echo "Done expanding web data archives."
