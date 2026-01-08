#!/bin/bash
# collect_results.sh - Download all results from VMs to local machine
# Run from local machine

set -e

PROJECT="ee-knhuang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../all_results"

# VM configurations
declare -A VMS
VMS["osrm-india"]="asia-south1-c"
VMS["osrm-china"]="asia-east1-b"
VMS["osrm-asia-other"]="asia-southeast1-b"
VMS["osrm-global-small"]="europe-west1-b"
VMS["osrm-northam"]="us-central1-c"
VMS["osrm-europe"]="europe-west1-b"

echo "============================================================"
echo "Collecting Results from All VMs - $(date)"
echo "============================================================"
echo ""

mkdir -p "$RESULTS_DIR"

total_files=0

for vm in "${!VMS[@]}"; do
    zone=${VMS[$vm]}
    echo "=== Downloading from $vm ==="

    # Count matrix files (*.json) and route files (*.geojson)
    matrix_count=$(gcloud compute ssh $vm --zone=$zone --project=$PROJECT --tunnel-through-iap --command="ls ~/results/*_matrix.json 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    routes_count=$(gcloud compute ssh $vm --zone=$zone --project=$PROJECT --tunnel-through-iap --command="ls ~/results/*_routes.geojson 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    echo "  Matrix files: $matrix_count"
    echo "  Route files: $routes_count"

    total_on_vm=$((matrix_count + routes_count))
    if [ "$total_on_vm" -gt 0 ]; then
        # Create VM-specific directory
        mkdir -p "$RESULTS_DIR/$vm"

        # Download all results (json and geojson)
        gcloud compute scp "$vm:~/results/*" "$RESULTS_DIR/$vm/" --zone=$zone --project=$PROJECT 2>/dev/null

        downloaded_json=$(ls "$RESULTS_DIR/$vm"/*.json 2>/dev/null | wc -l)
        downloaded_geojson=$(ls "$RESULTS_DIR/$vm"/*.geojson 2>/dev/null | wc -l)
        echo "  Downloaded: $downloaded_json json, $downloaded_geojson geojson"
        total_files=$((total_files + downloaded_json + downloaded_geojson))
    fi
    echo ""
done

echo "============================================================"
echo "Total files downloaded: $total_files"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"
