#!/bin/bash
# launch_all_processing.sh - Check downloads and start processing on all VMs
# Run from local machine

set -e

PROJECT="ee-knhuang"

# VM configurations
declare -A VMS
VMS["osrm-india"]="asia-south1-c"
VMS["osrm-china"]="asia-east1-b"
VMS["osrm-asia-other"]="asia-southeast1-b"
VMS["osrm-global-small"]="europe-west1-b"
VMS["osrm-northam"]="us-central1-c"
VMS["osrm-europe"]="europe-west1-b"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "OSRM Processing Launcher - $(date)"
echo "============================================================"
echo ""

for vm in "${!VMS[@]}"; do
    zone=${VMS[$vm]}
    echo -e "${YELLOW}=== $vm ===${NC}"

    # Check if downloads are complete
    result=$(gcloud compute ssh $vm --zone=$zone --project=$PROJECT --tunnel-through-iap --command="
        # Check if wget is running
        if pgrep -f wget > /dev/null; then
            echo 'STATUS:DOWNLOADING'
            exit 0
        fi

        # Check if processing is already running
        if pgrep -f 'process_cities.py' > /dev/null; then
            completed=\$(ls ~/results/*.json 2>/dev/null | wc -l)
            echo \"STATUS:RUNNING:\$completed\"
            exit 0
        fi

        # Check if OSM files exist
        osm_count=\$(ls ~/osrm-data/*.osm.pbf 2>/dev/null | wc -l)
        if [ \"\$osm_count\" -eq 0 ]; then
            echo 'STATUS:NO_OSM'
            exit 0
        fi

        # Ready to start
        echo \"STATUS:READY:\$osm_count\"
    " 2>/dev/null)

    status=$(echo "$result" | grep "STATUS:" | cut -d: -f2)

    case $status in
        "DOWNLOADING")
            echo -e "  ${YELLOW}Downloads in progress - waiting...${NC}"
            ;;
        "RUNNING")
            completed=$(echo "$result" | grep "STATUS:" | cut -d: -f3)
            echo -e "  ${GREEN}Already running - $completed cities completed${NC}"
            ;;
        "NO_OSM")
            echo -e "  ${RED}No OSM files found - check downloads${NC}"
            ;;
        "READY")
            osm_count=$(echo "$result" | grep "STATUS:" | cut -d: -f3)
            echo -e "  ${GREEN}Ready with $osm_count OSM files - starting processing...${NC}"

            # Start processing
            gcloud compute ssh $vm --zone=$zone --project=$PROJECT --tunnel-through-iap --command="
                source ~/osrm_env/bin/activate
                cd ~
                nohup python3 ~/process_cities.py > ~/processing.log 2>&1 &
                echo 'Processing started!'
            " 2>/dev/null
            ;;
        *)
            echo -e "  ${RED}Unknown status: $result${NC}"
            ;;
    esac
    echo ""
done

echo "============================================================"
echo "Use ./monitor_all.sh to check progress"
echo "Use ./check_issues.sh to check for problems"
echo "============================================================"
