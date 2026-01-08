#!/bin/bash
# monitor_all.sh - Monitor progress on all OSRM VMs
# Usage: ./monitor_all.sh [--loop]

set -e

# VM configurations
declare -A VMS
VMS["osrm-india"]="asia-south1-c:3248"
VMS["osrm-china"]="asia-east1-b:1850"
VMS["osrm-asia-other"]="asia-southeast1-b:2639"
VMS["osrm-global-small"]="europe-west1-b:3967"
VMS["osrm-northam"]="us-central1-c:372"
VMS["osrm-europe"]="europe-west1-b:1059"

PROJECT="ee-knhuang"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

check_vm() {
    local vm=$1
    local zone=$(echo ${VMS[$vm]} | cut -d: -f1)
    local total=$(echo ${VMS[$vm]} | cut -d: -f2)

    echo -e "${BLUE}=== $vm ===${NC}"

    # Check VM status
    status=$(gcloud compute instances describe $vm --zone=$zone --project=$PROJECT --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [ "$status" != "RUNNING" ]; then
        echo -e "  Status: ${RED}$status${NC}"
        return
    fi

    # Get progress info
    result=$(gcloud compute ssh $vm --zone=$zone --project=$PROJECT --tunnel-through-iap --command="
        completed=\$(ls ~/results/*.json 2>/dev/null | wc -l)
        disk=\$(df -h / | tail -1 | awk '{print \$5}')
        last_log=\$(tail -1 ~/processing.log 2>/dev/null || echo 'No log')
        errors=\$(grep -c 'ERROR' ~/processing.log 2>/dev/null || echo '0')
        running=\$(docker ps -q --filter ancestor=osrm/osrm-backend | wc -l)
        python_running=\$(pgrep -f 'process_cities.py' | wc -l)
        echo \"COMPLETED:\$completed\"
        echo \"DISK:\$disk\"
        echo \"ERRORS:\$errors\"
        echo \"DOCKER:\$running\"
        echo \"PYTHON:\$python_running\"
        echo \"LOG:\$last_log\"
    " 2>/dev/null || echo "CONNECTION_FAILED")

    if [ "$result" == "CONNECTION_FAILED" ]; then
        echo -e "  Status: ${RED}Connection failed${NC}"
        return
    fi

    # Parse results
    completed=$(echo "$result" | grep "COMPLETED:" | cut -d: -f2)
    disk=$(echo "$result" | grep "DISK:" | cut -d: -f2)
    errors=$(echo "$result" | grep "ERRORS:" | cut -d: -f2)
    docker_running=$(echo "$result" | grep "DOCKER:" | cut -d: -f2)
    python_running=$(echo "$result" | grep "PYTHON:" | cut -d: -f2)
    last_log=$(echo "$result" | grep "LOG:" | cut -d: -f2-)

    # Calculate progress
    if [ -n "$completed" ] && [ "$completed" -gt 0 ]; then
        pct=$((completed * 100 / total))
    else
        pct=0
        completed=0
    fi

    # Display with colors
    if [ "$pct" -eq 100 ]; then
        echo -e "  Progress: ${GREEN}$completed/$total ($pct%)${NC}"
    elif [ "$pct" -gt 0 ]; then
        echo -e "  Progress: ${YELLOW}$completed/$total ($pct%)${NC}"
    else
        echo -e "  Progress: $completed/$total ($pct%)"
    fi

    echo "  Disk: $disk"

    if [ "$errors" -gt 0 ]; then
        echo -e "  Errors: ${RED}$errors${NC}"
    else
        echo "  Errors: 0"
    fi

    if [ "$python_running" -gt 0 ]; then
        echo -e "  Processing: ${GREEN}Running${NC}"
    else
        echo -e "  Processing: ${RED}Not running${NC}"
    fi

    echo "  Last: ${last_log:0:60}"
    echo ""
}

# Main monitoring
monitor_once() {
    echo "============================================================"
    echo "OSRM Processing Monitor - $(date)"
    echo "============================================================"
    echo ""

    for vm in "${!VMS[@]}"; do
        check_vm "$vm"
    done

    echo "============================================================"
    echo "Summary"
    echo "============================================================"
    total_completed=0
    total_cities=0
    for vm in "${!VMS[@]}"; do
        zone=$(echo ${VMS[$vm]} | cut -d: -f1)
        cities=$(echo ${VMS[$vm]} | cut -d: -f2)
        total_cities=$((total_cities + cities))
        completed=$(gcloud compute ssh $vm --zone=$zone --project=$PROJECT --tunnel-through-iap --command="ls ~/results/*.json 2>/dev/null | wc -l" 2>/dev/null || echo "0")
        total_completed=$((total_completed + completed))
    done
    echo "Total completed: $total_completed / $total_cities"
    if [ "$total_cities" -gt 0 ]; then
        echo "Overall progress: $((total_completed * 100 / total_cities))%"
    fi
}

# Check for loop mode
if [ "$1" == "--loop" ]; then
    while true; do
        clear
        monitor_once
        echo ""
        echo "Refreshing in 5 minutes... (Ctrl+C to stop)"
        sleep 300
    done
else
    monitor_once
fi
