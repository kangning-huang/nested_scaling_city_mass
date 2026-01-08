#!/bin/bash
# check_issues.sh - Check for issues on all OSRM VMs and report problems
# Usage: ./check_issues.sh

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
ISSUES_FOUND=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "OSRM Issue Checker - $(date)"
echo "============================================================"
echo ""

for vm in "${!VMS[@]}"; do
    zone=$(echo ${VMS[$vm]} | cut -d: -f1)
    total=$(echo ${VMS[$vm]} | cut -d: -f2)

    echo "Checking $vm..."

    # Check VM status
    status=$(gcloud compute instances describe $vm --zone=$zone --project=$PROJECT --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [ "$status" == "NOT_FOUND" ]; then
        echo -e "  ${RED}ISSUE: VM not found${NC}"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
        continue
    fi

    if [ "$status" != "RUNNING" ]; then
        echo -e "  ${RED}ISSUE: VM is $status (not running)${NC}"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
        continue
    fi

    # Get detailed status
    result=$(gcloud compute ssh $vm --zone=$zone --project=$PROJECT --tunnel-through-iap --command="
        # Check disk usage
        disk_pct=\$(df / | tail -1 | awk '{print \$5}' | tr -d '%')
        echo \"DISK:\$disk_pct\"

        # Check if processing is running
        python_running=\$(pgrep -f 'process_cities.py' | wc -l)
        echo \"PYTHON:\$python_running\"

        # Check recent errors
        recent_errors=\$(tail -100 ~/processing.log 2>/dev/null | grep -c 'ERROR' || echo '0')
        echo \"RECENT_ERRORS:\$recent_errors\"

        # Check if Docker is working
        docker_ok=\$(docker ps >/dev/null 2>&1 && echo '1' || echo '0')
        echo \"DOCKER_OK:\$docker_ok\"

        # Check completed count
        completed=\$(ls ~/results/*.json 2>/dev/null | wc -l)
        echo \"COMPLETED:\$completed\"

        # Check if stuck (same log for 10 minutes)
        last_mod=\$(stat -c %Y ~/processing.log 2>/dev/null || echo '0')
        now=\$(date +%s)
        diff=\$((now - last_mod))
        echo \"LOG_AGE:\$diff\"

        # Check memory
        mem_avail=\$(free -m | grep Mem | awk '{print \$7}')
        echo \"MEM_AVAIL:\$mem_avail\"
    " 2>/dev/null || echo "CONNECTION_FAILED")

    if [ "$result" == "CONNECTION_FAILED" ]; then
        echo -e "  ${RED}ISSUE: Cannot connect to VM${NC}"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
        continue
    fi

    # Parse results
    disk_pct=$(echo "$result" | grep "DISK:" | cut -d: -f2)
    python_running=$(echo "$result" | grep "PYTHON:" | cut -d: -f2)
    recent_errors=$(echo "$result" | grep "RECENT_ERRORS:" | cut -d: -f2)
    docker_ok=$(echo "$result" | grep "DOCKER_OK:" | cut -d: -f2)
    completed=$(echo "$result" | grep "COMPLETED:" | cut -d: -f2)
    log_age=$(echo "$result" | grep "LOG_AGE:" | cut -d: -f2)
    mem_avail=$(echo "$result" | grep "MEM_AVAIL:" | cut -d: -f2)

    # Check for issues
    issues_for_vm=0

    # Disk usage > 90%
    if [ -n "$disk_pct" ] && [ "$disk_pct" -gt 90 ]; then
        echo -e "  ${RED}ISSUE: Disk usage at ${disk_pct}% (critical!)${NC}"
        issues_for_vm=$((issues_for_vm + 1))
    elif [ -n "$disk_pct" ] && [ "$disk_pct" -gt 80 ]; then
        echo -e "  ${YELLOW}WARNING: Disk usage at ${disk_pct}%${NC}"
    fi

    # Processing not running (but not complete)
    if [ -n "$python_running" ] && [ "$python_running" -eq 0 ] && [ -n "$completed" ] && [ "$completed" -lt "$total" ]; then
        echo -e "  ${RED}ISSUE: Processing stopped (${completed}/${total} complete)${NC}"
        issues_for_vm=$((issues_for_vm + 1))
    fi

    # Many recent errors
    if [ -n "$recent_errors" ] && [ "$recent_errors" -gt 10 ]; then
        echo -e "  ${YELLOW}WARNING: ${recent_errors} recent errors in log${NC}"
    fi

    # Docker not working
    if [ -n "$docker_ok" ] && [ "$docker_ok" -eq 0 ]; then
        echo -e "  ${RED}ISSUE: Docker not responding${NC}"
        issues_for_vm=$((issues_for_vm + 1))
    fi

    # Log not updated for 30 minutes (might be stuck)
    if [ -n "$log_age" ] && [ "$log_age" -gt 1800 ] && [ -n "$completed" ] && [ "$completed" -lt "$total" ]; then
        echo -e "  ${YELLOW}WARNING: No log activity for $((log_age / 60)) minutes${NC}"
    fi

    # Low memory
    if [ -n "$mem_avail" ] && [ "$mem_avail" -lt 1000 ]; then
        echo -e "  ${YELLOW}WARNING: Only ${mem_avail}MB memory available${NC}"
    fi

    if [ $issues_for_vm -eq 0 ]; then
        echo -e "  ${GREEN}OK${NC} - ${completed:-0}/${total} completed"
    fi

    ISSUES_FOUND=$((ISSUES_FOUND + issues_for_vm))
done

echo ""
echo "============================================================"
if [ $ISSUES_FOUND -gt 0 ]; then
    echo -e "${RED}Found $ISSUES_FOUND issue(s) requiring attention${NC}"
    echo ""
    echo "Recommended actions:"
    echo "  - For stopped processing: SSH in and restart process_cities.py"
    echo "  - For disk issues: Run cleanup or increase disk size"
    echo "  - For Docker issues: Restart Docker service"
    echo "  - For VM not running: gcloud compute instances start VM_NAME --zone=ZONE"
else
    echo -e "${GREEN}All VMs running normally${NC}"
fi
echo "============================================================"

exit $ISSUES_FOUND
