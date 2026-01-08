#!/bin/bash
#
# Master orchestration script for processing all 13,135 cities
# Submits SLURM job arrays for each country
#
# Usage:
#   bash submit_all_countries.sh           # Submit all countries
#   bash submit_all_countries.sh --dry-run # Show what would be submitted
#   bash submit_all_countries.sh --wave 1  # Submit only wave 1 (large countries)

set -e

WORK_DIR=/scratch/kh3657/osrm
SLURM_DIR=$WORK_DIR/slurm
CITY_LISTS=$WORK_DIR/city_lists
RESULTS_DIR=$WORK_DIR/results
MAX_CONCURRENT=20

DRY_RUN=false
WAVE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --wave)
            WAVE=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to count remaining cities for a country
count_remaining() {
    local country=$1
    local city_list=$CITY_LISTS/${country}_cities.txt

    if [ ! -f "$city_list" ]; then
        echo 0
        return
    fi

    local total=$(wc -l < $city_list)
    local done=0

    while read city_id; do
        if [ -f "$RESULTS_DIR/${city_id}_matrix.json" ]; then
            ((done++)) || true
        fi
    done < $city_list

    echo $((total - done))
}

# Function to get OSM file for a country
get_osm_file() {
    local country=$1

    # Read from country_summary.json
    python3 -c "
import json
with open('$CITY_LISTS/country_summary.json') as f:
    data = json.load(f)
for c in data:
    if c['filename'].replace('_cities.txt', '') == '$country':
        osm_path = c['osm_path']
        if osm_path != 'UNKNOWN':
            print(osm_path.replace('/', '_') + '-latest.osm.pbf')
        break
" 2>/dev/null || echo ""
}

# Function to submit a country job
submit_country() {
    local country=$1
    local n_cities=$2
    local osm_file=$3

    if [ -z "$osm_file" ] || [ "$osm_file" == "" ]; then
        echo "  SKIP: No OSM file mapping for $country"
        return
    fi

    local remaining=$(count_remaining $country)

    if [ $remaining -eq 0 ]; then
        echo "  SKIP: $country already complete"
        return
    fi

    echo "  Submitting $country: $remaining/$n_cities cities remaining"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] sbatch --job-name=osrm_${country} --array=1-${n_cities}%${MAX_CONCURRENT} $SLURM_DIR/process_country.slurm $country $osm_file"
    else
        sbatch --job-name=osrm_${country} \
               --array=1-${n_cities}%${MAX_CONCURRENT} \
               $SLURM_DIR/process_country.slurm $country $osm_file
        sleep 1  # Avoid overwhelming scheduler
    fi
}

echo "===== OSRM City Processing - Submit All ====="
echo "Work directory: $WORK_DIR"
echo "Max concurrent per country: $MAX_CONCURRENT"
echo "Dry run: $DRY_RUN"
echo ""

# =============================================================================
# Wave 1: Large Countries (>500 cities)
# =============================================================================
if [ -z "$WAVE" ] || [ "$WAVE" = "1" ]; then
    echo "=== Wave 1: Large Countries ==="

    submit_country "india" 3248 "$(get_osm_file india)"
    submit_country "china" 1850 "$(get_osm_file china)"
    submit_country "ethiopia" 557 "$(get_osm_file ethiopia)"
    submit_country "nigeria" 483 "$(get_osm_file nigeria)"

    echo ""
fi

# =============================================================================
# Wave 2: Medium Countries (100-500 cities)
# =============================================================================
if [ -z "$WAVE" ] || [ "$WAVE" = "2" ]; then
    echo "=== Wave 2: Medium Countries ==="

    submit_country "indonesia" 393 "$(get_osm_file indonesia)"
    submit_country "brazil" 349 "$(get_osm_file brazil)"
    submit_country "united_states" 324 "$(get_osm_file united_states)"
    submit_country "bangladesh" 301 "$(get_osm_file bangladesh)"
    submit_country "pakistan" 301 "$(get_osm_file pakistan)"
    submit_country "russia" 209 "$(get_osm_file russia)"
    submit_country "egypt" 190 "$(get_osm_file egypt)"
    submit_country "iran" 182 "$(get_osm_file iran)"
    submit_country "mexico" 168 "$(get_osm_file mexico)"
    submit_country "philippines" 160 "$(get_osm_file philippines)"
    submit_country "tanzania" 156 "$(get_osm_file tanzania)"
    submit_country "vietnam" 144 "$(get_osm_file vietnam)"
    submit_country "sudan" 138 "$(get_osm_file sudan)"
    submit_country "myanmar" 137 "$(get_osm_file myanmar)"
    submit_country "democratic_republic_of_the_congo" 134 "$(get_osm_file democratic_republic_of_the_congo)"
    submit_country "turkey" 132 "$(get_osm_file turkey)"
    submit_country "kenya" 127 "$(get_osm_file kenya)"
    submit_country "south_africa" 121 "$(get_osm_file south_africa)"
    submit_country "uganda" 115 "$(get_osm_file uganda)"
    submit_country "japan" 109 "$(get_osm_file japan)"
    submit_country "algeria" 104 "$(get_osm_file algeria)"
    submit_country "morocco" 101 "$(get_osm_file morocco)"

    echo ""
fi

# =============================================================================
# Wave 3: Small Countries (<100 cities)
# =============================================================================
if [ -z "$WAVE" ] || [ "$WAVE" = "3" ]; then
    echo "=== Wave 3: Small Countries ==="

    # Read all country files and submit those not already covered
    for city_list in $CITY_LISTS/*_cities.txt; do
        country=$(basename $city_list _cities.txt)

        # Skip pilot
        if [ "$country" = "pilot" ]; then
            continue
        fi

        # Skip large/medium countries (already handled in waves 1-2)
        case $country in
            india|china|ethiopia|nigeria|indonesia|brazil|united_states|\
            bangladesh|pakistan|russia|egypt|iran|mexico|philippines|\
            tanzania|vietnam|sudan|myanmar|democratic_republic_of_the_congo|\
            turkey|kenya|south_africa|uganda|japan|algeria|morocco)
                continue
                ;;
        esac

        n_cities=$(wc -l < $city_list)
        osm_file=$(get_osm_file $country)

        submit_country "$country" "$n_cities" "$osm_file"
    done

    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "===== Submission Complete ====="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] No jobs were submitted"
else
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  python3 $WORK_DIR/scripts/check_progress.py"
fi
