#!/bin/bash
#
# Batch upload POI data to Google Earth Engine using earthengine CLI
# Automatically finds all *_pois_9cats.gpkg files and uploads those not already in GEE
#
# Usage:
#   bash 02b_batch_upload_pois_earthengine.sh
#   bash 02b_batch_upload_pois_earthengine.sh --dry-run  # Test without uploading

set -e  # Exit on error

# Configuration
POI_DIR="../data/raw/pois"
ASSET_FOLDER="projects/ee-knhuang/assets/global_cities_POIs"
DRY_RUN=false

# Earth Engine CLI path (use full path from virtual environment)
EARTHENGINE_CMD="/Users/kangninghuang/.venvs/nyu_china_grant_env/bin/earthengine"

# Parse arguments
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No uploads will be performed"
fi

echo "============================================================"
echo "POI Batch Upload to Google Earth Engine"
echo "============================================================"
echo ""

# Check if earthengine CLI is available
if ! command -v "$EARTHENGINE_CMD" &> /dev/null; then
    echo "❌ ERROR: earthengine CLI not found at: $EARTHENGINE_CMD"
    echo "   Please ensure Earth Engine CLI is installed"
    echo "   Install: pip install earthengine-api"
    echo "   Authenticate: earthengine authenticate"
    exit 1
fi

echo "✓ Earth Engine CLI found"
echo ""

# Get list of existing assets in GEE
echo "📋 Checking existing assets in GEE..."
echo "   Folder: $ASSET_FOLDER"

# Create folder if it doesn't exist
if ! "$EARTHENGINE_CMD" ls "$ASSET_FOLDER" &> /dev/null; then
    echo "   ⚠️  Folder does not exist. Creating: $ASSET_FOLDER"
    if [[ "$DRY_RUN" == false ]]; then
        "$EARTHENGINE_CMD" create folder "$ASSET_FOLDER" || true
    fi
fi

# Get existing assets
EXISTING_ASSETS=$("$EARTHENGINE_CMD" ls "$ASSET_FOLDER" 2>/dev/null || echo "")

if [[ -z "$EXISTING_ASSETS" ]]; then
    echo "   No existing assets found"
    EXISTING_COUNT=0
else
    EXISTING_COUNT=$(echo "$EXISTING_ASSETS" | wc -l | tr -d ' ')
    echo "   Found $EXISTING_COUNT existing assets:"
    echo "$EXISTING_ASSETS" | sed 's/^/     - /'
fi

echo ""

# Find all POI geopackage files
echo "🔍 Searching for POI files in: $POI_DIR"
POI_FILES=$(find "$POI_DIR" -name "*_pois_9cats.gpkg" -type f | sort)

if [[ -z "$POI_FILES" ]]; then
    echo "❌ No POI files found matching pattern: *_pois_9cats.gpkg"
    exit 1
fi

TOTAL_FILES=$(echo "$POI_FILES" | wc -l | tr -d ' ')
echo "   Found $TOTAL_FILES POI files"
echo ""

# Process each file
UPLOADED=0
SKIPPED=0
FAILED=0

echo "============================================================"
echo "Processing Files"
echo "============================================================"
echo ""

while IFS= read -r poi_file; do
    # Extract city name from filename
    filename=$(basename "$poi_file")
    city_name="${filename%_pois_9cats.gpkg}"

    # Get absolute path
    abs_poi_file=$(cd "$(dirname "$poi_file")" && pwd)/$(basename "$poi_file")

    # Construct asset ID
    asset_id="$ASSET_FOLDER/${city_name}_pois_9cats"

    echo "─────────────────────────────────────────────────────────"
    echo "City: $city_name"
    echo "  File: $filename"
    echo "  Asset: $asset_id"

    # Check if asset already exists
    if echo "$EXISTING_ASSETS" | grep -q "${city_name}_pois_9cats"; then
        echo "  Status: ⏭️  SKIPPED (already exists)"
        ((SKIPPED++))
        echo ""
        continue
    fi

    # Get file size
    file_size=$(du -h "$abs_poi_file" | cut -f1)
    echo "  Size: $file_size"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  Status: 🔍 WOULD UPLOAD (dry run)"
        ((UPLOADED++))
    else
        echo "  Status: ⬆️  UPLOADING..."

        # Upload to GEE (use absolute path)
        if "$EARTHENGINE_CMD" upload table --asset_id="$asset_id" "$abs_poi_file" 2>&1; then
            echo "  Status: ✅ UPLOAD STARTED"
            ((UPLOADED++))
        else
            echo "  Status: ❌ UPLOAD FAILED"
            ((FAILED++))
        fi
    fi

    echo ""

done <<< "$POI_FILES"

# Summary
echo "============================================================"
echo "Upload Summary"
echo "============================================================"
echo ""
echo "Total files found:    $TOTAL_FILES"
echo "Already in GEE:       $SKIPPED"
echo "Newly uploaded:       $UPLOADED"
echo "Failed uploads:       $FAILED"
echo ""

if [[ "$DRY_RUN" == false ]] && [[ $UPLOADED -gt 0 ]]; then
    echo "============================================================"
    echo "Monitor Upload Progress"
    echo "============================================================"
    echo ""
    echo "The upload tasks have been started. Monitor progress using:"
    echo ""
    echo "  Option 1: GEE Code Editor → Tasks tab"
    echo "           https://code.earthengine.google.com/"
    echo ""
    echo "  Option 2: Command line"
    echo "           earthengine task list"
    echo ""
    echo "  Option 3: Check specific task"
    echo "           earthengine task info TASK_ID"
    echo ""
    echo "Uploads will appear as table ingestion tasks and may take"
    echo "several minutes to hours depending on file size."
    echo ""
elif [[ "$DRY_RUN" == true ]]; then
    echo "To perform actual upload, run:"
    echo "  bash $0"
    echo ""
fi

# Exit with error if any uploads failed
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
