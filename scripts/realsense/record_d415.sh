#!/usr/bin/env bash
# ============================================================================
# Record RGB-D data with Intel RealSense D415 (no IMU required)
# ============================================================================
# Records color and depth streams from a RealSense D415 camera to a .bag
# file using pyrealsense2 directly. The .bag format preserves raw sensor
# data and allows replay for re-processing with different parameters.
#
# Unlike the D435i/D455 pipeline (which uses SpectacularAI for VIO),
# this script records raw streams only. Camera poses are estimated later
# using COLMAP during the processing step.
#
# Requirements:
#   - RealSense D415 connected via USB 3.0
#   - pyrealsense2 installed (pip install pyrealsense2)
#
# Usage:
#   ./scripts/realsense/record_d415.sh [OPTIONS]
#
# Options:
#   --output PATH       Output .bag file path (default: ./data/d415_<timestamp>.bag)
#   --width W           Stream width (default: 1280)
#   --height H          Stream height (default: 720)
#   --fps N             Frame rate (default: 30)
#   --no-preview        Disable live preview window
#   -h, --help          Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
OUTPUT=""
WIDTH="1280"
HEIGHT="720"
FPS="30"
PREVIEW=true

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)     OUTPUT="$2"; shift 2 ;;
        --width)      WIDTH="$2"; shift 2 ;;
        --height)     HEIGHT="$2"; shift 2 ;;
        --fps)        FPS="$2"; shift 2 ;;
        --no-preview) PREVIEW=false; shift ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# Generate default output path with timestamp
if [[ -z "$OUTPUT" ]]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT="${PROJECT_DIR}/data/d415_${TIMESTAMP}.bag"
fi

# Ensure output parent directory exists
mkdir -p "$(dirname "$OUTPUT")"

echo "============================================"
echo "  RealSense D415 Recording"
echo "============================================"
echo "  Output:     ${OUTPUT}"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  FPS:        ${FPS}"
echo "  Preview:    ${PREVIEW}"
echo "============================================"

# Check that pyrealsense2 is available
if ! python -c "import pyrealsense2" 2>/dev/null; then
    echo "[ERROR] pyrealsense2 not found."
    echo "  Install with: pip install pyrealsense2"
    exit 1
fi

# Build command
CMD=(python "${PROJECT_DIR}/dn_splatter/scripts/record_realsense.py"
     --output "$OUTPUT"
     --width "$WIDTH"
     --height "$HEIGHT"
     --fps "$FPS")

if [[ "$PREVIEW" == false ]]; then
    CMD+=(--no-preview)
fi

echo ""
echo "[INFO] Starting recording... Press Ctrl+C to stop."
echo "[CMD]  ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "[OK] Recording saved to: ${OUTPUT}"
echo ""
echo "Next step: process the recording"
echo "  ./scripts/realsense/process_d415.sh ${OUTPUT} ./datasets/custom/my_scene"
