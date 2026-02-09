#!/usr/bin/env bash
# ============================================================================
# Record RGB-D data with Intel RealSense via SpectacularAI SDK
# ============================================================================
# Records color, depth, and IMU data from a RealSense D435i/D455 camera.
# The SpectacularAI SDK runs Visual-Inertial SLAM during recording to
# compute accurate 6-DoF camera poses.
#
# Requirements:
#   - RealSense D435i or D455 connected via USB 3.0
#   - SpectacularAI SDK installed (pip install spectacularAI[full])
#
# Usage:
#   ./scripts/realsense/record.sh [OPTIONS]
#
# Options:
#   --output DIR        Output directory (default: ./data/realsense_<timestamp>)
#   --no-preview        Disable live preview (lower CPU usage)
#   --recording-only    Only record, skip live VIO processing
#   -h, --help          Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
OUTPUT=""
PREVIEW=true
RECORDING_ONLY=false

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)     OUTPUT="$2"; shift 2 ;;
        --no-preview) PREVIEW=false; shift ;;
        --recording-only) RECORDING_ONLY=true; shift ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# Generate default output path with timestamp
if [[ -z "$OUTPUT" ]]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT="${PROJECT_DIR}/data/realsense_${TIMESTAMP}"
fi

# Ensure output parent directory exists
mkdir -p "$(dirname "$OUTPUT")"

echo "============================================"
echo "  RealSense Recording (SpectacularAI)"
echo "============================================"
echo "  Output: ${OUTPUT}"
echo "  Preview: ${PREVIEW}"
echo "  Recording only: ${RECORDING_ONLY}"
echo "============================================"

# Check that sai-record-realsense is available
if ! command -v sai-record-realsense &> /dev/null; then
    echo "[ERROR] sai-record-realsense not found."
    echo "  Install with: pip install spectacularAI[full]"
    exit 1
fi

# Build command
CMD=(sai-record-realsense --output "$OUTPUT")

if [[ "$PREVIEW" == false ]]; then
    CMD+=(--no_preview)
fi

if [[ "$RECORDING_ONLY" == true ]]; then
    CMD+=(--recording_only)
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
echo "  ./scripts/realsense/process.sh ${OUTPUT} ./datasets/custom/my_scene"
