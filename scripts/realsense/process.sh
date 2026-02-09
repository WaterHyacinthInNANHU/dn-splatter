#!/usr/bin/env bash
# ============================================================================
# Process RealSense recording into DN-Splatter format
# ============================================================================
# Takes a SpectacularAI recording and converts it to a nerfstudio-compatible
# dataset with transforms.json, color images, depth maps, and sparse point cloud.
#
# Uses the existing dn_splatter/scripts/process_sai.py wrapper which runs
# sai-cli process with parameters tuned for DN-Splatter.
#
# Usage:
#   ./scripts/realsense/process.sh INPUT_DIR [OUTPUT_DIR] [OPTIONS]
#
# Arguments:
#   INPUT_DIR           Path to SpectacularAI recording folder (or .zip)
#   OUTPUT_DIR          Output directory (default: datasets/custom/<input_name>)
#
# Options:
#   --preview           Show 3D preview during processing
#   --key-frame-dist M  Minimum keyframe distance in meters (default: 0.1)
#                       Use 0.05 for tabletop, 0.15 for room-scale
#   --dry-run           Print commands without executing
#   -h, --help          Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
INPUT=""
OUTPUT=""
PREVIEW=false
KEY_FRAME_DIST="0.1"
DRY_RUN=false

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preview)         PREVIEW=true; shift ;;
        --key-frame-dist)  KEY_FRAME_DIST="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        -h|--help)         usage ;;
        -*)                echo "Unknown option: $1"; usage ;;
        *)
            if [[ -z "$INPUT" ]]; then
                INPUT="$1"
            elif [[ -z "$OUTPUT" ]]; then
                OUTPUT="$1"
            else
                echo "Unexpected argument: $1"; usage
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "[ERROR] Input directory is required."
    usage
fi

echo "============================================"
echo "  Process RealSense Recording"
echo "============================================"
echo "  Input:          ${INPUT}"
echo "  Output:         ${OUTPUT:-<auto>}"
echo "  Keyframe dist:  ${KEY_FRAME_DIST}m"
echo "  Preview:        ${PREVIEW}"
echo "============================================"

# Build command
CMD=(python "${PROJECT_DIR}/dn_splatter/scripts/process_sai.py"
     "$INPUT"
     --key_frame_distance "$KEY_FRAME_DIST")

if [[ -n "$OUTPUT" ]]; then
    CMD+=("$OUTPUT")
fi

if [[ "$PREVIEW" == true ]]; then
    CMD+=(--preview)
fi

if [[ "$DRY_RUN" == true ]]; then
    CMD+=(--dry_run)
fi

echo ""
echo "[CMD] ${CMD[*]}"
echo ""

"${CMD[@]}"

# Determine actual output path for user message
if [[ -n "$OUTPUT" ]]; then
    FINAL_OUTPUT="$OUTPUT"
else
    BASENAME=$(basename "$INPUT")
    BASENAME="${BASENAME%.zip}"
    FINAL_OUTPUT="datasets/custom/${BASENAME}"
fi

echo ""
echo "[OK] Dataset processed to: ${FINAL_OUTPUT}"
echo ""
echo "Output structure:"
echo "  ${FINAL_OUTPUT}/"
echo "  ├── transforms.json      (camera poses + intrinsics)"
echo "  ├── sparse_pc.ply        (sparse point cloud)"
echo "  ├── images/"
echo "  │   ├── frame_00001.png  (color images)"
echo "  │   ├── depth_00001.png  (depth maps)"
echo "  │   └── ..."
echo "  └── colmap/sparse/0/     (COLMAP-compatible files)"
echo ""
echo "Next step: train DN-Splatter"
echo "  ./scripts/realsense/train.sh ${FINAL_OUTPUT}"
