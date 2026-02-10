#!/usr/bin/env bash
# ============================================================================
# Process D415 recording into DN-Splatter format
# ============================================================================
# Takes a RealSense D415 .bag recording (or directory of frames) and converts
# it to a nerfstudio-compatible dataset:
#   1. Extracts aligned RGB + depth frames from the .bag file
#   2. Runs COLMAP for camera pose estimation (no IMU available)
#   3. Generates transforms.json with sensor depth paths
#
# Usage:
#   ./scripts/realsense/process_d415.sh INPUT [OUTPUT_DIR] [OPTIONS]
#
# Arguments:
#   INPUT               Path to .bag file or directory with color/ and depth/
#   OUTPUT_DIR          Output directory (default: datasets/custom/<input_name>)
#
# Options:
#   --every-n N         Save every Nth frame from .bag (default: 15, ~2fps)
#   --matching-method M COLMAP matching: exhaustive (default), sequential, vocab_tree
#   --skip-extraction   Skip frame extraction (reuse already-extracted frames)
#   --skip-colmap       Skip COLMAP (reuse existing results)
#   --dry-run           Print commands without executing
#   -h, --help          Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
INPUT=""
OUTPUT=""
EVERY_N="15"
MATCHING_METHOD="exhaustive"
SKIP_EXTRACTION=false
SKIP_COLMAP=false
DRY_RUN=false

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --every-n)          EVERY_N="$2"; shift 2 ;;
        --matching-method)  MATCHING_METHOD="$2"; shift 2 ;;
        --skip-extraction)  SKIP_EXTRACTION=true; shift ;;
        --skip-colmap)      SKIP_COLMAP=true; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        -h|--help)          usage ;;
        -*)                 echo "Unknown option: $1"; usage ;;
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
    echo "[ERROR] Input path is required."
    usage
fi

echo "============================================"
echo "  Process D415 Recording"
echo "============================================"
echo "  Input:          ${INPUT}"
echo "  Output:         ${OUTPUT:-<auto>}"
echo "  Every N frames: ${EVERY_N}"
echo "  COLMAP method:  ${MATCHING_METHOD}"
echo "============================================"

# Build command - positional args (input, output_dir) must come before flags
CMD=(python "${PROJECT_DIR}/dn_splatter/scripts/process_d415.py"
     "$INPUT")

if [[ -n "$OUTPUT" ]]; then
    CMD+=("$OUTPUT")
fi

CMD+=(--every-n-frames "$EVERY_N"
      --matching-method "$MATCHING_METHOD")

if [[ "$SKIP_EXTRACTION" == true ]]; then
    CMD+=(--skip-extraction)
fi

if [[ "$SKIP_COLMAP" == true ]]; then
    CMD+=(--skip-colmap)
fi

if [[ "$DRY_RUN" == true ]]; then
    CMD+=(--dry-run)
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
    BASENAME="${BASENAME%.bag}"
    BASENAME="${BASENAME%.zip}"
    FINAL_OUTPUT="datasets/custom/${BASENAME}"
fi

echo ""
echo "[OK] Dataset processed to: ${FINAL_OUTPUT}"
echo ""
echo "Next steps:"
echo "  1. Visualize poses + point cloud (sanity check):"
echo "     ./scripts/realsense/visualize.sh ${FINAL_OUTPUT}"
echo "     ./scripts/realsense/visualize.sh ${FINAL_OUTPUT} --dense --max-depth 5.0"
echo ""
echo "  2. Train DN-Splatter:"
echo "     ./scripts/realsense/train.sh ${FINAL_OUTPUT}"
