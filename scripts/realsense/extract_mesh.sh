#!/usr/bin/env bash
# ============================================================================
# Extract mesh from a trained DN-Splatter model
# ============================================================================
# Runs mesh extraction on a trained DN-Splatter/AGS-Mesh checkpoint.
#
# Usage:
#   ./scripts/realsense/extract_mesh.sh CONFIG_PATH [OPTIONS]
#
# Arguments:
#   CONFIG_PATH         Path to the training config.yml
#
# Options:
#   --method METHOD     Mesh extraction method (default: o3dtsdf)
#                       Options: dn, tsdf, o3dtsdf, sugar-coarse, gaussians, marching
#   --output-dir DIR    Output directory (default: meshes/<timestamp>)
#   --voxel-size F      Voxel size for TSDF (default: auto)
#   --extra ARGS        Extra arguments passed to gs-mesh
#   -h, --help          Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
CONFIG_PATH=""
MESH_METHOD="o3dtsdf"
OUTPUT_DIR=""
VOXEL_SIZE=""
EXTRA_ARGS=""

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)      MESH_METHOD="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --voxel-size)  VOXEL_SIZE="$2"; shift 2 ;;
        --extra)       EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)     usage ;;
        -*)            echo "Unknown option: $1"; usage ;;
        *)
            if [[ -z "$CONFIG_PATH" ]]; then
                CONFIG_PATH="$1"
            else
                echo "Unexpected argument: $1"; usage
            fi
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config path is required."
    echo ""
    echo "Find your config with:"
    echo "  ls -t outputs/*/nerfstudio_models/../config.yml 2>/dev/null || ls -t outputs/*/*/*/config.yml"
    usage
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Generate default output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="${PROJECT_DIR}/meshes/${MESH_METHOD}_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Mesh Extraction"
echo "============================================"
echo "  Config:  ${CONFIG_PATH}"
echo "  Method:  ${MESH_METHOD}"
echo "  Output:  ${OUTPUT_DIR}"
echo "============================================"

# Build command
CMD=(gs-mesh "${MESH_METHOD}"
    --load-config "${CONFIG_PATH}"
    --output-dir "${OUTPUT_DIR}")

if [[ -n "$VOXEL_SIZE" ]]; then
    CMD+=(--voxel-size "${VOXEL_SIZE}")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    CMD+=($EXTRA_ARGS)
fi

echo ""
echo "[CMD] ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "[OK] Mesh extracted to: ${OUTPUT_DIR}"
echo ""
echo "Output files:"
ls -la "${OUTPUT_DIR}"/*.ply 2>/dev/null || ls -la "${OUTPUT_DIR}"/ 2>/dev/null
