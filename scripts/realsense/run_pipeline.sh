#!/usr/bin/env bash
# ============================================================================
# DN-Splatter + RealSense: Full Pipeline (Record -> Process -> Train -> Mesh)
# ============================================================================
# End-to-end pipeline that records with a RealSense camera, processes data,
# trains a DN-Splatter model, and extracts a mesh.
#
# Usage:
#   ./scripts/realsense/run_pipeline.sh [OPTIONS]
#
# Options:
#   --scene NAME        Scene name for output organization (default: scene_<timestamp>)
#   --recording DIR     Path to an existing recording (skips recording step)
#   --dataset DIR       Path to already-processed dataset (skips recording + processing)
#   --config PATH       Path to trained config.yml (skips to mesh extraction)
#   --method METHOD     Training method: dn-splatter (default), ags-mesh
#   --mesh-method M     Mesh extraction: o3dtsdf (default), tsdf, dn, gaussians
#   --key-frame-dist M  Keyframe distance in meters (default: 0.1)
#   --max-iter N        Max training iterations (default: 30000)
#   --skip-mesh         Skip mesh extraction after training
#   -h, --help          Show this help
#
# Examples:
#   # Full pipeline from scratch
#   ./scripts/realsense/run_pipeline.sh --scene living_room
#
#   # Start from existing recording
#   ./scripts/realsense/run_pipeline.sh --recording ./data/my_recording --scene table
#
#   # Start from processed dataset
#   ./scripts/realsense/run_pipeline.sh --dataset ./datasets/custom/my_scene
#
#   # Only extract mesh from trained model
#   ./scripts/realsense/run_pipeline.sh --config outputs/dn-splatter/.../config.yml
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
step()  { echo -e "\n${BOLD}${YELLOW}>>> $* <<<${NC}\n"; }

# Defaults
SCENE=""
RECORDING=""
DATASET=""
CONFIG=""
METHOD="dn-splatter"
MESH_METHOD="o3dtsdf"
KEY_FRAME_DIST="0.1"
MAX_ITER="30000"
SKIP_MESH=false

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)           SCENE="$2"; shift 2 ;;
        --recording)       RECORDING="$2"; shift 2 ;;
        --dataset)         DATASET="$2"; shift 2 ;;
        --config)          CONFIG="$2"; shift 2 ;;
        --method)          METHOD="$2"; shift 2 ;;
        --mesh-method)     MESH_METHOD="$2"; shift 2 ;;
        --key-frame-dist)  KEY_FRAME_DIST="$2"; shift 2 ;;
        --max-iter)        MAX_ITER="$2"; shift 2 ;;
        --skip-mesh)       SKIP_MESH=true; shift ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown option: $1"; usage ;;
    esac
done

# Generate scene name if not provided
if [[ -z "$SCENE" ]]; then
    SCENE="scene_$(date +"%Y%m%d_%H%M%S")"
fi

RECORDING_DIR="${PROJECT_DIR}/data/${SCENE}"
DATASET_DIR="${PROJECT_DIR}/datasets/custom/${SCENE}"
MESH_DIR="${PROJECT_DIR}/meshes/${SCENE}"

echo "============================================================"
echo "  DN-Splatter + RealSense: Full Pipeline"
echo "============================================================"
echo "  Scene:           ${SCENE}"
echo "  Method:          ${METHOD}"
echo "  Mesh method:     ${MESH_METHOD}"
echo "  Keyframe dist:   ${KEY_FRAME_DIST}m"
echo "  Max iterations:  ${MAX_ITER}"
echo "============================================================"

START_TIME=$SECONDS

# ---- Step 1: Record -------------------------------------------------------
if [[ -z "$CONFIG" && -z "$DATASET" && -z "$RECORDING" ]]; then
    step "Step 1/4: Recording with RealSense"
    info "Output: ${RECORDING_DIR}"
    info "Press Ctrl+C to stop recording when done."
    echo ""

    "${SCRIPT_DIR}/record.sh" --output "${RECORDING_DIR}"

    RECORDING="${RECORDING_DIR}"
    ok "Recording complete: ${RECORDING}"
else
    if [[ -n "$RECORDING" ]]; then
        info "Skipping recording (using existing: ${RECORDING})"
    elif [[ -n "$DATASET" ]]; then
        info "Skipping recording + processing (using existing dataset: ${DATASET})"
    else
        info "Skipping to mesh extraction (using config: ${CONFIG})"
    fi
fi

# ---- Step 2: Process ------------------------------------------------------
if [[ -z "$CONFIG" && -z "$DATASET" ]]; then
    step "Step 2/4: Processing recording"
    info "Input: ${RECORDING}"
    info "Output: ${DATASET_DIR}"

    "${SCRIPT_DIR}/process.sh" "${RECORDING}" "${DATASET_DIR}" \
        --key-frame-dist "${KEY_FRAME_DIST}"

    DATASET="${DATASET_DIR}"
    ok "Processing complete: ${DATASET}"
fi

# ---- Step 3: Train --------------------------------------------------------
if [[ -z "$CONFIG" ]]; then
    step "Step 3/4: Training ${METHOD}"
    info "Data: ${DATASET}"

    "${SCRIPT_DIR}/train.sh" "${DATASET}" \
        --method "${METHOD}" \
        --max-iter "${MAX_ITER}" \
        --experiment "${SCENE}"

    # Find the latest config
    CONFIG=$(find "${PROJECT_DIR}/outputs" -name "config.yml" -newer "${DATASET}/transforms.json" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

    if [[ -z "$CONFIG" ]]; then
        echo "[WARN] Could not auto-detect config path. Searching outputs/..."
        CONFIG=$(ls -t "${PROJECT_DIR}"/outputs/"${METHOD}"/*/"${SCENE}"/*/config.yml 2>/dev/null | head -1 || true)
    fi

    if [[ -z "$CONFIG" ]]; then
        echo "[ERROR] Could not find training config. Check outputs/ directory."
        exit 1
    fi

    ok "Training complete. Config: ${CONFIG}"
fi

# ---- Step 4: Extract Mesh -------------------------------------------------
if [[ "$SKIP_MESH" == false ]]; then
    step "Step 4/4: Extracting mesh (${MESH_METHOD})"
    info "Config: ${CONFIG}"
    info "Output: ${MESH_DIR}"

    "${SCRIPT_DIR}/extract_mesh.sh" "${CONFIG}" \
        --method "${MESH_METHOD}" \
        --output-dir "${MESH_DIR}"

    ok "Mesh extraction complete: ${MESH_DIR}"
else
    info "Skipping mesh extraction (--skip-mesh)"
fi

# ---- Summary ---------------------------------------------------------------
ELAPSED=$(( SECONDS - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo -e "  ${GREEN}${BOLD}Pipeline Complete!${NC}"
echo "============================================================"
echo "  Scene:      ${SCENE}"
echo "  Recording:  ${RECORDING:-skipped}"
echo "  Dataset:    ${DATASET:-skipped}"
echo "  Config:     ${CONFIG}"
if [[ "$SKIP_MESH" == false ]]; then
echo "  Mesh:       ${MESH_DIR}"
fi
echo "  Duration:   ${MINUTES}m ${SECS}s"
echo "============================================================"
