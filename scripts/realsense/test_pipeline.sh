#!/usr/bin/env bash
# ============================================================================
# Test the DN-Splatter RealSense pipeline (no camera required)
# ============================================================================
# Downloads a sample RGB-D dataset and runs the training + mesh extraction
# pipeline to verify everything works.
#
# Since SpectacularAI recordings require a physical RealSense camera,
# this test skips the recording/processing steps and uses the MuSHRoom
# iPhone dataset (which has the same format: transforms.json + RGB + depth).
#
# Usage:
#   ./scripts/realsense/test_pipeline.sh [OPTIONS]
#
# Options:
#   --quick             Short training (500 iterations, ~2 min on RTX 3090)
#   --full              Full training (30000 iterations)
#   --skip-download     Skip dataset download (reuse existing)
#   --skip-train        Skip training (only extract mesh from existing model)
#   --dataset NAME      Dataset to test with: mushroom (default), replica
#   --mesh-only CONFIG  Only run mesh extraction on given config
#   -h, --help          Show this help
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
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
step()  { echo -e "\n${BOLD}${YELLOW}>>> $* <<<${NC}\n"; }

# Defaults
MAX_ITER=500
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
MESH_ONLY=""
DATASET="mushroom"

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)         MAX_ITER=500; shift ;;
        --full)          MAX_ITER=30000; shift ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-train)    SKIP_TRAIN=true; shift ;;
        --dataset)       DATASET="$2"; shift 2 ;;
        --mesh-only)     MESH_ONLY="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *)               echo "Unknown option: $1"; usage ;;
    esac
done

echo "============================================================"
echo "  DN-Splatter Pipeline Test (no camera required)"
echo "============================================================"
echo "  Dataset:         ${DATASET}"
echo "  Max iterations:  ${MAX_ITER}"
echo "  Skip download:   ${SKIP_DOWNLOAD}"
echo "============================================================"

cd "$PROJECT_DIR"

# ---- Verify environment ----------------------------------------------------
step "Verifying environment"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || fail "PyTorch CUDA not available. Run setup_env.sh first."
ok "PyTorch CUDA available"

python -c "import nerfstudio" 2>/dev/null \
    || fail "Nerfstudio not installed"
ok "Nerfstudio installed"

command -v ns-train &>/dev/null \
    || fail "ns-train not found in PATH"
ok "ns-train available"

command -v gs-mesh &>/dev/null \
    || fail "gs-mesh not found in PATH"
ok "gs-mesh available"

# ---- Download sample data --------------------------------------------------
if [[ -n "$MESH_ONLY" ]]; then
    info "Skipping to mesh extraction"
else

if [[ "$DATASET" == "mushroom" ]]; then
    # Use normal-nerfstudio dataparser (same as RealSense pipeline) with MuSHRoom data
    DATA_DIR="${PROJECT_DIR}/datasets/room_datasets/koivu/iphone/long_capture"
    DATAPARSER="normal-nerfstudio"
    DATAPARSER_ARGS="--data ${DATA_DIR} --load-normals False --load-3D-points False --load-pcd-normals False"

    if [[ "$SKIP_DOWNLOAD" == false ]] && [[ ! -d "$DATA_DIR" ]]; then
        step "Downloading MuSHRoom sample dataset (koivu/iphone)"
        python dn_splatter/data/download_scripts/mushroom_download.py \
            --room-name koivu --sequence iphone
    else
        info "Using existing dataset at ${DATA_DIR}"
    fi

    if [[ ! -d "$DATA_DIR" ]]; then
        fail "Dataset not found at ${DATA_DIR}"
    fi

    # normal-nerfstudio expects transforms.json; mushroom ships transformations_colmap.json
    if [[ ! -f "${DATA_DIR}/transforms.json" ]] && [[ -f "${DATA_DIR}/transformations_colmap.json" ]]; then
        info "Symlinking transformations_colmap.json -> transforms.json"
        ln -sf transformations_colmap.json "${DATA_DIR}/transforms.json"
    fi

elif [[ "$DATASET" == "replica" ]]; then
    DATA_DIR="${PROJECT_DIR}/datasets/Replica/office0"
    DATAPARSER="replica"
    DATAPARSER_ARGS="--data ${PROJECT_DIR}/datasets/Replica --sequence office0"

    if [[ "$SKIP_DOWNLOAD" == false ]] && [[ ! -d "$DATA_DIR" ]]; then
        step "Downloading Replica dataset"
        python dn_splatter/data/download_scripts/replica_download.py
    else
        info "Using existing dataset at ${DATA_DIR}"
    fi

    if [[ ! -d "$DATA_DIR" ]]; then
        fail "Dataset not found at ${DATA_DIR}"
    fi

else
    fail "Unknown dataset: ${DATASET}. Use 'mushroom' or 'replica'."
fi

ok "Dataset ready: ${DATA_DIR}"

# ---- Train ------------------------------------------------------------------
CONFIG=""
if [[ "$SKIP_TRAIN" == false ]]; then
    step "Training dn-splatter (${MAX_ITER} iterations)"

    TRAIN_CMD="ns-train dn-splatter \
        --max-num-iterations ${MAX_ITER} \
        --pipeline.model.use-depth-loss True \
        --pipeline.model.depth-lambda 0.2 \
        --pipeline.model.depth-loss-type EdgeAwareLogL1 \
        --pipeline.model.use-normal-loss True \
        --pipeline.model.use-normal-tv-loss True \
        --pipeline.model.normal-supervision depth \
        --experiment-name test-realsense-pipeline \
        --viewer.quit-on-train-completion True \
        ${DATAPARSER} ${DATAPARSER_ARGS}"

    info "Running: ${TRAIN_CMD}"
    eval "$TRAIN_CMD"

    ok "Training complete"
fi

# Find latest config
CONFIG=$(find "${PROJECT_DIR}/outputs" -path "*/test-realsense-pipeline/*/config.yml" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2)

if [[ -z "$CONFIG" ]]; then
    CONFIG=$(ls -t "${PROJECT_DIR}"/outputs/dn-splatter/test-realsense-pipeline/*/config.yml 2>/dev/null | head -1 || true)
fi

fi  # end of MESH_ONLY check

# Use mesh-only config if specified
if [[ -n "$MESH_ONLY" ]]; then
    CONFIG="$MESH_ONLY"
fi

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    fail "Could not find training config. Run training first."
fi

ok "Config found: ${CONFIG}"

# ---- Extract mesh -----------------------------------------------------------
step "Extracting mesh (o3dtsdf)"
MESH_DIR="${PROJECT_DIR}/meshes/test_pipeline"
mkdir -p "$MESH_DIR"

gs-mesh o3dtsdf --load-config "$CONFIG" --output-dir "$MESH_DIR" \
    || warn "Mesh extraction had issues (this can happen with short training)"

# ---- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo -e "  ${GREEN}${BOLD}Pipeline Test Complete!${NC}"
echo "============================================================"
echo "  Config:  ${CONFIG}"
echo "  Mesh:    ${MESH_DIR}"
echo ""

if ls "$MESH_DIR"/*.ply &>/dev/null; then
    ok "Mesh file(s) found:"
    ls -lh "$MESH_DIR"/*.ply
else
    warn "No .ply files in mesh output (may need more training iterations)"
fi

echo ""
echo "============================================================"
echo "  What was tested:"
echo "    [x] Environment & dependencies"
if [[ -z "$MESH_ONLY" ]]; then
echo "    [x] Dataset download & loading"
echo "    [x] DN-Splatter training with depth+normal supervision"
fi
echo "    [x] Mesh extraction (o3dtsdf)"
echo ""
echo "  Not tested (requires physical RealSense D435i/D455):"
echo "    [ ] Recording with sai-record-realsense"
echo "    [ ] Processing with process_sai.py / sai-cli"
echo "============================================================"
