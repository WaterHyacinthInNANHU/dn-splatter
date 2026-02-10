#!/usr/bin/env bash
# ============================================================================
# Train DN-Splatter on RealSense data
# ============================================================================
# Trains a DN-Splatter Gaussian Splatting model with depth and normal
# supervision using data captured from a RealSense camera.
#
# Usage:
#   ./scripts/realsense/train.sh DATA_DIR [OPTIONS]
#
# Arguments:
#   DATA_DIR            Path to processed dataset (with transforms.json)
#
# Options:
#   --method METHOD     Training method: dn-splatter (default), ags-mesh, dn-splatter-big
#   --depth-lambda F    Depth loss weight (default: 0.2)
#   --normal-sup MODE   Normal supervision: depth (default) or mono
#   --max-iter N        Max training iterations (default: 30000)
#   --experiment NAME   Experiment name for output organization
#   --extra ARGS        Extra arguments passed directly to ns-train
#   -h, --help          Show this help
# ============================================================================

set -euo pipefail

# Ensure CUDA toolkit is in PATH (needed by gsplat for JIT compilation)
if ! command -v nvcc &>/dev/null; then
    for cuda_dir in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11; do
        if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
            export PATH="${cuda_dir}/bin:${PATH}"
            export CUDA_HOME="${cuda_dir}"
            break
        fi
    done
fi

# Defaults
DATA_DIR=""
METHOD="dn-splatter"
DEPTH_LAMBDA="0.2"
NORMAL_SUP="depth"
MAX_ITER="30000"
EXPERIMENT=""
EXTRA_ARGS=""

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)       METHOD="$2"; shift 2 ;;
        --depth-lambda) DEPTH_LAMBDA="$2"; shift 2 ;;
        --normal-sup)   NORMAL_SUP="$2"; shift 2 ;;
        --max-iter)     MAX_ITER="$2"; shift 2 ;;
        --experiment)   EXPERIMENT="$2"; shift 2 ;;
        --extra)        EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)      usage ;;
        -*)             echo "Unknown option: $1"; usage ;;
        *)
            if [[ -z "$DATA_DIR" ]]; then
                DATA_DIR="$1"
            else
                echo "Unexpected argument: $1"; usage
            fi
            shift
            ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "[ERROR] Data directory is required."
    usage
fi

# Validate data directory
if [[ ! -f "${DATA_DIR}/transforms.json" ]]; then
    echo "[ERROR] transforms.json not found in ${DATA_DIR}"
    echo "  Make sure you have processed the recording first:"
    echo "  ./scripts/realsense/process.sh <recording_dir> ${DATA_DIR}"
    exit 1
fi

echo "============================================"
echo "  Train DN-Splatter (RealSense data)"
echo "============================================"
echo "  Data:            ${DATA_DIR}"
echo "  Method:          ${METHOD}"
echo "  Depth lambda:    ${DEPTH_LAMBDA}"
echo "  Normal sup:      ${NORMAL_SUP}"
echo "  Max iterations:  ${MAX_ITER}"
echo "============================================"

# Build training command
CMD=(ns-train "${METHOD}"
    --max-num-iterations "${MAX_ITER}"
    --pipeline.model.use-depth-loss True
    --pipeline.model.depth-lambda "${DEPTH_LAMBDA}"
    --pipeline.model.depth-loss-type EdgeAwareLogL1
    --pipeline.model.use-normal-loss True
    --pipeline.model.use-normal-tv-loss True
    --pipeline.model.normal-supervision "${NORMAL_SUP}")

# Add experiment name if provided
if [[ -n "$EXPERIMENT" ]]; then
    CMD+=(--experiment-name "${EXPERIMENT}")
fi

# For ags-mesh, also enable depth confidence masks
if [[ "$METHOD" == "ags-mesh" ]]; then
    CMD+=(normal-nerfstudio --data "${DATA_DIR}" --load-depth-confidence-masks True)
else
    CMD+=(normal-nerfstudio --data "${DATA_DIR}")
fi

# When using depth-based normals, skip loading pre-computed normal images from disk
if [[ "$NORMAL_SUP" == "depth" ]]; then
    CMD+=(--load-normals False)
fi

# Append any extra arguments
if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    CMD+=($EXTRA_ARGS)
fi

echo ""
echo "[CMD] ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "[OK] Training complete!"
echo ""
echo "The config file is saved in the outputs/ directory."
echo "Find it with: ls -t outputs/${METHOD}/*/config.yml | head -1"
echo ""
echo "Next step: extract mesh"
echo "  ./scripts/realsense/extract_mesh.sh \$(ls -t outputs/${METHOD}/*/config.yml | head -1)"
