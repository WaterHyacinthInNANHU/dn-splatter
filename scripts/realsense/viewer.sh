#!/usr/bin/env bash
# ============================================================================
# Launch Nerfstudio Viewer for a trained DN-Splatter model
# ============================================================================
# Opens an interactive 3D viewer in the browser to visualize the trained
# Gaussian Splatting model. Renders RGB, depth, and normals.
#
# Usage:
#   ./scripts/realsense/viewer.sh CONFIG_PATH [OPTIONS]
#
# Arguments:
#   CONFIG_PATH         Path to the training config.yml
#                       If omitted, uses the most recent training run.
#
# Options:
#   --host HOST         Bind address (default: 0.0.0.0)
#   --port PORT         Viewer port (default: 7007)
#   --gpu GPU_ID        GPU to use (default: 0)
#   -h, --help          Show this help
#
# After launch, open http://<host>:<port> in your browser.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
CONFIG_PATH=""
HOST="0.0.0.0"
PORT="7007"
GPU_ID="0"

usage() {
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)  HOST="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --gpu)   GPU_ID="$2"; shift 2 ;;
        -h|--help) usage ;;
        -*)      echo "Unknown option: $1"; usage ;;
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

# Auto-find latest config if not specified
if [[ -z "$CONFIG_PATH" ]]; then
    CONFIG_PATH=$(find "${PROJECT_DIR}/outputs" -name "config.yml" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
    if [[ -z "$CONFIG_PATH" ]]; then
        echo "[ERROR] No config.yml found in outputs/."
        echo "  Train a model first: ./scripts/realsense/train.sh <DATA_DIR>"
        exit 1
    fi
    echo "[INFO] Auto-selected latest config: ${CONFIG_PATH}"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}"
    exit 1
fi

echo "============================================"
echo "  Nerfstudio Viewer"
echo "============================================"
echo "  Config:  ${CONFIG_PATH}"
echo "  GPU:     ${GPU_ID}"
echo "  Host:    ${HOST}"
echo "  Port:    ${PORT}"
echo "============================================"
echo ""
echo "  Open in browser: http://${HOST}:${PORT}"
echo ""

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Disable torch.compile — its inductor backend conflicts with nerfstudio's
# viewer interrupt handling (IOChangeException during dynamo tracing)
export TORCHDYNAMO_DISABLE=1

ns-viewer \
    --load-config "${CONFIG_PATH}" \
    --viewer.websocket-host "${HOST}" \
    --viewer.websocket-port "${PORT}"
