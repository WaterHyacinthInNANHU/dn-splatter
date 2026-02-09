#!/usr/bin/env bash
# ============================================================================
# DN-Splatter + RealSense: One-Step Environment Setup
# ============================================================================
# Sets up a complete environment for DN-Splatter with RealSense camera support.
# Uses conda for environment management.
#
# Modes:
#   full     Install everything — recording + processing + training (default)
#   collect  Data collection only — pyrealsense2, SpectacularAI, ffmpeg
#            (lightweight, no GPU required)
#   train    Processing + training + mesh — PyTorch, nerfstudio, DN-Splatter,
#            COLMAP, pyrealsense2 (GPU required)
#
# Prerequisites:
#   - conda (Miniconda or Anaconda)
#   - NVIDIA GPU with CUDA support (for train/full modes)
#   - Intel RealSense D415, D435i, or D455
#   - USB 3.0 port and cable
#
# Usage:
#   ./scripts/realsense/setup_env.sh [--mode full|collect|train]
# ============================================================================

set -euo pipefail

# ---- Configuration --------------------------------------------------------
ENV_NAME="${DN_SPLATTER_ENV:-dn-splatter-rs}"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"
TORCH_VERSION="2.1.2"
TORCHVISION_VERSION="0.16.2"
DN_SPLATTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="full"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

usage() {
    echo "Usage: $0 [--mode full|collect|train]"
    echo ""
    echo "Modes:"
    echo "  full     Install everything (default)"
    echo "  collect  Data collection only (no GPU required)"
    echo "  train    Processing + training + mesh (GPU required)"
    exit 0
}

# ---- Parse arguments ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            if [[ "$MODE" != "full" && "$MODE" != "collect" && "$MODE" != "train" ]]; then
                error "Invalid mode: $MODE (must be full, collect, or train)"
            fi
            shift 2
            ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

info "Install mode: ${MODE}"

# ---- Pre-flight checks ----------------------------------------------------
info "Running pre-flight checks..."

# Check conda
if ! command -v conda &> /dev/null; then
    error "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
fi

# Check NVIDIA GPU (train/full only)
if [[ "$MODE" != "collect" ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. Install NVIDIA drivers first."
    fi
    if ! nvidia-smi &> /dev/null; then
        error "nvidia-smi failed. Check your GPU drivers."
    fi
fi

ok "Pre-flight checks passed"

# ---- Detect conda shell integration ----------------------------------------
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- Create or reuse conda environment ------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Conda environment '${ENV_NAME}' already exists"
    read -r -p "Recreate it from scratch? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        info "Removing existing environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "${ENV_NAME}" -y
    else
        info "Reusing existing environment"
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

info "Activating environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# ---- Install system dependencies via conda --------------------------------
if [[ "$MODE" == "full" ]]; then
    info "Installing system-level dependencies via conda..."
    conda install -c conda-forge ffmpeg colmap -y
elif [[ "$MODE" == "collect" ]]; then
    info "Installing ffmpeg via conda..."
    conda install -c conda-forge ffmpeg -y
elif [[ "$MODE" == "train" ]]; then
    info "Installing COLMAP via conda..."
    conda install -c conda-forge colmap -y
fi

# ---- Install PyTorch with CUDA (train/full) --------------------------------
if [[ "$MODE" != "collect" ]]; then
    info "Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}..."
    pip install torch=="${TORCH_VERSION}+cu118" torchvision=="${TORCHVISION_VERSION}+cu118" \
        --extra-index-url https://download.pytorch.org/whl/cu118
fi

# ---- Install nerfstudio (train/full) --------------------------------------
if [[ "$MODE" != "collect" ]]; then
    info "Installing nerfstudio 1.1.3..."
    pip install nerfstudio==1.1.3
fi

# ---- Install DN-Splatter (train/full) -------------------------------------
if [[ "$MODE" != "collect" ]]; then
    # PyMCubes 0.1.2 (pinned in pyproject.toml) fails to build with NumPy 2.x.
    # Install 0.1.6 first (numpy 2.x compatible).
    info "Pre-installing PyMCubes 0.1.6 (numpy 2.x compatible)..."
    pip install "PyMCubes==0.1.6"

    info "Installing DN-Splatter from ${DN_SPLATTER_DIR}..."
    cd "${DN_SPLATTER_DIR}"
    pip install setuptools==69.5.1
    pip install -e .
fi

# ---- Install SpectacularAI SDK (collect/full) ------------------------------
if [[ "$MODE" != "train" ]]; then
    info "Installing SpectacularAI SDK..."
    pip install spectacularAI[full]
fi

# ---- Install pyrealsense2 (all modes) -------------------------------------
info "Installing pyrealsense2..."
pip install pyrealsense2

# ---- Install Linux udev rules for RealSense (optional) --------------------
if [[ "$(uname)" == "Linux" && "$MODE" != "train" ]]; then
    info "Checking RealSense udev rules..."
    if [[ ! -f /etc/udev/rules.d/99-realsense-libusb.rules ]]; then
        warn "RealSense udev rules not found."
        warn "If your camera is not detected, install librealsense2 udev rules:"
        warn "  sudo apt install librealsense2-utils"
        warn "  OR download from: https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules"
    else
        ok "RealSense udev rules found"
    fi
fi

# ---- Verify installation ---------------------------------------------------
info "Verifying installation..."

echo ""
echo "-------- Verification (mode: ${MODE}) --------"

# Python
python_ver=$(python --version 2>&1)
echo "  Python:          ${python_ver}"

if [[ "$MODE" != "collect" ]]; then
    # PyTorch + CUDA
    torch_ver=$(python -c "import torch; print(f'{torch.__version__} (CUDA {torch.version.cuda}, available={torch.cuda.is_available()})')" 2>&1)
    echo "  PyTorch:         ${torch_ver}"

    # Nerfstudio
    ns_ver=$(python -c "import nerfstudio; print(nerfstudio.__version__)" 2>&1 || echo "NOT FOUND")
    echo "  Nerfstudio:      ${ns_ver}"

    # DN-Splatter (check if method is registered)
    dn_check=$(python -c "
from nerfstudio.configs.method_configs import all_methods
print('registered' if 'dn_splatter' in str(all_methods) or 'dn-splatter' in str(all_methods) else 'NOT registered')
" 2>&1 || echo "check failed")
    echo "  DN-Splatter:     ${dn_check}"

    # COLMAP
    colmap_ok=$(command -v colmap 2>/dev/null && echo "found" || echo "NOT in PATH")
    echo "  COLMAP:          ${colmap_ok}"

    # gs-mesh
    gs_mesh=$(command -v gs-mesh 2>/dev/null && echo "found" || echo "NOT in PATH")
    echo "  gs-mesh:         ${gs_mesh}"
fi

if [[ "$MODE" != "train" ]]; then
    # SpectacularAI
    sai_ver=$(python -c "import spectacularAI; print('installed')" 2>&1 || echo "NOT FOUND")
    echo "  SpectacularAI:   ${sai_ver}"

    # sai-cli
    sai_cli=$(command -v sai-cli 2>/dev/null && echo "found" || echo "NOT in PATH")
    echo "  sai-cli:         ${sai_cli}"

    # sai-record-realsense
    sai_rec=$(command -v sai-record-realsense 2>/dev/null && echo "found" || echo "NOT in PATH")
    echo "  sai-record-rs:   ${sai_rec}"
fi

# pyrealsense2
rs_ver=$(python -c "import pyrealsense2 as rs; print(rs.__version__ if hasattr(rs, '__version__') else 'installed')" 2>&1 || echo "NOT FOUND")
echo "  pyrealsense2:    ${rs_ver}"

echo "------------------------------"
echo ""

ok "Environment setup complete! (mode: ${MODE})"
echo ""
info "To activate this environment in the future, run:"
echo "  conda activate ${ENV_NAME}"

if [[ "$MODE" == "collect" ]]; then
    echo ""
    info "Data collection quick start:"
    echo "  # D435i/D455 (with IMU)"
    echo "  ./scripts/realsense/record.sh --output ./data/my_scene"
    echo ""
    echo "  # D415 (no IMU)"
    echo "  ./scripts/realsense/record_d415.sh --output ./data/my_scene.bag"
    echo ""
    info "Transfer recordings to the training machine, then process + train there."
elif [[ "$MODE" == "train" ]]; then
    echo ""
    info "Processing + training quick start:"
    echo "  # Process a D415 .bag recording transferred from collection machine"
    echo "  ./scripts/realsense/process_d415.sh ./data/my_scene.bag ./datasets/custom/my_scene"
    echo ""
    echo "  # Train"
    echo "  ./scripts/realsense/train.sh ./datasets/custom/my_scene"
    echo ""
    echo "  # Or full pipeline from recording"
    echo "  ./scripts/realsense/run_pipeline.sh --camera d415 --recording ./data/my_scene.bag --scene my_scene"
else
    echo ""
    info "Quick start (D435i/D455 with IMU):"
    echo "  ./scripts/realsense/run_pipeline.sh --scene my_room"
    echo ""
    info "Quick start (D415, no IMU — uses COLMAP for poses):"
    echo "  ./scripts/realsense/run_pipeline.sh --camera d415 --scene my_desk"
fi
