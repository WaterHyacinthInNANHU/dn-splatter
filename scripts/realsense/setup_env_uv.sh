#!/usr/bin/env bash
# ============================================================================
# DN-Splatter + RealSense: One-Step Setup with uv
# ============================================================================
# Sets up a complete environment using uv (fast Python package manager).
# uv handles the virtual environment and all pip installs. You manage
# CUDA drivers and ffmpeg system-level.
#
# Modes:
#   full     Install everything — recording + processing + training (default)
#   collect  Data collection only — pyrealsense2, SpectacularAI, ffmpeg
#            (lightweight, no GPU required)
#   train    Processing + training + mesh — PyTorch, nerfstudio, DN-Splatter,
#            COLMAP, pyrealsense2 (GPU required)
#
# Prerequisites:
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - NVIDIA GPU + drivers (for train/full modes)
#   - ffmpeg (for collect/full modes): sudo apt install ffmpeg
#
# Usage:
#   ./scripts/realsense/setup_env_uv.sh [--mode full|collect|train]
#
# After setup, activate with:
#   source .venv/bin/activate
# ============================================================================

set -euo pipefail

# ---- Configuration --------------------------------------------------------
PYTHON_VERSION="3.10"
DN_SPLATTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${DN_SPLATTER_DIR}/.venv"
MODE="full"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check uv
if ! command -v uv &> /dev/null; then
    error "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
ok "uv found: $(uv --version)"

# Check NVIDIA GPU (train/full only)
if [[ "$MODE" != "collect" ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. Install NVIDIA drivers first."
    fi
    nvidia-smi &> /dev/null || error "nvidia-smi failed. Check GPU drivers."
    ok "NVIDIA GPU detected"
fi

# Check ffmpeg (collect/full only)
if [[ "$MODE" != "train" ]]; then
    if ! command -v ffmpeg &> /dev/null; then
        warn "ffmpeg not found. Install with: sudo apt install ffmpeg"
        warn "ffmpeg is required for SpectacularAI recording/processing."
        read -r -p "Continue without ffmpeg? [y/N] " response
        [[ "$response" =~ ^[Yy]$ ]] || exit 1
    else
        ok "ffmpeg found"
    fi
fi

# Install COLMAP (train/full only)
if [[ "$MODE" != "collect" ]]; then
    if ! command -v colmap &> /dev/null; then
        info "COLMAP not found. Installing via conda..."
        if command -v conda &> /dev/null; then
            conda install -c conda-forge colmap -y
            ok "COLMAP installed via conda"
        else
            warn "conda not found, cannot auto-install COLMAP."
            warn "Install manually: sudo apt install colmap"
            warn "  OR install conda first, then re-run this script."
        fi
    else
        ok "COLMAP found"
    fi
fi

# ---- Create virtual environment --------------------------------------------
info "Creating virtual environment with Python ${PYTHON_VERSION}..."

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at ${VENV_DIR}"
    read -r -p "Recreate it? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        info "Reusing existing venv"
    fi
fi

cd "$DN_SPLATTER_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    uv venv --python "${PYTHON_VERSION}" "$VENV_DIR"
fi

# Activate for subsequent commands
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

ok "Virtual environment ready at ${VENV_DIR}"

# ---- Install PyTorch with CUDA (train/full) --------------------------------
if [[ "$MODE" != "collect" ]]; then
    info "Installing PyTorch with CUDA 11.8..."
    uv pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118
fi

# ---- Install nerfstudio (train/full) --------------------------------------
if [[ "$MODE" != "collect" ]]; then
    info "Installing nerfstudio 1.1.3..."
    uv pip install nerfstudio==1.1.3
fi

# ---- Install DN-Splatter (train/full) -------------------------------------
if [[ "$MODE" != "collect" ]]; then
    # PyMCubes 0.1.2 (pinned in pyproject.toml) fails to build with NumPy 2.x.
    # Install 0.1.6 (numpy 2.x compatible) first; pip will reuse it.
    info "Pre-installing PyMCubes 0.1.6 (numpy 2.x compatible)..."
    uv pip install "PyMCubes==0.1.6"

    info "Installing DN-Splatter..."
    uv pip install setuptools==69.5.1
    uv pip install -e . --override <(echo "PyMCubes>=0.1.2")
fi

# ---- Install SpectacularAI (collect/full) ----------------------------------
if [[ "$MODE" != "train" ]]; then
    info "Installing SpectacularAI SDK and RealSense support..."
    uv pip install "spectacularAI[full]"
fi

# ---- Install pyrealsense2 (all modes) -------------------------------------
info "Installing pyrealsense2..."
uv pip install pyrealsense2

# ---- Verify ----------------------------------------------------------------
info "Verifying installation..."

echo ""
echo "-------- Verification (mode: ${MODE}) --------"

python_ver=$(python --version 2>&1)
echo "  Python:          ${python_ver}"

if [[ "$MODE" != "collect" ]]; then
    torch_info=$(python -c "
import torch
cuda_ok = torch.cuda.is_available()
print(f'{torch.__version__} (CUDA {torch.version.cuda}, available={cuda_ok})')
" 2>&1)
    echo "  PyTorch:         ${torch_info}"

    ns_ver=$(python -c "import nerfstudio; print(nerfstudio.__version__)" 2>&1 || echo "NOT FOUND")
    echo "  Nerfstudio:      ${ns_ver}"

    colmap_ok=$(command -v colmap &>/dev/null && echo "found" || echo "NOT in PATH")
    echo "  COLMAP:          ${colmap_ok}"

    gs_mesh_ok=$(command -v gs-mesh &>/dev/null && echo "found" || echo "NOT in PATH")
    echo "  gs-mesh:         ${gs_mesh_ok}"
fi

if [[ "$MODE" != "train" ]]; then
    sai_ok=$(python -c "import spectacularAI; print('installed')" 2>&1 || echo "NOT FOUND")
    echo "  SpectacularAI:   ${sai_ok}"
fi

rs_ok=$(python -c "import pyrealsense2; print('installed')" 2>&1 || echo "NOT FOUND")
echo "  pyrealsense2:    ${rs_ok}"

echo "------------------------------"
echo ""

ok "Setup complete with uv! (mode: ${MODE})"
echo ""
info "To activate this environment:"
echo "  source ${VENV_DIR}/bin/activate"

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
    info "Or use uv run to execute commands directly:"
    echo "  uv run ns-train --help"
    echo "  uv run gs-mesh --help"
    echo ""
    info "Quick start (D435i/D455 with IMU):"
    echo "  source .venv/bin/activate"
    echo "  ./scripts/realsense/run_pipeline.sh --scene my_room"
    echo ""
    info "Quick start (D415, no IMU — uses COLMAP for poses):"
    echo "  source .venv/bin/activate"
    echo "  ./scripts/realsense/run_pipeline.sh --camera d415 --scene my_desk"
fi
