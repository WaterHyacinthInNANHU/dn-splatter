#!/usr/bin/env bash
# ============================================================================
# DN-Splatter + RealSense: One-Step Setup with uv
# ============================================================================
# Sets up a complete environment using uv (fast Python package manager).
# uv handles the virtual environment and all pip installs. You manage
# CUDA drivers and ffmpeg system-level.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support + drivers installed
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - ffmpeg (sudo apt install ffmpeg)
#
# Usage:
#   chmod +x scripts/realsense/setup_env_uv.sh
#   ./scripts/realsense/setup_env_uv.sh
#
# After setup, activate with:
#   source .venv/bin/activate
# ============================================================================

set -euo pipefail

# ---- Configuration --------------------------------------------------------
PYTHON_VERSION="3.10"
DN_SPLATTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${DN_SPLATTER_DIR}/.venv"

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

# ---- Pre-flight checks ----------------------------------------------------
info "Running pre-flight checks..."

# Check uv
if ! command -v uv &> /dev/null; then
    error "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
ok "uv found: $(uv --version)"

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found. Install NVIDIA drivers first."
fi
nvidia-smi &> /dev/null || error "nvidia-smi failed. Check GPU drivers."
ok "NVIDIA GPU detected"

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    warn "ffmpeg not found. Install with: sudo apt install ffmpeg"
    warn "ffmpeg is required for SpectacularAI recording/processing."
    read -r -p "Continue without ffmpeg? [y/N] " response
    [[ "$response" =~ ^[Yy]$ ]] || exit 1
else
    ok "ffmpeg found"
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

# ---- Install PyTorch with CUDA ---------------------------------------------
info "Installing PyTorch with CUDA 11.8..."
uv pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# ---- Install nerfstudio ---------------------------------------------------
info "Installing nerfstudio 1.1.3..."
uv pip install nerfstudio==1.1.3

# ---- Pre-install PyMCubes (version compat fix) -----------------------------
# PyMCubes 0.1.2 (pinned in pyproject.toml) fails to build with NumPy 2.x.
# Install 0.1.6 (numpy 2.x compatible) first; pip will reuse it.
info "Pre-installing PyMCubes 0.1.6 (numpy 2.x compatible)..."
uv pip install "PyMCubes==0.1.6"

# ---- Install DN-Splatter --------------------------------------------------
info "Installing DN-Splatter..."
uv pip install setuptools==69.5.1
uv pip install -e . --override <(echo "PyMCubes>=0.1.2")

# ---- Install SpectacularAI + RealSense ------------------------------------
info "Installing SpectacularAI SDK and RealSense support..."
uv pip install "spectacularAI[full]"
uv pip install pyrealsense2

# ---- Verify ----------------------------------------------------------------
info "Verifying installation..."

echo ""
echo "-------- Verification --------"

python_ver=$(python --version 2>&1)
echo "  Python:          ${python_ver}"

torch_info=$(python -c "
import torch
cuda_ok = torch.cuda.is_available()
print(f'{torch.__version__} (CUDA {torch.version.cuda}, available={cuda_ok})')
" 2>&1)
echo "  PyTorch:         ${torch_info}"

ns_ver=$(python -c "import nerfstudio; print(nerfstudio.__version__)" 2>&1 || echo "NOT FOUND")
echo "  Nerfstudio:      ${ns_ver}"

sai_ok=$(python -c "import spectacularAI; print('installed')" 2>&1 || echo "NOT FOUND")
echo "  SpectacularAI:   ${sai_ok}"

rs_ok=$(python -c "import pyrealsense2; print('installed')" 2>&1 || echo "NOT FOUND")
echo "  pyrealsense2:    ${rs_ok}"

gs_mesh_ok=$(command -v gs-mesh &>/dev/null && echo "found" || echo "NOT in PATH")
echo "  gs-mesh:         ${gs_mesh_ok}"

echo "------------------------------"
echo ""

ok "Setup complete with uv!"
echo ""
info "To activate this environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
info "Or use uv run to execute commands directly:"
echo "  uv run ns-train --help"
echo "  uv run gs-mesh --help"
echo ""
info "Quick start:"
echo "  source .venv/bin/activate"
echo "  ./scripts/realsense/run_pipeline.sh --scene my_room"
