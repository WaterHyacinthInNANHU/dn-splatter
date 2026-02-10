#!/usr/bin/env bash
# ============================================================================
# Visualize camera poses + point clouds from a processed dataset
# ============================================================================
# Quick sanity check after preprocessing and before training.
# Shows COLMAP-estimated camera frustums and sparse/dense point clouds
# in an interactive web-based viser viewer (accessible via browser).
#
# Usage:
#   ./scripts/realsense/visualize.sh <DATA_DIR> [OPTIONS]
#
# Arguments:
#   DATA_DIR            Path to processed dataset (must have transforms.json)
#
# Options:
#   --dense             Also backproject depth frames into dense point clouds
#   --every-n N         Backproject every Nth frame (default: 10)
#   --max-depth M       Max depth in meters (default: 10.0)
#   --frustum-scale S   Camera frustum size (default: 0.15)
#   --no-sparse         Don't load sparse_pc.ply
#   --host HOST         Bind address (default: 0.0.0.0)
#   --port PORT         Viewer port (default: 8890)
#   -h, --help          Show this help
#
# Then open http://<host>:<port> in your browser.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    sed -n '/^# Usage:/,/^# ====/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
fi

cd "${PROJECT_DIR}"
python dn_splatter/scripts/visualize_dataset.py "$@"
