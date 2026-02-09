# DN-Splatter + RealSense Camera Guide

End-to-end guide for 3D Gaussian Splatting and mesh reconstruction using an Intel RealSense depth camera with DN-Splatter. Uses [SpectacularAI](https://www.spectacularai.com/mapping) SDK for Visual-Inertial SLAM preprocessing.

## Table of Contents

- [Prerequisites](#prerequisites)
- [One-Step Installation](#one-step-installation)
- [Installation with uv (Alternative)](#installation-with-uv-alternative)
- [Manual Installation](#manual-installation)
- [Testing Without a Camera](#testing-without-a-camera)
- [Pipeline Overview](#pipeline-overview)
- [Step 1: Record Data](#step-1-record-data)
- [Step 2: Process Recording](#step-2-process-recording)
- [Step 3: Train DN-Splatter](#step-3-train-dn-splatter)
- [Step 4: Extract Mesh](#step-4-extract-mesh)
- [Full Pipeline (One Command)](#full-pipeline-one-command)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Prerequisites

### Hardware
| Requirement | Details |
|---|---|
| **GPU** | NVIDIA GPU with CUDA support (RTX 2070+ recommended, 8 GB+ VRAM) |
| **Camera** | Intel RealSense **D435i** or **D455** (the "i" suffix is required — it has an IMU) |
| **USB** | USB 3.0 cable connected to a USB 3.0 port (USB 2.0 will not work) |

> **Note:** The D435 (without "i") is **not supported** — SpectacularAI requires the onboard IMU for Visual-Inertial SLAM.

### Software
| Requirement | Details |
|---|---|
| **OS** | Ubuntu 20.04 / 22.04 / 24.04 (other Linux distros may work) |
| **Python env** | [conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) **or** [uv](https://docs.astral.sh/uv/) |
| **NVIDIA drivers** | `nvidia-smi` must work |

---

## One-Step Installation

The install script creates a conda environment with all dependencies:

```bash
git clone https://github.com/maturk/dn-splatter
cd dn-splatter
chmod +x scripts/realsense/setup_env.sh
./scripts/realsense/setup_env.sh
```

This installs:
- Python 3.10 in a `dn-splatter-rs` conda environment
- PyTorch 2.1.2 with CUDA 11.8
- Nerfstudio 1.1.3
- DN-Splatter (editable install)
- SpectacularAI SDK (with visualization tools)
- pyrealsense2
- ffmpeg

After installation, activate the environment:

```bash
conda activate dn-splatter-rs
```

To customize the environment name:

```bash
DN_SPLATTER_ENV=my-custom-name ./scripts/realsense/setup_env.sh
```

---

## Installation with uv (Alternative)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager from Astral (the Ruff team). It's ~10-100x faster than pip for dependency resolution and installs.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Make sure ffmpeg is installed system-wide
sudo apt install ffmpeg

# Run the uv setup script
git clone https://github.com/maturk/dn-splatter
cd dn-splatter
chmod +x scripts/realsense/setup_env_uv.sh
./scripts/realsense/setup_env_uv.sh
```

This creates a `.venv/` in the project directory. Activate with:

```bash
source .venv/bin/activate
```

> **conda vs uv:** conda manages CUDA toolkit and system libs (ffmpeg) inside the environment. uv is faster but relies on system-installed CUDA drivers and ffmpeg. If you already have NVIDIA drivers and ffmpeg on your system, uv is the simpler choice.

---

## Manual Installation

If you prefer step-by-step control:

```bash
# 1. Create conda environment
conda create -n dn-splatter-rs python=3.10 -y
conda activate dn-splatter-rs
conda install -c conda-forge ffmpeg -y

# 2. Install PyTorch with CUDA
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Install Nerfstudio
pip install nerfstudio==1.1.3

# 4. Install DN-Splatter
cd dn-splatter/
pip install setuptools==69.5.1
pip install -e .

# 5. Install SpectacularAI SDK + RealSense
pip install spectacularAI[full]
pip install pyrealsense2

# 6. (Linux) Install RealSense udev rules if camera not detected
sudo apt install librealsense2-utils
```

---

## Testing Without a Camera

You can verify the full training + mesh extraction pipeline without a physical RealSense camera. The test script downloads a sample RGB-D dataset (MuSHRoom) and runs through training and mesh extraction.

```bash
# Quick test (~2 min on RTX 3090, 500 iterations)
./scripts/realsense/test_pipeline.sh --quick

# Full test (30000 iterations)
./scripts/realsense/test_pipeline.sh --full

# Use Replica dataset instead
./scripts/realsense/test_pipeline.sh --quick --dataset replica
```

### What Gets Tested

| Step | Tested? | Notes |
|---|---|---|
| Environment & deps | Yes | Verifies PyTorch CUDA, nerfstudio, gs-mesh |
| Dataset loading | Yes | Downloads MuSHRoom or Replica sample data |
| Training | Yes | Runs dn-splatter with depth + normal supervision |
| Mesh extraction | Yes | Runs `gs-mesh o3dtsdf` |
| RealSense recording | **No** | Requires physical D435i/D455 |
| SAI processing | **No** | Requires a SpectacularAI recording |

### Why Not .bag Files?

SpectacularAI uses its own recording format (`data.jsonl` + `data.mkv` + `calibration.json`), not RealSense `.bag` files. There is no converter between the two formats. The recording step (`sai-record-realsense`) requires a live camera connection. Once you have a recording, you can replay it offline with `sai-cli process` without the camera.

---

## Pipeline Overview

```
  RealSense Camera
        │
        ▼
  ┌─────────────┐    sai-record-realsense
  │  1. Record   │    Captures RGB + depth + IMU
  └──────┬──────┘
         │  recording/ (data.jsonl, data.mkv, calibration.json)
         ▼
  ┌─────────────┐    process_sai.py → sai-cli process
  │  2. Process  │    Visual-Inertial SLAM, loop closure, keyframe selection
  └──────┬──────┘
         │  dataset/ (transforms.json, images/, depth/, sparse_pc.ply)
         ▼
  ┌─────────────┐    ns-train dn-splatter
  │  3. Train    │    3D Gaussian Splatting with depth + normal supervision
  └──────┬──────┘
         │  outputs/ (config.yml, checkpoints)
         ▼
  ┌─────────────┐    gs-mesh o3dtsdf
  │  4. Mesh     │    TSDF fusion → marching cubes → triangle mesh
  └─────────────┘
         │
         ▼
      mesh.ply
```

---

## Step 1: Record Data

Connect your RealSense D435i/D455 via USB 3.0, then:

```bash
./scripts/realsense/record.sh --output ./data/my_scene
```

A live preview window shows the camera feed and VIO tracking. Move the camera slowly around the scene. Press `Ctrl+C` to stop.

### Options

| Flag | Description |
|---|---|
| `--output DIR` | Where to save the recording (default: `./data/realsense_<timestamp>`) |
| `--no-preview` | Disable the live preview window (useful for headless servers) |
| `--recording-only` | Only record raw data, skip live VIO |

### Recording Tips

- **Move slowly and smoothly** — fast motion causes blur and tracking loss
- **Overlap** — ensure good overlap between different viewing angles
- **Lighting** — avoid very dark rooms or direct IR interference
- **Coverage** — capture the scene from multiple heights and angles
- **Loop** — return to starting position to enable loop closure
- **Duration** — 1-3 minutes is usually sufficient for a single room

### Output Structure

```
data/my_scene/
├── data.jsonl          # Frame metadata + IMU data
├── data.mkv            # Encoded video streams
├── calibration.json    # Camera intrinsics
└── vio_config.yaml     # SDK configuration
```

---

## Step 2: Process Recording

Convert the raw recording into a DN-Splatter compatible dataset:

```bash
./scripts/realsense/process.sh ./data/my_scene ./datasets/custom/my_scene
```

This runs SpectacularAI's SLAM pipeline with parameters tuned for DN-Splatter:
- Keyframe selection at 10 cm intervals
- PNG output for lossless quality
- Optimized feature tracking (2000 keypoints, 50 optimizer iterations)

### Options

| Flag | Description |
|---|---|
| `--preview` | Show 3D point cloud and trajectory during processing |
| `--key-frame-dist M` | Keyframe spacing in meters. Use `0.05` for tabletop, `0.10` (default) for desk-scale, `0.15` for room-scale |
| `--dry-run` | Print commands without executing |

### Output Structure

```
datasets/custom/my_scene/
├── transforms.json         # Camera poses + intrinsics (nerfstudio format)
├── sparse_pc.ply           # Sparse 3D point cloud
├── images/
│   ├── frame_00001.png     # Color images
│   ├── depth_00001.png     # 16-bit depth maps (millimeters)
│   ├── frame_00002.png
│   ├── depth_00002.png
│   └── ...
└── colmap/sparse/0/        # COLMAP-compatible files
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

---

## Step 3: Train DN-Splatter

Train a Gaussian Splatting model with depth and normal supervision:

```bash
./scripts/realsense/train.sh ./datasets/custom/my_scene
```

This runs `ns-train` with settings optimized for RealSense sensor depth:
- **EdgeAwareLogL1** depth loss (best for real sensor depth)
- Normal supervision derived from depth gradients
- Normal total-variation smoothing

Training opens a Nerfstudio viewer in your browser at `http://localhost:7007`.

### Options

| Flag | Description |
|---|---|
| `--method METHOD` | `dn-splatter` (default), `ags-mesh` (better meshes), `dn-splatter-big` (more Gaussians) |
| `--depth-lambda F` | Depth loss weight (default: `0.2`, increase for more depth fidelity) |
| `--normal-sup MODE` | `depth` (default, from sensor) or `mono` (requires pretrained normals) |
| `--max-iter N` | Training iterations (default: `30000`) |
| `--experiment NAME` | Name for organizing outputs |

### How Many Iterations?

With RealSense sensor depth, training converges faster than RGB-only methods:

| Scene Size | Iterations | Time (RTX 3090) | Notes |
|---|---|---|---|
| Small object / tabletop | 7,000 - 15,000 | ~4-8 min | Desk-scale capture |
| Single room | 30,000 (default) | ~15-20 min | Good balance of quality vs speed |
| Large room / multi-room | 30,000 - 50,000 | ~20-30 min | May need `dn-splatter-big` |

**30,000 iterations** is the recommended default for most indoor scenes. Going beyond 50k rarely helps — if quality is still poor, improve the capture (more coverage, slower motion, better lighting) or tune `--depth-lambda` instead.

### Using Monocular Normal Supervision

For potentially better results, you can generate monocular normals first:

```bash
# Download Omnidata weights (one-time)
python dn_splatter/data/download_scripts/download_omnidata.py

# Generate normal maps
python dn_splatter/scripts/normals_from_pretrain.py \
    --data-dir ./datasets/custom/my_scene \
    --resolution low

# Train with monocular normals
./scripts/realsense/train.sh ./datasets/custom/my_scene --normal-sup mono
```

Or using DSINE normals (no manual weight download):

```bash
python dn_splatter/scripts/normals_from_pretrain.py \
    --data-dir ./datasets/custom/my_scene \
    --normal-format dsine

./scripts/realsense/train.sh ./datasets/custom/my_scene \
    --normal-sup mono \
    --extra "--normal-format dsine"
```

### AGS-Mesh (Better Mesh Reconstruction)

For higher quality meshes, use AGS-Mesh with depth/normal consistency filtering:

```bash
# Generate depth confidence masks
python dn_splatter/scripts/depth_normal_consistency.py \
    --data-dir ./datasets/custom/my_scene \
    --transforms_name transforms.json \
    --normal-format omnidata

# Train with AGS-Mesh
./scripts/realsense/train.sh ./datasets/custom/my_scene --method ags-mesh
```

---

## Step 4: Extract Mesh

After training, extract a 3D mesh:

```bash
# Find your config file
CONFIG=$(ls -t outputs/dn-splatter/*/config.yml 2>/dev/null | head -1)

# Extract mesh (Open3D TSDF — recommended)
./scripts/realsense/extract_mesh.sh "$CONFIG"
```

### Options

| Flag | Description |
|---|---|
| `--method METHOD` | Extraction method (see table below) |
| `--output-dir DIR` | Where to save the mesh |
| `--voxel-size F` | TSDF voxel size (smaller = more detail, more memory) |

### Mesh Extraction Methods

| Method | Command | Best For |
|---|---|---|
| **Open3D TSDF** | `--method o3dtsdf` | General use (recommended) |
| TSDF Fusion | `--method tsdf` | Small objects (use `--voxel-size 0.004`) |
| Depth+Normal Poisson | `--method dn` | Smooth surfaces (requires normals) |
| Gaussian Poisson | `--method gaussians` | Direct from Gaussian positions |
| Marching Cubes | `--method marching` | Voxelized scenes |
| IsoOctree (AGS-Mesh) | See below | Smoothest surfaces |

### IsoOctree Extraction (Advanced)

For the smoothest meshes (from the AGS-Mesh paper):

```bash
python dn_splatter/scripts/isooctree_dn.py <training_output_root> \
    --transformation_path <path/transforms.json> \
    --tsdf_rel 0.03 \
    --output_mesh_file ./meshes/my_scene.ply \
    --subdivision_threshold=100
```

---

## Full Pipeline (One Command)

Run everything from recording to mesh in one command:

```bash
./scripts/realsense/run_pipeline.sh --scene living_room
```

This will:
1. Open the camera for recording (stop with `Ctrl+C`)
2. Process the recording into a dataset
3. Train DN-Splatter
4. Extract a mesh

### Pipeline Options

```bash
# Full pipeline with custom settings
./scripts/realsense/run_pipeline.sh \
    --scene kitchen \
    --method ags-mesh \
    --mesh-method o3dtsdf \
    --key-frame-dist 0.15 \
    --max-iter 30000

# Skip recording (use existing recording)
./scripts/realsense/run_pipeline.sh \
    --recording ./data/my_recording \
    --scene my_scene

# Skip recording + processing (use existing dataset)
./scripts/realsense/run_pipeline.sh \
    --dataset ./datasets/custom/my_scene

# Only extract mesh from a trained model
./scripts/realsense/run_pipeline.sh \
    --config outputs/dn-splatter/.../config.yml \
    --scene my_scene
```

---

## Tips & Troubleshooting

### Camera Not Detected

```
# Check USB connection
lsusb | grep -i intel

# Install udev rules
sudo apt install librealsense2-utils

# Or manually:
# Download 99-realsense-libusb.rules from the librealsense repo
# Copy to /etc/udev/rules.d/ and run: sudo udevadm control --reload-rules
```

### USB 2.0 Warning

If you see bandwidth errors or dropped frames, your camera is likely on a USB 2.0 port. Check with:

```bash
# Should show "5000M" for USB 3.0
lsusb -t | grep -i realsense
```

### Out of GPU Memory

- Reduce `--max-iter` to shorten training
- Use `dn-splatter` instead of `dn-splatter-big`
- Reduce image resolution during processing: edit `process_sai.py` and add `"inputResolution": "400p"` to `SAI_CLI_PROCESS_PARAMS`
- Close other GPU-consuming applications

### Poor Reconstruction Quality

- **Blurry results:** Move the camera more slowly during capture
- **Missing regions:** Ensure full scene coverage from multiple angles
- **Noisy depth:** RealSense depth is noisier beyond 3-4 meters; stay close to surfaces
- **Floating artifacts:** Try `ags-mesh` method for better depth filtering, or increase `--depth-lambda` to `0.5`

### Training Loss Not Converging

- Check that `transforms.json` has reasonable camera poses (positions shouldn't be all zeros)
- Verify depth maps exist: `ls datasets/custom/my_scene/images/depth_*.png | wc -l`
- Try reducing depth influence: `--depth-lambda 0.1`

### SpectacularAI Processing Fails

- Ensure `ffmpeg` is installed: `ffmpeg -version`
- Check recording integrity: `ls -la data/my_scene/data.jsonl data/my_scene/data.mkv`
- Try with `--preview` flag to visualize the SLAM processing
- For very large scenes, increase `--key-frame-dist` to `0.2`

### Mesh Has Holes

- Use `o3dtsdf` method (most robust)
- Train for more iterations (`--max-iter 50000`)
- Ensure scene is well-covered from multiple angles
- For small objects, use `tsdf` with `--voxel-size 0.004`

---

## Script Reference

All scripts are in `scripts/realsense/`:

| Script | Purpose |
|---|---|
| `setup_env.sh` | One-step environment installation (conda) |
| `setup_env_uv.sh` | One-step environment installation (uv) |
| `record.sh` | Record RGB-D + IMU data with RealSense |
| `process.sh` | Convert recording to DN-Splatter dataset |
| `train.sh` | Train DN-Splatter model |
| `extract_mesh.sh` | Extract mesh from trained model |
| `run_pipeline.sh` | End-to-end pipeline (record -> mesh) |
| `test_pipeline.sh` | Test pipeline with sample data (no camera) |

Each script supports `--help` for detailed usage information.
