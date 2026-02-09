"""Process a RealSense D415 .bag recording into DN-Splatter format.

Takes a .bag file recorded from a D415 and produces a nerfstudio-compatible
dataset by:
  1. Extracting aligned RGB + depth frames from the .bag file
  2. Running COLMAP (via ns-process-data) for camera pose estimation
  3. Patching transforms.json with sensor depth file paths

The D415 has no IMU, so poses are estimated purely from visual features
using COLMAP's Structure-from-Motion pipeline.

Usage:
    python dn_splatter/scripts/process_d415.py recording.bag ./datasets/custom/my_scene
    python dn_splatter/scripts/process_d415.py recording.bag --every-n-frames 10
    python dn_splatter/scripts/process_d415.py ./data/frames_dir/ ./datasets/custom/my_scene --skip-extraction
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_OUT_FOLDER = "datasets/custom"


def extract_frames_from_bag(bag_path, output_dir, every_n=15):
    """Extract aligned RGB + depth frames from a .bag file.

    Args:
        bag_path: Path to the .bag file
        output_dir: Output directory for extracted frames
        every_n: Save every Nth frame (default 15 = ~2fps from 30fps)

    Returns:
        dict with intrinsics and frame count
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] pyrealsense2 not found. Install with: pip install pyrealsense2")
        sys.exit(1)

    images_dir = Path(output_dir) / "images"
    depth_dir = Path(output_dir) / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline for playback
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    print(f"[INFO] Opening .bag file: {bag_path}")
    profile = pipeline.start(config)

    # Disable real-time playback (process as fast as possible)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Get intrinsics from color stream
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    meta = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "w": intrinsics.width,
        "h": intrinsics.height,
        "depth_scale": depth_scale,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }

    print(f"[INFO] Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
          f"cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    print(f"[INFO] Resolution: {intrinsics.width}x{intrinsics.height}")
    print(f"[INFO] Depth scale: {depth_scale}")
    print(f"[INFO] Extracting every {every_n}th frame...")

    # Align depth to color
    align = rs.align(rs.stream.color)

    frame_count = 0
    saved_count = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                # End of .bag file
                break

            frame_count += 1

            if frame_count % every_n != 0:
                continue

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            saved_count += 1
            fname = f"frame_{saved_count:05d}.png"

            # Save color as BGR 8-bit PNG
            color_image = np.asanyarray(color_frame.get_data())
            import cv2
            cv2.imwrite(str(images_dir / fname), color_image)

            # Save depth as 16-bit PNG (values in depth sensor units, typically mm)
            depth_image = np.asanyarray(depth_frame.get_data())
            cv2.imwrite(str(depth_dir / fname), depth_image)

            if saved_count % 20 == 0:
                print(f"  Extracted {saved_count} frames (from {frame_count} total)...")

    except Exception as e:
        print(f"[WARN] Stopped extraction: {e}")
    finally:
        pipeline.stop()

    meta["frames_total"] = frame_count
    meta["frames_saved"] = saved_count
    print(f"[OK] Extracted {saved_count} frames from {frame_count} total")

    return meta


def extract_frames_from_directory(frames_dir, output_dir):
    """Copy frames from a pre-extracted directory.

    Expects:
        frames_dir/color/frame_NNNNN.png
        frames_dir/depth/frame_NNNNN.png
        frames_dir/recording_meta.json (optional)

    Args:
        frames_dir: Path to directory with color/ and depth/ subdirs
        output_dir: Output directory

    Returns:
        dict with intrinsics (from recording_meta.json if available)
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Determine source layout
    if (frames_dir / "color").exists():
        color_src = frames_dir / "color"
        depth_src = frames_dir / "depth"
    elif (frames_dir / "images").exists():
        color_src = frames_dir / "images"
        depth_src = frames_dir / "depth"
    else:
        print(f"[ERROR] Expected 'color/' or 'images/' subdirectory in {frames_dir}")
        sys.exit(1)

    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Copy color images
    color_files = sorted(color_src.glob("*.png"))
    print(f"[INFO] Copying {len(color_files)} color images...")
    for f in color_files:
        shutil.copy2(f, images_dir / f.name)

    # Copy depth images
    depth_files = sorted(depth_src.glob("*.png"))
    print(f"[INFO] Copying {len(depth_files)} depth images...")
    for f in depth_files:
        shutil.copy2(f, depth_dir / f.name)

    # Load metadata if available
    meta = {}
    meta_path = frames_dir / "recording_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[INFO] Loaded intrinsics from {meta_path}")

    meta["frames_saved"] = len(color_files)
    return meta


def run_colmap(output_dir, matching_method="exhaustive"):
    """Run COLMAP via ns-process-data to get poses and transforms.json.

    Args:
        output_dir: Dataset directory (must have images/ subdirectory)
        matching_method: COLMAP matching method
    """
    images_dir = Path(output_dir) / "images"

    cmd = [
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(output_dir),
        "--skip-image-processing",
        "--matching-method", matching_method,
    ]

    print(f"[CMD] {' '.join(cmd)}")
    print("[INFO] Running COLMAP (this may take a while)...")
    subprocess.check_call(cmd)
    print("[OK] COLMAP complete")


def patch_transforms_with_depth(output_dir):
    """Add depth_file_path to each frame in transforms.json.

    Args:
        output_dir: Dataset directory with transforms.json and depth/
    """
    output_dir = Path(output_dir)
    transforms_path = output_dir / "transforms.json"
    depth_dir = output_dir / "depth"

    with open(transforms_path) as f:
        transforms = json.load(f)

    # Build set of available depth files
    available_depths = {f.name for f in depth_dir.glob("*.png")}

    patched = 0
    missing = 0
    for frame in transforms["frames"]:
        # file_path is like "images/frame_00001.png" or "./images/frame_00001.png"
        image_name = Path(frame["file_path"]).name
        if image_name in available_depths:
            frame["depth_file_path"] = f"depth/{image_name}"
            patched += 1
        else:
            missing += 1

    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)

    print(f"[OK] Patched transforms.json: {patched} frames with depth, {missing} without")
    if missing > 0:
        print(f"[WARN] {missing} frames have no matching depth file "
              "(COLMAP may have dropped some frames or names don't match)")


def process(args):
    input_path = Path(args.input)
    is_bag = input_path.suffix == ".bag"

    # Determine output directory
    if args.output_dir is None:
        name = input_path.stem
        output_dir = Path(DEFAULT_OUT_FOLDER) / name
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("============================================")
    print("  Process D415 Recording")
    print("============================================")
    print(f"  Input:          {input_path}")
    print(f"  Output:         {output_dir}")
    print(f"  Is .bag:        {is_bag}")
    if is_bag:
        print(f"  Every N frames: {args.every_n_frames}")
    print(f"  COLMAP method:  {args.matching_method}")
    print("============================================")
    print()

    if args.dry_run:
        print("[DRY RUN] Would extract frames, run COLMAP, and patch transforms.json")
        return

    # Step 1: Extract or copy frames
    if is_bag and not args.skip_extraction:
        print(">>> Step 1/3: Extracting frames from .bag <<<")
        meta = extract_frames_from_bag(input_path, output_dir, every_n=args.every_n_frames)
    elif not is_bag:
        print(">>> Step 1/3: Copying frames from directory <<<")
        meta = extract_frames_from_directory(input_path, output_dir)
    else:
        print(">>> Step 1/3: Skipping extraction (--skip-extraction) <<<")
        meta = {}

    # Step 2: Run COLMAP
    if not args.skip_colmap:
        print()
        print(">>> Step 2/3: Running COLMAP for pose estimation <<<")
        run_colmap(output_dir, matching_method=args.matching_method)
    else:
        print()
        print(">>> Step 2/3: Skipping COLMAP (--skip-colmap) <<<")

    # Step 3: Patch transforms.json with depth paths
    print()
    print(">>> Step 3/3: Adding depth paths to transforms.json <<<")
    patch_transforms_with_depth(output_dir)

    # Save processing metadata
    if meta:
        meta_path = output_dir / "processing_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    print()
    print(f"[OK] Dataset processed to: {output_dir}")
    print()
    print("Output structure:")
    print(f"  {output_dir}/")
    print(f"  ├── transforms.json      (camera poses + depth paths)")
    print(f"  ├── sparse_pc.ply        (sparse point cloud)")
    print(f"  ├── images/")
    print(f"  │   ├── frame_00001.png  (color images)")
    print(f"  │   └── ...")
    print(f"  ├── depth/")
    print(f"  │   ├── frame_00001.png  (16-bit depth, mm)")
    print(f"  │   └── ...")
    print(f"  └── colmap/sparse/0/     (COLMAP reconstruction)")
    print()
    print("Next step: train DN-Splatter")
    print(f"  ./scripts/realsense/train.sh {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=str,
                        help="Path to .bag file or directory with color/ and depth/ subdirs")
    parser.add_argument("output_dir", type=str, nargs="?", default=None,
                        help="Output dataset directory (default: datasets/custom/<input_name>)")
    parser.add_argument("--every-n-frames", type=int, default=15,
                        help="Save every Nth frame from .bag (default: 15, ~2fps from 30fps)")
    parser.add_argument("--matching-method", type=str, default="exhaustive",
                        choices=["exhaustive", "sequential", "vocab_tree"],
                        help="COLMAP matching method (default: exhaustive)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip frame extraction (reuse already-extracted frames)")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip COLMAP (reuse existing colmap results)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing")
    args = parser.parse_args()

    process(args)
