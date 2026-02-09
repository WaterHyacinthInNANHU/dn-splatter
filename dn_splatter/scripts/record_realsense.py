"""Record RGB-D data from Intel RealSense D415 to .bag file.

Records color and depth streams from a RealSense D415 camera using
pyrealsense2. The D415 does not have an IMU, so this script records
only RGB-D streams. The .bag format preserves raw sensor data and
allows replay for re-processing with different parameters.

Usage:
    python dn_splatter/scripts/record_realsense.py --output ./data/my_scene.bag
    python dn_splatter/scripts/record_realsense.py --output ./data/my_scene.bag --no-preview
    python dn_splatter/scripts/record_realsense.py --output ./data/my_scene.bag --width 1280 --height 720 --fps 30
"""

import argparse
import os
import signal
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, required=True, help="Output .bag file path")
    parser.add_argument("--width", type=int, default=1280, help="Stream width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Stream height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window")
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] pyrealsense2 not found. Install with: pip install pyrealsense2")
        sys.exit(1)

    # Ensure output path ends with .bag
    output_path = args.output
    if not output_path.endswith(".bag"):
        output_path += ".bag"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_record_to_file(output_path)

    print("============================================")
    print("  RealSense D415 Recording")
    print("============================================")
    print(f"  Output:     {output_path}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  FPS:        {args.fps}")
    print(f"  Preview:    {not args.no_preview}")
    print("============================================")
    print()

    # Start pipeline
    print("[INFO] Starting RealSense pipeline...")
    profile = pipeline.start(config)

    # Print device info
    device = profile.get_device()
    print(f"[INFO] Device: {device.get_info(rs.camera_info.name)}")
    print(f"[INFO] Serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"[INFO] Firmware: {device.get_info(rs.camera_info.firmware_version)}")

    # Print intrinsics
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Color intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
          f"cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    print(f"[INFO] Depth scale: {depth_scale} ({1.0/depth_scale:.0f} units per meter)")
    print()

    # Graceful shutdown on Ctrl+C
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[INFO] Stopping recording...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Optional preview
    cv2 = None
    if not args.no_preview:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            print("[WARN] OpenCV not found, disabling preview. Install with: pip install opencv-python")

    # Auto-exposure warmup
    print("[INFO] Warming up auto-exposure (1 second)...")
    warmup_frames = args.fps  # ~1 second
    for _ in range(warmup_frames):
        try:
            pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            break

    print("[INFO] Recording... Press Ctrl+C to stop.")
    frame_count = 0

    while running:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            continue

        frame_count += 1

        if cv2 is not None and frame_count % 3 == 0:  # Update preview every 3rd frame
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                # Add frame counter overlay
                cv2.putText(color_image, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("RealSense D415 Recording", color_image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC to stop
                    running = False

    # Stop pipeline
    pipeline.stop()

    if cv2 is not None:
        cv2.destroyAllWindows()

    print()
    print(f"[OK] Recording saved: {output_path}")
    print(f"[OK] Total frames: {frame_count}")
    print()
    print("Next step: process the recording")
    print(f"  python dn_splatter/scripts/process_d415.py {output_path} ./datasets/custom/my_scene")


if __name__ == "__main__":
    main()
