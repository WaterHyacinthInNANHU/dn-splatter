"""Visualize camera poses and point clouds from a processed dataset.

Loads transforms.json and sparse_pc.ply (plus optionally dense depth
backprojections) and shows them in an interactive web-based viser viewer.
Useful as a sanity check after preprocessing and before training.

Usage:
    python dn_splatter/scripts/visualize_dataset.py datasets/custom/my_scene
    python dn_splatter/scripts/visualize_dataset.py datasets/custom/my_scene --dense --every-n 10
    python dn_splatter/scripts/visualize_dataset.py datasets/custom/my_scene --max-depth 3.0 --port 8080
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def rotation_matrix_to_wxyz(R):
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def backproject_depth(depth_path, color_path, fx, fy, cx, cy, pose, max_depth=10.0, stride=4):
    """Backproject a depth image into world-space points + colors.

    Returns:
        (points_Nx3, colors_Nx3) as float32/float32 numpy arrays.
        Colors are in [0, 1].
    """
    from PIL import Image

    depth = np.array(Image.open(depth_path)).astype(np.float64)
    color = np.array(Image.open(color_path)).astype(np.float64) / 255.0
    h, w = depth.shape
    depth_m = depth * 0.001

    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u, v)

    d = depth_m[vv, uu].flatten()
    c = color[vv, uu].reshape(-1, 3)

    mask = (d > 0) & (d < max_depth)
    d = d[mask]
    c = c[mask]
    uu_flat = uu.flatten()[mask]
    vv_flat = vv.flatten()[mask]

    x = (uu_flat - cx) / fx * d
    y = (vv_flat - cy) / fy * d
    z = d
    pts_cam = np.stack([x, y, z], axis=1)

    R = pose[:3, :3]
    t = pose[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    return pts_world.astype(np.float32), c.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=str, help="Path to processed dataset directory")
    parser.add_argument("--dense", action="store_true", help="Also backproject depth frames into dense point clouds")
    parser.add_argument("--every-n", type=int, default=10, help="Backproject every Nth frame for dense clouds (default: 10)")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Max depth in meters for dense backprojection (default: 10.0)")
    parser.add_argument("--frustum-scale", type=float, default=0.15, help="Camera frustum size in world units (default: 0.15)")
    parser.add_argument("--point-size", type=float, default=0.005, help="Point size for point clouds (default: 0.005)")
    parser.add_argument("--no-sparse", action="store_true", help="Don't load sparse_pc.ply")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8890, help="Server port (default: 8890)")
    args = parser.parse_args()

    import viser

    dataset_dir = Path(args.dataset_dir)
    transforms_path = dataset_dir / "transforms.json"

    if not transforms_path.exists():
        print(f"[ERROR] transforms.json not found in {dataset_dir}")
        sys.exit(1)

    with open(transforms_path) as f:
        transforms = json.load(f)

    fx = transforms["fl_x"]
    fy = transforms["fl_y"]
    cx = transforms["cx"]
    cy = transforms["cy"]
    w = transforms["w"]
    h = transforms["h"]
    frames = transforms["frames"]

    print(f"[INFO] Dataset: {dataset_dir}")
    print(f"[INFO] Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"[INFO] Resolution: {w}x{h}")
    print(f"[INFO] Frames with poses: {len(frames)}")

    # Start viser server
    server = viser.ViserServer(host=args.host, port=args.port)
    server.set_up_direction("+z")
    server.world_axes.visible = True

    # Load sparse point cloud
    sparse_path = dataset_dir / "sparse_pc.ply"
    if sparse_path.exists() and not args.no_sparse:
        import trimesh
        mesh = trimesh.load(str(sparse_path))
        pts = np.array(mesh.vertices, dtype=np.float32)
        if hasattr(mesh, 'colors') and mesh.colors is not None:
            colors = np.array(mesh.colors[:, :3], dtype=np.uint8)
        else:
            colors = np.full((len(pts), 3), 128, dtype=np.uint8)
        server.add_point_cloud(
            "/sparse_pc",
            points=pts,
            colors=colors,
            point_size=args.point_size * 2,
            point_shape="circle",
        )
        print(f"[INFO] Loaded sparse_pc.ply: {len(pts)} points")

    # Add camera frustums
    fov_y = 2.0 * math.atan2(h / 2.0, fy)
    aspect = w / h
    print(f"[INFO] Drawing {len(frames)} camera frustums (fov_y={math.degrees(fov_y):.1f} deg)...")

    for i, frame in enumerate(frames):
        c2w = np.array(frame["transform_matrix"])
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        wxyz = rotation_matrix_to_wxyz(R)

        # Color gradient: blue (first) -> red (last)
        frac = i / max(len(frames) - 1, 1)
        r = int(255 * frac)
        g = 50
        b = int(255 * (1 - frac))

        frame_name = Path(frame.get("file_path", f"frame_{i:05d}")).stem
        server.add_camera_frustum(
            f"/cameras/{frame_name}",
            fov=fov_y,
            aspect=aspect,
            scale=args.frustum_scale,
            color=(r, g, b),
            wxyz=wxyz,
            position=t,
        )

    # Dense depth backprojection
    if args.dense:
        print(f"[INFO] Backprojecting depth (every {args.every_n} frames, max_depth={args.max_depth}m)...")
        all_pts = []
        all_colors = []
        for i, frame in enumerate(frames):
            if i % args.every_n != 0:
                continue
            depth_key = frame.get("depth_file_path")
            if not depth_key:
                continue
            depth_path = dataset_dir / depth_key
            color_path = dataset_dir / frame["file_path"]
            if not depth_path.exists() or not color_path.exists():
                continue

            pose = np.array(frame["transform_matrix"])
            pts, colors = backproject_depth(
                depth_path, color_path, fx, fy, cx, cy, pose,
                max_depth=args.max_depth, stride=4,
            )
            all_pts.append(pts)
            all_colors.append(colors)
            print(f"  Frame {i}: {len(pts)} points")

        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)
            all_colors = (np.concatenate(all_colors, axis=0) * 255).astype(np.uint8)
            server.add_point_cloud(
                "/dense_depth",
                points=all_pts,
                colors=all_colors,
                point_size=args.point_size,
                point_shape="circle",
            )
            print(f"[INFO] Dense point cloud: {len(all_pts)} total points")

    print(f"\n[INFO] Viewer ready at http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down.")
        server.stop()


if __name__ == "__main__":
    main()
