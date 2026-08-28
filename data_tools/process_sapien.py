#!/usr/bin/env python3
"""Verify regenerated transforms.json / point_cloud.ply against the shipped ones.

Usage:
    python verify_preprocess.py \
        --shipped data/videoartgs/sapien/168 \
        --rerun   new_data/videoartgs/sapien/168

Verdict levels:
    [BYTE-IDENTICAL]    md5 equal -> released code + released inputs fully
                        reproduce the shipped artifact. Strongest evidence.
    [NUMERIC-IDENTICAL] values equal up to floating-point noise (same
                        algorithm; environment/library differences only).
    [NEAR-IDENTICAL]    sub-voxel discrepancies (voxel-boundary assignment
                        drift across open3d/CUDA versions). Acceptable.
    [DIVERGENT]         structural differences -> inspect the reported fields.
"""
import os
import json
import hashlib
import argparse
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

SCALAR_FIELDS = ["camera_angle_x", "camera_angle_y", "focal_x", "focal_y",
                 "cx", "cy", "w", "h"]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ------------------------------------------------------------ transforms ---

def compare_transforms(pa, pb):
    print("=" * 62)
    print("transforms.json")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if md5(pa) == md5(pb):
        print("  [BYTE-IDENTICAL] md5 match: byte-for-byte reproduction, "
              "nothing more to check.")
        return

    a, b = json.load(open(pa)), json.load(open(pb))

    # 1) global intrinsics fields
    for k in SCALAR_FIELDS:
        if not np.isclose(a[k], b[k], rtol=0, atol=1e-9):
            print(f"  [DIVERGENT] intrinsics field {k}: {a[k]} vs {b[k]}")
            return
    # 2) frame count
    if len(a["frames"]) != len(b["frames"]):
        print(f"  [DIVERGENT] frame count differs: "
              f"{len(a['frames'])} vs {len(b['frames'])}")
        return
    # 3) per frame: file_path/state must match exactly; time/pose numerically
    max_pose_diff, max_time_diff = 0.0, 0.0
    for i, (fa, fb) in enumerate(zip(a["frames"], b["frames"])):
        if fa["file_path"] != fb["file_path"] or fa["state"] != fb["state"]:
            print(f"  [DIVERGENT] frame {i} file_path/state mismatch: "
                  f"{fa['file_path']},{fa['state']} vs "
                  f"{fb['file_path']},{fb['state']}")
            return
        max_time_diff = max(max_time_diff, abs(fa["time"] - fb["time"]))
        diff = np.abs(np.array(fa["transform_matrix"]) -
                      np.array(fb["transform_matrix"])).max()
        max_pose_diff = max(max_pose_diff, diff)

    print(f"  frame count matches ({len(a['frames'])}); "
          f"state/time segmentation matches")
    print(f"  max |pose diff| = {max_pose_diff:.3e},  "
          f"max |time diff| = {max_time_diff:.3e}")
    if max_pose_diff < 1e-9 and max_time_diff < 1e-12:
        print("  [NUMERIC-IDENTICAL] semantically identical "
              "(md5 mismatch comes from serialization format only)")
    else:
        print("  [DIVERGENT] numeric differences; check that camera.json is "
              "the same source and num_cano == 150")


# ----------------------------------------------------------- point cloud ---

def load_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    try:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)
    except ValueError:
        rgb = None
    return xyz, rgb


def compare_pointclouds(pa, pb, tol_numeric=1e-6):
    print("=" * 62)
    print("point_cloud.ply")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if md5(pa) == md5(pb):
        print("  [BYTE-IDENTICAL] md5 match: byte-for-byte reproduction.")
        return

    A, ca = load_ply(pa)
    B, cb = load_ply(pb)
    print(f"  points: shipped={len(A)}, rerun={len(B)}"
          f"  (relative diff {abs(len(A) - len(B)) / len(A) * 100:.3f}%)")

    # Bidirectional nearest neighbor:
    #   A->B proves every shipped point is reproduced;
    #   B->A proves the rerun contains no extra / displaced points.
    d_ab, idx = cKDTree(B).query(A)
    d_ba, _ = cKDTree(A).query(B)
    print(f"  A->B  max NN = {d_ab.max():.3e} m,  mean = {d_ab.mean():.3e} m")
    print(f"  B->A  max NN = {d_ba.max():.3e} m,  mean = {d_ba.mean():.3e} m")

    same_count = len(A) == len(B)
    numeric = same_count and d_ab.max() < tol_numeric and d_ba.max() < tol_numeric

    if numeric and ca is not None and cb is not None:
        cdiff = np.abs(ca - cb[idx]).max()
        print(f"  matched-point color max |RGB diff| = {cdiff:.3e} (0-255 scale)")

    if numeric:
        print("  [NUMERIC-IDENTICAL] same point set; differences are float "
              "noise -> same algorithm + same inputs")
    elif d_ab.max() < 0.01 and d_ba.max() < 0.01:   # within voxel size eps/5 = 1 cm
        print("  [NEAR-IDENTICAL] sub-voxel differences -> same algorithm; "
              "voxel-boundary assignment drift from open3d/CUDA version "
              "differences. Acceptable.")
    else:
        print("  [DIVERGENT] structural differences above the voxel scale; "
              "check num_cano / eps / the input frame set")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", default="data/videoartgs/sapien/168",
                    help="official shipped scene directory")
    ap.add_argument("--rerun", default="new_data/videoartgs/sapien/168",
                    help="scene directory containing your regenerated artifacts")
    args = ap.parse_args()

    compare_transforms(os.path.join(args.shipped, "transforms.json"),
                       os.path.join(args.rerun, "transforms.json"))
    compare_pointclouds(os.path.join(args.shipped, "point_cloud.ply"),
                        os.path.join(args.rerun, "point_cloud.ply"))