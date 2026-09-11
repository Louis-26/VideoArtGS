"""
PAT_5 dataset: per-frame articulated point clouds for the VideoArtGS sapien scenes.

PAT consumes a static point cloud, so "frame t" is materialised by posing the
object with the GT joint values of frame t:

  gt/part_*.ply        exact per-part meshes (part_0 = static base)
  gt/mobility_v2.json  exact joint axes (read_gt applies the R_coord rotation
                       that puts them in the point_cloud.ply frame)
  gt/part_info.json    per-frame joint values, part_move[frame] -> 3-vector

Verified on all 20 sapien scenes:
  * read_gt()[k-1] corresponds to gt/part_k.ply
  * filtered.npz frame f corresponds to part_move frame 150 + f
    (frames 0..149 are the static canonical block), so the real 3D point
    tracks can calibrate the sign of each joint's motion instead of guessing.

Poses are generated with delta = value(t) - value(0), so frame 0 reproduces the
canonical cloud exactly. Any rotation about the correct axis is a valid
(geometry, axis) training pair, which is what makes this robust: the GT axis is
fixed in world coordinates and does not depend on the pose.
"""

import glob
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from utils.metrics import read_gt  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "videoartgs", "sapien")
CANONICAL_FRAMES = 150  # part_move frames 0..149 are the static canonical block


def rodrigues(axis, angle):
    """Rotation matrix for `angle` radians about the unit vector `axis`."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def load_scene(scene, samples_total=16384, seed=0):
    """Meshes sampled once in the canonical pose, plus GT axes and joint values."""
    d = os.path.join(DATA_PATH, scene)
    part_paths = sorted(glob.glob(os.path.join(d, "gt", "part_*.ply")),
                        key=lambda p: int(os.path.basename(p)[5:-4]))
    joints = read_gt(os.path.join(d, "gt", "mobility_v2.json"))
    if len(joints) != len(part_paths) - 1:
        raise ValueError(f"{scene}: {len(joints)} gt joints vs {len(part_paths)-1} movable parts")

    meshes = [trimesh.load(p, process=False, force="mesh") for p in part_paths]
    areas = np.array([max(float(m.area), 1e-9) for m in meshes])
    # Area-proportional budget with a floor so thin parts stay represented.
    floor = max(256, samples_total // (8 * len(meshes)))
    quota = np.maximum(floor, (areas / areas.sum() * samples_total).astype(int))
    quota = np.maximum(1, (quota * samples_total / quota.sum()).astype(int))

    rng = np.random.RandomState(seed)
    pts, nrm, pid = [], [], []
    for k, (m, q) in enumerate(zip(meshes, quota)):
        p, fidx = trimesh.sample.sample_surface(m, int(q), seed=int(rng.randint(1 << 30)))
        pts.append(np.asarray(p, dtype=np.float64))
        nrm.append(np.asarray(m.face_normals[fidx], dtype=np.float64))
        pid.append(np.full(int(q), k, dtype=np.int64))

    info = json.load(open(os.path.join(d, "gt", "part_info.json")))
    T = len(info[0]["part_move"])
    move = np.array([[info[j]["part_move"][str(t)] for j in range(len(info))] for t in range(T)])
    # Each joint drives exactly one component of its 3-vector.
    comp = [int(np.argmax(np.abs(move[:, j]).sum(0))) for j in range(move.shape[1])]
    values = np.stack([move[:, j, comp[j]] for j in range(move.shape[1])], axis=1)  # (T, J)
    if values.shape[1] != len(joints):
        raise ValueError(f"{scene}: part_info has {values.shape[1]} joints, mobility_v2 has {len(joints)}")

    axes = []
    for j in joints:
        dvec = np.asarray(j["direction"], dtype=np.float64)
        axes.append(dict(
            direction=dvec / (np.linalg.norm(dvec) + 1e-12),
            origin=np.asarray(j["origin"], dtype=np.float64),
            joint_type=j["joint_type"],
        ))

    return dict(scene=scene, parts_xyz=pts, parts_nrm=nrm, parts_pid=pid,
                axes=axes, values=values, num_parts=len(part_paths))


def pose(scene_data, frame, signs):
    """Articulate the canonical samples to `frame`; returns xyz, normals, part_ids."""
    values, axes = scene_data["values"], scene_data["axes"]
    delta = values[frame] - values[0]
    xyz, nrm = [scene_data["parts_xyz"][0]], [scene_data["parts_nrm"][0]]
    for k, ax in enumerate(axes, start=1):
        p, n = scene_data["parts_xyz"][k], scene_data["parts_nrm"][k]
        dv = signs[k - 1] * delta[k - 1]
        if ax["joint_type"] == "r":
            R = rodrigues(ax["direction"], dv)
            p = (p - ax["origin"]) @ R.T + ax["origin"]
            n = n @ R.T
        else:
            p = p + dv * ax["direction"][None]
        xyz.append(p)
        nrm.append(n)
    return (np.concatenate(xyz).astype(np.float32),
            np.concatenate(nrm).astype(np.float32),
            np.concatenate(scene_data["parts_pid"]))


def calibrate_signs(scene_data, verbose=True):
    """Pick each joint's motion sign using the real 3D tracks in filtered.npz.

    filtered.npz frame f == part_move frame CANONICAL_FRAMES + f, and the video
    moves one joint per 50-frame segment, so each joint is calibrated against the
    frames where only it is moving.
    """
    scene = scene_data["scene"]
    coords = np.load(os.path.join(DATA_PATH, scene, "filtered.npz"))["coords"]  # (F, P, 3)
    F, J = coords.shape[0], len(scene_data["axes"])
    seg = F // J

    base_signs = np.ones(J)
    xyz0, _, pid0 = pose(scene_data, 0, base_signs)
    # Track labels come from the canonical pose, where tracks and meshes agree.
    track_pid = pid0[cKDTree(xyz0).query(coords[0], k=1)[1]]

    signs, report = np.ones(J), []
    for k in range(J):
        f = min(seg * (k + 1) - 1, F - 1)          # last frame of joint k's segment
        frame = CANONICAL_FRAMES + f
        sel = coords[f][track_pid == k + 1]
        if len(sel) < 8 or abs(scene_data["values"][frame][k] - scene_data["values"][0][k]) < 1e-3:
            report.append((k, float("nan"), float("nan"), 1.0))
            continue
        err = {}
        for s in (1.0, -1.0):
            trial = signs.copy()
            trial[k] = s
            xyz, _, pid = pose(scene_data, frame, trial)
            # Compare only against the part that moved.
            err[s] = float(np.median(cKDTree(xyz[pid == k + 1]).query(sel, k=1)[0]))
        signs[k] = 1.0 if err[1.0] <= err[-1.0] else -1.0
        report.append((k, err[1.0], err[-1.0], signs[k]))

    if verbose:
        ext = float(np.linalg.norm(xyz0.max(0) - xyz0.min(0)))
        for k, e_pos, e_neg, s in report:
            jt = scene_data["axes"][k]["joint_type"]
            best = min(e_pos, e_neg) if e_pos == e_pos else float("nan")
            print(f"    {scene:>7} joint{k}[{jt}] err(+)={e_pos:.4f} err(-)={e_neg:.4f} "
                  f"-> sign {s:+.0f}  residual/extent={best/ext:.4f}")
    return signs, report


def select_frames(scene_data, max_poses):
    """Unique joint states, timestamped by first occurrence, spread over the timeline."""
    values = scene_data["values"]
    _, first = np.unique(np.round(values, 6), axis=0, return_index=True)
    frames = np.sort(first)
    if len(frames) > max_poses:
        frames = frames[np.linspace(0, len(frames) - 1, max_poses).astype(int)]
    return frames
