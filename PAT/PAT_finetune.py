"""
PAT_finetune.py — LoRA fine-tuning of PAT_B on VideoArtGS scenes, optionally with the
extra tracking / VGGT input modalities (VideoArtGS extension, see PAT/pat_extra_feats.py).

Uses the OFFICIAL Particulate training path (verified against RuiningLi/particulate):
`model(...)` (not `model.infer(...)`) computes all 9 losses internally, INCLUDING the
Hungarian matching between predicted part slots and GT part ids (run_matching=True).

Pipeline:
  1. builds GT tensors in the exact format of particulate/datasets.py:
     per-point part_ids, motion class (0 none / 1 revolute / 2 prismatic),
     plucker in the normalized [-0.5,0.5]^3 frame, per-point closest-point-on-axis,
     parent->child part_structure_matrix
  2. injects a hand-rolled LoRA into every Linear inside blocks[*].attn{1,2,3}; the new
     `extra_embeds` branches (if --extra_feats is given) are trained fully; everything
     else is frozen
  3. trains with the official loss weights (range losses off: joint_infos has no range GT)
  4. evaluates axis angle error + revolute origin error on TEST_SCENES BEFORE and AFTER
  5. merges LoRA back and saves particulate/model_ckpt/<save_name> (+ a sidecar
     <save_name>.json describing the extra inputs), verified to load with strict=True.
     PAT/init_deform_PAT.py reads the sidecar and rebuilds the same inputs at inference.

Extra input modalities (--extra_feats track_geo,track_tapip,vggt):
  track_geo    56-d explicit trajectory + motion statistics from filtered.npz (TAPIP3D tracks)
  track_tapip 384-d TAPIP3D updater hidden state   (needs data_tools/extract_tapip3d_feats.py)
  vggt        128-d PCA-reduced VGGT tokens         (needs data_tools/extract_vggt_feats.py)

GT part labels (--labels):
  sphere  joint center + dist_max spheres (original heuristic; crude)
  track   points whose nearest TAPIP3D tracks move get the nearest type-compatible moving
          joint, the rest is base; falls back to sphere labels if a part ends up empty
  gt      nearest surface sample of the GT part meshes gt/part_k.ply (real 3D segmentation GT)
Articulation GT (--joint_gt): joint_infos.json (motion analysis) or gt = gt/mobility_v2.json
(read with utils.metrics.read_gt, i.e. the evaluation's frame and joint filtering).

MUST CHECK BY HAND before trusting results:
  [A] open the saved <out_dir>/gt_labels_<scene>.ply files: the colored regions should
      roughly cover the actual moving parts.
  [B] Frame consistency: --pat_up_dir must equal the one used in init_deform_PAT.py.

Run from the VideoArtGS repo root, e.g.
  python PAT/PAT_finetune.py --epochs 150 --extra_feats track_geo,track_tapip,vggt \
      --labels track --train_on_all --save_name trained_PAT_model.pt
"""

import os
import sys
import json
import copy
import random
import time
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particulate.models import PAT_B, load_pat_checkpoint
from particulate.articulation_utils import axis_point_to_plucker, plucker_to_axis_point
try:
    from particulate.partfield_utils import get_partfield_model, obtain_partfield_feats
except ImportError:  # upstream repo keeps it at top level
    from partfield_utils import get_partfield_model, obtain_partfield_feats
from pat_extra_feats import (NormFrame, parse_extra_names, extra_feat_dims,
                             compute_scene_extra_feats, track_part_labels)
from utils.metrics import read_gt

TEST_SCENES = ["100481", "101284", "103811", "45194", "47648"]
DATA_PATH = os.path.join(ROOT, "data/videoartgs")
CKPT_PATH = os.path.join(ROOT, "particulate/model_ckpt")

NUM_ENC_POINTS = 40000  # PartField encoder points (official infer.py value)
MAX_PARTS = 16          # model's max_parts; ALL part-dim GT must be zero-padded to this

# Exact configuration of the released checkpoint (configs/particulate-B.yaml).
MODEL_KWARGS = dict(
    input_dim=448, dropout=0.1, use_normals=True, max_parts=MAX_PARTS,
    use_part_id_embedding=True, use_raw_coords=True,
    use_point_features_for_motion_decoding=False,
    num_mask_hypotheses=1, motion_representation="per_point_closest",
)

# Same convention as init_deform_PAT.py / official infer.py (xyz @ R.T).
UP_DIR_ROTATIONS = {
    "X":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
    "-X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
    "Y":  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    "-Y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
    "Z":  np.eye(3, dtype=np.float32),
    "-Z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
}

# Official train-particulate-B.yaml weights; range losses off (no GT ranges here).
LOSS_WEIGHTS = {
    "point_mask_loss": 1.0,
    "dice_loss": 1.0,
    "motion_hierarchy_loss": 1.0,
    "part_motion_classification_loss": 1.0,
    "part_motion_axis_loss_revolute": 1.0,
    "part_motion_axis_loss_prismatic": 1.0,
    "part_motion_range_loss_revolute": 0.0,
    "part_motion_range_loss_prismatic": 0.0,
    "point_closest_point_on_axis_loss": 1.0,
}


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=dt) / r)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        delta = nn.functional.linear(nn.functional.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + delta * self.scaling


def inject_lora(model, r, alpha, dropout, name_pattern="attn"):
    """Wrap every nn.Linear whose qualified name contains name_pattern.

    For PAT_B this catches blocks[i].attn{1,2,3}.{to_q,to_k,to_v,to_out.0}
    = 6 layers x 3 attentions x 4 linears = 72 modules.
    """
    replaced = []
    snapshot = list(model.named_modules())
    for parent_name, parent in snapshot:
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and name_pattern in full:
                setattr(parent, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(full)
    if not replaced:
        raise RuntimeError(
            f"LoRA injection matched no Linear with pattern '{name_pattern}'. "
            "Refusing to continue with an unmodified model.")
    return replaced


def merge_and_unwrap(model):
    """W <- W + B@A * scaling, then restore the original module tree/key names."""
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight += (child.lora_B @ child.lora_A) * child.scaling
                setattr(parent, child_name, child.base)


# ----------------------------------------------------------------------------- data
def find_scene_dir(scene_name, subsets):
    for subset in subsets:
        d = os.path.join(DATA_PATH, subset, scene_name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "joint_infos.json")) \
                and os.path.exists(os.path.join(d, "point_cloud.ply")):
            return d
    return None


def find_all_scenes(subsets):
    scenes = []
    for subset in subsets:
        sp = os.path.join(DATA_PATH, subset)
        if os.path.isdir(sp):
            for s in sorted(os.listdir(sp)):
                if find_scene_dir(s, [subset]):
                    scenes.append(s)
    return scenes


def load_xyz_normals(ply_path):
    data = PlyData.read(ply_path).elements[0].data
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    normals = None
    if {"nx", "ny", "nz"} <= set(data.dtype.names):
        normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float32)
        if (np.linalg.norm(normals, axis=1) <= 1e-6).mean() > 0.5:
            normals = None
    if normals is None:
        print(f"    estimating normals with Open3D for {os.path.basename(os.path.dirname(ply_path))}")
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=16))
        pcd.orient_normals_consistent_tangent_plane(16)
        normals = np.asarray(pcd.normals, dtype=np.float32)
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(n > 1e-6, normals / np.maximum(n, 1e-8),
                       np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return xyz, normals


def save_label_ply(xyz, part_ids, out_path):
    palette = np.random.RandomState(0).randint(0, 255, (int(part_ids.max()) + 1, 3), dtype=np.uint8)
    colors = palette[part_ids.astype(np.int64)]
    v = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(v, "vertex")]).write(out_path)


def sphere_labels(scene_name, xyz_w, moving, scale, args):
    """0 = static base, 1..K = moving joints in json order, from center + dist_max spheres."""
    part_ids = np.zeros(len(xyz_w), dtype=np.int64)
    for k, j in enumerate(moving, start=1):
        c = np.asarray(j["center"], dtype=np.float32)
        d = np.linalg.norm(xyz_w - c, axis=1)
        radius = float(j.get("dist_max", args.label_radius_ratio * scale))
        hit = d < radius
        if int(hit.sum()) < args.label_min_pts:
            reason = ("radius too small" if d.min() < 0.25 * scale
                      else "CENTER LIKELY IN A DIFFERENT FRAME than point_cloud.ply")
            print(f"    [label] {scene_name} joint {k}: sphere r={radius:.4f} caught {int(hit.sum())} pts; "
                  f"nearest point {d.min():.4f} away, bbox {scale:.4f} -> {reason}")
            if args.label_knn_fallback:
                hit = np.zeros(len(xyz_w), dtype=bool)
                hit[np.argsort(d)[:args.label_min_pts]] = True
                print(f"    [label] fallback: labelling the {args.label_min_pts} nearest points "
                      f"(THESE LABELS ARE SUSPECT — check the ply)")
        part_ids[hit] = k
    return part_ids


def track_labels(scene_name, scene_dir, xyz_w, frame, moving, fallback, args):
    """Track-derived labels (see pat_extra_feats.track_part_labels); -1 (no nearby track) is
    filled from the nearest labelled point. Falls back to `fallback` if a part is empty."""
    labels = track_part_labels(scene_dir, xyz_w, frame, moving, knn_k=args.knn_k, valid_thr=args.valid_thr)
    known = labels >= 0
    if known.sum() == 0:
        print(f"    [label] {scene_name}: no track-based labels at all, using sphere labels")
        return fallback
    if (~known).any():
        _, nn_idx = cKDTree(xyz_w[known]).query(xyz_w[~known])
        labels[~known] = labels[known][nn_idx]
    counts = [int((labels == p).sum()) for p in range(len(moving) + 1)]
    if min(counts) < args.label_min_pts:
        print(f"    [label] {scene_name}: track labels leave a part nearly empty {counts}, using sphere labels")
        return fallback
    return labels


def gt_part_meshes(scene_dir):
    """gt/part_0.ply (static base), gt/part_k.ply (k-th moving joint of read_gt, same order as
    utils/metrics.eval_CD pairs them). Triangle meshes in the dataset world frame."""
    import glob
    files = sorted(glob.glob(os.path.join(scene_dir, "gt", "part_*.ply")),
                   key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))
    if not files:
        raise FileNotFoundError(f"no gt/part_*.ply under {scene_dir}")
    return files


def gt_labels(scene_name, scene_dir, xyz_w, n_samples=300000, min_per_part=5000):
    """Per-point labels from the GT part meshes: sample every part surface uniformly
    (area-proportional, >= min_per_part points), label each point by its nearest sample.
    Returns (labels 0..K, nearest-sample distance)."""
    import open3d as o3d
    files = gt_part_meshes(scene_dir)
    meshes = [o3d.io.read_triangle_mesh(f) for f in files]
    areas = np.array([max(m.get_surface_area(), 1e-8) for m in meshes])
    samples, labels = [], []
    for k, m in enumerate(meshes):
        n = int(max(min_per_part, round(n_samples * areas[k] / areas.sum())))
        pts = np.asarray(m.sample_points_uniformly(number_of_points=n).points, dtype=np.float32)
        samples.append(pts)
        labels.append(np.full(len(pts), k, dtype=np.int64))
    samples, labels = np.concatenate(samples), np.concatenate(labels)
    d, idx = cKDTree(samples).query(xyz_w)
    print(f"    [label] {scene_name}: GT-mesh labels from {len(files)} parts; point-to-GT-surface "
          f"median {np.median(d):.4f}, p95 {np.percentile(d, 95):.4f}, >5cm {float((d > 0.05).mean()):.1%}")
    return labels[idx], d


def gt_moving_joints(scene_dir):
    """Moving joints from gt/mobility_v2.json in the dataset world frame (utils.metrics.read_gt:
    same rotation and filtering as the evaluation), in the order matching gt/part_{k}.ply."""
    gts = read_gt(os.path.join(scene_dir, "gt", "mobility_v2.json"))
    moving = []
    for g in gts:
        d = np.asarray(g["direction"], dtype=np.float32)
        moving.append({"joint_type": g["joint_type"],
                       "direction": (d / (np.linalg.norm(d) + 1e-8)).tolist(),
                       "origin": np.asarray(g["origin"], dtype=np.float32).tolist(),
                       "gt_idx": int(g.get("idx", -1))})
    return moving


def prepare_scene(scene_name, args, partfield_model, out_dir, extra_names):
    """Load one scene and precompute everything reusable across training steps.

    Returns None if the scene has no moving ('r'/'p') joints.
    All GT tensors follow particulate/datasets.py conventions, in the
    normalized [-0.5, 0.5]^3 frame (after the optional up-dir rotation).
    """
    scene_dir = find_scene_dir(scene_name, args.subsets)
    if scene_dir is None:
        return None
    xyz_w, normals_w = load_xyz_normals(os.path.join(scene_dir, "point_cloud.ply"))
    with open(os.path.join(scene_dir, "joint_infos.json")) as f:
        joint_infos = json.load(f)
    if args.joint_gt == "gt":
        # real articulation GT (gt/mobility_v2.json); centres only feed the sphere-label fallback
        moving = gt_moving_joints(scene_dir)
        n_gt_parts = len(gt_part_meshes(scene_dir))
        if n_gt_parts - 1 != len(moving):
            print(f"    [skip] {scene_name}: {n_gt_parts} GT part meshes but {len(moving)} GT joints")
            return None
        import open3d as o3d
        for k, j in enumerate(moving, start=1):
            j["center"] = np.asarray(o3d.io.read_triangle_mesh(gt_part_meshes(scene_dir)[k]).vertices).mean(0).tolist()
    else:
        moving = [j for j in joint_infos if j["joint_type"] in ("r", "p")]
    if not moving:
        return None

    R = UP_DIR_ROTATIONS[args.pat_up_dir]
    xyz_r = xyz_w @ R.T
    normals_r = normals_w @ R.T
    bbmin, bbmax = xyz_r.min(0), xyz_r.max(0)
    center = (bbmin + bbmax) / 2
    scale = float((bbmax - bbmin).max())
    xyz_n = ((xyz_r - center) / scale).astype(np.float32)
    frame = NormFrame(R, center, scale)

    # --- GT part labels: 0 = static base, 1..K = moving joints in json order. ---
    P = len(moving) + 1
    if P > MAX_PARTS:
        print(f"    [skip] {scene_name}: {P} parts exceeds max_parts={MAX_PARTS}")
        return None
    if args.labels == "gt":
        part_ids, _ = gt_labels(scene_name, scene_dir, xyz_w)
        if part_ids.max() != P - 1:
            print(f"    [skip] {scene_name}: GT labels have {part_ids.max() + 1} parts, expected {P}")
            return None
    else:
        part_ids = sphere_labels(scene_name, xyz_w, moving, scale, args)
        if args.labels == "track":
            part_ids = track_labels(scene_name, scene_dir, xyz_w, frame, moving, part_ids, args)
    save_label_ply(xyz_w, part_ids, os.path.join(out_dir, f"gt_labels_{scene_name}.ply"))
    counts = [int((part_ids == p).sum()) for p in range(P)]
    if min(counts) == 0:
        print(f"    [skip] {scene_name}: some GT part got 0 points (radii wrong?) counts={counts}")
        return None

    # --- Per-part motion GT in the normalized frame (zero-padded to MAX_PARTS). ---
    motion_class = np.zeros(MAX_PARTS, dtype=np.int64)   # 0 none / 1 revolute / 2 prismatic
    plucker = np.zeros((MAX_PARTS, 6), dtype=np.float32)
    prismatic_axis = np.zeros((MAX_PARTS, 3), dtype=np.float32)
    for k, j in enumerate(moving, start=1):
        d = np.asarray(j["direction"], dtype=np.float32)
        d = (R @ d)
        d /= (np.linalg.norm(d) + 1e-8)
        if j["joint_type"] == "r":
            motion_class[k] = 1
            o = np.asarray(j["origin"], dtype=np.float32)
            o_n = ((R @ o) - center) / scale
            plucker[k] = axis_point_to_plucker(d, o_n.astype(np.float32))
        else:
            motion_class[k] = 2
            prismatic_axis[k] = d

    # --- Per-point closest point on the revolute axis (datasets.py formula). ---
    cpoa = np.zeros((len(xyz_n), 3), dtype=np.float32)
    for k in range(P):
        if motion_class[k] == 1:
            axis_dir, axis_pt = plucker_to_axis_point(plucker[k])
            pts = xyz_n[part_ids == k]
            proj = axis_pt + axis_dir * np.dot(pts - axis_pt, axis_dir)[:, None]
            cpoa[part_ids == k] = proj

    # --- Hierarchy: base is the parent of every moving part. ---
    struct = np.zeros((MAX_PARTS, MAX_PARTS), dtype=bool)
    struct[0, 1:P] = True

    # --- Extra input modalities for the WHOLE cloud (VideoArtGS extension). ---
    extra = {}
    if extra_names:
        t0 = time.time()
        feats_extra, _ = compute_scene_extra_feats(scene_dir, xyz_w, frame, extra_names,
                                                   k_frames=args.k_frames, knn_k=args.knn_k,
                                                   valid_thr=args.valid_thr)
        extra = {n: torch.from_numpy(v) for n, v in feats_extra.items()}
        summary = ", ".join(f"{n}:{tuple(v.shape)}" for n, v in extra.items())
        if "track_geo" in extra:
            summary += f", track-valid {float(extra['track_geo'][:, -1].mean()):.2f}"
        print(f"    extra feats {summary} ({time.time() - t0:.0f}s)")

    # --- PartField features for the WHOLE cloud, computed once (chunked). ---
    N = len(xyz_n)
    enc_idx = np.random.choice(N, min(N, NUM_ENC_POINTS), replace=False)
    enc = torch.from_numpy(xyz_n[enc_idx]).float().cuda().unsqueeze(0)
    feats_chunks = []
    with torch.no_grad():
        for s in range(0, N, args.feat_chunk):
            dec = torch.from_numpy(xyz_n[s:s + args.feat_chunk]).float().cuda().unsqueeze(0)
            feats_chunks.append(obtain_partfield_feats(partfield_model, enc, dec).float().cpu())
    feats_full = torch.cat(feats_chunks, dim=1)[0]      # (N, 448) on CPU
    assert feats_full.shape[-1] == 448, feats_full.shape
    torch.cuda.empty_cache()

    return dict(
        name=scene_name, num_parts=P, norm_scale=scale,
        xyz=torch.from_numpy(xyz_n),
        normals=torch.from_numpy(normals_r.astype(np.float32)),
        feats=feats_full,
        extra=extra,
        part_ids=torch.from_numpy(part_ids),
        cpoa=torch.from_numpy(cpoa),
        motion_class=torch.from_numpy(motion_class),
        plucker=torch.from_numpy(plucker),
        prismatic_axis=torch.from_numpy(prismatic_axis),
        struct=torch.from_numpy(struct),
        moving_types=[j["joint_type"] for j in moving],
    )


def gather_extra(scene, idx, dropout_p=0.0, training=False):
    """{name: (1, n, D)} extra inputs for the sampled points; whole-modality dropout in training."""
    out = {}
    for name, feat in scene["extra"].items():
        f = feat[idx].cuda().unsqueeze(0)
        if training and dropout_p > 0 and random.random() < dropout_p:
            f = torch.zeros_like(f)
        out[name] = f
    return out


def sample_batch(scene, num_points, dropout_p=0.0):
    """Random point subset each step = natural augmentation (official recipe
    also resamples 2048 points per step)."""
    N = scene["xyz"].shape[0]
    idx = torch.from_numpy(np.random.choice(N, min(N, num_points), replace=False))
    return dict(
        xyz=scene["xyz"][idx].cuda().unsqueeze(0),
        feats=scene["feats"][idx].cuda().unsqueeze(0),
        normals=scene["normals"][idx].cuda().unsqueeze(0),
        extra_feats=gather_extra(scene, idx, dropout_p, training=True),
        part_ids=scene["part_ids"][idx].cuda().unsqueeze(0),
        num_valid_parts=torch.tensor([scene["num_parts"]], dtype=torch.long).cuda(),
        part_structure_matrix=scene["struct"].cuda().unsqueeze(0),
        gt_part_motion_class=scene["motion_class"].cuda().unsqueeze(0),
        gt_revolute_plucker=scene["plucker"].cuda().unsqueeze(0),
        gt_prismatic_axis=scene["prismatic_axis"].cuda().unsqueeze(0),
        gt_closest_point_on_axis=scene["cpoa"][idx].cuda().unsqueeze(0),
    )


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, scenes, num_points=4096, seed=1234, tag=""):
    """Axis angle (deg, sign-invariant) + revolute origin line distance (world
    units), with prediction slots aligned to GT part ids via infer(gt_part_ids=...,
    run_matching=True). Also reports the point-segmentation accuracy w.r.t. the labels."""
    model.eval()
    rng = np.random.RandomState(seed)
    rows, angles, seg_accs = [], [], []
    for scene in scenes:
        N = scene["xyz"].shape[0]
        idx = rng.choice(N, min(N, num_points), replace=False)
        idx_t = torch.from_numpy(idx)
        out = model.infer(
            xyz=scene["xyz"][idx_t].cuda().unsqueeze(0),
            feats=scene["feats"][idx_t].cuda().unsqueeze(0),
            normals=scene["normals"][idx_t].cuda().unsqueeze(0),
            extra_feats=gather_extra(scene, idx_t),
            gt_part_ids=scene["part_ids"][idx_t].cuda().unsqueeze(0),
            run_matching=True, min_part_confidence=0.0,
        )[0]
        plk = to_np(out["revolute_plucker"])
        pax = to_np(out["prismatic_axis"])
        pred_ids = to_np(out["part_ids"]).reshape(-1)
        gt_ids = to_np(scene["part_ids"][idx_t])
        seg_accs.append(float((pred_ids == gt_ids).mean()))
        for k, jt in enumerate(scene["moving_types"], start=1):
            if jt == "r":
                gt_axis, gt_pt = plucker_to_axis_point(to_np(scene["plucker"][k]))
                pr_axis, pr_pt = plucker_to_axis_point(plk[k])
                ang = np.degrees(np.arccos(np.clip(abs(float(gt_axis @ pr_axis)), 0, 1)))
                od = float(np.linalg.norm(np.cross(pr_axis, gt_pt - pr_pt))) * scene["norm_scale"]
                rows.append((scene["name"], k, "r", ang, od))
            else:
                gt_axis = to_np(scene["prismatic_axis"][k])
                pr = pax[k] / (np.linalg.norm(pax[k]) + 1e-8)
                ang = np.degrees(np.arccos(np.clip(abs(float(gt_axis @ pr)), 0, 1)))
                rows.append((scene["name"], k, "p", ang, float("nan")))
            angles.append(rows[-1][3])
    model.train()
    print(f"\n[Eval {tag}]  scene | joint | type | axis angle err (deg) | rev origin err (world)")
    for name, k, jt, ang, od in rows:
        od_s = f"{od:.4f}" if od == od else "   --"
        print(f"  {name:>8} |  {k}  |  {jt}  |  {ang:8.2f}  |  {od_s}")
    mean_ang = float(np.mean(angles)) if angles else float("nan")
    mean_seg = float(np.mean(seg_accs)) if seg_accs else float("nan")
    print(f"  mean axis angle error: {mean_ang:.2f} deg over {len(angles)} joints; "
          f"mean point-label accuracy (matched slots): {mean_seg:.3f}\n")
    return {"rows": rows, "mean_angle_deg": mean_ang, "mean_seg_acc": mean_seg}


# ----------------------------------------------------------------------------- main
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="passes over the train scenes")
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_points", type=int, default=2048,
                        help="decoder points per step (official training used 2048)")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--feat_chunk", type=int, default=16384)
    parser.add_argument("--pat_up_dir", type=str, default="Z", choices=list(UP_DIR_ROTATIONS),
                        help="CHECK [B]: must match the up_dir you use in init_deform_PAT.py")
    parser.add_argument("--subsets", type=str, default="sapien",
                        help="comma separated data/videoartgs subsets to use (default: sapien only)")
    parser.add_argument("--label_radius_ratio", type=float, default=0.25,
                        help="fallback label sphere radius (fraction of bbox) if 'dist_max' missing")
    parser.add_argument("--label_min_pts", type=int, default=256,
                        help="a GT part with fewer points than this triggers the KNN fallback")
    parser.add_argument("--label_knn_fallback", action=BooleanOptionalAction,
                        default=True, help="label the N nearest points when the sphere catches none "
                                           "(keeps the scene instead of skipping it; verify the ply)")
    parser.add_argument("--labels", type=str, default="sphere", choices=["sphere", "track", "gt"],
                        help="part-label source: sphere / track pseudo labels, or gt = gt/part_*.ply meshes")
    parser.add_argument("--joint_gt", type=str, default="joint_infos", choices=["joint_infos", "gt"],
                        help="articulation GT: joint_infos.json (motion analysis) or gt = gt/mobility_v2.json")
    parser.add_argument("--bridge", type=str, default="legacy", choices=["legacy", "pat_vlm"],
                        help="recorded in the sidecar: how init_deform_PAT.py should turn PAT outputs into "
                             "deform-field slots (legacy = override joint_infos axes; pat_vlm = PAT centres/"
                             "extents/axes for the joint_infos.json slot list)")
    parser.add_argument("--w_mask", type=float, default=None, help="override point_mask loss weight")
    parser.add_argument("--w_dice", type=float, default=None, help="override dice loss weight")
    parser.add_argument("--train_heads", action="store_true",
                        help="also unfreeze the motion decoder heads (not LoRA-only)")
    parser.add_argument("--train_on_all", action="store_true",
                        help="train on ALL scenes incl. TEST_SCENES (train = test; for overfitting checks)")
    parser.add_argument("--limit_train_scenes", type=int, default=0, help="smoke test: use only N scenes")
    parser.add_argument("--scenes", type=str, default="",
                        help="comma separated explicit scene list (overrides the directory scan)")
    # --- extra input modalities (VideoArtGS extension) ---
    parser.add_argument("--extra_feats", type=str, default="",
                        help="comma separated subset of track_geo,track_tapip,vggt")
    parser.add_argument("--extra_dropout", type=float, default=0.2,
                        help="probability of zeroing a whole extra modality for a training step")
    parser.add_argument("--extra_lr", type=float, default=1e-3, help="learning rate of the extra_embeds branches")
    parser.add_argument("--k_frames", type=int, default=8, help="time stamps in the track_geo descriptor")
    parser.add_argument("--knn_k", type=int, default=4, help="tracks averaged per point")
    parser.add_argument("--valid_thr", type=float, default=0.02,
                        help="max point-to-track distance (normalized units) for a valid transfer")
    parser.add_argument("--save_name", type=str, default="updated_pat_model.pt",
                        help="output checkpoint name under particulate/model_ckpt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=os.path.join(CKPT_PATH, "finetune_debug"))
    args = parser.parse_args()
    args.subsets = [s for s in args.subsets.split(",") if s]

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    weights = dict(LOSS_WEIGHTS)
    if args.w_mask is not None: weights["point_mask_loss"] = args.w_mask
    if args.w_dice is not None: weights["dice_loss"] = args.w_dice

    extra_names = parse_extra_names(args.extra_feats)
    dims = extra_feat_dims(extra_names)
    print(f"[FT] extra input modalities: {dims if dims else 'none'}")

    all_scenes = find_all_scenes(args.subsets)
    if args.scenes:
        wanted = [s for s in args.scenes.split(",") if s]
        missing = [s for s in wanted if s not in all_scenes]
        if missing:
            raise RuntimeError(f"--scenes not found under {DATA_PATH}: {missing}")
        all_scenes = wanted
    if args.train_on_all:
        train_names = list(all_scenes)
        print("[FT] --train_on_all: TEST_SCENES are INCLUDED in training (train = test)")
    else:
        train_names = [s for s in all_scenes if s not in TEST_SCENES]
    test_names = [s for s in all_scenes if s in TEST_SCENES]
    if args.limit_train_scenes:
        train_names = train_names[:args.limit_train_scenes]
    print(f"[FT] {len(all_scenes)} scenes found: {len(train_names)} train, {len(test_names)} test")
    if not train_names:
        raise RuntimeError(f"No training scenes under {DATA_PATH} ({args.subsets})")

    # --- Base model ---
    model = PAT_B(**MODEL_KWARGS, extra_feat_dims=dims).cuda()
    base_sd = torch.load(os.path.join(CKPT_PATH, "pat_model.pt"), map_location="cpu")
    load_pat_checkpoint(model, base_sd)
    print("[FT] loaded pat_model.pt (all released weights; extra branches zero-initialised)")

    for p in model.parameters():
        p.requires_grad_(False)
    replaced = inject_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    print(f"[FT] LoRA injected into {len(replaced)} Linears, e.g. {replaced[0]}")
    extra_params = list(model.extra_embeds.parameters())
    for p in extra_params:
        p.requires_grad_(True)
    if args.train_heads:
        for head in [model.revolute_motion_decoder, model.prismatic_motion_decoder,
                     model.point_motion_decoder, model.part_motion_classifier]:
            for p in head.parameters():
                p.requires_grad_(True)
        print("[FT] motion heads unfrozen as well")
    extra_ids = {id(p) for p in extra_params}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in extra_ids]
    trainable = lora_params + extra_params
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[FT] trainable {n_tr/1e6:.2f}M / total {n_all/1e6:.1f}M ({100*n_tr/n_all:.2f}%), "
          f"of which extra_embeds {sum(p.numel() for p in extra_params)/1e6:.2f}M")

    # --- Data (features precomputed once per scene) ---
    print("[FT] loading PartField and precomputing per-scene features...")
    partfield_model = get_partfield_model(device="cuda")
    train_scenes, test_scenes = [], []
    for name in sorted(set(train_names + test_names), key=lambda s: (s not in train_names, s)):
        s = prepare_scene(name, args, partfield_model, args.out_dir, extra_names)
        if s is None:
            print(f"    [skip] {name}")
            continue
        if name in train_names:
            train_scenes.append(s)
        if name in TEST_SCENES:
            test_scenes.append(s)
        print(f"    {name}: {s['xyz'].shape[0]} pts, {s['num_parts']-1} moving joint(s) {s['moving_types']}")
    del partfield_model
    torch.cuda.empty_cache()
    if not train_scenes:
        raise RuntimeError("All training scenes were skipped; check labels / data layout.")
    print(f"[FT] check the label dumps under {args.out_dir}/gt_labels_*.ply before trusting anything")

    eval_tag = "BEFORE (zero-shot)" + (" [test scenes are in train]" if args.train_on_all else "")
    metrics_before = evaluate(model, test_scenes, tag=eval_tag) if test_scenes else None

    # --- Train ---
    param_groups = [{"params": lora_params, "lr": args.lr}]
    if extra_params:
        param_groups.append({"params": extra_params, "lr": args.extra_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = args.epochs * len(train_scenes)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 50) * 0.5 * (1 + np.cos(np.pi * min(step, total_steps) / max(total_steps, 1))))
    model.train()
    step = 0
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        order = np.random.permutation(len(train_scenes))
        pbar = tqdm(order, desc=f"epoch {epoch}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        ep_losses = {}
        for i, si in enumerate(pbar):
            batch = sample_batch(train_scenes[si], args.num_points, args.extra_dropout)
            losses, _ = model(run_matching=True, **batch)
            # scenes without revolute parts give an empty-mean NaN in
            # point_closest_point_on_axis_loss (zero gradient); mask it so the
            # logged total stays finite
            losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
            loss = sum(weights.get(k, 0.0) * v for k, v in losses.items())
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            for k, v in losses.items():
                if weights.get(k, 0.0) > 0:
                    ep_losses[k] = ep_losses.get(k, 0.0) + float(v) / len(order)
            pbar.set_postfix({k.replace("part_motion_", "").replace("_loss", ""): f"{float(v):.3f}"
                              for k, v in losses.items() if weights.get(k, 0.0) > 0 and "range" not in k})
        total = sum(weights[k] * v for k, v in ep_losses.items())
        history.append({"epoch": epoch, "total": total, **ep_losses})
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"[FT] epoch {epoch:3d}/{args.epochs}  total {total:.3f}  " +
                  "  ".join(f"{k.replace('part_motion_', '').replace('_loss', '')} {v:.3f}"
                            for k, v in ep_losses.items() if "range" not in k) +
                  f"  ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    metrics_after = evaluate(model, test_scenes, tag="AFTER (LoRA-FT)") if test_scenes else None

    # --- Merge, verify, save ---
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if "lora_" in k or k.startswith("extra_embeds.")}, os.path.join(args.out_dir, "lora_adapter.pt"))
    merge_and_unwrap(model)

    fresh = PAT_B(**MODEL_KWARGS, extra_feat_dims=dims)
    fresh.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)

    changed = any(not torch.equal(fresh.state_dict()[k], base_sd[k]) for k in base_sd)
    if not changed:
        raise RuntimeError("Merged weights are identical to pat_model.pt — training had no effect; do NOT ship this file.")

    save_path = os.path.join(CKPT_PATH, args.save_name)
    torch.save(fresh.state_dict(), save_path)
    sidecar = {
        "extra_feats": extra_names,
        "extra_feat_dims": dims,
        "k_frames": args.k_frames,
        "knn_k": args.knn_k,
        "valid_thr": args.valid_thr,
        "pat_up_dir": args.pat_up_dir,
        "model_kwargs": MODEL_KWARGS,
        "labels": args.labels,
        "joint_gt": args.joint_gt,
        "bridge": args.bridge,
        "train_scenes": [s["name"] for s in train_scenes],
        "train_on_all": args.train_on_all,
        "base_checkpoint": "pat_model.pt",
    }
    sidecar_path = os.path.splitext(save_path)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[FT] saved merged checkpoint to {save_path} "
          f"({os.path.getsize(save_path)/1e6:.1f} MB, strict-load verified, differs from base)")
    print(f"[FT] sidecar with the input configuration written to {sidecar_path}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"before": metrics_before, "after": metrics_after, "history": history,
                   "args": vars(args)}, f, indent=2, default=str)
    print(f"[FT] before/after metrics written to {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
