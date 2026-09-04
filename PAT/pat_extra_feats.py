"""
Turn VGGT / TAPIP3D tracking information into per-point input features for PAT.
Shared by PAT/PAT_finetune.py (training) and PAT/init_deform_PAT.py (inference)
so both sides build exactly the same tensors.

Modalities (per canonical point, expressed in PAT's normalized frame):
  track_geo    56  explicit 3D trajectory + motion statistics derived from
                   <scene>/filtered.npz (TAPIP3D tracks used by the VideoArtGS pipeline)
                   24 displacement at K=8 normalized time stamps | 8 visibility at those
                   stamps | 3 max-displacement, path length, mean speed | 5 trajectory PCA
                   main direction + two variance ratios | 4 motion-type one-hot
                   (unknown/static/prismatic/revolute) | 6 fitted revolute axis dir + origin
                   | 3 fitted prismatic dir | 1 visibility ratio | 1 kNN confidence | 1 valid
  track_tapip 384  TAPIP3D EfficientUpdateFormer hidden state per track
                   (<scene>/pat_extra/tapip3d_feats.npz from data_tools/extract_tapip3d_feats.py)
  vggt        128  PCA-reduced VGGT aggregator tokens per point
                   (<scene>/pat_extra/vggt128.npy from data_tools/extract_vggt_feats.py)

Track-level quantities are transferred to the points with an inverse-distance
weighted k-NN (k=4) from the tracks' first-frame positions; points farther than
`valid_thr` (normalized units, default 0.02 ~ 3 cm) from every track get zeros and
valid=0 so the model can learn to ignore them.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_tools"))

EXTRA_FEAT_DIMS = {"track_geo": 56, "track_tapip": 384, "vggt": 128}

# motion_analysis.analyze_trajectory thresholds for the sapien subset
SAPIEN_MOTION_THRESHOLDS = dict(static_threshold=0.1, line_threshold=0.01, circle_threshold=0.01)


def parse_extra_names(spec):
    names = [n.strip() for n in str(spec).split(",") if n.strip()]
    for n in names:
        if n not in EXTRA_FEAT_DIMS:
            raise ValueError(f"unknown extra feature '{n}', choose from {list(EXTRA_FEAT_DIMS)}")
    return names


def extra_feat_dims(names):
    return {n: EXTRA_FEAT_DIMS[n] for n in names}


class NormFrame:
    """World -> PAT frame: rotate by R (up-dir), then map the bbox to [-0.5, 0.5]^3."""

    def __init__(self, R, center, scale):
        self.R = np.asarray(R, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = float(scale)

    def pts(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (((x.reshape(-1, 3) @ self.R.T) - self.center) / self.scale).reshape(x.shape).astype(np.float32)

    def vec(self, v):  # displacement-like quantities: rotate + scale, no shift
        v = np.asarray(v, dtype=np.float64)
        return ((v.reshape(-1, 3) @ self.R.T) / self.scale).reshape(v.shape).astype(np.float32)

    def dir(self, d):  # unit directions: rotate only
        d = np.asarray(d, dtype=np.float64)
        d = (d.reshape(-1, 3) @ self.R.T)
        d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
        return d.reshape(np.asarray(d).shape).astype(np.float32)


def canon_sign(v):
    """Resolve the sign ambiguity of axis-like vectors: largest-magnitude component > 0."""
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    j = np.argmax(np.abs(v), axis=-1)
    s = np.sign(np.take_along_axis(v, j[..., None], axis=-1))
    s[s == 0] = 1
    return v * s


def build_knn(query_xyz, ref_xyz, k=4):
    """Inverse-distance weights from every query point to its k nearest reference points."""
    k = min(k, len(ref_xyz))
    d, idx = cKDTree(ref_xyz).query(query_xyz, k=k)
    d, idx = d.reshape(len(query_xyz), k), idx.reshape(len(query_xyz), k)
    w = 1.0 / (d + 1e-4)
    w = w / w.sum(axis=1, keepdims=True)
    return idx, w.astype(np.float32), d[:, 0].astype(np.float32)


def transfer(track_feats, idx, w):
    return np.einsum("nk,nkd->nd", w, np.asarray(track_feats, dtype=np.float32)[idx])


# ----------------------------------------------------------------------------- motion classification
def classify_tracks(coords_w, visibs, cache_path=None, **thresholds):
    """Per-track motion type (-1 unknown / 0 static / 1 prismatic / 2 revolute) and the
    fitted world-frame axis parameters, via motion_analysis.filter_unreasonable_motion.
    Slow (RANSAC line + circle fit per track), so the result is cached per scene."""
    if cache_path and os.path.exists(cache_path):
        c = np.load(cache_path)
        if c["mtypes"].shape[0] == coords_w.shape[1]:
            return c["mtypes"], c["rev_dir"], c["rev_origin"], c["pris_dir"]
    from motion_analysis import filter_unreasonable_motion  # heavy import (matplotlib, sklearn)
    thr = dict(SAPIEN_MOTION_THRESHOLDS)
    thr.update(thresholds)
    _, _, mtypes, mparams = filter_unreasonable_motion(coords_w.copy(), visibs.copy(), **thr)
    Nt = coords_w.shape[1]
    rev_dir, rev_origin, pris_dir = np.zeros((Nt, 3), np.float32), np.zeros((Nt, 3), np.float32), np.zeros((Nt, 3), np.float32)
    for n in range(Nt):
        p = mparams[n]
        if mtypes[n] == 2 and p:
            rev_dir[n], rev_origin[n] = p["direction"], p["origin"]
        elif mtypes[n] == 1 and p:
            pris_dir[n] = p["direction"]
    mtypes = np.asarray(mtypes, dtype=np.int64)
    if cache_path:
        np.savez(cache_path, mtypes=mtypes, rev_dir=rev_dir, rev_origin=rev_origin, pris_dir=pris_dir)
    return mtypes, rev_dir, rev_origin, pris_dir


# ----------------------------------------------------------------------------- track_geo
def track_descriptors(coords_w, visibs, frame, mtypes, rev_dir, rev_origin, pris_dir, k_frames=8):
    """(N_track, 54) per-track descriptor in the normalized frame."""
    T, Nt, _ = coords_w.shape
    x = frame.pts(coords_w)                                             # (T, Nt, 3)
    vis = visibs.astype(np.float32)

    ts = np.linspace(0, T - 1, k_frames + 1).round().astype(int)[1:]    # K stamps, t=0 excluded
    disp = (x[ts] - x[0:1]).transpose(1, 0, 2).reshape(Nt, -1)          # (Nt, 3K)
    vis_k = vis[ts].T                                                   # (Nt, K)

    d_all = np.linalg.norm(x - x[0:1], axis=-1)                         # (T, Nt)
    max_disp = d_all.max(0)
    step = np.linalg.norm(np.diff(x, axis=0), axis=-1)                  # (T-1, Nt)
    path_len = step.sum(0)
    mean_speed = path_len / max(T - 1, 1)

    xc = x - x.mean(0, keepdims=True)
    cov = np.einsum("tni,tnj->nij", xc, xc) / T
    evals, evecs = np.linalg.eigh(cov)                                  # ascending
    pdir = canon_sign(evecs[:, :, -1])
    ev = np.clip(evals, 0, None)
    tot = ev.sum(1) + 1e-12
    ratios = np.stack([ev[:, -1] / tot, ev[:, -2] / tot], 1)

    onehot = np.zeros((Nt, 4), np.float32)
    onehot[np.arange(Nt), np.clip(mtypes, -1, 2) + 1] = 1

    is_rev = (mtypes == 2)[:, None]
    is_pris = (mtypes == 1)[:, None]
    rev_dir_n = canon_sign(frame.dir(rev_dir)) * is_rev
    rev_org_n = frame.pts(rev_origin) * is_rev
    pris_dir_n = canon_sign(frame.dir(pris_dir)) * is_pris
    vis_ratio = vis.mean(0)[:, None]

    desc = np.concatenate([
        disp, vis_k,
        max_disp[:, None], path_len[:, None], mean_speed[:, None],
        pdir, ratios, onehot, rev_dir_n, rev_org_n, pris_dir_n, vis_ratio], axis=1).astype(np.float32)
    assert desc.shape[1] == 54, desc.shape
    return np.nan_to_num(desc)


def load_filtered_tracks(scene_dir):
    d = np.load(os.path.join(scene_dir, "filtered.npz"))
    return d["coords"].astype(np.float32), d["visibs"].astype(bool)


def track_geo_feats(scene_dir, xyz_n, frame, k_frames=8, knn_k=4, valid_thr=0.02, cache=True):
    coords_w, visibs = load_filtered_tracks(scene_dir)
    cache_path = os.path.join(scene_dir, "pat_extra", "track_motion_cache.npz") if cache else None
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mtypes, rev_dir, rev_origin, pris_dir = classify_tracks(coords_w, visibs, cache_path)
    desc = track_descriptors(coords_w, visibs, frame, mtypes, rev_dir, rev_origin, pris_dir, k_frames)
    idx, w, dmin = build_knn(xyz_n, frame.pts(coords_w[0]), k=knn_k)
    valid = dmin < valid_thr
    conf = np.exp(-(dmin / valid_thr) ** 2).astype(np.float32)
    feats = transfer(desc, idx, w) * valid[:, None]
    out = np.concatenate([feats, conf[:, None], valid[:, None].astype(np.float32)], axis=1)
    assert out.shape[1] == EXTRA_FEAT_DIMS["track_geo"], out.shape
    return out.astype(np.float32), dict(mtypes=mtypes, knn=(idx, w, dmin), coords_w=coords_w)


# ----------------------------------------------------------------------------- track_tapip
def track_tapip_feats(scene_dir, xyz_n, frame, knn_k=4, valid_thr=0.02):
    path = os.path.join(scene_dir, "pat_extra", "tapip3d_feats.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run data_tools/extract_tapip3d_feats.py first")
    d = np.load(path)
    coords0 = d["coords"][0].astype(np.float32)                         # (Nt, 3) world, first dynamic frame
    feats = d["track_feats"].astype(np.float32)                         # (Nt, 384)
    ok = np.isfinite(feats).all(1) & np.isfinite(coords0).all(1)
    coords0, feats = coords0[ok], feats[ok]
    idx, w, dmin = build_knn(xyz_n, frame.pts(coords0), k=knn_k)
    valid = dmin < valid_thr
    out = transfer(feats, idx, w) * valid[:, None]
    assert out.shape[1] == EXTRA_FEAT_DIMS["track_tapip"], out.shape
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- vggt
def vggt_feats(scene_dir, num_points):
    path = os.path.join(scene_dir, "pat_extra", "vggt128.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run data_tools/extract_vggt_feats.py first")
    f = np.load(path).astype(np.float32)
    assert f.shape == (num_points, EXTRA_FEAT_DIMS["vggt"]), (f.shape, num_points)
    return np.nan_to_num(f)


# ----------------------------------------------------------------------------- public API
def compute_scene_extra_feats(scene_dir, xyz_world, frame, names, k_frames=8, knn_k=4, valid_thr=0.02):
    """Return {name: (N, D) float32} for ALL points of xyz_world (world frame, ply order)."""
    names = list(names)
    xyz_n = frame.pts(xyz_world)
    out, aux = {}, {}
    if "track_geo" in names:
        out["track_geo"], aux["track_geo"] = track_geo_feats(scene_dir, xyz_n, frame, k_frames, knn_k, valid_thr)
    if "track_tapip" in names:
        out["track_tapip"] = track_tapip_feats(scene_dir, xyz_n, frame, knn_k, valid_thr)
    if "vggt" in names:
        out["vggt"] = vggt_feats(scene_dir, len(xyz_world))
    for n in names:
        assert out[n].shape == (len(xyz_world), EXTRA_FEAT_DIMS[n]), (n, out[n].shape)
    return out, aux


def track_part_labels(scene_dir, xyz_world, frame, moving_joints, knn_k=4, valid_thr=0.02):
    """Pseudo part labels from the tracks: 0 = static base, k = k-th moving joint of
    `moving_joints` (json order). A point is 'moving' when the inverse-distance vote of
    its k nearest tracks is dominated by prismatic/revolute tracks; it is then assigned to
    the nearest moving joint centre whose type matches the dominant track motion type.
    Points without a nearby track get label -1 (caller decides the fallback)."""
    coords_w, visibs = load_filtered_tracks(scene_dir)
    cache_path = os.path.join(scene_dir, "pat_extra", "track_motion_cache.npz")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mtypes, _, _, _ = classify_tracks(coords_w, visibs, cache_path)
    xyz_n = frame.pts(xyz_world)
    idx, w, dmin = build_knn(xyz_n, frame.pts(coords_w[0]), k=knn_k)
    valid = dmin < valid_thr
    vote_p = (w * (mtypes[idx] == 1)).sum(1)
    vote_r = (w * (mtypes[idx] == 2)).sum(1)
    moving = valid & ((vote_p + vote_r) > 0.5)
    labels = np.where(valid, 0, -1).astype(np.int64)
    if not moving.any() or not moving_joints:
        return labels
    centers = np.array([j["center"] for j in moving_joints], dtype=np.float32)
    types = np.array([j["joint_type"] for j in moving_joints])
    dist = np.linalg.norm(xyz_world[moving][:, None] - centers[None], axis=-1)     # (M, J)
    want_r = (vote_r > vote_p)[moving]
    compat = np.where(want_r[:, None], types[None] == "r", types[None] == "p")
    dist_c = np.where(compat, dist, dist + 1e6)                                       # prefer type-compatible joints
    labels[moving] = dist_c.argmin(1) + 1
    return labels
