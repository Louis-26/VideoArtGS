#!/usr/bin/env python3
"""Verify regenerated REALSCAN preprocessing artifacts against shipped ones.

Covers the six realscan artifacts, in pipeline order:
    masks/               SAM2 + human prompts       -> IoU-level match
    data.npz             ST2 + process_vggt bundle  -> per-key comparison,
                         poses via gauge-free relative motion
    point_cloud.ply      fused from predicted depth -> NN, gauge-aware
    joint_infos_vlm.json GPT-4o prior               -> semantic (K + types)
    joint_infos.json     motion-analysis prior      -> per-joint, gauge-aware
    filtered.npz         TAPIP3D tracks (filtered)  -> distribution, gauge-aware

Gauge note: the realscan world frame is defined by pca_align on the
reconstructed mesh (process_vggt.py L156), so two independent runs may differ
by a global rigid transform (PCA rotation / sign flips). Geometry comparators
therefore retry after a PCA-initialized rigid ICP alignment when the raw
comparison fails, and report the estimated gauge offset.

Usage:
    python compare_files_vag_rs.py \
        --shipped data/videoartgs/realscan/coffeemachine_2r \
        --rerun   new_data/videoartgs/realscan/coffeemachine_2r

Verdict levels:
    ✅ [BYTE-IDENTICAL]     md5 equal.
    ✅ [SEMANTIC-IDENTICAL] downstream-consumed fields equal.
    ✅ [CONSISTENT]         within expected stochastic variation; may carry a
                            "(after rigid gauge alignment)" note.
    ❌ [DIVERGENT]          differences that change pipeline behavior.
"""
import os
import json
import hashlib
import argparse
import itertools
import numpy as np
from glob import glob
from collections import Counter
from plyfile import PlyData
from scipy.spatial import cKDTree

VLM_TYPE_MAP = {"hinge": "r", "slider": "p"}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def both_exist(pa, pb, label, is_dir=False):
    ok = True
    check = os.path.isdir if is_dir else os.path.exists
    for tag, p in (("shipped", pa), ("rerun", pb)):
        if not check(p):
            print(f"  ⚠️  [skip] {label}: {tag} missing ({p})")
            ok = False
    return ok


def _unit(v):
    v = np.asarray(v, np.float64).reshape(3)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def axis_angle_deg(da, db):
    cos = np.clip(np.dot(_unit(da), _unit(db)), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return min(ang, 180.0 - ang)


def rot_geodesic_deg(Ra, Rb):
    cos = np.clip((np.trace(Ra @ Rb.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def line_dist_cm(o1, d1, o2, d2):
    d1, d2 = _unit(d1), _unit(d2)
    w = np.asarray(o2, np.float64) - np.asarray(o1, np.float64)
    cross = np.cross(d1, d2)
    n = np.linalg.norm(cross)
    if n < 1e-8:
        return float(np.linalg.norm(w - np.dot(w, d1) * d1)) * 100.0
    return float(abs(np.dot(w, cross)) / n) * 100.0


# ------------------------------------------------- rigid gauge estimation --

def icp_rigid(B, A, R0=None, t0=None, iters=8, cap=3000):
    """Refine a rigid (R, t) mapping cloud B onto cloud A (no scale)."""
    B = B[:: max(1, len(B) // cap)].astype(np.float64)
    A = A[:: max(1, len(A) // cap)].astype(np.float64)
    tree = cKDTree(A)
    R = np.eye(3) if R0 is None else np.asarray(R0, np.float64)
    t = np.zeros(3) if t0 is None else np.asarray(t0, np.float64)
    for _ in range(iters):
        Bt = B @ R.T + t
        corr = A[tree.query(Bt)[1]]
        cb, ca = Bt.mean(0), corr.mean(0)
        H = (Bt - cb).T @ (corr - ca)
        U, _, Vt = np.linalg.svd(H)
        Rd = Vt.T @ U.T
        if np.linalg.det(Rd) < 0:
            Vt[-1] *= -1
            Rd = Vt.T @ U.T
        R = Rd @ R
        t = Rd @ t + (ca - Rd @ cb)
    return R, t


def _pca_frame(X):
    c = X.mean(0)
    w, V = np.linalg.eigh((X - c).T @ (X - c) / len(X))
    V = V[:, ::-1]                       # descending principal axes
    if np.linalg.det(V) < 0:
        V[:, -1] *= -1
    return c, V


def estimate_gauge(B, A, cap=3000):
    """Estimate the global rigid gauge mapping B onto A. Initialized with
    PCA principal-axis alignment (4 proper sign combinations, since PCA axis
    signs are arbitrary), refined with ICP. Returns (R, t, mean_nn)."""
    Bs = B[:: max(1, len(B) // cap)].astype(np.float64)
    As = A[:: max(1, len(A) // cap)].astype(np.float64)
    ca, Va = _pca_frame(As)
    cb, Vb = _pca_frame(Bs)
    best = (np.eye(3), np.zeros(3), np.inf)
    for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        R0 = Va @ np.diag(s) @ Vb.T
        t0 = ca - R0 @ cb
        R, t = icp_rigid(Bs, As, R0, t0, iters=6)
        Bt = Bs @ R.T + t
        score = max(cKDTree(As).query(Bt)[0].mean(),
                    cKDTree(Bt).query(As)[0].mean())
        if score < best[2]:
            best = (R, t, score)
    return best


def nn_stats(A, B):
    d_ab = cKDTree(B).query(A)[0]
    d_ba = cKDTree(A).query(B)[0]
    return d_ab, d_ba


# -------------------------------------------------------------- masks/ -----

def compare_masks(da_dir, db_dir, tol_iou):
    print("=" * 62)
    print("masks/  (SAM2 + human prompts: IoU-level match expected)")
    print(f"  shipped: {da_dir}\n  rerun  : {db_dir}")
    if not both_exist(da_dir, db_dir, "masks/", is_dir=True):
        return
    fa = sorted(glob(os.path.join(da_dir, "*.npy")))
    fb = sorted(glob(os.path.join(db_dir, "*.npy")))
    print(f"  files: shipped={len(fa)}, rerun={len(fb)}")
    if len(fa) != len(fb) or len(fa) == 0:
        print("  ❌ [DIVERGENT] frame count differs or empty")
        return
    n = len(fa)
    sample = sorted(set([0, n // 2, n - 1] + list(range(0, n, max(1, n // 10)))))
    ious = []
    for i in sample:
        ma = np.load(fa[i]).squeeze() > 0.5
        mb = np.load(fb[i]).squeeze() > 0.5
        if ma.shape != mb.shape:
            print(f"  ❌ [DIVERGENT] frame {i}: shape {ma.shape} vs {mb.shape}")
            return
        if ma.ndim == 2:
            ma, mb = ma[None], mb[None]
        for k in range(ma.shape[0]):
            u = np.logical_or(ma[k], mb[k]).sum()
            if u:
                ious.append(np.logical_and(ma[k], mb[k]).sum() / u)
    ious = np.array(ious)
    print(f"  IoU over {len(sample)} sampled frames: "
          f"mean = {ious.mean():.3f}, min = {ious.min():.3f}")
    if ious.mean() >= tol_iou:
        print(f"  ✅ [CONSISTENT] mean IoU >= {tol_iou}; residuals are "
              "SAM2 prompt/version differences")
    else:
        print(f"  ❌ [DIVERGENT] mean IoU below {tol_iou}; re-annotate or "
              "check object/hand channel order")


# ------------------------------------------------------------- data.npz ----

def compare_data_npz(pa, pb, tol_rot, tol_trans, tol_depth, tol_iou):
    print("=" * 62)
    print("data.npz  (realscan bundle: per-key comparison; poses compared "
          "via gauge-free relative motion)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "data.npz"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match.")
        return
    da, db = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    for k in ("video", "depths", "intrinsics", "extrinsics", "masks", "poses"):
        if k in da and k in db and da[k].shape != db[k].shape:
            print(f"  ❌ [DIVERGENT] key '{k}' shape {da[k].shape} vs {db[k].shape}")
            return
    ok = True

    v = np.abs(da["video"].astype(np.int16) - db["video"].astype(np.int16))
    print(f"  video: max |diff| = {v.max()}, mean = {v.mean():.4f} (uint8) "
          f"{'-> same source frames' if v.max() <= 2 else ''}")
    ok &= v.max() <= 2
    print(f"  intrinsics: max |diff| = "
          f"{np.abs(da['intrinsics'] - db['intrinsics']).max():.3e}")

    Pa, Pb = da["poses"], db["poses"]
    g0 = rot_geodesic_deg(Pa[0, :3, :3], Pb[0, :3, :3])
    rel_r, rel_t = [], []
    for i in range(len(Pa) - 1):
        dA = np.linalg.inv(Pa[i]) @ Pa[i + 1]
        dB = np.linalg.inv(Pb[i]) @ Pb[i + 1]
        rel_r.append(rot_geodesic_deg(dA[:3, :3], dB[:3, :3]))
        rel_t.append(np.linalg.norm(dA[:3, 3] - dB[:3, 3]) * 100)
    rel_r, rel_t = np.array(rel_r), np.array(rel_t)
    print(f"  poses (relative motion): rot diff mean/max = "
          f"{rel_r.mean():.3f}/{rel_r.max():.3f} deg, "
          f"trans diff mean/max = {rel_t.mean():.3f}/{rel_t.max():.3f} cm  "
          f"| global gauge offset = {g0:.2f} deg")
    ok &= rel_r.max() <= tol_rot and rel_t.max() <= tol_trans

    Da, Db = da["depths"], db["depths"]
    both = (Da > 0) & (Db > 0)
    rel = np.abs(Da[both] - Db[both]) / np.clip(Da[both], 1e-6, None)
    pos_diff = abs(int((Da > 0).sum()) - int((Db > 0).sum())) / max((Da > 0).sum(), 1)
    print(f"  depths: mean rel diff on joint-valid pixels = {rel.mean()*100:.2f}%, "
          f"valid-pixel count rel diff = {pos_diff*100:.2f}%")
    ok &= rel.mean() <= tol_depth

    Ma, Mb = da["masks"] > 0.5, db["masks"] > 0.5
    n = Ma.shape[0]
    ious = []
    for i in range(0, n, max(1, n // 10)):
        for k in range(Ma.shape[1]):
            u = np.logical_or(Ma[i, k], Mb[i, k]).sum()
            if u:
                ious.append(np.logical_and(Ma[i, k], Mb[i, k]).sum() / u)
    ious = np.array(ious)
    print(f"  masks (in bundle): mean IoU = {ious.mean():.3f}")
    ok &= ious.mean() >= tol_iou

    if ok:
        print(f"  ✅ [CONSISTENT] all keys within tolerances (rot <= {tol_rot} "
              f"deg, trans <= {tol_trans} cm, depth <= {tol_depth*100:.0f}%, "
              f"IoU >= {tol_iou})")
    else:
        print("  ❌ [DIVERGENT] one or more keys exceed tolerances (see above); "
              "check ST2 checkpoint parity and mask annotation")


# ----------------------------------------------------------- point cloud ---

def load_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    return xyz


def compare_pointclouds(pa, pb, tol_gauge_cm=0.5):
    """Returns the estimated gauge (R, t) when a rigid alignment succeeded,
    so joint/track comparators can reuse it. None otherwise."""
    print("=" * 62)
    print("point_cloud.ply  (fused from predicted depth/poses: gauge-aware)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "point_cloud.ply"):
        return None
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match.")
        return None

    A = load_ply(pa)
    B = load_ply(pb)
    print(f"  points: shipped={len(A)}, rerun={len(B)}"
          f"  (relative diff {abs(len(A) - len(B)) / len(A) * 100:.3f}%)")
    d_ab, d_ba = nn_stats(A, B)
    print(f"  raw     A->B mean NN = {d_ab.mean()*100:.3f} cm, "
          f"B->A = {d_ba.mean()*100:.3f} cm")
    if d_ab.mean() * 100 <= tol_gauge_cm and d_ba.mean() * 100 <= tol_gauge_cm:
        print("  ✅ [CONSISTENT] same frame, sub-tolerance geometry")
        return None

    R, t, _ = estimate_gauge(B, A)
    Bt = B @ R.T + t
    d_ab2, d_ba2 = nn_stats(A, Bt)
    print(f"  aligned A->B mean NN = {d_ab2.mean()*100:.3f} cm, "
          f"B->A = {d_ba2.mean()*100:.3f} cm  "
          f"(gauge: rot {rot_geodesic_deg(R, np.eye(3)):.2f} deg, "
          f"trans {np.linalg.norm(t)*100:.2f} cm)")
    if d_ab2.mean() * 100 <= tol_gauge_cm and d_ba2.mean() * 100 <= tol_gauge_cm:
        print("  ✅ [CONSISTENT] geometry matches after rigid gauge alignment "
              "-> PCA-frame difference, not a content difference")
        return (R, t)
    print("  ❌ [DIVERGENT] geometry differs even after rigid alignment; "
          "check model checkpoint / canonical frame set")
    return None


# ---------------------------------------------------- joint_infos_vlm ------

def _vlm_parse(path):
    raw = json.load(open(path))
    movable = [e for e in raw if e.get("joint") in VLM_TYPE_MAP]
    types = [VLM_TYPE_MAP[e["joint"]] for e in movable]
    names = [e.get("name") for e in movable]
    pairs = [(e["joint"], e.get("name")) for e in movable]
    base = [e.get("joint") for e in raw if e.get("joint") not in VLM_TYPE_MAP]
    return types, names, pairs, base


def compare_vlm_joint_infos(pa, pb):
    print("=" * 62)
    print("joint_infos_vlm.json  (unordered; consumed = K + type multiset)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "joint_infos_vlm.json"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match.")
        return
    ta, na, qa, ba = _vlm_parse(pa)
    tb, nb, qb, bb = _vlm_parse(pb)
    print(f"  consumed prior  shipped: K={len(ta)}, "
          f"type counts={dict(sorted(Counter(ta).items()))}")
    print(f"  consumed prior  rerun  : K={len(tb)}, "
          f"type counts={dict(sorted(Counter(tb).items()))}")
    if len(ta) != len(tb) or Counter(ta) != Counter(tb):
        print("  ❌ [DIVERGENT] K or type multiset differs -> Step 3 changes")
        return
    notes = []
    if Counter(qa) == Counter(qb):
        if qa != qb:
            notes.append("entry ordering differs (unordered file)")
    else:
        oa = list((Counter(na) - Counter(nb)).elements())
        ob = list((Counter(nb) - Counter(na)).elements())
        notes.append(f"part naming drift: shipped-only={oa}, rerun-only={ob}"
                     if oa or ob else "joint-name pairing differs (not consumed)")
    if Counter(ba) != Counter(bb):
        notes.append(f"base label differs ({ba} vs {bb})")
    print("  ✅ [SEMANTIC-IDENTICAL] downstream-consumed prior identical")
    if notes:
        print("  ignored-field diffs: " + "; ".join(notes))


# ------------------------------------------- joint_infos (motion analysis) --

def _load_joint_infos(path):
    info = json.load(open(path))
    static = [e for e in info if e.get("joint_type") == "s"]
    movable = [e for e in info if e.get("joint_type") in ("r", "p")]
    return static, movable


def _apply_gauge(joints, gauge):
    R, t = gauge
    out = []
    for e in joints:
        e = dict(e)
        e["direction"] = list(R @ np.asarray(e["direction"], np.float64))
        e["origin"] = list(R @ np.asarray(e["origin"], np.float64) + t)
        e["center"] = list(R @ np.asarray(e["center"], np.float64) + t)
        out.append(e)
    return out


def _pairwise_report(ma, mb, tol_ang, tol_pos, tag=""):
    ok = True
    for jt in ("r", "p"):
        ia = [i for i, e in enumerate(ma) if e["joint_type"] == jt]
        ib = [i for i, e in enumerate(mb) if e["joint_type"] == jt]
        if not ia:
            continue
        best, best_cost = None, np.inf
        for perm in itertools.permutations(ib, len(ia)):
            cost = sum(axis_angle_deg(ma[i]["direction"], mb[j]["direction"]) +
                       (line_dist_cm(ma[i]["origin"], ma[i]["direction"],
                                     mb[j]["origin"], mb[j]["direction"])
                        if jt == "r" else
                        np.linalg.norm(np.subtract(ma[i]["center"],
                                                   mb[j]["center"])) * 100)
                       for i, j in zip(ia, perm))
            if cost < best_cost:
                best_cost, best = cost, list(zip(ia, perm))
        for i, j in best:
            ang = axis_angle_deg(ma[i]["direction"], mb[j]["direction"])
            pos = (line_dist_cm(ma[i]["origin"], ma[i]["direction"],
                                mb[j]["origin"], mb[j]["direction"])
                   if jt == "r" else 0.0)
            cd = np.linalg.norm(np.subtract(ma[i]["center"],
                                            mb[j]["center"])) * 100
            extra = f", origin line dist = {pos:.2f} cm" if jt == "r" else ""
            print(f"  {tag}{jt}-joint shipped#{i} <-> rerun#{j}: "
                  f"axis angle = {ang:.2f} deg{extra}, center dist = {cd:.2f} cm")
            ok &= ang <= tol_ang and pos <= tol_pos
    return ok


def compare_joint_infos(pa, pb, tol_ang, tol_pos, gauge=None):
    print("=" * 62)
    print("joint_infos.json  (Step-3 product: statistical consistency, "
          "gauge-aware)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "joint_infos.json"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match.")
        return
    sa, ma = _load_joint_infos(pa)
    sb, mb = _load_joint_infos(pb)
    ca_ = Counter(e["joint_type"] for e in ma)
    cb_ = Counter(e["joint_type"] for e in mb)
    print(f"  movable joints  shipped: {dict(sorted(ca_.items()))},  "
          f"rerun: {dict(sorted(cb_.items()))}")
    if ca_ != cb_:
        print("  ❌ [DIVERGENT] per-type joint counts differ; check the vlm prior")
        return
    if _pairwise_report(ma, mb, tol_ang, tol_pos):
        print(f"  ✅ [CONSISTENT] within tolerances (angle <= {tol_ang} deg, "
              f"r-origin <= {tol_pos} cm)")
        return
    if gauge is not None:
        print("  raw comparison exceeded tolerances; retrying after the rigid "
              "gauge alignment estimated from point_cloud.ply:")
        if _pairwise_report(ma, _apply_gauge(mb, gauge), tol_ang, tol_pos,
                            tag="aligned "):
            print("  ✅ [CONSISTENT] (after rigid gauge alignment) -> PCA-frame "
                  "difference, not a content difference")
            return
    print("  ❌ [DIVERGENT] joints differ beyond tolerances; inspect tracks "
          "and cluster assignment for this scene")


# --------------------------------------------------------- filtered.npz ----

def _frame_cloud(coords, visibs, f, cap=4000):
    pts = coords[f][visibs[f] > 0.5]
    return pts[:: max(1, len(pts) // cap)] if len(pts) > cap else pts


def compare_filtered(pa, pb, tol_nn_cm, tol_count=0.20, gauge=None):
    print("=" * 62)
    print("filtered.npz  (Step-3 product: distribution consistency, "
          "gauge-aware)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "filtered.npz"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match.")
        return
    da, db = np.load(pa), np.load(pb)
    ca, va = da["coords"], da["visibs"]
    cb, vb = db["coords"], db["visibs"]
    print(f"  coords: shipped {ca.shape}, rerun {cb.shape}  |  "
          f"mean visibility: {va.mean():.3f} vs {vb.mean():.3f}")
    if ca.shape[0] != cb.shape[0]:
        print("  ❌ [DIVERGENT] frame count differs -> wrong n_canonical")
        return
    rel_n = abs(ca.shape[1] - cb.shape[1]) / max(ca.shape[1], 1)
    T = ca.shape[0]

    def worst_nn(transform=None):
        worst = 0.0
        for f in (0, T // 2, T - 1):
            A = _frame_cloud(ca, va, f)
            B = _frame_cloud(cb, vb, f)
            if transform is not None:
                R, t = transform
                B = B @ R.T + t
            if len(A) == 0 or len(B) == 0:
                continue
            d_ab, d_ba = nn_stats(A, B)
            m = max(d_ab.mean(), d_ba.mean()) * 100
            worst = max(worst, m)
            print(f"  frame {f:3d}: bidirectional mean NN = {m:.3f} cm "
                  f"({len(A)} vs {len(B)} visible pts)"
                  + ("" if transform is None else "  [aligned]"))
        return worst

    worst = worst_nn()
    if rel_n <= tol_count and worst <= tol_nn_cm:
        print(f"  ✅ [CONSISTENT] track count within {tol_count*100:.0f}% and "
              f"per-frame mean NN <= {tol_nn_cm} cm")
        return
    if gauge is None:  # self-estimate from the first-frame clouds
        A0, B0 = _frame_cloud(ca, va, 0), _frame_cloud(cb, vb, 0)
        if len(A0) > 10 and len(B0) > 10:
            R, t, _ = estimate_gauge(B0, A0)
            gauge = (R, t)
    if gauge is not None:
        print(f"  raw comparison exceeded tolerances; retrying with rigid "
              f"gauge alignment (rot "
              f"{rot_geodesic_deg(gauge[0], np.eye(3)):.2f} deg):")
        if rel_n <= tol_count and worst_nn(gauge) <= tol_nn_cm:
            print("  ✅ [CONSISTENT] (after rigid gauge alignment)")
            return
    print(f"  ❌ [DIVERGENT] rel track diff {rel_n*100:.1f}% or NN beyond "
          f"{tol_nn_cm} cm even after alignment")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped",
                    default="data/videoartgs/realscan/coffeemachine_2r",
                    help="official shipped scene directory")
    ap.add_argument("--rerun",
                    default="new_data/videoartgs/realscan/coffeemachine_2r",
                    help="scene directory containing your regenerated artifacts")
    ap.add_argument("--tol-ang", type=float, default=5.0)
    ap.add_argument("--tol-pos", type=float, default=5.0)
    ap.add_argument("--tol-nn", type=float, default=2.0)
    ap.add_argument("--tol-iou", type=float, default=0.90)
    ap.add_argument("--tol-rot", type=float, default=1.0,
                    help="data.npz: max relative-motion rotation diff (deg)")
    ap.add_argument("--tol-trans", type=float, default=2.0,
                    help="data.npz: max relative-motion translation diff (cm)")
    ap.add_argument("--tol-depth", type=float, default=0.05,
                    help="data.npz: max mean relative depth diff (fraction)")
    args = ap.parse_args()

    j = os.path.join
    compare_masks(j(args.shipped, "masks"), j(args.rerun, "masks"),
                  args.tol_iou)
    compare_data_npz(j(args.shipped, "data.npz"), j(args.rerun, "data.npz"),
                     args.tol_rot, args.tol_trans, args.tol_depth, args.tol_iou)
    gauge = compare_pointclouds(j(args.shipped, "point_cloud.ply"),
                                j(args.rerun, "point_cloud.ply"))
    compare_vlm_joint_infos(j(args.shipped, "joint_infos_vlm.json"),
                            j(args.rerun, "joint_infos_vlm.json"))
    compare_joint_infos(j(args.shipped, "joint_infos.json"),
                        j(args.rerun, "joint_infos.json"),
                        args.tol_ang, args.tol_pos, gauge=gauge)
    compare_filtered(j(args.shipped, "filtered.npz"),
                     j(args.rerun, "filtered.npz"), args.tol_nn, gauge=gauge)