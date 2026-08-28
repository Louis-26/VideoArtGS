#!/usr/bin/env python3
"""Verify regenerated preprocessing artifacts against the shipped ones.

Compares, per scene directory (pipeline order):
    Step 1: transforms.json        (exact / numeric)
            point_cloud.ply        (bidirectional nearest-neighbor + color)
    Step 2: joint_infos_vlm.json   (semantic + order-free)
    Step 3: joint_infos.json       (per-joint statistical consistency)
            filtered.npz           (trajectory distribution consistency)

Step-3 artifacts come from GPU inference (TAPIP3D) + clustering, so
byte/numeric identity is NOT expected there; the pass tier is
statistical consistency within tolerances.

Usage:
    python verify_preprocess.py \
        --shipped data/videoartgs/sapien/168 \
        --rerun   new_data/videoartgs/sapien/168

Verdict levels:
    ✅ [BYTE-IDENTICAL]     md5 equal -> full byte-for-byte reproduction.
    ✅ [NUMERIC-IDENTICAL]  equal up to floating-point noise.
    ✅ [SEMANTIC-IDENTICAL] downstream-consumed fields equal.
    ✅ [CONSISTENT]         within expected stochastic variation (Step 3).
    🟡 [NEAR-IDENTICAL]     sub-voxel drift across library versions.
    ❌ [DIVERGENT]          differences that change pipeline behavior.
"""
import os
import json
import hashlib
import argparse
import itertools
import numpy as np
from collections import Counter
from plyfile import PlyData
from scipy.spatial import cKDTree

SCALAR_FIELDS = ["camera_angle_x", "camera_angle_y", "focal_x", "focal_y",
                 "cx", "cy", "w", "h"]

# Mirror utils/metrics.read_joint_infos_vlm: only hinge/slider entries are
# consumed downstream. Entry order / names / base label are never read.
VLM_TYPE_MAP = {"hinge": "r", "slider": "p"}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def both_exist(pa, pb, label):
    ok = True
    for tag, p in (("shipped", pa), ("rerun", pb)):
        if not os.path.exists(p):
            print(f"  ⚠️  [skip] {label}: {tag} file missing ({p})")
            ok = False
    return ok


def _unit(v):
    v = np.asarray(v, np.float64).reshape(3)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def axis_angle_deg(da, db):
    """Direction-agnostic angle between two axis directions, degrees."""
    cos = np.clip(np.dot(_unit(da), _unit(db)), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return min(ang, 180.0 - ang)


def line_dist_cm(o1, d1, o2, d2):
    """Common-perpendicular distance between two 3D lines, centimeters."""
    d1, d2 = _unit(d1), _unit(d2)
    w = np.asarray(o2, np.float64) - np.asarray(o1, np.float64)
    cross = np.cross(d1, d2)
    n = np.linalg.norm(cross)
    if n < 1e-8:
        return float(np.linalg.norm(w - np.dot(w, d1) * d1)) * 100.0
    return float(abs(np.dot(w, cross)) / n) * 100.0


# ------------------------------------------------------------ transforms ---

def compare_transforms(pa, pb):
    print("=" * 62)
    print("transforms.json")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "transforms.json"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match: byte-for-byte reproduction, "
              "nothing more to check.")
        return

    a, b = json.load(open(pa)), json.load(open(pb))
    for k in SCALAR_FIELDS:
        if not np.isclose(a[k], b[k], rtol=0, atol=1e-9):
            print(f"  ❌ [DIVERGENT] intrinsics field {k}: {a[k]} vs {b[k]}")
            return
    if len(a["frames"]) != len(b["frames"]):
        print(f"  ❌ [DIVERGENT] frame count differs: "
              f"{len(a['frames'])} vs {len(b['frames'])}")
        return
    max_pose_diff, max_time_diff = 0.0, 0.0
    for i, (fa, fb) in enumerate(zip(a["frames"], b["frames"])):
        if fa["file_path"] != fb["file_path"] or fa["state"] != fb["state"]:
            print(f"  ❌ [DIVERGENT] frame {i} file_path/state mismatch")
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
        print("  ✅ [NUMERIC-IDENTICAL] semantically identical "
              "(md5 mismatch comes from serialization format only)")
    else:
        print("  ❌ [DIVERGENT] numeric differences; check that camera.json "
              "is the same source and num_cano == 150")


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
    if not both_exist(pa, pb, "point_cloud.ply"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match: byte-for-byte reproduction.")
        return

    A, ca = load_ply(pa)
    B, cb = load_ply(pb)
    print(f"  points: shipped={len(A)}, rerun={len(B)}"
          f"  (relative diff {abs(len(A) - len(B)) / len(A) * 100:.3f}%)")
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
        print("  ✅ [NUMERIC-IDENTICAL] same point set; differences are float "
              "noise -> same algorithm + same inputs")
    elif d_ab.max() < 0.01 and d_ba.max() < 0.01:
        print("  🟡 [NEAR-IDENTICAL] sub-voxel differences -> same algorithm; "
              "library-version drift. Acceptable.")
    else:
        print("  ❌ [DIVERGENT] structural differences above the voxel scale; "
              "check num_cano / eps / the input frame set")


# ---------------------------------------------------- joint_infos_vlm ------

def _vlm_parse(path):
    raw = json.load(open(path))
    movable = [e for e in raw if e.get("joint") in VLM_TYPE_MAP]
    types = [VLM_TYPE_MAP[e["joint"]] for e in movable]
    names = [e.get("name") for e in movable]
    pairs = [(e["joint"], e.get("name")) for e in movable]
    base = [e.get("joint") for e in raw if e.get("joint") not in VLM_TYPE_MAP]
    return types, names, pairs, base


def _fmt_counts(items):
    return dict(sorted(Counter(items).items()))


def compare_vlm_joint_infos(pa, pb):
    print("=" * 62)
    print("joint_infos_vlm.json  (unordered registry; consumed = K + type multiset)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "joint_infos_vlm.json"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match: byte-for-byte reproduction.")
        return

    ta, na, qa, ba = _vlm_parse(pa)
    tb, nb, qb, bb = _vlm_parse(pb)
    print(f"  consumed prior  shipped: K={len(ta)}, type counts={_fmt_counts(ta)}")
    print(f"  consumed prior  rerun  : K={len(tb)}, type counts={_fmt_counts(tb)}")

    if len(ta) != len(tb) or Counter(ta) != Counter(tb):
        print("  ❌ [DIVERGENT] K or the joint-type multiset differs -> Step 3 "
              "behavior changes (nq = 4 + K//2, per-type cluster counts)")
        return

    notes = []
    if Counter(qa) == Counter(qb):
        if qa != qb:
            notes.append("entry ordering differs (unordered file, content identical)")
    else:
        only_a = list((Counter(na) - Counter(nb)).elements())
        only_b = list((Counter(nb) - Counter(na)).elements())
        if only_a or only_b:
            notes.append(f"part naming drift: shipped-only={only_a}, "
                         f"rerun-only={only_b}")
        else:
            notes.append("joint-name pairing differs (same names, same types, "
                         "different association)")
    if Counter(ba) != Counter(bb):
        notes.append(f"base label differs ({ba} vs {bb})")

    print("  ✅ [SEMANTIC-IDENTICAL] downstream-consumed prior identical "
          "(nq, cluster counts, model slots all unchanged)")
    if notes:
        print("  ignored-field diffs: " + "; ".join(notes))


# ------------------------------------------- joint_infos (motion analysis) --

def _load_joint_infos(path):
    """Schema (motion_analysis.py L831-865): entry 0 = static slot ('s'),
    then prismatic entries, then revolute entries."""
    info = json.load(open(path))
    static = [e for e in info if e.get("joint_type") == "s"]
    movable = [e for e in info if e.get("joint_type") in ("r", "p")]
    return static, movable


def _match_by_type(ma, mb):
    """Match same-type joints (brute force; counts are tiny) minimizing
    axis angle + (line distance for 'r' / center distance for 'p')."""
    pairs = []
    for jt in ("r", "p"):
        ia = [i for i, e in enumerate(ma) if e["joint_type"] == jt]
        ib = [i for i, e in enumerate(mb) if e["joint_type"] == jt]
        if not ia:
            continue
        best, best_cost = None, np.inf
        for perm in itertools.permutations(ib, len(ia)):
            cost = 0.0
            for i, j in zip(ia, perm):
                cost += axis_angle_deg(ma[i]["direction"], mb[j]["direction"])
                if jt == "r":
                    cost += line_dist_cm(ma[i]["origin"], ma[i]["direction"],
                                         mb[j]["origin"], mb[j]["direction"])
                else:
                    cost += np.linalg.norm(np.subtract(ma[i]["center"],
                                                       mb[j]["center"])) * 100
            if cost < best_cost:
                best_cost, best = cost, list(zip(ia, perm))
        pairs += best
    return pairs


def compare_joint_infos(pa, pb, tol_ang, tol_pos):
    print("=" * 62)
    print("joint_infos.json  (Step-3 product: statistical consistency, "
          "byte identity not expected)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "joint_infos.json"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match (unusual for a stochastic "
              "product, but fine).")
        return

    sa, ma = _load_joint_infos(pa)
    sb, mb = _load_joint_infos(pb)
    ca_, cb_ = Counter(e["joint_type"] for e in ma), \
               Counter(e["joint_type"] for e in mb)
    print(f"  movable joints  shipped: {dict(sorted(ca_.items()))},  "
          f"rerun: {dict(sorted(cb_.items()))}")
    if ca_ != cb_:
        print("  ❌ [DIVERGENT] per-type joint counts differ -> clustering "
              "found different structure; check joint_infos_vlm.json first")
        return

    if sa and sb:
        c_dist = np.linalg.norm(np.subtract(sa[0]["center"],
                                            sb[0]["center"])) * 100
        dm_rel = abs(sa[0]["dist_max"] - sb[0]["dist_max"]) / \
            max(sa[0]["dist_max"], 1e-9) * 100
        print(f"  static slot: center dist = {c_dist:.2f} cm, "
              f"dist_max rel diff = {dm_rel:.1f}%")

    worst_ang, worst_pos, ok = 0.0, 0.0, True
    for i, j in _match_by_type(ma, mb):
        jt = ma[i]["joint_type"]
        ang = axis_angle_deg(ma[i]["direction"], mb[j]["direction"])
        cdist = np.linalg.norm(np.subtract(ma[i]["center"],
                                           mb[j]["center"])) * 100
        line = (f", origin line dist = "
                f"{line_dist_cm(ma[i]['origin'], ma[i]['direction'], mb[j]['origin'], mb[j]['direction']):.2f} cm"
                if jt == "r" else "")
        pos = line_dist_cm(ma[i]["origin"], ma[i]["direction"],
                           mb[j]["origin"], mb[j]["direction"]) if jt == "r" else 0.0
        print(f"  {jt}-joint shipped#{i} <-> rerun#{j}: "
              f"axis angle = {ang:.2f} deg{line}, center dist = {cdist:.2f} cm")
        worst_ang, worst_pos = max(worst_ang, ang), max(worst_pos, pos)
        ok &= ang <= tol_ang and pos <= tol_pos

    if ok:
        print(f"  ✅ [CONSISTENT] all matched joints within tolerances "
              f"(angle <= {tol_ang} deg, r-origin <= {tol_pos} cm); "
              "residuals are TAPIP3D/clustering stochasticity")
    else:
        print(f"  ❌ [DIVERGENT] worst axis angle {worst_ang:.2f} deg / "
              f"origin dist {worst_pos:.2f} cm exceed tolerances; inspect "
              "this scene's tracks and cluster assignment")


# --------------------------------------------------------- filtered.npz ----

def _frame_cloud(coords, visibs, f, cap=4000):
    pts = coords[f][visibs[f] > 0.5]
    if len(pts) > cap:
        pts = pts[:: max(1, len(pts) // cap)]
    return pts


def compare_filtered(pa, pb, tol_nn_cm, tol_count=0.20):
    print("=" * 62)
    print("filtered.npz  (Step-3 product: distribution consistency, "
          "byte identity not expected)")
    print(f"  shipped: {pa}\n  rerun  : {pb}")
    if not both_exist(pa, pb, "filtered.npz"):
        return
    if md5(pa) == md5(pb):
        print("  ✅ [BYTE-IDENTICAL] md5 match (unusual for a stochastic "
              "product, but fine).")
        return

    da, db = np.load(pa), np.load(pb)
    ca, va = da["coords"], da["visibs"]
    cb, vb = db["coords"], db["visibs"]
    print(f"  coords: shipped {ca.shape}, rerun {cb.shape}  |  "
          f"mean visibility: {va.mean():.3f} vs {vb.mean():.3f}")
    if ca.shape[0] != cb.shape[0]:
        print("  ❌ [DIVERGENT] frame count differs -> different n_canonical "
              "or input segmentation")
        return
    rel_n = abs(ca.shape[1] - cb.shape[1]) / max(ca.shape[1], 1)
    T = ca.shape[0]
    worst = 0.0
    for f in (0, T // 2, T - 1):
        A, B = _frame_cloud(ca, va, f), _frame_cloud(cb, vb, f)
        if len(A) == 0 or len(B) == 0:
            continue
        d_ab = cKDTree(B).query(A)[0].mean()
        d_ba = cKDTree(A).query(B)[0].mean()
        m = max(d_ab, d_ba) * 100
        worst = max(worst, m)
        print(f"  frame {f:3d}: bidirectional mean NN = {m:.3f} cm "
              f"({len(A)} vs {len(B)} visible pts)")

    if rel_n <= tol_count and worst <= tol_nn_cm:
        print(f"  ✅ [CONSISTENT] surviving-track count within "
              f"{tol_count*100:.0f}% and per-frame mean NN <= {tol_nn_cm} cm; "
              "both runs track the same surfaces")
    else:
        print(f"  ❌ [DIVERGENT] track count rel diff {rel_n*100:.1f}% or "
              f"worst mean NN {worst:.2f} cm exceed tolerances")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", default="data/videoartgs/sapien/168",
                    help="official shipped scene directory")
    ap.add_argument("--rerun", default="new_data/videoartgs/sapien/168",
                    help="scene directory containing your regenerated artifacts")
    ap.add_argument("--tol-ang", type=float, default=5.0,
                    help="joint_infos: max acceptable axis angle diff (deg)")
    ap.add_argument("--tol-pos", type=float, default=5.0,
                    help="joint_infos: max acceptable r-origin line dist (cm)")
    ap.add_argument("--tol-nn", type=float, default=2.0,
                    help="filtered.npz: max acceptable per-frame mean NN (cm)")
    args = ap.parse_args()

    j = os.path.join
    compare_transforms(j(args.shipped, "transforms.json"),
                       j(args.rerun, "transforms.json"))
    compare_pointclouds(j(args.shipped, "point_cloud.ply"),
                        j(args.rerun, "point_cloud.ply"))
    compare_vlm_joint_infos(j(args.shipped, "joint_infos_vlm.json"),
                            j(args.rerun, "joint_infos_vlm.json"))
    compare_joint_infos(j(args.shipped, "joint_infos.json"),
                        j(args.rerun, "joint_infos.json"),
                        args.tol_ang, args.tol_pos)
    compare_filtered(j(args.shipped, "filtered.npz"),
                     j(args.rerun, "filtered.npz"), args.tol_nn)