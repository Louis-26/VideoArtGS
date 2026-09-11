"""Pipeline-faithful axis error: replicates init_deform_PAT.bridge_pat_to_original.

The training metric feeds PAT the GT part ids, which is not what the pipeline
does. The pipeline uses PAT's own argmax segmentation, drops fragment parts,
Hungarian-matches GT joint centres to PAT part centroids under a type-
compatibility penalty and a distance gate, and keeps the original joint_infos
initialisation for anything that does not match. That fallback is why a badly
segmented scene can still score well -- and why a confident but wrong PAT part
is the thing that actually hurts.

Running it with --base reproduces the no-PAT baseline, which is the check that
this proxy tracks the number the real pipeline reports.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from scipy.optimize import linear_sum_assignment

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "PAT"))
from particulate.articulation_utils import plucker_to_axis_point
from particulate.models import PAT_B
from utils.metrics import read_gt
from PAT_train_full import CACHE_DIR, MAX_PARTS, PAT_KWARGS, DATA_PATH

MATCH_DIST_RATIO = 0.3
FRAGMENT_FRAC = 0.001


def gt_joint_for_slot(scene, joint_infos_moving):
    """Map each joint_infos moving slot to a GT joint via GT part centroids."""
    import trimesh
    parts = sorted(glob.glob(os.path.join(DATA_PATH, scene, "gt", "part_*.ply")),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    gts = read_gt(os.path.join(DATA_PATH, scene, "gt", "mobility_v2.json"))
    cent = np.stack([np.asarray(trimesh.load(p, process=False, force="mesh").vertices).mean(0)
                     for p in parts[1:]])
    jc = np.array([j["center"] for j in joint_infos_moving], dtype=np.float64)
    rows, cols = linear_sum_assignment(np.linalg.norm(jc[:, None] - cent[None], axis=-1))
    return {int(r): gts[int(c)] for r, c in zip(rows, cols)}


def line_dist(d1, o1, d2, o2):
    """Distance between two axis lines (utils.metrics.axis_metrics convention)."""
    c = np.cross(d1, d2); n = np.linalg.norm(c)
    if n < 1e-8:
        w = o2 - o1
        return float(np.linalg.norm(w - (w @ d1) * d1))
    return float(abs((o2 - o1) @ c) / n)


def angle(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))


def bridge_scene(model, scene, cache, use_pat=True, keep_origin=False, verbose=False):
    ji = json.load(open(os.path.join(DATA_PATH, scene, "joint_infos.json")))
    moving = [j for j in ji if j["joint_type"] in ("r", "p")]
    gtmap = gt_joint_for_slot(scene, moving)
    final = [np.asarray(j["direction"], float).copy() for j in moving]
    final_o = [np.asarray(j.get("origin", [0, 0, 0]), float).copy() for j in moving]
    orig_o = [np.asarray(j.get("origin", [0, 0, 0]), float).copy() for j in moving]
    replaced = [False] * len(moving)

    if use_pat:
        xyz_n = cache["real_xyz"]; center = cache["real_center"]; scale = float(cache["real_scale"])
        xyz_w = xyz_n * scale + center
        r = model.infer(
            xyz=torch.as_tensor(xyz_n, device="cuda").unsqueeze(0),
            feats=torch.as_tensor(cache["real_feats"].astype(np.float32), device="cuda").unsqueeze(0),
            normals=torch.as_tensor(cache["real_normals"], device="cuda").unsqueeze(0),
        )[0]
        pid = r["part_ids"]; is_rev = r["is_part_revolute"]; is_pris = r["is_part_prismatic"]
        uniq = np.unique(pid)
        min_pts = max(16, int(FRAGMENT_FRAC * len(pid)))
        uniq = np.array([p for p in uniq if (pid == p).sum() >= min_pts])
        if len(uniq) and len(moving):
            cents = np.stack([xyz_w[pid == p].mean(0) for p in uniq])
            jc = np.array([j["center"] for j in moving], float)
            dist = np.linalg.norm(jc[:, None] - cents[None], axis=-1)
            compat = np.stack([[bool(is_rev[p]) if j["joint_type"] == "r" else bool(is_pris[p])
                                for p in uniq] for j in moving])
            rows, cols = linear_sum_assignment(dist + np.where(compat, 0.0, 1e6))
            gate = MATCH_DIST_RATIO * scale
            for k, c in zip(rows, cols):
                if not compat[k, c] or dist[k, c] > gate:
                    continue
                p = uniq[c]
                if moving[k]["joint_type"] == "r":
                    d, pt = plucker_to_axis_point(r["revolute_plucker"][p])
                    if np.linalg.norm(d) < 1e-8:
                        continue
                    if not keep_origin:
                        final_o[k] = np.asarray(pt, float) * scale + center
                else:
                    d = r["prismatic_axis"][p]
                    if np.linalg.norm(d) < 1e-8:
                        continue
                final[k] = np.asarray(d, float); replaced[k] = True

    out = []
    for k in range(len(moving)):
        if k not in gtmap:
            continue
        gd = np.asarray(gtmap[k]["direction"], float); gd /= np.linalg.norm(gd) + 1e-12
        go = np.asarray(gtmap[k]["origin"], float)
        rev = moving[k]["joint_type"] == "r"
        nd = final[k] / (np.linalg.norm(final[k]) + 1e-12)
        od = np.asarray(moving[k]["direction"], float)
        od = od / (np.linalg.norm(od) + 1e-12)
        out.append(dict(scene=scene, slot=k, typ=moving[k]["joint_type"],
                        ang=angle(final[k], gd),
                        orig=angle(od, gd),
                        pos=line_dist(nd, final_o[k], gd, go) if rev else np.nan,
                        pos_orig=line_dist(od, orig_o[k], gd, go) if rev else np.nan,
                        replaced=replaced[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--base", action="store_true", help="no PAT: score joint_infos as-is")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--keep_origin", action="store_true",
                    help="take PAT's direction but keep joint_infos' origin")
    a = ap.parse_args()

    model = None
    if not a.base:
        model = PAT_B(**PAT_KWARGS).cuda()
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu"), strict=True)
        model.eval()

    rows = []
    for f in sorted(os.listdir(CACHE_DIR)):
        s = f[:-4]
        z = np.load(os.path.join(CACHE_DIR, f))
        cache = {k: z[k] for k in ("real_xyz", "real_normals", "real_feats",
                                   "real_center", "real_scale")} if not a.base else {}
        rows += bridge_scene(model, s, cache, use_pat=not a.base, keep_origin=a.keep_origin)
    ang = np.array([r["ang"] for r in rows]); org = np.array([r["orig"] for r in rows])
    pos = np.array([r["pos"] for r in rows], float); pos = pos[~np.isnan(pos)]
    poso = np.array([r["pos_orig"] for r in rows], float); poso = poso[~np.isnan(poso)]
    nrep = sum(r["replaced"] for r in rows)
    tag = "NO-PAT (joint_infos as-is)" if a.base else os.path.basename(a.ckpt or "?")
    tag += "  [keep_origin]" if a.keep_origin else ""
    print(f"\n{tag}")
    print(f"  axis angle  mean {ang.mean():6.3f}° ± {ang.std():.3f}   median {np.median(ang):.3f}°   "
          f"n={len(ang)}  replaced_by_PAT={nrep}")
    print(f"  (original joint_infos reference: mean {org.mean():6.3f}° ± {org.std():.3f})")
    print(f"  revolute origin (axis-line dist, world units)  PAT {pos.mean():.4f} ± {pos.std():.4f}   "
          f"median {np.median(pos):.4f}   |  orig {poso.mean():.4f} ± {poso.std():.4f}  n={len(pos)}")
    if not a.quiet:
        bad = sorted(rows, key=lambda r: -r["ang"])[:8]
        for r in bad:
            pz = "  --  " if np.isnan(r['pos']) else f"{r['pos']:.4f}"
            po = "  --  " if np.isnan(r['pos_orig']) else f"{r['pos_orig']:.4f}"
            print(f"    {r['scene']:>7} slot{r['slot']}[{r['typ']}] {r['ang']:7.2f}° (orig {r['orig']:5.2f}°) "
                  f"pos {pz} (orig {po}) replaced={r['replaced']}")
        print("  worst revolute origin:")
        for r in sorted([x for x in rows if not np.isnan(x['pos'])], key=lambda y: -y['pos'])[:8]:
            print(f"    {r['scene']:>7} slot{r['slot']}[r] pos {r['pos']:.4f} (orig {r['pos_orig']:.4f}) "
                  f"ang {r['ang']:5.2f}° replaced={r['replaced']}")
    return ang.mean()


if __name__ == "__main__":
    main()
