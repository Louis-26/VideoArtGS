"""
Extract VGGT image features for the canonical point cloud of each scene and
reduce them to a per-point 128-d descriptor for use as an extra PAT input.

Backbone: SpatialTrackerV2 front-end `VGGT4Track` (the VGGT-1B architecture:
DINOv2 ViT-L/14 patch embedding, 24 alternating frame/global attention blocks,
embed_dim 1024). We take the LAST aggregator layer, whose tokens are the
frame-attention and global-attention outputs concatenated -> 2048-d per patch
(37 x 37 patches at 518 x 518).

Per scene:
  1. pick --num_views canonical (static) frames uniformly from the first --n_canonical
  2. run the aggregator, keep the last layer's patch tokens        (S, 37, 37, 2048)
  3. project every point of point_cloud.ply into each view with the GT camera
     (transforms.json, Blender -> OpenCV), z-test against the GT depth map, and
     average the tokens of the patches it lands in                    (N, 2048)
  4. after all scenes: fit a PCA (2048 -> --pca_dim) on a subsample of all
     valid points, transform every scene                              (N, 128)

Outputs
  <scene>/pat_extra/vggt_raw.npy    (N, 2048) fp16   multi-view mean token
  <scene>/pat_extra/vggt_valid.npy  (N,)      bool   point seen in >= 1 view
  <scene>/pat_extra/vggt128.npy     (N, 128)  fp32   PCA-reduced (zeros where invalid)
  particulate/model_ckpt/pca_vggt.npz         mean / components / explained variance

Run inside the `st2` conda environment (needs the SpatialTrackerV2 deps):
  python data_tools/extract_vggt_feats.py --data_dir ./data/videoartgs/sapien
"""
import os
import sys
import json
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ST2_DIR = os.path.join(ROOT, "third_party", "SpatialTrackerV2")
sys.path.insert(0, ST2_DIR)
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track          # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image     # noqa: E402

PATCH = 14
PROC_SIZE = 518


def load_ply_xyz(path):
    d = PlyData.read(path).elements[0].data
    return np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32)


def load_cameras(scene_dir):
    meta = json.load(open(os.path.join(scene_dir, "transforms.json")))
    K = np.array([[meta["focal_x"], 0, meta["cx"]], [0, meta["focal_y"], meta["cy"]], [0, 0, 1]], dtype=np.float64)
    frames = meta["frames"]
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    c2w[:, :3, :3] = c2w[:, :3, :3] @ np.diag([1, -1, -1])          # Blender camera -> OpenCV camera
    w2c = np.linalg.inv(c2w)
    paths = [f["file_path"] for f in frames]
    return K, w2c, paths, int(meta["w"]), int(meta["h"])


@torch.no_grad()
def extract_scene(model, scene_dir, num_views, n_canonical, depth_tol, device):
    xyz = load_ply_xyz(os.path.join(scene_dir, "point_cloud.ply"))
    N = len(xyz)
    K, w2c, paths, W, H = load_cameras(scene_dir)
    view_ids = np.linspace(0, n_canonical - 1, num_views).round().astype(int)

    imgs, depths = [], []
    for i in view_ids:
        rgba = np.array(Image.open(os.path.join(scene_dir, paths[i])).convert("RGBA")).astype(np.float32)
        alpha = rgba[..., 3:4] / 255.0
        rgb = rgba[..., :3] * alpha + 255.0 * (1 - alpha)                # white background, as prepare_data()
        imgs.append(torch.from_numpy(rgb).permute(2, 0, 1))              # (3, H, W) in [0, 255]
        dpath = os.path.join(scene_dir, paths[i].replace("images", "depth"))
        depths.append(cv2.imread(dpath, -1).astype(np.float32) / 1e3)    # uint16 mm -> m
    imgs = torch.stack(imgs)                                             # (S, 3, H, W)
    imgs_proc = (preprocess_image(imgs) / 255.0).clamp(0, 1)             # (S, 3, 518, 518) for 800x800 input
    S, _, Hp, Wp = imgs_proc.shape
    assert Hp == PROC_SIZE and Wp == PROC_SIZE, (Hp, Wp)
    gh, gw = Hp // PATCH, Wp // PATCH                                    # 37 x 37

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tokens_list, p0 = model.aggregator(imgs_proc[None].to(device))
    tok = tokens_list[-1][0, :, p0:, :].float()                          # (S, 1369, 2048)
    del tokens_list
    C = tok.shape[-1]
    tok = tok.view(S, gh, gw, C)

    xyz_t = torch.from_numpy(xyz).to(device)
    ones = torch.ones(N, 1, device=device)
    xyz_h = torch.cat([xyz_t, ones], dim=1).double()                    # (N, 4)
    K_t = torch.from_numpy(K).to(device)
    feat_sum = torch.zeros(N, C, device=device)
    cnt = torch.zeros(N, device=device)
    sx, sy = Wp / W, Hp / H
    for s, i in enumerate(view_ids):
        cam = (torch.from_numpy(w2c[i]).to(device) @ xyz_h.T).T[:, :3]   # (N, 3) camera coords
        z = cam[:, 2]
        uv = (K_t @ cam.T).T
        u = uv[:, 0] / z.clamp(min=1e-6)
        v = uv[:, 1] / z.clamp(min=1e-6)
        ui, vi = u.round().long(), v.round().long()
        inb = (z > 1e-3) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        dmap = torch.from_numpy(depths[s]).to(device)
        d_gt = torch.zeros(N, device=device, dtype=torch.float64)
        d_gt[inb] = dmap[vi[inb], ui[inb]].double()
        vis = inb & (d_gt > 0) & ((z - d_gt).abs() < depth_tol)
        pu = ((u * sx) // PATCH).long().clamp(0, gw - 1)
        pv = ((v * sy) // PATCH).long().clamp(0, gh - 1)
        feat_sum[vis] += tok[s, pv[vis], pu[vis]]
        cnt[vis] += 1
    valid = cnt > 0
    feat = torch.zeros(N, C, device=device)
    feat[valid] = feat_sum[valid] / cnt[valid, None]
    return feat.half().cpu().numpy(), valid.cpu().numpy(), float(valid.float().mean())


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/videoartgs/sapien")
    parser.add_argument("--scenes", type=str, default="")
    parser.add_argument("--num_views", type=int, default=24)
    parser.add_argument("--n_canonical", type=int, default=150)
    parser.add_argument("--depth_tol", type=float, default=0.02, help="z-test tolerance in metres")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--pca_points_per_scene", type=int, default=25000)
    parser.add_argument("--pca_path", type=str, default=os.path.join(ROOT, "particulate", "model_ckpt", "pca_vggt.npz"))
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--skip_extract", action="store_true", help="only (re)fit PCA from existing vggt_raw.npy")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    scenes = [s for s in args.scenes.split(",") if s] or sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    device = "cuda"

    if not args.skip_extract:
        model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(device)
        for scene in scenes:
            out_dir = os.path.join(data_dir, scene, "pat_extra")
            os.makedirs(out_dir, exist_ok=True)
            raw_path = os.path.join(out_dir, "vggt_raw.npy")
            if os.path.exists(raw_path) and not args.reprocess:
                print(f"[{scene}] vggt_raw.npy exists, skipping extraction", flush=True)
                continue
            t0 = time.time()
            feat, valid, frac = extract_scene(model, os.path.join(data_dir, scene), args.num_views,
                                              args.n_canonical, args.depth_tol, device)
            np.save(raw_path, feat)
            np.save(os.path.join(out_dir, "vggt_valid.npy"), valid)
            print(f"[{scene}] raw tokens {feat.shape}, visible fraction {frac:.3f} ({time.time() - t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    # ---- PCA over all scenes (train = test here, so fit on everything) ----
    rng = np.random.RandomState(0)
    samples = []
    for scene in scenes:
        out_dir = os.path.join(data_dir, scene, "pat_extra")
        feat = np.load(os.path.join(out_dir, "vggt_raw.npy"))
        valid = np.load(os.path.join(out_dir, "vggt_valid.npy"))
        idx = np.where(valid)[0]
        idx = rng.choice(idx, min(len(idx), args.pca_points_per_scene), replace=False)
        samples.append(feat[idx].astype(np.float32))
    X = torch.from_numpy(np.concatenate(samples, 0)).to(device)
    mean = X.mean(0, keepdim=True)
    U, Sv, V = torch.pca_lowrank(X - mean, q=args.pca_dim, center=False, niter=4)
    comps = V.T.contiguous()                                              # (pca_dim, 2048)
    var = (Sv ** 2) / (X.shape[0] - 1)
    total_var = ((X - mean) ** 2).sum() / (X.shape[0] - 1)
    explained = (var / total_var).cpu().numpy()
    np.savez(args.pca_path, mean=mean[0].cpu().numpy(), components=comps.cpu().numpy(),
             explained_variance_ratio=explained, pca_dim=args.pca_dim, scenes=np.array(scenes))
    print(f"PCA fitted on {X.shape[0]} points: {args.pca_dim} comps explain {explained.sum():.3f} of variance -> {args.pca_path}")

    for scene in scenes:
        out_dir = os.path.join(data_dir, scene, "pat_extra")
        feat = torch.from_numpy(np.load(os.path.join(out_dir, "vggt_raw.npy")).astype(np.float32)).to(device)
        valid = np.load(os.path.join(out_dir, "vggt_valid.npy"))
        red = ((feat - mean) @ comps.T).cpu().numpy().astype(np.float32)
        red[~valid] = 0
        np.save(os.path.join(out_dir, f"vggt{args.pca_dim}.npy"), red)
        print(f"[{scene}] vggt{args.pca_dim}.npy {red.shape}", flush=True)


if __name__ == "__main__":
    main()
