"""
Re-run TAPIP3D on the dynamic frames of each scene and export the tracker's
INTERNAL per-track features for use as an extra PAT input modality.

Per scene this writes  <scene>/pat_extra/tapip3d_feats.npz  with
    coords       (T, N, 3)  world-frame 3D tracks of this run (N ~ 8192 raw tracks)
    visibs       (T, N)     visibility
    query_points (N, 4)     (t, x, y, z) query of every track
    track_feats  (N, 384)   EfficientUpdateFormer hidden state (last refinement
                            iteration, mean over sliding windows and frames), fp16

It never touches filtered.npz / joint_infos.json, which the VideoArtGS pipeline
keeps using unchanged. Requires third_party/TAPIP3D/inference.py with the
--save_track_feats flag (VideoArtGS addition).

    python data_tools/extract_tapip3d_feats.py --data_dir ./data/videoartgs/sapien --scenes 168,1280
"""
import os
import sys
import time
import subprocess
from argparse import ArgumentParser

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_tools"))
from extract_tapip3d_track import prepare_data          # noqa: E402
from utils.metrics import read_joint_infos_vlm           # noqa: E402


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/videoartgs/sapien")
    parser.add_argument("--scenes", type=str, default="", help="comma separated; default: all scenes in data_dir")
    parser.add_argument("--tapip3d_dir", type=str, default="./third_party/TAPIP3D")
    parser.add_argument("--n_query_points", type=int, default=8192)
    parser.add_argument("--n_canonical", type=int, default=150, help="static canonical frames to skip (sapien: 150)")
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    tapip3d_dir = os.path.abspath(args.tapip3d_dir)
    scenes = [s for s in args.scenes.split(",") if s] or sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))

    for scene in scenes:
        t0 = time.time()
        out_dir = os.path.join(data_dir, scene, "pat_extra")
        os.makedirs(out_dir, exist_ok=True)
        final_path = os.path.join(out_dir, "tapip3d_feats.npz")
        if os.path.exists(final_path) and not args.reprocess:
            print(f"[{scene}] tapip3d_feats.npz exists, skipping")
            continue

        # same query-set count as data_tools/extract_tapip3d_track.py
        joint_infos = read_joint_infos_vlm(os.path.join(data_dir, scene, "joint_infos_vlm.json"))
        nq = 4 + len(joint_infos) // 2

        input_path = os.path.join(out_dir, f"{scene}_tapin.npz")
        prepare_data(data_dir, scene, args.n_canonical, input_path)

        cmd = (f"cd {tapip3d_dir} && {sys.executable} inference.py --input_path {input_path} "
               f"--n_query_frames {nq} --n_query_points {args.n_query_points} "
               f"--output_dir {out_dir} --save_track_feats")
        print(f"[{scene}] running: {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=True)

        raw_path = os.path.join(out_dir, f"{scene}_tapin.n{nq}.npz")
        d = np.load(raw_path)
        assert "track_feats" in d.files, f"{raw_path} has no track_feats; is inference.py patched?"
        np.savez_compressed(
            final_path,
            coords=d["coords"].astype(np.float32),
            visibs=d["visibs"].astype(bool),
            query_points=d["query_points"].astype(np.float32),
            track_feats=d["track_feats"].astype(np.float16),
        )
        os.remove(raw_path)
        if os.path.exists(input_path):
            os.remove(input_path)
        print(f"[{scene}] saved {final_path}: coords {d['coords'].shape}, track_feats {d['track_feats'].shape} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
